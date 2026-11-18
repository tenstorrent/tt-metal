# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""LSE outputs and the KV pair split of MLA attention.

- ``paged_flash_multi_latent_attention_decode(..., return_lse=True)`` returns ``(out, lse)``, lse fp32
  [1, B, nh, 1] = log(sum_k exp(scale * q.k)) over each user's visible keys (-1e30 and a zero output for a user
  at cur_pos -1).
- ``chunked_flash_mla_prefill(..., return_lse=True)`` returns ``(out, lse)``, lse fp32 [1, nh, S, 1].
- ``chunked_flash_mla_prefill(..., kv_pair_parity_tensor=c)``: the cache holds chip c's half of every 128-token
  global block (local key l is global position G(l, c) = 128 * (l >> 6) + 64 * c + (l & 63)); query i of the
  chunk is at global position start + i and sees local key l iff G(l, c) <= start + i.

Two chips of a pair are emulated by two calls on one device; their (out, lse) merged with the LSE weights must
equal full attention over the whole cache.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc

TILE = 32
DK = 192  # K head dim (q.k width)
DV = 128  # V head dim
NH = 16  # Q heads of the KV group
BLOCK = 64  # cache block (local tokens)
NO_KEY_LSE = -1e30


# ---------------------------------------------------------------- pair-split index maps (see the spec)
def own(p, c):
    return ((p >> 6) & 1) == c


def local_index(p):
    return 64 * (p >> 7) + (p & 63)


def global_index(l, c):
    return 128 * (l >> 6) + 64 * c + (l & 63)


def visible_local(p, c):
    """L(p, c): local keys on chip c visible at global position p (inclusive)."""
    if p < 0:
        return 0
    return 64 * (p >> 7) + min(max((p & 127) + 1 - 64 * c, 0), 64)


def split_global(x, c):
    """Chip c's local rows of a global [..., N, D] sequence (N a multiple of 128): local l <- global G(l, c)."""
    n = x.shape[-2]
    idx = torch.tensor([global_index(l, c) for l in range(n // 2)])
    return x[..., idx, :]


def merge(o_a, lse_a, o_b, lse_b):
    """Exact merge of two partial attentions over disjoint key sets (fp32). o [..., D], lse [..., 1]."""
    m = torch.maximum(lse_a, lse_b)
    w_a, w_b = torch.exp(lse_a - m), torch.exp(lse_b - m)
    return (w_a * o_a + w_b * o_b) / (w_a + w_b)


def merge_tt(o_a, lse_a, o_b, lse_b):
    """The merge with ttnn eltwise ops on device: the lse tensors ([..., 1], fp32) broadcast against the outputs."""
    m = ttnn.maximum(lse_a, lse_b)
    w_a = ttnn.exp(ttnn.subtract(lse_a, m))
    w_b = ttnn.exp(ttnn.subtract(lse_b, m))
    inv = ttnn.reciprocal(ttnn.add(w_a, w_b))
    alpha_a, alpha_b = ttnn.multiply(w_a, inv), ttnn.multiply(w_b, inv)
    return ttnn.add(ttnn.multiply(o_a, alpha_a), ttnn.multiply(o_b, alpha_b))


def ref_attention(q, k, v, scale, visible):
    """q [R, DK], k [S, DK], v [S, DV], visible [R, S] bool -> (out [R, DV], lse [R]); no visible key: 0, -1e30."""
    scores = (q.float() @ k.float().T) * scale
    scores = scores.masked_fill(~visible, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    out = probs @ v.float()
    none = ~visible.any(dim=-1)
    out[none] = 0.0
    lse[none] = NO_KEY_LSE
    return out, lse


def compute_config(fp32_acc=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


def assert_lse_close(got, expected, atol, what):
    valid = expected > NO_KEY_LSE / 2
    assert torch.all(got[~valid] == NO_KEY_LSE), f"{what}: rows with no visible key must have lse -1e30"
    if valid.any():
        err = (got[valid] - expected[valid]).abs().max().item()
        assert err < atol, f"{what}: max |lse - ref| = {err} (atol {atol})"


# ---------------------------------------------------------------- decode
def decode_program(batch, table_tokens):
    """MiMo's global split decode program (attention._split_sdpa_program): the plain decode program from 32 users."""
    if batch >= TILE:
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=0,
        k_chunk_size=256 if table_tokens % 256 == 0 else 128,
        exp_approx_mode=False,
        max_cores_per_head_batch=max(16, min(64, 64 // batch)),
    )


def q_decode_tt(device, q, sharded):
    """q [1, B, NH, DK] -> device, height-sharded one user per core in L1 (MiMo) or DRAM interleaved."""
    batch = q.shape[1]
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    if sharded:
        memory_config = ttnn.create_sharded_memory_config(
            shape=(TILE, DK),
            core_grid=ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    return ttnn.from_torch(q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)


class PagedCache:
    """Per-user local sequences [B, n_local, D] in a shuffled paged cache [B * blocks, 1, BLOCK, D]."""

    def __init__(self, device, seqs, page_table, dtype):
        batch, n_local, dim = seqs.shape
        blocks = n_local // BLOCK
        cache = torch.zeros(batch * blocks, 1, BLOCK, dim)
        for b in range(batch):
            cache[page_table[b], 0] = seqs[b].reshape(blocks, BLOCK, dim)
        self.tt = ttnn.from_torch(cache, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        stored = ttnn.to_torch(self.tt).float()
        # The cache values as stored (block float rounding), back in sequence order: the reference's K / V.
        self.seqs = torch.stack([stored[page_table[b], 0].reshape(n_local, dim) for b in range(batch)])


def run_decode(device, q, k_cache, v_cache, page_table_tt, cur_pos, scale, sharded, return_lse, keep_tt=None):
    batch = q.shape[1]
    table_tokens = page_table_tt.shape[-1] * BLOCK
    result = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        q_decode_tt(device, q, sharded),
        k_cache.tt,
        v_cache.tt,
        head_dim_v=DV,
        page_table_tensor=page_table_tt,
        cur_pos_tensor=ttnn.from_torch(torch.tensor(cur_pos, dtype=torch.int32), device=device, dtype=ttnn.int32),
        scale=scale,
        program_config=decode_program(batch, table_tokens),
        compute_kernel_config=compute_config(),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        return_lse=return_lse,
    )
    if not return_lse:
        return ttnn.to_torch(result).float()
    out, lse = result
    assert tuple(lse.shape) == (1, batch, NH, 1), lse.shape
    assert lse.dtype == ttnn.float32 and lse.layout == ttnn.TILE_LAYOUT
    if keep_tt is not None:
        keep_tt.append((out, lse))
    return ttnn.to_torch(out).float(), ttnn.to_torch(lse).float()


def make_page_table(batch, blocks):
    return torch.randperm(batch * blocks).reshape(batch, blocks)


@pytest.mark.parametrize("sharded_q", [True, False], ids=["q_l1_sharded", "q_dram"])
@pytest.mark.parametrize(
    "cur_pos",
    [
        [-1, 0, 31, 63, 64, 300, 1023, 517],
        [700, -1],
        [(-1 if b % 7 == 3 else (37 * b * b + 11 * b) % 1024) for b in range(32)],
    ],
    ids=["b8", "b2", "b32"],
)
def test_mla_decode_return_lse(device, cur_pos, sharded_q, function_level_defaults, reset_seeds):
    """return_lse=True: lse matches log-sum-exp of the scaled scores, the output matches return_lse=False and the
    reference, and a skipped user (cur_pos -1) gets output 0 and lse -1e30."""
    batch, n = len(cur_pos), 1024
    scale = DK**-0.5
    q = torch.randn(1, batch, NH, DK)
    page_table = make_page_table(batch, n // BLOCK)
    k_cache = PagedCache(device, torch.randn(batch, n, DK), page_table, ttnn.bfloat8_b)
    v_cache = PagedCache(device, torch.randn(batch, n, DV), page_table, ttnn.bfloat8_b)
    page_table_tt = ttnn.from_torch(page_table.to(torch.int32), device=device, dtype=ttnn.int32)

    plain = run_decode(device, q, k_cache, v_cache, page_table_tt, cur_pos, scale, sharded_q, return_lse=False)
    out, lse = run_decode(device, q, k_cache, v_cache, page_table_tt, cur_pos, scale, sharded_q, return_lse=True)

    for b, p in enumerate(cur_pos):
        visible = (torch.arange(n) <= p)[None, :].expand(NH, n)
        ref_out, ref_lse = ref_attention(q[0, b], k_cache.seqs[b], v_cache.seqs[b], scale, visible)
        assert_lse_close(lse[0, b, :, 0], ref_lse, 0.05, f"user {b} at {p}")
        if p < 0:
            assert torch.all(out[0, b] == 0), f"user {b}: skipped user output must be 0"
            continue
        # The LSE output only adds an output: same attention as return_lse=False.
        assert torch.equal(out[0, b], plain[0, b]), f"user {b}: output differs from return_lse=False"
        passing, message = comp_pcc(ref_out, out[0, b], 0.999)
        assert passing, f"user {b} at {p}: {message}"


@pytest.mark.parametrize("batch_positions", [[0, 63, 64, 127, 128, 129, 191, 192, 1000, 1023, 2047, 1536]])
def test_mla_decode_pair_split_merge(device, batch_positions, function_level_defaults, reset_seeds):
    """Pair split decode: chip c caches only its half of every 128-token block; each chip's SDPA at the local
    position L(p, c) - 1 with return_lse, merged by the LSE weights, equals full attention over the global cache.
    Positions straddle the 64 / 128 boundaries, and p < 64 leaves chip 1 with nothing (cur_pos -1)."""
    batch, n = len(batch_positions), 2048
    scale = DK**-0.5
    q = torch.randn(1, batch, NH, DK)
    k_global, v_global = torch.randn(batch, n, DK), torch.randn(batch, n, DV)
    page_table = make_page_table(batch, n // 2 // BLOCK)  # both chips use the same table
    page_table_tt = ttnn.from_torch(page_table.to(torch.int32), device=device, dtype=ttnn.int32)

    parts, k_chips, v_chips, parts_tt = [], [], [], []
    for c in (0, 1):
        k_cache = PagedCache(device, split_global(k_global, c), page_table, ttnn.bfloat8_b)
        v_cache = PagedCache(device, split_global(v_global, c), page_table, ttnn.bfloat8_b)
        pos = [visible_local(p, c) - 1 for p in batch_positions]
        out, lse = run_decode(
            device, q, k_cache, v_cache, page_table_tt, pos, scale, True, return_lse=True, keep_tt=parts_tt
        )
        for b, p_local in enumerate(pos):
            visible = (torch.arange(n // 2) <= p_local)[None, :].expand(NH, n // 2)
            _, ref_lse = ref_attention(q[0, b], k_cache.seqs[b], v_cache.seqs[b], scale, visible)
            assert_lse_close(lse[0, b, :, 0], ref_lse, 0.05, f"chip {c} user {b}")
        parts.append((out, lse))
        k_chips.append(k_cache.seqs)
        v_chips.append(v_cache.seqs)

    # The global cache as the two chips store it.
    k_stored, v_stored = torch.zeros(batch, n, DK), torch.zeros(batch, n, DV)
    for c in (0, 1):
        idx = torch.tensor([global_index(l, c) for l in range(n // 2)])
        k_stored[:, idx], v_stored[:, idx] = k_chips[c], v_chips[c]

    (o_0, lse_0), (o_1, lse_1) = parts
    merged = merge(o_0, lse_0, o_1, lse_1)
    assert torch.isfinite(merged).all()
    # The same merge with ttnn ops on device: lse [1, B, nh, 1] fp32 broadcasts against out [1, B, nh, 128].
    (o_0_tt, lse_0_tt), (o_1_tt, lse_1_tt) = parts_tt
    merged_tt = ttnn.to_torch(merge_tt(o_0_tt, lse_0_tt, o_1_tt, lse_1_tt)).float()
    assert merged_tt.shape == merged.shape, merged_tt.shape
    for b, p in enumerate(batch_positions):
        visible = (torch.arange(n) <= p)[None, :].expand(NH, n)
        ref_out, _ = ref_attention(q[0, b], k_stored[b], v_stored[b], scale, visible)
        passing, message = comp_pcc(ref_out, merged[0, b], 0.999)
        assert passing, f"user {b} at {p}: merged pair vs full attention: {message}"
        passing, message = comp_pcc(ref_out, merged_tt[0, b], 0.999)
        assert passing, f"user {b} at {p}: on-device merge vs full attention: {message}"


def grouped_decode_program(reqs, table_tokens, cands):
    """The grouped decode's program: decode_program's; from 8 candidates 128-token K chunks and at most 32 cores per
    request (8 Q tiles beside 256-token chunks, or one more tree-reduction round, overflow L1)."""
    program = decode_program(reqs, table_tokens)
    if cands >= 8:
        program.k_chunk_size = 128
        program.max_cores_per_head_batch = min(program.max_cores_per_head_batch, 32)
    return program


def run_grouped_decode(
    device, q_slots, k_cache, v_cache, page_table_tt, cur_pos, scale, cands, return_lse=True, packed=False
):
    """A speculative verify's decode with one KV scan per request (spec_multi_pos_tiles): q_slots [1, R * T, NH, DK],
    candidate j of request r in slot r * T + j, goes in as Q [1, R, T * 32, DK] (DRAM interleaved) with candidate j's
    heads in rows 0..NH-1 of its 32-row tile j (``packed``: spec_multi_pos_packed, computed two candidates per tile);
    ``page_table_tt`` has one row per request and ``cur_pos`` one entry per candidate. Returns (out [1, R * T, NH, DV],
    lse [1, R * T, NH] or None without ``return_lse``) per slot, as the per-slot decode does. Packed, a candidate's
    output rows NH..31 must repeat rows 0..NH-1 and its lse rows NH..31 be -1e30."""
    slots = q_slots.shape[1]
    reqs = slots // cands
    rows = TILE  # Q / output rows per candidate
    q = torch.zeros(1, reqs, cands, rows, DK)
    q[:, :, :, :NH] = q_slots.reshape(1, reqs, cands, NH, DK)
    result = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        ttnn.from_torch(
            q.reshape(1, reqs, cands * rows, DK), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        ),
        k_cache.tt,
        v_cache.tt,
        head_dim_v=DV,
        page_table_tensor=page_table_tt,
        cur_pos_tensor=ttnn.from_torch(torch.tensor(cur_pos, dtype=torch.int32), device=device, dtype=ttnn.int32),
        scale=scale,
        program_config=grouped_decode_program(reqs, page_table_tt.shape[-1] * BLOCK, cands),
        compute_kernel_config=compute_config(),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        return_lse=return_lse,
        spec_multi_pos_tiles=cands,
        spec_multi_pos_packed=packed,
    )
    if not return_lse:
        out, lse = result, None
    else:
        out, lse = result
        assert tuple(lse.shape) == (1, reqs, cands * rows, 1), lse.shape
        lse = ttnn.to_torch(lse).float().reshape(1, reqs, cands, rows)
        if packed:
            assert torch.all(lse[..., NH:] == NO_KEY_LSE), "packed: lse padding rows must be -1e30"
        lse = lse[:, :, :, :NH].reshape(1, slots, NH)
    assert tuple(out.shape) == (1, reqs, cands * rows, DV), out.shape
    out = ttnn.to_torch(out).float().reshape(1, reqs, cands, rows, DV)
    if packed:
        assert torch.equal(out[..., NH : 2 * NH, :], out[..., :NH, :]), "packed: output padding rows repeat rows 0..15"
    out = out[:, :, :, :NH].reshape(1, slots, NH, DV)
    return out, lse


@pytest.mark.parametrize("layout", ["global", "pair_split"])
@pytest.mark.parametrize(
    "cands, starts",
    [
        (4, [1000]),
        (4, [61, 2044 - 3]),
        (4, [0, 30, 126, 1500]),
        (8, [123]),
        (2, [62, 127, 700, 2046]),
    ],
    ids=["t4_r1", "t4_r2", "t4_r4", "t8_r1", "t2_r4"],
)
@pytest.mark.parametrize("packed", [False, True], ids=["tiles", "packed"])
def test_mla_decode_spec_multi_pos(device, layout, cands, starts, packed, function_level_defaults, reset_seeds):
    """A speculative verify's MLA decode with one KV scan per request (spec_multi_pos_tiles = T candidates per page
    table row, return_lse): every candidate's output and lse match the per-slot decode (one row per candidate, the
    request's table row repeated) and the torch reference. Request r's candidates are at global positions
    starts[r] + j; with a pair split (each chip caching half of every 128-token block, both chips run) a candidate's
    position is the chip's local L(p, c) - 1, so near 64 chip 1 has leading -1 candidates (no local key yet) and
    repeated bounds. The last request of a call leaves its last two candidates as gaps (-1, a verify row with fewer
    drafts). A -1 candidate gets output 0 and lse -1e30. The two decodes round differently (the per-slot one runs
    16-row half tiles), so the grouped one must be as close to the reference as the per-slot one, not bit-equal.
    ``packed``: spec_multi_pos_packed, two candidates per compute tile (each half its own mask)."""
    n = 2048  # global tokens per request
    scale = DK**-0.5
    reqs = len(starts)
    slots = reqs * cands
    q_slots = torch.randn(1, slots, NH, DK)
    glob = [[s + j for j in range(cands)] for s in starts]
    glob[-1][-2:] = [-1, -1]
    k_global, v_global = torch.randn(reqs, n, DK), torch.randn(reqs, n, DV)
    chips = (None,) if layout == "global" else (0, 1)
    n_local = n if layout == "global" else n // 2
    page_table = make_page_table(reqs, n_local // BLOCK)
    group_table_tt = ttnn.from_torch(page_table.to(torch.int32), device=device, dtype=ttnn.int32)
    slot_table = page_table.repeat_interleave(cands, dim=0)  # the per-slot decode: each slot its request's row
    slot_table_tt = ttnn.from_torch(slot_table.to(torch.int32), device=device, dtype=ttnn.int32)
    for c in chips:
        k_seqs = k_global if c is None else split_global(k_global, c)
        v_seqs = v_global if c is None else split_global(v_global, c)
        k_cache = PagedCache(device, k_seqs, page_table, ttnn.bfloat8_b)
        v_cache = PagedCache(device, v_seqs, page_table, ttnn.bfloat8_b)
        pos = [p if (p < 0 or c is None) else visible_local(p, c) - 1 for row in glob for p in row]
        what = f"{layout} chip {c}"
        out, lse = run_grouped_decode(
            device, q_slots, k_cache, v_cache, group_table_tt, pos, scale, cands, packed=packed
        )
        k_slots = PagedCache(device, k_seqs.repeat_interleave(cands, dim=0), slot_table, ttnn.bfloat8_b)
        v_slots = PagedCache(device, v_seqs.repeat_interleave(cands, dim=0), slot_table, ttnn.bfloat8_b)
        ref_out_tt, ref_lse_tt = run_decode(
            device, q_slots, k_slots, v_slots, slot_table_tt, pos, scale, False, return_lse=True
        )
        for s, p in enumerate(pos):
            r = s // cands
            tag = f"{what} request {r} candidate {s % cands} at local {p}"
            if p < 0:
                assert torch.all(out[0, s] == 0), f"{tag}: a -1 candidate's output must be 0"
                assert torch.all(lse[0, s] == NO_KEY_LSE), f"{tag}: a -1 candidate's lse must be -1e30"
                continue
            visible = (torch.arange(n_local) <= p)[None, :].expand(NH, n_local)
            ref_out, ref_lse = ref_attention(q_slots[0, s], k_cache.seqs[r], v_cache.seqs[r], scale, visible)
            assert_lse_close(lse[0, s], ref_lse, 0.05, tag)
            assert_lse_close(ref_lse_tt[0, s, :, 0], ref_lse, 0.05, f"{tag}, the per-slot decode")
            grouped = float(comp_pcc(ref_out, out[0, s], 0.999)[1])
            per_slot = float(comp_pcc(ref_out, ref_out_tt[0, s], 0.999)[1])
            assert grouped >= 0.999 and grouped >= per_slot - 2e-4, f"{tag}: PCC {grouped} (per-slot {per_slot})"


@pytest.mark.parametrize("return_lse", [True, False], ids=["lse", "no_lse"])
@pytest.mark.parametrize("packed", [False, True], ids=["tiles", "packed"])
def test_mla_decode_spec_multi_pos_skipped(device, return_lse, packed, function_level_defaults, reset_seeds):
    """-1 candidates of spec_multi_pos_tiles with and without return_lse: request 0's group is all -1 (the whole row
    is skipped), request 1's has -1 on both sides of its two real candidates, request 2's none. A -1 candidate's output
    is 0 (and its lse -1e30) even where the output buffer held another call's results: a first call with every
    candidate real fills it, and the second call's output usually takes the same L1. The real candidates match the
    torch reference. ``packed``: two candidates per compute tile, so request 1's -1 candidates share tiles with real
    ones."""
    cands, n = 4, 2048
    scale = DK**-0.5
    pos = [-1, -1, -1, -1, -1, 300, 301, -1, 1000, 1001, 1002, 1003]
    reqs = len(pos) // cands
    q_slots = torch.randn(1, len(pos), NH, DK)
    k_seqs, v_seqs = torch.randn(reqs, n, DK), torch.randn(reqs, n, DV)
    page_table = make_page_table(reqs, n // BLOCK)
    table_tt = ttnn.from_torch(page_table.to(torch.int32), device=device, dtype=ttnn.int32)
    k_cache = PagedCache(device, k_seqs, page_table, ttnn.bfloat8_b)
    v_cache = PagedCache(device, v_seqs, page_table, ttnn.bfloat8_b)
    every = [500 * r + 100 + j for r in range(reqs) for j in range(cands)]
    first, _ = run_grouped_decode(device, q_slots, k_cache, v_cache, table_tt, every, scale, cands, return_lse, packed)
    assert torch.all(first.abs().amax(dim=(-2, -1)) > 0), "the first call must leave non-zero outputs"
    out, lse = run_grouped_decode(device, q_slots, k_cache, v_cache, table_tt, pos, scale, cands, return_lse, packed)
    for s, p in enumerate(pos):
        r = s // cands
        tag = f"request {r} candidate {s % cands} at {p} (return_lse {return_lse})"
        if p < 0:
            assert torch.all(out[0, s] == 0), f"{tag}: a -1 candidate's output must be 0"
            if return_lse:
                assert torch.all(lse[0, s] == NO_KEY_LSE), f"{tag}: a -1 candidate's lse must be -1e30"
            continue
        visible = (torch.arange(n) <= p)[None, :].expand(NH, n)
        ref_out, ref_lse = ref_attention(q_slots[0, s], k_cache.seqs[r], v_cache.seqs[r], scale, visible)
        if return_lse:
            assert_lse_close(lse[0, s], ref_lse, 0.05, tag)
        passing, message = comp_pcc(ref_out, out[0, s], 0.999)
        assert passing, f"{tag}: {message}"


# ---------------------------------------------------------------- chunked prefill
def prefill_program(q_chunk, k_chunk):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def int32_scalar(device, value):
    return ttnn.from_torch(torch.tensor([value], dtype=torch.int32), device=device, dtype=ttnn.int32)


def run_prefill(
    device,
    q_chunk_t,
    k_cache,
    v_cache,
    page_table_tt,
    start,
    program,
    fp32_acc,
    return_lse,
    parity=None,
    scalar=False,
    keep_tt=None,
):
    """One chunked_flash_mla_prefill call on q [1, NH, S, DK] at chunk start ``start`` (a device tensor, or the
    scalar chunk_start_idx with ``scalar``)."""
    kwargs = dict(
        scale=DK**-0.5,
        program_config=program,
        compute_kernel_config=compute_config(fp32_acc),
        input_tensor_v=v_cache.tt,
        return_lse=return_lse,
    )
    if scalar:
        kwargs["chunk_start_idx"] = start
    else:
        kwargs["chunk_start_idx_tensor"] = int32_scalar(device, start)
    if parity is not None:
        kwargs["kv_pair_parity_tensor"] = int32_scalar(device, parity)
    result = ttnn.transformer.chunked_flash_mla_prefill(
        ttnn.from_torch(q_chunk_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        k_cache.tt,
        DV,
        page_table_tt,
        **kwargs,
    )
    if not return_lse:
        return ttnn.to_torch(result).float()
    out, lse = result
    seq = q_chunk_t.shape[2]
    assert tuple(lse.shape) == (1, NH, seq, 1), lse.shape
    assert lse.dtype == ttnn.float32 and lse.layout == ttnn.TILE_LAYOUT
    if keep_tt is not None:
        keep_tt.append((out, lse))
    return ttnn.to_torch(out).float(), ttnn.to_torch(lse).float()


def check_rows(got_out, got_lse, ref_out, ref_lse, what, lse_atol=0.05):
    """got/ref out [NH, S, DV], lse [NH, S]."""
    assert_lse_close(got_lse, ref_lse, lse_atol, what)
    none = ref_lse <= NO_KEY_LSE / 2
    assert torch.all(got_out[none] == 0), f"{what}: rows with no visible key must be 0"
    passing, message = comp_pcc(ref_out[~none], got_out[~none], 0.999)
    assert passing, f"{what}: {message}"


@pytest.mark.parametrize("fp32_acc", [False, True], ids=["streaming", "fp32_dest"])
@pytest.mark.parametrize("chunk", [128, 256])
def test_chunked_mla_prefill_return_lse(device, chunk, fp32_acc, function_level_defaults, reset_seeds):
    """return_lse=True on the plain chunked prefill: lse matches log-sum-exp of the scaled causal scores, and the
    output equals return_lse=False bit for bit."""
    n = 1024
    q = torch.randn(1, NH, n, DK)
    page_table = make_page_table(1, n // BLOCK)
    k_cache = PagedCache(device, torch.randn(1, n, DK), page_table, ttnn.bfloat8_b)
    v_cache = PagedCache(device, torch.randn(1, n, DV), page_table, ttnn.bfloat8_b)
    page_table_tt = ttnn.from_torch(page_table.to(torch.int32), device=device, dtype=ttnn.int32)
    program = prefill_program(chunk, chunk)
    for start in (0, chunk, n - 2 * chunk):
        q_chunk = q[:, :, start : start + 2 * chunk]
        # The scalar chunk_start_idx path for one start.
        scalar = start == chunk
        args = (device, q_chunk, k_cache, v_cache, page_table_tt, start, program, fp32_acc)
        plain = run_prefill(*args, False, scalar=scalar)
        out, lse = run_prefill(*args, True, scalar=scalar)
        assert torch.equal(out, plain), f"start {start}: output differs from return_lse=False"
        end = start + 2 * chunk
        visible = torch.arange(end)[None, :] <= torch.arange(start, end)[:, None]
        for h in range(NH):
            ref_out, ref_lse = ref_attention(
                q_chunk[0, h], k_cache.seqs[0, :end], v_cache.seqs[0, :end], DK**-0.5, visible
            )
            check_rows(out[0, h], lse[0, h, :, 0], ref_out, ref_lse, f"start {start} head {h}")


class PairSplitCaches:
    """One user's random global K / V of ``n`` tokens (a multiple of 128) split over the two chips of a pair: chip
    c's local cache through one shared page table (``k[c]``, ``v[c]``), and the global cache as the chips store it
    (``k_stored``, ``v_stored``: the full-attention reference of the merge)."""

    def __init__(self, device, n):
        k_global, v_global = torch.randn(1, n, DK), torch.randn(1, n, DV)
        self.page_table = make_page_table(1, n // 2 // BLOCK)  # both chips use the same table
        self.page_table_tt = ttnn.from_torch(self.page_table.to(torch.int32), device=device, dtype=ttnn.int32)
        self.k = [PagedCache(device, split_global(k_global, c), self.page_table, ttnn.bfloat8_b) for c in (0, 1)]
        self.v = [PagedCache(device, split_global(v_global, c), self.page_table, ttnn.bfloat8_b) for c in (0, 1)]
        self.k_stored, self.v_stored = torch.zeros(n, DK), torch.zeros(n, DV)
        for c in (0, 1):
            idx = torch.tensor([global_index(l, c) for l in range(n // 2)])
            self.k_stored[idx], self.v_stored[idx] = self.k[c].seqs[0], self.v[c].seqs[0]

    def table_tt(self, device, blocks):
        """The page table's first ``blocks`` columns (the local tokens [0, 64 * blocks)) as a device tensor."""
        return ttnn.from_torch(self.page_table[:, :blocks].to(torch.int32), device=device, dtype=ttnn.int32)


def check_pair_split_start(device, caches, q, start, seq, program, fp32_acc):
    """Pair-split chunked prefill of q[:, :, start : start + seq] on both chips (return_lse): each chip matches the
    pair-split reference, and the two merged by the LSE weights (in torch and with ttnn ops) equal full causal
    attention over the global cache. Returns the program cache entries each chip's call added."""
    scale = DK**-0.5
    end = start + seq
    q_chunk_t = q[:, :, start:end]
    pos = torch.arange(start, end)
    parts, parts_tt, added = [], [], []
    for c in (0, 1):
        programs_before = device.num_program_cache_entries()
        out, lse = run_prefill(
            device,
            q_chunk_t,
            caches.k[c],
            caches.v[c],
            caches.page_table_tt,
            start,
            program,
            fp32_acc,
            True,
            parity=c,
            keep_tt=parts_tt,
        )
        added.append(device.num_program_cache_entries() - programs_before)
        n_local = end // 2  # the op reads local keys [0, (start + S) / 2)
        g = torch.tensor([global_index(l, c) for l in range(n_local)])
        visible = g[None, :] <= pos[:, None]
        for h in range(NH):
            ref_out, ref_lse = ref_attention(
                q_chunk_t[0, h], caches.k[c].seqs[0, :n_local], caches.v[c].seqs[0, :n_local], scale, visible
            )
            check_rows(out[0, h], lse[0, h, :, 0], ref_out, ref_lse, f"start {start} chip {c} head {h}")
        parts.append((out, lse))

    (o_0, lse_0), (o_1, lse_1) = parts
    merged = merge(o_0, lse_0, o_1, lse_1)
    assert torch.isfinite(merged).all(), f"start {start}: merged output not finite"
    # The same merge with ttnn ops on device: lse [1, nh, S, 1] fp32 broadcasts against out [1, nh, S, 128].
    merged_tt = ttnn.to_torch(merge_tt(*parts_tt[0], *parts_tt[1])).float()
    visible = torch.arange(end)[None, :] <= pos[:, None]
    for h in range(NH):
        ref_out, _ = ref_attention(q_chunk_t[0, h], caches.k_stored[:end], caches.v_stored[:end], scale, visible)
        passing, message = comp_pcc(ref_out, merged[0, h], 0.999)
        assert passing, f"start {start} head {h}: merged pair vs full attention: {message}"
        passing, message = comp_pcc(ref_out, merged_tt[0, h], 0.999)
        assert passing, f"start {start} head {h}: on-device merge vs full attention: {message}"
    return added


def check_pair_split_lse_off(device, caches, q, start, seq, c, program, fp32_acc):
    """Without return_lse the pair-split output is the same bit for bit (rows with no key are 0 too)."""
    q_chunk_t = q[:, :, start : start + seq]
    args = (device, q_chunk_t, caches.k[c], caches.v[c], caches.page_table_tt, start, program, fp32_acc)
    plain = run_prefill(*args, False, parity=c)
    out, _ = run_prefill(*args, True, parity=c)
    assert torch.equal(out, plain), f"start {start} chip {c}: pair split output differs with return_lse"


@pytest.mark.parametrize("fp32_acc", [False, True], ids=["streaming", "fp32_dest"])
@pytest.mark.parametrize(
    "q_chunk, k_chunk, seq",
    [(128, 128, 256), (256, 256, 512), (64, 128, 256), (128, 256, 384)],
    ids=["c128", "c256", "q64_k128", "q128_k256"],
)
def test_chunked_mla_prefill_pair_split_merge(
    device, q_chunk, k_chunk, seq, fp32_acc, function_level_defaults, reset_seeds
):
    """Pair split: each chip's cache holds its half of every 128-token block; the two chips' chunked prefills (parity
    0 / 1, return_lse) each match the pair-split reference, and merged by the LSE weights they equal full causal
    attention over the global cache. Chunk starts are multiples of 128, including 0 (chip 1 sees no key for global
    positions 0..63: output 0, lse -1e30). Both chips and every start share one program."""
    n = 2048  # global tokens per user; each chip caches n / 2
    q = torch.randn(1, NH, n, DK)
    caches = PairSplitCaches(device, n)
    program = prefill_program(q_chunk, k_chunk)

    programs = None
    align = max(q_chunk, k_chunk)
    starts = [s for s in sorted({0, 128, 256, n - seq - 128, n - seq}) if s % align == 0]
    for start in starts:
        for c, added in enumerate(check_pair_split_start(device, caches, q, start, seq, program, fp32_acc)):
            if programs is None:
                programs = added  # the first call compiles the program
            else:
                assert added == 0, f"start {start} parity {c} added a program"

    for start, c in ((0, 1), (starts[-1], 0)):
        check_pair_split_lse_off(device, caches, q, start, seq, c, program, fp32_acc)


def q_chunks_per_core(seq, q_chunk, cores=64):
    """(fewest, most) Q chunks on a busy core of the chunked causal prefill's global Q scheduling
    (sdpa_program_factory.cpp): the NH * S / q_chunk chunks of one user are dealt to the cores in zigzag pairs of a
    head when each head has an even count, else one by one."""
    per_head = seq // q_chunk
    unit, units = (2, NH * per_head // 2) if per_head % 2 == 0 else (1, NH * per_head)
    base, extra = divmod(units, cores)
    return unit * max(base, 1), unit * (base + (1 if extra else 0))


@pytest.mark.parametrize("fp32_acc", [False, True], ids=["streaming", "fp32_dest"])
@pytest.mark.parametrize(
    "chunk, seq, starts",
    [(256, 2048, (0, 1024, 2048)), (128, 2048, (0, 2048)), (256, 4096, (0, 4096))],
    ids=["c256_s2048", "c128_s2048", "c256_s4096"],
)
def test_chunked_mla_prefill_pair_split_multi_chunk_per_core(
    device, chunk, seq, starts, fp32_acc, function_level_defaults, reset_seeds
):
    """MiMo's prefill shapes (attention._prefill_sdpa_program: q / k chunk 256 from S = 2048, 128 below): every busy
    core runs several Q chunks (c256_s2048: one zigzag pair of 8 chunks a head, 2 a core; c128_s2048: 4 a core;
    c256_s4096: 8 a core), so the LSE statistics CBs, the double-buffered Q and the pair-split row map hand over
    from one Q chunk to the next on a core. Both chips match the pair-split reference, their merge equals full
    attention at starts 0 and >= S, and return_lse=False gives the same output."""
    fewest, most = q_chunks_per_core(seq, chunk)
    assert fewest >= 2, (fewest, most)
    n = starts[-1] + seq  # global tokens per user; each chip caches n / 2
    q = torch.randn(1, NH, n, DK)
    caches = PairSplitCaches(device, n)
    program = prefill_program(chunk, chunk)
    for start in starts:
        check_pair_split_start(device, caches, q, start, seq, program, fp32_acc)
    check_pair_split_lse_off(device, caches, q, 0, seq, 1, program, fp32_acc)
    check_pair_split_lse_off(device, caches, q, starts[-1], seq, 0, program, fp32_acc)


def grouped_program(device, chunk, heads, splits):
    """The grouped-heads / K split chunked prefill program (MiMo's later chunks from 64K: attention.later_chunk_split)
    on the full grid."""
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
        q_heads_per_unit=heads,
        k_splits=splits,
    )


@pytest.mark.parametrize(
    "heads, splits, chunk",
    [(2, 1, 256), (4, 1, 128), (1, 3, 256), (2, 15, 256)],
    ids=["g2", "g4_c128", "s3", "g2s15"],  # four heads' accumulators at chunk 256 exceed L1
)
def test_chunked_mla_prefill_pair_split_grouped_ksplit(
    device, heads, splits, chunk, function_level_defaults, reset_seeds
):
    """The pair split on the grouped-heads / K split program (SDPAProgramConfig.q_heads_per_unit / k_splits): each
    chip matches the pair-split reference, LSE included (with a K split the merge forms it from the pieces'
    statistics), the merged pair equals full attention, every start shares the programs, and return_lse=False gives
    the same output. At start 0 chip 1's first 64 rows see no key (output 0, lse -1e30) and most K slices are empty."""
    seq, n = 512, 3072
    q = torch.randn(1, NH, n, DK)
    caches = PairSplitCaches(device, n)
    program = grouped_program(device, chunk, heads, splits)
    programs = None
    for start in (0, 256, n - seq):
        for c, added in enumerate(check_pair_split_start(device, caches, q, start, seq, program, False)):
            if programs is None:
                programs = added
            else:
                assert added == 0, f"start {start} parity {c} added a program"
    check_pair_split_lse_off(device, caches, q, 0, seq, 1, program, False)
    check_pair_split_lse_off(device, caches, q, n - seq, seq, 0, program, False)


def test_chunked_mla_prefill_pair_split_tight_page_table(device, function_level_defaults, reset_seeds, expect_error):
    """Pair split only reads the local keys below the chunk's end, (start + S) / 2 local tokens, which can be fewer
    than S: at start 0 with S = 2048 a 16-block table (1024 local tokens) is enough. A tight table gives the same
    output and LSE as the full table bit for bit; one below the start-0 amount (S / 2) is rejected. Table widths
    are multiples of 8 blocks (the flexible chunk start reads the page-table row in 32-byte units)."""
    seq, chunk, n = 2048, 256, 4096
    q = torch.randn(1, NH, n, DK)
    caches = PairSplitCaches(device, n)
    program = prefill_program(chunk, chunk)
    for start in (0, 1024):
        blocks = (start + seq) // 2 // BLOCK  # 16 at start 0: fewer local tokens than S
        tight_tt = caches.table_tt(device, blocks)
        q_chunk_t = q[:, :, start : start + seq]
        for c in (0, 1):
            full = run_prefill(
                device, q_chunk_t, caches.k[c], caches.v[c], caches.page_table_tt, start, program, False, True, parity=c
            )
            tight = run_prefill(
                device, q_chunk_t, caches.k[c], caches.v[c], tight_tt, start, program, False, True, parity=c
            )
            assert torch.equal(tight[0], full[0]), f"start {start} chip {c}: output differs with a tight page table"
            assert torch.equal(tight[1], full[1]), f"start {start} chip {c}: lse differs with a tight page table"

    too_small = caches.table_tt(device, seq // 2 // BLOCK - 8)
    with expect_error(RuntimeError, "Pair split: the page table must cover"):
        run_prefill(device, q[:, :, :seq], caches.k[0], caches.v[0], too_small, 0, program, False, True, parity=0)
