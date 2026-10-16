# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the ``spec_multi_pos_tiles`` sanity and nightly tests."""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import is_wormhole_b0
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

BLOCK_SIZE = 64
HEAD_DIM = 256
NUM_KV_HEADS = 1
NUM_Q_HEADS = 6  # valid q heads per candidate; rows 6..31 of each Q tile are padding
TILE = 32
SCALE = 1.0 / (HEAD_DIM**0.5)

# Spec mode makes every Q-shaped CB T times larger (PNHt == T), so the (k_chunk_size,
# cores-per-head) budget shrinks sharply as T grows. These are the largest points that fit
# Blackhole L1 (1,461,248 B of CB space per core) at head_dim=256 — see the explicit L1
# guard in the program factory, which reports the numbers when a config does not fit.
#
#   T=4,  chunk=128, 64 cores/head -> 1,310,720 B
#   T=7,  chunk=64,   4 cores/head -> 1,329,152 B
#   T=11, chunk=32,   1 core /head -> 1,447,936 B
#
# The sweep below uses core counts chosen so the reference (B == T rows spread over the same
# grid) lands on the SAME cores-per-head as spec mode (B == 1). With an identical reduction
# tree the two paths agree bit-for-bit, which isolates the mode from flash-decode's bf16
# partial-sum ordering. ``test_spec_multi_pos_wide_reduction_split`` covers the other case,
# where the trees deliberately differ.
SPEC_CONFIG = {
    4: {"max_cores": 16, "k_chunk_size": 128},
    7: {"max_cores": 4, "k_chunk_size": 64},
    11: {"max_cores": 1, "k_chunk_size": 32},
}

# Wormhole has less CB space (1,393,440 B), so the T=11 point above does not fit there.
WORMHOLE_MAX_T = 7


def skip_if_exceeds_l1(T):
    if is_wormhole_b0() and T > WORMHOLE_MAX_T:
        pytest.skip(f"T={T} spec config does not fit Wormhole L1")


# ── Helpers ────────────────────────────────────────────────────────────────


def build_kv(seq_len, seed):
    """One logical user's K/V, bf16-rounded so the paged cache round-trips exactly."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(NUM_KV_HEADS, seq_len, HEAD_DIM, generator=g).bfloat16().float()
    v = torch.randn(NUM_KV_HEADS, seq_len, HEAD_DIM, generator=g).bfloat16().float()
    return k, v


def paged_layout(k, v, page_table_row):
    """Scatter the user's K/V into a paged buffer keyed by physical block id."""
    num_blocks = page_table_row.numel()
    paged_k = torch.zeros(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    paged_v = torch.zeros_like(paged_k)
    for virtual_block in range(num_blocks):
        physical_block = int(page_table_row[virtual_block])
        lo = virtual_block * BLOCK_SIZE
        hi = lo + BLOCK_SIZE
        paged_k[physical_block] = k[:, lo:hi, :]
        paged_v[physical_block] = v[:, lo:hi, :]
    return paged_k, paged_v


def torch_reference(q_heads, k, v, cur_pos):
    """fp32 causal reference: candidate j attends to KV positions [0, cur_pos[j]]."""
    T = q_heads.shape[0]
    out = torch.zeros(T, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32)
    for j in range(T):
        pos = int(cur_pos[j])
        k_j = k[0, : pos + 1, :].float()
        v_j = v[0, : pos + 1, :].float()
        scores = (q_heads[j].float() @ k_j.T) * SCALE
        out[j] = torch.softmax(scores, dim=-1) @ v_j
    return out


def program_config(device, max_cores, k_chunk_size):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=k_chunk_size,  # 0 -> dynamic k-chunk size, chosen in-kernel from cur_pos
        exp_approx_mode=False,
        max_cores_per_head_batch=max_cores,
    )


def config_for(device, T):
    cfg = SPEC_CONFIG[T]
    return program_config(device, cfg["max_cores"], cfg["k_chunk_size"])


def build_inputs(device, T, p, seq_len, seed):
    """Everything both calls share, plus the two call-specific Q / page-table tensors."""
    assert seq_len % BLOCK_SIZE == 0
    num_blocks = seq_len // BLOCK_SIZE
    cur_pos = torch.tensor([p + j for j in range(T)], dtype=torch.int32)
    assert int(cur_pos[-1]) < seq_len, "cur_pos must stay inside the cache"

    k, v = build_kv(seq_len, seed)
    g = torch.Generator().manual_seed(seed + 1)
    # Shuffled blocks: the two calls must resolve the same physical blocks through the
    # page table, not through a coincidentally identity mapping.
    page_row = torch.randperm(num_blocks, generator=g).to(torch.int32)
    paged_k, paged_v = paged_layout(k, v, page_row)

    # Q: one 32-row tile per candidate, rows 0..NUM_Q_HEADS-1 valid, the rest zero padding.
    # [1, T, 32, DH] and [1, 1, T*32, DH] are byte-identical once tilized, which is exactly
    # the equivalence the mode relies on.
    q_heads = torch.randn(T, NUM_Q_HEADS, HEAD_DIM, generator=g).bfloat16().float()
    q_batched = torch.zeros(1, T, TILE, HEAD_DIM)
    q_batched[0, :, :NUM_Q_HEADS, :] = q_heads

    return {
        "k": k,
        "v": v,
        "q_heads": q_heads,
        "cur_pos": cur_pos,
        "page_row": page_row,
        "k_tt": ttnn.Tensor(paged_k, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        "v_tt": ttnn.Tensor(paged_v, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        "cur_pos_tt": ttnn.Tensor(cur_pos, ttnn.int32).to(device),
        "q_batched": q_batched,
        "q_spec": q_batched.reshape(1, 1, T * TILE, HEAD_DIM),
    }


def run_reference(device, inp, T, program_config):
    """Today's mode: B == T pseudo-users, T identical (aliased) page-table rows."""
    page_table = inp["page_row"].unsqueeze(0).repeat(T, 1).contiguous()
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        ttnn.Tensor(inp["q_batched"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        inp["k_tt"],
        inp["v_tt"],
        page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
        cur_pos_tensor=inp["cur_pos_tt"],
        scale=SCALE,
        program_config=program_config,
    )
    torch_out = ttnn.to_torch(out)
    assert tuple(torch_out.shape) == (1, T, TILE, HEAD_DIM)
    return torch_out[0, :, :NUM_Q_HEADS, :].float()


def run_spec(device, inp, T, program_config):
    """Spec mode: the same Q bytes on ONE batch row, one page-table row."""
    page_table = inp["page_row"].unsqueeze(0).contiguous()
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        inp["k_tt"],
        inp["v_tt"],
        page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
        cur_pos_tensor=inp["cur_pos_tt"],
        scale=SCALE,
        program_config=program_config,
        spec_multi_pos_tiles=T,
    )
    torch_out = ttnn.to_torch(out)
    # Byte-identical layout to the B=T output [1, T, 32, DH].
    assert tuple(torch_out.shape) == (1, 1, T * TILE, HEAD_DIM)
    return torch_out.reshape(1, T, TILE, HEAD_DIM)[0, :, :NUM_Q_HEADS, :].float()


def straddles_chunk_boundary(T, p):
    """True when the T bounds span two k-chunks.

    This matters for how tightly the two paths can be expected to agree. Spec mode scans up
    to the LARGEST bound, so when the bounds straddle a boundary it scans one chunk more than
    the reference rows whose own bound sits in the lower chunk. That extra chunk is fully
    masked and contributes exactly zero, but it changes the chunk count, hence the
    chunk-to-core distribution, hence the order in which flash decode sums its bf16 partial
    results — so bit-equality is not available for these cases even with matched core counts.
    """
    chunk = SPEC_CONFIG[T]["k_chunk_size"]
    return (p // chunk) != ((p + T - 1) // chunk)


def check(device, T, p, seq_len, seed=0, program_config=None, pcc_ref_vs_spec=None):
    inp = build_inputs(device, T, p, seq_len, seed)
    program_config = program_config if program_config is not None else config_for(device, T)
    if pcc_ref_vs_spec is None:
        pcc_ref_vs_spec = 0.999 if straddles_chunk_boundary(T, p) else 0.9999

    ref = run_reference(device, inp, T, program_config)
    spec = run_spec(device, inp, T, program_config)
    torch_ref = torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    where = f"T={T}, p={p}, seq_len={seq_len}"

    # 1) The two device paths must agree. With a matched reduction tree and bounds inside one
    #    k-chunk they come out bit-identical; when the bounds straddle a chunk boundary the
    #    chunk counts differ and only bf16 partial-sum accuracy is available (see
    #    straddles_chunk_boundary).
    eq, msg = comp_pcc(ref, spec, pcc=pcc_ref_vs_spec)
    assert eq, f"spec vs batched ({where}): {msg}"
    assert torch.allclose(
        ref, spec, rtol=2e-2, atol=2e-2
    ), f"spec vs batched ({where}): max abs diff {(ref - spec).abs().max().item():.5f}"

    # 2) Both must match a plain fp32 reference that applies each candidate's bound
    #    independently — this is what catches an off-by-one or a dropped per-row bound.
    eq, msg = comp_pcc(torch_ref, spec, pcc=0.99)
    assert eq, f"spec vs torch ({where}): {msg}"
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"batched vs torch ({where}): {msg}"


def check_bit_exact(device, T, p, seq_len):
    """Matched reduction tree, bounds in one k-chunk: spec must equal batched bit-for-bit."""
    assert not straddles_chunk_boundary(T, p)
    inp = build_inputs(device, T, p, seq_len, seed=23)
    pc = config_for(device, T)
    ref = run_reference(device, inp, T, pc)
    spec = run_spec(device, inp, T, pc)
    assert torch.equal(ref, spec), (
        f"expected bit-identical output (T={T}, p={p}, seq_len={seq_len}); "
        f"max abs diff {(ref - spec).abs().max().item():.8f}"
    )


def minimal_spec_args(device, T=4, seq_len=512, p=100):
    inp = build_inputs(device, T, p, seq_len, seed=3)
    return inp, config_for(device, T)


# ── Batch groups (B >= 1) ──────────────────────────────────────────────────
#
# Everything above runs B == 1: one batch row carrying all T candidates. That row's CBs all
# scale with T, so L1 caps how many cores can reduce it (Tg=4 -> 64 cores/head, Tg=7 -> 4),
# and a single group cannot fill the grid past its own cap. B groups of Tg lift that: batch b
# owns cur_pos[b*Tg : (b+1)*Tg] and gets its own reduction group of cores, so B=2, Tg=4 runs 8
# candidates across ~2x the cores that one 8-tile row could ever use.
#
# The reference is unchanged — the legacy B == B*Tg pseudo-user call. The claim is the same
# one the B == 1 sweep makes, now per group: candidate b*Tg+j must come out exactly as legacy
# batch row b*Tg+j did.

GROUP_CHUNK = 128  # k_chunk_size for the group sweep: 4 tiles, the spec-mode dynamic cap


def cores_per_head(device, batch, max_cores):
    """The program factory's core split, mirrored (see 'Core Allocation' in the factory).

    Bit-equality between the spec call (B rows) and the legacy reference (B*Tg rows) is only
    available when both land on the same cores-per-head, since that fixes the flash-decode
    reduction tree and hence the order the bf16 partial results are summed.
    """
    grid = device.compute_with_storage_grid_size()
    available = grid.x * grid.y
    return max(1, min(available, max_cores * batch) // batch)


def run_spec_groups(device, inp, B, Tg, program_config):
    """Spec mode with B groups: the same Q bytes as [1, B, Tg*32, DH], B aliased page rows."""
    T = B * Tg
    assert inp["cur_pos"].numel() == T
    # [1, T, 32, DH] -> [1, B, Tg*32, DH] is a pure reshape, so the tilized bytes are
    # identical to the reference's Q — candidate b*Tg+j lands on batch b, row-tile j.
    q_spec = inp["q_batched"].reshape(1, B, Tg * TILE, HEAD_DIM)
    page_table = inp["page_row"].unsqueeze(0).repeat(B, 1).contiguous()
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        ttnn.Tensor(q_spec, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        inp["k_tt"],
        inp["v_tt"],
        page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
        cur_pos_tensor=inp["cur_pos_tt"],
        scale=SCALE,
        program_config=program_config,
        spec_multi_pos_tiles=Tg,
    )
    torch_out = ttnn.to_torch(out)
    assert tuple(torch_out.shape) == (1, B, Tg * TILE, HEAD_DIM)
    return torch_out.reshape(1, T, TILE, HEAD_DIM)[0, :, :NUM_Q_HEADS, :].float()


def group_straddles(B, Tg, p, chunk):
    """True when some group's Tg bounds span two k-chunks.

    Spec mode scans to the group's LARGEST bound, so a straddling group scans one chunk more
    than the reference rows whose own bound sits in the lower chunk. That chunk is fully
    masked and contributes zero, but it changes the chunk count and hence the chunk-to-core
    distribution — so bit-equality is off the table even with a matched tree.
    """
    return any((p + b * Tg) // chunk != (p + (b + 1) * Tg - 1) // chunk for b in range(B))


def check_groups(device, B, Tg, p, seq_len, max_cores, seed=0, k_chunk_size=GROUP_CHUNK, pcc=None):
    T = B * Tg
    torch.manual_seed(seed)
    inp = build_inputs(device, T, p, seq_len, seed)
    pc = program_config(device, max_cores=max_cores, k_chunk_size=k_chunk_size)
    matched_tree = cores_per_head(device, B, max_cores) == cores_per_head(device, T, max_cores)
    if pcc is None:
        pcc = 0.9999 if matched_tree and not group_straddles(B, Tg, p, k_chunk_size) else 0.999

    ref = run_reference(device, inp, T, pc)
    spec = run_spec_groups(device, inp, B, Tg, pc)
    torch_ref = torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    where = f"B={B}, Tg={Tg}, p={p}, seq_len={seq_len}, max_cores={max_cores}"

    eq, msg = comp_pcc(ref, spec, pcc=pcc)
    assert eq, f"spec groups vs batched ({where}): {msg}"
    assert torch.allclose(
        ref, spec, rtol=2e-2, atol=2e-2
    ), f"spec groups vs batched ({where}): max abs diff {(ref - spec).abs().max().item():.5f}"

    # The fp32 check is what catches a group picking up the wrong slice of cur_pos: a
    # candidate given its neighbour group's bound still matches the device reference's shape
    # and magnitude, but not the independently-bounded softmax.
    eq, msg = comp_pcc(torch_ref, spec, pcc=0.99)
    assert eq, f"spec groups vs torch ({where}): {msg}"
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"batched vs torch ({where}): {msg}"


def check_groups_bit_exact(device, seq_len, p):
    """B=2 spec vs B=8 batched on matched cores-per-head (max_cores=8): must be bit-identical."""
    B, Tg, max_cores = 2, 4, 8
    T = B * Tg
    if cores_per_head(device, B, max_cores) != cores_per_head(device, T, max_cores):
        pytest.skip("grid too small to give the spec and reference calls a matched reduction tree")
    assert not group_straddles(B, Tg, p, GROUP_CHUNK)

    torch.manual_seed(41)
    inp = build_inputs(device, T, p, seq_len, seed=43)
    pc = program_config(device, max_cores=max_cores, k_chunk_size=GROUP_CHUNK)
    ref = run_reference(device, inp, T, pc)
    spec = run_spec_groups(device, inp, B, Tg, pc)
    assert torch.equal(ref, spec), (
        f"expected bit-identical output (B={B}, Tg={Tg}, p={p}, seq_len={seq_len}); "
        f"max abs diff {(ref - spec).abs().max().item():.8f}"
    )
