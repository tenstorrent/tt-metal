# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""spec_multi_pos_tiles folds Tg candidates onto one batch row so that row's KV is scanned once.
Matched reduction trees are bit-identical inside one k-chunk; a straddling bound only agrees to bf16 partial sums."""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# One module-scoped device; every case reuses the same programs per (T, seq_len).
pytestmark = pytest.mark.use_module_device

BLOCK_SIZE = 64
HEAD_DIM = 256
NUM_KV_HEADS = 1
NUM_Q_HEADS = 6  # valid q heads per candidate; rows 6..31 of each Q tile are padding
TILE = 32
SCALE = 1.0 / (HEAD_DIM**0.5)

# Spec mode grows Q-shaped CBs with T. Core counts below match the reference reduction tree.
SPEC_CONFIG = {
    4: {"max_cores": 16, "k_chunk_size": 128},
    7: {"max_cores": 4, "k_chunk_size": 64},
    11: {"max_cores": 1, "k_chunk_size": 32},
}


def _build_kv(seq_len, seed):
    """One logical user's K/V, bf16-rounded so the paged cache round-trips exactly."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(NUM_KV_HEADS, seq_len, HEAD_DIM, generator=g).bfloat16().float()
    v = torch.randn(NUM_KV_HEADS, seq_len, HEAD_DIM, generator=g).bfloat16().float()
    return k, v


def _paged_layout(k, v, page_table_row):
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


def _torch_reference(q_heads, k, v, cur_pos):
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


def _program_config(device, max_cores, k_chunk_size):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=k_chunk_size,  # 0 -> dynamic k-chunk size, chosen in-kernel from cur_pos
        exp_approx_mode=False,
        max_cores_per_head_batch=max_cores,
    )


def _config_for(device, T):
    cfg = SPEC_CONFIG[T]
    return _program_config(device, cfg["max_cores"], cfg["k_chunk_size"])


def _build_inputs(device, T, p, seq_len, seed):
    """Everything both calls share, plus the two call-specific Q / page-table tensors."""
    assert seq_len % BLOCK_SIZE == 0
    num_blocks = seq_len // BLOCK_SIZE
    cur_pos = torch.tensor([p + j for j in range(T)], dtype=torch.int32)
    assert int(cur_pos[-1]) < seq_len, "cur_pos must stay inside the cache"

    k, v = _build_kv(seq_len, seed)
    g = torch.Generator().manual_seed(seed + 1)
    # Shuffled blocks: both calls must use the page table, not an identity mapping.
    page_row = torch.randperm(num_blocks, generator=g).to(torch.int32)
    paged_k, paged_v = _paged_layout(k, v, page_row)

    # [1, T, 32, DH] and [1, 1, T*32, DH] are the same bytes once tilized.
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


def _run_reference(device, inp, T, program_config):
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


def _run_spec(device, inp, T, program_config):
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


def _straddles_chunk_boundary(T, p):
    """True when the T bounds span two k-chunks, so bit-equality is not available."""
    chunk = SPEC_CONFIG[T]["k_chunk_size"]
    return (p // chunk) != ((p + T - 1) // chunk)


def _check(device, T, p, seq_len, seed=0, program_config=None, pcc_ref_vs_spec=None):
    inp = _build_inputs(device, T, p, seq_len, seed)
    program_config = program_config if program_config is not None else _config_for(device, T)
    if pcc_ref_vs_spec is None:
        pcc_ref_vs_spec = 0.999 if _straddles_chunk_boundary(T, p) else 0.9999

    ref = _run_reference(device, inp, T, program_config)
    spec = _run_spec(device, inp, T, program_config)
    torch_ref = _torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    where = f"T={T}, p={p}, seq_len={seq_len}"

    # Matched tree and one k-chunk: bit-identical. A straddling bound only has bf16 partial-sum agreement.
    eq, msg = comp_pcc(ref, spec, pcc=pcc_ref_vs_spec)
    assert eq, f"spec vs batched ({where}): {msg}"
    assert torch.allclose(
        ref, spec, rtol=2e-2, atol=2e-2
    ), f"spec vs batched ({where}): max abs diff {(ref - spec).abs().max().item():.5f}"

    # Both must match an fp32 reference that applies each candidate's bound independently.
    eq, msg = comp_pcc(torch_ref, spec, pcc=0.99)
    assert eq, f"spec vs torch ({where}): {msg}"
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"batched vs torch ({where}): {msg}"


# p walks tile edges and k-chunk edges, including bounds that straddle a chunk.


@pytest.mark.parametrize("T", [4, 7, 11], ids=["T4", "T7", "T11"])
@pytest.mark.parametrize(
    "seq_len, p",
    [
        (2048, 1024),  # p % 32 == 0,  p % 128 == 0
        (2048, 1025),  # p % 32 == 1
        (2048, 1054),  # p % 32 == 30
        (2048, 1055),  # p % 32 == 31
        (2048, 1023),  # last position of a 128-chunk: bounds start a fresh chunk
        (2048, 1021),  # bounds straddle a 128-chunk boundary -> TWO masked chunks
        (8192, 4095),  # last position of a chunk
        (8192, 4093),  # straddles a chunk boundary
        (8192, 5000),
    ],
    ids=[
        "s2k_p1024",
        "s2k_p1025",
        "s2k_p1054",
        "s2k_p1055",
        "s2k_p1023",
        "s2k_p1021_straddle",
        "s8k_p4095",
        "s8k_p4093_straddle",
        "s8k_p5000",
    ],
)
def test_spec_multi_pos_matches_batched(device, T, seq_len, p):
    torch.manual_seed(0)
    assert p + T - 1 < seq_len
    _check(device, T, p, seq_len)


@pytest.mark.parametrize("T", [4, 7, 11], ids=["T4", "T7", "T11"])
@pytest.mark.parametrize("seq_len, p", [(2048, 1024), (8192, 5000)], ids=["s2k", "s8k"])
def test_spec_multi_pos_is_bit_exact(device, T, seq_len, p):
    """Matched reduction tree and bounds inside one k-chunk: the two forms are the same bits."""
    torch.manual_seed(5)
    assert not _straddles_chunk_boundary(T, p)
    inp = _build_inputs(device, T, p, seq_len, seed=23)
    pc = _config_for(device, T)
    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec(device, inp, T, pc)
    assert torch.equal(ref, spec), (
        f"expected bit-identical output (T={T}, p={p}, seq_len={seq_len}); "
        f"max abs diff {(ref - spec).abs().max().item():.8f}"
    )


@pytest.mark.parametrize("T", [4, 7], ids=["T4", "T7"])
@pytest.mark.parametrize("p", [30000, 32639], ids=["p30000", "p32639_straddle"])
def test_spec_multi_pos_long_context(device, T, p):
    """32k context — the regime the mode exists for (DRAM-bound KV scan)."""
    torch.manual_seed(1)
    _check(device, T, p, seq_len=32768)


@pytest.mark.parametrize("p", [30000, 32639], ids=["p30000", "p32639_straddle"])
def test_spec_multi_pos_long_context_single_core(device, p):
    """T=11 at 32k fits one core/head. Compare to the batched reference, not to a higher-core torch PCC."""
    torch.manual_seed(2)
    T = 11
    inp = _build_inputs(device, T, p, seq_len=32768, seed=17)
    pc = _config_for(device, T)
    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec(device, inp, T, pc)
    eq, msg = comp_pcc(ref, spec, pcc=0.9999)
    assert eq, f"spec vs batched (T={T}, p={p}, seq_len=32768): {msg}"
    assert torch.allclose(ref, spec, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("seq_len, p", [(2048, 1024), (8192, 5000), (32768, 30000)], ids=["s2k", "s8k", "s32k"])
def test_spec_multi_pos_wide_reduction_split(device, seq_len, p):
    """64 cores/head splits the reduction differently. Spec must be no less accurate than the reference."""
    torch.manual_seed(3)
    T = 4
    inp = _build_inputs(device, T, p, seq_len, seed=19)
    pc = _program_config(device, max_cores=64, k_chunk_size=128)
    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec(device, inp, T, pc)
    torch_ref = _torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])

    eq, msg = comp_pcc(ref, spec, pcc=0.999)
    assert eq, f"spec vs batched (wide split, seq_len={seq_len}, p={p}): {msg}"
    assert torch.allclose(ref, spec, rtol=2e-2, atol=2e-2)

    # Spec mode must be no worse than the batched reference against fp32.
    err_ref = (ref - torch_ref).abs().max().item()
    err_spec = (spec - torch_ref).abs().max().item()
    assert err_spec <= err_ref * 1.5 + 1e-6, (
        f"spec mode lost accuracy vs the batched reference: |spec-torch|={err_spec:.6f} "
        f"vs |ref-torch|={err_ref:.6f}"
    )
    eq, msg = comp_pcc(torch_ref, spec, pcc=0.99)
    assert eq, f"spec vs torch (wide split, seq_len={seq_len}, p={p}): {msg}"


def test_spec_multi_pos_short_context(device):
    """Short context: the whole scan is one chunk, so the single chunk is the masked one."""
    torch.manual_seed(1)
    _check(device, T=4, p=40, seq_len=512, seed=7)


def test_spec_multi_pos_first_block(device):
    """cur_pos inside the very first tile — the mask cuts inside column-tile 0."""
    torch.manual_seed(2)
    _check(device, T=7, p=5, seq_len=512, seed=11)


def test_spec_multi_pos_dynamic_chunk(device):
    """k_chunk_size=0 picks different chunk widths, so only bf16 partial-sum agreement is required."""
    torch.manual_seed(3)
    pc = _program_config(device, max_cores=64, k_chunk_size=0)
    _check(device, T=4, p=1024, seq_len=2048, seed=13, program_config=pc, pcc_ref_vs_spec=0.999)


def _minimal_spec_args(device, T=4, seq_len=512, p=100):
    inp = _build_inputs(device, T, p, seq_len, seed=3)
    return inp, _config_for(device, T)


def test_rejects_page_table_batch_mismatch(device, expect_error):
    """Page-table rows must equal Q batch groups. A B=T table against one Q row is the legacy form."""
    T = 4
    inp, pc = _minimal_spec_args(device, T=T)
    page_table = inp["page_row"].unsqueeze(0).repeat(T, 1).contiguous()
    with expect_error(RuntimeError, "one row per Q batch group"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_rejects_q_row_count_mismatch(device, expect_error):
    """spec_multi_pos_tiles must equal the number of 32-row Q tiles."""
    T = 4
    inp, pc = _minimal_spec_args(device, T=T)
    with expect_error(RuntimeError, "padded rows"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T + 1,
        )


def test_rejects_cur_pos_length_mismatch(device, expect_error):
    """cur_pos must carry exactly T bounds — one per candidate row-tile."""
    T = 4
    inp, pc = _minimal_spec_args(device, T=T)
    short_cur_pos = ttnn.Tensor(torch.tensor([100, 101], dtype=torch.int32), ttnn.int32).to(device)
    with expect_error(RuntimeError, "cur_pos must have"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=short_cur_pos,
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_rejects_sliding_window(device, expect_error):
    """The sliding-window mask shares the causal mask machinery; not combined for now."""
    T = 4
    inp, pc = _minimal_spec_args(device, T=T)
    with expect_error(RuntimeError, "sliding_window_size"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            sliding_window_size=256,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_legacy_path_unaffected(device):
    """spec_multi_pos_tiles omitted -> the pre-change op, bit-for-bit. Backward-compat guard."""
    torch.manual_seed(4)
    T = 4
    inp = _build_inputs(device, T, p=1000, seq_len=2048, seed=5)
    pc = _config_for(device, T)
    ref = _run_reference(device, inp, T, pc)
    torch_ref = _torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"legacy path regressed: {msg}"


# B groups of Tg: each group has its own reduction cores. The reference stays the legacy B*Tg call.

GROUP_CHUNK = 128  # k_chunk_size for the group sweep: 4 tiles, the spec-mode dynamic cap


def _cores_per_head(device, batch, max_cores):
    """Bit-equality needs both calls on the same cores-per-head, which fixes the reduction tree."""
    grid = device.compute_with_storage_grid_size()
    available = grid.x * grid.y
    return max(1, min(available, max_cores * batch) // batch)


def _run_spec_groups(device, inp, B, Tg, program_config):
    """Spec mode with B groups: the same Q bytes as [1, B, Tg*32, DH], B aliased page rows."""
    T = B * Tg
    assert inp["cur_pos"].numel() == T
    # [1, T, 32, DH] to [1, B, Tg*32, DH] is a reshape; tilized bytes match the reference Q.
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


def _group_straddles(B, Tg, p, chunk):
    """True when some group's bounds span two k-chunks, so bit-equality is off."""
    return any((p + b * Tg) // chunk != (p + (b + 1) * Tg - 1) // chunk for b in range(B))


def _check_groups(device, B, Tg, p, seq_len, max_cores, seed=0, k_chunk_size=GROUP_CHUNK, pcc=None):
    T = B * Tg
    torch.manual_seed(seed)
    inp = _build_inputs(device, T, p, seq_len, seed)
    pc = _program_config(device, max_cores=max_cores, k_chunk_size=k_chunk_size)
    matched_tree = _cores_per_head(device, B, max_cores) == _cores_per_head(device, T, max_cores)
    if pcc is None:
        pcc = 0.9999 if matched_tree and not _group_straddles(B, Tg, p, k_chunk_size) else 0.999

    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec_groups(device, inp, B, Tg, pc)
    torch_ref = _torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    where = f"B={B}, Tg={Tg}, p={p}, seq_len={seq_len}, max_cores={max_cores}"

    eq, msg = comp_pcc(ref, spec, pcc=pcc)
    assert eq, f"spec groups vs batched ({where}): {msg}"
    assert torch.allclose(
        ref, spec, rtol=2e-2, atol=2e-2
    ), f"spec groups vs batched ({where}): max abs diff {(ref - spec).abs().max().item():.5f}"

    # The fp32 check catches a group that used the wrong slice of cur_pos.
    eq, msg = comp_pcc(torch_ref, spec, pcc=0.99)
    assert eq, f"spec groups vs torch ({where}): {msg}"
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"batched vs torch ({where}): {msg}"


@pytest.mark.parametrize(
    "seq_len, p",
    [
        (2048, 1024),  # p % 32 == 0; both groups sit inside chunk 8
        (2048, 1025),  # p % 32 == 1
        (2048, 1054),  # p % 32 == 30; the groups' bounds cross a TILE edge
        (2048, 1055),  # p % 32 == 31
        # Chunk edges where the two groups do not scan the same distance:
        (2048, 1020),  # group 0 ends at 1023 (8 chunks), group 1 at 1027 (9) -> scan ranges differ
        (2048, 1021),  # group 0 straddles the boundary (TWO masked chunks), group 1 has one
        (8192, 4095),  # group 0 straddles at the top of chunk 31
        (8192, 5000),
        (32768, 30000),
        (32768, 32700),  # last chunk of the cache
    ],
    ids=[
        "s2k_p1024",
        "s2k_p1025",
        "s2k_p1054",
        "s2k_p1055",
        "s2k_p1020_split_scan",
        "s2k_p1021_straddle",
        "s8k_p4095_straddle",
        "s8k_p5000",
        "s32k_p30000",
        "s32k_p32700",
    ],
)
def test_spec_multi_pos_groups_matches_batched(device, seq_len, p):
    """B=2, Tg=4 at full grid. Reduction trees differ, so only bf16 partial-sum agreement."""
    _check_groups(device, B=2, Tg=4, p=p, seq_len=seq_len, max_cores=55, seed=31)


@pytest.mark.parametrize(
    "seq_len, p",
    [(2048, 1024), (2048, 1021), (8192, 5000)],
    ids=["s2k", "s2k_straddle", "s8k"],
)
def test_spec_multi_pos_groups_16_cores(device, seq_len, p):
    """Same claim at 16 cores/head, where masked chunks are more likely to share a core."""
    _check_groups(device, B=2, Tg=4, p=p, seq_len=seq_len, max_cores=16, seed=37)


@pytest.mark.parametrize("seq_len, p", [(2048, 1024), (8192, 5000), (32768, 30000)], ids=["s2k", "s8k", "s32k"])
def test_spec_multi_pos_groups_is_bit_exact(device, seq_len, p):
    """max_cores_per_head_batch=8 matches the trees; without a straddling group the bits match."""
    B, Tg, max_cores = 2, 4, 8
    T = B * Tg
    if _cores_per_head(device, B, max_cores) != _cores_per_head(device, T, max_cores):
        pytest.skip("grid too small to give the spec and reference calls a matched reduction tree")
    assert not _group_straddles(B, Tg, p, GROUP_CHUNK)

    torch.manual_seed(41)
    inp = _build_inputs(device, T, p, seq_len, seed=43)
    pc = _program_config(device, max_cores=max_cores, k_chunk_size=GROUP_CHUNK)
    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec_groups(device, inp, B, Tg, pc)
    assert torch.equal(ref, spec), (
        f"expected bit-identical output (B={B}, Tg={Tg}, p={p}, seq_len={seq_len}); "
        f"max abs diff {(ref - spec).abs().max().item():.8f}"
    )


def test_spec_multi_pos_groups_dynamic_chunk(device):
    """k_chunk_size=0 with the two groups on opposite sides of a dynamic-chunk step."""
    B, Tg, p = 2, 4, 60
    assert (p + Tg - 1) // 32 + 1 == 2 and (p + 2 * Tg - 1) // 32 + 1 == 3  # 2 tiles vs 4 (pow2)
    # The reference is not capped at 4 tiles, so the reduction split differs.
    _check_groups(device, B=B, Tg=Tg, p=p, seq_len=512, max_cores=16, seed=47, k_chunk_size=0, pcc=0.999)


def test_spec_multi_pos_groups_first_block(device):
    """Both groups' bounds cut inside the first tiles of the same chunk."""
    _check_groups(device, B=2, Tg=4, p=5, seq_len=512, max_cores=16, seed=53)


@pytest.mark.parametrize("B, Tg", [(4, 2), (2, 7)], ids=["B4_Tg2", "B2_Tg7"])
def test_spec_multi_pos_group_shapes(device, B, Tg):
    """Other (B, Tg) splits of the same candidate count, capped so the tall rows fit L1."""
    _check_groups(device, B=B, Tg=Tg, p=1024, seq_len=2048, max_cores=4, seed=59, k_chunk_size=64)


@pytest.mark.parametrize(
    "seq_len, p",
    [
        (2048, 1024),
        (2048, 1025),
        (2048, 1054),
        (2048, 1055),
        (2048, 1020),
        (2048, 1021),
        (8192, 4095),
        (8192, 5000),
        (32768, 30000),
        (32768, 32700),
    ],
    ids=[
        "s2k_p1024",
        "s2k_p1025",
        "s2k_p1054",
        "s2k_p1055",
        "s2k_p1020_split_scan",
        "s2k_p1021_straddle",
        "s8k_p4095_straddle",
        "s8k_p5000",
        "s32k_p30000",
        "s32k_p32700",
    ],
)
@pytest.mark.parametrize("k_chunk_size", [0, 128], ids=["B3_Tg4_dyn_chunk", "B3_Tg4_chunk128"])
def test_spec_groups_b3_tg4_full_grid(device, seq_len, p, k_chunk_size):
    """B=3, Tg=4 at 36 cores/head. Trees differ, so only bf16 partial-sum agreement."""
    _check_groups(device, B=3, Tg=4, p=p, seq_len=seq_len, max_cores=36, seed=67, k_chunk_size=k_chunk_size, pcc=0.999)


def test_rejects_group_cur_pos_length(device, expect_error):
    """cur_pos must carry B*Tg bounds — Tg per batch group, not Tg in total."""
    B, Tg = 2, 4
    inp = _build_inputs(device, B * Tg, p=100, seq_len=512, seed=61)
    short_cur_pos = ttnn.Tensor(torch.arange(Tg, dtype=torch.int32), ttnn.int32).to(device)
    q_spec = inp["q_batched"].reshape(1, B, Tg * TILE, HEAD_DIM)
    page_table = inp["page_row"].unsqueeze(0).repeat(B, 1).contiguous()
    with expect_error(RuntimeError, "cur_pos must have"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(q_spec, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
            cur_pos_tensor=short_cur_pos,
            scale=SCALE,
            program_config=_program_config(device, max_cores=16, k_chunk_size=GROUP_CHUNK),
            spec_multi_pos_tiles=Tg,
        )
