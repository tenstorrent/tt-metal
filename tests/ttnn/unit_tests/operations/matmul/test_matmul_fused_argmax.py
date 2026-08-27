# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.matmul(..., fused_argmax_partials=...): the opt-in fused
greedy-argmax epilogue on the Blackhole DRAM-sharded decode matmul (LM-head
path). TRISC2 scans the freshly packed bf16 logit tiles in the pack shadow
and each DRAM-bank worker emits a 32 x (global_index, bf16_value_bits)
partials page; the final reduce is num_banks compares per row on host.

Gating: the fused epilogue kernel JIT-compiles with the in-tree opt-in
(ComputeConfigDescriptor::enable_trisc2_rvv, which adds zve32f to the TRISC2
compile). Compile-side coverage runs device-free in CI
(tests/tt_metal/tt_metal/jit_build/test_trisc2_rvv.cpp); these tests execute
on silicon, so they stay behind TT_ENABLE_RVV_TESTS=1 until a Blackhole
runner picks them up; otherwise they skip.
"""

import math
import os

import numpy as np
import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="matmul fused_argmax_partials is Blackhole-only (TRISC2 Zve32f)"),
    pytest.mark.skipif(
        os.environ.get("TT_ENABLE_RVV_TESTS") != "1",
        reason="RVV device-test gate: set TT_ENABLE_RVV_TESTS=1 to run the fused-argmax tests on a Blackhole device",
    ),
]

TILE = 32


def _bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16)


def _mono(x: np.ndarray) -> np.ndarray:
    """Monotone image of the bfloat16_greater sign-magnitude total order."""
    x = x.astype(np.uint16)
    return np.where(x & 0x8000, x ^ 0xFFFF, x | 0x8000).astype(np.uint16)


_MONO_NEG_INF = int(_mono(np.array([0xFF80], dtype=np.uint16))[0])  # 0x007F


def _reference_argmax_row(bits_row: np.ndarray):
    """Incumbent semantics: bfloat16_greater order, smallest index, -inf init."""
    m = _mono(bits_row)
    idx = int(m.argmax())  # first occurrence of the max
    if int(m[idx]) <= _MONO_NEG_INF:
        return 0, 0xFF80
    return idx, int(bits_row[idx])


def _combine_partials(partials: np.ndarray, valid_rows: int, shard_w_tiles: int, valid_n_tiles: int):
    """Host-side final reduce: num_banks (idx, val) pairs -> 1 per row.
    Ascending bank order + first-max keeps the lowest global index on ties."""
    num_banks = partials.shape[0]
    nvalid = min(num_banks, (valid_n_tiles + shard_w_tiles - 1) // shard_w_tiles)
    idx = partials[:nvalid, 0::2].astype(np.int64)  # [w, 32]
    raw = (partials[:nvalid, 1::2] & 0xFFFF).astype(np.uint16)  # [w, 32]
    m = _mono(raw)
    w_best = np.argmax(m, axis=0)  # first max = lowest bank
    out = []
    for r in range(valid_rows):
        if int(m[w_best[r], r]) <= _MONO_NEG_INF:
            out.append((0, 0xFF80))
        else:
            out.append((int(idx[w_best[r], r]), int(raw[w_best[r], r])))
    return out


def _find_largest_divisor(n, max_divisor=8):
    for d in range(max_divisor, 0, -1):
        if n % d == 0:
            return d
    return 1


def _lm_head_storage_grid(k):
    """Mirror model_config's lm_head_core_grid derivation (8x8 down)."""
    rows, cols = 8, 8
    while k % (TILE * rows * cols) != 0:
        rows -= 1
        if rows == 0:
            cols -= 1
            rows = 8
            assert cols > 0
    return ttnn.CoreGrid(y=rows, x=cols)


def _dram_sharded_setup(device, torch_in0, torch_w, in1_dtype):
    """Production DRAM-sharded LM-head matmul operands + config
    (models/tt_transformers lm_head semantics)."""
    k = torch_in0.shape[-1]
    v = torch_w.shape[-1]
    assert torch_w.shape[-2] == k
    nb = device.dram_grid_size().x
    n_padded = math.ceil(v / (TILE * nb)) * (TILE * nb)
    grid = _lm_head_storage_grid(k)
    nc = grid.num_cores

    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, TILE, k),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in0_t = ttnn.from_torch(
        torch_in0.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem
    )

    in1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
    in1_spec = ttnn.ShardSpec(in1_grid, (k, n_padded // nb), ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_spec)
    in1_t = ttnn.from_torch(
        torch_w.bfloat16(), dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mem
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_find_largest_divisor(k // TILE // nc),
        per_core_M=1,
        per_core_N=math.ceil(v / (TILE * nc)),
        fused_activation=None,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2 if in1_dtype == ttnn.bfloat16 else ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    out_mem = ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1)
    return dict(
        in0_t=in0_t,
        in1_t=in1_t,
        program_config=program_config,
        compute_config=compute_config,
        out_mem=out_mem,
        nb=nb,
        shard_w_tiles=n_padded // TILE // nb,
        valid_n_tiles=v // TILE,
        valid_rows=torch_in0.shape[-2],
    )


def _fresh_partials(device, nb):
    return ttnn.from_torch(
        torch.zeros(1, 1, nb, 64, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_fused(device, torch_in0, torch_w, in1_dtype):
    """Run plain and fused matmul; return (plain bits, fused bits, combined
    (idx, val) per row, setup dict)."""
    s = _dram_sharded_setup(device, torch_in0, torch_w, in1_dtype)
    kwargs = dict(
        program_config=s["program_config"],
        memory_config=s["out_mem"],
        dtype=ttnn.bfloat16,
        compute_kernel_config=s["compute_config"],
    )
    plain = ttnn.matmul(s["in0_t"], s["in1_t"], **kwargs)
    partials = _fresh_partials(device, s["nb"])
    fused = ttnn.matmul(s["in0_t"], s["in1_t"], fused_argmax_partials=partials, **kwargs)

    plain_bits = _bf16_bits(ttnn.to_torch(plain))
    fused_bits = _bf16_bits(ttnn.to_torch(fused))
    # Device truth at logical extent (defensive 4D slice like the campaign harness).
    v = torch_w.shape[-1]
    logits_bits = _bf16_bits(ttnn.to_torch(fused)[0, 0, : s["valid_rows"], :v])
    part = ttnn.to_torch(partials).numpy().astype(np.uint32).reshape(s["nb"], 64)
    combined = _combine_partials(part, s["valid_rows"], s["shard_w_tiles"], s["valid_n_tiles"])
    return plain_bits, fused_bits, logits_bits, combined, s


def _check_case(device, torch_in0, torch_w, in1_dtype):
    plain_bits, fused_bits, logits, combined, s = _run_fused(device, torch_in0, torch_w, in1_dtype)

    # 1. Logits byte-identical with the epilogue on vs off.
    assert np.array_equal(plain_bits, fused_bits), "fused epilogue changed the logits bytes"

    # 2. Combined (idx, value-bits) bit-exact against the incumbent-semantics
    # host reference computed from the same device logits (valid region only).
    rows = s["valid_rows"]
    for r in range(rows):
        ref_idx, ref_val = _reference_argmax_row(logits[r])
        got_idx, got_val = combined[r]
        assert (got_idx, got_val) == (
            ref_idx,
            ref_val,
        ), f"row {r}: got (idx={got_idx}, val={got_val:#06x}), want (idx={ref_idx}, val={ref_val:#06x})"


K = 1024


@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
@pytest.mark.parametrize("rows", [1, 32], ids=["b1", "b32"])
def test_matmul_fused_argmax_random(device, in1_dtype, rows):
    torch.manual_seed(1000 + rows)
    v = 4096
    torch_in0 = torch.randn(1, 1, rows, K) * 0.5
    torch_w = torch.randn(1, 1, K, v) * 0.5
    _check_case(device, torch_in0, torch_w, in1_dtype)


def test_matmul_fused_argmax_planted_and_ties(device):
    """Planted max columns at worker seams and cross-worker ties: the lowest
    GLOBAL index must win, matching the incumbent rule across cores."""
    torch.manual_seed(7)
    v = 4096
    nb_cols = v // 8  # columns per DRAM-bank worker at nb=8
    torch_in0 = torch.randn(1, 1, 1, K) * 0.1
    torch_w = torch.randn(1, 1, K, v) * 0.1
    boost = torch_in0.flatten() * 4.0
    # A strong column right at the worker 3 / worker 4 seam, and an identical
    # copy in worker 6: worker-3's (lower global index) must win the combine.
    for col in (4 * nb_cols - 1, 4 * nb_cols, 6 * nb_cols + 17):
        torch_w[0, 0, :, col] = boost
    _check_case(device, torch_in0, torch_w, ttnn.bfloat16)


def test_matmul_fused_argmax_padded_vocab_no_leak(device):
    """V not a multiple of TILE*num_banks: bank-pad tiles are zero-weight, so
    with all-negative valid logits any scan leak past the logical width would
    win with 0.0. The scan must stop at the logical width."""
    torch.manual_seed(11)
    v = 2080  # 65 tiles over 8 banks -> last worker holds valid + pad tiles
    torch_in0 = torch.rand(1, 1, 1, K) * 0.5 + 0.1  # positive
    torch_w = -(torch.rand(1, 1, K, v) * 0.5 + 0.1)  # negative -> all logits < 0
    _check_case(device, torch_in0, torch_w, ttnn.bfloat16)


def test_matmul_fused_argmax_off_path_unchanged(device):
    """Without the flag the op must behave exactly as before (and produce the
    same bytes as with the flag, which test cases above already assert)."""
    torch.manual_seed(3)
    v = 4096
    torch_in0 = torch.randn(1, 1, 1, K) * 0.5
    torch_w = torch.randn(1, 1, K, v) * 0.5
    s = _dram_sharded_setup(device, torch_in0, torch_w, ttnn.bfloat16)
    kwargs = dict(
        program_config=s["program_config"],
        memory_config=s["out_mem"],
        dtype=ttnn.bfloat16,
        compute_kernel_config=s["compute_config"],
    )
    out_a = ttnn.matmul(s["in0_t"], s["in1_t"], **kwargs)
    out_b = ttnn.matmul(s["in0_t"], s["in1_t"], **kwargs)
    assert np.array_equal(_bf16_bits(ttnn.to_torch(out_a)), _bf16_bits(ttnn.to_torch(out_b)))
