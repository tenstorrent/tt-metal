# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated repro for the trisc0 build failure of the Quasar matmul compute kernel at a large-N config.

The llama e2e reaches (after the fill_cache fix let it past prefill) a 1D-mcast matmul with a wide output:
    MatmulMultiCoreReuseMultiCast1DProgramConfig(grid=2x1, in0_block_w=4, out_subblock_h=1, out_subblock_w=4,
        out_block_h=1, out_block_w=84, per_core_M=1, per_core_N=84, mcast_in0=1, untilize_out=0)
Building it FATALs:
    Failed to generate binaries for bmm_large_block_zm_fused_bias_activation_metal2 (build.cpp:149)
    trisc0 build failed.
Root cause (from the kernel cache: trisck.o + trisc0.elf are produced, but the post-link .xip.elf step
fails): the pack_init additions that fix the pack-destination-switch bug (issue #58488) push trisc0 over a
size/relocation limit at this large config -- the same kernel builds fine at the small test_linear_small_grid
config. The e2e's mcast_in0 path is "correct by coincidence" without those pack_inits (cb_out aliases
cb_intermed0), so the fix is retained for asserts-on / non-aliasing configs and this large-N build limit is
tracked here.

FIXED: the pack-destination retarget (pack_reconfig_data_format + pack_init) is now emitted once via a
noinline qsr_pack_retarget() helper instead of inlined at each switch site, so trisc0 fits at per_core_N=84
(measured text+data ~8.2KB was ~40B over the ~8KB trisc L1 budget). This test reconstructs that exact
program config + matching shapes and PCC-checks the result, guarding against a regression of that overflow.
Inputs built via quasar.tilize (from_torch(TILE) faults on Quasar wide-short tensors).

Run (Quasar sim, 2-node emulator, SLOW dispatch):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_matmul_large_n_build.py
"""

import pytest
import torch
from loguru import logger

import ttnn

# Exact compile-time args of the failing e2e matmul.
GRID = (2, 1)  # compute_with_storage_grid_size = 2x1 (the 2-node emulator)
IN0_BLOCK_W = 4  # K tiles per block
PER_CORE_M = 1  # M tiles / core
PER_CORE_N = 84  # N tiles / core (the wide output that overflows trisc0)
OUT_SUBBLOCK_H = 1
OUT_SUBBLOCK_W = 4  # -> out_block_w / out_subblock_w = 21 subblocks unrolled
OUT_BLOCK_H = 1
OUT_BLOCK_W = 84

M = PER_CORE_M * 32  # 32
K = IN0_BLOCK_W * 32  # 128 (one K block)
N = PER_CORE_N * GRID[0] * 32  # mcast_in0: N split across the 2 cores -> 168 tiles = 5376


def _tile_bf16_dram(t_bf16, mesh_device):
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    qt = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qt or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.timeout(3600)
def test_matmul_large_n_build(mesh_device):
    """Build the exact large-N 1D-mcast matmul in isolation and check PCC. Guards against a regression of the
    trisc0 code-size overflow: bmm_large_block_zm_fused_bias_activation_metal2's pack-destination retarget
    (pack_reconfig_data_format + pack_init) is emitted once via the noinline qsr_pack_retarget helper instead
    of inlined at each switch site, which is what makes trisc0 fit at per_core_N=84."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    at = _tile_bf16_dram(a, mesh_device)
    bt = _tile_bf16_dram(b, mesh_device)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(GRID[0], GRID[1]),
        in0_block_w=IN0_BLOCK_W,
        out_subblock_h=OUT_SUBBLOCK_H,
        out_subblock_w=OUT_SUBBLOCK_W,
        out_block_h=OUT_BLOCK_H,
        out_block_w=OUT_BLOCK_W,
        per_core_M=PER_CORE_M,
        per_core_N=PER_CORE_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )

    out = ttnn.matmul(at, bt, program_config=program_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)

    ref = a.float().reshape(M, K) @ b.float().reshape(K, N)
    pcc = _pcc(o, ref)
    logger.info(f"[matmul-large-n] out {tuple(o.shape)} finite={torch.isfinite(o).all().item()} PCC={pcc:.5f}")
    assert tuple(o.shape) == (1, 1, M, N), f"unexpected shape {tuple(o.shape)}"
    assert pcc > 0.99, f"matmul PCC {pcc}"
