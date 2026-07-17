# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the resnet50/quasar final fully-connected (fc) LINEAR op.

This isolates the single call the model makes at the end of resnet50.run():

    x = self.fc(x)   # ttnn.experimental.quasar.linear(act, weight, bias=..., program_config=...)

so the LLK team can test / fix the fc matmul in isolation with a PCC check, without running
the whole network.

What is reproduced (verbatim from ttnn_functional_resnet50.py: ResnetLinear + fit_fc_grid + run()):
  * fc dims: ResNet50 fc is K=2048 -> N=1000. N is tile-padded to 1024 (32 tiles); K=2048 = 64 tiles.
  * The 1D-mcast program config: ttnn._ttnn.operations.experimental.quasar.
        MatmulMultiCoreReuseMultiCast1DProgramConfig(out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=..., fuse_batch=True, mcast_in0=True, in0_block_w=...).
  * The activation is WIDTH_SHARDED on the SAME rectangular grid as the matmul (mcast_in0 requires
    the input sharding to match the matmul grid). Grid / per_core_N / in0_block_w are derived from the
    device via fit_fc_grid (stock 8x4=32 grid, per_core_N=1, in0_block_w=2 on a full Quasar part; a
    smaller rectangle on the 2x3 emulator / craq-sim grid).

IMPORTANT PATH NOTE:
  The 2D-mcast matmul path (MatmulMultiCoreReuseMultiCastProgramConfig) currently HANGS on Quasar
  (LLK dest-sync; those kernels are no-op'd). This fc uses the 1D-mcast config
  (MatmulMultiCoreReuseMultiCast1DProgramConfig, mcast_in0=True) which is a DIFFERENT kernel path.
  This test exercises that 1D path as-is; if it also hangs / mismatches, that is a distinct LLK bug
  from the 2D-mcast one and should be reported separately.

Run (craq-sim, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_linear.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import nearest_32
from tests.ttnn.utils_for_testing import assert_with_pcc


def _fit_fc_grid(device, n_tiles, k_tiles):
    """Inlined copy of ttnn_functional_resnet50.fit_fc_grid (kept local so this op test stays
    self-contained). Pick the largest rectangular core grid that fits the device AND evenly tiles
    the N output dim; raise per_core_N so every N tile is covered. Returns
    (grid_x, grid_y, num_cores, per_core_N, in0_block_w)."""
    grid = device.compute_with_storage_grid_size()
    best_gx, best_gy, best_nc = 1, 1, 1
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            nc = gx * gy
            if n_tiles % nc == 0 and nc > best_nc:
                best_gx, best_gy, best_nc = gx, gy, nc
    per_core_N = n_tiles // best_nc
    kt_per_core = k_tiles // best_nc  # best_nc | n_tiles | k_tiles, so exact
    in0_block_w = 2 if kt_per_core % 2 == 0 else kt_per_core
    return best_gx, best_gy, best_nc, per_core_N, in0_block_w


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_resnet50_fc_linear(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # resnet50 fc dims. N=1000 is tile-padded to 1024 (32 tiles); K=2048 (64 tiles). We build the
    # weight/bias at the padded width and compare all 1024 columns (the model trims to 1000 later via
    # untilize_with_unpadding, which is a separate layout op not under test here).
    K = 2048  # in features
    N_padded = 1024  # 1000 padded up to 32-tile-multiple
    M = 32  # one tile of rows (batch is fused into a single tile row; padded from batch=1)
    n_tiles = N_padded // 32  # 32
    k_tiles = K // 32  # 64

    fc_gx, fc_gy, fc_num_cores, per_core_N, in0_block_w = _fit_fc_grid(device, n_tiles=n_tiles, k_tiles=k_tiles)
    fc_matmul_grid = (fc_gx, fc_gy)

    # --- 1D-mcast program config, verbatim from ResnetLinear ---
    matmul_config = ttnn._ttnn.operations.experimental.quasar.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=fc_matmul_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # --- torch operands ---
    act_torch = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    # ttnn.linear computes act @ weight, so the weight is stored [K, N] (not the torch.nn.Linear
    # [N, K] orientation). Build it directly at [1, 1, K, N_padded].
    weight_torch = torch.randn((1, 1, K, N_padded), dtype=torch.bfloat16)
    bias_torch = torch.randn((1, 1, 1, N_padded), dtype=torch.bfloat16)

    # golden: act @ weight + bias  (broadcast bias over the M rows)
    golden = torch.matmul(act_torch.float(), weight_torch.float()) + bias_torch.float()  # [1,1,M,N_padded]

    # --- ttnn operands ---
    # Activation: WIDTH_SHARDED on the fc grid (mcast_in0 requires input sharding == matmul grid).
    fc_core_grid = ttnn.CoreGrid(x=fc_gx, y=fc_gy)
    act_mem_config = ttnn.create_sharded_memory_config_(
        [nearest_32(M), K // fc_num_cores],
        fc_core_grid,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )
    act = ttnn.from_torch(act_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    act = act.to(device, act_mem_config)

    # Weight + bias: interleaved on device (the model does ttnn...to_device on the preprocessed tensors).
    weight = ttnn.from_torch(
        weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bias = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # --- the exact fc call ---
    out = ttnn.experimental.quasar.linear(
        act,
        weight,
        bias=bias,
        program_config=matmul_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )

    got = ttnn.to_torch(out).float()
    assert tuple(got.shape) == (1, 1, M, N_padded), got.shape
    # bf16 + LoFi over a K=2048 reduction is noisy -> 0.98 (raise toward 0.99 once LLK is happy).
    assert_with_pcc(golden, got, pcc=0.98)
