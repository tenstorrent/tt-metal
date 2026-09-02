# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated test for the resnet50 FC matmul on the 2-core Quasar emulator (#48552).

The full e2e reaches the FC last (2048 -> 1000), where the old no-spill fc config aliased cb_out (per_core_N)
and cb_intermed0 (out_block_w) at DIFFERENT sizes -> "Aliased DFBs cb_out and cb_intermed0 different total sizes"
FATAL. The fix K-SPILLS the fc (in0_block_w << full K) so out_block_w can be the FULL per_core_N -> the two
aliased CBs match, while the weights stream per K-block. This test exercises the EXACT model path (fit_fc_grid +
ResnetLinear) at the fc dims so the fix can be checked in seconds instead of a ~30-min full-model run.

Run (K-spill matmul still DPRINT-masked):
  unset TT_METAL_LLK_ASSERTS
  TT_METAL_DPRINT_CORES=all TT_METAL_FORCE_JIT_COMPILE=1 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
  pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_fc_kspill.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import nearest_32
from models.demos.vision.classification.resnet50.quasar.tt.ttnn_functional_resnet50 import ResnetLinear, fit_fc_grid
from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_fc_kspill(mesh_device):
    """resnet50 fc (2048 -> 1000) via the real ResnetLinear/fit_fc_grid path. PASS => the fc alias-size FATAL is
    fixed and the K-spilled fc computes correctly."""
    device = mesh_device
    torch.manual_seed(0)

    K = 2048  # in features
    N = 1000  # classes
    N_pad = nearest_32(N)  # 1024
    k_tiles = K // TILE  # 64
    n_tiles = N_pad // TILE  # 32

    # torch golden: [1, K] @ [K, N] + bias
    act_torch = torch.randn((1, K), dtype=torch.bfloat16).float()
    w_torch = torch.randn((N, K), dtype=torch.bfloat16).float()  # torch Linear weight [out, in]
    b_torch = torch.randn((N,), dtype=torch.bfloat16).float()
    golden = torch.nn.functional.linear(act_torch, w_torch, b_torch)  # [1, N]

    # --- device tensors, matching the model's fc feed ---
    # activation: [1,1,M(=1 tile pad),K] WIDTH_SHARDED on the fc grid (single core on Quasar).
    gx, gy, num_cores, per_core_N, in0_block_w, out_block_w, out_subblock_w = fit_fc_grid(device, n_tiles, k_tiles)
    act_padded = torch.zeros((1, 1, TILE, K), dtype=torch.bfloat16).float()
    act_padded[0, 0, 0, :] = act_torch[0]
    act_core_grid = ttnn.CoreGrid(x=gx, y=gy)
    act_mem = ttnn.create_sharded_memory_config_(
        [TILE, K // num_cores],
        act_core_grid,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )
    tt_act = ttnn.from_torch(act_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device, act_mem)

    # weight [K, N_pad] (ttnn matmul act @ weight), bias [1,1,1,N_pad] -- in DRAM so the K-spill streams them.
    w_kn = torch.zeros((1, 1, K, N_pad), dtype=torch.bfloat16).float()
    w_kn[0, 0, :, :N] = w_torch.t()
    b_row = torch.zeros((1, 1, 1, N_pad), dtype=torch.bfloat16).float()
    b_row[0, 0, 0, :N] = b_torch
    tt_w = ttnn.from_torch(
        w_kn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_b = ttnn.from_torch(
        b_row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )
    print(
        f"  DIAG fc: K={K}(t{k_tiles}) N={N}->{N_pad}(t{n_tiles}) | grid=({gx},{gy}) num_cores={num_cores} "
        f"per_core_N={per_core_N} in0_block_w={in0_block_w} out_block_w={out_block_w} out_subblock_w={out_subblock_w} "
        f"num_k_blocks={k_tiles // in0_block_w}"
    )

    fc = ResnetLinear(
        tt_w,
        tt_b,
        ttnn.L1_MEMORY_CONFIG,
        {"ACTIVATIONS_DTYPE": ttnn.bfloat16},
        compute_config,
        matmul_grid=(gx, gy),
        per_core_N=per_core_N,
        in0_block_w=in0_block_w,
        out_block_w=out_block_w,
        out_subblock_w=out_subblock_w,
    )
    out = fc(tt_act)

    got = ttnn.to_torch(ttnn.from_device(out)).float().reshape(-1)[:N].reshape(1, N)
    assert_with_pcc(golden, got, pcc=0.98)
    dev_top1 = int(torch.argmax(got, dim=1).item())
    gold_top1 = int(torch.argmax(golden, dim=1).item())
    print(f"  fc K-spill PASSED (num_k_blocks={k_tiles // in0_block_w}) dev_top1={dev_top1} golden_top1={gold_top1}")
    assert dev_top1 == gold_top1, f"fc top-1 mismatch dev={dev_top1} golden={gold_top1}"
