# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated test for the resnet50 FC matmul on the 2-core Quasar emulator (#48552).

The full e2e reaches the FC last (2048 -> 1000), where the old no-spill fc config aliased cb_out (per_core_N)
and cb_intermed0 (out_block_w) at DIFFERENT sizes -> "Aliased DFBs cb_out and cb_intermed0 different total sizes"
FATAL. The fix K-SPILLS the fc (in0_block_w << full K) so out_block_w can be the FULL per_core_N -> the two
aliased CBs match, while the weights stream per K-block. This test exercises that config at the fc dims so the fix
can be checked in seconds instead of a ~30-min full-model run. The model now runs the fc with no program config on
Quasar (the unified matmul); no_program_config runs the same fc that way.

Run (K-spill matmul still DPRINT-masked):
  unset TT_METAL_LLK_ASSERTS
  TT_METAL_DPRINT_CORES=all TT_METAL_FORCE_JIT_COMPILE=1 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
  pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_fc_kspill.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_quasar, nearest_32
from models.demos.vision.classification.resnet50.quasar.tt.ttnn_functional_resnet50 import fit_fc_grid
from tests.ttnn.utils_for_testing import assert_allclose, assert_with_pcc

TILE = 32


def _kspill_fc_config(device, n_tiles, k_tiles):
    """The fc's 1D-mcast config as the model built it before the unified matmul: on Quasar one core, K-spilled so
    out_block_w is the full per_core_N; elsewhere fit_fc_grid's rectangle, as ResnetLinear still builds it.
    Returns (grid_x, grid_y, num_cores, program_config)."""
    if is_quasar():
        gx, gy, num_cores, per_core_N = 1, 1, 1, n_tiles
        in0_block_w = max(b for b in range(1, 9) if k_tiles % b == 0)
        out_subblock_w = max(w for w in range(1, 9) if per_core_N % w == 0)
        out_block = {"out_block_h": 1, "out_block_w": per_core_N}
    else:
        gx, gy, num_cores, per_core_N, in0_block_w = fit_fc_grid(device, n_tiles, k_tiles)
        out_subblock_w = 1
        out_block = {}
    program_config = ttnn._ttnn.operations.experimental.quasar.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        **out_block,
    )
    return gx, gy, num_cores, program_config


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("use_program_config", [True, False], ids=["kspill_1d", "no_program_config"])
def test_quasar_fc_kspill(mesh_device, use_program_config):
    """resnet50 fc (2048 -> 1000) with the K-spilled 1D-mcast config, or with no program config (the unified matmul
    on Quasar). PASS => the fc alias-size FATAL is fixed and the fc computes correctly."""
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
    if use_program_config:
        gx, gy, num_cores, program_config = _kspill_fc_config(device, n_tiles, k_tiles)
    else:
        gx, gy, num_cores, _, _ = fit_fc_grid(device, n_tiles, k_tiles)
        program_config = None
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
    print(f"  DIAG fc: K={K}(t{k_tiles}) N={N}->{N_pad}(t{n_tiles}) | grid=({gx},{gy}) | {program_config}")

    out = ttnn.experimental.quasar.linear(
        tt_act,
        tt_w,
        bias=tt_b,
        program_config=program_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config,
    )

    got = ttnn.to_torch(ttnn.from_device(out)).float().reshape(-1)[:N].reshape(1, N)
    assert_with_pcc(golden, got, pcc=0.98)
    if not use_program_config:
        # bf16 partial sums over K: WH reaches a few % of the largest output.
        assert_allclose(golden, got, rtol=0.1, atol=0.1 * golden.abs().max().item())
    dev_top1 = int(torch.argmax(got, dim=1).item())
    gold_top1 = int(torch.argmax(golden, dim=1).item())
    print(f"  fc PASSED dev_top1={dev_top1} golden_top1={gold_top1}")
    assert dev_top1 == gold_top1, f"fc top-1 mismatch dev={dev_top1} golden={gold_top1}"
