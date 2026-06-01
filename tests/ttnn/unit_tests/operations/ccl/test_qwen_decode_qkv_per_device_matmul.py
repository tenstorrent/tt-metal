# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for Qwen Galaxy decode QKV `per_device` matmul (job 11751 hang).

The DRAM-sharded program config (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`)
hangs on BH no-prefetch. This test contrasts two configs against the same per-chip
matmul shape (32x1280 @ 1280x1280):

  - `dram_sharded`: DRAM width-sharded wqkv + `XQKV_DECODE_PER_DEVICE_PROGCFG` (HANGS)
  - `interleaved_auto`: interleaved DRAM wqkv + `program_config=None` (MLP w2 pattern, OK)

Run on BH 8x4:
  pytest tests/ttnn/unit_tests/operations/ccl/test_qwen_decode_qkv_per_device_matmul.py -v
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

CLUSTER_SHAPE = (8, 4)
M = 32
K_LOCAL = 1280  # dim // 4
N_LOCAL = 1280  # qkv_n_local
_DIM = 5120
_TORCH_SEED = 42051


def _sync_mesh(mesh_device):
    ttnn.synchronize_device(mesh_device)


def _wqkv_torch():
    rows, cols = CLUSTER_SHAPE
    full_k = cols * K_LOCAL
    full_n = rows * N_LOCAL
    return torch.randn(1, 1, full_k, full_n) * 0.02


def _build_wqkv_weights_dram_sharded(mesh_device, model_args, w_pt, dtype=ttnn.bfloat8_b):
    return ttnn.from_torch(
        w_pt,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_args.create_dram_sharded_mem_config(K_LOCAL, N_LOCAL),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=CLUSTER_SHAPE),
    )


def _build_wqkv_weights_interleaved(mesh_device, w_pt, dtype=ttnn.bfloat8_b):
    return ttnn.from_torch(
        w_pt,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=CLUSTER_SHAPE),
    )


def _build_act_attn_input(mesh_device, model_args, batch=M):
    pt = torch.randn(batch, 1, _DIM) * 0.05
    return model_args.prepare_residual_tensor_decode(pt, model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"])


def _run_matmul_dram_sharded(mesh_device, model_args, act, weights, label):
    pc = model_args.model_config["XQKV_DECODE_PER_DEVICE_PROGCFG"]
    logger.info(f"[qkv-matmul-repro] {label}: start (DRAM-sharded pc in0_block_w={pc.in0_block_w})")
    t0 = time.monotonic()
    out = ttnn.matmul(
        act,
        weights,
        program_config=pc,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=model_args.compute_kernel_config_hifi2,
        dtype=ttnn.bfloat16,
    )
    _sync_mesh(mesh_device)
    elapsed = time.monotonic() - t0
    logger.info(f"[qkv-matmul-repro] {label}: done in {elapsed:.2f}s")
    return out, elapsed


def _run_matmul_interleaved_auto(mesh_device, model_args, act, weights, label):
    """Interleaved DRAM wqkv + auto program config (MLP w2 pattern). Activation -> DRAM first."""
    logger.info(f"[qkv-matmul-repro] {label}: start (interleaved auto program config)")
    act_dram = ttnn.to_memory_config(act, ttnn.DRAM_MEMORY_CONFIG)
    t0 = time.monotonic()
    out = ttnn.linear(
        act_dram,
        weights,
        program_config=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=None,
        compute_kernel_config=model_args.compute_kernel_config_hifi2,
        dtype=ttnn.bfloat16,
    )
    _sync_mesh(mesh_device)
    elapsed = time.monotonic() - t0
    logger.info(f"[qkv-matmul-repro] {label}: done in {elapsed:.2f}s")
    ttnn.deallocate(act_dram)
    return out, elapsed


@pytest.mark.parametrize("mesh_device", [pytest.param(CLUSTER_SHAPE, id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
@pytest.mark.timeout(180)
def test_qwen_decode_qkv_per_device_matmul_dram_sharded(mesh_device):
    """Reproduces the hanging path: `SHARDED_ATTN_INPUT_MEMCFG` act + DRAM-sharded wqkv/progcfg."""
    if ttnn.get_arch_name().lower() != "blackhole":
        pytest.skip("Blackhole Galaxy only")
    assert mesh_device.get_num_devices() == CLUSTER_SHAPE[0] * CLUSTER_SHAPE[1]

    torch.manual_seed(_TORCH_SEED)
    model_args = TtQwenModelArgs(mesh_device, dummy_weights=False, max_batch_size=M, max_seq_len=256)
    model_args.use_prefetcher = False

    w_pt = _wqkv_torch()
    weights = _build_wqkv_weights_dram_sharded(mesh_device, model_args, w_pt)
    act = _build_act_attn_input(mesh_device, model_args)
    out, elapsed = _run_matmul_dram_sharded(mesh_device, model_args, act, weights, "dram_sharded")
    ttnn.deallocate(out)
    logger.info(f"[qkv-matmul-repro] dram_sharded elapsed={elapsed:.2f}s")
    assert elapsed < 120.0, f"matmul took {elapsed:.1f}s — likely hang (Slurm cancelled 11751 here)"


@pytest.mark.parametrize("mesh_device", [pytest.param(CLUSTER_SHAPE, id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
@pytest.mark.timeout(180)
def test_qwen_decode_qkv_per_device_matmul_interleaved_auto(mesh_device):
    """Non-hanging alternative: interleaved DRAM wqkv + auto program config (MLP w2 pattern)."""
    if ttnn.get_arch_name().lower() != "blackhole":
        pytest.skip("Blackhole Galaxy only")
    assert mesh_device.get_num_devices() == CLUSTER_SHAPE[0] * CLUSTER_SHAPE[1]

    torch.manual_seed(_TORCH_SEED)
    model_args = TtQwenModelArgs(mesh_device, dummy_weights=False, max_batch_size=M, max_seq_len=256)
    model_args.use_prefetcher = False

    w_pt = _wqkv_torch()
    weights = _build_wqkv_weights_interleaved(mesh_device, w_pt)
    act = _build_act_attn_input(mesh_device, model_args)
    out, elapsed = _run_matmul_interleaved_auto(mesh_device, model_args, act, weights, "interleaved_auto")
    ttnn.deallocate(out)
    logger.info(f"[qkv-matmul-repro] interleaved_auto elapsed={elapsed:.2f}s")
    assert elapsed < 120.0, f"matmul took {elapsed:.1f}s — likely hang"
