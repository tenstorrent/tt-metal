# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single-device GDN forward_decode perf benchmark (synthetic TP=4).

"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn import TtGatedDeltaNet
from models.demos.qwen35_27b.tt.model_config import (
    GDN_CONV_KERNEL_SIZE,
    GDN_KEY_DIM,
    GDN_QKV_DIM,
    GDN_VALUE_DIM,
    GDN_Z_DIM,
    GDN_Dv,
    GDN_Nk,
    GDN_Nv,
    Qwen35ModelArgs,
    _replicate,
    _shard_small,
    _shard_w,
    create_activation_shard_config,
    create_dram_sharded_matmul_program_config,
    create_dram_sharded_mem_config,
)


def _override_args_for_synthetic_tp(args, synthetic_tp):
    """Re-derive the GDN TP fields and their dependent configs to simulate
    one device of a `synthetic_tp`-device tensor-parallel split."""
    args.gdn_nk_tp = GDN_Nk // synthetic_tp
    args.gdn_nv_tp = GDN_Nv // synthetic_tp
    args.gdn_qkv_dim_tp = GDN_QKV_DIM // synthetic_tp
    args.gdn_z_dim_tp = GDN_Z_DIM // synthetic_tp
    args.gdn_qkvz_dim_tp = (GDN_QKV_DIM + GDN_Z_DIM) // synthetic_tp
    args.gdn_value_dim_tp = GDN_VALUE_DIM // synthetic_tp
    args.gdn_key_dim_tp = GDN_KEY_DIM // synthetic_tp
    args.gdn_qkvz_weight_memcfg = create_dram_sharded_mem_config(args.dim, args.gdn_qkvz_dim_tp)
    args.gdn_out_weight_memcfg = create_dram_sharded_mem_config(args.gdn_value_dim_tp, args.dim)
    args.gdn_qkvz_progcfg = create_dram_sharded_matmul_program_config(1, args.dim, args.gdn_qkvz_dim_tp)
    args.gdn_out_progcfg = create_dram_sharded_matmul_program_config(1, args.gdn_value_dim_tp, args.dim)
    args.act_shard_gdn_value = create_activation_shard_config(args.gdn_value_dim_tp)


def _build_synthetic_gdn_weights(mesh, args, cache_dir):
    """Build a tw dict with the same keys/shapes as `_load_gdn_weights_for_layer`
    but with random bf16 tensors of per-device shapes. Reuses `_shard_w`,
    `_shard_small`, `_replicate` - on a (1,1) mesh, sharding is identity, so
    feeding per-device-shaped tensors yields per-device-shaped weights."""
    os.makedirs(cache_dir, exist_ok=True)
    dim = args.dim
    nv_tp = args.gdn_nv_tp
    qkvz_tp = args.gdn_qkvz_dim_tp
    value_tp = args.gdn_value_dim_tp
    qkv_tp = args.gdn_qkv_dim_tp

    # Linear weights: _shard_w expects [out, in], transposes internally to [in, out].
    qkvz_w = torch.randn(qkvz_tp, dim) * 0.02
    ab_w = torch.randn(2 * nv_tp, dim) * 0.02
    out_w = torch.randn(dim, value_tp) * 0.02  # [in_dim, out_dim_tp]; transpose to [value_tp, dim]

    # Per-head and small tensors
    A_log = torch.randn(nv_tp) * 0.5
    dt_bias = torch.randn(nv_tp) * 0.1
    norm_w = torch.ones(GDN_Dv)
    conv_taps = [torch.randn(qkv_tp) * 0.1 for _ in range(GDN_CONV_KERNEL_SIZE)]

    return {
        "qkvz": _shard_w(
            qkvz_w, mesh, dim=-1, memory_config=args.gdn_qkvz_weight_memcfg, cache_path=os.path.join(cache_dir, "qkvz")
        ),
        "ab": _shard_w(
            ab_w, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=os.path.join(cache_dir, "ab")
        ),
        "out": _shard_w(
            out_w, mesh, dim=0, memory_config=args.gdn_out_weight_memcfg, cache_path=os.path.join(cache_dir, "out")
        ),
        "A_log": _shard_small(A_log, mesh, os.path.join(cache_dir, "A_log")),
        "dt_bias": _shard_small(dt_bias, mesh, os.path.join(cache_dir, "dt_bias")),
        "norm_w": _replicate(norm_w, mesh, os.path.join(cache_dir, "norm_w")),
        "conv_taps": [_shard_small(t, mesh, os.path.join(cache_dir, f"conv_tap_{j}")) for j, t in enumerate(conv_taps)],
    }


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_gdn_decode_single_device_synthetic_tp4(mesh_device, reset_seeds, ensure_gc):
    """Single-device forward_decode perf benchmark with per-device dims matching
    one device of a 4-way TP split. Random synthetic weights; sanity asserts only.

    Run with tracy to collect a per-op breakdown of the full layer:
        python -m tracy -p -r -v -m pytest \\
            models/demos/qwen35_27b/tests/test_gdn_single_card.py::test_gdn_decode_single_device_synthetic_tp4
    """
    SYNTHETIC_TP = 4
    batch_size = 32
    max_seq_len = 256

    # Use the local config stub (Qwen35ModelArgs.LOCAL_HF_PARAMS) to avoid needing
    # the real model on disk. With dummy_weights=True, the framework loads config
    # from models/demos/qwen35_27b/model_params/Qwen3.5-27B-FP8/config.json and
    # skips tokenizer/processor creation.
    #
    # The model_type "qwen3_5" isn't in HF transformers' CONFIG_MAPPING (the real
    # repo brings modeling code via trust_remote_code). Register two Qwen2Config
    # subclasses with the right model_type so AutoConfig.from_pretrained accepts
    # our local stub. Qwen35ModelArgs reads fields directly from the resulting
    # config dict, so the registered class is only used to parse it.
    from transformers import AutoConfig, Qwen2Config
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    class _Qwen35StubConfig(Qwen2Config):
        model_type = "qwen3_5"

    class _Qwen35TextStubConfig(Qwen2Config):
        model_type = "qwen3_5_text"

    for cfg_cls in (_Qwen35StubConfig, _Qwen35TextStubConfig):
        if cfg_cls.model_type not in CONFIG_MAPPING:
            AutoConfig.register(cfg_cls.model_type, cfg_cls)

    # qwen35_27b/tt/model_config.py hardcodes DRAM_CORES=8 (right for WH/P150, wrong
    # for P100 which has 7 DRAM banks). Patch the module constants from the actual
    # device's DRAM grid size before constructing args, so DRAM-sharded weight
    # memcfgs and program configs land on the available DRAM cores.
    from models.demos.qwen35_27b.tt import model_config as qwen_cfg

    dram_cores_dev = mesh_device.dram_grid_size().x
    qwen_cfg.DRAM_CORES = dram_cores_dev
    qwen_cfg.DRAM_GRID = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores_dev - 1, 0))}
    )
    logger.info(f"Patched qwen35_27b DRAM_CORES = {dram_cores_dev} for this device")

    os.environ["HF_MODEL"] = "Qwen3.5-27B-FP8"
    args = Qwen35ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=True)
    _override_args_for_synthetic_tp(args, SYNTHETIC_TP)
    logger.info(
        f"Synthetic TP={SYNTHETIC_TP}: Nv_TP={args.gdn_nv_tp}, Nk_TP={args.gdn_nk_tp}, "
        f"value_dim_tp={args.gdn_value_dim_tp}, qkvz_dim_tp={args.gdn_qkvz_dim_tp}"
    )

    cache_dir = "/tmp/gdn_synthetic_tp4_cache"
    tw = _build_synthetic_gdn_weights(mesh_device, args, cache_dir)

    gdn = TtGatedDeltaNet(
        mesh_device=mesh_device,
        tt_ccl=None,  # single-device; tt_all_reduce short-circuits on (1,1)
        args=args,
        state_dict={},
        weight_cache_path=Path(cache_dir),
        layer_num=0,
        dtype=ttnn.bfloat8_b,
        transformation_mats=None,
        configuration=args,
    )
    gdn.set_weights(tw)
    gdn.reset_state()

    def _input():
        x = ttnn.from_torch(
            torch.randn(1, 1, batch_size, args.dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    # Warmup: first step compiles per-op kernels
    logger.info("Warmup forward_decode...")
    out0 = gdn.forward_decode(_input())
    ttnn.deallocate(out0)

    # Measured step
    logger.info("Measured forward_decode...")
    out = gdn.forward_decode(_input())
    out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    logger.info(f"Output shape: {out_t.shape}, range=[{out_t.min():.4f}, {out_t.max():.4f}]")
    assert out_t.shape[-1] == args.dim, f"Expected last dim {args.dim}, got {out_t.shape[-1]}"
    assert not torch.isnan(out_t).any(), "Output contains NaN"
    assert not torch.isinf(out_t).any(), "Output contains Inf"
    assert out_t.abs().max() > 0, "Output is all zeros"
    logger.info("PASSED: synthetic-TP=4 forward_decode produced valid output")
