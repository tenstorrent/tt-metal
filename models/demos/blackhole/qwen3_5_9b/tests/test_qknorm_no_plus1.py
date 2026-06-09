# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the 27B-TP long-context (64k) degeneration root cause.

The Qwen3.5-27B-FP8 checkpoint stores the attention q_norm/k_norm weights as the already-effective
(1+w) scale, so the TP loader (load_attention_weights_tp) must load them DIRECTLY, WITHOUT adding
a +1 offset. An earlier build added +1 ("zero-centered correction"), which double-counted and
over-scaled Q/K by ~2.3x -> softmax over-sharpens -> 64k retrieval collapses into repetition. The
working reference (qwen35_27b) loads them without +1; this test pins that so the +1 can't return.

Fast: builds a tiny synthetic state_dict and checks the loaded q_norm/k_norm equal the input
weights (no +1), unlike the main RMSNorms which DO keep their +1 (handled elsewhere, by the
framework norm, not this loader).

Run:
  source python_env/bin/activate
  MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.5-27B-FP8 \
    pytest -svq models/demos/blackhole/qwen3_5_9b/tests/test_qknorm_no_plus1.py
"""
import os

import pytest
import torch
from loguru import logger

import ttnn


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
def test_qknorm_loaded_without_plus1(mesh_device):
    """load_attention_weights_tp must NOT add +1 to q_norm/k_norm (matches the reference 27B)."""
    from models.demos.blackhole.qwen3_5_9b.tt.attention.tp import load_attention_weights_tp

    nd = mesh_device.get_num_devices()
    HD = 128
    NH = 8 * nd  # arbitrary; sharded by dim=-1 across the mesh
    NKV = nd
    DIM = 1024

    torch.manual_seed(0)
    # Distinctive q_norm/k_norm values centered well away from 0 (like the FP8 ckpt's ~0.75),
    # so a stray +1 would be unmistakable.
    q_norm_w = torch.full((HD,), 0.75) + 0.01 * torch.randn(HD)
    k_norm_w = torch.full((HD,), 0.60) + 0.01 * torch.randn(HD)
    state_dict = {
        "q_proj.weight": torch.randn(DIM, NH * HD * 2) * 0.02,  # fused [Q,gate], column-parallel
        "k_proj.weight": torch.randn(DIM, NKV * HD) * 0.02,
        "v_proj.weight": torch.randn(DIM, NKV * HD) * 0.02,
        "o_proj.weight": torch.randn(NH * HD, DIM) * 0.02,
        "q_norm.weight": q_norm_w,
        "k_norm.weight": k_norm_w,
    }

    class _Args:
        n_local_heads = NH // nd
        n_local_kv_heads = max(1, NKV // nd)
        head_dim = HD
        rope_head_dim = 64
        max_batch_size = 1
        max_seq_len = 128

        def ccl_topology(self):
            return ttnn.Topology.Linear

    tw = load_attention_weights_tp(mesh_device, state_dict, _Args(), cache_dir=None)

    # Gather a single replica and compare against the RAW input (no +1).
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if nd > 1 else None
    q_loaded = ttnn.to_torch(tw["q_norm"], mesh_composer=comp).float().reshape(-1)[:HD]
    k_loaded = ttnn.to_torch(tw["k_norm"], mesh_composer=comp).float().reshape(-1)[:HD]

    q_err_raw = (q_loaded - q_norm_w).abs().max().item()
    q_err_plus1 = (q_loaded - (q_norm_w + 1.0)).abs().max().item()
    k_err_raw = (k_loaded - k_norm_w).abs().max().item()
    logger.info(f"q_norm: |loaded-raw|={q_err_raw:.4f}  |loaded-(raw+1)|={q_err_plus1:.4f}")
    logger.info(f"k_norm: |loaded-raw|={k_err_raw:.4f}")

    # bf16 round-trip tolerance ~0.01; a stray +1 would be ~1.0 off.
    assert q_err_raw < 0.05, f"q_norm should load WITHOUT +1, but |loaded-raw|={q_err_raw:.4f} (regressed +1?)"
    assert k_err_raw < 0.05, f"k_norm should load WITHOUT +1, but |loaded-raw|={k_err_raw:.4f} (regressed +1?)"
    assert q_err_plus1 > 0.5, "sanity: loaded q_norm must clearly NOT equal raw+1"
    logger.info("PASSED: 27B-TP q_norm/k_norm loaded without the +1 offset (matches reference)")
