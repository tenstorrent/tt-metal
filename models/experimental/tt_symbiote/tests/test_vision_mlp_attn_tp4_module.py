# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Module-level TP4 unit tests for the dots.ocr vision MLP and attention.

Unlike the ``*_n_k_shard_*`` tests (which exercise the raw sharded matmul ops),
these build the actual ``TTNNModule`` subclasses end-to-end -- ``from_torch`` ->
``set_device`` -> ``preprocess_weights`` -> ``move_weights_to_device`` ->
``forward`` -- and PCC each per-device output against the HF reference:

  * ``TTNNDotsVisionMLPTP4BH``       (N-shard fc1/fc3, K-shard fc2 + Ring all_reduce)
  * ``TTNNDotsVisionAttentionTP4BH`` (L1 activations, swept QKV/o_proj/SDPA, 2D RoPE)

Both reduce to a replicated full-hidden output, so every one of the TP=4 device
shards must match the reference. Pretrained dots.ocr weights are loaded via
``from_pretrained`` (``DOTS_OCR_MODEL_PATH`` or the HF cache).

Run::

    MESH_DEVICE=P150x4 pytest -s \\
        models/experimental/tt_symbiote/tests/test_vision_mlp_attn_tp4_module.py
"""

from __future__ import annotations

# Re-use the TP4 mesh fixture (bh_tp4_mesh_device) + RoPE/attn-reference helpers.
pytest_plugins = ("models.experimental.tt_symbiote.tests.test_vision_attn_tp4_n_k_shard_blackhole",)

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.tt_symbiote.modules.dots_ocr_vision import (
    TTNNDotsVision2DRoPE,
    TTNNDotsVisionAttentionTP4BH,
    TTNNDotsVisionMLPTP4BH,
)
from models.experimental.tt_symbiote.utils.device_management import set_device

from models.experimental.tt_symbiote.tests.test_vision_attn_tp4_n_k_shard_blackhole import (
    GRID_H,
    GRID_W,
    HIDDEN,
    PCC_THRESHOLD,
    SEQ_LEN,
    SPATIAL_MERGE_SIZE,
    TP,
    _device_params,
    _rope_cos_sin_torch,
    _torch_ref as _torch_attn_ref,
)

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"
INTERMEDIATE = 4224
NUM_HEADS = 12
HEAD_DIM = 128


def _first_child(module, *names):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"{type(module).__name__!r} has none of {names!r}")


def _raw_ttnn(t):
    inner = getattr(t, "ttnn_tensor", None)
    return inner if inner is not None else t


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


def _load_vision_block():
    """blocks[0] of the dots.ocr vision tower with pretrained weights (+ attn/mlp)."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        pytest.skip("transformers required for dots.ocr weight loading")
    model_path = _resolve_model_path()
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).eval()
    except Exception as exc:
        pytest.skip(f"dots.ocr pretrained weights unavailable at {model_path!r}: {exc}")
    block = _first_child(hf_model.vision_tower, "blocks", "layers")[0]
    attn = _first_child(block, "attn", "attention", "self_attn")
    mlp = _first_child(block, "mlp", "feed_forward")
    return block, attn, mlp


def _x_input():
    torch.manual_seed(0xB10C_F00D)
    return torch.randn(1, 1, SEQ_LEN, HIDDEN, dtype=torch.bfloat16) * 0.1


def _to_device_l1(x, mesh_device):
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _check_all_devices(out_tt, ref, mesh_device, label):
    """PCC every TP device shard of a replicated output against ``ref``."""
    device_tensors = ttnn.get_device_tensors(_raw_ttnn(out_tt))
    assert len(device_tensors) == TP, f"expected {TP} device tensors, got {len(device_tensors)}"
    failures = []
    for dev_idx, dev_t in enumerate(device_tensors):
        out_dev = dev_t.cpu().to_torch().float().reshape(ref.shape)
        passed, pcc = comp_pcc(ref, out_dev, PCC_THRESHOLD)
        logger.info(f"[{label} arch={mesh_device.arch().name} TP={TP} dev={dev_idx}] pcc={float(pcc):.6f}")
        if not passed:
            failures.append((dev_idx, float(pcc)))
    assert not failures, f"{label} failed PCC on " + ", ".join(f"dev{i} pcc={p:.6f}" for i, p in failures)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@run_for_blackhole()
def test_vision_mlp_tp4_module(bh_tp4_mesh_device):
    """``TTNNDotsVisionMLPTP4BH`` end-to-end vs HF vision MLP (replicated output)."""
    mesh_device = bh_tp4_mesh_device
    assert GRID_H * GRID_W == SEQ_LEN
    _, _, hf_mlp = _load_vision_block()
    assert int(hf_mlp.fc1.weight.shape[1]) == HIDDEN and int(hf_mlp.fc1.weight.shape[0]) == INTERMEDIATE
    # K-shard fc2 bias would be summed TP times across the all_reduce; zero for PCC.
    if hf_mlp.fc2.bias is not None:
        hf_mlp.fc2.bias.data.zero_()

    x = _x_input()
    with torch.no_grad():
        ref = hf_mlp(x).float()

    tt_mlp = TTNNDotsVisionMLPTP4BH.from_torch(hf_mlp)
    set_device(tt_mlp, mesh_device, register_forward_hook=False, dump_visualization=False)
    tt_mlp.preprocess_weights()
    tt_mlp.move_weights_to_device()

    out_tt = tt_mlp.forward(_to_device_l1(x, mesh_device))
    ttnn.synchronize_device(mesh_device)
    _check_all_devices(out_tt, ref, mesh_device, "vision_mlp_tp4_module")


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@run_for_blackhole()
def test_vision_attn_tp4_module(bh_tp4_mesh_device):
    """``TTNNDotsVisionAttentionTP4BH`` end-to-end vs HF vision attention + 2D RoPE."""
    mesh_device = bh_tp4_mesh_device
    assert GRID_H * GRID_W == SEQ_LEN
    assert NUM_HEADS % TP == 0
    _, hf_attn, _ = _load_vision_block()
    qkv = _first_child(hf_attn, "qkv", "qkv_proj")
    proj = _first_child(hf_attn, "proj", "o_proj", "out_proj")
    assert int(qkv.weight.shape[1]) == HIDDEN

    x = _x_input()
    cos_torch, sin_torch = _rope_cos_sin_torch()
    with torch.no_grad():
        ref = _torch_attn_ref(qkv.weight.data, proj.weight.data, x, cos_torch, sin_torch).float()

    tt_attn = TTNNDotsVisionAttentionTP4BH.from_torch(hf_attn, hidden_size=HIDDEN, num_heads=NUM_HEADS)
    set_device(tt_attn, mesh_device, register_forward_hook=False, dump_visualization=False)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    # Build the device 2D-RoPE tables (DRAM -> L1), as the block/tower do.
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import rot_mats_l1

    rope = TTNNDotsVision2DRoPE(
        device=mesh_device, head_dim=HEAD_DIM, spatial_merge_size=SPATIAL_MERGE_SIZE, theta=10000.0
    )
    rot_mats_dram, _ = rope.build(torch.tensor([[1, GRID_H, GRID_W]], dtype=torch.int32), SEQ_LEN)
    rot_mats = rot_mats_l1(rot_mats_dram)
    for t in rot_mats_dram:
        ttnn.deallocate(t)

    out_tt = tt_attn.forward(_to_device_l1(x, mesh_device), rot_mats=rot_mats)
    ttnn.synchronize_device(mesh_device)
    _check_all_devices(out_tt, ref, mesh_device, "vision_attn_tp4_module")
