# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 unit test for the full dots.ocr vision block on Blackhole 4×P150c.

Exercises ``TTNNDotsVisionBlockTP4BH.forward`` (norm1 → attn → residual →
norm2 → mlp → residual).  Every local activation tensor (op in/out) is L1
interleaved; weights and RoPE tables start in DRAM and are promoted to L1
before use.  Same Blackhole TP4 paths as the standalone attention and MLP tests:

  - ``TTNNDotsVisionAttentionTP4BH``: L1 activations, swept QKV / o_proj / SDPA PCs
  - ``TTNNDotsVisionMLPTP4BH``: N-shard fc1/fc3, K-shard fc2 + Ring all_reduce

Loads pretrained dots.ocr weights via ``from_pretrained`` (set ``DOTS_OCR_MODEL_PATH``
to a local checkpoint or rely on the HF cache for ``rednote-hilab/dots.ocr``).

Run::

    MESH_DEVICE=P150x4 pytest \\
        models/experimental/tt_symbiote/tests/test_vision_block_tp4_n_k_shard_blackhole.py -s

    MESH_DEVICE=P300x2 pytest \\
        models/experimental/tt_symbiote/tests/test_vision_block_tp4_n_k_shard_blackhole.py -s
"""

from __future__ import annotations

# Re-use the TP4 mesh fixture from the attention test module.
pytest_plugins = ("models.experimental.tt_symbiote.tests.test_vision_attn_tp4_n_k_shard_blackhole",)

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.tt_symbiote.modules.dots_ocr_vision import (
    TTNNDotsVision2DRoPE,
    TTNNDotsVisionBlockTP4BH,
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


def _first_child(module, *names: str):
    """Return the first registered child ``module.<name>`` without nested getattr defaults."""
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"{type(module).__name__!r} has none of {names!r}")


def _raw_ttnn(t):
    """Unwrap ``TorchTTNNTensor`` to the underlying ``ttnn.Tensor`` (if wrapped)."""
    inner = getattr(t, "ttnn_tensor", None)
    return inner if inner is not None else t


def _resolve_model_path() -> str:
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


def _load_vision_block():
    """Return blocks[0] from the dots.ocr vision tower with pretrained weights."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        pytest.skip("transformers required for dots.ocr weight loading")

    model_path = _resolve_model_path()
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
    except Exception as exc:
        pytest.skip(f"dots.ocr pretrained weights unavailable at {model_path!r}: {exc}")

    vision_tower = hf_model.vision_tower
    blocks = _first_child(vision_tower, "blocks", "layers")
    block = blocks[0]

    attn = _first_child(block, "attn", "attention", "self_attn")
    mlp = _first_child(block, "mlp", "feed_forward")
    qkv = _first_child(attn, "qkv", "qkv_proj")
    proj = _first_child(attn, "proj", "o_proj", "out_proj")

    assert int(qkv.weight.shape[1]) == HIDDEN
    assert int(mlp.fc1.weight.shape[1]) == HIDDEN
    assert int(mlp.fc1.weight.shape[0]) == INTERMEDIATE

    # K-shard fc2: bias would be summed TP times across all_reduce; zero for PCC.
    if mlp.fc2.bias is not None:
        mlp.fc2.bias.data.zero_()

    logger.info(f"loaded dots.ocr vision block[0] from {model_path!r}")
    return block, qkv, proj


def _torch_vision_block_ref(hf_block, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Float reference matching ``TTNNDotsVisionBlock.forward`` (post-norm + 2D RoPE)."""
    attn = _first_child(hf_block, "attn", "attention", "self_attn")
    qkv_w = _first_child(attn, "qkv", "qkv_proj").weight.data
    o_w = _first_child(attn, "proj", "o_proj", "out_proj").weight.data
    mlp = _first_child(hf_block, "mlp", "feed_forward")

    with torch.no_grad():
        residual = x
        normed = hf_block.norm1(x)
        attn_out = _torch_attn_ref(qkv_w, o_w, normed, cos, sin).to(residual.dtype)
        hidden = residual + attn_out
        residual = hidden
        normed2 = hf_block.norm2(hidden)
        mlp_out = mlp(normed2)
        out = residual + mlp_out
    return out.float()


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@run_for_blackhole()
def test_vision_block_tp4_n_k_shard_blackhole(bh_tp4_mesh_device):
    """Full vision block TP4 on Blackhole: norm1 → attn → norm2 → mlp with L1 TP4 paths."""
    mesh_device = bh_tp4_mesh_device
    mesh_env = os.environ.get("MESH_DEVICE", "P150x4")

    assert GRID_H * GRID_W == SEQ_LEN

    hf_block, qkv_mod, proj_mod = _load_vision_block()
    _ = qkv_mod, proj_mod

    torch.manual_seed(0xB10C_4B1E)
    x = torch.randn(1, 1, SEQ_LEN, HIDDEN, dtype=torch.bfloat16) * 0.1
    cos_torch, sin_torch = _rope_cos_sin_torch()

    with torch.no_grad():
        ref = _torch_vision_block_ref(hf_block, x, cos_torch, sin_torch)

    tt_block = TTNNDotsVisionBlockTP4BH.from_torch(hf_block, hidden_size=HIDDEN, num_heads=12)
    set_device(tt_block, mesh_device, register_forward_hook=False, dump_visualization=False)
    tt_block.preprocess_weights()
    tt_block.move_weights_to_device()

    def _log_pc(label, pc):
        if pc is None:
            logger.info(f"  {label}: auto-config")
            return
        if hasattr(pc, "q_chunk_size"):
            g = pc.compute_with_storage_grid_size
            logger.info(
                f"  {label}: q_chunk={pc.q_chunk_size} k_chunk={pc.k_chunk_size} "
                f"grid=({g.x},{g.y}) exp_approx={pc.exp_approx_mode}"
            )
            return
        tm = "T" if pc.transpose_mcast else "F"
        logger.info(
            f"  {label}: grid={pc.compute_with_storage_grid_size} tm={tm} "
            f"M={pc.per_core_M} N={pc.per_core_N} obh={pc.out_block_h} "
            f"ibw={pc.in0_block_w} sub=({pc.out_subblock_h},{pc.out_subblock_w})"
        )

    attn = tt_block.attn
    mlp = tt_block.mlp
    logger.info(f"[vision_block_tp4_bh program configs arch={mesh_device.arch().name}]")
    _log_pc("qkv", getattr(attn, "_bh_tp4_qkv_pc", None))
    _log_pc("o_proj bf8", getattr(attn, "_bh_tp4_o_pc", None))
    _log_pc("sdpa", getattr(attn, "_bh_tp4_sdpa_pc", None))
    _log_pc("mlp gate", getattr(mlp, "_bh_tp4_gate_pc", None))
    _log_pc("mlp up", getattr(mlp, "_bh_tp4_up_pc", None))
    _log_pc("mlp down", getattr(mlp, "_bh_tp4_down_pc", None))

    grid_thw = torch.tensor([[1, GRID_H, GRID_W]], dtype=torch.int32)
    rope = TTNNDotsVision2DRoPE(
        device=mesh_device,
        head_dim=128,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        theta=10000.0,
    )
    rot_mats_dram, _ = rope.build(grid_thw, SEQ_LEN)
    from models.experimental.tt_symbiote.modules.vision_tp4_bh import rot_mats_l1

    rot_mats = rot_mats_l1(rot_mats_dram)
    cos_dram, sin_dram = rot_mats_dram
    ttnn.deallocate(cos_dram)
    ttnn.deallocate(sin_dram)

    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    hidden_states = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=rep,
    )

    # Call ``forward`` directly so the return value is a raw ``ttnn.Tensor``, not
    # a ``TorchTTNNTensor`` wrapper from ``module.call()``.
    out_tt = tt_block.forward(
        hidden_states,
        rot_mats=rot_mats,
    )
    out_tt = _raw_ttnn(out_tt)
    ttnn.synchronize_device(mesh_device)

    device_tensors = ttnn.get_device_tensors(out_tt)
    assert len(device_tensors) == TP, f"expected {TP} device tensors, got {len(device_tensors)}"

    failures = []
    for dev_idx, dev_t in enumerate(device_tensors):
        out_dev = dev_t.cpu().to_torch().float().reshape(ref.shape)
        passed_dev, pcc_dev = comp_pcc(ref, out_dev, PCC_THRESHOLD)
        logger.info(
            f"[vision_block_tp4_n_k_shard arch={mesh_device.arch().name} env={mesh_env} "
            f"TP={TP} S={SEQ_LEN} H={HIDDEN} I={INTERMEDIATE} "
            f"grid=({GRID_H},{GRID_W}) dev={dev_idx}] "
            f"pcc={float(pcc_dev):.6f} (threshold {PCC_THRESHOLD})"
        )
        if not passed_dev:
            failures.append((dev_idx, float(pcc_dev)))

    ttnn.deallocate(out_tt)
    ttnn.deallocate(hidden_states)
    cos_tt, sin_tt = rot_mats
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)

    assert not failures, (
        "Vision block TP4 N/K-shard failed PCC on "
        + ", ".join(f"dev{i} pcc={p:.6f}" for i, p in failures)
        + f" (threshold {PCC_THRESHOLD})"
    )
