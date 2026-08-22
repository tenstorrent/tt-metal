# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 RMSNorm — uses HF Gemma4RMSNorm as reference.

``test_rms_norm_with_scale`` / ``_without_scale`` use HF's *constructor* init,
which for Gemma4RMSNorm is all-ones — no dynamic range at all.
``test_rms_norm_real_weights`` loads the trained scale straight out of the
checkpoint (12B layer-0 ``input_layernorm`` spans [-143, 193], the final
``norm`` peaks at 604) so the bf16 weight cast and the width-sharded reduction
are exercised at the magnitudes the model actually runs at.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K (RMSNorm is replicated, no TP sharding)
    HF_MODEL=google/gemma-4-31B-it pytest models/demos/gemma4/tests/unit/test_rms_norm.py -k real_weights
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.rms_norm import RMSNorm

from ...tests.test_factory import (
    TestFactory,
    _get_model_path,
    compare_tensors,
    get_pcc_threshold,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
    real_weight_index,
    skip_unless_real_weights,
)


@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_rms_norm_with_scale(batch_size, seq_len, mesh_device, reset_seeds, request):
    """Test RMSNorm (with_scale=True) against HF Gemma4RMSNorm."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    hf_config = TestFactory.create_hf_config()
    hidden_size = hf_config.hidden_size

    hf_norm = Gemma4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps, with_scale=True)
    hf_norm.eval()
    state_dict = {"weight": hf_norm.weight.data.clone()}

    tt_norm = RMSNorm(mesh_device=mesh_device, hf_config=hf_config, state_dict=state_dict, with_scale=True)

    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = hf_norm(x_torch.float()).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_norm.forward(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"RMSNorm with_scale PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq()
def test_rms_norm_without_scale(batch_size, seq_len, mesh_device, reset_seeds, request):
    """Test RMSNorm (with_scale=False) against HF Gemma4RMSNorm."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    hf_config = TestFactory.create_hf_config()
    hidden_size = hf_config.hidden_size

    hf_norm = Gemma4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps, with_scale=False)
    hf_norm.eval()

    tt_norm = RMSNorm(mesh_device=mesh_device, hf_config=hf_config, state_dict={}, with_scale=False)

    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = hf_norm(x_torch.float()).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_norm.forward(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"RMSNorm without_scale PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric()
def test_rms_norm_keep_sharded_matches_interleaved(mesh_device, reset_seeds, request):
    """keep_sharded L1 output must match the historical S2I→DRAM path numerically."""
    from types import SimpleNamespace

    from models.demos.gemma4.tt.rms_norm import _SHARDED_NORM_MAX_HEIGHT

    hidden_size = 2048  # tile-aligned; divides cleanly for width-shard grids
    seq_len = min(128, _SHARDED_NORM_MAX_HEIGHT)
    hf_config = SimpleNamespace(rms_norm_eps=1e-6, hidden_size=hidden_size)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16)
    state_dict = {"weight": weight}

    tt_norm = RMSNorm(mesh_device=mesh_device, hf_config=hf_config, state_dict=state_dict, with_scale=True)

    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    x_tt = ttnn.from_torch(
        x_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
    )
    out_dram = tt_norm.forward(x_tt, keep_sharded=False)
    x_tt2 = ttnn.from_torch(
        x_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
    )
    out_sh = tt_norm.forward(x_tt2, keep_sharded=True)
    assert out_sh.is_sharded(), "keep_sharded must leave width-sharded L1 output"
    out_sh_as_dram = ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG)

    t_dram = ttnn.to_torch(ttnn.get_device_tensors(out_dram)[0]) if is_mesh else ttnn.to_torch(out_dram)
    t_sh = ttnn.to_torch(ttnn.get_device_tensors(out_sh_as_dram)[0]) if is_mesh else ttnn.to_torch(out_sh_as_dram)
    passing, pcc_msg = compare_tensors(t_sh, t_dram, pcc_threshold=0.999)
    assert passing, f"keep_sharded vs S2I path diverged: {pcc_msg}"


# ── Real-weight PCC ────────────────────────────────────────────────────────

# Checkpoint key patterns per norm site. The plain text checkpoints name these
# "model.layers.0.…" and the unified/multimodal ones "model.language_model.
# layers.0.…", so match on the suffix instead of hardcoding a full key.
_REAL_NORM_SITES = {
    "final_norm": lambda k: k.endswith(".norm.weight") and ".layers." not in k,
    "input_layernorm": lambda k: k.endswith(".layers.0.input_layernorm.weight"),
    "post_ffw_layernorm": lambda k: k.endswith(".layers.0.post_feedforward_layernorm.weight"),
}

# The vision / audio towers carry their own norms of a different width
# (e.g. vision_embedder.pos_norm); screen them out of the suffix match.
_NON_TEXT_TOWERS = ("vision", "audio")


def _is_text_tower(key):
    return not any(tower in key for tower in _NON_TEXT_TOWERS)


def _load_real_norm_weight(site):
    """Trained RMSNorm scale for ``site``, straight from the checkpoint."""
    skip_unless_real_weights()
    index = real_weight_index()

    matches = sorted(k for k in index if _is_text_tower(k) and _REAL_NORM_SITES[site](k))
    if not matches:
        pytest.skip(f"Checkpoint {_get_model_path()} has no '{site}' RMSNorm weight")
    key = matches[0]

    from safetensors import safe_open

    with safe_open(index[key], framework="pt") as f:
        weight = f.get_tensor(key)
    return key, weight.to(torch.bfloat16)


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("site", sorted(_REAL_NORM_SITES), ids=sorted(_REAL_NORM_SITES))
@parametrize_batch_seq(configs=[(1, 1), (1, 1024)], ids=["decode", "prefill_1024"])
def test_rms_norm_real_weights(batch_size, seq_len, site, mesh_device, reset_seeds, request):
    """RMSNorm vs HF Gemma4RMSNorm using the checkpoint's trained scale.

    The all-ones constructor init the other tests inherit hides everything
    weight magnitude can break: bf16 rounding of large scales and the
    cross-core gather in the width-sharded reduction. Real Gemma4 norm scales
    are neither unit nor centred (12B final ``norm`` reaches 604), so this is
    the gate that would catch a regression there.
    """
    skip_unless_real_weights()

    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    hf_config = TestFactory.create_hf_config()
    hidden_size = hf_config.hidden_size

    key, weight = _load_real_norm_weight(site)
    assert tuple(weight.shape) == (hidden_size,), f"{key} is {tuple(weight.shape)}, expected ({hidden_size},)"
    w_f32 = weight.float()
    logger.info(
        f"{site} -> {key}: min={w_f32.min():.4g} max={w_f32.max():.4g} "
        f"mean={w_f32.mean():.4g} std={w_f32.std():.4g}"
    )

    # Both paths get the same bf16-cast weight so the PCC isolates the TT op
    # under real dynamic range instead of re-measuring the fp32->bf16 cast.
    hf_norm = Gemma4RMSNorm(hidden_size, eps=hf_config.rms_norm_eps, with_scale=True)
    with torch.no_grad():
        hf_norm.weight.data.copy_(weight)
    hf_norm.eval()

    tt_norm = RMSNorm(
        mesh_device=mesh_device, hf_config=hf_config, state_dict={"weight": weight.clone()}, with_scale=True
    )

    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = hf_norm(x_torch.float()).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_norm.forward(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"RMSNorm real-weight ({site}) PCC too low: {pcc_msg}"
