# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device fuse-mode LoRA swap for the LTX-2.3 pipeline.

Unlike the host-fuse path (``fuse_loras_into`` + full weight reload), swapping here is a
``bind_active`` on each LoRA Linear: ``weight.data += scale * A@B`` on device, no reload.

Correctness anchor: the host-layout algebra (loader A/B == ``fuse_loras_into``) is already
proven CPU-side. This test closes the remaining gap — that ``LoRAMixin._apply_delta`` applies
that delta correctly on the *sharded device* weight — by comparing the on-device fused weight
against a host recompute of the same registered A/B.

The device cases need the real 22B checkpoint + a LoRA file + a mesh; they mirror the opt-in
style of ``test_ltx_lora_runtime.py``. Set ``RUN_LORA_GEN=1`` for an end-to-end base/lora/unloaded
generate A/B.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.experimental.lora.ltx_adapter_loader import iter_lora_modules
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.utils import tensor
from models.tt_dit.utils.test import line_params


def _default_checkpoint() -> str:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    if os.path.exists(local):
        return local
    return "Lightricks/LTX-2.3:ltx-2.3-22b-dev.safetensors"


def _default_lora() -> str:
    explicit = os.environ.get("LORA_PATH")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="Lightricks/LTX-2.3", filename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors")


def _read_weight(param) -> torch.Tensor:
    """Reconstruct a sharded Parameter's full weight on host."""
    return tensor.to_torch(param.data, mesh_axes=param.mesh_axes)


def _first_qkv_lora_module(transformer):
    """First self-attn to_qkv LoRA module (path, module) — the fused/permuted/interleaved
    path, i.e. the one most sensitive to a layout bug."""
    for path, mod in iter_lora_modules(transformer):
        if path.endswith("attn1.to_qkv"):
            return path, mod
    return None, None


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, True],
    ],
    ids=["2x2sp0tp1"],
    indirect=["mesh_device", "device_params"],
)
def test_ondevice_lora_bind_matches_host_delta(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
):
    """bind_active changes the weight to match base + host(scale*A@B); unbind restores the base
    (within bf16 add/subtract drift); the transformer Module is reused throughout."""
    ckpt = _default_checkpoint()
    lora = _default_lora()
    strength = float(os.environ.get("LORA_STRENGTH", "1.0"))

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    pipeline = LTXPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=None,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        mode="av",
        num_frames=int(os.environ.get("NUM_FRAMES", "25")),
        height=int(os.environ.get("HEIGHT", "256")),
        width=int(os.environ.get("WIDTH", "256")),
        lora_enabled=True,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        logger.info("Skipping assertions on non-zero rank")
        return

    transformer_id = id(pipeline.transformer)
    assert pipeline._active_lora is None, "fresh pipeline must have no active LoRA"

    path, module = _first_qkv_lora_module(pipeline.transformer)
    assert module is not None, "no attn1.to_qkv LoRA module found — is lora_enabled wired?"
    base_w = _read_weight(module.weight).clone()

    # --- bind (on-device fuse) ---------------------------------------------
    pipeline.load_lora_weights(lora, strength=strength)
    assert id(pipeline.transformer) == transformer_id, "model object was rebuilt, not reused"
    assert module.is_lora_active, f"{path} was not bound"
    fused_w = _read_weight(module.weight)
    assert not torch.allclose(fused_w, base_w), "bind did not change the weight"

    # host recompute of the SAME registered delta; scale folded in exactly as _apply_delta does.
    adapter = module.lora_bank[module.active_idx]
    delta_oi = module.active_scale * (adapter.B.to(torch.float32) @ adapter.A.to(torch.float32))  # [out,in]
    delta = delta_oi if delta_oi.shape == base_w.shape else delta_oi.T
    assert delta.shape == base_w.shape, f"delta {tuple(delta.shape)} vs weight {tuple(base_w.shape)}"
    expected = base_w.to(torch.float32) + delta

    ok, pcc = comp_pcc(expected, fused_w.to(torch.float32), pcc=0.999)
    logger.info(f"[{path}] on-device bind vs host delta: {pcc}")
    assert ok, f"on-device fused weight disagrees with host delta recompute: {pcc}"

    # --- unbind ------------------------------------------------------------
    pipeline.unload_lora_weights()
    assert pipeline._active_lora is None
    assert id(pipeline.transformer) == transformer_id, "model object was rebuilt, not reused"
    restored_w = _read_weight(module.weight)
    # fuse-mode unbind subtracts in bf16 — close, not bit-exact (see lora.py drift note).
    assert torch.allclose(
        restored_w.to(torch.float32), base_w.to(torch.float32), atol=1e-2
    ), "unbind did not restore the base weight within bf16 drift"
    logger.info("On-device LoRA bind/unbind weight checks passed.")

    if os.environ.get("RUN_LORA_GEN", "0") in ("1", "true", "True"):
        steps = int(os.environ.get("STEPS", "6"))
        num_frames = int(os.environ.get("NUM_FRAMES", "25"))
        height = int(os.environ.get("HEIGHT", "256"))
        width = int(os.environ.get("WIDTH", "256"))
        seq = pipeline.gemma_encoder_pair.sequence_length
        v_p = torch.zeros(1, seq, pipeline.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, seq, pipeline.gemma_encoder_pair.audio_dim)
        for tag, spec in (("base", None), ("lora", lora), ("unloaded", None)):
            if spec is not None:
                pipeline.load_lora_weights(spec, strength=strength)
            else:
                pipeline.unload_lora_weights()
            pipeline._prepare_transformer(0)
            v_lat, a_lat = pipeline.call_av(
                video_prompt_embeds=v_p,
                audio_prompt_embeds=a_p,
                neg_video_prompt_embeds=v_p,
                neg_audio_prompt_embeds=a_p,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=steps,
                seed=0,
                ge_gamma=0.0,
            )
            logger.info(f"[{tag}] generate ran; video latent {tuple(v_lat.shape)}")
            assert id(pipeline.transformer) == transformer_id, "model object was rebuilt during generation"
