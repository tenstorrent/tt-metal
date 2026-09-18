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
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["2x2sp0tp1", "bh_4x8sp1tp0"],
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
    assert not pipeline._active_lora, "fresh pipeline must have no active LoRA"

    path, module = _first_qkv_lora_module(pipeline.transformer)
    assert module is not None, "no attn1.to_qkv LoRA module found — is lora_enabled wired?"
    base_w = _read_weight(module.weight).clone()

    # --- bind (on-device fuse) ---------------------------------------------
    pipeline.load_lora_weights(lora, strength=strength)
    assert id(pipeline.transformer) == transformer_id, "model object was rebuilt, not reused"
    assert module.is_lora_active, f"{path} was not bound"
    fused_w = _read_weight(module.weight)
    assert not torch.allclose(fused_w, base_w), "bind did not change the weight"

    # host recompute of the SAME registered delta; scale folded in exactly as _apply_delta_stack does.
    assert len(module.active_stack) == 1, f"expected a single bound adapter, got {module.active_stack}"
    bank_idx, bound_scale = module.active_stack[0]
    adapter = module.lora_bank[bank_idx]
    delta_oi = bound_scale * (adapter.B.to(torch.float32) @ adapter.A.to(torch.float32))  # [out,in]
    delta = delta_oi if delta_oi.shape == base_w.shape else delta_oi.T
    assert delta.shape == base_w.shape, f"delta {tuple(delta.shape)} vs weight {tuple(base_w.shape)}"
    expected = base_w.to(torch.float32) + delta

    ok, pcc = comp_pcc(expected, fused_w.to(torch.float32), pcc=0.999)
    logger.info(f"[{path}] on-device bind vs host delta: {pcc}")
    assert ok, f"on-device fused weight disagrees with host delta recompute: {pcc}"

    # --- unbind ------------------------------------------------------------
    pipeline.unload_lora_weights()
    assert not pipeline._active_lora
    assert id(pipeline.transformer) == transformer_id, "model object was rebuilt, not reused"
    restored_w = _read_weight(module.weight).to(torch.float32)

    # Unbind must actually move the weight back off the merged value. Comparing
    # against the fused weight is the check that discriminates: a residue-vs-base
    # bound cannot, because a correct unbind and an unbind that did nothing both
    # leave a residue on the order of the delta.
    assert not torch.allclose(restored_w, fused_w.to(torch.float32)), "unbind did not change the weight"

    # Residue is precision-dependent and only logged, never asserted.
    delta_mag = delta.abs().max().item()
    residue = (restored_w - base_w.to(torch.float32)).abs().max().item()
    logger.info(f"[{path}] residue after subtract-unbind: {residue:.5f} (delta max {delta_mag:.5f})")

    # The invariant production actually relies on: a page-out and reload re-seeds
    # the base weights, so that residue can never accumulate across generations.
    # LTX gets this on every generation for free because the VAE is a coresident
    # exclusion of the transformer. Without it, repeated bind/unbind walks W away
    # from the base — fastest when the weights are quantized.
    pipeline.transformer.deallocate_weights()
    pipeline._prepare_transformer(0)
    reload_residue = (_read_weight(module.weight).to(torch.float32) - base_w.to(torch.float32)).abs().max().item()
    logger.info(f"[{path}] residue after evict + reload: {reload_residue:.7f}")
    assert reload_residue <= 1e-6, f"evict + reload did not restore the base weight: residue {reload_residue}"
    logger.info("On-device LoRA bind/unbind weight checks passed.")

    # --- stack: several adapters bound at once ------------------------------
    # Same file registered twice at different scales -> two bank slots.
    #
    # Re-read the base here rather than reusing base_w: the bind/unbind cycle
    # above leaves a small residue in W by design (see the drift note in
    # layers/lora.py), and these checks are about the stack, not that residue.
    stack_base_w = _read_weight(module.weight).clone().to(torch.float32)

    h1 = pipeline.register_lora_adapter(lora, scale=0.5, name="stack_a")
    h2 = pipeline.register_lora_adapter(lora, scale=0.25, name="stack_b")
    pipeline.set_active_loras([h1, h2])
    assert len(module.active_stack) == 2, f"expected a 2-deep stack, got {module.active_stack}"

    stacked_w = _read_weight(module.weight).to(torch.float32)
    d_sum = None
    for bank_idx, sc in module.active_stack:
        a = module.lora_bank[bank_idx]
        d = sc * (a.B.to(torch.float32) @ a.A.to(torch.float32))
        d_sum = d if d_sum is None else d_sum + d
    d_sum = d_sum if d_sum.shape == stack_base_w.shape else d_sum.T
    ok, pcc = comp_pcc(stack_base_w + d_sum, stacked_w, pcc=0.999)
    logger.info(f"[{path}] 2-deep stack vs summed host deltas: {pcc}")
    assert ok, f"stacked weight disagrees with the sum of the host deltas: {pcc}"

    # The A/B LRU has to cover the whole stack, else each bind and unbind
    # re-uploads whichever members fell off the LRU end.
    assert module.lora_cache_capacity >= 2, f"cache capacity not raised for the stack: {module.lora_cache_capacity}"

    # Retuning one member's strength must re-bind, never re-register — that is
    # what keeps a strength tweak cheap for a multi-GB adapter.
    bank_len = len(module.lora_bank)
    pipeline.set_active_loras([(h1, 0.75), (h2, 0.25)])
    assert len(module.lora_bank) == bank_len, "re-scaling grew the bank; it must only re-bind"
    assert module.active_stack[0][1] == 0.75, module.active_stack

    # Unregistering one member leaves the rest of the stack merged.
    pipeline.set_active_loras([h1, h2])
    two_deep_w = _read_weight(module.weight).clone()
    module.unregister_lora(module.active_stack[0][0])
    assert len(module.active_stack) == 1, f"unregister should leave one member, got {module.active_stack}"
    assert not torch.allclose(
        _read_weight(module.weight), two_deep_w
    ), "unregistering a stacked member did not change the merged weight"

    pipeline.set_active_loras([])
    assert not module.active_stack and not module.is_lora_active
    logger.info("On-device LoRA stack checks passed.")

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
