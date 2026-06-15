# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Runtime (post-init) LoRA load / fuse / unload for the base LTX-2.3 pipeline.

Covers the feature requested in the LoRA-serving issue: load, fuse, and unload LoRA
adaptors *after* the model has been instantiated, running successive forward()/generate()
calls without tearing the model down, and leaving the non-LoRA path unaffected.

The `test_lora_spec_coercion` case is device-free (CI-safe). The `test_runtime_lora_*`
cases need the real 22B checkpoint + a LoRA file + a mesh device, so they mirror the
opt-in style of `test_pipeline_ltx_two_stages.py`.
"""

import os
import tempfile

import pytest
import torch
from loguru import logger
from safetensors.torch import save_file

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline, latent_grid
from models.tt_dit.utils import tensor
from models.tt_dit.utils.fuse_loras import LoraSpec
from models.tt_dit.utils.test import line_params
from models.tt_dit.utils.video import export_video_audio


def _default_checkpoint() -> str:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    if os.path.exists(local):
        return local
    return "Lightricks/LTX-2.3:ltx-2.3-22b-dev.safetensors"


def _default_lora() -> str:
    """Resolve a LoRA to fuse at runtime: env var > local distilled LoRA > HF download."""
    explicit = os.environ.get("LORA_PATH")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="Lightricks/LTX-2.3", filename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors")


def _find_param(module, suffix: str, prefix: str = ""):
    """First Parameter whose dotted name ends with ``suffix`` (depth-first)."""
    for name, p in module.named_parameters():
        if f"{prefix}{name}".endswith(suffix):
            return p
    for name, child in module.named_children():
        found = _find_param(child, suffix, f"{prefix}{name}.")
        if found is not None:
            return found
    return None


def _read_weight(param) -> torch.Tensor:
    """Reconstruct a sharded Parameter's full weight on host for comparison."""
    return tensor.to_torch(param.data, mesh_axes=param.mesh_axes)


# ---------------------------------------------------------------------------
# Device-free logic
# ---------------------------------------------------------------------------


def test_lora_spec_coercion():
    """`load_lora_weights` argument normalisation + LoRA-tagged cache naming."""
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "styleA.safetensors")
        b = os.path.join(d, "styleB.safetensors")
        for p in (a, b):
            save_file({"dummy": torch.zeros(1)}, p)

        # bare path -> LoraSpec with the given strength
        specs = LTXPipeline._coerce_lora_specs(a, 0.8)
        assert specs == [LoraSpec(path=a, strength=0.8)]

        # explicit LoraSpec keeps its own strength; mixed list preserved in order
        specs = LTXPipeline._coerce_lora_specs([LoraSpec(path=a, strength=0.5), b], 1.0)
        assert specs == [LoraSpec(path=a, strength=0.5), LoraSpec(path=b, strength=1.0)]

        # None / empty -> no specs
        assert LTXPipeline._coerce_lora_specs(None, 1.0) == []
        assert LTXPipeline._coerce_lora_specs([], 1.0) == []

        # the base (no-LoRA) cache name must be a strict prefix of the fused name, so the
        # unmodified base never collides with a fused variant on disk.
        base_name = LTXPipeline._build_transformer_cache_name(a, [])
        fused_name = LTXPipeline._build_transformer_cache_name(a, [LoraSpec(path=a, strength=0.8)])
        assert fused_name != base_name
        assert ".lora-" in fused_name
        assert "@0.8" in fused_name


# ---------------------------------------------------------------------------
# Hardware: load / unload correctness (no teardown, base path unaffected)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(3600)  # one 46 GB load + LoRA fuse(s); override the 300s default
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, True],
    ],
    ids=["2x2sp0tp1"],
    indirect=["mesh_device", "device_params"],
)
def test_runtime_lora_load_unload(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
):
    """load_lora -> weights change; unload_lora -> weights restored bit-identically; the
    transformer Module object is reused throughout (no teardown). Optionally runs a
    generate() in each state when RUN_LORA_GEN=1."""
    ckpt = _default_checkpoint()
    lora = _default_lora()

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "25"))
    height = int(os.environ.get("HEIGHT", "256"))
    width = int(os.environ.get("WIDTH", "256"))

    # gemma_path=None -> zero prompt embeddings (Gemma is gated on this box). The LoRA
    # feature under test is independent of the text encoder; content is noise.
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
        num_frames=num_frames,
        height=height,
        width=width,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        logger.info("Skipping assertions on non-zero rank")
        return

    transformer_id = id(pipeline.transformer)
    assert not pipeline._active_lora_specs, "fresh pipeline must have no active LoRA"

    # A LoRA-affected fused-QKV weight; snapshot the pristine base.
    param = _find_param(pipeline.transformer, "to_qkv.weight")
    assert param is not None, "could not locate a to_qkv weight to probe"
    base_w = _read_weight(param).clone()

    # --- load + fuse -------------------------------------------------------
    pipeline.load_lora_weights(lora, strength=1.0)
    assert pipeline._active_lora_specs == [LoraSpec(path=lora, strength=1.0)]
    assert id(pipeline.transformer) == transformer_id, "model object was rebuilt, not reused"
    fused_w = _read_weight(param)
    assert not torch.equal(fused_w, base_w), "LoRA fuse did not change the weight"

    # idempotent re-request is a no-op (weights unchanged)
    pipeline.load_lora_weights(lora, strength=1.0)
    assert torch.equal(_read_weight(param), fused_w)

    # --- unload ------------------------------------------------------------
    pipeline.unload_lora_weights()
    assert pipeline._active_lora_specs == []
    assert id(pipeline.transformer) == transformer_id, "model object was rebuilt, not reused"
    restored_w = _read_weight(param)
    assert torch.equal(restored_w, base_w), "unload did not restore base weights bit-identically"

    logger.info("Runtime LoRA load/unload weight checks passed.")

    if os.environ.get("RUN_LORA_GEN", "0") in ("1", "true", "True"):
        # End-to-end proof that forward() runs in each LoRA state without teardown. Uses the
        # gemma-free zero-embedding call_av path (mirrors test_ltx_gen_bh_qb); content is noise.
        steps = int(os.environ.get("STEPS", "6"))
        seq = pipeline.gemma_encoder_pair.sequence_length
        v_p = torch.zeros(1, seq, pipeline.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, seq, pipeline.gemma_encoder_pair.audio_dim)
        for tag, spec in (("base", None), ("lora", lora), ("unloaded", None)):
            if spec is not None:
                pipeline.load_lora_weights(spec)
            else:
                pipeline.unload_lora_weights()
            pipeline._prepare_transformer(0)  # active transformer before denoise (mirrors generate())
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
            pipeline._prepare_vae()
            lf, lh, lw = latent_grid(num_frames, height, width)
            px = pipeline.decode_latents(v_lat, lf, lh, lw)
            audio = pipeline.decode_audio(a_lat, num_frames, fps=24)
            out = f"ltx_lora_runtime_{tag}.mp4"
            export_video_audio(px, out, fps=24, audio=audio)
            assert os.path.exists(out) and os.path.getsize(out) > 0, f"{tag} generation produced no mp4"
            logger.info(f"[{tag}] wrote {out} ({os.path.getsize(out)} B), video={tuple(px.shape)}")
            assert id(pipeline.transformer) == transformer_id, "model object was rebuilt during generation"
