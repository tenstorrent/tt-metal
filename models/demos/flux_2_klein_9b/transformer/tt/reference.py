# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The HF golden -- the ONLY place in this package where HF is called to compute.

Why the golden lives in its own module
-------------------------------------
`tt/pipeline.py` is a pure TTNN forward: it reads the reference module's
*weights* at build time and its *config*, and never calls it. The PCC number
those two produce is only meaningful if the other side of the comparison is the
real thing, so this module calls
`diffusers.Flux2Transformer2DModel.forward` directly, in float32, on the very
same input dict `tt/inputs.py` builds.

Keeping it separate is what makes the "TT-only hot path" claim checkable: every
HF call in this package is in this file, plus the one constant-seeding call in
`Flux2KleinTransformerPipeline.denoise_trace_setup`.

Loading the checkpoint
----------------------
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)` is a **diffusers**
checkpoint (`config._class_name = Flux2Transformer2DModel`), not a transformers
one: no `model_type`, no `auto_map`, so `AutoModel` cannot see it. Source B
already solved this -- including the fact that the pinned `diffusers==0.35.1`
predates the class and has to be side-loaded -- in
`models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/_reference_loader.py`.
That loader is reused here verbatim (imported by path, because its directory is
not a package) rather than reimplemented, so the golden this module returns is
byte-identical to the one every per-component PCC test was graded against.

The two goldens
---------------
`_hf_reference_denoise_step`     -- Call 1: one forward, the velocity prediction.
`_hf_reference_denoise_latents`  -- Call 2: that forward driven around the
                                    flow-match Euler loop, on the SAME sigma
                                    list `tt/inputs.py::sigma_schedule` hands
                                    the TT loop, for the SAME number of steps.

Depth
-----
Both accept `dual_layers` / `single_layers`. At full depth they call
`model(...)` -- the reference callable, unmodified, which is the golden
`e2e_plan.json` names. At a capped depth (the fast wiring loop) they run the
same chain over sliced `ModuleList`s, so a capped TT build is compared against a
capped reference and the PCC still measures the port rather than the missing
blocks. Nothing is monkey-patched: the slice is taken at the call site and the
module is left exactly as loaded.
"""

from __future__ import annotations

import importlib.util
import sys

import torch

from .stubs import HF_MODEL_ID, REPO_ROOT, SOURCE_B

# Source B's loader: diffusers-version resolution, real shipped weights, eval()
# + requires_grad_(False). Its directory has no __init__.py (the PCC tests import
# their siblings by path too), so it is loaded by file path.
_LOADER_PATH = REPO_ROOT / SOURCE_B / "tests" / "pcc" / "_reference_loader.py"

_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def _reference_loader():
    """Import Source B's `_reference_loader.py` by path, once."""
    name = "_flux2_e2e_reference_loader"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    if not _LOADER_PATH.is_file():
        raise FileNotFoundError(
            f"Source B's reference loader is missing ({_LOADER_PATH}); the golden cannot be built "
            "without it, and reimplementing the diffusers-version resolution here would let the e2e "
            "golden drift from the one the per-component PCCs were graded against"
        )
    spec = importlib.util.spec_from_file_location(name, _LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_model(model_id: str = HF_MODEL_ID):
    """The reference `Flux2Transformer2DModel`, in eval mode, with real weights.

    Cached per `model_id`: it is 9.08 B parameters (~36 GB in float32), the
    pipeline build reads its weights and the golden runs its forward, and both
    e2e tests plus the trace contract test want the same object.
    """
    model = _MODEL_CACHE.get(model_id)
    if model is None:
        model = _reference_loader().load_reference_model(model_id)
        _MODEL_CACHE[model_id] = model
    return model


# ------------------------------------------------------------------ the goldens


def _float32(inputs: dict) -> dict:
    """The input dict in the reference's own precision.

    The reference is loaded in float32 (Source B's loader leaves `torch_dtype`
    unset, which upcasts the bfloat16 shards), so the golden is computed in
    float32 -- the higher-precision answer a bfloat16 TTNN port should be graded
    against. `tt/inputs.py` already builds float32, so this is normally a no-op;
    it exists so a caller that lowered the dtype cannot silently lower the
    golden too.
    """
    out = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            out[key] = value.to(torch.float32)
        else:
            out[key] = value
    return out


def _depths(model, dual_layers, single_layers):
    full_dual = len(model.transformer_blocks)
    full_single = len(model.single_transformer_blocks)
    dual = full_dual if dual_layers is None else max(1, min(int(dual_layers), full_dual))
    single = full_single if single_layers is None else max(1, min(int(single_layers), full_single))
    return dual, single, full_dual, full_single


def _hf_reference_denoise_step(model, inputs: dict, dual_layers=None, single_layers=None) -> torch.Tensor:
    """CALL 1 GOLDEN -- the velocity prediction, `[B, S_img, 128]`, float32.

    At full depth this is literally `model(...)`: the checkpoint ships no
    `generate()` and no task head other than this forward, so the reference
    callable IS the golden.

    At a capped depth the same chain is run over `transformer_blocks[:dual]` and
    `single_transformer_blocks[:single]`. It is a transcription of
    `Flux2Transformer2DModel.forward` (diffusers 0.37.1,
    `models/transformers/transformer_flux2.py`), with the two `for` loops
    iterating a slice; every other line, including the `timestep * 1000` scaling
    and the text-first RoPE concatenation, is unchanged.
    """
    args = _float32(inputs)
    dual, single, full_dual, full_single = _depths(model, dual_layers, single_layers)

    hidden_states = args["hidden_states"]
    encoder_hidden_states = args["encoder_hidden_states"]
    timestep = args["timestep"]
    img_ids = args["img_ids"]
    txt_ids = args["txt_ids"]

    with torch.no_grad():
        if dual == full_dual and single == full_single:
            return model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                return_dict=False,
            )[0]

        num_txt_tokens = encoder_hidden_states.shape[1]
        t = timestep.to(hidden_states.dtype) * 1000
        temb = model.time_guidance_embed(t, None)

        mod_img = model.double_stream_modulation_img(temb)
        mod_txt = model.double_stream_modulation_txt(temb)
        mod_single = model.single_stream_modulation(temb)

        x = model.x_embedder(hidden_states)
        ctx = model.context_embedder(encoder_hidden_states)

        ids_img = img_ids[0] if img_ids.ndim == 3 else img_ids
        ids_txt = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
        image_rotary_emb = model.pos_embed(ids_img)
        text_rotary_emb = model.pos_embed(ids_txt)
        rope = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        for block in model.transformer_blocks[:dual]:
            ctx, x = block(
                hidden_states=x,
                encoder_hidden_states=ctx,
                temb_mod_img=mod_img,
                temb_mod_txt=mod_txt,
                image_rotary_emb=rope,
                joint_attention_kwargs=None,
            )

        x = torch.cat([ctx, x], dim=1)
        for block in model.single_transformer_blocks[:single]:
            x = block(
                hidden_states=x,
                encoder_hidden_states=None,
                temb_mod=mod_single,
                image_rotary_emb=rope,
                joint_attention_kwargs=None,
            )
        x = x[:, num_txt_tokens:, ...]

        x = model.norm_out(x, temb)
        return model.proj_out(x)


def _hf_reference_denoise_latents(
    model,
    inputs: dict,
    num_inference_steps: int = 4,
    dual_layers=None,
    single_layers=None,
) -> dict:
    """CALL 2 GOLDEN -- the final denoised latents after the Euler loop.

    Identical schedule to the TT side: `tt/inputs.py::sigma_schedule` is called
    once, here, with the same `num_inference_steps` and the same
    `image_seq_len`, and the same update

        lat <- lat + (sigma_next - sigma) * v

    is applied on host in float32. No TT tensor is injected at any joint -- this
    is the whole loop, run independently, which is what makes the final-latents
    PCC a real end-to-end number rather than a one-step one.
    """
    from . import inputs as tt_inputs

    args = _float32(inputs)
    lat = args["hidden_states"].clone()
    image_seq_len = int(lat.shape[-2])
    sigmas = tt_inputs.sigma_schedule(num_inference_steps, image_seq_len)
    steps = len(sigmas) - 1

    per_step = []
    for i in range(steps):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        step_inputs = dict(args)
        step_inputs["hidden_states"] = lat
        step_inputs["timestep"] = torch.full((lat.shape[0],), sigma, dtype=torch.float32)
        velocity = _hf_reference_denoise_step(model, step_inputs, dual_layers=dual_layers, single_layers=single_layers)
        lat = lat + (sigma_next - sigma) * velocity
        per_step.append(lat)

    return {"latents": lat, "per_step": per_step, "sigmas": list(sigmas)}
