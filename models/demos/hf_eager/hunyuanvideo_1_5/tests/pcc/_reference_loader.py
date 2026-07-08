# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reference-model loader for ``tencent/HunyuanVideo-1.5`` (text/image-to-video DiT).

Why a custom loader is needed
------------------------------
``tencent/HunyuanVideo-1.5`` is **not** a ``transformers`` model. Its
``config.json`` has no ``model_type`` / ``auto_map`` key, so
``AutoConfig`` / ``AutoModel.from_pretrained`` raise
"Unrecognized model" and the generic PCC harness fallback fails. The repo
ships the model in Tencent's **native ``hyvideo`` format** (the transformer
``config.json`` carries ``_class_name: "HunyuanVideo_1_5_DiffusionTransformer"``
from ``hyvideo.models.transformers.hunyuanvideo_1_5_transformer`` — NOT a
diffusers-format checkpoint), and the weights are sharded across many task
variants (``transformer/720p_t2v``, ``transformer/480p_i2v``, ...).

The tt-metal port — and therefore every per-component PCC test and every
``_stubs/*.py`` in this bring-up — was planned against the **diffusers**
re-implementation ``diffusers.HunyuanVideo15Transformer3DModel`` (the stub /
component names ``hunyuan_video15_transformer_block``, ``ada_layer_norm_zero``,
``pix_art_alpha_text_projection``, ``combined_timestep_text_proj_embeddings``,
etc. are diffusers module class names, and the candidate submodule paths
— ``transformer_blocks.0``, ``context_embedder.token_refiner.refiner_blocks.0``,
``x_embedder``, ``rope``, ``norm_out`` — are diffusers attribute paths).
So the correct reference architecture is the diffusers class, loaded with the
model's **real shipped config** (read from the HF cache).

Weight-loading strategy (documented limitation — strategy #5)
-------------------------------------------------------------
The real weights cannot be loaded into the diffusers class here:

  * The shipped checkpoints are in the native ``hyvideo`` key layout, which is
    not key-compatible with ``HunyuanVideo15Transformer3DModel`` and has no
    in-tree diffusers conversion for the ``tencent/HunyuanVideo-1.5`` repo.
  * The full model (``num_layers=54`` at ``hidden_size=2048``) is ~8B params
    (~32 GB in fp32) — infeasible to instantiate on the CPU-only torch build
    used for bring-up, and the weights are not present locally regardless.

This is **valid for the per-component structural PCC** these tests run: each
test extracts ONE submodule (e.g. ``transformer_blocks.0``) and compares the
ttnn port against this reference, where the ttnn port copies its weights FROM
this very module. Both sides therefore use identical (random-but-deterministic)
weights, so the comparison validates the ttnn op implementation, not the
trained values. To keep the module buildable on CPU while preserving the exact
per-block structure, the two *repeated* stacks are shrunk to a small count
(``transformer_blocks`` and ``token_refiner.refiner_blocks``); every tested
submodule is structurally identical to the real 54-layer checkpoint. All other
config values are taken verbatim from the repo's real ``transformer`` config.

The loader is import-safe (no work at import time) and deterministic (weights
seeded, global RNG state restored so downstream test input generation is
unperturbed).
"""

from __future__ import annotations

import json
import os

import torch

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"

# The transformer variant whose config we read. The repo ships several
# task/resolution variants that share the same block structure; 720p_t2v is
# the canonical base text-to-video transformer.
_CONFIG_SUBFOLDERS = (
    "transformer/720p_t2v",
    "transformer/480p_t2v",
    "transformer/720p_i2v",
    "transformer/480p_i2v",
)

# Repeated-stack depths are shrunk from the real values (num_layers=54,
# num_refiner_layers=2) so the module fits in CPU memory. Per-component PCC
# only ever resolves index 0 of each stack; every block is structurally
# identical to the real checkpoint. Override via env if a deeper model is
# needed (WARNING: num_layers=54 is ~8B params / ~32 GB fp32).
_REF_NUM_LAYERS = int(os.environ.get("TT_HY15_REF_NUM_LAYERS", "2"))
_REF_NUM_REFINER_LAYERS = int(os.environ.get("TT_HY15_REF_NUM_REFINER_LAYERS", "2"))

# Deterministic seed for the random reference weights.
_REF_SEED = int(os.environ.get("TT_HY15_REF_SEED", "0"))

# Fallback native (hyvideo-format) transformer config, verbatim from
# ``transformer/720p_t2v/config.json`` of the repo. Used only if the cached
# config cannot be read (keeps the loader offline-safe and self-contained).
_FALLBACK_NATIVE_CONFIG = {
    "heads_num": 16,
    "hidden_size": 2048,
    "mm_double_blocks_depth": 54,
    "in_channels": 32,
    "out_channels": 32,
    "mlp_width_ratio": 4,
    "patch_size": [1, 1, 1],
    "qk_norm_type": "rms",
    "text_states_dim": 3584,
    "text_states_dim_2": None,
    "vision_states_dim": 1152,
    "rope_theta": 256,
    "rope_dim_list": [16, 56, 56],
    "ideal_task": "t2v",
    "use_meanflow": False,
}


def _read_native_config(model_id: str) -> dict:
    """Return the repo's real (native ``hyvideo``-format) transformer config.

    Reads from the local HF cache (offline); falls back to the pinned native
    config if no variant is cached."""
    try:
        from huggingface_hub import hf_hub_download

        for subfolder in _CONFIG_SUBFOLDERS:
            try:
                path = hf_hub_download(model_id, "config.json", subfolder=subfolder, local_files_only=True)
                with open(path) as fh:
                    return json.load(fh)
            except Exception:
                continue
    except Exception:
        pass
    return dict(_FALLBACK_NATIVE_CONFIG)


def _map_to_diffusers_config(native: dict) -> dict:
    """Map the native ``hyvideo`` transformer config to the kwargs accepted by
    ``diffusers.HunyuanVideo15Transformer3DModel``."""
    heads = int(native.get("heads_num", 16))
    hidden = int(native.get("hidden_size", 2048))
    head_dim = hidden // heads

    patch = native.get("patch_size", [1, 1, 1])
    if isinstance(patch, int):
        patch = [patch, patch, patch]

    rope_dim = list(native.get("rope_dim_list") or [16, 56, 56])
    qk_norm_type = str(native.get("qk_norm_type", "rms")).lower()
    task = str(native.get("ideal_task", "t2v")).lower()

    cfg = dict(
        in_channels=int(native.get("in_channels", 32)),
        out_channels=int(native.get("out_channels", 32)),
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        num_layers=_REF_NUM_LAYERS,
        num_refiner_layers=_REF_NUM_REFINER_LAYERS,
        mlp_ratio=float(native.get("mlp_width_ratio", 4)),
        patch_size=int(patch[-1]),
        patch_size_t=int(patch[0]),
        qk_norm="rms_norm" if qk_norm_type.startswith("rms") else "layer_norm",
        text_embed_dim=int(native.get("text_states_dim", 3584)),
        image_embed_dim=int(native.get("vision_states_dim", 1152)),
        rope_theta=float(native.get("rope_theta", 256.0)),
        rope_axes_dim=tuple(rope_dim),
        task_type="t2v" if task.startswith("t2v") else "i2v",
        use_meanflow=bool(native.get("use_meanflow", False)),
    )

    # Native config leaves text_states_dim_2 null; the byT5 projection dim
    # defaults to byt5-small's d_model (1472) in the diffusers class. Only
    # override when the repo config provides a concrete positive dim.
    t2 = native.get("text_states_dim_2")
    if isinstance(t2, int) and t2 > 0:
        cfg["text_embed_2_dim"] = t2

    return cfg


def load_reference_model(model_id: str):
    """Return an ``nn.Module`` (eval mode) equivalent in structure to the HF
    reference transformer for ``tencent/HunyuanVideo-1.5``.

    The module is the diffusers ``HunyuanVideo15Transformer3DModel`` — the same
    architecture the tt-metal port and every per-component stub target — built
    from the repo's real shipped config with deterministic random weights and
    the two repeated block stacks shrunk for CPU feasibility (see module
    docstring for the full rationale)."""
    from diffusers import HunyuanVideo15Transformer3DModel

    native = _read_native_config(model_id or HF_MODEL_ID)
    diffusers_cfg = _map_to_diffusers_config(native)

    # Deterministic weights; restore the global RNG afterwards so the test's
    # own seeded input generation (which runs after this call) is unaffected.
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(_REF_SEED)
        model = HunyuanVideo15Transformer3DModel(**diffusers_cfg)
    finally:
        torch.set_rng_state(rng_state)

    model.eval()
    model.requires_grad_(False)
    return model


if __name__ == "__main__":
    # Self-check: build the module and run a real forward.
    m = load_reference_model(HF_MODEL_ID)
    import torch as _t

    assert isinstance(m, _t.nn.Module)
    assert not m.training, "model must be in eval mode"
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[self-check] built {type(m).__name__} with {n_params / 1e6:.1f}M params (eval={not m.training})")

    with _t.no_grad():
        out = m(
            hidden_states=_t.randn(1, m.config.in_channels, 2, 4, 4),
            timestep=_t.tensor([500.0]),
            encoder_hidden_states=_t.randn(1, 8, m.config.text_embed_dim),
            encoder_attention_mask=_t.ones(1, 8, dtype=_t.long),
            encoder_hidden_states_2=_t.randn(1, 4, m.config.text_embed_2_dim),
            encoder_attention_mask_2=_t.ones(1, 4, dtype=_t.long),
            image_embeds=_t.randn(1, 5, m.config.image_embed_dim),
            return_dict=False,
        )
    print(f"[self-check] full forward OK -> output shape {tuple(out[0].shape)}")
