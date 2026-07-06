# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-local reference loader for ``meituan-longcat/LongCat-Image``.

Why this file exists
--------------------
``meituan-longcat/LongCat-Image`` is a *diffusers text-to-image pipeline*, not a single
``transformers`` checkpoint. Its top-level ``config.json`` is literally
``{"model_name": "LongCat-Image"}`` — no ``model_type`` key — so both
``AutoConfig.from_pretrained`` and ``AutoModel[.ForCausalLM].from_pretrained`` fail
("Unrecognized model ... should have a `model_type` key"). The tt_hw_planner bring-up
capture and every per-component PCC test hit that wall, so the loader-resolver
(``scripts/tt_hw_planner/reference_loader_resolver.py``) asks for this file.

Strategy — #1 in the resolver ladder: load HF/diffusers-native weights with the right class
-------------------------------------------------------------------------------------------
The repo ships ordinary diffusers/transformers safetensors under per-component subfolders
(``model_index.json`` describes the pipeline). Although the class names
``LongCatImagePipeline`` / ``LongCatImageTransformer2DModel`` look bespoke and NO ``.py``
files ship in the repo, the *installed* ``diffusers`` (>=0.38) registers both natively and
``transformers`` (>=5.10) registers ``Qwen2_5_VLForConditionalGeneration`` — so the real
weights load with native classes and ``subfolder=``; no ``trust_remote_code`` custom code is
needed. This is strategy #1 (real, coherent weights), NOT strategy #4 (random init).

Components (from ``model_index.json``):
  * ``text_encoder`` — ``Qwen2_5_VLForConditionalGeneration`` (Qwen2.5-VL 7B, ~16.5 GB bf16)
  * ``transformer``  — ``LongCatImageTransformer2DModel`` (Flux-style MMDiT, 6.27 B params, ~12.5 GB)
  * ``vae``          — ``AutoencoderKL`` (Flux VAE, 16 latent channels, ~168 MB)

What is returned
----------------
A single container ``nn.Module`` (``LongCatImageReference``) whose children are named
``text_encoder`` / ``transformer`` / ``vae`` to match the repo layout, so the PCC harness can
resolve submodule paths such as ``transformer.transformer_blocks.0`` or ``vae.decoder``, and
component discovery can walk the genuine module tree. The container's ``forward`` delegates to
the ``transformer`` (the core denoiser) purely so ``model(...)`` is exercisable; the PCC
harness itself never calls the container's ``forward`` — it resolves and runs a submodule.

Robustness / honesty
--------------------
Each component is loaded with its real weights. If a single component's real load genuinely
fails (e.g. its shards are unreachable), that component *only* falls back to a
random-initialised (but deterministically seeded) build from its own config, so the container
is always structurally complete and the other components keep their real weights. The
per-component provenance is recorded on ``model.reference_provenance`` and printed. Real
weights are the default and expected path here.

Contract: this module is import-safe (no network, no model construction, no global mutation at
import time) and deterministic (eval mode; any random fallback is seeded). The dtype can be
overridden with ``LONGCAT_REFERENCE_DTYPE`` (default ``bfloat16`` to fit host RAM for the
~29 GB of weights); set it to ``float32`` if you have the RAM and want a higher-precision golden.
"""
from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn as nn

_DEFAULT_MODEL_ID = "meituan-longcat/LongCat-Image"
_SEED = 0

_DTYPE_ALIASES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def _resolve_dtype() -> torch.dtype:
    """Pick the load dtype. Default bfloat16 keeps the full ~29 GB of weights within host RAM;
    override via ``LONGCAT_REFERENCE_DTYPE`` (e.g. ``float32``)."""
    raw = os.environ.get("LONGCAT_REFERENCE_DTYPE", "bfloat16").strip().lower()
    return _DTYPE_ALIASES.get(raw, torch.bfloat16)


class LongCatImageReference(nn.Module):
    """Container holding the real LongCat-Image pipeline sub-networks as named children.

    Children (matching the repo's ``model_index.json`` keys):
      * ``text_encoder`` : ``Qwen2_5_VLForConditionalGeneration``
      * ``transformer``  : ``LongCatImageTransformer2DModel`` (the denoiser)
      * ``vae``          : ``AutoencoderKL``
    """

    def __init__(self, text_encoder: nn.Module, transformer: nn.Module, vae: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.vae = vae
        # Provenance: which children carry real weights vs. a random-init fallback. Set by the
        # loader after construction. Plain attribute -> not registered as a module/buffer.
        self.reference_provenance: Dict[str, str] = {}

    def forward(self, *args, **kwargs):
        """Delegate to the core denoiser so a forward on the container is meaningful.

        The tt_hw_planner PCC harness resolves and runs an individual *submodule*, so it never
        relies on this; it exists only so ``model(...)`` is directly exercisable."""
        return self.transformer(*args, **kwargs)


def _load_transformer(model_id: str, dtype: torch.dtype):
    """Real ``LongCatImageTransformer2DModel``; random-init fallback keyed off its own config."""
    from diffusers import LongCatImageTransformer2DModel

    try:
        m = LongCatImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True
        )
        return m, "real"
    except Exception as exc:  # noqa: BLE001
        print(
            f"[longcat-loader] transformer real load failed ({type(exc).__name__}: {exc}); "
            f"falling back to random-init from config"
        )
        cfg = LongCatImageTransformer2DModel.load_config(model_id, subfolder="transformer")
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        torch.manual_seed(_SEED)
        return LongCatImageTransformer2DModel.from_config(cfg).to(dtype), "random(config)"


def _load_vae(model_id: str, dtype: torch.dtype):
    """Real ``AutoencoderKL``; random-init fallback keyed off its own config."""
    from diffusers import AutoencoderKL

    try:
        m = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True)
        return m, "real"
    except Exception as exc:  # noqa: BLE001
        print(
            f"[longcat-loader] vae real load failed ({type(exc).__name__}: {exc}); "
            f"falling back to random-init from config"
        )
        cfg = AutoencoderKL.load_config(model_id, subfolder="vae")
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        torch.manual_seed(_SEED)
        return AutoencoderKL.from_config(cfg).to(dtype), "random(config)"


def _load_text_encoder(model_id: str, dtype: torch.dtype):
    """Real ``Qwen2_5_VLForConditionalGeneration``; random-init fallback keyed off its own config."""
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    try:
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True
        )
        return m, "real"
    except Exception as exc:  # noqa: BLE001
        print(
            f"[longcat-loader] text_encoder real load failed ({type(exc).__name__}: {exc}); "
            f"falling back to random-init from config"
        )
        cfg = AutoConfig.from_pretrained(model_id, subfolder="text_encoder")
        torch.manual_seed(_SEED)
        # Plain constructor (random init via post_init) is version-robust across transformers releases.
        return Qwen2_5_VLForConditionalGeneration(cfg).to(dtype), "random(config)"


def load_reference_model(model_id: str = _DEFAULT_MODEL_ID) -> nn.Module:
    """Return an ``nn.Module`` (eval mode) equivalent to the HF reference for LongCat-Image.

    Loads the three real pipeline sub-networks (``text_encoder``, ``transformer``, ``vae``) with
    their native diffusers/transformers classes and real safetensors weights, wrapped in a single
    container so component discovery can walk the genuine tree and the PCC harness can resolve
    per-component submodules. ``model_id`` may be a HF repo id or a local pipeline directory.
    """
    model_id = model_id or _DEFAULT_MODEL_ID
    dtype = _resolve_dtype()

    transformer, tf_prov = _load_transformer(model_id, dtype)
    vae, vae_prov = _load_vae(model_id, dtype)
    text_encoder, te_prov = _load_text_encoder(model_id, dtype)

    model = LongCatImageReference(text_encoder=text_encoder, transformer=transformer, vae=vae)
    model.reference_provenance = {
        "text_encoder": te_prov,
        "transformer": tf_prov,
        "vae": vae_prov,
    }
    model.eval()
    print(
        f"[longcat-loader] loaded LongCat-Image reference (dtype={dtype}); " f"provenance={model.reference_provenance}"
    )
    return model


if __name__ == "__main__":  # pragma: no cover - manual self-check
    m = load_reference_model(_DEFAULT_MODEL_ID)
    assert isinstance(m, nn.Module)
    assert {"text_encoder", "transformer", "vae"}.issubset(dict(m.named_children()))
    dt = _resolve_dtype()
    with torch.no_grad():
        # VAE decode: cheap end-to-end forward on a tiny latent.
        z = torch.randn(1, m.vae.config.latent_channels, 8, 8, dtype=dt)
        dec = m.vae.decode(z).sample
        print("[self-check] vae.decode ->", tuple(dec.shape))
        # Transformer (container.forward) denoiser step on a tiny Flux-style input.
        # forward: (hidden_states[B,img_len,in_channels], encoder_hidden_states[B,txt_len,joint_dim],
        #           timestep[B], img_ids[img_len,3], txt_ids[txt_len,3]); no pooled projection.
        cfg = m.transformer.config
        img_len, txt_len = 16, 8
        hs = torch.randn(1, img_len, cfg.in_channels, dtype=dt)
        ehs = torch.randn(1, txt_len, cfg.joint_attention_dim, dtype=dt)
        ts = torch.tensor([1.0], dtype=dt)
        img_ids = torch.zeros(img_len, 3)
        txt_ids = torch.zeros(txt_len, 3)
        out = m(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            timestep=ts,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]
        print("[self-check] transformer forward ->", tuple(out.shape))
    print("[self-check] OK", m.reference_provenance)
