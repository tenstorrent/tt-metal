# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared, test-framework-free helpers to build the REAL HF ``DFlashDraftModel`` and run its context-KV
forward — the ground truth the device drafter is PCC'd against.

These were factored out of ``tests/dflash_prefill/conftest.py`` so BOTH the pytest integration test and the
production prefill runtime (``TtPrefillRuntime.dflash_pcc_check``, issue #49586) can reach them. Nothing here
imports ``pytest``: ``load_hf_drafter`` RAISES (``DrafterUnavailable`` for a missing/incomplete checkpoint,
``RuntimeError`` for a genuine build failure) instead of calling ``pytest.skip``; the conftest wraps it and
translates ``DrafterUnavailable`` into a skip for the test context."""

import os

import torch
from transformers import AutoConfig

from models.demos.deepseek_v3_d_p.reference.dflash_prefill.dflash import DFlashDraftModel
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig


class DrafterUnavailable(RuntimeError):
    """The DFlash drafter checkpoint is absent or incomplete — a soft/expected condition. Tests translate
    this to a skip; production translates it to a hard error (the drafter was already built from the same
    checkpoint, so reaching it here means a real misconfiguration)."""


def is_drafter(m) -> bool:
    return all(hasattr(m, a) for a in ("fc", "hidden_norm", "layers", "target_layer_ids"))


def normalize_rope_config(config):
    """K2.6 ships its yarn rope under DeepSeek's ``deepseek_yarn`` type in the new ``rope_parameters``
    schema. transformers' ROPE_INIT_FUNCTIONS has no ``deepseek_yarn`` entry, so the reference's rotary
    embedding raises ``KeyError: 'deepseek_yarn'`` at build time. Remap it to the standard ``yarn`` init in
    whichever field this transformers version actually reads (``rope_parameters`` and/or ``rope_scaling``),
    keeping factor/beta*/mscale so yarn's attention_factor still resolves to 1 — numerically equivalent to
    deepseek_yarn here since mscale == mscale_all_dim == 1. Also lift rope_theta to the top level so the base
    (50000, not the 10000 default) isn't lost."""

    def _fix_type(d):
        if isinstance(d, dict):
            for key in ("rope_type", "type"):
                if d.get(key) == "deepseek_yarn":
                    d[key] = "yarn"

    theta = None
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict):
        theta = rp.get("rope_theta")
        _fix_type(rp)  # the field new transformers reads — remap in place

    rs = getattr(config, "rope_scaling", None)
    if isinstance(rs, dict):
        theta = theta or rs.get("rope_theta")
        _fix_type(rs)
    elif isinstance(rp, dict):
        # Older transformers read rope_scaling instead; mirror the (now-fixed) yarn params into it.
        config.rope_scaling = {
            k: rp[k]
            for k in (
                "rope_type",
                "factor",
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
            if k in rp
        }

    if theta is not None:
        config.rope_theta = float(theta)
    return config


def load_hf_drafter(path: str, *, load_weights: bool = True):
    """Build the REAL z-lab DFlashDraftModel (fp32, eager) from the VENDORED reference modeling code
    (``reference/dflash_prefill``) + the checkout's config (+ safetensors when ``load_weights``). The
    model *code* is always the in-repo reference; only config/weights come from ``path``. With
    ``load_weights=False`` (random mode) no safetensors is loaded — the caller supplies random weights.

    Raises ``DrafterUnavailable`` when the checkpoint is missing/incomplete or the built model is not a
    drafter (a soft condition — a skip in tests); ``RuntimeError`` when the model/config genuinely fails to
    build (a hard error in both contexts)."""
    if not path or not os.path.exists(path):
        raise DrafterUnavailable(
            f"DFLASH drafter checkpoint not found: {path!r} (dir with config.json [+ model.safetensors])"
        )
    try:
        config = normalize_rope_config(AutoConfig.from_pretrained(path, trust_remote_code=True))
        model = DFlashDraftModel(config).float().eval()
        if load_weights:
            from safetensors.torch import load_file

            sd = load_file(os.path.join(path, "model.safetensors"))
            missing, _ = model.load_state_dict(sd, strict=False)
    except Exception as e:
        raise RuntimeError(
            f"could not build DFlashDraftModel (reference/dflash_prefill) from {path}: {type(e).__name__}: {e}"
        ) from e

    if load_weights:
        required = ["fc.weight", "hidden_norm.weight"] + [
            f"layers.{i}.self_attn.{p}.weight"
            for i in range(config.num_hidden_layers)
            for p in ("k_proj", "v_proj", "k_norm")
        ]
        absent = [k for k in required if k in missing]
        if absent:
            raise DrafterUnavailable(f"checkpoint missing required drafter tensors, e.g. {absent[:3]}")

    if not is_drafter(model):
        raise DrafterUnavailable("built model is not a DFlashDraftModel (missing fc/hidden_norm/target_layer_ids)")
    model.config._attn_implementation = "eager"  # force eager so the synthetic forward runs on CPU
    return model


def drafter_cfg_from_hf(c) -> DFlashDrafterConfig:
    """Build the device config from the HF model's config so dims + rope params match the checkpoint."""
    rs = dict(getattr(c, "rope_scaling", None) or getattr(c, "rope_parameters", None) or {})
    dfc = dict(getattr(c, "dflash_config", None) or {})
    d = DFlashDrafterConfig()  # defaults fill anything the config omits
    return DFlashDrafterConfig(
        hidden_size=c.hidden_size,
        head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        num_hidden_layers=c.num_hidden_layers,
        rms_norm_eps=c.rms_norm_eps,
        target_layer_ids=tuple(dfc.get("target_layer_ids", d.target_layer_ids)),
        rope_theta=float(rs.get("rope_theta") or getattr(c, "rope_theta", None) or d.rope_theta),
        rope_factor=float(rs.get("factor", d.rope_factor)),
        rope_beta_fast=float(rs.get("beta_fast", d.rope_beta_fast)),
        rope_beta_slow=float(rs.get("beta_slow", d.rope_beta_slow)),
        rope_orig_max_pos=int(rs.get("original_max_position_embeddings", d.rope_orig_max_pos)),
        rope_mscale=float(rs.get("mscale", d.rope_mscale)),
        rope_mscale_all_dim=float(rs.get("mscale_all_dim", d.rope_mscale_all_dim)),
    )


def cache_kv(pkv, i):
    """Pull layer i's (key, value) from a DynamicCache across transformers API variants."""
    if hasattr(pkv, "key_cache") and len(pkv.key_cache) > i:
        return pkv.key_cache[i], pkv.value_cache[i]
    if hasattr(pkv, "layers"):
        return pkv.layers[i].keys, pkv.layers[i].values
    kv = pkv[i]
    return kv[0], kv[1]


@torch.inference_mode()
def hf_context_kv(model, cfg: DFlashDrafterConfig, ctx: torch.Tensor, q_len: int = None):
    """Run the REAL drafter forward and return per-layer (k_ctx, v_ctx) as [kv_heads, ctx_len, head_dim] fp32.

    The context K/V depend only on ``target_hidden`` (shared across layers), so the noise block content
    is irrelevant — zeros suffice — and the forward's noise/attention path need not be numerically
    meaningful for the captured context slice to be correct. ``q_len`` defaults to the drafter's
    ``block_size`` (only affects the noise block that is sliced off).
    """
    from transformers import DynamicCache

    if q_len is None:
        q_len = int(getattr(model.config, "block_size", cfg.block_size))
    ctx_len = ctx.shape[1]
    total = ctx_len + q_len
    noise = torch.zeros(1, q_len, cfg.hidden_size, dtype=ctx.dtype)
    position_ids = torch.arange(total).unsqueeze(0)
    pkv = DynamicCache()
    try:
        model(
            target_hidden=ctx,
            noise_embedding=noise,
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=pkv,
            use_cache=True,
            cache_position=torch.arange(total),
        )
    except Exception as e:
        raise RuntimeError(f"HF drafter forward failed (reference/dflash_prefill): {type(e).__name__}: {e}") from e

    out = {}
    for i in range(cfg.num_hidden_layers):
        k, v = cache_kv(pkv, i)  # [1, kv_heads, total, head_dim]
        out[i] = (k[0, :, :ctx_len, :].float(), v[0, :, :ctx_len, :].float())
    return out
