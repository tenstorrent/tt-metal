# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures + helpers for the DFlash drafter tests (issue #49586).

Mirrors the ``deepseek_v3_d_p/tests/conftest.py`` style: env-resolved *resource* fixtures — the real HF
reference drafter, its device config, and its weights — built on top of ``_``-prefixed support helpers
(the deepseek conftest does the same with ``_resolve_*`` / ``_check_pretrained_available``). Both
``test_dflash.py`` (device-vs-HF on a synthetic context feature) and ``test_dflash_prefill_integration.py``
(device-vs-HF on the real 61-layer verifier's hiddens) consume these instead of hand-rolling the setup.

``$DFLASH_HF_MODEL`` must point at a Kimi-*-DFlash checkout dir (``config.json`` [+ ``model.safetensors``
for the pretrained axis]); the modeling *code* is the vendored ``reference/speculative_decoding/dflash/
dflash.py`` — not a ``.py`` in the checkout. Fixtures ``pytest.skip`` cleanly when the model can't be found
or built.

The ``use_pretrained`` fixture parametrizes both tests over ``[random, pretrained]``; the drafter
resources depend on it, so the SAME weights (seeded-random or the real checkpoint) feed BOTH the device
drafter and the HF reference — required for a meaningful PCC.
"""

import os

import pytest
import torch

from models.demos.deepseek_v3_d_p.tt.speculative_decoding.dflash.dflash_drafter_config import DFlashDrafterConfig

HF_ENV = "DFLASH_HF_MODEL"


# --------------------------------------------------------------------------------------- support helpers
def _is_drafter(m) -> bool:
    return all(hasattr(m, a) for a in ("fc", "hidden_norm", "layers", "target_layer_ids"))


def _normalize_rope_config(config):
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


def _load_hf_drafter(load_weights: bool = True):
    """Build the REAL z-lab DFlashDraftModel (fp32, eager) from the VENDORED reference modeling code
    (``reference/speculative_decoding/dflash``) + the checkout's config (+ safetensors when pretrained). The
    model *code* is always the in-repo reference; only config/weights come from ``$DFLASH_HF_MODEL``. With
    load_weights=False (random mode) no safetensors is loaded — the caller supplies random weights."""
    path = os.environ.get(HF_ENV)
    if not path or not os.path.exists(path):
        pytest.skip(f"set {HF_ENV}=/path/to/Kimi-K2.x-DFlash (dir with config.json [+ model.safetensors])")
    try:
        from transformers import AutoConfig

        from models.demos.deepseek_v3_d_p.reference.speculative_decoding.dflash.dflash import DFlashDraftModel

        config = _normalize_rope_config(AutoConfig.from_pretrained(path, trust_remote_code=True))
        model = DFlashDraftModel(config).float().eval()
        if load_weights:
            from safetensors.torch import load_file

            sd = load_file(os.path.join(path, "model.safetensors"))
            missing, _ = model.load_state_dict(sd, strict=False)
            required = ["fc.weight", "hidden_norm.weight"] + [
                f"layers.{i}.self_attn.{p}.weight"
                for i in range(config.num_hidden_layers)
                for p in ("k_proj", "v_proj", "k_norm")
            ]
            absent = [k for k in required if k in missing]
            if absent:
                pytest.skip(f"checkpoint missing required drafter tensors, e.g. {absent[:3]}")
    except Exception as e:  # transformers missing / qwen3 or deepseek_yarn unsupported / build error
        pytest.skip(
            f"could not build DFlashDraftModel (reference/speculative_decoding/dflash): {type(e).__name__}: {e}"
        )

    if not _is_drafter(model):
        pytest.skip("built model is not a DFlashDraftModel (missing fc/hidden_norm/target_layer_ids)")
    model.config._attn_implementation = "eager"  # force eager so the synthetic forward runs on CPU
    return model


def _drafter_cfg_from_hf(c) -> DFlashDrafterConfig:
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


def _random_state_dict(cfg: DFlashDrafterConfig, seed: int = 42) -> dict:
    """Seeded random weights for the 20-tensor prefill subset: proj ~ N(0, initializer_range), norm gains
    = ones (the same seeded-random convention as the deepseek prefill tests). Self-contained; fed
    identically to the HF model and the device."""
    g = torch.Generator().manual_seed(seed)
    H, kv, D, std = cfg.hidden_size, cfg.kv_dim, cfg.head_dim, cfg.initializer_range

    def _lin(out_dim: int, in_dim: int) -> torch.Tensor:
        return (torch.randn(out_dim, in_dim, generator=g) * std).to(torch.bfloat16)

    sd: dict = {
        "fc.weight": _lin(H, cfg.target_feature_size),
        "hidden_norm.weight": torch.ones(H, dtype=torch.bfloat16),
    }
    for i in range(cfg.num_hidden_layers):
        sd[f"layers.{i}.self_attn.k_proj.weight"] = _lin(kv, H)
        sd[f"layers.{i}.self_attn.v_proj.weight"] = _lin(kv, H)
        sd[f"layers.{i}.self_attn.k_norm.weight"] = torch.ones(D, dtype=torch.bfloat16)
    return sd


def _cache_kv(pkv, i):
    """Pull layer i's (key, value) from a DynamicCache across transformers API variants."""
    if hasattr(pkv, "key_cache") and len(pkv.key_cache) > i:
        return pkv.key_cache[i], pkv.value_cache[i]
    if hasattr(pkv, "layers"):
        return pkv.layers[i].keys, pkv.layers[i].values
    kv = pkv[i]
    return kv[0], kv[1]


@torch.inference_mode()
def _hf_context_kv(model, cfg: DFlashDrafterConfig, ctx: torch.Tensor, q_len: int):
    """Run the REAL drafter forward and return per-layer (k_ctx, v_ctx) as [kv_heads, ctx_len, head_dim] fp32.

    The context K/V depend only on ``target_hidden`` (shared across layers), so the noise block content
    is irrelevant — zeros suffice — and the forward's noise/attention path need not be numerically
    meaningful for the captured context slice to be correct.
    """
    from transformers import DynamicCache

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
        pytest.skip(f"HF drafter forward failed (likely a transformers version detail): {type(e).__name__}: {e}")

    out = {}
    for i in range(cfg.num_hidden_layers):
        k, v = _cache_kv(pkv, i)  # [1, kv_heads, total, head_dim]
        out[i] = (k[0, :, :ctx_len, :].float(), v[0, :, :ctx_len, :].float())
    return out


# --------------------------------------------------------------------------------------- fixtures
@pytest.fixture
def use_pretrained(request) -> bool:
    """Weight axis for the drafter (and, in the integration test, the verifier): ``random`` = seeded
    weights, no checkpoint; ``pretrained`` = the real drafter/verifier checkpoints.

    INDIRECT fixture: each test supplies the values via
    ``@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)``
    so the axis is visible in the test's own params (matching ``test_prefill_transformer.py``), while the
    drafter resource fixtures (``hf_drafter``/``drafter_cfg``/``drafter_state_dict``/``hf_context_kv``) can
    still depend on it. A test using those fixtures MUST parametrize ``use_pretrained``."""
    return request.param


@pytest.fixture
def hf_drafter(use_pretrained):
    """The REAL HF ``DFlashDraftModel`` (fp32, eager) with the correct weights: ``pretrained`` → the
    checkpoint safetensors; ``random`` → the seeded 20-tensor subset loaded over HF random-init (only the
    context-KV weights matter). Skips if ``$DFLASH_HF_MODEL`` is unset / unbuildable."""
    model = _load_hf_drafter(load_weights=use_pretrained)
    if not use_pretrained:
        model.load_state_dict(_random_state_dict(_drafter_cfg_from_hf(model.config)), strict=False)
    return model


@pytest.fixture
def drafter_cfg(hf_drafter) -> DFlashDrafterConfig:
    """The device ``DFlashDrafterConfig`` derived from the HF checkpoint's config (dims + rope params)."""
    return _drafter_cfg_from_hf(hf_drafter.config)


@pytest.fixture
def drafter_state_dict(use_pretrained, hf_drafter, drafter_cfg) -> dict:
    """Weights fed to the DEVICE drafter — the SAME as ``hf_drafter`` holds: its ``state_dict()`` when
    pretrained, or the identical seeded ``_random_state_dict`` (same seed) when random."""
    return hf_drafter.state_dict() if use_pretrained else _random_state_dict(drafter_cfg)


@pytest.fixture
def hf_context_kv(hf_drafter, drafter_cfg):
    """Callable ``ctx -> {layer: (k_ctx, v_ctx)}``: runs the real HF drafter forward on the concatenated
    context feature ``[1, seq, n*H]`` and returns the per-layer context K/V slice — the ground truth the
    device drafter is PCC'd against."""

    def _run(ctx, q_len=None):
        if q_len is None:
            q_len = int(getattr(hf_drafter.config, "block_size", drafter_cfg.block_size))
        return _hf_context_kv(hf_drafter, drafter_cfg, ctx, q_len)

    return _run
