# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from transformers import AutoConfig

from models.demos.deepseek_v3_d_p.reference.dflash_prefill.dflash import DFlashDraftModel
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig

HF_ENV = "DFLASH_HF_MODEL"


# HF drafter reference (ground truth for the device PCC): build the real DFlashDraftModel from the vendored
# reference code + checkout config. _load_hf_drafter pytest.skips if the checkpoint is missing, raises on build failure.
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


def _load_hf_drafter(load_weights: bool = True):
    """Build the REAL z-lab DFlashDraftModel (fp32, eager) from the VENDORED reference modeling code
    (``reference/dflash_prefill``) + the checkout's config (+ safetensors when ``load_weights``). The
    model *code* is always the in-repo reference; only config/weights come from ``$DFLASH_HF_MODEL``. With
    ``load_weights=False`` (random mode) no safetensors is loaded — the caller supplies random weights.

    Skips (``pytest.skip``) when the checkpoint is missing/incomplete or the built model is not a drafter
    (a soft/expected condition); raises ``RuntimeError`` when the model/config genuinely fails to build."""
    path = os.environ.get(HF_ENV)
    if not path or not os.path.exists(path):
        pytest.skip(f"set {HF_ENV}=/path/to/Kimi-K2.x-DFlash (dir with config.json [+ model.safetensors])")
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
            pytest.skip(f"checkpoint missing required drafter tensors, e.g. {absent[:3]}")

    if not is_drafter(model):
        pytest.skip("built model is not a DFlashDraftModel (missing fc/hidden_norm/target_layer_ids)")
    model.config._attn_implementation = "eager"  # force eager so the synthetic forward runs on CPU
    return model


def cache_kv(pkv, i):
    """Pull layer i's (key, value) from a DynamicCache across transformers API variants."""
    if hasattr(pkv, "key_cache") and len(pkv.key_cache) > i:
        return pkv.key_cache[i], pkv.value_cache[i]
    if hasattr(pkv, "layers"):
        return pkv.layers[i].keys, pkv.layers[i].values
    kv = pkv[i]
    return kv[0], kv[1]


@torch.inference_mode()
def _hf_context_kv(model, cfg: DFlashDrafterConfig, ctx: torch.Tensor, q_len: int = None):
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


# fixtures
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
        model.load_state_dict(_random_state_dict(DFlashDrafterConfig.from_hf_config(model.config)), strict=False)
    return model


@pytest.fixture
def drafter_cfg(hf_drafter) -> DFlashDrafterConfig:
    """The device ``DFlashDrafterConfig`` derived from the HF checkpoint's config (dims + rope params)."""
    return DFlashDrafterConfig.from_hf_config(hf_drafter.config)


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
        return _hf_context_kv(hf_drafter, drafter_cfg, ctx, q_len)

    return _run
