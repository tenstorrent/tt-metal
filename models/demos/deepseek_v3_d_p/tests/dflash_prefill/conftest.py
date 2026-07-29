# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.dflash_prefill.reference_kv import DrafterUnavailable
from models.demos.deepseek_v3_d_p.reference.dflash_prefill.reference_kv import (
    drafter_cfg_from_hf as _drafter_cfg_from_hf,
)
from models.demos.deepseek_v3_d_p.reference.dflash_prefill.reference_kv import hf_context_kv as _hf_context_kv
from models.demos.deepseek_v3_d_p.reference.dflash_prefill.reference_kv import load_hf_drafter
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig

HF_ENV = "DFLASH_HF_MODEL"


# helpers
def _load_hf_drafter(load_weights: bool = True):
    """Thin test wrapper over ``reference_kv.load_hf_drafter``: reads ``$DFLASH_HF_MODEL`` and translates a
    'checkpoint unavailable/incomplete' soft failure (``DrafterUnavailable``) into a ``pytest.skip``; a
    genuine build failure (``RuntimeError``) still propagates as a test error. The drafter model *code* is
    always the vendored reference; only config/weights come from ``$DFLASH_HF_MODEL``. With
    ``load_weights=False`` (random mode) no safetensors is loaded — the caller supplies random weights."""
    try:
        return load_hf_drafter(os.environ.get(HF_ENV), load_weights=load_weights)
    except DrafterUnavailable as e:
        pytest.skip(f"set {HF_ENV}=/path/to/Kimi-K2.x-DFlash — {e}")


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
        return _hf_context_kv(hf_drafter, drafter_cfg, ctx, q_len)

    return _run
