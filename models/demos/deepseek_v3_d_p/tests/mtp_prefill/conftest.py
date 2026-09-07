# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the GLM-5.2 MTP prefill tests.

Weight axis follows ``tests/dflash_prefill/conftest.py``: an INDIRECT ``use_pretrained`` fixture so
the axis shows up in each test's own params, with the resource fixtures hanging off it.

There is no HF reference *model* to build here (contrast ``_load_hf_drafter``) — GLM-5.2 ships MTP
weights with no MTP code — so the ground truth is the composed CPU reference in
``reference/glm_5_2/mtp.py``. See issue #53533.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.reference import pretrained_mla_weights
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.weights import _dequant_fp8
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config, glm_5_2_hf_config
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.mtp_config import MTPConfig
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import load_mtp_state_dict
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

HF_ENV = "GLM52_HF_MODEL"
DEFAULT_GLM52_PATH = "/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8"


def glm52_checkpoint_path() -> str | None:
    """The GLM-5.2 checkout, from ``$GLM52_HF_MODEL`` or the adapter's default. None if absent."""
    path = os.environ.get(HF_ENV) or DEFAULT_GLM52_PATH
    return path if os.path.exists(os.path.join(path, "config.json")) else None


def random_mtp_state_dict(cfg: MTPConfig, seed: int = 42) -> dict:
    """Seeded random MTP weights, same conventions as the GLM block test's helpers.

    Norm gains ``randn*0.1 + 1`` (``_glm_norm_weight``) and the projection scaled by ``1/sqrt(fan_in)``
    (``_glm_random_*_weights``). ``eh_proj`` is generated in **HF layout** ``[H, 2H]``, un-transposed,
    exactly as the checkpoint stores it, so the random and pretrained legs feed the module identically.
    """
    g = torch.Generator().manual_seed(seed)
    h = cfg.hidden_size

    def _gain():
        return (torch.randn(h, generator=g) * 0.1 + 1.0).to(torch.bfloat16)

    return {
        "enorm": _gain(),
        "hnorm": _gain(),
        "eh_proj": (torch.randn(h, 2 * h, generator=g) * (2 * h) ** -0.5).to(torch.bfloat16),
        "shared_head_norm": _gain(),
    }


@pytest.fixture
def use_pretrained(request) -> bool:
    """Weight axis: ``random`` = seeded weights, no checkpoint; ``pretrained`` = the real GLM-5.2 MTP
    tensors. INDIRECT — every test using the fixtures below MUST parametrize it:
    ``@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)``.
    """
    return request.param


@pytest.fixture
def mtp_cfg(use_pretrained) -> MTPConfig:
    """The device ``MTPConfig``. Pretrained reads (and verifies against) the checkpoint; random builds
    it from ``glm_5_2_hf_config()`` so the leg runs with no checkpoint at all."""
    if not use_pretrained:
        return MTPConfig.from_hf_config(glm_5_2_hf_config())
    path = glm52_checkpoint_path()
    if path is None:
        pytest.skip(f"set {HF_ENV}=/path/to/GLM-5.2 (dir with config.json + MTP weights)")
    return MTPConfig.from_pretrained(path)


@pytest.fixture
def mtp_state_dict(use_pretrained, mtp_cfg) -> dict:
    """The four MTP tensors in HF layout, fed identically to the device module and the CPU reference.

    Pretrained loads only the shards that hold them (four tensors out of 141 shards), so this leg is
    cheap — it does not touch the layer's MLA or its 256 fp8 experts.
    """
    if not use_pretrained:
        return random_mtp_state_dict(mtp_cfg)
    path = glm52_checkpoint_path()
    if path is None:
        pytest.skip(f"set {HF_ENV}=/path/to/GLM-5.2 (dir with config.json + MTP weights)")
    sd = load_mtp_state_dict(path, layer_idx=mtp_cfg.mtp_layer_idx)
    return {k: v.to(torch.bfloat16) for k, v in sd.items()}


_PROJ = ("gate_proj", "up_proj", "down_proj")


def glm_layer_state_dict(config, model_dir: str, layer_idx: int, num_routed_experts: int) -> dict:
    """One GLM decoder layer's real weights, in ``TtPrefillBlock``'s state_dict format.

    Layer 78 -- the MTP layer -- is a *full* GLM block: MLA carrying its own indexer weights, both
    RMSNorm gains, and a 256-expert MoE. The experts and the MLA projections are fp8 with a
    per-128x128-block scale; the device's weight cache stores them dequantized, so the block scale
    is undone here. One set of bf16 tensors then feeds both the device and the CPU reference, which
    is what makes their PCC a statement about the ops rather than about two different weight loads.

    ~19 GiB resident for the experts alone. ``dict.pop`` frees each fp8 tensor as it is converted,
    so the peak is the bf16 result plus one shard, not both formats at full size.
    """
    model_dir = str(model_dir)
    prefix = f"model.layers.{layer_idx}."

    # MLA + indexer through the sparse-MLA loader: it emits the canonical dict that feeds BOTH
    # ttMLA and SparseMLAReference, and unlike the block extractor it carries GLM's indexer tensors.
    weight_map = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(f"{prefix}self_attn.")})
    assert shards, f"no self_attn tensors for layer {layer_idx} in {model_dir}"
    mla_weights = pretrained_mla_weights(
        config, layer=layer_idx, checkpoint_path=[str(Path(model_dir) / s) for s in shards]
    )

    norms = load_hf_state_dict_filtered(model_dir, [f"{prefix}input_layernorm.", f"{prefix}post_attention_layernorm."])
    raw = load_hf_state_dict_filtered(model_dir, [f"{prefix}mlp."])

    def _pop(key):
        t = raw.pop(key)
        if t.dtype == torch.float8_e4m3fn:
            t = _dequant_fp8(t, raw.pop(f"{key}_scale_inv"))
        return t.to(torch.bfloat16)

    return {
        "attn_norm_weight": norms[f"{prefix}input_layernorm.weight"].to(torch.bfloat16),
        "mla_weights": mla_weights,
        "ffn_norm_weight": norms[f"{prefix}post_attention_layernorm.weight"].to(torch.bfloat16),
        "gate_weights": {
            "weight": raw.pop(f"{prefix}mlp.gate.weight").to(torch.bfloat16),
            "e_score_correction_bias": raw.pop(f"{prefix}mlp.gate.e_score_correction_bias").float(),
        },
        "routed_expert_weights": [
            {p: _pop(f"{prefix}mlp.experts.{e}.{p}.weight") for p in _PROJ} for e in range(num_routed_experts)
        ],
        "shared_expert_weights": {p: _pop(f"{prefix}mlp.shared_experts.{p}.weight") for p in _PROJ},
    }


@pytest.fixture
def mtp_layer_state_dict(use_pretrained, mtp_cfg, config_only) -> dict:
    """The MTP layer's own DECODER-block weights: MLA + indexer, both norms, the 256-expert MoE.

    The other, much larger half of ``mtp_state_dict``, which is only the four MTP-specific tensors.
    Loading it is what lets a test be pretrained on both sides of the MTP boundary; it is ~19 GiB
    and several minutes of fp8 dequant, so only a test that needs the real layer should ask for it.
    """
    if not use_pretrained:
        pytest.skip("the MTP layer's real decoder weights exist only in the checkpoint")
    path = glm52_checkpoint_path()
    if path is None:
        pytest.skip(f"set {HF_ENV}=/path/to/GLM-5.2 (dir with config.json + MTP weights)")
    return glm_layer_state_dict(config_only, path, mtp_cfg.mtp_layer_idx, GLM52Config.NUM_ROUTED_EXPERTS)


@pytest.fixture
def mtp_config_and_glm_config(mtp_cfg):
    """``(MTPConfig, glm_hf_config)`` pair with matching hidden size, for tests that need both."""
    config = glm_5_2_hf_config()
    assert config.hidden_size == mtp_cfg.hidden_size == GLM52Config.EMB_SIZE
    return mtp_cfg, config
