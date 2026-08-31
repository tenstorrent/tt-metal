# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Host-side consistency checks for ``KimiK27Config``. No TTNN, no device code, no weights.

``KimiK27Config`` inherits every value from ``KimiK26Config`` because K2.7-Code is architecturally
identical to K2.6. That reuse is the whole reason the K2.7 migration was cheap -- and it is trusted
rather than checked, which is what these two tests fix. K3 has had this guard since it was added
(``test_kimi_k3_mla_reference.test_k3_constants_match_the_vendored_checkpoint_config``); K2.x never
did, so a divergence would have surfaced as an unattributable device PCC miss instead of a one-line
host failure.

Two distinct failure modes, hence two tests:

  * **transcription drift** -- the values are hand-transcribed from a checkpoint config, so a
    mistyped digit would pass the entire suite. ``test_..._vendored_config`` pins them against
    ``reference/kimi_k2_7/config.json``, vendored verbatim from the K2.7-Code checkpoint the same way
    ``reference/kimi_k3/config.json`` is. Hermetic: runs everywhere, needs no mount.

    Note the vendored file is read with plain ``json.load``, never ``AutoConfig``: its ``auto_map``
    names ``configuration_kimi_k25`` modules that are not vendored, so a ``trust_remote_code`` load
    would fail. That is also why ``KimiK27Adapter.hf_model_default`` still resolves to
    ``reference/kimi_k2_6`` -- that dir carries the vendored *modeling* code, which is genuinely
    shared, and only this config file needed a K2.7 copy.
  * **the inheritance premise breaking** -- if a future K2.7 checkpoint revision changes a dimension,
    inheriting K2.6's numbers becomes wrong. ``test_..._staged_checkpoint`` pins it against the real
    staged K2.7 checkpoint. Skips cleanly when that is not mounted, so it is advisory on a dev box
    and load-bearing in CI.

If the second test ever fails, the fix is to override the moved fields on ``KimiK27Config`` -- which
is precisely why it exists as a subclass rather than a bare alias.
"""

import json
import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config

# Default matches KimiK27Adapter.default_local_path. $KIMI_K2_7_HF_MODEL (the adapter's env_var)
# overrides it, so this follows whatever a CI leg or a developer already exported.
K2_7_CHECKPOINT_ENV = "KIMI_K2_7_HF_MODEL"
K2_7_CHECKPOINT_DEFAULT = Path("/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized")

VENDORED_CONFIG = Path(__file__).parents[2] / "reference" / "kimi_k2_7" / "config.json"


def _text_config(path: Path) -> dict:
    """Load a checkpoint config and return the LM fields.

    Both configs read here are the multimodal wrapper (``KimiK25ForConditionalGeneration``), so the
    LM fields live under ``text_config``. The ``.get`` fallback keeps this working against a flat
    text-only config too -- the shape ``reference/kimi_k2_6/config.json`` has -- so the helper is
    reusable if a leg is ever pointed at one.
    """
    with open(path) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def _expected_from(text_config: dict) -> dict:
    """Map ``KimiK27Config`` attribute names onto the checkpoint's own field names."""
    rope = text_config["rope_scaling"]
    return {
        "EMB_SIZE": text_config["hidden_size"],
        "MOE_INTERMEDIATE_SIZE": text_config["moe_intermediate_size"],
        "INTERMEDIATE_SIZE": text_config["intermediate_size"],
        "NUM_ROUTED_EXPERTS": text_config["n_routed_experts"],
        "NUM_EXPERTS_PER_TOKEN": text_config["num_experts_per_tok"],
        "NUM_SHARED_EXPERTS": text_config["n_shared_experts"],
        "NUM_EXPERT_GROUPS": text_config["n_group"],
        "NUM_LIMITED_GROUPS": text_config["topk_group"],
        "ROUTE_SCALE": text_config["routed_scaling_factor"],
        "NUM_LAYERS": text_config["num_hidden_layers"],
        "NUM_DENSE_LAYERS": text_config["first_k_dense_replace"],
        "VOCAB_SIZE": text_config["vocab_size"],
        "NUM_ATTENTION_HEADS": text_config["num_attention_heads"],
        "NUM_KEY_VALUE_HEADS": text_config["num_key_value_heads"],
        "Q_LORA_RANK": text_config["q_lora_rank"],
        "KV_LORA_RANK": text_config["kv_lora_rank"],
        "QK_NOPE_HEAD_DIM": text_config["qk_nope_head_dim"],
        "QK_ROPE_HEAD_DIM": text_config["qk_rope_head_dim"],
        "V_HEAD_DIM": text_config["v_head_dim"],
        "RMS_NORM_EPS": text_config["rms_norm_eps"],
        "ROPE_THETA": text_config["rope_theta"],
        "MAX_POSITION_EMBEDDINGS": text_config["max_position_embeddings"],
        # YaRN. Unlike K3 (NoPE), K2.x must have these -- the mscale factors feed the softmax scale,
        # so a silent change here is a numerics change, not a metadata change.
        "ROPE_SCALING_FACTOR": rope["factor"],
        "ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS": rope["original_max_position_embeddings"],
        "ROPE_SCALING_BETA_FAST": rope["beta_fast"],
        "ROPE_SCALING_BETA_SLOW": rope["beta_slow"],
        "ROPE_SCALING_MSCALE": rope["mscale"],
        "ROPE_SCALING_MSCALE_ALL_DIM": rope["mscale_all_dim"],
    }


def _mismatches(expected: dict) -> dict:
    return {
        name: (getattr(KimiK27Config, name), want)
        for name, want in expected.items()
        if getattr(KimiK27Config, name) != want
    }


def test_k2_7_constants_match_the_vendored_config():
    """Every ``KimiK27Config`` constant equals the vendored ``reference/kimi_k2_7/config.json`` value.

    Catches a transcription slip in the hand-written config class. Hermetic -- no mount, no network.
    """
    mismatched = _mismatches(_expected_from(_text_config(VENDORED_CONFIG)))
    assert not mismatched, f"KimiK27Config disagrees with {VENDORED_CONFIG}: {mismatched}"


def test_k2_7_constants_match_the_staged_checkpoint():
    """Every ``KimiK27Config`` constant equals the staged K2.7-Code checkpoint's value.

    This is the test that guards the migration's central premise: that K2.7 can inherit K2.6's
    numbers. Skips when the checkpoint is not mounted rather than failing, so it does not punish a
    dev box -- but on any machine that runs the prefill suite the mount is present by definition,
    since the weight cache lives on it.
    """
    root = Path(os.environ.get(K2_7_CHECKPOINT_ENV) or K2_7_CHECKPOINT_DEFAULT)
    config_path = root / "config.json"
    if not config_path.is_file():
        pytest.skip(f"K2.7 checkpoint config not staged at {config_path}; set ${K2_7_CHECKPOINT_ENV}")

    mismatched = _mismatches(_expected_from(_text_config(config_path)))
    assert not mismatched, (
        f"KimiK27Config no longer matches the K2.7 checkpoint at {config_path}: {mismatched}. "
        "K2.7 has diverged from K2.6 -- override the moved fields on KimiK27Config rather than "
        "editing KimiK26Config, which K2.6 still depends on."
    )
