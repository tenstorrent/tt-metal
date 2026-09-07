# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5-128B dimension constants + the vendored HF config loader.

Every constant here is a transcription of ``configs/Mistral-Medium-3.5-128B/config.json``
(``text_config`` — the text backbone; the Pixtral vision tower is out of scope for prefill), and
``tests/unit/test_reference_config.py`` asserts each one against that file so a drift fails a test
rather than a run. Read dims from here; read TP/SP/seq/dtypes from ``spec.py``.

Architecture, for orientation:
  * dense GQA — 96 Q heads / 8 KV heads / head_dim 128, no QK-norm, no sinks, no sliding window
    (``sliding_window: null`` for every layer), no q/k/v/o bias;
  * dense SwiGLU MLP (silu) at intermediate 28672 — no MoE anywhere;
  * RMSNorm eps 1e-5, plain (no Gemma ``1 + w`` fold);
  * full rotary with YaRN scaling (theta 1e6, factor 64, original_max_position 4096, beta_fast 4,
    beta_slow 1, mscale 1.0 / mscale_all_dim 0.0 — see :mod:`..tt.rope` for what HF makes of those);
  * ``llama_4_scaling_beta: 0`` — the Ministral3 per-position query scale degenerates to x1.0 (see
    ``LLAMA4_SCALING_BETA`` below);
  * fp8 checkpoint, ``activation_scheme: static`` and ``weight_block_size: null`` => PER-TENSOR
    ``weight_scale``, not DeepSeek blockwise ``weight_scale_inv``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "Mistral-Medium-3.5-128B"
CONFIG_JSON = CONFIG_DIR / "config.json"


class MistralMedium35Config:
    """Static model dimensions. Mirrors the ``model_config`` class every prefill adapter exposes."""

    MODEL_NAME = "Mistral-Medium-3.5-128B"
    HF_REPO_ID = "mistralai/Mistral-Medium-3.5-128B"

    # --- text backbone dims ---
    NUM_LAYERS = 88
    HIDDEN_SIZE = 12288
    INTERMEDIATE_SIZE = 28672
    NUM_ATTENTION_HEADS = 96
    NUM_KEY_VALUE_HEADS = 8
    HEAD_DIM = 128
    VOCAB_SIZE = 131072
    RMS_NORM_EPS = 1e-5
    HIDDEN_ACT = "silu"
    MAX_POSITION_EMBEDDINGS = 262144

    # --- attention family flags (all absences, spelled out so a test can assert them) ---
    SLIDING_WINDOW = None  # every layer is full-causal
    ATTENTION_BIAS = False  # q/k/v/o carry no bias
    USE_QK_NORM = False
    HAS_ATTENTION_SINKS = False
    NUM_EXPERTS = 0  # dense: no MoE on any layer

    # --- rope (YaRN) ---
    ROPE_TYPE = "yarn"
    ROPE_THETA = 1000000.0
    YARN_FACTOR = 64.0
    YARN_ORIG_MAX_POS = 4096
    YARN_BETA_FAST = 4.0
    YARN_BETA_SLOW = 1.0
    YARN_MSCALE = 1.0
    YARN_MSCALE_ALL_DIM = 0.0
    # HF reads `rope_parameters["truncate"]` with a default of True; the config carries no such key,
    # so the correction dims ARE floored/ceiled. (The gpt-oss donor sets truncate=False and its
    # comment about keeping float dims does NOT transfer — see tt/rope.py.)
    YARN_TRUNCATE = True
    # Ministral3Attention multiplies Q by 1 + beta*log(1 + floor(pos/orig_max_pos)). beta == 0 here,
    # so the factor is exactly 1.0 for every position and the TT attention omits the term entirely.
    LLAMA4_SCALING_BETA = 0

    # --- checkpoint quantization ---
    QUANT_METHOD = "fp8"
    ACTIVATION_SCHEME = "static"
    WEIGHT_BLOCK_SIZE = None  # null => per-tensor weight_scale
    MODULES_TO_NOT_CONVERT = ("model.vision_tower", "model.multi_modal_projector", "lm_head")

    # --- serving-engine constant (PrefillModelAdapter.model_config contract) ---
    # Bytes of hidden state moved per fabric packet on the D2D activation socket. Single-rank
    # prefill never sends one, but the adapter contract requires the attribute.
    FABRIC_PAYLOAD_SIZE = 4096

    @property
    def num_key_value_groups(self) -> int:
        return self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS


@lru_cache(maxsize=1)
def raw_config(path: str | Path = CONFIG_JSON) -> dict:
    """The vendored ``config.json`` verbatim (the Mistral3 wrapper, vision tower included)."""
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def load_text_config(path: str | Path = CONFIG_DIR):
    """The HF ``Ministral3Config`` for the text backbone, unwrapped from the Mistral3 wrapper.

    ``Mistral3ForConditionalGeneration`` wraps a Ministral3 text model plus a Pixtral vision tower;
    prefill brings up the text backbone only, so every consumer wants ``cfg.text_config``. Returns
    the HF config object (not a dict) because the reference model, the TT modules and the golden
    runner all read attributes off it.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
    return getattr(cfg, "text_config", cfg)


def reduced_text_config(*, num_hidden_layers=None, hidden_size=None, intermediate_size=None, vocab_size=None):
    """A copy of the text config with selected dims shrunk — for host-side reference tests.

    A full 88-layer / 12288-hidden CPU forward is ~121 B parameters, which no host test can build.
    Shrinking layer count and/or width keeps the same code path (same attention family, same rope
    constants, same norm) at a size a test can instantiate. Head count and head_dim are NOT
    shrinkable here: the GQA head layout is the thing under test.
    """
    import copy

    cfg = copy.deepcopy(load_text_config())
    if num_hidden_layers is not None:
        cfg.num_hidden_layers = num_hidden_layers
    if hidden_size is not None:
        cfg.hidden_size = hidden_size
    if intermediate_size is not None:
        cfg.intermediate_size = intermediate_size
    if vocab_size is not None:
        cfg.vocab_size = vocab_size
    return cfg
