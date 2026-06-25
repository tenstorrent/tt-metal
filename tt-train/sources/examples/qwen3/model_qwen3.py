# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 HF weight loading and re-exports.

The model implementation lives in ``ttml.models.qwen3``.  This module provides
``load_weights_from_hf`` for loading HuggingFace checkpoints into the
single-device ``Qwen3`` model, and re-exports symbols consumed by
``model_qwen3_distributed.py``, ``model_factory.py`` and external callers.
"""

import ttml

# Re-export shared components so existing callers (model_qwen3_distributed,
# model_factory, etc.) continue to work with ``from model_qwen3 import ...``
from ttml.models.qwen3 import Qwen3, Qwen3Config  # noqa: F401

# The HF -> ttml weight loader and its helpers now live in the shared
# ttml.models.qwen3.weights module (single source of truth, shared with the GRPO
# example). Re-export them so existing ``from model_qwen3 import ...`` callers
# keep working.
from ttml.models.qwen3.weights import (  # noqa: F401
    load_weights_from_hf,
    torch_to_ttml,
    unpermute_proj_rows,
    unpermute_norm_weights,
    build_weight_mapping_single,
)

# Backwards-compat alias: callers that imported ``Qwen3ForCausalLM`` from this
# module now get the canonical ``Qwen3`` from ``ttml.models.qwen3``.  Tying,
# checkpointing and runner mode are configured through ``Qwen3Config`` (see
# ``create_qwen3_config_from_hf`` and ``utils/model_factory.create_ttml_model``).
Qwen3ForCausalLM = Qwen3


def linear(x, weight, bias=None):
    return ttml.ops.linear.linear(x, weight, bias)
