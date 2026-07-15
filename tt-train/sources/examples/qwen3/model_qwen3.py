# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 ``linear`` helper and backwards-compat alias.

The model implementation and HF weight loader live in ``ttml.models.qwen3`` /
``ttml.models.qwen3.weights``; import those directly. This module provides the
``linear`` helper (used by ``model_qwen3_distributed``) and the
``Qwen3ForCausalLM`` alias for callers that predate the rename.
"""

import ttml
from ttml.models.qwen3 import Qwen3

# Backwards-compat alias: callers that imported ``Qwen3ForCausalLM`` from this
# module now get the canonical ``Qwen3`` from ``ttml.models.qwen3``.  Tying,
# checkpointing and runner mode are configured through ``Qwen3Config`` (see
# ``create_qwen3_config_from_hf`` and ``utils/model_factory.create_ttml_model``).
Qwen3ForCausalLM = Qwen3


def linear(x, weight, bias=None):
    return ttml.ops.linear.linear(x, weight, bias)
