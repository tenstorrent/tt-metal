# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatibility re-exports for the Qwen3 GRPO completer.

Everything this module used to define now lives in shared ttml locations and is
re-exported here so existing ``from .qwen3_support import ...`` call sites keep
working:

  - The weight name mapping, HF -> ttml weight conversion, the QK / RoPE row
    permutations, the ``torch_to_ttml`` upload helper and the
    ``load_weights_from_hf`` loader live in :mod:`ttml.models.qwen3.weights`.
  - :func:`build_mesh`, the named-mesh builder shared with the other training
    scripts, lives in :mod:`ttml.common.utils`.
"""

from __future__ import annotations

# Re-exported shared Qwen3 weight helpers (single source of truth in
# ttml.models.qwen3.weights).
from ttml.models.qwen3.weights import (  # noqa: F401
    torch_to_ttml,
    unpermute_proj_rows,
    unpermute_norm_weights,
    build_weight_mapping_single,
    load_weights_from_hf,
)

# Re-exported shared named-mesh builder (single source of truth in
# ttml.common.utils; also used by examples/train/train.py).
from ttml.common.utils import build_mesh  # noqa: F401
