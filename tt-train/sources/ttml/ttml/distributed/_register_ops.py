# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Auto-register and monkey-patch selected ttml.ops entry points.

Call ``init_ops()`` to replace the public Python entry points for a curated
set of ops with dispatch-wrapped versions. The raw callables are saved in
``dispatch._RAW_OPS`` so dispatch can still call the underlying C++ kernel
after planning and redistribution.

Only ops that participate in the TP / distributed path are patched.
User-defined ops are not patched: users register a rule and wrap their op
with ``register_op(op_name, raw_callable)`` under their own names (see docs).
"""

from __future__ import annotations

# Import all rule modules so that @register_rule decorators fire
from .rules import (  # noqa: F401
    elementwise,
    matmul,
    normalization,
    attention,
    loss,
    collectives,
)

_initialized = False


def _patch(submodule, attr_name: str, op_name: str):
    """Replace *submodule.attr_name* with a dispatch wrapper and register the raw callable."""
    from .dispatch import register_op

    raw = getattr(submodule, attr_name)
    wrapper = register_op(op_name, raw)
    setattr(submodule, attr_name, wrapper)


def init_ops():
    """Monkey-patch ttml.ops.* with dispatch-wrapped versions.

    Call this function before using distributed dispatch to ensure ops
    go through the layout-aware dispatch layer.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    import ttml

    # -- Linear / matmul --------------------------------------------------------

    _patch(ttml.ops.linear, "linear", "linear")
    _patch(ttml.ops.matmul, "matmul_op", "matmul")

    # -- Binary elementwise ------------------------------------------------------

    _patch(ttml.ops.binary, "add", "add")
    _patch(ttml.ops.binary, "mul", "mul")
    _patch(ttml.ops.binary, "sub", "sub")
    _patch(ttml.ops.binary, "div", "div")

    # -- Unary elementwise -------------------------------------------------------

    _patch(ttml.ops.unary, "relu", "relu")
    _patch(ttml.ops.unary, "gelu", "gelu")
    _patch(ttml.ops.unary, "silu", "silu")

    # -- Normalization -----------------------------------------------------------

    _patch(ttml.ops.rmsnorm, "rmsnorm", "rmsnorm")
    _patch(ttml.ops.rmsnorm, "rmsnorm_composite", "rmsnorm_composite")
    _patch(ttml.ops.layernorm, "layernorm", "layernorm")
    _patch(ttml.ops.layernorm, "composite_layernorm", "composite_layernorm")

    # -- Attention ---------------------------------------------------------------

    _patch(ttml.ops.attention, "scaled_dot_product_attention", "sdpa")
    _patch(
        ttml.ops.distributed, "ring_attention_sdpa", "ring_sdpa"
    )  # Same rule as sdpa, distinct name for trace

    # -- Embedding ---------------------------------------------------------------

    _patch(ttml.ops.embedding, "embedding", "embedding")

    # -- Multi-head utils --------------------------------------------------------

    _patch(
        ttml.ops.multi_head_utils, "grouped_heads_creation", "grouped_heads_creation"
    )
    _patch(ttml.ops.multi_head_utils, "heads_fusion", "heads_fusion")
    _patch(ttml.ops.multi_head_utils, "heads_creation", "heads_creation")

    # -- RoPE -------------------------------------------------------------------

    _patch(ttml.ops.rope, "rope", "rope")

    # -- Reshape -----------------------------------------------------------------

    _patch(ttml.ops.reshape, "reshape", "reshape")

    # -- Dropout -----------------------------------------------------------------

    _patch(ttml.ops.dropout, "dropout", "dropout")

    # -- Loss --------------------------------------------------------------------

    _patch(ttml.ops.loss, "cross_entropy_loss", "cross_entropy_loss")

    _patch(ttml.ops.distributed, "broadcast", "broadcast")
    _patch(ttml.ops.distributed, "all_gather", "all_gather")
    _patch(ttml.ops.distributed, "all_reduce", "all_reduce")
    _patch(ttml.ops.distributed, "scatter", "scatter")
