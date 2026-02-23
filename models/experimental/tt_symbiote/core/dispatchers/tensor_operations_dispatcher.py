# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN operation dispatch handlers and mapping."""


from models.experimental.tt_symbiote.core.dispatchers.default_dispatcher import (
    can_dispatch_to_ttnn as default_can_dispatch_to_ttnn,
)
from models.experimental.tt_symbiote.core.dispatchers.default_dispatcher import (
    handle_add,
    handle_addmm,
    handle_bmm,
    handle_cat,
    handle_div,
    handle_expand,
    handle_mul,
    handle_neg,
    handle_permute,
    handle_power,
    handle_slice,
    handle_squeeze,
    handle_stack,
    handle_sub,
    handle_transpose,
    handle_unsqueeze,
    handle_view,
)

# ========== Helper Functions ==========


# Mapping of ATen operations to TTNN handlers
func_to_ttnn_compatible = {
    "aten::view": handle_view,
    "aten::transpose.int": handle_transpose,
    "aten::mul.Tensor": handle_mul,
    "aten::sub.Tensor": handle_sub,
    "aten::div.Tensor": handle_div,
    "aten::slice.Tensor": handle_slice,
    "aten::neg": handle_neg,
    "aten::cat": handle_cat,
    "aten::add.Tensor": handle_add,
    "aten::unsqueeze": handle_unsqueeze,
    "aten::squeeze.dim": handle_squeeze,
    "aten::expand": handle_expand,
    "aten::mul.Scalar": handle_mul,
    "aten::sub.Scalar": handle_sub,
    "aten::add.Scalar": handle_add,
    # "aten::add_.Tensor": handle_add_inplace,
    "aten::bmm": handle_bmm,
    "aten::pow.Tensor_Scalar": handle_power,
    "aten::stack": handle_stack,
    "aten::addmm": handle_addmm,
    "aten::permute": handle_permute,
    "aten::mm": handle_bmm,
}


def can_dispatch_to_ttnn(func_name: str, args=None, kwargs=None) -> bool:
    """Check if operation can be dispatched to TTNN backend."""
    return func_name in func_to_ttnn_compatible and default_can_dispatch_to_ttnn(func_name, args, kwargs)


def dispatch_to_ttnn(func_name, args, kwargs):
    """Dispatch operation to TTNN handler."""
    return func_to_ttnn_compatible[func_name](func_name, args, kwargs)
