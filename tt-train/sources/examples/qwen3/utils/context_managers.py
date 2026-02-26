# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Context managers for training control flow.

``no_grad``
    Temporarily disable gradient tracking, restoring the previous mode on exit.

``empty_init``
    Skip tensor value initialisation during model construction — all
    weight/bias tensors are allocated directly on device via ``ttnn.empty``
    (no CPU data, no host→device transfer, no tilisation).  Use when you
    plan to load pretrained weights immediately after construction::

        with empty_init():
            model = Qwen3ForCausalLM(config)
        load_weights_from_hf(model, state_dict, config)
"""

from contextlib import contextmanager

import ttml


# ------------------------------------------------------------------
# Gradient-mode context manager
# ------------------------------------------------------------------


@contextmanager
def no_grad():
    """Temporarily disable gradient tracking, restoring the previous mode on exit."""
    ctx = ttml.autograd.AutoContext.get_instance()
    prev = ctx.get_gradient_mode() if hasattr(ctx, "get_gradient_mode") else None
    ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    try:
        yield ctx
    finally:
        if prev is not None:
            ctx.set_gradient_mode(prev)
        else:
            ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)


# ------------------------------------------------------------------
# Empty-init context manager
# ------------------------------------------------------------------

_empty_init = False


class empty_init:
    """Context manager that sets the global ``_empty_init`` flag."""

    def __enter__(self):
        global _empty_init
        self._prev = _empty_init
        _empty_init = True
        return self

    def __exit__(self, *args):
        global _empty_init
        _empty_init = self._prev


def is_empty_init():
    return _empty_init
