# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Context managers for training control flow.

``empty_init``
    Skip tensor value initialisation during model construction — all
    weight/bias tensors are allocated directly on device via ``ttnn.empty``
    (no CPU data, no host→device transfer, no tilisation).  Use when you
    plan to load pretrained weights immediately after construction::

        with empty_init():
            model = Qwen3ForCausalLM(config)
        load_weights_from_hf(model, state_dict, config)
"""


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
