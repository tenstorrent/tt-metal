# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 CPU reference: only what GLM-5.4 adds to GLM-5.3 (damp gate, two-bank sparse MLA with window carry,
loop schedule) plus a looped stack built on the GLM-5.3 reference. Assumptions A1-A9: ``glm_5_4_config``.

The schedule (``schedule``) is torch-free and imported eagerly; the ops and the stack load on first use, so device
code that imports only the schedule (TtPrefillTransformer) does not import the CPU reference stack.
"""

import importlib

from models.demos.deepseek_v3_d_p.reference.glm_5_4.schedule import LoopStep, loop_schedule

_LAZY = {
    "Glm54StackResult": "loop",
    "LoopPass2MLAReference": "loop",
    "Pass2Result": "loop",
    "glm54_pass2_layer_reference": "loop",
    "glm54_stack_reference": "loop",
    "damp_gate_reference": "ops",
    "pass1_indices": "ops",
    "two_bank_sparse_mla": "ops",
    "window_indices": "ops",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


__all__ = ["LoopStep", "loop_schedule", *_LAZY]
