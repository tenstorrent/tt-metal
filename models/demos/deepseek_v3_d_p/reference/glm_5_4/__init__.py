# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 CPU reference: only what GLM-5.4 adds to GLM-5.3 (damp gate, two-bank sparse MLA with window carry,
loop schedule) plus a looped stack built on the GLM-5.3 reference. Assumptions A1-A9: ``glm_5_4_config``."""

from models.demos.deepseek_v3_d_p.reference.glm_5_4.loop import (
    Glm54StackResult,
    LoopPass2MLAReference,
    LoopStep,
    Pass2Result,
    glm54_pass2_layer_reference,
    glm54_stack_reference,
    loop_schedule,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4.ops import (
    damp_gate_reference,
    pass1_indices,
    two_bank_sparse_mla,
    window_indices,
)

__all__ = [
    "Glm54StackResult",
    "LoopPass2MLAReference",
    "LoopStep",
    "Pass2Result",
    "damp_gate_reference",
    "glm54_pass2_layer_reference",
    "glm54_stack_reference",
    "loop_schedule",
    "pass1_indices",
    "two_bank_sparse_mla",
    "window_indices",
]
