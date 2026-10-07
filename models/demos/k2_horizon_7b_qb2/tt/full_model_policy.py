# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-6 precision policy for the TP4 full stack.

Layer indices are zero based. The owner-authorized accuracy repair keeps the
optimized decoder's partitioning, residual, cache and collective contracts.
Stage 8 starts from this policy; the precision experiment ledger lives under
doc/full_model/precision. No policy is selected dynamically during inference.
"""

from .optimized_decoder import MatmulGeometry, PrecisionPolicy


def stage6_precision_policy(layer_index: int) -> PrecisionPolicy:
    if not 0 <= layer_index < 36:
        raise ValueError("Expected a K2-Horizon layer index in [0, 36)")
    # Layer 0 repairs repeated-BOS continuation; layers 3..10 preserve the
    # shared-suite free-running answer quality. Both subgroup reversions fail.
    promoted_mlp = layer_index == 0 or 3 <= layer_index <= 10
    return PrecisionPolicy(
        attention="bfloat8_b",
        mlp="bfloat8_b" if promoted_mlp else "bfloat4_b",
        down="bfloat8_b",
        attention_fidelity="HiFi2",
        mlp_fidelity="HiFi2" if promoted_mlp else "LoFi",
        down_fidelity="HiFi2",
        sdpa_fidelity="HiFi2",
        prefill_fp32=True,
        prefill_m_block=4,
        prefill_qkv_activation="bfloat16" if layer_index == 1 else "bfloat8_b",
        prefill_mlp_activation="bfloat8_b",
        qkv_geometry=MatmulGeometry(16, 32, 1, True),
        o_geometry=MatmulGeometry(32, 4, 2, True),
        mlp_geometry=MatmulGeometry(32, 8, 3, True),
        down_geometry=MatmulGeometry(32, 12, 1, True),
    )
