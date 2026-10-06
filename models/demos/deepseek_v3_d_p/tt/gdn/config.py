# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-program configuration of the GDN layer on the KDA path (design gdn-on-kda §3.5)."""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig

# Local rows of the supported geometries: 640 per SP rank (Galaxy SP8xTP4, LoudBox LB-A / LB-B); 640 and 1280 on
# one SP rank (the 1x4 comparison with the current GDN implementation and single-chip runs).
_SEQUENCE_PARALLEL_ROWS = (640,)
_SINGLE_RANK_ROWS = (640, 1280)


def gdn_program_config(
    *, active_seq_len_local: int, sequence_parallel_size: int, tp_ccl_topology: ttnn.Topology
) -> KDAProgramConfig:
    """Return the GDN program configuration of one supported geometry; reject any other.

    SP > 1 forces the grouped recurrence; two groups of 10 chunks at 640 rows (G = 2) measured 150 us faster than
    one group of 20 at 32 heads on LoudBox (tt_metal_tracker-g1b.5.13; layer-level check g1b.5.15). One SP rank uses
    the direct scan, which beats every grouped schedule at 32 heads (208 us, g1b.5.13). The other fields follow the
    Kimi-K3 production numerics except the FP32 q/k/v convolution accumulation; the tuned projection schedules were
    measured at K3 shapes only, so they stay off.
    """
    if sequence_parallel_size <= 0:
        raise ValueError(f"sequence_parallel_size must be positive, got {sequence_parallel_size}")
    if sequence_parallel_size > 1:
        if active_seq_len_local not in _SEQUENCE_PARALLEL_ROWS:
            raise ValueError(
                f"no GDN recurrence configuration for SP{sequence_parallel_size} at local T={active_seq_len_local}; "
                f"supported: {_SEQUENCE_PARALLEL_ROWS}"
            )
        recurrence = KDARecurrenceProgramConfig(local_scan_strategy="grouped", summary_group_chunks=10)
    else:
        if active_seq_len_local not in _SINGLE_RANK_ROWS:
            raise ValueError(
                f"no GDN recurrence configuration for one SP rank at local T={active_seq_len_local}; "
                f"supported: {_SINGLE_RANK_ROWS}"
            )
        recurrence = KDARecurrenceProgramConfig(local_scan_strategy="direct")
    return KDAProgramConfig(
        recurrence=recurrence,
        qkv_channel_chunk_size=512,
        tp_ccl_topology=tp_ccl_topology,
        gated_rms_output_dtype=ttnn.bfloat16,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        tuned_projection_matmuls=False,
        # A BF16 tap-sum carry perturbed a near-orthogonal q/k pair into a 0.70 output error on 2.4T real text
        # (tt_metal_tracker-g1b.5.19); FP32 accumulation lowers GDN output and state error in every matrix cell.
        fp32_convolution_accumulation=True,
    )
