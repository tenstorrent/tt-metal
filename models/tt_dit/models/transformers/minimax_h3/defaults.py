# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Default values of the MiniMax-H3 switches: the measured-best configuration on a 4x8 Blackhole Galaxy.

Each switch's own comment (where it is read) describes what it does; there "default off" means the switch's
code-level default. This module turns the best set on with `os.environ.setdefault` at import, so an explicit value in
the environment, including "0" or "", still wins (e.g. `MINIMAX_H3_LOFI=` keeps full fidelity). Modules that read
these switches import this module first. The sparse ring op reads its `TT_SPARSE_RING_*` switches from the process
environment at program build, after this has run.

`MINIMAX_H3_MK_E2B_QK_NORM` depends on the length and is set by the pipeline (see `e2b_default`).
"""

from __future__ import annotations

import os

DEFAULTS: dict[str, str] = {
    # Pipeline (host side, bit-identical)
    "MINIMAX_H3_TRACE_DENOISE": "1",
    "MINIMAX_H3_MODULATION_CACHE": "8",
    "MINIMAX_H3_LATENT_READBACK_SLICE": "1",
    "MINIMAX_H3_PREAMBLE_OVERLAP": "1",
    "MINIMAX_H3_REFINER_PERSISTENT": "1",
    "MINIMAX_H3_PER_FORWARD_ONCE": "1",
    "MINIMAX_H3_LATENT_UPLOAD_REAL_ROWS": "1",
    "MINIMAX_H3_RUNG_VIDEO_ARENA": "1",
    "MINIMAX_H3_LAYOUT_CACHE": "1",
    "MINIMAX_H3_LEVELS_ONCE": "1",
    "MINIMAX_H3_O2_SLOT_TRACES": "1",
    "MINIMAX_H3_AUDIO_BESIDE_VIDEO": "1",
    "MINIMAX_H3_STEP_SYNC": "0",
    "MINIMAX_H3_ENCODER_TP8": "1",
    # Precision: LoFi matmuls with 5-bit rounded weights, 8-bit activations where measured
    "MINIMAX_H3_LOFI": "all",
    "MINIMAX_H3_LOFI_ROUND_WEIGHTS": "1",
    "MINIMAX_H3_TO_OUT_ROW_PARALLEL": "1",
    "MINIMAX_H3_TO_OUT_BFP8_WEIGHT": "1",
    "MINIMAX_H3_BF8_NATIVE_QK": "1",
    "MINIMAX_H3_BF8_V_BEFORE_SPLIT": "1",
    "MINIMAX_H3_BF8_POOLED_V": "1",
    "MINIMAX_H3_FF1_OUT_BF8": "1",
    # Sparse ring attention
    "MINIMAX_H3_SPARSE_RING": "1",
    "MINIMAX_H3_SPARSE_RING_DTYPE": "bf8",
    "MINIMAX_H3_SOL_PREFIX_SPREAD": "1",
    "MINIMAX_H3_SOL_PREFIX_WEIGHTS": "1",
    "MINIMAX_H3_SOL_RUNTIME_LEN": "1",
    "TT_SPARSE_RING_Q_PERM": "1",
    "TT_SPARSE_RING_FAST_PBIAS": "2",
    "TT_SPARSE_RING_DYN_SPLIT": "on",
    "TT_SPARSE_RING_DYN_EXTRA_PASS": "1",
    "TT_SPARSE_RING_KV_PAGES": "max",
    "TT_SPARSE_RING_MEMBER_RESCALE": "1",
    "TT_SPARSE_RING_MULTIPASS_PAIRS": "1",
    "TT_SPARSE_RING_BANKED_READS": "1",
    # Fused layouts in the DiT block
    "MINIMAX_H3_MK_S4_GATE_TILEROW": "1",
    "MINIMAX_H3_MK_S2L_ONE_GATHER": "1",
    "MINIMAX_H3_MK_S1_TOKEN_MAJOR_O": "layout",
    "MINIMAX_H3_MK_E2A_HEAD_SPLIT": "1",
    # Matmul blockings
    "MINIMAX_H3_FF2_MMRS_BLOCKING": "1",
    "MINIMAX_H3_TO_OUT_MMRS_BLOCKING": "1",
}

for _name, _value in DEFAULTS.items():
    os.environ.setdefault(_name, _value)


def e2b_default(rows_per_device: int, qkv_n_per_device: int) -> str:
    """Default of MINIMAX_H3_MK_E2B_QK_NORM for a pipeline whose shortest rung has `rows_per_device` rows: on where the
    qkv matmul already runs on its non-transposed grid (M <= N, e.g. 5 s on 4x8). Longer lengths keep it off: the fused
    norm moves qkv off its transposed pin there, which costs more than the norm saves."""
    return "1" if rows_per_device <= qkv_n_per_device else "0"
