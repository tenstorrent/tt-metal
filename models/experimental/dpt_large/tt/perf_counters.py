# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DPTPerfCounters:
    """
    Lightweight, process-local counters for catching silent CPU/host fallbacks.

    These are intentionally simple (no threading guarantees). The DPT demo
    runner/eval scripts run single-threaded.
    """

    vit_backbone_fallback_count: int = 0
    reassembly_readout_fallback_count: int = 0

    def reset(self) -> None:
        self.vit_backbone_fallback_count = 0
        self.reassembly_readout_fallback_count = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "vit_backbone_fallback_count": int(self.vit_backbone_fallback_count),
            "reassembly_readout_fallback_count": int(self.reassembly_readout_fallback_count),
        }


PERF_COUNTERS = DPTPerfCounters()


def reset_perf_counters() -> None:
    PERF_COUNTERS.reset()


def inc_vit_backbone_fallback() -> int:
    PERF_COUNTERS.vit_backbone_fallback_count += 1
    return int(PERF_COUNTERS.vit_backbone_fallback_count)


def inc_reassembly_readout_fallback() -> int:
    PERF_COUNTERS.reassembly_readout_fallback_count += 1
    return int(PERF_COUNTERS.reassembly_readout_fallback_count)

