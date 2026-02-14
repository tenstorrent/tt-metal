# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field


# Perf runs should surface when TTNN program configs are not actually being used.
# Keep strictness as a simple module-level flag so runners can toggle behavior
# without threading config objects throughout the model code.
_STRICT_PROGRAM_CONFIG: bool = False


@dataclass
class DPTPerfCounters:
    """
    Lightweight, process-local counters for catching silent CPU/host fallbacks.

    These are intentionally simple (no threading guarantees). The DPT demo
    runner/eval scripts run single-threaded.
    """

    vit_backbone_fallback_count: int = 0
    reassembly_readout_fallback_count: int = 0
    upsample_host_fallback_count: int = 0
    # Stage-2 hybrid attention bookkeeping: SDPA currently requires interleaved
    # operands, so we explicitly interleave/reshard around the SDPA "island".
    attn_island_interleave_count: int = 0
    attn_island_reshard_count: int = 0
    attn_island_interleave_ms_total: float = 0.0
    attn_island_reshard_ms_total: float = 0.0
    ln_island_interleave_count: int = 0
    ln_island_reshard_count: int = 0
    ln_island_interleave_ms_total: float = 0.0
    ln_island_reshard_ms_total: float = 0.0
    program_config_fallback_total: int = 0
    program_config_fallback_by_op: dict[str, int] = field(default_factory=dict)
    program_config_fallback_by_reason: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.vit_backbone_fallback_count = 0
        self.reassembly_readout_fallback_count = 0
        self.upsample_host_fallback_count = 0
        self.attn_island_interleave_count = 0
        self.attn_island_reshard_count = 0
        self.attn_island_interleave_ms_total = 0.0
        self.attn_island_reshard_ms_total = 0.0
        self.ln_island_interleave_count = 0
        self.ln_island_reshard_count = 0
        self.ln_island_interleave_ms_total = 0.0
        self.ln_island_reshard_ms_total = 0.0
        self.program_config_fallback_total = 0
        self.program_config_fallback_by_op.clear()
        self.program_config_fallback_by_reason.clear()

    def snapshot(self) -> dict[str, object]:
        return {
            "vit_backbone_fallback_count": int(self.vit_backbone_fallback_count),
            "reassembly_readout_fallback_count": int(self.reassembly_readout_fallback_count),
            "upsample_host_fallback_count": int(self.upsample_host_fallback_count),
            "attn_island_interleave_count": int(self.attn_island_interleave_count),
            "attn_island_reshard_count": int(self.attn_island_reshard_count),
            "attn_island_interleave_ms_total": float(self.attn_island_interleave_ms_total),
            "attn_island_reshard_ms_total": float(self.attn_island_reshard_ms_total),
            "ln_island_interleave_count": int(self.ln_island_interleave_count),
            "ln_island_reshard_count": int(self.ln_island_reshard_count),
            "ln_island_interleave_ms_total": float(self.ln_island_interleave_ms_total),
            "ln_island_reshard_ms_total": float(self.ln_island_reshard_ms_total),
            "program_config_fallback_total": int(self.program_config_fallback_total),
            # Nested dicts are JSON-serializable and keep the top-level keys stable.
            "program_config_fallback_by_op": dict(self.program_config_fallback_by_op),
            "program_config_fallback_by_reason": dict(self.program_config_fallback_by_reason),
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


def inc_upsample_host_fallback() -> int:
    PERF_COUNTERS.upsample_host_fallback_count += 1
    return int(PERF_COUNTERS.upsample_host_fallback_count)


def inc_attn_island_interleave(ms: float) -> int:
    PERF_COUNTERS.attn_island_interleave_count += 1
    PERF_COUNTERS.attn_island_interleave_ms_total += float(ms)
    return int(PERF_COUNTERS.attn_island_interleave_count)


def inc_attn_island_reshard(ms: float) -> int:
    PERF_COUNTERS.attn_island_reshard_count += 1
    PERF_COUNTERS.attn_island_reshard_ms_total += float(ms)
    return int(PERF_COUNTERS.attn_island_reshard_count)


def inc_ln_island_interleave(ms: float) -> int:
    PERF_COUNTERS.ln_island_interleave_count += 1
    PERF_COUNTERS.ln_island_interleave_ms_total += float(ms)
    return int(PERF_COUNTERS.ln_island_interleave_count)


def inc_ln_island_reshard(ms: float) -> int:
    PERF_COUNTERS.ln_island_reshard_count += 1
    PERF_COUNTERS.ln_island_reshard_ms_total += float(ms)
    return int(PERF_COUNTERS.ln_island_reshard_count)


def set_strict_program_config(strict: bool) -> None:
    global _STRICT_PROGRAM_CONFIG
    _STRICT_PROGRAM_CONFIG = bool(strict)


def strict_program_config_enabled() -> bool:
    return bool(_STRICT_PROGRAM_CONFIG)


def inc_program_config_fallback(*, op: str, reason: str) -> int:
    op = str(op) if op is not None else "unknown"
    reason = str(reason) if reason is not None else "unknown"
    PERF_COUNTERS.program_config_fallback_total += 1
    PERF_COUNTERS.program_config_fallback_by_op[op] = int(PERF_COUNTERS.program_config_fallback_by_op.get(op, 0)) + 1
    PERF_COUNTERS.program_config_fallback_by_reason[reason] = (
        int(PERF_COUNTERS.program_config_fallback_by_reason.get(reason, 0)) + 1
    )
    return int(PERF_COUNTERS.program_config_fallback_total)
