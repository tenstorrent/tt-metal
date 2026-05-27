# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug-trace helpers for staged per-module PCC analysis of Voxtral TTS."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger

from models.common.utility_functions import comp_pcc


class VoxtralTTSDebugTrace:
    """Lightweight key→tensor store collected during a single forward pass."""

    def __init__(self) -> None:
        self._data: dict[str, torch.Tensor] = {}

    def set(self, key: str, value: torch.Tensor) -> None:
        self._data[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._data


@dataclass
class StagedPCCReport:
    first_low_stage: str | None = None
    first_semantic_argmax_mismatch_step: int | None = None


def _log_pcc_entry(
    label: str,
    ref_t: torch.Tensor,
    tt_t: torch.Tensor,
    target: float,
    report: StagedPCCReport,
) -> None:
    try:
        ref_f = ref_t.reshape(-1).float()
        tt_f = tt_t.reshape(-1).float()
        n = min(ref_f.numel(), tt_f.numel())
        _, pcc = comp_pcc(ref_f[:n], tt_f[:n], pcc=0.0)
        pcc_val = float(pcc)
        status = "PASS" if pcc_val >= target else "LOW"
        logger.info(f"  {label}: PCC={pcc_val:.4f}  target>={target:.4f}  [{status}]")
        if pcc_val < target and report.first_low_stage is None:
            report.first_low_stage = label
    except Exception as exc:
        logger.warning(f"  {label}: PCC computation failed: {exc}")


def log_voxtral_staged_pcc_report(
    cpu_trace: VoxtralTTSDebugTrace,
    tt_trace: VoxtralTTSDebugTrace,
    *,
    target: float,
    ref_waveform: torch.Tensor,
    tt_waveform: torch.Tensor,
) -> StagedPCCReport:
    """Compare CPU vs TT traces stage-by-stage; return first low-PCC stage and first semantic mismatch."""
    report = StagedPCCReport()

    for key in ("embeds.prompt", "text.prefill.hidden"):
        cpu_t = cpu_trace.get(key)
        tt_t = tt_trace.get(key)
        if cpu_t is not None and tt_t is not None:
            _log_pcc_entry(key, cpu_t, tt_t, target, report)

    step = 0
    while True:
        prefix = f"step.{step}"
        if not any(k.startswith(f"{prefix}.") for k in tt_trace.keys()):
            break

        for suffix in ("text.hidden_in", "acoustic.codes", "text.hidden_out"):
            key = f"{prefix}.{suffix}"
            cpu_t = cpu_trace.get(key)
            tt_t = tt_trace.get(key)
            if cpu_t is not None and tt_t is not None:
                _log_pcc_entry(key, cpu_t, tt_t, target, report)

        cpu_codes = cpu_trace.get(f"{prefix}.acoustic.codes")
        tt_codes = tt_trace.get(f"{prefix}.acoustic.codes")
        if cpu_codes is not None and tt_codes is not None:
            cpu_sem = int(cpu_codes.reshape(-1)[0].item())
            tt_sem = int(tt_codes.reshape(-1)[0].item())
            if cpu_sem != tt_sem and report.first_semantic_argmax_mismatch_step is None:
                report.first_semantic_argmax_mismatch_step = step

        step += 1

    for key in ("tokenizer.latent", "tokenizer.mel"):
        cpu_t = cpu_trace.get(key)
        tt_t = tt_trace.get(key)
        if cpu_t is not None and tt_t is not None:
            _log_pcc_entry(key, cpu_t, tt_t, target, report)

    if ref_waveform is not None and tt_waveform is not None:
        _log_pcc_entry("output.waveform", ref_waveform, tt_waveform, target, report)

    return report
