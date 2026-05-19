# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L3 program_config_tuner — pick program_config from the reference DB."""

from __future__ import annotations

from typing import List

from ..ceilings import BoxSpec
from ..join import JoinedRow
from ..reference_db import find_best
from .base import (
    Finding,
    ModelSource,
    Patch,
    PatchKind,
    Severity,
    SourceLocation,
    VerificationResult,
)


def _shape_sig(row: JoinedRow) -> str:
    if not row.inputs:
        return ""
    t = row.inputs[0]
    return "x".join(str(t.get(k, "")) for k in ("W", "Z", "Y", "X"))


class ProgramConfigTuner:
    name = "program_config_tuner"
    level = 3
    requires: List[str] = []

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box: BoxSpec) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        seen = set()
        for r in joined:
            if r.cluster_id in seen:
                continue
            seen.add(r.cluster_id)
            if r.region not in ("B", "C"):
                continue
            sig = _shape_sig(r)
            ref = find_best(r.op_code, sig, box.name)
            if ref is None:
                continue
            observed = (r.device_kernel_ns or 0) / 1e3
            if observed <= ref.observed_device_us * 1.10:
                continue
            cost_ms = (r.device_kernel_ns or 0) / 1e6
            findings.append(
                Finding(
                    block_name=self.name,
                    cluster_id=r.cluster_id or "?",
                    block_path=r.block_path,
                    severity=Severity.WARN,
                    cost_ms=cost_ms,
                    evidence={
                        "observed_device_us": observed,
                        "reference_device_us": ref.observed_device_us,
                        "ratio": observed / max(ref.observed_device_us, 1e-3),
                        "reference_source_run": ref.source_run,
                        "proposed_program_config": ref.kwargs.get("program_config"),
                    },
                    suggestion=(
                        f"Cluster runs at {observed:.1f} us, reference is "
                        f"{ref.observed_device_us:.1f} us "
                        f"({observed / ref.observed_device_us:.2f}x slower). "
                        f"Apply reference program_config."
                    ),
                    source_locations=[
                        SourceLocation(
                            path="models/tt_transformers/tt/model_config.py",
                            variable=f"program_config[{r.op_code}/{sig}]",
                        )
                    ],
                    expected_gain_ms=cost_ms * (1 - ref.observed_device_us / max(observed, 1e-3)),
                )
            )
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:  # noqa: ARG002
        patches: List[Patch] = []
        for f in findings:
            pc = f.evidence.get("proposed_program_config") or {}
            patches.append(
                Patch(
                    kind=PatchKind.TUNING_TABLE_ENTRY,
                    target=SourceLocation(
                        path="models/tt_transformers/tt/model_config.py",
                        variable=f"program_config[{f.cluster_id}]",
                    ),
                    new_kwargs={"program_config": pc, "cluster_id": f.cluster_id},
                    rationale=(
                        f"reference DB best @ same shape on this box; expected "
                        f"-{f.expected_gain_ms:.2f} ms; source: {f.evidence.get('reference_source_run')}"
                    ),
                    extra={"finding": f.cluster_id},
                )
            )
        return patches

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        b = sum(r.device_kernel_ns or 0 for r in before)
        a = sum(r.device_kernel_ns or 0 for r in after)
        delta_ms = (a - b) / 1e6
        return VerificationResult(
            ok=delta_ms < 0,
            delta_ms=delta_ms,
            accuracy_held=True,
            reason=("reference config helped" if delta_ms < 0 else "no improvement; reverting"),
        )
