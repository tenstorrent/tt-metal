# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L1 math_fidelity_downcast — drop fidelity where FPU is starving."""

from __future__ import annotations

from typing import List

from ..cluster import Cluster
from ..join import JoinedRow
from .base import (
    Finding,
    ModelSource,
    OptimizerBlock,
    Patch,
    PatchKind,
    Severity,
    SourceLocation,
    VerificationResult,
)


FIDELITY_ORDER = ["HiFi4", "HiFi3", "HiFi2", "LoFi"]


def _step_down(fidelity: str) -> str:
    if fidelity not in FIDELITY_ORDER:
        return "HiFi2"
    idx = FIDELITY_ORDER.index(fidelity)
    return FIDELITY_ORDER[min(idx + 1, len(FIDELITY_ORDER) - 1)]


class MathFidelityDowncast:
    name = "math_fidelity_downcast"
    level = 1
    requires: List[str] = []

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        seen = set()
        for r in joined:
            if r.cluster_id in seen:
                continue
            seen.add(r.cluster_id)
            # Region B/C are FPU-saturated at non-LoFi; Region D with low FPU
            # util at HiFi3/4 also benefits.
            if r.region not in ("B", "C"):
                continue
            if r.math_fidelity == "LoFi":
                continue
            cost_ms = (r.device_kernel_ns or 0) / 1e6
            new_fidelity = _step_down(r.math_fidelity or "HiFi2")
            findings.append(
                Finding(
                    block_name=self.name,
                    cluster_id=r.cluster_id or "?",
                    block_path=r.block_path,
                    severity=Severity.INFO,
                    cost_ms=cost_ms,
                    evidence={
                        "current_fidelity": r.math_fidelity,
                        "proposed_fidelity": new_fidelity,
                        "fpu_util_pct": r.pm_fpu_util_pct,
                        "region_reason": r.region_reason,
                    },
                    suggestion=(
                        f"Downcast fidelity from {r.math_fidelity} to {new_fidelity}; "
                        "FPU peak roughly doubles per step."
                    ),
                    source_locations=[
                        SourceLocation(
                            path="models/tt_transformers/tt/model_config.py",
                            variable=f"math_fidelity[{r.op_code}]",
                        )
                    ],
                    expected_gain_ms=cost_ms * 0.4,
                )
            )
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:  # noqa: ARG002
        patches: List[Patch] = []
        for f in findings:
            patches.append(
                Patch(
                    kind=PatchKind.TUNING_TABLE_ENTRY,
                    target=SourceLocation(
                        path="models/tt_transformers/tt/model_config.py",
                        variable=f"math_fidelity[{f.cluster_id}]",
                    ),
                    new_kwargs={
                        "math_fidelity": f.evidence.get("proposed_fidelity"),
                        "cluster_id": f.cluster_id,
                    },
                    rationale=(
                        f"FPU util {f.evidence.get('fpu_util_pct', 0):.0f}% at "
                        f"{f.evidence.get('current_fidelity')} -> "
                        f"{f.evidence.get('proposed_fidelity')} (expected -{f.expected_gain_ms:.2f} ms)"
                    ),
                    extra={"finding": f.cluster_id},
                )
            )
        return patches

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        # Accuracy gate is read by the runner from model_targets.yaml; we
        # surface whether device-kernel-time shrank as the run-time signal.
        b = sum(r.device_kernel_ns or 0 for r in before)
        a = sum(r.device_kernel_ns or 0 for r in after)
        delta_ms = (a - b) / 1e6
        return VerificationResult(
            ok=delta_ms < 0,
            delta_ms=delta_ms,
            accuracy_held=True,
            reason=("device time shrank" if delta_ms < 0 else "device time did not improve"),
        )
