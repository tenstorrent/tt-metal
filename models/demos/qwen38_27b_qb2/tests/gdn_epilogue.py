# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only acceptance rules for the physical GDN epilogue experiment."""

import math
import statistics

BATCHES = (32, 16, 8, 1)
PLACEMENTS = ("dram", "l1")


def predecessor_ready(properties, receipt, invocation):
    if properties.get("InvocationID") not in ("", invocation):
        raise ValueError("Predecessor invocation changed")
    if properties.get("MainPID") != "0" or properties.get("ActiveState") not in ("inactive", "failed"):
        return False
    if properties.get("LoadState") not in ("loaded", "not-found"):
        return False
    if properties.get("LoadState") == "loaded" and properties.get("Result") != "success":
        raise ValueError("Predecessor exited unsuccessfully")
    if not receipt or receipt.get("state") != "completed" or receipt.get("cleanup_completed") is not True:
        raise ValueError("Predecessor lacks completed qualification and cleanup")
    return True


def compare_timings(cases):
    if len(cases) != 3 or [case["variant"] for case in cases] != ["native", "fused", "native"]:
        raise ValueError("Require native/fused/native timing brackets")
    samples = [case["traced_call_us"] for case in cases]
    if any(len(values) != 5 for values in samples) or any(
        not math.isfinite(value) or value <= 0 for values in samples for value in values
    ):
        raise ValueError("Invalid or incomplete timing samples")
    medians = [statistics.median(values) for values in samples]
    drift = abs(medians[2] / medians[0] - 1)
    native = statistics.median(samples[0] + samples[2])
    qualified = drift <= 0.03
    return dict(
        native_us=native,
        fused_us=medians[1],
        control_drift_fraction=drift,
        timing_comparison_qualified=qualified,
        qualified_speedup=native / medians[1] if qualified else None,
        # Linear addition is a projection, not a traced full-model measurement.
        projected_48_layer_saving_ms=(native - medians[1]) * 48 / 1000 if qualified else None,
        full_model_speedup_measured=False,
    )
