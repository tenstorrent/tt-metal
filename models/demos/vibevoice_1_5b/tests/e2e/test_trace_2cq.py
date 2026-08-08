# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command 3 gate for microsoft/VibeVoice-1.5B: trace+2CQ contract + host-free.

Builds the resident pipeline via the module-level `build_pipeline` factory and:
  * `trace_capture_selftest`: for each PIPELINE_STAGE captures ONE host-op-free step
    in begin/end_trace_capture, execute_trace, verify PCC vs the eager reference, and
    RELEASE before the next stage. A stage whose kernel is not cleanly trace-capturable
    (e.g. the acoustic decoder's conv zero-padding) DEGRADES to single-CQ and PRINTS the
    fallback (never silently dropped); it still verifies PCC.
  * `host_op_selftest`: runs the on-device stage kernels under observe_host_ops with all
    encoding/weight-build/uploads OUTSIDE the observed region — a truly on-device forward
    fires ZERO host aten ops.
Both functions open/close their own device (trace region + 2 command queues).
"""

from __future__ import annotations

from models.demos.vibevoice_1_5b.tt import pipeline as P


def test_trace_capture():
    ok = P.trace_capture_selftest()
    print(f"PIPELINE_STAGES={P.PIPELINE_STAGES}")
    assert ok, "trace_capture_selftest: a stage failed to verify (even after single-CQ fallback)"


def test_host_free():
    verdict = P.host_op_selftest()
    print(f"host_op verdict: {verdict}")
    assert verdict["on_device"], f"host compute in the forward: {verdict['host_ops']}"
