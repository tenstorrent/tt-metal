# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""COMMAND 3 gate for ``tencent/HunyuanVideo-1.5``: host-free forward + trace+2CQ.

These exercise the SAME module-level hooks the perf/2CQ + host-op probes call:

  * ``host_op_selftest()`` — the authoritative fully-on-device check.  Runs the
    model math under ``host_op_observer.observe_host_ops`` (encoding + weight
    build outside the observed region) and asserts ZERO host aten ops fire, for
    every task head.
  * ``trace_capture_selftest()`` — captures / executes / releases a trace for
    each stage in ``PIPELINE_STAGES`` and asserts the trace output matches the
    HF golden (PCC >= 0.95).  Uses the per-stage ``denoise_trace_setup /
    denoise_trace_step / denoise_write_inputs`` contract and the
    ``build_pipeline`` factory.
"""

from __future__ import annotations

from models.demos.hf_eager.hunyuanvideo_1_5.tt import pipeline as P


def test_pipeline_stages_declared():
    assert P.PIPELINE_STAGES == ["denoise"]
    pipe_has = P.HunyuanVideo15Pipeline
    for suffix in ("trace_setup", "trace_step", "write_inputs"):
        assert hasattr(pipe_has, f"denoise_{suffix}"), f"missing denoise_{suffix} contract hook"
    assert callable(P.build_pipeline)


def test_host_op_selftest_fully_on_device():
    verdict = P.host_op_selftest()
    print(
        f"host_op_selftest verdict: on_device={verdict['on_device']} "
        f"n_host_ops={verdict['n_host_ops']} host_ops={verdict['host_ops']}"
    )
    assert verdict["on_device"], f"host aten ops fired in the forward: {verdict['host_ops']}"


def test_trace_capture_selftest_per_stage():
    ok = P.trace_capture_selftest()
    print(f"trace_capture_selftest -> {ok}")
    assert ok, "trace+2CQ per-stage capture/PCC self-test failed"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-s", "-v"]))
