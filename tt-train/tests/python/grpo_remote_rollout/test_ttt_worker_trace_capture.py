# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for TttGenerationWorker trace-capture warmup.

TttGenerationWorker.generate() must set ``warmup_prefill=True`` when it
calls ``Generator.prefill_forward_text``. If it does not, the first
``generate(..., enable_trace=True)`` call might raise this TT_FATAL from
``tt_metal/distributed/mesh_workload.cpp``:

    Cannot load new binaries during trace capture. This program is not
    yet in program cache. Warm up before capturing a trace.

The fatal comes from the on-device sampling kernels. They are not in
the program cache when the decode trace capture opens. ``warmup_prefill=
True`` runs ``warmup_model_prefill`` before the capture, which puts the
kernels in the cache.

This test does one traced ``generate()`` call and shows that it does
not raise. Uses dummy weights, so no HF token or weight download is
needed.

Run (needs >= 2 chips):
    cd tt-train/tests/python/grpo_remote_rollout
    python3 -m pytest -s test_ttt_worker_trace_capture.py
"""

from __future__ import annotations

import gc

import pytest

MESH_SHAPE = (1, 2)
# One short prompt is enough. The fatal comes from the decode trace
# capture at prefill time, not from the decode loop itself.
PROMPT_LEN = 8
MAX_NEW_TOKENS = 2


@pytest.mark.timeout(0)
def test_traced_generate_does_not_fatal_on_first_call():
    import ttnn

    from _completer_utils import _TRACE_REGION_SIZE, build_completer

    if len(ttnn.get_device_ids()) < MESH_SHAPE[0] * MESH_SHAPE[1]:
        pytest.skip(f"needs >= {MESH_SHAPE[0] * MESH_SHAPE[1]} chips")

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        trace_region_size=_TRACE_REGION_SIZE,
    )
    completer = None
    try:
        completer = build_completer(mesh_device, dummy_weights=True, max_batch_size=1)

        prompt = list(range(1, PROMPT_LEN + 1))

        try:
            completions = completer.generate(
                [prompt],
                max_new_tokens=MAX_NEW_TOKENS,
                enable_trace=True,
            )
        except RuntimeError as e:
            msg = str(e)
            if "Cannot load new binaries during trace capture" in msg:
                pytest.fail("First traced generate() tripped the trace-capture " "fatal.")
            raise

        assert len(completions) == 1, f"expected 1 completion, got {len(completions)}"
    finally:
        completer = None
        gc.collect()
        ttnn.close_mesh_device(mesh_device)
