# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`Qwen/Qwen3-Coder-Next` profiling harness with Tracy signposts.

This is NOT a second implementation of the demo. It drives the same
`tt/pipeline.py` the demo and the gates run, and adds only the phase markers a
profiler needs to split one capture into prefill and decode:

    PREFILL_START ... PREFILL_END      the prompt's single forward
    DECODE_START  ... DECODE_END       the autoregressive steps

A harness without these markers yields an empty per-phase report, because the
profiler cannot tell which device ops belong to which phase; `tt_hw_planner`
falls back to a full 'start'/'stop' capture and `tt-opt` refuses the run.

Run under tracy:

    python_env/bin/python -m tracy -r -p --op-support-count 40000 -o <outdir> \
        -m pytest -- models/demos/qwen3_coder_next/demo/text_demo_signpost.py::test_qwen3_signpost -x

Knobs (shared with the rest of the package, so the profile describes the same
build the gates measure):
    TT_QWEN3_LAYERS    depth            (default: reference.DEFAULT_LAYERS)
    TT_QWEN3_CAPACITY  pinned capacity  (default: pipeline.DEFAULT_CAPACITY = 64)
    TT_QWEN3_TP        tensor-parallel degree per replica
    TT_QWEN3_PROMPT    prompt text
    TT_QWEN3_MAX_NEW_TOKENS  decode steps to profile

WHAT "DECODE" MEANS HERE, AND WHY THE NUMBER LOOKS LARGE
The graduated `gated_delta_net` port covers one delta-rule chunk from a zero
recurrent state, so the pipeline carries NO KV cache and re-runs the whole
prefix at the pinned capacity on every step (see tt/pipeline.py). Each marked
decode step is therefore a full prefix forward, not an incremental one, and
per-step cost grows with prefix length. Profile numbers from this harness are
honest measurements of that loop -- they are not comparable to a KV-cached
decode, and the loop's shape is the first thing worth fixing.
"""
from __future__ import annotations

import os

import pytest

from models.demos.qwen3_coder_next import device_harness
from models.demos.qwen3_coder_next.tt.pipeline import (
    DEFAULT_CAPACITY,
    DEFAULT_PROMPT,
    build_pipeline,
)
from models.demos.qwen3_coder_next.tt.reference import (
    DEFAULT_LAYERS,
    encode_prompt,
    load_reference,
)

try:
    # tracy.signpost writes a marker into the profiler timeline. Outside a tracy
    # capture the import fails, and the harness must still run -- otherwise the
    # only way to exercise it is under the profiler.
    from tracy import signpost
except Exception:  # pragma: no cover - tracy is present under `-m tracy`

    def signpost(header, message=None):  # type: ignore[misc]
        print(f"[signpost] {header}", flush=True)


LAYERS = int(os.environ.get("TT_QWEN3_LAYERS", DEFAULT_LAYERS))
CAPACITY = int(os.environ.get("TT_QWEN3_CAPACITY", DEFAULT_CAPACITY))


@pytest.mark.timeout(0)
def test_qwen3_signpost():
    """One prefill and N decode steps, each bracketed by a Tracy signpost."""
    prompt_text = os.environ.get("TT_QWEN3_PROMPT", DEFAULT_PROMPT)

    model, tokenizer = load_reference(LAYERS)
    device, shape = device_harness.open_mesh()
    try:
        pipe = build_pipeline(
            device, model=model, layers=LAYERS, tokenizer=tokenizer, capacity=CAPACITY
        )
        pipe._mesh_shape = shape

        prompt = encode_prompt(tokenizer, prompt_text).reshape(-1)
        prompt_len = int(prompt.numel())
        if prompt_len >= CAPACITY:
            pytest.skip(
                f"prompt of {prompt_len} tokens leaves no room to decode inside the pinned "
                f"capacity C={CAPACITY} (gated_delta_net single-chunk limit)"
            )

        # `horizon` derives the step count from the model, clamped by capacity, so the
        # profiled length is never an invented constant.
        want = os.environ.get("TT_QWEN3_MAX_NEW_TOKENS")
        steps = pipe.horizon(prompt_len, int(want) if want else None)

        print(
            f"[signpost-harness] layers={pipe.depth} capacity={CAPACITY} "
            f"prompt_len={prompt_len} steps={steps}",
            flush=True,
        )

        # ---------------------------- PREFILL ----------------------------
        # decode_prefill seeds the resident token buffer and runs the prefix once.
        signpost("PREFILL_START")
        pipe.decode_prefill(prompt)
        signpost("PREFILL_END")

        # ---------------------------- DECODE -----------------------------
        # Steady state. Every op inside this bracket is the decode phase.
        signpost("DECODE_START")
        for _ in range(steps):
            pipe.decode_step()
        signpost("DECODE_END")

        print(f"[signpost-harness] completed {steps} decode step(s)", flush=True)
    finally:
        device_harness.close_mesh(device)
