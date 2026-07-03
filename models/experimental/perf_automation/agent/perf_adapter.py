# SPDX-License-Identifier: Apache-2.0
"""One GENERIC PerfAdapter for trace-replay per-token measurement — valid for ANY model.

`measure_adapter` (agent/trace_replay.py) needs a PerfAdapter (setup/step/refresh_inputs, + optional
write_inputs). Rather than a hand-written adapter per model, this module ships a SINGLE adapter that
wraps any pipeline conforming to a tiny standard DECODE CONTRACT — so the only per-model artifact is
the pipeline's own `decode_step`, which the structural decode lever (or emit-e2e) produces; the
adapter itself is model-agnostic and lives here once.

DECODE CONTRACT (duck-typed on the built pipeline object):
    decode_step(state) -> state          REQUIRED. Exactly one steady-state decode token: reads
                                          persistent on-device buffers, samples on-device, writes the
                                          next id back into `state`. NO host reads (to_torch/.item()).
    decode_prefill(input_ids) -> state    OPTIONAL. Process the prompt once, return the initial cache/
                                          state. If absent, `state` starts as None (fixed-input loop).
    decode_write_inputs(state) -> None    OPTIONAL. Stage the next step's inputs (issued on CQ1). Its
                                          presence flips measure_adapter's auto mode into the 2CQ path.

A pipeline WITHOUT decode_step (repeat-prefill / host-argmax decode) raises AttributeError in setup;
the perf test's guard then falls back to FORWARD_WALL_MS and the detector reports 'repeat_prefill'.
"""

from __future__ import annotations

from typing import Callable


class PipelineDecodeAdapter:
    """Generic PerfAdapter over any pipeline exposing the decode contract above.

    build_fn    device -> pipeline. Builds the pipeline EXACTLY as the demo/perf test does, on the
                measurement device (so the trace captures the real program).
    prompt_ids  small prompt fed to decode_prefill to build the initial state (ignored if the
                pipeline has no decode_prefill).
    batch       users in the batch — forwarded so trace_replay derives tokens_per_sec.
    """

    def __init__(self, build_fn: Callable[[object], object], prompt_ids=None, *, batch: int = 1) -> None:
        self._build = build_fn
        self._prompt = prompt_ids
        self.batch = int(batch or 1)
        self._pipe = None
        self._state = None

    def setup(self, device) -> None:
        self._pipe = self._build(device)
        step = getattr(self._pipe, "decode_step", None)
        if not callable(step):
            raise AttributeError(
                "pipeline exposes no decode_step(state); its decode is repeat-prefill — "
                "run the structural decode lever to add a cached single-token step"
            )
        prefill = getattr(self._pipe, "decode_prefill", None)
        self._state = prefill(self._prompt) if callable(prefill) else None
        wi = getattr(self._pipe, "decode_write_inputs", None)
        if callable(wi):
            self.write_inputs = lambda: wi(self._state)

    def step(self):
        self._state = self._pipe.decode_step(self._state)
        return self._state

    def refresh_inputs(self) -> None:
        pass
