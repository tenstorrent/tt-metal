# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only execution evidence around the unchanged canonical generator."""
import json
import os
import time

from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

from .generator import OrnithGenerator


class ObservedOrnithGenerator(OrnithGenerator):
    def _capture_trace(self, key, call):
        started = time.monotonic()
        super()._capture_trace(key, call)
        self.event(
            "capture",
            key=key,
            trace=str(self.traces[key]),
            elapsed=time.monotonic() - started,
            signature=dict(kind=key, fixed_slots=self.batch_size, cache_blocks=self.num_blocks, context=self.context),
            captures=self.counters["captures"],
        )

    def _warmup(self):
        started = time.monotonic()
        super()._warmup()
        self.event(
            "compilation_warmup",
            elapsed=time.monotonic() - started,
            program_cache=self.device.num_program_cache_entries(),
        )

    def event(self, kind, **fields):
        if os.environ.get("ORNITH_VLLM_TRACE_EVENTS") == "1":
            print("ORNITH_TRACE " + json.dumps(dict(event=kind, time=time.monotonic(), **fields)), flush=True)

    def prepare(self, *, capture=True):
        before = dict(self.traces)
        started = time.monotonic()
        super().prepare(capture=capture)
        if self.traces != before:
            self.event(
                "preparation_complete",
                elapsed=time.monotonic() - started,
                captures=self.counters["captures"],
                retirements=self.counters["teardown_releases"],
                traces={k: str(v) for k, v in self.traces.items()},
                buckets=self.buckets,
                process_pid=os.getpid(),
                trace_allocation_tracking=bool(TRACE_ALLOC_TRACKING),
                worker_environment={
                    key: os.environ.get(key)
                    for key in (
                        "TT_METAL_TRACE_ALLOC_TRACKING",
                        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                        "ORNITH_VLLM_TRACE_EVENTS",
                        "TT_METAL_DEVICE_PROFILER",
                        "TT_METAL_WATCHER",
                    )
                },
            )

    def _refresh_prefill(self, n, ids, pos, slot):
        if self.prepared:
            self.event(
                "prefill_chunk",
                length=n,
                start=pos,
                slot=slot,
                active_decode_slots=[
                    i
                    for i in range(self.batch_size)
                    if i not in getattr(self, "_submitted_prefill_slots", ())
                    and self._active[i]
                    and self._slot_ends[i] > 0
                ],
                trace=str(self.traces.get("prefill_128" if n == 128 else f"prefill_1_{slot}"))
                if n in (1, 128)
                else None,
            )
        return super()._refresh_prefill(n, ids, pos, slot)

    def prefill_forward(self, tokens, **kwargs):
        self._submitted_prefill_slots = tuple(kwargs.get("empty_slots") or range(len(kwargs["prompt_lens"])))
        logit_reads = self.counters["logit_readbacks"]
        token_reads = self.counters["token_readbacks"]

        def draws():
            return [
                n + (self._sampling_clock - epoch if active else 0)
                for n, epoch, active in zip(self._seed_draws, self._seed_epochs, self._active)
            ]

        before_draws = draws()
        try:
            return super().prefill_forward(tokens, **kwargs)
        finally:
            self._submitted_prefill_slots = ()
            self.event(
                "prefill_complete",
                captures=self.counters["captures"],
                retirements=self.counters["teardown_releases"],
                program_cache=self.device.num_program_cache_entries(),
                logit_readbacks=self.counters["logit_readbacks"] - logit_reads,
                token_readbacks=self.counters["token_readbacks"] - token_reads,
                seed_draws_before=before_draws,
                seed_draws_after=draws(),
            )

    def replay(self, *, sample=True):
        self.event(
            "decode",
            positions=[self._position(i) for i in range(self.batch_size)],
            model_trace=str(self.traces.get("decode")),
            sampling_trace=str(self.traces.get(self._sampling_mode)) if sample else None,
            captures=self.counters["captures"],
            retirements=self.counters["teardown_releases"],
        )
        return super().replay(sample=sample)

    def close(self):
        self.event("shutdown_begin", captures=self.counters["captures"], retirements=self.counters["teardown_releases"])
        super().close()
        self.event("shutdown_end", captures=self.counters["captures"], retirements=self.counters["teardown_releases"])
