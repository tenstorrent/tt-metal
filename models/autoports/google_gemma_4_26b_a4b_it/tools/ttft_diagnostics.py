# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in host phase timing; no device profiler, synchronization, or reads."""

import functools
import json
import time


def install(adapter):
    gen = adapter.generator

    def state():
        return dict(
            model_trace=gen.trace_id is not None,
            prefill_trace=gen.prefill_trace_id is not None,
            prepared=gen.prefill_prepared is not None,
            sampled_mode=gen.sampled_mode,
            captured_sampled_mode=getattr(gen, "_trace_sampled_mode", None),
            penalties=gen.sampler._penalties_active,
            logprobs=getattr(gen.sampler, "_log_probs_active", None),
            counters=dict(gen.counters),
        )

    def wrap(owner, name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            before = state()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            record = dict(phase=name, host_ms=(time.perf_counter() - start) * 1000, before=before, after=state())
            if name.startswith("can_reuse"):
                record["result"] = bool(result)
            if name == "can_reuse_serving_prefill":
                record["tokens_shape"] = list(args[0].shape)
                record["prompt_lens"] = [int(n) for n in kwargs["prompt_lens"]]
                record["tables"] = [(type(t).__name__, list(t.shape)) for t in gen._tables(kwargs["page_table"])]
                record["incoming_key"] = repr(
                    gen._serving_prefill_key(args[0], kwargs["page_table"], kwargs["kv_cache"], kwargs["prompt_lens"])
                )
                record["stored_key"] = repr(gen.prefill_prepared["key"]) if gen.prefill_prepared else None
            if name == "configure_sampling":
                record["params"] = repr(args[0])
                record["reuse_requested"] = kwargs.get("_reuse_trace", False)
            print("TTFT_DIAGNOSTIC " + json.dumps(record), flush=True)
            return result

        setattr(owner, name, measured)

    for name in (
        "configure_sampling",
        "can_reuse_serving_prefill",
        "can_reuse_serving_decode",
        "serving_prefill_tokens",
        "prefill_forward",
        "sample_prefill",
        "_capture",
    ):
        wrap(gen, name)
    wrap(adapter, "prefill_forward")
