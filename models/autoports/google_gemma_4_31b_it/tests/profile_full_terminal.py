# SPDX-License-Identifier: Apache-2.0
"""Reduced real-weight wrapper profile, ONLY layers0/5, never full-stack metrics.

Collection (run from the configured bringup environment, serialized):
  TT_METAL_TRACE_ALLOC_TRACKING=0 python -m tracy -r -p -v --no-web-server \
    --op-support-count 16384 -o models/autoports/google_gemma_4_31b_it/doc/full_model/tracy_reduced \
    -m models.autoports.google_gemma_4_31b_it.tests.profile_full_terminal --attribution \
    --result models/autoports/google_gemma_4_31b_it/doc/full_model/reduced_profile.json

For the resulting ops_perf_results*.csv, run e.g.:
  tt-perf-report <csv> --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
    --csv <decode_perf_report.csv> > <decode_perf_report.console.log>
  tt-perf-report <csv> --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
    --no-summary > <decode_perf_report.txt>
Repeat for PERF_PREFILL, PERF_MODEL_ONLY, PERF_SAMPLER_ONLY and PERF_TERMINAL,
using the corresponding _END signpost. Advice remains enabled.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tracy import signpost

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.generator import Gemma4Generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 8, 16, 32), default=1)
    parser.add_argument("--attribution", action="store_true")
    parser.add_argument("--drain-steps", type=int, default=2)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    if args.drain_steps < 1:
        raise ValueError("Profiler drain cadence must be positive")
    for key in ("TT_METAL_WATCHER", "TT_METAL_DPRINT_CORES"):
        if os.environ.get(key):
            raise ValueError(f"Profiler must not share a run with {key}")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0") != "0":
        raise ValueError("Collect allocation-tracking acceptance separately from profiling")
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(mesh, layer_indices=(0, 5), batch_size=args.batch, max_seq_len=262144)
        assert gen.model.layer_indices == (0, 5), "Full-stack profiling is forbidden"
        assert not gen.sampler.force_argmax_sampling, "Profile the delivered normal split sampler"
        run(gen, args)
    finally:
        if gen is not None:
            for trace in (gen.model_trace, gen.sample_trace):
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


def run(gen, args):
    mesh = gen.mesh
    drain_seconds = 0.0

    def drain():
        nonlocal drain_seconds
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        ttnn.ReadDeviceProfiler(mesh)
        drain_seconds += time.perf_counter() - start

    # This is a fixed-token diagnostic prompt, not a chat-quality measurement.
    fragment = gen.tokenizer.encode(
        "Explain why careful measurements help compare two implementations. ", add_special_tokens=False
    )
    prompt = ([gen.tokenizer.bos_token_id or 2] + fragment * (4096 // len(fragment) + 1))[:4096]
    assert len(prompt) == 4096
    prompts = [list(prompt) for _ in range(args.batch)]
    host_tokens = torch.tensor(prompts, dtype=torch.int64)
    original_hidden = gen.prefill.hidden

    def drained_hidden(**kwargs):
        hidden = original_hidden(**kwargs)
        # Two real layers per chunk; draining prevents setup/tail buckets from
        # overrunning the profiler ring. It is not generator production behavior.
        drain()
        return hidden

    gen.prefill.hidden = drained_hidden
    if args.attribution:
        # Standalone terminal gets real reduced-model prefill activations. Do it
        # before generator preparation: extra profiling traces never live across
        # preparation, and their temporary state cannot invalidate model traces.
        gen.reset()
        gen._reserve_requests([4096] * args.batch, 128)
        selected = []
        for slot in range(args.batch):
            for start in range(0, 4096, 1024):
                gen.prefill.stage(
                    host_tokens[slot, start : start + 1024], slot=slot, start=start, size=1024, valid=1024
                )
                hidden = gen.prefill.hidden(size=1024, start=start, kv_cache=gen.kv_cache)
                if start == 3072:
                    selected.append(gen.prefill.last_row(hidden))
                del hidden
        terminal_input = selected[0] if args.batch == 1 else ttnn.concat(selected, dim=2)
        warm = gen.model.terminal(terminal_input)
        del warm
        drain()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        terminal_output = gen.model.terminal(terminal_input)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            drain()
            signpost("PERF_TERMINAL")
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            signpost("PERF_TERMINAL_END")
            drain()
        finally:
            ttnn.release_trace(mesh, trace)
        del terminal_output, terminal_input, selected

    gen.prepare_decode()
    gen.set_sampling(temperature=0.0, top_k=1, top_p=1.0, seed=0)
    # A complete 4096-token prefix plus one decode warms real offsets/read windows.
    gen.generate_batch(prompts, 2, ignore_eos=True)
    drain()
    captures = gen.counters["captures"]
    gen.reset()
    gen._reserve_requests([4096] * args.batch, 128)
    drain()
    before = dict(gen.counters)
    signpost("PERF_PREFILL")
    first = gen.prefill_forward(
        host_tokens, page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[4096] * args.batch
    )
    signpost("PERF_PREFILL_END")
    drain()
    gen._copy(torch.full((1, args.batch), 4096, dtype=torch.int32), gen.positions)
    gen.counters["position_refreshes"] += 1
    output = [[int(token)] for token in first]
    # 128 generated tokens total: first token above, followed by127 feedback steps.
    for step in range(127):
        # One steady-state step is the ranked op window. The entire128-token
        # request still executes, with periodic diagnostic profiler drains.
        if step == 63:
            drain()
            signpost("PERF_DECODE")
        tokens = gen._replay()
        if step == 63:
            signpost("PERF_DECODE_END")
            drain()
        for row, token in zip(output, tokens):
            row.append(int(token))
        if (step + 1) % args.drain_steps == 0:
            drain()
    drain()
    after = dict(gen.counters)
    assert gen.counters["captures"] == captures, "Profiling must reuse prepared traces"
    assert all(len(row) == 128 for row in output)
    if args.attribution:
        # Attribution adds synchronization only in these explicitly separate
        # diagnostic windows, after the normal combined token-out request.
        signpost("PERF_MODEL_ONLY")
        ttnn.execute_trace(mesh, gen.model_trace, cq_id=0, blocking=True)
        signpost("PERF_MODEL_ONLY_END")
        drain()
        signpost("PERF_SAMPLER_ONLY")
        ttnn.execute_trace(mesh, gen.sample_trace, cq_id=0, blocking=True)
        signpost("PERF_SAMPLER_ONLY_END")
        drain()
    result = dict(
        scope="reduced real-weight layers0/5 only; not full-model latency or accuracy",
        layer_indices=[0, 5],
        layer_types=[layer.layer_type for layer in gen.model.layers],
        input_tokens=4096,
        generated_tokens=128,
        batch=args.batch,
        concurrency=args.batch,
        configured_context=gen.context,
        pool_tokens=gen.pool_tokens,
        cache_shapes=[[list(t.shape) for t in cache] for cache in gen.kv_cache],
        cache_dtypes=[[str(t.dtype) for t in cache] for cache in gen.kv_cache],
        page_table_shapes=[list(t.shape) for t in gen.page_table],
        counters={key: after[key] - before[key] for key in before},
        decode_report_step=63,
        decode_absolute_position=4159,
        profiler_drain_seconds=drain_seconds,
        profiler_drain_steps=args.drain_steps,
        attribution=args.attribution,
        output_token_ids=output,
        notes=[
            "Synthetic fixed-length diagnostic prompt; reduced model text is not a quality gate.",
            "PERF_DECODE includes model+normal sampling traces and caller token readback.",
            "PERF_MODEL_ONLY includes terminal; PERF_TERMINAL is a separate traced activation control.",
            "No latency/roofline claims: profiler drains and diagnostics perturb the run.",
        ],
    )
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("REDUCED_PROFILE_COMPLETE layers=0,5 input=4096 output=128", flush=True)


if __name__ == "__main__":
    main()
