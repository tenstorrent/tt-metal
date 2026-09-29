# Local reproduction commands

Run only with exclusive hardware ownership. These commands use the owned image
Python, `/tmp` working directory and explicit mounted-source precedence recorded
in the evidence README. Do not profile a live server. The serving launcher is
`tools/ttft_server.py --output <cohort-directory>`; add `--no-async-scheduling`
for the synchronous control. Capacity, precision and wire configuration are in
each `launch.json`; benchmark commands are saved verbatim as `.command.json`.

## Head families

Module: `models.autoports.google_gemma_4_26b_a4b_it.tests.probe_full_lm_head`.
Every row below shares these arguments:

```
--weight-dtype bfloat4_b --fidelity LoFi --baseline-block 4
--rounds 5 --replays 20
--fixture /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/terminal_input.pt
```

Add `--output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/tsu_optimization/<artifact>`.
The fixture SHA256 and every legal/failing geometry and raw timing round are
recorded in the JSON. Timing includes conversions, slicing and concatenation.

| Artifact | Remaining arguments |
|---|---|
| `head_bfp4_dram_retry.json` | `--chunks 65536 --blocks 1 2 4 --readers 1 2 3` |
| `head_bfp4_dram_chunks.json` | `--chunks 32768 16384 --blocks 1 2 4 --readers 1 2 3` |
| `head_bfp4_tiles.json` | `--interleaved-only --grids 11x10 --per-core-n 19 20 22 24 32 --blocks 2 4 8 11 22 44 88` |
| `head_bfp4_l1.json` | `--interleaved-only --grids 11x10 --per-core-n 19 20 22 24 32 --blocks 1 2 4 8 --input-memory l1` |
| `head_bfp4_smallk.json` | `--interleaved-only --grids 11x10 11x8 8x10 8x8 --blocks 1 2 4 8 11 22 44 88` |
| `head_bfp4_smallchunks.json` | `--chunks 4096 8192 --blocks 8 11 22 44 88 --readers 1 2 3` |

Full-generator control module `tests.measure_tsu_paths`, arguments
`--head-blocks 4 2 1 4 --repeats 5 --output <head_full_paths.json>`;
input4096/output128/cache262144 are explicit defaults recorded in the JSON.

## Safety

Module `tests.check_tsu_prefill_trace --reuse-eager --lengths 1025 4095 4096
4097 8192 8193 16384 --output <final_watcher.json>`, environment
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`.
Worker checks remain enabled; inherited Ethernet instrumentation capacity
limitation is documented in the work log.

The separate final tracked run uses lengths4096/4097 and additionally
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, output `final_tracking.json`.

## Reduced profile

No live server, only real layers0/5 plus full terminal/sampler/recorder:

```
python -m tracy -r -p -v --check-exit-code --op-support-count 100000 \
  --no-op-info-cache --disable-device-data-dump-to-files \
  --disable-device-data-push-to-tracy -o <profile-root> -n <run-name> \
  -m models.autoports.google_gemma_4_26b_a4b_it.tests.measure_tsu_paths \
  --reduced --profile --input-length 128 --output <paths.json>
```

The earlier4K capture uses input4096 and K4 production head; the final short
capture uses K1. Per-SDPA comparisons are like-for-like; whole-window differences
also include the head change. `--check-exit-code` was added to the final short
capture; the invalid interrupted first4K wrapper result is explicitly rejected.

After device/server jobs end, analyze saved CSV with `tt-perf-report
--start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --tracing-mode
--active-experts 8 --no-color <csv>` for the advice-backed human table, and
add `--csv <destination>` in a separate invocation for machine rows.
