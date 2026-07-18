# Multichip decoder work log

Date: 2026-07-18 UTC

## Scope and starting point

- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Baseline: completed `tt/optimized_decoder.py` at starting HEAD `58166c9d6d2`.
- Stage-owned runtime: `tt/multichip_decoder.py`, its tests, and this stage's
  documentation.  Full-model and vLLM work were not started.
- An unrelated pre-existing edit to
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` was preserved and
  excluded from this stage.

## Hardware and pre-code decision

`tt-smi -ls --local`, `ttnn.get_num_devices()`, and a mesh open/close probe
found four local Blackhole P300c devices.  `MeshShape(1,4)` reported matching
physical/logical degree histograms `{2:4}` and 34,178,731,008 bytes of
allocator-visible DRAM/device.  The pre-code plan therefore selected full-mesh
1-D TP=4 and recorded all tensor, activation, cache, CCL, memory, padding, and
rejected-alternative calculations in `mesh_plan.md` before the final path was
implemented.

The context contract was recalculated for Q8/KV2 ownership and preserves the
Hugging Face 131,072-token context.  The BF16 full-stack conservative plan is
13,866,369,024 bytes/device.  The archived TP2 test was pointed at a TP2
contract snapshot so it continues to test its historical stage rather than
the evolving repo-local pipeline contract.

## Implementation

`MultiChipDecoder` inherits from and names `OptimizedDecoder` as its
single-chip baseline.  Load time packs rank-local Q8/K2/V2 projection regions,
column-shards QKV/gate/up, row-shards O/down, replicates norms and positions,
and shards caches by KV head.  Runtime attention and SwiGLU stay local until
two BF16 asynchronous ring all-reduces restore the replicated residual.  One
shared `TT_CCL` owner supplies persistent trace-safe semaphores to a decoder
stack.  Paged and contiguous caches share the same two-local-head contract.

The selected geometry is a 16-way projection storage/N partition plus a
16-core residual/norm grid; BFP4/LoFi projection weights; BF16 collectives;
two links; separate gate/up projections.  The DRAM factory maps that storage
geometry to eight weight-owning DRAM-bank workers and an 80-core kernel
bounding launch.  Static inspection and a fallback-forbidden execution audit
found no host fallback in runtime methods.

## Correctness, cache, trace, and stack gates

Final command:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
timeout 300 pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k multichip_correctness
```

Result: `2 passed, 8 deselected`.  The final log is
`logs/fallback_audit_final.log`.  Principal PCC results were prefill
`0.9999993671`, decode `0.9999886729`, stacked prefill `0.9999982966`, and
stacked decode `0.9999547372`.  The test also passed non-aligned lengths 7 and
31; positions 63, 64, and 65; arbitrary page placement; exact local head
reconstruction; unwritten-page zero checks; deterministic eager runs; and five
bitwise-equal trace replays.

The complete default test file then passed `5 passed, 5 skipped`; all five
skips are the documented watcher, topology, fused-MM+RS, and two performance opt-in items.
See `logs/pytest_final.log`.  The earlier `correctness_final.log` and
`fallback_audit.log` are retained pre-stack checkpoints, not the final gate.

A final static compatibility run covered the TP4 runtime/capacity/CCL-owner
contracts and their archived TP2 counterparts: `6 passed, 13 deselected`.
See `logs/static_final.log`.

The first stack-test draft retained an unrelated decode L1 output while
starting a prefill stack, causing a static-CB/L1 live-set collision.  The
autofix isolation changed only the test lifetime to match a real stack:
prefill feeds prefill, then its live set is released; decode is regenerated
and its L1 output feeds decode directly.  Both direct boundaries pass.

Watcher command and result are recorded verbatim in `watcher_summary.txt`.
Four-device batch-32 paged execution and ten decode trace replays passed with
PCC 1.0, bitwise determinism, no watcher error/assert, and a zero exit.

## Topology and tuning

The exact fractured-residual probe command was:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
RUN_MULTICHIP_DECODER_TOPOLOGY_PROBE=1 timeout 300 \
pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k fractured_residual_topology_probe
```

The selected replicated boundary measured 0.084869 ms.  Separate
reduce-scatter, local add, distributed RMSNorm, all-gather, and BFP4 QKV
measured 0.085773 ms, 1.066% slower, while adding 12,288 bytes/layer of stats
traffic.  Rank PCCs were at least `0.9999708059`.

A direct generic fused all-gather-matmul TP4 probe hung.  The `$autofix` loop
used `$autotriage` to reproduce and inspect it.  Source and runtime evidence
show the generic factory hardcodes four transfers in each of two directions,
advertising eight slices for a four-rank ring.  Existing tests are TP8.  The
core fix lies outside this stage's decoder/tests/docs scope, so the unsafe
launch was removed and the safe separate candidate retained.  See
`AUTOTRIAGE.md` and `AUTOFIX.md`.

The independent review then requested an API-complete fused matmul plus
reduce-scatter audit. The generic 2-D multicast API passed both exact TP4-local
BFP4/LoFi row projections, but O was 0.066379 versus 0.052418 ms (26.6342%
slower) and down was 0.112457 versus 0.098161 ms (14.5637% slower), with
minimum rank PCC 0.999999940395 for both. The minimal-strided API is explicitly
disabled for Blackhole in the repository because issue `#46181` records a
nondeterministic producer/consumer race. See `fused_mm_rs_audit.md` and
`logs/fused_mm_rs_probe.log`.

The complete warmed sweep is in `candidate_results.csv`.  It rejected O32 for
decode regression; O8/residual32 and full eight-way storage geometry for slower
decode; BFP8 collectives and one link for decode regression; and packed gate/up
for prefill regression.

The first independent stage review identified two missing projection audits.
The `$autofix` remediation derived the hidden DRAM-factory output subblocks and
active-core mapping from source, then implemented and measured a complete
TP4-shape-adapted interleaved BFP4/LoFi 1-D decoder family.  The 1-D family
passed output PCC `0.9999998039` and warmed trace replay but regressed prefill
from `0.733909` to `0.989849 ms` and decode from `0.320058` to `0.405666 ms`.
The eight-way DRAM geometry also regressed decode to `0.331311 ms`.  Exact
analysis is in `profiler_geometry_audit.md`; logs are
`logs/perf_explore_interleaved_1d.log` and `logs/perf_explore_geometry8.log`.

Final performance command:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_VARIANT=default \
MULTICHIP_DECODER_BATCH=1 MULTICHIP_DECODER_SEQ_LEN=18 \
MULTICHIP_DECODER_PREFILL_REPEATS=50 MULTICHIP_DECODER_TRACE_REPLAYS=1000 \
timeout 300 pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k warmed_perf
```

Result: single/TP4 prefill `1.243475/0.733909 ms` (1.694317x, 42.3579%
efficiency); single/TP4 decode `0.581578/0.320058 ms` (1.817101x, 45.4275%
efficiency).  Output PCC was `0.9999998071`.  See `logs/perf_final.log`.

## Profiling and finalization

The exact Tracy capture and `tt-perf-report` commands, canonical losslessly
compressed raw op CSV, four filtered CSVs, four summary CSVs, four provenance logs, and four
human-readable tables are indexed in `tracy/README.md`. The final live-source
capture is `2026_07_18_09_34_42`. The TP4 decode report attributes 36.20% to
matmul, 20.76% to reduce-scatter, 12.67% to all-gather, and about 14.46% to
explicit DM plus TM. It is communication-sensitive rather
than DRAM-saturated.

Independent `$stage-review` verdict: `clean-pass`, with no required work. See
`stage_review.md`. The final scoped implementation/evidence commit SHA is
recorded by the docs-only bookkeeping commit that follows it.
