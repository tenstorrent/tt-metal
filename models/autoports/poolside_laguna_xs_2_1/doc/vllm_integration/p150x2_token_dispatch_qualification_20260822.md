<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# P150x2 MoE token-dispatch qualification — 2026-08-22

Status: production integration is **default off** and not yet promoted. Both
the Stage-2 prototype and the production stacked-weight path pass the one-layer
8192-token accuracy, warm-latency, memory, and program-cache-stability gates.
Cumulative 39-layer and cross-bucket qualification remains outstanding.

## Scope and configuration

- Hardware: one P150x2 mesh, physical chips 2 and 3 for the captured run.
- Model/layer: Laguna-XS-2.1 routed MoE layer 1.
- Tokens/top-k/experts: 8192 / 8 / 256 global, 128 per ASIC.
- Dispatch mapping: mesh-axis-0 groups of size one on the 1x2 mesh; each ASIC
  packs only its contiguous local experts. No dispatch/combine CCL occurs.
- Router and shared expert preserve the established 256-row slice semantics.
- Compact expert FFN: BF16 activations, BFP4 weights, LoFi, 16 M tiles (512
  rows) per kernel chunk.
- Combine is an unweighted route-slot permutation. Existing fused
  `post_combine_reduce` applies router weights, masks non-local routes, and
  sums top-k slots before the existing EP all-reduce.

Source log: `/tmp/laguna_token_dispatch_weighted_20260822.log`.

## Captured command

```bash
env -u TT_METAL_HOME \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/home/ttuser/dev/laguna/tt-metal \
  TT_VISIBLE_DEVICES=2,3 \
  LAGUNA_PROFILE=p150x2 \
  TT_RUN_LAGUNA_TOKEN_DISPATCH_PROBE=1 \
  TT_LAGUNA_MOE_PREFILL_TILE_SPARSE=0 \
  TT_LAGUNA_TOKEN_DISPATCH_ACTIVATIONS=bf16 \
  TT_LAGUNA_TOKEN_DISPATCH_FIDELITY=lofi \
  TT_LAGUNA_TOKEN_DISPATCH_CHUNK_M_TILES=16 \
  TT_LAGUNA_TOKEN_DISPATCH_PRESERVE_SLICE_SEMANTICS=1 \
  /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python \
  -m pytest -s -vv --tb=short \
  /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/test_token_dispatch_prototype_hardware.py::test_laguna_d2_whole_8192_token_dispatch_probe
```

Result: `1 passed in 16.24s`; the mesh closed cleanly.

## Results

| Metric | Baseline / gate | Prototype | Verdict |
|---|---:|---:|---|
| Final output PCC | >= 0.995 | **0.9997239543** | pass |
| Final max absolute error | record | 0.0546875 | recorded |
| Warm one-layer latency | baseline 255.742401 ms | **53.693918 ms** | pass |
| Warm speedup | >= 1x; stretch >= 3x | **4.76297x** | pass/stretch pass |
| Cold one-layer latency | baseline 295.503778 ms | 947.243716 ms | compile cost recorded |
| Peak allocated bytes | <= 1,540,210,688 | **1,523,400,704** | pass |
| Allocated after output free | no leak/fragmentation regression | 545,841,152 | pass |
| Largest contiguous free per bank after free | record | 3,999,937,920 | recorded |

The dispatch plan selected exactly 65,536 routes and preserved every selected
input row (`PCC=1.0`). Tile-aligned expert regions occupied 68,800 rows. The
compact kernel executed 123,392 padded rows total, versus 754,688 and 880,128
expert rows on the two ASICs in the established 32x256 slice loop.

Stage isolation:

| Stage | PCC |
|---|---:|
| FF1 gate vs packed production path | 0.9998795834 |
| FF1 up vs packed production path | 0.9998799264 |
| SwiGLU activation | 0.9997291777 |
| Separate FF2 | 0.9996974167 |
| Fused FF2 vs packed production path | 0.9997464597 |
| Raw combine permutation | 1.0 |
| Weighted post-combine reduction | 0.9999935172 |

## Accuracy findings that are contractual

Running the router once across all 8192 rows is not equivalent to the current
32x256 program sequence: only 82.286% of top-k slots and 81.787% of per-token
top-k sets matched. The whole-M shared expert also differed slightly (PCC
0.9998248). The production candidate therefore keeps both router and shared
expert in 256-row slices while dispatching the routed FFNs across the whole
outer bucket.

The original prototype omitted the post-combine routing-weight multiply and
stalled below the production accuracy gate. Restoring the existing fused
weighted reduction raised the final PCC to 0.999724. An unweighted combine is
not a valid fallback or optimization.

## Production-integration envelope

The candidate switch is `TT_LAGUNA_MOE_TOKEN_DISPATCH=1`; unset or `0` keeps
the established path. Values other than literal `0` or `1` fail at setup.
The launcher additionally requires `TT_LAGUNA_PREFIX_CACHE=0`, one sequence,
streaming prefill on, uniform KV, and every DFlash/context/multi-sequence/
tile-sparse experiment off. Acknowledging experiments does not permit stacking
independently qualified paths. Even when requested, the runtime falls back
unless all of these hold:

- p150x2 (`D=2`) and a routed layer 1 through 39;
- interleaved prefill bucket 1024, 2048, 4096, or 8192;
- production packed gate/up weights and the exact 256/128 expert partition;
- hidden/intermediate/top-k = 2048/512/8;
- BF16 activation and CCL, BFP4 gate/up/down, LoFi MoE fidelity.

The integration reads existing expert-sharded
`exp_gate_up [1,128,2048,1024]` and
`exp_down [1,128,512,2048]` tensors directly. It must not allocate duplicated
per-expert weights. Short buckets, decode/sharded calls, D1/D4, dense layer 0,
and alternate precision policies remain on the original 256-row sparse-MoE
implementation.

## Remaining promotion gates

The production stacked-weight layer-1/8192 gate below is complete. Promotion
still requires all of the following in one reviewed qualification tranche:

1. Prove selected-row, weighted reduction, and final-output accuracy for every
   enabled bucket (1024/2048/4096/8192) at routed layers 1, 20, and 39.
2. Build and run the actual 40-layer model, checking cumulative setup/residency
   and end-to-end logits and greedy tokens.
3. Boot the actual p150x2 vLLM server with all 39 routed layers, run its complete
   prefill warmup lifecycle, freeze program-cache misses after trace capture,
   and compare cache-off baseline/candidate TTFT and TPOT at each enabled bucket.
4. Keep the switch default off until every numerical, performance, memory,
   cache, and health gate below passes and its artifacts are reviewed.

## Production stacked-weight rerun

The production candidate was rerun after adding direct reader indexing for the
existing expert-sharded `exp_gate_up` and `exp_down` tensors. It did not create
the prototype's duplicated per-expert tensor lists.

Source log: `/tmp/laguna_token_dispatch_stacked_production_20260822.log`.

```bash
cd /tmp && script -q -c 'env -u TT_METAL_HOME PYTHONNOUSERSITE=1 PYTHONPATH=/home/ttuser/dev/laguna/tt-metal TT_VISIBLE_DEVICES=2,3 LAGUNA_PROFILE=p150x2 TT_RUN_LAGUNA_STACKED_TOKEN_DISPATCH_PROBE=1 TT_LAGUNA_MOE_TOKEN_DISPATCH=1 TT_LAGUNA_MOE_PREFILL_TILE_SPARSE=0 TT_LAGUNA_TOKEN_DISPATCH_ACTIVATIONS=bf16 TT_LAGUNA_TOKEN_DISPATCH_FIDELITY=lofi TT_LAGUNA_TOKEN_DISPATCH_CHUNK_M_TILES=16 TT_LAGUNA_TOKEN_DISPATCH_PRESERVE_SLICE_SEMANTICS=1 /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python -m pytest -s -vv --tb=short /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/test_token_dispatch_prototype_hardware.py::test_laguna_d2_production_stacked_8192_token_dispatch_probe' /tmp/laguna_token_dispatch_stacked_production_20260822.log
```

Result: `1 passed in 57.32s`; only physical chips 2 and 3 were opened and the
mesh closed cleanly.

| Metric | Gate | Production stacked | Verdict |
|---|---:|---:|---|
| Final output PCC | >= 0.995 | **0.9997239543** | pass |
| Final max absolute error | record | 0.0546875 | recorded |
| Baseline warm latency | reference | 255.861025 ms | recorded |
| Candidate warm latency | < 255.936 ms | **53.281487 ms** | pass |
| Warm speedup | >= 3x | **4.802062x** | pass |
| Baseline cold latency | record | 555.821380 ms | recorded |
| Candidate cold latency | record | 53,246.028540 ms | recorded; see below |
| Tracked peak allocated bytes | <= 1,540,210,688 | **892,739,584** | pass |
| Resident weights + input | record | 309,436,416 bytes | recorded |
| Allocated after output free | record | 317,841,408 bytes | recorded |
| Largest contiguous free/bank after free | >= 3,999,000,000 | **4,017,737,088** | pass |
| Program-cache entries cold/warm/repeat | unchanged after cold | 351 / 351 / 351 | pass |

The first stacked invocation built 130 previously uncached artifacts (JIT cache
stats 1065/1195 hits), producing a material 53.2-second one-time cold cost.
Warm and repeated passes added no program-cache entries and retained the 53.3 ms
latency. Serving warmup must absorb and verify this cost before promotion. The
shape and local-expert-id program keys are shared across layers, but that reuse
has not yet been demonstrated during a full 39-layer model load/run.

## Next bounded promotion gate (planned; not yet run)

This gate deliberately uses production stacked weights and the real full-model
and serving paths. It does not use the prototype's duplicated per-expert tensor
lists. The feature remains default off throughout. Chips 2 and 3 are the only
allowed devices, the server uses fixed port 8000, and chips 0 and 1 must not be
opened or reset.

### Phase A: representative layer/bucket matrix in one mesh lifetime

Add one opt-in entry point beside the production-stacked probe which opens one
p150x2 mesh, loads real checkpoint weights for routed layers 1, 20, and 39, and
tests 1024, 2048, 4096, and 8192 rows for each layer. Each case must compare the
established 256-row slice loop with the default-off production dispatch path,
using three synchronized warm repetitions. Keep the mesh and TTNN program cache
alive for the entire 12-case matrix; this is required to prove cross-layer
amortization rather than merely observing a persistent on-disk JIT hit in a new
process.

The proposed bounded command, after that matrix entry point is reviewed, is:

```bash
cd /tmp && script -q -c 'env -u TT_METAL_HOME PYTHONNOUSERSITE=1 PYTHONPATH=/home/ttuser/dev/laguna/tt-metal TT_VISIBLE_DEVICES=2,3 LAGUNA_PROFILE=p150x2 TT_RUN_LAGUNA_STACKED_TOKEN_DISPATCH_MATRIX=1 TT_LAGUNA_MOE_TOKEN_DISPATCH=1 TT_LAGUNA_MOE_PREFILL_TILE_SPARSE=0 TT_LAGUNA_TOKEN_DISPATCH_ACTIVATIONS=bf16 TT_LAGUNA_TOKEN_DISPATCH_FIDELITY=lofi TT_LAGUNA_TOKEN_DISPATCH_CHUNK_M_TILES=16 TT_LAGUNA_TOKEN_DISPATCH_PRESERVE_SLICE_SEMANTICS=1 /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python -m pytest -s -vv --tb=short /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/test_token_dispatch_prototype_hardware.py::test_laguna_d2_production_stacked_cross_bucket_layer_matrix' /tmp/laguna_token_dispatch_matrix_20260822.log
```

Hard gates for all 12 cases:

- dispatch plan covers exactly `T*8` selected rows and every selected source
  row compares bit-exactly (`PCC=1.0`); raw combine remains a bit-exact route
  permutation;
- weighted routed reduction PCC is at least 0.9999, final layer output PCC is
  at least 0.995, and final maximum absolute error is at most 0.125;
- candidate median warm latency is no more than 1.03 times its paired baseline
  in any case, the geometric-mean candidate/baseline ratio across all cases is
  at most 0.90, and the layer-1/8192 case retains at least a 3x speedup;
- no warm repetition adds a program-cache entry; after a bucket is first built
  at layer 1, layers 20 and 39 add zero entries for that bucket; finally forbid
  misses and repeat all 12 calls without an error or cardinality change;
- peak transient allocation is no larger than the measured 1,540,210,688-byte
  one-layer ceiling, output cleanup returns to within 32 MiB of the pre-case
  allocation, free DRAM remains at least 10%, and the largest contiguous free
  region remains at least 128 MiB per bank.

The observed 53.2 seconds is kernel JIT/build cost, not warm execution. The
matrix must record wall time, JIT hit/build counts, and program-cache cardinality
before and after every cold and warm call. The expected behavior is one build
of the common dispatch/FFN/combine kernels at the first eligible shape, new
shape-specialized program entries only on the first layer for each bucket, and
reuse by the other 38 routed layers. Do not delete or mutate a shared compiler
cache to manufacture a cold run; the already captured 53.2-second result is the
cold-cost oracle. A later serving boot must budget that full cost before ready.

Estimated duration: 10–15 minutes, including the one-time compile, three warm
repetitions, host PCC composition, and clean mesh close.

### Phase B: actual 40-layer logits and token gate

Run the existing full-model `prefill_autoreg` acceptance path twice, first with
the established implementation as the oracle and then with the candidate. Each
run loads all 40 real layers (39 routed), evaluates the AIME reference logits,
allocates the qualified p150x2 context, captures/replays decode, and writes
generated-token artifacts. This is not a reduced-layer or synthetic-weight
test. Close the first mesh cleanly and verify device health before starting the
second process.

Flag-off oracle:

```bash
cd /tmp && script -q -c 'env -u TT_METAL_HOME PYTHONNOUSERSITE=1 PYTHONPATH=/home/ttuser/dev/laguna/tt-metal TT_VISIBLE_DEVICES=2,3 LAGUNA_PROFILE=p150x2 TT_LAGUNA_MOE_TOKEN_DISPATCH=0 TT_LAGUNA_MOE_PREFILL_TILE_SPARSE=0 TT_LAGUNA_PREFIX_CACHE=0 TT_LAGUNA_HYBRID_KV=0 TT_LAGUNA_DFLASH=0 TT_LAGUNA_STREAMING_PREFILL=1 /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/full_model_checks.py prefill_autoreg --profile p150x2 --acceptance --gen-len 100 --enforce-memory-margin --outdir doc/full_model/token_dispatch_oracle_20260822' /tmp/laguna_token_dispatch_full_model_oracle_20260822.log
```

Candidate:

```bash
cd /tmp && script -q -c 'env -u TT_METAL_HOME PYTHONNOUSERSITE=1 PYTHONPATH=/home/ttuser/dev/laguna/tt-metal TT_VISIBLE_DEVICES=2,3 LAGUNA_PROFILE=p150x2 TT_LAGUNA_MOE_TOKEN_DISPATCH=1 TT_LAGUNA_MOE_PREFILL_TILE_SPARSE=0 TT_LAGUNA_PREFIX_CACHE=0 TT_LAGUNA_HYBRID_KV=0 TT_LAGUNA_DFLASH=0 TT_LAGUNA_STREAMING_PREFILL=1 /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/full_model_checks.py prefill_autoreg --profile p150x2 --acceptance --gen-len 100 --enforce-memory-margin --outdir doc/full_model/token_dispatch_candidate_20260822' /tmp/laguna_token_dispatch_full_model_20260822.log
```

Hard gates:

- the full-logit AIME scores remain top-1 >= 0.90, top-5 >= 0.98, and
  top-100 = 1.0; before running, extend the qualification artifact to retain
  the evaluated logits rows so their PCC against a flag-off oracle can be
  checked at >= 0.995;
- all 100 greedy token IDs exactly match the flag-off artifact for the same
  prompt and seed; no NaN/Inf is present in the saved logits;
- the real 40-layer build completes, all 39 routed layers report eligible
  production dispatch state without duplicated expert tensors, and the weight,
  prefill, and trace snapshots each retain >=10% free DRAM and >=128 MiB
  contiguous free per bank;
- teardown closes the mesh cleanly with no allocator, watcher, fabric, or CCL
  error. Any fallback in a nominally supported layer/bucket is a failure, not a
  partial pass.

Estimated duration: 15–25 minutes per process, 30–50 minutes total.

### Phase C: real vLLM boot, warmup, and cache-off latency comparison

Run two fresh server processes sequentially on the same chips and port. Prefix
caching is explicitly off in both runs so request reuse cannot hide dispatch
cost; hybrid KV, DFlash, speculative decode, and tile-sparse MoE are off. The
baseline differs only by `TT_LAGUNA_MOE_TOKEN_DISPATCH=0`. The candidate needs
the launcher's explicit experimental acknowledgement. Both force program-cache
misses off after prefill warmup and decode trace capture.

Baseline launch:

```bash
cd /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1 && env -u TT_METAL_HOME -u TT_MESH_GRAPH_DESC_PATH -u TT_LAGUNA_SPEC_DECODE PYTHONNOUSERSITE=1 TT_VISIBLE_DEVICES=2,3 LAGUNA_PROFILE=p150x2 LAGUNA_MAX_MODEL_LEN=131072 LAGUNA_MAX_NUM_SEQS=1 TT_LAGUNA_PREFIX_CACHE=0 TT_LAGUNA_HYBRID_KV=0 TT_LAGUNA_DFLASH=0 TT_LAGUNA_STREAMING_PREFILL=1 TT_LAGUNA_MOE_TOKEN_DISPATCH=0 TT_LAGUNA_MOE_PREFILL_TILE_SPARSE=0 TT_LAGUNA_FREEZE_PROGRAM_CACHE=1 TT_LAGUNA_ENFORCE_MEMORY_MARGIN=1 TT_LAGUNA_MIN_DRAM_FREE_FRACTION=0.10 TT_LAGUNA_MIN_CONTIGUOUS_MIB=128 LAGUNA_LOG_DIR=/home/ttuser/laguna-qualification/token-dispatch-20260822/baseline ./serve_vllm.sh
```

Candidate launch, only after the baseline process group is closed and chips 2
and 3 are healthy again:

```bash
cd /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1 && env -u TT_METAL_HOME -u TT_MESH_GRAPH_DESC_PATH -u TT_LAGUNA_SPEC_DECODE PYTHONNOUSERSITE=1 TT_VISIBLE_DEVICES=2,3 LAGUNA_PROFILE=p150x2 LAGUNA_MAX_MODEL_LEN=131072 LAGUNA_MAX_NUM_SEQS=1 LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1 TT_LAGUNA_PREFIX_CACHE=0 TT_LAGUNA_HYBRID_KV=0 TT_LAGUNA_DFLASH=0 TT_LAGUNA_STREAMING_PREFILL=1 TT_LAGUNA_MOE_TOKEN_DISPATCH=1 TT_LAGUNA_MOE_PREFILL_TILE_SPARSE=0 TT_LAGUNA_FREEZE_PROGRAM_CACHE=1 TT_LAGUNA_ENFORCE_MEMORY_MARGIN=1 TT_LAGUNA_MIN_DRAM_FREE_FRACTION=0.10 TT_LAGUNA_MIN_CONTIGUOUS_MIB=128 LAGUNA_LOG_DIR=/home/ttuser/laguna-qualification/token-dispatch-20260822/candidate ./serve_vllm.sh
```

Both environment contracts were run through `serve_vllm.sh config` without
device access. They resolved to physical devices `2,3`, p150x2/ring/two-link,
131072 context, one sequence, streaming prefill on, prefix/hybrid/DFlash/tile
sparse off, and the qualified memory margins. The baseline reported no
experimental overrides; the candidate reported exactly
`TT_LAGUNA_MOE_TOKEN_DISPATCH=1 (qualified=0)` and no other override.

Each timestamped log, rather than a pre-existing `latest.log`, is the source of
truth. Wait for that file's `Application startup complete`, verify
`curl -fsS http://127.0.0.1:8000/health`, and then use a host-only raw-token
qualification client to issue exactly 1024, 2048, 4096, and 8192 input tokens
plus 32 greedy output tokens, three times per length. Save prompt hashes,
returned token IDs, TTFT, TPOT, E2E, the resolved launch header, and the log byte
offset for every run. The candidate client consumes the baseline JSON as its
oracle. This small client/mode must be landed and device-free unit-tested before
the server commands are approved; it should reuse the exact-token streaming
parser in `tests/prefix_cache_qualification.py` rather than a tokenizer or a
text-only benchmark.

Server hard gates:

- the resolved header is p150x2, physical devices `2,3`, port 8000, context
  131072, one sequence, streaming prefill on, and prefix/hybrid/DFlash/spec/tile
  sparse off; candidate dispatch is on only under the printed experimental
  acknowledgement;
- warmup reaches every standard D2 prefill bucket, including 1K/2K/4K/8K,
  across the actual 40-layer stack; `prefill_warmup` and `trace` memory lines
  appear before `Application startup complete`;
- candidate time-to-ready is <= baseline time-to-ready + 120 seconds and is
  <20 minutes absolute. The 53.2-second cold build must occur, if needed, only
  before ready. The log must then show a nonzero frozen program-cache entry
  count and no compile/build activity or forbidden miss during requests;
- every candidate run returns exactly the baseline token IDs. Median candidate
  TTFT is <=1.03x baseline at every length, its geometric-mean TTFT ratio is
  <=0.90, and 8192-token TTFT improves by at least 1.5x. Candidate TPOT and E2E
  are each <=1.03x baseline at every length;
- at `weights`, `prefill_warmup`, and `trace`, free DRAM is >=10% and largest
  contiguous free is >=128 MiB per bank. Candidate steady-state allocation may
  exceed baseline by at most 128 MiB, and candidate trace contiguity may trail
  baseline by at most 64 MiB per bank. Used bytes after every request return to
  within 32 MiB of the post-trace value;
- `/health` passes before and after every request and at the end. The new log
  tail contains no traceback, OOM, allocator error, program-cache miss, watcher
  exception, device hang, NaN/Inf, fabric/CCL timeout, or fallback warning.

Do not use `serve_vllm.sh stop` while chips 0 and 1 are owned by another run,
because that helper resets all devices. Close only this server's process group,
then reset only chips 2 and 3:

```bash
kill -TERM -- "-$(cat /tmp/laguna_vllm_srv.$(id -u).pid)"
sleep 10
kill -KILL -- "-$(cat /tmp/laguna_vllm_srv.$(id -u).pid)" 2>/dev/null || true
tt-smi -r 2 3
```

Estimated duration is 10–15 minutes per server boot plus 3–5 minutes per
request suite. The complete Phase A/B/C tranche is expected to take 70–110
minutes; use a two-hour hard timeout. Stop immediately on a numerical, memory,
program-cache, or device-health failure. Passing this tranche justifies review
for promotion; it does not itself change the default.
