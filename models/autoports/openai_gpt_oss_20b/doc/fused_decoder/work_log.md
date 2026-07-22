# Fused decoder work log

Date: 2026-07-22 UTC

## Scope, source, and device

The functional source commit is
`f972e13d4d699e7bfa3ca6bde66a9dc069aa8993`. Work is limited to
`tt/fused_decoder.py`, its fused tests/drivers, `doc/fused_decoder/`, and the
fused memory update in `doc/context_contract.json`. No optimized-decoder,
multichip, full-model, or vLLM source was changed.

```text
timeout 60 tt-smi -s
PASS: four local Blackhole P300c devices healthy; device 0 selected; no reset.

TT_VISIBLE_DEVICES=0
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

Firmware is 19.8.0; the runtime warns that 19.5.0 is the newest fully tested
Blackhole bundle. The P150 descriptor intentionally opens one chip from the
P300c endpoint. All hardware commands were serialized. Watcher and profiler
were never enabled together.

Initial synchronized synthetic baseline before rewrites:

| Path | Mean | Minimum | Iterations |
| --- | ---: | ---: | ---: |
| Functional warmed prefill | 8.241 ms | 8.047 ms | 5 |
| Functional eager decode | 6.558 ms | not retained | 5 |
| Functional traced decode | 5.996 ms | 5.979 ms | 100 |

## Accepted rewrites

1. Replaced manual prefill attention with sink-aware causal SDPA over logical
   K/V while preserving the cache-fill side effect.
2. Removed decode's explicit mask slice/repeat and used causal decode SDPA with
   a position tensor.
3. Cached RoPE and update-index views after warmup. Before this fix, changing a
   scalar RoPE index reused the first compiled value and K PCC fell to 0.970 on
   the second update; positions 3–18 now remain correct.
4. Removed decode Q-to-DRAM and concat-to-DRAM copies, kept concat output
   sharded through O projection, and replaced a permutation with reshape.
5. Folded biases into `linear`, sigmoid into multiply, and reused safe
   elementwise buffers with `output_tensor`.
6. Split gate/up weights once at construction for multi-token prefill. Kept the
   single wide gate/up projection for one-token decode because it wins latency.

## Candidate matrix and rejected options

Synthetic candidate fixture (five prefill calls, 100 trace replays):

| Policy | Prefill mean/min | Traced decode mean/min | Interpretation |
| --- | ---: | ---: | --- |
| `auto` (final) | 7.278/7.105 ms | 5.893/5.886 ms | split prefill, wide decode |
| `wide` | 7.471/7.424 ms | 5.901/5.887 ms | prefill slower than split |
| `split` | 7.166/7.090 ms | 6.793/6.784 ms | decode 0.900 ms slower than auto |
| `sparse` | 18.415/18.161 ms | 2.255/2.242 ms | timing only; synthetic expert weights are zero |
| `sparse_split` | 16.709/16.430 ms | 1.852/1.845 ms | timing only; synthetic expert weights are zero |

Real layer-12 controls use the authoritative Float32-random-to-BF16 input:

| Decode policy | PCC | Traced mean/min | Decision |
| --- | ---: | ---: | --- |
| wide | 0.999298 | 5.906/5.892 ms | final decode topology |
| split | 0.999298 | 6.795/6.786 ms | correct, slower |
| packed BF16 sparse | 0.702504 | failed correctness gate | rejected |
| split BF16 sparse | 0.605508 | failed correctness gate | rejected |

Sparse follow-ups also rejected BF8 weights (PCC 0.702890) and
`in0_block_w=1` (PCC 0.697125); `in0_block_w=2` was restored. A 5x6 grid is
the largest valid tested sparse receiver/worker topology; 8x8 fails its core
contract. Output subblock width 2 slightly regressed device time, and explicit
L1 input copies regressed trace latency. These experiments and corrected
hypotheses are in `AUTODEBUG.md`, `AUTOFIX.md`, and the retained candidate
logs. `$autofix` succeeded: sparse remains evidence-only and default `auto`
returns to the exact split-prefill/wide-decode graph.

Other constrained candidates:

| Candidate | Result | Decision |
| --- | --- | --- |
| Sharded GQA decode-SDPA output | `Sharded output not supported for GQA` | retain one required conversion |
| Remove post-RoPE prefill slices | K logical 32 versus V logical 4 at non-aligned input | retain logical slices |
| Residual-add + RMSNorm | residual sum is needed again; recompute/copy removes dispatch benefit | reject |
| Generic GLU/SwiGLU | cannot express both clamps, `up+1`, and 1.703125 coefficient | reject |
| GPT-OSS `moe_compute`/TTMoEDecode | packed low-precision and per-device expert contract mismatch | reject |

The exhaustive pattern disposition is in `README.md`. Every candidate that
could preserve this single-device graph was tested or rejected from its op
contract; convolution, embedding, distributed-norm, and fused-collective
patterns do not occur in Stage 02.

## Correctness, stress, watcher, and fallback gates

Final complete suite:

```text
env TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  script -q -e -c \
  'pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_fused_decoder.py' \
  models/autoports/openai_gpt_oss_20b/doc/fused_decoder/logs/final_suite.log

PASS: 9 tests in 26.04s; COMMAND_EXIT_CODE=0.
```

The tests prove a distinct fused runtime, non-aligned S=3/17/33, batch 2,
bitwise determinism, real layer 12 sliding attention, real layer 13 full
attention, exact real split-prefill MoE at three lengths, paged positions
3–18, full-prefix/unchanged-suffix cache integrity, ten deterministic trace
replays, and a same-process performance win. PCC details are in `README.md`.

Watcher/fallback-clean correctness, run separately from profiling:

```text
env TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  script -q -e -c \
  'pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_fused_decoder.py -k "not beats"' \
  models/autoports/openai_gpt_oss_20b/doc/fused_decoder/logs/watcher_correctness.log

PASS: 8 tests, 1 deselected in 29.09s; COMMAND_EXIT_CODE=0; watcher attached,
checked, and detached without error; fallback exceptions enabled.
```

The batch-2 case emits an internal TTNN reshape memory-config warning and
continues on-device; it is not a Torch/host fallback. The measured batch-1
graph has the exact layout operations reported below.

## Final latency gate

The canonical same-process gate was rerun alone after the final topology:

```text
env TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  script -q -e -c \
  'pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_fused_decoder.py -k beats' \
  models/autoports/openai_gpt_oss_20b/doc/fused_decoder/logs/performance_gate.log

functional_perf={'prefill_mean_ms': 8.2064880291,
                 'prefill_min_ms': 7.9693612643,
                 'decode_traced_mean_ms': 5.9883454186,
                 'decode_traced_min_ms': 5.9737297706}
fused_perf={'prefill_mean_ms': 7.1583688259,
            'prefill_min_ms': 7.0873210207,
            'decode_traced_mean_ms': 5.8913813694,
            'decode_traced_min_ms': 5.8830478229}
PASS: 1 test, 8 deselected in 7.05s; COMMAND_EXIT_CODE=0.
```

Fused improves prefill mean/minimum by 12.77%/11.07% and traced decode
mean/minimum by 1.62%/1.52%. The complete suite independently records the same
directional win.

## Final profiler commands and conclusions

```text
env TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  timeout 300 python -m tracy -p -r --check-exit-code \
  -o models/autoports/openai_gpt_oss_20b/doc/fused_decoder/perf/reports \
  -n fused_prefill_final \
  models/autoports/openai_gpt_oss_20b/tests/fused_decoder_profile.py --path prefill

env TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  timeout 300 python -m tracy -p -r --check-exit-code \
  -o models/autoports/openai_gpt_oss_20b/doc/fused_decoder/perf/reports \
  -n fused_decode_final \
  models/autoports/openai_gpt_oss_20b/tests/fused_decoder_profile.py --path decode
```

`tt-perf-report` selected the `FUSED_PREFILL`/`FUSED_PREFILL_END` and
`FUSED_DECODE`/`FUSED_DECODE_END` signposts from the final raw op CSVs and
emitted the processed CSV/PNG artifacts listed in `README.md`.

Prefill sums to 7,028.28 us across 52 ops: matmul is 85.70%, there is one
dedicated SDPA, and there is no explicit layout-conversion op. Decode sums to
5,855.63 us across 56 ops: five matmuls are 80.46%, there is one decode SDPA,
two paged cache updates, and exactly one 0.58 us interleaved-to-sharded GQA
handoff. No Torch conversion, host fallback, collective, or explicit reshard
appears. Internal TTNN tilize/untilize work around exact router/MoE formats was
compared against the split candidate; removing it costs about 0.90 ms traced.

## Context and stage-review bookkeeping

- `doc/context_contract.json` now includes the 2.708 GB fused static weight
  footprint and advances the tested boundary to S=33 without reducing cache
  extent or the prior supported boundary.
- Initial independent review returned `more-work-needed`: trace/cache integrity
  was incomplete, candidate evidence was not durable, and memory accounting
  was stale. The tests now compare complete caches and unchanged suffixes in
  eager/trace loops, all candidate logs/drivers are retained, and the context
  contract includes split-weight storage.
- Fresh independent rereview verdict: `clean-pass`. The reviewer independently
  matched the 17:56 prefill/decode raw op sequences to the current
  split-prefill/wide-decode source, found no material hard-check gap, and
  confirmed all three prior findings closed. See `stage_review.md`.
- Local Stage 02 commit SHA: pending.
- No push will be performed.
