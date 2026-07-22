# GPT-OSS 20B optimized decoder

This stage provides the single-device optimized decoder layer in
`tt/optimized_decoder.py`. It starts from the fused decoder and owns both
prefill and decode forwards. It does not include multichip, full-model, or vLLM
work.

## Selected implementation

The primary batch-one path keeps packed QKV and TTNN composite attention, then
replaces dense all-expert execution with the repo GPT-OSS routed `Experts`
module and `ttnn.sparse_matmul`:

- BFP8_B/LoFi routed gate, up, and down expert matmuls;
- 9x10 grids, `in0_block_w=45`, 1x1 output subblocks;
- advisor-seeded L1 decode norm, packed-QKV, O-projection, residual, and router
  layouts;
- explicit 8x8 decode SDPA with a 32-token K chunk;
- BFP8_B persistent K/V cache, while current-token K/V remain BF16 before
  cache writes;
- explicit 10x4 2D QKV/O program configs at S=128;
- internal expert padding and output slicing for arbitrary logical sequence
  lengths.

The batch-one constructor releases inherited dense expert tensors after sparse
weights load, so the measured path cannot silently fall back. Batch 2 retains
the exact fused dense expert compatibility path because the shared sparse
expert module is currently batch-one-only.

## Result

On one Blackhole P300c endpoint, the final same-process S=17 gate measured:

| Path | Warmed prefill mean | Traced decode mean |
| --- | ---: | ---: |
| fused | 7.331439 ms | 5.904240 ms |
| optimized | **4.107129 ms** | **0.846833 ms** |

This is a 44.0% prefill reduction and an 85.7% decode reduction. It also beats
the earlier optimized artifact's 0.928144 ms decode by 8.8%. At S=128 the
selected 10x4 2D sparse path is correct but measures 13.04211 ms versus
9.56244 ms fused because the padded group activates all 32 experts.

Final real-weight PCC remains above the 0.99 functional bar:

| Layer kind | Prefill output | Decode output |
| --- | ---: | ---: |
| layer 12, sliding attention | 0.99024635 | 0.99576329 |
| layer 13, full attention | 0.99308115 | 0.99419248 |

S=3/17/33 non-aligned prefill, S=128 boundary prefill, batch 2, repeated paged
decode positions 3-18, deterministic reruns, and ten trace replays all pass.
The validated context boundary advances from 33 to the configured extent 128.

## Required advisor and profiling artifacts

The mandatory fresh shard-advisor capture is under `shard_advise/`:

- `report.json` — 43 ops, 40 choices, zero spills, no unfixable ops;
- `final_ir.mlir` — the compiler-validated layout/program graph;
- `advise_gpt_oss.py` — the rewritten dense capture harness.

Advisor attention/residual/router recommendations are applied. The complete
advisor dense-MoE L1 chain was adapted and measured: it is correct and improves
dense decode to 5.70996 ms, but is rejected versus 0.84683 ms routed sparse
decode. See `work_log.md` for every recommendation and candidate disposition.

Compact final `tt-perf-report` artifacts are under `tracy/final/`. The profiled
decode window contains 860.461 us of device ops plus 80.932 us of gaps. Sparse
gate/up/down account for 45.6% and are visibly BFP8/LoFi in the report. The
profile and watcher were collected in separate runs.

## Validation

```bash
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py -s
```

Final correctness: 8 passed, 2 intentional opt-in skips. The performance gate
is enabled with `RUN_OPTIMIZED_DECODER_PERF=1`. The profiler window is enabled
with `RUN_OPTIMIZED_DECODER_PROFILE=1` and must not be combined with watcher.

The final watcher run uses `TT_METAL_WATCHER=10`, disables Ethernet watcher
checks for this one-chip endpoint, enables TTNN fallback exceptions, and passes
both real layer kinds plus repeated and traced cache integrity without a device
assert or NoC error.

## Evidence map

- `work_log.md`: topology audit, commands, PCC/performance matrices, advisor
  dispositions, profiler interpretation, limitations, and optimize checklist.
- `logs/run_20260722_final_correctness_rerun.log`: final full suite.
- `logs/run_20260722_final_perf.log`: final same-process baseline comparison.
- `logs/run_20260722_watcher.log`: watcher-only correctness run.
- `tracy/final/{prefill,decode}_perf_report.{csv,txt}`: compact op reports.
- `../context_contract.json`: updated BFP8 cache and S=128 validation contract.
