# Fresh verification — 2026-07-31

The no-fusion optimized-decoder checkpoint was reapplied to functional base
`c3cc345a10b` as local commit `c55a8c067c8` and rerun on the current four-chip
Blackhole host. Every model command used one 1x1 mesh and closed it before the
next command.

## Static and contract checks

```text
pytest -q \
  models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py \
  models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py \
  models/autoports/qwen_qwen3_6_27b/tests/test_linear_prefill_long_state.py
```

Result: 9 passed. This includes optimized-path independence, the runtime
host/fallback source audit, the unchanged context contract, and non-aligned
near-context state logic.

## Fresh correctness and warmed timing

All numerical commands set `ttnn.CONFIG.throw_exception_on_fallback = True`.
Decode numbers are traced state-mutating replays. Prefill lengths 33 and 5 are
intentionally not tile/chunk aligned.

| Path | Batch | PCC | Warmed latency |
|---|---:|---:|---:|
| full decode step 1 / 2 | 1 | 0.999008319 / 0.999583101 | 1.066906 ms |
| full decode step 1 / 2 | 32 | 0.999592518 / 0.999824190 | 1.368461 ms |
| linear decode step 1 / 2 | 1 | 0.999986797 / 0.999987234 | 2.289250 ms |
| linear decode step 1 / 2 | 32 | 0.999967587 / 0.999990453 | 20.616257 ms |
| full prefill, seq 33 | 1 | 0.999991614 | 3.110356 ms |
| full prefill, seq 33 | 32 | 0.999991089 | 49.799153 ms |
| linear prefill, seq 5 | 1 | 0.999997419 | 12.277131 ms |
| linear prefill, seq 5 | 32 | 0.999997010 | 294.746846 ms |

The optimized BFP8 paged-cache prefill-to-decode test passed at PCC
0.999993917 and verified distinct physical page occupancy for key and value.

Fresh functional batch-32 prefill baselines used the corresponding full-layer
smoke harnesses with one warmup and one signposted measurement. Full attention
seq33 measured 72.456282 ms versus optimized 49.799153 ms; linear attention
seq5 measured 316.626799 ms versus optimized 294.746846 ms. Both optimized B32
paths improve. Complete PTY-captured stdout/stderr, recorder command, exit
status, fallback configuration, and device closure are retained separately in
`candidates/functional_full_prefill_b32_raw.log` and
`candidates/functional_linear_prefill_b32_raw.log`.

The original candidate matrix was recovered with its source-session provenance
under `candidates/recovered_matrix.log`. It contains the B1/B32 topology,
precision, cache, and geometry comparisons plus official-weight controls and
exact L1 blockers.

The rereview autofix added two missing topology/config families. The legal
residual-sharded chain improved full decode at both batches and is now the
default; see `candidates/residual_sharded_chain_autofix.log`. Explicit 2D
prefill configs were adapted through multiple L1/grid failures and measured at
both batches; B1 was slower and B32 retained a hard L1 blocker, as recorded in
`candidates/large_prefill_2d_autofix.log`.

Fresh final-default profiler evidence is under
`tracy/fresh_final_full_b1/`. The advice-enabled report and filtered CSV identify
the dominant rows as:

- packed QKV: 162 us, LoFi BF16 x BFP8 -> BF16, DRAM-sharded;
- output projection: 72 us, LoFi BF16 x BFP8 -> BF16, DRAM-sharded;
- packed gate/up: 328 us, LoFi BF16 x BFP4 -> BF16, 109-core L1 path;
- down projection: 189 us, LoFi BF16 x BFP8 -> BF16, DRAM-sharded.

## Watcher

Separate, non-profiled batch-32 runs passed for both layer kinds:

```text
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  python models/autoports/qwen_qwen3_6_27b/tests/optimized_traced_synthetic_pcc.py \
  --kind full --batch 32

TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  python models/autoports/qwen_qwen3_6_27b/tests/optimized_traced_synthetic_pcc.py \
  --kind linear --batch 32
```

The scoped ETH disable avoids unrelated active-Ethernet watcher-buffer noise
for this single-device stage. Compute, NoC, CB/L1, and kernel assertion checks
remained enabled. Both commands closed devices normally.
