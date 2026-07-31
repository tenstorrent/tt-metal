# Optimized decoder work log

## 2026-07-31 — rereview autofix

- The first remediated rereview found missing large-prefill program-config,
  residual-chain, and raw B32 functional-control evidence.
- Added and adapted an explicit 2D multicast prefill candidate for QKV, O,
  packed gate/up, and down. Full/linear B1 were correct but did not beat the
  retained path. Multiple reduced-block B32 retries still exceeded the hard
  1,572,864-byte L1 limit; exact configs and errors are in
  `candidates/large_prefill_2d_autofix.log`.
- Measured the longest compatible width-sharded full-decode residual chain.
  It reduced B1 from 1.218025 to 1.066906 ms and B32 from 1.518185 to
  1.368461 ms, so it was promoted to the default. PCC remained above the
  functional bar. The next packed gate/up consumer's L1-interleaved boundary
  is explicit and profiled.
- Saved complete separate functional B32 stdout/stderr streams, recorder
  commands, exit status, fallback configuration, and device closure in
  `candidates/functional_{full,linear}_prefill_b32_raw.log`.
- Reran 9 static/contract tests, all four traced decode shapes, a watcher-clean
  full B32 run, and a separate final B1 Tracy profile after promotion.

## 2026-07-31 — fresh checkpoint verification

- Reapplied the completed no-fusion optimized-decoder stage directly to
  functional checkpoint `c3cc345a10b`; local stage commit:
  `c55a8c067c8`.
- Confirmed the stage-owned commit contains only the optimized implementation,
  Qwen tests, context metadata, and optimized-decoder documentation/evidence.
  It does not add fused-decoder, multichip, full-model, or vLLM source.
- Reran static/contract tests, traced state-mutating decode at batch 1 and 32
  for both layer kinds, non-aligned prefill at both batches, paged-cache
  prefill-to-decode routing, and separate watcher-10 batch-32 runs.
- Fresh results reproduce the selected path. See
  `fresh_verification_20260731.md` for commands and exact PCC/latency values.
- Recovered the missing compact candidate matrix from the original stage
  session with provenance in `candidates/recovered_matrix.log`.
- Added fresh same-shape functional B32 prefill baselines and a reduced final
  B1 Tracy profile whose raw/filtered rows prove the selected runtime
  dtype/fidelity policy.
- Four Blackhole devices were visible before testing. Hardware remained
  healthy and every command closed its 1x1 mesh normally; no reset or recovery
  was required.

Date: 2026-07-29. Hardware: one Blackhole p300c from a four-device host.
Starting commit: `5c27b175b68`. Stage commits are recorded at the end.

## Commands and evidence

Hardware commands were serialized. Watcher and profiler were separate runs.

```bash
python models/autoports/qwen_qwen3_6_27b/tests/optimized_traced_synthetic_pcc.py --kind {full,linear} --batch {1,32}
python models/autoports/qwen_qwen3_6_27b/tests/optimized_{full,linear}_attention_synthetic_pcc.py --mode prefill --sequence {33,5} --batch {1,32}
python models/autoports/qwen_qwen3_6_27b/tests/optimized_linear_attention_real_pcc.py
python models/autoports/qwen_qwen3_6_27b/tests/optimized_full_attention_cache_pcc.py
TT_METAL_WATCHER=10 python models/autoports/qwen_qwen3_6_27b/tests/optimized_traced_synthetic_pcc.py --kind <kind> --batch <batch>
python -m tracy -r -p -v -o <artifact-dir> <test-script> <arguments>
tt-perf-report <ops-csv> --start-signpost <PERF_DECODE|PERF_PREFILL> --end-signpost <..._END> --csv <filtered.csv> --summary-file <summary>
```

Static checks:

```bash
python -m py_compile models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py models/autoports/qwen_qwen3_6_27b/tests/optimized_*py
pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py
```

## Advisor command

Part B was run in fresh shells. The bootstrap rewrote path variables because
the checkout is symlinked, so the repo paths were restored after sourcing.
The advisor virtualenv also needed tt-metal's `tools` on `PYTHONPATH` for
`tracy`.

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source "$TTMLIR_ADVISOR_HOME/scripts/bootstrap.sh"
export TT_METAL_HOME=/home/mvasiljevic/tt-metal
export RUNTIME_ROOT=/home/mvasiljevic/tt-metal
export BUILD_HOME=/home/mvasiljevic/tt-metal/build_Release
export PYTHONPATH=/home/mvasiljevic/tt-metal:/home/mvasiljevic/tt-metal/tools:$PYTHONPATH
ttnn-advise capture models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/advise_qwen.py \
  --output-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/shard_advise
```

The required report and final IR exist. An additional batch-1 capture is in
`shard_advise_batch1`. See README for every applied/rejected recommendation.

## Failures that changed the implementation

- Sharing decode DRAM-sharded weights with prefill caused matmul memory-contract
  failures. Persistent interleaved prefill and DRAM-sharded decode copies fix
  this without runtime transfers.
- Batch-32 full prefill exposed a rank/layout-dependent RoPE reshape. Using
  `unsqueeze_to_4D` plus transpose preserves batch 1 and enables batch 32.
- Packed gate/up DRAM-sharded `in0_block_w=10` and `5` exceeded L1. The adapted
  `2` retry passed and tied the advisor 1D path.
- Official-weight full attention initially catastrophically disagreed with HF.
  The audit found an inherited contiguous Q/gate split where Qwen requires a
  per-head split. Fixing the optimized path raises real PCC to 0.998369. The
  corrected oracle rejects BFP4 attention at 0.987799.

## Optimize checklist

- [x] operation-topology audit recorded
- [x] functional and best-correct baselines recorded
- [x] real shapes, both layer kinds, batches 1 and 32
- [x] precision/fidelity policy swept and profiler-verified
- [x] packed same-input QKV and gate/up compared with split control
- [x] shard-advise run this pass; mandatory report and final IR saved
- [x] advisor layouts/configs applied first, then measured
- [x] DRAM-sharded decode matmuls and multiple gate/up block widths tried
- [x] explicit large-prefill 2D configs adapted and measured at B1/B32
- [x] longest feasible residual-sharded chain measured and promoted
- [x] composite paged/chunked SDPA retained
- [x] warmed non-aligned prefill and traced decode measured separately
- [x] before/after PCC and latency recorded
- [x] no runtime Torch/from_torch/to_torch or host fallback
- [x] paged cache, repeated trace mutation, determinism/alias guards
- [x] watcher-clean and separate final profiler runs
- [x] context contract updated for BFP8 KV capacity
- [x] independent stage rereview clean-pass after evidence remediation
- [x] local implementation/evidence checkpoint `c55a8c067c8`, never pushed

## Commits

- functional base: `c3cc345a10b`
- optimized implementation and historical evidence: `c55a8c067c8`
- residual-chain optimization, evidence remediation, and clean rereview:
  `81438cb467b`
