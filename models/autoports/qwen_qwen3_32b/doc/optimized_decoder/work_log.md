# Optimized decoder work log

Date: 2026-07-16 UTC
Model: `Qwen/Qwen3-32B`
Hardware: one Blackhole p300c selected with `TT_VISIBLE_DEVICES=2,3`
Representative layer: 32 of 64 (the sole dense layer kind)
Starting checkpoint: `f42cc399282`

## Scope and evidence identity

Work was restricted to `tt/optimized_decoder.py`, its model-local tests, and
documentation. No multichip, full-model, generation, or vLLM work was started.
The functional decoder remained the correctness/performance control.

Final source identities embedded in correctness/performance artifacts:

| File | SHA-256 |
|---|---|
| `tt/optimized_decoder.py` | `b136bfe65d8453a991b8870c591c8601d39ccad391111629408d2e73b25197b4` |
| `tests/test_optimized_decoder.py` | `e249c009d8a8addc70da4c5bd7ff70cae14b55e7299f1b70a013bfff67cbe936` |
| `tests/test_context_capacity.py` | `3246ee7a1ccd4b4c4706354404ffa83beddb522625ceecf020f6817deffdbdbe` |
| `doc/context_contract.json` | `c003fad34c835d9a2ac33b96749d45fc7c62614296af6e9d9fae44e636c56b7c` |

Real-weight/activation evidence uses HF checkpoint revision
`9216db5781bf21249d130ec9da846c4624c16137`. The captured input to
`model.layers.32` is `activations/layer32_prompt_inputs.pt`, SHA-256
`03cd6658c365761d09a30b9f1d357bbd8b6daec6420eec72e3d95c4fe691ad3e`,
shape `[1,21,5120]`, BF16. One prompt-derived boundary activation is repeated
across the emitted batch-32 slots. `capture_layer32_activations.py` records the
prompt-token hash, index/config hashes, and every checkpoint shard hash.
The stronger non-aligned/repeated-run gate uses
`activations/layer32_prompt_repeated35_inputs.pt`, SHA-256
`bfa06cb3e398927b72c51f8873100bdebb092520265e88e08fcd2c88e1a27e07`,
shape `[1,35,5120]`. It runs HF layers 0-31 on the original real prompt-token
sequence repeated only to reach 35 tokens; it is not a repeated hidden vector.

## Audit and rewrite order

Before knob tuning, the operation graph was audited for repeated same-input
matmuls, composite replacements, layout conversions, fusion, and data movement.
Actions were:

1. Retain/own packed QKV instead of split Q/K/V projections.
2. Use TTNN QKV head creation/concat, RoPE, paged cache updates, and prefill/
   decode SDPA rather than hand-built attention primitives.
3. Fold SiLU into the gate/up multiply.
4. Preserve a 40-core width-sharded decode residual/norm/gate/up chain, use the
   independently selected 32-core down projection, and only cross layouts at
   composite/core-geometry boundaries.
5. Use DRAM width-sharded decode weights and role-specific matmul configs.
6. Replace per-position trace state with a bounded, fixed-address slot and
   preallocated embedding outputs.
7. Materialize the final public DRAM output directly.
8. Internally chunk long/non-aligned prefill without public divisibility rules.

The full current-op/candidate/action/evidence table is in `README.md`.

## Mandatory shard-advisor gate (OPT-015)

The advisor ran this pass after the rewrite, in a separate bootstrap shell as
required by `.agents/skills/shard-advise/SETUP.md` Part B:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
cd "$TTMLIR_ADVISOR_HOME"
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
export PYTHONPATH=/home/mvasiljevic/tt-metal/python_env/lib/python3.12/site-packages:/home/mvasiljevic/tt-metal:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-}
ttnn-advise capture \
  /home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_32b/doc/optimized_decoder/advise_qwen3_32b.py:decode \
  --out /home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_32b/doc/optimized_decoder/shard_advise
```

Result: 26 captured ops, 23 final choices, spill pass ran with one spill.
Required artifacts are nonempty:

| Artifact | SHA-256 |
|---|---|
| `shard_advise/report.json` | `af538b90b1c0da0fe0eb05851892208e5c3c888cb5109725521fd646398453f2` |
| `shard_advise/final_ir.mlir` | `f01e034478e12a86960528aa8d2a956097e1bd9c9c640fb3495614704a37334a` |

The report was translated into an executable
`decode_matmul_mode="shard_advisor"` candidate: initial/final residual L1
interleaved; residual norms B11; QKV W107; Q norm B40 on 4x10; K norm B32 on
4x8; Q/K RoPE H64/H32; gate/up W100; down input W80; exact 1-D program configs
including output blocks/subblocks. `nlp_concat_heads_decode` requires a sharded
input (also the report's unfixable constraint), so the legal trial preserves
that input and explicitly materializes the advised interleaved O input.

| Recommendation | Decision | Evidence |
|---|---|---|
| BF8 attention/lower-precision MLP grouping | Applied as the advisor seed | Precision was independently swept; final uses BFP4 projections and BFP8 cache |
| Sharded residual/norm/linear skeleton | Applied directionally | Final keeps the measured 40-core L1 width-sharded chain |
| Full advised head/norm/RoPE/layout family | Rejected | Decode PCC 0.985269 fails; 5.627/1.743 ms |
| Advised 1-D projection configs with production head layouts | Rejected | Correct at 0.999272 decode PCC, but 5.645/1.737 ms |
| DRAM-sharded production matmuls | Selected over advisor | Final is correct at 0.998990 and 5.477/1.217 ms |

Raw evidence: `results/candidates/advisor_full_vs_matmuls.json`. The first
constraint failure was not treated as a rejection; the full legal family and a
matmul-only isolate were both run.

## Candidate evidence

Unless marked synthetic, rows use real layer-32 weights, prompt-derived HF
boundary activations, batch 32, prefill length 17, position 17, warmed prefill,
and traced warmed decode. The acceptance bar is PCC >= 0.99.

### Precision and fidelity

| Candidate | Prefill PCC | Decode PCC | Prefill ms | Decode ms | Decision |
|---|---:|---:|---:|---:|---|
| Functional BF16 | 0.999999627 | 0.999869502 | 83.252 | 82.101 | Control |
| All BFP8, HiFi2 | 0.999999628 | 0.999931387 | 6.283 | 2.017 | Correct, slower |
| All BFP8, LoFi | 0.999999346 | 0.999889741 | 5.940 | 1.395 | Correct, slower |
| BFP4 MLP, MLP HiFi2 | 0.999997490 | 0.999248607 | 6.121 | 1.971 | Correct, slower |
| BFP4 MLP, attention HiFi2 | 0.999997490 | 0.999248607 | 5.626 | 1.368 | Earlier correct baseline |
| Gate/up BFP4, down BFP8 | 0.999998397 | 0.999511423 | 5.735 | 1.425 | Correct, slower |
| BFP4 MLP, BF16 KV | 0.999997490 | 0.999280187 | 5.339 | 1.402 | Prefill win, decode loss |
| Attention BFP4 | 0.999999113 | 0.999686546 | 6.142 | 1.869 | Correct on real input, slower |
| BFP4 MLP + attention LoFi | 0.999997442 | 0.999275695 | 5.570 | 1.247 | Correct earlier family |
| All projections BFP4, down 40 | 0.999996978 | 0.999002692 | 5.467 | 1.219 | Correct real candidate |
| All projections BFP4, down 32 | 0.999996978 | 0.998990068 | 5.477 | 1.217 | Selected; precise 500-replay win |

The early short synthetic sweep is retained as historical stress evidence in
`results/candidates/precision_selection_stress.json`:

| Policy | Synthetic prefill PCC | Synthetic decode PCC | Decision |
|---|---:|---:|---|
| BFP4 MLP, attention HiFi2 | 0.991965 | 0.991702 | Pass |
| BFP4 MLP, attention LoFi | 0.990090 | 0.990386 | Pass and select |
| All projections BFP4 | 0.975395 | 0.972469 | Fails this artificial distribution; investigate on real boundaries |

Per OPT-012, the artificial random distribution is not allowed to veto a
policy that passes real model boundaries. All-BFP4 was therefore rerun on the
original prompt boundary, advancing positions 17-20, and the independent
35-token HF boundary artifact at non-aligned length 31 plus positions 31-34.
All selected-policy output/cache PCCs pass 0.99: length-31 prefill is 0.999995,
decode is 0.998826-0.999354, and cache PCC is 0.994387 or better. The seeded
random test remains useful as a separate diagnostic and explicitly runs the
conservative optimized BFP8-attention/BFP4-MLP policy; it cannot silently
replace or reject the final path.

### Geometry, programs, layouts, and composites

| Sweep | Correct results (traced decode ms unless prefill) | Decision |
|---|---|---|
| Original shared MLP cores 16/20/32/40/80 | 1.458 / 1.461 / 1.368 / 1.367 / 1.406 | Seed 40; follow with role-specific sweep |
| Coherent all-BFP4 gate/up 16/20/32/40/80 | block-10 L1 fail; block-8 L1 fail; 1.224; 1.219; 1.247 | Phase-specific input shards exercise 10/8/5/4/2 tiles; keep 40 |
| All-BFP4 down 16 blocks 50/25/10/5 | L1 fail / 1.220 / 1.232 / 1.256 | Largest legal passing block 25 is near-tie; precisely slower |
| All-BFP4 down 20 blocks 40/20/10/8/5 | L1 fail / 1.223 / 1.234 / 1.240 / 1.259 | Largest legal passing block 20 is best for 20 cores but loses |
| Precise down/gate tie-break, 500 replays | down32 1.217996; down40 1.219028; down16/b25 1.219652; down20/b20 1.222648; gate32 1.224401 | Select down 32 |
| Packed gate+up, 40 cores | 1.551, PCC 0.999292 | Reject versus separate; initial L1 failure was adapted to legal block 2 |
| Decode `in0_block_w` cap 2/5/10/default | 1.564 / 1.413 / 1.378 / 1.367 | Keep role-specific uncapped defaults |
| Decode SDPA 8x4/8x8 | 1.375 / 1.367 | Keep 8x8 |
| Decode approximate/exact exponent | 1.368 / 1.367 | Keep exact |
| Prefill block 2/8/10 | 5.975 / 5.593 / 5.588 | Compare 8/10 more precisely |
| Prefill block 8/10, 25 iterations | 5.638 / 5.590 | Keep 10 |
| Prefill block 16 | L1 request 2,044,672 > 1,572,864 bytes | Retry/adapt to legal block 10; structured evidence saved |
| Prefill MLP L1-input chain | 6.219 versus 5.635 contemporary DRAM input | Reject whole adapted chain |
| Full advisor / advisor matmuls only | 1.743 fail-PCC / 1.737 correct | Reject both versus DRAM-sharded 1.217 |

Candidate JSONs under `results/candidates/` preserve full configs, PCCs,
latencies, and source hashes. No configuration was rejected solely on its first
TTNN/API error: all-BFP8 was retried with legal prefill block 4 after block 8
exceeded L1; packed MLP was retried with block 2; prefill block 16 was adapted
to block 10; advisor head layouts were isolated from its matmuls. The final
review remediation added a phase-specific gate/up input shard and reran the
all-BFP4 family in `all_bfp4_role_geometry_v2.json`: gate16/block10 requests
2,093,312 L1 bytes and gate20/block8 requests 1,739,520 versus 1,572,864, while
gate16/block5, gate20/block4, gate32/block5, gate40/block4, and gate80/block2
all execute. Down16/block50 requests 2,087,168 bytes and down20/block40 requests
1,700,608; the intervening blocks 25/10/5 and 20/10/8/5 all execute. The
500-replay ranking is saved in `all_bfp4_role_geometry_v2_precise.json`.

## Final commands and outcomes

### Ordinary optimized regression

```bash
TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
pytest -q models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py -s --disable-warnings
```

Result: 5 passed and the two opt-in performance/profile tests skipped in
55.85 s. This independently reproduces the conservative synthetic diagnostic,
the final real length-17 and length-31 gates, and
advancing-trace PCCs without Watcher or profiler instrumentation.

### Same-run before/after timing

```bash
TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_RUN_PERF=1 \
QWEN3_32B_PERF_CANDIDATES=functional_bf16,optimized_all_bfp4_lofi_down32c \
QWEN3_32B_PREFILL_PERF_ITERATIONS=25 \
QWEN3_32B_DECODE_PERF_ITERATIONS=200 \
QWEN3_32B_RESULTS_DIR=models/autoports/qwen_qwen3_32b/doc/optimized_decoder/results/final \
QWEN3_32B_RESULT_NAME=before_after.json \
pytest -q models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_candidates -s --disable-warnings
```

Result: functional 83.252/82.101 ms and optimized 5.477/1.217 ms; optimized
PCC 0.999996978/0.998990068. The final default beats the strongest retained
earlier correct 1.366718 ms trace by 10.93%; that row is preserved in
`results/candidates/decode_program_configs.json` with PCC and full config.

### Watcher correctness, stress, repeats, and advancing trace

```bash
TT_VISIBLE_DEVICES=2,3 TT_METAL_WATCHER=10 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_RESULTS_DIR=models/autoports/qwen_qwen3_32b/doc/optimized_decoder/results/final \
pytest -q \
  models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py::test_optimized_runtime_is_implementation_owned_and_host_free \
  models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py::test_optimized_synthetic_non_aligned_prefill_decode_and_repeats \
  models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py::test_optimized_real_recorded_non_aligned_prefill_decode_and_repeats \
  models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py::test_optimized_real_weight_prefill_decode \
  models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py::test_traced_decode_advances_position_and_cache \
  -s --disable-warnings
```

Result: 5 passed in 59.29 s. The real-weight test inspected
`generated/watcher/watcher.log` for `error/assert/hang/stuck/timeout`; no matches.
The selected policy also passed non-aligned length 31 and positions 31-34.
Advancing positions 17-20 passed output/key/value history checks with stable
buffer addresses and changing outputs. The timing test separately asserts
bitwise-equal repeated trace output for unchanged inputs/state.

### Capacity

```bash
TT_VISIBLE_DEVICES=2,3 QWEN3_32B_CONTEXT_DECODER=optimized \
QWEN3_32B_CONTEXT_PROBE_LEN=8192 \
QWEN3_32B_CONTEXT_RESULTS_DIR=models/autoports/qwen_qwen3_32b/doc/optimized_decoder/results/final/capacity \
pytest -q models/autoports/qwen_qwen3_32b/tests/test_context_capacity.py::test_batch32_prefill_capacity_probe -s --disable-warnings

TT_VISIBLE_DEVICES=2,3 QWEN3_32B_CONTEXT_DECODER=optimized \
QWEN3_32B_CONTEXT_PROBE_LEN=16384 QWEN3_32B_CONTEXT_EXPECT_OOM=1 \
QWEN3_32B_CONTEXT_RESULTS_DIR=models/autoports/qwen_qwen3_32b/doc/optimized_decoder/results/final/capacity \
pytest -q models/autoports/qwen_qwen3_32b/tests/test_context_capacity.py::test_batch32_prefill_capacity_probe -s --disable-warnings
```

Results: 8192 passes with output `[1,32,8192,5120]`. The expected 16384 run
fails a 10,737,418,240-byte DRAM allocation (29,524,352-byte largest free
block). Both tests pass as contract probes and write source-hashed JSON.

### Tracy and `tt-perf-report`

```bash
TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_RUN_PROFILE=1 \
QWEN3_32B_RESULTS_DIR=models/autoports/qwen_qwen3_32b/doc/optimized_decoder/results/final \
python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen3_32b/doc/optimized_decoder/tracy_final_all_bfp4 \
  -m pytest -q models/autoports/qwen_qwen3_32b/tests/test_optimized_decoder.py::test_profile_selected_decoder -s --disable-warnings

tt-perf-report models/autoports/qwen_qwen3_32b/doc/optimized_decoder/tracy_final_all_bfp4/reports/2026_07_16_16_56_12/ops_perf_results_2026_07_16_16_56_12.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --no-color --no-host-ops --no-summary --raw-op-codes
```

The same report command with `PERF_PREFILL`/`PERF_PREFILL_END` produced prefill
evidence. Tracy and Watcher were kept separate. `perf_report.md` reconciles
device rows, wall timing, and roofline.

## Optimize final checklist

- [x] Functional semantics/PCC bar, paged KV, repeat, and advancing traced replay
  pass on the implementation-owned optimized path.
- [x] Runtime audit has no measured host fallback or unnecessary
  torch/from_torch/to_torch, tilize/untilize, copy, or reshard.
- [x] Decode uses a coherent L1 width-sharded residual/norm/gate/up chain with
  an evidenced 32-core down boundary; prefill uses DRAM-interleaved activations
  and explicit large 2-D configs.
- [x] Operation-topology audit and `$graph-rewrite` work preceded knob tuning;
  packed QKV, composite attention, fused SwiGLU, bounded trace state, and direct
  output materialization are PCC-verified.
- [x] `$shard-advise` ran this pass; report/IR are saved; every layout/config was
  seeded in a legal executable family and applied or rejected with evidence.
- [x] Best-candidate comparison includes functional, earlier correct optimized,
  advisor, precision/fidelity, geometry, program, layout, SDPA, and packed-MLP
  candidates. Final default reproduces the winner.
- [x] Dominant runtime rows verify BFP4/LoFi QKV/O/gate/up/down and BFP8 KV cache.
- [x] Attention precision was swept separately, including real BFP4 attention;
  MLP gate/up/down BFP4 and LoFi/HiFi2 were swept.
- [x] Important ops have explicit memory/program/compute-kernel configs; core
  grids, role-specific block widths, SDPA grid/exp, and output layouts were swept.
- [x] DRAM-sharded decode matmuls are selected. Packed gate/up was adapted and
  measured but loses after split/layout overhead.
- [x] No collective/CCL topology, fused CCL-matmul, persistent CCL, MoE,
  LM-head, or sampling item applies to this single-device dense-layer stage.
- [x] Reduced precision uses real weights and prompt-derived HF activations;
  synthetic stress is a guardrail, not the sole selector.
- [x] Performance is reconciled across 0.540 ms traffic-only roofline, 1.202 ms
  profiled device time, 1.217 ms 200-replay wall time, and 1.243 ms profiled-run
  wall time.
- [x] Public batch-32 capability is preserved and tested. Batch 1 is not part of
  the compiler-derived functional contract, so a separate batch-1 latency claim
  is not applicable here.
- [x] Non-aligned length 31, repeated decode, Watcher, 8K capacity, and hard 16K
  OOM evidence are final-source clean.

## Device health and commits

`tt-smi -s` before and after final validation reported both p300c devices with
`dram_status=true`, 800 MHz AICLK, advancing equal heartbeats, and final ASIC
temperatures 33.4/35.0 C. Local commit SHAs are appended after the independent
stage review. No push is performed.
