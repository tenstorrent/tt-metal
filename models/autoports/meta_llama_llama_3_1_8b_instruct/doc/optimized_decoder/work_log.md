# Optimized decoder work log

Target: `meta-llama/Llama-3.1-8B-Instruct`, representative dense layer 16,
single Blackhole P300 device selected from the repo's already-restricted
`TT_VISIBLE_DEVICES=2,3` endpoint pair.

## 2026-07-16: startup and device contract

- Branch at stage start: `mvasiljevic/model/meta-llama-llama-3.1-8b-instruct`.
- Starting checkpoint: `46b4ebfb352`.
- Worktree was clean before stage-owned files were created.
- `timeout 60 tt-smi -ls --local`: two P300c UMD endpoints visible.
- Minimal `MeshShape(1, 1)` open/close: pass; selected logical device `[1]`,
  architecture `BLACKHOLE`, compute/storage grid `11x10`, DRAM grid `8x1`.
- Hardware commands are serialized. Watcher and profiler runs will be separate.

## Operation-topology audit and graph rewrite

Audit source: the completed `functional_decoder.py`, the TTNN common attention,
MLP, and RMSNorm modules, and `tech_reports/LLMs/llms.md` section 4. A current
perf report is added after the first traced baseline; this table records the
code-derived topology that preceded knob tuning, as required.

| Phase / subgraph | Functional operation and movement | Candidate | Dtype / contract constraint | Initial action |
| --- | --- | --- | --- | --- |
| input norm | DRAM-interleaved RMSNorm with op defaults | width-sharded L1 residual + sharded RMSNorm program | norm weights/activation BF16 | implemented for decode; prefill stays DRAM |
| QKV projection | one packed BF16 matmul, DRAM input/output | retain packed QKV; DRAM-sharded BFP8/BFP4 weights; geometry sweep | packed order must remain Q,K,V; head op requires BF16 output | implemented packed DRAM-sharded path |
| decode head creation | permute, reshape, dedicated `nlp_create_qkv_heads_decode` | replace singleton-dimension permute with metadata reshape | public `[1,batch,1,H]`; internal `[1,1,batch,H]` | implemented; one dispatch removed |
| RoPE | dedicated rotary embedding for Q and K | retain dedicated op | Q/K update tensors BF16 | retained |
| cache update | position slice + repeat; two `paged_update_cache` calls | retain update contract; test BFP8 cache separately | decode update tensors remain BF16; logical users must not become tile-padding users | retained |
| decode attention | query L1→DRAM, default decode SDPA | explicit SDPA program/compute config | same cache shape, current-position tensor, batch, and context | explicit 8x8 config implemented |
| head concatenation | DRAM→height-sharded, dedicated concat, L1→DRAM, slice, permute | retain dedicated concat; keep output L1 width-sharded and metadata-reshape final output | batch-13 slice must preserve logical rows | implemented; DRAM round trip and second singleton permute removed |
| O projection + residual | BF16 DRAM matmul, DRAM add | DRAM-sharded reduced-weight matmul, L1-sharded add | output grid must rejoin residual grid before add | implemented |
| post-attention norm | DRAM-interleaved RMSNorm | same sharded residual/norm chain | BF16 norm | implemented |
| MLP gate/up | two same-input BF16 DRAM matmuls, separate SiLU, multiply | compare packed gate/up against tuned split; fold SiLU into binary op | real-weight evidence decides BFP4; packed result must count slicing/layout costs | split+BFP4/LoFi implemented first; packed knob implemented for A/B |
| MLP down + residual | BF16 DRAM matmul and DRAM add | BFP8/BFP4 DRAM-sharded matmul, phase-specific input shard, L1 residual add | intermediate 14336 and hidden 4096 must divide selected grids | BFP8/HiFi2 path implemented first |
| prefill projections | BF16 DRAM matmuls with defaults | DRAM-interleaved activations, DRAM-sharded reduced weights, explicit 2-D configs | valid non-aligned logical sequence lengths must remain accepted | implemented; seq=7 real op contract exercised |
| prefill attention | dedicated SDPA with defaults | retain composite and use explicit SDPA config | causal mask and true logical sequence slicing | implemented |
| host/device boundaries | setup-only `from_torch`; test-only `to_torch` | no runtime host fallback | forwards must contain no torch/from_torch/to_torch | preserved |
| collectives | none after functional stage collapsed 1x4 TP to dense 1x1 math | not applicable in this single-device stage | do not start multichip work | no action |

Existing dedicated/fused ops were confirmed in the TTNN op library and model
usage: packed QKV head creation, rotary embedding, SDPA/FlashAttention,
decode-head concatenation, RMSNorm, and SwiGLU's binary activation argument.
No primitive attention or normalization subgraph remains.

## First rewritten correctness candidate

Policy: BF16 activations/norms; BFP8 QKV/O/down weights; BFP4 gate/up weights;
HiFi2 attention/down; LoFi gate/up; BF16 cache; explicit prefill and decode
program configs; 32-core residual/projection grids.

Focused command shape: synthetic batch 1, non-aligned prefill `seq=7`, then
decode at position 7. TTNN contracts all executed successfully. Output PCC was
0.990479 prefill and 0.989683 decode. The latter is retained as a synthetic
stress diagnostic, not a precision veto.

Real layer-16 control with the same policy, batch 1, prefill 18 then decode 18:

| Gate | PCC |
| --- | ---: |
| prefill output | 0.9999213201 |
| decode output | 0.9999025958 |
| decode K append | 0.9999058373 |
| decode V append | 0.9999090097 |

This passes the functional bar (`0.99`) and proves that the reduced policy is
valid on target weights. Per OPT-012, the synthetic-only delta does not select
a slower higher-precision fallback. Geometry, cache dtype, fidelity, packed
MLP, advisor layout, trace, and performance remain to be measured.

## Mandatory shard-advisor pass (OPT-015)

The advisor was run once this pass after the dense decode graph rewrite. Source
was `/home/mvasiljevic/tt-mlir` at `3f8b9c0a2587` on branch
`ttnn-jit-shard-advisor`; its tt-metal submodule was `13adda80`. The required
separate-shell setup and capture were:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
cd /home/mvasiljevic/tt-mlir
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
export LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib
export PYTHONPATH=${PYTHONPATH}:/home/mvasiljevic/tt-metal:/home/mvasiljevic/tt-metal/python_env/lib/python3.12/site-packages
export TT_VISIBLE_DEVICES=0,1
cd /home/mvasiljevic/tt-metal
ttnn-advise capture models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise/advise_llama.py:decode \
  --out models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise
```

The advisor's bundled UMD could discover the system but could not safely map
host system memory on this firmware (`unexpected NOC address`). A bounded
`tt-smi list`, reset, list cycle confirmed healthy hardware but did not change
that version-bound failure. `advise_llama.py` therefore preserves a transparent
capture-only adapter: tensors keep the real metadata, a synthetic Blackhole
8x8 device supplies graph capture, `SYSTEM_DESC_PATH` supplies the captured
system, and rotary/decode-SDPA have graph-only handlers. Runtime measurements
below use normal TTNN and real devices. No tt-mlir source was modified.

`report.json` records 24 operations, 21 final choices, and a completed spill
pass with one spill. `final_ir.mlir` is nonempty and contains the final configs.
The advisor cannot legally model the decode-head concat shard after its custom
SDPA handler and reverted that single choice; runtime explicitly converts
SDPA output to the concat operator's required height shard.

| Advisor recommendation | First candidate | Decision and evidence |
| --- | --- | --- |
| QKV 1-D: 11x9, `in0_block_w=2`, `per_core_N=2`, subblock 2, 96 output shards | executed | applied |
| O 1-D: 11x6, `in0_block_w=8`, `per_core_N=2`, subblock 2, 64 output shards | executed | applied |
| gate/up 1-D: 11x9, `in0_block_w=2`, `per_core_N=5`, subblock 5, 90 output shards | executed | applied; subblock 5 improved 100-replay decode from 0.717096 to 0.713226 ms |
| down 1-D: 11x6, `in0_block_w=8`, `per_core_N=2`, subblock 2, 64 output shards | executed | applied |
| exact norm/residual chain: input norm block-11, QKV/residual width-64, post norm width-22, gated input width-56, final L1-interleaved/DRAM output | `advisor_exact_chain` executed end-to-end on real weights | rejected as default: PCC 0.999805/0.999855, prefill 3.334983 ms, traced BFP8-cache decode 0.749051 ms versus chosen 3.359613/0.713226 ms |
| concat input shard | advisor reverted choice | runtime uses the dedicated concat contract; report records the compiler-side revert and real path is profile-verified |

Required preserved artifacts are `shard_advise/report.json`,
`shard_advise/final_ir.mlir`, `shard_advise/report.txt`, and the reproducible
capture adapter `shard_advise/advise_llama.py`.

## Precision, fidelity, topology, and configuration search

Every timing below is warmed and uses the same real layer-16 harness unless a
failure is explicitly named. Main selection numbers use ten prefill iterations
and 100 trace replays; exploratory rows used 20 trace replays. All passing
reduced-precision cumulative candidates were checked on real weights.

### Cumulative candidates

| Candidate | Prefill ms | Traced decode ms | Real prefill/decode PCC | Decision |
| --- | ---: | ---: | ---: | --- |
| first DRAM-sharded rewrite | 4.009 | 0.895 | 0.999921/0.999903 at batch 1 | baseline optimized candidate |
| advisor 1-D, original precision | 4.622 | 0.827 | pass | advisor wins decode, loses prefill; retained as search base |
| BFP8/LoFi attention | 4.177 | 0.834 | pass | valid |
| BFP4/LoFi attention | 4.177 | 0.828 | pass | selected over BFP8 |
| BFP4/LoFi down on DRAM candidate | 3.920 | 0.803 | pass | selected |
| advisor + BFP4 down | 3.664 | 0.770 | 0.999887/0.999889 | selected cumulatively |
| advisor + all BFP4/LoFi | 3.564 | 0.755 | 0.999805/0.999863 | selected cumulatively |
| final, BF16 cache, 10/100 | 3.318 | 0.750 | same chosen projection policy | final same-cache result |
| final, BFP8 cache, 10/100 | 3.346 | 0.713 | decode 0.999862 | final fastest traced decode; literal final file reproduced 3.345963/0.712697 |
| exact advisor residual chain, BFP8, 10/100 | 3.335 | 0.749 | 0.999805/0.999855 | rejected; slower decode |

The strongest correct pre-stage traced baseline is the functional BF16 path at
37.656058 ms prefill and 36.948320 ms decode. The final same-cache path is
3.317967/0.750455 ms; optional BFP8 cache reaches 3.359613/0.713226 ms. Batch-1
BF16 measurements are 1.460630/1.368892 ms functional and
1.262574/0.583138 ms optimized. An attempted functional+BFP8 comparison failed
at the documented functional `fill_cache` dtype assertion; the fair baseline
table uses BF16 on both sides.

### Role-specific sweeps and rejected candidates

| Opportunity | Adapted attempts and evidence | Result |
| --- | --- | --- |
| attention fidelity | BFP4 HiFi2 4.262083/0.892559; BFP4 LoFi 4.176661/0.828447 | LoFi selected |
| down fidelity | BFP4 HiFi2 4.293043/0.892559; BFP4 LoFi 3.919681/0.803130 | LoFi selected |
| gate/up fidelity | final HiFi2 4.554736/0.757960; LoFi about 3.564/0.755 before final repeats | LoFi selected; material prefill win |
| DRAM-sharded all-BFP8/HiFi2 | grid 8 first exceeded L1 (CB1 1,676,032 > 1,572,864); adapted grid 10 passed at 4.826649/1.079736 | rejected after successful retry |
| all-role DRAM core geometry | 32 cores 0.803130 ms; 16 cores 0.813913; 8 cores hit an L1 clash | 32 selected |
| down-only geometry | 32/16/8 cores: 0.803130/0.803235/0.802709 ms | differences are noise; 32 retained for clean division/grid use |
| gate/up geometry | 32 cores 0.803130; 16 0.822046; 8 first exceeded L1; adapted `in0_block_w=8` passed at 0.869705 | 32 selected |
| advisor gate/up output subblock | width 1: 3.345287/0.717096; advisor width 5: 3.359613/0.713226 (BFP8, 10/100) | width 5 selected for decode |
| exact advisor residual chain | full coherent chain implemented, real PCC passed, 3.334983/0.749051 | rejected as slower than mixed 32-way residual chain |
| packed gate/up | grid 8 exceeded L1 (CB2,192,128 > 1,572,864); adapted prefill grid 11 still exceeded L1 (CB1,669,888 > 1,572,864) | rejected with two legal grid/layout attempts; tuned separate projections remain valid |
| SDPA | default 3.804547/0.753551; explicit grid 8 about 3.564/0.755; grid 10 3.777365/0.751803; grid 11 3.778317/0.752733 | explicit 8x8 selected for prefill; decode deltas are noise |
| prefill matmul grid | 8x10 about 3.564; 10x10 3.969520; 11x10 4.082396 | 8x10 selected |

Packed QKV was already the legal fastest topology and remains packed. Separate
gate/up is retained only after the packed candidates were adapted across grids
and still exceeded L1. The `ttnn.mul` activation argument folds SiLU into the
binary step. Singleton permutes and the decode concat DRAM round trip were
removed. There are no collectives, CCL buffers, experts, LM head, or sampling
in this single-device decoder-layer stage, so the related optimize items are
not applicable.

## Final correctness, stress, trace, and watcher evidence

Real batch-32 sequence-18 PCC is 0.9998048729 prefill and 0.9998617278 decode;
key and value append PCC are 0.9928925315 and 0.9933177982. The BFP8-cache
decode PCC is 0.9998621340. Batch 13, logical sequence 7 passed three repeated
runs with exact deterministic outputs and PCC 0.9997397117 prefill and
0.9998280717 decode. Five synthetic trace replays were bit-exact.

The static runtime audit checks that optimized forwards and MLP helpers are
owned by `optimized_decoder.py` and contain no functional forward call,
`torch`, `from_torch`, or `to_torch`. The measured runtime has no host fallback,
tilize/untilize pair, or needless public-boundary conversion. Internal reshards
remaining in the profile feed a consumer's required shard or bridge the chosen
mixed chain.

Watcher was run separately from the profiler:

```bash
TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k 'real_weight_prefill_decode_and_cache_contract or non_aligned_sequence_batch_and_repeated_run_determinism'
```

Both tests passed before and after the final candidate switch was added. The
earlier combined log had 1,103 clean lines; the literal-final-code rerun left a
556-line current watcher log with clean detach/stack summaries and zero matches
for error/assert/hang/timeout/illegal/corrupt/NoC-fail patterns. No
watcher/profiler overlap occurred.

## Tracy and `tt-perf-report` conclusions

The final BFP8-cache default was captured with Tracy and processed with advice
enabled. Preserved files are under `tracy/layer16/`. Decode has 35 operations,
0.697 ms device time, 0.034 ms inter-op gap, and aggregate 156 GB/s DRAM
throughput (30.6% of the 512 GB/s Blackhole figure used by TTNN). Dominant rows:

| Row | Device time | Share | Verified policy/config |
| --- | ---: | ---: | --- |
| QKV | 91 us | 12.6% | BF16 x BFP4, LoFi, advisor 11x9/2/2/2 |
| O | 33 us | 4.6% | BF16 x BFP4, LoFi, advisor 11x6/8/2/2 |
| gate | 113 us | 15.6% | BF16 x BFP4, LoFi, advisor 11x9/2/5/5 |
| up | 107 us | 14.8% | BF16 x BFP4, LoFi, advisor 11x9/2/5/5 |
| down | 110 us | 15.2% | BF16 x BFP4, LoFi, advisor 11x6/8/2/2 |
| decode SDPA | 43 us | 6.7% | dedicated composite, explicit 8x8 |

The report marked the projection configs optimized. Its fidelity suggestion was
already tested role by role; HiFi2 lost while LoFi passed the real PCC bar.
Prefill reports 3.027 ms device time plus 0.724 ms profiler-instrumented gaps.
Advice attributes about 0.298 ms to the batch-32 cache-fill loop. The contiguous
`fill_cache` API accepts a scalar `batch_idx`, so the per-user loop is required;
unprofiled warmed wall time is 3.318-3.360 ms and decode trace removes dispatch.

The five BFP4 matrices contain 218,103,808 parameters, or 109,051,904 bytes.
Including about 1.25 MB of position-18 BFP8 cache reads gives an approximate
0.215 ms bandwidth floor at 512 GB/s. The measured 0.697 ms device time is 3.2x
that floor; 0.713 ms wall time reconciles with device time plus the 0.034 ms
profile gap within run-to-run/profile perturbation.

## Optimize completion checklist

- [x] Functional PCC, paged cache, trace replay, runtime fallback, repeated-run,
  watcher, warmed before/after, and advice-backed profile gates pass.
- [x] Decode uses L1 width shards across every material region; the coherent
  exact advisor chain was measured, and the faster mixed 32/64/90-way chain was
  selected with explicit boundary reshards.
- [x] Prefill remains DRAM interleaved with explicit 2-D projection configs.
- [x] Operation topology was audited and `$graph-rewrite` applied before local
  knob tuning: packed QKV/composite SDPA retained; singleton permutes and concat
  round trip removed; SiLU folded into multiply.
- [x] `$shard-advise` ran this pass; required artifacts exist; every emitted
  dense layout/program family was applied or rejected with measured evidence.
- [x] Final default beats the strongest correct traced baseline and reproduces
  the selected 10/100 result. Batch 1 and batch 32 are both measured/covered.
- [x] Profiler rows verify BFP4 projection weights, LoFi projection math, and
  the chosen advisor program configs reached all five dominant matmuls.
- [x] Packed/split projection decisions have evidence: QKV packed; gate/up
  separate after two adapted packed attempts exceeded L1.
- [x] Important memory, program, compute-kernel, SDPA, prefill-grid, per-role
  core-grid, `in0_block_w`, output-subblock, dtype, and fidelity choices were
  swept. BFP4/LoFi was tried for attention, gate/up, and down.
- [x] DRAM-sharded decode matmuls were implemented and swept across geometry;
  the advisor 1-D path with interleaved DRAM weights won cumulative traced
  decode and is therefore the final default permitted by OPT-004/OPT-015.
- [x] Avoidable movement is removed; remaining reshards serve an incompatible
  dedicated-op input or are faster than the exact lower-movement chain.
- [x] Precision experiments used real layer weights/activations; full-model
  top-k frontier selection remains correctly deferred to datatype-sweep.
- [x] Roofline, device time, traced wall time, and remaining gaps reconcile.
- [x] Multi-device/collective/fused-CCL/persistent-CCL, MoE, LM-head, sampling,
  full-model, serving, and qualitative-generation checklist items are not
  applicable to this explicitly single-device decoder-layer stage.

## Stage-review finding and AutoFix remediation

The first independent `$stage-review` returned `more-work-needed`: the original
trace regression proved repeated bit equality but did not compare real replay
outputs to Hugging Face or prove that refreshed trace input was consumed. The
finding was treated as required work.

Per `$tt-enable-tracing` and `$autofix`, a fresh source-only AutoDebug pass was
requested. The nested AutoDebug runner could not read the checkout because its
sandbox lacked `bubblewrap`; the fresh forked inspector recorded that runner
limitation and completed a temporary `AUTODEBUG.md` by direct read-only inspection;
the evidence is preserved in this stage-owned log. Its
highest-confidence hypothesis was that the new regression test itself read the
capture destination before the first `ttnn.execute_trace`. Capture records the
commands and persistent output address; that destination is not valid output
until replay.

The focused experiment changed only that sequencing and retained all stronger
assertions. Result:

| AutoFix check | Result |
| --- | ---: |
| eager real decode vs HF | PCC 0.9998610513 |
| five actual trace replays vs HF | PCC 0.9998610513 each |
| repeated replay determinism | bit exact |
| trace key/value append vs HF | PCC 0.9931565 / 0.9934396 |
| refreshed persistent-input trace vs second HF decode | PCC 0.9998641238; output changed |
| refreshed key/value append vs HF | PCC 0.9930171 / 0.9934953 |

Verdict: hypothesis verified and the apparent trace failure fixed at the test
boundary. Input/cache identity mismatch, precision drift, L1 output-view
lifetime, and cache-ordering hypotheses are refuted by the passing actual
replays, refreshed-input response, and both cache comparisons. The durable test
is `test_real_weight_traced_decode_replay_correctness_and_input_refresh`. It
also passed with `TT_METAL_WATCHER=10`; the resulting 556-line watcher log had
zero error/assert/hang/timeout/illegal/corrupt/NoC-fail pattern matches.

The required fresh rereview after remediation returned `clean-pass`. It
explicitly confirmed that the prior P1 is closed by the five HF-checked actual
replays, bit-exact repeats, original/refreshed K/V append checks, and changed
HF-matching output after refreshing the exact captured input. No material
blockers remained.

## Local commits

- Stage implementation/evidence commit: `8b2656be7cc` (`Add optimized Llama
  3.1 8B decoder`). All repository pre-commit hooks passed.
- This work-log-only follow-up records the stage SHA. Nothing was pushed.
