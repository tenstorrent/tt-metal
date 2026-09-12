# Optimized decoder work log

Stage 3; Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
Starting commit ad43d1388fd, clean worktree. Scope: optimized_decoder.py, tests and docs.
Status: complete; runtime gates passed and independent stage review returned clean-pass.

## Environment

Installed tt-autodebug inventory: codex-home/config.toml has enabled=true.
Package environment resolved by scripts/environment.py through the experiment wrapper.
Existing python_env and compiled runtime reused. Actual device nodes 0..3 exist;
serialized 1x1 mesh open/close passed, Blackhole grid 11x10, eight DRAM banks,
firmware 19.8.0, KMD 2.8.0. The supplied generic AGENTS environment description
(no accelerator) does not match this measured host. No toolchain installation.
`timeout 60 python_env/bin/tt-smi -ls --local` failed because that executable is absent;
TTNN device open/close itself succeeded. No reset or recovery necessary.

## Initial topology audit (before tuning)

Source: fused_decoder.py and fused_decoder/tracy_final/fused_l{0,3}/{prefill,decode}_perf_report.txt.
Best accepted baseline: corrected_linear_benchmark.json (3.07446 ms prefill,
2.49408 ms traced decode), corrected_full_benchmark.json (2.75436 / 2.26043 ms).
All projection weights BF16/HiFi4/FP32 accumulation, BF16 activations/KV,
FP32 recurrence, row-major BF16 convolution history, logical B1 with tile padding.
262144 context; B32 at257; no capacity reduction permitted.

| Current operations | Material costs / movement | Candidate | Constraints / planned action |
|---|---|---|---|
| Input norm, attention residual, post norm, final residual | Two single-core DRAM norms: 210 us linear /172 us full decode | L1 width-sharded residual/norm through MLP | Exact FP32 norm policy for linear; preserve B>1 row semantics; measure coherent working grids |
| Packed linear QKV/Z/B/A or full Q/K/V/gate projection | 445 /383 us decode, interleaved BF16 weights | Group-specific BFP8/BFP4, LoFi/HiFi2; DRAM sharding/readers | Preserve packed ordering/padding; compare legal tuned separate family |
| Linear projection slice/untilize, native conv+SiLU, delta prep/scan, gated norm | Native row-major conv boundary; FP32 recurrent state | Retain composites, reduce boundary movement if legal | Prior adapted matmul-untilize failed PCC; recheck current compatible projection configs |
| Full native heads, Q/K norm, RoPE, fused cache update, SDPA | Head-specific L1 reshards; SDPA24 us at128 | Reduced KV, explicit SDPA config, lower movement heads | Logical batch separate from padded rows; cache fill casts, BF16 decode updates |
| Attention output projection | 163 us decode | BFP4/BFP8 and DRAM sharding | Attention precision independent from MLP |
| Separate gate/up (same normalized input), gate SiLU epilogue, multiply, down | ~1380 us, 56% linear /61% full decode | BFP4/LoFi cross geometry; packed vs best fused separate | Count split/movement/elementwise; real weights and real activations decide precision |
| Chunked prefill (128) | Repeated weight traffic, short-M projections | Larger chunks, explicit 2D configs/minimal matmul | Arbitrary tails, continuation, largest context remain valid |

No MoE, collectives, LM head or sampling in this dense single-device decoder.
Those checklist entries are inapplicable to this stage's mathematical path.

## Experiments

Exact invocations appended to commands.log by run_optimization_experiment.sh;
per-run source hashes, logs and JSON use the experiment name.

### Initial precision and topology screen

Real checkpoint weights with synthetic input: BFP8/HiFi2 passes full-kind
PCC0.998679 prefill/0.997962 decode and traces at1.527ms. All-MLP BFP4/LoFi
fails the synthetic prefill diagnostic at0.993451; it is NOT a policy veto.
`record_decoder_activations.py` records tokenized prompt embeddings and CPU HF
preceding-layer outputs for target layers0/3 in the external tensor cache.
No TT full-model stage or generator is constructed. The real-activation BFP4
MLP test passes at0.999968 prefill/0.999981 decode,1.363ms. All-projection
BFP4/LoFi passes both meaningful layer kinds. The real-weight/real-activation
same-harness fused controls are `real_fused_l{0,3}.json`.

DRAM sharding plus sharded norms cuts full decode to0.9445ms; a first residual
carry attempt is0.9665ms and thus not accepted solely on topology. Two readers
with block4 lowers it to0.7886ms. Packed gate/up at this geometry is0.7450ms
versus0.7886ms tuned separate; the per-role sweep will revisit this comparison.
Linear packed two-reader decode is0.9074ms. These are candidate medians,
not final default numbers. First block16/BFP8 and block8/BFP4 attempts exceed
L1 circular-buffer budgets; exact allocation errors are retained in logs.

### AutoFix: three-reader output storage

Failing contract: all legal per-bank reader geometries must have enough output
storage for padded reader work. Three-reader down projection K17408,N5120
schedules24 workers x7 tiles=168 tiles, while the old ceil(160/8)=20 per-core
storage reserves160 tiles. `geometry_r3_b4.log` raises 'Worker 7-2 has no storage
area assigned'. Fresh AutoDebug inspected the native assignment arithmetic.
One-variable experiment `readers3_storage_fixed` sizes `per_core_N` from the
DRAM shard's padded width x eight banks. Exact original reader/block policy
then passes prefill0.99990386/decode0.99990386/changed-token0.99988580 and30
bitwise repeated traced replays. No native code change. Whole-layer0.901ms is
slower than two-reader0.789ms at this geometry, so legality does not imply
selection. Source-only report: `AUTODEBUG_readers.md`.

A residual10 norm trial failed with subblock16 exceeding FP32 destination
capacity4. The adapted configuration retains block16 and uses subblock4;
this changes only internal norm blocking, not residual width or precision.

### Prefill and cache screen

With explicit 8x8 2D matmuls, logical511 prefill in128-token chunks costs
12.18ms linear/11.22ms full. A single512 bucket costs4.55/3.67ms, with PCC
0.999789/0.999888. Chunk1024 produces the same physical512 work at this length;
it is not evidence about a genuinely longer input. Larger-context/tail tests
and direct comparisons against untuned large-M kernels remain required.
BFP8 full KV fill is explicitly cast while decode update remains BF16.
`attention_cache_{10,11,12}.json` passes reduced cache and explicit SDPA config
candidates; final same-context rows will decide the retained config.

### Precision-locked geometry, packing and movement closure

`projection_sweep_l3.json` sweeps all five dominant roles plus packed gate/up;
`projection_sweep_l0.json` independently checks the wider linear-attention pack.
`projection_results.csv` retains every legal attempted block/core/reader combination
and exact validation/L1 failures. Quantized-weight PCC here checks implementation,
not model precision: whole-decoder real-HF PCC is the precision gate.
The three-reader 80-storage-core/block2 attention/gate/up family wins over the
40-core/block4 family and fewer-core larger-block families at the same BFP4/LoFi.
Output prefers48 cores/block4. Down initially preferred two readers/8 cores/block17
in isolation; `movement_down3_l{0,3}` shows three readers/32 cores/block17 wins
whole-decoder latency by avoiding surrounding movement. This is the relevant
selection metric. No assumption that more cores, packing or lower op count wins.

`hifi2_*`, `bfp8_lofi_*`, and `activation8_*` hold geometry and other policy fixed.
HiFi2 passes but loses for every dominant group; BFP8 attention/gate and BF8
activations also lose. BF16 activations plus BFP4/LoFi weights are retained.
`closure_split_attention_l{0,3}` is legal and passes but loses to packed attention
(1.031/0.837 ms versus0.833/0.671 ms). Separate gate/up with SiLU fused into the
multiply beats packed gate/up (0.833/0.671 versus0.839/0.677 ms) under matching
precision, geometry and rectangular residual layout. Native minimal SwiGLU is
legal after tile-interleaving weights, but1.456/1.299 ms loses. All timings are
layer0/layer3 and are intermediate candidates, not final default claims.

### AutoFix: logical batch versus physical rows

`batch3_probe.log` exposed a height32 shard receiving public [3,1,H], whose
physical row count is96. `AUTODEBUG_batch.md` gives the source diagnosis.
Internal norm/MLP/residual rows now stay packed [1,1,B,H]; public boundaries
interleave before unpacking B>1. The original batch3/layer0/257 continuation
passes (`batch3_fixed_l0.json`): prefill0.999783802, continuation0.999783743,
trace0.999845982, changed-input0.999804378, four sequential trace steps all>.99978.
B1 controls preserve correctness and latency. Independent follow-through is in
`AUTOFIX_batch.md`. The optional carry-output branch is now gated by the residual
branch so it cannot return packed attention to a generic public-rank residual.
Final larger-batch and watcher evidence remains a separate required gate.

### Long-prefill search

`prefill_matrix.json`, `closure_matrix.json`, and their named logs compare actual
4097-token real-HF inputs. Generic chunks1024 improve on512. Explicit2D1024
initially exceeded L1; reducing output-block height to1 fixes legality but loses
performance at1024/2048/4096. The candidate is not rejected at its first API error.
Native minimal matmul is slower at128 but faster at1024/2048/4096;2048 is the best
initial chunk size, with full/linear prefills35.32/25.61 ms before SDPA tuning.
M8/K16/N16 and combined configs are measured separately. Short-prefill sharded
norms and2D grids are also measured instead of assuming large-M winners apply.

### Profiler-driven follow-up

`tracy/candidate_fixed_l3` has actual BF16 x BFP4 LoFi rows for all five projections,
40 device ops, zero host ops,638 us kernel sum and38 us gaps. The packed output
projection unnecessarily interleaved then reshared; `movement_carryboth_*` tests
removal plus carrying B1 input through its residual boundary. BF16-to-BF16 cache
fill casts seen in prefill were removed by an explicit dtype equality check.
The report's reader matmuls display8 cores although native configs use24 workers;
its utilization percentages can therefore exceed100%. Program configs and raw
attributes, not that core-count estimate, determine the actual reader geometry.

### Final policy, then capability repairs

The selected policy is installed as `DEFAULT_POLICY`, with full experimental
policy replacement optional. `test_optimized_decoder.py` patches both older
constructors to fail; all shapes, real per-user PCC, continuation, changed input,
page permutation, 30 restored bitwise repeats and four consuming trace steps
exercise the optimized default. Native minimal matmul uses M4/K8/N16, chunk2048
for long prefill; output/down also use it from128 tokens. Short prefill totaling
at most32 rows uses DRAM-sharded projections. A first128-row DRAM attempt hit
the native M==1 tile constraint; adapted32-row matmul chunking passed but lost
(3.86/3.50 ms full/linear versus1.80/2.08 ms). It is not selected at128.
At31 tokens it wins (1.365/1.391 versus1.652/1.967 ms). Five correct traced
prefill repetitions also quantify the suggested dispatch-gap reduction without
making the decoder own a caller's request trace.

The first full suite is retained as `stress_initial_failure.{log,xml}`: three
parameter groups passed, five failed. Four failures share a continuation
alignment bug in new128-token SDPA blocks; the fifth is batch32 linear decode
L1 pressure. CQ-id errors in pytest rendering occur after device close while
formatting dead tensors; the causal assertions are preserved before them.
No hang, device reset, weakened PCC bar or capacity reduction occurred.

Continuation now reduces Q and K blocks independently until they divide the
absolute prefix. A separate source audit found rounded SDPA reads can overrun a
minimal page table even when masked numerical output passes. Prefill therefore
also requires rounded logical end to fit mapped capacity; decode selects K that
divides mapped capacity, so every valid device position's rounded read fits.
Caller allocation remains32-token pages, with no128-token allocation requirement.
This reclassifies intermediate64/128-block latency candidates whose rounded
reads exceeded their mapped table as unsafe candidates, not correct baselines.
`AUTODEBUG_continuation.md` records the native reader/factory reasoning.

For batch32, a public packed attention tensor [32,1,16512] physically expands to
[32,32,16512],33,816,576 BF16 bytes. Its L1 slices overlap the GDN prep CB budget
(ends1,115,136; live allocation starts618,496). One-knob public-DRAM control
`batch32_dram_boundary` passes. The narrowed `batch32_attention_boundary` moves
only the attention projection to DRAM before public repacking and also passes:
prefill0.999783529, continuation0.999783395, traced decode0.999847829, four
sequential steps>.99978. B1 retains L1; larger batches from8 use the bounded
attention boundary. New B8/B16 tests cover that switch; final B32 is rerun.

### Validation and bounded final profiling

`bash models/autoports/qwen_qwen3_8_27b/tests/validate_optimized_decoder.sh`
reran after both capability fixes:12 pytest groups /68 cases passed in193.90s,
including B1/B2/B3/B8/B16/B32, nonaligned long tails, per-user PCC and output shapes.
`stress_summary.json` records minimum prefill0.9995203 and decode0.9993832.
Watcher B1/B32 both kinds passed with disabled features=None; `watcher_long_tails`
also covers2049/4097/31 and continuation on the minimal mapped tables.

The initial final_l0 Tracy invocation unnecessarily included five traced prefill
replays and30 timed decode replays in addition to correctness tests. Its device
run passed, but postprocessing failed: Op422915 missing from the C++ device CSV.
It is retained as `oversized_profile_l0.{json,log,source.sha256}` and archived raw
profiles. This is a profiler collection/postprocessing failure, not a device hang
or a model pass. Removing redundant profiled benchmarking yielded complete
prefill/decode CSV for both kinds. The profile harness now times only its actual
signposted warmed windows; long stress/median benchmarks remain separate.
No reset was performed; every device process closed normally.

Final profile advice: all five decode projections are reported optimized.
Prefill's L1-input advice was executed at128/2048 on both kinds (`advice_matrix`).
DRAM is retained: L1 worsens both linear cases, full2048, and traced full128.
MinimalMatmul's 'no program_config' advice is a parser field-name mismatch: the
explicit `MinimalMatmulConfig` uses M4/K8/N16/1x4 and was swept. Fidelity advice
was already tried per role with actual HF PCC. Decode's remaining7us first-norm
gap is present inside an actual captured trace; it is not an untraced Python op.
The same-source profiles show44/38 device ops and zero host ops (linear/full).

Normal benchmark artifacts are `final_benchmark_l{0,3}` and
`baseline_benchmark_l{0,3}`, same real recorded128-token inputs and wait strategy.
Baseline→default traced decode is2.49543→0.82180 ms linear and2.26245→0.66357 ms
full; warmed prefill3.08808→2.16072 and2.70662→1.79324 ms. Source hashes accompany
both. `performance_summary.json` reconciles separately instrumented same-run
roofline, kernel sum, gaps and host time; it does not mix profiled device time
with unprofiled host medians. A final blocking-trace API comparison will check
whether the small host wait remainder can be reduced.

Capacity commands, serialized after profiles/watcher:
`PYTHONPATH=. timeout -k 10 7200 python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/run_optimized_context.py --snapshot /home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --layer <0|3> --activations /home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations_long --output models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/context_l<0|3>.json`
with stdout/stderr in the matching `.log`. These compare full outputs against
the frozen fused implementation and trace the last legal decode position;
no fresh262144-token HF oracle is claimed for repeated recorded inputs.

Capacity closure: both context runners exited 0. Complete 262144-token output PCC
is 0.9997950792 linear / 0.9996872416 full; last-valid-position traced decode
is 0.9998990297 / 0.9999096990, with bitwise repeats. Context contract is now
validated with no reduction. Watcher long-tail summary is included. Independent
review additionally checked all 68 default policies and source hashes, the native
untilize contract, full-capacity rows, and actual profiler attributes.

### Close candidate and trace-wait follow-through

`comparison_matrix.json` was executed with the same long recorded inputs and
30 traced repetitions per case. ABBA full-attention decode medians were final
0.664148/0.664614 ms versus prior best 0.663687/0.663156 ms. Final prefill was
1.715495/1.751744 ms versus 2.116080/2.116130 ms. These small decode losses are
recorded explicitly; the stricter user winning gate triggered another SDPA grid
search at final BFP4/LoFi/BFP8 precision (`sdpagrid_matrix.json`).

Blocking execute_trace versus nonblocking execute_trace plus synchronize did
not remove the gap: linear 0.821093 versus 0.821803 ms; full 0.664855 versus
0.663571 ms. Fused blocking controls remain 2.494894/2.263953 ms. This rejects
the host API change; reported normal medians keep the original wait strategy.
The BF16-cache blocking control (0.663346 ms) identifies a roughly 1.5 us short
cache-precision cost, not a large orchestration error. The native 7 us trace
first-norm gap remains after carrying input/residual shards and this wait trial.

Final benchmark harness anomaly: selected_matrix empty policies merged the
sweep script historical BASE and selected old block4/one-reader geometry.
The saved effective policies exposed it; those rows are not final-default
evidence. Added explicit default_policy matrix selection, which passes {}
to the constructor. verified_matrix reruns default/control/default, while
pytest/profile already invoke defaults directly. No decoder rollback.

### SDPA grid selection

Final precision-locked short decode grids (ms):11x10 ~0.6644;10x8 0.663066;
8x8 0.663281;8x4 0.660826;4x4 0.661753;8x2 0.660732;8x1 0.661699;
4x1 0.665145;2x1 0.686245;1x1 0.728220. All real-weight checks passed.
The8x2 grid loses for longer contexts:2048 0.723451 versus wide0.690667,
4097 0.791982 versus wide0.705100. It is selected only for B1 and fewer
than16 mapped pages; larger batches/contexts retain the wide grid. This
reduces SDPA dispatch/reduction overhead without reducing capability or
changing cache allocation. Prefill and linear paths are unchanged. Full
capacity probes therefore still exercise identical code/config/storage.


### Final verified default

`verified_matrix.json` uses the repaired explicit default-policy selection.
Linear prefill/trace-prefill/decode: 2.260554/1.911012/0.822062 ms.
Full: 1.783167/1.671455/0.661752 ms. Full default/control/default decode is
0.661752/0.663265/0.661923 ms. Effective policies match `final_policy.json`.
Final default PCC is 0.999771059/0.999833167 linear and
0.999900520/0.999903858 full (prefill/decode), all above 0.995.

The final 12-group/68-case pytest rerun passed in107.70s (`selected_stress.xml`)
with unchanged minima and guards. Watcher selected31/129/257/511 passed with
all features enabled and normal detach. Final full profile `selected_l3`
contains38 device/zero host ops, 0.625718 ms kernels,0.036491 ms gaps and
0.685662 ms same-run host time. SDPA uses16 cores at23.140 us; all five
projections remain actual BF16 x BFP4 LoFi. Raw profile and inspector files
are archived without deletion, with exact mapping in `tracy/raw_archive.json`.

Host checks: Black --check --target-version py312 on the eight new Python
implementation/test files; cached isort --check-only; cached autoflake --check
--remove-all-unused-imports --ignore-init-module-imports; py_compile; bash -n
on all three shell runners; staged git diff --check for authored Python, shell,
Markdown and JSON. Generated reports use native CSV line endings; immutable source snapshots
retain exact bytes. The first commit hooks normalized whitespace/EOF in 58
generated text artifacts. Their original indexed bytes were preserved under
pre_commit_originals in the external archive, with SHA256 and paths in
tracy/raw_archive.json. One 621 KB profiler log exceeded the repository limit
and is committed losslessly as candidate_fixed_l3.log.gz. All code hooks passed. Python/tests/docs only, so no
C++/CMake build was required or run. No dependencies installed or hardware reset.

Independent fresh xhigh stage-review returned **clean-pass**, with no required
work remaining. See `stage_review.md`, including the resolved harness finding,
source SHA256, recomputed profile/performance/PCC evidence and residual limits.

## Local checkpoint provenance

Repository: tt-metal

Branch: `mvasiljevic/qwen38-full-bringup`

Stage implementation, tests, evidence and clean review commit:
`3db7870d0653b7ba7ceab4296720dd3d54769f96`

All applicable repository commit hooks passed on retry, including formatting,
YAML, merge-conflict, large-file and repository-specific checks. This provenance
entry is recorded in a following documentation-only local commit. No commits
were pushed; no other repository was changed.
