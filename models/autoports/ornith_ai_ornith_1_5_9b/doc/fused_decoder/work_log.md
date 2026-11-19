# Fused decoder work log

Stage 2, ornith-ai/Ornith-1.5-9B, revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
Starting checkout d55afc09, clean working tree. Enabled plugins verified in Codex config.
Hardware: P300c Blackhole chip 0, serial jobs; topology discovery retained, no TT_VISIBLE_DEVICES override.
No devices were held by another process at startup. No resets authorized or performed.
Environment: python_env, HF_HOME=/home/ttuser/ornith-9b/hf; installed environment.py exports applied.
Functional precision/context/acceptance unchanged: BF16 weights/activations, HiFi4 FP32 accumulation,
FP32 recurrence, 262144 logical context, PCC >= 0.995.

## Topology audit

| Subgraph | Movement / dtype | Candidate |
|---|---|---|
| Input RMSNorm | BF16 DRAM | Already dedicated |
| Linear z/b/a/QKV projections | Same input, separate matmuls; RM/tilize reshapes | Pack projections; native tile reshape |
| Four-tap causal convolution | Slice/typecast/mul/add, FP32 accumulation | Dedicated causal convolution; preserve accumulation |
| Q/K L2 norm | FP32 square/sum/epsilon/rsqrt/mul | Dedicated RMS normalization with equivalent scale/epsilon |
| Gated delta recurrence | FP32 state; dedicated prefill and decode | Already dedicated |
| GDN output norm and SiLU gate | BF16 norm/silu/multiply | Input activation fusion |
| Full Q-gate/K/V projections | Shared LHS; separate projections and reshapes | Packed projections / head split |
| Full Q/K norm then partial RoPE | Primitive slices/neg/concat/multiply/add | Dedicated rotate-half op |
| Full attention | Shard Q/K/V at decode; paged fill/update and SDPA | Already dedicated; head concat |
| Residual add then norm | BF16 DRAM | Residual RMSNorm (residual still needed) |
| MLP gate/up/down | Separate gate/up, SiLU/multiply | Packed gate/up; fused activation |

## Measurements

Final matched measurements are in performance_summary.json and README.md. The chronological experiments below retain intermediate source hashes and rejected candidates.

### First candidates (serial, real weights, S=1024)

Benchmark command: `timeout 1200 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_fused --layer {0|3} --lengths 1024 --benchmark --output <artifact>.json` (`--baseline` for controls).
Each JSON includes whole-output and contribution PCC, cache PCC for full attention, three deterministic
replays, seven warmed prefill timings and seven groups of 50 nonblocking traced decode replays.
Timing runs disable watcher/profiler/allocation tracking. The real-weight HF warning about a missing
optional fast path describes the CPU reference only; device_only guards the measured TT path.

- `baseline_linear.json`: 13.6448 ms prefill / 1.69798 ms traced decode.
- `baseline_full.json`: 8.71046 / 1.89806 ms.
- Dedicated partial RoPE: initial prefill passed; decode returned padded heads and concat failed.
  Slicing the logical head extent fixes the op contract (`rope_adapted_full.json`): 8.60739 / 1.86133 ms. Kept.
- Equivalent L2 via FP32 RMSNorm, epsilon=1e-6/128 and FP32 weight=1/sqrt(128):
  `l2_linear.json`: 12.8578 / 1.65266 ms. Kept. Source `l2_candidate.py.txt`.
- Dedicated QKV causal convolution/SiLU: the program config is exported at ttnn root; its kernels
  require packer_l1_acc=False. Both initial API/config errors retained, then corrected without changing
  matmul precision. Internal padding covers decode and logical tails; state updates take logical rows only.
  `conv_config_linear.json`: 8.88470 / 1.54077 ms. Kept; `conv_candidate.py.txt`.
- Native tiled reshape instead of unconditional untilize/tilize: `reshape_linear.json` 8.7888 / 1.5436 ms
  (decode noise-level +0.18%); full `reshape_full.json` 8.30994 / 1.77887 ms. Kept provisionally for integrated measurement.
- Dedicated concat-heads: first decode call requires sharded SDPA output; request that output directly
  instead of adding an interleaved-to-sharded round trip. Initial failure retained in `concat_heads_full.log`.

### Projection, activation and movement candidates

- Head concat prefill and adapted decode: `concat_heads_reshard_full.json` 7.97485 / 1.76731 ms.
  Direct sharded GQA output is explicitly rejected by sdpa_decode_device_operation.cpp:547;
  the measured adapted path includes reshard + concat + interleaving. Kept pending final audit.
- Packed shared-LHS input projections: linear `packed_linear.json` 8.88163 / 1.49015 ms;
  full `packed_full.json` 7.94310 / 1.70061 ms. Kept.
- Repack Q/gate weights by head at setup and use dedicated prefill head splitting:
  `split_heads_adapted_full.json` 6.91836 / 1.72603 ms. Decode splitter required interleaving before
  slicing a 16-head extent (otherwise invalid 16-row physical shard). The adapted decode remained
  slower and was rejected; prefill retained. The final decode reads the reordered weight columns directly.
- Dedicated minimal_matmul fused SwiGLU: `minimal_mlp_linear.json` 8.26980 / 1.72274 ms;
  `minimal_mlp_full.json` 6.57442 / 1.93613 ms. Explicit decode geometry M=1,K=16,N=12,
  subblock=1x2, grid=8x8 was also measured (`minimal_config_linear.json`: 8.27685 / 1.92456 ms).
  Keep only for prefill. All these tests use BF16/HiFi4 and real model weights.
- Packed gate/up ordinary decode matmul: `gateup_linear.json` 8.24917 / 1.48889 ms.
  Merge SiLU into binary inputs and sigmoid into output gating: `binary_linear.json`
  8.19162 / 1.48241 ms; `binary_full.json` 6.53399 / 1.69351 ms. Kept.
- Additional same-policy alternatives via `tests/fusion_candidate.py` and `tests/run_fusion_candidates.sh`:
  separate gate matmul with fused SiLU 1.48919 ms; separate gate/up with binary SiLU 1.48332 ms;
  one tile-interleaved MLP weight reused by decode with reshapes/slices 1.50974 ms.
  All pass. Packed decode is retained, with an additional 192 MiB phase-specific MLP weight.
  Packed ordinary prefill measured 8.49724 ms versus fused minimal prefill 8.19–8.25 ms and was rejected.
- Full profiler showed two entire 262144x64 RoPE tables untilized per decode (394.394 us total).
  Setup-prepared row-major lookups remove those full-context conversions: `rope_lookup_full.json`
  6.58663 / 1.29500 ms. Combining cos/sin into one lookup further gives
  `rope_packed_full.json` 6.55927 / 1.28651 ms. Both preserve device-valued positions.
- Fused convolution now reads the packed projection directly. For n>=3 update only its last three
  logical rows; n<3 retains preceding history. This removes whole-prefix concat and QKV materialization:
  `conv_movement_linear.json` 7.90047 / 1.48393 ms (decode variation 0.10%). Kept for prefill.

Profiling commands from the repository root: `timeout 1900 bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_fused_profiles.sh {0|3}`. The script changes directory internally.
`python models/autoports/ornith_ai_ornith_1_5_9b/tests/fused_perf_reports.py {linear|full}` emits advice-enabled tables and CSV.
Intermediate profiler evidence is provisional until the final source is reprofiled.

### GDN graph restructuring

- Raw flat Q/K/V feed the existing chunk_size=32 GDN prep, which fuses L2 normalization,
  query scale and GQA head expansion. Non-aligned lengths pad g=0 and beta=0 so inactive
  tail transitions preserve state. `flat_gdn_linear.json`: 6.60061 / 1.48392 ms.
  `linear_flat_battery.json` passes lengths 1,2,3,4,31,32,33,1023,1024,1025,2051,17,
  continuation and batch 32 before subsequent changes.
- Recurrent ring alias eliminates a state copy (`ring_linear.json`: 6.65710 / 1.47189 ms).
- Carry head-major output through per-head RMSNorm and dedicated concat; flatten the SiLU gate only
  after norm: `headmajor_linear.json` 5.92468 / 1.45787 ms.
- The padded flat chunk path also wins for one logical decode token. `chunk_decode_linear.json`
  5.98484 / 1.37726 ms versus recurrent 1.45787 ms. Selected; the ring path is now superseded and
  removed. This eliminates the public recurrent wrapper's internal untilize/tilize relayouts.
- Softplus merged into decay multiply; unified default `unified_gdn_linear.json`
  5.96222 / 1.37952 ms. This 0.16% decode difference is timing noise, not a claimed speedup.

### Advice follow-up

`tt-perf-report` recommends DRAM-sharded implementations of the four packed projection roles.
Tests keep BF16/HiFi4 precision and include all input/output conversions in end-to-end latency.
The first experiment used unsupported Python slicing of TTNN Shape; corrected to integer indexing.
The adapted single-reader config with maximum input shard block hit a measured 1652672-byte L1
requirement versus 1572864 bytes available. Retrying a smaller block, then reader-count alternatives.
No timeout, hang, reset, or weakened precision gate is involved.

### Stop rule and final default

The same-precision DRAM-sharded search is complete in `dram_candidates_driver.log` and
`dram_r*_c*_b*_linear.{json,log}`. Single-reader controls pass at ~1.972 ms when L1 fits, slower
than the interleaved 1.3795 ms. The larger single-reader blocks fail the recorded L1 bound.
Two/four reader pipelines pass; maximum legal input-shard blocks were tested under BF16/HiFi4
and were slower or exceeded L1. The selected configuration is four readers per bank,
64-core activation/output layout target, and in0_block_w=1. per_core_N follows output shard width.

Best-correct sequence near the stop point (traced decode):

| Candidate | Linear ms | Incremental gain |
|---|---:|---:|
| 2 readers, 64 cores, block=2 | 1.218638 | initial DRAM candidate |
| 2 readers, 32 cores, block=1 | 1.2147 | <1% |
| 2 readers, 64 cores, block=1 | 1.2115 | <1% |
| 4 readers, 32 cores, block=1 | 1.2101 | 0.12% |
| 4 readers, 64 cores, block=1 | 1.2078 | 0.19% |

The full-attention selected candidate is 1.12978 ms (`dram_r4_c64_b1_full.json`).
The final default reproduces linear at 1.207738 ms and prefill at 5.873293 ms (`default_linear.json`).
These sub-percent refinements are near timing noise; the material DRAM realization win is measured
against both the interleaved graph and the slower single-reader controls.

Stop Rule 2 applies: the per-layer bandwidth lower bounds are approximately 0.86 ms linear
(active BF16 projection weights plus recurrent read/write) and 0.827 ms full (active BF16 weights
plus 1025-token KV reads), using the installed tt-perf-report Blackhole peak of 512 GB/s.
Both selected traced-decode times exceed 70% of those roofline estimates. The last two kept
layout refinements each gain less than 3%. Do not count the prefill-only duplicate weights as
per-token decode traffic. Final arithmetic and same-run profiler reconciliation are recorded in
performance_summary.json after final profile collection.

No optimized-decoder, multichip, full-model, sampling or serving work has begun. All production
changes are in tt/fused_decoder.py; experiments and evidence remain in this stage's tests/docs.

## Not tried (stop rule)

These are hand-off notes under Stop Rule 2, not correctness exceptions:

- Adapt `experimental.kda.sigmoid_gated_rms_norm` to SiLU by multiplying by z after the fused norm/sigmoid. The final head-major GDN output makes this a plausible three-to-two-op rewrite. Requires packed-gate/padding adaptation and PCC; speedup is unmeasured and bounded by the final norm/concat/gating budget.
- Carry width-sharded residual/norm/gate tensors across compatible projection boundaries; expected
  gain is the measured conversion/norm budget, not a demonstrated speedup. Current conversions are
  required at the selected interleaved specialized-op / sharded-matmul boundaries.
- Additional DRAM reader counts, role-specific core grids and intermediate legal K blocks. Tested
  1/2/4 readers, 32/64-core layouts, block=1/2 and maximum shard-width blocks. No precision sweep.
- BFP8/BFP4 weight or cache policies and lower compute fidelity. This stage preserves BF16/HiFi4
  and FP32 recurrent state; precision selection belongs to the later pipeline stages.
- Dense multi-user graph fusion, multi-chip collective families, full-model terminal operations,
  vLLM and token feedback: explicitly outside this goal. Existing ragged batch-32 semantics remain.

Every graph-fusing skill pattern has an assessment in pattern_audit.md. Applicable graph rewrites
were measured, have a concrete operator/math contract reason, or are explicitly listed above as untried after Stop Rule 2; none is silently omitted.

## Final gates and artifact closure

`tests/run_fused_evidence.sh` completed with EVIDENCE_QUEUE_PASS; `tests/run_fused_contracts.sh` completed with FUSED_CONTRACTS_PASS (468 seconds). Exact repository-root commands and flags are in README.md and both scripts. Final implementation SHA256 is `8f094843786f8f388b241449c0322a684fc398244500274f34ae3862fb07cc10`.

- linear: sampled long-output minimum PCC 0.998594105, contribution minimum 0.998479128; final decode 262144.
- full: sampled long-output minimum PCC 0.998201311, contribution minimum 0.998122931; final decode 262144.

Both complete 262139/262143/262144-token forwards and final-slot traced decode pass with allocation tracking and no program-cache exclusion. Context contract updated with preserved capacity and extra packed/sharded weight and RoPE storage. Boundary parity, 128 successive decode tokens, batch32, resume negative controls, bitwise replay determinism, synthetic pytest and separate watcher batteries pass. Watcher log hashes/audit are in watcher_audit.json.

Matched final S=1024 warmed prefill/traced decode: linear baseline13.615562/1.698006 ms, fused5.867048/1.207541 ms; full baseline8.682155/1.897858 ms, fused6.533450/1.129713 ms. Roofline fractions71.57%/73.24%. The final source reproduces the selected best candidate. performance_summary.json derives the arithmetic and same-run instrumented profiler reconciliation. Final tables/CSV are in tracy_final/{linear,full}; advice-enabled profiling confirms BF16/HiFi4 sharded decode and removal of full RoPE-table untilizes.

No target job timed out or hung; no device reset or process termination was needed. Initial configuration errors are retained and adapted, not used as unsupported rejection evidence. Raw logs are archived losslessly by tests/archive_fused_evidence.py after all device jobs finish. No later pipeline stage started.

## Independent review and closure

Fresh xhigh stage review round 1 returned **clean-pass**, with no Required Work: `stage_review_round1.md`. It independently checked final source identity, all positive PCC, archive hashes, candidate/default ordering, profiler trace metadata and stop-rule arithmetic. Production source is unchanged after review.

Closed nonblocking hard-check note: `tests/validate_fused.py` now explicitly asserts ordinary contribution PCC >=0.995. `contribution_gate_audit.json` applies that same assertion to all existing final/watcher/profile contribution records; all pass. The reviewer explicitly states no device rerun is needed to establish these values. This tightens future test enforcement without changing the decoder or accepted quality threshold.

## Carried to next stage

- Try reusing row-major convolution history in the n<3 update to avoid the additional tiled concat/slice conversions. Unmeasured; Stop Rule 2 already met.
- Geometry convergence was measured on the shared projection policy using linear attention; full attention validates the selected policy. Do not claim an exhaustive independent full-attention geometry sweep.
- Profiler rendered DRAM core counts/FLOP percentages are inconsistent with raw metadata. Raw durations and bandwidth accounting are valid. Investigate reporting separately; do not infer super-physical compute efficiency.
- Sustained traced token feedback and multi-signature lifetime remain generator-stage work. Current 128-token accumulating parity is eager, while separate tracked trace tests cover changed requests and deterministic replay.
- Other untried post-stop graph/layout candidates remain listed under Not tried (stop rule).

All stage-owned changes are confined to the allowed fused implementation, tests and docs/context contract. Local checkpoint SHA will be recorded after committing. No push is authorized or performed.

## Local checkpoint

Repository: `/home/ttuser/ornith-9b/tt-metal`; branch: `main`.
Source, tests, context contract, evidence and independent clean-pass review committed locally as
`7407000e680c06a75b7ace770f92f547da5772ea` (`Add validated fused Ornith decoder and stage evidence`).
All commit hooks passed. No push performed. The following documentation-only commit records this
immutable checkpoint identity; the production source remains at the reviewed SHA256.
