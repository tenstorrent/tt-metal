# Qwen3.6-27B optimized decoder

The selected implementation is `tt/optimized_decoder.py`, frozen delivery194
SHA256 `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.
It accelerates the fused single-device decoder, retains the 262144-token context
contract and accepts non-aligned logical lengths. All58 final-source suite jobs and4 pytest cases pass. The fresh independent [stage review](stage_review_round1.md) returned
**clean-pass** with no required work. Implementation and evidence are committed
locally as `2d86c2f5734198ffd9121fb9a9066d289a6a6e08`; nothing was pushed.

This is stage3, optimized-decoder only. Layer0 is gated delta-network linear
attention (GDN); layer3 is full attention. Both use real checkpoint weights at
revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`. Measurements use physical
Blackhole chip0 on a P300, one1×1 mesh. No multichip, full-model or vLLM work
is included. Starting checkout: `4c79322a4c142187308864977f00cbe5012c0352`.

## Measurements and acceptance

[Final measurement tables](final_measurements.md) contain matched before/after
warmed prefill and traced warmed decode, all final shape rows, per-kind stress
PCC and profiler accounting. `build_final_tables.py` derives them directly from
JSON. Final performance rows have profiler/allocation tracking disabled;
correctness rows enable allocation tracking including program-cache allocations.
The fused baseline uses the same legacy `[B,1,H]` interface. Preferred compact
`[1,B,H]` decode avoids two boundary reshapes; its timings are reported separately.

The unchanged acceptance bar is PCC≥.995. Every meaningful kind is checked
against HF, plus fused output/contribution/state equivalence and independent
HF recurrent state. Short tests check bit equality on repeated inputs and
changed-output behavior after trace input updates. Stress changes inputs across
128 steps at B1 and512 steps at B32, retaining each per-user PCC. A fast short
candidate does not override a long-sequence failure.

GDN uses higher effective input precision to pass the long gate. Primary and
quantized residual matrices are both BFP8; QKV/B/A use both terms, Z and output
use one. This is not equivalent to single-term BFP8 storage or accuracy.
`capacity_delivery194.json` accounts for the extra term and phase-specific
prefill copies. Final GDN B1/S512 prefill is6.213297ms versus fused5.577254ms;
B32/S512 also has a precision-related prefill cost, explicitly reported in the
final tables. Decode gains do not erase these prefill regressions.

The functional GDN state oracle is defective on correlated real-text inputs
(PCC .017 against HF after prefill). The harness uses fused state plus an
independent HF state comparison; repaired evidence is in
`oracle_repaired_text.json` and `AUTODEBUG_functional_state_oracle.md`.
Fused GDN also fails the separate512-step accuracy test, whereas the selected
optimized path passes. It remains a historical performance baseline, not an
eligible correct long-sequence candidate. Original failing evidence is retained.

## Selected configuration

| Area | Selected behavior and evidence |
|---|---|
| Precision | GDN attention BFP8 primary/residual, HiFi4, packer accumulation off; MLP BFP8/HiFi2, packer on. Full attention BFP8/LoFi, packer on. BF16 activations/state with FP32 GDN recurrence and current readout before state rounding. Full KV BFP8. |
| Decode matmuls | DRAM-sharded role-specific configs: GDN input C40/R5/K1, output C32/R4/K1; MLP pipeline2 readers/shared input. Real-weight geometry, fidelity and reader sweeps are retained. |
| Packing | GDN primary/residual input projections remain separate matmuls; narrower residual physical width11520 excludes Z. Full decode packed QKV plus separate gate; full prefill packed QKVG. Decode MLP gate/up remain separate; native tile-pair SwiGLU prefill is fused. Legal packed/separate controls were measured. |
| Prefill | Public chunk512 handles arbitrary tails; GDN internal row cap640, grid11×10, primary K8, residual K10/SW3; native SwiGLU block2/8/16. Larger legal configs were slower. Nonaligned multi-user unflatten uses exact aligned NoC face-row transfers. |
| Normalization | Exact FP32 reduction tree; temporary folded/inverse L1 scratch and80-core output. Full-attention BF16 clone preserves rounding and private scratch is released before large matmul allocations. |
| GDN kernels | Packed Q/K preparation, balanced recurrence owners, exact delta row arithmetic, fused persistent-state RNE, affine/gate direct transfer and row-broadcast, exact native mixed-input FPU add/softplus/sigmoid gate. |
| SDPA/cache | Native SDPA; B1 decode grid8×4, B32 paged8×8. Short allocated tables use decode key64; large tables retain512. Page32 semantics, shuffled physical pages, mixed per-row positions and continuation preserved. |
| Runtime movement | Compact residuals, direct sharded projections, exact decode convolution and staging kernels. Direct-call audit, independent inspection of selected indirect helpers, and runtime conversion sentinels find no torch/from_torch/to_torch or host fallback. Necessary device layout conversions remain where measured alternatives lose or violate exact rounding. |

[Kernel contracts](kernel_contracts.md) and
[attention/gate kernel contracts](kernel_contracts_attention.md) document
arithmetic order, NoC/CB synchronization, padding, scratch and trace lifetime,
and exact helper equivalence. Two old recurrence comments say one-head ring
depth; actual capacity is two heads, with one-head reserve/push/pop payloads.
The code and proof agree; the comments do not describe the capacity correctly.

## Profiling, decisions and stop rule

The initial operation-topology audit is the first table in `work_log.md`.
[Profiler conclusions](final_profile_conclusions.md) explain dominant rows and
chunk/tail costs. [Checklist](checklist.md) maps every applicable optimization family to measured
choices, repaired failures or exact blockers. [Work log](work_log.md) records
commands, before/after numbers, rejected options and untried hand-off work.

Advice-enabled tt-perf-report tables and CSV for both phases are under
`perf/final194_profile_profile_l{0,3}_b{1,32}_s{512,33,513,1025}/` for the
recorded shape combinations. Each retains compressed raw ops, report commands,
source hash, metrics and same-run accounting. `perf/profile80_bfp4_0/` and
BFP4 precision rows measure real weights; stage8 datatype-sweep makes the final
full-model precision choice. This stage keeps BFP8 by default.

Material advice was tried and repaired before rejection: direct SwiGLU input
L1 was slower (26.573/6.755ms versus26.462/6.749); primary-prefill K10 first
exceeded L1, then legal BH1 gave174.670ms versus residual-only174.323ms at
B32/S512; larger legal row caps1024/1280 gave220.092/200.836ms versus188.435.
Corrected separate attention projections were slower and failed long PCC;
packed primary+residual correction pairs were legal but slower. Legal MLP
L1 output and BF16-KV SDPA repairs also lost. Exact norm, packed Q/K, narrow
correction, convolution movement, QKV staging and affine/gate changes were
kept only after whole-layer accuracy and timing. No first API/allocation error
is used alone to reject these material families.

`stop_rule_selection.json` records ≥70% same-run bandwidth roofline and the
last two kept improvements below3%:185→189→193 gives1.592330→1.586212→1.579975ms,
gains0.384237% and0.393176%. Profile195 gives1.123648ms lower bound divided by
1.602493ms measured host latency =70.119%. Delivery194 reproduces70.120%.
Actual11520-column residual payload is counted. This is a bandwidth lower
bound, not a claim of70% compute utilization; no8h-device-time claim is made.
The required `Not tried (stop rule)` section is in the work log.

## Context, validation and limitations

All four final-source B1 context runs pass:262143/262144 prefill for both kinds,
traced decode at context262144, and full-attention high-position continuation.
[Context contract](../context_contract.json) and
[capacity accounting](capacity_delivery194.json) preserve262144 without a
capability reduction. Largest-context HF compares selected output rows after
processing every token into state/cache; TT executes the entire decoder on
all tokens. Full-context B32 capacity is not claimed.

Final gate launchers are `final_jobs_v2.py` (26 correctness,4 context,14
performance,8 profile jobs) and `final_long_prefill.py` (6 B32 prefill jobs).
Exit-code ledgers and result provenance bind all58 runs to the delivery SHA.
`test_optimized_decoder.py` asserts the optimized class is selected and adds
four pytest cases. Watcher runs separately from profiling and covers both
kinds, contract/continuation, trace reuse, mixed positions and B3/S513. Stress
and trace-reuse gates include allocation tracking with program-cache coverage
and positive controls. The hard-gate inventory is [final_evidence_check.json](final_evidence_check.json);
all58 runs plus4 pytest cases pass on the frozen source.
`delivery194_manifest.json` records review and commit status.

Dense MLP means MoE active-expert execution is not applicable. CCL, terminal
norm/head, sampling and token feedback belong to later stages. Real-text
activations here are decoder precision probes, not generated-text evaluation.
No device reset was needed. Benign motherboard discovery/subset-MMIO warnings
refer to the host's multi-chip inventory; only chip0 is opened for computation.

## Reproduction

From the repository root, with the pinned checkpoint cached:

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
unset TT_VISIBLE_DEVICES QWEN_OPT_CANDIDATE QWEN_OPT_POLICY QWEN_FUSION_CANDIDATE
export QWEN_DECODER=optimized
python -m models.autoports.qwen_qwen3_6_27b.tests.prepare_text_activations --batch 32 --length 33 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/text_activations
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/final_jobs_v2.py --suite correctness --tag reproduce_correctness
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/final_jobs_v2.py --suite context --tag reproduce_context
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/final_jobs_v2.py --suite performance --tag reproduce_performance
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/final_jobs_v2.py --suite profile --tag reproduce_profile
```

Use the exact commands in the work log for final pytest and long-prefill runs.
`run_job.sh` archives source, serializes hardware with flock, bounds each job
with timeout and the installed skill wait_for, and keeps complete logs.
Do not set TT_VISIBLE_DEVICES=0: this checkout's topology discovery requires
explicit physical_device_ids=[0] with the variable unset.

`candidate_summary.csv` / `sequence_summary.csv` summarize exploratory rows;
source JSON, `candidates/` and `source_archives/` retain immutable measured
implementations and policies. `.pt` diagnostics and raw Tracy directories are
ignored; `ignored_tensor_inventory.json` records local tensor hashes and the
text preparation command above regenerates final fixtures. Rendered reports
and compressed raw ops are retained in git. No push is authorized or performed.
