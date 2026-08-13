# Optimized decoder work log

Date: 2026-07-30 UTC

## Scope and starting state

- Model: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Functional-stage commit: `c3cc345a10b`.
- Stage scope: `tt/optimized_decoder.py`, its tests, and optimized-decoder /
  context documentation only.
- Device: one Blackhole p300c from the healthy four-device host; all device
  commands were serialized.
- `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi` list and 1x1 mesh open/close were
  healthy. The recurring MPI `/dev/shm` warning did not affect execution.
- Unrelated `tt_metal/third_party/tt-cluster-descriptors/` remained untouched.

## Baseline

Warmed functional measurements:

| Kind / phase | B1 | B32 |
|---|---:|---:|
| full traced decode | 2.324382 ms | 2.529912 ms |
| linear traced decode | 3.006078 ms | 21.440020 ms |
| full prefill S=33 | 3.820037 ms | 68.680565 ms |
| linear prefill S=5 | 11.663751 ms | 313.205223 ms |

The topology audit preceded tuning. Functional decode kept norms, projections,
residuals, and MLP in DRAM; full attention used three same-input Q/K/V
projections; MLP gate/up were separate; linear attention had packed `in_qkv`
but separate z/b/a; full attention crossed DRAM/L1 around head creation.

## Implementation and experiments

The optimized class reuses proven allocation/state helpers through
`FunctionalDecoder.from_state_dict.__func__(cls, ...)`, but the runtime object
is an `OptimizedDecoder` and its public prefill/decode entry points and
full-attention implementation are independently owned.

Main changes:

- phase-specific prefill and decode weights;
- DRAM-width-sharded decode matmuls;
- L1-width-sharded decode residual, explicit sharded RMSNorm;
- packed Q/K/V/gate and explicit SDPA configs;
- split gate/up MLP with fused SiLU;
- explicit 8x10 prefill 2-D matmul programs;
- physical-row-aware non-aligned B32 prefill config and RoPE reshape;
- BFP8 paged KV cache and BFP8 linear recurrent state without reducing the
  context contract.

Adaptation ledger:

1. Decode block width 20 overcommitted packed-QKV L1 (2,243,328 bytes).
2. Block width 5 reached packed MLP but requested 1,585,408 bytes.
3. Block width 4 overlapped the persistent L1 residual.
4. Block width 2 made the aggressive BFP4 path legal for both layer kinds.
5. Prefill K block 8 requested 1,676,032 bytes for packed MLP; block 4 fit.
6. B32 physical sequence padding required program M from `padded_shape`, not
   logical rows.
7. B32 non-aligned RoPE required a logical and physical reshape shape.
8. BF16 B32 packed attention required prefill K block 2.
9. Direct host-to-DRAM-sharded BF16 tilize requested a 2.34 MiB one-core CB;
   staging through DRAM interleaved enabled a distributed device reshard.
10. The first S=32769 optimized run applied the serving 2-D prefill program to
    the full M dimension and requested a 13,226,752-byte worker CB. Large-M
    projections now use TTNN's general program selection; serving B1/B32 S=33
    retains the measured explicit program.
11. The first S=192511 run then reached a hard DRAM OOM during full-sequence
    RoPE (295,698,432-byte request, 4,092,087,616 bytes/bank allocated). The
    optimized long path now fills K/V and computes Q/gate/O in 32K chunks and
    bounds MLP chunks to 2,048 physical rows. The retry completed at S=192511.
12. The first independent stage review found that full-attention projection
    roles were still coupled and that the inherited linear recurrent composite
    had not been experimentally attacked. Focused source-analysis subagents
    `autofix_full_role_sweep` and `autofix_linear_mixer` were used under the
    optimize workflow; both findings were treated as blockers rather than
    documentation-only items. The retained `AUTOFIX.md` refers specifically to
    the earlier official-weight correctness repair described below.
13. Full QKV, O, gate, up, and down block widths were split into independent
    fields. QKV and O also received independent compute-kernel fidelity so
    HiFi2 projection trials no longer silently change HiFi4 SDPA.
14. The selected full cumulative policy is BF16 QKV/O HiFi2, HiFi4 SDPA,
    QKV width 2, O width 3, split BFP4/LoFi gate/up width 5, and down width 17.
    Its official HF PCC is 0.997612 at B1 and 0.998095 at B32.
15. Linear decode now owns a packed BF16 DRAM-sharded
    `[qkv,z,beta,decay]` projection and BF16 DRAM-sharded output projection.
    It also replaces the K=1 `key-transpose @ delta` batched matmul with the
    equivalent broadcast outer product. The cumulative path preserves
    official HF PCC 0.998852.
16. Linear packed block width 4 was retried after the working width-2 path and
    failed with an exact 1,872,640-byte static-CB request against 1,572,864
    bytes of worker L1. A Tile(4,32) DRAM-height-sharded recurrent challenger
    is blocked by the current Tile(32,32) producer contract: no generic device
    retile exists and matmul requires output tile height to match in0.
17. The determinism runner's decode tensor was B1-shaped as
    `[1,batch,1,H]`; B32 exposed the test bug with a width-shard physical-height
    assertion. The runner now uses `[1,1,batch,H]`. Fresh B32 decode runs for
    both final layer kinds are bit-exact.
18. A second independent stage review required retained, machine-readable
    evidence for the precision-locked full sweep and the two remaining
    recurrent matmuls. The candidate harness now writes command, exit status,
    resolved policy, PCC, latency, and exact device failures as JSON.
19. Full gate/up widths 10 and 20 were attempted at B1 and B32. Width 10
    collided with persistent L1; width 20 requested a 2,690,816-byte static CB
    against 1,572,864 bytes/core. A coherent four-core storage/shard candidate
    also produced exact L1/CB collisions at both batches.
20. Same-BFP4 MLP HiFi2 preserved the official PCC exactly but regressed
    whole traced decode from 1.268/1.454 to 1.674/1.859 ms at B1/B32.
    Role rows show gate/up/down rising from 225.6/159.9/154.4 us to
    358.7/296.6/291.5 us, so LoFi remains selected.
21. The two linear recurrent matmuls now use an explicit program and compute
    policy. Widths 1/2/4, a 1x2 output-subblock candidate, and HiFi4 were
    measured at B1 and B32 and rerun under the final BFP8 state. Width-4
    HiFi2 won at 1.927/16.190 ms, with the B32 recurrent row mean reduced
    from 4.261 ms at width 1 and 2.486 ms at width 2 to 1.717 ms.
22. L1-sharding was closed with exact source contracts: a sharded 1-D
    mcast-in0 tensor requires `fuse_batch=True`, while fused batch requires the
    second input batch to be one (the state has 48/1536 batches). Tile32 B32 A
    would also require 3 MiB/core on four cores. The Tile(4,32)
    DRAM-height-sharded family remains inexpressible without a producer retile.
    `artifacts/program_contracts.json` records source lines and calculations.
23. A fresh AutoFix pass swept the physical persistent recurrent state while
    holding the selected linear topology fixed. Official-weight S=65 prefill
    plus four decode steps gave minimum PCC 0.997950 for FP32, 0.997950 for
    BF16, 0.997965 for BFP8, and 0.993340 for BFP4. BFP8 was selected; BFP4
    was rejected below the 0.995 bar. The final BFP8 policy traced at
    1.929964/16.198048 ms B1/B32.
24. The final BFP8 path passed an S=513 plus 16-step transition, a repeated
    B32 transition bit-exactly, fresh B32 prefill/decode determinism, watcher
    stress, and S=192511 capacity. Capacity retained 985,656,229 nonzero output
    elements and 129,830 nonzero recurrent elements in physical BFP8 storage.
25. The cumulative full policy's complete one-role matrix was run at both
    batches. QKV w4 and O w12 exceeded L1; gate/up w2 and w4 and down w4 were
    slower than the selected w5/w5/w17; gate/up w10/w20, down w34/w68, and
    coherent grid4 hit retained exact L1/CB limits. O w4/w6/w8 were
    neutral-to-slower at B32 than selected w3. MLP HiFi2 preserved official
    PCC but regressed to 1.673865/1.859167 ms.
26. The final prefill profiler was made phase-aware with fixed full S=33 and
    linear S=5 workloads. Five linear iterations completed correctly but
    overflowed Tracy's device-op join (`Op 1117187 not present in
    cpp_device_perf_report.csv`). Reducing only the high-op-count linear
    window to three warmed iterations retained plural repeated measurements
    and produced complete reports; full attention retains five iterations.

AutoFix was invoked after official full-attention PCC was 0.078047. The focused
layout A/B and the per-head q/gate finding are in `AUTOFIX.md` and
`AUTODEBUG.md`.

## Representative commands

Correctness and final performance:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/full_attention_real_pcc.py --candidate default --batch 1 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final_full_real_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/full_attention_real_pcc.py --candidate default --batch 32 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final_full_real_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_real_pcc.py --optimized --candidate default --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final4_linear_real_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/full_attention_synthetic_pcc.py --mode prefill --sequence 33 --batch 32 --iterations 5 --optimized
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py --mode prefill --sequence 65 --batch 1 --optimized
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind full --batch 1 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final_full_traced_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind full --batch 32 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final_full_traced_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 1 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final4_linear_traced_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 32 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final4_linear_traced_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_recurrent_state_transition.py --candidate default --batch 1 --prefill-sequence 513 --decode-steps 16 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final_linear_long_transition_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_recurrent_state_transition.py --candidate default --batch 32 --prefill-sequence 5 --decode-steps 8 --repeat-runs 2 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final_linear_transition_determinism_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/optimized_determinism.py --kind full --mode decode --batch 32 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final4_full_decode_determinism_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/full_attention_synthetic_pcc.py --mode prefill --sequence 32769 --capacity-only --optimized --batch 1
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/full_attention_synthetic_pcc.py --mode prefill --sequence 192511 --capacity-only --optimized --batch 1
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py --mode prefill --sequence 192511 --capacity-only --optimized --batch 1 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final4_linear_capacity_s192511_b1.json
```

Watcher stress, separately from profiling:

```bash
TT_METAL_WATCHER=1 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind full --batch 32 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final_full_watcher_b32.json
TT_METAL_WATCHER=1 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 32 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final4_linear_watcher_b32.json
```

Both watcher runs returned zero, checked and detached all devices, and
contained no error/assert/hang signature. Their JSON has
`"watcher_enabled": true`.

Exact final profiler commands:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind full --batch 1 --candidate default --artifact-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/final4_full_decode_b1
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind full --batch 32 --candidate default --artifact-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/final4_full_decode_b32
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind linear --batch 1 --candidate default --artifact-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/final4_linear_decode_b1
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind linear --batch 32 --candidate default --artifact-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/final4_linear_decode_b32
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind full --phase prefill --batch 1 --candidate default --artifact-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/final4_full_prefill_b1
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind full --phase prefill --batch 32 --candidate default --artifact-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/final4_full_prefill_b32
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind linear --phase prefill --batch 1 --candidate default --artifact-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/final4_linear_prefill_b1
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind linear --phase prefill --batch 32 --candidate default --artifact-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/final4_linear_prefill_b32
```

The wrapper invokes Tracy and `tt-perf-report` with the phase-matched
`PERF_DECODE` or `PERF_PREFILL` signposts, then writes exact argv, exit
statuses, resolved policy output, raw report path, and report output to
`profile_run.json`. Each of the eight `final4_*` directories retains that
provenance plus `perf.csv`, `summary.csv`, and `summary.png`. Full prefill uses
five warmed iterations; linear prefill uses three because its larger op count
overflowed the five-iteration device-profiler join. Raw Tracy `.logs` and
`reports` were moved to desktop trash after filtering and remain recoverable.

## Final evidence

- Full official-weight HF PCC: 0.997612 B1 / 0.998095 B32.
- Linear official-weight HF PCC: 0.998852.
- Official-weight state-transition minimum PCC:
  FP32/BF16/BFP8/BFP4 = 0.997950/0.997950/0.997965/0.993340; BFP8 selected,
  BFP4 rejected.
- Non-aligned prefill PCC: full S=33 B1/B32 0.999994; linear S=5/S=65
  0.999996.
- Traced B1/B32 decode PCC over 10 steps: all representative results
  >=0.999009.
- Exact fresh-run B32 prefill and decode determinism: bit-exact for both kinds;
  the repeated BFP8-state B32 prefill-to-decode transition is also bit-exact.
- B32 10-step traced stress: both kinds pass; watcher-enabled runs clean.
- Long BFP8-state transition: S=513 prefill plus 16 decode steps passes, with
  PCC at least 0.999996.
- Runtime fallback hard-failure mode: clean.
- Optimized full-attention capacity: S=32769 and S=192511 completed with
  nonzero output and populated BFP8 paged cache. The latter output shape was
  `(1, 192511, 5120)`.
- Optimized linear default capacity: S=192511 completed after decode packing
  was enabled, with 985,656,229 nonzero output elements and 129,830 nonzero
  physical BFP8 recurrent-state elements
  (`artifacts/candidates/final4_linear_capacity_s192511_b1.json`).
- Post-capacity short-path regression: full S=33 B1/B32 retained PCC 0.999994.
- Final traced decode: full 1.268103/1.453556 ms B1/B32; linear
  1.670179/15.949088 ms B1/B32.
- Final exact-provenance linear prefill S=5: 11.086341/275.375946 ms versus
  functional 11.663751/313.205223 ms. BFP8 state conversion reduces the
  earlier FP32-state gain but final prefill still improves B1 and B32.
- Final4 profiler accounting: full 1.200407/1.278195 ms and linear
  1.521271/15.893577 ms device time per replay at B1/B32; see `README.md`.
- Final4 prefill profiler accounting: full 3.054999/16.310691 ms and linear
  10.517650/275.038348 ms device time per iteration at B1/B32. Profiled wall
  medians were 3.255676/16.559895 and 11.086341/275.375946 ms; linear reports
  explicitly contain BFP8-to-FP32 expansion and FP32-to-BFP8 writeback.
- Static optimized-path tests and syntax checks: see final gate entry below.

## Final gate and commits

- Independent stage review: `clean-pass`.
- Optimized-decoder checkpoint: `9c5c2811eed260b60b5a85c87274309dd6668088`.
- No push was performed.

## Independent projection-policy AutoFix continuation (2026-07-30)

The fresh review found one remaining P1 evidence gap: packed-input and output
linear projections were fixed at BF16/HiFi2. AutoFix introduced independent
input/output weight-dtype and compute-fidelity policy fields and retained
isolated and cumulative candidates. The BFP8 recurrent-state policy,
grid4_w4 recurrence, packed topology, MLP policy, public sequence semantics,
and context capacity were held fixed.

Serialized ten-step commands used:

```bash
for candidate in linear_proj_bf16_hifi2 linear_input_bf16_lofi linear_output_bf16_lofi linear_both_bf16_lofi linear_input_bfp8_hifi2 linear_input_bfp8_lofi linear_output_bfp8_hifi2 linear_output_bfp8_lofi linear_input_bfp4_lofi linear_output_bfp4_lofi linear_both_bfp4_lofi; do
  for batch in 1 32; do
    python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_optimized_candidate.py --kind linear --batch "${batch}" --candidate "${candidate}" --steps 10 --output "models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/${candidate}_b${batch}.json"
  done
done
```

Candidates were BF16/HiFi2 baseline, input/output/both BF16/LoFi,
input/output BFP8/HiFi2 and BFP8/LoFi, input/output BFP4/LoFi, and cumulative
BFP4/LoFi. All passed B1/B32 synthetic PCC. Exact results are in
`artifacts/candidate_matrix.csv`; the principal traced medians were:

- BF16/HiFi2: 1.925025/16.198853 ms B1/B32.
- input BFP8/LoFi: 1.771583/16.044518 ms.
- input BFP4/LoFi: 1.765712/16.027662 ms.
- output BFP4/LoFi: 1.873437/16.138073 ms.
- both BFP4/LoFi: 1.710466/15.986414 ms.

Official transition commands used S=65 prefill plus four decode steps:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_recurrent_state_transition.py --candidate linear_input_bfp8_lofi --batch 1 --prefill-sequence 65 --decode-steps 4 --real-weights --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/linear_input_bfp8_lofi_transition_real_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_recurrent_state_transition.py --candidate linear_input_bfp4_lofi --batch 1 --prefill-sequence 65 --decode-steps 4 --real-weights --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/linear_input_bfp4_lofi_transition_real_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_recurrent_state_transition.py --candidate linear_both_bfp4_lofi --batch 1 --prefill-sequence 65 --decode-steps 4 --real-weights --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/linear_both_bfp4_lofi_transition_real_b1.json
```

Their minimum PCC values were 0.997965, 0.997432, and 0.997175. Cumulative
BFP4/LoFi is therefore the fastest passing B1 candidate and improves B32.

Focused profiler commands used `tests/run_profiled_candidate.py` at B1 and
B32 for BF16/HiFi2, input BFP8/LoFi, input BFP4/LoFi, output BFP4/LoFi, and
cumulative BFP4/LoFi. Device time per replay was respectively
1.776752/16.146869, 1.621828/16.000345, 1.615684/15.989006,
1.719812/16.089277, and 1.562287/15.933102 ms. Raw Tracy `.logs` and
`reports` were moved recoverably to desktop trash after compact
`profile_run.json`, `perf.csv`, `summary.csv`, and `summary.png` were retained.

After promoting cumulative BFP4/LoFi, final default evidence was refreshed:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_real_pcc.py --optimized --candidate default --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final5_linear_real_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_recurrent_state_transition.py --candidate default --batch 1 --prefill-sequence 65 --decode-steps 4 --real-weights --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final5_linear_transition_real_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 1 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final5_linear_traced_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 32 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final5_linear_traced_b32.json
TT_METAL_WATCHER=1 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 32 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final5_linear_watcher_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/optimized_determinism.py --kind linear --mode decode --batch 32 --candidate default --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final5_linear_decode_determinism_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/optimized_determinism.py --kind linear --mode prefill --batch 32 --candidate default --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final5_linear_prefill_determinism_b32.json
```

Final official decode PCC is 0.998677 and transition minimum PCC is 0.997175.
Ten-step medians are 1.709965/15.990722 ms. The B32 watcher run was clean,
and fresh B32 decode and prefill were bit-exact. Refreshed final4 compact
profiler windows report decode device/wall time
1.562080/1.724553 ms at B1 and 15.925412/16.006359 ms at B32; prefill
device/wall time is 10.514529/11.045196 ms at B1 and
275.007068/275.355469 ms at B32. Post-run `tt-smi -s` reported all four p300c
devices healthy with DRAM enabled and no uncorrectable GDDR errors.
Final static gates passed: `py_compile` for optimized decoder and affected
runners, 117/117 optimized-path pytest cases, 82 candidate-matrix rows with
exactly 12 CSV columns, and seven valid final5 JSON artifacts.

## Precision-locked projection geometry AutoFix (2026-07-30)

The subsequent review required the selected BFP4/LoFi input/output precision
to be crossed with material geometry. The BFP8 state, grid4_w4 recurrence,
BFP4/LoFi MLP, eight-core residual storage, and all public semantics were held
fixed. Exact B1/B32 candidate commands were:

```bash
for candidate in linear_final_input_w1 linear_final_input_w4 linear_final_input_w5 linear_final_input_w10 linear_final_input_w20 linear_final_output_w1 linear_final_output_w2 linear_final_output_w4 linear_final_output_w6 linear_final_output_w8 linear_final_output_w12 linear_final_output_w24 linear_final_input_w5_output_w8 linear_final_input_w5_output_w12 linear_final_input_w5_output_w24 linear_final_grid4; do
  for batch in 1 32; do
    python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_optimized_candidate.py --kind linear --batch "${batch}" --candidate "${candidate}" --steps 10 --output "models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/${candidate}_b${batch}.json"
  done
done
```

Packed-input widths 1/4/5 passed at both batches; width 5 won at
1.684406/15.959816 ms. Width 10 hit an exact L1/static-CB overlap and width 20
required 2,587,136 bytes of static CB versus 1,572,864 bytes of L1. Output
widths 1/2/4/6/8/12/24 all passed. Crossing the leaders with input width 5
produced:

- output width 8: 1.674006/15.957983 ms;
- output width 12: 1.670349/15.942844 ms;
- output width 24: 1.672102/15.943516 ms.

The four-core storage adaptation failed with an exact L1/CB overlap at B1 and
B32. Every passing contender was profiled with:

```bash
for candidate in linear_final_input_w1 linear_final_input_w4 linear_final_input_w5 linear_final_output_w1 linear_final_output_w2 linear_final_output_w4 linear_final_output_w6 linear_final_output_w8 linear_final_output_w12 linear_final_output_w24 linear_final_input_w5_output_w8 linear_final_input_w5_output_w12 linear_final_input_w5_output_w24; do
  for batch in 1 32; do
    python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/run_profiled_candidate.py --kind linear --batch "${batch}" --candidate "${candidate}" --artifact-dir "models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/tracy/${candidate}_b${batch}"
  done
done
```

Selected width 5/12 device time is 1.521726/15.890707 ms. The output-subblock
review found no exposed subblock or worker-grid field in
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`; the factory internally
selects 1x8 for packed input and 1x7 for output. Exact API/source paths,
geometry contracts, and the failed grid control are in
`artifacts/program_contracts.json`.

After promotion, final commands were:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_real_pcc.py --optimized --candidate default --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final6_linear_real_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/linear_recurrent_state_transition.py --candidate default --batch 1 --prefill-sequence 65 --decode-steps 4 --real-weights --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final6_linear_transition_real_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 1 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final6_linear_traced_b1.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 32 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final6_linear_traced_b32.json
TT_METAL_WATCHER=1 python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --kind linear --batch 32 --optimized --steps 10 --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final6_linear_watcher_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/optimized_determinism.py --kind linear --mode decode --batch 32 --candidate default --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final6_linear_decode_determinism_b32.json
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/optimized_determinism.py --kind linear --mode prefill --batch 32 --candidate default --result-json models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/artifacts/candidates/final6_linear_prefill_determinism_b32.json
```

Final official decode PCC is 0.998717 and real transition minimum PCC is
0.997167. Final ten-step trace medians are 1.670179/15.949088 ms. Watcher is
clean and B32 decode/prefill are bit-exact. Refreshed final4 decode
device/wall is 1.521271/1.687251 ms at B1 and 15.893577/15.975878 ms at B32.
Refreshed prefill device/wall is 10.517650/11.086341 ms and
275.038348/275.375946 ms.
Final static gates passed: Black check, `py_compile`, 165/165 optimized-path
pytest cases, 114 candidate-matrix rows with exactly 12 columns, parseable
program-contract and final6 JSON, 26 successful compact contender profiles,
no retained raw `.logs`/`reports`, scoped `git diff --check`, and parser
`--help` validation for every documented runner flag.
