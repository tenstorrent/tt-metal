# Exact review of four native layout warnings

The four warnings are benign **for this captured composite operation**: native `slice` correctly writes both requested outputs to DRAM. No production/native change or hardware rerun was needed. This is not a general exemption for ignored `memory_config` warnings.

[Capture](capture.log) SHA256 `9e732a7f939a3f81a39e9ca06dee9419417dd3a021144a8e08d5db123ce309ec`; [raw CSV](raw/reports/2026_11_13_11_40_05/ops_perf_results_2026_11_13_11_40_05.csv) SHA256 `2a9102451b9a1ecd9089f69788c108b91fd7d827b5190e91c779f9763c297cea`. Captured decoder is ab28, and the ordinary validator still verifies its complete harness/source contract.

## Source chain and actual placements

Frozen [ab28](../v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot):1364–1365 explicitly calls `ttnn.slice(..., memory_config=DRAM)` on generalized-gate scores and indices. These are two separate singleton full-attention decode outputs, BF16 and UINT16 respectively; both start ROW_MAJOR HEIGHT_SHARDED L1.

Native `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp`:382 requests ROW_MAJOR input via `to_layout(input, ROW_MAJOR, ..., memory_config)`. `ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp`:73–82 warns because the input already has that layout and the requested DRAM memory differs; it returns the original L1 tensor. Crucially, `slice.cpp`:433–444 then passes the original requested `output_memory_config` into `prim::slice`. This **output** operation performs the intended L1→DRAM placement. It does not rely on the preceding no-op layout check to move memory.

The raw CSV proves the actual output, not just the intended code. Gate scores/indices are ROW_MAJOR L1_HEIGHT_SHARDED; both following slice outputs are ROW_MAJOR DRAM_INTERLEAVED `[1,1,1,8]`, with unchanged BF16/UINT16 dtype. Scores subsequently tilize for the existing math; indices remain the UINT16 ROW_MAJOR single stick required by sparse matmul. All44slice outputs—two each from eager decode, trace construction, and20replays—satisfy these exact checks.

| Capture warning line / UTC | Source call | Raw zero-based CSV row / global call count | Actual slice output |
|---|---|---|---|
| 5692 /11:39:55.181 | Eager score slice, ab28:1364 | 266 /271360 | BF16 ROW_MAJOR DRAM, `[1,1,1,8]` |
| 5790 /11:39:55.712 | Eager index slice, ab28:1365 | 268 /273408 | UINT16 ROW_MAJOR DRAM, `[1,1,1,8]` |
| 5863 /11:39:56.049 | Trace-construction score slice | 336 /343040 | BF16 ROW_MAJOR DRAM, `[1,1,1,8]` |
| 5864 /11:39:56.049 | Trace-construction index slice | 338 /345088 | UINT16 ROW_MAJOR DRAM, `[1,1,1,8]` |

Mapping follows the sequential score-then-index composite call order, the two gate instances in eager/capture metadata, and warning placement before eager completion/trace completion. The warning itself contains no global operation ID; it is emitted during the composite input-layout check before the underlying slice is submitted. Trace replays execute captured programs and do not reissue the host layout warning. Captured repeated operand/program metadata are independently identical across all20replays.

## Bounded validator extension

Original [validator snapshot](validate_capture_host.initial.py.snapshot), SHA256 `21df996bcc949c0e2aab94e99c2f33a8bf06a6bcceae692c77518b6f8f7716d8`, and [initial failure](validation.initial.log), SHA256 `b4bfd750dc96234a3419a143b6e652a829b664ff54ea2d7694b24050422b2034`, are preserved. The original `validation.log` and plan validator were not changed.

New [reviewed validator](../v6_profile_ab28_warmed_s256_plan/validate_capture_host_layout_review.py), SHA256 `ceba8d15d2695ea45f8a77fa8ce23e7080694247aef421509546d193377ba97a`, permits only these exact four reviewed warning records after validating layer3, the complete capture/raw hashes, original validator hash, native slice/layout source hashes, four precise line/timestamp pairs and all44slice metadata. It records [detailed review](layout_warning_review.json). Negative controls change output memory to L1, dtype to UINT32, or width to32; all are rejected. Other warnings retain the original rejection rule.

[Scope proof](layout_warning_validator_scope.json) restores the entire original validator AST by removing only two review functions and two review assignments, then restoring its original unknown-warning assertion. PCC, harness/source provenance, signposts, completeness, durations, session fingerprints, drop/failure detection, overlap and report checks are unchanged. There is no blanket warning allowlist and no deletion or alteration of capture data.

## Completed same-capture validation

[Validation log](validation.layout_review.log) exits0. [Completeness](profiler_completeness.json) reports104warmed-prefill operations and68operations per decode replay; all20trace sequences identical,16measured decode replays, complete positive executed durations, no missing/drop markers and zero overlap. [Prefill](prefill_perf_report.txt) and [decode](decode_perf_report.txt) text reports plus their CSV tables exist, generated with active-experts8 and exact PERF_PREFILL/PERF_DECODE windows.

Kernel-duration sums are8,721.485μs/prefill and549.4005μs/decode; they exclude dispatch/profiler gaps and are not clean wall-latency measurements. Largest classes are prefill grouped matmul6,540.014μs (about75.0%), and decode matmul189.527μs plus sparse matmul173.219μs (together66.0%). [Topology](op_topology.md) and [summary](perf_summary.json) retain full counts/placement and report conclusions. Full stage acceptance still requires the separate correctness, clean benchmark, context and review gates.

Host-only reproduction (the validator intentionally refuses an existing completeness file, preserving this completed result):

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_profile_ab28_warmed_s256_plan/validate_capture_host_layout_review.py \
  --artifact models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_profile_ab28_warmed_s256_layer3 --layer 3
```
