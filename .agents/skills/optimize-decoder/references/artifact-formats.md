# Optimized Decoder Artifact Formats

All paths in JSON are relative to `models/demos/<model>/doc/optimized_decoder/` unless stated otherwise. Use strict JSON, UTF-8, two-space indentation, and `"schema_version": 1`. Use `null` for unavailable values and empty arrays for no entries.

## Directory Contract

The final golden optimized-decoder evidence lives directly under:

```text
models/demos/<model>/doc/optimized_decoder/
```

Do not create per-debug-run or per-attempt directories in this doc tree. Keep functional baseline evidence in `doc/functional_decoder/` and future phase evidence in sibling directories such as `doc/multidevice/`.

## Status Values

Use these exact status strings:

- `pass`
- `fail`
- `blocked`
- `skipped`

`fail`, `blocked`, and `skipped` are valid status values for diagnostic artifacts, but final optimized-decoder success requires `manifest.status == "pass"`. Stress is required for optimized decoder: a final passing artifact must record stress as `pass`, never `skipped`.

Use these exact mode strings:

- `prefill`
- `decode`

Use lowercase hyphen-case for `layer_kind_id` and optimization ids.

## manifest.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_manifest",
  "step_id": "optimized_decoder",
  "status": "pass",
  "model": {
    "id": "org/example-model",
    "slug": "example-model",
    "hf_revision": null
  },
  "repo": {
    "git_commit": "abcdef1234567890",
    "git_short_sha": "abcdef1",
    "git_dirty": true
  },
  "environment": {
    "host": "hostname",
    "hardware": "blackhole-p100",
    "arch": "BLACKHOLE",
    "ttnn_version": null,
    "python_version": "3.10.12",
    "tt_metal_home": "/path/to/tt-metal"
  },
  "functional_baseline": {
    "artifact_dir": "../functional_decoder",
    "manifest": "../functional_decoder/manifest.json"
  },
  "started_utc": "2026-05-26T14:00:00Z",
  "ended_utc": "2026-05-26T15:00:00Z",
  "paths": {
    "report": "optimized_decoder.md",
    "commands": "commands.sh",
    "baseline_summary": "baseline_summary.json",
    "functional_regression_results": "functional_regression_results.json",
    "optimization_plan": "optimization_plan.json",
    "precision_results": "precision_results.json",
    "program_config_results": "program_config_results.json",
    "data_movement_audit": "data_movement_audit.md",
    "fallback_audit": "fallback_audit.md",
    "tt_perf_advice": "tt_perf_advice.json",
    "performance_results": "performance_results.json",
    "pcc_results": "results/pcc_results.json",
    "kv_cache_results": "results/kv_cache_results.json",
    "determinism_results": "results/determinism_results.json",
    "stress_results": "results/stress_results.json",
    "watcher_summary": "watcher/watcher_summary.json"
  },
  "commands": [
    {
      "id": "pytest_dense_decode",
      "purpose": "optimized decode PCC",
      "command": "pytest ...",
      "exit_code": 0,
      "log": "pytest/dense_decode.log"
    }
  ]
}
```

## baseline_summary.json

`baseline_status` must be `pass`. If the functional baseline did not pass, stop and rerun or repair functional bringup before producing final optimized artifacts.

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_baseline_summary",
  "functional_artifact_dir": "../functional_decoder",
  "baseline_status": "pass",
  "layer_kinds": [
    {
      "layer_kind_id": "dense",
      "functional_prefill_pcc": 0.9991,
      "functional_decode_pcc": 0.9990,
      "functional_prefill_device_ms": 12.3,
      "functional_decode_device_ms": 0.45
    }
  ]
}
```

## functional_regression_results.json

This file proves the tests from `doc/functional_decoder/` still pass against the optimized implementation. Do not weaken assertions to make this pass; use explicit optimized-mode parametrization only when it preserves the same checks.

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_functional_regression_results",
  "functional_artifact_dir": "../functional_decoder",
  "status": "pass",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "functional_test_id": "test_decoder_decode_pcc",
      "optimized_mode": true,
      "assertions_preserved": true,
      "status": "pass",
      "command_id": "pytest_dense_decode",
      "log": "pytest/dense_decode.log",
      "notes": []
    }
  ]
}
```

## optimization_plan.json

For MoE models, add a required step with id `single-user-moe-active-experts`, status `pass`, and evidence pointing to the report section or audit artifact that proves optimized execution follows the gate-selected active-expert path rather than a dense all-expert path.

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_plan",
  "required_steps": [
    {"id": "functional-regression", "status": "pass", "evidence": "functional_regression_results.json"},
    {"id": "stress", "status": "pass", "evidence": "results/stress_results.json"},
    {"id": "watcher-clean", "status": "pass", "evidence": "watcher/watcher_summary.json"},
    {"id": "runtime-contract", "status": "pass", "evidence": "fallback_audit.md"},
    {"id": "l1-sharded-activations", "status": "pass", "evidence": "performance_results.json"},
    {"id": "program-configs", "status": "pass", "evidence": "program_config_results.json"},
    {"id": "dram-sharded-matmuls", "status": "pass", "evidence": "program_config_results.json"},
    {"id": "bfp8-weights", "status": "pass", "evidence": "precision_results.json"},
    {"id": "bfp8-kv-cache", "status": "pass", "evidence": "precision_results.json"},
    {"id": "mlp-bfp4", "status": "pass", "evidence": "precision_results.json"},
    {"id": "compute-kernel-fidelity", "status": "pass", "evidence": "program_config_results.json"},
    {"id": "tt-perf-report-advice", "status": "pass", "evidence": "tt_perf_advice.json"}
  ],
  "notes": []
}
```

## precision_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_precision_results",
  "pcc_threshold": 0.995,
  "results": [
    {
      "layer_kind_id": "dense",
      "tensor_group": "ff1-ff3",
      "candidate_dtype": "bfloat4_b",
      "accepted": true,
      "prefill_pcc": 0.997,
      "decode_pcc": 0.996,
      "pcc_delta_vs_functional": -0.0021,
      "performance_delta_pct": 8.2,
      "notes": []
    }
  ]
}
```

## program_config_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_program_config_results",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "op_group": "ff1-ff3",
      "program_config_type": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
      "memory_config": "L1_WIDTH_SHARDED",
      "compute_kernel_config": {
        "math_fidelity": "LoFi",
        "math_approx_mode": false,
        "fp32_dest_acc_en": false,
        "packer_l1_acc": true
      },
      "core_grid": {"x": 8, "y": 4},
      "accepted": true,
      "reason": "DRAM-bound decode matmul improved without PCC regression"
    }
  ]
}
```

## tt_perf_advice.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_tt_perf_advice",
  "advice": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "op_id": 42,
      "op_name": "ttnn.linear",
      "advice_text": "Use DRAM-sharded matmul",
      "action": "accepted",
      "before_device_us": 120.0,
      "after_device_us": 82.0,
      "pcc_after": 0.997,
      "rejection_reason": null
    }
  ]
}
```

## performance_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_performance_results",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "baseline_device_ms": 0.45,
      "optimized_device_ms": 0.31,
      "speedup": 1.45,
      "dominant_bottleneck": "dram-bandwidth",
      "ops_csv": "tracy/dense/decode_ops.csv",
      "perf_report_csv": "tracy/dense/decode_perf_report.csv",
      "perf_report_txt": "tracy/dense/decode_perf_report.txt",
      "notes": []
    }
  ]
}
```

## results/pcc_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_pcc_results",
  "threshold": 0.995,
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "pcc": 0.997,
      "passed": true,
      "functional_baseline_pcc": 0.999,
      "pcc_delta_vs_functional": -0.002,
      "trace_replay_used": true,
      "command_id": "pytest_dense_decode",
      "log": "pytest/dense_decode.log"
    }
  ]
}
```

## results/kv_cache_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_kv_cache_results",
  "results": [
    {
      "layer_kind_id": "dense",
      "cache_dtype": "bfloat8_b",
      "paged": true,
      "max_decode_seq_len_tested": 131072,
      "page_size": 64,
      "prefill_boundary_supported": true,
      "decode_pcc": 0.997,
      "accepted": true,
      "rejection_reason": null,
      "command_id": "pytest_dense_decode",
      "log": "pytest/dense_decode.log"
    }
  ]
}
```

## watcher/watcher_summary.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_watcher_summary",
  "runs": [
    {
      "layer_kind_id": "dense",
      "command_id": "pytest_dense_watcher",
      "clean": true,
      "env": {
        "TT_METAL_WATCHER": "2",
        "TT_METAL_WATCHER_NOINLINE": "1",
        "TT_METAL_WATCHER_TEST_MODE": "1",
        "TT_METAL_WATCHER_DISABLE_ETH": "1",
        "TT_METAL_LOGS_PATH": "watcher/dense"
      },
      "logs": {
        "pytest": "pytest/dense_watcher.log",
        "watcher": "watcher/dense/generated/watcher/watcher.log",
        "kernel_names": "watcher/dense/generated/watcher/kernel_names.txt",
        "kernel_elf_paths": "watcher/dense/generated/watcher/kernel_elf_paths.txt"
      },
      "detected_messages": [],
      "false_positive_justifications": [],
      "status": "pass"
    }
  ]
}
```

## results/determinism_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_determinism_results",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "passed": true,
      "repeat_count": 5,
      "method": "bytewise_equal_after_to_torch_at_output_boundary",
      "nondeterministic_tensors": [],
      "command_id": "pytest_dense_decode"
    }
  ]
}
```

## results/stress_results.json

Include one entry for each representative layer kind and optimized stress mode that the test suite exercises. All entries must pass for a final optimized-decoder `pass`; do not use `skipped` for stress in final golden artifacts.

```json
{
  "schema_version": 1,
  "artifact_type": "optimized_decoder_stress_results",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "status": "pass",
      "duration_seconds": 300,
      "iteration_count": 1200,
      "reason": null,
      "command_id": "pytest_dense_decode_stress",
      "log": "pytest/dense_stress.log"
    }
  ]
}
```

## data_movement_audit.md

Use these exact headings:

```markdown
# Data Movement Audit

## Runtime Path

## Sharding Plan

## Removed Movement

## Remaining Movement

## Justified Movement

## Residual Risk
```

## fallback_audit.md

Use these exact headings:

```markdown
# Runtime Fallback Audit

## Prefill Runtime Boundary

## Decode Runtime Boundary

## Torch Usage

## TTNN Host Transfers

## Trace Capture Allocations

## Evidence

## Residual Risk
```

## optimized_decoder.md

Use these exact headings:

```markdown
# Optimized Decoder

## Result

## Functional Baseline

## Optimization Summary

## Precision

## Program And Memory Configs

## Functional Regression

## Performance

## TT-Perf-Report Advice

## Correctness

## Watcher

## Data Movement

## Commands

## Residual Risk
```
