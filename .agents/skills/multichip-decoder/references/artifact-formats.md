# Multichip Decoder Artifact Formats

All paths in JSON are relative to `models/demos/<model>/doc/multichip_decoder/` unless stated otherwise. Use strict JSON, UTF-8, two-space indentation, and `"schema_version": 1`. Use `null` for unavailable values and empty arrays for no entries.

## Directory Contract

Final golden multi-chip evidence lives directly under:

```text
models/demos/<model>/doc/multichip_decoder/
```

Do not create per-debug-run or per-attempt directories in this doc tree. Keep earlier phase evidence in sibling directories:

```text
models/demos/<model>/doc/functional_decoder/
models/demos/<model>/doc/optimized_decoder/
models/demos/<model>/doc/multichip_decoder/
```

## Status Values

Use exact status strings:

- `pass`
- `fail`
- `blocked`
- `skipped`

Final success requires `manifest.status == "pass"`.

Use exact mode strings:

- `prefill`
- `decode`

Use lowercase hyphen-case for `layer_kind_id` and strategy ids.

## manifest.json

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_manifest",
  "step_id": "multichip_decoder",
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
    "hardware": "t3k",
    "arch": "WORMHOLE_B0",
    "ttnn_version": null,
    "python_version": "3.10.12",
    "tt_metal_home": "/path/to/tt-metal"
  },
  "baseline": {
    "step_id": "optimized_decoder",
    "artifact_dir": "../optimized_decoder",
    "manifest": "../optimized_decoder/manifest.json",
    "status": "pass"
  },
  "target_mesh": {
    "mesh_shape": [1, 8],
    "num_devices": 8,
    "strategy_id": "1d-tp",
    "tp": 8,
    "ep": 1,
    "sp": 1
  },
  "started_utc": "2026-05-26T14:00:00Z",
  "ended_utc": "2026-05-26T15:00:00Z",
  "paths": {
    "report": "multichip_decoder.md",
    "commands": "commands.sh",
    "baseline_summary": "baseline_summary.json",
    "mesh_contract": "mesh_contract.json",
    "parallelization_plan": "parallelization_plan.json",
    "memory_capacity_plan": "memory_capacity_plan.json",
    "baseline_regression_results": "baseline_regression_results.json",
    "fallback_audit": "fallback_audit.md",
    "data_movement_audit": "data_movement_audit.md",
    "tt_perf_advice": "tt_perf_advice.json",
    "performance_results": "performance_results.json",
    "sequence_limits": "sequence_limits.json",
    "pcc_results": "results/pcc_results.json",
    "kv_cache_results": "results/kv_cache_results.json",
    "determinism_results": "results/determinism_results.json",
    "stress_results": "results/stress_results.json",
    "watcher_summary": "watcher/watcher_summary.json"
  },
  "commands": [
    {
      "id": "pytest_dense_decode",
      "purpose": "target mesh decode PCC",
      "command": "pytest ...",
      "exit_code": 0,
      "log": "pytest/dense_decode.log"
    }
  ]
}
```

## commands.sh

Plain executable shell transcript. Each command block must start with a command id comment that matches `manifest.json`:

```bash
# command_id: pytest_dense_decode
# purpose: target mesh decode PCC
pytest models/demos/example/tests/test_layer.py -k "dense and decode and multichip" -vv 2>&1 | tee pytest/dense_decode.log
```

## baseline_summary.json

The baseline may be `functional_decoder` or `optimized_decoder`. `baseline_status` must be `pass`.

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_baseline_summary",
  "baseline_step_id": "optimized_decoder",
  "baseline_artifact_dir": "../optimized_decoder",
  "baseline_status": "pass",
  "layer_kinds": [
    {
      "layer_kind_id": "dense",
      "baseline_prefill_pcc_vs_hf": 0.9991,
      "baseline_decode_pcc_vs_hf": 0.9990,
      "baseline_prefill_device_ms": 12.3,
      "baseline_decode_device_ms": 0.45,
      "baseline_output_layout": "DRAM_WIDTH_SHARDED",
      "notes": []
    }
  ]
}
```

## mesh_contract.json

This file documents the tensor and mesh contract that lets decoder layers chain without extra ops.

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_mesh_contract",
  "mesh": {
    "mesh_shape": [1, 8],
    "num_devices": 8,
    "fabric_config": "FABRIC_1D_RING",
    "topology": "Ring",
    "tp_axis": 1,
    "ep_axis": null,
    "sp_axis": null
  },
  "layout_contract": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "input_layout": "DRAM_WIDTH_SHARDED",
      "output_layout": "DRAM_WIDTH_SHARDED",
      "chainable_without_extra_ops": true,
      "residual_sharding": "hidden-dim-sharded",
      "notes": []
    }
  ],
  "weight_sharding": [
    {
      "layer_kind_id": "dense",
      "tensor_group": "wqkv",
      "placement": "column-parallel",
      "mesh_mapping": "PlacementShard(-1)",
      "notes": []
    }
  ],
  "kv_cache": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "paged": true,
      "local_kv_heads_per_device": 1,
      "cache_sharding": "local-kv-heads",
      "page_table_layout": "replicated",
      "notes": []
    }
  ],
  "collectives": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "location": "attention-output",
      "op": "reduce_scatter",
      "axis": 1,
      "dim": 3,
      "reason": "restore residual layout after row-parallel WO"
    }
  ]
}
```

## parallelization_plan.json

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_parallelization_plan",
  "selected_strategy": {
    "strategy_id": "1d-tp",
    "reason": "target hardware is 1x8 and dense decoder fits common TP pattern",
    "expected_bottleneck": "communication-bound WO reduction",
    "alternatives_rejected": [
      {
        "strategy_id": "replicated-activation-local-rmsnorm",
        "reason": "extra gather cost outweighed local RMSNorm benefit in measured decode"
      }
    ]
  },
  "required_steps": [
    {"id": "baseline-pass", "status": "pass", "evidence": "baseline_summary.json"},
    {"id": "mesh-contract", "status": "pass", "evidence": "mesh_contract.json"},
    {"id": "single-chip-comparison", "status": "pass", "evidence": "results/pcc_results.json"},
    {"id": "paged-kv-cache", "status": "pass", "evidence": "results/kv_cache_results.json"},
    {"id": "traced-decode", "status": "pass", "evidence": "results/pcc_results.json"},
    {"id": "watcher-clean", "status": "pass", "evidence": "watcher/watcher_summary.json"},
    {"id": "runtime-contract", "status": "pass", "evidence": "fallback_audit.md"},
    {"id": "performance-report", "status": "pass", "evidence": "performance_results.json"}
  ],
  "notes": []
}
```

For Galaxy MoE plans, add a required step with id `galaxy-moe-memory-fit` and evidence in `memory_capacity_plan.json` before accepting expert replication.

## memory_capacity_plan.json

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_memory_capacity_plan",
  "target_mesh": [4, 8],
  "max_sequence_length": 131072,
  "all_layers_loaded": true,
  "kv_cache_full_length": true,
  "estimated_per_device_dram_bytes": {
    "weights": 1000000000,
    "kv_cache": 2000000000,
    "activations": 500000000,
    "program_cache_and_slack": 500000000,
    "total": 4000000000,
    "available": 12000000000
  },
  "expert_replication": {
    "considered": true,
    "accepted": true,
    "replication_factor": 8,
    "fits_with_full_model_and_kv_cache": true,
    "evidence": []
  },
  "notes": []
}
```

## baseline_regression_results.json

Proves the chosen baseline's tests still pass with the multi-chip implementation enabled where applicable.

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_baseline_regression_results",
  "baseline_step_id": "optimized_decoder",
  "status": "pass",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "baseline_test_id": "test_decoder_decode_pcc",
      "multichip_enabled": true,
      "assertions_preserved": true,
      "status": "pass",
      "command_id": "pytest_dense_decode",
      "log": "pytest/dense_decode.log",
      "notes": []
    }
  ]
}
```

## results/pcc_results.json

Compare target multi-chip TTNN output to the single-chip TTNN baseline output, not directly to HF.

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_pcc_results",
  "comparison": "target-multichip-ttnn-vs-single-chip-ttnn",
  "threshold": 0.995,
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "pcc": 0.9998,
      "passed": true,
      "single_chip_command_id": "pytest_dense_decode_single_chip_capture",
      "multichip_command_id": "pytest_dense_decode",
      "trace_replay_used": true,
      "output_shape": [1, 1, 1, 4096],
      "output_layout": "DRAM_WIDTH_SHARDED",
      "input_seed": 12345,
      "notes": []
    }
  ],
  "component_results": []
}
```

## performance_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_performance_results",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "single_chip_device_ms": 0.45,
      "multichip_device_ms": 0.12,
      "speedup": 3.75,
      "parallel_efficiency": 0.46875,
      "target_num_devices": 8,
      "tracy_ops_csv": "tracy/dense/decode_ops.csv",
      "tt_perf_report_csv": "tracy/dense/decode_perf_report.csv",
      "tt_perf_report_txt": "tracy/dense/decode_perf_report.txt",
      "bottleneck": "communication-bound",
      "notes": []
    }
  ]
}
```

## sequence_limits.json

```json
{
  "schema_version": 1,
  "artifact_type": "multichip_decoder_sequence_limits",
  "layer_limits": [
    {
      "layer_kind_id": "dense",
      "baseline_max_prefill_seq_len": 32768,
      "multichip_tested_max_prefill_seq_len": 32768,
      "baseline_max_decode_context_len": 32768,
      "multichip_tested_max_decode_context_len": 32768,
      "reduced": false,
      "reduction_reason": null,
      "evidence": []
    }
  ]
}
```

## Other Results JSON

Use the same schema style as the earlier decoder phases for:

- `results/kv_cache_results.json`: cache shape, local head counts, page-table shape, tested positions, tested user slots, comparison method, and pass/fail.
- `results/determinism_results.json`: repeated-run count, equality method, pass/fail, and nondeterministic tensors.
- `results/stress_results.json`: duration seconds, iteration count, pass/fail, or reason skipped when the baseline did not require stress.
- `watcher/watcher_summary.json`: watcher env, command id, log paths, clean boolean, detected messages, false-positive justifications, and final status.
- `tt_perf_advice.json`: tt-perf-report advice, attempted changes, accepted or rejected status, and evidence.

## multichip_decoder.md

Use these headings:

```markdown
# Multichip Decoder

## Baseline
## Target Mesh
## Parallelization Strategy
## Mesh Contract
## RMSNorm And KV Cache
## PCC Results
## Determinism And Stress
## Watcher
## Trace And Runtime Fallback Audit
## Performance
## tt-perf-report Advice
## Sequence Limits
## Remaining Risks
```

Keep the report concise and evidence-led. Link every final claim to the JSON, log, or Tracy artifact that proves it.
