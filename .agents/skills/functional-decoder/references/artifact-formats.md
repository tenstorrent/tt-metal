# Functional Decoder Artifact Formats

All paths in JSON are relative to `models/autoports/<model>/doc/functional_decoder/` unless stated otherwise. Use strict JSON, UTF-8, two-space indentation, and `"schema_version": 1`. Use `null` for unavailable values and empty arrays for no entries. Timestamps are UTC ISO-8601 strings ending in `Z`.

## Directory Contract

The final golden functional-decoder evidence lives directly under:

```text
models/autoports/<model>/doc/functional_decoder/
```

Do not create per-debug-run or per-attempt directories in this doc tree. Later bringup phases should use sibling directories such as:

```text
models/autoports/<model>/doc/optimized_decoder/
models/autoports/<model>/doc/multichip_decoder/
```

The functional implementation itself must live at:

```text
models/autoports/<model>/tt/functional_decoder.py
```

## Status Values

Use these exact status strings:

- `pass`
- `fail`
- `blocked`
- `skipped`

Use these exact mode strings:

- `prefill`
- `decode`

Use lowercase hyphen-case for `layer_kind_id`.

Final `pass` artifacts must satisfy the sequence-length contract. If any representative layer kind uses reduced prefill or decode length, `sequence_limits.json` must prove a measured capacity blocker or cite an explicit user-approved reduced scope. Reduced length for convenience, "tractability", profiler/watcher cost, runtime, or small-proof speed is not valid for `manifest.status == "pass"`.

## manifest.json

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_manifest",
  "step_id": "functional_decoder",
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
  "started_utc": "2026-05-26T14:00:00Z",
  "ended_utc": "2026-05-26T14:30:00Z",
  "paths": {
    "report": "functional_decoder.md",
    "commands": "commands.sh",
    "model_facts": "model_facts.json",
    "layer_kinds": "layer_kinds.json",
    "implementation_contract": "implementation_contract.json",
    "weight_stats": "weight_stats.json",
    "sequence_limits": "sequence_limits.json",
    "fallback_audit": "fallback_audit.md",
    "pcc_results": "results/pcc_results.json",
    "kv_cache_results": "results/kv_cache_results.json",
    "determinism_results": "results/determinism_results.json",
    "stress_results": "results/stress_results.json",
    "watcher_summary": "watcher/watcher_summary.json"
  },
  "commands": [
    {
      "id": "pytest_dense_prefill",
      "purpose": "prefill PCC",
      "command": "pytest ...",
      "exit_code": 0,
      "log": "pytest/dense_prefill.log"
    }
  ]
}
```

## commands.sh

Plain executable shell transcript. Each command block must start with a command id comment that matches `manifest.json`:

```bash
# command_id: pytest_dense_prefill
# purpose: prefill PCC
pytest models/autoports/example/tests/test_layer.py -k "dense and prefill" -vv 2>&1 | tee pytest/dense_prefill.log
```

If full prefill/decode length is reduced, include the capacity probe command(s) here. These are final evidence commands, not debug attempts, and their ids must appear in `sequence_limits.json`.

## model_facts.json

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_model_facts",
  "model_id": "org/example-model",
  "hf_config": {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 32768,
    "rope_theta": 10000.0,
    "rope_scaling": null,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "attention_bias": false,
    "layer_types": ["full_attention"]
  },
  "hf_classes": {
    "decoder_layer": "ExampleDecoderLayer",
    "attention": "ExampleAttention",
    "mlp": "ExampleMLP",
    "moe": null,
    "cache": "DynamicCache"
  },
  "source_files": [
    "transformers/models/example/modeling_example.py"
  ],
  "notes": []
}
```

## layer_kinds.json

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_layer_kinds",
  "layer_kinds": [
    {
      "layer_kind_id": "dense",
      "representative_layer_index": 0,
      "reason_unique": "dense MLP decoder path",
      "hf_layer_type": "full_attention",
      "features": ["paged-attention", "gqa", "rope", "dense-mlp"],
      "implementation_files": [
        "models/autoports/example/tt/functional_decoder.py"
      ],
      "pytest_ids": [
        "test_example_decoder_layer[dense-prefill]",
        "test_example_decoder_layer[dense-decode]"
      ],
      "expected_max_prefill_seq_len": 32768,
      "expected_max_decode_context_len": 32768
    }
  ]
}
```

## implementation_contract.json

Record the concrete Python interface that the later optimized, multichip, and full-model ports will consume. `from_state_dict` is the general weight-loading API, not a synthetic-only test hook.

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_implementation_contract",
  "implementation": {
    "path": "models/autoports/example/tt/functional_decoder.py",
    "class_name": "FunctionalDecoder",
    "base_class": "models.common.lightweightmodule.LightweightModule",
    "subclasses_lightweight_module": true
  },
  "methods": {
    "from_state_dict": {
      "kind": "classmethod",
      "signature": "from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs)",
      "loads_real_weights": true,
      "loads_synthetic_weights": true,
      "real_weights_are_canonical": true,
      "allowed_setup_work": [
        "torch state-dict tensor reads",
        "weight reshape or transpose",
        "dtype conversion",
        "ttnn.as_tensor or equivalent device loading",
        "KV-cache and static tensor allocation"
      ],
      "notes": []
    },
    "prefill_forward": {
      "signature": "prefill_forward(self, hidden_states, *, page_table, position_ids, **kwargs)",
      "runtime_torch_allowed": false,
      "runtime_from_torch_allowed": false,
      "runtime_to_torch_allowed": false,
      "input_layout": "model-specific",
      "output_layout": "model-specific",
      "notes": []
    },
    "decode_forward": {
      "signature": "decode_forward(self, hidden_states, *, page_table, position_ids, **kwargs)",
      "runtime_torch_allowed": false,
      "runtime_from_torch_allowed": false,
      "runtime_to_torch_allowed": false,
      "trace_replay_supported": true,
      "input_layout": "model-specific",
      "output_layout": "model-specific",
      "notes": []
    }
  },
  "state_dict_contract": {
    "real_weight_supported": true,
    "synthetic_weight_supported": true,
    "real_weight_evidence": {
      "status": "pass",
      "command_id": "pytest_dense_real_weight_load",
      "reason_not_run": null
    },
    "synthetic_weight_evidence": {
      "status": "pass",
      "command_id": "pytest_dense_prefill"
    },
    "required_keys_by_layer_kind": [
      {
        "layer_kind_id": "dense",
        "keys": [
          "model.layers.0.self_attn.q_proj.weight"
        ]
      }
    ],
    "key_transforms": [],
    "notes": []
  }
}
```

Set `real_weight_evidence.status` to `pass` when a real state dict was available and the load path was exercised. Use `skipped` with a concrete `reason_not_run` only when no real weights were available for the final artifact. The synthetic path must pass for CI-oriented final artifacts.

## weight_stats.json

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_weight_stats",
  "source": {
    "weight_access": "partial|full",
    "hf_revision": null,
    "notes": []
  },
  "tensors": [
    {
      "layer_kind_id": "dense",
      "layer_index": 0,
      "name": "model.layers.0.self_attn.q_proj.weight",
      "shape": [4096, 4096],
      "dtype": "torch.bfloat16",
      "mean": 0.0001,
      "std": 0.0123,
      "synthetic_seed": 12345
    }
  ]
}
```

## sequence_limits.json

Every layer kind must either test the reference maximums or prove why the reserved hardware cannot. If `reduced` is `true`, `reduction_kind` must be `"capacity"` or `"explicit-user-scope"`. For `"capacity"`, include nonempty `capacity_evidence` with command ids, logs, attempted lengths, failure signatures or byte calculations, available device memory, and the largest feasible tested lengths. For `"explicit-user-scope"`, include the user instruction that approved the reduced scope. Do not use "tractability" or runtime cost as a capacity reason.

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_sequence_limits",
  "layer_limits": [
    {
      "layer_kind_id": "dense",
      "reference_max_prefill_seq_len": 32768,
      "tested_max_prefill_seq_len": 32768,
      "reference_max_decode_context_len": 32768,
      "tested_max_decode_context_len": 32768,
      "reduced": false,
      "reduction_kind": null,
      "reduction_reason": null,
      "capacity_evidence": {
        "available_device_dram_bytes": null,
        "estimated_required_kv_cache_bytes": null,
        "attempted_prefill_command_id": null,
        "attempted_prefill_log": null,
        "attempted_decode_command_id": null,
        "attempted_decode_log": null,
        "failure_signature": null,
        "largest_feasible_prefill_seq_len": 32768,
        "largest_feasible_decode_context_len": 32768,
        "notes": []
      },
      "explicit_user_scope": null,
      "evidence": [
        "pytest/dense_prefill.log",
        "pytest/dense_decode.log"
      ]
    }
  ]
}
```

## results/pcc_results.json

For MoE models, entries in `results` are full decoder outputs after the TTNN gate/router selects experts and the TTNN expert path computes, weights, and reduces those selected experts. Gate-only and expert-only PCCs are optional diagnostics; put them in `component_results` and do not treat them as a substitute for the full decoder result.

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_pcc_results",
  "threshold": 0.995,
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "prefill",
      "pcc": 0.9991,
      "passed": true,
      "output_shape": [1, 1, 128, 4096],
      "dtype": "bfloat16",
      "input_seed": 20260526,
      "synthetic_weight_seed": 12345,
      "trace_replay_used": false,
      "command_id": "pytest_dense_prefill",
      "log": "pytest/dense_prefill.log"
    }
  ],
  "component_results": [
    {
      "layer_kind_id": "moe",
      "mode": "prefill",
      "component": "gate",
      "pcc": 0.999,
      "passed": true,
      "diagnostic_only": true,
      "command_id": "pytest_moe_prefill_components",
      "log": "pytest/moe_prefill_components.log"
    }
  ]
}
```

## results/kv_cache_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_kv_cache_results",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "passed": true,
      "cache_shape": [1024, 8, 32, 128],
      "page_table_shape": [32, 1024],
      "page_table_seed": 777,
      "tested_positions": [0, 17, 32767],
      "tested_user_slots": [0, 3, 31],
      "comparison_method": "TT cache update compared against HF cache-equivalent K/V for selected pages",
      "notes": []
    }
  ]
}
```

## results/determinism_results.json

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_determinism_results",
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

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_stress_results",
  "results": [
    {
      "layer_kind_id": "dense",
      "mode": "decode",
      "status": "skipped",
      "duration_seconds": 0,
      "iteration_count": 0,
      "reason": "stress mode not requested for this proof",
      "command_id": null
    }
  ]
}
```

## watcher/watcher_summary.json

```json
{
  "schema_version": 1,
  "artifact_type": "functional_decoder_watcher_summary",
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

## fallback_audit.md

Use these exact headings:

```markdown
# Fallback Audit

## Scope

## Runtime Boundary

## Search Commands

## Findings

## Allowed Host Interactions

## Disallowed Interactions Fixed

## Residual Risk
```

## functional_decoder.md

Use these exact headings:

```markdown
# Functional Decoder

## Result

## Layer Kinds

## Correctness

## KV Cache

## Trace And Performance

## Watcher

## Determinism And Stress

## Sequence Limits

## Fallback Audit

## Commands

## Residual Risk
```
