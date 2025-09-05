# DeepSeek V3 — Module/Op Result Capture and Replay Plan

This document summarizes the instrumentation we added, how to use it, and what remains for the hardware team to validate. The work is opt‑in and designed to run with or without access to TTNN runtimes.

## Goals

- Module results CSV: capture one line per pytest invocation (metadata, durations, pass/fail, PCC metrics, error text) to enable tracking and dashboards.
- Op tracing: capture every TTNN op call performed by DeepSeek modules during tests, including input shapes, memory/program configs, and outputs — enabling per‑op unit test generation.
- Serializer: robust JSON serializer/deserializer for DeepSeek config dataclasses and TTNN configs, with placeholders for non‑serializable objects (MeshDevice, Tensor).
- Op replay: helper to reconstruct and execute a single captured op call as a self‑contained unit test (random inputs, real shapes/configs) on hardware.

## What’s Implemented

1) Module Results CSV (opt‑in)

- Plugin: `models/demos/deepseek_v3/utils/results_csv_plugin.py` (loaded by conftest).
- Enable:
  - CLI: `--results-csv /path/to/results.csv`
  - Env: `TT_TEST_RESULTS_CSV=/path/to/results.csv`
- Captured columns include test id, module name, mode/seq_len/batch_size, hf_max_seq_len, mesh info, metrics JSON (PCC/allclose/ulp), outcome, wallclock duration, error, run/session metadata.
- Disabled by default — zero overhead when not enabled.

2) TTNN Op Tracing (opt‑in)

- Plugin: `models/demos/deepseek_v3/utils/op_capture_plugin.py` (loaded by conftest).
- Enable:
  - CLI: `--ops-jsonl /path/to/op_calls.jsonl`
  - Env: `TT_OP_RESULTS_JSONL=/path/to/op_calls.jsonl`
- Uses an import hook to wrap `ttnn` callables, capturing inputs/kwargs/output for every op call.
- Writes JSON Lines with linkage (run_id, test_nodeid, module, op_index), op info, shapes/configs, and errors if any.
- Disabled by default.

3) Config Serializer

- Module: `models/demos/deepseek_v3/utils/serialize_configs.py`.
- `to_jsonable(obj)`: serializes DeepSeek dataclasses, TTNN MemoryConfig/ProgramConfig, enums, paths, and Tensor/Device placeholders.
- `from_jsonable(obj, mesh_device=None)`: reconstructs objects when possible; keeps placeholders for TensorRef; injects provided MeshDevice instead of deserializing it.
- Unit tests (no TTNN required): `models/demos/deepseek_v3/tests/test_serialize_configs.py`.

4) Op Replay Helper

- Module: `models/demos/deepseek_v3/utils/op_replay.py`.
- Programmatic: `replay_op_record(record_dict, mesh_device, rng_seed=0)`.
- CLI: `python -m models.demos.deepseek_v3.utils.op_replay --jsonl op_calls.jsonl --index 0`.
- Reconstructs inputs (random tensors with recorded dtype/layout/memory_config and sharding), configs (via serializer), and invokes the captured `ttnn` op.

## What’s Left / Validation Plan (hardware team)

1) Validate Serializer on Runtime Types

- Run: `pytest models/demos/deepseek_v3/tests/test_serialize_configs.py`.
- Then, with TTNN available, add a small sanity to build a real `ttnn.MemoryConfig` and a known `ProgramConfig` class and ensure `from_jsonable` reconstructs usable objects (optional add‑on test).

2) Validate Op Tracing on Hardware

- Enable op capture and run one or more module tests:
  - `pytest models/demos/deepseek_v3/tests/test_mlp_1d.py --ops-jsonl /tmp/op_calls.jsonl -q`.
  - Optionally capture module results too: `--results-csv /tmp/results.csv`.
- Inspect `/tmp/op_calls.jsonl` to ensure many records are emitted.

3) Replay a Handful of Captured Ops

- `python -m models.demos.deepseek_v3.utils.op_replay --jsonl /tmp/op_calls.jsonl --index 0`.
- Repeat with a few indices representing different ops (e.g., linear, mul, reduce_scatter_async, to_memory_config, embedding).
- Expected: replay should run and print the output tensor shape; failures indicate missing reconstruction detail — please capture logs and the specific records.

4) Generate and Run Op Unit Tests (next phase)

- Using the serializer and replay helper, auto‑generate parametric pytest cases grouped by (op_name, shape/config signature) and run them on hardware.
- We can add a `tests/ttnn/op_repros/` suite that ingests JSONL and emits tests; keep it opt‑in.

## Notes / Design Choices

- Plugins are opt‑in to avoid overhead and clutter during normal dev workflows.
- The import‑hook tracer ensures broad coverage of new TTNN ops without touching module code.
- We serialize MeshDevice as a placeholder and ask the user to provide a real device at replay time to avoid serializing hardware handles.
- Tensor data is never serialized — only shapes/dtypes/memory configs; replay uses random values.

## Caveats / Corrections

- ProgramConfig reconstruction is best‑effort: constructors may expect specific typed fields (e.g., `CoreCoord`). The serializer tries kwargs and then attribute assignment fallback. If an op requires a strongly‑typed program config at call time, replay may need adjustments to map the field dicts to constructor types.
- Dtype strings may vary depending on TTNN stringification; the replay helper uses substring matching to map to TTNN dtypes. If you observe mismatches, please share examples so we can refine mappings.
- MemoryConfig shard mapping: the helper chooses `ShardTensor2dMesh` with recorded `dims`, or replicates if none. Module code sometimes uses different mappers; adjust if a specific op expects an alternative.
- The current unit tests validate serializer logic without TTNN; add hardware‑backed tests to validate reconstructability in your environment.

Please file issues/PRs for extension points (new TTNN configs/types, mapper heuristics) as you discover them.

