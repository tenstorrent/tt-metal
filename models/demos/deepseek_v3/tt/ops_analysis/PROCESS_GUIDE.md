# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# How to Produce Exact TTNN Op Analyses and Isolated Unit Tests

This guide explains how to analyze a DeepSeek TT module and create isolated, per‑op unit tests that exactly match the original operator signatures: shapes, dtypes, memory configs, and program configs.

The goal is precision: tests must reproduce every non‑tensor argument the production forward path passes to each TTNN op.

## Compulsory Reading
Before starting analysis or writing tests for a module, read the following end‑to‑end (not skimming):
- The target module source under `models/demos/deepseek_v3/tt/<module>.py` and any base classes it extends (e.g., `decoder_block_base.py`, `rms_norm_base.py`).
- Related submodules called from the module’s forward path (e.g., `utils/composite_ops.py` for mesh_scatter, `tt/mla_1d.py` if invoked).
- `models/demos/deepseek_v3/utils/*` that the module references:
  - `config_dataclass.py` (OpConfig shapes/fields and program/compute config types)
  - `config_helpers.py` (helper math, memory/program config builders)
  - `run_config.py` (how model/weight/state configs merge; `FromWeightConfig`, `MeshDeviceStub` semantics)
  - `shared_state_addon.py` (shared state patterns, if used)
- `models/demos/deepseek_v3/demo/run_config_output.txt` — locate the exact section for your module; also read its early sections to understand mesh, norms, and memory conventions.
- Existing tests under `models/demos/deepseek_v3/tests/` to mirror fixtures, composers, and PCC checks.
- The TTNN op definitions and signatures in `ttnn/ttnn/operations/**` and relevant helpers in `ttnn/ttnn/core.py`.
- If shape semantics are unclear, check the HF reference model/config under `models/demos/deepseek_v3/reference/**` (e.g., `modeling_deepseek.py`, `configuration_deepseek.py`).

## 1) Read the Code and Run Config
- Read the target module under `models/demos/deepseek_v3/tt/...` thoroughly (including its `convert_weights`, `prefill_model_config`, `decode_model_config`, and `forward_*`).
- Read any utils it calls (e.g., `utils/config_helpers.py`, `utils/composite_ops.py`).
- Inspect `models/demos/deepseek_v3/demo/run_config_output.txt` for the module’s section to confirm the actual dtypes, memory configs, and program configs at runtime.

## 2) Derive Shapes from HF Config + Mesh
- Use `hf_config` (e.g., hidden_size, vocab_size, num_heads) and the mesh shape to derive per‑device shapes.
- Write comments in the test that show how each dimension is computed from `hf_config` and the mesh.

## 3) Reproduce the Construction Path from Source
For each operator and weight:
- Identify exactly how the source module constructs the tensors and configs. Then mirror that sequence in your test.
- Concretely, follow the same functions and order that the source module uses:
  - Weight construction: match data types, transposes/permutes, reshapes, and sharding mappers (e.g., `shard_tensor_to_mesh_mapper`, `ShardTensor2dMesh`, `ReplicateTensorToMesh`).
  - Memory configs: use the same helper(s) and parameters (e.g., `create_sharded_memory_config` / `create_sharded_memory_config_`, sharding strategy/orientation, core range selection, height/width, tile alignment).
  - Program configs: instantiate the same program config class and reproduce how its parameters are derived (either by calling the same generator function used by the module or by re‑implementing the same calculation in the test if the generator is not imported).
  - Compute kernel config: choose the same compute fidelity/flags (e.g., LoFi vs HiFi) as the module.
- Do not “pick something reasonable” — match the module’s construction path and arguments exactly.

## 4) Build Isolated Unit Tests (No Model Converters)
- Do not import or call `convert_weights` or module config builders in the test. The test must be isolated and locally reconstruct the semantics.
- Create random CPU tensors sized per the derived shapes.
- Create TTNN tensors with the same dtype, layout, and mesh mapping as the module.
- Construct memory configs and program configs in the test exactly as the module does (same helper calls, same arguments).
- Build a local cfg dict that mirrors how forward expands kwargs, and call ops via `**` expansion.
  - Example: `run_cfg = {"linear": LinearConfig(...), "input_memory_config": <memcfg>, ...}` then `ttnn.linear(x, **run_cfg["linear"])`.
  - Prefer using the same OpConfig dataclasses from `utils/config_dataclass.py` to ensure field names/types match.
- Call the TTNN op with the identical signature: `ttnn.op(..., **run_cfg[op_name])` or explicitly include `memory_config`, `program_config`, `compute_kernel_config` if not wrapped in an OpConfig.
- Convert per‑device outputs back to Torch with the correct composer matching the sharding topology.

### 4a) Embed and Validate the Relevant Run Config Snippet
- For each unit test, embed a triple‑quoted string of the expected pretty‑printed config snippet for the op(s) under test, taken from `run_config_output.txt` and/or constructed from the same logic.
- Generate the pretty string for your test cfg using the same utility as the demo logger:
  - `from models.demos.deepseek_v3.utils.run_config import _convert_run_config_to_pretty_print`
  - `pretty = _convert_run_config_to_pretty_print(run_cfg_subset)` (e.g., `run_cfg["linear"]` or `{"linear": run_cfg["linear"]}`)
- Compare the generated string to your embedded expected snippet. Prefer exact string equality; if needed, normalize whitespace.
- This ensures the test’s cfg faithfully matches the runtime configuration format and content.

## 5) Composite Ops: Test Their TTNN Primitives
If the module uses composite helpers (e.g., `mesh_scatter`), test the underlying TTNN primitives separately with the same argument patterns and topology:
- `ttnn.to_memory_config` for interleaved ↔ sharded transitions used by the composite.
- `ttnn.point_to_point` with the same sender/receiver traversal and topology.
- `ttnn.copy` if the composite uses it to restore into the original buffer.

## 6) Compare to Torch/CPU and Assert PCC
- Compute a Torch baseline directly (e.g., `x @ W`, `layernorm`, etc.).
- Gather TTNN results into a Torch tensor using the correct composer (mesh or mesh2d, correct dims).
- Check PCC via `models.utility_functions.comp_pcc` (use the same threshold the project uses for that op).
- Also assert expected memory configs for outputs to ensure layout fidelity.

## 7) Keep Results Organized
- Put each module’s ops analysis in `tt/ops_analysis/<ModuleName>_ops.md`.
- Put each module’s op tests in `models/demos/deepseek_v3/tests/test_<module>_ops.py`.
- Maintain `ops_analysis/TODO.md` and `ops_analysis/TEST_TODO.md` as single sources of truth for progress.

## 8) Review Against run_config_output
Before finalizing:
- Cross‑check that dtypes, memory layouts, and program configs match the run‑config printout.
- Ensure any collectives or point‑to‑point patterns used in forward are represented by dedicated primitive‑level tests.

Following this process will produce precise, reproducible operator analyses and tests that mirror the module’s forward behavior exactly, without relying on converters or module configs.
