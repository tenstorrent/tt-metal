<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Multi-Device (Tensor-Parallel) Guidance

This section is appended to your prompt because `spec.num_devices > 1`.
`spec.mesh_shape` (e.g. `[1, 4]`) is the target mesh. ALL on-device code
targets ONE `mesh_device` handle — never N separate single devices, and
never `ttnn.open_device(device_id=...)`.

## Phase scoping

- **reference worker**: ignore everything below. Goldens are computed
  UNSHARDED on host CPU — never shard or split the torch reference.
- **architecture worker**: COMPUTE the parallelism plan — do not eyeball it.
  Extract per-component facts from the HF config (param bytes, cadence:
  `per_token` for anything in the decode loop, `per_input` for run-once
  encoders, q/kv head counts, and `production_tokens` — the encoder
  sequence length at REAL input sizes, e.g. a full document page, not the
  unit-test shape), then:

  ```python
  from skills.orchestrator.lib.parallelism import plan_parallelism
  from skills.orchestrator.lib.device import device_info
  out = plan_parallelism(component_facts, spec["num_devices"],
                         device_info(spec["device"])["dram_bytes"])
  ```

  Record each component's `placement` + `rationale` in the inventory
  `notes`, resolve any `judgments` explicitly (state your reasoning), and
  summarize the plan in `ARCHITECTURE.md`.
- **ttnn / debug / optimization / real_weights / generation / perf
  workers**: everything below applies.

## Mesh open + fabric

Fabric MUST be configured before the mesh is opened:

```python
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*spec["mesh_shape"]))
...
ttnn.close_mesh_device(mesh_device)
```

In pytest, prefer the existing fixtures (they handle fabric setup/teardown):

```python
@pytest.mark.parametrize("mesh_device", [tuple(spec_mesh_shape)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_block(mesh_device, ...):
```

(`bh_1d_mesh_device` in the repo-root `conftest.py` is the Blackhole-generic
alternative; it opens the mesh as `(num_devices, 1)`.)

All CCL calls use `topology=ttnn.Topology.Linear` on a 1xN line. Validated
Blackhole multi-chip references:
`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/Sys_eng_smoke_tests/test_ccl_smoke_test_p300.py`
and `.../test_ccl_smoke_test_qb.py`.

## Weight sharding recipe

Distribute weights at load time via `mesh_mapper` on
`ttnn.from_torch` / `ttnn.as_tensor`:

| Weight | Strategy | mesh_mapper |
| :--- | :--- | :--- |
| QKV / gate / up projections | column-parallel: shard the OUTPUT feature dim | `ttnn.ShardTensorToMesh(mesh_device, dim=-1)` |
| o_proj / down projections | row-parallel: shard the INPUT feature dim; CCL after the matmul | `ttnn.ShardTensorToMesh(mesh_device, dim=-2)` |
| norms, biases of replicated ops, embedding (small models) | replicate | `ttnn.ReplicateTensorToMesh(mesh_device)` |
| LM head | shard vocab dim, concat logits on host (or all_gather) | `ttnn.ShardTensorToMesh(mesh_device, dim=-1)` |

The activation stays replicated into column-parallel matmuls; each device
produces its local slice of heads / hidden features; row-parallel matmuls
produce PARTIAL sums that must be combined with an all-reduce
(reduce_scatter + all_gather, or all_gather + local add). Pattern source:
`models/tt_transformers/tt/attention.py`, `mlp.py`, `lm_head.py`.

## CCL ops

Use the `TT_CCL` helper pattern from `models/tt_transformers/tt/ccl.py`
(`ttnn.experimental.all_gather_async`,
`ttnn.experimental.reduce_scatter_minimal_async`) — these need pre-created
global semaphores, `num_links`, and `topology=ttnn.Topology.Linear`. Simpler
synchronous `ttnn.all_gather(tensor, dim, topology=ttnn.Topology.Linear)` is
acceptable for first-pass correctness; switch to the async variants in the
optimization phase.

## Follow the recorded parallelism plan

The architecture phase computed and recorded a per-component placement
plan (inventory `notes` + `ARCHITECTURE.md`) via
`skills.orchestrator.lib.parallelism.plan_parallelism`. Implement what it
says; do not re-derive placement ad hoc. The two recurring outcomes:

- **`kv_replication: N`** (GQA with `n_kv_heads < num_devices`): DO NOT
  copy tt_transformers' divisibility assert. Replicate each KV head's
  weights and cache onto the N devices serving its Q-head group; Q heads
  still shard `num_devices`-ways, so SDPA stays chip-local.
- **`placement: replicate`** (run-once encoders): all weights
  `ReplicateTensorToMesh`; outputs stay replicated, so the handoff into
  column-parallel TP layers needs no CCL.

## PCC verification on a mesh

- Sharded output: `torch_out = ttnn.to_torch(tt_out,
  mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=<shard_dim>))`.
- Replicated output: compare ONE device's copy —
  `ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])`. Concatenating
  replicas inflates the tensor `num_devices`× and breaks shape checks.
- debug worker: on PCC mismatch, also compare PER-SHARD
  (`ttnn.get_device_tensors`) against the corresponding golden slice to
  localize which device or CCL stage diverges.

## Hangs

CCL deadlocks hang ALL chips, not one. Report `hang_detected=true` as
usual; the orchestrator's `tt-smi -r` resets the whole box (expected —
the device lock represents the entire mesh).
