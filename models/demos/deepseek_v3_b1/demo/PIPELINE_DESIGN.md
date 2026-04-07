# Pod Pipeline Design

This document describes the pipeline orchestration abstractions in `pipeline.py`, used by the demo CLI and the LM head sampling tests (4-stage single-galaxy and 16-stage single-pod).

## Overview

The pipeline is a **ring** of stages: each process runs one stage. Data flows as tokens (host → device) or activations/tokens (device-to-device) between stages. Stage 0 is special: it has H2D/D2H sockets and performs embedding lookup; the last stage in the ring sends the sampled token back to stage 0 for D2H.

The design separates:

1. **What each stage does** — `StageKind` implementations (Embedding, LMHead, Passthrough).
2. **Which stage does what** — `PipelineConfiguration`: a mapping from stage ID to `StageKind`.
3. **Per-process execution** — `Pipeline`: owns one `StageKind`, runs a 4-phase lifecycle, and delegates token I/O to the underlying `PipelineBlock`.

The low-level `PipelineBlock` (from `micro_ops.pipeline_block`) is **not** defined here; it provides socket wiring, entry/exit nodes, and H2D/D2H for the first stage.

---

## PipelineBlock (external)

Defined in `models/demos/deepseek_v3_b1/micro_ops/pipeline_block/op.py`.

- **Role**: Per-stage building block. Creates socket interfaces (D2D, and optionally H2D/D2H on stage 0), entry/exit node coords, and optional embedding tensor for the first stage.
- **Key constructor options**: FIFO/page sizes, `embedding_tensor`, `entry_node_downstream`, `exit_node_upstream`, H2D/D2H socket sizes (stage 0 only).
- **Key methods**: `run()`, `terminate()`, `write_token()`, `read_output()`, `get_downstream_socket()`, `get_upstream_socket()`.

The pipeline topology (how many stages, which stage has H2D/D2H) comes from `ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)`; `PipelineBlock` is created per stage with parameters that depend on that stage’s kind.

---

## StageContext

A small dataclass passed into every `StageKind` method:

| Field           | Type           | Description                                      |
|----------------|----------------|--------------------------------------------------|
| `mesh_device`  | `ttnn.MeshDevice` | The mesh device for this process.               |
| `pipeline_config` | `list`       | Result of `generate_blitz_decode_pipeline(mesh_device)`. |
| `my_mesh_id`   | `int`         | This process’s stage index (0 .. num_stages-1). |

---

## StageKind

Abstract base class that defines what a stage does in three steps:

| Method | Purpose | Default |
|--------|--------|--------|
| `create_pipeline_block(ctx)` | Build and return the `PipelineBlock` for this stage (FIFO sizes, H2D/D2H, entry/exit cores, embedding, etc.). | **Abstract** — must implement. |
| `setup(ctx, pipeline_block)` | After the block exists: allocate tensors, weights, semaphores on device. | No-op. |
| `launch_compute(ctx, pipeline_block)` | After `pipeline_block.run()`: launch the stage’s compute kernel (e.g. LMHead op). | No-op. |

These map to **Phase 1**, **Phase 2**, and **Phase 4** of the `Pipeline` lifecycle (Phase 3 is `pipeline_block.run()`).

### EmbeddingStage

- **Stage 0 only.** H2D + embedding lookup; forwards activation downstream; loopback receives token from last stage and feeds D2H.
- **Constructor**: `embedding_tensor: torch.Tensor` (shape `[iterations, 1, 1, K]` or `[1, 1, 1, K]`).
- **create_pipeline_block**: Converts embedding to ttnn, puts on device (DRAM), returns `PipelineBlock` with token upstream/downstream, activation downstream, H2D/D2H sockets, and `embedding_tensor`.
- **setup / launch_compute**: No-op (embedding lookup runs inside the pipeline block kernel).

### LMHeadStage

- **Receives** activation from the previous stage, runs RMSNorm + matmul + sampling, **sends** token downstream.
- **Constructor**: `weights: LMHeadWeights`, `lm_head_fp32_dest_acc_en: bool = True`, `lm_head_persistent_mode: bool = True`.
- **create_pipeline_block**: Returns `PipelineBlock` with activation upstream, token downstream, and custom `entry_node_downstream` / `exit_node_upstream` so data goes to/from the LMHead input core and argmax output core.
- **setup**: Allocates mesh tensors (input, intermediate, gamma, weight matrix, scores, indices, output index), scratch buffer, semaphores (and optionally `persistent_next_iter_semaphore` when `lm_head_persistent_mode` is True). Gets sockets from `pipeline_block.get_downstream_socket()` / `get_upstream_socket()`. Stores everything in `_lmhead_state` for `launch_compute`.
- **launch_compute**: Calls `LMHeadSampling.op(...)` with the tensors and semaphores from `setup`, and `lm_head_persistent_mode` / `persistent_next_iter_semaphore` as configured.

### PassthroughStage

- **No compute.** Forwards payload (activation or token) from upstream to downstream.
- **Constructor**: `payload: PassthroughPayload` — either `ACTIVATION` or `TOKEN`.
- **create_pipeline_block**: Returns `PipelineBlock` with FIFO/page sizes set from the payload type (activation-sized vs token-sized).
- **setup / launch_compute**: No-op.

### PassthroughPayload

Enum: `ACTIVATION` | `TOKEN`. Used to choose FIFO and page sizes in `PassthroughStage`.

---

## LMHeadWeights

Dataclass of **torch** tensors supplied to the LMHead stage (conversion to ttnn and placement is done inside `LMHeadStage.setup()`):

| Field           | Shape        | Description                    |
|----------------|-------------|--------------------------------|
| `gamma`        | `[M, K]`    | RMSNorm weights.              |
| `weight_matrix`| `[K, n_total]` | LM head projection.        |
| `indices`      | `[1, n_total]` | Vocabulary indices for sampling. |

Constants: `M=1`, `K=7168`, `n_total=16160` (101 cores × 160 per core). Real weights (e.g. from cache) can be adapted into this interface with appropriate reshaping/typing.

---

## PipelineConfiguration

**Role**: Defines the full pipeline as a **mapping from stage ID to StageKind**. The mapping itself is the topology: number of stages = `len(stages)`, and the kind at each index defines embedding, LMHead, or passthrough.

- **Constructor**: `PipelineConfiguration(stages: dict[int, StageKind])`.
- **Properties**: `num_stages` (len of mapping), `__getitem__(stage_id)` to get the `StageKind` for a stage.
- **Method**: `build_pipeline(mesh_device)` — returns a `Pipeline` for **this process’s** stage by looking up `mesh_device.get_system_mesh_id()` in the mapping.

Configuration instances are not built by constructor in callers; they are created by **factory functions** (see below).

---

## Pipeline configuration factory functions

Standalone functions that return `PipelineConfiguration`:

| Function | Topology |
|----------|----------|
| `create_single_galaxy_pipeline_configuration(...)` | 4 stages: 0=Embedding, 1=LMHead, 2–3=Token passthrough. |
| `create_single_pod_pipeline_configuration(...)`     | 16 stages: 0=Embedding, 1–13=Activation passthrough, 14=LMHead, 15=Token passthrough. |
| `create_pipeline_configuration_from_num_procs(num_procs, ...)` | Dispatches: 4 → single-galaxy, 16 → single-pod; otherwise raises. |

Common arguments: `weight_provider` (`WeightProvider` protocol from `weight_provider.py`), `lm_head_fp32_dest_acc_en`, `lm_head_persistent_mode`, and optional layer-id overrides for scaleout. The demo CLI constructs the provider from `ModelPipeline` (`weights_mode`: cached tensorbin, HF safetensors + `prepare_*`, or synthetic). See [demo/README.md](README.md) for `--weights` / `--cache-path` / `--model-path`.

---

## Pipeline

**Role**: Per-process **orchestrator**. Holds one `StageKind` and one `PipelineBlock` (created during the lifecycle). Runs a strict 4-phase setup and exposes token I/O and barrier/terminate.

**Constructor**: `Pipeline(mesh_device, stage_kind)`. Builds `StageContext` and `pipeline_config` via `generate_blitz_decode_pipeline(mesh_device)`.

**Four-phase lifecycle**

| Phase | Method              | What it does |
|-------|---------------------|--------------|
| 1     | `configure_block()` | `stage_kind.create_pipeline_block(ctx)` → sets `_pipeline_block`. |
| 2     | `setup()`           | `stage_kind.setup(ctx, pipeline_block)` (tensors, weights, semaphores). |
| 3     | `start_pipeline()` | `pipeline_block.run()` (socket interfaces). |
| 4     | `start_compute()`  | `stage_kind.launch_compute(ctx, pipeline_block)` (e.g. LMHeadSampling.op). |

**Convenience**: `setup_and_run()` runs all four phases in order. Callers typically use this after `config.build_pipeline(mesh_device)`.

**I/O and control**

- `write_token(token_tensor)` — stage 0 only; delegates to `pipeline_block.write_token()`.
- `read_output(output_tensor)` — stage 0 only; delegates to `pipeline_block.read_output()`.
- `barrier()` — `ttnn.distributed_context_barrier()`.
- `terminate()` — calls `pipeline_block.terminate()` if the block exists (e.g. one-shot tests).

**Property**: `my_mesh_id` — this process’s stage index.

---

## Weight helpers

Used by tests and demo to produce inputs for the pipeline (no pipeline wiring):

- **`create_synthetic_weights(iterations)`**
  Deterministic one-hot embedding and weight matrix; returns `(embedding_tensor, lmhead_weights, expected_indices)`. Used for multi-iteration persistent runs and validation.

The 4-stage one-shot test (`test_lm_head_sampling_pipeline_block_4stage_single_galaxy`) uses a local helper `_create_random_weights_single_iteration(seed=5449)` in the test file to build random weights for a single token; it is not part of the pipeline module.

---

## Data flow (conceptual)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │  Stage 0 (EmbeddingStage)                               │
  Host ── H2D ──►   │  token → embedding lookup → activation ────────────────┼──► D2D ──► Stage 1
                    │  D2H ◄── token (loopback from last stage)               │
                    └─────────────────────────────────────────────────────────┘

  Stage 1..(L-1)    activation or token passthrough (PassthroughStage or LMHeadStage)
  Stage L (LMHead)  activation → LMHeadSampling.op → token
  Stage L+1 .. N-1  token passthrough back to Stage 0
```

- **Single-galaxy (4 stages)**: 0=Embed, 1=LMHead, 2–3=Token.
- **Single-pod (16 stages)**: 0=Embed, 1–13=Activation, 14=LMHead, 15=Token.

---

## Usage pattern (demo / tests)

1. Create weights: `embedding_tensor, lmhead_weights, _ = create_synthetic_weights(iterations)` (or a test-local helper for one-shot).
2. Build configuration: `config = create_single_galaxy_pipeline_configuration(...)` or `create_single_pod_pipeline_configuration(...)` or `create_pipeline_configuration_from_num_procs(num_procs, ...)`.
3. Build pipeline for this process: `pipeline = config.build_pipeline(mesh_device)`.
4. Run: `pipeline.setup_and_run()`.
5. Stage 0 only: loop `pipeline.write_token(...)` / `pipeline.read_output(...)`.
6. `pipeline.barrier()`.
7. Optionally `pipeline.terminate()` (e.g. one-shot tests).

All processes execute steps 2–6; only mesh_id 0 does the token write/read loop.
