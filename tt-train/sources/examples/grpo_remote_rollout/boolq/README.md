# BoolQ training example

GRPO fine-tuning of `meta-llama/Llama-3.2-1B-Instruct` on
`google/boolq`, with a Yes/No correctness reward. The training loop
itself is the generic `GRPOTrainer` from `ttml.trainers` (see
[`tt-train/docs/GRPO_TRAINER.md`](../../../../docs/GRPO_TRAINER.md));
this directory only adds the deployment-level wiring needed to run it
across two MPI ranks.

---

## Two-rank architecture

Generation is the slow part of GRPO and benefits from running inside
a captured ttnn trace. Training-side `ttml.Llama` and inference-side
`tt-transformers.Transformer` are different model implementations
with different mesh-shape constraints, so this example splits them
across two MPI ranks: the trainer keeps a free policy mesh, the
worker keeps a captured decode trace, and weights are pushed from one
to the other every step.

```text
                 mpirun (tt-run, world_size = 2)
                ─────────────────────────────────

  rank 0 (TTML)                    rank 1 (TTT)
  ───────────────                   ──────────────
  ttml.Llama policy                 Nx tt-transformers Transformer
  GRPOTrainer + optimizer           Nx TttGenerationWorker
  mesh: [1, N] (DDP)                mesh: [1, N] -> Nx [1, 1] submesh
       │                                ▲
       │  MPIRolloutClient    OP_GENERATE / OP_TRANSFER / OP_SHUTDOWN
       └──────────────► MPI ───────────► MPIRolloutServer
       │                                │
       └────── WeightBridge socket ─────┘
```

`N` is the per-rank mesh width, picked by `--split` on the Python
entrypoint: `2` for `2-2` (4 chips total) or `4` for `4-4` (8 chips
total). `--split` must match the `device_topology` in the
`mgd.textproto` you point tt-run at. See [How to run](#how-to-run).

`GRPOTrainer` is unaware of the rank split. It calls
`completer.generate(...)`; `LlamaCompleterRemoteRollout` hides the cross-rank
RPC inside that call. The [trainer doc](../../../../docs/GRPO_TRAINER.md)
covers the model- and rank-agnostic API; everything below is
specific to this two-rank deployment.

---



## Components


| Class                         | Side | Role                                                                                                                                                                                                                           |
| ----------------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `LlamaCompleterRemoteRollout` | TTML | Concrete `GRPOCompleter`. Owns the ttml policy. Routes `generate(...)` and `push_weights()` to the peer rank via `inference_client`.                                                                                           |
| `MPIRolloutClient`            | TTML | MPI client + `WeightBridge` owner. Constructed before the completer; its constructor blocks until the peer's server is up.                                                                                                     |
| `MPIRolloutServer`            | TTT  | Dispatches `OP_GENERATE` / `OP_TRANSFER` / `OP_SHUTDOWN` to user-supplied callbacks. Blocks in `serve_forever()` until shutdown.                                                                                               |
| `TttGenerationWorker`         | TTT  | Hosts the `tt-transformers.Transformer` and a captured decode trace. Exposes `generate` and `update_weights` callbacks.                                                                                                        |
| `WeightBridge`                | both | Replicated-tensor transport (ABC). `HostWeightBridge` moves each weight to host via MPI and re-uploads it to each receiver submesh. Wire-format spec: [`LLAMA_WEIGHT_TRANSFER.md`](../../../../docs/LLAMA_WEIGHT_TRANSFER.md). |
| `WeightSyncCallback`          | TTML | `TrainerCallback` that calls `completer.push_weights()` every `every` optimizer steps. Opt-in.                                                                                                                                 |


---



## LlamaCompleterRemoteRollout

```python
from utils.llama_grpo_completer import LlamaCompleterRemoteRollout, LlamaCompletionCtx
```

Llama-specific implementation of `GRPOCompleter`. Loads the ttml
policy from a HuggingFace ID or local safetensors directory, manages
the KV cache, and dispatches generation requests over MPI to the TTT
rank.

```python
completer = LlamaCompleterRemoteRollout(
    ctx=LlamaCompletionCtx(
        max_tokens_to_complete=256,
        temperature=0.7,
        completions_per_prompt=8,
    ),
    transformer_config=transformer_config,   # TransformerConfig
    mesh_device=mesh_device,                 # opened ttnn.MeshDevice
    model_source="meta-llama/Llama-3.2-1B-Instruct",
    inference_client=client,                 # MPIRolloutClient
    enable_ddp=True,
)
```


| Parameter            | Type                 | Description                                                                                                           |
| -------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `ctx`                | `LlamaCompletionCtx` | Generation parameters (max tokens, temperature, completions per prompt).                                              |
| `transformer_config` | `TransformerConfig`  | Model architecture config (parsed from the YAML training config).                                                     |
| `mesh_device`        | `ttnn.MeshDevice`    | Already-opened TTML mesh. The caller owns its lifetime; the completer does not open or close it.                      |
| `model_source`       | `str`                | HuggingFace model ID or path to a local directory containing `model.safetensors`.                                     |
| `inference_client`   | `MPIRolloutClient`   | RPC client to the TTT rank. The completer routes `generate` and `push_weights` calls through this.                    |
| `enable_ddp`         | `bool`               | Enable distributed data parallelism across the TTML mesh. Must agree with the `enable_ddp` in the YAML device config. |




### Methods used by the user


| Method           | Description                                                                                                                               |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `push_weights()` | Export the current ttml policy as an HF-keyed dict and push it to the TTT rank. Used once at startup, before the first `trainer.train()`. |


For per-step pushes, register `WeightSyncCallback(completer, every=N)`
as a trainer callback; it calls `push_weights()` after every `N`
optimizer steps.

---



## TTML rank skeleton (rank 0)

```python
import os
from datasets import load_dataset
from utils.mpi_rollout import MPIRolloutClient
from utils.llama_grpo_completer import (
    LlamaCompletionCtx, LlamaCompleterRemoteRollout, WeightSyncCallback,
)
from ttml.trainers import GRPOTrainer, get_grpo_config

TTML_RANK, TTT_RANK = 0, 1
mesh_device = ...                         # opened from YAML config

# Bridge handshake: blocks until rank 1 also constructs its server.
client = MPIRolloutClient(peer_rank=TTT_RANK, device=mesh_device)

dataset = load_dataset("google/boolq", split="train").map(format_example)

completer = LlamaCompleterRemoteRollout(
    ctx=LlamaCompletionCtx(
        max_tokens_to_complete=256,
        temperature=0.7,
        completions_per_prompt=8,
    ),
    transformer_config=transformer_config,    # parsed from YAML
    mesh_device=mesh_device,
    model_source="meta-llama/Llama-3.2-1B-Instruct",
    inference_client=client,
    enable_ddp=True,
)

# One-off: replace the worker's dummy boot weights with real instruct
# weights before the first generate call.
completer.push_weights()

trainer = GRPOTrainer(
    completer=completer,
    dataset=dataset,
    config=get_grpo_config(yaml_dict, output_dir=output_dir),
    reward_func=my_reward,
    optimizer_dict={"type": "MorehAdamW", "lr": 5.0e-6},
    callbacks=[WeightSyncCallback(completer, every=1)],   # push policy every step
    model_source="meta-llama/Llama-3.2-1B-Instruct",
)
try:
    trainer.train()
finally:
    client.shutdown()                     # must run before the TTML mesh closes
```

---



## TTT rank skeleton (rank 1)

```python
import ttnn
from utils.mpi_rollout import MPIRolloutServer
from utils.ttt_generation_worker import TttGenerationWorker
from utils.weight_bridge import HostWeightBridge
from utils.llama_ttt_presets import (
    bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad,
)

ttnn.init_distributed_context()
parent_mesh = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 4),
    offset=ttnn.MeshCoordinate(0, 0),
)
submeshes = parent_mesh.create_submeshes(ttnn.MeshShape(1, 1))   # four [1, 1] submeshes

stop_token_ids, pad_token_id = llama_stop_and_pad("meta-llama/Llama-3.2-1B-Instruct")

workers = [
    TttGenerationWorker(
        mesh_device=submesh,
        model_source="meta-llama/Llama-3.2-1B-Instruct",
        max_batch_size=32,
        max_seq_len=2048,
        instruct=True,
        optimizations=bf16_attn_bfp8_mlp_optimizations,
        stop_token_ids=stop_token_ids,
        pad_token_id=pad_token_id,
        temperature=0.7, top_k=0, top_p=1.0, seed=None,
    )
    for submesh in submeshes
]

# The bridge replicates each transferred policy onto every submesh.
bridge = HostWeightBridge.init_receiver(mesh=parent_mesh, peer_rank=0, submeshes=submeshes)
server = MPIRolloutServer(
    peer_rank=0,
    bridge=bridge,
    generate_fn=workers[0].generate,          # generation served by submesh 0
    on_weights_received=lambda per_submesh: [
        w.update_weights(d) for w, d in zip(workers, per_submesh)
    ],
)
server.serve_forever()                    # blocks until rank 0 sends OP_SHUTDOWN
```

---



## Single-file dispatch

Both ranks live in the same Python file
([`boolq_training_example.py`](boolq_training_example.py)) and are
dispatched on the MPI rank set by `mpirun` / `tt-run`:

```python
if int(os.environ["OMPI_COMM_WORLD_RANK"]) == 0:
    ttml_main()
else:
    ttt_main()
```

---



## Configuration

Everything runtime-tunable lives in a single training YAML,
[`grpo_boolq_llama_1b_remote_rollout.yaml`](../../../../configs/training_configs/grpo_boolq_llama_1b_remote_rollout.yaml).
Both ranks load it: the TTML rank consumes `training_config` /
`device_config` (standard `GRPOTrainer` inputs, described in the
[trainer doc](../../../../docs/GRPO_TRAINER.md)); the TTT rank
consumes `remote_rollout_config` plus the same `grpo_config.temperature`
so both sides bake in matching sampling.

### `remote_rollout_config` — TTT rollout-rank knobs

Consumed only by the TTT rank (`_ttt_main` in
[`boolq_training_example.py`](boolq_training_example.py)) when it
opens the rollout mesh and constructs `TttGenerationWorker`.

| Field            | Type        | Default      | Description                                                                                                                                                                                                                    |
| ---------------- | ----------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mesh_shape`     | `list[int]` | `[1, 2]`     | Shape of the TTT parent mesh `[rows, cols]`. The worker splits it into `rows * cols` `[1, 1]` submeshes; each hosts one `tt-transformers.Transformer` copy and generation runs data-parallel across them.                     |
| `max_batch_size` | `int`       | `32`         | Per-submesh concurrent decode capacity. Global generation batch = `max_batch_size * num_submeshes`. Sets the paged KV-cache block budget on each submesh — over-sizing costs L1 / DRAM; under-sizing rejects large requests. |
| `max_seq_len`    | `int`       | `2048`       | Max prompt + completion length. Sets the KV-cache page-table depth: `max_num_blocks_per_user = ceil(max_seq_len / block_size)`. Must fit the longest prompt (post chat-template) plus `grpo_config.max_completion_length`.    |

### `training_config` extras

Set alongside the standard `GRPOTrainer` blocks under `training_config`:

| Field               | Type   | Default                                | Description                                                                                                              |
| ------------------- | ------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `model_id`          | `str`  | `"meta-llama/Llama-3.2-1B-Instruct"`   | HF repo path. TTML rank uses it for tokenizer + `LlamaCompleterRemoteRollout`; TTT rank uses it for `llama_stop_and_pad` and `TttGenerationWorker.model_source`. |
| `weight_sync_every` | `int`  | `1`                                    | Cadence (in optimizer steps) at which `WeightSyncCallback` pushes fresh policy weights from the TTML rank to the TTT worker. |

The rest of `training_config` (GRPO / optimizer knobs) and `device_config`
(trainer-mesh shape / DDP) are the standard `GRPOTrainer` blocks —
see [`tt-train/docs/GRPO_TRAINER.md`](../../../../docs/GRPO_TRAINER.md).

---



## How to run

### 1. Hardware

The default `configurations/split_2_2` targets a Blackhole loudbox or
quietbox populated with P100 / P150 cards — 4 single-chip PCIe devices
total, 2 pinned to each MPI rank. Other cards (N300, P300) are
supported but require the config edits below because those cards
expose 2 chips per PCIe device.

Minimum: at least 1 chip per side (2 chips total). The launcher opens
a `[1, N]` mesh on each rank; scaling to more chips per rank requires
matching bumps in both meshes (see [1.1.3](#113-different-split-eg-4-4)).

#### 1.1. Adapting configurations for other hardware

The two files that encode host-specific assumptions are
`configurations/split_2_2/rank_bindings.yaml` (PCIe-device pinning per
rank) and `configurations/split_2_2/mgd.textproto` (arch + mesh shape).

##### 1.1.1. Wormhole vs Blackhole — change `arch` in `mgd.textproto`

Edit the `mesh_descriptors { ... arch: ... }` field:

- `arch: BLACKHOLE` — P100 / P150 (default).
- `arch: WORMHOLE_B0` — N100 / N150 / N300.

##### 1.1.2. N300 / P300 — remap `TT_VISIBLE_DEVICES` in `rank_bindings.yaml`

`TT_VISIBLE_DEVICES` indexes PCIe devices, not chips. On P150 each
PCIe device exposes one chip, so the default `"0,1"` / `"2,3"` pins 4
single-chip cards to the two ranks. On N300 / P300, one PCIe device
exposes two chips, so the same `[1, 2]` per-rank mesh needs only one
device per rank:

```yaml
- rank: 0
  env_overrides: { TT_VISIBLE_DEVICES: "0" }    # one 2-chip board
- rank: 1
  env_overrides: { TT_VISIBLE_DEVICES: "1" }    # a disjoint 2-chip board
```

##### 1.1.3. Different split (e.g. 4-4)

To run a `[1, 4]` mesh per rank (8 chips total) you need to change
`device_config.mesh_shape` and `remote_rollout_config.mesh_shape` in
[`grpo_boolq_llama_1b_remote_rollout.yaml`](../../../../configs/training_configs/grpo_boolq_llama_1b_remote_rollout.yaml)
to `[1, 4]`, expand `device_config.device_ids`, and add a
`configurations/split_4_4/` dir with `hosts.txt`, `mgd.textproto`
(mesh `device_topology { dims: [ 1, 4 ] }`), and `rank_bindings.yaml`
(4 chips per rank via `TT_VISIBLE_DEVICES`). See the weight-transfer
test's `configurations/4-4/` at
[`tt-train/tests/python/grpo_remote_rollout/weight_transfer/configurations/4-4/`](../../../../tests/python/grpo_remote_rollout/weight_transfer/configurations/4-4/)
for a working `[1, 4]` template. Also point `runner.sh` at the new
directory (`CONFIG_DIR`).


### 2. Environment Variables

Set these before running:

- `TT_METAL_RUNTIME_ROOT`, `TT_METAL_HOME` — path to the tt-metal repository root.
- `HF_TOKEN` — HuggingFace token for gated model access.

### 3. Run

`./tt-train/sources/examples/grpo_remote_rollout/runner.sh`

### 4. Observe the outputs

- `generated/tt-train/grpo_run/grpo_metrics.csv` — per-step CSV
written by `GRPOMonitor`.
- `generated/tt-train/grpo_run/checkpoints/grpo_step_{N}/` — full HF
checkpoint directories (see
[Checkpointing](../../../../docs/GRPO_TRAINER.md#checkpointing) in
the trainer doc for the layout).

To train on a different model or dataset, copy this directory and
swap `MODEL_ID`, the YAML path, and the dataset / reward function.
Keep the two-rank dispatch and the `WeightSyncCallback` wiring — that
is what keeps the inference worker in sync with the trainer.

---



## See also

- [`tt-train/docs/GRPO_TRAINER.md`](../../../../docs/GRPO_TRAINER.md)
  — generic trainer API (rank- and model-agnostic).
- [`tt-train/docs/LLAMA_WEIGHT_TRANSFER.md`](../../../../docs/LLAMA_WEIGHT_TRANSFER.md)
  — wire format used by `WeightBridge` to ship policy weights from
  the TTML rank to the TTT rank.
