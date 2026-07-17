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
       └────── WeightBridge socket ─────┘    (replicated weights every step)
```

`N = 2` on this deployment: each rank drives a `[1, 2]` mesh over 2 P150 boards
on a BH Quietbox (4 P150 boards / Blackhole ASICs total, wired as `configurations/local4`).
See [How to run](#how-to-run).

`GRPOTrainer` is unaware of the rank split. It calls
`completer.generate(...)`; `LlamaCompleterRemoteRollout` hides the cross-rank
RPC inside that call. The [trainer doc](../../../../docs/GRPO_TRAINER.md)
covers the model- and rank-agnostic API; everything below is
specific to this two-rank deployment.

---

## Components

| Class | Side | Role |
|-------|------|------|
| `LlamaCompleterRemoteRollout`  | TTML | Concrete `GRPOCompleter`. Owns the ttml policy. Routes `generate(...)` and `push_weights()` to the peer rank via `inference_client`. |
| `MPIRolloutClient`  | TTML | MPI client + `WeightBridge` owner. Constructed before the completer; its constructor blocks until the peer's server is up. |
| `MPIRolloutServer`  | TTT  | Dispatches `OP_GENERATE` / `OP_TRANSFER` / `OP_SHUTDOWN` to user-supplied callbacks. Blocks in `serve_forever()` until shutdown. |
| `TttGenerationWorker` | TTT  | Hosts the `tt-transformers.Transformer` and a captured decode trace. Exposes `generate` and `update_weights` callbacks. |
| `WeightBridge`        | both | Replicated-tensor transport (ABC). `HostWeightBridge` moves each weight to host via MPI and re-uploads it to each receiver submesh. Wire-format spec: [`LLAMA_WEIGHT_TRANSFER.md`](../../../../docs/LLAMA_WEIGHT_TRANSFER.md). |
| `WeightSyncCallback`  | TTML | `TrainerCallback` that calls `completer.push_weights()` every `every` optimizer steps. Opt-in. |

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `LlamaCompletionCtx` | Generation parameters (max tokens, temperature, completions per prompt). |
| `transformer_config` | `TransformerConfig` | Model architecture config (parsed from the YAML training config). |
| `mesh_device` | `ttnn.MeshDevice` | Already-opened TTML mesh. The caller owns its lifetime; the completer does not open or close it. |
| `model_source` | `str` | HuggingFace model ID or path to a local directory containing `model.safetensors`. |
| `inference_client` | `MPIRolloutClient` | RPC client to the TTT rank. The completer routes `generate` and `push_weights` calls through this. |
| `enable_ddp` | `bool` | Enable distributed data parallelism across the TTML mesh. Must agree with the `enable_ddp` in the YAML device config. |

### Methods used by the user

| Method | Description |
|--------|-------------|
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

## How to run

```bash
# 2->2 split on a BH Quietbox (4 P150 boards, 2 P150s per rank).
HF_TOKEN=hf_... ./runner.sh
```

`runner.sh` invokes `tt-run` with `world_size == 2` and dispatches the
two ranks into `boolq_training_example.py`:

| Bindings | TTML mesh (YAML) | TTT parent → submeshes | Chips |
|----------|------------------|------------------------|-------|
| `configurations/local4` | `[1, 2]` DDP ([`grpo_boolq_llama_1b_ddp_2dev.yaml`](../../../../configs/training_configs/grpo_boolq_llama_1b_ddp_2dev.yaml)) | `[1, 2]` → 2× `[1, 1]` | 4 |

`local4` pins rank 0 to P150 boards `0,2` and rank 1 to boards `1,3` via
`TT_VISIBLE_DEVICES`. The pairing is not arbitrary: the BH Quietbox eth
topology is bipartite (`{0,1} × {2,3}`, with no direct `0<->1` or `2<->3`
eth), so each rank must own one chip from each side to form a valid
`[1, 2]` intra-mesh — see the note in
[`configurations/local4/rank_bindings.yaml`](configurations/local4/rank_bindings.yaml).
Override the bindings/hostfile for other machines with `--rank-bindings`
/ `--hostfile`.

### Outputs

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

## Single-rank alternative

`GRPOTrainer` itself supports single-rank training: write a
`GRPOCompleter` whose `generate(...)` runs locally on the same mesh
as the policy, drop the `inference_client` kwarg, and skip the
`WeightSyncCallback`. The current `LlamaCompleterRemoteRollout` requires
`inference_client`, so a single-rank Llama use-case would need a new
completer subclass; the trainer is unchanged.

---

## See also

- [`tt-train/docs/GRPO_TRAINER.md`](../../../../docs/GRPO_TRAINER.md)
  — generic trainer API (rank- and model-agnostic).
- [`tt-train/docs/LLAMA_WEIGHT_TRANSFER.md`](../../../../docs/LLAMA_WEIGHT_TRANSFER.md)
  — wire format used by `WeightBridge` to ship policy weights from
  the TTML rank to the TTT rank.
