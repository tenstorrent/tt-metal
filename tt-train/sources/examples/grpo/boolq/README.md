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
  ttml.Llama policy                 tt-transformers Transformer
  GRPOTrainer + optimizer           TttGenerationWorker
  mesh: [1, N] (DDP)                mesh: [1, 1]
       │                                ▲
       │  TttInferenceClient    OP_GENERATE / OP_TRANSFER / OP_SHUTDOWN
       └──────────────► MPI ───────────► TttInferenceServer
       │                                │
       └────── WeightBridge socket ─────┘    (replicated weights every step)
```

`GRPOTrainer` is unaware of the rank split. It calls
`completer.generate(...)`; `LlamaGRPOCompleter` hides the cross-rank
RPC inside that call. The [trainer doc](../../../../docs/GRPO_TRAINER.md)
covers the model- and rank-agnostic API; everything below is
specific to this two-rank deployment.

---

## Components

| Class | Side | Role |
|-------|------|------|
| `LlamaGRPOCompleter`  | TTML | Concrete `GRPOCompleter`. Owns the ttml policy. Routes `generate(...)` and `push_weights()` to the peer rank via `inference_client`. |
| `TttInferenceClient`  | TTML | MPI client + `WeightBridge` owner. Constructed before the completer; its constructor blocks until the peer's server is up. |
| `TttInferenceServer`  | TTT  | Dispatches `OP_GENERATE` / `OP_TRANSFER` / `OP_SHUTDOWN` to user-supplied callbacks. Blocks in `serve_forever()` until shutdown. |
| `TttGenerationWorker` | TTT  | Hosts the `tt-transformers.Transformer` and a captured decode trace. Exposes `generate` and `update_weights` callbacks. |
| `WeightBridge`        | both | Replicated-tensor transport over a `MeshSocket`. Wire-format spec: [`LLAMA_WEIGHT_TRANSFER.md`](../../../../docs/LLAMA_WEIGHT_TRANSFER.md). |
| `WeightSyncCallback`  | TTML | `TrainerCallback` that calls `completer.push_weights()` every `every` optimizer steps. Opt-in. |

---

## LlamaGRPOCompleter

```python
from utils.llama_grpo_completer import LlamaGRPOCompleter, LlamaCompletionCtx
```

Llama-specific implementation of `GRPOCompleter`. Loads the ttml
policy from a HuggingFace ID or local safetensors directory, manages
the KV cache, and dispatches generation requests over MPI to the TTT
rank.

```python
completer = LlamaGRPOCompleter(
    ctx=LlamaCompletionCtx(
        max_tokens_to_complete=256,
        temperature=0.7,
        completions_per_prompt=8,
    ),
    transformer_config=transformer_config,   # TransformerConfig
    mesh_device=mesh_device,                 # opened ttnn.MeshDevice
    model_source="meta-llama/Llama-3.2-1B-Instruct",
    inference_client=client,                 # TttInferenceClient
    enable_ddp=True,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `LlamaCompletionCtx` | Generation parameters (max tokens, temperature, completions per prompt). |
| `transformer_config` | `TransformerConfig` | Model architecture config (parsed from the YAML training config). |
| `mesh_device` | `ttnn.MeshDevice` | Already-opened TTML mesh. The caller owns its lifetime; the completer does not open or close it. |
| `model_source` | `str` | HuggingFace model ID or path to a local directory containing `model.safetensors`. |
| `inference_client` | `TttInferenceClient` | RPC client to the TTT rank. The completer routes `generate` and `push_weights` calls through this. |
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
from utils.inference_bridge import TttInferenceClient
from utils.llama_grpo_completer import (
    LlamaCompletionCtx, LlamaGRPOCompleter, WeightSyncCallback,
)
from ttml.trainers import GRPOTrainer, get_grpo_config

TTML_RANK, TTT_RANK = 0, 1
mesh_device = ...                         # opened from YAML config

# Bridge handshake: blocks until rank 1 also constructs its server.
client = TttInferenceClient(peer_rank=TTT_RANK, device=mesh_device)

dataset = load_dataset("google/boolq", split="train").map(format_example)

completer = LlamaGRPOCompleter(
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
from utils.inference_bridge import TttInferenceServer
from utils.ttt_generation_worker import TttGenerationWorker
from utils.llama_ttt_presets import (
    bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad,
)

ttnn.init_distributed_context()
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    offset=ttnn.MeshCoordinate(0, 0),
)

stop_token_ids, pad_token_id = llama_stop_and_pad("meta-llama/Llama-3.2-1B-Instruct")

worker = TttGenerationWorker(
    mesh_device=mesh_device,
    model_source="meta-llama/Llama-3.2-1B-Instruct",
    max_batch_size=32,
    max_seq_len=2048,
    instruct=True,
    optimizations=bf16_attn_bfp8_mlp_optimizations,
    stop_token_ids=stop_token_ids,
    pad_token_id=pad_token_id,
    temperature=0.7, top_k=0, top_p=1.0, seed=None,
)

server = TttInferenceServer(
    peer_rank=0,
    device=mesh_device,
    generate_fn=worker.generate,
    on_weights_received=worker.update_weights,
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
HF_TOKEN=hf_... ./runner.sh
```

`runner.sh` invokes `tt-run` with `world_size == 2` and dispatches the
two ranks into `boolq_training_example.py`. Mesh / DDP configuration
for the TTML rank is read from
[`grpo_boolq_llama_2dev_ddp_gas_4.yaml`](../../../../configs/training_configs/grpo_boolq_llama_2dev_ddp_gas_4.yaml)
(`mesh_shape: [1, 2]`, `enable_ddp: true`). The TTT rank uses a fixed
`[1, 1]` mesh hardcoded in the entrypoint.

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
`WeightSyncCallback`. The current `LlamaGRPOCompleter` requires
`inference_client`, so a single-rank Llama use-case would need a new
completer subclass; the trainer is unchanged.

---

## See also

- [`tt-train/docs/GRPO_TRAINER.md`](../../../../docs/GRPO_TRAINER.md)
  — generic trainer API (rank- and model-agnostic).
- [`tt-train/docs/LLAMA_WEIGHT_TRANSFER.md`](../../../../docs/LLAMA_WEIGHT_TRANSFER.md)
  — wire format used by `WeightBridge` to ship policy weights from
  the TTML rank to the TTT rank.
