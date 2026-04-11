# GRPO Trainer

Group Relative Policy Optimization (GRPO) trainer for reinforcement learning from
human/automated feedback on Tenstorrent devices.
The API follows [TRL's GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)
conventions where possible so that users familiar with TRL face minimal friction.

---

## Quick Start

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.grpo_trainer import GrpoConfig, GrpoTrainer, TrainerCallback

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 1. Prepare dataset — must have a "prompt" column
def format_example(example):
    messages = [
        {"role": "system", "content": "Answer Yes or No."},
        {"role": "user", "content": example["question"]},
    ]
    return {
        "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        "answer": "yes" if example["answer"] else "no",
    }

dataset = load_dataset("google/boolq", split="train").map(format_example)

# 2. Define a reward function
def my_reward(completions, answer, **kwargs):
    return [2.0 if c.strip().lower().startswith(a) else -1.0
            for c, a in zip(completions, answer)]

# 3. Define a callback for logging
class LogCallback(TrainerCallback):
    def on_step_end(self, trainer, step, metrics):
        print(f"Step {step} | Reward: {metrics['reward_mean']:.4f} | Len: {metrics['mean_completion_len']:.2f}")

# 4. Train
trainer = GrpoTrainer(
    model_source=model_id,
    dataset=dataset,
    config=GrpoConfig(
        batch_size=4,
        num_generations=8,
        max_completion_length=256,
        gradient_accumulation_steps=4,
        logging_steps=1,
        prompts_to_train=1600,
    ),
    reward_func=my_reward,
    transformer_config={...},   # see Transformer Config below
    optimizer_config={"type": "MorehAdamW", "lr": 5e-6},
    device_config={"enable_ddp": True, "mesh_shape": [1, 2]},
    callbacks=[LogCallback()],
)
trainer.train()
```

---

## GrpoConfig

`GrpoConfig` is a dataclass that controls the GRPO training loop.

```python
from utils.grpo_trainer import GrpoConfig
```

| Parameter | Type | Default | TRL equivalent | Description |
|-----------|------|---------|----------------|-------------|
| `batch_size` | `int` | — | `per_device_train_batch_size` | Number of prompts sampled per batch, **across all devices combined** (not per device). With DDP each device processes `batch_size // num_devices` prompts. |
| `num_generations` | `int` | — | `num_generations` | Number of completions generated per prompt. Each prompt produces this many candidate responses for reward scoring. |
| `max_completion_length` | `int` | — | `max_completion_length` | Maximum number of tokens to generate per completion. |
| `micro_batch_size` | `int` | — | `generation_batch_size` | Number of completions processed in a single forward pass during loss computation. Controls memory usage. |
| `gradient_accumulation_steps` | `int` | — | `gradient_accumulation_steps` | Number of batches accumulated before each optimizer step. Effective batch = `batch_size * gradient_accumulation_steps`. |
| `num_iterations` | `int` | — | `num_iterations` | Number of training passes over each batch of completions (mini-epochs). |
| `epsilon` | `float` | — | `epsilon` | Clipping parameter for the GRPO surrogate loss (analogous to PPO clip range). |
| `prompts_to_train` | `int` | — | *(use `max_steps`)* | Total number of prompts to train on. Unlike TRL which uses `max_steps`, this directly specifies the data budget. Equivalent to `max_steps * batch_size * gradient_accumulation_steps`. |
| `temperature` | `float` | — | `temperature` | Sampling temperature for completion generation. |
| `warmup_steps` | `int` | — | `warmup_steps` | Number of linear learning rate warmup steps. |
| `output_dir` | `str` | — | `output_dir` | Directory for logs, metrics CSV, and checkpoints. |
| `checkpointing` | `bool` | — | *(use `save_steps`)* | Whether to save checkpoints during training. |
| `checkpoint_interval` | `int` | — | *(use `save_steps`)* | Save a checkpoint every *N* optimizer steps (when `checkpointing=True`). |
| `logging_steps` | `int` | — | `logging_steps` | Fire `on_step_end` callbacks every *N* optimizer steps (0 or negative to disable). The trainer has no built-in logging; all output is callback-driven. |

---

## GrpoTrainer

```python
from utils.grpo_trainer import GrpoTrainer
```

### Constructor

```python
GrpoTrainer(
    model_source,
    dataset,
    config,
    reward_func,
    transformer_config,
    optimizer_config,
    device_config,
    callbacks=None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_source` | `str` | HuggingFace model ID (e.g. `"meta-llama/Llama-3.2-1B-Instruct"`) or path to a local directory containing `model.safetensors`. |
| `dataset` | `Dataset` | HuggingFace `datasets.Dataset` with at least a `"prompt"` column. All other columns are passed to the reward function. |
| `config` | `GrpoConfig` | Training configuration (see above). |
| `reward_func` | `Callable` | Reward function. Receives decoded completions and any dataset columns (see [Reward Functions](#reward-functions)). |
| `transformer_config` | `dict` | Model architecture config (see [Transformer Config](#transformer-config)). |
| `optimizer_config` | `dict` | Optimizer config dict passed to the [ttml optimizer registry](../../docs/TTML_ONBOARDING.md). Must include a `"type"` key. |
| `device_config` | `dict` | Device mesh configuration (see [Device Config](#device-config)). |
| `callbacks` | `list[TrainerCallback] \| None` | Hooks into the training loop (see [Callbacks](#callbacks)). |

### Methods

| Method | Description |
|--------|-------------|
| `train()` | Run the full GRPO training loop. Handles device setup, model loading, generation, reward computation, policy gradient updates, and checkpointing. |

---

## Reward Functions

Reward functions follow TRL conventions. The trainer inspects the function signature
and passes only the arguments it requests:

```python
# Receives completions + specific dataset column by name
def accuracy_reward(completions, answer, **kwargs):
    return [2.0 if c.strip().lower().startswith(a) else -1.0
            for c, a in zip(completions, answer)]

# Receives only completions (no dataset columns needed)
def brevity_reward(completions):
    return [-0.1 * (len(c) / 20) ** 2 for c in completions]

# Receives everything via **kwargs
def custom_reward(completions, **kwargs):
    answers = kwargs["answer"]
    prompts = kwargs["prompts"]
    ...
```

The dispatcher automatically matches parameter names to available data:
- `completions` — decoded completion strings (always available)
- `prompts` — decoded prompt strings (always available)
- Any dataset column name (e.g. `answer`, `category`) — matched by name

If the function declares `**kwargs`, all available data is passed. If it does not,
only explicitly named parameters are passed.

> **Note:** Unlike TRL, which accepts a list of reward functions (`reward_funcs=[f1, f2]`)
> and sums their outputs, `GrpoTrainer` takes a single `reward_func`. To combine
> multiple reward signals, sum them in your function:
>
> ```python
> def combined_reward(completions, answer, **kwargs):
>     acc = [2.0 if c.strip().lower().startswith(a) else -1.0
>            for c, a in zip(completions, answer)]
>     brev = [-0.1 * (len(c) / 20) ** 2 for c in completions]
>     return [a + b for a, b in zip(acc, brev)]
> ```

---

## Callbacks

Subclass `TrainerCallback` and override any hooks you need:

```python
import csv
import os
from utils.grpo_trainer import TrainerCallback

class GRPOMonitor(TrainerCallback):
    def __init__(self, output_dir):
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["step", "reward", "avg_length"])

    def on_step_end(self, trainer, step, metrics):
        reward = metrics["reward_mean"]
        length = metrics["mean_completion_len"]
        print(f"Step {step} | Reward: {reward:.4f} | Len: {length:.2f} tokens")
        with open(self.file_path, mode="a", newline="") as f:
            csv.writer(f).writerow([step, reward, length])

trainer = GrpoTrainer(..., callbacks=[GRPOMonitor(output_dir)])
```

| Hook | Signature | When |
|------|-----------|------|
| `on_step_end` | `(trainer, step, metrics)` | Every `logging_steps` optimizer steps. `metrics` is a dict with `reward_mean`, `reward_std`, `mean_completion_len`, and `lr`. |
| `on_train_end` | `(trainer)` | After the final batch. |

The trainer has no built-in logging. All monitoring, CSV writing, and console
output is handled through callbacks. The `trainer` argument gives callbacks
access to `trainer.model` and `trainer.config`.

---

## Transformer Config

Model architecture parameters passed as a plain dict:

```python
transformer_config = {
    "model_type": "llama",
    "num_heads": 32,
    "num_groups": 8,
    "embedding_dim": 2048,
    "intermediate_dim": 8192,
    "dropout_prob": 0.0,
    "num_blocks": 16,
    "weight_tying": "enabled",
    "vocab_size": 32000,
    "max_sequence_length": 1024,
    "runner_type": "memory_efficient",
    "theta": 500000.0,
    "rope_scaling": {
        "scaling_factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_context_length": 8192,
    },
}
```

---

## Optimizer Config

Optimizer parameters passed as a dict with a `"type"` key. Forwarded to the
ttml optimizer registry:

```python
optimizer_config = {
    "type": "MorehAdamW",
    "lr": 5.0e-6,
    "beta1": 0.9,
    "beta2": 0.99,
    "epsilon": 1.0e-8,
    "weight_decay": 0.01,
}
```

Any optimizer registered with `ttml.optimizers.register_optimizer` can be used.
See [TTML Onboarding — Optimizers](../../docs/TTML_ONBOARDING.md) for the full list of
built-in optimizers.

---

## Device Config

Device mesh and distributed training configuration:

```python
device_config = {
    "enable_ddp": True,
    "mesh_shape": [1, 2],       # [rows, cols] of the device mesh
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_ddp` | `bool` | `False` | Enable distributed data-parallel training. |
| `mesh_shape` | `list[int]` | `[1, 1]` | Shape of the device mesh `[rows, cols]`. Total devices = `rows * cols`. |
| `device_ids` | `list[int] \| None` | `None` | Specific device IDs to use (default: auto-select). |
| `enable_tp` | `bool` | `False` | Enable tensor parallelism (mutually exclusive with DDP). |

`GrpoTrainer.train()` handles device setup automatically — calling `enable_fabric`,
`open_device`, and `initialize_parallelism_context` based on this config.

---

## DDP / Multi-device

DDP is configured entirely through `device_config`. When `enable_ddp=True`:

1. The device mesh is opened with the specified `mesh_shape`.
2. Input tensors are sharded across devices along the batch dimension.
3. Gradients are synchronized via `ttml.core.distributed.synchronize_gradients`
   before each optimizer step.

`batch_size` specifies the **global** batch size across all devices. Each device
processes `batch_size // total_devices` prompts per batch.

---

## Datasets

The trainer accepts any HuggingFace `datasets.Dataset` object. The only requirement
is a `"prompt"` column containing formatted prompt strings. All other columns
are preserved and passed to the reward function.

```python
from datasets import load_dataset

dataset = load_dataset("google/boolq", split="train").map(format_fn)
# dataset must have: "prompt" (str)
# dataset may have:  "answer", "category", ... (passed to reward_func)
```

---

## Key Differences from TRL

| Aspect | TRL `GRPOTrainer` | ttml `GrpoTrainer` |
|--------|-------------------|---------------------|
| **Model** | Passed as a `transformers` model object | Specified via `model_source` (HF ID or local path); built internally |
| **Reward functions** | List of functions (`reward_funcs=[f1, f2]`), summed | Single function (`reward_func=f`) |
| **Batch size** | `per_device_train_batch_size` (per device) | `batch_size` (global, across all devices) |
| **Training budget** | `max_steps` (optimizer steps) | `prompts_to_train` (total prompts) |
| **Optimizer** | String name (`optim="adamw_bnb_8bit"`) | Config dict (`{"type": "MorehAdamW", ...}`) |
| **Device setup** | Handled by HF Accelerate | Handled by `device_config` dict |
| **KL penalty** | `beta` parameter | Not implemented (equivalent to `beta=0.0`) |
| **Callbacks** | HF `TrainerCallback` with `on_log(args, state, control, logs)` | `TrainerCallback` with `on_step_end(trainer, step, metrics)` |

---

## Examples

### Training

[`boolq_training_example.py`](boolq_training_example.py) — trains
Llama-3.2-1B-Instruct on BoolQ using `GrpoTrainer` with a custom reward
function, CSV logging via `GRPOMonitor` callback, and DDP on 2 devices.

```bash
python3 boolq_training_example.py
```

### Accuracy Evaluation

[`boolq_accuracy_example.py`](boolq_accuracy_example.py) — evaluates a
model on the BoolQ validation set with greedy decoding (`temperature=0`)
and writes per-question results to a CSV. Runs on 2 devices (p150x2) with
`PROMPTS_TO_VALIDATE=20` by default.

```bash
python3 boolq_accuracy_example.py
```

To evaluate a fine-tuned checkpoint, change `MODEL_ID` to the directory
containing `model.safetensors`.

### Tests

[`test_batched_vs_single_completion.py`](test_batched_vs_single_completion.py) —
verifies that batched inference produces identical outputs to one-by-one
inference under greedy decoding. Runs on a single device.

```bash
pytest test_batched_vs_single_completion.py
```

---

## Environment Variables

Set these before running:

- `TT_METAL_RUNTIME_ROOT` — path to the tt-metal repository root.
- `HF_TOKEN` — HuggingFace token for gated model access.
- `TT_MESH_GRAPH_DESC_PATH` — path to the mesh graph descriptor.
