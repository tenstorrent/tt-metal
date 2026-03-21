# SFT Trainer

Supervised Fine-Tuning (SFT) trainer for adapting language models on Tenstorrent devices.
The API is inspired by [TRL's SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) so that
users who know TRL face minimal friction.

---

## Quick Start

```python
from functools import partial
from datasets import load_dataset
from ttml.trainers import SFTConfig, SFTTrainer
from ttml.datasets import InMemoryDataloader, sft_collate_fn

# 1. Build model (any ttml model — Llama, NanoGPT, etc.)
model = ...

# 2. Prepare data
dataset = load_dataset("trl-lib/Capybara", split="train")
collate = partial(sft_collate_fn, max_seq_len=1024, pad_token_id=tokenizer.pad_token_id)
train_loader = InMemoryDataloader(dataset, collate, batch_size=8, shuffle=True)

# 3. Train
trainer = SFTTrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=None,
    config=SFTConfig(max_steps=5_000, learning_rate=2e-5),
)
trainer.train()
```

---

## SFTConfig

`SFTConfig` is a dataclass that controls training-loop behaviour. Optimizer-specific
parameters are passed separately via the `optimizer` argument on `SFTTrainer`.

```python
from ttml.trainers import SFTConfig
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | `int` | `1000` | Total number of optimizer steps. |
| `gradient_accumulation_steps` | `int` | `1` | Micro-batches accumulated before each optimizer step. |
| `eval_interval` | `int` | `200` | Run evaluation every *N* steps (0 to disable). |
| `save_interval` | `int` | `500` | Save a checkpoint every *N* steps (0 to disable). |
| `checkpoint_dir` | `str` | `"checkpoints"` | Directory for checkpoint files. |
| `seed` | `int \| None` | `None` | RNG seed for reproducibility. |
| `max_seq_len` | `int` | `1024` | Maximum sequence length (used for the causal mask). |
| `learning_rate` | `float` | `2e-5` | Peak learning rate; also used for the default AdamW optimizer. |
| `warmup_steps` | `int` | `0` | Linear warmup steps for the default schedule. |
| `max_grad_norm` | `float` | `0.0` | Maximum gradient L2 norm for clipping (0 to disable). |
| `log_interval` | `int` | `1` | Progress-bar update and `on_step_end` callback frequency. |
| `gradient_checkpointing` | `bool` | `False` | Recompute activations during backward to save memory (~30% extra compute). |

---

## SFTTrainer

```python
from ttml.trainers import SFTTrainer
```

### Constructor

```python
SFTTrainer(
    model,
    train_dataloader,
    eval_dataloader,
    config,
    optimizer=None,
    peft_config=None,
    lr_schedule=None,
    callbacks=None,
    compute_loss_func=None,
    loss_composer=None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `AbstractModuleBase` | Model to fine-tune. |
| `train_dataloader` | `TTMLDataloader` | Training data loader. |
| `eval_dataloader` | `TTMLDataloader \| None` | Evaluation data loader. |
| `config` | `SFTConfig` | Trainer configuration. |
| `optimizer` | `OptimizerBase \| dict \| None` | Optimizer instance, config dict (see [Optimizer](#optimizer)), or `None` for default AdamW. |
| `peft_config` | `LoraConfig \| None` | When provided, wraps the model with `LoraModel` automatically (see [LoRA / PEFT](#lora--peft)). |
| `lr_schedule` | `Callable[[int], float] \| None` | Custom `step -> lr` schedule. Overrides the default linear-warmup-then-constant. |
| `callbacks` | `list[TrainerCallback] \| None` | Hooks into the training loop (see [Callbacks](#callbacks)). |
| `compute_loss_func` | `Callable \| None` | Custom `(logits, batch) -> loss` replacing the default masked cross-entropy. |
| `loss_composer` | `Any \| None` | Mesh composer for multi-device loss aggregation. Defaults to `concat_mesh_to_tensor_composer(device, 0)` which works for both single-device and DDP. Pass a custom composer to override. |

### Methods

| Method | Description |
|--------|-------------|
| `train()` | Run the full training loop for `config.max_steps` steps. Automatically cycles the dataloader. |

---

## Optimizer

The `optimizer` parameter accepts three forms:

```python
# 1. Pre-built optimizer instance
optimizer = ttml.optimizers.AdamW(model.parameters(), adamw_config)
SFTTrainer(..., optimizer=optimizer)

# 2. Config dict — forwarded to the optimizer registry
SFTTrainer(..., optimizer={"type": "AdamW", "lr": 1e-4, "weight_decay": 0.01})

# 3. None (default) — creates AdamW with config.learning_rate
SFTTrainer(..., optimizer=None)
```

Any optimizer registered with `ttml.optimizers.register_optimizer` can be used via the dict form.
See [TTML Onboarding — Optimizers](TTML_ONBOARDING.md) for the full list of built-in optimizers.

---

## LoRA / PEFT

Pass a `LoraConfig` via `peft_config` to automatically wrap the model with LoRA adapters.
Base model parameters are frozen; only LoRA weights are trained.

```python
from ttml.trainers import SFTConfig, SFTTrainer, LoraConfig

trainer = SFTTrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=None,
    config=SFTConfig(learning_rate=1e-4),  # higher LR typical for LoRA
    peft_config=LoraConfig(
        rank=8,
        alpha=16.0,
        target_modules=[".*q_proj.*", ".*v_proj.*"],
    ),
)
trainer.train()
```

`LoraConfig` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rank` | `int` | `8` | LoRA rank (bottleneck dimension). |
| `alpha` | `float` | `16.0` | Scaling factor (`alpha / rank`). |
| `target_modules` | `list[str]` | `[]` | Regex patterns matching module names to wrap. |
| `use_rslora` | `bool` | `False` | Use rank-stabilized scaling (`alpha / sqrt(rank)`). |
| `is_bias_trainable` | `bool` | `False` | Unfreeze bias terms in wrapped layers. |
| `trainable_modules` | `list[str]` | `[]` | Additional module name prefixes to unfreeze. |
| `lora_dropout` | `float` | `0.0` | Dropout applied to LoRA input during training. |

---

## Callbacks

Subclass `TrainerCallback` and override any hooks you need:

```python
from ttml.trainers import TrainerCallback

class WandbLogger(TrainerCallback):
    def on_step_end(self, trainer, step, loss, lr):
        wandb.log({"loss": loss, "lr": lr}, step=step)

    def on_eval_end(self, trainer, step, eval_loss):
        wandb.log({"eval_loss": eval_loss}, step=step)

trainer = SFTTrainer(..., callbacks=[WandbLogger()])
```

| Hook | Signature | When |
|------|-----------|------|
| `on_train_begin` | `(trainer)` | After `model.train()`, before first step. |
| `on_before_optimizer_step` | `(trainer)` | After backward / gradient accumulation, before grad clipping and `optimizer.step()`. |
| `on_step_end` | `(trainer, step, loss, lr)` | Every `log_interval` steps. |
| `on_eval_end` | `(trainer, step, eval_loss)` | After each evaluation pass. |
| `on_save` | `(trainer, step, path)` | After each checkpoint save. |
| `on_train_end` | `(trainer)` | After the final step. |

---

## Datasets and Collation

### Batch

All dataloaders must yield `Batch` objects:

```python
@dataclass
class Batch:
    input_ids: Tensor   # [B, 1, 1, T]  uint32
    labels: Tensor      # [B, T]         uint32
    loss_mask: Tensor   # [B, 1, T, 1]  bfloat16
```

**Loss mask contract:** `loss_mask.sum() == B * T` so that `mean(cross_entropy * loss_mask)`
gives the per-completion-token loss.

> **Warning — custom collate functions must respect the loss mask contract.**
>
> The default loss computation multiplies the per-token cross-entropy by `loss_mask` and
> takes the **mean** over `B * T` elements. This only produces a correct per-completion-token
> loss when `loss_mask` is normalized so that `loss_mask.sum() == B * T`. The built-in
> `sft_collate_fn` handles this automatically, but if you write your own `collate_fn` you
> **must** normalize `loss_mask` the same way — for example, by scaling each sequence's mask
> so the batch total equals `B * T`. If the mask is not normalized (e.g. it contains raw 0/1
> values), your reported losses will be incorrectly scaled and will not reflect the true
> per-completion-token loss.

### TTMLDataloader

Abstract base class. Subclass it and implement `__iter__` and `__len__`:

```python
class TTMLDataloader(ABC):
    def __init__(self, dataset, collate_fn, batch_size): ...

    @abstractmethod
    def __iter__(self) -> Iterator[Batch]: ...

    @abstractmethod
    def __len__(self) -> int: ...
```

### InMemoryDataloader

Ready-to-use concrete dataloader for in-memory and HuggingFace datasets:

```python
from ttml.datasets import InMemoryDataloader

loader = InMemoryDataloader(
    dataset,           # indexable dataset (list, HF Dataset, etc.)
    collate_fn,        # Callable[[list], Batch]
    batch_size=8,
    shuffle=True,      # reshuffle each epoch
    drop_last=True,    # drop incomplete final batch
)
```

### sft_collate_fn

Built-in collate function for SFT with prompt/completion masking:

```python
from functools import partial
from ttml.datasets import sft_collate_fn

collate = partial(sft_collate_fn, max_seq_len=1024, pad_token_id=tokenizer.pad_token_id)
```

Expects each example to be a dict with `"input_ids"` and `"labels"` (prompt positions
set to `-100`). Handles padding, truncation, and loss mask normalization.

---

## Gradient Checkpointing

Enable activation recomputation to reduce memory at the cost of extra compute:

```python
SFTConfig(gradient_checkpointing=True)
```

This sets the model's `runner_type` to `RunnerType.MemoryEfficient`, which re-runs each
transformer block's forward pass during backward instead of caching activations.

---

## Gradient Clipping

```python
SFTConfig(max_grad_norm=1.0)
```

Uses `ttml.core.clip_grad_norm` (L2 norm) after gradient accumulation and before the
optimizer step. Set to `0.0` (default) to disable.

---

## DDP / Multi-device

`SFTTrainer` supports distributed data-parallel (DDP) training without any
DDP-specific configuration flags. Two extension points are combined:

1. **Collate function** -- create batch tensors with a `shard_tensor_to_mesh_mapper`
   so that each device receives a slice of the global batch.
2. **`on_before_optimizer_step` callback** -- call
   `ttml.core.distributed.synchronize_gradients` to all-reduce gradients before
   the optimiser step.

Loss aggregation across devices is handled automatically via a default
`concat_mesh_to_tensor_composer(device, 0)`.  Pass a custom `loss_composer`
to the `SFTTrainer` constructor to override.

```python
import ttml
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback

# -- device setup (caller responsibility) --
autograd_ctx = ttml.autograd.AutoContext.get_instance()
ttml.core.distributed.enable_fabric(num_devices)
autograd_ctx.open_device([1, num_devices])
autograd_ctx.initialize_parallelism_context(
    ttml.autograd.DistributedConfig(enable_ddp=True)
)

device = autograd_ctx.get_device()
shard_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)

# -- gradient-sync callback --
class DDPCallback(TrainerCallback):
    def on_before_optimizer_step(self, trainer):
        ttml.core.distributed.synchronize_gradients(trainer.model.parameters())

# -- collate that shards along the batch dimension --
def my_collate(examples, mapper=None):
    ...  # pass mapper to Tensor.from_numpy(...)

trainer = SFTTrainer(
    ...,
    callbacks=[DDPCallback()]
)
trainer.train()
```

See [`sources/examples/lora_llama/train_lora_llama_sft.py`](../sources/examples/lora_llama/train_lora_llama_sft.py)
for a complete DDP + LoRA example.

---

## Complete Example

See [`sources/examples/gsm8k_finetune/gsm8k_finetune.py`](../sources/examples/gsm8k_finetune/gsm8k_finetune.py)
for a full working example that fine-tunes GPT-2 on the GSM8K math dataset using
`SFTTrainer` with a custom collate function and SpeedrunScheduler.
