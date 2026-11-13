# Unified Trainer

`ttml.unified_trainer` fine-tunes HuggingFace models on Tenstorrent devices from one validated config. The same
trainer runs from the command line and inside services such as tt-inference-server, which builds it once and runs one
job after another on it.

`unified_trainer` is a working name. The existing `ttml.trainers` package is unchanged and works as before.

---

## Quick start

From the command line, with a recipe from `tt-train/configs/training_configs/unified_trainer/`:

```bash
python -m ttml.unified_trainer \
  --config tt-train/configs/training_configs/unified_trainer/llama_3_2_1b_sst2_lora.yaml \
  --set max_steps=100 --set model.lora.r=8
```

Each `--set key=value` overrides one field; dotted keys reach into sections, and values are parsed as YAML.

From Python:

```python
from ttml.unified_trainer import CheckpointCallback, MetricsCallback, SFTTrainer, load_config

trainer = SFTTrainer(callbacks=[MetricsCallback(), CheckpointCallback()])
try:
    trainer.setup(load_config("recipe.yaml"))
    trainer.train()
finally:
    trainer.cleanup()
```

`SFTTrainerConfig(...)` builds the same config in code.

---

## What it trains

**Models.** HuggingFace Llama checkpoints (`model_type: llama`), as a hub id or a local model directory, in bf16.
Checkpoints the ttml Llama can't represent are rejected before the device opens: quantized weights, attention or MLP
biases, an activation other than SiLU, and RoPE scaling other than `llama3` or the default. Other architectures plug in
as model providers (see [Adding a model](#adding-a-model)).

**LoRA or every weight.** With a `lora` section in `model`, the run trains LoRA adapters and nothing else. Without
one, it trains every weight. `target_modules` uses HF projection names. ttml's Llama fuses some projections into one layer, which has
one LoRA pair, so a fused group is targeted whole or not at all:

| HF projections | ttml Llama layer |
|---|---|
| `q_proj`, `k_proj`, `v_proj` | `attention.qkv_linear` |
| `o_proj` | `attention.out_linear` |
| `gate_proj`, `up_proj` | `mlp.w_gate_up` |
| `down_proj` | `mlp.w2` |

`[q_proj, k_proj, v_proj, o_proj]` trains attention, and adding `gate_proj, up_proj, down_proj` trains every projection.
`[q_proj, v_proj]` is rejected, because LoRA on `qkv_linear` also updates `k_proj`.

**Data.** The `id` of the `dataset` section picks one:

- `sst2` (GLUE SST-2 sentiment), with prompts `Review: <sentence>\nOutput:` and responses `{"label": "positive"}`.
- `alpaca` (`tatsu-lab/alpaca`), with the Alpaca prompt; 2% of it is held out for validation.
- `custom`: local files, for example JSONL, named in the section's `custom_options` and formatted with the `alpaca`
  template. The template reads `instruction` and `output`, and `input` if present. `column_mapping` maps those fields
  to columns with other names.

```yaml
dataset:
  id: custom
  custom_options:
    file_type: jsonl
    train_dataset_path: data/train.jsonl
    val_dataset_path: data/val.jsonl
    column_mapping: {instruction: question, output: answer}
```

Examples longer than `max_length` tokens are dropped. The loss is the cross-entropy of the response tokens; the prompt
and padding don't count.

---

## Configuration

Unknown fields are errors, so a typo fails instead of being ignored. Required fields have no default.

| Field | Default | Description |
|---|---|---|
| `dataset` | | `id`, one of `sst2`, `alpaca` or `custom`, is required. `custom_options` is required when `id` is `custom`: `train_dataset_path`, `val_dataset_path` (no validation if unset), `file_type` (`json`; `jsonl` works too), `template` (`alpaca`), `column_mapping`. |
| `model` | | `name`, an HF hub id or a local HF model directory, is required. `lora` (unset) trains LoRA adapters: `r`, `alpha` (the scaling is `alpha / r`), `target_modules`, `dropout` (0.0), `task_type` (`CAUSAL_LM`); unset trains every weight. `init_strategy` (unset) is a checkpoint to start from; see [Starting from a checkpoint](#starting-from-a-checkpoint). |
| `dtype` | `bfloat16` | Must be `bfloat16`. |
| `max_length` | | Tokens per example: a multiple of 32, at most the model's `max_position_embeddings`. |
| `batch_size` | | Examples per micro-batch. |
| `gradient_accumulation_steps` | `1` | Micro-batches per optimizer step. |
| `num_epochs` | `1` | |
| `max_steps` | `0` | Stop after this many optimizer steps; 0 runs every epoch. |
| `max_grad_norm` | `0.0` | Clip the global gradient norm to this; 0 disables clipping. |
| `val_steps_freq` | `0` | Validate before training and every this many optimizer steps; 0 disables validation. |
| `gradient_checkpointing` | `false` | Recompute activations in the backward pass to save memory. |
| `master_dtype` | `bfloat16` | `float32` keeps fp32 master weights for AdamW; full fine-tuning needs it (see below). |
| `seed` | `0` | Seeds initialization, LoRA dropout and data shuffling. |
| `optimizer` | | One optimizer, or a list (see [Several optimizers](#several-optimizers)). Each takes `lr` (constant), `modules`, and `type`, which picks the optimizer and the other fields it takes. `adamw` (the default, and the only one so far) takes `weight_decay` (0.0, decoupled), `beta1`, `beta2`, `epsilon`, `amsgrad` and `stochastic_rounding`; unset, those keep ttml's defaults. |
| `metrics` | | `steps_freq` (10): train metrics are averaged over and logged every this many steps. `train_metrics` (`[loss]`, `learning_rate` also works), `val_metrics` (`[loss]`). |
| `logging` | | `log_level` (`INFO`), `use_wandb` (false), `wandb_project`, `wandb_run_name`, `wandb_tags`. |
| `checkpoint` | | See [Checkpoints](#checkpoints). |

**Full fine-tuning precision.** At fine-tuning learning rates, an AdamW update is smaller than bf16 can resolve next
to most weights, so a bf16 weight doesn't move. `master_dtype: float32` stores the trainable weights in fp32; the
forward and backward passes still run in bf16. LoRA doesn't need it, as its adapters start at zero. See
[fp32 master weights](MIXED_PRECISION.md#fp32-master-weights).

### Several optimizers

A list under `optimizer` trains different weights with different optimizers or settings:

```yaml
optimizer:
  - lr: 1.0e-3
    modules: [attention_norm, mlp_norm]  # these weights train here
  - lr: 1.0e-4                           # no modules: every other trainable weight
    weight_decay: 0.01
```

A trainable weight goes to the first optimizer whose `modules` lists a `/`-separated part of its ttml name (for
example `llama/blocks/0/attention/qkv_linear/lora_A`), as in ttml's param groups, else to the one optimizer without
`modules`. Every trainable weight must get an optimizer and every optimizer a weight; `setup` raises otherwise, so a
weight is never left untrained by a missing pattern. One optimizer without `modules` trains every trainable weight.
With several optimizers, the `learning_rate` metric is logged per optimizer, as `learning_rate/0`, `learning_rate/1`
and so on, in list order.

---

## Checkpoints

A checkpoint holds what the run trains, in the format HuggingFace tools load:

| Run | Directory | Loads with |
|---|---|---|
| LoRA | `adapter_config.json`, `adapter_model.safetensors` | `PeftModel.from_pretrained(base_model, path)` |
| Every weight | `config.json`, `model.safetensors` (bf16), tokenizer files | `AutoModelForCausalLM.from_pretrained(path)` |

Both also hold `trainer_state.json`: the step, epoch and metrics at the save. With `save_optim`, they hold
`optimizer_state.pkl` too: the optimizer state, the trained weights at the precision they train in (a full
checkpoint's `model.safetensors` is bf16) and the RNG state, which a resumed run needs. The adapter of a fused layer
is written per HF projection, with the projections sharing one `lora_A` and splitting `lora_B` by rows. A full
checkpoint writes a tied embedding once, as HF does.

The `checkpoint` section:

| Field | Default | Description |
|---|---|---|
| `save_strategy` | `type: none` | When to save; see below. Unless `none`, the final checkpoint is also saved at the end. |
| `rotation_strategy` | `type: last` | Which checkpoints to keep; see below. |
| `project_dir` | `runs/unified_trainer` | Checkpoints go in `<project_dir>/checkpoints/`. |
| `final_checkpoint_name` | `final` | |
| `save_optim` | `false` | Also save the optimizer state, so a run can resume from the checkpoint. |

Each strategy is a section whose `type` picks it, and with it the other fields it takes:

```yaml
val_steps_freq: 50
checkpoint:
  save_strategy:
    type: step
    frequency: 100
  rotation_strategy:
    type: best
    keep_n: 2
  save_optim: true
  project_dir: runs/my_run
```

A field another `type` doesn't take is an error, so to change the type with `--set`, replace the whole section:
`--set "checkpoint.save_strategy={type: epoch}"`.

| `save_strategy` | Saves |
|---|---|
| `step` | Every `frequency` (100) optimizer steps. |
| `epoch` | At the end of every `frequency`-th (1) epoch. |
| `none` | Nothing, not even the final checkpoint. |

| `rotation_strategy` | Keeps |
|---|---|
| `last` | The `keep_n` (3) most recent checkpoints. |
| `best` | The `keep_n` (3) with the best `metric` (`val/loss`, from the latest validation, so it needs `val_steps_freq`): the lowest with `mode: min` (the default), the highest with `mode: max`. And the latest checkpoint. |
| `none` | Every checkpoint. |

Rotation never deletes the checkpoint just saved, and only deletes checkpoints of the same run: an earlier run's in the
same `project_dir`, such as the one a run resumed from, are kept. `<project_dir>/checkpoints/checkpoint_history.json`
lists the run's checkpoints that are kept, with their step, epoch and metrics. A checkpoint is written to a staging
directory and moved into place when complete, so a failed save leaves the previous one intact.

### Starting from a checkpoint

`model.init_strategy` names a checkpoint directory, which `setup` loads once it has built the model and optimizer.
Unset, the run starts from the pretrained weights of `model.name`.

```yaml
model:
  name: meta-llama/Llama-3.2-1B-Instruct
  init_strategy:
    type: resume
    path: runs/my_run/checkpoints/final
```

Its `type` is one of:

- `init`: the trained weights only. Training starts at step 0 with a fresh optimizer.
- `resume`: also the optimizer state, which the checkpoint holds if it was saved with `checkpoint.save_optim`.
  Training continues after the checkpoint's step, on the batches the run would have trained on next, so keep
  `batch_size`, `gradient_accumulation_steps`, `optimizer`, the data and the `seed` as they were.

The trainer loads it whichever callbacks it has, so a run needs no `CheckpointCallback` to start from a checkpoint. A
checkpoint that doesn't load fails `setup`. A checkpoint of the other kind is an error: a LoRA run can't load a full
model (make it the `model.name` instead), and a full run can't load an adapter.

---

## Running inside a service

A host builds one trainer with its callbacks and runs each job as:

```python
try:
    trainer.setup(config)  # validates, loads data, opens the device, builds the model and optimizer,
                           # and loads the checkpoint model.init_strategy names
    trainer.train()
finally:
    trainer.cleanup()  # also after a failed setup
```

- **Validation first.** `setup` checks the config against the model before it opens the device or downloads weights.
  A config the model can't train raises `UnsupportedTrainingConfig`, a `ValueError` whose message starts with the field
  at fault, for example `model.lora.target_modules: ...`.
- **Cleanup.** `cleanup` frees the model, optimizer and data, then closes the device. It is safe to call more than
  once and after any failure. What the trainer held before its first run, such as the callbacks, is kept for the next
  job.
- **Callbacks** are objects with any of these methods, each called with the trainer first: `on_setup_error`,
  `on_train_start`, `on_train_epoch_start`, `on_train_batch_start`, `on_forward_start`, `on_forward_end`,
  `on_backward_start`, `on_backward_end`, `on_optimizer_step_start`, `on_optimizer_step_end`, `on_step_end`,
  `on_train_batch_end`, `on_train_epoch_end`, `on_validation_start`, `on_validation_batch_start`,
  `on_validation_batch_end`, `on_validation_end`, `on_train_end` and `on_train_error`. They don't have to subclass
  `Callback`.
- **Failures.** An exception in `setup` goes to every callback's `on_setup_error`, and one in `train` to
  `on_train_error`; then the method re-raises it. So a host reports both through its callbacks, with nothing to catch.
- **Stopping.** An exception from a callback ends the run: `on_train_error` sees it, then `train` raises it. To
  cancel a job, raise an exception of your own and catch it around `train`.
- **Reading the run.** `trainer.loss_to_float(loss)`, `trainer.current_learning_rate()` (a list, one LR per
  optimizer in `optimizer` order), `trainer.parameter_counts()` (total, trainable), `trainer.global_step` and
  `trainer.epoch`.
- **Saving on your own schedule.** `trainer.save_checkpoint(path, metadata)` writes a checkpoint, and
  `trainer.load_checkpoint(path)` loads one. With `optimizer=True` both also cover the optimizer state, and
  `trainer.resume_from_checkpoint(path)`, before `train`, continues the run of a checkpoint saved that way.

---

## Adding a model

A model provider builds the ttml model for an HF `model_type` and maps its weights to HF's. It implements the
`ModelProvider` protocol in `ttml/unified_trainer/providers/base.py`:

- `check(config, hf_config)`: raise `UnsupportedTrainingConfig` for anything the model can't train. It runs before the
  device opens.
- `build(config, hf_config)`: the model with pretrained weights, wrapped in LoRA when `config.model.lora` is set.
- `export_adapter` / `import_adapter`: the LoRA weights as PEFT tensors and adapter config, and back.
- `export_weights` / `import_weights`: every weight, as an HF model directory, and back.

Register it in `PROVIDERS` in `ttml/unified_trainer/providers/__init__.py`. `LlamaProvider` in `providers/llama.py` is
the reference.

---

## Tests

```bash
pytest tt-train/tests/python/unified_trainer
```

Most tests are device-free: the training loop and callbacks on a fake trainer, configs and recipes, data, the PEFT
mapping and checkpoint files. The tests marked `requires_device` train a tiny random Llama on a device. They check
that loss decreases and that HF PEFT and `from_pretrained` load the checkpoints with the same logits as ttml.
