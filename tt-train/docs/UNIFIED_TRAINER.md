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
  --config tt-train/configs/training_configs/unified_trainer/llama_3_2_1b_instruct_sst2_lora.yaml \
  --set stopping_criteria.limit=100 --set model.lora.r=8
```

Each `--set key=value` overrides one field; dotted keys reach into sections, and values are parsed as YAML.
`--trainer` names the trainer the recipe is for; `sft`, the default, is the only one so far.

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

**LoRA only.** A run trains the LoRA adapters its `model.lora` section describes, and nothing else. Full fine-tuning
(every weight) is not supported yet: a config without `model.lora` is rejected. `target_modules` uses HF projection
names, and takes any of them. ttml's Llama fuses some projections into one layer:

| HF projections | ttml Llama layer |
|---|---|
| `q_proj`, `k_proj`, `v_proj` | `attention.qkv_linear` |
| `o_proj` | `attention.out_linear` |
| `gate_proj`, `up_proj` | `mlp.w_gate_up` |
| `down_proj` | `mlp.w2` |

Each targeted projection still gets its own adapter, as in PEFT. A fused layer stacks the `lora_A` of its targeted
projections in one weight, and holds each one's `lora_B` in that projection's rows of another; the rows of a projection
that isn't targeted stay zero. So `[q_proj, v_proj]` trains adapters on q and v and leaves k as it is, and
`[q_proj, k_proj, v_proj, o_proj]` trains attention. The one difference from PEFT is that the projections of a fused
layer share a dropout mask.

**Optimizer.** AdamW at a constant learning rate; it is the only optimizer supported so far.

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
and padding don't count. Other datasets plug in as dataset providers (see [Adding a dataset](#adding-a-dataset)).

---

## Configuration

Unknown fields are errors, so a typo fails instead of being ignored. Required fields have no default.

| Field | Default | Description |
|---|---|---|
| `model` | | `name`, an HF hub id or a local HF model directory, is required. `lora` is the LoRA adapters to train, and a run needs it, as only LoRA training is supported so far: `r`, `alpha` (the scaling is `alpha / r`), `target_modules`, `dropout` (0.0), `task_type` (`CAUSAL_LM`). `gradient_checkpointing` (`false`) builds the model to recompute activations in the backward pass, which saves memory. |
| `dataset` | | `id`, one of `sst2`, `alpaca` or `custom`, is required. `custom_options` is required when `id` is `custom`: `train_dataset_path`, `val_dataset_path` (no validation if unset), `file_type` (`json`; `jsonl` works too), `template` (`alpaca`), `column_mapping`. |
| `optimizer` | | One optimizer, or a list (see [Several optimizers](#several-optimizers)). Each takes `lr` (constant), `modules`, and `type`, which picks the optimizer and the other fields it takes. `adamw` (the default, and the only one supported so far) takes `weight_decay` (0.0, decoupled), `beta1`, `beta2`, `epsilon`, `amsgrad` and `stochastic_rounding`; unset, those keep ttml's defaults. |
| `device` | | `mesh` (`[1, 1]`), the size of each dimension of the device mesh the run opens, such as `[rows, columns]`. A single chip, `[1, 1]`, is the only mesh supported so far. |
| `dtype` | `bfloat16` | Must be `bfloat16`. |
| `max_length` | | Tokens per example: a multiple of 32, at most the model's `max_position_embeddings`. |
| `batch_size` | | Examples per micro-batch. |
| `gradient_accumulation_steps` | `1` | Micro-batches per optimizer step. |
| `max_grad_norm` | `0.0` | Clip the global gradient norm to this before every optimizer step; 0 disables clipping. |
| `seed` | `0` | Seeds initialization, LoRA dropout and data shuffling. |
| `stopping_criteria` | `type: epoch`, `limit: 1` | When the run ends; see [Stopping criteria](#stopping-criteria). |
| `metrics` | | Two sections, `train` and `validation`, each with `steps_freq` and `metric_names`; see [Metrics and validation](#metrics-and-validation). |
| `logging` | | `log_level` (`INFO`), `use_wandb` (false), `wandb_project`, `wandb_run_name`, `wandb_tags`. |
| `checkpoint` | | See [Checkpoints](#checkpoints). |

### Stopping criteria

`stopping_criteria` ends the run after `limit` epochs or optimizer steps, as its `type` says. Both fields are required
in the section; without the section, the run trains for one epoch.

```yaml
stopping_criteria:
  type: step
  limit: 500
```

| `type` | The run ends after |
|---|---|
| `epoch` | `limit` passes over the train data. |
| `step` | `limit` optimizer steps, over as many epochs as they take. It ends in the middle of an epoch if the limit falls there. |

A step limit needs every epoch to make an optimizer step, so `setup` rejects one on train data with fewer batches than
`gradient_accumulation_steps`.

### Metrics and validation

The `metrics` section has a `train` and a `validation` section with the same fields:

```yaml
metrics:
  train:
    steps_freq: 10
    metric_names: [loss, learning_rate]
  validation:
    steps_freq: 50
    metric_names: [loss]
```

| Field | `train` | `validation` |
|---|---|---|
| `steps_freq` | `10`. The metrics are averaged over and logged every this many optimizer steps. | `0`. The run validates before training and every this many optimizer steps; 0 disables validation. |
| `metric_names` | `[loss]`; `learning_rate` also works. | `[loss]`. |

Train metrics are logged as `train/<name>` and validation metrics as `val/<name>`.

### Several optimizers

A list under `optimizer` trains different weights with different optimizers or settings:

```yaml
optimizer:
  - lr: 1.0e-3
    modules: [lora_B]  # these weights train here
  - lr: 1.0e-4         # no modules: every other trainable weight
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

A checkpoint is a directory holding the model's weights in the format `checkpoint.format` picks, and
`trainer_state.json`: the step, epoch and metrics at the save, the format, and the base model the run trains
(`name_or_path`, and `revision`: the hub commit it was downloaded at, `null` for a local directory).

```yaml
checkpoint:
  format:
    type: hf  # hf or tt
```

| `type` | Training | The directory holds |
|---|---|---|
| `hf` | LoRA | The LoRA adapters as a PEFT adapter: `adapter_config.json` and `adapter_model.safetensors`. |
| `hf` | Full fine-tuning | The model, as `save_pretrained` writes one. |
| `tt` | LoRA | The LoRA adapters, in `checkpoint.pkl`. |
| `tt` | Full fine-tuning | Every parameter, in `checkpoint.pkl`. |

`hf` is HuggingFace's layout. The adapter of a fused layer is written per HF projection, so the PEFT adapter is the one
PEFT would have trained. A model is laid out as `save_pretrained` writes one: `config.json`, the tokenizer files, and
the bf16 weights in `model.safetensors`, or in shards of at most 5 GB listed in `model.safetensors.index.json`.

`tt` is ttml's own layout, the file GRPO saves with `ttml.checkpointing`: the weights as the model holds them, with q,
k and v fused, the q and k rows permuted for RoPE, and the vocabulary padded.

The base model comes back from what the checkpoint records:

- A PEFT adapter names it in `adapter_config.json`, as `base_model_name_or_path` and `revision`, so
  `AutoPeftModelForCausalLM.from_pretrained(path)` loads the adapter on the model it was trained on.
- A `tt` checkpoint names it in the header of `checkpoint.pkl`, with the LoRA `r`, `alpha` and `target_modules`.

The `checkpoint` section:

| Field | Default | Description |
|---|---|---|
| `save_strategy` | `type: none` | When to save; see below. Unless `none`, the final checkpoint is also saved at the end. |
| `rotation_strategy` | `type: last` | Which checkpoints to keep; see below. |
| `init_strategy` | unset | A checkpoint to start from; see [Starting from a checkpoint](#starting-from-a-checkpoint). Unset starts from the pretrained weights. |
| `format` | `type: hf` | How a checkpoint stores the model; see above. |
| `project_dir` | `runs/unified_trainer` | Checkpoints go in `<project_dir>/checkpoints/`. |
| `final_checkpoint_name` | `final` | |

Each strategy is a section whose `type` picks it, and with it the other fields it takes:

```yaml
metrics:
  validation:
    steps_freq: 50
checkpoint:
  save_strategy:
    type: step
    frequency: 100
  rotation_strategy:
    type: best
    keep_n: 2
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
| `best` | The `keep_n` (3) with the best `metric` (`val/loss`, from the latest validation, so it needs `metrics.validation.steps_freq`): the lowest with `mode: min` (the default), the highest with `mode: max`. And the latest checkpoint. |
| `none` | Every checkpoint. |

Rotation never deletes the checkpoint just saved, and only deletes checkpoints of the same run: an earlier run's in the
same `project_dir`, such as the one a run started from, are kept. `<project_dir>/checkpoints/checkpoint_history.json`
lists the run's checkpoints that are kept, with their step, epoch and metrics. A checkpoint is written to a staging
directory and moved into place when complete, so a failed save leaves the previous one intact.

### Starting from a checkpoint

`checkpoint.init_strategy` names a checkpoint directory, which `setup` loads onto the model it builds. Unset, the run
starts from the pretrained weights of `model.name`.

```yaml
checkpoint:
  init_strategy:
    type: init
    path: runs/my_run/checkpoints/final
```

`init`, the only `type` so far, loads the checkpoint's weights onto the pretrained weights of `model.name`: training
starts at step 0 with a fresh optimizer. The checkpoint must be in the layout the run saves, by `checkpoint.format` and
whether it trains LoRA (see the table above), and of the run's `model.name`, with the run's LoRA `r`, `alpha` and
`target_modules`.

It takes effect with or without `CheckpointCallback`, and a checkpoint that doesn't load fails `setup`.

---

## Running inside a service

A host builds one trainer with its callbacks and runs each job as:

```python
try:
    trainer.setup(config)  # validates, loads data, opens the device, builds the model and optimizer
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
- **Saving on your own schedule.** `trainer.save_checkpoint(path, metadata)` writes a checkpoint as the run saves
  them.

---

## Adding a model

A model provider builds the ttml model for an HF `model_type`, and loads and exports its weights. It subclasses
`ModelProvider` in `ttml/models/provider/base.py`, with the config as `self.config` and the model's HF `config.json`
as `self.hf_config`, and implements its four methods:

- `check()`: raise `UnsupportedTrainingConfig` for anything the model can't be built or trained with, including a
  `device.mesh` it can't be built on. It must not download weights or open the device.
- `build()`: the model, with the pretrained weights of `model.name`.
- `load_weights(model, directory)`: the weights `export_weights` wrote to `directory`, into a model `build` made.
- `export_weights(model, directory)`: reads the weights from the device and writes them to `directory`.

What a checkpoint holds and how it is laid out is the provider's: `LlamaProvider` writes the layouts in the table under
[Checkpoints](#checkpoints), with the shared helpers of `ttml/models/utils/` for PEFT adapters.

Write it in `ttml/models/provider/<model>.py` and add it to `providers()` in `ttml/models/provider/__init__.py`, keyed
by the `model_type`. `LlamaProvider` in `ttml/models/provider/llama.py` is the reference.

---

## Adding a dataset

A dataset provider serves one split of a dataset as batches: the trainer builds one for training and one for
validation, and asks each for its next batch. It subclasses `DatasetProvider` in `ttml/datasets/provider/base.py`,
which tokenizes the examples, masks the prompt, drops the examples longer than `max_length`, and batches them. The
subclass adds what is the dataset's own:

- `_load_raw_dataset()`: the HF dataset of `self.split`.
- `_prompt_and_response(example)`: the prompt and the response of one example.
- `train_val_split_ratio`, for a source with a train split only: the share of it to train on, the rest being held out
  for validation.
- `has_split(config, split)`, for a source that may lack a split: the trainer then runs without it.

Write it in `ttml/datasets/provider/<dataset>.py`, with its prompt templates in
`ttml/datasets/utils/templates/<dataset>.py`, and add it to `providers()` in `ttml/datasets/provider/__init__.py`,
keyed by its `dataset.id`. `SST2Provider` in `ttml/datasets/provider/sst2.py` is the reference.

What a trainer uses of a provider: `next_batch()`, the next batch of the epoch as host arrays, or `None` once the
epoch is over; iterating it, which takes the batches left in the epoch; `len()`, the batches of an epoch; and
`seek(epoch)`, which starts epoch `epoch`, in the order that epoch shuffles the train split in.

---

## Adding a trainer

A trainer subclasses `Trainer` in `ttml/unified_trainer/implementations/base.py`, which runs the training loop, and
implements its abstract methods:

- Setup: `_load_dataloaders()`, the train and validation batches (`None` without validation data);
  `_create_device_manager()`, which opens the device; `_load_model()`, the model with its pretrained weights and those
  of `checkpoint.init_strategy`; and `_load_optimizer()`, over the trainable weights.
- Steps: `_prepare_batch(batch)`, which moves a batch to the device; `_forward(batch)`, the batch's unscaled loss, in
  training and validation alike; `_backward(loss)`; `_accumulate_loss(running, loss)`, a running sum of losses,
  detached from the graph; `_clip_gradients(max_norm)`; `_optimizer_step()`, which also zeroes the gradients; and
  `_zero_grad()`.
- Modes: `_train_mode()`, `_eval_mode()`, and `_no_grad()`, a context manager that disables gradient tracking.
- For callbacks and hosts: `loss_to_float(loss)`, `parameter_counts()`, `current_learning_rate()` and
  `save_checkpoint(path, metadata)`, as [Running inside a service](#running-inside-a-service) describes.

It sets `config_class` to its config, a subclass of `TrainerConfig`, which a recipe for it validates as.

Write it in `ttml/unified_trainer/implementations/<trainer>.py` and add it to `TRAINERS` in
`ttml/unified_trainer/implementations/__init__.py`, keyed by the name `--trainer` takes. `SFTTrainer` in
`ttml/unified_trainer/implementations/sft.py` is the reference.
