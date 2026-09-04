# GRPO Trainer

Group Relative Policy Optimization (GRPO) trainer for reinforcement learning from
human/automated feedback on Tenstorrent devices.
The API follows [TRL's GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)
conventions where possible so that users familiar with TRL face minimal friction.

---

## Quick Start

```python
from datasets import load_dataset
from ttml.trainers import GRPOConfig, GRPOTrainer

# 1. A GRPOCompleter handles model loading, text generation, and
#    forward passes. The trainer is agnostic to which one you pass.
#    See "GRPOCompleter" below for the contract.
completer = MyCompleter(...)

# 2. Dataset must have a "prompt" column. All other columns are
#    forwarded by name to the reward function.
dataset = load_dataset("...", split="train").map(format_example)

# 3. Reward function. The trainer matches parameter names to
#    available data; declare **kwargs to receive everything.
def my_reward(completions, answer, **kwargs):
    return [2.0 if c.strip().lower().startswith(a) else -1.0
            for c, a in zip(completions, answer)]

# 4. Train — the trainer auto-appends a `GRPOMonitor` from `GRPOConfig`, so
#    `output_dir/grpo_metrics.csv` and per-step console lines come for free.
trainer = GRPOTrainer(
    completer=completer,
    dataset=dataset,
    config=GRPOConfig(
        epsilon=0.2,
        per_device_train_batch_size=8,
        num_iterations=1,
        gradient_accumulation_steps=4,
        prompts_to_train=1600,
        logging_steps=1,
        # Logging surface (all optional):
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",                # or "wandb" — the built-in GRPOMonitor calls wandb.init() for you
        run_name=None,                   # optional wandb run name; project/entity/mode come from WANDB_* env vars
    ),
    reward_func=my_reward,
    optimizer_dict={"type": "MorehAdamW", "lr": 5.0e-6},
    callbacks=[],                        # optional; see "Callbacks" below
    model_source="...",                  # used only for HF config in checkpoints
)
trainer.train()
```

`GRPOTrainer` is agnostic to model architecture, device topology, and
rank count. It only calls `completer.generate(...)`,
`completer.compute_nlog_probs(...)`, and the standard
`TrainerCallback` hooks. Where generation runs — in-process on the
same mesh as the policy, or on a peer MPI rank — is the completer's
choice. For a worked-out two-rank deployment (separate trainer and
inference ranks, weight push every step), see the
[BoolQ example](../sources/examples/grpo_remote_rollout/boolq/README.md).

---

## Architecture

GRPO training is split into two components:

- **`GRPOCompleter`** — abstract base class that handles model-specific concerns:
  model loading, device setup, text generation, and forward passes for log-prob
  computation.
- **`GRPOTrainer`** — model-agnostic training loop that drives reward computation,
  advantage estimation, and policy gradient updates.

This separation means the trainer does not need to know anything
about model architecture, device topology, or rank count. The trainer
only calls `completer.generate(...)`, `completer.compute_nlog_probs(...)`,
and the standard `TrainerCallback` hooks; whether generation runs
in-process on the same mesh as the policy or on a peer MPI rank is
the completer's choice. To support a new model family, implement a
new `GRPOCompleter` subclass (see [GRPOCompleter](#grpocompleter)).

---

## GRPOCompleter

```python
from ttml.trainers import GRPOCompleter
```

Abstract base class for model-specific completion engines. Subclass this for each
model architecture (Llama, Qwen, etc.).

### Required properties

| Property | Type | Description |
|----------|------|-------------|
| `tokenizer` | any | The tokenizer used by this completion engine. |
| `model` | any | The underlying tt model used for forward passes and optimization. |

### Required methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `(prompts: List[List[int]]) -> List[List[int]]` | Generate completions for a batch of tokenised prompts. |
| `generate_str` | `(prompt_strs: List[str]) -> List[str]` | Generate completions from string prompts, returning decoded strings. |
| `compute_nlog_probs` | `(prompts, completions) -> (nlog_probs, mask)` | Compute per-token negative log probabilities for prompt+completion pairs. |

The detailed API contract can be found in `tt-train/sources/ttml/ttml/trainers/grpo_trainer.py`.

### Available completer implementations

Three concrete completers ship today:

- `LlamaGRPOCompleter`
  ([`sources/examples/grpo/utils/llama_completer.py`](../sources/examples/grpo/utils/llama_completer.py))
  — single-process Llama; owns its own mesh via `setup_device`.
- `Qwen3GRPOCompleter`
  ([`sources/examples/grpo/utils/qwen3_completer.py`](../sources/examples/grpo/utils/qwen3_completer.py))
  — single-process Qwen3 with FSDP (see below).
- `LlamaCompleterRemoteRollout`
  ([`sources/examples/grpo_remote_rollout/utils/llama_grpo_completer.py`](../sources/examples/grpo_remote_rollout/utils/llama_grpo_completer.py))
  — two-rank Llama; receives an already-opened mesh and delegates
  generation to a peer MPI rank. Documented alongside the
  [BoolQ example](../sources/examples/grpo_remote_rollout/boolq/README.md).

### Qwen3GRPOCompleter

```python
from utils.qwen3_completer import Qwen3GRPOCompleter, Qwen3CompletionCtx
```

Qwen3-specific implementation of `GRPOCompleter`. Drives the pure-Python ttml
Qwen3 model (`ttml.models.qwen3.Qwen3`) and shards it across the `"fsdp"` mesh
axis with `ttml.fsdp.fully_shard`. The model architecture is read from the
HuggingFace config of `model_source`; only `max_sequence_length` is taken from
`transformer_config` (to bound the generation horizon).

```python
from ttml.common.config import DeviceConfig

completer = Qwen3GRPOCompleter(
    ctx=Qwen3CompletionCtx(
        max_tokens_to_complete=256,
        temperature=1.0,
        completions_per_prompt=8,
    ),
    transformer_config=transformer_config,   # max_sequence_length only
    device_config=DeviceConfig(
        {"device_config": {"enable_fsdp": True, "mesh_shape": [32, 1]}}
    ),
    model_source="Qwen/Qwen3-32B",
)
```

`DeviceConfig` is defined in
[`ttml/common/config.py`](../sources/ttml/ttml/common/config.py); its
constructor accepts either a full YAML dict (with a top-level
`device_config:` block) or a path to a YAML file. In practice the
BoolQ script loads a training YAML and passes the raw dict, i.e.
`DeviceConfig(raw)` (see
[`boolq_training_example.py`](../sources/examples/grpo/boolq_training_example.py)).

Unlike the Llama completer, `setup_device` opens a **named** mesh via
`ttml.open_device_mesh` so an `"fsdp"` axis exists. By default
(`lazy_parameter_init=True`) the model is built lazily, each block plus the root
model is wrapped with `fully_shard`, the parameters are materialized
already-sharded, and the HuggingFace weights are then streamed in sharded (the
full unsharded model is never materialized on one chip). With
`lazy_parameter_init=False` it instead loads the (still replicated) weights
first and then wraps with `fully_shard`. Either way, parameters, gradients, and
optimizer state end up sharded `1/N` across the FSDP axis.

---

## GRPOConfig

`GRPOConfig` is a dataclass that controls the GRPO training loop.

```python
from ttml.trainers import GRPOConfig
```

| Parameter | Type | Default | TRL equivalent | Description |
|-----------|------|---------|----------------|-------------|
| `per_device_train_batch_size` | `int` | — | `per_device_train_batch_size` | Number of completions processed on a **single device** within one micro-batch. The across-mesh micro-batch holds `per_device_train_batch_size * num_devices` completions and always shards evenly along axis 0. The per-microbatch prompt count is **derived** as `per_device_train_batch_size * num_devices / num_generations`. |
| `num_generations` | `int` | — | `num_generations` | Number of completions generated per prompt. Each prompt produces this many candidate responses for reward scoring. |
| `max_completion_length` | `int` | — | `max_completion_length` | Maximum number of tokens to generate per completion. |
| `gradient_accumulation_steps` | `int` | — | `gradient_accumulation_steps` | Number of micro-batches per generation (effective) batch. Each generation batch generates `gradient_accumulation_steps` times the per-micro-batch completions, and the trainer accumulates gradients over that many micro-batches before a single optimizer step. Effective batch size (in completions) = `per_device_train_batch_size * num_devices * gradient_accumulation_steps`. |
| `num_iterations` | `int` | — | `num_iterations` | Number of training passes over each batch of completions (mini-epochs). |
| `epsilon` | `float` | — | `epsilon` | Clipping parameter for the GRPO surrogate loss (analogous to PPO clip range). |
| `prompts_to_train` | `int` | — | *(use `max_steps`)* | Total number of prompts to train on. Unlike TRL which uses `max_steps`, this directly specifies the data budget. Equivalent to `max_steps * (per_device_train_batch_size * num_devices * gradient_accumulation_steps / num_generations)`. **Currently, `prompts_to_train` must be divisible by the generation batch size in prompts (`per_device_train_batch_size * num_devices * gradient_accumulation_steps / num_generations`) to avoid a ragged final batch.** |
| `temperature` | `float` | — | `temperature` | Sampling temperature for completion generation. |
| `warmup_steps` | `int` | — | `warmup_steps` | Number of linear learning rate warmup steps. |
| `output_dir` | `str` | — | `output_dir` | Directory for logs, metrics CSV, and checkpoints. |
| `checkpointing` | `bool` | — | *(use `save_steps`)* | Whether to save checkpoints during training. |
| `checkpoint_interval` | `int` | — | *(use `save_steps`)* | Save a checkpoint every *N* optimizer steps (when `checkpointing=True`). |
| `logging_steps` | `int` | — | `logging_steps` | Cadence (in optimizer steps) at which the built-in `GRPOMonitor` emits a CSV row / wandb log. All callbacks fire every step; metrics are accumulated every step and emitted every *N* steps as an interval-mean (except `min_completion_len`, `max_completion_len`, `lr` where minimum, maximum, last value is emitted respectively). Set to `0` or negative to disable emission. |
| `log_completions` | `bool` | `False` | `log_completions` | When `True`, the built-in [`GRPOMonitor`](#built-in-grpomonitor) prints a sample of `(prompt, completion, reward)` triples every logging step (and, if `report_to == "wandb"`, logs them as a `wandb.Table`). |
| `num_completions_to_print` | `int` | `0` | `num_completions_to_print` | Upper bound on how many completions per step the monitor prints when `log_completions=True`. `0` disables the sample dump even when `log_completions` is set (a warning is emitted). |
| `report_to` | `str` | `"none"` | `report_to` | Where the built-in monitor forwards scalar metrics and the optional completions table. Only `"none"` and `"wandb"` are accepted in this framework (TRL takes a list; here it must be a single string). When `"wandb"` is set, the built-in `GRPOMonitor` calls `wandb.init(...)` itself in `on_train_begin` (matching the TRL / `transformers` `WandbCallback` convention); `project` / `entity` / `mode` come from the `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_MODE` env vars, and `name` from `run_name` below. If the caller already opened a wandb run before constructing the trainer, the monitor logs into it and leaves its lifecycle alone. |
| `run_name` | `str \| None` | `None` | `run_name` | Optional wandb run name, forwarded to `wandb.init` when the monitor opens the run. `None` lets wandb auto-generate one. |
| `disable_default_monitor` | `bool` | `False` | *(no direct equivalent)* | Opt out of the auto-appended `GRPOMonitor`. Use this if you need a fully custom logging pipeline; you can still pass your own callbacks via the trainer constructor. |

---

## GRPOTrainer

```python
from ttml.trainers import GRPOTrainer
```

### Constructor

```python
GRPOTrainer(
    completer,
    dataset,
    config,
    reward_func,
    optimizer_dict,
    callbacks=None,
    model_source=None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `completer` | `GRPOCompleter` | A model-specific completion engine (e.g. `LlamaGRPOCompleter`, `Qwen3GRPOCompleter`, or `LlamaCompleterRemoteRollout`). Responsible for generation, forward passes, and device setup. |
| `dataset` | `Dataset` | HuggingFace `datasets.Dataset` with at least a `"prompt"` column. All other columns are passed to the reward function. |
| `config` | `GRPOConfig` | Training configuration (see above). |
| `reward_func` | `Callable` | Reward function. Receives decoded completions and any dataset columns (see [Reward Functions](#reward-functions)). |
| `optimizer_dict` | `dict` | Optimizer config dict passed to the [ttml optimizer registry](TTML_ONBOARDING.md). Must include a `"type"` key. |
| `callbacks` | `list[TrainerCallback] \| None` | Hooks into the training loop (see [Callbacks](#callbacks)). |
| `model_source` | `str \| None` | HuggingFace model ID or local path. Used only for saving HF config in checkpoints. |

### Methods

| Method | Description |
|--------|-------------|
| `train()` | Run the full GRPO training loop. Handles generation, reward computation, policy gradient updates, and checkpointing. |

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
> and sums their outputs, `GRPOTrainer` takes a single `reward_func`. To combine
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

## Built-in GRPOMonitor

`ttml.trainers.GRPOMonitor` is the framework's default step logger. `GRPOTrainer`
auto-appends one to its callback list unless `GRPOConfig.disable_default_monitor`
is set, so you do not need to construct one yourself.

At `on_train_begin` it creates `<output_dir>/` if needed and snapshots the
callback classes present at that moment, but does **not** write the CSV
header yet. On the first logging step, `GRPOMonitor` derives the column list
from `trainer.metrics` at that moment and writes the header line together
with the first data row (both are committed to `<output_dir>/grpo_metrics.csv`
before the next step runs). The columns are a fixed base set —

```
step, reward_mean, reward_std, mean_completion_len, min_completion_len,
max_completion_len, lr, step_time_s, generation_time_s
```

— followed by one `<CallbackClassName>_time_s` column per callback present at
`on_train_begin` (**excluding `GRPOMonitor` itself**), followed by every extra
numeric key populated on `trainer.metrics` before the first logging step
fires. Once the header is on disk the column set is frozen for the rest of
the run; keys that first appear later are dropped with a one-time warning.

Each `<Callback>_time_s` column is the callback's total wall-clock cost for
the **current** step, accumulated across every hook the trainer fired for
that step (`on_before_optimizer_step`, `on_step_end`, and — on checkpoint
steps — `on_save`). The counter is reset at the top of every optimizer step,
and each entry is refreshed in the metrics dict immediately after that
callback's hook returns, so `GRPOMonitor` (which runs last, after
`step_time_s` is sealed) sees current-step totals for every other callback
when it writes the row.

`step_time_s` is the **total per-step wall time** — it covers generation, host
post-generation work, the reference log-probs pass, the training loop, every
non-monitor `on_step_end` callback, and any checkpoint save on this step (this
matches TRL's `step_time`, which is also the full step wall time including
generation). `GRPOMonitor`'s own cost is deliberately outside `step_time_s`
and is not written as a CSV column.

**Interval aggregation.** `GRPOMonitor.on_step_end` runs every optimizer step
and pushes every numeric scalar in `trainer.metrics` into constant-memory
running stats. It only emits a CSV row / wandb log once every `logging_steps`
steps, at which point each metric is aggregated as the interval-mean over the
window, then the running stats are cleared. Two policy exceptions apply:
`min_completion_len` and `max_completion_len` are emitted as the min / max
across the window (a mean of per-step mins is meaningless), and `lr` is
emitted as the current-step value (matching TRL's `_get_learning_rate`).
Non-numeric payloads (`prompts` / `completions` / `rewards`) always snapshot
the values from the step that triggered the emission. This matches TRL, which
also accumulates per-step samples and emits their mean at logging cadence.

When `report_to == "wandb"`, the monitor calls `wandb.init(...)` itself in
`on_train_begin` (matching the TRL / `transformers` `WandbCallback`
convention). `project` / `entity` / `mode` are read from the `WANDB_PROJECT` /
`WANDB_ENTITY` / `WANDB_MODE` env vars, `name` comes from `GRPOConfig.run_name`,
and the `config` payload is the full `GRPOConfig` (plus `model_source` and the
optimizer dict). If the caller already opened a run before constructing the
trainer, the monitor logs into that existing run and does **not** call
`wandb.finish()` at the end — the caller owns the lifecycle.

Every numeric scalar metric is forwarded to wandb under the `grpo/` namespace
(`grpo/reward_mean`, `grpo/step_time_s`, `grpo/EvalCallback_time_s`, ...). If
`log_completions=True` and `num_completions_to_print > 0`, the first *K*
`(prompt, completion, reward)` triples for the step are printed via
`logging.info` and also logged as a `wandb.Table` under `grpo/completions`.

If the `wandb` package is not installed, the monitor logs a one-time warning
and quietly falls back to console + CSV only.

If you have a pre-existing custom step logger (from before this framework
absorbed `GRPOMonitor`), the auto-append is skipped for any callback that is
either a subclass of `ttml.trainers.GRPOMonitor` or whose class is named
`GRPOMonitor`. If your custom logger uses a different class name, set
`GRPOConfig.disable_default_monitor=True` to opt out of the built-in one
and avoid duplicate CSV writes.

## Callbacks

Subclass `TrainerCallback` and override any hooks you need. In practice most
users write a small callback that computes an auxiliary metric and injects it
into the CSV — like the reverse-text example's eval callback:

```python
from difflib import SequenceMatcher
from ttml.trainers import TrainerCallback

class EvalCallback(TrainerCallback):
    def __init__(self, completer, ctx, dataset, num_examples):
        rows = dataset.select(range(min(num_examples, len(dataset))))
        self.completer = completer
        self.ctx = ctx
        self.prompts = list(rows["prompt"])
        self.answers = list(rows["answer"])

    def on_step_end(self, trainer, step, **kwargs):
        similarity = self._greedy_similarity()
        # `trainer.metrics` is a mutable dict populated by the trainer for
        # this step. Any scalar you write here lands in the CSV row that the
        # built-in GRPOMonitor writes AFTER this callback runs.
        trainer.metrics["eval_similarity"] = similarity

trainer = GRPOTrainer(..., callbacks=[EvalCallback(...)])   # GRPOMonitor is auto-added
```

Two contracts are worth calling out:

- **`trainer.metrics` is mutable and shared**: callbacks earlier in
  `trainer.callbacks` can add scalar columns by writing into it, and any
  callback later in the list (including the built-in `GRPOMonitor`) will read
  the merged view. The dict is rebuilt at the top of every optimizer step, so
  writes do not leak between steps; `GRPOMonitor` accumulates values across
  steps in its own running-stats state.
- **Column-set freeze**: `GRPOMonitor` writes the CSV header on the first
  logging step, deriving the columns from `trainer.metrics` at that moment.
  Any key populated by the trainer or by a callback's `on_step_end` before
  the first logging step fires will appear as a column. Keys that first
  appear after that step are dropped with a one-time warning — the schema is
  not churned mid-run.

| Hook | Signature | When |
|------|-----------|------|
| `on_train_begin` | `(trainer)` | Before the first batch. |
| `on_step_end` | `(trainer, step, **kwargs)` | Every optimizer step (matches the base `TrainerCallback.on_step_end` contract; not gated by `logging_steps`). Keyword args are the current step's `trainer.metrics`: `reward_mean`, `reward_std`, `mean_completion_len`, `min_completion_len`, `max_completion_len`, `lr`, `generation_time_s`, and one `<CallbackClassName>_time_s` per non-monitor callback (accumulated for the current step across every hook fired so far — a callback later in the list sees the current-step total for every callback before it). `step_time_s` is only present for `GRPOMonitor`, which runs after the timer is sealed; other callbacks see `trainer.metrics` without it. When `log_completions=True`, truncated `prompts` / `completions` / `rewards` lists are also passed. The built-in `GRPOMonitor` accumulates these values across every step and only emits a CSV / wandb row every `logging_steps` steps (see [Built-in GRPOMonitor](#built-in-grpomonitor)). |
| `on_before_optimizer_step` | `(trainer)` | After gradient accumulation, before `optimizer.step()`. |
| `on_save` | `(trainer, step, path)` | After a checkpoint is saved. `path` is the checkpoint directory. |
| `on_train_end` | `(trainer)` | After the final batch. |

The only built-in logging is the auto-added `GRPOMonitor`; other CSV writing,
progress bars, or dashboards belong in additional callbacks.

> **Cross-rank weight transfer**: a completer that runs generation on
> a peer MPI rank can use a `TrainerCallback` to push freshly-updated
> policy weights to the peer after each optimizer step. The trainer
> itself does not know about this — it just fires `on_step_end`. See
> the [BoolQ example](../sources/examples/grpo_remote_rollout/boolq/README.md)
> for the shipped pattern (`WeightSyncCallback` + `MPIRolloutClient`).

---

## Transformer Config

Model architecture parameters passed as a plain dict. In YAML configs, the
transformer config can be provided inline under `transformer_config:` or as a
path to a separate file via `transformer_config_path:`. The path may use
`${TT_METAL_RUNTIME_ROOT}` (recommended, matching other tt-train configs) or
be a plain absolute/relative path; relative paths are resolved against the
YAML file's directory. The external file must contain a top-level
`transformer_config` mapping. See
[`configs/model_configs/`](../configs/model_configs/) for available model
config files.

```yaml
# Option 1: reference an external model config file
transformer_config_path: "${TT_METAL_RUNTIME_ROOT}/tt-train/configs/model_configs/llama3_2_1B.yaml"

# Option 2: inline (still supported)
transformer_config:
  model_type: "llama"
  num_heads: 32
  ...
```

---

## Optimizer Config

Optimizer parameters passed as a dict with a `"type"` key. Forwarded to the
ttml optimizer registry:

```python
optimizer_dict = {
    "type": "MorehAdamW",
    "lr": 5.0e-6,
    "beta1": 0.9,
    "beta2": 0.99,
    "epsilon": 1.0e-8,
    "weight_decay": 0.01,
}
```

Any optimizer registered with `ttml.optimizers.register_optimizer` can be used.
See [TTML Onboarding — Optimizers](TTML_ONBOARDING.md) for the full list of
built-in optimizers.

---

## Device Config

The abstract `GRPOCompleter` and `GRPOTrainer` impose no
`device_config` of their own — concrete completers decide how (and
whether) to consume one. Either way, the mesh is configured from a
YAML training config (`device_config:` block) applied by your
entrypoint:

```yaml
device_config:
  mesh_shape: [1, 2]
  enable_ddp: true
```

**In-process completers (`LlamaGRPOCompleter`, `Qwen3GRPOCompleter`)**
accept a `DeviceConfig` object (see
[`ttml/common/config.py`](../sources/ttml/ttml/common/config.py))
and open the mesh themselves inside `setup_device`. The entrypoint
just constructs it from the loaded YAML dict:

```python
from ttml.common.config import DeviceConfig

raw = load_config(config_path)
device_config = DeviceConfig(raw)
completer = LlamaGRPOCompleter(..., device_config=device_config)
```

**Remote-rollout completer (`LlamaCompleterRemoteRollout`)** instead
receives an already-opened `mesh_device`; the entrypoint opens it
before construction:

```python
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(*device_config.mesh_shape), ...
)
completer = LlamaCompleterRemoteRollout(
    ..., mesh_device=mesh_device, enable_ddp=device_config.enable_ddp
)
```

### FSDP

When the completer opens a named mesh with an `"fsdp"` axis (size > 1), the
`GRPOTrainer` automatically:

1. Slices each micro-batch across the whole mesh (dim 0): the across-mesh
   micro-batch is `per_device_train_batch_size * num_devices` completions, with
   `per_device_train_batch_size` landing on each device.
   `per_device_train_batch_size * num_devices` must be divisible by
   `num_generations` so each prompt's GRPO group stays intact within the batch.
2. Synchronizes gradients with `ttml.sync_gradients(params, axis_names=("dp", "fsdp"))`
   each optimizer step. FSDP-managed parameters skip the `"fsdp"` axis (their
   gradients were already reduce-scattered by the FSDP backward hook); any
   replicated parameter is all-reduced across the axis.

Checkpointing is unsupported under FSDP (the checkpoint would store per-rank
shards rather than full tensors) — set `checkpointing: false`.

---

## DDP / Multi-device

When the YAML config sets `enable_ddp: true` and `mesh_shape: [1, N]`,
the policy is replicated across the N chips of the trainer's mesh and
data parallelism is applied within that mesh:

1. The completer initialises ttml's parallelism context against the
   already-opened mesh.
2. Input tensors are sharded across the N chips along the batch
   dimension.
3. Gradients are synchronized via
   `ttml.core.distributed.synchronize_gradients` before each
   optimizer step.

`per_device_train_batch_size` specifies the number of completions on
a **single device** per micro-batch. The whole mesh therefore
processes `per_device_train_batch_size * total_devices` completions
per micro-batch, and the per-micro-batch prompt count is derived as
`per_device_train_batch_size * total_devices / num_generations`.

This section describes only the trainer's own mesh — the in-process
data parallelism that the trainer drives. If your completer also runs
generation on a peer MPI rank, the topology over there is independent
of `GRPOTrainer` and lives in the completer / its example doc.

> Sharded TP / CP is not exercised by `GRPOTrainer` today; this
> section assumes replicated parameters with batch-dim sharding.

---

## Checkpointing

When `checkpointing=True`, the trainer saves a full checkpoint every
`checkpoint_interval` optimizer steps into `output_dir/checkpoints/grpo_step_{step}/`.

Each checkpoint directory contains:

| File | Contents | Source |
|------|----------|--------|
| `model.safetensors` | Model weights in safetensors format | `model.parameters()` exported as float32 numpy arrays |
| `config.json` | HuggingFace model configuration | `AutoConfig.from_pretrained(model_source)` |
| `tokenizer_config.json` | Tokenizer configuration | `tokenizer.save_pretrained()` |
| `tokenizer.json` | Full tokenizer (vocabulary, merges, etc.) | `tokenizer.save_pretrained()` |
| `generation_config.json` | Generation parameters (temperature, max tokens, special token IDs) | Built from `GRPOConfig` and tokenizer |
| `trainer_state.json` | Training progress (global step, learning rate) | Current optimizer step and LR |
| `scheduler.pt` | Learning rate scheduler state (base LR, warmup config, step) | `torch.save()` |
| `rng_state.pth` | Python, NumPy, and PyTorch RNG states for reproducibility | `torch.save()` |
| `training_args.bin` | Full `GRPOConfig` dataclass serialized as a dict | `torch.save(dataclasses.asdict(grpo_config))` |
| `timestamp.txt` | UTC timestamp of when the checkpoint was saved | `datetime.now(timezone.utc)` |

To resume from a checkpoint, point `model_source` at the checkpoint directory
(it contains `model.safetensors` and the tokenizer files).

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

| Aspect | TRL `GRPOTrainer` | ttml `GRPOTrainer` |
|--------|-------------------|---------------------|
| **Model** | Passed as a `transformers` model object | Built by a `GRPOCompleter` (e.g. `LlamaGRPOCompleter`, `Qwen3GRPOCompleter`, or `LlamaCompleterRemoteRollout`) from a HF ID or local path |
| **Reward functions** | List of functions (`reward_funcs=[f1, f2]`), summed | Single function (`reward_func=f`) |
| **Training budget** | `max_steps` (optimizer steps) | `prompts_to_train` (total prompts) |
| **Optimizer** | String name (`optim="adamw_bnb_8bit"`) | Config dict (`{"type": "MorehAdamW", ...}`) |
| **Device setup** | Handled by HF Accelerate | Caller opens the mesh from a YAML `device_config:` block and hands it to the completer |
| **KL penalty** | `beta` parameter | Not implemented (equivalent to `beta=0.0`) |
| **Callbacks** | HF `TrainerCallback` with `on_log(args, state, control, logs)` | `TrainerCallback` with `on_step_end(trainer, step, **kwargs)` |
| **`report_to`** | List of tracker names (`"wandb"`, `"tensorboard"`, `"trackio"`, ...) | Single string, only `"none"` or `"wandb"` accepted (no list form) |

---

## Examples

Two BoolQ examples ship today. Both train the same policy on the
same dataset (`google/boolq`, Yes/No correctness reward) with the
same `GRPOTrainer`; they differ in where token generation runs.

### Single-process, ttml-only

- [`tt-train/sources/examples/grpo/`](../sources/examples/grpo/)
  — **Single-process, ttml-only.** Both the training forward/backward
  and the rollout token generation run inside the same ttml process on
  one device mesh. The completer (`LlamaGRPOCompleter` or
  `Qwen3GRPOCompleter` under
  [`utils/`](../sources/examples/grpo/utils/)) owns the ttml policy
  model and drives generation itself. Entry point:
  [`boolq_training_example.py`](../sources/examples/grpo/boolq_training_example.py)
  (`--model llama` or `--model qwen3`, optional `--config <yaml>`).
  Also ships an accuracy-eval sibling
  ([`boolq_accuracy_example.py`](../sources/examples/grpo/boolq_accuracy_example.py))
  and a plotting helper
  ([`boolq_plot_example.py`](../sources/examples/grpo/boolq_plot_example.py)).

Each task lives in its own subdirectory; the model-specific completers in
[`utils/`](utils/) are shared between them.

#### BoolQ Training

[`boolq/boolq_training_example.py`](boolq/boolq_training_example.py) — trains
Llama-3.2-1B-Instruct on BoolQ using `GRPOTrainer` with a custom reward
function, CSV logging via the framework's built-in `GRPOMonitor`, and DDP on 2 devices.

```bash
python3 boolq/boolq_training_example.py
```

To train Qwen3 32B sharded across all 32 galaxy cards with FSDP:

```bash
python3 boolq/boolq_training_example.py --model qwen3 \
    --config ${TT_METAL_RUNTIME_ROOT}/tt-train/configs/training_configs/grpo_boolq_qwen3_32b_fsdp.yaml
```

#### BoolQ Accuracy Evaluation

[`boolq/boolq_accuracy_example.py`](boolq/boolq_accuracy_example.py) — evaluates a
model on the BoolQ validation set with greedy decoding (`temperature=0`)
and writes per-question results to a CSV. Runs on 1 device (p150) with
`PROMPTS_TO_VALIDATE=20` by default.

```bash
python3 boolq/boolq_accuracy_example.py
```

To evaluate a fine-tuned checkpoint, change `MODEL_ID` to the directory
containing `model.safetensors`.

#### Reverse Text Training

[`reverse_text/reverse_text_training_example.py`](reverse_text/reverse_text_training_example.py) —
trains Qwen3-0.6B to reverse text character-by-character on a single p150,
ported from the prime-rl / verifiers TRL example. Rewards the similarity ratio
between the text in the `<reversed_text>` tags and the true reversal, and runs a
greedy eval on a held-out split every step.

```bash
python3 reverse_text/reverse_text_training_example.py
```

#### Plotting

[`boolq/boolq_plot_example.py`](boolq/boolq_plot_example.py) — plots any column of the `grpo_metrics.csv` written by the built-in `GRPOMonitor`.

```bash
python3 boolq/boolq_plot_example.py <output_dir>/grpo_metrics.csv reward_mean
```

[`reverse_text/reverse_text_plot_example.py`](reverse_text/reverse_text_plot_example.py) —
plots every reverse-text metric (reward, the three eval scores, completion
length, and the step / generation times) as one grid. With no arguments it picks
the newest run under `generated/tt-train/grpo_reverse_text_run/` and writes
`grpo_metrics.png` beside its CSV.

```bash
python3 reverse_text/reverse_text_plot_example.py
```

### Two-rank MPI, ttml + tt-transformers

- [`tt-train/sources/examples/grpo_remote_rollout/boolq/`](../sources/examples/grpo_remote_rollout/boolq/)
  — **Two-rank MPI, ttml + tt-transformers.** Rollout generation is
  offloaded to a peer rank running `tt-transformers.Transformer`
  inside a captured ttnn trace (much faster than ttml decode). Rank 0
  runs the ttml policy and `GRPOTrainer`; rank 1 runs generation
  workers.

Both examples plug into `GRPOTrainer` through the same
`GRPOCompleter` abstraction — the trainer itself does not know which
of the two deployments it's in.

---

## Environment Variables

Set these before running:

- `TT_METAL_RUNTIME_ROOT` — path to the tt-metal repository root.
- `HF_TOKEN` — HuggingFace token for gated model access.
- `TT_MESH_GRAPH_DESC_PATH` — path to the mesh graph descriptor.
