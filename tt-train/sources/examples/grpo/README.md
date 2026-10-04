# Single-process GRPO examples

Single-rank GRPO training with `GRPOTrainer` on Tenstorrent devices.
Both the training forward/backward and the rollout token generation
run inside one ttml process on a single device mesh. For a two-rank
deployment that offloads rollout to a peer `tt-transformers` rank,
see [`../grpo_remote_rollout/boolq/`](../grpo_remote_rollout/boolq/).

The trainer API and its rank-agnostic conventions are documented in
[`tt-train/docs/GRPO_TRAINER.md`](../../../docs/GRPO_TRAINER.md); this
directory holds the concrete scripts and per-run notes.

---

## Examples

### Training

[`boolq_training_example.py`](boolq_training_example.py) — trains a
Llama or Qwen3 policy on `google/boolq` with a Yes/No correctness
reward and a `GRPOMonitor` callback that writes per-step CSV metrics.

Default (Llama-3.2-1B on a single device, config
[`grpo_boolq_llama_1b_1dev.yaml`](../../../configs/training_configs/grpo_boolq_llama_1b_1dev.yaml)):

```bash
python3 boolq_training_example.py \
    --config tt-train/configs/training_configs/grpo_boolq_llama_1b_1dev.yaml
```

To use a wider DDP config, pass a different `--config` (e.g. 4-device
[`grpo_boolq_llama_1b_ddp_4dev.yaml`](../../../configs/training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml)
or 32-device
[`grpo_boolq_llama_1b_ddp_32dev.yaml`](../../../configs/training_configs/grpo_boolq_llama_1b_ddp_32dev.yaml)):

```bash
python3 boolq_training_example.py \
    --config tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml
```

To train **Qwen3 32B sharded across all 32 Galaxy cards with FSDP**, pick the
Qwen3 config:

```bash
python3 boolq_training_example.py \
    --config ${TT_METAL_RUNTIME_ROOT}/tt-train/configs/training_configs/grpo_boolq_qwen3_32b_fsdp.yaml
```

Notes:

- The model family comes from `model_type` in the config's model yaml
  (`"llama"` or `"qwen3"`), and the runner (gradient checkpointing or not)
  from its `runner_type`. The configs set `rollout_source: "ttml"` in
  `grpo_config`, so `GRPOTrainer` builds a `TTMLRolloutSampler` for that model;
  for Qwen3 FSDP see [FSDP](../../../docs/GRPO_TRAINER.md#fsdp) in the trainer doc.
- `model_source` in the training config selects the HuggingFace ID or local
  path (default: `meta-llama/Llama-3.2-1B-Instruct`).
- **WandB logging**: set `report_to: wandb` (and optionally `run_name`) in
  `grpo_config`; project / entity / mode come from the `WANDB_PROJECT` /
  `WANDB_ENTITY` / `WANDB_MODE` env vars.

### Accuracy Evaluation

[`boolq_accuracy_example.py`](boolq_accuracy_example.py) — evaluates
a model on the BoolQ validation set with greedy decoding
(`temperature=0`) and writes per-question results to CSV. Runs on 1
device (p150) with `PROMPTS_TO_VALIDATE=20` by default; see
[`boolq_accuracy_example.yaml`](boolq_accuracy_example.yaml) for the
device / transformer config.

```bash
python3 boolq_accuracy_example.py
```

To evaluate a fine-tuned checkpoint, change `MODEL_ID` at the top of
the script to the directory containing `model.safetensors`.

A companion plotting helper
([`boolq_plot_example.py`](boolq_plot_example.py)) turns the
per-step CSV from `GRPOMonitor` into a training curve.

---

## Device Config

`TTMLRolloutSampler` (built by `GRPOTrainer`) opens the trainer's
device mesh from the `device_config:` block of the training YAML,
wrapped in a `DeviceConfig` object (defined in
[`ttml/common/config.py`](../../ttml/ttml/common/config.py)):

```yaml
device_config:
  enable_ddp: true
  mesh_shape: [1, 2]       # [rows, cols] of the device mesh
```

| Field         | Type              | Default   | Description                                                                                                                                                              |
| ------------- | ----------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `enable_ddp`  | `bool`            | `false`   | Enable distributed data-parallel training across the trainer's mesh.                                                                                                     |
| `enable_fsdp` | `bool`            | `false`   | Enable fully-sharded data parallel. Supported for Qwen3 by `TTMLRolloutSampler`; shards params / grads / optimizer state across the `"fsdp"` mesh axis.                |
| `mesh_shape`  | `list[int]`       | `[1, 1]`  | Shape of the device mesh `[rows, cols]`. Total devices = `rows * cols`.                                                                                                  |
| `device_ids`  | `list[int]`       | `null`    | Specific device IDs to use (default: auto-select).                                                                                                                       |

Device setup (`enable_fabric`, `open_device`,
`initialize_parallelism_context`) is performed inside the
`TTMLRolloutSampler` constructor, not the trainer. FSDP-specific behavior — including the
requirement that `checkpointing: false` — is documented in the
[FSDP subsection](../../../docs/GRPO_TRAINER.md#fsdp) of the trainer
doc.

---

## See also

- [`tt-train/docs/GRPO_TRAINER.md`](../../../docs/GRPO_TRAINER.md)
  — model- and rank-agnostic `GRPOTrainer` API.
- [`../grpo_remote_rollout/boolq/`](../grpo_remote_rollout/boolq/)
  — two-rank MPI deployment with `tt-transformers` rollout.
