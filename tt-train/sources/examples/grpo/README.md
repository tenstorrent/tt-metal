# GRPO examples

GRPO training with `GRPOTrainer` on Tenstorrent devices. Where rollouts are
generated is a config choice, not a different script:

- `rollout_backend: "ttml_in_process_sync"` — training and
  rollout generation run in one ttml process on one device mesh.
- `rollout_backend: "ttt_remote_sync"` — the same script
  runs on two tt-run ranks; rank 1 generates with `tt-transformers`. See
  [Remote rollout (two ranks)](#remote-rollout-two-ranks).

The trainer API and its rank-agnostic conventions are documented in
[`tt-train/docs/GRPO_TRAINER.md`](../../../docs/GRPO_TRAINER.md); this
directory holds the concrete scripts and per-run notes.

---

## Examples

### Training

[`boolq/boolq_training_example.py`](boolq/boolq_training_example.py) — trains a
Llama or Qwen3 policy on `google/boolq` with a Yes/No correctness
reward and a `GRPOMonitor` callback that writes per-step CSV metrics.

Default (Llama-3.2-1B on a single device, config
[`grpo_boolq_llama_1b_1dev.yaml`](../../../configs/training_configs/grpo_boolq_llama_1b_1dev.yaml)):

```bash
python3 boolq/boolq_training_example.py \
    --config tt-train/configs/training_configs/grpo_boolq_llama_1b_1dev.yaml
```

To use a wider DDP config, pass a different `--config` (e.g. 4-device
[`grpo_boolq_llama_1b_ddp_4dev.yaml`](../../../configs/training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml)
or 32-device
[`grpo_boolq_llama_1b_ddp_32dev.yaml`](../../../configs/training_configs/grpo_boolq_llama_1b_ddp_32dev.yaml)):

```bash
python3 boolq/boolq_training_example.py \
    --config tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml
```

To train **Qwen3 32B sharded across all 32 Galaxy cards with FSDP**, pick the
Qwen3 config:

```bash
python3 boolq/boolq_training_example.py \
    --config ${TT_METAL_RUNTIME_ROOT}/tt-train/configs/training_configs/grpo_boolq_qwen3_32b_fsdp.yaml
```

Notes:

- The model family comes from `model_type` in the config's model yaml
  (`"llama"` or `"qwen3"`), and the runner (gradient checkpointing or not)
  from its `runner_type`. `GRPOTrainer` builds that model, and since these
  configs set `rollout_backend: "ttml_in_process_sync"` in `grpo_config`, it generates rollouts
  with it through a `TTMLRolloutSampler`; the remote config instead generates
  on a second rank (see [Remote rollout](#remote-rollout-two-ranks));
  for Qwen3 FSDP see [FSDP](../../../docs/GRPO_TRAINER.md#fsdp) in the trainer doc.
- `model_source` in the training config selects the HuggingFace ID or local
  path (default: `meta-llama/Llama-3.2-1B-Instruct`).
- **WandB logging**: set `report_to: wandb` (and optionally `run_name`) in
  `grpo_config`; project / entity / mode come from the `WANDB_PROJECT` /
  `WANDB_ENTITY` / `WANDB_MODE` env vars.

### Accuracy Evaluation

[`boolq/boolq_accuracy_example.py`](boolq/boolq_accuracy_example.py) — evaluates
a model on the BoolQ validation set with greedy decoding
(`temperature=0`) through a `TTMLRolloutSampler` on the model built by
`setup_ttml_model`, and writes per-question results to CSV. Runs on 1
device (p150) with `PROMPTS_TO_VALIDATE=20` by default; see
[`boolq/boolq_accuracy_example.yaml`](boolq/boolq_accuracy_example.yaml) for the
device / transformer config.

```bash
python3 boolq/boolq_accuracy_example.py
```

To evaluate a fine-tuned checkpoint, change `MODEL_ID` at the top of
the script to a `GRPOTrainer` checkpoint directory (the one containing
`checkpoint.pkl`).

A companion plotting helper
([`boolq/boolq_plot_example.py`](boolq/boolq_plot_example.py)) turns the
per-step CSV from `GRPOMonitor` into a training curve.

### GSM8K (Qwen3-0.6B, remote rollouts)

[`gsm8k/gsm8k_training_example.py`](gsm8k/gsm8k_training_example.py) — trains
the think-tagged SFT checkpoint `ichovpanTT/qwen3-0.6b-base-think-sft` on
`openai/gsm8k` with
[`grpo_gsm8k_qwen3_p6b_remote_rollout.yaml`](../../../configs/training_configs/grpo_gsm8k_qwen3_p6b_remote_rollout.yaml):
rank 0 trains the ttml Qwen3 policy on one chip, rank 1 generates with
`tt-transformers` on another. Prompts use the plain "Question:/Answer:" layout
the SFT checkpoint was trained on, not a chat template. The reward is the sum of
five signals, each logged as a `{name}_mean` CSV column: correctness of the
number in `<answer>`, tag counts, loose and strict `<think>`/`<answer>` format,
and an integer answer.

```bash
./tt-train/sources/examples/grpo/gsm8k/run_remote_rollout.sh
```

[`gsm8k/run_remote_rollout.sh`](gsm8k/run_remote_rollout.sh) uses the rank
bindings in [`gsm8k/configurations/split_1_1/`](gsm8k/configurations/split_1_1/)
(one P150 per rank) and takes the same `--config`, `--script`, `--hostfile` and
`--rank-bindings` overrides as the BoolQ launcher. The rollout rank boots with
dummy weights, so the checkpoint must be registered in tt-transformers'
`LOCAL_HF_PARAMS` (`qwen3-0.6b-base-think-sft` is).

[`gsm8k/eval_gsm8k.py`](gsm8k/eval_gsm8k.py) prints pass@k tables (correct
answer, tags present, tags exactly once, format regex) on the GSM8K test split
for one or more HuggingFace-format checkpoints, generating on tt-transformers
with `TTTRolloutSampler`:

```bash
cd tt-train/sources/examples/grpo/gsm8k
python3 eval_gsm8k.py --models Qwen/Qwen3-0.6B-Base=base ichovpanTT/qwen3-0.6b-base-think-sft=sft \
    --n-samples 16 --temperature 1.0 --limit 200
```

---

## Device Config

`GRPOTrainer` opens its device mesh from the `device_config:` block of
the training YAML,
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
| `enable_fsdp` | `bool`            | `false`   | Enable fully-sharded data parallel. Supported for Qwen3; shards params / grads / optimizer state across the `"fsdp"` mesh axis.                                     |
| `fsdp_replicate_params` | `list[str]` | `[]` | Regex patterns of parameters FSDP keeps replicated instead of sharding. Meant for params too small to shard into whole tiles; see [Keeping parameters replicated](../../../docs/FSDP.md#keeping-parameters-replicated). |
| `mesh_shape`  | `list[int]`       | `[1, 1]`  | Shape of the device mesh `[rows, cols]`. Total devices = `rows * cols`.                                                                                                  |
| `device_ids`  | `list[int]`       | `null`    | Specific device IDs to use (default: auto-select).                                                                                                                       |

Device setup (`enable_fabric`, `open_device`,
`initialize_parallelism_context`) is performed by `setup_ttml_model`,
which the `GRPOTrainer` constructor calls. FSDP-specific behavior is documented in the
[FSDP subsection](../../../docs/GRPO_TRAINER.md#fsdp) of the trainer
doc.

---

## Remote rollout (two ranks)

Generation is the slow part of GRPO and benefits from running inside a
captured ttnn trace. With
[`grpo_boolq_llama_1b_remote_rollout.yaml`](../../../configs/training_configs/grpo_boolq_llama_1b_remote_rollout.yaml)
the BoolQ training script runs on two tt-run ranks: rank 0 trains the ttml
policy, rank 1 generates with `tt-transformers` (`TTTRolloutSampler`).

```bash
./tt-train/sources/examples/grpo/boolq/run_remote_rollout.sh
```

[`boolq/run_remote_rollout.sh`](boolq/run_remote_rollout.sh) launches
`boolq/boolq_training_example.py --config <remote yaml>` on both ranks with
the rank bindings in
[`boolq/configurations/split_2_2/`](boolq/configurations/split_2_2/)
(2 TTML chips + 2 TTT chips on one host). `--config`, `--script`,
`--hostfile` and `--rank-bindings` override the defaults. `HF_TOKEN` must be
set: both ranks load `meta-llama/Llama-3.2-1B-Instruct`.

Both ranks construct the same `GRPOTrainer`, which picks each rank's role:

- **Rank 0** builds the ttml policy, calls `dataset_func`, and for each batch
  sends the prompts to rank 1, receives the completions, tt-transformers'
  per-token log-probs and rank 1's weight version, and computes the rewards.
  At a generation-batch boundary once `weight_sync_every` optimizer steps have
  passed since the last push, it pushes the policy weights.
- **Rank 1** opens the rollout mesh, builds a `TTTRolloutSampler` and an
  `MPIRolloutServer`, never loads the dataset, and serves requests until
  rank 0 is done.

Rank 1 boots with dummy weights at weight version -1 and refuses to generate
until rank 0 pushes its weights as version 0, before the first batch.

### Remote config fields

| Field | Description |
| ----- | ----------- |
| `training_config.model_source` | HuggingFace Hub id. Rank 0 loads its weights from it and rank 1 reads its config and tokenizer; a local path is rejected in `ttt_remote_sync`. |
| `grpo_config.rollout_backend` | `"ttt_remote_sync"`. |
| `grpo_config.weight_sync_every` | Push the policy weights to rank 1 at a generation-batch boundary once *N* optimizer steps have passed since the last push (default 1). |
| `remote_rollout_config.mesh_shape` | Rank 1's parent mesh `[rows, cols]`, split into `rows * cols` `[1, 1]` submeshes that generate data-parallel. |
| `remote_rollout_config.max_batch_size` | Per-submesh decode capacity. One tt-transformers call serves `max_batch_size * num_submeshes` completions; larger batches are split into several calls. |
| `remote_rollout_config.max_seq_len` | Max prompt + completion length. A prompt longer than `max_seq_len - max_completion_length` raises `ValueError`; prompts are never truncated. |
| `remote_rollout_config.top_k` / `top_p` | On-device sampling settings (defaults 32 / 1.0). |

### Adapting the rank bindings to other hardware

The default `configurations/split_2_2` targets a Blackhole loudbox or
quietbox with P100 / P150 cards: 4 single-chip PCIe devices, 2 pinned to
each rank. Two files encode the host-specific assumptions:

- `rank_bindings.yaml` pins PCIe devices per rank with `TT_VISIBLE_DEVICES`.
  It indexes PCIe devices, not chips: on P300 one device exposes two
  chips, so a `[1, 2]` mesh per rank needs one device per rank
  (`TT_VISIBLE_DEVICES: "0"` and `"1"`).
- `mgd.textproto` sets the arch (`BLACKHOLE`) and the mesh shape.

For a `[1, 4]` mesh per rank (8 chips), set `device_config.mesh_shape` and
`remote_rollout_config.mesh_shape` to `[1, 4]`, expand
`device_config.device_ids`, and add a `configurations/split_4_4/` directory
(`hosts.txt`, `mgd.textproto` with `device_topology { dims: [ 1, 4 ] }`,
`rank_bindings.yaml` with 4 chips per rank). The weight-transfer test's
[`configurations/4-4/`](../../../tests/python/remote_rollout/weight_transfer/configurations/4-4/)
is a working `[1, 4]` template. Point `CONFIG_DIR` in
`run_remote_rollout.sh` (or `--rank-bindings` / `--hostfile`) at the new
directory.

### Limitations

A failure on either rank ends the tt-run job and logs its traceback. Cross-rank
calls are blocking MPI sends and receives without deadlines, so a stuck peer
still hangs the other rank.

---

## See also

- [`tt-train/docs/GRPO_TRAINER.md`](../../../docs/GRPO_TRAINER.md)
  — model- and rank-agnostic `GRPOTrainer` API, including
  [Rollout modes](../../../docs/GRPO_TRAINER.md#rollout-modes).
- [`tt-train/docs/LLAMA_WEIGHT_TRANSFER.md`](../../../docs/LLAMA_WEIGHT_TRANSFER.md)
  — wire format used to ship policy weights from the TTML rank to the TTT rank.
