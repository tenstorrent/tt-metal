# BoolQ training example

GRPO fine-tuning of `meta-llama/Llama-3.2-1B-Instruct` on
`google/boolq`, with a Yes/No correctness reward, with rollouts
generated on a second MPI rank. The script
([`boolq_training_example.py`](boolq_training_example.py)) is a plain
`GRPOTrainer` script (see
[`tt-train/docs/GRPO_TRAINER.md`](../../../../docs/GRPO_TRAINER.md));
its config selects `rollout_source: "ttt"` and
`rollout_mode: "remote_sync"`, and `runner.sh` launches it on both ranks.

---

## Two-rank architecture

Generation is the slow part of GRPO and benefits from running inside
a captured ttnn trace. Training-side `ttml.Llama` and inference-side
`tt-transformers.Transformer` are different model implementations
with different mesh-shape constraints, so this example splits them
across two MPI ranks: the trainer keeps a free policy mesh, the
rollout rank keeps a captured decode trace, and weights are pushed
from one to the other every `weight_sync_every` steps.

```text
                 mpirun (tt-run, world_size = 2)
                ─────────────────────────────────

  rank 0 (TTML)                      rank 1 (TTT)
  ───────────────                     ──────────────
  ttml.Llama policy                   TTTRolloutSampler
  GRPOTrainer + optimizer               Nx tt-transformers Transformer
  SyncRemoteRolloutBatchSource        MPIRolloutServer
  mesh: [1, N] (DDP)                  mesh: [1, N] -> Nx [1, 1] submesh
       │                                   ▲
       │  MPIRolloutClient   OP_GENERATE / OP_REQUEST_TRANSFER / OP_SHUTDOWN
       └──────────────► MPI ──────────────►┘
       │                                   │
       └──────── HostWeightBridge ─────────┘
```

`N` is the per-rank mesh width: `device_config.mesh_shape` on rank 0 and
`remote_rollout_config.mesh_shape` on rank 1. It must match the
`device_topology` in the `mgd.textproto` you point tt-run at. See
[How to run](#how-to-run).

Both ranks construct the same `GRPOTrainer`; the constructor picks the
role from the rank:

- **Rank 0** builds the ttml policy, calls `dataset_func`, and builds a
  `SyncRemoteRolloutBatchSource`. For each batch it sends the prompts to
  rank 1, gets back the completions, tt-transformers' per-token log-probs
  and rank 1's weight version, and computes the rewards. After every
  `weight_sync_every` optimizer steps it pushes the policy weights.
- **Rank 1** opens the rollout mesh, builds a `TTTRolloutSampler` and an
  `MPIRolloutServer`, and never loads the dataset. `trainer.train()` serves
  requests until rank 0 is done.

Both ranks boot with the Hub weights of `model_source` (weight version
0), so no initial weight push is needed.

---

## Components

| Class                          | Side | Role |
| ------------------------------ | ---- | ---- |
| `SyncRemoteRolloutBatchSource` | TTML | The trainer's `RolloutBatchSource` in `remote_sync`: remote generate, reward scoring, versioned weight pushes. |
| `MPIRolloutClient`             | TTML | `remote_generate(prompts) -> (completions, logprobs, weight_version)` and `send_weights(hf_dict, version=)`. Its constructor blocks until the peer's server is up. |
| `MPIRolloutServer`             | TTT  | Serves a `RolloutSampler`: `OP_GENERATE` calls `sampler.generate`, `OP_REQUEST_TRANSFER` calls `sampler.update_weights` with the version from the request header. Blocks in `serve_forever()` until `OP_SHUTDOWN`. |
| `TTTRolloutSampler`            | TTT  | Hosts the `tt-transformers.Transformer` copies and a captured decode trace; returns completions with tt-transformers' on-device sampled-token log-probs. |
| `WeightBridge`                 | both | Replicated-tensor transport (ABC). `HostWeightBridge` moves each weight to host via MPI and re-uploads it to each receiver submesh. Wire-format spec: [`LLAMA_WEIGHT_TRANSFER.md`](../../../../docs/LLAMA_WEIGHT_TRANSFER.md). |

---

## Configuration

Everything runtime-tunable lives in a single training YAML,
[`grpo_boolq_llama_1b_remote_rollout.yaml`](../../../../configs/training_configs/grpo_boolq_llama_1b_remote_rollout.yaml).
Both ranks load it. `training_config`, `device_config` and `grpo_config`
are the standard `GRPOTrainer` blocks (see the
[trainer doc](../../../../docs/GRPO_TRAINER.md)); the remote-specific
fields are:

| Field | Description |
| ----- | ----------- |
| `training_config.model_source` | HuggingFace Hub id. Both ranks load their weights from it; a local path is rejected in `remote_sync`. |
| `grpo_config.rollout_source` / `rollout_mode` | `"ttt"` / `"remote_sync"`. |
| `grpo_config.weight_sync_every` | Push the policy weights to rank 1 every *N* optimizer steps (default 1). |
| `grpo_config.temperature` | Baked into rank 1's decode trace at construction. tt-transformers emits no log-probs on its greedy path, so `temperature: 0` is rejected. |

### `remote_rollout_config` — rollout-rank knobs

| Field            | Type        | Default      | Description |
| ---------------- | ----------- | ------------ | ----------- |
| `mesh_shape`     | `list[int]` | —            | Shape of the TTT parent mesh `[rows, cols]`. The sampler splits it into `rows * cols` `[1, 1]` submeshes; each hosts one `tt-transformers.Transformer` copy and generation runs data-parallel across them. |
| `max_batch_size` | `int`       | —            | Per-submesh concurrent decode capacity. One tt-transformers call serves `max_batch_size * num_submeshes` completions; larger batches are split into several calls. Sets the paged KV-cache block budget on each submesh. |
| `max_seq_len`    | `int`       | —            | Max prompt + completion length. A prompt longer than `max_seq_len - grpo_config.max_completion_length` (after the chat template) raises `ValueError`; prompts are never truncated. |
| `seed`           | `int`       | `None`       | On-device sampling seed. |
| `top_k`          | `int`       | `32`         | tt-transformers samples from at most the top 32 tokens; the reported log-prob is the full-vocab one. |
| `top_p`          | `float`     | `1.0`        | Nucleus sampling threshold. |

### Limitations

Cross-rank calls are blocking MPI sends and receives without deadlines.
An exception on either rank ends the tt-run job, but a rank that exits
cleanly without reaching its peer's matching call, or a stuck peer,
hangs the other rank.

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

`./tt-train/sources/examples/grpo_remote_rollout/boolq/runner.sh`

### 4. Observe the outputs

- `generated/tt-train/grpo_run/grpo_metrics.csv` — per-step CSV
written by `GRPOMonitor`.
- `generated/tt-train/grpo_run/checkpoints/grpo_step_{N}/` — full HF
checkpoint directories (see
[Checkpointing](../../../../docs/GRPO_TRAINER.md#checkpointing) in
the trainer doc for the layout).

To train on a different Llama model or dataset, change `model_source`
in the YAML and the dataset / reward functions in the script. Any
`GRPOTrainer` script runs in this mode when launched through tt-run
with a `remote_sync` config; for example the single-process
[`grpo/boolq/boolq_training_example.py`](../../grpo/boolq/boolq_training_example.py)
accepts this YAML via `--config`.

---



## See also

- [`tt-train/docs/GRPO_TRAINER.md`](../../../../docs/GRPO_TRAINER.md)
  — generic trainer API (rank- and model-agnostic).
- [`tt-train/docs/LLAMA_WEIGHT_TRANSFER.md`](../../../../docs/LLAMA_WEIGHT_TRANSFER.md)
  — wire format used by `WeightBridge` to ship policy weights from
  the TTML rank to the TTT rank.
