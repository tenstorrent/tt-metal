# Qwen3.8-27B LoRA fine-tuning on GovReport (seq_len 8192)

LoRA fine-tuning of the real Qwen3.8-27B checkpoint on GovReport long-document summarization, on one
4-chip Blackhole mesh (e.g. a QuietBox with 4× p150b). Metrics go to stdout and Weights & Biases.

| | |
|---|---|
| Model | Qwen3.8-27B (48 Gated DeltaNet + 16 gated-attention layers), base frozen, bf16 |
| Adapters | LoRA rank 16, alpha 32 (rsLoRA), on attention `q/k/v/o_proj` and DeltaNet `in_proj_qkv/in_proj_z/out_proj`: 22.7M trainable params |
| Parallelism | TP=4 + sequence parallelism, full block recompute in backward |
| Data | GovReport reports with at least 3,000 tokens whose report + summary fits in 8,192 tokens: 6,711 train / 318 validation examples, ~5.8k tokens on average |
| Loss | cross-entropy on the summary tokens only (and the final eos), so lowering it requires reading the report |
| Optimizer | AdamW (β 0.9/0.95), lr 1e-4, 20 warmup updates, cosine decay to 10%, 4 examples per update |

## Files

| File | Purpose |
|---|---|
| `prepare_govreport.py` | downloads and tokenizes GovReport, filters to examples that fit in 8k |
| `finetune_govreport.py` | the training loop: masked loss, eval, wandb, LoRA export |
| `run_govreport.sh` | sets up the environment and launches `finetune_govreport.py` |
| `validate_vs_hf.py` | checks the ttml model against HF transformers (CPU) on a truncated model, plus a causality check |
| `forensics.py` | optional logger for host memory, card telemetry and `dmesg`, plus tt-triage on a stall (for debugging crashes) |
| `../LORA_A_GRADIENT.md` | the column-parallel LoRA gradient bug that this setup depends on being fixed |

## 1. Build

On branch `umales/qwen38`, from the tt-metal root:

```bash
./build_metal.sh --build-tt-train      # produces build_Release/ (with tt-train's _ttml module)
./create_venv.sh                       # produces python_env/ (skip if it already exists)
python_env/bin/pip install "transformers>=5.12" datasets wandb
```

`transformers` 5.x is needed for the `qwen3_5` tokenizer and for the HF reference in `validate_vs_hf.py`.

The `gated_delta_net_backward` kernels are compiled at runtime, so changing them doesn't need a rebuild.
The ttml C++ ops (`rmsnorm_*`, `sdpa_*`, `gated_rmsnorm`, …) do: rebuild with
`cmake --build build_Release --target install` after changing them.

## 2. Model and data

```bash
# ~54 GB checkpoint
hf download Qwen/Qwen3.8-27B --local-dir /path/to/Qwen3.8-27B

# Download + tokenize GovReport (a few minutes, 32 processes) -> train.npz, validation.npz
python_env/bin/python tt-train/sources/examples/qwen38_lora/finetune/prepare_govreport.py \
    --model-path /path/to/Qwen3.8-27B --out /path/to/govreport_tok
```

Expected output: `kept 6711` (train) and `kept 318` (validation).

## 3. wandb

```bash
python_env/bin/wandb login        # or put the key in ~/.netrc
```

Runs go to the project `qwen38-lora-govreport` (`--wandb-project`) under your default entity
(override with `WANDB_ENTITY`). Pass `--no-wandb` to log to stdout only.

## 4. Check the setup (recommended on a new machine)

**4-layer real-weight check against HF.** The HF side runs on CPU in fp32 and takes a few minutes.

```bash
cd tt-train/sources/examples/qwen38_lora/finetune
export QWEN38_MODEL_PATH=/path/to/Qwen3.8-27B GOVREPORT_DATA=/path/to/govreport_tok
export TT_METAL_HOME=$(cd ../../../../.. && pwd) ARCH_NAME=blackhole
export PYTHONPATH=$TT_METAL_HOME/build_Release/tt-train/sources/ttml:$TT_METAL_HOME/tt-train/sources/ttml:$TT_METAL_HOME
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt-train/configs/mgd/bh_qb_1_4_ring_ring.textproto
PY=$TT_METAL_HOME/python_env/bin/python

$PY validate_vs_hf.py hf --out /tmp/hf_L4.npz
$PY validate_vs_hf.py ttml --sequence-parallel --out /tmp/tt_L4.npz
$PY validate_vs_hf.py compare /tmp/hf_L4.npz /tmp/tt_L4.npz       # expect PASS, Pearson r ~0.9998
$PY validate_vs_hf.py ttml --sequence-parallel --seq 8192 --causal-check --out /tmp/tt_8k.npz   # expect CAUSAL
```

**4-layer training smoke test.** About 3 minutes. It checks that gradients are finite and that the replicated
`lora_A` copies stay identical on all 4 chips:

```bash
MODEL_PATH=/path/to/Qwen3.8-27B DATA_DIR=/path/to/govreport_tok OUT=/tmp/smoke ./run_govreport.sh \
    --layers 4 --updates 4 --grad-accum 2 --warmup 1 --eval-examples 2 --eval-every 0 --save-every 0 \
    --check-lora-a --debug-grads --no-wandb
```

Every `[grads]` line should report `0 non-finite tensors`, and every `[check]` line `= 0.000e+00`.

## 5. Full run

Run it in tmux so it doesn't depend on your terminal:

```bash
cd tt-train/sources/examples/qwen38_lora/finetune
tmux new-session -d -s qwen38_train \
  "MODEL_PATH=/path/to/Qwen3.8-27B DATA_DIR=/path/to/govreport_tok OUT=/path/to/runs/govreport \
   ./run_govreport.sh --updates 300 --grad-accum 4 --lr 1e-4 --warmup 20 \
       --eval-every 25 --eval-examples 32 --save-every 100 --wandb-name qwen38-27b-lora-govreport-8k"

tmux attach -t qwen38_train                  # watch it (detach: Ctrl-b d)
tail -f /path/to/runs/govreport/train.log    # or follow the log
```

Prefix `FORENSICS=1` to also run `forensics.py`. It writes to `$OUT/forensics/forensics.log` and runs tt-triage
if the log stalls; set `FORENSICS_CARD_INTERVAL=0.1` for 10 Hz card telemetry.

### What to expect (4× p150b)

| | |
|---|---|
| Checkpoint load + model build | ~12.5 min |
| Eval on 32 validation examples | ~90 s (at step 0 and every `--eval-every` updates) |
| One update (4 × 8,192 tokens) | ~32 s, ~1,020 tokens/s, ~19.5% MFU |
| 300 updates | ~2.7 h of training plus ~15 min of evals |
| Step-0 eval loss | **3.36** nats/token (summary tokens) |
| Train loss after ~20 updates | ~0.7–1.3 |
| Host RAM | ~1 GB process RSS after loading |

### Outputs (in `$OUT`)

- `train.log`: everything printed, including one line of metrics per update.
- `config.json`: arguments and dataset/parameter counts.
- `lora_step00100.npz`, …: the trainable LoRA tensors every `--save-every` updates and at the end. Each
  array is `[4, ...]`: one entry per chip, read device by device. Column-parallel `lora_B` and
  row-parallel `lora_A` are shards; the rest are identical copies.
- `wandb/`: the local wandb run data.

### Logged metrics

`train/loss` (mean over the update's examples, weighted by summary tokens), `train/lr`,
`train/step_seconds`, `train/examples_seen`, `train/epoch`, `perf/tokens_per_s`, `perf/mfu`,
`eval/loss` (over the summary tokens of the fixed eval set), `eval/seconds`.

### Useful flags

| Flag | Default | |
|---|---|---|
| `--updates` | 400 | optimizer steps |
| `--grad-accum` | 4 | examples per optimizer step |
| `--lr`, `--warmup`, `--min-lr-ratio` | 1e-4, 20, 0.1 | linear warmup, then cosine decay to `lr × min-lr-ratio` |
| `--rank`, `--alpha` | 16, 32 | LoRA size (rsLoRA scaling `alpha / sqrt(rank)`) |
| `--eval-every`, `--eval-examples` | 50, 32 | eval cadence and size |
| `--save-every` | 100 | LoRA `.npz` export cadence |
| `--ckpt-every` + `--resume` | 0 (off) | full checkpoint (LoRA + AdamW + data position) and resume |
| `--seed` | 42 | sets the eval set and data order |
| `--layers N` | all | truncate the model (smoke tests only) |
| `--check-lora-a`, `--debug-grads` | off | correctness checks (slow; for debugging) |

## Fixes this setup depends on (all on this branch)

Before these fixes, only random-init (MFU) runs had been tested. The first real-weight training step
produced inf gradients.

1. **`gated_delta_net_backward` with real weights.** Its in-kernel UT-transform inverse used Neumann
   doubling. When a chunk's keys are strongly aligned (real checkpoints), the intermediate matrix powers
   reach ~1e18 while the inverse itself is O(1), so the gradients came out ~1e9 instead of ~1e-4. The
   backward now takes the forward's stable inverse through a new `tinv` input
   (`chunk_gated_delta_rule(..., output_intermediates=True)`; see `fused_delta_rule.py` and
   `gated_delta_net_backward.py`). It matches a float64 reference at 8k with PCC ≥ 0.9999.
2. **Column-parallel LoRA `lora_A` gradient** was never summed across TP, so the 4 copies drifted apart
   from update 2 on. See `../LORA_A_GRADIENT.md`.
3. **Loader** crashed on truncated (`--layers`) models; the conv taps of the skipped layers are now ignored.

## Known issue: host power loss on `bh-qb-16`

On `bh-qb-16` the whole machine lost power three times, after 10–12 minutes of full training load
(2026-10-28 at 11:40, 13:17 and 14:59 UTC):

- The last 10 Hz telemetry before each loss was normal: cards 57–63 °C and ≤ 265 W, host memory flat,
  nothing in `dmesg`.
- The BMC event log shows only the power-on after each event, and once its own clock reset, which means
  the BMC lost power too.
- A single update on its own and an 18-update run (9 minutes) both completed fine.

This points to the machine's power or cooling under sustained ~1 kW card load, not to the training code.
On a new machine, consider `FORENSICS=1` for the first full run.
