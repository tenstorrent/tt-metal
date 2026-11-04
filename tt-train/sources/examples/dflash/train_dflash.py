# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Train a DFlash drafter on TT hardware from speculators offline data (TT_PORT_PLAN D2).

Same data contract and step as ``speculators/scripts/train.py`` on one replica:

* rows: ``prepare_data.py`` Arrow dataset + ``hs_<idx>.safetensors`` hidden states;
* packing: speculators multipack sampler (seeded per epoch), ``--total-seq-len`` tokens
  per row;
* anchors: up to ``--max-anchors`` per packed sequence, padded to a fixed
  ``--anchor-budget`` (fixed device shapes);
* step: KL loss with ``exp(-(k-1)/gamma)`` decay, clip_grad_norm 1.0, AdamW (fp32
  master weights), linear/cosine schedule with ``total // 100`` warmup.

The initial checkpoint is a speculators-format directory (``config.json`` +
``model.safetensors``) that must include the frozen target tensors
(``embed_tokens``, ``lm_head``, ``verifier_norm``); produce one with the speculators
repo's ``scripts/tt/export_dflash_init.py``. Checkpoints are written back in the same
format (speculators names, fp32 master weights).

Example::

    python tt-train/sources/examples/dflash/train_dflash.py \\
        --init-checkpoint /data/dflash/init --data /data/dflash/prepared \\
        --hidden-states /data/dflash/prepared/hidden_states \\
        --total-seq-len 4096 --anchor-budget 512 --max-steps 2000 --out /data/dflash/run1
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np


def lr_lambda(kind: str, step: int, warmup: int, total: int) -> float:
    """transformers get_{linear,cosine}_schedule_with_warmup multipliers."""
    if kind == "none":
        return 1.0
    if step < warmup:
        return step / max(1, warmup)
    if kind == "linear":
        return max(0.0, (total - step) / max(1, total - warmup))
    if kind == "cosine":
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(kind)


def load_checkpoint(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    from safetensors.torch import load_file  # noqa: PLC0415

    config = json.loads((path / "config.json").read_text())
    state = {k: v.float().numpy() for k, v in load_file(str(path / "model.safetensors")).items()}
    return config, state


def save_checkpoint(out: Path, init_dir: Path, init_state: dict, stepper, spec_shapes, meta: dict) -> None:
    """Trainable tensors from the fp32 master weights; frozen ones copied bit-exact from init."""
    from safetensors.numpy import save_file  # noqa: PLC0415

    from ttml.models.dflash.weights import read_master_weights, ttml_to_speculators  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    trained = ttml_to_speculators(read_master_weights(stepper.optimizer), spec_shapes)
    state = {**init_state, **trained}
    save_file({k: np.ascontiguousarray(v, np.float32) for k, v in state.items()}, str(out / "model.safetensors"))
    shutil.copy(init_dir / "config.json", out / "config.json")
    (out / "tt_train_state.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init-checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--hidden-states", type=Path, default=None)
    ap.add_argument("--split-ratio", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--total-seq-len", type=int, default=8192)
    ap.add_argument("--max-anchors", type=int, default=256, help="per packed sequence (speculators --max-anchors)")
    ap.add_argument("--anchor-budget", type=int, required=True, help="fixed anchor slots per row (shape constant)")
    ap.add_argument(
        "--anchor-overflow",
        choices=["subsample", "raise"],
        default="subsample",
        help="rows needing more anchors than --anchor-budget: subsample uniformly (logged) or fail",
    )
    ap.add_argument("--gamma", type=float, default=4.0, help="speculators --dflash-decay-gamma")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--scheduler-type", choices=["linear", "cosine", "none"], default="linear")
    ap.add_argument("--scheduler-warmup-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="sampler seed (speculators default 0)")
    ap.add_argument("--anchor-seed", type=int, default=1234)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args()

    import ttml  # noqa: PLC0415
    from ttml.models.dflash import (  # noqa: PLC0415
        DFlashDeviceBatch,
        DFlashDrafter,
        DFlashDrafterConfig,
        DFlashTrainStep,
        build_host_batch,
        sample_anchors,
    )
    from ttml.models.dflash.host import required_anchor_count  # noqa: PLC0415
    from ttml.models.dflash.data import OfflineDFlashDataset, packed_batches  # noqa: PLC0415
    from ttml.models.dflash.weights import load_speculators_weights  # noqa: PLC0415

    spec_config, state = load_checkpoint(args.init_checkpoint)
    cfg = DFlashDrafterConfig.from_speculators_config(spec_config)
    if (args.anchor_budget * cfg.block_size) % 32 or args.total_seq_len % 32:
        raise SystemExit("anchor_budget * block_size and total_seq_len must be multiples of 32")
    spec_shapes = {k: v.shape for k, v in state.items()}
    missing = [k for k in ("embed_tokens.weight", "lm_head.weight", "verifier_norm.weight") if k not in state]
    if missing:
        raise SystemExit(f"{args.init_checkpoint} lacks frozen target tensors {missing}; see export_dflash_init.py")

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.get_device()
    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    model = DFlashDrafter(cfg)
    load_speculators_weights(model, state)  # before the optimizer: it snapshots masters
    stepper = DFlashTrainStep(model, lr=args.lr, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm)

    dataset = OfflineDFlashDataset(args.data, args.hidden_states, args.split_ratio)
    rows_per_epoch = len(packed_batches(dataset.lengths, args.total_seq_len, args.seed, 0))
    total = args.max_steps or args.epochs * rows_per_epoch
    warmup = args.scheduler_warmup_steps or total // 100
    rng = np.random.default_rng(args.anchor_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "train_log.jsonl").open("a")
    print(f"{len(dataset)} samples, ~{rows_per_epoch} rows/epoch, {total} steps, warmup {warmup}, cfg={cfg}")

    step = 0
    for epoch in range(args.epochs):
        for indices, row in dataset.epoch_rows(args.total_seq_len, args.seed, epoch):
            if step >= total:
                break
            t0 = time.perf_counter()
            wanted = required_anchor_count(row["loss_mask"], row["document_ids"], cfg.block_size, args.max_anchors)
            anchors, valid = sample_anchors(
                rng,
                row["loss_mask"],
                row["document_ids"],
                cfg.block_size,
                args.max_anchors,
                args.anchor_budget,
                overflow=args.anchor_overflow,
            )
            hb = build_host_batch(cfg, anchor_positions=anchors, anchor_valid=valid, gamma=args.gamma, **row)
            batch = DFlashDeviceBatch.from_host(hb)
            lr = args.lr * lr_lambda(args.scheduler_type, step, warmup, total)
            stepper.set_lr(lr)
            loss, grad_norm = stepper(batch)
            dt = time.perf_counter() - t0
            if not math.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}: {loss}")
            if step % args.log_every == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "lr": lr,
                    "anchors": int(valid.sum()),
                    "anchors_dropped": max(0, wanted - args.anchor_budget),
                    "samples": [int(i) for i in indices],
                    "sec": dt,
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(
                    f"step {step} loss {loss:.4f} gnorm {grad_norm:.3f} lr {lr:.2e} anchors {rec['anchors']} {dt:.2f}s"
                )
            step += 1
            if step % args.save_every == 0 or step == total:
                save_checkpoint(
                    args.out / f"checkpoint_{step}",
                    args.init_checkpoint,
                    state,
                    stepper,
                    spec_shapes,
                    {"step": step, "epoch": epoch, "args": {k: str(v) for k, v in vars(args).items()}},
                )
    ctx.close_device()


if __name__ == "__main__":
    main()
