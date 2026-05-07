"""Full-split evaluation of a TinyLlama .pkl checkpoint on validation and test.

Loads the checkpoint produced by ``pretrain_tinyllama.py``, opens the TT mesh,
builds the model from the saved config, and sweeps the *entire* validation and
test splits of a pre-tokenized dataset (no eval_iters cap). Reports token-level
mean cross-entropy loss and perplexity per split.

Usage (under srun, with the same mesh used for training):
    python -u eval_tinyllama_checkpoint.py \
        --checkpoint /data/awliu/test_logs_slimpajama-6B/tinyllama_ckpts/tinyllama_slimpajama_glx_final.pkl \
        --tokenized-data-dir /data/awliu/datasets/SlimPajama-6B-tokenized \
        --batch-size 5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# Reuse all the heavy lifting from the training script: model build, checkpoint
# loader, packed dataset, collate, loss extraction, causal mask.
import pretrain_tinyllama
from pretrain_tinyllama import (
    PackedTokenDataset,
    _load_packed_split,
    collate_packed,
    get_loss_value,
    load_model_from_checkpoint,
)

# Checkpoints were pickled when pretrain_tinyllama.py was __main__, so any
# class defined there (e.g. _PackedTokenizerStub) is recorded with module path
# __main__.X. When unpickling from this script, __main__ is *this* file and
# the lookup fails. Alias the relevant symbols into __main__ so pickle finds
# them regardless of which module actually defined them.
import sys as _sys

_main_mod = _sys.modules["__main__"]
for _name in dir(pretrain_tinyllama):
    if _name.startswith("_") and not _name.startswith("__"):
        # Private helpers like _PackedTokenizerStub, _numpy_entry_to_tensor.
        setattr(_main_mod, _name, getattr(pretrain_tinyllama, _name))
    elif not _name.startswith("__"):
        # Public classes (Model, ModelConfig, TrainingConfig, etc.) that may
        # also appear in the pickle.
        if not hasattr(_main_mod, _name):
            setattr(_main_mod, _name, getattr(pretrain_tinyllama, _name))

import ttnn
from ttml import autograd as _ttml_autograd  # noqa: F401  (ensures ttml import order)
import ttml
from ttml.common.data import build_causal_mask


def _eval_split(
    model,
    pds: PackedTokenDataset,
    batch_size: int,
    seq_len: int,
    attn_mask,
    split_name: str,
) -> tuple[float, int, int]:
    """Sweep an entire packed split and return (mean_loss, total_tokens, n_batches).

    Loss is averaged with token-count weights so the final number is the true
    token-level cross-entropy across the whole split (not a batch-mean-of-means,
    which would skew toward whatever batch ended up smaller). With a wrap=False
    iterator, the only short batch is the trailing one — but we'd rather report
    the exact token-weighted mean either way.
    """
    model.eval()
    it = iter(pds)
    total_loss_x_tokens = 0.0
    total_tokens = 0
    n_batches = 0
    t0 = time.time()
    last_log = t0
    while True:
        blocks = []
        try:
            for _ in range(batch_size):
                blocks.append(next(it))
        except StopIteration:
            pass
        if not blocks:
            break

        inp, tgt = collate_packed(blocks, seq_len)
        logits = model(inp, attn_mask)
        loss = ttml.ops.loss.cross_entropy_loss(logits, tgt, reduce=ttml.ops.ReduceType.MEAN)
        loss_val = get_loss_value(loss)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        # Each block contributes seq_len next-token predictions.
        batch_tokens = len(blocks) * seq_len
        total_loss_x_tokens += loss_val * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1

        now = time.time()
        if now - last_log >= 10.0:
            running = total_loss_x_tokens / total_tokens
            print(
                f"  [{split_name}] batch {n_batches}: "
                f"running_loss={running:.4f} ({total_tokens:,} tokens, {now - t0:.1f}s)",
                flush=True,
            )
            last_log = now

    elapsed = time.time() - t0
    if total_tokens == 0:
        print(f"  [{split_name}] no batches — split is empty or smaller than one batch")
        return float("nan"), 0, 0

    mean_loss = total_loss_x_tokens / total_tokens
    print(
        f"  [{split_name}] DONE: {n_batches} batches, {total_tokens:,} tokens, {elapsed:.1f}s",
        flush=True,
    )
    return mean_loss, total_tokens, n_batches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path, help="Path to .pkl checkpoint.")
    p.add_argument(
        "--tokenized-data-dir",
        required=True,
        type=Path,
        help="Root of pre-tokenized dataset (expects validation/ and test/ subdirs).",
    )
    p.add_argument("--val-split", default="validation")
    p.add_argument("--test-split", default="test")
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Per-step batch size. Match training config to keep memory/MFU comparable.",
    )
    p.add_argument(
        "--n-chunks",
        type=int,
        default=8,
        help="PackedTokenDataset n_chunks (mmap window). Doesn't affect numerics.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=5489,
        help="Seed for PackedTokenDataset (no shuffling, but used for reproducibility).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.tokenized_data_dir.is_dir():
        raise SystemExit(f"Tokenized data dir not found: {args.tokenized_data_dir}")

    # Open device first — the checkpoint loader allocates ttml tensors that
    # require an open device context.
    print("1. Opening TT device...")
    ttml.autograd.AutoContext.get_instance().open_device()
    ttml.autograd.AutoContext.get_instance().get_device()
    ttml.autograd.AutoContext.get_instance().set_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n2. Loading checkpoint: {args.checkpoint}")
    (
        model,
        tokenizer,
        model_config,
        training_config,
        loaded_step,
        _opt_state,
        _opt_lr,
        _train_iter_state,
    ) = load_model_from_checkpoint(str(args.checkpoint))
    seq_len = model_config.max_sequence_length
    print(f"   - Loaded step:    {loaded_step}")
    print(f"   - Vocab size:     {model_config.vocab_size}")
    print(f"   - Sequence len:   {seq_len}")
    print(
        f"   - Architecture:   {model_config.num_blocks} layers, "
        f"{model_config.embedding_dim} embd, {model_config.num_heads} heads"
    )

    print("\n3. Building causal attention mask...")
    mask_np = build_causal_mask(seq_len)
    mask = ttml.autograd.Tensor.from_numpy(mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    # GPT-2 / Llama use fused SDPA's native causal path when mask is None.
    # DeepSeek's composite SDPA needs an explicit mask.
    attn_mask = mask if model_config.model_type == "deepseek" else None

    results: dict[str, dict] = {}

    for split_label, split_name in (("validation", args.val_split), ("test", args.test_split)):
        print(f"\n4. Loading {split_label} split: {split_name}")
        try:
            pds, meta = _load_packed_split(
                tokenized_dir=str(args.tokenized_data_dir),
                split=split_name,
                block_size=seq_len,
                n_chunks=args.n_chunks,
                seed=args.seed,
                shuffle=False,
                wrap=False,
            )
        except FileNotFoundError as e:
            print(f"   - Skipping {split_label}: {e}")
            continue

        if meta is not None:
            print(
                f"   - Tokens in split: {int(meta.get('total_tokens', 0)):,}, " f"shards: {len(meta.get('shards', []))}"
            )

        print(f"\n5. Evaluating {split_label} (full sweep, batch_size={args.batch_size})...")
        mean_loss, n_tokens, n_batches = _eval_split(
            model=model,
            pds=pds,
            batch_size=args.batch_size,
            seq_len=seq_len,
            attn_mask=attn_mask,
            split_name=split_label,
        )
        ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
        results[split_label] = {
            "loss": mean_loss,
            "ppl": ppl,
            "tokens": n_tokens,
            "batches": n_batches,
        }

    print("\n=== Results ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Step:       {loaded_step}")
    for split_label, r in results.items():
        print(
            f"  {split_label:<10s}  loss={r['loss']:.6f}  ppl={r['ppl']:.4f}  "
            f"({r['tokens']:,} tokens, {r['batches']} batches)"
        )

    ttml.autograd.AutoContext.get_instance().close_device()
    return 0


if __name__ == "__main__":
    sys.exit(main())
