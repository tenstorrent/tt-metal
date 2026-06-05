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
    build_mesh,
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
from ttml.common.config import load_config, DeviceConfig
from ttml.common.utils import get_loss_over_devices, get_tt_metal_runtime_root, no_grad


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

    Under DDP (active mesh with a ``dp`` axis of size > 1) ``batch_size`` is the
    *global* number of blocks drawn per step; ``collate_packed`` shards that
    leading dim across the ``dp`` axis so each chip evaluates ``batch_size /
    dp_size`` blocks in parallel (≈ dp_size× wall-clock speedup over a single
    card). The per-step loss is then the mean of the per-rank cross-entropies
    via ``get_loss_over_devices``; since every rank gets the same block count
    that mean equals the token-weighted batch mean. Each ``collate_packed`` call
    must receive a dp-divisible block count, so the trailing ``len(blocks) %
    dp_size`` blocks of the final (short) batch are dropped — at most
    ``dp_size - 1`` blocks, i.e. a negligible fraction of a multi-hundred-M-token
    split — and reported.
    """
    mesh = ttml.maybe_mesh()
    use_dp = mesh is not None and mesh.has_axis("dp") and mesh.axis_size("dp") > 1
    dp_size = mesh.axis_size("dp") if use_dp else 1
    if batch_size % dp_size != 0:
        raise ValueError(
            f"--batch-size ({batch_size}) must be divisible by the DP axis size ({dp_size}) so the "
            f"global batch shards evenly across chips. Pick a multiple of {dp_size} "
            f"(e.g. {max(1, batch_size // dp_size) * dp_size} or {(batch_size // dp_size + 1) * dp_size})."
        )

    model.eval()
    it = iter(pds)
    total_loss_x_tokens = 0.0
    total_tokens = 0
    n_batches = 0
    dropped_blocks = 0
    t0 = time.time()
    last_log = t0
    exhausted = False
    while not exhausted:
        blocks = []
        try:
            for _ in range(batch_size):
                blocks.append(next(it))
        except StopIteration:
            exhausted = True

        # Each collate_packed call shards dim 0 across the dp axis, so the block
        # count must be a multiple of dp_size. Full batches already satisfy this
        # (batch_size % dp_size == 0); only the trailing short batch can violate
        # it, so trim its sub-dp_size remainder.
        if use_dp:
            remainder = len(blocks) % dp_size
            if remainder:
                dropped_blocks += remainder
                blocks = blocks[: len(blocks) - remainder]
        if not blocks:
            break

        # Disable autograd for the whole forward: with grads on, every op would
        # retain its inputs for a backward that never happens, pinning the full
        # activation graph for all layers until reset_graph(). Under no_grad the
        # framework frees intermediates as the forward proceeds, so peak memory
        # is ~one layer's activations instead of all of them — this is what lets
        # eval use a much larger per-chip batch than training.
        with no_grad():
            inp, tgt = collate_packed(blocks, seq_len)
            logits = model(inp, attn_mask)
            loss = ttml.ops.loss.cross_entropy_loss(logits, tgt, reduce=ttml.ops.ReduceType.MEAN)
            loss_val = float(get_loss_over_devices(loss)) if use_dp else get_loss_value(loss)
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
    if dropped_blocks:
        print(
            f"  [{split_name}] dropped {dropped_blocks} trailing block(s) "
            f"({dropped_blocks * seq_len:,} tokens) not divisible by dp_size={dp_size}.",
            flush=True,
        )
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
        "--config",
        type=str,
        default=None,
        help=(
            "Optional training YAML to read device_config (mesh_shape, enable_ddp) from, so eval "
            "runs on the same DP mesh as training. Overridden by --mesh-shape/--enable-ddp if given."
        ),
    )
    p.add_argument(
        "--mesh-shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROWS", "COLS"),
        help="Explicit mesh shape, e.g. --mesh-shape 32 1. Overrides --config.",
    )
    p.add_argument(
        "--enable-ddp",
        action="store_true",
        help="Force-enable data parallelism on the active mesh axis (used with --mesh-shape).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help=(
            "Blocks drawn per step. Under DDP this is the GLOBAL batch (sharded across the dp axis, "
            "so per-chip = batch_size / dp_size) and must be divisible by the dp axis size. "
            "Match training (global 160 = 32 chips × 5 per chip) to keep memory/MFU comparable."
        ),
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

    # Resolve the device mesh. Precedence: explicit --mesh-shape, then --config
    # YAML (mirrors training's device_config), then single-device (the original
    # non-DDP behavior). Multi-device meshes are evaluated data-parallel: the
    # global batch is sharded across the dp axis for a ≈dp_size× speedup.
    if args.mesh_shape is not None:
        rows, cols = args.mesh_shape
        device_config = DeviceConfig(
            {"device_config": {"mesh_shape": [rows, cols], "enable_ddp": args.enable_ddp or (rows * cols > 1)}}
        )
    elif args.config is not None:
        configs_root = f"{get_tt_metal_runtime_root()}/tt-train/configs/training_configs"
        yaml_config = load_config(args.config, configs_root)
        device_config = DeviceConfig(yaml_config)
    else:
        device_config = DeviceConfig({})

    if device_config.enable_tp:
        raise SystemExit(
            "Tensor parallelism is not supported for checkpoint eval "
            "(pickle checkpoints are not TP-sharded). Use DDP (enable_ddp) only."
        )

    # Open the mesh first — the checkpoint loader allocates ttml tensors (model
    # weights, replicated across all chips under DDP) that require an open mesh.
    mesh = build_mesh(device_config)
    print(f"1. Opening TT mesh: shape={mesh.shape}, axis_names={mesh.axis_names}")
    ttml.open_device_mesh(mesh, tuple(device_config.device_ids) if device_config.device_ids else None)
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

        _eval_mesh = ttml.maybe_mesh()
        _dp_size = _eval_mesh.axis_size("dp") if (_eval_mesh is not None and _eval_mesh.has_axis("dp")) else 1
        if _dp_size > 1:
            print(
                f"\n5. Evaluating {split_label} (full sweep, global batch_size={args.batch_size} "
                f"= {_dp_size} chips × {args.batch_size // _dp_size} per chip)..."
            )
        else:
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
