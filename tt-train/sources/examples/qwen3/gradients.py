# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 backward pass comparison: HuggingFace (PyTorch) vs ttml.

Compares per-parameter gradients between HuggingFace (CPU/bfloat16) and
ttml (Tenstorrent device/bfloat16) after a single forward+backward pass
with cross-entropy loss.

Usage:
    # Single device:
    python gradients.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \\
        --max_seq_len 128

    # With tensor parallelism:
    python gradients.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \\
        --max_seq_len 128 --mesh_shape 1 8

    # With gradient checkpointing:
    python gradients.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \\
        --max_seq_len 128 --mesh_shape 1 8 --checkpoint

    # Data parallelism:
    python gradients.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \\
        --max_seq_len 128 --mesh_shape 4 1

    # Combined DP + TP:
    python gradients.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \\
        --max_seq_len 128 --mesh_shape 4 8 --checkpoint
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import ttnn
import ttml

from generate import create_causal_mask_tensor
from utils.tensor_utils import (
    create_input_tensor_from_torch as create_input_tensor,
    create_input_tensor_dp,
)
from utils.sharded_loss import sharded_cross_entropy_loss
from utils.memory import MemoryUsageTracker, finalize_memory
from utils.param_utils import (
    repermute_proj_rows,
    repermute_norm_weights,
    _build_grad_mapping_single,
    _build_grad_mapping_distributed,
    _extract_grad_distributed,
)


def _compare_gradients(hf_grads, ttml_grads, mapping, inv_transforms):
    """Compare per-parameter gradients between HF and TTML.

    Metrics (a = HF ground truth, b = TTML, eps = 1e-4):
      AbsDiff  = |a - b|          → mean and max
      RelDiff  = |a - b| / (|a| + eps)  → mean and max
      CosSim   = dot(a,b) / (||a|| * ||b|| + eps)  → cosine similarity
      CosDist  = 1 - CosSim                         → cosine distance
    RelDiff accounts for bfloat16 weight precision by normalising against |a|.
    Output uses fixed-point notation with decimal-point alignment per column.
    """
    REL_EPS = 1.0e-2

    rows = []
    abs_mean_sum = 0.0
    abs_max_worst = 0.0
    rel_mean_sum = 0.0
    rel_max_worst = 0.0
    cos_sim_sum = 0.0
    cos_dist_worst = 0.0
    count = 0
    skipped = []

    for hf_name, ttml_name in sorted(mapping.items()):
        if hf_name not in hf_grads:
            skipped.append((hf_name, "no HF grad"))
            continue
        if ttml_name not in ttml_grads:
            skipped.append((hf_name, "no TTML grad"))
            continue

        hf_grad = hf_grads[hf_name].float()
        ttml_grad_raw = ttml_grads[ttml_name].float()
        ttml_grad = ttml_grad_raw.squeeze()

        if hf_grad.shape != ttml_grad.shape:
            if hf_grad.dim() == 2 and ttml_grad.dim() == 2:
                r = min(hf_grad.shape[0], ttml_grad.shape[0])
                c = min(hf_grad.shape[1], ttml_grad.shape[1])
                hf_grad = hf_grad[:r, :c]
                ttml_grad = ttml_grad[:r, :c]
            elif hf_grad.dim() == 1 and ttml_grad.dim() == 1:
                d = min(hf_grad.shape[0], ttml_grad.shape[0])
                hf_grad = hf_grad[:d]
                ttml_grad = ttml_grad[:d]
            else:
                skipped.append(
                    (
                        hf_name,
                        f"shape mismatch: HF {hf_grad.shape} vs TTML {ttml_grad.shape}",
                    )
                )
                continue

        if hf_name in inv_transforms:
            tr = inv_transforms[hf_name]
            if tr[0] == "repermute_proj":
                ttml_grad = repermute_proj_rows(ttml_grad, num_heads=tr[1])
            elif tr[0] == "repermute_norm":
                ttml_grad = repermute_norm_weights(ttml_grad)

        hf_flat = hf_grad.flatten()
        ttml_flat = ttml_grad.flatten()

        abs_diff = (hf_flat - ttml_flat).abs()
        ad_mean = abs_diff.mean().item()
        ad_max = abs_diff.max().item()

        rel_diff = abs_diff / (hf_flat.abs() + REL_EPS)
        rd_mean = rel_diff.mean().item()
        rd_max = rel_diff.max().item()

        hf_norm_t = hf_flat.norm()
        ttml_norm_t = ttml_flat.norm()
        cos_sim = (
            torch.dot(hf_flat, ttml_flat) / (hf_norm_t * ttml_norm_t + 1e-8)
        ).item()
        cos_dist = 1.0 - cos_sim
        hf_norm_val = hf_norm_t.item()
        ttml_norm_val = ttml_norm_t.item()

        abs_mean_sum += ad_mean
        abs_max_worst = max(abs_max_worst, ad_max)
        rel_mean_sum += rd_mean
        rel_max_worst = max(rel_max_worst, rd_max)
        cos_sim_sum += cos_sim
        cos_dist_worst = max(cos_dist_worst, cos_dist)
        count += 1

        rows.append(
            {
                "name": hf_name,
                "ad_mean": ad_mean,
                "ad_max": ad_max,
                "rd_mean": rd_mean,
                "rd_max": rd_max,
                "cos_sim": cos_sim,
                "cos_dist": cos_dist,
                "hf_norm": hf_norm_val,
                "ttml_norm": ttml_norm_val,
                "norm_diff": hf_norm_val - ttml_norm_val,
            }
        )

    if not rows:
        if skipped:
            print(f"\n  Skipped {len(skipped)} parameters:")
            for name, reason in skipped:
                print(f"    {name}: {reason}")
        return

    col_keys = [
        "ad_mean",
        "ad_max",
        "rd_mean",
        "rd_max",
        "cos_sim",
        "cos_dist",
        "hf_norm",
        "ttml_norm",
        "norm_diff",
    ]
    col_headers = [
        "AbsMean",
        "AbsMax",
        "RelMean",
        "RelMax",
        "CosSim",
        "CosDist",
        "HF Norm",
        "TTML Norm",
        "Norm Diff",
    ]
    COL_SEP = "  "

    def _dec_places(values):
        """Pick decimal places so the smallest non-zero value shows ~4 sig figs."""
        abs_vals = [abs(v) for v in values if v != 0]
        if not abs_vals:
            return 4
        min_val = min(abs_vals)
        if min_val >= 100:
            return 2
        if min_val >= 1:
            return 4
        if min_val >= 0.01:
            return 6
        if min_val >= 0.0001:
            return 8
        places = int(-math.log10(min_val)) + 4
        return min(places, 12)

    col_fmt = {}
    for key in col_keys:
        values = [r[key] for r in rows]
        dp = _dec_places(values)
        max_int_w = 1
        for v in values:
            int_part = f"{v:.{dp}f}".split(".")[0]
            max_int_w = max(max_int_w, len(int_part))
        col_fmt[key] = [dp, max_int_w]

    for key, hdr in zip(col_keys, col_headers):
        dp, int_w = col_fmt[key]
        total = int_w + 1 + dp
        if total < len(hdr):
            col_fmt[key][1] += len(hdr) - total

    def _col_w(key):
        return col_fmt[key][1] + 1 + col_fmt[key][0]

    def _fmt(value, key):
        dp, int_w = col_fmt[key]
        parts = f"{value:.{dp}f}".split(".")
        return f"{parts[0]:>{int_w}}.{parts[1]}"

    name_w = max(max(len(r["name"]) for r in rows), len("HF Parameter"))
    header = COL_SEP.join(
        [f"{'HF Parameter':>{name_w}}"]
        + [f"{h:^{_col_w(k)}}" for k, h in zip(col_keys, col_headers)]
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            COL_SEP.join([f"{r['name']:>{name_w}}"] + [_fmt(r[k], k) for k in col_keys])
        )

    print("-" * len(header))
    if count > 0:
        summaries = [
            ("Avg AbsDiff Mean", abs_mean_sum / count),
            ("Worst AbsDiff Max", abs_max_worst),
            ("Avg RelDiff Mean", rel_mean_sum / count),
            ("Worst RelDiff Max", rel_max_worst),
            ("Avg CosSim", cos_sim_sum / count),
            ("Worst CosDist", cos_dist_worst),
        ]
        for label, value in summaries:
            print(f"{label:>{name_w}}: {value:.10f}")
        print(f"{'Parameters compared':>{name_w}}: {count}")
    if skipped:
        print(f"\n  Skipped {len(skipped)} parameters:")
        for name, reason in skipped:
            print(f"    {name}: {reason}")


def run_backward_comparison(
    hf_model,
    ttml_model,
    config,
    sequences,
    max_seq_len,
    device,
    tie_word_embeddings=False,
    distributed=False,
    shard_dim=None,
    track_memory=False,
    tokenizer=None,
    sharded_loss=False,
    dp_size=1,
):
    """Run backward on both HF and TTML models, compare per-parameter gradients.

    sequences: list of token lists (one per batch element / DP group).

    Uses cross-entropy loss with next-token prediction targets — the standard
    LLM training objective.  For vocab-sharded logits (sharded_loss=True) uses
    a distributed cross-entropy that communicates only tiny [B,1,S,1] correction
    tensors instead of gathering the full vocabulary.
    Supports single-device, TP, DP, and combined DP+TP configurations.
    """
    print("\n" + "=" * 70)
    parts = []
    if dp_size > 1:
        parts.append(f"DP={dp_size}")
    tp_size_disp = (
        device.shape[shard_dim] if (distributed and shard_dim is not None) else 1
    )
    if tp_size_disp > 1:
        parts.append(f"TP={tp_size_disp}")
    mode_str = "distributed " + ", ".join(parts) if parts else "single-device"
    loss_type = "distributed cross-entropy" if sharded_loss else "cross-entropy"
    effective_batch = len(sequences)
    print(
        f"Backward pass: gradient comparison ({mode_str}, "
        f"batch_size={effective_batch}, loss={loss_type})"
    )
    print("=" * 70)

    vocab_size = config.vocab_size
    vocab_padded = ((vocab_size + 31) // 32) * 32

    for i, seq in enumerate(sequences):
        print(f"  [{i}]: {len(seq)} tokens")

    seq_lens = [len(s) for s in sequences]
    max_seq_in_batch = max(seq_lens)
    assert (
        max_seq_in_batch <= max_seq_len
    ), f"Longest sequence ({max_seq_in_batch}) exceeds max_seq_len ({max_seq_len})"

    # ------------------------------------------------------------------
    # 1. HuggingFace backward (CPU, float32)
    # ------------------------------------------------------------------
    print("\n[HF] Forward + backward ...")
    t0 = time.time()

    hf_model.train()
    hf_model.zero_grad()

    hf_input = torch.zeros(effective_batch, max_seq_len, dtype=torch.long)
    for b in range(effective_batch):
        hf_input[b, : seq_lens[b]] = torch.tensor(sequences[b], dtype=torch.long)

    hf_outputs = hf_model(input_ids=hf_input, use_cache=False)
    hf_logits = hf_outputs.logits  # (effective_batch, max_seq_len, vocab_size)

    hf_target = torch.zeros(effective_batch, max_seq_len, dtype=torch.long)
    for b in range(effective_batch):
        if seq_lens[b] > 1:
            hf_target[b, : seq_lens[b] - 1] = torch.tensor(
                sequences[b][1:], dtype=torch.long
            )
    hf_loss = F.cross_entropy(
        hf_logits.view(-1, hf_logits.size(-1)),
        hf_target.view(-1),
        reduction="mean",
    )
    hf_loss.backward()

    hf_time = time.time() - t0
    print(f"  Loss: {hf_loss.item():.6f}  ({hf_time:.2f}s)")

    hf_grads = {}
    for name, param in hf_model.named_parameters():
        if param.grad is not None:
            hf_grads[name] = param.grad.float().clone()
    print(f"  Gradients collected: {len(hf_grads)}")

    # ------------------------------------------------------------------
    # 2. TTML backward (device, bfloat16)
    # ------------------------------------------------------------------
    print("\n[TTML] Forward + backward ...")
    t0 = time.time()

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    ttml_model.train()

    causal_mask = create_causal_mask_tensor(max_seq_len, device)

    if dp_size > 1:
        padded_np = np.zeros((dp_size, 1, 1, max_seq_len), dtype=np.int32)
        for d in range(dp_size):
            padded_np[d, 0, 0, : seq_lens[d]] = np.array(sequences[d], dtype=np.int32)
        input_tensor = create_input_tensor_dp(padded_np, device)
        logits = ttml_model(input_tensor, causal_mask, input_ids_np=padded_np)
    else:
        padded_input = torch.zeros(
            effective_batch, 1, 1, max_seq_len, dtype=torch.int32
        )
        for b in range(effective_batch):
            padded_input[b, 0, 0, : seq_lens[b]] = torch.tensor(
                sequences[b], dtype=torch.int32
            )
        input_tensor = create_input_tensor(padded_input, device)
        logits = ttml_model(
            input_tensor, causal_mask, input_ids_np=padded_input.numpy()
        )
    if track_memory:
        MemoryUsageTracker.snapshot("FORWARD_PASS")
    print("  Forward pass complete.")
    print(f"  [Backward] logits shape: {list(logits.shape())}")

    tp_size = device.shape[shard_dim] if (distributed and shard_dim is not None) else 1
    if sharded_loss:
        print(
            f"  [Backward] sharded_loss: logits kept sharded, "
            f"per-device vocab dim = {list(logits.shape())[3]}"
        )

    target_np = np.zeros((effective_batch, max_seq_len), dtype=np.uint32)
    for b in range(effective_batch):
        if seq_lens[b] > 1:
            target_np[b, : seq_lens[b] - 1] = np.array(
                sequences[b][1:], dtype=np.uint32
            )

    if sharded_loss:
        print(
            f"  [Backward] using distributed cross-entropy "
            f"(tp_size={tp_size}, dp_size={dp_size})"
        )
        ttml_loss = sharded_cross_entropy_loss(
            logits, target_np, vocab_padded, tp_size, tp_axis=shard_dim, dp_size=dp_size
        )
    elif dp_size > 1:
        dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0, 0)
        target_tensor = ttml.autograd.Tensor.from_numpy(
            target_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, dp_mapper
        )
        print(f"  [Backward] target ttml shape: {list(target_tensor.shape())}")
        ttml_loss = ttml.ops.loss.cross_entropy_loss(
            logits, target_tensor, ttml.ops.ReduceType.MEAN
        )
    else:
        target_torch = torch.from_numpy(target_np).to(torch.int32)
        target_host = ttnn.from_torch(target_torch, dtype=ttnn.uint32)
        target_dev = ttnn.to_device(target_host, device)
        target_tensor = ttml.autograd.create_tensor(target_dev, requires_grad=False)
        print(f"  [Backward] target ttml shape: {list(target_tensor.shape())}")
        ttml_loss = ttml.ops.loss.cross_entropy_loss(
            logits, target_tensor, ttml.ops.ReduceType.MEAN
        )
    print(f"  [Backward] loss shape: {list(ttml_loss.shape())}")

    if distributed:
        loss_composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        loss_all = ttnn.to_torch(ttml_loss.get_value(), mesh_composer=loss_composer).to(
            torch.float32
        )
        if dp_size > 1:
            loss_val = loss_all[::tp_size].mean().item()
        else:
            loss_val = loss_all[0].item()
    else:
        loss_val = ttnn.to_torch(ttml_loss.get_value()).to(torch.float32).item()

    print(f"  [Backward] Starting backward with loss={loss_val:.6f}...")
    ttml_loss.backward(False)

    if dp_size > 1:
        ttml.core.distributed.synchronize_gradients(ttml_model.parameters())

    if track_memory:
        MemoryUsageTracker.snapshot("BACKWARD_PASS")
    ttml_time = time.time() - t0
    print(f"  Loss: {loss_val:.6f}  ({ttml_time:.2f}s)")

    # ------------------------------------------------------------------
    # 3. Extract TTML gradients
    # ------------------------------------------------------------------
    ttml_params = ttml_model.parameters()
    any_key = next(iter(ttml_params))
    root_prefix = any_key.split("/")[0]

    if distributed:
        mapping, inv_transforms, gs = _build_grad_mapping_distributed(
            config, root_prefix, tie_word_embeddings
        )

        ttml_grads = {}
        for hf_name, ttml_name in mapping.items():
            if ttml_name not in ttml_params:
                continue
            tensor = ttml_params[ttml_name]
            strategy = gs[hf_name]
            grad = _extract_grad_distributed(tensor, device, strategy, dp_size=dp_size)
            if grad is not None:
                ttml_grads[ttml_name] = grad
    else:
        mapping, inv_transforms, _ = _build_grad_mapping_single(
            config, root_prefix, tie_word_embeddings
        )

        ttml_grads = {}
        for name, tensor in ttml_params.items():
            if tensor.is_grad_initialized():
                grad_tt = tensor.get_grad()
                ttml_grads[name] = ttnn.to_torch(grad_tt).to(torch.float32)

    print(f"  Gradients collected: {len(ttml_grads)}")

    ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)

    # ------------------------------------------------------------------
    # 4. Compare per-parameter gradients
    # ------------------------------------------------------------------
    print("\nPer-parameter gradient diff (a=HF, b=TTML, eps=1e-4):")
    _compare_gradients(hf_grads, ttml_grads, mapping, inv_transforms)

    ctx.reset_graph()


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3: backward gradient comparison (HF vs ttml)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--mesh_shape",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("ROWS", "COLS"),
        help="Device mesh shape [rows, cols].  rows = DP degree, "
        "cols = TP degree.  Default: 1 1 (single device).",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (activation recomputation). "
        "Trades compute for memory during backward.",
    )
    parser.add_argument(
        "--sharded_loss",
        action="store_true",
        default=False,
        help="Keep LM head output sharded (no all-gather) and compute "
        "loss on per-device vocab shards.  Dramatically reduces "
        "peak memory for the LM head forward/backward.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for backward pass. With bs>1 inputs are "
        "'1. <prompt>', '2. <prompt>', ... (default: 1)",
    )
    parser.add_argument(
        "--track_memory",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Enable memory usage tracking. Optional int N = snapshot every N-th "
        "layer (default: every layer). Use --track_memory or --track_memory 4.",
    )
    args = parser.parse_args()

    dp_size, tp_size = args.mesh_shape

    distributed = tp_size > 1 or dp_size > 1
    use_distributed_model = tp_size > 1

    # ------------------------------------------------------------------
    # 1. Load HuggingFace model (pretrained)
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HuggingFace model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    hf_model.eval()
    hf_config = hf_model.config
    hf_state_dict = hf_model.state_dict()

    # ------------------------------------------------------------------
    # 2. Build prompts and tokenize
    # ------------------------------------------------------------------
    if dp_size > 1:
        count = dp_size
    elif args.batch_size > 1:
        count = args.batch_size
    else:
        count = 1

    if count > 1:
        width = len(str(count))
        prompts = [f"{str(i+1).zfill(width)}. {args.prompt}" for i in range(count)]
    else:
        prompts = [args.prompt]

    all_prompt_tokens = [tokenizer.encode(p) for p in prompts]
    token_lens = [len(pt) for pt in all_prompt_tokens]
    if len(set(token_lens)) > 1:
        print(f"  WARNING: prompt token lengths differ: {token_lens}")
    for i, p in enumerate(prompts):
        print(f"Prompt[{i}]: {p!r}  →  {len(all_prompt_tokens[i])} tokens")

    # ------------------------------------------------------------------
    # 3. Set up Tenstorrent device (single or mesh)
    # ------------------------------------------------------------------
    from utils.device_setup import setup_device
    from utils.model_factory import create_ttml_model, load_hf_weights

    ctx, device = setup_device(dp_size, tp_size)

    memory_guard = None
    if args.track_memory:
        print("Memory tracking enabled")
        memory_guard = MemoryUsageTracker.begin_capture()
        MemoryUsageTracker.snapshot("BEFORE_MODEL_CREATION")

    # ------------------------------------------------------------------
    # 4. Create ttml model and load weights
    # ------------------------------------------------------------------
    ttml_model, config, tie, shard_dim, mode_str = create_ttml_model(
        hf_config,
        args.max_seq_len,
        dp_size=dp_size,
        tp_size=tp_size,
        checkpoint=args.checkpoint,
        track_memory=args.track_memory,
        sharded_loss=args.sharded_loss,
    )
    if not use_distributed_model and args.sharded_loss:
        args.sharded_loss = False

    if args.track_memory:
        MemoryUsageTracker.snapshot("AFTER_MODEL_CREATION")

    load_hf_weights(
        ttml_model,
        hf_state_dict,
        config,
        tie=tie,
        tp=use_distributed_model,
        shard_dim=shard_dim,
    )

    if args.track_memory:
        MemoryUsageTracker.snapshot("AFTER_WEIGHT_LOADING")

    del hf_state_dict

    # ------------------------------------------------------------------
    # 5. Build sequences and run backward comparison
    # ------------------------------------------------------------------
    sequences = [list(all_prompt_tokens[i]) for i in range(len(prompts))]

    for i, seq in enumerate(sequences):
        label = f"DP[{i}]" if dp_size > 1 else f"Batch[{i}]"
        print(
            f"\n  {label} backward sequence ({len(seq)} tokens): "
            f"{tokenizer.decode(seq)!r}"
        )

    run_backward_comparison(
        hf_model,
        ttml_model,
        config,
        sequences,
        args.max_seq_len,
        device,
        tie_word_embeddings=tie,
        distributed=distributed,
        shard_dim=shard_dim,
        track_memory=args.track_memory,
        tokenizer=tokenizer,
        sharded_loss=getattr(ttml_model, "sharded_loss", False),
        dp_size=dp_size,
    )

    if args.track_memory and memory_guard is not None:
        finalize_memory(
            memory_guard,
            label="FIRST_ITERATION_COMPLETE",
            title="Memory Usage Report (after first iteration / compilation)",
        )
        memory_guard = None

    # ------------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------------
    if args.track_memory:
        finalize_memory(memory_guard)

    ctx.close_device()


if __name__ == "__main__":
    main()
