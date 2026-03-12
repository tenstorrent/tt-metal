# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 training script for ttml (single-device and tensor-parallel).

Loads a pretrained Qwen3 model from HuggingFace, converts it to ttml, and
fine-tunes on a next-token-prediction objective using cross-entropy loss.

Supports:
  - Single-device training (default)
  - Tensor-parallel training (--mesh_shape 1 8)
  - Data-parallel + tensor-parallel (--mesh_shape 4 8)
  - Gradient accumulation (--gradient_accumulation_steps)
  - Cosine LR schedule with linear warmup (--warmup_steps, --lr)
  - Periodic evaluation and text generation (--eval_every, --gen_every)
  - Gradient clipping (--clip_grad_norm)
  - Training on HuggingFace datasets or local text files
  - Frozen embedding/lm_head for finetuning (--freeze_embeddings, default on)
  - LoRA finetuning (--lora_rank, --lora_alpha, --lora_targets)
  - Checkpointing to safetensors (--save_dir, --save_every, --resume_from)
  - HF-compatible export after training (--export_hf_dir)
  - TensorBoard logging (automatic when --save_dir is set)

Usage:
    # Single device, fine-tune on wikitext:
    python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 --batch_size 1

    # Save checkpoints every 100 steps:
    python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 \\
        --save_dir ./checkpoints --save_every 100

    # Resume from a checkpoint:
    python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 \\
        --save_dir ./checkpoints --resume_from ./checkpoints/step_200

    # Export fine-tuned model for use with HuggingFace:
    python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 \\
        --export_hf_dir ./my_finetuned_model

    # Tensor-parallel fine-tune:
    python train.py --model_path Qwen/Qwen3-8B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 --mesh_shape 1 8

    # Fine-tune on a local text file:
    python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \\
        --dataset /path/to/corpus.txt --steps 1000 --lr 5e-5

    # Single device with gradient checkpointing (saves memory):
    python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 --checkpoint

    # Tensor-parallel with gradient checkpointing (saves memory):
    python train.py --model_path Qwen/Qwen3-8B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 --mesh_shape 1 8 --checkpoint

    # LoRA finetuning (trains only low-rank adapters, base weights frozen):
    python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 --lora_rank 8

    # Checkpoint with scattered intermediates (further memory savings;
    # scatters saved activations across TP devices, requires batch_size % tp_size == 0):
    python train.py --model_path Qwen/Qwen3-8B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 --mesh_shape 1 8 --checkpoint \\
        --scatter_intermediates

    # Save checkpoints + TensorBoard logs (view with `tensorboard --logdir ./output/logs`):
    python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \\
        --dataset wikitext --steps 500 --lr 1e-4 --save_dir ./output
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import ttml

from utils.sharded_loss import sharded_cross_entropy_loss
from utils.lora import LORA_TARGETS_ALL, inject_adapter_in_model
from utils.memory import MemoryUsageTracker, finalize_memory
from ttml.common.utils import no_grad
from utils.tensor_utils import (
    create_causal_mask,
    create_input_tensor,
    create_target_tensor,
    get_loss_value,
    extract_logits,
)
from utils.save_load import (
    export_hf_model,
    load_checkpoint,
    load_model_from_safetensors,
    save_checkpoint,
    _load_hf_dict_into_ttml,
)


# =====================================================================
# Learning rate schedule
# =====================================================================


def cosine_lr_schedule(step, warmup_steps, total_steps, max_lr, min_lr):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def constant_lr_schedule(step, warmup_steps, max_lr):
    """Linear warmup then constant at peak LR."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    return max_lr


# =====================================================================
# Data loading — on-the-fly tokenization
# =====================================================================
from utils.dataset import TextDataset, load_text_datasets  # noqa: E402


# =====================================================================
# Evaluation
# =====================================================================


def evaluate(
    model,
    val_dataset,
    seq_len,
    batch_size,
    num_batches,
    causal_mask,
    distributed=False,
    dp_mapper=None,
    sharded_loss=False,
    vocab_padded=0,
    tp_size=1,
    shard_dim=1,
):
    """Compute average validation loss over num_batches."""
    model.eval()
    ctx = ttml.autograd.AutoContext.get_instance()
    with no_grad():
        losses = []
        for _ in range(num_batches):
            x_np, y_np = val_dataset.get_batch(batch_size)
            x_np = x_np.reshape(batch_size, 1, 1, seq_len)

            input_tensor = create_input_tensor(x_np, dp_mapper)

            logits = model(input_tensor, causal_mask, input_ids_np=x_np)
            if sharded_loss:
                loss = sharded_cross_entropy_loss(
                    logits, y_np, vocab_padded, tp_size, tp_axis=shard_dim
                )
            else:
                target_tensor = create_target_tensor(y_np, dp_mapper)
                loss = ttml.ops.loss.cross_entropy_loss(
                    logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN
                )
            losses.append(get_loss_value(loss, distributed))
            ctx.reset_graph()

    model.train()
    return float(np.mean(losses))


# =====================================================================
# Text generation (for periodic evaluation)
# =====================================================================


def generate_text(
    model, config, tokenizer, prompt, max_tokens, max_seq_len, device, distributed=False
):
    """Generate text using the model (greedy decoding)."""
    model.eval()
    ctx = ttml.autograd.AutoContext.get_instance()
    with no_grad():
        orig_vocab = config.vocab_size
        causal_mask = create_causal_mask(max_seq_len)

        prompt_tokens = tokenizer.encode(prompt)
        current_tokens = list(prompt_tokens)
        generated = []

        for _ in range(max_tokens):
            padded = np.zeros((1, 1, 1, max_seq_len), dtype=np.uint32)
            start_idx = max(0, len(current_tokens) - max_seq_len)
            tokens_window = current_tokens[start_idx:]
            padded[0, 0, 0, : len(tokens_window)] = np.array(
                tokens_window, dtype=np.uint32
            )

            input_tensor = create_input_tensor(padded)
            logits = model(input_tensor, causal_mask, input_ids_np=padded)
            logits_np = extract_logits(logits, distributed)

            pred_pos = len(tokens_window) - 1
            next_token_logits = logits_np[0, 0, pred_pos, :orig_vocab]
            next_token = int(np.argmax(next_token_logits))

            generated.append(next_token)
            current_tokens.append(next_token)

            ctx.reset_graph()

            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and next_token == eos_id:
                break

    model.train()
    return tokenizer.decode(generated)


# =====================================================================
# Main training function
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 training with ttml (single-device, TP, and DP+TP)"
    )
    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model path (e.g., Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--mesh_shape",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("ROWS", "COLS"),
        help="Device mesh shape [rows, cols].  rows = DP degree, "
        "cols = TP degree.  Default: 1 1 (single device).",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset: 'wikitext', a HF dataset name, path to .txt file, "
        "(tokenized on-the-fly, no cache)",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1, help="Micro-batch size")
    parser.add_argument("--steps", type=int, default=500, help="Total optimizer steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate (cosine schedule floor)",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=50, help="Linear warmup steps"
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["cosine", "constant"],
        help="LR schedule after warmup: 'cosine' decays to min_lr, 'constant' holds at peak LR",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay"
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps per optimizer step",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.0,
        help="Max gradient norm for clipping (0=disabled)",
    )

    # Evaluation
    parser.add_argument(
        "--eval_every", type=int, default=50, help="Evaluate every N optimizer steps"
    )
    parser.add_argument(
        "--valid_mul",
        type=int,
        default=4,
        help="Validation size as a multiplier of the total training batch size "
        "(batch_size * gradient_accumulation_steps * dp_size). "
        "E.g. with batch_size=4, gradient_accumulation=8, valid_mul=4 "
        "the validation uses 4*8*4=128 samples.",
    )
    parser.add_argument(
        "--gen_every",
        type=int,
        default=100,
        help="Generate sample text every N optimizer steps",
    )
    parser.add_argument(
        "--gen_prompt",
        type=str,
        default="Once upon a time",
        help="Prompt for periodic text generation",
    )
    parser.add_argument(
        "--gen_tokens",
        type=int,
        default=32,
        help="Number of tokens to generate for evaluation",
    )

    # Memory / checkpointing
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (activation recomputation). "
        "Trades compute for memory during backward.",
    )
    parser.add_argument(
        "--scatter_intermediates",
        action="store_true",
        default=False,
        help="When used with --checkpoint, scatter saved activations across "
        "TP devices to reduce per-device memory by tp_size. "
        "Requires batch_size %% tp_size == 0.",
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
    parser.add_argument(
        "--sharded_loss",
        action="store_true",
        default=False,
        help="Keep LM-head output vocab-sharded and use distributed cross-entropy "
        "loss (avoids all-gathering the full vocabulary).",
    )

    # Finetuning
    parser.add_argument(
        "--freeze_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze embedding and lm_head layers during finetuning "
        "(default: True). Use --no-freeze_embeddings to train all params.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="LoRA rank (0=disabled). Typical values: 4, 8, 16, 32.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA scaling alpha (default: same as rank, giving scaling=1.0)",
    )
    parser.add_argument(
        "--lora_targets",
        type=str,
        nargs="+",
        default=None,
        help="LoRA target modules (default: all projections). "
        "Choices: q_proj, k_proj, v_proj, o_proj, "
        "gate_proj, up_proj, down_proj",
    )
    # Checkpointing
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory. Checkpoints are saved under "
        "<save_dir>/checkpoints/step_<N> and TensorBoard logs "
        "under <save_dir>/logs/{train,valid}. "
        "If None, no checkpoints or TB logs are written.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save a checkpoint every N optimizer steps. "
        "0 (default) saves only at the end of training. "
        "Requires --save_dir.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from. "
        "Loads model weights, optimizer state, and LoRA adapters "
        "(if applicable), then continues training from the saved step.",
    )
    parser.add_argument(
        "--export_hf_dir",
        type=str,
        default=None,
        help="If set, export the final fine-tuned model in HF-compatible "
        "safetensors format to this directory after training completes. "
        "LoRA adapters are merged into the base weights if LoRA was used.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_log",
        type=str,
        default="train_log.txt",
        help="Path to output training log file",
    )
    parser.add_argument(
        "--token_cache_dir",
        type=str,
        default=".token_cache",
        help="Directory for pre-tokenized dataset cache files "
        "(default: .token_cache). Pass an empty string to disable caching.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        default=False,
        help="Print per-step timing breakdown for each training operation "
        "(zero_grad, forward, loss, backward, opt_step, eval, etc.)",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # TensorBoard writers (created when --save_dir is set)
    tb_train_writer = None
    tb_val_writer = None
    if args.save_dir:
        from torch.utils.tensorboard import SummaryWriter

        tb_train_writer = SummaryWriter(
            log_dir=os.path.join(args.save_dir, "logs", "train")
        )
        tb_val_writer = SummaryWriter(
            log_dir=os.path.join(args.save_dir, "logs", "valid")
        )

    # ------------------------------------------------------------------
    # 1. Load HuggingFace model and tokenizer
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HuggingFace model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    hf_config = hf_model.config
    hf_state_dict = hf_model.state_dict()

    # Free HF model (we only need its weights)
    del hf_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------
    # 2. Load dataset and pre-tokenize (with disk cache)
    # ------------------------------------------------------------------

    train_texts, val_texts = load_text_datasets(args.dataset)
    cache_dir = args.token_cache_dir or ".token_cache"
    print("\nPre-tokenizing datasets...")
    train_dataset = TextDataset(
        train_texts,
        tokenizer,
        args.max_seq_len,
        cache_dir=cache_dir,
        split="train",
    )
    val_dataset = TextDataset(
        val_texts,
        tokenizer,
        args.max_seq_len,
        cache_dir=cache_dir,
        split="val",
    )

    # ------------------------------------------------------------------
    # 3. Set up Tenstorrent device
    # ------------------------------------------------------------------
    dp_size = args.mesh_shape[0]
    tp_size = args.mesh_shape[1]
    distributed = tp_size > 1 or dp_size > 1
    use_distributed_model = tp_size > 1

    from utils.device_setup import setup_device

    ctx, device = setup_device(dp_size, tp_size, seed=args.seed)

    # Start memory tracking if enabled
    memory_guard = None
    if args.track_memory:
        print("Memory tracking enabled")
        memory_guard = MemoryUsageTracker.begin_capture()

    # DP mapper: shards batch dim (0) across DP groups (mesh dim 0)
    dp_mapper = None
    if dp_size > 1:
        dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0, 0)

    # ------------------------------------------------------------------
    # 4. Create ttml model and load weights
    # ------------------------------------------------------------------
    # Memory snapshot before model creation
    if args.track_memory:
        MemoryUsageTracker.snapshot("BEFORE_MODEL_CREATION")

    # Build LoRA config (None when disabled)
    lora_config = None
    if args.lora_rank > 0:
        lora_alpha = (
            args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank)
        )
        if args.lora_targets is not None:
            # Handle comma-separated targets (e.g. "q_proj,k_proj" → ["q_proj", "k_proj"])
            expanded = []
            for t in args.lora_targets:
                expanded.extend(t.split(","))
            lora_targets = [t.strip() for t in expanded if t.strip()]
            invalid = [t for t in lora_targets if t not in LORA_TARGETS_ALL]
            if invalid:
                parser.error(
                    f"Invalid --lora_targets: {invalid}. "
                    f"Valid choices: {', '.join(LORA_TARGETS_ALL)}"
                )
        else:
            lora_targets = list(LORA_TARGETS_ALL)
        lora_config = {
            "rank": args.lora_rank,
            "alpha": lora_alpha,
            "targets": lora_targets,
        }
        print(
            f"\nLoRA enabled: rank={args.lora_rank}, alpha={lora_alpha}, "
            f"targets={lora_targets}"
        )

    from utils.model_factory import create_ttml_model, load_hf_weights
    from utils.tensor_utils import tile_pad

    ttml_model, config, tie, shard_dim, mode_str = create_ttml_model(
        hf_config,
        args.max_seq_len,
        dp_size=dp_size,
        tp_size=tp_size,
        checkpoint=args.checkpoint,
        scatter_intermediates=args.scatter_intermediates,
        track_memory=args.track_memory,
        sharded_loss=args.sharded_loss,
    )
    if not (tp_size > 1) and args.sharded_loss:
        args.sharded_loss = False
    vocab_padded = tile_pad(config.vocab_size)

    # Memory snapshot after model creation
    if args.track_memory:
        MemoryUsageTracker.snapshot("AFTER_MODEL_CREATION")

    # ------------------------------------------------------------------
    # 4b. Validate checkpoint args early (before weight loading)
    # ------------------------------------------------------------------
    if args.save_every > 0 and not args.save_dir:
        parser.error("--save_every requires --save_dir to be set.")
    if args.resume_from and not os.path.isdir(args.resume_from):
        parser.error(f"--resume_from path does not exist: {args.resume_from}")

    # Always load the original HF weights first — this guarantees ALL parameters
    # (including frozen embed_tokens / lm_head) are properly initialized.
    # For full fine-tune resume, checkpoint weights overwrite the trainable
    # params afterwards; frozen params keep their HF values.
    print("\nLoading HF weights into ttml model...")
    t0 = time.time()
    load_hf_weights(
        ttml_model,
        hf_state_dict,
        config,
        tie=tie,
        tp=use_distributed_model,
        shard_dim=shard_dim,
    )
    print(f"  HF weight loading took {time.time() - t0:.2f}s")
    del hf_state_dict

    if args.resume_from and lora_config is None:
        ckpt_model_path = os.path.join(args.resume_from, "model.safetensors")
        if not os.path.exists(ckpt_model_path):
            parser.error(f"No model checkpoint found at {ckpt_model_path}")
        print(f"\nOverwriting trainable weights from checkpoint: {args.resume_from}")
        t0 = time.time()
        ckpt_state_dict = load_model_from_safetensors(args.resume_from)
        _load_hf_dict_into_ttml(
            ttml_model,
            ckpt_state_dict,
            config,
            tie_word_embeddings=tie,
            distributed=use_distributed_model,
            shard_dim=shard_dim if use_distributed_model else None,
        )
        del ckpt_state_dict
        print(f"  Checkpoint weight loading took {time.time() - t0:.2f}s")

    if lora_config is not None:
        print("\nInjecting LoRA adapters...")
        inject_adapter_in_model(ttml_model, lora_config)
        print(
            f"  LoRA injected: rank={lora_config['rank']}, alpha={lora_config['alpha']}, "
            f"targets={lora_config['targets']}"
        )

    # Memory snapshot after weight loading
    if args.track_memory:
        MemoryUsageTracker.snapshot("AFTER_WEIGHT_LOADING")

    # ------------------------------------------------------------------
    # 5. Set up optimizer and training state
    # ------------------------------------------------------------------

    # Select trainable parameters
    all_params = ttml_model.parameters()
    if lora_config is not None:
        trainable_params = {
            name: param for name, param in all_params.items() if "lora" in name
        }
        frozen_count = len(all_params) - len(trainable_params)
        print(
            f"\nLoRA finetuning: {len(trainable_params)} LoRA params trainable, "
            f"{frozen_count} base params frozen"
        )
    elif args.freeze_embeddings:
        trainable_params = {
            name: param
            for name, param in all_params.items()
            if "embed_tokens" not in name and "lm_head" not in name
        }
        frozen_count = len(all_params) - len(trainable_params)
        print(
            f"\nFreezing embedding + lm_head: {frozen_count} params frozen, "
            f"{len(trainable_params)} trainable"
        )
    else:
        trainable_params = all_params
        print(f"\nAll {len(trainable_params)} params are trainable")

    """
    non_trainable_params = {
        name: param for name, param in all_params.items() if name not in trainable_params
    }
    for name, weight in non_trainable_params.items():
        weight.tensor.set_requires_grad(False)
    if non_trainable_params:
        print(f"Set requires_grad=False for {len(non_trainable_params)} non-trainable params")
    """

    print("Setting up optimizer...")
    adamw_config = ttml.optimizers.AdamWConfig.make(
        args.lr,
        args.beta1,
        args.beta2,
        args.eps,
        args.weight_decay,
    )
    optimizer = ttml.optimizers.AdamW(trainable_params, adamw_config)

    # Memory snapshot after optimizer creation
    if args.track_memory:
        MemoryUsageTracker.snapshot("AFTER_OPTIMIZER_CREATION")

    # ------------------------------------------------------------------
    # 5b. Resume from checkpoint (optimizer state + optional LoRA adapters)
    # ------------------------------------------------------------------
    resume_step = 0
    if args.resume_from:
        # Model weights for non-LoRA resume were already loaded in section 4b
        # (before optimizer creation) to avoid uninitialized tensor issues.
        resume_step = load_checkpoint(
            ttml_model,
            optimizer,
            args.resume_from,
            config,
            tie_word_embeddings=tie,
            distributed=use_distributed_model,
            device=device if use_distributed_model else None,
            shard_dim=shard_dim if use_distributed_model else None,
            dp_size=dp_size,
            lora_config=lora_config,
            skip_model_weights=True,
        )

    # Causal mask (shared across all steps)
    causal_mask = create_causal_mask(args.max_seq_len)

    # Training state
    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    ttml_model.train()

    seq_len = args.max_seq_len
    micro_batch = args.batch_size
    # In DP mode, total batch = micro_batch * dp_size (each device gets micro_batch)
    accum_steps = args.gradient_accumulation_steps
    total_steps = args.steps
    use_clip = args.clip_grad_norm > 0.0

    tokens_per_step = micro_batch * dp_size * seq_len * accum_steps
    eval_batches = max(1, accum_steps * args.valid_mul)
    print(f"\nTraining config:")
    print(f"  Steps: {total_steps}")
    print(f"  Micro-batch size per device: {micro_batch}")
    if dp_size > 1:
        print(f"  Global batch size: {micro_batch * dp_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Gradient accumulation: {accum_steps}")
    print(f"  Tokens per optimizer step: {tokens_per_step:,}")
    print(f"  Peak LR: {args.lr}")
    print(f"  LR schedule: {args.lr_schedule}")
    print(f"  Warmup steps: {args.warmup_steps}")
    eval_bs = micro_batch * dp_size if dp_size > 1 else micro_batch
    eval_samples = eval_batches * eval_bs
    print(
        f"  Validation: {args.valid_mul}x total_batch = {eval_samples} samples ({eval_batches} batches of {eval_bs})"
    )
    print(
        f"  Gradient clipping: {'%.1f' % args.clip_grad_norm if use_clip else 'disabled'}"
    )
    ckpt_status = "disabled"
    if args.checkpoint:
        ckpt_status = "enabled" + (
            " + scatter_intermediates" if args.scatter_intermediates else ""
        )
    print(f"  Gradient checkpointing: {ckpt_status}")
    print(f"  Freeze embedding/lm_head: {args.freeze_embeddings}")
    if lora_config:
        print(
            f"  LoRA: rank={lora_config['rank']}, alpha={lora_config['alpha']}, "
            f"targets={lora_config['targets']}"
        )
    else:
        print(f"  LoRA: disabled")
    print(f"  Distributed: {mode_str}")
    if args.save_dir:
        print(f"  Output dir: {args.save_dir}")
        print(
            f"    Checkpoints: {os.path.join(args.save_dir, 'checkpoints')} (every {args.save_every or 'final'} steps)"
        )
        print(f"    TensorBoard: {os.path.join(args.save_dir, 'logs')} (train + valid)")
        tb_train_writer.add_text("config", f"```\n{vars(args)}\n```", 0)
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from} (step {resume_step})")
    if args.export_hf_dir:
        print(f"  HF export dir: {args.export_hf_dir}")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Initial validation loss (before any training)
    # ------------------------------------------------------------------
    print("\nEvaluating initial validation loss (before training)...")
    initial_val_loss = evaluate(
        ttml_model,
        val_dataset,
        seq_len,
        eval_bs,
        eval_batches,
        causal_mask,
        distributed,
        dp_mapper,
        sharded_loss=args.sharded_loss,
        vocab_padded=vocab_padded,
        tp_size=tp_size,
        shard_dim=shard_dim if shard_dim is not None else 1,
    )
    print(f"  Initial val loss: {initial_val_loss:.4f}")
    if tb_val_writer is not None:
        tb_val_writer.add_scalar("loss", initial_val_loss, resume_step)

    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    ttml_model.train()

    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}\n")

    # Flag to track if first iteration is complete (for memory tracking)
    is_everything_compiled = False

    train_losses = []
    val_losses = [(resume_step, initial_val_loss)]
    last_val_loss = initial_val_loss
    log_lines = [f"step={resume_step}, val_loss={initial_val_loss:.4f}"]
    total_tokens = 0
    train_start = time.time()

    timings = args.timings

    def _tlog(step, tag, t0):
        """Print a timing line if --timings is enabled. Returns current time."""
        if timings:
            elapsed = time.time() - t0
            print(f"  [{step}] {tag:<16s} {elapsed:.3f}s")
        return time.time()

    bar = tqdm(range(resume_step + 1, total_steps + 1), desc="Training")
    for step in bar:
        step_start = time.time()
        if timings:
            print(f">>> STEP {step} START")

        # Learning rate schedule
        if args.lr_schedule == "constant":
            lr_now = constant_lr_schedule(step - 1, args.warmup_steps, args.lr)
        else:
            lr_now = cosine_lr_schedule(
                step - 1, args.warmup_steps, total_steps, args.lr, args.min_lr
            )
        # Update optimizer LR
        optimizer.set_lr(lr_now)

        # Zero gradients
        t0 = time.time()
        optimizer.zero_grad()
        t0 = _tlog(step, "zero_grad", t0)

        # Gradient accumulation loop
        accum_loss = 0.0
        for micro_step in range(accum_steps):
            if timings:
                print(f"  [{step}] micro_step {micro_step}/{accum_steps}")

            # In DP mode, sample micro_batch * dp_size tokens and stack
            # so batch dim [dp_size * micro_batch, ...] gets sharded by dp_mapper
            if dp_size > 1:
                global_bs = micro_batch * dp_size
                x_np, y_np = train_dataset.get_batch(global_bs)
                x_np = x_np.reshape(global_bs, 1, 1, seq_len)
            else:
                x_np, y_np = train_dataset.get_batch(micro_batch)
                x_np = x_np.reshape(micro_batch, 1, 1, seq_len)

            t0 = time.time()
            input_tensor = create_input_tensor(x_np, dp_mapper)
            t0 = _tlog(step, "create_input", t0)

            # Forward pass
            logits = ttml_model(input_tensor, causal_mask, input_ids_np=x_np)
            t0 = _tlog(step, "forward", t0)

            # Memory snapshot after forward pass (only during first iteration)
            if args.track_memory and not is_everything_compiled and micro_step == 0:
                MemoryUsageTracker.snapshot("FORWARD_PASS")

            # Cross-entropy loss
            if args.sharded_loss:
                loss = sharded_cross_entropy_loss(
                    logits, y_np, vocab_padded, tp_size, tp_axis=shard_dim
                )
            else:
                target_tensor = create_target_tensor(y_np, dp_mapper)
                loss = ttml.ops.loss.cross_entropy_loss(
                    logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN
                )
            t0 = _tlog(step, "loss_compute", t0)

            accum_loss += get_loss_value(loss, distributed)
            t0 = _tlog(step, "loss_sync", t0)

            # Scale for gradient accumulation
            if accum_steps > 1:
                loss = loss * (1.0 / float(accum_steps))

            # Backward pass
            loss.backward(False)
            t0 = _tlog(step, "backward", t0)

            # Memory snapshot after backward pass (only during first iteration)
            if args.track_memory and not is_everything_compiled and micro_step == 0:
                MemoryUsageTracker.snapshot("BACKWARD_PASS")

            ctx.reset_graph()
            t0 = _tlog(step, "reset_graph", t0)

        # Synchronize gradients across DP groups (all-reduce along mesh dim 0)
        if dp_size > 1:
            t0 = time.time()
            ttml.core.distributed.synchronize_gradients(ttml_model.parameters())
            t0 = _tlog(step, "sync_grads", t0)

        # Average accumulated loss
        step_loss = accum_loss / accum_steps
        train_losses.append(step_loss)

        # Gradient clipping
        if use_clip:
            t0 = time.time()
            ttml.core.clip_grad_norm(
                ttml_model.parameters(),
                args.clip_grad_norm,
                2.0,  # L2 norm
                False,  # error_if_nonfinite
            )
            t0 = _tlog(step, "clip_grad", t0)

        # Optimizer step
        t0 = time.time()
        optimizer.step()
        t0 = _tlog(step, "opt_step", t0)

        # Print memory usage after first iteration (compilation complete)
        if args.track_memory and not is_everything_compiled:
            is_everything_compiled = True
            finalize_memory(
                memory_guard,
                label="FIRST_ITERATION_COMPLETE",
                title="Memory Usage Report (after first iteration / compilation)",
            )
            memory_guard = None

        # Periodic checkpoint
        if args.save_dir and args.save_every > 0 and step % args.save_every == 0:
            ckpt_dir = os.path.join(args.save_dir, "checkpoints", f"step_{step}")
            save_checkpoint(
                step=step,
                ttml_model=ttml_model,
                optimizer=optimizer,
                config=config,
                save_dir=ckpt_dir,
                tie_word_embeddings=tie,
                distributed=use_distributed_model,
                device=device if use_distributed_model else None,
                shard_dim=shard_dim if use_distributed_model else None,
                dp_size=dp_size,
                lora_config=lora_config,
                args_dict=vars(args),
            )

        total_tokens += tokens_per_step
        step_time = time.time() - step_start
        tokens_per_sec = tokens_per_step / step_time

        if tb_train_writer is not None:
            tb_train_writer.add_scalar("loss", step_loss, step)
            tb_train_writer.add_scalar("lr", lr_now, step)
            tb_train_writer.add_scalar(
                "throughput/tokens_per_sec", tokens_per_sec, step
            )
            tb_train_writer.add_scalar("throughput/step_time_sec", step_time, step)

        # Update progress bar
        postfix = {
            "loss": f"{step_loss:.4f}",
            "lr": f"{lr_now:.2e}",
            "tok/s": f"{tokens_per_sec:.0f}",
        }

        # Periodic evaluation
        if step % args.eval_every == 0 or step == resume_step + 1:
            t0 = time.time()
            val_loss = evaluate(
                ttml_model,
                val_dataset,
                seq_len,
                eval_bs,
                eval_batches,
                causal_mask,
                distributed,
                dp_mapper,
                sharded_loss=args.sharded_loss,
                vocab_padded=vocab_padded,
                tp_size=tp_size,
                shard_dim=shard_dim if shard_dim is not None else 1,
            )
            _tlog(step, "eval", t0)
            val_losses.append((step, val_loss))
            last_val_loss = val_loss
            if tb_val_writer is not None:
                tb_val_writer.add_scalar("loss", val_loss, step)
            if timings:
                print(f"  [{step}] val_loss={val_loss:.4f}")

            # Re-enable training mode
            ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            ttml_model.train()

        if last_val_loss is not None:
            postfix["val_loss"] = f"{last_val_loss:.4f}"

        bar.set_postfix(postfix, refresh=False)

        # Periodic text generation
        if step % args.gen_every == 0:
            print(f"\n--- Generation at step {step} ---")
            gen_text = generate_text(
                ttml_model,
                config,
                tokenizer,
                args.gen_prompt,
                args.gen_tokens,
                seq_len,
                device,
                distributed,
            )
            print(f"Prompt: {args.gen_prompt!r}")
            print(f"Output: {gen_text!r}")
            print("---")
            if tb_train_writer is not None:
                tb_train_writer.add_text(
                    "generation",
                    f"**Prompt:** {args.gen_prompt}  \n**Output:** {gen_text}",
                    step,
                )

            # Re-enable training mode
            ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            ttml_model.train()

        if timings:
            print(f"  [{step}] STEP COMPLETE  {time.time() - step_start:.3f}s")

        # Log
        log_line = (
            f"step={step}, loss={step_loss:.4f}, lr={lr_now:.6e}, "
            f"tok/s={tokens_per_sec:.0f}, total_tokens={total_tokens:,}"
        )
        if val_losses and val_losses[-1][0] == step:
            log_line += f", val_loss={val_losses[-1][1]:.4f}"
        log_lines.append(log_line)

    # ------------------------------------------------------------------
    # 7. Final summary
    # ------------------------------------------------------------------
    train_time = time.time() - train_start
    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"{'=' * 70}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total time: {train_time:.1f}s")
    print(f"  Avg tokens/sec: {total_tokens / train_time:.0f}")
    print(f"  Initial val loss: {initial_val_loss:.4f}")
    if train_losses:
        print(f"  Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Final val loss: {val_losses[-1][1]:.4f}")

    # Final generation
    print(f"\n--- Final generation ---")
    gen_text = generate_text(
        ttml_model,
        config,
        tokenizer,
        args.gen_prompt,
        args.gen_tokens,
        seq_len,
        device,
        distributed,
    )
    print(f"Prompt: {args.gen_prompt!r}")
    print(f"Output: {gen_text!r}")

    # ------------------------------------------------------------------
    # Save final checkpoint
    # ------------------------------------------------------------------
    if args.save_dir:
        ckpt_dir = os.path.join(args.save_dir, "checkpoints", f"step_{total_steps}")
        save_checkpoint(
            step=total_steps,
            ttml_model=ttml_model,
            optimizer=optimizer,
            config=config,
            save_dir=ckpt_dir,
            tie_word_embeddings=tie,
            distributed=use_distributed_model,
            device=device if use_distributed_model else None,
            shard_dim=shard_dim if use_distributed_model else None,
            dp_size=dp_size,
            lora_config=lora_config,
            args_dict=vars(args),
        )

    # ------------------------------------------------------------------
    # Export HF-compatible model
    # ------------------------------------------------------------------
    if args.export_hf_dir:
        export_hf_model(
            ttml_model=ttml_model,
            config=config,
            save_dir=args.export_hf_dir,
            tie_word_embeddings=tie,
            original_model_path=args.model_path,
            distributed=use_distributed_model,
            device=device if use_distributed_model else None,
            shard_dim=shard_dim if use_distributed_model else None,
            dp_size=dp_size,
            lora_config=lora_config,
        )

    # Save training log
    if args.output_log:
        with open(args.output_log, "w") as f:
            f.write(f"# Qwen3 Training Log\n")
            f.write(f"# Model: {args.model_path}\n")
            f.write(f"# Mode: {mode_str}\n")
            f.write(f"# Dataset: {args.dataset}\n")
            f.write(
                f"# Micro-batch: {micro_batch}, DP: {dp_size}, Seq len: {seq_len}\n"
            )
            f.write(f"# LR: {args.lr}, Warmup: {args.warmup_steps}\n")
            f.write(f"# Checkpoint: {ckpt_status}\n")
            f.write(f"# Freeze embedding/lm_head: {args.freeze_embeddings}\n")
            if lora_config:
                f.write(
                    f"# LoRA: rank={lora_config['rank']}, alpha={lora_config['alpha']}, "
                    f"targets={lora_config['targets']}\n"
                )
            if args.save_dir:
                f.write(f"# TensorBoard: {os.path.join(args.save_dir, 'logs')}\n")
            f.write(f"# Total time: {train_time:.1f}s\n\n")
            for line in log_lines:
                f.write(line + "\n")
        print(f"\nTraining log saved to: {args.output_log}")

    # ------------------------------------------------------------------
    # 8. Cleanup
    # ------------------------------------------------------------------
    if tb_train_writer is not None:
        tb_train_writer.close()
    if tb_val_writer is not None:
        tb_val_writer.close()
    ctx.close_device()


if __name__ == "__main__":
    main()
