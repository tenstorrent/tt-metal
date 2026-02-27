# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 generation comparison: HuggingFace (PyTorch) vs ttml.

Key differences:
  - HF:   CPU/bfloat16, KV cache, processes one new token per step
  - ttml: Tenstorrent device/bfloat16, optional KV cache (--kv_cache) or
          full padded sequence each step (default, fixed shape avoids
          kernel recompilation)

For backward pass / gradient comparison, see gradients.py.

Usage:
    python generate.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \\
        --max_tokens 32 --max_seq_len 128

    # With tensor parallelism:
    python generate.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \\
        --max_tokens 32 --max_seq_len 128 --mesh_shape 1 8

    # With data parallelism:
    python generate.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \\
        --max_tokens 32 --max_seq_len 128 --mesh_shape 4 1

    # Combined DP + TP:
    python generate.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \\
        --max_tokens 32 --max_seq_len 128 --mesh_shape 4 8

    # On-device sampling (no large D2H logits transfer):
    python generate.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \\
        --max_tokens 32 --max_seq_len 128 --mesh_shape 1 8 --no_logits

    # Sharded loss + on-device sampling (all_gather before sample):
    python generate.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \\
        --max_tokens 32 --max_seq_len 128 --mesh_shape 1 8 --sharded_loss --no_logits

    # Batched generation (4 samples per device, 8 total with DP=2):
    python generate.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \\
        --max_tokens 32 --max_seq_len 128 --mesh_shape 2 1 --batch_size 4
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import ttnn
import ttml

from utils.kv_cache import (
    KVCache,
    _to_device_tiled,
    _causal_mask,
    _decode_mask,
    _step_attn_mask,
)
from utils.memory import MemoryUsageTracker, finalize_memory
from utils.tensor_utils import (
    create_input_tensor_from_torch,
    create_input_tensor_dp,
    gather_mesh_to_cpu,
)

create_causal_mask_tensor = _causal_mask  # public alias used by gradients.py


def _sample_logits_mask(orig_vocab, padded_vocab, device):
    """Mask for padded vocabulary entries (subtractive: 0=valid, 1e4=padding)."""
    if orig_vocab >= padded_vocab:
        return None
    mask = torch.zeros((1, 1, 1, padded_vocab), dtype=torch.bfloat16)
    mask[:, :, :, orig_vocab:] = 1e4
    return _to_device_tiled(mask, device)


# =====================================================================
# Step preparation
# =====================================================================


def _build_padded_input(
    current_tokens, batch_size, max_seq_len, step, kv_cache, past_kv
):
    """Build padded token tensor and prediction positions for one step.

    Returns (padded [B,1,1,S], pred_positions list[int], seq_len_for_mask).
    """
    if kv_cache and step == 0:
        seq_len = max(len(t) for t in current_tokens)
        padded = torch.zeros((batch_size, 1, 1, seq_len), dtype=torch.int32)
        pred_positions = []
        for b in range(batch_size):
            toks = current_tokens[b]
            padded[b, 0, 0, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
            pred_positions.append(len(toks) - 1)
        return padded, pred_positions, seq_len

    if kv_cache and step > 0:
        padded = torch.zeros((batch_size, 1, 1, 1), dtype=torch.int32)
        for b in range(batch_size):
            padded[b, 0, 0, 0] = current_tokens[b][-1]
        return padded, [0] * batch_size, None

    padded = torch.zeros((batch_size, 1, 1, max_seq_len), dtype=torch.int32)
    pred_positions = []
    for b in range(batch_size):
        start = max(0, len(current_tokens[b]) - max_seq_len)
        window = current_tokens[b][start:]
        padded[b, 0, 0, : len(window)] = torch.tensor(window, dtype=torch.int32)
        pred_positions.append(len(window) - 1)
    return padded, pred_positions, None


# =====================================================================
# Sampling
# =====================================================================


def _sample_on_device(
    logits,
    pred_positions,
    batch_size,
    device,
    distributed,
    dp_size,
    tp_size,
    per_device_batch,
    temperature,
    logits_mask,
):
    """Sample next tokens on device. Transfers only O(batch*seq) uint32."""
    seed = int(np.random.randint(1, int(1e7)))
    sample_result = ttml.ops.sample.sample_op(logits, temperature, seed, logits_mask)

    result_cpu = gather_mesh_to_cpu(
        sample_result.get_value(),
        device,
        distributed,
        dp_size,
        tp_size,
        per_device_batch,
    )
    return [int(result_cpu[b, 0, pred_positions[b], 0]) for b in range(batch_size)]


def _collect_logits_and_sample(
    logits,
    pred_positions,
    batch_size,
    orig_vocab,
    device,
    distributed,
    dp_size,
    tp_size,
    per_device_batch,
    temperature,
    logits_lists,
):
    """Transfer full logits to CPU, record them, sample on CPU."""
    logits_cpu = gather_mesh_to_cpu(
        logits.get_value(),
        device,
        distributed,
        dp_size,
        tp_size,
        per_device_batch,
    ).to(torch.float32)

    tokens = []
    for b in range(batch_size):
        next_logits = logits_cpu[b, 0, pred_positions[b], :orig_vocab]
        logits_lists[b].append(next_logits.clone())

        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1).item()
        else:
            token = torch.argmax(next_logits).item()
        tokens.append(token)
    return tokens


# =====================================================================
# HuggingFace generation (reference, CPU)
# =====================================================================


def generate_hf(hf_model, tokenizer, all_prompt_tokens, max_tokens, temperature=0.0):
    """Generate with HuggingFace model using KV cache on CPU."""
    from transformers import DynamicCache

    hf_model.eval()
    batch_size = len(all_prompt_tokens)
    max_prompt_len = max(len(pt) for pt in all_prompt_tokens)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    input_ids = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_prompt_len, dtype=torch.long)
    for b in range(batch_size):
        plen = len(all_prompt_tokens[b])
        offset = max_prompt_len - plen
        input_ids[b, offset:] = torch.tensor(all_prompt_tokens[b], dtype=torch.long)
        attention_mask[b, offset:] = 1

    all_generated = [[] for _ in range(batch_size)]
    all_logits = [[] for _ in range(batch_size)]
    past = DynamicCache()

    with torch.no_grad():
        for step in tqdm(range(max_tokens)):
            cur_ids = input_ids if step == 0 else next_tokens_t
            outputs = hf_model(
                input_ids=cur_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )

            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :].float()

            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_logits, dim=-1)

            next_tokens_t = next_tokens.unsqueeze(1)
            for b in range(batch_size):
                all_generated[b].append(next_tokens[b].item())
                all_logits[b].append(next_logits[b].clone())

            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, 1, dtype=torch.long)], dim=1
            )

    for b in range(batch_size):
        text = tokenizer.decode(all_generated[b])
        prefix = f"[{b+1}] " if batch_size > 1 else ""
        print(f"{prefix}>>> {tokenizer.decode(all_prompt_tokens[b])}{text}")

    return all_generated, all_logits


# =====================================================================
# TTML generation
# =====================================================================


def generate_ttml(
    model,
    config,
    tokenizer,
    all_prompt_tokens,
    max_tokens,
    max_seq_len,
    device,
    *,
    temperature=0.0,
    collect_logits=False,
    distributed=False,
    sharded_loss=False,
    dp_size=1,
    tp_size=1,
    kv_cache=False,
    track_memory=False,
):
    """Generate with ttml model.

    When sharded_loss=True, logits arrive vocab-sharded across TP devices.
    We all_gather them to full vocab before sampling or collection — this
    is simpler and avoids the distributed argmax+Gumbel approximation.
    """
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(
        ttml.autograd.GradMode.DISABLED
    )
    model.eval()

    if isinstance(all_prompt_tokens[0], int):
        all_prompt_tokens = [all_prompt_tokens]
    batch_size = len(all_prompt_tokens)
    is_dp = dp_size > 1
    per_device_batch = batch_size // dp_size if is_dp else batch_size

    orig_vocab = config.vocab_size
    padded_vocab = ((orig_vocab + 31) // 32) * 32

    past_kv = KVCache(config.num_hidden_layers, max_seq_len) if kv_cache else None
    causal_mask = None if kv_cache else _causal_mask(max_seq_len, device)

    logits_mask = None
    if not collect_logits:
        logits_mask = _sample_logits_mask(orig_vocab, padded_vocab, device)

    current_tokens = [list(pt) for pt in all_prompt_tokens]
    generated = [[] for _ in range(batch_size)]
    logits_lists = [[] for _ in range(batch_size)] if collect_logits else None
    step_times: list[float] = []

    for step in tqdm(range(max_tokens)):
        _step_t0 = time.perf_counter()
        # --- build input tensors for this step ---
        padded, pred_positions, prefill_len = _build_padded_input(
            current_tokens, batch_size, max_seq_len, step, kv_cache, past_kv
        )

        attn_mask = _step_attn_mask(
            step, kv_cache, past_kv, prefill_len, causal_mask, device
        )

        if is_dp:
            input_tensor = create_input_tensor_dp(padded.numpy(), device)
            input_ids_np = padded.numpy()
        else:
            input_tensor = create_input_tensor_from_torch(padded, device)
            input_ids_np = padded.numpy()

        # --- forward ---
        logits = model(
            input_tensor, attn_mask, past_key_values=past_kv, input_ids_np=input_ids_np
        )

        if track_memory and step == 0:
            MemoryUsageTracker.snapshot("GENERATION_STEP_0")

        # --- if vocab-sharded, gather to full vocab on device ---
        if sharded_loss:
            logits = ttml.ops.distributed.all_gather(logits, dim=3, cluster_axis=1)

        # --- sample ---
        if collect_logits:
            tokens = _collect_logits_and_sample(
                logits,
                pred_positions,
                batch_size,
                orig_vocab,
                device,
                distributed,
                dp_size,
                tp_size,
                per_device_batch,
                temperature,
                logits_lists,
            )
        else:
            tokens = _sample_on_device(
                logits,
                pred_positions,
                batch_size,
                device,
                distributed,
                dp_size,
                tp_size,
                per_device_batch,
                temperature,
                logits_mask,
            )

        for b in range(batch_size):
            generated[b].append(tokens[b])
            current_tokens[b].append(tokens[b])

        if not kv_cache:
            ttml.autograd.AutoContext.get_instance().reset_graph()
        step_times.append(time.perf_counter() - _step_t0)

    for b in range(batch_size):
        text = tokenizer.decode(generated[b])
        print(f"[{b+1}] >>> {tokenizer.decode(all_prompt_tokens[b])}{text}")

    # Per-token timing summary
    print(f"\n{'Step':>5} {'Time (ms)':>10} {'Tok/s':>8}")
    print("-" * 26)
    for i, t in enumerate(step_times):
        label = "prefill" if (kv_cache and i == 0) else f"{i:5d}"
        print(f"{label:>5} {t * 1000:>10.1f} {1.0 / t:>8.1f}")
    if step_times:
        decode_times = step_times[1:] if kv_cache else step_times
        if decode_times:
            avg = sum(decode_times) / len(decode_times)
            print(f"\n  Decode avg: {avg * 1000:.1f} ms/tok  ({1.0 / avg:.1f} tok/s)")

    ttml.autograd.AutoContext.get_instance().reset_graph()
    return generated, logits_lists


# =====================================================================
# Comparison
# =====================================================================


def compare_logits(all_hf_logits, all_ttml_logits, vocab_size, tokenizer):
    """Compare per-step logits between HF and ttml, aggregated across batch."""
    if not isinstance(all_hf_logits[0], list):
        all_hf_logits = [all_hf_logits]
        all_ttml_logits = [all_ttml_logits]

    batch_size = len(all_hf_logits)
    num_steps = len(all_hf_logits[0])

    batch_hdr = f"  (averaged over {batch_size} samples)" if batch_size > 1 else ""
    print(f"\n{'Step':>5} {'AvgPCC':>10} {'MaxDiff':>10} " f"{'Match':>8}{batch_hdr}")
    print("-" * (40 + len(batch_hdr)))
    total_matches = 0
    total_pcc = 0.0

    for i in range(num_steps):
        step_pcc = 0.0
        step_maxdiff = 0.0
        step_matches = 0
        for b in range(batch_size):
            hf_v = all_hf_logits[b][i][:vocab_size].float()
            ttml_v = all_ttml_logits[b][i][:vocab_size].float()
            hf_c = hf_v - hf_v.mean()
            ttml_c = ttml_v - ttml_v.mean()
            pcc = ((hf_c * ttml_c).sum() / (hf_c.norm() * ttml_c.norm() + 1e-8)).item()
            step_pcc += pcc
            step_maxdiff = max(step_maxdiff, (hf_v - ttml_v).abs().max().item())
            if torch.argmax(hf_v).item() == torch.argmax(ttml_v).item():
                step_matches += 1

        avg_pcc = step_pcc / batch_size
        total_pcc += step_pcc
        total_matches += step_matches
        match_str = (
            f"{step_matches}/{batch_size}"
            if batch_size > 1
            else ("Y" if step_matches else "N")
        )
        print(f"{i:>5} {avg_pcc:>10.6f} {step_maxdiff:>10.4f} " f"{match_str:>8}")

    total = num_steps * batch_size
    print("-" * (40 + len(batch_hdr)))
    print(f"Average PCC:      {total_pcc / total:.6f}")
    print(
        f"Top-1 match rate: {total_matches}/{total} "
        f"({100 * total_matches / total:.1f}%)"
    )


# =====================================================================
# Device setup
# =====================================================================


def _create_model(hf_config, hf_state_dict, args, dp_size, tp_size):
    """Create ttml model and load HF weights. Returns (model, config, sharded_loss, mode_str)."""
    from utils.model_factory import create_ttml_model, load_hf_weights

    use_tp = tp_size > 1
    model, config, tie, shard_dim, mode_str = create_ttml_model(
        hf_config,
        args.max_seq_len,
        dp_size=dp_size,
        tp_size=tp_size,
        track_memory=args.track_memory,
        sharded_loss=args.sharded_loss,
    )
    sharded_loss = args.sharded_loss and use_tp
    load_hf_weights(
        model,
        hf_state_dict,
        config,
        tie=tie,
        tp=use_tp,
        shard_dim=shard_dim,
    )
    return model, config, sharded_loss, mode_str


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3: generation comparison (HF vs ttml)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--mesh_shape",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("ROWS", "COLS"),
        help="Device mesh [rows, cols]. rows=DP, cols=TP. Default: 1 1.",
    )
    parser.add_argument(
        "--sharded_loss",
        action="store_true",
        default=False,
        help="Keep LM head output vocab-sharded; all_gather before sampling.",
    )
    parser.add_argument(
        "--track_memory",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Enable memory tracking. Optional int N = snapshot every N-th layer.",
    )
    parser.add_argument(
        "--kv_cache",
        action="store_true",
        default=False,
        help="Use KV cache: prefill prompt then decode one token at a time.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size. Total prompts = batch_size * DP. " "Default: 1.",
    )
    parser.add_argument(
        "--no_logits",
        action="store_true",
        default=False,
        help="On-device sampling without collecting logits (faster, "
        "disables HF comparison).",
    )
    args = parser.parse_args()

    dp_size, tp_size = args.mesh_shape
    distributed = tp_size > 1 or dp_size > 1

    # 1. Load HuggingFace model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HuggingFace model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    hf_model.eval()

    # 2. Build prompts (total = batch_size * dp_size)
    total_prompts = args.batch_size * dp_size
    if total_prompts > 1:
        width = len(str(total_prompts))
        prompts = [
            f"{str(i+1).zfill(width)}. {args.prompt}" for i in range(total_prompts)
        ]
    else:
        prompts = [args.prompt]

    all_prompt_tokens = [tokenizer.encode(p) for p in prompts]
    print(
        f"Batch size: {args.batch_size}/device, "
        f"{total_prompts} total (DP={dp_size})"
    )
    for i, p in enumerate(prompts):
        print(f"Prompt[{i}]: {p!r}  ->  {len(all_prompt_tokens[i])} tokens")

    # 3. Set up device
    from utils.device_setup import setup_device

    ctx, device = setup_device(dp_size, tp_size)

    memory_guard = None
    if args.track_memory:
        print("Memory tracking enabled")
        memory_guard = MemoryUsageTracker.begin_capture()
        MemoryUsageTracker.snapshot("BEFORE_MODEL_CREATION")

    # 4. Create ttml model and load weights
    model, config, sharded_loss, mode_str = _create_model(
        hf_model.config, hf_model.state_dict(), args, dp_size, tp_size
    )

    if args.track_memory:
        MemoryUsageTracker.snapshot("AFTER_WEIGHT_LOADING")

    # 5. Generate with HuggingFace
    num_samples = len(prompts)
    total_tok = args.max_tokens * num_samples

    print("\n" + "=" * 70)
    print("HuggingFace generation (CPU, bfloat16):")
    print("=" * 70)
    t0 = time.time()
    all_hf_tokens, all_hf_logits = generate_hf(
        hf_model, tokenizer, all_prompt_tokens, args.max_tokens, args.temperature
    )
    hf_time = time.time() - t0
    print(f"  Time: {hf_time:.2f}s ({total_tok / hf_time:.1f} tok/s)")

    # 6. Generate with ttml
    gen_mode = "KV-cache" if args.kv_cache else "full-recompute"
    print("\n" + "=" * 70)
    print(f"TTML generation ({mode_str}, bfloat16, {gen_mode}):")
    print("=" * 70)

    collect_logits = not args.no_logits
    t0 = time.time()
    all_ttml_tokens, all_ttml_logits = generate_ttml(
        model,
        config,
        tokenizer,
        all_prompt_tokens,
        args.max_tokens,
        args.max_seq_len,
        device,
        temperature=args.temperature,
        collect_logits=collect_logits,
        distributed=distributed,
        sharded_loss=sharded_loss,
        dp_size=dp_size,
        tp_size=tp_size,
        kv_cache=args.kv_cache,
        track_memory=args.track_memory,
    )
    ttml_time = time.time() - t0
    print(f"  Time: {ttml_time:.2f}s ({total_tok / ttml_time:.1f} tok/s)")

    # 7. Compare
    if all_ttml_logits is not None and all_ttml_logits[0] is not None:
        print(f"\n{'=' * 70}")
        print("Logits comparison (HF vs TTML):")
        print("=" * 70)
        compare_logits(all_hf_logits, all_ttml_logits, config.vocab_size, tokenizer)

    total_matches = 0
    total_tokens = 0
    for i in range(num_samples):
        hf_tok = all_hf_tokens[i]
        ttml_tok = all_ttml_tokens[i]
        m = sum(1 for a, b in zip(hf_tok, ttml_tok) if a == b)
        total_matches += m
        total_tokens += len(hf_tok)
        prefix = f"[{i+1}] " if num_samples > 1 else ""
        print(
            f"{prefix}HF  : "
            f"{tokenizer.decode(all_prompt_tokens[i])}"
            f"{tokenizer.decode(hf_tok)}"
        )
        print(
            f"{prefix}TTML: "
            f"{tokenizer.decode(all_prompt_tokens[i])}"
            f"{tokenizer.decode(ttml_tok)}"
        )
    print(
        f"\nToken match: {total_matches}/{total_tokens} "
        f"({100 * total_matches / total_tokens:.1f}%)"
    )

    # 8. Cleanup
    if args.track_memory:
        finalize_memory(memory_guard)

    ctx.close_device()


if __name__ == "__main__":
    main()
