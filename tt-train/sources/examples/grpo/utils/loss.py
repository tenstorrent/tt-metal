from typing import List, Tuple
import ttnn
import ttml
import numpy as np
from .setup import InferenceCtx
from .gsm8k import extract_hash_answer
from .inference import tokens_to_tensor, create_causal_mask, round_up
from .ttml_operators import Exp, Clip, Min


def compute_rewards_advantages(ctx: InferenceCtx, answers: List[float], completions: List[List[int]]):
    assert len(answers) == len(completions)

    completions_strs = [ctx.tokenizer.decode(c, skip_special_tokens=True) for c in completions]

    guesses: List[float | None] = [extract_hash_answer(s) for s in completions_strs]

    rewards_np = np.zeros(len(completions), dtype=np.float32)
    for i, (g, a) in enumerate(zip(guesses, answers)):
        if g is None or a is None:
            rewards_np[i] = 0.0
            continue
        rewards_np[i] = 1.0 if abs(g - a) < 1e-3 else 0.0  # or -1.0 for wrong if you prefer

    advantages_np = np.zeros_like(rewards_np)
    G = ctx.group_size
    for start in range(0, len(rewards_np), G):
        end = min(start + G, len(rewards_np))
        rg = rewards_np[start:end]
        mu = float(rg.mean())
        advantages_np[start : start + G] = rg - mu

    return rewards_np, advantages_np


def compute_grpo_loss(
    nlog_probs_old: ttml.autograd.Tensor,
    nlog_probs_new: ttml.autograd.Tensor,
    mask: ttml.autograd.Tensor,
    adv_tt: ttml.autograd.Tensor,
    B: int,
    Tp: int,
    completions_batch_len: int,
    eps: float,
) -> Tuple[ttml.autograd.Tensor, list]:
    ratio = Exp.apply(nlog_probs_old - nlog_probs_new)
    clipped_ratio = Clip.apply(ratio, 1.0 - eps, 1.0 + eps)

    surr1 = ratio * adv_tt
    surr2 = clipped_ratio * adv_tt
    surr = Min.apply(surr1, surr2)

    mask_np = mask.to_numpy()
    tokens_per_completion = np.maximum(mask_np.sum(axis=1, keepdims=True), 1.0)
    weight_np = (mask_np / tokens_per_completion).astype(np.float32)
    weight_tt = ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16)

    weighted_surr = surr * weight_tt
    weighted_surr_4d = ttml.ops.reshape.reshape(weighted_surr, [1, 1, B, Tp])
    loss = ttml.ops.unary.mean(weighted_surr_4d) * (-float(Tp) / completions_batch_len)

    return loss


def compute_nlog_probs(
    ctx: InferenceCtx,
    prompts: List[List[int]],
    completions: List[List[int]],
) -> tuple[ttml.autograd.Tensor, ttml.autograd.Tensor, int]:
    """
    Compute per-token negative log probabilities for prompt+completion sequences.
    Concatenates each (prompt, completion) pair, applies the standard next-token-prediction
    shift (input = sequence[:-1], target = sequence[1:]), runs the model, and returns
    the cross-entropy at every position.
    Sequences shorter than the longest are left-padded with ctx.pad_token.
    Args:
        prompts:     List of B token-id lists (one per sequence).
        completions: List of B token-id lists (one per sequence).
    Returns:
        nlog_probs: Tensor [B, Tp] — negative log-probability of each target token.
                    Positions beyond T (the true sequence length) are tile padding and meaningless.
        mask:       Tensor [B, Tp] — binary mask, 1.0 on completion-token positions only
                    (0.0 on prompt tokens, left-pad, and tile-pad). Accounts for the
                    one-token input/target shift so mask[i,t]=1 where the model is
                    predicting a completion token.
        Tp:         int — T rounded up to the tile boundary (multiple of ctx.tile_size).
                    T = max(len(p)+len(c)) - 1 across the batch.
    """

    assert len(completions) == len(prompts)

    B = len(completions)
    ctx._B = B  # create_causal_mask() reads this

    lengths = [len(p) + len(c) for p, c in zip(prompts, completions)]
    T = max(lengths) - 1
    assert T >= 1

    inputs_np = np.full((B, T), ctx.pad_token, dtype=np.uint32)
    targets_np = np.full((B, T), ctx.pad_token, dtype=np.uint32)
    loss_mask_np = np.zeros((B, T), dtype=np.float32)
    pad_lengths = []

    for i, (p, c) in enumerate(zip(prompts, completions)):
        sequence = p + c
        L = len(sequence) - 1
        shift = T - L  # left-padding amount in full sequence
        pad_lengths.append(shift)  # left pads in inputs/targets

        if len(p) < 2:
            raise ValueError("Prompt is too short")

        if len(sequence) < 2:
            raise ValueError("Sequence is too short")

        inputs_np[i, -L:] = np.asarray(sequence[:-1], dtype=np.uint32)
        targets_np[i, -L:] = np.asarray(sequence[1:], dtype=np.uint32)

        if c:
            start = -1 + shift + len(p)  # because of the one-token shift we subtract 1
            end = min(start + len(c), T)
            if start < end:
                loss_mask_np[i, start:end] = 1.0

    x_tt = tokens_to_tensor(ctx, inputs_np, B)  # [B,1,1,padded(T)]
    mask_tt = create_causal_mask(ctx, prompt_len=0, query_len=T, pad_lengths=pad_lengths)
    logits = ctx.tt_model(x_tt, mask_tt)  # [B,1,T,V]

    Tp = round_up(ctx, T)
    targets_pad = np.full((B, Tp), ctx.pad_token, dtype=np.uint32)
    targets_pad[:, :T] = targets_np  # or align however your sequence axis is arranged

    targets_tt = ttml.autograd.Tensor.from_numpy(targets_pad, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)

    nlog = ttml.ops.loss.cross_entropy_loss(logits, targets_tt, ttml.ops.ReduceType.NONE)
    nlog = ttml.ops.reshape.reshape(nlog, [B, Tp])

    loss_mask_pad = np.zeros((B, Tp), dtype=np.float32)
    loss_mask_pad[:, :T] = loss_mask_np

    loss_mask_tt = ttml.autograd.Tensor.from_numpy(loss_mask_pad, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16)

    assert nlog.shape() == [B, Tp]
    assert loss_mask_tt.shape() == [B, Tp]
    return nlog, loss_mask_tt, Tp
