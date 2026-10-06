# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small, deterministic CPU semantics for Qwen3.8-Flash-Next bring-up.

These routines intentionally contain no TTNN imports.  They are the executable
contract used while porting individual kernels and state transitions.  The
formulas are pinned against the Qwen4Exp Transformers implementation and the
day-zero SGLang Qwen4Exp/MTP implementation recorded in ``RUN_MANIFEST.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


def zero_centered_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int | None = None,
) -> torch.Tensor:
    """Qwen4Exp RMSNorm: FP32 variance and a learned ``1 + weight`` scale."""

    input_dtype = x.dtype
    x_float = x.float()
    if group_size is None:
        normalized = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    else:
        if x.shape[-1] % group_size:
            raise ValueError(f"last dimension {x.shape[-1]} is not divisible by group_size={group_size}")
        grouped = x_float.unflatten(-1, (x.shape[-1] // group_size, group_size))
        normalized = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
        normalized = normalized.flatten(-2)
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"norm weight shape {tuple(weight.shape)} does not match width {x.shape[-1]}")
    return (normalized * (1.0 + weight.float())).to(input_dtype)


def gated_residual_read(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one H-wide block input from the four-branch gated residual."""

    expected_width = hc_count * hidden_size
    if residual.shape[-1] != expected_width:
        raise ValueError(f"expected residual width {expected_width}, got {residual.shape[-1]}")
    normalized = zero_centered_rms_norm(residual, norm_weight, eps, group_size=hidden_size)
    low_rank = F.silu(F.linear(normalized, down_weight) / hc_count)
    gate = torch.sigmoid(F.linear(low_rank, up_weight)).unflatten(-1, (hc_count, hidden_size))
    block_input = (gate * normalized.unflatten(-1, (hc_count, hidden_size))).mean(dim=-2)
    return block_input, normalized


def gated_residual_write(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normalized_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    """Write one H-wide block result back into every residual branch."""

    expected_width = hc_count * hidden_size
    if residual.shape[-1] != expected_width or normalized_residual.shape != residual.shape:
        raise ValueError("residual and normalized residual do not have the expected hyper-connection shape")
    if block_output.shape[:-1] != residual.shape[:-1] or block_output.shape[-1] != hidden_size:
        raise ValueError("block output is not aligned with the residual batch and hidden size")
    coefficient = 2 * torch.sigmoid(F.linear(normalized_residual, inject_weight) / hc_count)
    branches = residual.unflatten(-1, (hc_count, hidden_size))
    return (branches + coefficient.unsqueeze(-1) * block_output.unsqueeze(-2)).flatten(-2)


def mtp_input_fusion(
    input_embedding: torch.Tensor,
    hidden_state: torch.Tensor,
    embedding_norm_weight: torch.Tensor,
    hidden_norm_weight: torch.Tensor,
    fc_embedding_weight: torch.Tensor,
    fc_hidden_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    eps: float,
) -> torch.Tensor:
    """Exact Qwen4Exp MTP input mixing for the four-branch GR state.

    ``fc_hidden`` is shared across the four branch views.  The projected token
    embedding is broadcast and added to each branch; there is no 2H->H concat
    projection in the released Qwen3.8-Flash-Next checkpoint.
    """

    if input_embedding.shape[:-1] != hidden_state.shape[:-1]:
        raise ValueError("embedding and hidden state prefixes differ")
    if input_embedding.shape[-1] != hidden_size or hidden_state.shape[-1] != hc_count * hidden_size:
        raise ValueError("MTP input widths do not match hidden_size/hc_count")
    embedding = zero_centered_rms_norm(input_embedding, embedding_norm_weight, eps)
    embedding = F.linear(embedding, fc_embedding_weight)
    # The released Qwen4Exp MTP module constructs pre_fc_norm_hidden at
    # hc_count * hidden_size and applies it before viewing the result as
    # [hc_count, hidden_size].  This is one global 4H normalization; only the
    # following fc_hidden projection is shared independently by each branch.
    hidden = zero_centered_rms_norm(hidden_state, hidden_norm_weight, eps)
    hidden = F.linear(hidden.unflatten(-1, (hc_count, hidden_size)), fc_hidden_weight)
    return (hidden + embedding.unsqueeze(-2)).flatten(-2)


def router_topk(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 softmax, top-k, and exact selected-score renormalization."""

    if not 0 < top_k <= logits.shape[-1]:
        raise ValueError(f"top_k={top_k} is outside [1, {logits.shape[-1]}]")
    probabilities = torch.softmax(logits.float(), dim=-1)
    scores, indices = torch.topk(probabilities, top_k, dim=-1)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    return scores.to(logits.dtype), indices


def _l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gated_delta_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    l2_normalize_qk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serial FP32 Gated DeltaNet recurrence used as prefill/decode oracle."""

    if query.shape != key.shape or query.shape[:-1] != value.shape[:-1]:
        raise ValueError("query, key, and value shapes are incompatible")
    if log_decay.shape != query.shape[:-1] or beta.shape != query.shape[:-1]:
        raise ValueError("decay/beta shapes must be [batch, sequence, heads]")
    input_dtype = query.dtype
    if l2_normalize_qk:
        query = _l2_norm(query)
        key = _l2_norm(key)
    query, key, value, beta, log_decay = [
        tensor.transpose(1, 2).contiguous().float() for tensor in (query, key, value, beta, log_decay)
    ]
    batch, heads, sequence, key_dim = key.shape
    value_dim = value.shape[-1]
    query = query * (1.0 / math.sqrt(key_dim))
    state = (
        torch.zeros(batch, heads, key_dim, value_dim, dtype=torch.float32, device=value.device)
        if initial_state is None
        else initial_state.to(device=value.device, dtype=torch.float32)
    )
    output = torch.empty(batch, heads, sequence, value_dim, dtype=torch.float32, device=value.device)
    for index in range(sequence):
        q_t = query[:, :, index]
        k_t = key[:, :, index]
        v_t = value[:, :, index]
        state = state * log_decay[:, :, index].exp().unsqueeze(-1).unsqueeze(-1)
        remembered = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - remembered) * beta[:, :, index].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    return output.transpose(1, 2).contiguous().to(input_dtype), state


def causal_depthwise_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    dilation: int = 1,
    activation: Literal["silu"] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise convolution with an explicit rolling state."""

    if hidden_states.ndim != 3 or weight.ndim != 2:
        raise ValueError("expected hidden_states [B,C,S] and weight [C,K]")
    batch, channels, sequence = hidden_states.shape
    if weight.shape[0] != channels or dilation <= 0:
        raise ValueError("convolution channels/dilation are invalid")
    minimum_state_len = (weight.shape[-1] - 1) * dilation
    if initial_state is None:
        state_len = minimum_state_len
        state = hidden_states.new_zeros((batch, channels, state_len))
    else:
        if initial_state.shape[:2] != (batch, channels) or initial_state.shape[-1] < minimum_state_len:
            raise ValueError("initial convolution state is too short or has the wrong shape")
        state_len = initial_state.shape[-1]
        state = initial_state
    combined = torch.cat([state, hidden_states], dim=-1).to(weight.dtype)
    output = F.conv1d(combined, weight.unsqueeze(1), groups=channels, dilation=dilation)
    output = output[..., -sequence:]
    if activation == "silu":
        output = F.silu(output)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation!r}")
    next_state = combined[..., -state_len:].to(state.dtype) if state_len else combined[..., :0].to(state.dtype)
    return output.to(hidden_states.dtype), next_state


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _next_prime(value: int) -> int:
    candidate = value + 1
    while not _is_prime(candidate):
        candidate += 1
    return candidate


@dataclass(frozen=True)
class NGramHashSpec:
    ngram_size: int
    heads_per_ngram: int
    layer_multipliers: torch.Tensor
    head_vocab_sizes: torch.Tensor
    head_offsets: torch.Tensor
    padded_vocab_size: int


def build_ngram_hash_spec(
    unigram_vocab_size: int,
    ngram_size: int,
    heads_per_ngram: int,
    ngram_vocab_size_base: int,
    ple_layer_index: int,
    seed: int,
    divisible_by: int,
) -> NGramHashSpec:
    """Build the checkpoint-compatible SplitMix64 multipliers and prime heads."""

    if unigram_vocab_size <= 0 or ngram_size < 2 or heads_per_ngram <= 0 or divisible_by <= 0:
        raise ValueError("invalid n-gram hash dimensions")
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // unigram_vocab_size) // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)

    num_heads = (ngram_size - 1) * heads_per_ngram
    prime = ngram_vocab_size_base - 1
    vocab_sizes = []
    first_global_head = ple_layer_index * num_heads
    for global_head in range(first_global_head + num_heads):
        prime = _next_prime(prime)
        if global_head >= first_global_head:
            vocab_sizes.append(prime)
    offsets = []
    total = 0
    for size in vocab_sizes:
        offsets.append(total)
        total += size
    padded = math.ceil(total / divisible_by) * divisible_by
    return NGramHashSpec(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        layer_multipliers=torch.tensor(multipliers, dtype=torch.long),
        head_vocab_sizes=torch.tensor(vocab_sizes, dtype=torch.long),
        head_offsets=torch.tensor(offsets, dtype=torch.long),
        padded_vocab_size=padded,
    )


def _shift_right_ignore_eos(token_ids: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    if shift == 0:
        return token_ids
    batch, sequence = token_ids.shape
    positions = torch.arange(sequence, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat([eos_positions.new_full((batch, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch, -1)
    shifted = token_ids.gather(1, gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


def ngram_token_ids(
    input_ids: torch.Tensor,
    previous_context: torch.Tensor | None,
    eos_token_id: int,
    spec: NGramHashSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all bigram/trigram head IDs and the exact raw-token history state."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, sequence]")
    input_ids = input_ids.long()
    context_len = spec.ngram_size - 1
    if previous_context is None:
        previous_context = input_ids.new_full((input_ids.shape[0], context_len), eos_token_id)
    elif previous_context.shape != (input_ids.shape[0], context_len):
        raise ValueError("previous n-gram context has the wrong shape")
    history = torch.cat([previous_context.long(), input_ids], dim=-1)
    multipliers = spec.layer_multipliers.to(history.device)
    vocab_sizes = spec.head_vocab_sizes.to(history.device)
    offsets = spec.head_offsets.to(history.device)
    shifted = [_shift_right_ignore_eos(history, index, eos_token_id) for index in range(spec.ngram_size)]
    blocks = []
    for ngram in range(2, spec.ngram_size + 1):
        start = (ngram - 2) * spec.heads_per_ngram
        end = start + spec.heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        ids = torch.remainder(mixed.unsqueeze(-1), vocab_sizes[start:end].view(1, 1, -1))
        blocks.append(ids + offsets[start:end].view(1, 1, -1))
    ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
    next_context = history[:, -context_len:].clone() if context_len else history[:, :0].clone()
    return ids, next_context


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotary = rotary * cos + _rotate_half(rotary) * sin
    return torch.cat([rotary, passthrough], dim=-1)


def qsa_selected_token_mask(
    query: torch.Tensor,
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    token_budget: int,
    compress_ratio: int,
    eps: float,
) -> torch.Tensor:
    """Dense CPU oracle for QSA block selection and the causal incomplete tail."""

    batch, query_length, index_heads, head_dim = query.shape
    if raw_keys.shape[:2] != (batch, attention_mask.shape[-1]) or raw_keys.shape[-1] != head_dim:
        raise ValueError("raw QSA key shape does not match mask/query")
    if token_budget % compress_ratio:
        raise ValueError("QSA token budget must divide by the compression ratio")
    current_cos = full_cos[:, -query_length:, :].unsqueeze(2)
    current_sin = full_sin[:, -query_length:, :].unsqueeze(2)
    query = zero_centered_rms_norm(query, q_norm_weight, eps)
    query = _apply_partial_rope(query, current_cos, current_sin)
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    selected = torch.full(
        (batch, query_length, token_budget + compress_ratio - 1),
        -1,
        dtype=torch.int32,
        device=query.device,
    )
    block_topk = token_budget // compress_ratio
    for batch_index in range(batch):
        for query_index in range(query_length):
            local_visible = torch.nonzero(visible[batch_index, 0, query_index], as_tuple=False).flatten()
            complete = local_visible.numel() // compress_ratio
            if complete:
                block_tokens = local_visible[: complete * compress_ratio].view(complete, compress_ratio)
                key_groups = raw_keys[batch_index].index_select(0, block_tokens.flatten())
                pooled = key_groups.view(complete, compress_ratio, head_dim).float().mean(1).to(raw_keys.dtype)
                pooled = zero_centered_rms_norm(pooled, k_norm_weight, eps)
                starts = block_tokens[:, 0]
                block_keys = _apply_partial_rope(
                    pooled,
                    full_cos[batch_index].index_select(0, starts),
                    full_sin[batch_index].index_select(0, starts),
                )
                scores = torch.matmul(
                    query[batch_index, query_index].float(), block_keys.float().transpose(-1, -2)
                ).transpose(-1, -2)
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(head_dim)
                chosen = scores.topk(min(block_topk, complete), dim=0).indices
                chosen_tokens = block_tokens.index_select(0, chosen).flatten()
            else:
                chosen_tokens = local_visible.new_empty((0,))
            tail = local_visible[complete * compress_ratio :]
            chosen_tokens = torch.cat([chosen_tokens, tail]).to(torch.int32)
            selected[batch_index, query_index, : chosen_tokens.numel()] = chosen_tokens

    key_length = attention_mask.shape[-1]
    output = torch.zeros((*selected.shape[:-1], key_length + 1), device=query.device, dtype=torch.bool)
    scatter = torch.where(selected >= 0, selected, key_length).long()
    output = output.scatter(-1, scatter, True)[..., :key_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        output = torch.where(output, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
    return output


def speculative_commit_state(per_step_state: torch.Tensor, accepted_drafts: torch.Tensor) -> torch.Tensor:
    """Select verifier state after current-token + ``accepted_drafts`` positions."""

    if per_step_state.ndim < 2 or accepted_drafts.shape != (per_step_state.shape[0],):
        raise ValueError("per-step state and accepted-draft batch shapes do not align")
    accepted_drafts = accepted_drafts.to(device=per_step_state.device, dtype=torch.long)
    if torch.any(accepted_drafts < 0) or torch.any(accepted_drafts >= per_step_state.shape[1]):
        raise ValueError("accepted draft count is outside the verifier window")
    rows = torch.arange(per_step_state.shape[0], device=per_step_state.device)
    return per_step_state[rows, accepted_drafts]


def qsa_shared_draft_indices(
    captured_indices: torch.Tensor,
    captured_length: torch.Tensor,
    current_position: torch.Tensor,
    tail_width: int,
) -> torch.Tensor:
    """Reuse frozen target-aligned QSA indices and append the in-flight causal tail."""

    if captured_indices.ndim != 2 or captured_length.shape != current_position.shape:
        raise ValueError("captured QSA state shapes are inconsistent")
    if captured_indices.shape[0] != captured_length.shape[0] or tail_width < 0:
        raise ValueError("captured QSA batch/tail width is invalid")
    offsets = torch.arange(tail_width, device=captured_indices.device, dtype=torch.long)
    tail = captured_length.long().unsqueeze(1) + offsets.unsqueeze(0)
    valid = tail <= current_position.long().unsqueeze(1)
    tail = torch.where(valid, tail, -1).to(captured_indices.dtype)
    return torch.cat([captured_indices, tail], dim=-1)
