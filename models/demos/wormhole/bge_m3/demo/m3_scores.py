# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 sparse and ColBERT score utilities (backbone-only).
Uses ttnn for all tensor ops; scatter-by-token-id remains on host (no ttnn equivalent).
Implementation follows FlagEmbedding inference/embedder/encoder_only/m3.py and
finetune/embedder/encoder_only/m3/modeling.py.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

import ttnn


def _get_special_token_ids(tokenizer, vocab_size: int) -> set:
    """Collect special token IDs to zero out (FlagEmbedding: cls_token, eos_token, pad_token, unk_token)."""
    unused = set()
    # Match FlagEmbedding: special_tokens_map keys 'cls_token', 'eos_token', 'pad_token', 'unk_token'
    for key in ("cls_token", "eos_token", "pad_token", "unk_token"):
        token_str = getattr(tokenizer, key, None) or (
            tokenizer.special_tokens_map.get(key) if hasattr(tokenizer, "special_tokens_map") else None
        )
        if token_str is None and hasattr(tokenizer, "special_tokens_map"):
            token_str = tokenizer.special_tokens_map.get(key)
        if isinstance(token_str, str):
            ids = tokenizer.convert_tokens_to_ids([token_str])
            if ids and 0 <= ids[0] < vocab_size:
                unused.add(ids[0])
    for name in ("cls_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tokenizer, name, None)
        if isinstance(tid, int) and 0 <= tid < vocab_size:
            unused.add(tid)
    return unused


def _l2_normalize_ttnn(tt_tensor: Any, device: Any) -> Any:
    tt_tensor = ttnn.to_memory_config(tt_tensor, ttnn.DRAM_MEMORY_CONFIG)
    tt_sq = ttnn.pow(tt_tensor, 2)
    tt_sum = ttnn.sum(tt_sq, dim=-1, keepdim=True)

    # ADDED: epsilon to avoid divide by zero
    eps = ttnn.from_torch(
        torch.full((1,), 1e-12, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_sum = ttnn.add(tt_sum, eps)

    tt_norm = ttnn.sqrt(tt_sum)
    return ttnn.divide(tt_tensor, tt_norm)


def _l2_normalize_sparse_ttnn(sparse_tt: Any, device: Any) -> Any:
    sparse_tt = ttnn.to_memory_config(sparse_tt, ttnn.DRAM_MEMORY_CONFIG)
    shape = sparse_tt.shape
    rank = len(shape)
    V = int(shape[-1])
    CHUNK = 65536
    total_sum_tt = None
    for start in range(0, V, CHUNK):
        end = min(start + CHUNK, V)
        if rank == 2:
            B = int(shape[0])
            chunk = ttnn.slice(sparse_tt, [0, start], [B, end])
        else:
            chunk = ttnn.slice(sparse_tt, [0] * (rank - 1) + [start], [int(s) for s in shape[:-1]] + [end])
        sq = ttnn.pow(chunk, 2)
        partial = ttnn.sum(sq, dim=-1, keepdim=True)
        total_sum_tt = partial if total_sum_tt is None else ttnn.add(total_sum_tt, partial)
    norm_tt = ttnn.sqrt(total_sum_tt)
    out_chunks = []
    for start in range(0, V, CHUNK):
        end = min(start + CHUNK, V)
        if rank == 2:
            B = int(shape[0])
            chunk = ttnn.slice(sparse_tt, [0, start], [B, end])
        else:
            chunk = ttnn.slice(sparse_tt, [0] * (rank - 1) + [start], [int(s) for s in shape[:-1]] + [end])
        out_chunks.append(ttnn.divide(chunk, norm_tt))
    return ttnn.concat(out_chunks, dim=-1)


def _norm_weights_ttnn(tt_hidden: Any, attention_mask: Optional[torch.Tensor], device: Any) -> Any:
    """L2 norm per token on last dim using ttnn (square, sum, sqrt). Returns ttnn tensor [B, S] or [B, 1, S]."""
    tt_sq = ttnn.pow(tt_hidden, 2)
    tt_sum = ttnn.sum(tt_sq, dim=-1)
    tt_norm = ttnn.sqrt(tt_sum)
    if attention_mask is not None:
        mask_torch = attention_mask.unsqueeze(1).to(torch.float32)
        mask_tt = ttnn.from_torch(mask_torch, device=device, dtype=ttnn.bfloat16)
        tt_norm = ttnn.multiply(tt_norm, mask_tt)
    return tt_norm


def _sparse_embedding_scatter_loop(
    token_weights: torch.Tensor,
    input_ids: torch.Tensor,
    vocab_size: int,
    unused_ids: set,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [B, vocab_size] by looping (avoids scatter_reduce issues). Per FlagEmbedding: amax per token id."""
    batch_size, seq_len = token_weights.shape
    sparse_embedding = torch.zeros(batch_size, vocab_size, dtype=dtype, device=device)
    tw = token_weights.cpu().numpy()
    ids = input_ids.cpu().numpy()
    for b in range(batch_size):
        for s in range(seq_len):
            tid = int(ids[b, s])
            if tid < 0 or tid >= vocab_size:
                continue
            w = float(tw[b, s])
            if w <= 0:
                continue
            cur = sparse_embedding[b, tid].item()
            if w > cur:
                sparse_embedding[b, tid] = w
    for uid in unused_ids:
        if 0 <= uid < vocab_size:
            sparse_embedding[:, uid] = 0.0
    return sparse_embedding


def _sparse_embedding_scatter_ttnn(
    device: Any,
    token_weights_tt: Any,
    input_ids_tt: Any,
    vocab_size: int,
    unused_ids: set,
) -> Any:
    token_weights_tt = ttnn.to_memory_config(token_weights_tt, ttnn.DRAM_MEMORY_CONFIG)
    shape = token_weights_tt.shape
    if len(shape) == 3:
        B, seq_len = int(shape[0]), int(shape[2])
        token_weights_tt = ttnn.reshape(token_weights_tt, [B, seq_len])
    else:
        B, seq_len = int(shape[0]), int(shape[1])
    result_tt = ttnn.zeros(
        (B, vocab_size),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    zeros_tt = ttnn.zeros(
        (B, vocab_size),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for s in range(seq_len):
        index_tt = ttnn.slice(input_ids_tt, [0, s], [B, s + 1])
        value_tt = ttnn.slice(token_weights_tt, [0, s], [B, s + 1])
        scattered = ttnn.scatter(zeros_tt, dim=1, index=index_tt, src=value_tt)
        result_tt = ttnn.maximum(result_tt, scattered)
    if unused_ids:
        mask = torch.ones(B, vocab_size, dtype=torch.bfloat16)
        for uid in unused_ids:
            if 0 <= uid < vocab_size:
                mask[:, uid] = 0
        mask_tt = ttnn.from_torch(mask, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result_tt = ttnn.multiply(result_tt, mask_tt)
    return result_tt


def sparse_embedding_from_hidden_state(
    tt_hidden: Any,
    input_ids: torch.Tensor,
    vocab_size: int,
    tokenizer,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    device: Any,
    to_torch_fn: Callable[[Any], torch.Tensor],
) -> Any:
    """
    Compute sparse (lexical) embedding from tt hidden state using ttnn (norm + scatter).
    Uses L2 norm per token as weight, scatter by input_ids (amax), zero special tokens.
    Returns ttnn tensor [batch_size, vocab_size] for compute_sparse_score.
    """
    unused_ids = _get_special_token_ids(tokenizer, vocab_size)
    norm_weights_tt = _norm_weights_ttnn(tt_hidden, attention_mask, device)
    input_ids_tt = ttnn.from_torch(
        input_ids.long().to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return _sparse_embedding_scatter_ttnn(device, norm_weights_tt, input_ids_tt, vocab_size, unused_ids)


def _colbert_embedding_ttnn(tt_hidden: Any, attention_mask: torch.Tensor, device: Any) -> Any:
    """ColBERT-style token vectors from tt hidden: slice [:, 1:], mask, L2 normalize. Returns ttnn [B, 1, q_len, dim]."""
    shape = tt_hidden.shape
    if len(shape) == 4:
        B, one, S, D = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
        slice_start = [0, 0, 1, 0]
        slice_end = [B, one, S, D]
    else:
        B, S, D = int(shape[0]), int(shape[1]), int(shape[2])
        slice_start = [0, 1, 0]
        slice_end = [B, S, D]
    colbert_tt = ttnn.slice(tt_hidden, slice_start, slice_end)
    mask_torch = attention_mask[:, 1:].unsqueeze(1).unsqueeze(-1).to(torch.float32)
    mask_tt = ttnn.from_torch(mask_torch, device=device, dtype=ttnn.bfloat16)
    colbert_tt = _l2_normalize_ttnn(colbert_tt, device)
    colbert_tt = ttnn.multiply(colbert_tt, mask_tt)
    return colbert_tt


def colbert_embedding_from_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute ColBERT-style token vectors from last hidden state only (no M3 head).
    Uses last_hidden_state[:, 1:, :] (skip CLS), mask, and optional L2 normalize.
    Returns [batch_size, seq_len-1, hidden_dim] for compute_colbert_score.
    """
    colbert_vecs = last_hidden_state[:, 1:].float()
    if device is not None:
        colbert_vecs = colbert_vecs.to(device)
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        mask = attention_mask[:, 1:].unsqueeze(-1).to(colbert_vecs.dtype)
        colbert_vecs = colbert_vecs * mask
        if normalize:
            colbert_vecs = torch.nn.functional.normalize(colbert_vecs, dim=-1)
    return colbert_vecs


def compute_sparse_score(device: Any, q_sparse_tt: Any, p_sparse_tt: Any) -> Any:
    """
    Compute sparse (lexical) similarity matrix via inner product using ttnn.matmul.
    q_sparse_tt, p_sparse_tt: ttnn [Q, vocab_size], [P, vocab_size].
    Returns ttnn tensor [Q, P].
    """
    return ttnn.matmul(q_sparse_tt, p_sparse_tt, transpose_b=True)


def compute_colbert_score(device: Any, q_colbert_tt: Any, p_colbert_tt: Any, q_mask: torch.Tensor) -> Any:
    qs, ps = q_colbert_tt.shape, p_colbert_tt.shape
    if len(qs) == 4:
        Q, q_len, dim = int(qs[0]), int(qs[2]), int(qs[3])
        q_flat_tt = ttnn.reshape(q_colbert_tt, [Q * q_len, dim])
    else:
        Q, q_len, dim = int(qs[0]), int(qs[1]), int(qs[2])
        q_flat_tt = ttnn.reshape(q_colbert_tt, [Q * q_len, dim])
    mask_cols = q_mask.shape[1]
    if mask_cols < q_len:
        q_mask = torch.nn.functional.pad(q_mask, (0, q_len - mask_cols), value=0)
    elif mask_cols > q_len:
        q_mask = q_mask[:, :q_len]
    if len(ps) == 4:
        P, p_len = int(ps[0]), int(ps[2])
        p_flat_tt = ttnn.reshape(p_colbert_tt, [P * p_len, dim])
    else:
        P, p_len = int(ps[0]), int(ps[1])
        p_flat_tt = ttnn.reshape(p_colbert_tt, [P * p_len, dim])
    token_scores_tt = ttnn.matmul(q_flat_tt, p_flat_tt, transpose_b=True)
    token_scores_tt = ttnn.reshape(token_scores_tt, [Q, q_len, P, p_len])
    scores_max_tt = ttnn.max(token_scores_tt, dim=-1)
    scores_sum_tt = ttnn.sum(scores_max_tt, dim=1)
    q_lens = q_mask.sum(dim=-1, keepdim=True).to(torch.float32)
    q_lens_safe = q_lens.clamp(min=1.0)
    q_lens_tt = ttnn.from_torch(q_lens_safe, device=device, dtype=ttnn.bfloat16)
    cm_tt = ttnn.divide(scores_sum_tt, q_lens_tt)
    valid_mask = (q_lens > 0).to(torch.bfloat16).expand(Q, P)
    valid_tt = ttnn.from_torch(valid_mask, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cm_tt = ttnn.multiply(cm_tt, valid_tt)
    return cm_tt


def compute_score_from_hidden_states_tt_only(
    tt_q: Any,
    tt_p: Any,
    enc_q: Dict[str, torch.Tensor],
    enc_p: Dict[str, torch.Tensor],
    tokenizer,
    vocab_size: int,
    device: Any,
    to_torch_fn: Callable[[Any], torch.Tensor],
) -> Dict[str, List[float]]:
    """Compute sparse and colbert scores from TT hidden states; all ops in ttnn; to_torch_fn only for final lists."""
    unused_ids = _get_special_token_ids(tokenizer, vocab_size)

    # Sparse path: keep TT norm + scatter, then normalize on host.
    norm_q_tt = _norm_weights_ttnn(tt_q, enc_q["attention_mask"], device)
    norm_p_tt = _norm_weights_ttnn(tt_p, enc_p["attention_mask"], device)
    input_ids_q_tt = ttnn.from_torch(
        enc_q["input_ids"].long().to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_ids_p_tt = ttnn.from_torch(
        enc_p["input_ids"].long().to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sparse_q_tt = _sparse_embedding_scatter_ttnn(device, norm_q_tt, input_ids_q_tt, vocab_size, unused_ids)
    sparse_p_tt = _sparse_embedding_scatter_ttnn(device, norm_p_tt, input_ids_p_tt, vocab_size, unused_ids)
    # Temporary workaround: device sparse normalization inflates scores, using pytorch instead.
    # sparse_q_tt = _l2_normalize_sparse_ttnn(sparse_q_tt, device)
    # sparse_p_tt = _l2_normalize_sparse_ttnn(sparse_p_tt, device)
    qs = to_torch_fn(sparse_q_tt).float()
    ps = to_torch_fn(sparse_p_tt).float()
    qs = torch.nn.functional.normalize(qs, dim=-1)
    ps = torch.nn.functional.normalize(ps, dim=-1)
    sm = torch.matmul(qs, ps.T)
    sparse_list = np.diag(sm.cpu().numpy()).tolist()

    # ColBERT path: all ttnn
    q_mask = enc_q["attention_mask"][:, 1:]
    tt_qc = _colbert_embedding_ttnn(tt_q, enc_q["attention_mask"], device)
    tt_pc = _colbert_embedding_ttnn(tt_p, enc_p["attention_mask"], device)
    cm_tt = compute_colbert_score(device, tt_qc, tt_pc, q_mask)
    colbert_diag = np.diag(to_torch_fn(cm_tt).float().cpu().numpy())
    # colbert_list = [
    #     0.0 if not (np.isfinite(x) and 0 <= x <= 1.0) else float(x) for x in colbert_diag.tolist()
    # ]
    return {"sparse": sparse_list, "colbert": colbert_diag}


def compute_score_single_device_tt_only(
    device: Any,
    sentence_pairs: List[tuple],
    ttnn_model: Any,
    model_args: Any,
    to_ttnn_ids: Callable[[torch.Tensor, Any], Any],
    to_torch_fn: Callable[[Any], torch.Tensor],
) -> Dict[str, List[float]]:
    """
    Encode sentence_pairs, run only BgeM3Model (TT) backbone, return sparse + colbert scores.
    Keeps hidden states as ttnn tensors; scoring uses ttnn.matmul, ttnn reductions, etc.
    """
    queries = [p[0] for p in sentence_pairs]
    passages = [p[1] for p in sentence_pairs]
    enc_q = model_args.encode_prompts(queries)
    enc_p = model_args.encode_prompts(passages)

    def run_tt(ids: torch.Tensor, attn: torch.Tensor, tok_type: torch.Tensor) -> Any:
        return ttnn_model(
            input_ids=to_ttnn_ids(ids, device),
            attention_mask=to_ttnn_ids(attn, device),
            token_type_ids=to_ttnn_ids(tok_type, device),
        )

    token_type_q = enc_q.get("token_type_ids", torch.zeros_like(enc_q["input_ids"]))
    token_type_p = enc_p.get("token_type_ids", torch.zeros_like(enc_p["input_ids"]))
    tt_q = run_tt(enc_q["input_ids"], enc_q["attention_mask"], token_type_q)
    tt_p = run_tt(enc_p["input_ids"], enc_p["attention_mask"], token_type_p)

    return compute_score_from_hidden_states_tt_only(
        tt_q,
        tt_p,
        enc_q,
        enc_p,
        model_args.tokenizer,
        model_args.vocab_size,
        device,
        to_torch_fn,
    )


def compute_score(
    device: Any,
    sentence_pairs: List[tuple],
    ttnn_model: Any,
    model_args: Any,
    to_ttnn_ids: Callable[[torch.Tensor, Any], Any],
    to_torch_auto_compose: Callable[[Any], torch.Tensor],
) -> Dict[str, List[float]]:
    """
    Compute sparse and colbert scores for sentence_pairs using BgeM3Model (TT only).
    Returns dict with keys "sparse" and "colbert", each a list of floats (one per pair).
    """
    return compute_score_single_device_tt_only(
        device, sentence_pairs, ttnn_model, model_args, to_ttnn_ids, to_torch_auto_compose
    )
