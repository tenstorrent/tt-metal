# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence

import torch

from specfr.base_runtime import DecodeState


@dataclass(frozen=True)
class PathVerification:
    accepted_prefix_lengths: list[int]
    proposed_paths: list[list[int]]
    verification_batch_size: int | None = None
    verification_seq_len: int | None = None
    base_argmax_per_path: list[list[int]] | None = None
    base_argmax_pos0: int | None = None
    # Per-path bonus token: base argmax at the first rejection point (or after
    # the last accepted token when the full path is accepted). Standard
    # speculative decoding always yields accepted_len+1 tokens from one verify.
    bonus_token_per_path: list[int] | None = None


def _accept_with_mode(
    logits_1d: torch.Tensor,
    expected_token: int,
    *,
    acceptance_mode: str,
    rng: random.Random | None,
    draft_prob: float | None = None,
) -> bool:
    if expected_token < 0 or expected_token >= logits_1d.shape[-1]:
        return False
    if acceptance_mode == "argmax":
        predicted_token = int(torch.argmax(logits_1d, dim=-1).item())
        return predicted_token == int(expected_token)
    if acceptance_mode == "probabilistic":
        probs = torch.softmax(logits_1d.float(), dim=-1)
        base_prob = float(probs[int(expected_token)].item())
        if draft_prob is not None and draft_prob > 0:
            accept_prob = min(1.0, base_prob / draft_prob)
        else:
            accept_prob = base_prob
        sample = rng.random() if rng is not None else random.random()
        return sample < accept_prob
    raise ValueError(f"Unknown acceptance_mode='{acceptance_mode}'")


def verify_paths_from_decode_state(
    decode_state: DecodeState,
    proposed_paths: Sequence[Sequence[int]],
    *,
    clone_decode_state: Callable[[DecodeState], DecodeState],
    advance_decode_state: Callable[[DecodeState, int], DecodeState],
    acceptance_mode: str = "argmax",
    rng: random.Random | None = None,
    draft_probs_per_path: Sequence[Sequence[float]] | None = None,
    return_base_argmax: bool = False,
) -> PathVerification:
    if len(proposed_paths) == 0:
        return PathVerification(accepted_prefix_lengths=[], proposed_paths=[], verification_batch_size=0, verification_seq_len=0)
    if any(len(path) == 0 for path in proposed_paths):
        raise ValueError("All proposed paths must contain at least one token.")

    base_argmax_pos0 = int(torch.argmax(decode_state.next_token_logits.reshape(-1), dim=-1).item())
    accepted_prefix_lengths: list[int] = []
    base_argmax_per_path: list[list[int]] | None = [] if return_base_argmax else None
    bonus_token_per_path: list[int] = []
    for path_idx, path in enumerate(proposed_paths):
        path_probs = draft_probs_per_path[path_idx] if draft_probs_per_path is not None and path_idx < len(draft_probs_per_path) else None
        local_state = clone_decode_state(decode_state)
        accepted = 0
        this_argmax: list[int] = []
        for path_offset, expected_token in enumerate(path):
            logits_1d = local_state.next_token_logits.reshape(-1)
            if return_base_argmax:
                this_argmax.append(int(torch.argmax(logits_1d, dim=-1).item()))
            draft_prob = path_probs[path_offset] if path_probs is not None and path_offset < len(path_probs) else None
            if not _accept_with_mode(
                logits_1d,
                int(expected_token),
                acceptance_mode=acceptance_mode,
                rng=rng,
                draft_prob=draft_prob,
            ):
                break
            accepted += 1
            if path_offset + 1 < len(path):
                local_state = advance_decode_state(local_state, int(expected_token))
        accepted_prefix_lengths.append(accepted)
        if return_base_argmax:
            base_argmax_per_path.append(this_argmax)
        if accepted == len(path) and len(path) > 0:
            local_state = advance_decode_state(local_state, int(path[-1]))
        bonus_token_per_path.append(int(torch.argmax(local_state.next_token_logits.reshape(-1), dim=-1).item()))

    return PathVerification(
        accepted_prefix_lengths=accepted_prefix_lengths,
        proposed_paths=[list(path) for path in proposed_paths],
        verification_batch_size=len(proposed_paths),
        verification_seq_len=None,
        base_argmax_per_path=base_argmax_per_path,
        base_argmax_pos0=base_argmax_pos0,
        bonus_token_per_path=bonus_token_per_path,
    )


def verify_paths_batched_single_pass(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    device: torch.device,
    prefix_token_ids: Sequence[int],
    proposed_paths: Sequence[Sequence[int]],
    acceptance_mode: str = "argmax",
    rng: random.Random | None = None,
    draft_probs_per_path: Sequence[Sequence[float]] | None = None,
    return_base_argmax: bool = False,
    per_path_forward: bool = False,
) -> PathVerification:
    """Verify each path against the base model."""
    if len(prefix_token_ids) == 0:
        raise ValueError("prefix_token_ids must be non-empty.")
    if len(proposed_paths) == 0:
        return PathVerification(accepted_prefix_lengths=[], proposed_paths=[], verification_batch_size=0, verification_seq_len=0)
    if any(len(path) == 0 for path in proposed_paths):
        raise ValueError("All proposed paths must contain at least one token.")

    prefix = list(prefix_token_ids)
    prefix_len = len(prefix)
    accepted_prefix_lengths: list[int] = []
    base_argmax_per_path: list[list[int]] | None = [] if return_base_argmax else None
    base_argmax_pos0: int | None = None
    bonus_token_per_path: list[int] = []

    if per_path_forward:
        for path in proposed_paths:
            seq = prefix + list(path)
            path_probs = None
            if draft_probs_per_path is not None and len(draft_probs_per_path) > len(accepted_prefix_lengths):
                path_probs = draft_probs_per_path[len(accepted_prefix_lengths)]
            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
            accepted = 0
            this_argmax: list[int] = []
            for path_offset, expected_token in enumerate(path):
                logits_at_pos = logits[0, prefix_len - 1 + path_offset, :]
                if return_base_argmax:
                    this_argmax.append(int(torch.argmax(logits_at_pos, dim=-1).item()))
                draft_prob = path_probs[path_offset] if path_probs is not None and path_offset < len(path_probs) else None
                if _accept_with_mode(
                    logits_at_pos,
                    int(expected_token),
                    acceptance_mode=acceptance_mode,
                    rng=rng,
                    draft_prob=draft_prob,
                ):
                    accepted += 1
                    continue
                break
            if base_argmax_pos0 is None:
                base_argmax_pos0 = int(torch.argmax(logits[0, prefix_len - 1, :], dim=-1).item())
            accepted_prefix_lengths.append(accepted)
            if return_base_argmax:
                base_argmax_per_path.append(this_argmax)
            bonus_token_per_path.append(int(torch.argmax(logits[0, prefix_len - 1 + accepted, :], dim=-1).item()))
        max_seq_len = max(len(prefix) + len(path) for path in proposed_paths)
    else:
        sequences = [prefix + list(path) for path in proposed_paths]
        max_seq_len = max(len(seq) for seq in sequences)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        input_ids = torch.full(
            (len(sequences), max_seq_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros((len(sequences), max_seq_len), dtype=torch.long, device=device)
        for batch_idx, seq in enumerate(sequences):
            input_ids[batch_idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[batch_idx, : len(seq)] = 1
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        base_argmax_pos0 = int(torch.argmax(logits[0, prefix_len - 1, :], dim=-1).item())
        for batch_idx, path in enumerate(proposed_paths):
            path_probs = draft_probs_per_path[batch_idx] if draft_probs_per_path is not None and batch_idx < len(draft_probs_per_path) else None
            accepted = 0
            this_argmax = []
            for path_offset, expected_token in enumerate(path):
                logits_at_pos = logits[batch_idx, prefix_len - 1 + path_offset, :]
                if return_base_argmax:
                    this_argmax.append(int(torch.argmax(logits_at_pos, dim=-1).item()))
                draft_prob = path_probs[path_offset] if path_probs is not None and path_offset < len(path_probs) else None
                if _accept_with_mode(
                    logits_at_pos,
                    int(expected_token),
                    acceptance_mode=acceptance_mode,
                    rng=rng,
                    draft_prob=draft_prob,
                ):
                    accepted += 1
                    continue
                break
            accepted_prefix_lengths.append(accepted)
            if return_base_argmax:
                base_argmax_per_path.append(this_argmax)
            bonus_token_per_path.append(int(torch.argmax(logits[batch_idx, prefix_len - 1 + accepted, :], dim=-1).item()))

    return PathVerification(
        accepted_prefix_lengths=accepted_prefix_lengths,
        proposed_paths=[list(path) for path in proposed_paths],
        verification_batch_size=len(proposed_paths),
        verification_seq_len=max_seq_len,
        base_argmax_per_path=base_argmax_per_path,
        base_argmax_pos0=base_argmax_pos0,
        bonus_token_per_path=bonus_token_per_path,
    )


def verify_paths_flattened_tree(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    device: torch.device,
    prefix_token_ids: Sequence[int],
    proposed_paths: Sequence[Sequence[int]],
    acceptance_mode: str = "argmax",
    rng: random.Random | None = None,
    draft_probs_per_path: Sequence[Sequence[float]] | None = None,
    return_base_argmax: bool = False,
) -> PathVerification:
    """Verify paths with a single forward over a flattened tree with custom attention mask."""
    if len(prefix_token_ids) == 0:
        raise ValueError("prefix_token_ids must be non-empty.")
    if len(proposed_paths) == 0:
        return PathVerification(accepted_prefix_lengths=[], proposed_paths=[], verification_batch_size=0, verification_seq_len=0)
    if any(len(path) == 0 for path in proposed_paths):
        raise ValueError("All proposed paths must contain at least one token.")

    prefix = list(prefix_token_ids)
    prefix_len = len(prefix)
    paths = [list(p) for p in proposed_paths]
    path_lens = [len(p) for p in paths]
    path_starts: list[int] = []
    s = prefix_len
    for L in path_lens:
        path_starts.append(s)
        s += L
    total_len = s

    flat_tokens: list[int] = prefix[:]
    for path in paths:
        flat_tokens.extend(path)
    input_ids = torch.tensor([flat_tokens], dtype=torch.long, device=device)

    dtype_float = torch.float32
    if hasattr(model, "dtype") and model.dtype is not None:
        dtype_float = model.dtype
    attention_mask = torch.full((total_len, total_len), float("-inf"), dtype=dtype_float, device=device)
    for i in range(prefix_len):
        attention_mask[i, : i + 1] = 0
    for k, (start, L) in enumerate(zip(path_starts, path_lens)):
        for j in range(L):
            pos = start + j
            attention_mask[pos, :prefix_len] = 0
            attention_mask[pos, start : start + j + 1] = 0
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits_2d = logits[0]

    base_argmax_pos0 = int(torch.argmax(logits_2d[prefix_len - 1, :], dim=-1).item())
    accepted_prefix_lengths: list[int] = []
    base_argmax_per_path: list[list[int]] | None = [] if return_base_argmax else None
    bonus_token_per_path: list[int] = []

    for k, path in enumerate(paths):
        path_probs = draft_probs_per_path[k] if draft_probs_per_path is not None and k < len(draft_probs_per_path) else None
        start = path_starts[k]
        accepted = 0
        this_argmax: list[int] = []
        for j, expected_token in enumerate(path):
            logits_at_pos = logits_2d[start - 1 + j, :]
            if return_base_argmax:
                this_argmax.append(int(torch.argmax(logits_at_pos, dim=-1).item()))
            draft_prob = path_probs[j] if path_probs is not None and j < len(path_probs) else None
            if not _accept_with_mode(
                logits_at_pos,
                int(expected_token),
                acceptance_mode=acceptance_mode,
                rng=rng,
                draft_prob=draft_prob,
            ):
                break
            accepted += 1
        accepted_prefix_lengths.append(accepted)
        if return_base_argmax:
            base_argmax_per_path.append(this_argmax)
        bonus_token_per_path.append(int(torch.argmax(logits_2d[start - 1 + accepted, :], dim=-1).item()))

    return PathVerification(
        accepted_prefix_lengths=accepted_prefix_lengths,
        proposed_paths=paths,
        verification_batch_size=1,
        verification_seq_len=total_len,
        base_argmax_per_path=base_argmax_per_path,
        base_argmax_pos0=base_argmax_pos0,
        bonus_token_per_path=bonus_token_per_path,
    )
