# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate a complete public prefill submission before any device mutation.

Only CPU request tensors and cache metadata are inspected. This module never
prepares programs, reads device payloads, uploads tensors, or changes state.
Off-batch page ownership remains the scheduler/admission layer's responsibility.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from .cache_contract import validate_page_ownership
from .serving import _validated_pages

_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint32, torch.uint64)


@dataclass(frozen=True)
class PrefillSubmission:
    tokens: torch.Tensor
    slots: tuple[int, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    requested_indices: tuple[tuple[int, ...], ...] | None
    page_tables_per_layer: tuple[torch.Tensor, ...] | None


def _integers(value, name):
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a one-dimensional integer array")
        value = value.tolist()
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 1 or value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name} must be a CPU one-dimensional integer tensor")
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list, tuple, or CPU integer vector")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise ValueError(f"{name} must contain integer values")
    return tuple(int(item) for item in value)


def validate_prefill(
    generator,
    tokens,
    *,
    prompt_lens,
    start_pos=None,
    slots=None,
    empty_slots=None,
    kv_cache=None,
    page_table=None,
    page_tables_per_layer=None,
    return_all_logits=False,
    device_output=False,
    logit_positions=None,
):
    """Normalize a whole batch or raise before preparation/cache mutation.

    ``tokens`` carries complete token rows. Each request consumes
    ``tokens[row, starts[row]:ends[row]]`` at those absolute positions.
    Columns beyond each logical end are ignored and normalized to zero.
    ``logit_positions`` are relative to each submitted suffix and retain caller
    order/duplicates. Page-table rows use submission order, not physical slots;
    additional capacity rows in plugin-provided tables are permitted.
    """
    if generator.closed:
        raise RuntimeError("Prefill requires an open generator")
    if generator.slot_state.poisoned or generator.reload_executor.poisoned:
        raise RuntimeError("Reset the poisoned generator before accepting prefill")
    cache = generator.cache if kv_cache is None else kv_cache
    if cache is not generator.cache:
        raise ValueError("Bind externally owned cache before generator preparation")
    if type(return_all_logits) is not bool or type(device_output) is not bool:
        raise TypeError("Prefill output selectors must be booleans")
    if return_all_logits and device_output:
        raise ValueError("All-logits compatibility output is host-chunked; device output returns the final row")
    if logit_positions is not None and not return_all_logits:
        raise ValueError("logit_positions requires return_all_logits=True")
    if slots is not None and empty_slots is not None:
        raise ValueError("Specify slots or empty_slots, not both")
    ends = _integers(prompt_lens, "prompt_lens")
    count = len(ends)
    if not 1 <= count <= generator.max_batch_size:
        raise ValueError("Prefill requires 1..max_batch_size rows")
    selected = empty_slots if slots is None else slots
    rows = tuple(range(count)) if selected is None else _integers(selected, "slots")
    starts = (0,) * count if start_pos is None else _integers(start_pos, "start_pos")
    if len(rows) != count or len(starts) != count:
        raise ValueError("Prefill lengths, starts and slots must have equal row counts")
    if len(set(rows)) != count or any(slot < 0 or slot >= generator.max_batch_size for slot in rows):
        raise ValueError("Prefill slots must be unique and within configured capacity")
    source = torch.as_tensor(tokens)
    if source.device.type != "cpu" or source.ndim != 2 or source.shape[0] != count:
        raise ValueError("Prefill tokens must be a CPU [rows, full-token-width] tensor")
    if source.dtype not in _INTEGER_DTYPES:
        raise ValueError("Prefill tokens must have integer dtype")
    width = source.shape[1]
    for start, end in zip(starts, ends):
        if not 0 <= start < end <= generator.model.max_seq_len:
            raise ValueError("Invalid absolute logical prefill interval")
        if end > width:
            raise ValueError("Absolute prompt length exceeds the supplied token row width")
    # Python integer comparison avoids UINT64 -> INT64 overflow concealing an
    # invalid token. Only logical input columns count; padding is not a token.
    normalized = torch.zeros((count, width), dtype=torch.int32)
    for row, end in enumerate(ends):
        logical = source[row, :end].tolist()
        if any(token < 0 or token >= generator.model.vocab_size for token in logical):
            raise ValueError("Logical prompt contains a token outside the real vocabulary")
        normalized[row, :end] = torch.tensor(logical, dtype=torch.int32)
    requested = None
    if logit_positions is not None:
        if isinstance(logit_positions, torch.Tensor):
            if logit_positions.device.type != "cpu" or logit_positions.ndim != 2:
                raise ValueError("Requested logit positions must provide one CPU vector per row")
            indices = list(logit_positions)
        elif isinstance(logit_positions, (list, tuple)):
            indices = list(logit_positions)
        else:
            raise TypeError("Requested logit positions must provide one integer sequence per row")
        if len(indices) != count:
            raise ValueError("Requested logit-position rows must match prompt rows")
        requested = tuple(_integers(value, "logit_positions") for value in indices)
        for values, start, end in zip(requested, starts, ends):
            if not values or any(position < 0 or position >= end - start for position in values):
                raise ValueError("Requested logits must be nonempty and lie within each logical suffix")
    if len(cache) != len(generator.model.layers) or any(len(states) != generator.max_batch_size for states in cache):
        raise ValueError("Bound cache must cover every layer and configured slot")
    full = [states for states in cache if "page_table" in states[0]]
    for states in full:
        for slot, end in zip(rows, ends):
            if end > states[slot]["context_length"]:
                raise ValueError("Logical prefill exceeds allocated KV context")
    if full and count > 1 and page_table is None and page_tables_per_layer is None:
        raise ValueError("Mixed prefill requires explicit page mappings for the aggregate physical pool")
    normalized_pages = _validated_pages(
        generator, page_table, page_tables_per_layer, rows, rows, tuple(end - 1 for end in ends)
    )
    if normalized_pages is not None:
        ownership_pages = normalized_pages
    else:
        ownership_pages = []
        for states in full:
            shadows = [states[slot].get("_host_pages") for slot in rows]
            if any(shadow is None for shadow in shadows):
                raise ValueError("No CPU shadow exists for the bound page table; provide explicit mappings")
            ownership_pages.append(torch.cat(shadows, dim=0))
    # This establishes intra-submission ownership only. Admission must also
    # validate remembered off-batch live extents; they cannot be inferred here.
    validate_page_ownership(cache, ownership_pages, slots=rows, prompt_lens=ends)
    return PrefillSubmission(
        tokens=normalized,
        slots=rows,
        starts=starts,
        ends=ends,
        requested_indices=requested,
        page_tables_per_layer=None if normalized_pages is None else tuple(normalized_pages),
    )
