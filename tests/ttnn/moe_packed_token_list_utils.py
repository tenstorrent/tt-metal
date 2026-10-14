# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Test-only decoding and layout goldens for packed MoE token lists."""

from typing import Sequence

####################################################################################################
# Packed token lists of the local output path (moe_ring_common.h, namespace moe_ring::token_list).
# MUST stay in sync with the constexpr equivalents there. The op's output 2 on that path is one
# uint32 page: token_list_header_words() words of segment starts (offsets[0..experts_per_device],
# the rest padding), then every local expert's 4-B entries (k_slot << 24 | token_id) back to back,
# expert e's count_e entries at offsets[e]; segment starts are aligned to 16 entries.
####################################################################################################
TOKEN_LIST_TOKEN_BITS = 24
TOKEN_LIST_SEGMENT_ALIGN_ENTRIES = 16
TOKEN_LIST_ENTRY_BYTES = 4


def token_list_align_entries(entries: int) -> int:
    return -(-entries // TOKEN_LIST_SEGMENT_ALIGN_ENTRIES) * TOKEN_LIST_SEGMENT_ALIGN_ENTRIES


def token_list_header_words(experts_per_device: int) -> int:
    return token_list_align_entries(experts_per_device + 1)


def token_list_entry_capacity(
    tokens: int, selected_experts_k: int, experts_per_device: int, tokens_per_chunk: int = 32
):
    """Every (token, k slot) at most once, per-segment alignment padding, and one chunk of tail (dm1 reads whole chunks)."""
    return (
        token_list_align_entries(
            tokens * selected_experts_k + (TOKEN_LIST_SEGMENT_ALIGN_ENTRIES - 1) * experts_per_device
        )
        + tokens_per_chunk
    )


def token_list_page_words(tokens: int, selected_experts_k: int, experts_per_device: int, tokens_per_chunk: int = 32):
    return token_list_header_words(experts_per_device) + token_list_entry_capacity(
        tokens, selected_experts_k, experts_per_device, tokens_per_chunk
    )


def token_list_segment_starts(counts: Sequence[int]) -> list[int]:
    """offsets[0..E] from the per-expert counts: offsets[e + 1] = align16(offsets[e] + count_e)."""
    starts = [0]
    for count in counts:
        starts.append(token_list_align_entries(starts[-1] + int(count)))
    return starts


def decode_packed_token_lists(page, experts_per_device: int, counts: Sequence[int] | None = None):
    """``[(token_ids, k_slots)]`` per local expert from one device's packed page (a 1-D integer tensor or list).
    With ``counts`` (output 0's per-expert counts) each list is trimmed to its expert's entries; without, a list
    runs to the next segment start and includes the alignment padding."""
    import torch

    words = torch.as_tensor(page).flatten().to(torch.int64) & 0xFFFFFFFF
    starts = words[: experts_per_device + 1].tolist()
    entries = words[token_list_header_words(experts_per_device) :]
    lists = []
    for local_expert in range(experts_per_device):
        stop = starts[local_expert] + int(counts[local_expert]) if counts is not None else starts[local_expert + 1]
        segment = entries[starts[local_expert] : stop]
        lists.append((segment & ((1 << TOKEN_LIST_TOKEN_BITS) - 1), segment >> TOKEN_LIST_TOKEN_BITS))
    return lists
