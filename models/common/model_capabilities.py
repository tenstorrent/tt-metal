# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

# Maximum number of tokens that can simultaneously occupy the KV cache
# across all concurrent users. This fallback applies to model/device
# configurations not covered by a model-specific override.
# Derived from the default branch of the per-model KV-cache rules in
# the TT vLLM worker (tenstorrent/vllm#315).
# See also: https://github.com/tenstorrent/vllm/issues/315
FALLBACK_MAX_TOKENS_ALL_USERS = 131_072


class ModelCapabilitiesMixin:
    """Defines interface for hardware- or model-specific configurations.

    NOTE: The default values here and per-model overrides will eventually be
    unified with the corresponding vLLM scheduler configuration so that both
    paths derive from the same source of truth.

    Generator classes also carry a class-level ``model_capabilities`` dict that
    the vLLM TT plugin reads off the bridge class at config time. A key that is
    absent means "not supported"; the plugin never assumes a default. Recognized
    keys:

    ``supports_prefix_caching`` (bool)
        The generator accepts a nonzero ``start_pos`` for a prompt whose prefix
        is already in the KV cache.
    ``supports_async_decode`` (bool)
        ``decode_forward(..., read_from_device=False)`` followed by
        ``read_decode_output(..., async_read=True)`` is implemented, so the
        engine may overlap scheduling with device execution.
    ``supports_sample_on_device`` (bool)
        The full on-device sampling pipeline is implemented.
    ``supports_chunked_prefill`` (bool)
        One prompt may be prefilled across several engine steps. Independent of
        ``supports_prefix_caching``: they share the resume plumbing but gate on
        different validation.
    ``chunked_prefill_token_alignment`` (positive int)
        Required alongside ``supports_chunked_prefill``. The largest
        ``q_chunk_size`` the model's chunked-SDPA program config can be built
        with. That op requires ``chunk_start_idx`` to be a multiple of it and
        reads the wrong prefix rather than raising when it is not, so every
        resume offset is floored to :func:`resume_offset_alignment` of this and
        the paged KV ``block_size``. Declare it whether or not the model traces
        prefill: tracing only removes the chance to derive the config from the
        offset instead.
    """

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs) -> int:
        """Returns the fallback all-user KV-cache token capacity.

        Used when no model- or device-specific override applies.
        """
        return FALLBACK_MAX_TOKENS_ALL_USERS


def resume_offset_alignment(block_size: int, declared_alignment: int | None) -> int:
    """Multiple every prefill resume offset must land on.

    Two independent divisibility requirements, so the answer is their least
    common multiple, not the larger of the two:

    ``block_size``
        The chunk's K/V is written through the page-table slice
        ``[:, chunk_start // block_size : ...]``. An offset off that multiple
        shifts every write by ``chunk_start % block_size`` positions.

    ``declared_alignment``
        ``chunked_prefill_token_alignment``: the q_chunk_size the model's traced
        chunked-SDPA program is captured with.

    Both current pairs (256/64, 128/64) are powers of two, where the LCM and the
    maximum coincide. They are not required to be, and the maximum of 96 and 64
    satisfies neither.
    """
    if not declared_alignment:
        return block_size
    return math.lcm(block_size, int(declared_alignment))


def floor_to_alignment(offsets, alignment: int) -> list[int]:
    """Floor each resume offset down to ``alignment``.

    Flooring recomputes at most ``alignment - 1`` tokens whose K/V is rewritten
    identically into the same blocks, so it is semantically a no-op.
    """
    return [(int(offset) // alignment) * alignment for offset in offsets]
