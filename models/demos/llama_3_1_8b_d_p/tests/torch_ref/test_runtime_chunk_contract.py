# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The runtime's chunk contract, asserted without a device.

`TtPrefillRuntime.prefill_chunk` guards `[actual_start, actual_end)` on entry. Every one of those
guards exists because the failure it catches is **silent**: an out-of-contract range does not raise
anywhere downstream, it writes real values to wrong block-cyclic cache addresses, and the damage
first surfaces as a KV PCC drop many layers later.

That makes the guards worth testing directly rather than trusting them to fire in a device run —
and they are pure arithmetic, so they need no mesh. The model, mesh device and CCL manager are
stubbed to the few attributes the constructor touches; nothing here executes an op.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.llama_3_1_8b_d_p.reference.config import LlamaConfigConstants
from models.demos.llama_3_1_8b_d_p.tt.attention.kv_cache import cache_capacity
from models.demos.llama_3_1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntimeConfig, TtPrefillRuntime

CHUNK = 5120  # the spec's chunk size
SP, TP = 8, 4


class _StubMeshDevice:
    """Only `.shape` is read by the constructor."""

    shape = (SP, TP)


def _runtime(max_seq_len=10240, chunk_size=CHUNK, num_users=1):
    config = LlamaConfigConstants.from_json()
    mesh_config = SimpleNamespace(sp=SP, tp=TP, sp_axis=0, tp_axis=1)
    model = SimpleNamespace(hf_config=config.to_hf_config(), rope_setup=None)
    rt_config = TtPrefillRuntimeConfig(
        max_seq_len=max_seq_len, chunk_size=chunk_size, num_layers=32, num_users=num_users
    )
    return TtPrefillRuntime(model, _StubMeshDevice(), mesh_config, rt_config)


# The chunk input the guards inspect: only `.shape[-1]` is read, and it must be the PER-DEVICE
# token count (chunk_size / sp).
def _chunk_input(chunk_size=CHUNK, sp=SP):
    return SimpleNamespace(shape=(1, 1, chunk_size // sp))


def test_capacity_rounds_up_and_is_alignment_safe():
    """The runtime's capacity is `max_seq_len` rounded up to a whole chunk, and stays 32*sp aligned."""
    rt = _runtime(max_seq_len=131072, chunk_size=CHUNK)
    assert rt.config.capacity == cache_capacity(131072, CHUNK) == 133120
    assert rt.config.capacity % (32 * SP) == 0
    # An already-aligned length is not inflated by a whole chunk.
    assert _runtime(max_seq_len=10240, chunk_size=CHUNK).config.capacity == 10240


def test_config_rejects_a_misaligned_chunk_size():
    """chunk_size is the block-cyclic addressing period; a misaligned one corrupts addresses silently."""
    with pytest.raises(AssertionError, match="block-cyclic"):
        TtPrefillRuntimeConfig(max_seq_len=10240, chunk_size=5000, num_layers=32)


def test_runtime_rejects_a_chunk_size_not_divisible_by_32_sp():
    """32 alone is not enough — it must be a multiple of 32*sp for the SP block-cyclic walk."""
    with pytest.raises(AssertionError, match=r"32\*sp"):
        _runtime(max_seq_len=4096, chunk_size=64)  # 64 % 32 == 0 but 64 % 256 != 0


@pytest.mark.parametrize(
    "actual_start,actual_end,match",
    [
        (-CHUNK, 0, "non-negative"),
        (1, CHUNK, "chunk-aligned"),  # not a multiple of chunk_size
        (CHUNK, CHUNK, "empty chunk"),  # start == end
        (CHUNK, CHUNK - 1, "empty chunk"),  # end before start
        (0, CHUNK + 1, "more than one chunk"),  # spans past one chunk
        (2 * CHUNK, 2 * CHUNK + CHUNK, "past the cache capacity"),  # beyond a 10240 cache
    ],
)
def test_out_of_contract_ranges_are_rejected(actual_start, actual_end, match):
    """Each of these would silently write to the wrong cache addresses if it got through."""
    rt = _runtime()
    with pytest.raises(AssertionError, match=match):
        rt.prefill_chunk(_chunk_input(), object(), slot_id=0, actual_start=actual_start, actual_end=actual_end)


def test_in_contract_ranges_reach_the_model():
    """The two legal chunks of a 2-chunk sequence pass every guard.

    `model.forward` is stubbed to record its arguments, so this checks the guards let a valid range
    through AND that the range is threaded on correctly — `cached_len` is the write offset and
    `logical_n` the valid prefix end, and swapping them would still pass every assertion above.
    """
    rt = _runtime()
    calls = []
    rt.model.forward = lambda chunk_input, **kw: calls.append(kw) or "out"
    rt.rope_mats = lambda: ("cos", "sin")  # the real one needs a device

    assert rt.prefill_chunk(_chunk_input(), object(), slot_id=0, actual_start=0, actual_end=CHUNK) == "out"
    assert rt.prefill_chunk(
        _chunk_input(), object(), slot_id=0, actual_start=CHUNK, actual_end=2 * CHUNK
    ) == "out"

    assert [c["cached_len"] for c in calls] == [0, CHUNK], "cached_len must be the cache WRITE OFFSET"
    assert [c["logical_n"] for c in calls] == [CHUNK, 2 * CHUNK], "logical_n must be the valid prefix END"
    assert all(c["indexed_rope"] for c in calls), "the chunked path uses on-device indexed rope"
    assert all(c["return_logits"] is False for c in calls), "filling a KV cache needs no vocab projection"


def test_a_short_final_chunk_is_allowed():
    """`actual_end < actual_start + chunk_size` is legal: the tail of a final chunk may be padding.

    The chunk still OCCUPIES a full chunk of physical positions; only the real-token count differs.
    """
    rt = _runtime()
    rt.model.forward = lambda chunk_input, **kw: kw
    rt.rope_mats = lambda: ("cos", "sin")
    kw = rt.prefill_chunk(_chunk_input(), object(), slot_id=0, actual_start=CHUNK, actual_end=CHUNK + 32)
    assert kw["logical_n"] == CHUNK + 32


def test_slot_id_is_range_checked():
    """A slot beyond num_users addresses another user's cache region."""
    rt = _runtime(num_users=1)
    with pytest.raises(AssertionError, match="slot_id"):
        rt.prefill_chunk(_chunk_input(), object(), slot_id=1, actual_start=0, actual_end=CHUNK)


def test_wrong_per_device_token_count_is_rejected():
    """The chunk input must carry chunk_size/sp tokens per device, not the global count."""
    rt = _runtime()
    with pytest.raises(AssertionError, match="tokens per device"):
        rt.prefill_chunk(
            SimpleNamespace(shape=(1, 1, CHUNK)),  # global count, not per-device
            object(),
            slot_id=0,
            actual_start=0,
            actual_end=CHUNK,
        )


def test_make_chunk_input_rejects_an_oversized_chunk():
    """More tokens than chunk_size cannot be one chunk."""
    rt = _runtime()
    with pytest.raises(AssertionError, match="exceeds chunk_size"):
        rt.make_chunk_input([0] * (CHUNK + 1))
