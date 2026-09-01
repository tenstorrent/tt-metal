# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-only tests for the resumed-prefill offset alignment in
``models.tt_transformers.tt.generator.Generator``.

A resumed prefill (prefix caching, or a prompt split across engine steps) hands
the traced chunked SDPA a ``chunk_start_idx``. That op reads the wrong prefix
instead of raising when the offset is not a multiple of the ``q_chunk_size`` its
program was captured with, so the offset is floored to
``lcm(block_size, q_chunk_size)`` before use.

These are pure functions of ``(num_cached, seq_len, block_size, q_chunk_size)``.
The tests bind the real methods to a stub so they run without a device.
"""

import math
from types import SimpleNamespace

import pytest

from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import Generator

# The buckets Llama-3.1-8B traces on T3K, and the ceiling warmup applies to them.
TRACED_BUCKETS = (128, 1024, 2048, 4096, 8192)
CAPPED_WARMUP_SEQ_LEN = 131072


def _q_chunk_size(seq_len):
    """The rule ``ModelArgs.get_attn_sdpa_prefill_program_config`` applies at chunk_start_idx=0."""
    return 256 if seq_len >= 2048 else 64


class _StubModelArgs:
    def __init__(self, q_chunk_size_fn=_q_chunk_size):
        self._q_chunk_size_fn = q_chunk_size_fn
        self.capped_warmup_seq_len = CAPPED_WARMUP_SEQ_LEN

    def get_attn_sdpa_program_config(self, mode, seq_len, chunk_start_idx, _unused):
        return SimpleNamespace(q_chunk_size=self._q_chunk_size_fn(seq_len))


class _StubModelArgsNoProgramConfig:
    """A model whose args cannot describe the pin, e.g. Gemma4ModelArgs."""

    def __init__(self):
        self.capped_warmup_seq_len = CAPPED_WARMUP_SEQ_LEN


class _StubGenerator:
    """Borrows the real methods so the tests exercise shipped code, not a copy."""

    _traced_sdpa_q_chunk_size = Generator._traced_sdpa_q_chunk_size
    _resume_offset_alignment = Generator._resume_offset_alignment
    _assert_uniform_resume_alignment = Generator._assert_uniform_resume_alignment
    _align_resume_offsets = Generator._align_resume_offsets
    # Read off the class a staticmethod is a plain function, which would rebind here.
    _resumed_warmup_prompt_len = staticmethod(Generator._resumed_warmup_prompt_len)

    def __init__(self, block_size, model_args=None, model_capabilities=None):
        args = model_args if model_args is not None else [_StubModelArgs()]
        self.model_args = args
        self.data_parallel = len(args)
        self.model_capabilities = model_capabilities or {}
        self._block_size = block_size

    def _paged_prefill_block_size(self, _kv_cache):
        return self._block_size

    def align(self, num_cached_per_user, prompt_lens):
        # kv_cache only has to be non-None: _paged_prefill_block_size is stubbed.
        return self._align_resume_offsets(num_cached_per_user, prompt_lens, kv_cache=[object()])


@pytest.mark.parametrize("block_size", [32, 64, 128, 256])
@pytest.mark.parametrize("bucket", TRACED_BUCKETS)
def test_alignment_is_lcm_of_block_size_and_pin(block_size, bucket):
    gen = _StubGenerator(block_size)
    alignment = gen._resume_offset_alignment(bucket, block_size)
    expected = math.lcm(block_size, _q_chunk_size(bucket))
    assert alignment == expected
    assert alignment % block_size == 0, "paged ops need the page-table slice to land on a block"
    assert alignment % _q_chunk_size(bucket) == 0, "the traced SDPA program pins this q_chunk_size"


def test_zero_offset_short_circuits_without_consulting_the_alignment():
    """An ordinary prefill passes a zero-filled start_pos and must not need an alignment."""
    gen = _StubGenerator(64, model_args=[_StubModelArgsNoProgramConfig()], model_capabilities={})
    assert gen.align([0, 0], [1024, 4096]) == [0, 0]


def test_undeclarable_alignment_raises_on_a_real_resume(expect_error):
    gen = _StubGenerator(64, model_args=[_StubModelArgsNoProgramConfig()], model_capabilities={})
    with expect_error(ValueError, "resumed_prefill_token_alignment"):
        gen.align([128], [4096])


def test_declared_alignment_is_used_when_the_pin_cannot_be_derived():
    """Gemma 4's case: chunked_prefill_sdpa pins 128 and the model args cannot say so."""
    gen = _StubGenerator(
        64,
        model_args=[_StubModelArgsNoProgramConfig()],
        model_capabilities={"resumed_prefill_token_alignment": 128},
    )
    assert gen.align([200], [4096]) == [128]


def test_already_aligned_offset_is_unchanged():
    gen = _StubGenerator(64)
    # 4096 pins 256; 512 is already a multiple of lcm(64, 256) = 256.
    assert gen.align([512], [8192]) == [512]


def test_settle_loop_takes_the_second_hop():
    """One pass is not enough: flooring lengthens the suffix, which can raise the pin.

    block_size 32, seq_len 1120, start_pos 100. A single pass floors 100 against the
    suffix at offset 96, whose padded length is 1024 and pins 64, giving 64. The
    longer suffix at offset 64 pads to 2048 and pins 256, so the offset must settle
    lower still.
    """
    gen = _StubGenerator(32)
    (settled,) = gen.align([100], [1120])
    assert settled == 0
    # The result is a fixed point: re-running does not move it.
    assert gen.align([settled], [1120]) == [settled]


def test_settled_offset_is_a_fixed_point_across_the_bucket_range():
    gen = _StubGenerator(64)
    for seq_len in (200, 1120, 2100, 4500, 9000):
        for start_pos in range(0, seq_len, 37):
            (settled,) = gen.align([start_pos], [seq_len])
            assert 0 <= settled <= start_pos, "flooring never raises the offset"
            assert gen.align([settled], [seq_len]) == [settled], "not a fixed point"


def test_each_user_is_aligned_independently():
    gen = _StubGenerator(64)
    assert gen.align([0, 100, 512], [1024, 1120, 8192]) == [0, 0, 512]


@pytest.mark.parametrize("block_size", [64, 128, 256, 1024])
@pytest.mark.parametrize("bucket", TRACED_BUCKETS)
def test_resumed_warmup_prompt_reaches_the_intended_bucket(block_size, bucket):
    """Regression for the block_size >= 128 startup failure.

    Spanning the bucket alone gave a suffix of ``bucket - num_cached``, which is 0 at
    block_size 128 and negative at 256, so warmup tripped
    ``assert 0 <= num_cached < seq_len`` and the server never started.
    """
    gen = _StubGenerator(block_size)
    num_cached = gen._resume_offset_alignment(bucket, block_size)
    total_seq_len = gen._resumed_warmup_prompt_len(bucket, num_cached, CAPPED_WARMUP_SEQ_LEN)
    suffix = total_seq_len - num_cached

    assert suffix > 0, "a resumed warmup with no suffix has nothing to prefill"
    assert get_padded_prefill_len(suffix) == bucket, "the captured trace would be for another bucket"
    assert 0 <= num_cached < total_seq_len, "would trip the guard in _prefill_forward_text_impl"


def test_resumed_warmup_prompt_respects_the_warmup_ceiling():
    """When the ceiling equals the bucket the prompt is capped, and the suffix still fits."""
    gen = _StubGenerator(64)
    bucket = 8192
    num_cached = gen._resume_offset_alignment(bucket, 64)
    total_seq_len = gen._resumed_warmup_prompt_len(bucket, num_cached, capped_warmup_seq_len=bucket)
    assert total_seq_len == bucket
    assert get_padded_prefill_len(total_seq_len - num_cached) == bucket


def test_heterogeneous_replicas_are_rejected_rather_than_silently_aligned_to_replica_zero(expect_error):
    replica_0 = _StubModelArgs()
    replica_1 = _StubModelArgs(q_chunk_size_fn=lambda seq_len: 512)
    gen = _StubGenerator(64, model_args=[replica_0, replica_1])
    with expect_error(AssertionError, "replica 1"):
        gen.align([300], [4096])
