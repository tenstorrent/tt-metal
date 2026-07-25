# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the borrowed-prefix ownership contract (#51080 roadmap item 2).

The fixed full-span prefix read can hand back the model-owned KV cache itself instead of
cloning it (~2 whole-cache copies per layer per step of block-invariant data). Two independent
guards make that safe, and BOTH were needed — the second one was found only on device, after
the first one alone still produced ``TT_FATAL: Input Tensor is not allocated``:

1. ``denoise_hidden_forward`` consults ``MutablePrefixKVReader.owns_result`` before freeing the
   per-layer prompt source.
2. ``denoise_attention`` compares BUFFERS, not object identity, before freeing its
   ``ttnn.to_memory_config`` result. ``to_memory_config`` returns a *fresh Tensor object that
   aliases the input buffer* when no conversion is needed (device-observed:
   ``distinct_buffer=False, same_object=False``), so the original ``is not`` check deallocated
   the model KV cache.

CPU-only; no device required.

COVERAGE BOUNDARY — read before trusting this file alone. It tests the *predicate*
(`_is_distinct_buffer`) with fakes; it does NOT execute the two real call sites in
`denoise_attention`, and no CPU test invokes `denoise_attention` at all (building a faithful fake
for the whole attention path is not worth it). Those call sites are covered by:

* the consumer-side guard test
  `test_denoise_forward.py::test_denoise_hidden_forward_honours_prompt_source_ownership`, which
  drives the real `denoise_hidden_forward` and is mutation-verified — deleting the `owns_result`
  guard makes it fail; and
* the device A/B `doc/optimize_perf/verify_prefix_borrow.sh`, where `DG_PREFIX_BORROW=1` vs `0`
  must produce an identical `committed_sha256` on the full 30-layer traced path. A regression that
  freed the borrowed cache would abort that run outright.

So a revert of `_is_distinct_buffer` to an `is not` identity check is caught by the device gate,
not by CPU CI. If you touch the prefix ownership contract, run that A/B.
"""

from models.experimental.diffusion_gemma.tt.diffusion_attention import _is_distinct_buffer


class _FakeTensor:
    """Minimal stand-in exposing the buffer_address() surface the guard relies on."""

    def __init__(self, address, *, raises=False):
        self._address = address
        self._raises = raises
        self.freed = False

    def buffer_address(self):
        if self._raises:
            raise RuntimeError("buffer_address unavailable for this storage type")
        return self._address

    def deallocate(self, force=True):
        self.freed = True


def test_alias_of_the_same_buffer_is_not_freed():
    """The device-observed case: different object, SAME buffer -> must not be freed."""
    source = _FakeTensor(0xDEAD0000)
    alias = _FakeTensor(0xDEAD0000)
    assert alias is not source, "the whole point is that object identity says 'distinct'"
    assert _is_distinct_buffer(alias, source) is False


def test_genuine_copy_is_freed():
    """A real conversion produces a new buffer we own and must free, or it leaks per layer."""
    source = _FakeTensor(0xDEAD0000)
    copy = _FakeTensor(0xBEEF0000)
    assert _is_distinct_buffer(copy, source) is True


def test_same_object_is_not_freed():
    source = _FakeTensor(0xDEAD0000)
    assert _is_distinct_buffer(source, source) is False


def test_unknowable_buffer_defaults_to_not_freeing():
    """If ownership cannot be proven, leak rather than free a caller-owned tensor.

    A leaked conversion is recoverable; freeing the model-owned KV cache is not.
    """
    source = _FakeTensor(0xDEAD0000)
    opaque = _FakeTensor(0xBEEF0000, raises=True)
    assert _is_distinct_buffer(opaque, source) is False
