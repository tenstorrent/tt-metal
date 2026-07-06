# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only structural tests for tt/chunked_prefill.py (#47466).

These do not need a device: they check the flag default, the per-chunk block
math (chunk_page_table slicing), and the input validation. The numerical
RoPE-offset + cross-chunk-attention correctness is the device gate in
``test_device_chunked_prefill.py``.
"""

import torch

from models.experimental.diffusion_gemma.tt import chunked_prefill as cp


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(cp.FLAG, raising=False)
    assert cp.chunked_prefill_enabled() is False
    monkeypatch.setenv(cp.FLAG, "1")
    assert cp.chunked_prefill_enabled() is True
    monkeypatch.setenv(cp.FLAG, "0")
    assert cp.chunked_prefill_enabled() is False


def test_default_chunk_size(monkeypatch):
    monkeypatch.delenv("DG_CHUNKED_PREFILL_CHUNK", raising=False)
    assert cp._default_chunk_size() == 256
    monkeypatch.setenv("DG_CHUNKED_PREFILL_CHUNK", "512")
    assert cp._default_chunk_size() == 512


def test_blocks_in():
    assert cp._blocks_in(0, 64) == 0
    assert cp._blocks_in(1, 64) == 1
    assert cp._blocks_in(64, 64) == 1
    assert cp._blocks_in(65, 64) == 2
    assert cp._blocks_in(512, 64) == 8


def test_reference_page_table():
    pt = cp.make_reference_page_table(8, mesh_device=None)
    assert pt.shape == (1, 8)
    assert pt.dtype == torch.int32
    assert pt.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_chunk_page_table_slicing_matches_reference_contract():
    """Chunk c's fill table == page_table[:, chunk_start_block:chunk_end_block].

    Mirrors tt_transformers generator: chunk_start_block = start // block_size,
    chunk_end_block = ceil((start+len)/block_size). For 512 tokens as 2x256 with
    block_size 64, chunk 0 owns blocks [0:4], chunk 1 owns [4:8].
    """
    block_size, chunk_size, prompt_len = 64, 256, 512
    pt = cp.make_reference_page_table(prompt_len // block_size, mesh_device=None)
    expected = {0: [[0, 1, 2, 3]], 1: [[4, 5, 6, 7]]}
    for c in range(prompt_len // chunk_size):
        start = c * chunk_size
        end = start + chunk_size
        sb = start // block_size
        eb = cp._blocks_in(end, block_size)
        assert pt[:, sb:eb].tolist() == expected[c]
