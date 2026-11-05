# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Token packing plan and layout (host only, no device)."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.chronos_forecast.tt.model import PACK_MAX_SEQ, TtChronos


def _plan(batch, seq_len, *, pack_tokens=None, chips=1):
    # l1_chunk_tokens=None skips the device-grid blocking check, so no device is needed.
    host = SimpleNamespace(pack_tokens=pack_tokens, num_devices=chips, l1_chunk_tokens=None)
    return TtChronos._pack_plan(host, batch, seq_len)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_small_batch_stays_unpacked(batch):
    assert _plan(batch, 37) is None


def test_pack_tokens_false_disables_packing():
    assert _plan(1024, 37, pack_tokens=False) is None


def test_paper_shape_stays_unpacked():
    assert _plan(1024, 133) is None  # ctx2048/h64: S=133 saves < 25%


@pytest.mark.parametrize("chips", [1, 4])
def test_ctx512_h64_packs(chips):
    k, sp, seqs = _plan(1024, 37, chips=chips)  # 32 context + 4 future + REG
    assert k >= 2 and sp % 32 == 0 and k * 37 <= sp <= PACK_MAX_SEQ
    assert seqs * k >= 1024 and seqs % chips == 0


@pytest.mark.parametrize("k,num_context,tail_len,num_reg", [(6, 32, 5, 1), (3, 32, 3, 1), (2, 4, 2, 1)])
def test_packed_layout(k, num_context, tail_len, num_reg):
    seq_len = num_context + tail_len
    sp = -(-k * seq_len // 32) * 32
    slot, token, segment, position = TtChronos._packed_layout(k, num_context, tail_len, num_reg, sp)
    assert slot.numel() == token.numel() == k * seq_len
    num_future = tail_len - num_reg
    unpacked = torch.cat(
        [
            torch.arange(num_context),
            torch.arange(num_context + num_reg, num_context + num_reg + num_future),
            torch.arange(num_context, num_context + num_reg),
        ]
    )
    for j in range(k):
        rows = (slot == j).nonzero().flatten()
        assert sorted(token[rows].tolist()) == list(range(seq_len))  # every token of slot j exactly once
        assert torch.equal(position[rows], unpacked[token[rows]])  # RoPE restarts per series
    assert bool((segment[k * seq_len :] == k).all())  # pad rows form their own segment
    same = segment.unsqueeze(0) == segment.unsqueeze(1)
    assert bool(same.any(-1).all())  # no all -inf mask row
    assert bool((slot[: k * tail_len] == torch.arange(k).repeat_interleave(tail_len)).all())  # tails first
