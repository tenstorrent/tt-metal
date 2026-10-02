# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for audit fixes (no device): the flow's token-major mask and pad block are keyed by B2,
and the codec rejects out-of-range codes before the device lookup."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flowmod
from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder


def _mask(fake, rows_pad):
    with patch.object(flowmod.ttnn, "from_torch", side_effect=lambda t, **kw: t):
        return flowmod.TtVoxtralFlow._tm_mask(fake, rows_pad)


def test_tm_mask_is_rebuilt_for_a_different_b2_with_the_same_padding():
    fake = SimpleNamespace(_sched={}, _tm_B2=8, device=None)
    m8 = _mask(fake, 32)  # 3*8 = 24 rows -> 32
    fake._tm_B2 = 10
    m10 = _mask(fake, 32)  # 3*10 = 30 rows -> 32: same padding, different owners
    assert not torch.equal(m8, m10)
    r = torch.arange(32)
    owner = torch.where(r < 30, r % 10, 10 + r)
    expect = torch.where(owner.reshape(-1, 1) == owner.reshape(1, -1), 0.0, -1e9).to(torch.bfloat16)
    assert torch.equal(m10.reshape(32, 32), expect)


def test_tm_pad_rows_are_rebuilt_for_a_different_b2_with_the_same_padding():
    """Sibling of the mask key: the zero block _trunk_tm appends must have rows_pad - 3*B2 rows for each B2."""
    made = []
    fake = SimpleNamespace(
        _sched={},
        _tm_rows=lambda B2: 32,
        _up=lambda t: made.append(t) or t,
        layers=[],
        _norm=lambda x, w: x,
        norm=None,
        proj={"acoustic_codebook_output": None},
    )
    with patch.object(
        flowmod.ttnn, "concat", side_effect=lambda ts, dim, memory_config=None: torch.cat(ts, dim=dim)
    ), patch.object(flowmod.ttnn, "linear", side_effect=lambda x, w, **kw: x), patch.object(
        flowmod.ttnn, "slice", side_effect=lambda x, a, b: x
    ):
        for B2 in (8, 10):  # 24 and 30 rows: both pad to 32
            p = [torch.zeros(1, B2, flowmod.FM_INPUT_DIM)] * 3
            assert flowmod.TtVoxtralFlow._trunk_tm(fake, *p, B2).shape[1] == 32
    assert [t.shape[1] for t in made] == [8, 2]


@pytest.mark.parametrize(
    "bad",
    [(0, -1, 0), (0, 8192, 0), (1, 0, -1), (1, 0, 21)],
)
def test_codec_rejects_out_of_range_codes(bad, expect_error):
    which, sem, ac = bad
    codes = torch.zeros(1, 37, 4, dtype=torch.long)
    codes[0, 0, 2] = sem
    codes[0, 5, 1] = ac
    with expect_error(ValueError, "codes must be in"):
        TtVoxtralCodecDecoder.check_codes(codes)


def test_codec_accepts_the_full_valid_range():
    codes = torch.zeros(1, 37, 3, dtype=torch.long)
    codes[0, 0] = torch.tensor([0, 4096, 8191])
    codes[0, 1:] = torch.tensor([0, 10, 20])
    TtVoxtralCodecDecoder.check_codes(codes)
