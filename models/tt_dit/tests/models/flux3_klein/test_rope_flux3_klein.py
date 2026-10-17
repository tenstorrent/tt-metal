# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only: ``rope_flux3_klein.rope_cos_sin`` is bit-equal to the M1 conversion of the reference ``rope_embed``.

The M1 block test builds its tables as ``emb = rope_embed(ids[None].float())``, ``cos = emb[..., 0, 0]``,
``sin = emb[..., 1, 0]``, each ``repeat_interleave(2, dim=-1)`` and zero-angle padded (``test_transformer_block_flux3_klein.py``
``rope_tables``); that conversion is reproduced here verbatim (``m1_rope_tables``) and is the oracle. The reference
package is imported only by this test, never by ``rope_flux3_klein.py`` (``packing.py`` docstring, the same rule).
"""

from __future__ import annotations

import pytest
import torch

from ....models.transformers.flux3_klein.rope_flux3_klein import (
    FLUX3_KLEIN_ROPE_AXES_DIM,
    FLUX3_KLEIN_ROPE_DIM,
    FLUX3_KLEIN_ROPE_THETA,
    rope_cos_sin,
)
from ....pipelines.flux3_klein.packing import image_ids, padded_length, reference_ids, text_ids
from ....reference.flux3_klein.dit import AXES_DIM, THETA, rope_embed


def m1_rope_tables(ids: torch.Tensor, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The M1 builder, copied: ``(cos, sin)`` ``[1, 1, padded, 128]`` fp32 with ``cos 1 / sin 0`` pad rows."""
    emb = rope_embed(ids[None].float())  # (1, 1, L, 64, 2, 2) with [[cos, -sin], [sin, cos]] per pair
    cos = emb[..., 0, 0].repeat_interleave(2, dim=-1)
    sin = emb[..., 1, 0].repeat_interleave(2, dim=-1)
    pad = padded - cos.shape[-2]
    assert pad >= 0
    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad), value=1.0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad), value=0.0)
    return cos, sin


def g_order_ids() -> torch.Tensor:
    """The single-stage ids of a 1-reference request: ``[image | image_cond | text]`` (``dit.py:230`` promotes the
    concatenation to fp32, as ``torch.cat`` of fp32 and int64 does)."""
    return torch.cat([image_ids(64, 64), reference_ids(64, 64, 0), text_ids(320).float()])


CASES = {
    "image_64x64": lambda: image_ids(64, 64),
    "image_56x70": lambda: image_ids(56, 70),
    **{f"reference_8x10_k{k}": (lambda k=k: reference_ids(8, 10, k)) for k in range(10)},
    "reference_64x64_k0": lambda: reference_ids(64, 64, 0),
    "text_4096_int64": lambda: text_ids(4096),
    "text_320_fp32": lambda: text_ids(320).float(),
    "g_order_1ref": g_order_ids,
}


def test_constants_match_reference() -> None:
    assert FLUX3_KLEIN_ROPE_THETA == THETA
    assert FLUX3_KLEIN_ROPE_AXES_DIM == tuple(AXES_DIM)
    assert FLUX3_KLEIN_ROPE_DIM == 128


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@pytest.mark.parametrize("sp", [1, 2, 4], ids=["sp1", "sp2", "sp4"])
def test_rope_cos_sin_equals_m1_builder(case: str, sp: int) -> None:
    ids = CASES[case]()
    length = ids.shape[0]
    padded = padded_length(length, sp)
    cos, sin = rope_cos_sin(ids, padded)
    cos_m1, sin_m1 = m1_rope_tables(ids, padded)
    assert cos.shape == sin.shape == (1, 1, padded, FLUX3_KLEIN_ROPE_DIM)
    assert cos.dtype == sin.dtype == torch.float32
    assert torch.equal(cos, cos_m1), f"{case}: cos differs from the M1 builder"
    assert torch.equal(sin, sin_m1), f"{case}: sin differs from the M1 builder"
    # Pad rows are the identity rotation; real rows are not all identity (the tables carry information).
    tail_cos, tail_sin = cos[0, 0, length:], sin[0, 0, length:]
    assert torch.equal(tail_cos, torch.ones_like(tail_cos))
    assert torch.equal(tail_sin, torch.zeros_like(tail_sin))
    assert cos[0, 0, :length].min() < 1.0 or length == 0
    # Channels 2p and 2p+1 share pair p's angle.
    assert torch.equal(cos[..., 0::2], cos[..., 1::2]) and torch.equal(sin[..., 0::2], sin[..., 1::2])


def test_int64_and_fp32_ids_agree() -> None:
    unscaled_grid = torch.stack(
        [
            torch.zeros(64, dtype=torch.long),
            torch.arange(64) // 8,
            torch.arange(64) % 8,
            -torch.ones(64, dtype=torch.long),
        ],
        -1,
    )
    for ids in (text_ids(4096), text_ids(80), unscaled_grid):
        assert ids.dtype == torch.int64
        a = rope_cos_sin(ids, padded_length(ids.shape[0], 4))
        b = rope_cos_sin(ids.float(), padded_length(ids.shape[0], 4))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_axis_layout_is_axis_major() -> None:
    """Axis ``a`` only moves channels ``[32a, 32a + 32)``: a text id (only axis 3 non-zero) leaves the first 96
    channels at the identity, an image id (axes 1 and 2) leaves channels 0..31 and 96..127 at the identity."""
    cos_t, sin_t = rope_cos_sin(text_ids(8), 8)
    assert torch.equal(cos_t[..., :96], torch.ones_like(cos_t[..., :96]))
    assert torch.equal(sin_t[..., :96], torch.zeros_like(sin_t[..., :96]))
    assert cos_t[0, 0, 1:, 96:].min() < 1.0
    cos_i, sin_i = rope_cos_sin(image_ids(2, 3), 6)
    for channels in (slice(0, 32), slice(96, 128)):
        assert torch.equal(cos_i[..., channels], torch.ones_like(cos_i[..., channels]))
        assert torch.equal(sin_i[..., channels], torch.zeros_like(sin_i[..., channels]))
    assert cos_i[0, 0, 1:, 32:96].min() < 1.0


def test_rejects_bad_inputs(expect_error) -> None:
    with expect_error(ValueError, "ids must be"):
        rope_cos_sin(torch.zeros(5, 3), 32)
    with expect_error(ValueError, "cannot pad"):
        rope_cos_sin(text_ids(80), 64)
