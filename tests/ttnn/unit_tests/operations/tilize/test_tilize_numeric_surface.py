# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — the numeric surface: integer dtype family, rank 0, and the two
padding EXCLUSIONS that are now supported instead of refused.

DO NOT DELETE — `lever_ledger.json` names `test_uint8_requires_fp32_dest` by
`assert_test`, and these are the pins that keep the three Refinement-4 claims
from quietly regressing:

  * an integer dtype is EXACT (tilize does no arithmetic — never assert PCC here);
  * `uint8` needs fp32 DEST, and the failure mode when it does not get it is a
    ZERO tile, not a noisy one (so an exactness assert is the only detector);
  * the writer's OUTPUT-format pad stamp fires ONLY on a cast that loses the fill,
    so every other cell's kernel is byte-identical to Refinement 3's.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


_TORCH_OF = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint8: torch.uint8,
    ttnn.uint16: torch.int32,
    ttnn.uint32: torch.int32,
    ttnn.int32: torch.int32,
}

_INT_DTYPES = [
    pytest.param(ttnn.uint32, id="uint32"),
    pytest.param(ttnn.uint16, id="uint16"),
    pytest.param(ttnn.int32, id="int32"),
    pytest.param(ttnn.uint8, id="uint8"),
]


def _make_input(dtype, shape):
    if dtype == ttnn.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if dtype == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype in (ttnn.uint16, ttnn.uint32):
        return torch.randint(0, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


def _tilize_and_read(torch_input, dtype, *, device, out_dtype=None, padded=None, pad_value=None, use_multicore=True):
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kwargs = {}
    if pad_value is not None:
        kwargs["pad_value"] = pad_value
    if padded is not None:
        kwargs["output_padded_shape"] = padded
    out = tilize(
        tt_input,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=out_dtype if out_dtype is not None else dtype,
        use_multicore=use_multicore,
        **kwargs,
    )
    if padded is None and pad_value is None:
        return ttnn.to_torch(out)
    return out.cpu().to_torch_with_padded_shape()


def _pad_reference(torch_input, padded, pad_value, compare_dtype):
    x = torch_input.to(compare_dtype)
    if x.dim() < len(padded):
        x = x.reshape((1,) * (len(padded) - x.dim()) + tuple(x.shape))
    pads = tuple(j for i in reversed(range(x.dim())) for j in (0, padded[i] - x.shape[i]))
    return F.pad(x, pads, value=pad_value)


# ---------------------------------------------------------------------------
# 1. The integer dtype family — EXACT, never PCC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _INT_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [[1, 1, 32, 64], [1, 1, 64, 128], [2, 32, 64], [1, 1, 2048, 64]],
    ids=["one_block", "multi_block", "rank3", "tall_narrow"],
)
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_integer_identity_is_exact(dtype, shape, use_multicore, device):
    """An integer datum is a WIDTH, not a value change: tilize must be a byte
    permutation, so compare exactly. `uint8`'s historical failure is a STRIDED
    tile (every other row zero) — shape-correct and value-wrong, which a PCC
    threshold would happily pass."""
    torch_input = _make_input(dtype, shape)
    got = _tilize_and_read(torch_input, dtype, device=device, use_multicore=use_multicore)
    assert torch.equal(got.to(_TORCH_OF[dtype]), torch_input.to(_TORCH_OF[dtype]))


@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_integer_padded_identity_is_exact(dtype, device):
    """The same exactness through the R_PAD reader, including whole pad tiles.

    A 50-element row is 50 B for `uint8` — a non-word-aligned fill offset, which
    is what `fill_l1_with_val`'s head/tail element stores exist for.
    """
    torch_input = _make_input(dtype, [1, 1, 50, 50])
    padded = [1, 1, 128, 128]
    pad_value = 7 if dtype != ttnn.int32 else -7
    got = _tilize_and_read(torch_input, dtype, device=device, padded=padded, pad_value=pad_value)
    expected = _pad_reference(torch_input, padded, pad_value, _TORCH_OF[dtype])
    assert torch.equal(got.to(_TORCH_OF[dtype]), expected)


def test_uint8_requires_fp32_dest(device):
    """Ledger pin (F25's 8-bit arm): fp32 DEST is a CORRECTNESS requirement for a
    1-byte datum, not a precision preference.

    The tilize LLK's 8-bit path (`IS_8BIT_FORMAT`) is only validated with DEST
    accumulation enabled; with a 16-bit DEST the packed tile comes out ZERO. That
    is why this knob is NOT a measurable perf lever with a keepable off arm — the
    off arm has no correct output. Pin both the config decision and the dtype gate
    (`bfloat8_b` also reports 1 byte per element and must NOT be treated as an
    8-bit datum).
    """
    shape = [1, 1, 64, 128]
    torch_input = _make_input(ttnn.uint8, shape)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.uint8, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = pd  # keep the module reference explicit for the reader
    from ttnn.operations.tilize.tilize import validate

    descriptor = plan.create_program_descriptor(tt_input, out, validate(tt_input, ttnn.DRAM_MEMORY_CONFIG))
    assert descriptor.kernels[2].config.fp32_dest_acc_en, "uint8 must run with fp32 DEST"

    assert ttnn.uint8 in pd.EIGHT_BIT_DTYPES
    assert ttnn.bfloat8_b not in pd.EIGHT_BIT_DTYPES  # block float, not an 8-bit datum
    assert ttnn.bfloat8_b in pd.BLOCK_FLOAT_DTYPES


# ---------------------------------------------------------------------------
# 2. rank 0 — the scalar, reachable only through the pad path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.uint8], ids=["bf16", "fp32", "uint32", "uint8"]
)
def test_rank0_scalar_pads_to_one_tile(dtype, device):
    """A rank-0 input has no tile dims of its own: the pad target synthesizes them.
    The data region is the single value; every other position is exactly the fill."""
    torch_input = _make_input(dtype, [])
    pad_value = 42 if dtype in (ttnn.uint8, ttnn.uint32) else 42.0
    got = _tilize_and_read(torch_input, dtype, device=device, padded=[32, 32], pad_value=pad_value)
    compare = _TORCH_OF[dtype]
    assert list(got.shape) == [32, 32]
    expected = _pad_reference(torch_input, [32, 32], pad_value, compare)
    assert torch.equal(got.to(compare), expected)


def test_rank0_auto_pad_synthesizes_the_tile_dims(device):
    """`pad_mode="auto"` on a scalar rounds nothing — it SYNTHESIZES [32, 32], the
    same promotion `_expand_rank` performs for the kernels' geometry view."""
    torch_input = _make_input(ttnn.bfloat16, [])
    got = _tilize_and_read(torch_input, ttnn.bfloat16, device=device, pad_value=-3.5)
    assert list(got.shape) == [32, 32]
    assert torch.equal(got.to(torch.bfloat16), _pad_reference(torch_input, [32, 32], -3.5, torch.bfloat16))


# ---------------------------------------------------------------------------
# 3. The two ex-EXCLUSIONS
# ---------------------------------------------------------------------------


def test_exclusions_is_empty():
    """Both Phase-0 EXCLUSIONS are gone; only the Refinement-1 sharded-vs-single-core
    rows remain, and they are placement, not numeric-surface."""
    from ttnn.operations.tilize import EXCLUSIONS

    # A dtype/pad EXCLUSION keyed ONLY on the numeric surface is what Refinement 4
    # removed. Refinement 5 added one keyed on the TILE GEOMETRY as well
    # (tile_height=16 x a block-float output — a platform pack gap, see the op
    # file), which is a tile-geometry refusal that happens to name a dtype, not a
    # numeric-surface one: at the default 32-row tile every dtype is supported.
    numeric = [
        e
        for e in EXCLUSIONS
        if ({"dtype", "output_dtype", "pad_mode", "pad_value"} & set(e)) and "tile_height" not in e
    ]
    assert numeric == [], f"a numeric-surface EXCLUSION came back: {numeric}"


@pytest.mark.parametrize("pad_value", [0.0, 10.2, -18.0], ids=["zero", "pos", "neg"])
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16_in", "fp32_in"])
def test_padded_bfloat8_b_output(in_dtype, pad_value, device):
    """Ex-EXCLUSION 1: padding into a block-float output.

    Nothing had to be built — the fill is materialized in the INPUT CB and the
    packer builds the shared exponent over pad and data alike. bf8b is lossy at
    pack time, so this is the one cell in this file that is a PCC check.
    """
    torch_input = _make_input(in_dtype, [1, 1, 50, 50])
    padded = [1, 1, 64, 64]
    got = _tilize_and_read(
        torch_input, in_dtype, device=device, out_dtype=ttnn.bfloat8_b, padded=padded, pad_value=pad_value
    )
    expected = _pad_reference(torch_input, padded, pad_value, torch.bfloat16)
    a, b = got.flatten().float(), expected.flatten().float()
    pcc = 1.0 if torch.equal(a, b) else float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    assert pcc > 0.99, f"bf8b padded PCC {pcc}"


@pytest.mark.parametrize("pad_value", [10.2, -18.3, 3.5], ids=["pos", "neg", "half"])
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_widening_cast_pad_is_exact(pad_value, use_multicore, device):
    """Ex-EXCLUSION 2: bf16 -> fp32 with a fill bf16 cannot hold.

    The reader's fill is packed in the INPUT format (a hard contract), so without
    the writer's second, OUTPUT-format stamp the pad region would arrive
    bf16-rounded (10.2 -> 10.1875) in an fp32 tensor. Exact is the bar.
    """
    torch_input = _make_input(ttnn.bfloat16, [1, 1, 50, 50])
    padded = [1, 1, 128, 128]  # W tail + H tail + whole pad tiles, all three
    got = _tilize_and_read(
        torch_input,
        ttnn.bfloat16,
        device=device,
        out_dtype=ttnn.float32,
        padded=padded,
        pad_value=pad_value,
        use_multicore=use_multicore,
    )
    expected = _pad_reference(torch_input, padded, pad_value, torch.float32)
    assert torch.equal(got, expected)


@pytest.mark.parametrize(
    "in_dtype, out_dtype, pad_value, expected",
    [
        # No cast: the input-format fill IS the output-format fill.
        (ttnn.bfloat16, ttnn.bfloat16, 10.2, False),
        (ttnn.float32, ttnn.float32, 10.2, False),
        (ttnn.uint32, ttnn.uint32, 7, False),
        # Widening cast, fill EXACT in the input format -> still nothing to fix.
        (ttnn.bfloat16, ttnn.float32, 0.0, False),
        (ttnn.bfloat16, ttnn.float32, -18.0, False),
        # Widening cast, fill INEXACT in bf16 -> the stamp is the only way to be exact.
        (ttnn.bfloat16, ttnn.float32, 10.2, True),
        (ttnn.bfloat16, ttnn.float32, 3.3, True),
        # Narrowing cast: the packer's own truncation already lands the right value.
        (ttnn.float32, ttnn.bfloat16, 10.2, False),
        # Block float: an element word cannot be stamped into a shared-exponent tile.
        (ttnn.bfloat16, ttnn.bfloat8_b, 10.2, False),
    ],
)
def test_out_fill_gate_fires_only_when_the_round_trip_loses_the_fill(in_dtype, out_dtype, pad_value, expected):
    """The gate is what keeps every other cell byte-identical to Refinement 3: the
    stamp costs L1 stores, so it must fire exactly when — and only when — the
    input-format fill cannot reproduce the output-format value."""
    assert pd.needs_output_format_fill(pad_value, in_dtype, out_dtype) is expected


def test_out_fill_is_off_on_the_unpadded_hot_path(device):
    """No pad region -> no stamp, whatever the cast. The `out_fill` compile-time arg
    is index 11 of the writer's args."""
    shape = [1, 1, 64, 128]
    torch_input = _make_input(ttnn.bfloat16, shape)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    from ttnn.operations.tilize.tilize import validate

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = validate(tt_input, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    descriptor = pd.create_program_descriptor(tt_input, out, plan)
    assert descriptor.kernels[1].compile_time_args[11] == 0
