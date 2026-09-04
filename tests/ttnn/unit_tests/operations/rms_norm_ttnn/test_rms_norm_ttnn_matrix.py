# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extended coverage for rms_norm_ttnn — the corners the golden harness cannot
reach on this ttnn build, plus the operand x placement rectangle.

DO NOT DELETE.

WHY THIS FILE EXISTS.  Two groups of golden cells never reach the op in this
checkout, for reasons on the HARNESS side rather than the op's:

  * `eval/golden_tests/rms_norm_ttnn/program_config.py::sharded_config_for`
    reads `CoreRange.end_coord` / `.start_coord`; this ttnn build exposes
    `.start` / `.end`, so the stand-in raises AttributeError BEFORE the op is
    called.  That takes out the whole `program_config` loose group (10 cells)
    and 3 of test_validation.py's refusal tests.
  * `eval/metrics.py`'s comparison calls `torch.max()` on the readback, which
    raises on a 0-element tensor -- so the three zero-volume loose cells fail
    AFTER the op has returned a correctly-shaped, correctly-placed empty
    tensor (shape / dtype / layout are all checked first, and all pass).

Neither is an op defect and neither is this file's job to patch: the golden
suite is the specification.  What this file does is cover the same OP behaviour
with a local stand-in that mirrors the real config object field-for-field, so
the `program_config` contract (variant detection, the restatement checks,
`subblock_w` honoured, `inplace` identity, every refusal) and the zero-volume
contract are actually exercised somewhere.

The rest of the file is the operand x placement rectangle the acceptance test
does not carry: all six optional-operand combinations against all four
placements at both layouts (so the cross-core width combine, the HEIGHT-shard
local reduce and the ROW_MAJOR BAND scheme each see a residual and a bias), the
blocked (Wt, 32) per-channel form, the L1-tight wide geometries, and poisoned
tile padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import ttnn

from eval.sharding import auto_shard_config, shard_config
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

_ML = ttnn.TensorMemoryLayout
PLACEMENTS = [_ML.INTERLEAVED, _ML.HEIGHT_SHARDED, _ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED]
LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]

#: (weight, bias, residual), named as the golden suite's `gamma_mode` axis does.
PRESENCE = {
    "no_gamma": (False, False, False),
    "gamma": (True, False, False),
    "gamma_bias": (True, True, False),
    "bias": (False, True, False),
    "residual": (False, False, True),
    "gamma_bias_residual": (True, True, True),
}

# pcc / relative-rms gates, matching the golden suite's bf16 entry.
PCC = 0.995
RMS = 0.04


# ---------------------------------------------------------------------------
# the local `program_config` stand-in
# ---------------------------------------------------------------------------
#
# Field-for-field the same object the golden suite builds (two variants, the
# same names, types, defaults and keyword-only construction) -- only the shard
# bounding-box accessor differs, which is the whole reason this copy exists.
# The op reads the FIELDS off whatever object it is handed and never checks its
# type, so a config built here is the same call the real object would make.


@dataclass(kw_only=True)
class DefaultProgramConfig:
    legacy_reduction: bool = False
    legacy_rsqrt: bool = False
    use_welford: bool = False


@dataclass(kw_only=True)
class ShardedProgramConfig:
    compute_with_storage_grid_size: object
    subblock_w: int
    block_h: int
    block_w: int
    inplace: bool
    legacy_reduction: bool = False
    legacy_rsqrt: bool = False
    use_welford: bool = False


def sharded_config_for(tensor, **overrides):
    """The derivation the op performs when the argument is omitted: grid from the
    shard grid's bounding box, block_h / block_w from the shard shape in tiles,
    subblock_w 1, inplace off."""
    spec = tensor.memory_config().shard_spec
    assert spec is not None, "a sharded program_config needs a sharded input"
    bbox = spec.grid.bounding_box()
    config = ShardedProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(
            bbox.end.x - bbox.start.x + 1,
            bbox.end.y - bbox.start.y + 1,
        ),
        subblock_w=1,
        block_h=spec.shape[0] // 32,
        block_w=spec.shape[1] // 32,
        inplace=False,
    )
    for name, value in overrides.items():
        assert hasattr(config, name), f"no field {name!r}"
        setattr(config, name, value)
    return config


# ---------------------------------------------------------------------------
# shared runner
# ---------------------------------------------------------------------------


def _memory_config(shape, memory_layout, layout, dtype, device, shard=None):
    if memory_layout == _ML.INTERLEAVED:
        return ttnn.DRAM_MEMORY_CONFIG
    if shard is not None:
        return shard_config(shard[0], shard[1], memory_layout, layout=layout, dtype=dtype, device=device)
    return auto_shard_config(list(shape), memory_layout, layout=layout, dtype=dtype, device=device)


def _run(
    device,
    shape,
    *,
    mode="gamma",
    layout=ttnn.TILE_LAYOUT,
    memory_layout=_ML.INTERLEAVED,
    dtype=ttnn.bfloat16,
    operand_dtype=None,
    operand_layout=None,
    blocked_operand=False,
    poison=None,
    shard=None,
    program_config=None,
    epsilon=1e-12,
    expect_input_identity=False,
):
    """Build every operand `mode` names, dispatch, and gate on pcc + relative rms.

    Returns the ttnn output so identity contracts can be asserted on it.
    """
    has_weight, has_bias, has_residual = PRESENCE[mode]
    operand_dtype = operand_dtype or dtype
    operand_layout = operand_layout or layout
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    operand_torch_dtype = torch.float32 if operand_dtype == ttnn.float32 else torch.bfloat16
    width = shape[-1] if len(shape) else 1

    torch.manual_seed(0)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    mc = _memory_config(shape, memory_layout, layout, dtype, device, shard)
    ttnn_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device, memory_config=mc)

    def _per_channel(seed):
        torch.manual_seed(seed)
        values = torch.randn(width, dtype=torch.float32).to(operand_torch_dtype)
        if blocked_operand:
            # The (Wt, 32) ROW_MAJOR form: one tile column per row, trailing lanes
            # of the last row zero-padded.  Same values, a different tensor.
            wt = (width + 31) // 32
            padded = torch.zeros(wt * 32, dtype=torch.float32).to(operand_torch_dtype)
            padded[:width] = values
            staged = padded.reshape(wt, 32)
        else:
            staged = values.reshape(1, 1, 1, width)
        return values, ttnn.from_torch(staged, dtype=operand_dtype, layout=operand_layout, device=device)

    kwargs = {"epsilon": epsilon}
    torch_kwargs = {"epsilon": epsilon}
    if has_weight:
        torch_w, kwargs["weight"] = _per_channel(1)
        torch_kwargs["weight"] = torch_w
    if has_bias:
        torch_b, kwargs["bias"] = _per_channel(2)
        torch_kwargs["bias"] = torch_b
    if has_residual:
        torch.manual_seed(3)
        torch_r = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
        # The residual carries the input's placement EXACTLY, shard spec included.
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            torch_r, dtype=dtype, layout=layout, device=device, memory_config=ttnn_x.memory_config()
        )
        torch_kwargs["residual_input_tensor"] = torch_r

    if poison is not None and layout == ttnn.TILE_LAYOUT:
        # Fill the implicit tile padding with a loud value: `expected` is built
        # from the LOGICAL torch tensors and never sees padding, so an op that
        # folds padding into the reduction diverges hard instead of by
        # sqrt(W_padded/W) - 1, which for a wide row is under bf16's own step.
        ttnn_x = ttnn.fill_implicit_tile_padding(ttnn_x, poison)
        if has_residual:
            kwargs["residual_input_tensor"] = ttnn.fill_implicit_tile_padding(kwargs["residual_input_tensor"], poison)
        if operand_layout == ttnn.TILE_LAYOUT:
            for key in ("weight", "bias"):
                if key in kwargs:
                    kwargs[key] = ttnn.fill_implicit_tile_padding(kwargs[key], poison)

    if memory_layout != _ML.INTERLEAVED:
        # The common norm contract: a sharded input asks for a matching output.
        kwargs["memory_config"] = ttnn_x.memory_config()
    if program_config is not None:
        kwargs["program_config"] = program_config

    ttnn_out = rms_norm_ttnn(ttnn_x, **kwargs)

    if expect_input_identity:
        assert ttnn_out is ttnn_x, (
            "program_config.inplace=True must return the INPUT tensor object itself, not a new "
            "tensor holding the same values -- the caller reads their own tensor back"
        )
    assert list(ttnn_out.shape) == list(shape), f"shape {list(ttnn_out.shape)} != {list(shape)}"
    assert ttnn_out.layout == layout, f"layout {ttnn_out.layout} != {layout}"
    assert ttnn_out.dtype == dtype

    expected = torch_rms_norm_ttnn(torch_x, **torch_kwargs).float().flatten()
    actual = ttnn.to_torch(ttnn_out).float().flatten()
    if actual.numel() == 0:
        return ttnn_out
    assert torch.isfinite(actual).all(), "output carries Inf/NaN"
    if actual.numel() > 1:
        pcc = torch.corrcoef(torch.stack([actual, expected]))[0, 1].item()
        rms = ((actual - expected).pow(2).mean().sqrt() / expected.std()).item()
        assert pcc > PCC and rms < RMS, f"pcc={pcc:.6f} (>{PCC}) rms={rms:.4f} (<{RMS})"
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    return ttnn_out


# ---------------------------------------------------------------------------
# 1. operand presence x placement x layout
# ---------------------------------------------------------------------------
#
# The acceptance test sweeps presence x layout on interleaved DRAM only.  This
# adds the placements, which is where each presence value meets a DIFFERENT
# reduction: HEIGHT keeps it local, WIDTH/BLOCK gather partials to a group root
# and multicast the finalized stat back, and a ROW_MAJOR width/block shard takes
# the BAND scheme (staged out of the core's own L1 in the tensor's global tile
# frame).  A residual has to ride all three, and under a native scheme it must be
# consumed zero-copy rather than re-read over the NoC.


@pytest.mark.parametrize("mode", list(PRESENCE))
@pytest.mark.parametrize("memory_layout", PLACEMENTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_operands_across_placements(device, mode, memory_layout, layout):
    _run(device, (1, 1, 256, 512), mode=mode, layout=layout, memory_layout=memory_layout)


@pytest.mark.parametrize("mode", ["gamma_bias", "residual", "gamma_bias_residual"])
@pytest.mark.parametrize("memory_layout", PLACEMENTS)
@pytest.mark.parametrize("shape", [(1, 1, 32, 50), (1, 1, 47, 64), (1, 1, 17, 50)])
def test_operands_on_non_aligned_shapes(device, mode, memory_layout, shape):
    """Non-tile-aligned H and/or W with the operands present.

    The masked reduce and the H-tail interact with `cb_x_sum` (whose pad lanes
    are `x_pad + r_pad`) and with the per-channel operands' own tile padding, so
    this is where a dropped mask shows up.
    """
    _run(device, shape, mode=mode, memory_layout=memory_layout)


# ---------------------------------------------------------------------------
# 2. poisoned tile padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("memory_layout", PLACEMENTS)
@pytest.mark.parametrize("shape", [(1, 1, 32, 40), (1, 1, 32, 72), (1, 1, 40, 40), (1, 1, 224, 72)])
def test_poisoned_padding_with_every_operand(device, memory_layout, shape):
    """W small enough that one tile of padding is 10-38% of the row, and the
    padding filled with 1000.0.

    An op that folds padding into the reduction is then wrong by 6-27%, and an
    op that masks the VALUES but still divides by the padded width fails on the
    pad fraction alone.  Both the residual's padding and the per-channel
    operands' padding are poisoned, not just the input's.
    """
    _run(device, shape, mode="gamma_bias_residual", memory_layout=memory_layout, poison=1000.0)


# ---------------------------------------------------------------------------
# 3. the blocked (Wt, 32) ROW_MAJOR per-channel form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["gamma", "gamma_bias", "gamma_bias_residual"])
@pytest.mark.parametrize("memory_layout", PLACEMENTS)
@pytest.mark.parametrize("shape", [(1, 1, 32, 128), (1, 1, 32, 72), (1, 1, 256, 512), (1, 1, 32, 4096)])
def test_blocked_per_channel_form(device, mode, memory_layout, shape):
    """A ROW_MAJOR operand laid out one tile column per row.

    The same weights as the flat (1, 1, 1, W) vector and a different physical
    tensor -- its Wt pages are interleaved across DRAM banks, so the flat form's
    single wide read would fetch the wrong bytes.  W = 72 is the case where the
    two forms diverge most (the last row is 24/32 padding).
    """
    _run(
        device,
        shape,
        mode=mode,
        memory_layout=memory_layout,
        operand_layout=ttnn.ROW_MAJOR_LAYOUT,
        blocked_operand=True,
    )


# ---------------------------------------------------------------------------
# 4. the L1-tight geometries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 16384),  # one tile-row, 512 tiles: must split the width across cores
        (1, 1, 32, 7168),  # decode profile, DeepSeek-V3 width
        (1, 1, 64, 12288),  # two tile-rows on a 110-core grid
        (1, 1, 160, 11008),  # Llama FFN width
        (1, 1, 32, 4064),  # Wt = 127, prime -> the chunk-divisor cliff
    ],
)
def test_wide_rows_with_every_operand(device, shape):
    """Wide rows with all three operands present.

    Every activation crossing doubles with a residual, so these are the shapes
    where the L1 solve re-decides the regime (RESIDENT -> ROW_RESIDENT ->
    STREAM) and where the width split has to carry `cb_x_sum` as well.
    """
    _run(device, shape, mode="gamma_bias_residual")


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("memory_layout", [_ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED])
def test_widest_row_major_band_with_every_operand(device, dtype, memory_layout):
    """The widest ROW_MAJOR band in the suite, at both activation dtypes.

    fp32 here is the configuration that pushed the CB region past L1 before the
    band scheme got its staging fallback (D30): three activation staging rings
    plus cb_x_sum plus two per-channel staging rings, all at 4-byte elements on a
    25-tile band.  Kept as a live regression pin on that fallback.
    """
    _run(
        device,
        (128, 8192),
        mode="gamma_bias_residual",
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_layout=memory_layout,
    )


# ---------------------------------------------------------------------------
# 5. mixed per-channel formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operand_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("operand_layout", LAYOUTS)
def test_mixed_per_channel_format(device, operand_dtype, operand_layout):
    """bf16 activations with an fp32 / bf16 / bf8b weight AND bias.

    The two per-channel operands must share a LAYOUT but not a dtype, so each CB
    declares its own data format and the chain's reconfig fold must not elide the
    second switch.
    """
    if operand_dtype == ttnn.bfloat8_b and operand_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("a block-compressed format has no row-major physical form")
    _run(
        device,
        (1, 1, 64, 128),
        mode="gamma_bias",
        operand_dtype=operand_dtype,
        operand_layout=operand_layout,
    )


def test_weight_and_bias_at_different_dtypes(device):
    """weight fp32, bias bf16 -- independent formats on the same layout."""
    torch.manual_seed(0)
    shape, width = (1, 1, 64, 128), 128
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn(width, dtype=torch.float32)
    b = torch.randn(width, dtype=torch.float32).to(torch.bfloat16)
    out = ttnn.to_torch(
        rms_norm_ttnn(
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            epsilon=1e-12,
            weight=ttnn.from_torch(
                w.reshape(1, 1, 1, width), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
            ),
            bias=ttnn.from_torch(
                b.reshape(1, 1, 1, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            ),
        )
    ).float()
    expected = torch_rms_norm_ttnn(x, epsilon=1e-12, weight=w, bias=b).float()
    pcc = torch.corrcoef(torch.stack([out.flatten(), expected.flatten()]))[0, 1].item()
    assert pcc > PCC, f"pcc={pcc:.6f}"


# ---------------------------------------------------------------------------
# 6. degenerate ranks and volumes, WITH the operands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(), (64,), (50,), (2, 1, 1, 32, 64), (2, 1, 1, 47, 50)])
def test_degenerate_ranks_consume_every_operand(device, shape):
    """Rank 0, 1 and 5 must consume every operand, in ONE device program.

    Rank 0 is the interesting one: mean(t^2) over a one-element row is t^2, and
    the per-channel operands are 1-element vectors, so the scalar rides the
    ordinary path rather than a short-circuit.
    """
    _run(device, shape, mode="gamma_bias_residual", layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("shape", [(1, 1, 0, 64), (0, 64), (1, 1, 32, 0), (0,), (1, 0, 32, 64)])
def test_zero_volume_returns_an_empty_tensor(device, shape):
    """A zero-volume input returns a copy at the requested placement.

    Asserted on shape / layout / dtype / element count rather than on values --
    there are none.  (This is the contract the golden suite's three zero-volume
    cells check; its metric code cannot compare 0-element tensors on this build,
    so the assertion lives here.)
    """
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm_ttnn(ttnn_x)
    assert list(out.shape) == list(shape)
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.dtype == ttnn.bfloat16
    assert ttnn.to_torch(out).numel() == 0


def test_zero_volume_with_every_operand(device):
    """Zero volume is still ONE device program even with all three operands.

    Nothing is read, so nothing can be wrong numerically -- what is under test is
    that the degenerate program builds and dispatches with the operands bound.
    """
    shape = (1, 1, 0, 64)
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    w = ttnn.from_torch(
        torch.ones(1, 1, 1, 64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    r = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = rms_norm_ttnn(x, weight=w, bias=w, residual_input_tensor=r)
    assert list(out.shape) == list(shape)
    assert ttnn.to_torch(out).numel() == 0


# ---------------------------------------------------------------------------
# 7. program_config — consumed, honoured, refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [{}, {"legacy_reduction": True}, {"legacy_rsqrt": True}, {"legacy_reduction": True, "legacy_rsqrt": True}],
    ids=["plain", "legacy_reduction", "legacy_rsqrt", "both_legacy"],
)
@pytest.mark.parametrize("shape", [(1, 1, 384, 768), (1, 1, 96, 104)])
def test_default_program_config_variant(device, shape, overrides):
    """The DEFAULT variant against an interleaved input.

    `legacy_reduction` / `legacy_rsqrt` are algorithm switches this op is
    numerically correct at EITHER setting of, so a caller who flips one gets the
    same answer rather than a different algorithm.  (1, 1, 96, 104) puts the
    caller's config on the masked-reduce path rather than only the clean one.
    """
    _run(device, shape, program_config=DefaultProgramConfig(**overrides))


@pytest.mark.parametrize(
    "shape, memory_layout",
    [
        ((1, 1, 384, 768), _ML.HEIGHT_SHARDED),
        ((1, 1, 64, 1536), _ML.WIDTH_SHARDED),
        ((1, 1, 384, 768), _ML.BLOCK_SHARDED),
    ],
)
def test_sharded_program_config_variant(device, shape, memory_layout):
    """The SHARDED variant, derived from the input's own shard spec.

    The blocking the op is handed is the blocking its input already has, so this
    is the "omitted argument" derivation made explicit -- and it must build the
    same program the omitted case does.
    """
    mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    probe = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    _run(device, shape, memory_layout=memory_layout, program_config=sharded_config_for(probe))


@pytest.mark.parametrize("subblock_w", [1, 2, 4, 8])
def test_subblock_w_is_honoured(device, subblock_w):
    """`subblock_w` binds the pass-B DEST-lane block size.

    Over a PINNED shard geometry (block_w = 8, so 1/2/4/8 all divide it): a
    blocking the derivation would never have chosen, and the answer must be the
    same at every one of them.  Accepting the object and then blocking
    differently is worse than refusing it, because the caller cannot see it.
    """
    shape = (1, 1, 64, 1536)
    shard = ([64, 256], (6, 1))
    mc = shard_config(
        shard[0], shard[1], _ML.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    probe = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    _run(
        device,
        shape,
        memory_layout=_ML.WIDTH_SHARDED,
        shard=shard,
        program_config=sharded_config_for(probe, subblock_w=subblock_w),
    )


@pytest.mark.parametrize(
    "shape, memory_layout",
    [
        ((1, 1, 256, 1024), _ML.HEIGHT_SHARDED),
        ((1, 1, 64, 1024), _ML.WIDTH_SHARDED),
        ((1, 1, 256, 1024), _ML.BLOCK_SHARDED),
    ],
)
@pytest.mark.parametrize("mode", ["gamma", "gamma_bias_residual"])
def test_inplace_returns_the_input_tensor(device, shape, memory_layout, mode):
    """`inplace` is a contract about what SURVIVES the call.

    The caller reads their own tensor back afterwards, so the op must hand back
    that very object -- a copy holding the right values fails here, which is the
    only way to test an identity contract.  Exercised with the operands too,
    because the output CB then aliases the input buffer while pass B is reading
    it through the intervening scale and bias stages.
    """
    mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    probe = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    _run(
        device,
        shape,
        mode=mode,
        memory_layout=memory_layout,
        program_config=sharded_config_for(probe, inplace=True),
        expect_input_identity=True,
    )


# --- refusals --------------------------------------------------------------

SHARDED_SHAPE = (1, 1, 384, 768)


@pytest.fixture
def sharded_input(device):
    mc = auto_shard_config(
        list(SHARDED_SHAPE), _ML.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    return ttnn.from_torch(
        torch.zeros(SHARDED_SHAPE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )


@pytest.fixture
def interleaved_input(device):
    return ttnn.from_torch(
        torch.zeros((1, 1, 64, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def test_refuses_use_welford(interleaved_input, expect_error):
    """There is no Welford single-pass path; the flag is an error, not a request
    for a different algorithm."""
    with expect_error((ValueError, RuntimeError), "(?i)welford"):
        rms_norm_ttnn(interleaved_input, program_config=DefaultProgramConfig(use_welford=True))


def test_refuses_use_welford_on_the_sharded_variant(sharded_input, expect_error):
    with expect_error((ValueError, RuntimeError), "(?i)welford"):
        rms_norm_ttnn(
            sharded_input,
            memory_config=sharded_input.memory_config(),
            program_config=sharded_config_for(sharded_input, use_welford=True),
        )


@pytest.mark.parametrize("subblock_w", [0, -1])
def test_refuses_subblock_w_below_one(sharded_input, expect_error, subblock_w):
    """Zero is constructible and is used as a divisor, so it is refused before it
    reaches one."""
    with expect_error((ValueError, RuntimeError), "(?i)subblock_w"):
        rms_norm_ttnn(
            sharded_input,
            memory_config=sharded_input.memory_config(),
            program_config=sharded_config_for(sharded_input, subblock_w=subblock_w),
        )


def test_refuses_subblock_w_that_does_not_divide_block_w(sharded_input, expect_error):
    with expect_error((ValueError, RuntimeError), "(?i)subblock_w"):
        rms_norm_ttnn(
            sharded_input,
            memory_config=sharded_input.memory_config(),
            program_config=sharded_config_for(sharded_input, subblock_w=5),
        )


def test_refuses_subblock_w_above_the_dest_capacity(sharded_input, expect_error):
    """The DEST limit is the caller's own `fp32_dest_acc_en`, so the refusal names
    both operands of the constraint."""
    with expect_error((ValueError, RuntimeError), "(?i)subblock_w|dest"):
        rms_norm_ttnn(
            sharded_input,
            memory_config=sharded_input.memory_config(),
            program_config=sharded_config_for(sharded_input, subblock_w=24),
        )


@pytest.mark.parametrize("field", ["block_h", "block_w"])
def test_refuses_a_wrong_block_restatement(sharded_input, expect_error, field):
    """`block_h` / `block_w` restate the input's shard geometry.  No value of them
    can express a blocking the shard does not already fix, so a wrong restatement
    is refused rather than acted on."""
    with expect_error((ValueError, RuntimeError), f"(?i){field}"):
        rms_norm_ttnn(
            sharded_input,
            memory_config=sharded_input.memory_config(),
            program_config=sharded_config_for(sharded_input, **{field: 3}),
        )


def test_refuses_default_config_against_a_sharded_input(sharded_input, expect_error):
    with expect_error((ValueError, RuntimeError), "(?i)variant|placement|sharded"):
        rms_norm_ttnn(
            sharded_input,
            memory_config=sharded_input.memory_config(),
            program_config=DefaultProgramConfig(),
        )


def test_refuses_sharded_config_against_an_interleaved_input(sharded_input, interleaved_input, expect_error):
    """The config is derived from a genuinely sharded tensor and then handed to an
    interleaved call, so the variant is well-formed and only the pairing is
    wrong -- the refusal cannot be an accident of construction."""
    with expect_error((ValueError, RuntimeError), "(?i)variant|placement|interleaved"):
        rms_norm_ttnn(interleaved_input, program_config=sharded_config_for(sharded_input))


def test_refuses_memory_config_disagreeing_under_inplace(sharded_input, expect_error):
    """Under `inplace` the output IS the input, so a placement the op would have
    to discard is refused rather than accepted."""
    with expect_error((ValueError, RuntimeError), "(?i)inplace|memory_config|placement"):
        rms_norm_ttnn(
            sharded_input,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sharded_config_for(sharded_input, inplace=True),
        )


def test_grid_size_inside_the_range_changes_nothing(device, sharded_input):
    """X-02: `compute_with_storage_grid_size` is accepted and range-checked, and it
    is NOT a placement contract -- placement follows the input's shard spec, so
    every in-range value must produce the same answer."""
    reference = None
    for grid in (ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 3)):
        out = ttnn.to_torch(
            rms_norm_ttnn(
                sharded_input,
                memory_config=sharded_input.memory_config(),
                program_config=sharded_config_for(sharded_input, compute_with_storage_grid_size=grid),
            )
        ).float()
        if reference is None:
            reference = out
        else:
            torch.testing.assert_close(out, reference, rtol=0, atol=0)


def test_refuses_grid_size_outside_the_device_grid(device, sharded_input, expect_error):
    grid = device.compute_with_storage_grid_size()
    with expect_error((ValueError, RuntimeError), "(?i)grid"):
        rms_norm_ttnn(
            sharded_input,
            memory_config=sharded_input.memory_config(),
            program_config=sharded_config_for(
                sharded_input, compute_with_storage_grid_size=ttnn.CoreCoord(grid.x + 1, grid.y)
            ),
        )


# ---------------------------------------------------------------------------
# 8. output placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["gamma", "gamma_bias_residual"])
def test_sharded_input_interleaved_output(device, mode):
    """A sharded input may produce an interleaved output, and that pairing must
    NOT be refused -- the seed already serves it."""
    shape = (1, 1, 256, 512)
    torch.manual_seed(0)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    mc = auto_shard_config(list(shape), _ML.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    has_weight, has_bias, has_residual = PRESENCE[mode]
    kwargs = {"epsilon": 1e-12, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
    torch_kwargs = {"epsilon": 1e-12}
    width = shape[-1]
    if has_weight:
        torch.manual_seed(1)
        w = torch.randn(width, dtype=torch.float32).to(torch.bfloat16)
        kwargs["weight"] = ttnn.from_torch(
            w.reshape(1, 1, 1, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        torch_kwargs["weight"] = w
    if has_bias:
        torch.manual_seed(2)
        b = torch.randn(width, dtype=torch.float32).to(torch.bfloat16)
        kwargs["bias"] = ttnn.from_torch(
            b.reshape(1, 1, 1, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        torch_kwargs["bias"] = b
    if has_residual:
        torch.manual_seed(3)
        r = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn_x.memory_config()
        )
        torch_kwargs["residual_input_tensor"] = r

    out = rms_norm_ttnn(ttnn_x, **kwargs)
    assert out.memory_config().memory_layout == _ML.INTERLEAVED
    expected = torch_rms_norm_ttnn(torch_x, **torch_kwargs).float().flatten()
    actual = ttnn.to_torch(out).float().flatten()
    pcc = torch.corrcoef(torch.stack([actual, expected]))[0, 1].item()
    assert pcc > PCC, f"pcc={pcc:.6f}"
