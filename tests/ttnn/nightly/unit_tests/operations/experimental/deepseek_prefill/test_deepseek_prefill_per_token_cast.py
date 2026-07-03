# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8 and per_token_cast_back.

Mirrors DeepEP's reference (deepseek-ai/DeepEP deep_ep/utils/math.py):
- per_token_cast_to_fp8: for each 128-element block of a token,
    scale = clamp(max(|x|), 1e-4) / 448, e4m3 = round(x / scale)
- per_token_cast_back: out = decode(e4m3) * scale
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import comp_pcc, assert_equal

pytestmark = pytest.mark.use_module_device


BLOCK_W = 128
E4M3_MAX = 448.0

SHAPES = [
    (1, 1024),  # single row (partial tile-row)
    (30, 1152),  # partial tile-row + 9 scale blocks
    (640, 7168),
    (3200, 7168),
    (6400, 7168),
    (2, 3, 30, 1152),  # 4D + partial tile-row (M = 180)
    (2, 40, 1024),  # multi-row-tile batch with a partial final row-tile (R=40 -> tiles of 32 + 8)
]

ROUNDTRIP_SHAPES = [(32, 1024), (30, 1152), (4, 1, 128, 1024)]

# Masked-decompress configs: (H, local expert_token_counts, dispatch_group_size). Region offsets are
# derived with the real dispatch rule (each count padded up to TILE_SIZE, cumulative), so the buffer is
# byte-faithful to a single-device dispatch. Each valid row's per-128-block scales ride in the tail of
# its metadata row, so no separate scale tensor is needed. dispatch_group_size only feeds the
# counter_offset derivation (0 on one chip). Covers small/large (multi compute-block) regions, an empty
# expert, varying blocks-per-row (H/128), and the real width (H=7168, blocks_per_row=56 > tile_h so a
# single token spans multiple compute blocks).
MASKED_CONFIGS = [
    (256, [5, 3], 1),  # 2 blocks/row, two small regions
    (128, [40, 10], 1),  # 1 block/row, first region 40 rows > tile_h (2 compute blocks)
    (384, [8, 0, 12], 1),  # 3 blocks/row, middle expert empty
    (256, [6, 4], 2),  # dispatch_group_size > 1 (counter_offset still 0 on one chip)
    (256, [12], 4),
    (7168, [2], 1),  # real width: blocks_per_row=56 > tile_h, each token spans 2 compute blocks
    (7168, [40, 3], 1),  # real width + a region > tile_h rows
    (7168, [200], 1),  # single large region: full-grid split lands many cores' slices mid-expert
]


# Blackhole only operation
@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


# ---------------------------------------------------------------------------
# Tensor creation helpers.
# ---------------------------------------------------------------------------


def _scale_shape(shape):
    """Expected scale shape for an input shape: leading dims preserved, last dim -> H / 128."""
    return tuple(shape[:-1]) + (shape[-1] // BLOCK_W,)


def _make_e4m3_from_torch(x_fp8_torch, *, device):
    """ttnn.from_torch with dtype=fp8_e4m3 requires float32 input. Cast first."""
    return ttnn.from_torch(
        x_fp8_torch.float(),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------
# Reference math (mirrors DeepEP's per_token_cast_to_fp8 / per_token_cast_back).
# ---------------------------------------------------------------------------


def _ref_scale(x_fp32):
    """Per-token reference scale [.., H/128] = clamp(amax over each 128-wide block, 1e-4) / 448."""
    *leading, H = x_fp32.shape
    blocks = x_fp32.reshape(*leading, H // BLOCK_W, BLOCK_W)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    return amax / E4M3_MAX


# ---------------------------------------------------------------------------
# Quality assertion helper.
# ---------------------------------------------------------------------------


def assert_quality(result, ref, *, pcc_threshold, rtol, atol, label=""):
    """Assert PCC >= pcc_threshold and torch.allclose(result, ref, rtol, atol), with logging."""
    passed_pcc, pcc = comp_pcc(result, ref, pcc_threshold)
    close = torch.allclose(result, ref, rtol=rtol, atol=atol)
    status = "PASS" if passed_pcc and close else "FAIL"
    logger.info(f"[{status}] {label}: pcc={pcc:.6f} (min={pcc_threshold}), allclose={close} (rtol={rtol}, atol={atol})")
    assert passed_pcc, f"{label}: PCC {pcc:.6f} below {pcc_threshold}"
    assert close, f"{label}: values not allclose (rtol={rtol}, atol={atol})"


# ---------------------------------------------------------------------------
# per_token_cast_to_fp8: scale and quantization value tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_cast_to_fp8_scale(device, dtype, layout):
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    M = 1024
    # In this test, we want to verify that cast_to_fp8 scale is lossless if 128-consecutives values are the same
    # and if they can be represented exactly in fp8.
    # Build one row with one exactly representable value per 128-wide block, then repeat it
    # vertically across M rows. These values also produce exactly representable power-of-two scales.
    block_values = torch.tensor([-448, -224, -112, -56, 56, 112, 224, 448], dtype=torch_dtype)
    x_row = block_values.repeat_interleave(BLOCK_W)
    x = x_row.repeat([M, 1])
    x_tt = ttnn.from_torch(x, dtype=ttnn_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref = _ref_scale(x.float())
    assert_equal(scale, ref)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_to_fp8_scale_values(device, dtype, shape, layout):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(*shape) * 5.0).to(torch_dtype)
    x_tt = ttnn.from_torch(x, dtype=ttnn_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref = _ref_scale(x.float())

    assert tuple(x_tt.shape) == shape
    assert tuple(scale_tt.shape) == _scale_shape(shape)
    assert tuple(output_e4m3_tt.shape) == shape
    assert output_e4m3_tt.dtype == ttnn.fp8_e4m3
    assert x_tt.dtype == ttnn_dtype
    assert scale_tt.dtype == ttnn.float32
    assert x_tt.layout == layout
    assert output_e4m3_tt.layout == ttnn.ROW_MAJOR_LAYOUT  # outputs are always ROW_MAJOR
    assert scale_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    max_rel = ((scale - ref).abs() / ref.abs().clamp_min(1e-9)).max().item()
    logger.info(f"scale {dtype} shape={shape}: max_rel={max_rel:.4f}")
    assert_quality(scale, ref, pcc_threshold=0.999, rtol=1e-2, atol=1e-9, label=f"scale {dtype} shape={shape}")


# ---------------------------------------------------------------------------
# per_token_cast_back: dequantization value tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_back_dequant(device, out_dtype, shape):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, out_dtype)
    ttnn_dtype = getattr(ttnn, out_dtype)

    input_e4m3 = (torch.randn(*shape) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    input_scale = (torch.rand(*_scale_shape(shape)) * 4.0 - 2.0).to(torch.float32)

    e4m3_tt = _make_e4m3_from_torch(input_e4m3, device=device)
    scale_tt = ttnn.from_torch(
        input_scale,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn_dtype)
    out = ttnn.to_torch(out_tt).float()

    golden = input_e4m3.float() * input_scale.repeat_interleave(BLOCK_W, dim=-1)
    if out_dtype == "bfloat16":
        golden = golden.to(torch_dtype).float()

    assert tuple(out_tt.shape) == shape
    assert out_tt.dtype == ttnn_dtype
    assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    # Restrict to normal e4m3 values where relative tolerance is meaningful.
    normal = input_e4m3.float().abs() > 2.0**-6
    assert_quality(
        out[normal],
        golden[normal],
        pcc_threshold=0.999,
        rtol=1e-2,
        atol=1e-3,
        label=f"dequant {out_dtype} shape={shape}",
    )


# ---------------------------------------------------------------------------
# Round-trip: per_token_cast_to_fp8 followed by per_token_cast_back.
# ---------------------------------------------------------------------------


# Only the forward op's input layout is parametrized; per_token_cast_back is ROW_MAJOR-only and
# always receives the forward op's ROW_MAJOR e4m3 / scale outputs.
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", ROUNDTRIP_SHAPES)
def test_round_trip_random(device, dtype, shape, layout):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(*shape) * 5.0).to(torch_dtype)
    x_in = x.float()
    x_tt = ttnn.from_torch(x, dtype=ttnn_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.float32)
    y = ttnn.to_torch(y_tt).float()

    # fp8 quantization (~12% worst-case relative error) bounds the reconstruction.
    assert_quality(y, x_in, pcc_threshold=0.999, rtol=0.1, atol=0.2, label=f"roundtrip {dtype} shape={shape}")


# ---------------------------------------------------------------------------
# Masked per_token_cast_back: decompress only the valid expert-region rows of a dispatch buffer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("H, counts, dispatch_group_size", MASKED_CONFIGS, ids=lambda v: str(v))
def test_cast_back_masked(device, output_dtype, H, counts, dispatch_group_size):
    """Single-device masked decompress from a hand-built dispatch buffer.

    input_e4m3 is a dispatch buffer of T rows; only the rows in each local expert region
    [offsets[e], offsets[e]+counts[e]) are decompressed. Each valid row's per-128-block fp32 scales
    ride in the tail of its metadata row (fields 5.., bit-cast to int32), matching the fp8 dispatch
    path, so no separate scale tensor is needed (input_scale is unused here — a dummy is passed).
    Offsets follow the real dispatch rule (counts padded up to TILE_SIZE, cumulative), so the layout
    matches a single-device dispatch. counter_offset stays 0 (it is mesh-position-derived and cannot
    be exercised on one chip). Only valid rows are asserted; padding/trailing garbage is left untouched.
    """
    torch.manual_seed(0)

    n_blocks = H // BLOCK_W  # scale columns / 128-blocks per row; also the metadata scale-tail length
    metadata_len = 5 + n_blocks
    G = dispatch_group_size
    experts_per_chip = len(counts)  # local experts on this device
    num_routed_experts = experts_per_chip * G  # num_dispatch_groups = 1
    ttnn_out_dtype = getattr(ttnn, output_dtype)

    # Local region offsets = cumulative sum of tile-padded counts (get_gate_outputs, single group).
    offsets = []
    acc = 0
    for c in counts:
        offsets.append(acc)
        acc += ((c + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    T = acc + 8  # + trailing garbage rows past the last region

    valid = [offsets[e] + i for e in range(experts_per_chip) for i in range(counts[e])]

    # Dispatch buffer values already on the e4m3 grid (decode is then exact), and each row's scales.
    raw_e4m3 = (torch.randn(T, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    scales = (torch.rand(T, n_blocks) * 2.0 + 0.5).to(torch.float32)

    # metadata head [coord, token, topk, expert, weight] is unread by the masked path; the tail
    # (fields 5..) carries the token's per-128-block fp32 scales, bit-cast to int32 (dispatch layout).
    metadata = torch.zeros((1, 1, T, metadata_len), dtype=torch.int32)
    for r in valid:
        metadata[0, 0, r, 5:] = scales[r].view(torch.int32)

    e4m3_tt = _make_e4m3_from_torch(raw_e4m3.reshape(1, 1, T, H), device=device)
    # input_scale is unused in masked mode but still a required positional; pass a minimal dummy.
    scale_dummy_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, n_blocks, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _to_int32(vals):
        return ttnn.from_torch(
            torch.tensor(vals, dtype=torch.int32).reshape(1, 1, 1, num_routed_experts),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # counts/offsets span all num_routed_experts; only this device's first experts_per_chip entries are
    # read (counter_offset == 0 on one chip), the rest are padding for the other dispatch-group devices.
    counts_tt = _to_int32(counts + [0] * (num_routed_experts - experts_per_chip))
    offsets_tt = _to_int32(offsets + [0] * (num_routed_experts - experts_per_chip))
    metadata_tt = ttnn.from_torch(
        metadata, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(
        e4m3_tt,
        scale_dummy_tt,
        output_dtype=ttnn_out_dtype,
        expert_token_counts=counts_tt,
        expert_region_offsets=offsets_tt,
        metadata=metadata_tt,
        experts_per_chip=experts_per_chip,
        dispatch_group_size=G,
    )
    out = ttnn.to_torch(out_tt).float().reshape(T, H)

    # Reference: each valid row = decode(e4m3) * its own per-128-block scale (from the metadata tail).
    ref_valid = torch.stack([raw_e4m3[r] * scales[r].repeat_interleave(BLOCK_W) for r in valid])
    out_valid = torch.stack([out[r] for r in valid])

    # PCC is the correctness gate: masking, the metadata-tail scale read, and the block layout are exact
    # (PCC == 1.0). The Tensix eltwise multiply carries reduced-mantissa (tf32/bf16) precision even with
    # an fp32 dest, leaving a small absolute floor; tiny |values| turn that into a large relative error,
    # so allclose needs an absolute floor in atol. A bf16 output adds its own rounding.
    rtol, atol = (6e-2, 1.5e-1) if output_dtype == "bfloat16" else (2e-2, 5e-2)
    assert_quality(
        out_valid, ref_valid, pcc_threshold=0.999, rtol=rtol, atol=atol, label=f"masked {output_dtype} H={H} G={G}"
    )
