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
]

ROUNDTRIP_SHAPES = [(32, 1024), (30, 1152), (4, 1, 128, 1024)]


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


def _ref_power_of_two_scale(x_fp32):
    """Sparse MLA KV scale: 2^ceil(log2(clamp(amax, 1e-4) / E4M3_MAX))."""
    *leading, H = x_fp32.shape
    blocks = x_fp32.reshape(*leading, H // BLOCK_W, BLOCK_W)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    return torch.pow(2.0, torch.ceil(torch.log2(amax / E4M3_MAX)))


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


@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_cast_to_fp8_scale(device, dtype, input_layout):
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
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref = _ref_scale(x.float())
    assert_equal(scale, ref)


@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_to_fp8_scale_values(device, dtype, shape, input_layout):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(*shape) * 5.0).to(torch_dtype)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    scale = ttnn.to_torch(scale_tt).float()

    ref = _ref_scale(x.float())

    assert tuple(x_tt.shape) == shape
    assert tuple(scale_tt.shape) == _scale_shape(shape)
    assert tuple(output_e4m3_tt.shape) == shape
    assert output_e4m3_tt.dtype == ttnn.fp8_e4m3
    assert x_tt.dtype == ttnn_dtype
    assert scale_tt.dtype == ttnn.float32
    assert x_tt.layout == input_layout
    assert output_e4m3_tt.layout == ttnn.ROW_MAJOR_LAYOUT  # outputs are always ROW_MAJOR
    assert scale_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    max_rel = ((scale - ref).abs() / ref.abs().clamp_min(1e-9)).max().item()
    logger.info(f"scale {dtype} shape={shape}: max_rel={max_rel:.4f}")
    assert_quality(scale, ref, pcc_threshold=0.999, rtol=1e-2, atol=1e-9, label=f"scale {dtype} shape={shape}")


@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", [(1, 512), (30, 512), (2, 3, 32, 512)])
def test_cast_to_fp8_power_of_two_scale_for_sparse_kv(device, dtype, shape, input_layout):
    """Opt-in sparse-KV mode keeps the existing op contract but emits TT-safe UE8M0-style scales."""
    torch.manual_seed(23)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    x = (torch.randn(*shape) * 0.01).to(torch_dtype)
    # Give the four 128-wide blocks distinct dynamic ranges.
    x = x * torch.tensor([1.0, 8.0, 64.0, 512.0], dtype=torch_dtype).repeat_interleave(BLOCK_W)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt, round_scale_to_power_of_two=True)
    scale = ttnn.to_torch(scale_tt).float()
    ref_scale = _ref_power_of_two_scale(x.float())

    assert tuple(e4m3_tt.shape) == shape
    assert tuple(scale_tt.shape) == _scale_shape(shape)
    # The device row-max reduction can quantize an FP32 amax at a power-of-two
    # boundary. Its rounded scale must remain the same or an adjacent power.
    scale_ratio = scale / ref_scale
    assert torch.all((scale_ratio == 0.5) | (scale_ratio == 1.0) | (scale_ratio == 2.0)), (
        f"Expected scale ratio to be 0.5, 1.0, or 2.0; got ratios={scale_ratio.tolist()}, "
        f"scales={scale.tolist()}, reference_scales={ref_scale.tolist()}"
    )
    assert torch.all(
        torch.log2(scale) == torch.round(torch.log2(scale))
    ), f"Expected power-of-two scales; got scales={scale.tolist()}"

    # Blackhole's truncating packer reserves normalized magnitudes >= 480 for
    # 0x7F. Check the safety property against the unquantized host amax.
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // BLOCK_W, BLOCK_W)
    normalized_amax = blocks.abs().amax(dim=-1) / scale
    assert torch.all(normalized_amax < 480.0), (
        f"Normalized amax must be below 480; got max={normalized_amax.max().item()}, "
        f"values={normalized_amax.tolist()}"
    )

    y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.float32)
    y = ttnn.to_torch(y_tt).float()
    assert_quality(y, x.float(), pcc_threshold=0.999, rtol=0.15, atol=1e-3, label=f"sparse KV {dtype} {shape}")


def test_cast_to_fp8_power_of_two_scale_e4m3fn_boundary(device):
    """Exponent-15 E4M3FN values through 448 are finite and must not increase the scale."""
    block_maxima = torch.tensor([240.0, 256.0, 448.0, 449.0], dtype=torch.float32)
    x = torch.zeros(1, 4 * BLOCK_W, dtype=torch.float32)
    x[0, ::BLOCK_W] = block_maxima
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt, round_scale_to_power_of_two=True)

    scale = ttnn.to_torch(scale_tt).float().reshape(-1)
    expected_scale = torch.tensor([1.0, 1.0, 1.0, 2.0])
    assert torch.equal(
        scale, expected_scale
    ), f"Unexpected boundary scales: got {scale.tolist()}, expected {expected_scale.tolist()}"

    y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.float32)
    y = ttnn.to_torch(y_tt).float()
    # The hardware packer truncates instead of rounding to nearest. Values can
    # therefore land one E4M3FN ULP below the mathematical result.
    expected_output = torch.tensor([224.0, 240.0, 416.0, 448.0])
    actual_output = y[0, ::BLOCK_W]
    assert torch.equal(
        actual_output, expected_output
    ), f"Unexpected E4M3FN boundary outputs: got {actual_output.tolist()}, expected {expected_output.tolist()}"


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


# Output layout is always ROW_MAJOR.
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", ROUNDTRIP_SHAPES)
def test_round_trip_random(device, dtype, shape, input_layout):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    x = (torch.randn(*shape) * 5.0).to(torch_dtype)
    x_in = x.float()
    x_tt = ttnn.from_torch(
        x, dtype=ttnn_dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    e4m3_tt, scale_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)
    y_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.float32)
    y = ttnn.to_torch(y_tt).float()

    # fp8 quantization (~12% worst-case relative error) bounds the reconstruction.
    assert_quality(y, x_in, pcc_threshold=0.999, rtol=0.1, atol=0.2, label=f"roundtrip {dtype} shape={shape}")


MASKED_CASES = [
    ("dense", [130, 74, 200, 96, 41]),
    ("zeros_middle", [130, 0, 0, 74, 200]),
    ("zeros_leading", [0, 0, 130, 74, 200]),
    ("zeros_trailing", [130, 74, 200, 0, 0]),
]


def _ceil_tile(n):
    return ((n + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def create_u32_tensor(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


MAX_DISPATCH_BUFFER_TOKENS = 5 * 1024 * 8

# Metadata scale path: the dispatch metadata row is [METADATA_HEADER routing ints][H/128 fp32-bit scales].
METADATA_HEADER = 5


def _pack_scale_metadata(input_scale):
    """Bit-store fp32 per-token scales in the tail of an int32 dispatch-metadata row (the metadata scale
    path). Leading header columns are filled with a sentinel the kernel must ignore."""
    M, blocks = input_scale.shape
    meta = torch.full((M, METADATA_HEADER + blocks), 0x0BADF00D, dtype=torch.int32)
    meta[:, METADATA_HEADER:] = input_scale.contiguous().view(torch.int32)
    return meta


@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scales_from_metadata", [False, True])
@pytest.mark.parametrize("bf16_scale", [False, True])
@pytest.mark.parametrize("label, counts", MASKED_CASES, ids=[c[0] for c in MASKED_CASES])
def test_masked_cast_back(device, label, counts, bf16_scale, scales_from_metadata, output_dtype):
    torch.manual_seed(0)
    H = 1024

    experts_per_chip = len(counts)
    # This chip owns non-contiguous global ids (odd slots) out of a wider routed-expert space.
    num_routed_experts = 2 * experts_per_chip
    global_expert_idx_table = [2 * s + 1 for s in range(experts_per_chip)]

    # Packed region layout for this chip's experts; other global ids stay zero (never read).
    expert_region_offsets = [0] * num_routed_experts
    expert_token_counts = [0] * num_routed_experts
    running_offset = 0
    for local_slot, token_count in enumerate(counts):
        global_id = global_expert_idx_table[local_slot]
        expert_region_offsets[global_id] = running_offset
        expert_token_counts[global_id] = token_count
        running_offset += _ceil_tile(token_count)
    total_valid_rows = running_offset
    capacity = MAX_DISPATCH_BUFFER_TOKENS  # fixed flat buffer; [total_valid_rows, capacity) is untouched tail

    input_e4m3 = (torch.randn(capacity, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    input_scale = torch.rand(capacity, H // BLOCK_W) * 4.0 - 2.0

    e4m3_tt = _make_e4m3_from_torch(input_e4m3, device=device)
    # Feed the scales either as a plain (M, H/128) fp32 tensor or packed into the int32 metadata tail;
    # both drive the same math, so the golden is identical.
    if scales_from_metadata:
        scale_tt = None
        metadata_tt = ttnn.from_torch(
            _pack_scale_metadata(input_scale),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        metadata_tt = None
        scale_tt = ttnn.from_torch(
            input_scale,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    expert_region_offsets_tt = create_u32_tensor(device, expert_region_offsets)
    expert_token_counts_tt = create_u32_tensor(device, expert_token_counts)
    global_expert_idx_table_tt = create_u32_tensor(device, global_expert_idx_table)

    out_tt = ttnn.experimental.deepseek_prefill.masked_per_token_cast_back(
        e4m3_tt,
        scale_tt,
        expert_region_offsets_tt,
        expert_token_counts_tt,
        global_expert_idx_table_tt,
        experts_per_chip=experts_per_chip,
        output_dtype=output_dtype,
        metadata=metadata_tt,
        bf16_scale=bf16_scale,
    )
    out = ttnn.to_torch(out_tt).float()

    assert tuple(out_tt.shape) == (capacity, H)
    assert out_tt.dtype == output_dtype

    golden_scale = input_scale.to(torch.bfloat16).float() if bf16_scale else input_scale
    golden = input_e4m3.float() * golden_scale.repeat_interleave(BLOCK_W, dim=-1)
    # bf16_scale runs the whole compute datapath in bf16; a bf16 output additionally rounds at the packer.
    # Either narrows the result to bf16 precision; only fp32-output + fp32-scale stays full fp32.
    if bf16_scale or output_dtype == ttnn.bfloat16:
        golden = golden.to(torch.bfloat16).float()

    # The op sweeps [0, total_valid_rows) contiguously (valid tokens + end-of-region tile padding), so
    # every written row must equal e4m3 * scale; the tail beyond total_valid_rows is left untouched.
    prefix_out = out[:total_valid_rows]
    prefix_golden = golden[:total_valid_rows]
    normal = input_e4m3.float()[:total_valid_rows].abs() > 2.0**-6
    atol = 1e-2 if bf16_scale else 1e-3
    assert_quality(
        prefix_out[normal],
        prefix_golden[normal],
        pcc_threshold=0.999,
        rtol=1e-2,
        atol=atol,
        label=f"masked cast back {label} bf16_scale={bf16_scale} "
        f"metadata={scales_from_metadata} out={output_dtype}",
    )
