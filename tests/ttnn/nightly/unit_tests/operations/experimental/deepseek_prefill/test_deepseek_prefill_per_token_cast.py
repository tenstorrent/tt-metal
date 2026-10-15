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

from tt_py_test_utils_common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from tests.ttnn.utils_for_testing import comp_pcc, assert_equal
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

pytestmark = pytest.mark.use_module_device


BLOCK_W = 128
E4M3_MAX = 448.0

SHAPES = [
    (1, 1024),  # single row (partial tile-row)
    (30, 1152),  # partial tile-row + 9 scale blocks
    (PREFILL_CHUNK_TOKENS_PER_CHIP, 7168),
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


@pytest.mark.uncollect_if(pred=ci_pruning.bh_row_major_input)
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


@pytest.mark.uncollect_if(pred=ci_pruning.bh_row_major_input)
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


@pytest.mark.uncollect_if(pred=ci_pruning.bh_row_major_input)
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
    # Power-of-two scales and their reciprocals are exact, so every boundary
    # value is representable after scaling and must round-trip exactly.
    expected_output = (block_maxima / expected_scale).to(torch.float8_e4m3fn).float() * expected_scale
    actual_output = y[0, ::BLOCK_W]
    assert torch.equal(
        actual_output, expected_output
    ), f"Unexpected E4M3FN boundary outputs: got {actual_output.tolist()}, expected {expected_output.tolist()}"


# ---------------------------------------------------------------------------
# per_token_cast_back: dequantization value tests.
# ---------------------------------------------------------------------------


@pytest.mark.uncollect_if(pred=ci_pruning.bh_narrow_scales_to_bf16)
@pytest.mark.parametrize("narrow_scales_to_bf16", [False, True], ids=["scales_kept_at_fp32", "scales_narrow_to_bf16"])
@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_cast_back_dequant(device, out_dtype, shape, narrow_scales_to_bf16):
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
    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(
        e4m3_tt, scale_tt, output_dtype=ttnn_dtype, narrow_scales_to_bf16=narrow_scales_to_bf16
    )
    out = ttnn.to_torch(out_tt).float()

    # apples-to-apples: on the bf16 path the device narrows the scale to bf16 before the multiply,
    # so the golden narrows it too
    golden_scale = input_scale.to(torch.bfloat16).float() if narrow_scales_to_bf16 else input_scale
    golden = input_e4m3.float() * golden_scale.repeat_interleave(BLOCK_W, dim=-1)
    if out_dtype == "bfloat16":
        golden = golden.to(torch_dtype).float()

    assert tuple(out_tt.shape) == shape
    assert out_tt.dtype == ttnn_dtype
    assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    # Restrict to normal e4m3 values where relative tolerance is meaningful.
    normal = input_e4m3.float().abs() > 2.0**-6
    # narrow_scales_to_bf16 runs the multiply in bf16/HiFi2 (vs fp32/HiFi4)
    # which has a real precision cost, not a golden mismatch, so allow a slightly looser rtol.
    rtol = 1.5e-2 if narrow_scales_to_bf16 else 1e-2
    assert_quality(
        out[normal],
        golden[normal],
        pcc_threshold=0.999,
        rtol=rtol,
        atol=1e-3,
        label=f"dequant {out_dtype} narrow_scales_to_bf16={narrow_scales_to_bf16} shape={shape}",
    )


# ---------------------------------------------------------------------------
# Round-trip: per_token_cast_to_fp8 followed by per_token_cast_back.
# ---------------------------------------------------------------------------


@pytest.mark.uncollect_if(pred=ci_pruning.bh_row_major_input)
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


# Each number inside the list is how much tokens does each expert on a single chip have
# The length of the list is num_experts_per_chip
# Example: [130, 74, 200, 96, 41] means that we have 5 experts per chip, where the first expert has 130 tokens, etc...
TOKEN_COUNT_AWARE_CASES = [
    # dense - each expert has tokens
    ("dense", [130, 74, 200, 96, 41]),
    # zeros_middle, zeros_leading, zeros_trailing - some expert have no tokens
    # we test if those tokens are skipped correctly and if the operator
    # correctly handles the case where some experts have no tokens
    ("zeros_middle", [130, 0, 0, 74, 200]),
    ("zeros_leading", [0, 0, 130, 74, 200]),
    ("zeros_trailing", [130, 74, 200, 0, 0]),
    # overflow - the packed regions run past the flat buffer; the op must clamp to capacity, drop the
    # out-of-bounds tokens, and never hang.
    ("overflow", [9000, 9000, 9000, 9000, 9000]),
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
METADATA_HEADER = 3  # This may change in the future, but the actual number is not important for the test
# The number is just used to set the starting offset when reading the scales from the metadata row


def _pack_scale_metadata(input_scale):
    """Bit-store fp32 per-token scales in the tail of an int32 dispatch-metadata row (the metadata scale
    path). Leading header columns are filled with a sentinel the kernel must ignore."""
    M, blocks = input_scale.shape
    meta = torch.full((M, METADATA_HEADER + blocks), 0x0BADF00D, dtype=torch.int32)
    meta[:, METADATA_HEADER:] = input_scale.contiguous().view(torch.int32)
    return meta


@pytest.mark.uncollect_if(pred=ci_pruning.bh_scales_not_from_metadata)
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scales_from_metadata", [False, True])
@pytest.mark.parametrize("label, counts", TOKEN_COUNT_AWARE_CASES, ids=[c[0] for c in TOKEN_COUNT_AWARE_CASES])
def test_token_count_aware_cast_back(device, label, counts, scales_from_metadata, output_dtype):
    torch.manual_seed(0)
    H = KimiK27Config.EMB_SIZE

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
    capacity = MAX_DISPATCH_BUFFER_TOKENS  # fixed flat buffer; [total_valid_rows, capacity) is untouched tail
    # The op clamps its swept region to the buffer capacity, so rows past it are dropped (overflow case).
    total_valid_rows = min(running_offset, capacity)

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

    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(
        e4m3_tt,
        scale_tt,
        token_count_aware=True,
        expert_region_offsets=expert_region_offsets_tt,
        expert_token_counts=expert_token_counts_tt,
        global_expert_idx_table=global_expert_idx_table_tt,
        experts_per_chip=experts_per_chip,
        output_dtype=output_dtype,
        metadata=metadata_tt,
    )
    out = ttnn.to_torch(out_tt).float()

    assert tuple(out_tt.shape) == (capacity, H)
    assert out_tt.dtype == output_dtype

    golden = input_e4m3.float() * input_scale.repeat_interleave(BLOCK_W, dim=-1)
    # A bf16 output rounds the result at the packer; fp32 output stays full fp32.
    if output_dtype == ttnn.bfloat16:
        golden = golden.to(torch.bfloat16).float()

    # The op sweeps [0, total_valid_rows) contiguously (valid tokens + end-of-region tile padding), so
    # every written row must equal e4m3 * scale; the tail beyond total_valid_rows is left untouched.
    prefix_out = out[:total_valid_rows]
    prefix_golden = golden[:total_valid_rows]
    normal = input_e4m3.float()[:total_valid_rows].abs() > 2.0**-6
    assert_quality(
        prefix_out[normal],
        prefix_golden[normal],
        pcc_threshold=0.999,
        rtol=1e-2,
        atol=1e-3,
        label=f"token-count-aware cast back {label} metadata={scales_from_metadata} out={output_dtype}",
    )


# ---------------------------------------------------------------------------
# scale_block_w (32 / 64 / 128) and round-to-nearest-even. Kimi-K3 dispatches its 3584-wide latent
# as e4m3 with one fp32 scale per 32 elements (MXFP8 granularity). The I/O chunk stays 128 wide.
# ---------------------------------------------------------------------------

K3_LATENT = 3584
SCALE_BLOCK_SHAPES = [
    (30, 1152),  # partial tile-row
    (PREFILL_CHUNK_TOKENS_PER_CHIP, K3_LATENT),  # K3 latent, one chip's chunk
    (2, 3, 30, 1152),  # 4D + partial tile-row
]


def _ref_pow2_scale_blocks(x_fp32, block_w):
    *leading, H = x_fp32.shape
    amax = x_fp32.reshape(*leading, H // block_w, block_w).abs().amax(dim=-1).clamp(min=1e-4)
    return torch.pow(2.0, torch.ceil(torch.log2(amax / E4M3_MAX)))


def _e4m3_rtz(v):
    """Truncate fp32 to the e4m3 grid (normals): clear the low 20 mantissa bits."""
    bits = v.contiguous().view(torch.int32) & ~0xFFFFF
    return bits.view(torch.float32)


@pytest.mark.parametrize("round_to_nearest_even", [False, True], ids=["rtz", "rne"])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["rm", "tile"])
@pytest.mark.parametrize("scale_block_w", [32, 64, 128], ids=lambda b: f"block_w{b}")
@pytest.mark.parametrize("shape", SCALE_BLOCK_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_cast_to_fp8_scale_block_w(device, shape, scale_block_w, input_layout, round_to_nearest_even):
    torch.manual_seed(0)
    # Per-block magnitudes spread over several octaves so a wrong block boundary changes the scale.
    x = torch.randn(shape) * torch.pow(
        2.0, torch.randint(-6, 6, shape[:-1] + (shape[-1] // 32, 1)).float()
    ).repeat_interleave(32, -1).reshape(shape)
    x = x.to(torch.bfloat16)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    q_tt, s_tt = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        x_tt,
        round_scale_to_power_of_two=True,
        scale_block_w=scale_block_w,
        round_to_nearest_even=round_to_nearest_even,
    )
    assert tuple(s_tt.shape) == tuple(shape[:-1]) + (shape[-1] // scale_block_w,)
    s = ttnn.to_torch(s_tt).float()
    # FP8_E4M3 has no to_torch on a single device: decode it with cast_back against unit scales.
    ones_tt = ttnn.from_torch(
        torch.ones(s.shape),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.per_token_cast_back(
            q_tt, ones_tt, output_dtype=ttnn.float32, scale_block_w=scale_block_w
        )
    ).float()

    ref_s = _ref_pow2_scale_blocks(x.float(), scale_block_w)
    # pow2 scales are exact except where fp32 amax/448 sits on a power-of-two boundary (one step apart).
    ratio = s / ref_s
    assert torch.all((ratio == 1.0) | (ratio == 2.0) | (ratio == 0.5)), "scale off by more than one power of two"
    exact_scale = (ratio == 1.0).float().mean().item()
    assert exact_scale >= 0.999, f"only {exact_scale:.4%} of scales exact"

    # Quantized bytes against the device's own scales (isolates rounding from scale selection).
    scaled = x.float() / s.repeat_interleave(scale_block_w, dim=-1)
    if round_to_nearest_even:
        expected = scaled.to(torch.float8_e4m3fn).float()
    else:
        expected = _e4m3_rtz(scaled).to(torch.float8_e4m3fn).float()
    normal = scaled.abs() >= 2.0**-6  # e4m3 subnormals keep packer truncation (documented)
    match = (q[normal] == expected[normal]).float().mean().item()
    logger.info(f"block_w={scale_block_w} rne={round_to_nearest_even} {input_layout}: e4m3 match {match:.6%}")
    assert match >= 0.9999, f"e4m3 match {match:.4%} < 99.99%"

    # Round trip through per_token_cast_back with the same block width.
    back = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.per_token_cast_back(
            q_tt, s_tt, output_dtype=ttnn.float32, scale_block_w=scale_block_w
        )
    ).float()
    golden = q * s.repeat_interleave(scale_block_w, dim=-1)
    assert_quality(
        back, golden, pcc_threshold=0.99999, rtol=1e-6, atol=1e-30, label=f"cast_back block_w={scale_block_w}"
    )
    # Error vs the input: RNE <= half an e4m3 ulp (1/16 relative), RTZ < one ulp (1/8 relative).
    rel = ((back - x.float()).abs() / x.float().abs().clamp(min=1e-30))[normal]
    bound = 0.0625 if round_to_nearest_even else 0.125
    assert (rel <= bound + 1e-6).float().mean().item() >= 0.9999
    if round_to_nearest_even:
        bias = ((back - x.float()) * x.float().sign())[normal].mean().item()
        logger.info(f"rne signed bias {bias:.3e}")


@pytest.mark.parametrize("scale_block_w", [32, 128], ids=lambda b: f"block_w{b}")
def test_token_count_aware_cast_back_scale_block_w(device, scale_block_w):
    """Metadata-tail scales with a K3-shaped metadata row (3 routing words + H/scale_block_w scales)."""
    torch.manual_seed(0)
    H = K3_LATENT
    counts = [37, 0, 64, 5, 130]
    experts_per_chip = len(counts)
    num_routed_experts = 2 * experts_per_chip
    table = [2 * s + 1 for s in range(experts_per_chip)]
    region = [0] * num_routed_experts
    tok = [0] * num_routed_experts
    off = 0
    for slot, c in enumerate(counts):
        g = table[slot]
        region[g], tok[g] = off, c
        off += _ceil_tile(c)
    capacity = 1024
    e4m3 = (torch.randn(capacity, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    n_scales = H // scale_block_w
    scales = torch.pow(2.0, torch.randint(-8, 4, (capacity, n_scales)).float())
    meta = torch.full((capacity, 3 + n_scales), 0x0BADF00D, dtype=torch.int32)
    meta[:, 3:] = scales.contiguous().view(torch.int32)
    out = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.per_token_cast_back(
            _make_e4m3_from_torch(e4m3, device=device),
            None,
            token_count_aware=True,
            expert_region_offsets=create_u32_tensor(device, region),
            expert_token_counts=create_u32_tensor(device, tok),
            global_expert_idx_table=create_u32_tensor(device, table),
            experts_per_chip=experts_per_chip,
            output_dtype=ttnn.float32,
            metadata=ttnn.from_torch(
                meta,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            scale_block_w=scale_block_w,
        )
    ).float()
    golden = e4m3.float() * scales.repeat_interleave(scale_block_w, dim=-1)
    # e4m3 subnormals are not decoded exactly by the unpacker (same mask as test_token_count_aware_cast_back).
    normal = e4m3.float()[:off].abs() > 2.0**-6
    assert torch.equal(out[:off][normal], golden[:off][normal]), "count-aware metadata-tail dequant mismatch"


@pytest.mark.parametrize("block", [32, 128], ids=["block_w32", "block_w128"])
@pytest.mark.parametrize("out_l1", [False, True], ids=["dram_out", "l1_out"])
@pytest.mark.parametrize("counts", [[37, 0, 64, 5, 130], [0, 0, 0, 0, 1]], ids=["mixed", "one_token"])
def test_count_aware_packed_quant_expert_buffer(device, counts, out_l1, block):
    """Scaled fp8 combine source: quantize the valid prefix of a bf8_b TILE expert buffer into packed
    [H e4m3 | H/32 fp32] rows, then dequantize them with cast_back(scales_in_row_tail=True)."""
    torch.manual_seed(0)
    H, capacity = (K3_LATENT, 640) if block == 32 else (6144, 640)  # 1x128: the GLM 5.3 row
    experts_per_chip = len(counts)
    table = [2 * s + 1 for s in range(experts_per_chip)]
    region, tok = [0] * (2 * experts_per_chip), [0] * (2 * experts_per_chip)
    off = 0
    for slot, c in enumerate(counts):
        region[table[slot]], tok[table[slot]] = off, c
        off += _ceil_tile(c)
    x = torch.randn(capacity, H) * torch.pow(2.0, torch.randint(-4, 4, (capacity, 1)).float())
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_bf8 = ttnn.to_torch(x_tt).float()  # what the device actually holds
    packed_tt, _ = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        x_tt,
        round_scale_to_power_of_two=True,
        scale_block_w=block,
        round_to_nearest_even=True,
        token_count_aware=True,
        expert_region_offsets=create_u32_tensor(device, region),
        expert_token_counts=create_u32_tensor(device, tok),
        global_expert_idx_table=create_u32_tensor(device, table),
        experts_per_chip=experts_per_chip,
        pack_scales_in_row=True,
        memory_config=ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG,  # L1: the K3 combine source
    )
    assert tuple(packed_tt.shape) == (capacity, H + 4 * H // block)
    if out_l1:
        packed_tt = ttnn.to_memory_config(packed_tt, ttnn.DRAM_MEMORY_CONFIG)
    back = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.per_token_cast_back(
            packed_tt, None, output_dtype=ttnn.float32, scale_block_w=block, scales_in_row_tail=True
        )
    ).float()
    assert tuple(back.shape) == (capacity, H)
    valid = back[:off]
    ref = x_bf8[:off]
    s = _ref_pow2_scale_blocks(ref, block).repeat_interleave(block, dim=-1)
    expected = (ref / s).to(torch.float8_e4m3fn).float() * s
    normal = (ref / s).abs() >= 2.0**-6
    match = (valid[normal] == expected[normal]).float().mean().item()
    logger.info(f"count-aware packed quant {counts}: match {match:.6%} over {off} rows")
    assert match >= 0.999, f"packed quant/dequant match {match:.4%}"


@pytest.mark.parametrize("scale_block_w", [32], ids=["block_w32"])
def test_count_aware_packed_round_trip(device, scale_block_w):
    """Packed rows [H e4m3 | scales] quantized and dequantized count-aware (the dispatch-side DQ1 path):
    the dequant reads each row's scales from its own tail and only the valid expert prefix."""
    torch.manual_seed(5)
    H, capacity = K3_LATENT, 2048
    counts = [23 + (7 * i) % 40 for i in range(20)]
    epc = len(counts)
    table = [2 * s + 1 for s in range(epc)]
    region, tok = [0] * (2 * epc), [0] * (2 * epc)
    off = 0
    for slot, c in enumerate(counts):
        region[table[slot]], tok[table[slot]] = off, c
        off += _ceil_tile(c)
    x = torch.randn(capacity, H) * torch.pow(2.0, torch.randint(-4, 4, (capacity, 1)).float())
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    blk = dict(scale_block_w=scale_block_w)
    packed_tt, _ = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        x_tt, round_scale_to_power_of_two=True, pack_scales_in_row=True, **blk
    )
    ca = dict(
        token_count_aware=True,
        expert_region_offsets=create_u32_tensor(device, region),
        expert_token_counts=create_u32_tensor(device, tok),
        global_expert_idx_table=create_u32_tensor(device, table),
        experts_per_chip=epc,
    )
    plain = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.per_token_cast_back(
            packed_tt, None, output_dtype=ttnn.float32, scales_in_row_tail=True, **blk
        )
    ).float()
    aware = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.per_token_cast_back(
            packed_tt, None, output_dtype=ttnn.float32, scales_in_row_tail=True, **blk, **ca
        )
    ).float()
    assert tuple(aware.shape) == (capacity, H)
    assert torch.equal(aware[:off], plain[:off]), "count-aware row-tail dequant differs from the plain path"
    rel = ((aware[:off] - x[:off]).norm() / x[:off].norm()).item()
    logger.info(f"count-aware packed round trip ({scale_block_w}): rel err {rel:.4f}")
    assert rel < 0.08


@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
@pytest.mark.parametrize("block", [32, 128], ids=["block_w32", "block_w128"])
def test_count_aware_packed_tile_output(device, out_dtype, block):
    """1x32 / 1x128 (GLM) count-aware dequant straight to TILE output (the routed expert's tiled-x input): the valid prefix
    must match the row-major path (bf16 exactly; bf8_b within its shared-exponent rounding)."""
    torch.manual_seed(6)
    H, capacity = (K3_LATENT, 2048) if block == 32 else (6144, 1024)  # 1x128: the GLM 5.3 row
    counts = [23 + (7 * i) % 40 for i in range(20)]
    epc = len(counts)
    table = [2 * s + 1 for s in range(epc)]
    region, tok = [0] * (2 * epc), [0] * (2 * epc)
    off = 0
    for slot, c in enumerate(counts):
        region[table[slot]], tok[table[slot]] = off, c
        off += _ceil_tile(c)
    x = torch.randn(capacity, H) * torch.pow(2.0, torch.randint(-4, 4, (capacity, H // 32)).float()).repeat_interleave(
        32, dim=1
    )
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    packed_tt, _ = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        x_tt, round_scale_to_power_of_two=True, pack_scales_in_row=True, scale_block_w=block
    )
    ca = dict(
        token_count_aware=True,
        expert_region_offsets=create_u32_tensor(device, region),
        expert_token_counts=create_u32_tensor(device, tok),
        global_expert_idx_table=create_u32_tensor(device, table),
        experts_per_chip=epc,
        scales_in_row_tail=True,
        scale_block_w=block,
    )
    ref = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.per_token_cast_back(packed_tt, None, output_dtype=ttnn.float32, **ca)
    ).float()
    tiled_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(
        packed_tt, None, output_dtype=out_dtype, tile_output=True, **ca
    )
    assert tiled_tt.layout == ttnn.TILE_LAYOUT and tiled_tt.dtype == out_dtype
    tiled = ttnn.to_torch(tiled_tt).float()
    assert tuple(tiled.shape) == (capacity, H)
    if out_dtype == ttnn.bfloat16:
        assert torch.equal(tiled[:off], ref[:off]), "tile output differs from the row-major dequant"
    else:
        rel = ((tiled[:off] - ref[:off]).norm() / ref[:off].norm()).item()
        logger.info(f"tile output bf8 vs row-major: rel err {rel:.4f}")
        assert rel < 0.02


def test_count_aware_packed_tile_output_row_segments(device):
    """Rows that are a width-concat of per-slice packed rows (the all-gathered latent: each chip quantizes its
    896-column slice, rows padded to 64 B): tile-output dequant with row_segments must match the reference."""
    torch.manual_seed(8)
    H, capacity, segs = K3_LATENT, 2048, 4
    counts = [23 + (7 * i) % 40 for i in range(20)]
    epc = len(counts)
    table = [2 * s + 1 for s in range(epc)]
    region, tok = [0] * (2 * epc), [0] * (2 * epc)
    off = 0
    for slot, c in enumerate(counts):
        region[table[slot]], tok[table[slot]] = off, c
        off += _ceil_tile(c)
    x = torch.randn(capacity, H) * torch.pow(2.0, torch.randint(-4, 4, (capacity, H // 32)).float()).repeat_interleave(
        32, dim=1
    )
    seg_w = H // segs
    # Per-slice packed rows [seg_w e4m3 | seg_w/32 fp32 pow2 scales | pad to 1024 B], built on the host.
    parts = []
    for s_ in range(segs):
        xb = x[:, s_ * seg_w : (s_ + 1) * seg_w].reshape(capacity, seg_w // 32, 32)
        sc = torch.pow(2.0, torch.ceil(torch.log2(xb.abs().amax(-1, keepdim=True).clamp(min=1e-4) / 448.0)))
        q = (xb / sc).to(torch.float8_e4m3fn).view(torch.uint8).reshape(capacity, seg_w)
        pad = torch.zeros(capacity, 1024 - seg_w - 4 * (seg_w // 32), dtype=torch.uint8)
        parts.append(torch.cat([q, sc.squeeze(-1).contiguous().view(torch.uint8), pad], dim=-1))
    raw = torch.cat(parts, dim=-1)
    packed_tt = ttnn.from_torch(
        raw.view(torch.float8_e4m3fn).float(), dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ca = dict(
        token_count_aware=True,
        expert_region_offsets=create_u32_tensor(device, region),
        expert_token_counts=create_u32_tensor(device, tok),
        global_expert_idx_table=create_u32_tensor(device, table),
        experts_per_chip=epc,
        scales_in_row_tail=True,
        scale_block_w=32,
    )
    out = ttnn.to_torch(
        ttnn.experimental.deepseek_prefill.per_token_cast_back(
            packed_tt, None, output_dtype=ttnn.bfloat16, tile_output=True, row_segments=segs, **ca
        )
    ).float()
    assert tuple(out.shape) == (capacity, H)
    # Reference: the same per-slice quantization dequantized on the host.
    ref = torch.zeros(capacity, H)
    for s_ in range(segs):
        seg = raw[:, s_ * 1024 : (s_ + 1) * 1024]
        qb = seg[:, :seg_w].contiguous()
        q = qb.view(torch.float8_e4m3fn).float()
        q[(qb & 0x78) == 0] = 0.0  # the unpacker flushes e4m3 subnormals to zero
        sc = seg[:, seg_w : seg_w + 4 * (seg_w // 32)].contiguous().view(torch.float32)
        ref[:, s_ * seg_w : (s_ + 1) * seg_w] = q * sc.repeat_interleave(32, dim=1)
    bad = out[:off] != ref[:off].to(torch.bfloat16).float()
    assert not bad.any(), "segmented tile dequant mismatch"
    rel = ((out[:off] - x[:off]).norm() / x[:off].norm()).item()
    assert rel < 0.08, rel
