# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for nomic_qkv_rotary_heads (immutable spec; see ttnn/ttnn/operations/nomic_qkv_rotary_heads/op_design.md).

qkv = x @ W^T + b (fp32), split into 12 heads of 64 for q, k and v, q scaled by 1/8, NeoX rotate-half rotary (base 10000)
on q and k, written as three interleaved (B, 12, S, 64) bfloat16 TILE tensors. x is in the layout
nomic_residual_layer_norm leaves it in. Tolerances are the golden suite's
(eval/golden_tests/nomic_qkv_rotary_heads/helpers.py TOLERANCES): PCC >= 0.99998 and RMS error <= 0.0050 of the
reference's standard deviation, per output, against a float64 reference. Padding rows (s >= S) of q, k and v must be
exactly zero, also when x's padding rows hold inf or NaN.

Shapes pin every per-core M-block count of the plan (1, 2 and 4 blocks; 10 and 11 grid columns), a padded sequence
(3x400, 8x500), positions up to 1023 (1x1024) and the three focus shapes (2x512, 4x512, 8x512).
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.nomic_qkv_rotary_heads import nomic_qkv_rotary_heads
from ttnn.operations.nomic_qkv_rotary_heads import (
    default_compute_kernel_config,
    head_dim_order,
    prepare_qkv_weights,
    prepare_rotary_tables,
)

HIDDEN, HEADS, HEAD_DIM = 768, 12, 64
ROTARY_BASE = 10000.0
PCC_THRESHOLD = {ttnn.bfloat16: 0.99998}
RMS_THRESHOLD = {ttnn.bfloat16: 0.0050}

SHAPES = [
    (2, 1, 512, 768),  # M = 32, one M block a core; the 1k-token focus shape
    (1, 1, 1024, 768),  # M = 32, positions up to 1023
    (3, 1, 352, 768),  # M = 33
    (3, 1, 400, 768),  # M = 39, seq_padded, 10 grid columns, 2 M blocks
    (4, 1, 512, 768),  # M = 64, 2 M blocks; the 2k-token focus shape
    (8, 1, 500, 768),  # M = 128, seq_padded
    (8, 1, 512, 768),  # M = 128, 4 M blocks; the 4k-token focus shape
]

OUTPUTS = [
    ("l1", "preallocated"),
    ("l1", "new"),
    ("dram", "preallocated"),
    ("dram", "new"),
]


# ---------------------------------------------------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------------------------------------------------


def rotary_cos_sin(seqlen):
    inv_freq = 1.0 / (ROTARY_BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    angles = torch.arange(seqlen, dtype=torch.float64)[:, None] * inv_freq[None, :]
    return torch.cat((angles.cos(), angles.cos()), dim=-1), torch.cat((angles.sin(), angles.sin()), dim=-1)


def rotate_half(t):
    half = t.shape[-1] // 2
    return torch.cat((-t[..., half:], t[..., :half]), dim=-1)


def torch_reference(x, weight, bias):
    """(q, k, v), each (B, 12, S, 64) float64, reference head-dim order; q carries 1/sqrt(64)."""
    batch, _, seqlen, _ = x.shape
    qkv = x.double().reshape(batch, seqlen, HIDDEN) @ weight.double().T + bias.double()
    q, k, v = (t.reshape(batch, seqlen, HEADS, HEAD_DIM).permute(0, 2, 1, 3) for t in qkv.split(HIDDEN, dim=-1))
    cos, sin = rotary_cos_sin(seqlen)
    q = q * HEAD_DIM**-0.5
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin, v


# ---------------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------------


def make_inputs(shape):
    torch.manual_seed(42)
    x = torch.randn(shape).to(torch.bfloat16)
    weight = (torch.randn(3 * HIDDEN, HIDDEN) / math.sqrt(HIDDEN)).to(torch.bfloat16)
    bias = (0.02 * torch.randn(3 * HIDDEN)).to(torch.bfloat16)
    return x, weight, bias


def ln_layout(shape, device):
    """x's layout as the norm writes it (authoritative: the norm's own layout functions)."""
    from ttnn.operations.nomic_residual_layer_norm import (
        input_memory_config as norm_input_memory_config,
        output_memory_config as norm_output_memory_config,
    )

    return norm_output_memory_config(tuple(shape), device, norm_input_memory_config(tuple(shape), device))


def x_to_device(x, device, poison=None):
    """x in the norm's layout. With `poison`, the padding rows (s >= S of the last tile row) hold that value."""
    shape = tuple(x.shape)
    layout = ln_layout(shape, device)
    if poison is None:
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
    batch, _, seqlen, _ = shape
    padded_seqlen = math.ceil(seqlen / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    fill = torch.full((batch, 1, padded_seqlen - seqlen, HIDDEN), poison).to(torch.bfloat16)
    interleaved = ttnn.from_torch(
        torch.cat((x, fill), dim=2),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    interleaved = ttnn.reshape(interleaved, ttnn.Shape(shape), ttnn.Shape((batch, 1, padded_seqlen, HIDDEN)))
    tt_x = ttnn.to_memory_config(interleaved, layout)
    ttnn.deallocate(interleaved)
    return tt_x


def allocate_heads(shape, device, memory_config):
    """[q, k, v], allocated v, k, q as the model allocates them (v lands highest in L1)."""
    heads_shape = ttnn.Shape((shape[0], HEADS, shape[-2], HEAD_DIM))
    value, key, query = (
        ttnn.allocate_tensor_on_device(heads_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)
        for _ in range(3)
    )
    return [query, key, value]


# ---------------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------------


def pcc_and_relative_rms(actual, expected):
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    a = actual - actual.mean()
    e = expected - expected.mean()
    pcc = (a * e).sum() / torch.sqrt((a * a).sum() * (e * e).sum())
    rms = torch.sqrt(((actual - expected) ** 2).mean()) / expected.std()
    return pcc.item(), rms.item()


def check_outputs(outputs, expected, shape, memory_config, provided=None):
    batch, _, seqlen, _ = shape
    order = list(head_dim_order())
    assert len(order) == HEAD_DIM and sorted(order) == list(range(HEAD_DIM))
    assert len(outputs) == 3
    for index, (name, got, ref) in enumerate(zip("qkv", outputs, expected)):
        if provided is not None:
            assert (
                got.buffer_address() == provided[index].buffer_address()
            ), f"{name} must be written into output_tensors[{index}]"
        assert got.memory_config() == memory_config, f"{name}: memory config {got.memory_config()}"
        assert not got.is_sharded(), f"{name} must be interleaved (SDPA refuses sharded operands)"
        assert got.dtype == ttnn.bfloat16 and got.layout == ttnn.TILE_LAYOUT
        assert tuple(got.shape) == (batch, HEADS, seqlen, HEAD_DIM)

        reference = ref[..., order] if name in "qk" else ref
        actual = ttnn.to_torch(got).float()
        assert torch.isfinite(actual).all(), f"{name}: non-finite values in valid rows"
        pcc, rms = pcc_and_relative_rms(actual, reference)
        assert pcc >= PCC_THRESHOLD[ttnn.bfloat16], f"{name}: PCC {pcc:.6f} < {PCC_THRESHOLD[ttnn.bfloat16]}"
        assert rms <= RMS_THRESHOLD[ttnn.bfloat16], f"{name}: relative RMS {rms:.5f} > {RMS_THRESHOLD[ttnn.bfloat16]}"

        if seqlen % ttnn.TILE_SIZE:
            padding = got.cpu().to_torch_with_padded_shape().float()[..., seqlen:, :]
            assert torch.all(
                padding == 0
            ), f"{name}: padding rows past S={seqlen} must be exactly zero (max |{padding.abs().max().item():.3e}|)"


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("output_buffer,output_mode", OUTPUTS, ids=[f"{b}-{m}" for b, m in OUTPUTS])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{s[0]}x{s[2]}" for s in SHAPES])
def test_nomic_qkv_rotary_heads(shape, output_buffer, output_mode, device):
    x, weight, bias = make_inputs(shape)
    expected = torch_reference(x, weight, bias)

    weights = prepare_qkv_weights(weight, bias, device)
    tables = prepare_rotary_tables(shape[-2], device)
    tt_x = x_to_device(x, device)

    memory_config = ttnn.L1_MEMORY_CONFIG if output_buffer == "l1" else ttnn.DRAM_MEMORY_CONFIG
    provided = allocate_heads(shape, device, memory_config) if output_mode == "preallocated" else None
    kwargs = dict(memory_config=memory_config, compute_kernel_config=default_compute_kernel_config())
    if provided is not None:
        kwargs["output_tensors"] = provided

    outputs = nomic_qkv_rotary_heads(tt_x, weights, tables, **kwargs)
    check_outputs(outputs, expected, shape, memory_config, provided)


@pytest.mark.parametrize("poison", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("shape", [(3, 1, 400, 768), (8, 1, 500, 768)], ids=["3x400", "8x500"])
def test_padding_rows_are_zero_for_non_finite_inputs(shape, poison, device):
    x, weight, bias = make_inputs(shape)
    expected = torch_reference(x, weight, bias)
    tt_x = x_to_device(x, device, poison=poison)
    seqlen = shape[-2]
    poisoned = tt_x.cpu().to_torch_with_padded_shape().float()[..., seqlen:, :]
    if math.isnan(poison):
        assert torch.isnan(poisoned).all(), "x's padding lost its NaN poison"
    else:
        assert torch.isinf(poisoned).all(), "x's padding lost its inf poison"

    outputs = nomic_qkv_rotary_heads(
        tt_x,
        prepare_qkv_weights(weight, bias, device),
        prepare_rotary_tables(seqlen, device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=default_compute_kernel_config(),
    )
    check_outputs(outputs, expected, shape, ttnn.L1_MEMORY_CONFIG)


def test_tables_for_a_longer_sequence_are_accepted(device):
    """Tables cover S when built for S' >= S: rows depend only on the position."""
    shape = (2, 1, 500, 768)
    x, weight, bias = make_inputs(shape)
    expected = torch_reference(x, weight, bias)
    outputs = nomic_qkv_rotary_heads(
        x_to_device(x, device),
        prepare_qkv_weights(weight, bias, device),
        prepare_rotary_tables(512, device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    check_outputs(outputs, expected, shape, ttnn.L1_MEMORY_CONFIG)


def test_fp32_dest_accumulation_off_is_refused(device, expect_error):
    shape = (2, 1, 512, 768)
    x, weight, bias = make_inputs(shape)
    config = default_compute_kernel_config()
    config.fp32_dest_acc_en = False
    # SupportRefusal subclasses NotImplementedError; the message text is the op's own.
    with expect_error(NotImplementedError, ""):
        nomic_qkv_rotary_heads(
            x_to_device(x, device),
            prepare_qkv_weights(weight, bias, device),
            prepare_rotary_tables(shape[-2], device),
            compute_kernel_config=config,
        )


def test_tables_shorter_than_the_sequence_are_refused(device, expect_error):
    shape = (2, 1, 512, 768)
    x, weight, bias = make_inputs(shape)
    with expect_error(NotImplementedError, ""):
        nomic_qkv_rotary_heads(
            x_to_device(x, device),
            prepare_qkv_weights(weight, bias, device),
            prepare_rotary_tables(256, device),
        )
