# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 4 — Compute kernel config exposed to caller.

Verifies the new ``compute_kernel_config`` and ``unpack_to_dest_mode`` kwargs
on ``backward_softmax``:

1. **Default contract**: passing ``compute_kernel_config=None`` is
   bit-identical to the pre-R4 entry point (the R3 dtype-aware default is
   preserved).
2. **Override per dtype**: a custom ``WormholeComputeKernelConfig`` reaches
   the program descriptor — verified through (a) numerical correctness under
   each override (PCC within the per-dtype band) and (b) round-trip on the
   resolved fields by reaching into ``create_program_descriptor`` directly
   and inspecting the produced ``ComputeConfigDescriptor``.
3. **`math_fidelity = Invalid` fallback**: the no-arg
   ``WormholeComputeKernelConfig()`` constructor leaves
   ``math_fidelity = Invalid``. The descriptor must fall back to the
   dtype-default fidelity instead of letting that sentinel propagate to JIT.
4. **`unpack_to_dest_mode` override**: applying ``UnpackToDestMode.Default``
   to every CB (the safe override) still produces correct results; applying
   ``UnpackToDestFp32`` only to a non-matmul-input CB is the documented safe
   case from the verifier note in ``op_requirements.md``.
5. **Validation**: a malformed ``unpack_to_dest_mode`` (wrong length, wrong
   element type) is rejected at descriptor construction.

The hard-coded HiFi2+fp32_dest_acc+math_approx=True parametrised case on
bf16 (from the op_requirements ask) is exercised by
``test_backward_softmax_compute_config_per_dtype_overrides``.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.operations.backward_softmax import backward_softmax
from ttnn.operations.backward_softmax.backward_softmax_program_descriptor import (
    create_program_descriptor,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _torch_reference(grad_output: torch.Tensor, output: torch.Tensor, dim: int) -> torch.Tensor:
    grad_output = grad_output.float()
    output = output.float()
    s = (output * grad_output).sum(dim=dim, keepdim=True)
    return output * (grad_output - s)


def _torch_quantised_for(dtype: ttnn.DataType, t: torch.Tensor) -> torch.Tensor:
    if dtype == ttnn.bfloat16:
        return t.to(torch.bfloat16).to(torch.float32)
    return t


def _to_device(t: torch.Tensor, device, dtype: ttnn.DataType) -> ttnn.Tensor:
    if dtype == ttnn.bfloat16:
        host_t = t.to(torch.bfloat16)
    else:
        host_t = t
    return ttnn.from_torch(
        host_t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# Per-dtype tolerance bands cribbed from test_backward_softmax_dtype.py — they
# represent the *floor* the op delivers, not a config-dependent tolerance.
# A lower-fidelity override still has to hit these because the matmul reduce
# path dominates the precision floor on Wormhole B0 regardless of the FPU
# fidelity setting (see numerical_stability.md).
_DTYPE_TOLERANCES = {
    ttnn.float32: (0.999, 0.01),
    ttnn.bfloat16: (0.999, 0.05),
    ttnn.bfloat8_b: (0.95, 0.15),
}
_DTYPE_IDS = {
    ttnn.float32: "float32",
    ttnn.bfloat16: "bfloat16",
    ttnn.bfloat8_b: "bfloat8_b",
}


def _assert_within_dtype_band(actual: torch.Tensor, expected: torch.Tensor, dtype: ttnn.DataType, tag: str) -> None:
    pcc_threshold, rms_rel_threshold = _DTYPE_TOLERANCES[dtype]
    pcc_ok, pcc_msg = check_with_pcc(expected, actual, pcc=pcc_threshold)
    rms_rel = (actual - expected).pow(2).mean().sqrt().item() / max(expected.pow(2).mean().sqrt().item(), 1e-12)
    assert pcc_ok, f"[{tag}] PCC failed: {pcc_msg} (rms_rel={rms_rel:.4f})"
    assert rms_rel <= rms_rel_threshold, f"[{tag}] rms_rel={rms_rel:.4f} > {rms_rel_threshold:.4f}"


# -----------------------------------------------------------------------------
# 1. Default (compute_kernel_config=None) — bit-identical to R3
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
def test_backward_softmax_default_config_unchanged(device, dtype):
    """
    Passing ``compute_kernel_config=None`` (the new default) must produce
    bit-identical output to omitting the kwarg entirely — proves the R4
    default path is byte-for-byte the R3 path.
    """
    torch.manual_seed(0)
    shape = (1, 1, 64, 128)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    ttnn_dy = _to_device(torch_dy, device, dtype)
    ttnn_y = _to_device(torch_y, device, dtype)

    out_omit = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=-1)).float()
    out_none = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=-1, compute_kernel_config=None)).float()
    out_none_unpack = ttnn.to_torch(
        backward_softmax(ttnn_dy, ttnn_y, dim=-1, compute_kernel_config=None, unpack_to_dest_mode=None)
    ).float()

    assert torch.equal(out_omit, out_none), f"[{_DTYPE_IDS[dtype]}] omit vs None drifted"
    assert torch.equal(out_omit, out_none_unpack), f"[{_DTYPE_IDS[dtype]}] None+None vs omit drifted"


# -----------------------------------------------------------------------------
# 2. Per-dtype caller overrides — numerical correctness preserved
# -----------------------------------------------------------------------------


# The throughput-first regime callers would pick when they don't care about
# the precision floor and want max DST capacity and fewest FPU cycles.
_THROUGHPUT_CONFIG_PER_DTYPE = {
    ttnn.float32: ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    ),
    # The explicit case the op_requirements ask names: HiFi2 + fp32_dest_acc
    # + math_approx=True on bf16.
    ttnn.bfloat16: ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    ),
    ttnn.bfloat8_b: ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    ),
}


@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 32, 256), id="multi_tile_W"),
        pytest.param((1, 1, 64, 128), id="non_square_64x128"),
    ],
)
def test_backward_softmax_compute_config_per_dtype_overrides(device, dtype, shape):
    """
    A throughput-first ``compute_kernel_config`` per dtype must still satisfy
    the per-dtype PCC/rms_rel band. This is the test the op_requirements
    explicitly asked for (HiFi2 + fp32_dest_acc + math_approx=True on bf16).
    """
    torch.manual_seed(42)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    expected = _torch_reference(_torch_quantised_for(dtype, torch_dy), _torch_quantised_for(dtype, torch_y), dim=-1)

    ttnn_dy = _to_device(torch_dy, device, dtype)
    ttnn_y = _to_device(torch_y, device, dtype)

    cfg = _THROUGHPUT_CONFIG_PER_DTYPE[dtype]
    ttnn_grad_input = backward_softmax(ttnn_dy, ttnn_y, dim=-1, compute_kernel_config=cfg)

    actual = ttnn.to_torch(ttnn_grad_input).float()
    _assert_within_dtype_band(actual, expected, dtype, f"{_DTYPE_IDS[dtype]} shape={shape} override")


# -----------------------------------------------------------------------------
# 3. math_fidelity=Invalid fallback — no-arg WormholeComputeKernelConfig works
# -----------------------------------------------------------------------------


def test_backward_softmax_compute_config_no_arg_default_construct_works(device):
    """
    ``ttnn.WormholeComputeKernelConfig()`` constructs with
    ``math_fidelity = Invalid`` (per the binding default). The descriptor
    must substitute the dtype-default fidelity rather than passing Invalid
    through to JIT (which would either crash or produce nonsense).

    We exercise this with bf16 because the dtype-default for bf16 is HiFi2,
    so a working result confirms the fallback actually fired.
    """
    torch.manual_seed(1)
    shape = (1, 1, 32, 128)
    dtype = ttnn.bfloat16
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    expected = _torch_reference(_torch_quantised_for(dtype, torch_dy), _torch_quantised_for(dtype, torch_y), dim=-1)

    ttnn_dy = _to_device(torch_dy, device, dtype)
    ttnn_y = _to_device(torch_y, device, dtype)

    cfg = ttnn.WormholeComputeKernelConfig()
    assert cfg.math_fidelity == ttnn.MathFidelity.Invalid, "binding default unexpectedly changed; update the test"

    out = backward_softmax(ttnn_dy, ttnn_y, dim=-1, compute_kernel_config=cfg)
    actual = ttnn.to_torch(out).float()
    _assert_within_dtype_band(actual, expected, dtype, "bf16 no-arg config (Invalid fidelity fallback)")


# -----------------------------------------------------------------------------
# 4. unpack_to_dest_mode override — propagates and stays correct
# -----------------------------------------------------------------------------


def test_backward_softmax_unpack_to_dest_mode_all_default_matches_no_override(device):
    """
    Explicitly passing ``[UnpackToDestMode.Default] * 32`` must produce the
    same result as omitting ``unpack_to_dest_mode`` (the default state).
    Sanity check that the override doesn't smuggle in a different code path
    when the values are nominally identical.
    """
    torch.manual_seed(2)
    shape = (1, 1, 32, 128)
    dtype = ttnn.float32
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    ttnn_dy = _to_device(torch_dy, device, dtype)
    ttnn_y = _to_device(torch_y, device, dtype)

    out_no_override = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=-1)).float()
    all_default = [ttnn.UnpackToDestMode.Default] * 32
    out_explicit = ttnn.to_torch(backward_softmax(ttnn_dy, ttnn_y, dim=-1, unpack_to_dest_mode=all_default)).float()
    assert torch.equal(
        out_no_override, out_explicit
    ), "explicit [Default]*32 unpack_to_dest_mode differs from omitting the kwarg"


@pytest.mark.parametrize(
    "cb_index,cb_name",
    [
        (0, "CB_GRAD_OUTPUT"),
        (1, "CB_OUTPUT"),
        (24, "CB_PROD"),
    ],
)
def test_backward_softmax_unpack_to_dest_fp32_on_matmul_input_cb_breaks(device, cb_index, cb_name):
    """
    Empirical regression guard for the verifier note in op_requirements.md:
    ``UnpackToDestFp32`` is **incompatible** with any CB whose tiles are
    unpacked into SrcA/SrcB for the FPU. In backward_softmax that's every
    CB on the read path — ``CB_GRAD_OUTPUT`` (read by mul + sub via FPU),
    ``CB_OUTPUT`` (read by mul + mul via FPU), and ``CB_PROD`` (read by the
    matmul-based REDUCE_ROW SUM, the canonical breakage the verifier
    flagged).

    Without this test the caller might naively scatter ``UnpackToDestFp32``
    across all fp32 CBs trying to tighten precision, expecting it to be a
    no-op or improvement; instead it silently produces garbage (inf /
    overflow). The test pins the contract by *expecting* a numerical failure
    — if any of these CBs ever becomes safe under UnpackToDestFp32 (e.g.
    LLK changes), the test will start passing and we'll re-evaluate the
    docstring on the entry point.
    """
    torch.manual_seed(3)
    shape = (1, 1, 32, 128)
    dtype = ttnn.float32
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    expected = _torch_reference(torch_dy, torch_y, dim=-1)

    ttnn_dy = _to_device(torch_dy, device, dtype)
    ttnn_y = _to_device(torch_y, device, dtype)

    modes = [ttnn.UnpackToDestMode.Default] * 32
    modes[cb_index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    out = backward_softmax(ttnn_dy, ttnn_y, dim=-1, unpack_to_dest_mode=modes)
    actual = ttnn.to_torch(out).float()

    # Compute the same metric the band check uses, but assert it FAILS.
    pcc_threshold, _ = _DTYPE_TOLERANCES[dtype]
    pcc_ok, _ = check_with_pcc(expected, actual, pcc=pcc_threshold)
    assert not pcc_ok, (
        f"UnpackToDestFp32 on {cb_name} (index {cb_index}) is no longer "
        "destructive — re-evaluate the entry-point docstring's warning."
    )


# -----------------------------------------------------------------------------
# 5. Validation: malformed unpack_to_dest_mode rejected
# -----------------------------------------------------------------------------


def test_backward_softmax_unpack_to_dest_mode_wrong_length_rejected(device):
    """A length other than 32 must be rejected at descriptor construction."""
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    ttnn_dy = _to_device(torch_dy, device, ttnn.float32)
    ttnn_y = _to_device(torch_y, device, ttnn.float32)

    # 31 entries — short by one.
    too_short = [ttnn.UnpackToDestMode.Default] * 31
    with pytest.raises(ValueError, match="length-32"):
        backward_softmax(ttnn_dy, ttnn_y, dim=-1, unpack_to_dest_mode=too_short)


def test_backward_softmax_unpack_to_dest_mode_wrong_element_type_rejected(device):
    """Non-UnpackToDestMode elements must be rejected (catches str/int slips)."""
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    torch_dy = torch.randn(shape, dtype=torch.float32)
    torch_y = torch.randn(shape, dtype=torch.float32)
    ttnn_dy = _to_device(torch_dy, device, ttnn.float32)
    ttnn_y = _to_device(torch_y, device, ttnn.float32)

    bogus = [ttnn.UnpackToDestMode.Default] * 32
    bogus[5] = "default"  # type: ignore[list-item]
    with pytest.raises(ValueError, match="UnpackToDestMode"):
        backward_softmax(ttnn_dy, ttnn_y, dim=-1, unpack_to_dest_mode=bogus)


# -----------------------------------------------------------------------------
# 6. Descriptor introspection — fields propagate
# -----------------------------------------------------------------------------


def test_backward_softmax_compute_config_fields_propagate(device):
    """
    Build the program descriptor directly and inspect the
    ``ComputeConfigDescriptor`` on the compute kernel. Each
    ``WormholeComputeKernelConfig`` field the spec asks us to expose must
    end up on the descriptor.
    """
    torch.manual_seed(0)
    shape = (1, 1, 32, 64)
    dtype = ttnn.float32
    torch_t = torch.randn(shape, dtype=torch.float32)
    ttnn_dy = _to_device(torch_t, device, dtype)
    ttnn_y = _to_device(torch_t, device, dtype)
    ttnn_gi = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        dst_full_sync_en=True,
    )
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 32
    unpack_modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32

    descriptor = create_program_descriptor(
        ttnn_dy,
        ttnn_y,
        ttnn_gi,
        dim=-1,
        compute_kernel_config=cfg,
        unpack_to_dest_mode=unpack_modes,
    )

    # The compute kernel is the third kernel in the descriptor (reader,
    # writer, compute — see create_program_descriptor's return).
    compute_kernel = descriptor.kernels[2]
    compute_config = compute_kernel.config

    assert compute_config.math_fidelity == ttnn.MathFidelity.HiFi3, "math_fidelity did not propagate"
    assert compute_config.math_approx_mode is True, "math_approx_mode did not propagate"
    assert compute_config.fp32_dest_acc_en is False, "fp32_dest_acc_en did not propagate"
    assert compute_config.dst_full_sync_en is True, "dst_full_sync_en did not propagate"
    # unpack_to_dest_mode is exposed as a vector binding; index into it.
    propagated_mode = compute_config.unpack_to_dest_mode[0]
    assert propagated_mode == ttnn.UnpackToDestMode.UnpackToDestFp32, "unpack_to_dest_mode[0] did not propagate"


def test_backward_softmax_compute_config_default_falls_back_to_dtype_aware(device):
    """
    When ``compute_kernel_config=None``, the descriptor must use the
    R3 dtype-aware default. Per-dtype: fp32→HiFi4+fp32_dest_acc,
    bf16→HiFi2+fp32_dest_acc=False, bfp8→LoFi+fp32_dest_acc=False.
    """
    torch.manual_seed(0)
    shape = (1, 1, 32, 64)
    torch_t = torch.randn(shape, dtype=torch.float32)

    expected_per_dtype = {
        ttnn.float32: (ttnn.MathFidelity.HiFi4, True),
        ttnn.bfloat16: (ttnn.MathFidelity.HiFi2, False),
        ttnn.bfloat8_b: (ttnn.MathFidelity.LoFi, False),
    }

    for dtype, (expect_fid, expect_fp32_acc) in expected_per_dtype.items():
        ttnn_dy = _to_device(torch_t, device, dtype)
        ttnn_y = _to_device(torch_t, device, dtype)
        ttnn_gi = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)),
            dtype,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        descriptor = create_program_descriptor(ttnn_dy, ttnn_y, ttnn_gi, dim=-1)
        compute_config = descriptor.kernels[2].config
        assert (
            compute_config.math_fidelity == expect_fid
        ), f"[{_DTYPE_IDS[dtype]}] default fidelity {compute_config.math_fidelity} != expected {expect_fid}"
        assert compute_config.fp32_dest_acc_en == expect_fp32_acc, (
            f"[{_DTYPE_IDS[dtype]}] default fp32_dest_acc_en {compute_config.fp32_dest_acc_en} "
            f"!= expected {expect_fp32_acc}"
        )


def test_backward_softmax_compute_config_invalid_fidelity_falls_back(device):
    """
    A user-supplied config with ``math_fidelity = Invalid`` (the binding
    default for the no-arg constructor) must fall back to the dtype-default
    fidelity on the resulting descriptor — neither passing Invalid through
    nor mutating the caller's object.
    """
    torch.manual_seed(0)
    shape = (1, 1, 32, 64)
    dtype = ttnn.bfloat16
    torch_t = torch.randn(shape, dtype=torch.float32)
    ttnn_dy = _to_device(torch_t, device, dtype)
    ttnn_y = _to_device(torch_t, device, dtype)
    ttnn_gi = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    cfg = ttnn.WormholeComputeKernelConfig()  # math_fidelity defaults to Invalid
    assert cfg.math_fidelity == ttnn.MathFidelity.Invalid

    descriptor = create_program_descriptor(
        ttnn_dy,
        ttnn_y,
        ttnn_gi,
        dim=-1,
        compute_kernel_config=cfg,
    )
    compute_config = descriptor.kernels[2].config

    # Should be HiFi2 (bf16 dtype-default), NOT Invalid.
    assert (
        compute_config.math_fidelity == ttnn.MathFidelity.HiFi2
    ), f"Invalid fidelity not replaced with dtype-default; got {compute_config.math_fidelity}"
    # Caller's config object must not have been mutated.
    assert cfg.math_fidelity == ttnn.MathFidelity.Invalid, "caller's WormholeComputeKernelConfig was mutated"
