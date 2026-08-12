# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from math import isnan
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, assert_equal


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_mac_all_tensors(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand((h, w), dtype=torch.bfloat16)

    golden_fn = ttnn.get_golden_function(ttnn.mac)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn.mac(input_tensor, input_tensor1, input_tensor2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("scalar1", [5.5])
@pytest.mark.parametrize("scalar2", [-13.25])
def test_mac_tensor_with_2_scalaras(device, h, w, scalar1, scalar2):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = scalar1
    torch_input_tensor2 = scalar2

    golden_fn = ttnn.get_golden_function(ttnn.mac)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.mac(input_tensor, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=2)


def assert_where_exact(torch_input_tensor, torch_input1, torch_input2, device):
    """Run ``ttnn.where`` and assert bit-exact equality with the torch golden.

    ``where`` is a per-element selection between the two input branches with no arithmetic, so the
    output must match torch exactly.
    """

    def from_torch_if_tensor(x):
        if not isinstance(x, torch.Tensor):
            return x

        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor, input1, input2 = (
        from_torch_if_tensor(arg)
        for arg in ((torch_input_tensor > 0).to(torch_input_tensor.dtype), torch_input1, torch_input2)
    )
    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(torch_input_tensor > 0, torch_input1, torch_input2)
    output_tensor = ttnn.where(input_tensor, input1, input2)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "hc, ht, hf, wc, wt, wf",
    [
        [64, 64, 64, 128, 128, 1],
        [64, 64, 64, 128, 1, 128],
        [64, 64, 64, 1, 128, 128],
        [64, 64, 1, 128, 128, 128],
        [64, 1, 64, 128, 128, 128],
        [1, 64, 64, 128, 128, 128],
        [64, 64, 1, 128, 128, 1],
        [64, 1, 64, 128, 1, 128],
        [64, 1, 64, 128, 128, 1],
        [64, 64, 1, 128, 1, 128],
        [1, 1, 64, 128, 128, 1],
        [64, 1, 64, 1, 1, 128],
    ],
)
def test_where_bcast(device, dtype, hc, ht, hf, wc, wt, wf):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hc, wc), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((ht, wt), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor2 = torch.rand((hf, wf), dtype=dtype).uniform_(-100, 100)

    assert_where_exact(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, device)


def run_ternary_test_value(device, h, w, value, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)
    torch_input_tensor2 = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn_function(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("value", [15.5])
def test_addcdiv(device, h, w, value):
    run_ternary_test_value(device, h, w, value, ttnn.addcdiv)


@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("value", [15.5, 0, 1.0, -5.0])
@pytest.mark.parametrize(
    "hc, ht, hf, wc, wt, wf",
    [
        [64, 64, 64, 128, 128, 128],  # no bcast
        # Row / Col bcast cases
        [64, 64, 64, 128, 128, 1],
        [64, 64, 64, 128, 1, 128],
        [64, 64, 64, 1, 128, 128],
        [64, 64, 1, 128, 128, 128],
        [64, 1, 64, 128, 128, 128],
        [1, 64, 64, 128, 128, 128],
        [64, 64, 1, 128, 128, 1],
        [64, 1, 64, 128, 1, 128],
        [1, 64, 64, 1, 128, 128],
        [64, 1, 1, 128, 1, 1],  # scalar bcast case
    ],
)
def test_addcmul_with_bcast(device, tor_dtype, ttnn_dtype, hc, ht, hf, wc, wt, wf, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hc, wc), dtype=tor_dtype).uniform_(-100, 500)
    torch_input_tensor1 = torch.rand((ht, wt), dtype=tor_dtype).uniform_(-200, 200)
    torch_input_tensor2 = torch.rand((hf, wf), dtype=tor_dtype).uniform_(-300, 400)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn.addcmul(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.0,
        -0.5,
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        ((1, 2, 1088, 1024), (1, 2, 1, 1024), (1, 2, 1088, 1024)),  # Composite
        ((1, 2, 1088, 1024), (1, 2, 1, 1), (1, 2, 1088, 1024)),  # Composite
        ((4, 2, 1088, 1024), (1, 2, 1088, 1024), (1, 1, 1088, 1024)),  # HLK
    ],
)
def test_addcmul_with_bcast_bf8b(device, torch_dtype, ttnn_dtype, a_shape, b_shape, c_shape, value):
    """
    Test addcmul: Block format datatype inputs with subtile broadcast use composite Addcmul implementation.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(a_shape, dtype=torch_dtype)
    torch_input_tensor1 = torch.randn(b_shape, dtype=torch_dtype)
    torch_input_tensor2 = torch.randn(c_shape, dtype=torch_dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("value", [0.5, 3])
@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcmul(device, torch_dtype, ttnn_dtype, value, in_data1_shape, in_data2_shape, in_data3_shape):
    in_data1 = torch.full(in_data1_shape, 0.0031, dtype=torch_dtype)
    in_data2 = torch.full(in_data2_shape, 508.0, dtype=torch_dtype)
    in_data3 = torch.full(in_data3_shape, 748.0, dtype=torch_dtype)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_with_ulp(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcmul_with_int32_inputs(device, in_data1_shape, in_data2_shape, in_data3_shape):
    in_data1 = torch.randint(-100, 100, in_data1_shape, dtype=torch.int32)
    in_data2 = torch.randint(-100, 100, in_data2_shape, dtype=torch.int32)
    in_data3 = torch.randint(-100, 100, in_data3_shape, dtype=torch.int32)
    value = 6
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1), (1, 1, 32, 32), (1, 1, 1, 1)),
    ],
)
@pytest.mark.parametrize("value", [6, -6.0, float(2**31 + 7), 2**32 - 1])
def test_addcmul_with_uint32_inputs(device, in_data1_shape, in_data2_shape, in_data3_shape, value):
    # Use values spanning the full uint32 range. Store them as int32 bit-patterns;
    # TTNN interprets those bit-patterns as uint32, and the golden is compared via 32-bit wraparound.
    in_data1 = torch.randint(0, 2**32, in_data1_shape, dtype=torch.int64).to(torch.int32)
    in_data2 = torch.randint(0, 2**16, in_data2_shape, dtype=torch.int64).to(torch.int32)
    in_data3 = torch.randint(0, 2**16, in_data3_shape, dtype=torch.int64).to(torch.int32)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    # Float values are narrowed to float32 by the C++ binding, so apply the same rounding
    # to the golden to keep both sides in sync (e.g. float(2**31+7) → 2147483648.0 in float32).
    golden_value = float(torch.tensor(value, dtype=torch.float32).item()) if isinstance(value, float) else value

    # Promote to int64 before calling the golden function to prevent intermediate overflow,
    # then truncate back to int32 to match the 32-bit wraparound semantics of uint32 hardware.
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(
        in_data1.to(torch.int64), in_data2.to(torch.int64), in_data3.to(torch.int64), value=golden_value
    ).to(torch.int32)

    assert_equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("value", [0.5, 1.0, 7.7, 22.42, 107.6])
@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcdiv(device, torch_dtype, ttnn_dtype, value, in_data1_shape, in_data2_shape, in_data3_shape):
    torch.manual_seed(10)
    in_data1 = torch.empty(in_data1_shape, dtype=torch_dtype).uniform_(-100, 100)
    in_data2 = torch.empty(in_data2_shape, dtype=torch_dtype).uniform_(-100, 100)
    in_data3 = torch.empty(in_data3_shape, dtype=torch_dtype).uniform_(-100, 100)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcdiv(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcdiv)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    # Normalize: if input3 is zero and golden is nan and ttnn output is inf, change ttnn output to nan
    zero_mask = in_data3 == 0
    golden_nan_mask = torch.isnan(golden_tensor)
    output_inf_mask = torch.isinf(output_tensor)
    normalize_mask = zero_mask & golden_nan_mask & output_inf_mask
    if normalize_mask.any():
        output_tensor = torch.where(
            normalize_mask,
            torch.tensor(float("nan"), dtype=output_tensor.dtype, device=output_tensor.device),
            output_tensor,
        )

    assert_with_ulp(output_tensor, golden_tensor, ulp_threshold=1, allow_nonfinite=True)


def test_ternary_scalar_distinguishes_cache_entries(device):
    """Regression guard for the ternary scalar static/dynamic contract.

    addcmul(a, b, c, value) packs `value` (scalar_input_a) into the compute-kernel runtime args.
    The scalar is intentionally EXCLUDED from to_hash()/compute_program_hash (so calls differing
    only in the scalar cache-hit instead of recompiling) but must be re-applied to the cached
    program on every dispatch via TernaryDeviceOperation::get_dynamic_runtime_args(). Pins both
    halves of the contract:
      * a different scalar must NOT grow the cache  -> guards against re-adding the scalar to the
        hash (the recompile-per-scalar hack).
      * a different scalar must change the output   -> guards against the frozen-runtime-arg bug,
        where the cache hit reuses the first call's baked-in scalar (only correctly re-patched now
        that the factory registers buffer bindings and declares the scalar dynamic).
      * the same scalar must reproduce the output   -> guards against wrongly perturbing the scalar
        on the fast path.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    shape = (256, 256)
    torch.manual_seed(0)
    a = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    b = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    c = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    out_v1 = ttnn.to_torch(ttnn.addcmul(a, b, c, value=2.0)).float().clone()
    entries_after_first = device.num_program_cache_entries()
    assert entries_after_first == 1, f"Expected 1 cache entry after first addcmul, got {entries_after_first}"

    # Same scalar: must reuse the cached program AND reproduce the exact values.
    out_v1_again = ttnn.to_torch(ttnn.addcmul(a, b, c, value=2.0)).float().clone()
    assert device.num_program_cache_entries() == 1, "same scalar must reuse the cached program (no new cache entry)"
    assert torch.equal(out_v1, out_v1_again), "same scalar must reproduce identical output"

    # Different scalar: scalar is excluded from the hash, so it must STILL be a cache hit (no new
    # entry) and yet produce different values (scalar re-patched on the cache hit).
    out_v2 = ttnn.to_torch(ttnn.addcmul(a, b, c, value=7.0)).float().clone()
    assert device.num_program_cache_entries() == 1, (
        "a different scalar must NOT create a new cache entry -- the scalar is dynamic, not part of "
        "the program hash. A new entry here means the scalar was (re-)added to compute_program_hash."
    )
    assert not torch.equal(out_v1, out_v2), (
        "a different scalar must change the output (scalar re-patched on the fast path). Identical "
        "output here means the cache hit reused the first call's frozen scalar runtime arg."
    )

    device.disable_and_clear_program_cache()


def test_ternary_addcmul_cache_hit_refreshes_operand_addresses(device):
    """Regression guard for the ternary buffer-address static/dynamic contract (cache-hit bug).

    Two same-shaped addcmul calls reuse the same cached program (correct), but the cache-hit path
    must re-patch the runtime args holding the operand/output buffer ADDRESSES. Before the fix, the
    second same-shaped call whose operand lived in a DIFFERENT buffer ran against the first call's
    stale addresses and returned silent garbage (PCC ~0). Here the operands are distinct
    ttnn.chunk() slices of a single parent tensor, so they are the same shape but different buffers.
    TernaryDeviceOperation::get_dynamic_runtime_args() must refresh reader slots 0/1/2 and writer
    slot 0 on every dispatch.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    torch.manual_seed(0)
    N, H = 32, 64

    base_t = torch.rand((1, 1, N, H), dtype=torch.bfloat16)
    chunks_t = torch.rand((1, 1, 1, 4 * H), dtype=torch.bfloat16)

    base = ttnn.from_torch(base_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    chunks = ttnn.from_torch(chunks_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    chunk = ttnn.chunk(chunks, 4, -1)  # 4 shape-identical operands in different buffers

    def reference(i):
        return base_t + base_t * chunks_t[..., i * H : (i + 1) * H]

    # Pre-allocate BOTH outputs and keep them alive across both calls, passing each via
    # output_tensor. This forces the two dispatches to write to DISTINCT, simultaneously-live output
    # buffers, so the second call's writer slot 0 must be re-patched to out1's address -- otherwise it
    # writes into out0 and out1 stays zero. Without two live outputs the allocator could hand the
    # second call out0's just-freed address, masking a missing writer-address patch.
    out0 = ttnn.from_torch(
        torch.zeros((1, 1, N, H), dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    out1 = ttnn.from_torch(
        torch.zeros((1, 1, N, H), dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    # First call compiles the addcmul program.
    ttnn.addcmul(base, base, chunk[0], value=1.0, output_tensor=out0)
    entries_after_first = device.num_program_cache_entries()

    # Cache hit with a different-buffer operand AND a different output buffer of identical shape: the
    # addcmul program must be reused (no new cache entry), yet the stale-address regression made this
    # return ~0 PCC.
    ttnn.addcmul(base, base, chunk[2], value=1.0, output_tensor=out1)
    assert (
        device.num_program_cache_entries() == entries_after_first
    ), "same-shaped addcmul must reuse the cached program (no new cache entry)"

    assert_with_pcc(reference(0), ttnn.to_torch(out0).float(), 0.99)
    assert_with_pcc(reference(2), ttnn.to_torch(out1).float(), 0.99)

    device.disable_and_clear_program_cache()
