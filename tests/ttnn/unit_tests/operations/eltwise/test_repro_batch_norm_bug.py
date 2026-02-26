"""
Reproducer for batch_norm program cache collision.

Two bugs:
  (1) Dtype: run bf16 then fp32 back-to-back; hash must include dtype.
  (2) Weight/bias presence: run WITH weight+bias then WITHOUT (same shape/dtype);
      hash must include weight_has_value and bias_has_value so different programs are used.

Why (2) currently creates 2 hashes (no collision): compute_program_hash uses
get_optional_tensor_info(weight) and get_optional_tensor_info(bias), which return
  - when present: (optional(dtype), optional(memory_config))
  - when absent:  (nullopt, nullopt)
So the tuple passed to hash_operation differs and we get two cache entries. The audit
worried about explicit weight_has_value/bias_has_value; in practice the optional
(dtype, memory_config) already encodes presence (Some vs nullopt), so the hash differs.
"""
import pytest
import torch
import ttnn


def _run_batch_norm(device, input_shape, dtype=torch.float32):
    N, C, H, W = input_shape
    torch.manual_seed(42)
    inp = torch.rand(N, C, H, W, dtype=dtype) * 2 - 1
    mean = torch.rand(C, dtype=dtype).view(1, C, 1, 1)
    var = torch.rand(C, dtype=dtype).abs().view(1, C, 1, 1) + 0.1

    ttnn_dtype = ttnn.float32 if dtype == torch.float32 else ttnn.bfloat16
    inp_tt = ttnn.from_torch(inp, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    mean_tt = ttnn.from_torch(mean, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    var_tt = ttnn.from_torch(var, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    weight = torch.ones(C, dtype=dtype).view(1, C, 1, 1)
    bias = torch.zeros(C, dtype=dtype).view(1, C, 1, 1)
    weight_tt = ttnn.from_torch(weight, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    bias_tt = ttnn.from_torch(bias, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    golden_1 = torch.nn.functional.batch_norm(
        inp, mean.view(C), var.view(C), weight=weight.view(C), bias=bias.view(C), eps=1e-5
    )
    golden_2 = torch.nn.functional.batch_norm(inp, mean.view(C), var.view(C), eps=1e-5)
    golden_3 = torch.nn.functional.batch_norm(inp, mean.view(C), var.view(C), weight=weight.view(C), eps=1e-5)
    golden_4 = torch.nn.functional.batch_norm(inp, mean.view(C), var.view(C), bias=bias.view(C), eps=1e-5)

    out_tt_1 = ttnn.batch_norm(
        inp_tt, running_mean=mean_tt, running_var=var_tt, eps=1e-5, weight=weight_tt, bias=bias_tt
    )
    result_1 = ttnn.to_torch(out_tt_1)
    n_wrong_1 = (~torch.isclose(result_1, golden_1, atol=0.02, rtol=0.02)).sum().item()
    label_1 = f"batch_norm{list(input_shape)} dtype={dtype} call=1(with_wb)"
    if n_wrong_1 == 0:
        print(f"PASS  {label_1}")
    else:
        print(f"FAIL  {label_1}  —  {n_wrong_1}/{result_1.numel()} wrong")
    print(f"  program cache count: {device.num_program_cache_entries()}")

    out_tt_2 = ttnn.batch_norm(inp_tt, running_mean=mean_tt, running_var=var_tt, eps=1e-5)
    result_2 = ttnn.to_torch(out_tt_2)
    n_wrong_2 = (~torch.isclose(result_2, golden_2, atol=0.02, rtol=0.02)).sum().item()
    label_2 = f"batch_norm{list(input_shape)} dtype={dtype} call=2(no_wb)"
    if n_wrong_2 == 0:
        print(f"PASS  {label_2}")
    else:
        print(f"FAIL  {label_2}  —  {n_wrong_2}/{result_2.numel()} wrong")
    print(f"  program cache count: {device.num_program_cache_entries()}")

    out_tt_3 = ttnn.batch_norm(inp_tt, running_mean=mean_tt, running_var=var_tt, eps=1e-5, weight=weight_tt)
    result_3 = ttnn.to_torch(out_tt_3)
    n_wrong_3 = (~torch.isclose(result_3, golden_3, atol=0.02, rtol=0.02)).sum().item()
    label_3 = f"batch_norm{list(input_shape)} dtype={dtype} call=3(weight_only)"
    if n_wrong_3 == 0:
        print(f"PASS  {label_3}")
    else:
        print(f"FAIL  {label_3}  —  {n_wrong_3}/{result_3.numel()} wrong")
    print(f"  program cache count: {device.num_program_cache_entries()}")

    out_tt_4 = ttnn.batch_norm(inp_tt, running_mean=mean_tt, running_var=var_tt, eps=1e-5, bias=bias_tt)
    result_4 = ttnn.to_torch(out_tt_4)
    n_wrong_4 = (~torch.isclose(result_4, golden_4, atol=0.02, rtol=0.02)).sum().item()
    label_4 = f"batch_norm{list(input_shape)} dtype={dtype} call=4(bias_only)"
    if n_wrong_4 == 0:
        print(f"PASS  {label_4}")
    else:
        print(f"FAIL  {label_4}  —  {n_wrong_4}/{result_4.numel()} wrong")
    print(f"  program cache count: {device.num_program_cache_entries()}")

    return n_wrong_1 + n_wrong_2 + n_wrong_3 + n_wrong_4


def _make_tensors(shape, dtype, ttnn_dtype, device):
    N, C, H, W = shape
    torch.manual_seed(42)
    inp = torch.rand(N, C, H, W, dtype=dtype) * 2 - 1
    mean = torch.rand(C, dtype=dtype).view(1, C, 1, 1)
    var = torch.rand(C, dtype=dtype).abs().view(1, C, 1, 1) + 0.1
    weight = torch.ones(C, dtype=dtype).view(1, C, 1, 1)
    bias = torch.zeros(C, dtype=dtype).view(1, C, 1, 1)

    inp_tt = ttnn.from_torch(inp, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    mean_tt = ttnn.from_torch(mean, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    var_tt = ttnn.from_torch(var, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    weight_tt = ttnn.from_torch(weight, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    bias_tt = ttnn.from_torch(bias, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return inp, mean, var, weight, bias, inp_tt, mean_tt, var_tt, weight_tt, bias_tt


def _check(result, golden, label, device):
    n_wrong = (~torch.isclose(result, golden, atol=0.02, rtol=0.02)).sum().item()
    if n_wrong == 0:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}  —  {n_wrong}/{result.numel()} wrong")
    print(f"  program cache count: {device.num_program_cache_entries()}")
    return n_wrong


def _run_batch_norm_different_shapes(device, dtype=torch.float32):
    ttnn_dtype = ttnn.float32 if dtype == torch.float32 else ttnn.bfloat16
    total_wrong = 0

    # Call 1: weight + bias — shape (1, 4, 32, 32)
    shape_1 = (1, 4, 32, 32)
    inp, mean, var, weight, bias, inp_tt, mean_tt, var_tt, weight_tt, bias_tt = _make_tensors(
        shape_1, dtype, ttnn_dtype, device
    )
    C = shape_1[1]
    out_tt = ttnn.batch_norm(inp_tt, running_mean=mean_tt, running_var=var_tt, eps=1e-5, weight=weight_tt, bias=bias_tt)
    golden = torch.nn.functional.batch_norm(
        inp, mean.view(C), var.view(C), weight=weight.view(C), bias=bias.view(C), eps=1e-5
    )
    total_wrong += _check(
        ttnn.to_torch(out_tt), golden, f"batch_norm{list(shape_1)} dtype={dtype} call=1(with_wb)", device
    )

    # Call 2: no weight, no bias — shape (2, 8, 64, 64)
    shape_2 = (2, 8, 64, 64)
    inp, mean, var, weight, bias, inp_tt, mean_tt, var_tt, weight_tt, bias_tt = _make_tensors(
        shape_2, dtype, ttnn_dtype, device
    )
    C = shape_2[1]
    out_tt = ttnn.batch_norm(inp_tt, running_mean=mean_tt, running_var=var_tt, eps=1e-5)
    golden = torch.nn.functional.batch_norm(inp, mean.view(C), var.view(C), eps=1e-5)
    total_wrong += _check(
        ttnn.to_torch(out_tt), golden, f"batch_norm{list(shape_2)} dtype={dtype} call=2(no_wb)", device
    )

    # Call 3: weight only — shape (4, 16, 32, 64)
    shape_3 = (4, 16, 32, 64)
    inp, mean, var, weight, bias, inp_tt, mean_tt, var_tt, weight_tt, bias_tt = _make_tensors(
        shape_3, dtype, ttnn_dtype, device
    )
    C = shape_3[1]
    out_tt = ttnn.batch_norm(inp_tt, running_mean=mean_tt, running_var=var_tt, eps=1e-5, weight=weight_tt)
    golden = torch.nn.functional.batch_norm(inp, mean.view(C), var.view(C), weight=weight.view(C), eps=1e-5)
    total_wrong += _check(
        ttnn.to_torch(out_tt), golden, f"batch_norm{list(shape_3)} dtype={dtype} call=3(weight_only)", device
    )

    # Call 4: bias only — shape (1, 32, 64, 32)
    shape_4 = (1, 32, 64, 32)
    inp, mean, var, weight, bias, inp_tt, mean_tt, var_tt, weight_tt, bias_tt = _make_tensors(
        shape_4, dtype, ttnn_dtype, device
    )
    C = shape_4[1]
    out_tt = ttnn.batch_norm(inp_tt, running_mean=mean_tt, running_var=var_tt, eps=1e-5, bias=bias_tt)
    golden = torch.nn.functional.batch_norm(inp, mean.view(C), var.view(C), bias=bias.view(C), eps=1e-5)
    total_wrong += _check(
        ttnn.to_torch(out_tt), golden, f"batch_norm{list(shape_4)} dtype={dtype} call=4(bias_only)", device
    )

    return total_wrong


@pytest.mark.use_module_device
def test_batch_norm_cache_collision_dtype(device):
    """Bug 1: Run bf16 then fp32 back-to-back; hash must include dtype."""
    shape = (2, 4, 32, 32)
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    n1 = _run_batch_norm(device, shape, dtype=torch.bfloat16)
    n2 = _run_batch_norm(device, shape, dtype=torch.float32)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    assert n1 == 0 and n2 == 0, "batch_norm dtype cache collision: one or both cases failed"


@pytest.mark.use_module_device
def test_batch_norm_cache_collision_weight_bias(device):
    """Bug 2: Run WITH weight+bias then WITHOUT (same shape/dtype); hash must include weight_has_value/bias_has_value."""
    shape = (2, 4, 32, 32)
    dtype = torch.float32
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    # First: with weight and bias (populates cache)
    n1 = _run_batch_norm(device, shape, dtype=dtype)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    assert n1 == 0, "batch_norm weight/bias cache collision: one or both calls failed"


@pytest.mark.use_module_device
def test_batch_norm_different_shapes(device):
    """Run batch_norm with 4 different shapes, each with a different weight/bias combination."""
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    n = _run_batch_norm_different_shapes(device, dtype=torch.float32)
    print(f"program cache count (after): {device.num_program_cache_entries()}")
    assert n == 0, "batch_norm different shapes: one or more calls failed"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    # Bug 2: with vs without weight+bias -> 2 hashes because get_optional_tensor_info(weight/bias)
    # returns (dtype, mem_cfg) when present and (nullopt, nullopt) when absent (see batch_norm_device_operation.cpp).
    # print("=== Same shape, different weight/bias combos ===")
    # print(f"program cache count (before): {device.num_program_cache_entries()}")
    # _run_batch_norm(device, (2, 4, 32, 32), dtype=torch.float32)
    # print(f"program cache count (after): {device.num_program_cache_entries()}")

    print("\n=== Different shapes, different weight/bias combos ===")
    print(f"program cache count (before): {device.num_program_cache_entries()}")
    _run_batch_norm_different_shapes(device, dtype=torch.float32)
    print(f"program cache count (after): {device.num_program_cache_entries()}")

    # print("=== Bug 1: dtype (bf16 then fp32) ===")
    # print(f"program cache count (before): {device.num_program_cache_entries()}")
    # _run_batch_norm(device, (2, 4, 32, 32), dtype=torch.bfloat16)
    # _run_batch_norm(device, (2, 4, 32, 32), dtype=torch.float32)
    # print(f"program cache count (after): {device.num_program_cache_entries()}")

    ttnn.close_device(device)
