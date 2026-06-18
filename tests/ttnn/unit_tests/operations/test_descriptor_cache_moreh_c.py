# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Cache-correctness sweep for batch-C moreh ops + experimental matmul + tanh_bw (#46506).

Companion to test_descriptor_cache.py — same intent, same shape of assertions:

  * not stale        : every call's result matches the op's golden, even on a program-cache
                       HIT (a frozen per-call rt-arg — e.g. a tensor address not re-patched —
                       would make a later call wrong).
  * not over-caching : the op does not create a new program-cache entry for something it should
                       re-apply on hit (fresh input DATA at a fresh address must NOT rebuild).

Each test below COPIES its op's real invocation + torch reference VERBATIM from that op's own
nightly test file (test_moreh_nll_loss*.py / test_moreh_norm.py / test_moreh_sum.py /
matmul/test_attn_matmul.py) — no invented shapes/cases. We reuse those files' run_* helpers
where they exist, since each helper already builds fresh tensors and asserts correctness
internally (assert passing); we wrap them to bound program-cache entry growth.

HASH NOTE (important — drives the loop design):
  None of moreh_nll_loss(step1/step2), moreh_nll_loss_backward, moreh_nll_loss_unreduced_backward,
  moreh_norm_backward, moreh_sum_backward define a custom compute_program_hash. They therefore
  use the framework DEFAULT hash, which is
      hash_objects_with_default_seed(type_hash, operation_attributes, tensor_args)
  i.e. it reflects over the WHOLE operation_attributes_t struct. So the per-call scalars
  (reduction, ignore_index, p, dim, keepdim) ARE in the hash — varying them legitimately adds a
  cache entry and is NOT a bug (see the task NOTE). They are NOT frozen-args suspects, because a
  frozen scalar would only be a bug if baked into rt-args AND excluded from the hash, which is not
  the case here. So to probe the cache-HIT path we hold every operation_attribute FIXED and vary
  only the input DATA (fresh allocation -> fresh address) and bound the entry count on that loop.
  This is the same address-re-patch class that #46506 is about.

  attn_matmul / group_attn_matmul (the standard, non-from_cache calls used here) leave
  num_tokens/transpose_hw unset, so their explicit compute_program_hash reduces to shape/dtype/
  config — again, fix those and vary data. tanh_bw has a clean explicit hash and no per-call scalar.
"""

import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def cache_device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


# ===========================================================================
# moreh_nll_loss  (forward; internally dispatches moreh_nll_loss_step1 + step2)
# Copied from tests/ttnn/nightly/.../moreh/test_moreh_nll_loss.py::run_moreh_nll_loss
# (shape [5, 10], ignore_index=1, none_weight=False — verbatim case from test_moreh_nll_loss).
# ignore_index & reduction live in operation_attributes -> default-hashed -> held FIXED here;
# we vary input DATA across the loop. Each call's golden is the torch NLLLoss for that data.
# ===========================================================================
def test_moreh_nll_loss_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_nll_loss import run_moreh_nll_loss

    torch.manual_seed(0)
    shape = [5, 10]
    ignore_index = 1
    reduction = "mean"
    none_weight = False

    # run_moreh_nll_loss builds fresh torch+tt tensors and asserts pcc internally (not stale).
    # COMPOSITE op: the forward composes multiple distinct descriptor programs (step1 + step2), so
    # the first call settles the cache; later passes with the same config must add none.
    # WARM-UP then assert ZERO growth.
    run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, cache_device)
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        run_moreh_nll_loss(shape, ignore_index, reduction, none_weight, cache_device)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_nll_loss: cache grew past {base} across fresh data (over-caching / addr not patched)"


# ===========================================================================
# moreh_nll_loss_backward
# Copied from test_moreh_nll_loss.py::run_moreh_nll_loss_backward
# (shape [400, 300], ignore_index=1, reduction_mean=True, none_weight=False — verbatim case).
# Internally runs the forward (step1+step2) AND nll_loss_backward, asserting input.grad pcc.
# ===========================================================================
def test_moreh_nll_loss_backward_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_nll_loss import run_moreh_nll_loss_backward

    torch.manual_seed(0)
    shape = [400, 300]
    ignore_index = 1
    reduction_mean = True
    none_weight = False

    # COMPOSITE op: forward step1 + forward step2 + backward = multiple distinct descriptor programs
    # on the first miss; later passes with the same config must add none.
    # WARM-UP then assert ZERO growth.
    run_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, cache_device)
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        run_moreh_nll_loss_backward(shape, ignore_index, reduction_mean, none_weight, cache_device)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_nll_loss_backward: cache grew past {base} across fresh data (over-caching)"


# ===========================================================================
# moreh_nll_loss_unreduced_backward
# Copied from test_moreh_nll_loss_unreduced.py::run_moreh_nll_loss_unreduced_backward
# (shape [5, 10], ignore_index=1, none_weight=False — verbatim from test_moreh_nll_loss_unreduced).
# Pure backward op (no forward dispatch) -> single descriptor program.
# ===========================================================================
def test_moreh_nll_loss_unreduced_backward_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_nll_loss_unreduced import (
        run_moreh_nll_loss_unreduced_backward,
    )

    torch.manual_seed(0)
    shape = [5, 10]
    ignore_index = 1
    none_weight = False

    # WARM-UP then assert ZERO growth across fresh data.
    run_moreh_nll_loss_unreduced_backward(shape, ignore_index, none_weight, cache_device)
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        run_moreh_nll_loss_unreduced_backward(shape, ignore_index, none_weight, cache_device)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_nll_loss_unreduced_backward: cache grew past {base} across fresh data"


# ===========================================================================
# moreh_norm_backward
# Copied from test_moreh_norm.py::run_moreh_norm_backward
# (input_shape [32, 32], p=2.0, dim=0, rtol=atol=0.06, keepdim=False — verbatim test case).
# p / dim / keepdim are in operation_attributes -> default-hashed -> held FIXED; vary DATA.
# ===========================================================================
def test_moreh_norm_backward_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_norm import run_moreh_norm_backward

    torch.manual_seed(2024)
    input_shape = [32, 32]
    p = 2.0
    dim = 0
    rtol = atol = 0.06
    keepdim = False

    # run_moreh_norm_backward builds fresh tensors and asserts grad allclose internally.
    # WARM-UP then assert ZERO growth across fresh data (p/dim/keepdim held FIXED).
    run_moreh_norm_backward(input_shape, p, dim, rtol, atol, cache_device, keepdim=keepdim)
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        run_moreh_norm_backward(input_shape, p, dim, rtol, atol, cache_device, keepdim=keepdim)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_norm_backward: cache grew past {base} across fresh data (over-caching)"


# ===========================================================================
# moreh_sum_backward
# Copied from test_moreh_sum.py::moreh_sum_backward
# (input_shape [3, 2, 320-1, 320-1] = TILE*10-1, dim=0, keepdim=True, use_provide_output=True
#  — verbatim shape/dims from test_moreh_sum_backward). Helper builds fresh tensors + asserts pcc.
# ===========================================================================
def test_moreh_sum_backward_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_sum import (
        moreh_sum_backward,
        TILE_HEIGHT,
        TILE_WIDTH,
    )

    input_shape = [3, 2, TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1]
    dim = 0
    keepdim = True

    def _run():
        # moreh_sum_backward seeds torch internally and returns the pass/fail bool.
        passing = moreh_sum_backward(input_shape, dim, keepdim, True, False, cache_device)
        assert passing, "moreh_sum_backward stale (pcc fail on cache hit)"

    # WARM-UP then assert ZERO growth across fresh data.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_sum_backward: cache grew past {base} across fresh data (over-caching)"


# ===========================================================================
# experimental.attn_matmul  (standard path; num_tokens/transpose_hw unset)
# Copied from matmul/test_attn_matmul.py::test_attn_matmul_with_program_cache
# (first shape from generate_input_shapes(): [1,10,64,96] x [64,1,96,32], all bfloat16;
#  golden = (a.transpose(0,2) @ b).transpose(0,2); tolerances copied verbatim, k=96).
# ===========================================================================
def test_attn_matmul_cache(cache_device):
    from tests.ttnn.utils_for_testing import assert_numeric_metrics

    torch.manual_seed(0)
    input_shape_a = [1, 10, 64, 96]  # [q_len, q_heads, batch, K]
    input_shape_b = [64, 1, 96, 32]  # [batch, kv_heads, K, seq_len]
    in0_dtype = in1_dtype = out_dtype = ttnn.bfloat16
    k = input_shape_a[-1]  # 96
    compute_grid_size = cache_device.compute_with_storage_grid_size()

    def _run():
        input_tensor_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_b = torch.randn(input_shape_b).bfloat16()

        tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(cache_device)
        tt_input_tensor_b = ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(cache_device)

        tt_output_tensor_on_device = ttnn.experimental.attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=out_dtype,
        )
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)
        # not stale: golden recomputed per fresh input.
        assert_numeric_metrics(
            golden_output_tensor,
            tt_output_tensor,
            atol=0.02 * k,
            rtol=47.75 * k,
            frobenius_threshold=0.001 * k,
            check_ulp=False,
        )

    # WARM-UP then assert ZERO growth across fresh data.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"attn_matmul: cache grew past {base} across fresh data (over-caching / addr not patched)"


# ===========================================================================
# experimental.group_attn_matmul  (standard path; num_tokens/transpose_hw unset)
# Copied from matmul/test_attn_matmul.py::test_group_attn_matmul_with_program_cache
# (first shape: K=96, seq_len=512+64, q_heads=10, kv_heads=2, batch=32, all bfloat8_b, DRAM;
#  golden repeats kv across q_heads then (a.transpose(0,2) @ b).transpose(0,2); tolerances verbatim).
# That existing test asserts exactly 1 entry per distinct shape -> we hold one shape and bound <=1.
# ===========================================================================
def test_group_attn_matmul_cache(cache_device):
    from tests.ttnn.utils_for_testing import assert_numeric_metrics

    torch.manual_seed(0)
    interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG
    in0_dtype = in1_dtype = output_dtype = ttnn.bfloat8_b
    q_len = 1
    batch = 32
    K, seq_len, q_heads, kv_heads = 96, 512 + 64, 10, 2
    compute_grid_size = cache_device.compute_with_storage_grid_size()

    def _run():
        input_shape_a = [q_len, q_heads, batch, K]
        input_shape_b = [batch, kv_heads, K, seq_len]
        input_tensor_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_b = torch.randn(input_shape_b).bfloat16()

        tt_input_tensor_a = (
            ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(cache_device, interleaved_mem_config)
        )
        tt_input_tensor_b = (
            ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(cache_device, interleaved_mem_config)
        )

        tt_output_tensor_on_device = ttnn.experimental.group_attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            compute_with_storage_grid_size=compute_grid_size,
            memory_config=interleaved_mem_config,
            dtype=output_dtype,
        )
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        golden_a = input_tensor_a.to(torch.float)
        golden_b = torch.repeat_interleave(input_tensor_b.to(torch.float), q_heads // kv_heads, dim=1)
        golden_output_tensor = (golden_a.transpose(0, 2) @ golden_b).transpose(0, 2)
        # not stale
        assert_numeric_metrics(
            golden_output_tensor,
            tt_output_tensor,
            atol=0.021 * K,
            rtol=37 * K,
            frobenius_threshold=0.001 * K,
            check_ulp=False,
        )

    # WARM-UP then assert ZERO growth across fresh data.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"group_attn_matmul: cache grew past {base} across fresh data (over-caching)"


# ===========================================================================
# eltwise / unary_backward / tanh_bw  ->  ttnn.tanh_bw(grad, input)
# Invocation copied from test_descriptor_no_rebuild.py::test_tanh_bw_no_rebuild
# (1,1,64,64 bf16 grad + input). Reference is the analytic tanh derivative
#   d/dx tanh(x) = 1 - tanh(x)^2   ->   grad_input = grad * (1 - tanh(x)^2)
# (the attached ttnn golden uses torch autograd which needs a requires_grad leaf; the analytic
#  form is identical and avoids that, matching what tanh_bw computes on device).
# No per-call scalar; pure tensor op -> addresses-only -> expect <=1 entry, never stale.
# ===========================================================================
def test_tanh_bw_cache(cache_device):
    torch.manual_seed(0)
    shape = (1, 1, 64, 64)

    def _run():
        grad = torch.randn(shape, dtype=torch.bfloat16)
        inp = torch.randn(shape, dtype=torch.bfloat16)
        grad_t = ttnn.from_torch(grad, layout=ttnn.TILE_LAYOUT, device=cache_device)
        inp_t = ttnn.from_torch(inp, layout=ttnn.TILE_LAYOUT, device=cache_device)
        out = ttnn.to_torch(ttnn.tanh_bw(grad_t, inp_t)[0]).float()
        ref = (grad.float() * (1.0 - torch.tanh(inp.float()) ** 2)).float()
        assert torch.allclose(out, ref, atol=0.05, rtol=0.05), "tanh_bw stale (fresh data)"

    # WARM-UP then assert ZERO growth across fresh data.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"tanh_bw: cache grew past {base} across fresh data (over-caching / addr not patched)"
