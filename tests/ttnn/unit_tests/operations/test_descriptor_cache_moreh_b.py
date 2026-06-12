# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Cache-correctness sweep for moreh ops (batch B) migrated to the descriptor framework (#46506).

Companion to test_descriptor_cache.py — same intent, same structure. For each op we run its
SIMPLEST existing test invocation (the op's own `tt_*` / `moreh_*` helper, copied — not invented)
under an ENABLED program cache, and we assert two things hold:

  * not stale  : every call's result matches the op's golden, even on a program-cache HIT
                 (a frozen per-call rt-arg would make a later call wrong).
  * not over-caching : the op does not create a new program-cache entry for a value it should
                 re-apply on the cache-hit path.

WHAT VARIES PER OP (read from the device factories under
ttnn/cpp/ttnn/operations/moreh/<op>/device/):

  * moreh_group_norm / moreh_layer_norm (+ their _backward):
        eps is baked into kernel rt-args via std::bit_cast<uint32_t>(eps), BUT eps lives in
        operation_attributes_t and is therefore part of the program hash:
          - moreh_group_norm has a custom compute_program_hash that explicitly includes eps.
          - moreh_layer_norm / _backward use the DEFAULT hash, which hashes operation_attributes
            (and thus eps) + tensor_args.
        => varying eps LEGITIMATELY adds a cache entry (not a bug). There is NO scalar that is
           baked into rt-args yet EXCLUDED from the hash, so there is no frozen-stale risk.
        => the genuine cache-HIT probe is: hold eps (and all attrs) FIXED, vary input DATA
           (fresh allocation -> fresh address). The HIT must stay correct (addresses patched,
           eps re-applied from the cached program) and must not add entries.

  * moreh_sgd:
        lr / momentum / dampening / weight_decay are baked into reader rt-args via
        std::bit_cast<uint32_t>(...), and all live in operation_attributes_t. moreh_sgd uses the
        DEFAULT hash, so those scalars are part of the cache key.
        => varying lr/momentum LEGITIMATELY adds entries (not a bug). Same as above: no
           excluded-from-hash scalar => no frozen-stale risk. The HIT probe holds the scalars
           FIXED and varies input DATA.

  * moreh_matmul:
        pure tensor op; the per-call variation is fresh inputs + transpose flags. transpose
        flags live in operation_attributes_t (hashed), so flipping them adds an entry by design
        (the op's own test_moreh_matmul_enable_cache asserts exactly 2 entries across a
        flag-flip loop). The HIT probe varies input DATA with FIXED flags.

PREDICTED VERDICT (all six): PASS / no frozen-stale scalar. These ops route every per-call
non-address value through the program hash, so a cache HIT is by construction value-correct;
the only thing to defend is address patching + no over-caching on the fixed-attribute loop.
"""

import pytest
import torch

import ttnn

from tests.ttnn.unit_tests.operations.test_utils import TILE_HEIGHT, TILE_WIDTH, to_torch
from models.common.utility_functions import comp_allclose, comp_allclose_and_pcc


@pytest.fixture(scope="module")
def cache_device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# moreh_layer_norm — forward. Copied from test_moreh_layer_norm.py (tt_layer_norm /
# torch_layer_norm, input_shape [6, 2*TH, 2*TW] normalized_dims=2, the op's callback case).
# eps is hashed -> HIT probe holds eps FIXED and varies input DATA.
# ---------------------------------------------------------------------------
def test_moreh_layer_norm_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_layer_norm import (
        tt_layer_norm,
        torch_layer_norm,
        make_input_tensors,
    )

    torch.manual_seed(2024)
    input_shape = [6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH]
    normalized_dims = 2
    eps = 0.05
    elementwise_affine = True
    rtol = atol = 0.15  # normalized_dims == 2

    def _run():
        cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, normalized_dims, elementwise_affine)
        expected_output, _, _ = torch_layer_norm(
            cpu_input, normalized_dims=normalized_dims, eps=eps, gamma=cpu_gamma, beta=cpu_beta
        )
        actual_output, _, _ = tt_layer_norm(
            cpu_input,
            normalized_dims=normalized_dims,
            eps=eps,
            gamma=cpu_gamma,
            beta=cpu_beta,
            dtype=ttnn.bfloat16,
            device=cache_device,
        )
        passing, out = comp_allclose(expected_output, actual_output, rtol=rtol, atol=atol)
        assert passing, f"moreh_layer_norm stale on cache hit: {out}"

    # WARM-UP then assert ZERO growth (eps held FIXED; only input DATA varies).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_layer_norm: cache grew past {base} across fresh data (over-caching / addr not patched)"


# ---------------------------------------------------------------------------
# moreh_layer_norm_backward. Copied from test_moreh_layer_norm.py
# (tt_layer_norm_backward / torch_layer_norm_backward, the op's callback case).
# eps is hashed -> HIT probe holds eps FIXED and varies input DATA.
# ---------------------------------------------------------------------------
def test_moreh_layer_norm_backward_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_layer_norm import (
        tt_layer_norm_backward,
        torch_layer_norm_backward,
        make_input_tensors,
    )

    torch.manual_seed(2024)
    input_shape = [6, 2 * TILE_HEIGHT, 2 * TILE_WIDTH]
    normalized_dims = 2
    eps = 0.05
    elementwise_affine = True
    rtol = 0.1
    atol = 0.5

    def _run():
        cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(
            input_shape, normalized_dims, elementwise_affine, do_backward=True
        )
        expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_layer_norm_backward(
            cpu_input, cpu_output_grad, normalized_dims=normalized_dims, eps=eps, gamma=cpu_gamma, beta=cpu_beta
        )
        actual_input_grad, actual_gamma_grad, actual_beta_grad = tt_layer_norm_backward(
            cpu_input,
            cpu_output_grad,
            normalized_dims=normalized_dims,
            eps=eps,
            gamma=cpu_gamma,
            beta=cpu_beta,
            dtype=ttnn.bfloat16,
            device=cache_device,
        )
        pig, oig = comp_allclose(expected_input_grad, actual_input_grad, rtol=rtol, atol=atol)
        assert pig, f"moreh_layer_norm_backward input_grad stale on cache hit: {oig}"
        if expected_gamma_grad is not None:
            pgg, ogg = comp_allclose(expected_gamma_grad, actual_gamma_grad, rtol=rtol, atol=atol)
            assert pgg, f"moreh_layer_norm_backward gamma_grad stale on cache hit: {ogg}"
        if expected_beta_grad is not None:
            pbg, obg = comp_allclose(expected_beta_grad, actual_beta_grad, rtol=rtol, atol=atol)
            assert pbg, f"moreh_layer_norm_backward beta_grad stale on cache hit: {obg}"

    # WARM-UP then assert ZERO growth. COMPOSITE op: the public moreh_layer_norm_backward dispatches
    # multiple distinct device programs (input_grad + gamma_beta_grad), so the first call settles
    # the cache at N entries; later passes with the same config must add none, regardless of N.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_layer_norm_backward: cache grew past {base} across fresh data (over-caching)"


# ---------------------------------------------------------------------------
# moreh_group_norm — forward.
# SKIPPED: the op's own forward test (test_moreh_group_norm) is itself skipped at the top of
# run_test_moreh_group_norm with reason "libstdc++ issue". We do not enable a path the op's
# own suite disables. eps IS in the custom compute_program_hash (verified), so there is no
# frozen-stale risk regardless.
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="moreh_group_norm forward is skipped in its own test (test_moreh_group_norm: 'libstdc++ issue'); "
    "not enabling a path the op's suite disables. eps is in the custom compute_program_hash -> no frozen-stale risk."
)
def test_moreh_group_norm_cache(cache_device):
    pass


# ---------------------------------------------------------------------------
# moreh_group_norm_backward. Copied from test_moreh_group_norm.py
# (tt_group_norm_backward / torch_group_norm_backward, the op's callback case:
# N=2, C/num_groups=[4,1], HW=[23,23], affine=True, all grads required).
# eps participates via the custom compute_program_hash -> HIT probe holds eps FIXED, varies DATA.
# ---------------------------------------------------------------------------
def test_moreh_group_norm_backward_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_group_norm import (
        tt_group_norm_backward,
        torch_group_norm_backward,
        make_input_tensors,
    )

    torch.manual_seed(2024)
    N, C, num_groups, H, W = 2, 4, 1, 23, 23
    input_shape = (N, C, H, W)
    eps = 1e-05
    affine = True
    input_requires_grad = gamma_requires_grad = beta_requires_grad = True
    rtol = atol = 0.1

    Ht = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    Wt = (W + TILE_WIDTH - 1) // TILE_WIDTH
    divisor = N * C * Ht * Wt

    def _run():
        cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(input_shape, affine, do_backward=True)
        expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_group_norm_backward(
            cpu_input,
            cpu_output_grad,
            num_groups,
            input_requires_grad,
            gamma_requires_grad,
            beta_requires_grad,
            cpu_gamma,
            cpu_beta,
            eps,
        )
        actual_input_grad, actual_gamma_grad, actual_beta_grad = tt_group_norm_backward(
            cpu_input,
            cpu_output_grad,
            num_groups,
            input_requires_grad,
            gamma_requires_grad,
            beta_requires_grad,
            cpu_gamma,
            eps,
            cache_device,
        )
        if expected_input_grad is not None:
            pig, oig = comp_allclose(expected_input_grad, actual_input_grad, rtol=rtol, atol=atol)
            assert pig, f"moreh_group_norm_backward input_grad stale on cache hit: {oig}"
        # gamma/beta grad divided by divisor (bf16 sum error), per the op's own test.
        if expected_gamma_grad is not None:
            pgg, ogg = comp_allclose(expected_gamma_grad / divisor, actual_gamma_grad / divisor, rtol=rtol, atol=atol)
            assert pgg, f"moreh_group_norm_backward gamma_grad stale on cache hit: {ogg}"
        if expected_beta_grad is not None:
            pbg, obg = comp_allclose(expected_beta_grad / divisor, actual_beta_grad / divisor, rtol=rtol, atol=atol)
            assert pbg, f"moreh_group_norm_backward beta_grad stale on cache hit: {obg}"

    # WARM-UP then assert ZERO growth. COMPOSITE op: the public moreh_group_norm_backward dispatches
    # multiple distinct device programs (input_grad + gamma_beta_grad), so the first call settles
    # the cache at N entries; later passes with the same config must add none, regardless of N.
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_group_norm_backward: cache grew past {base} across fresh data (over-caching)"


# ---------------------------------------------------------------------------
# moreh_matmul. Copied from test_moreh_matmul.py (moreh_matmul helper + its
# test_moreh_matmul_enable_cache shapes). Pure tensor op; per-call variation is fresh inputs.
# Transpose flags are hashed (flipping them adds an entry by design — the op's own
# enable_cache test asserts exactly 2 entries across a flag-flip loop). HIT probe: FIXED flags,
# fresh DATA -> addresses must patch, one entry.
# ---------------------------------------------------------------------------
def test_moreh_matmul_cache(cache_device):
    from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_matmul import moreh_matmul

    # input, other, output shape, transpose_input, transpose_other (single-core case, fixed flags)
    params = ([32, 32], [32, 32], [32, 32], False, False)

    def _run():
        # moreh_matmul re-seeds torch and allocates fresh tt tensors each call (fresh addresses),
        # and asserts the PCC golden internally.
        passing = moreh_matmul(params, False, None, cache_device)
        assert passing, "moreh_matmul stale on cache hit (PCC failed)"

    # WARM-UP then assert ZERO growth across fresh data (transpose flags held FIXED).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_matmul: cache grew past {base} across fresh data with fixed flags (over-caching)"


# ---------------------------------------------------------------------------
# moreh_sgd. Copied from test_moreh_sgd.py test_moreh_sgd_callback (single shape [32,32],
# lr=3.0, momentum=7.7, dampening=0.5, momentum_initialized loop).
# lr/momentum/dampening/weight_decay are baked into rt-args AND hashed (default hash over
# operation_attributes) -> varying them adds entries by design. HIT probe holds the scalars
# FIXED and varies input DATA (fresh param/grad allocations).
# ---------------------------------------------------------------------------
def test_moreh_sgd_cache(cache_device):
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(0)
    shape = [32, 32]
    lr = 3.0
    momentum = 7.7
    dampening = 0.5
    weight_decay = 0.0
    nesterov = False
    momentum_initialized = False
    npu_dtype, cpu_dtype = ttnn.bfloat16, torch.bfloat16
    rtol = atol = 0.05

    def _run():
        # Build a fresh model + grad each iteration -> fresh data / fresh device addresses,
        # while lr/momentum/... (the hashed attrs) stay fixed.
        x_data = torch.rand(shape).to(cpu_dtype)
        y_data = torch.rand(shape).to(cpu_dtype)

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.weight = nn.Parameter(torch.randn(shape).to(cpu_dtype)).to(cpu_dtype)

            def forward(self, x):
                return torch.mul(x, self.weight)

        model = SimpleModel()
        criterion = nn.L1Loss()
        optimizer = optim.SGD(
            {model.weight}, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov
        )
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        loss.backward()

        cpu_param_in = model.weight.clone()
        cpu_momentum_in = None
        if momentum != 0 and 0 in optimizer.state_dict()["state"]:
            cpu_momentum_in = optimizer.state_dict()["state"][0]["momentum_buffer"].clone()

        optimizer.step()

        cpu_momentum_out = None
        if momentum != 0 and 0 in optimizer.state_dict()["state"]:
            cpu_momentum_out = optimizer.state_dict()["state"][0]["momentum_buffer"].clone()

        cpu_grad = model.weight.grad

        dev_param_in = ttnn.from_torch(cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=cache_device)
        dev_param_out = ttnn.from_torch(cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=cache_device)
        dev_grad = ttnn.from_torch(cpu_grad, npu_dtype, layout=ttnn.TILE_LAYOUT, device=cache_device)

        dev_momentum_buffer_in = None
        dev_momentum_buffer_out = None
        if momentum != 0:
            if momentum_initialized and cpu_momentum_in is not None:
                dev_momentum_buffer_in = ttnn.from_torch(
                    cpu_momentum_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=cache_device
                )
            dev_momentum_buffer_out = ttnn.from_torch(
                cpu_param_in, npu_dtype, layout=ttnn.TILE_LAYOUT, device=cache_device
            )

        dev_param_out, dev_momentum_buffer_out = ttnn.operations.moreh.sgd(
            dev_param_in,
            dev_grad,
            dev_momentum_buffer_in,
            dev_param_out,
            dev_momentum_buffer_out,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            momentum_initialized=momentum_initialized,
        )

        param_result = ttnn.to_torch(dev_param_out).to(cpu_dtype)
        passing, out = comp_allclose_and_pcc(model.weight, param_result, pcc=0.99, rtol=rtol, atol=atol)
        assert passing, f"moreh_sgd param stale on cache hit: {out}"
        if momentum != 0:
            momentum_buffer_result = ttnn.to_torch(dev_momentum_buffer_out).to(cpu_dtype)
            passing, out = comp_allclose_and_pcc(
                cpu_momentum_out, momentum_buffer_result, pcc=0.99, rtol=rtol, atol=atol
            )
            assert passing, f"moreh_sgd momentum stale on cache hit: {out}"

    # WARM-UP then assert ZERO growth across fresh data (lr/momentum/... held FIXED).
    _run()
    base = cache_device.num_program_cache_entries()
    for _ in range(3):
        _run()
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"moreh_sgd: cache grew past {base} across fresh data with fixed lr/momentum (over-caching)"
