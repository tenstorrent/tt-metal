# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Repro: ProgramSpecMeshWorkloadFactoryAdapter mis-binds aliased tensor arguments.

Any Metal 2.0 op whose factory declares two TensorParameters can be called with the
same tensor for both.  On that call, ProgramSpecMeshWorkloadFactoryAdapter::
resolve_bindings (ttnn/api/ttnn/mesh_device_operation_adapter.hpp) matches both
parameters by MeshTensor identity and cannot tell which slot each came from.  A tensor's
program hash covers only storage type and spec (ttnn/api/ttnn/tensor/tensor.hpp), so
without more, a later call with distinct but same-spec tensors would hit that cache entry
and replay its slot indices -- silently feeding one tensor to both parameters.  The
program-cache key of such ops therefore also covers which tensor_args slots alias
(compute_mesh_workload_cache_key), so the aliased and the distinct call patterns never share an entry.

batch_norm is used here because running_mean and running_var are separate
TensorParameters of identical shape, so aliasing them is legal, and eval-mode
batch_norm makes the substitution visible in the output:

    out = (input - running_mean) / sqrt(running_var + eps)

With input=5, mean=1, var=4:  correct = (5-1)/sqrt(4) = 2.0
If var is fed the mean tensor:  buggy   = (5-1)/sqrt(1) = 4.0
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device


def _channel_tensor(value, channels, device):
    """Per-channel [1, C, 1, 1] tensor filled with `value`."""
    return ttnn.from_torch(
        torch.full((1, channels, 1, 1), value, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def test_program_cache_aliased_then_distinct_tensor_args(device):
    channels = 2
    eps = 0.0

    input_tensor = ttnn.from_torch(
        torch.full((1, channels, 32, 32), 5.0, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    mean = _channel_tensor(1.0, channels, device)
    var = _channel_tensor(4.0, channels, device)

    # Call 1 -- ALIASED: the same tensor is passed for running_mean and running_var.
    # This is the call that poisons the cache entry (running_var's binding collapses
    # onto running_mean's slot).
    aliased = ttnn.batch_norm(input_tensor, running_mean=mean, running_var=mean, training=False, eps=eps)
    aliased_out = ttnn.to_torch(aliased)[0, 0, 0, 0].item()
    # (5 - 1) / sqrt(1) = 4.0 -- correct for THIS call, since mean and var really are equal here.
    assert aliased_out == pytest.approx(4.0, abs=0.05), f"call 1 (aliased) itself is wrong: {aliased_out}"

    # Call 2 -- DISTINCT: same specs as call 1, so it would hit the cache entry above if the
    # cache key ignored aliasing. running_var must be read from `var` (4.0), not from `mean` (1.0).
    distinct = ttnn.batch_norm(input_tensor, running_mean=mean, running_var=var, training=False, eps=eps)
    distinct_out = ttnn.to_torch(distinct)[0, 0, 0, 0].item()

    expected = 2.0  # (5 - 1) / sqrt(4)
    buggy = 4.0  # (5 - 1) / sqrt(1)  <- running_var silently bound to the mean tensor
    print(f"\ncall 1 (mean=var=1.0): {aliased_out}   [expected 4.0]")
    print(f"call 2 (mean=1.0, var=4.0): {distinct_out}   [expected {expected}, bug gives {buggy}]")

    assert distinct_out != pytest.approx(buggy, abs=0.05), (
        f"program-cache aliasing bug reproduced: call 2 returned {distinct_out}, which is "
        f"(input - mean) / sqrt(mean + eps) -- running_var was re-bound to the mean tensor "
        f"by the cache-hit path"
    )
    assert distinct_out == pytest.approx(expected, abs=0.05), f"call 2 returned {distinct_out}, expected {expected}"


def test_workload_descriptor_cache_hit_after_aliased_inputs_is_rejected(device, expect_error):
    """ring_distributed_sdpa(X, X, X) caches a WorkloadDescriptor program whose buffer bindings could not be
    resolved; a later same-spec call with distinct tensors must not silently reuse its stale addresses."""
    shape = (1, 1, 1024, 128)
    ring_size = 4
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
    )

    def to_device(value):
        return ttnn.from_torch(torch.full(shape, value), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def ring_sdpa(q, k, v):
        return ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            q, k, v, ring_size=ring_size, ring_id=0, program_config=program_config
        )

    device.clear_program_cache()
    x = to_device(1.0)
    ttnn.to_torch(ring_sdpa(x, x, x))
    ttnn.to_torch(ring_sdpa(x, x, x))

    q, k, v = to_device(1.0), to_device(1.0), to_device(1.0)
    with expect_error(RuntimeError, "cannot be re-patched"):
        ttnn.to_torch(ring_sdpa(q, k, v))

    device.clear_program_cache()
    ttnn.to_torch(ring_sdpa(q, k, v))
    device.clear_program_cache()


def test_program_cache_distinct_then_aliased_tensor_args(device):
    """The reverse ordering of test_program_cache_aliased_then_distinct_tensor_args: an entry primed
    by distinct running_mean / running_var must not serve an aliased call with the wrong tensor, and
    the distinct call must still be correct afterwards."""
    channels = 2
    eps = 0.0

    input_tensor = ttnn.from_torch(
        torch.full((1, channels, 32, 32), 5.0, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    mean = _channel_tensor(1.0, channels, device)
    var = _channel_tensor(4.0, channels, device)

    distinct = ttnn.batch_norm(input_tensor, running_mean=mean, running_var=var, training=False, eps=eps)
    distinct_out = ttnn.to_torch(distinct)[0, 0, 0, 0].item()
    assert distinct_out == pytest.approx(2.0, abs=0.05), f"distinct call returned {distinct_out}, expected 2.0"

    # running_mean and running_var are both `var` here: (5 - 4) / sqrt(4) = 0.5.
    aliased = ttnn.batch_norm(input_tensor, running_mean=var, running_var=var, training=False, eps=eps)
    aliased_out = ttnn.to_torch(aliased)[0, 0, 0, 0].item()
    assert aliased_out == pytest.approx(0.5, abs=0.05), f"aliased call returned {aliased_out}, expected 0.5"

    distinct_again = ttnn.batch_norm(input_tensor, running_mean=mean, running_var=var, training=False, eps=eps)
    distinct_again_out = ttnn.to_torch(distinct_again)[0, 0, 0, 0].item()
    assert distinct_again_out == pytest.approx(
        2.0, abs=0.05
    ), f"distinct call after the aliased one returned {distinct_again_out}, expected 2.0"


def test_program_cache_aliased_out_of_order_tensor_params_sampling(device):
    """Issue #57248: SpecBindingSupport::resolve_bindings matches each entry of ProgramRunArgs::tensor_args
    to an io slot by MeshTensor identity, and the factory's Table order need not follow the
    reflection order of the io tensors.

    SamplingInputs declares (input_values=0, input_indices=1, k=2, p=3, temp=4), whereas
    SamplingProgramFactory inserts SAMPLING_TEMP_TENSOR before SAMPLING_K_TENSOR and
    SAMPLING_P_TENSOR. With p and temp aliased, the miss cannot tell which slot each came from, so
    an aliased call must not share a cache entry with distinct calls: if it did, a later hit with
    distinct p=0, temp=1 tensors could swap them (supplying p=0 to TEMP and temp=1 to P), returning
    wrong sampled labels. The program-cache key folds in which tensor_args slots alias, so the
    aliased and distinct patterns each get their own entry and each reuses it on later calls.
    """
    host_values = (-torch.arange(64, dtype=torch.float32) / 256).to(torch.bfloat16).repeat(1, 1, 32, 1)
    host_labels = (1000 + torch.arange(64, dtype=torch.int32)).repeat(1, 1, 32, 1)
    expected_labels = torch.full((32,), 1000, dtype=torch.int64)

    values = ttnn.from_torch(host_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    labels = ttnn.from_torch(host_labels, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    k = ttnn.from_torch(
        torch.full((32,), 32, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ones_a = ttnn.from_torch(
        torch.ones(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ones_b = ttnn.from_torch(
        torch.ones(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    p_zero = ttnn.from_torch(
        torch.zeros(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    temp_one = ttnn.from_torch(
        torch.ones(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    assert ones_a is not ones_b and p_zero is not temp_one

    def reset_cache():
        ttnn.synchronize_device(device)
        device.clear_program_cache()
        device.enable_program_cache()
        assert device.num_program_cache_entries() == 0

    def run_sampling_call(name, p_tensor, temp_tensor, entries_before, entries_after):
        before = device.num_program_cache_entries()
        assert before == entries_before, f"{name}: expected {entries_before} cache entries before call, got {before}"
        out = ttnn.sampling(values, labels, k=k, p=p_tensor, temp=temp_tensor)
        actual = ttnn.to_torch(out).reshape(-1).to(torch.int64)
        after = device.num_program_cache_entries()
        assert actual.numel() == 32 and after == entries_after, (
            f"{name}: expected 32 labels and {entries_after} program-cache entries, "
            f"got numel={actual.numel()}, entries={after}"
        )
        return actual

    def assert_expected_labels(name, actual):
        wrong = int((actual != expected_labels).sum().item())
        assert wrong == 0, f"{name}: {wrong} of 32 labels wrong (got {actual.tolist()}, expected all 1000)"

    try:
        # 1. Distinct-primed control: equal-valued but distinct tensors bind TEMP -> 4 and P -> 3.
        reset_cache()
        run_sampling_call("distinct_prime", ones_a, ones_b, entries_before=0, entries_after=1)
        distinct_actual = run_sampling_call("distinct_then_normal", p_zero, temp_one, entries_before=1, entries_after=1)
        assert_expected_labels("distinct-input control", distinct_actual)

        # 2. Fresh-cache control: p=0, temp=1 on an empty program cache must return the unique argmax (1000).
        reset_cache()
        fresh_actual = run_sampling_call("fresh_normal", p_zero, temp_one, entries_before=0, entries_after=1)
        assert_expected_labels("fresh-cache control", fresh_actual)

        # 3. Alias-primed repro: the distinct call after an aliased one builds its own entry instead of
        # replaying the aliased entry's bindings, and each pattern then hits its own entry.
        reset_cache()
        run_sampling_call("alias_prime", ones_a, ones_a, entries_before=0, entries_after=1)
        alias_actual = run_sampling_call("alias_then_normal", p_zero, temp_one, entries_before=1, entries_after=2)
        assert_expected_labels("normal call after aliased priming", alias_actual)
        run_sampling_call("alias_repeat", ones_b, ones_b, entries_before=2, entries_after=2)
        post_alias_actual = run_sampling_call(
            "normal_after_re_alias", p_zero, temp_one, entries_before=2, entries_after=2
        )
        assert_expected_labels("normal call after a repeated aliased call", post_alias_actual)

        # 4. Distinct-then-alias: the aliased call after a distinct one also gets its own entry, and the
        # distinct entry keeps serving distinct calls correctly.
        reset_cache()
        normal_actual = run_sampling_call("normal_prime", p_zero, temp_one, entries_before=0, entries_after=1)
        assert_expected_labels("distinct priming call", normal_actual)
        run_sampling_call("alias_after_normal", ones_a, ones_a, entries_before=1, entries_after=2)
        normal_again_actual = run_sampling_call(
            "normal_after_alias", p_zero, temp_one, entries_before=2, entries_after=2
        )
        assert_expected_labels("distinct call after an aliased one", normal_again_actual)
    finally:
        ttnn.synchronize_device(device)
        device.clear_program_cache()
