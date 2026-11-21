# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_wormhole_b0, is_blackhole
from tt_py_test_utils_common.utility_functions import torch_random


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [-1])
def test_pre_and_post_operation_hooks_for_printing(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    def pre_hook_to_print_args_and_kwargs(operation, args, kwargs):
        print(f"Pre-hook called for {operation}. Args: {args}, kwargs: {kwargs}")

    def post_hook_to_print_output(operation, args, kwargs, output):
        print(f"Post-hook called for {operation}. Output: {output}")

    with ttnn.register_pre_operation_hook(pre_hook_to_print_args_and_kwargs), ttnn.register_post_operation_hook(
        post_hook_to_print_output
    ):
        ttnn.exp(input_tensor) * 2 + 1


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [-1])
def test_pre_operaiton_hook_for_storing_input_activations(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    activations = []

    def pre_hook_to_store_activations(operation, args, kwargs):
        for arg in args:
            if isinstance(arg, ttnn.Tensor):
                activations.append(arg)
        for kwarg in kwargs.values():
            if isinstance(kwarg, ttnn.Tensor):
                activations.append(kwarg)

    with ttnn.register_pre_operation_hook(pre_hook_to_store_activations):
        ttnn.exp(input_tensor) * 2 + 1

    # TODO(arakhmati): only store activations for top-level ops
    assert len(activations) == 3


def test_pre_operation_hook_runs_for_concurrent_top_level_operations(device):
    torch_input_tensor = torch_random((1, 32, 32), -1, 1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    worker_in_hook = threading.Event()
    main_op_done = threading.Event()
    hooked_threads = []
    operation_ids = []

    def pre_hook(operation, args, kwargs):
        hooked_threads.append(threading.current_thread().name)
        operation_ids.append(ttnn.decorators.get_current_operation_id())
        if threading.current_thread().name == "worker":
            worker_in_hook.set()
            main_op_done.wait(timeout=30)

    worker_errors = []

    def worker():
        try:
            ttnn.exp(input_tensor)
        except Exception as e:
            worker_errors.append(e)

    with ttnn.manage_config("enable_fast_runtime_mode", False), ttnn.register_pre_operation_hook(pre_hook):
        thread = threading.Thread(target=worker, name="worker")
        thread.start()
        try:
            assert worker_in_hook.wait(timeout=30), "worker operation never reached its pre-operation hook"
            ttnn.exp(input_tensor)
        finally:
            main_op_done.set()
            thread.join()

    assert not worker_errors, worker_errors
    assert sorted(hooked_threads) == sorted(["worker", threading.current_thread().name])
    assert len(set(operation_ids)) == 2, f"concurrent operations must keep their own operation ids: {operation_ids}"
