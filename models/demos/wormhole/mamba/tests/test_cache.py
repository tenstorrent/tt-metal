# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from models.demos.wormhole.mamba.tt.cache import TensorCache

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


@pytest.mark.parametrize("on_host", [True, False])
@pytest.mark.parametrize(
    "E, num_users, num_entries",
    (
        (32, 32, 1),
        (2560, 32, 1),
        (5120, 32, 4),
    ),
)
def test_cache(E, num_users, num_entries, on_host, device):
    values = [[torch.rand((1, 1, 1, E), dtype=torch.bfloat16) for _ in range(num_users)] for _ in range(num_entries)]

    cache = TensorCache(num_users, num_entries, E, device, on_host=on_host)
    for i in range(num_entries):
        for user in range(num_users):
            value = ttnn.from_torch(
                values[i][user],
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            cache.set(user, i, value)
            did_pass, output_pcc = comp_pcc(ttnn.to_torch(value), ttnn.to_torch(cache.get(user, i)), 1.0)
            assert did_pass

    for i in range(num_entries):
        conv_states = cache.concat_users(i)
        assert list(conv_states.shape) == [1, 1, num_users, E]

        expected = torch.concat(values[i], dim=2)
        did_pass, output_pcc = comp_pcc(expected, ttnn.to_torch(conv_states), 1.0)
        assert did_pass

    cache.reset()

    for i in range(num_entries):
        conv_states = cache.concat_users(i)
        assert list(conv_states.shape) == [1, 1, num_users, E]

        expected = torch.zeros((1, 1, num_users, E), dtype=torch.bfloat16)
        did_pass, output_pcc = comp_pcc(expected, ttnn.to_torch(conv_states), 1.0)
        assert did_pass
