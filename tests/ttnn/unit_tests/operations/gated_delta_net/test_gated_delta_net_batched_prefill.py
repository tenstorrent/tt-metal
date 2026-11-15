# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""batched_prefill (B > 1, T > 1): sequence boundaries inside a tile row, padding finite or not."""

import pytest
import torch

from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    check_call,
    compute_config,
    make_inputs,
    reference,
    to_device,
)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 10, 256), (2, 3, 32, 64)), id="b2_t5_boundary_in_tile"),
        pytest.param(((1, 66, 256), (2, 3, 32, 64)), id="b2_t33"),
        pytest.param(((1, 64, 2560), (2, 12, 128, 128)), id="b2_t32_chip"),
    ],
)
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
def test_batched_prefill(device, inputs, poison):
    torch.manual_seed(1)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    returned = gated_delta_net(*to_device(tensors, device, poison=poison), compute_kernel_config=compute_config())
    check_call(returned, expected, inputs)
