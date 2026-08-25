# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for LMHead2D on a Wormhole Galaxy."""

import gc

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_2d import LMHead2D
from models.common.utility_functions import comp_pcc


class _ColumnAllReduce:
    """Synchronous adapter satisfying LMHead2D's borrowed-input/owned-output contract."""

    cluster_axis = 1
    consumes_input = False
    returns_owned_output = True

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

    def __call__(self, tensor):
        return ttnn.all_reduce(
            tensor,
            cluster_axis=self.cluster_axis,
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=tensor.memory_config(),
        )


def _deallocate(tensor):
    if tensor is not None:
        tensor.deallocate(True)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
@pytest.mark.parametrize(
    "dim,vocab_size,padded_vocab_size",
    [
        pytest.param(8192, 128256, 128256, id="llama"),
        pytest.param(5120, 151936, 152064, id="qwen"),
    ],
)
def test_lm_head_2d_wh_galaxy_decode_reference(mesh_device, dim, vocab_size, padded_vocab_size):
    """Run each full-size decode geometry independently and reuse its allocations."""
    torch.manual_seed(2)
    weight = torch.randn((dim, padded_vocab_size), dtype=torch.bfloat16)
    if padded_vocab_size > vocab_size:
        weight[:, vocab_size:] = 0
    hidden = torch.randn((1, 1, 32, dim), dtype=torch.bfloat16)
    reference = torch.matmul(hidden, weight[:, :vocab_size])
    lazy_input = LazyWeight(source=hidden, device=mesh_device, dtype=ttnn.bfloat16)
    module = LMHead2D(
        [LazyWeight(source=weight, device=mesh_device, dtype=ttnn.bfloat8_b)],
        vocab_size,
        _ColumnAllReduce(mesh_device),
    )

    try:
        # PrefillRuntime extracts the final token rows before LM-head projection,
        # so both modes consume the same physical 32-row output batch here.
        for mode_forward in (module.decode_forward, module.prefill_forward):
            for _ in range(2):
                output = mode_forward(lazy_input)
                try:
                    actual = to_torch_auto_compose(output)
                    assert tuple(actual.shape[-2:]) == (32, padded_vocab_size)
                    assert torch.isneginf(actual[..., vocab_size:]).all()
                    passing, message = comp_pcc(reference, actual[..., :vocab_size].float(), 0.99)
                finally:
                    _deallocate(output)
                assert passing, message
    finally:
        _deallocate(lazy_input._value)
        lazy_input._value = None
        module.release()
        del hidden, lazy_input, module, reference, weight
        gc.collect()
