# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for Embedding2D on a Wormhole Galaxy."""

import gc

import pytest
import torch

from models.common.modules.embedding.embedding_2d import Embedding2D
from models.common.modules.lazy_weight import LazyWeight
from models.common.tests.modules._wh_galaxy_hardware import compose_2d_sharded_tensor
from models.common.utility_functions import comp_pcc


def _deallocate(tensor):
    if tensor is not None:
        tensor.deallocate(True)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4")], indirect=True)
@pytest.mark.parametrize(
    "vocab_size,dim,embed_scale",
    [
        pytest.param(128256, 8192, 1.0, id="llama"),
        pytest.param(151936, 5120, 5120**0.5, id="qwen"),
    ],
)
def test_embedding_2d_wh_galaxy_reference(mesh_device, vocab_size, dim, embed_scale):
    """Qualify decode batch 32 and sequential prefill lengths without co-allocation."""
    torch.manual_seed(0)
    weight = torch.randn((vocab_size, dim), dtype=torch.bfloat16)
    module = Embedding2D(LazyWeight(source=weight, device=mesh_device), embed_scale=embed_scale)

    try:
        for mode, token_count in (("decode", 32), ("prefill", 128), ("prefill", 2048)):
            token_ids = torch.randint(0, vocab_size, (1, token_count), dtype=torch.int32)
            reference = torch.nn.functional.embedding(token_ids.long(), weight).float() * embed_scale
            lazy_ids = LazyWeight(source=token_ids.reshape(1, 1, 1, token_count), device=mesh_device)

            for _ in range(2):
                output = module.forward(lazy_ids, mode=mode)
                try:
                    actual = compose_2d_sharded_tensor(output, mesh_device).reshape(1, token_count, dim).float()
                    passing, message = comp_pcc(reference, actual, 0.99)
                finally:
                    _deallocate(output)
                assert passing, message

            _deallocate(lazy_ids._value)
            lazy_ids._value = None
            del actual, lazy_ids, reference, token_ids
            gc.collect()
    finally:
        _deallocate(getattr(module, "weights", None))
        module.config.weights._value = None
        del module, weight
        gc.collect()
