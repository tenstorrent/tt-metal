# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer parity: TT ``DiffusionGemmaDenseMLP`` vs the actual
``transformers.models.diffusion_gemma.modeling_diffusion_gemma.DiffusionGemmaText4MLP``.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_dense_mlp.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma.dense_mlp import DiffusionGemmaDenseMLP
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, local_device_to_torch
from ....utils.test import line_params

PCC_THRESHOLD = 0.9995
ALLCLOSE_ATOL = 2e-2
ALLCLOSE_RTOL = 2e-2


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params"),
    [
        pytest.param((2, 4), 0, 1, line_params, id="bh_qb2_tp2"),
        pytest.param((4, 8), 0, 2, line_params, id="bh_galaxy_tp4"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [256])
def test_dense_mlp(mesh_device: ttnn.MeshDevice, tp_axis: int, num_links: int, seq_len: int) -> None:
    """TT vs HF DiffusionGemmaText4MLP."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaText4MLP,
        DiffusionGemmaTextConfig,
    )

    torch.manual_seed(0)
    dtype = torch.float32

    hidden_size = 2816
    intermediate_size = 2112
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if (intermediate_size // tp_factor) % ttnn.TILE_SIZE != 0:
        pytest.skip(f"intermediate_size={intermediate_size} / tp={tp_factor} not tile-aligned")

    hf_config = DiffusionGemmaTextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_activation="gelu_pytorch_tanh",
    )
    # layer_idx is unused by DiffusionGemmaText4MLP (it overrides parent's kv-share logic) but required by sig.
    torch_model = DiffusionGemmaText4MLP(hf_config, layer_idx=0).to(dtype).eval()

    x = torch.randn(B, seq_len, hidden_size, dtype=dtype)

    with torch.no_grad():
        torch_out = torch_model(x)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    tt_model = DiffusionGemmaDenseMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    tt_x = bf16_tensor(x.unsqueeze(0), device=mesh_device)
    tt_out = tt_model(tt_x)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(torch_out, tt_out_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
