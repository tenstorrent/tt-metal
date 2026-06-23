# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer parity: TT ``DiffusionGemmaSelfConditioning`` vs the actual
``transformers.models.diffusion_gemma.modeling_diffusion_gemma.DiffusionGemmaSelfConditioning``.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_self_conditioning.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma.self_conditioning import DiffusionGemmaSelfConditioning
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, local_device_to_torch
from ....utils.test import line_params, ring_params

PCC_THRESHOLD = 0.9995
ALLCLOSE_ATOL = 2e-2
ALLCLOSE_RTOL = 2e-2


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((4, 8), 0, 2, line_params, ttnn.Topology.Linear, id="bh_galaxy_tp4"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_tp2"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_self_conditioning(mesh_device: ttnn.MeshDevice, tp_axis: int, num_links: int, topology: ttnn.Topology) -> None:
    """TT vs HF DiffusionGemmaSelfConditioning."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaSelfConditioning as HFSelfConditioning,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextConfig

    torch.manual_seed(0)
    torch_dtype = torch.float32

    # DiffusionGemma-26B-A4B-it text config.
    hidden_size = 2816
    intermediate_size = 2112
    canvas_length = 256
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if (intermediate_size // tp_factor) % ttnn.TILE_SIZE != 0:
        pytest.skip(f"intermediate_size={intermediate_size} / tp={tp_factor} not tile-aligned")

    hf_config = DiffusionGemmaTextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
    )
    torch_model = HFSelfConditioning(hf_config).to(torch_dtype).eval()

    inputs_embeds = torch.randn(B, canvas_length, hidden_size, dtype=torch_dtype)
    sc_signal = torch.randn(B, canvas_length, hidden_size, dtype=torch_dtype)

    with torch.no_grad():
        torch_out = torch_model(inputs_embeds, sc_signal)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    tt_model = DiffusionGemmaSelfConditioning(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    tt_inputs_embeds = bf16_tensor(inputs_embeds.unsqueeze(0), device=mesh_device)
    tt_sc_signal = bf16_tensor(sc_signal.unsqueeze(0), device=mesh_device)

    tt_out = tt_model(tt_inputs_embeds, tt_sc_signal)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"torch_out: {torch_out.shape}, tt_out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(torch_dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(
        torch_out, tt_out_torch.to(torch_dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    ), f"allclose failed: max abs={abs_diff.max().item():.3e}"
