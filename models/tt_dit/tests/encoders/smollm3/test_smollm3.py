# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from loguru import logger

import ttnn
from models.tt_dit.encoders.smollm3 import SmolLm3Checkpoint
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis"),
    [pytest.param((2, 2), 0, id="2x2sp")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("masked", [pytest.param(True, id="masked")])
def test_transformer(*, mesh_device: ttnn.MeshDevice, sp_axis: int | None, masked: bool) -> None:
    torch.manual_seed(0)

    batch_size = 2
    sequence_length = 512
    tp_axis = 1
    sp_factor = mesh_device.shape[sp_axis] if sp_axis is not None else 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis) if sp_axis is not None else None,
    )

    model = SmolLm3Checkpoint("briaai/FIBO").build(
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    torch_model = transformers.AutoModelForCausalLM.from_pretrained(
        "briaai/FIBO", subfolder="text_encoder", dtype=torch.float32
    )

    tokens = torch.randint(0, torch_model.config.vocab_size, [batch_size, sequence_length])
    lengths = torch.randint(sequence_length // 4, 3 * sequence_length // 4, [batch_size])
    mask = torch.arange(sequence_length).flip([0]) < lengths.unsqueeze(1) if masked else None

    tt_tokens = tensor.from_torch(
        tokens,
        device=mesh_device,
        dtype=ttnn.uint32,
        mesh_axes=[None, sp_axis],
    )
    tt_mask = tensor.from_torch(mask, device=mesh_device) if mask is not None else None

    logger.info("running ttnn model...")
    tt_hidden_states = model.forward(
        tt_tokens,
        mask=tt_mask,
        skip_final_linear=True,
        output_hidden_states=True,
    )
    tt_hidden_states_torch = [tensor.to_torch(t, mesh_axes=[None, sp_axis, None]) for t in tt_hidden_states]

    logger.info("running torch model...")
    torch_mask_input = mask if mask is not None else torch.ones_like(tokens)
    with torch.no_grad():
        hidden_states = torch_model.forward(
            tokens, attention_mask=torch_mask_input, output_hidden_states=True
        ).hidden_states

    if mask is not None:
        # Masked positions on the start of the sequence contain undefined values from computing softmax over all -inf
        # so we remove them before comparison.
        _, _, d = hidden_states[0].shape
        hidden_states = [t.masked_select(mask.unsqueeze(-1)).view([-1, d]) for t in hidden_states]
        tt_hidden_states_torch = [t.masked_select(mask.unsqueeze(-1)).view([-1, d]) for t in tt_hidden_states_torch]

    assert len(hidden_states) == len(tt_hidden_states_torch)

    for x, tt_x in zip(hidden_states[-4:], tt_hidden_states_torch[-4:], strict=True):
        assert_quality(x, tt_x, pcc=0.9979, relative_rmse=0.065)
