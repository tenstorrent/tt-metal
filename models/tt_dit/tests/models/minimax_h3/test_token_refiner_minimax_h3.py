# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TokenRefiner as TorchMiniMaxH3TokenRefiner
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.token_refiner_minimax_h3 import MiniMaxH3TokenRefiner
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor
from ....utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links
from .common import randomize_norm_weights

# MiniMax-H3 transformer config, shared by the `transformer/` (t2va) and `transformer_ref/` partitions.
HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
FFN_DIM = 14336
NUM_REFINER_LAYERS = 2
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5
FINAL_NORM_EPS = 1e-5


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8), 1, 0, 2, ring_params_req_exact_devices, ttnn.Topology.Ring, False, id="4x8sp1tp0nl2_ring_is_fsdp0"
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "prompt_seq_len",
    [
        pytest.param(512, id="l512"),
        pytest.param(1024, id="l1024"),
    ],
)
def test_minimax_h3_token_refiner(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    prompt_seq_len: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    # Tight for the same reason as the block: the residual updates are unconditional, so the residual
    # stream dominates the output. Measured at real dims against the torch reference, with the norm
    # weights randomized (see `randomize_norm_weights`):
    #   norm weights never loaded  0.8870      swiglu up/gate swapped     0.9934
    #   final_norm skipped         0.8933      norm1/norm2 swapped        0.9861
    #   only 1 of 2 blocks         0.9909      qk-norm skipped            0.9992
    #   block order swapped        0.9992
    # The real implementation measures 0.999986, so 0.9995 clears the worst variant (0.9992).
    MIN_PCC = 0.9995

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    B = 1
    # The refiner runs over the text stream *before* it is scattered into the packed sequence, so it
    # sees a short standalone sequence. Unlike the block, it has no AdaLN, no RoPE and no mask: it is
    # a plain pre-norm transformer over the whole text stream.
    assert prompt_seq_len % ttnn.TILE_SIZE == 0

    torch_model = TorchMiniMaxH3TokenRefiner(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        num_layers=NUM_REFINER_LAYERS,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
        final_norm_eps=FINAL_NORM_EPS,
    ).to(torch.float32)
    # Without this the RMSNorm weights are all ones and norm weight loading is untested; see
    # `randomize_norm_weights`. Must happen before `state_dict()` is read below.
    randomize_norm_weights(torch_model)
    torch_model.eval()

    prompt_input = torch.randn((B, prompt_seq_len, HIDDEN_SIZE), dtype=torch.float32)

    logger.info(f"Running torch model, prompt_seq_len={prompt_seq_len}")
    with torch.no_grad():
        torch_out = torch_model(prompt_input)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_model = MiniMaxH3TokenRefiner(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        num_layers=NUM_REFINER_LAYERS,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
        final_norm_eps=FINAL_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # The text stream is short and every SP device needs the whole of it (each device later scatters
    # text rows into its own slice of the packed sequence), so it is replicated on SP and fractured
    # on TP only.
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3)
    logger.info(f"tt_prompt {tt_prompt.shape}")

    logger.info("Running TT model")
    tt_out = tt_model(tt_prompt)

    # Concat the SP axis onto dim 0 so the replicas can be inspected rather than silently averaged,
    # and the TP axis onto the hidden dim.
    concat_dims = [None, None]
    concat_dims[sp_axis] = 0
    concat_dims[tp_axis] = 3
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    assert tt_out.shape[0] == sp_factor

    # Every SP device must hold an identical copy: the refiner is replicated on that axis, so any
    # divergence means a device read something it should not have.
    for d in range(1, sp_factor):
        torch.testing.assert_close(tt_out[0], tt_out[d], rtol=0, atol=0, msg=f"SP replica {d} diverged from replica 0")

    tt_out = tt_out[:1]
    assert_quality(torch_out, tt_out, pcc=MIN_PCC)
