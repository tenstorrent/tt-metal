# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Per-block latency benchmark for the Ideogram 4.0 single-stream block. NOT part
# of the correctness gate (verify.py targets test_transformer_ideogram4.py).
# Reuses the correctness test's input builder; warms up the program cache, then
# times N forward passes with a device sync and logs ms/block per config.
#
# Run: pytest .../test_transformer_ideogram4_perf.py -s -k <config>
# =============================================================================

from time import time

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.transformer_ideogram4 import Ideogram4TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.padding import PaddingConfig
from ....utils.tensor import bf16_tensor
from .test_transformer_ideogram4 import (
    ADALN_DIM,
    EMB_DIM,
    HEAD_DIM,
    INTERMEDIATE_SIZE,
    NORM_EPS,
    NUM_HEADS,
    _build_inputs,
    _sp_padded_len,
)

WARMUP_ITERS = 2
TIMED_ITERS = 10


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 4), (1, 1), 0, 1, 1, id="tp1sp1"),
        pytest.param((2, 4), (1, 2), 0, 1, 1, id="tp2"),
        pytest.param((2, 4), (1, 4), 0, 1, 1, id="tp4"),
        pytest.param((2, 4), (2, 1), 0, 1, 1, id="sp2"),
        pytest.param((2, 4), (2, 2), 0, 1, 1, id="sp2tp2"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, id="sp2tp4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    ("batch_size", "text_len", "image_len"),
    [pytest.param(1, 128, 4096, id="b1_text128_img4096")],
)
def test_transformer_block_perf(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    text_len: int,
    image_len: int,
) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    seq_len = text_len + image_len

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sp_factor = tuple(submesh_device.shape)[sp_axis]
    tp_factor = tuple(submesh_device.shape)[tp_axis]

    x, adaln_input, _segment_ids, cos, sin, _attn_bias = _build_inputs(batch_size, text_len, image_len, torch_dtype)

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    ccl_manager = CCLManager(submesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    padding_config = (
        PaddingConfig.from_tensor_parallel_factor(NUM_HEADS, HEAD_DIM, tp_factor)
        if NUM_HEADS % tp_factor != 0
        else None
    )

    tt_block = Ideogram4TransformerBlock(
        hidden_size=EMB_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adaln_dim=ADALN_DIM,
        mesh_device=submesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    # Random weights are fine for a latency measurement.
    tt_block.load_torch_state_dict(_reference_state_dict(torch_dtype))

    padded_len = _sp_padded_len(seq_len, sp_factor)
    cos4 = cos.unsqueeze(1)
    sin4 = sin.unsqueeze(1)
    if sp_factor > 1:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded_len - seq_len))
        cos4 = torch.nn.functional.pad(cos4, (0, 0, 0, padded_len - seq_len))
        sin4 = torch.nn.functional.pad(sin4, (0, 0, 0, padded_len - seq_len))
        tt_x = bf16_tensor(x, device=submesh_device, mesh_axis=sp_axis, shard_dim=1)
        tt_cos = bf16_tensor(cos4, device=submesh_device, mesh_axis=sp_axis, shard_dim=2)
        tt_sin = bf16_tensor(sin4, device=submesh_device, mesh_axis=sp_axis, shard_dim=2)
    else:
        tt_x = bf16_tensor(x, device=submesh_device)
        tt_cos = bf16_tensor(cos4, device=submesh_device)
        tt_sin = bf16_tensor(sin4, device=submesh_device)
    tt_adaln = bf16_tensor(adaln_input, device=submesh_device)

    def _run():
        return tt_block(tt_x, cos=tt_cos, sin=tt_sin, adaln_input=tt_adaln, spatial_sequence_length=seq_len)

    for _ in range(WARMUP_ITERS):
        _run()
    ttnn.synchronize_device(submesh_device)

    start = time()
    for _ in range(TIMED_ITERS):
        out = _run()
    ttnn.synchronize_device(submesh_device)
    ms = (time() - start) * 1000 / TIMED_ITERS
    ttnn.deallocate(out)

    logger.info(
        f"IDEOGRAM4 BLOCK PERF | submesh={submesh_shape} tp={tp_factor} sp={sp_factor} "
        f"B={batch_size} seq={seq_len} | {ms:.2f} ms/block"
    )


def _reference_state_dict(torch_dtype):
    from ....reference.ideogram4 import modeling_ideogram4

    block = modeling_ideogram4.Ideogram4TransformerBlock(
        hidden_size=EMB_DIM,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adanln_dim=ADALN_DIM,
    ).to(dtype=torch_dtype)
    return block.state_dict()
