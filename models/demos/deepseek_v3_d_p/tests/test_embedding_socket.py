# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test that runs only the DeepSeek V3 parallel embedding on an 8x4 Galaxy mesh.

No RotaryEmbedding, no attention, no FFN — just construct TtParallelEmbedding,
forward a batch of random token IDs, and sanity-check the output shape.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_embedding_8x4_galaxy(mesh_device):
    sp_axis = 0
    tp_axis = 1
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

    seq_len_total = 5 * 1024
    assert seq_len_total % sp_factor == 0, f"seq_len_total ({seq_len_total}) must divide sp_factor ({sp_factor})"
    isl_per_chip = seq_len_total // sp_factor
    assert (
        isl_per_chip % ttnn.TILE_SIZE == 0
    ), f"isl_per_chip ({isl_per_chip}) must be a multiple of TILE_SIZE ({ttnn.TILE_SIZE})"

    vocab_size = DeepSeekV3Config.VOCAB_SIZE
    emb_dim = DeepSeekV3Config.EMB_SIZE

    logger.info(
        f"Running TtParallelEmbedding on {mesh_device.shape} "
        f"(seq_len_total={seq_len_total}, isl_per_chip={isl_per_chip}, vocab_size={vocab_size}, emb_dim={emb_dim})"
    )

    torch.manual_seed(42)
    torch_weight = torch.randn(vocab_size, emb_dim, dtype=torch.float32)
    torch_tokens = torch.randint(0, vocab_size, (sp_factor, 1, isl_per_chip))

    tt_emb = TtParallelEmbedding(
        mesh_device=mesh_device,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        torch_weight=torch_weight,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
    )

    # Push tokens through a persistent H2DStreamService. Worker_cores is set so
    # the service-core kernel multicasts a data-ready inc after each transfer;
    # the h2d_socket_sync op below waits on that, copies the backing tensor
    # into a fresh output, and acks the service core — replacing the host-side
    # `service.barrier()` round-trip.
    per_chip_bytes = isl_per_chip * 4  # uint32
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()],
        ),
    )
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    h2d_service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_chip_bytes,
        scratch_cb_size_bytes=per_chip_bytes,
        mapper=mapper,
        worker_cores=worker_cores,
    )
    logger.info(
        f"[h2d] H2DStreamService built: global_shape=({sp_factor},1,{isl_per_chip}) "
        f"uint32 ROW_MAJOR DRAM, per_chip_bytes={per_chip_bytes}, worker_cores={worker_cores}"
    )

    logger.info(f"torch_tokens shape: {torch_tokens.shape}")
    # uint32 bit pattern == int32 bit pattern for non-negative ids (vocab fits in 18 bits).
    flat_tokens = torch_tokens.to(torch.int32).contiguous().numpy()

    for i in range(20):
        h2d_service.forward_to_tensor_bytes(flat_tokens)
        # Device-side sync: workers wait on data_ready_sem, copy backing -> fresh
        # output, ack consumed_counter. No host barrier needed.
        tt_tokens = ttnn.experimental.deepseek_prefill.h2d_socket_sync(h2d_service)[0]
        logger.info(f"tt_tokens shape (synced from H2D service): {tt_tokens.shape}")

        tt_output = tt_emb(tt_tokens)

        # ttnn.embedding squeezes the leading singleton: per-chip output is
        # [1, isl_per_chip, emb_dim / tp_factor] (3D), not the 4D shape the docstring claims.
        expected_per_chip_shape = (1, isl_per_chip, emb_dim // tp_factor)
        assert (
            tuple(tt_output.shape) == expected_per_chip_shape
        ), f"Embedding output shape {tuple(tt_output.shape)} != expected {expected_per_chip_shape}"
        logger.info(f"Embedding output per-chip shape: {tuple(tt_output.shape)}")

        # Host reference: torch.nn.functional.embedding on the same tokens/weight.
        # F.embedding on tokens [sp_factor, 1, isl_per_chip] returns [sp_factor, 1, isl_per_chip, emb_dim];
        # squeeze the spare singleton to match the gathered TT shape [sp_factor, isl_per_chip, emb_dim].
        torch_output = F.embedding(torch_tokens, torch_weight).squeeze(1)

        # Gather: TP composer composes mesh dim 0 (SP) along tensor dim 0, mesh dim 1 (TP) along dim -1.
        tt_host = ttnn.to_torch(tt_output, mesh_composer=get_tp_mesh_composer(mesh_device), dtype=torch.bfloat16)
        logger.info(f"TT gathered host shape: {tuple(tt_host.shape)}")

        pcc_threshold = 0.999
        _, pcc = comp_pcc(torch_output.float(), tt_host.float())
        logger.info(f"Embedding PCC vs host reference: {pcc:.6f} (threshold {pcc_threshold})")
        assert pcc > pcc_threshold, f"Embedding PCC {pcc:.6f} below threshold {pcc_threshold}"
