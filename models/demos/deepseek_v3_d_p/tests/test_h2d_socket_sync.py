# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from tests.ttnn.utils_for_testing import comp_pcc

# 3 x uint32: [slot_id, actual_start, actual_end]
_METADATA_SIZE_BYTES = 12


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
def test_h2d_socket_sync_8x4_galaxy(mesh_device):
    sp_axis, tp_axis = 0, 1
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

    seq_len_total = 5 * 1024
    assert seq_len_total % sp_factor == 0
    isl_per_chip = seq_len_total // sp_factor
    assert isl_per_chip % ttnn.TILE_SIZE == 0

    vocab_size = DeepSeekV3Config.VOCAB_SIZE
    emb_dim = DeepSeekV3Config.EMB_SIZE

    torch.manual_seed(0)
    torch_weight = torch.randn(vocab_size, emb_dim, dtype=torch.float32)
    tt_emb = TtParallelEmbedding(
        mesh_device=mesh_device,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        torch_weight=torch_weight,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
    )

    per_chip_bytes = isl_per_chip * 4  # uint32
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
    )
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        max_socket_page_size_bytes=per_chip_bytes,
        mapper=mapper,
        worker_cores=worker_cores,
        metadata_size_bytes=_METADATA_SIZE_BYTES,
    )

    num_iters = 4
    op_cache_delta = []
    for i in range(num_iters):
        torch_tokens = torch.randint(0, vocab_size, (sp_factor, 1, isl_per_chip))
        flat_tokens = torch_tokens.to(torch.int32).contiguous().numpy()
        meta = struct.pack("<III", i, 0, isl_per_chip)  # slot_id=i, [actual_start, actual_end)

        service.forward_to_tensor_bytes(flat_tokens, metadata=meta)
        pre = mesh_device.num_program_cache_entries()
        tt_tokens, tt_meta = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            service, metadata_size_bytes=_METADATA_SIZE_BYTES
        )
        op_cache_delta.append(mesh_device.num_program_cache_entries() - pre)

        meta_host = ttnn.to_torch(tt_meta, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        meta_vals = meta_host.flatten().to(torch.int64)[:3].tolist()
        assert meta_vals == [i, 0, isl_per_chip], f"iter {i}: metadata mismatch {meta_vals}"

        tt_output = tt_emb(tt_tokens)
        torch_output = F.embedding(torch_tokens, torch_weight).squeeze(1)
        tt_host = ttnn.to_torch(tt_output, mesh_composer=get_tp_mesh_composer(mesh_device), dtype=torch.bfloat16)
        _, pcc = comp_pcc(torch_output.float(), tt_host.float())
        logger.info(f"iter {i}: token PCC={pcc:.6f}, h2d_op_cache_delta={op_cache_delta[-1]}")
        assert pcc > 0.999, f"iter {i}: token PCC {pcc:.6f} below threshold"

    assert op_cache_delta[0] >= 1, f"expected a program-cache entry on the first call: {op_cache_delta}"
    assert all(d == 0 for d in op_cache_delta[1:]), f"op recompiled instead of cache-hitting: {op_cache_delta}"

    service.barrier()
    del service
