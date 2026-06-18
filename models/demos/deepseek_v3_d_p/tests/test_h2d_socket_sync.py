# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Op test for the native C++ ``ttnn.experimental.deepseek_prefill.h2d_socket_sync`` (issue #46319).

Mirrors ``test_embedding_socket.py``'s proven H2DStreamService setup, but exercises
the new C++ op directly and asserts the properties that motivated the port:

  * correctness  — each pushed (per-iteration *varying*) token buffer is copied
    verbatim into a fresh output, validated end-to-end through the parallel
    embedding (PCC vs torch reference);
  * metadata     — the inline 3xuint32 control payload round-trips;
  * program cache — after the first call the op is a cache HIT (entry count stays
    constant). This is the whole point of the C++ port: no per-call program build;
  * no clobber   — tokens vary every iteration, so a later transfer overwriting an
    earlier iteration's output would surface as a PCC failure.

NOTE: this is a hardware test (8x4 Galaxy mesh). The mesh_device fixture must have
the program cache enabled (it is on the prefill path). The ConcatMeshToTensor /
to_torch(uint32) calls may need tweaking to match your local ttnn revision.
"""

import struct

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from tests.ttnn.utils_for_testing import comp_pcc

# 3 x uint32: [slot_id, actual_start, actual_end] — matches prefill_h2d_producer.py.
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

    # Persistent H2DStreamService with worker-sync + inline metadata enabled.
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
        fifo_size_bytes=8 * per_chip_bytes,
        scratch_cb_size_bytes=per_chip_bytes,
        mapper=mapper,
        worker_cores=worker_cores,
        metadata_size_bytes=_METADATA_SIZE_BYTES,
    )

    num_iters = 4
    # Measure program-cache growth around ONLY the h2d_socket_sync call, so the
    # check is decoupled from unrelated warmup (the embedding / to_torch
    # legitimately compile a new program on their second invocation).
    op_cache_delta = []
    for i in range(num_iters):
        # Vary tokens every iteration — a later push clobbering an earlier output
        # would make that iteration's PCC fail.
        torch_tokens = torch.randint(0, vocab_size, (sp_factor, 1, isl_per_chip))
        flat_tokens = torch_tokens.to(torch.int32).contiguous().numpy()
        meta = struct.pack("<III", i, 0, isl_per_chip)  # slot_id=i, [actual_start, actual_end)

        service.forward_to_tensor_bytes(flat_tokens, metadata=meta)
        pre = mesh_device.num_program_cache_entries()
        tt_tokens, tt_meta = h2d_socket_sync(service, worker_cores, metadata_size_bytes=_METADATA_SIZE_BYTES)
        op_cache_delta.append(mesh_device.num_program_cache_entries() - pre)

        # --- metadata round-trip (replicated across the mesh; read one copy) ---
        meta_host = ttnn.to_torch(tt_meta, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        meta_vals = meta_host.flatten().to(torch.int64)[:3].tolist()
        assert meta_vals == [i, 0, isl_per_chip], f"iter {i}: metadata mismatch {meta_vals}"

        # --- token correctness, end-to-end through the embedding (PCC) ---
        tt_output = tt_emb(tt_tokens)
        torch_output = F.embedding(torch_tokens, torch_weight).squeeze(1)
        tt_host = ttnn.to_torch(tt_output, mesh_composer=get_tp_mesh_composer(mesh_device), dtype=torch.bfloat16)
        _, pcc = comp_pcc(torch_output.float(), tt_host.float())
        logger.info(f"iter {i}: token PCC={pcc:.6f}, h2d_op_cache_delta={op_cache_delta[-1]}")
        assert pcc > 0.999, f"iter {i}: token PCC {pcc:.6f} below threshold"

    # --- program cache: the op builds its program on the first call, then every
    # subsequent identical call must be a pure cache HIT (zero new entries). A
    # recompiling op (the dispatch-tax regression #46319 targets) would add an
    # entry on every iteration.
    assert op_cache_delta[0] >= 1, f"expected the op to register a program-cache entry on first call: {op_cache_delta}"
    assert all(d == 0 for d in op_cache_delta[1:]), (
        f"h2d_socket_sync added program-cache entries after warmup (per-call deltas={op_cache_delta}): "
        "it is recompiling instead of hitting the cache — exactly the dispatch-tax regression #46319 targets"
    )

    # Drain in-flight writes, then drop the service while the device is still
    # alive so its destructor doesn't run at interpreter exit (which trips the
    # cq_id / service-core-L1 TT_FATALs after the mesh_device fixture closes).
    service.barrier()
    del service
