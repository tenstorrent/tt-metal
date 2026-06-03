# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Rigorous proof that the H2D socket path delivers byte-identical model input.

Generated-token comparison can't isolate the socket: prefill output is
non-deterministic run-to-run (same input → different first tokens across
processes). So instead we compare the *input tensor the model receives*, which
is fully deterministic and is the only thing the socket touches:

    reference = the device tensor TtDeepSeekPrefillPipeline._prepare_input_tensor
                would build for these token_ids (the non-socket path's input)
    delivered = the tensor h2d_socket_sync returns after the same token_ids are
                pushed through the H2DStreamService (the socket path's input)

If delivered == reference element-for-element (and the metadata round-trips
intact), the socket provably introduces no input degradation; any downstream
token variance is the known model non-determinism, not the socket.

No model is built — this is a fast, device-only transport check.
"""

import json
import struct
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync

# _tokens_to_host_tensor lives in the producer (the runner no longer carries a
# host-side input builder — its request loop reads via h2d_socket_sync). The
# producer's copy is identical reorder logic and asserts the same MAX_SEQ_LEN
# length this test pads to.
from models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer import _tokens_to_host_tensor
from models.demos.deepseek_v3_d_p.tt.runners.prefill_runner import (
    GLOBAL_MESH_SHAPE,
    H2D_MAPPER_CONFIG,
    H2D_METADATA_SIZE_BYTES,
    H2D_SYNC_WORKER_CORES,
    MAX_SEQ_LEN,
    _build_h2d_service,
)

_INPUT_JSON = Path(__file__).parents[1] / "tt" / "runners" / "standalone_input.json"


def _reference_input_tensor(mesh_device, token_ids):
    """Mirror TtDeepSeekPrefillPipeline._prepare_input_tensor (is_balanced path):
    balanced chunk reorder -> reshape (sp,1,isl) -> shard dim0 across sp."""
    sp_factor = GLOBAL_MESH_SHAPE[0]
    isl_per_chip = MAX_SEQ_LEN // sp_factor
    chunk_order = create_balanced_chunk_order(sp_factor)
    t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
    sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    return ttnn.from_torch(
        sharded,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=GLOBAL_MESH_SHAPE, dims=(0, None)),
    )


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
def test_h2d_socket_input_equivalence(mesh_device):
    # --- input: same JSON the runner/producer use, padded like the producer ---
    data = json.loads(_INPUT_JSON.read_text())
    token_ids = list(data["token_ids"])
    actual_isl = len(token_ids)
    if len(token_ids) < MAX_SEQ_LEN:
        token_ids = token_ids + [1] * (MAX_SEQ_LEN - len(token_ids))
    # PrefillMetadata wire format: (slot_id, actual_start, actual_end).
    # Single-chunk demo: whole prompt on slot 0 starting at KV pos 0.
    slot_id, actual_start, actual_end = 0, 0, actual_isl
    logger.info(f"[equiv] actual_isl={actual_isl} padded_len={len(token_ids)}")

    # --- reference: what _prepare_input_tensor would feed the model (no socket) ---
    reference = _reference_input_tensor(mesh_device, token_ids)
    ref_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(reference)]

    # --- socket: push the same tokens + metadata, pull via h2d_socket_sync ---
    service = _build_h2d_service(mesh_device)
    mapper = ttnn.create_mesh_mapper(mesh_device, H2D_MAPPER_CONFIG)
    host_tokens = _tokens_to_host_tensor(token_ids, mapper)
    metadata = struct.pack("<III", slot_id, actual_start, actual_end)
    service.forward_to_tensor(host_tokens, metadata=metadata)
    delivered, tt_metadata = h2d_socket_sync(
        service, H2D_SYNC_WORKER_CORES, metadata_size_bytes=H2D_METADATA_SIZE_BYTES
    )
    del_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(delivered)]

    # --- (1) token tensor: byte/element exact across every device shard ---
    assert len(del_shards) == len(ref_shards), f"shard count {len(del_shards)} != {len(ref_shards)}"
    mismatches = [i for i, (d, r) in enumerate(zip(del_shards, ref_shards)) if not torch.equal(d, r)]
    assert not mismatches, f"socket-delivered tokens differ from reference on device shards {mismatches}"
    logger.info(f"[equiv] PASS: {len(del_shards)} device shards byte-identical (socket == non-socket input)")

    # --- (2) metadata round-trips intact ---
    meta = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
    assert (
        int(meta[0]) == slot_id and int(meta[1]) == actual_start and int(meta[2]) == actual_end
    ), f"metadata mismatch: got {meta[:3].tolist()} expected [{slot_id}, {actual_start}, {actual_end}]"
    logger.info(f"[equiv] PASS: PrefillMetadata round-trip " f"[slot_id,actual_start,actual_end]={meta[:3].tolist()}")

    del service
