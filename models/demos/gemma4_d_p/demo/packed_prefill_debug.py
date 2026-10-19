# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One full-model packed prefill trace replay for debugging."""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE, _get_prefill_tokens
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment, pack_positions, pack_tokens


@torch.no_grad()
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_prefill_packed_4k_traced(mesh_device, reset_seeds):
    chunk_size, max_seq_len = 4096, 262144
    lengths = (256, 512, 768, 1024, 1536)
    mesh = MeshConfig(mesh_device)
    hf_model = os.environ["HF_MODEL"]
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(hf_model))
    caches = allocate_ring_kv_caches(
        mesh, args, num_users=len(lengths), max_seq_len=max_seq_len, prefill_chunk_size=chunk_size
    )
    _, model, _, _ = create_tt_model(
        mesh_config=mesh,
        hf_model_id=hf_model,
        prefill_chunk_size=chunk_size,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        ring_kv_caches=caches,
        stripe_size=32,
    )
    segments = tuple(PrefillSegment(slot, 0, length) for slot, length in enumerate(lengths))
    tokens_all = _get_prefill_tokens(hf_model, len(lengths) * chunk_size, args.vocab_size)[0]
    token_lists = [
        tokens_all[slot * chunk_size : slot * chunk_size + length].tolist() for slot, length in enumerate(lengths)
    ]
    metadata = [model.prefill_metadata] + [PrefillMetadata(mesh) for _ in lengths[1:]]
    for segment, request_metadata in zip(segments, metadata):
        request_metadata.update(slot_idx=segment.slot_id, kv_actual_global=0, valid_global=segment.actual_end)

    def upload(rows):
        return ttnn.from_torch(
            torch.tensor(rows, dtype=torch.int64).reshape(1, chunk_size),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (8, 4), dims=(1, None)),
        )

    tokens = upload(pack_tokens(token_lists, segments, chunk_size=chunk_size, cp=mesh.cp_degree))
    positions = upload(pack_positions(segments, chunk_size=chunk_size, cp=mesh.cp_degree))
    model.set_prefill_rope_positions(positions)
    model._prefill_metadata_external = True

    def forward():
        embeddings = model.transform_and_embed_prefill_inputs_device(tokens)
        return model(hidden_states=embeddings, prefill_batch=tuple(zip(lengths, metadata)))

    warmup = forward()
    ttnn.synchronize_device(mesh_device)
    warmup.deallocate(True)
    output = None
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        try:
            output = forward()
        finally:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        if output is not None:
            output.deallocate(True)
        tokens.deallocate(True)
        positions.deallocate(True)
