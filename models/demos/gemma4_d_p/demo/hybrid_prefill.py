# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-model sanity for fixed per-layer traces and eager striped attention."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE, _get_prefill_tokens
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.hybrid_prefill import Gemma4HybridPrefill
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment, pack_positions, pack_tokens


@torch.no_grad()
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_prefill_hybrid_traced(mesh_device, reset_seeds):
    chunk_size = int(os.environ.get("GEMMA4_HYBRID_CHUNK_SIZE", "8192"))
    if chunk_size < 4096 or chunk_size % 1024:
        raise ValueError("This sanity test needs a chunk size >=4096 and divisible by 1024")
    schedules = (
        ((chunk_size,), (0,)),
        ((256, 512, 768, 1024, 1536), (27936, 12896, 25248, 29408, 13888)),
        ((1600, 257), (288, 544)),
    )
    mesh = MeshConfig(mesh_device)
    hf_model = os.environ["HF_MODEL"]
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(hf_model))
    caches = allocate_ring_kv_caches(mesh, args, num_users=5, max_seq_len=262144, prefill_chunk_size=chunk_size)
    _, model, _, _ = create_tt_model(
        mesh_config=mesh,
        hf_model_id=hf_model,
        prefill_chunk_size=chunk_size,
        max_seq_len=262144,
        max_batch_size=1,
        ring_kv_caches=caches,
        stripe_size=32,
        max_prefill_requests=5,
    )
    text = _get_prefill_tokens(hf_model, chunk_size, args.vocab_size)[0].tolist()

    def host(rows):
        return ttnn.from_torch(
            torch.tensor(rows, dtype=torch.int64).reshape(1, chunk_size),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (8, 4), dims=(1, None)),
        )

    tokens = ttnn.to_device(host([0] * chunk_size), mesh_device)
    positions = ttnn.to_device(host([0] * chunk_size), mesh_device)
    model.set_prefill_rope_positions(positions)
    hybrid = Gemma4HybridPrefill(model, tokens)
    try:
        for iteration, (lengths, prefixes) in enumerate(schedules):
            segments = tuple(
                PrefillSegment(slot, prefix, prefix + length)
                for slot, (prefix, length) in enumerate(zip(prefixes, lengths))
            )
            logger.info("[hybrid_prefill] lengths={} prefixes={} (synthetic KV history)", lengths, prefixes)
            model.packed_prefill_metadata.update(
                (s.slot_id, s.actual_start, s.actual_end - s.actual_start, s.padded_length()) for s in segments
            )
            values = [text[:length] for length in lengths]
            ttnn.copy_host_to_device_tensor(host(pack_tokens(values, segments, chunk_size=chunk_size)), tokens)
            ttnn.copy_host_to_device_tensor(host(pack_positions(segments, chunk_size=chunk_size)), positions)
            if iteration == 0:
                hybrid.capture()
                trace_ids = tuple(hybrid.traces)
            hybrid.run(blocking=True)
            assert tuple(hybrid.traces) == trace_ids
        logger.info("[hybrid_prefill] reused {} traces for all three request layouts", len(hybrid.traces))
    finally:
        hybrid.close()
        tokens.deallocate(True)
        positions.deallocate(True)
