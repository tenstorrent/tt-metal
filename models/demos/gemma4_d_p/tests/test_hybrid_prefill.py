# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tests.test_packed_prefill import _natural_output, synthetic_model_inputs
from models.demos.gemma4_d_p.tt.ccl import CCLManager
from models.demos.gemma4_d_p.tt.hybrid_prefill import Gemma4HybridPrefill
from models.demos.gemma4_d_p.tt.model import Gemma4Model
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment, pack_positions, pack_tokens
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


@torch.no_grad()
@pytest.mark.timeout(600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 32 * 1024 * 1024})
def test_hybrid_prefill_layout_changes(mesh_device):
    args, state, reference = synthetic_model_inputs()
    mesh = MeshConfig(mesh_device)
    chunk_size = 8192
    caches = allocate_ring_kv_caches(mesh, args, num_users=6, max_seq_len=16384)
    model = Gemma4Model(
        mesh,
        args,
        state,
        CCLManager(mesh),
        prefill_chunk_size=chunk_size,
        precision=Gemma4Precision({"kv_cache": ttnn.bfloat8_b}),
        max_seq_len=16384,
        ring_kv_caches=caches,
        stripe_size=32,
        max_prefill_requests=6,
    )

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
    histories = [[] for _ in range(6)]
    schedules = (
        ((0, 0, 288), (1, 0, 544), (2, 0, 800)),
        ((2, 800, 257), (0, 288, 1600)),
        ((3, 0, 8192),),
        tuple((i, 0, n) for i, n in enumerate((256, 512, 768, 1024, 1536))),
        ((5, 0, 3840), (4, 0, 4352)),
        ((4, 0, 257),),
    )
    try:
        for iteration, schedule in enumerate(schedules):
            segments = tuple(PrefillSegment(slot, start, start + length) for slot, start, length in schedule)
            token_lists = [torch.randint(1, 256, (s.actual_end - s.actual_start,)).tolist() for s in segments]
            for segment, values in zip(segments, token_lists):
                histories[segment.slot_id] = histories[segment.slot_id][: segment.actual_start] + values
            model.packed_prefill_metadata.update(
                (s.slot_id, s.actual_start, s.actual_end - s.actual_start, s.padded_length()) for s in segments
            )
            ttnn.copy_host_to_device_tensor(host(pack_tokens(token_lists, segments, chunk_size=chunk_size)), tokens)
            ttnn.copy_host_to_device_tensor(host(pack_positions(segments, chunk_size=chunk_size)), positions)
            if iteration == 0:
                hybrid.capture()
                trace_ids = tuple(hybrid.traces)
                assert len(trace_ids) == 2 * len(model.layers)
            hybrid.run(blocking=True)
            assert tuple(hybrid.traces) == trace_ids
            output = _natural_output(hybrid.output)
            offset = 0
            for segment in segments:
                expected = reference(torch.tensor([histories[segment.slot_id]]), use_cache=False).last_hidden_state[0]
                expected = expected[segment.actual_start : segment.actual_end]
                packed = output[offset : offset + segment.padded_length()].reshape(-1, 8, 32, output.shape[-1])
                actual = packed.roll(-(segment.actual_start // 32 % 8), dims=1).reshape(-1, output.shape[-1])
                actual = actual[: len(expected)]
                assert torch.isfinite(actual).all()
                passed, pcc = comp_pcc(expected, actual, 0.98)
                assert passed, f"iteration={iteration} request={segment}: {pcc}"
                offset += segment.padded_length()
        for buffers in hybrid.buffers.values():
            output = ttnn.to_torch(ttnn.get_device_tensors(buffers.output)[0])
            assert torch.count_nonzero(output[:, :, 64:, :]) == 0
        programs = mesh_device.num_program_cache_entries()
        hybrid.run(blocking=True)
        assert mesh_device.num_program_cache_entries() == programs
        assert torch.equal(_natural_output(hybrid.output), _natural_output(hybrid.run(blocking=True)))
    finally:
        hybrid.close()
        hybrid.close()
        tokens.deallocate(True)
        positions.deallocate(True)
