# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import torch

import ttnn

from ..tt.token_history import append_tokens


def test_token_history_trace():
    from ..tt.generator_vllm import GraniteForCausalLM

    ttnn.set_fabric_config(**GraniteForCausalLM.model_capabilities["fabric_config"])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10000000)
    try:

        def put(x):
            return ttnn.from_torch(
                x.int(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        states = []
        for b in (1, 8, 16):
            tokens = put(torch.arange(b).reshape(1, 1, 1, b) + 100000)
            history = put(torch.zeros(257, 1, 1, b))
            cursor = put(torch.zeros(1))
            append_tokens(tokens, history, cursor)
            states.append((b, tokens, history, cursor))
        traces = []
        for b, tokens, history, cursor in states:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            append_tokens(tokens, history, cursor)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces.append(trace)
        for state, trace in zip(states, traces):
            b, tokens, history, cursor = state
            for step in range(1, 259):
                host = ttnn.from_torch(
                    (torch.arange(b).reshape(1, 1, 1, b) + 100000 + step * 16).int(),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                ttnn.copy_host_to_device_tensor(host, tokens)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            for shard in ttnn.get_device_tensors(history):
                actual = ttnn.to_torch(shard).reshape(257, b).long()
                assert torch.equal(
                    actual, (torch.arange(b)[None, :] + 100000 + torch.arange(257)[:, None] * 16)
                ), actual
            assert int(ttnn.to_torch(ttnn.get_device_tensors(cursor)[0])[0]) == 257
            print("HISTORY_OK", b, flush=True)
        for trace in traces:
            ttnn.release_trace(mesh, trace)
    finally:
        ttnn.close_mesh_device(mesh)
