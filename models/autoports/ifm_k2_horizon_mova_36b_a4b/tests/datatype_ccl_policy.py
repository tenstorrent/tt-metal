# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-mesh CCL policy construction, immutable slot reuse and repeated trace replay."""

import json
from pathlib import Path

import torch

import ttnn
from models.common.tt_ccl import get_tt_ccl

from ..tt.multichip_decoder import MultichipDecoder

DOC = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"


def main():
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    traces = []
    rows = []
    try:
        ccl = get_tt_ccl(mesh)
        layers = []
        for dtype in ("bfloat8_b", "bfloat16", "bfloat4_b", "bfloat8_b"):
            # Exercise the exact constructor helper without unrelated checkpoint loading.
            layer = MultichipDecoder.__new__(MultichipDecoder)
            layer.mesh_device = mesh
            layer.ccl = ccl
            layer.residual_sharded = False
            layer.options = dict(ccl_dtype=dtype, num_links=2)
            layer._prepare_all_reduce()
            assert all(t.dtype == getattr(ttnn, dtype) for t, _ in layer.ar_slots)
            layers.append(layer)
        assert layers[0].ar_slots is layers[3].ar_slots
        assert len({id(layer.ar_slots) for layer in layers}) == 3
        values = torch.arange(1, 5, dtype=torch.bfloat16)[:, None, None, None].expand(4, 1, 32, 2560).contiguous()
        source = ttnn.from_torch(
            values,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # Every persistent output, dtype, slot and program precedes capture.
        outputs = [
            ttnn.empty(
                (1, 1, 32, 2560),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=layer.ar_mem,
            )
            for layer in layers
            for _ in (0, 1)
        ]
        for index, layer in enumerate(layers):
            for slot in (0, 1):
                warm = layer._reduce(source, slot)
                ttnn.copy(warm, outputs[2 * index + slot])
                del warm
        prepared = []
        for index, layer in enumerate(layers):
            for slot in (0, 1):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                temporary = layer._reduce(source, slot)
                output = outputs[2 * index + slot]
                ttnn.copy(temporary, output)
                del temporary
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                traces.append(tid)
                prepared.append((index, slot, tid, output))
        for repeat in range(3):
            for index, slot, tid, output in prepared:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                for rank, tensor in enumerate(ttnn.get_device_tensors(output)):
                    assert torch.all(ttnn.to_torch(tensor) == 10), (index, slot, repeat, rank)
                rows.append(
                    dict(
                        policy_index=index,
                        dtype=str(layers[index].ar_dtype),
                        scratch_dtype=str(layers[index].ar_slots[slot][0].dtype),
                        slot=slot,
                        repeat=repeat,
                        trace_id=str(tid),
                        all_ranks_exact=True,
                    )
                )
        (DOC / "debug/ccl_policy_reuse.json").write_text(
            json.dumps(
                dict(
                    status="pass",
                    rows=rows,
                    return_to_bfp8_reuses_original_slots=True,
                    all_preparation_before_first_capture=True,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        for tid in traces:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
