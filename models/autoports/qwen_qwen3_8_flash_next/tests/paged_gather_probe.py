# SPDX-License-Identifier: Apache-2.0
"""Exact full-capacity pooled gather boundary test; serialized TT device use."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.paged_qsa import transfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    context, batch, physical_pages = 262144, 3, 48
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    rows = []
    try:

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        attention = SimpleNamespace(
            blocks=context // 4,
            batch_size=batch,
            device=mesh,
            max_context=context,
            pool_tokens=physical_pages * 32,
        )
        rng = torch.Generator().manual_seed(29)
        source = torch.randn(1, 1, physical_pages * 8, 128, generator=rng).bfloat16()
        device_source = upload(source)
        table = torch.randint(0, physical_pages, (batch, context // 32), generator=rng, dtype=torch.int32)
        device_table = upload(table, ttnn.int32)
        # Deliberately reuse the workspace across long-to-short prefixes and lanes.
        workspace = upload(torch.full((1, 1, context // 4, 128), float("nan")), layout=ttnn.TILE_LAYOUT)
        for start, seq, lane in [
            (context - 8, 8, 2),
            (0, 1, 0),
            (2, 1, 1),
            (3, 1, 1),
            (126, 1, 2),
            (127, 1, 0),
            (128, 8, 1),
            (context - 1, 1, 0),
        ]:
            positions = torch.zeros(batch, 32, dtype=torch.int32)
            positions[:, 0] = start
            device_positions = upload(positions, ttnn.uint32)
            transfer(attention, device_source, workspace, device_positions, device_table, seq=seq, lane=lane)
            complete = (start + seq) // 4
            groups = torch.arange(complete)
            expected = source[0, 0, table[lane, groups // 8].long() * 8 + groups % 8]
            for shard in ttnn.get_device_tensors(workspace):
                actual = ttnn.to_torch(shard)[0, 0, :complete]
                assert torch.equal(actual, expected), (start, seq, lane)
            rows.append(dict(start=start, seq=seq, lane=lane, complete_groups=complete, exact=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    status="pass",
                    context=context,
                    cases=rows,
                    scope="Exact valid-prefix gather; suffix is unspecified and must be masked by the consumer",
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
