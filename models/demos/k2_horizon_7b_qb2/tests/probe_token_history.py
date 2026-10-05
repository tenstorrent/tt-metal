"""Exact token history through traced FP32 cache writes (no host feedback)."""

import argparse
import json
from pathlib import Path

import torch

import ttnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--replays", type=int, default=64)
    args = parser.parse_args()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    trace = None
    try:

        def upload(x, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(
                x, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
            )

        host = torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32) + 261000
        tokens = upload(host, ttnn.uint32)
        pos = upload(
            torch.zeros((1, 1, 1, 32) if args.native else (1,), dtype=torch.int32),
            ttnn.uint32 if args.native else ttnn.int32,
        )
        history = ttnn.zeros(
            (1, 1, 128, 32),
            dtype=ttnn.uint32 if args.native else ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT if args.native else ttnn.TILE_LAYOUT,
            device=mesh,
        )
        if args.native:
            from models.common.sampling.token_history import history_program

            program = history_program(tokens, pos, history)
        memory = ttnn.create_sharded_memory_config(
            (32, 32),
            core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        compute = ttnn.init_device_compute_kernel_config(mesh.arch(), fp32_dest_acc_en=True)

        def record():
            if args.native:
                ttnn.generic_op([tokens, pos, history], program)
            else:
                tiled = ttnn.to_layout(tokens, ttnn.TILE_LAYOUT)
                values = ttnn.to_memory_config(ttnn.typecast(tiled, ttnn.float32), memory)
                ttnn.experimental.paged_update_cache(
                    history, values, update_idxs_tensor=pos, compute_kernel_config=compute
                )
                ttnn.plus_one(pos)
            ttnn.plus_one(tokens)

        record()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        record()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for _ in range(args.replays):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        steps = args.replays + 1
        rows = min(steps, 128)
        indices = torch.arange(rows)
        if args.native:
            indices += ((steps - 1 - indices) // 128) * 128
        expected = host.flatten().long()[None, :] + indices[:, None]
        for rank, device_history in enumerate(ttnn.get_device_tensors(history)):
            actual = ttnn.to_torch(device_history)[0, 0, :rows].long()
            assert torch.equal(actual, expected), (rank, actual, expected)
        if args.native:
            for device_index in ttnn.get_device_tensors(pos):
                assert int(ttnn.to_torch(device_index).flatten()[0]) == steps % 128
        result = {
            "pass": True,
            "steps": steps,
            "slots": 32,
            "ranks_checked": 4,
            "exact_uint32_tokens": True,
            "cursor_wrap_checked": steps > 128,
            "readbacks_inside_replay_loop": 0,
            "method": "native UINT32 copy" if args.native else "FP32 paged_update_cache",
        }
        suffix = "native" if args.native else "float"
        Path(f"models/demos/k2_horizon_7b_qb2/doc/optimized_full_model/token_history_{suffix}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result))
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
