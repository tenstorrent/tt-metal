# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare traced sliding attention with individually synchronized calls.

Run on an 8x4 Galaxy, without model assets:
    python models/demos/gemma4_d_p/tests/repro_sliding_attention.py
    python models/demos/gemma4_d_p/tests/repro_sliding_attention.py --separate-buffers

The shared-buffer failure is timing-dependent; --repeats controls the run length.
"""

import argparse
import os
from types import SimpleNamespace

import torch

import ttnn
from models.demos.common.prefill.runners.runner_utils import open_mesh_device


def run(mesh, separate_buffers, repeats):
    grid = mesh.compute_with_storage_grid_size()
    bank_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(mesh.dram_grid_size().x)]
    )
    cache_memory = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, 32, 256],
            grid=bank_grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    def tensor(host, dtype, dims, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=dims),
            memory_config=memory_config,
        )

    cases = []
    for phase in (12, 13):
        indices = torch.arange(16 * 16384 * 256, dtype=torch.float32).reshape(1, 16, 16384, 256)
        k = torch.sin(indices * 0.013 + phase).to(torch.bfloat16)
        v = torch.cos(indices * 0.017 + phase).to(torch.bfloat16)
        q = (k[:, :, :8192].repeat_interleave(2, dim=1) * 0.01).contiguous()
        cases.append(
            (
                tensor(q, ttnn.bfloat16, (2, 1)),
                tensor(k, ttnn.bfloat8_b, (2, 1), cache_memory),
                tensor(v, ttnn.bfloat8_b, (2, 1), cache_memory),
            )
        )
    buffers = [
        (
            tensor(torch.zeros(1, 16, 1024, 256), ttnn.bfloat8_b, (None, 1), cache_memory),
            tensor(torch.zeros(1, 16, 1024, 256), ttnn.bfloat8_b, (None, 1), cache_memory),
        )
        for _ in range(128 if separate_buffers else 1)
    ]
    slot = ttnn.from_torch(
        torch.zeros(1, 1, 1, 1, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    position = ttnn.from_torch(
        torch.zeros(1, 1, 1, 1, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
    semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    def attention(index):
        q, k, v = cases[index % 2]
        gathered_k, gathered_v = buffers[index if separate_buffers else 0]
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            None,
            None,
            None,
            persistent_output_buffer_k=gathered_k,
            persistent_output_buffer_v=gathered_v,
            joint_strategy="rear",
            logical_n=16384,
            program_config=program,
            scale=1.0,
            compute_kernel_config=compute,
            dim=2,
            multi_device_global_semaphore=semaphores,
            num_links=2,
            cluster_axis=0,
            mesh_device=mesh,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=ttnn.CoreCoord(grid.x - 1, 0),
            use_column_major_ccl=True,
            is_causal=True,
            is_balanced=False,
            slot_id=slot,
            kv_actual_isl_tensor=position,
            kv_cache_num_layers=1,
            kv_cache_layer_idx=0,
            sliding_window_size=1024,
        )[0]

    references = []
    for index in range(2):
        references.append(attention(index))
        ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    outputs = [attention(index) for index in range(128)]
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    ttnn.synchronize_device(mesh)
    try:
        for repeat in range(repeats):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            for index, output in enumerate(outputs):
                difference = ttnn.ne(output, references[index % 2])
                count = ttnn.sum(difference)
                ttnn.deallocate(difference)
                host = ttnn.from_device(count, blocking=True)
                ttnn.deallocate(count)
                counts = [float(ttnn.to_torch(x).sum()) for x in ttnn.get_device_tensors(host)]
                if any(counts):
                    actual = ttnn.from_device(output, blocking=True)
                    expected = ttnn.from_device(references[index % 2], blocking=True)
                    for device, (actual_shard, reference_shard) in enumerate(
                        zip(ttnn.get_device_tensors(actual), ttnn.get_device_tensors(expected))
                    ):
                        actual_shard = ttnn.to_torch(actual_shard)
                        reference_shard = ttnn.to_torch(reference_shard)
                        changed = (actual_shard != reference_shard).nonzero()
                        if changed.numel():
                            print(
                                f"Mismatch: replay={repeat} call={index} device={divmod(device, 4)} "
                                f"changed={changed.shape[0]} first={changed[0].tolist()} "
                                f"max_abs={(actual_shard - reference_shard).abs().max().item()}",
                                flush=True,
                            )
                    raise AssertionError(f"Attention differs at replay {repeat}, call {index}")
            if (repeat + 1) % 10 == 0:
                print(f"{repeat + 1}/{repeats} replays match", flush=True)
        print(f"PASS: {repeats * len(outputs)} attention calls match their synchronized references", flush=True)
    finally:
        ttnn.release_trace(mesh, trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separate-buffers", action="store_true")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(16)
    os.environ["PREFILL_FABRIC_MODE"] = "1d_ring"
    mesh = open_mesh_device((8, 4), SimpleNamespace(FABRIC_PAYLOAD_SIZE=8192), trace_region_size=256 * 1024 * 1024)
    try:
        run(mesh, args.separate_buffers, args.repeats)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
