# SPDX-License-Identifier: Apache-2.0
"""Device-profiler sweep for BGE-M3 TP2 ring-joint SDPA."""

import os

import pytest
import torch
from tracy import signpost

import ttnn

CHUNK_COMBOS = [(128, 256), (256, 256), (256, 512), (512, 256), (512, 512)]


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["tp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "worker_l1_size": 1_344_544,
            "trace_region_size": 1_000_000,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_ring_sdpa_tp2_worker(mesh_device):
    """Run one exact-shape configuration under the device profiler."""
    q_chunk = int(os.environ["BGE_RING_Q_CHUNK"])
    k_chunk = int(os.environ["BGE_RING_K_CHUNK"])
    fp32_dest = os.environ.get("BGE_RING_FP32_DEST", "1") == "1"
    batch, heads, seq_len, head_dim = 12, 16, 8192, 64

    print(
        f"RING_STEP setup q={q_chunk} k={k_chunk} fp32_dest={fp32_dest}",
        flush=True,
    )
    full_grid = mesh_device.compute_with_storage_grid_size()
    compute_grid = (full_grid.x - 1, full_grid.y)
    ccl_offset = (full_grid.x - 1, 0)
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([all_cores])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    semaphores = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(2)]

    input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 1), dims=(2, None))
    replicated_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    print("RING_STEP allocate inputs", flush=True)

    def make_input():
        return ttnn.from_torch(
            torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=input_mapper,
        )

    q, k, v = make_input(), make_input(), make_input()

    def make_replicated(tensor):
        return ttnn.from_torch(
            tensor,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicated_mapper,
        )

    persistent_k = make_replicated(torch.zeros(batch, heads, seq_len, head_dim, dtype=torch.bfloat16))
    persistent_v = make_replicated(torch.zeros(batch, heads, seq_len, head_dim, dtype=torch.bfloat16))
    dummy = make_replicated(torch.zeros(batch, heads, 0, head_dim, dtype=torch.bfloat16))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=compute_grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest,
        packer_l1_acc=False,
    )

    def run_once():
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            dummy,
            dummy,
            dummy,
            persistent_output_buffer_k=persistent_k,
            persistent_output_buffer_v=persistent_v,
            joint_strategy="rear",
            logical_n=seq_len,
            program_config=program_config,
            compute_kernel_config=compute_config,
            dim=2,
            multi_device_global_semaphore=semaphores,
            num_links=1,
            cluster_axis=0,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            subdevice_id=worker_sub_device_id,
            ccl_core_grid_offset=ccl_offset,
            is_causal=False,
            scale=1.0,
            use_column_major_ccl=True,
        )

    print("RING_STEP compile and warmup", flush=True)
    output, joint_output, lse = run_once()
    ttnn.synchronize_device(mesh_device)
    print(f"RING_STEP warmup complete dtype={output.dtype} shape={output.shape}", flush=True)
    ttnn.deallocate(output)
    ttnn.deallocate(joint_output)
    ttnn.deallocate(lse)

    print("RING_STEP profile one device operation", flush=True)
    signpost("start")
    output, joint_output, lse = run_once()
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    ttnn.deallocate(output)
    ttnn.deallocate(joint_output)
    ttnn.deallocate(lse)
    print("RING_STEP profile dispatch complete", flush=True)


@pytest.mark.timeout(3600)
def test_ring_sdpa_tp2_sweep():
    """Profile each chunk pair in a fresh subprocess and rank device time."""
    from tracy.process_model_log import run_device_profiler
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    streaming_only = os.environ.get("BGE_RING_STREAMING_ONLY", "0") == "1"
    combos = [(512, 512, False)] if streaming_only else [(q, k, True) for q, k in CHUNK_COMBOS]
    results = []
    for q_chunk, k_chunk, fp32_dest in combos:
        dest_label = "fp32" if fp32_dest else "streaming"
        label = f"q{q_chunk}_k{k_chunk}_{dest_label}"
        subdir = f"bge_ring_tp2_{label}"
        os.environ["BGE_RING_Q_CHUNK"] = str(q_chunk)
        os.environ["BGE_RING_K_CHUNK"] = str(k_chunk)
        os.environ["BGE_RING_FP32_DEST"] = "1" if fp32_dest else "0"
        command = (
            "pytest models/demos/wormhole/bge_m3/tests/sweeps/"
            "sweep_ring_sdpa_tp2.py::test_ring_sdpa_tp2_worker -s -q"
        )
        print(f"RING_SWEEP_START {label}", flush=True)
        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            perf = post_process_ops_log(
                subdir,
                float_columns=["DEVICE KERNEL DURATION [ns]"],
                sum_vals=False,
                has_signposts=True,
            )
            durations = perf["DEVICE KERNEL DURATION [ns]"]
            duration_ns = int(max(durations)) if len(durations) else 0
            print(f"RING_SWEEP_RESULT {label} device_ms={duration_ns / 1e6:.3f}", flush=True)
            results.append((duration_ns, label))
        except Exception as error:
            print(f"RING_SWEEP_FAILED {label}: {type(error).__name__}: {error}", flush=True)

    print("RING_SWEEP_RANKED", flush=True)
    for duration_ns, label in sorted(results):
        print(f"  {duration_ns / 1e6:8.3f} ms  {label}", flush=True)
    assert results, "No ring-joint SDPA configuration completed"
