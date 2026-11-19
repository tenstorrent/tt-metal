# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shape-faithful fused boundaries, including the sharded next norm consumer."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer, pcc
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best-separate", action="store_true")
    p.add_argument("--packed-swiglu", action="store_true")
    p.add_argument("--device-fence", action="store_true")
    p.add_argument("--family", choices=["mmrs", "agmm", "agminimal"], required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--role", choices=["attention", "mlp"], default="mlp")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--sync-replays",
        action="store_true",
        help="Diagnostic: synchronize every replay; timings include host fences and phase prints.",
    )
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(28)
    root = Path(__file__).resolve().parents[1]
    cfg = Qwen3_5TextConfig(**json.loads((root / "doc/hf_config.json").read_text())["text_config"])
    weights = load_layer(a.layer)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
    try:
        m = MultichipDecoder.from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=a.layer,
            mesh_device=mesh,
            policy=(
                {}
                if a.packed_swiglu
                else {
                    "packed_mlp": False,
                    "minimal_swiglu": False,
                    "fused_mmrs": False,
                    "attention_fidelity": "HiFi4" if a.layer == 0 else "LoFi",
                }
            ),
        )
        print("CCL_LINKS", m.ccl.num_links, flush=True)
        role = a.role
        key = (
            (
                "mlp.down_proj.weight"
                if role == "mlp"
                else ("linear_attn.out_proj.weight" if a.layer == 0 else "self_attn.o_proj.weight")
            )
            if a.family == "mmrs"
            else ("mlp.gate_proj.weight" if role == "mlp" else ("packed_gdn" if a.layer == 0 else "packed_qkv"))
        )
        if a.packed_swiglu:
            assert a.family == "agminimal" and role == "mlp"
            key = "mlp.gate_up.weight"
        w = ttnn.to_memory_config(m.weights[key], ttnn.DRAM_MEMORY_CONFIG)
        k, n = w.shape
        gather_buffer = None
        if a.family == "agminimal":
            # Own scratch across warmup, capture, and every replay. Linear's native
            # barrier argument does not provide an inter-launch entry fence.
            gather_buffer = m._upload(torch.zeros(1, 1, a.length, k, dtype=torch.bfloat16))
            grid = mesh.compute_with_storage_grid_size()
            semaphore_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
            )
            fused_semaphores = [ttnn.create_global_semaphore(mesh, semaphore_cores, 0) for _ in range(2)]
            fused_barrier = ttnn.create_global_semaphore(mesh, semaphore_cores, 0)
            fence_input = m._upload(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16))
            fence_output = m._upload(torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16))
            print("PHASE agminimal persistent scratch and dedicated semaphores ready", flush=True)

        def read(t):
            return torch.cat([ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)], -1)

        if a.family == "mmrs":
            x = m._upload_shards([torch.randn(1, a.length, k, dtype=torch.bfloat16) * 0.1 for _ in range(4)])
            residual = m.upload_residual(torch.randn(1, a.length, 5120, dtype=torch.bfloat16) * 0.1)

            def consumer(t):
                return m._norm(ttnn.add(residual, t), "post_attention_layernorm.weight")

            def separate():
                return consumer(m._reduce(m._linear(x, key)))

            def fused():
                y = ttnn.experimental.matmul_reduce_scatter_prefill(
                    x,
                    w,
                    cluster_axis=1,
                    scatter_dim=-1,
                    topology=ttnn.Topology.Linear,
                    num_links=m.ccl.num_links,
                    compute_kernel_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi2 if role == "mlp" else ttnn.MathFidelity.HiFi4
                    ),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                return consumer(y)

        else:
            x = m.upload_residual(torch.randn(1, a.length, 5120, dtype=torch.bfloat16) * 0.1)
            x4 = ttnn.reshape(x, (1, 1, a.length, 1280))
            w4 = ttnn.reshape(w, (1, 1, k, n))
            pm = math.ceil(a.length / 32 / 4)
            pn = math.ceil(n / 32 / 8)
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=2 if pn % 2 == 0 else 1,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )

            def separate():
                gathered = m._collective(x)
                if a.packed_swiglu:
                    return ttnn.experimental.minimal_matmul(
                        gathered,
                        m.swiglu_weight,
                        config=ttnn.MinimalMatmulConfig(
                            M_block_size=4,
                            K_block_size=4,
                            N_block_size=16,
                            subblock_h=1,
                            subblock_w=4,
                            compute_with_storage_grid_size=(11, 10),
                        ),
                        fuse_swiglu=True,
                        compute_kernel_config=m.matmul_configs[role],
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                if a.best_separate:
                    assert key == "mlp.gate_proj.weight", "This control includes baseline packed correction otherwise"
                    return m._linear(gathered, key)
                return ttnn.linear(
                    gathered,
                    w,
                    program_config=config,
                    compute_kernel_config=m.matmul_configs[role],
                    dtype=ttnn.float32 if key == "packed_gdn" else ttnn.bfloat16,
                )

            def fused():
                if a.family == "agminimal":
                    if a.device_fence:
                        ttnn.experimental.all_gather_async(
                            fence_input,
                            dim=3,
                            cluster_axis=1,
                            mesh_device=mesh,
                            topology=ttnn.Topology.Linear,
                            num_links=2,
                            multi_device_global_semaphore=m.ccl.get_ag_ping_pong_semaphore(),
                            barrier_semaphore=m.ccl.get_barrier_semaphore(),
                            persistent_output_tensor=fence_output,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    values = ttnn.experimental.all_gather_minimal_matmul_async(
                        x4,
                        w4,
                        config=ttnn.MinimalMatmulConfig(
                            M_block_size=4,
                            K_block_size=4,
                            N_block_size=4,
                            subblock_h=2,
                            subblock_w=2,
                            compute_with_storage_grid_size=(8, 8),
                        ),
                        multi_device_global_semaphore=fused_semaphores,
                        barrier_semaphore=fused_barrier,
                        persistent_output_buffer=gather_buffer,
                        num_links=m.ccl.num_links,
                        num_workers_per_link=8 // m.ccl.num_links,
                        cluster_axis=1,
                        topology=ttnn.Topology.Linear,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=m.matmul_configs[role],
                        dtype=ttnn.float32 if key == "packed_gdn" else ttnn.bfloat16,
                        force_transpose=False,
                    )
                    result = ttnn.reshape(values[0], (1, a.length, n))
                    if a.packed_swiglu:
                        gate, up = result[:, :, :4352], result[:, :, 4352:8704]
                        return (
                            ttnn.multiply(ttnn.silu(gate), up)
                            if m.policy["round_silu"]
                            else ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                        )
                    return result[:, :, : m.projection_info[key]["logical_n"]] if a.best_separate else result
                gathered, y = ttnn.experimental.all_gather_matmul_async(
                    x4,
                    w4,
                    dim=3,
                    multi_device_global_semaphore=m.ccl.get_ag_ping_pong_semaphore(),
                    all_gather_core_grid_offset=(0, 4),
                    barrier_semaphore=m.ccl.get_barrier_semaphore(),
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=config,
                    compute_kernel_config=m.matmul_configs[role],
                    dtype=ttnn.float32 if key == "packed_gdn" else ttnn.bfloat16,
                )
                return ttnn.reshape(y, (1, a.length, n))

        print("PHASE eager separate begin", flush=True)
        ref = read(separate())
        print("PHASE eager fused begin", flush=True)
        got = read(fused())
        correlation = pcc(ref, got)
        print("EAGER_PCC", correlation, flush=True)
        assert correlation >= 0.995 and torch.isfinite(got).all(), correlation
        timings = {}
        replay_pcc = {}
        for name, operation in [("separate", separate), ("fused", fused)]:
            for iteration in range(2):
                print(f"PHASE {name} warmup {iteration} begin", flush=True)
                warm = operation()
                ttnn.synchronize_device(mesh)
                del warm
                print(f"PHASE {name} warmup {iteration} complete", flush=True)
            ttnn.synchronize_device(mesh)
            print(f"PHASE {name} capture begin", flush=True)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = operation()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            print(f"PHASE {name} capture complete; first replay begin", flush=True)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            first_pcc = pcc(read(out), ref)
            assert first_pcc >= 0.995, first_pcc
            print(f"PHASE {name} first replay complete PCC={first_pcc}", flush=True)
            ttnn.synchronize_device(mesh)
            print(f"PHASE {name} repeated replay begin sync={a.sync_replays}", flush=True)
            start = time.perf_counter()
            for iteration in range(50):
                if a.sync_replays:
                    print(f"PHASE {name} replay {iteration} begin", flush=True)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                if a.sync_replays:
                    ttnn.synchronize_device(mesh)
                    print(f"PHASE {name} replay {iteration} complete", flush=True)
            ttnn.synchronize_device(mesh)
            timings[name] = (time.perf_counter() - start) * 1000 / 50
            last_pcc = pcc(read(out), ref)
            assert last_pcc >= 0.995, last_pcc
            replay_pcc[name] = {"first": first_pcc, "last": last_pcc}
            print(f"PHASE {name} repeated replay complete PCC={last_pcc}", flush=True)
            ttnn.release_trace(mesh, tid)
            del out
        report = dict(
            family=a.family,
            layer=a.layer,
            role=role,
            length=a.length,
            key=key,
            pcc=correlation,
            timing_ms=timings,
            next_consumer=(
                "sharded residual add + distributed norm + delayed gather"
                if a.family == "mmrs"
                else "column-parallel matmul"
            ),
            weight_copy="setup interleaved from DRAM-sharded; extra bytes included if selected",
            source_sha256=hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        )
        report.update(
            packed_swiglu=a.packed_swiglu,
            selected_policy=dict(m.policy),
            sync_replays=a.sync_replays,
            device_fence=a.device_fence,
            best_separate=a.best_separate,
            timing_scope=(
                "diagnostic wall time including per-replay host synchronization and prints"
                if a.sync_replays
                else "queued trace wall time with synchronization at endpoints"
            ),
            replay_pcc=replay_pcc,
            persistent_gather_bytes_per_device=(
                math.ceil(a.length / 32) * 32 * k * 2 if gather_buffer is not None else 0
            ),
            dedicated_fused_semaphores=a.family == "agminimal",
        )
        if a.family == "mmrs":
            report["native_plan"] = dict(
                ttnn._ttnn.operations.experimental.ccl_experimental._matmul_reduce_scatter_prefill_last_plan()
            )
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
