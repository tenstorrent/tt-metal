"""Shape-faithful WO -> residual add -> next HC read topology experiments.

All candidates include the next consumer, with no benchmark-boundary gather.
Real checkpoint weights; random attention activations are an op/layout probe,
not a substitute for complete-decoder precision validation.
"""

import argparse, hashlib, json, math, statistics, time
from pathlib import Path
import torch, ttnn
from transformers import AutoConfig
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    MeshOps,
    MeshPolicy,
    ParallelHyperConnection,
    _hidden_parts,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedHyperConnection
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_components import CHECKPOINT
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics


class ReplicatedHC(OptimizedHyperConnection):
    dram_readers = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["rs", "ar", "ag", "agmm", "agmm_minimal", "mmrs"], required=True)
    p.add_argument("--length", type=int, default=1)
    p.add_argument("--links", type=int, default=2)
    p.add_argument("--dtype", choices=["bf16", "bf8"], default="bf16")
    p.add_argument("--ring", action="store_true")
    p.add_argument("--lofi", action="store_true")
    p.add_argument("--persistent", action="store_true")
    p.add_argument("--tag", default="topology")
    p.add_argument("--block", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, "model.language_model.layers.0.")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if a.ring else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    trace = None
    try:
        ops = MeshOps(mesh, MeshPolicy(links=a.links))
        s = a.length
        hc = (
            ReplicatedHC.from_state_dict(state, prefix="mlp_hyper_connection", hf_config=cfg, mesh_device=mesh)
            if a.family == "ar"
            else ParallelHyperConnection(state, "mlp_hyper_connection", cfg, ops)
        )
        weight = state["linear_attn.out_proj.weight"].T.float()
        x = torch.randn(1, 1, s, 6144).bfloat16() * 0.03
        residual = torch.randn(1, 1, s, 10240).bfloat16() * 0.05
        inject = torch.sigmoid(torch.randn(1, 1, s, 4)).bfloat16()
        dtype = ttnn.bfloat8_b if a.dtype == "bf8" else ttnn.bfloat16
        xd = ops.shards(list(x.chunk(4, -1)), dtype)
        rd = ops.upload(residual) if a.family == "ar" else ops.shards(_hidden_parts(residual))
        inj = ops.upload(inject)
        row = ops.shards([w.unsqueeze(0).unsqueeze(0) for w in weight.chunk(4, 0)], ttnn.bfloat8_b)
        col = ops.shards([w.unsqueeze(0).unsqueeze(0) for w in weight.chunk(4, 1)], ttnn.bfloat8_b)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi if a.lofi else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        interm = ops.upload(torch.zeros(1, 1, s, 2560), dtype)
        rsout = ops.upload(torch.zeros(1, 1, s, 640), dtype)
        gather = ops.upload(torch.zeros(1, 1, s, 6144), dtype)
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=a.block,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=max(1, math.ceil(s / 32 / 8)),
            per_core_N=10,
            out_block_w=5,
            transpose_mcast=False,
            fuse_batch=False,
            allowed_worker_cores=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
        )
        agpc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(10, 4),
            in0_block_w=a.block,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=math.ceil(s / 32),
            per_core_N=1,
            fuse_batch=True,
            mcast_in0=True,
        )
        mini = ttnn.MinimalMatmulConfig(
            M_block_size=1,
            K_block_size=a.block,
            N_block_size=4,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        )

        def forward():
            if a.family == "rs":
                y = ops.scatter_reduce(ops.linear(xd, row, compute_kernel_config=compute))
            elif a.family == "ar":
                y = ops.reduce(ttnn.linear(xd, row, compute_kernel_config=compute))
            elif a.family == "ag":
                y = ops.linear(ops.gather(xd), col, compute_kernel_config=compute)
            elif a.family == "agmm_minimal":
                y = ttnn.experimental.all_gather_minimal_matmul_async(
                    input_tensor=xd,
                    weight_tensor=col,
                    config=mini,
                    compute_kernel_config=compute,
                    multi_device_global_semaphore=ops.ccl.get_and_cycle_ag_semaphore_handles(1),
                    num_links=a.links,
                    topology=ttnn.Topology.Ring if a.ring else ttnn.Topology.Linear,
                    cluster_axis=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=dtype,
                    force_transpose=True,
                    num_workers_per_link=8 // a.links,
                    num_buffers_per_channel=8,
                    persistent_output_buffer=gather if a.persistent else None,
                )[0]
            elif a.family == "agmm":
                _, y = ttnn.experimental.all_gather_matmul_async(
                    xd,
                    col,
                    persistent_output_buffer=gather if a.persistent else None,
                    dim=3,
                    multi_device_global_semaphore=ops.ccl.get_and_cycle_ag_semaphore_handles(1),
                    all_gather_core_grid_offset=(0, 4),
                    barrier_semaphore=ops.ccl.get_and_cycle_barrier_semaphore_handle(1),
                    num_links=a.links,
                    topology=ttnn.Topology.Ring if a.ring else ttnn.Topology.Linear,
                    memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=agpc,
                    compute_kernel_config=compute,
                )
            else:
                _, y = ttnn.experimental.matmul_reduce_scatter_async(
                    xd,
                    row,
                    persistent_intermediate_buffer=interm,
                    persistent_output_buffer=rsout,
                    dim=3,
                    multi_device_global_semaphore=ops.ccl.get_and_cycle_rs_semaphore_handles(),
                    barrier_semaphore=ops.ccl.get_and_cycle_barrier_semaphore_handle(),
                    reduce_scatter_core_grid_offset=(0, 8),
                    num_links=a.links,
                    topology=ttnn.Topology.Ring if a.ring else ttnn.Topology.Linear,
                    memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=pc,
                    compute_kernel_config=compute,
                )
            updated = hc.write(rd, y, inj)
            mixed, _ = hc.read(updated)
            return mixed

        for _ in range(2):
            out = forward()
            ttnn.synchronize_device(mesh)
        reference = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        samples = []
        for _ in range(5):
            before = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - before) * 1000 / 20)
        actual = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
        assert torch.equal(actual, reference)
        dest = Path("models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder") / a.tag
        dest.mkdir(parents=True, exist_ok=True)
        external = (
            Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip")
            / f"{a.tag}_{a.family}_s{s}_links{a.links}_persistent{a.persistent}_ring{a.ring}_{a.dtype}_block{a.block}.pt"
        )
        torch.save(actual, external)
        control = external.with_name(f"topology_rs_s{s}.pt")
        metric = pcc_metrics(torch.load(control, weights_only=True), actual) if control.exists() else None
        if metric:
            assert metric["pcc"] >= 0.995, metric
        result = dict(
            args=vars(a), median_ms=statistics.median(samples), samples_ms=samples, pcc=metric, output=str(external),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256((Path(__file__).parents[1] / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        )
        (dest / "runner.py.txt").write_bytes(Path(__file__).read_bytes())
        (
            dest / f"{a.family}_s{s}_links{a.links}_persistent{a.persistent}_ring{a.ring}_{a.dtype}_block{a.block}.json"
        ).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
