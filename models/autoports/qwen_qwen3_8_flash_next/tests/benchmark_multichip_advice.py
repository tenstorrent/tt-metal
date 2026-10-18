"""Whole-decoder, precision-locked matmul and CCL advice experiments."""

import argparse, math, sys, hashlib
from pathlib import Path
import torch, ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_native_experts import NativeExpertParallel
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--advice", required=True)
    p.add_argument("--role", default="all")
    p.add_argument("--cores", type=int, default=5)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--readers", type=int, default=1)
    p.add_argument("--lofi", action="store_true")
    p.add_argument("--weight4", action="store_true")
    p.add_argument("--prefill-only", action="store_true")
    p.add_argument("--validate", action="store_true", help="Run the boundary harness with this candidate")
    p.add_argument("--stack", action="store_true")
    p.add_argument("--long-context", action="store_true", help="Run full-context qualification with this candidate")
    a, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    NativeExpertParallel = target.ParallelMoE
    phase = {"decode": False}
    if a.advice in ("hc_decode_bf16", "qsa_decode_bfp4", "qsa_hybrid_quantized"):
        original_chunk = target.MultichipDecoder._chunk

        def chunk_with_phase(self, *args, **kwargs):
            phase["decode"] = kwargs["decode"] and self.batch_size == 1
            return original_chunk(self, *args, **kwargs)

        target.MultichipDecoder._chunk = chunk_with_phase
    if a.advice == "column_ag_l1":
        original_gather = target.MeshOps.gather

        def column_gather_l1(self, value, dim=-1):
            if tuple(value.shape) != (1, 1, 1, 1536) or dim % 4 != 3 or value.dtype != ttnn.bfloat8_b:
                return original_gather(self, value, dim)
            key = (tuple(value.shape), 3, value.dtype)
            if key not in self.gather_outputs:
                self.gather_outputs[key] = ttnn.empty(
                    (1, 1, 1, 6144),
                    dtype=value.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            return ttnn.experimental.all_gather_async(
                value,
                dim=3,
                mesh_device=self.mesh,
                cluster_axis=1,
                persistent_output_tensor=self.gather_outputs[key],
                num_links=self.policy.links,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        target.MeshOps.gather = column_gather_l1
    if a.advice == "ccl16":
        original_reduce = target.MeshOps.scatter_reduce

        def bf16_reduce(self, x):
            return original_reduce(self, ttnn.typecast(x, ttnn.bfloat16) if x.dtype == ttnn.bfloat8_b else x)

        target.MeshOps.scatter_reduce = bf16_reduce
    if a.advice in ("persistent", "ccl8", "persistent_ccl8"):
        gather = target.MeshOps.gather
        reduce = target.MeshOps.scatter_reduce

        def altered_gather(self, x, dim=-1):
            dtype = x.dtype
            if "ccl8" in a.advice and dtype == ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat8_b)
            if "persistent" in a.advice:
                dim = dim % len(x.shape)
                key = (tuple(x.shape), dim, x.dtype)
                if not hasattr(self, "gather_outputs"):
                    self.gather_outputs = {}
                if key not in self.gather_outputs:
                    shape = list(x.shape)
                    shape[dim] *= 4
                    self.gather_outputs[key] = ttnn.empty(
                        shape, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh
                    )
                y = ttnn.experimental.all_gather_async(
                    x,
                    dim=dim,
                    mesh_device=self.mesh,
                    cluster_axis=1,
                    num_links=self.policy.links,
                    topology=ttnn.Topology.Linear,
                    multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
                    barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    persistent_output_tensor=self.gather_outputs[key],
                )
            else:
                y = gather(self, x, dim)
            return ttnn.typecast(y, dtype) if y.dtype != dtype else y

        def altered_reduce(self, x):
            dtype = x.dtype
            if "ccl8" in a.advice:
                x = ttnn.typecast(x, ttnn.bfloat8_b)
            if "persistent" in a.advice:
                shape = tuple(x.shape)
                flat = ttnn.reshape(x, (1, 1, math.prod(shape[:-1]), shape[-1]))
                key = (tuple(flat.shape), flat.dtype)
                if not hasattr(self, "reduce_outputs"):
                    self.reduce_outputs = {}
                if key not in self.reduce_outputs:
                    outshape = list(flat.shape)
                    outshape[-1] //= 4
                    self.reduce_outputs[key] = [
                        ttnn.empty(
                            (2, *tuple(flat.shape)[1:]), dtype=flat.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh
                        ),
                        ttnn.empty(outshape, dtype=flat.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh),
                    ]
                y = ttnn.experimental.reduce_scatter_minimal_async(
                    flat,
                    dim=3,
                    persistent_output_buffers=self.reduce_outputs[key],
                    multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
                    barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                    num_links=self.policy.links,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                y = ttnn.reshape(y, (*shape[:-1], shape[-1] // 4))
            else:
                y = reduce(self, x)
            return ttnn.typecast(y, dtype) if y.dtype != dtype else y

        target.MeshOps.gather = altered_gather
        target.MeshOps.scatter_reduce = altered_reduce
    factory = target.MultichipDecoder.from_state_dict.__func__
    linear = ttnn.linear
    settings = {}
    records = []

    def configure(cls, *args, **kwargs):
        obj = factory(cls, *args, **kwargs)
        ops = obj.ops
        mesh = ops.mesh
        selected = []
        if a.advice in ("qsa_bf8_control", "qsa_bf8_to_bf4", "qsa_hybrid_quantized"):
            # Reconstruct from checkpoint rather than upcasting already-quantized BF4.
            from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_components import CHECKPOINT
            from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
            from types import MethodType

            if not isinstance(obj.attention, target.ParallelQSA):
                return obj
            assert obj.batch_size == 1
            state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{kwargs['layer_idx']}.self_attn.")
            qg = state["q_proj.weight"].reshape(24, 512, 2560)
            projections = []
            for rank in range(4):
                group = rank // 2
                projections.append(
                    torch.cat(
                        (
                            qg[rank * 6 : (rank + 1) * 6, :256].reshape(1536, 2560),
                            state["k_proj.weight"][group * 256 : (group + 1) * 256],
                            state["v_proj.weight"][group * 256 : (group + 1) * 256],
                            qg[rank * 6 : (rank + 1) * 6, 256:].reshape(1536, 2560),
                            state["indexer.index_qk_proj.weight"],
                        ),
                        0,
                    ).T
                )
            packed = ops.shards(projections, ttnn.bfloat8_b)
            obj.attention.packed_projection = (
                ttnn.typecast(packed, ttnn.bfloat4_b) if a.advice == "qsa_bf8_to_bf4" else packed
            )

            if a.advice == "qsa_hybrid_quantized":
                obj.attention.decode_projection = ttnn.typecast(packed, ttnn.bfloat4_b)

            def control_linear(owner, x, weight):
                compute = owner.projection_compute
                if weight is owner.packed_projection:
                    if a.advice == "qsa_hybrid_quantized" and phase["decode"]:
                        weight = owner.decode_projection
                    elif a.advice in ("qsa_bf8_control", "qsa_hybrid_quantized") and x.shape[-2] > 1:
                        compute = owner.prefill_projection_compute
                return owner.ops.linear(x, weight, compute_kernel_config=compute)

            obj.attention._linear = MethodType(control_linear, obj.attention)
            records.append(
                dict(
                    role="attn_packed_projection",
                    advice=a.advice,
                    selected_dtype=str(obj.attention.packed_projection.dtype),
                    quantization=(
                        "checkpoint→BF8 prefill + checkpoint→BF8→BF4 decode"
                        if a.advice == "qsa_hybrid_quantized"
                        else ("checkpoint→BF8" if a.advice == "qsa_bf8_control" else "checkpoint→BF8→BF4")
                    ),
                )
            )
            return obj
        if a.advice in ("persistent", "ccl8", "persistent_ccl8", "ccl16", "column_ag_l1"):
            return obj
        if a.role in ("all", "hc"):
            selected += [
                (hc, name, "hc_" + name) for hc in (obj.attn_hyper, obj.mlp_hyper) for name in ("read_weight", "up")
            ]
        if a.role in ("all", "ple") and obj.ple is not None:
            selected.append((obj.ple, "projection", "ple_projection"))
        if a.role in ("all", "attn", "attn_in", "attn_out"):
            selected += [
                (obj.attention, name, "attn_" + name)
                for name in ("packed_weight", "out_weight")
                if hasattr(obj.attention, name)
            ]
            selected += [
                (obj.attention, name, "attn_" + name)
                for name in ("packed_projection", "decode_projection", "wo")
                if hasattr(obj.attention, name)
            ]
        if a.role in ("attn_in", "attn_out"):
            selected = [item for item in selected if (item[1] in ("out_weight", "wo")) == (a.role == "attn_out")]
        if a.role == "column_wo":
            selected.append((obj.attention.output_projection, "column_weight", "attention_column_wo"))
        if a.role in ("all", "shared", "router", "shared_in", "shared_out"):
            names = {
                "all": ("router", "shared_gu", "shared_down"),
                "router": ("router",),
                "shared": ("shared_gu", "shared_down"),
                "shared_in": ("shared_gu",),
                "shared_out": ("shared_down",),
            }[a.role]
            selected += [(obj.mlp, name, "shared_" + name) for name in names]
        if a.advice == "expert_nd":
            banks = mesh.dram_grid_size().x
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))])
            for name, width in [("gates", 64), ("ups", 64), ("downs", 256)]:
                mem = ttnn.MemoryConfig(ttnn.BufferType.DRAM, ttnn.NdShardSpec(ttnn.Shape([32, width]), grid))
                setattr(obj.mlp, name, [ttnn.to_memory_config(w, mem) for w in getattr(obj.mlp, name)])
            return obj
        for owner, name, role in selected:
            w = getattr(owner, name)
            k, n = tuple(w.shape)[-2:]
            data = dict(role=role, k=k, n=n, original_dtype=str(w.dtype), advice=a.advice)
            if a.weight4:
                w = ttnn.typecast(w, ttnn.bfloat4_b)
                setattr(owner, name, w)
                data["selected_dtype"] = str(w.dtype)
            compute = None
            if a.advice == "qsa_decode_bfp4":
                settings[w.buffer_address()] = dict(decode_weight=ttnn.typecast(w, ttnn.bfloat4_b))
                data["decode_weight_dtype"] = "bfloat4_b"
            elif a.advice in ("hc_bf16", "hc_decode_bf16"):
                settings[w.buffer_address()] = dict(output_bf16=name == "read_weight")
            elif a.advice in ("bfp4", "lofi", "hifi2", "hifi4"):
                compute = ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity={"hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}.get(a.advice, ttnn.MathFidelity.LoFi),
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
                if a.advice == "bfp4":
                    w = ttnn.typecast(w, ttnn.bfloat4_b)
                    setattr(owner, name, w)
                settings[w.buffer_address()] = dict(compute=compute)
            elif a.advice == "l1":
                settings[w.buffer_address()] = dict(l1=True)
            elif a.advice == "activation8":
                settings[w.buffer_address()] = dict(input8=True)
            elif a.advice == "dram":
                count = min(a.cores, k // 32)
                while k // 32 % count:
                    count -= 1
                block = min(a.block, k // 32 // count)
                while (k // 32 // count) % block:
                    block -= 1
                banks = mesh.dram_grid_size().x
                sn = math.ceil(n / (32 * banks * a.readers)) * 32 * a.readers
                dg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))])
                mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dg, [k, sn], ttnn.ShardOrientation.ROW_MAJOR),
                )
                values = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(w)]
                wd = ttnn.from_torch(
                    torch.cat(values, 0),
                    dtype=w.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=mem,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                )
                coords = [ttnn.CoreCoord(i % min(8, count), i // min(8, count)) for i in range(count)]
                grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
                imem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, [32, k // count], ttnn.ShardOrientation.ROW_MAJOR),
                )
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=sn // 32 // a.readers,
                    num_workers_per_dram_bank=a.readers,
                )
                settings[w.buffer_address()] = dict(dram=(wd, imem, pc))
                data.update(cores=count, block=block, readers=a.readers, shard=[k, sn])
            elif a.advice == "geometry":
                # Whole K width and two/four-column output subblocks; N padding
                # is internal to the matmul writer.
                cores = min(a.cores, math.ceil(n / 32))
                pn = math.ceil(n / 32 / cores)
                sub = min(4, pn)
                while pn % sub:
                    sub -= 1
                block = min(a.block, k // 32)
                while (k // 32) % block:
                    block -= 1
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(min(10, cores), math.ceil(cores / 10)),
                    in0_block_w=block,
                    out_subblock_h=1,
                    out_subblock_w=sub,
                    per_core_M=1,
                    per_core_N=pn,
                    fuse_batch=True,
                    mcast_in0=True,
                )
                settings[w.buffer_address()] = dict(program=pc, l1=True)
                data.update(cores=cores, block=block, subblock=sub)
            else:
                raise ValueError(a.advice)
            if a.lofi:
                settings[w.buffer_address()]["compute"] = ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
                data["fidelity"] = "LoFi"
            records.append(data)
        return obj

    def tuned(x, w, **kw):
        setting = settings.get(w.buffer_address())
        if not setting:
            return linear(x, w, **kw)
        if "decode_weight" in setting and phase["decode"]:
            w = setting["decode_weight"]
        if setting.get("output_bf16") and (a.advice == "hc_bf16" or phase["decode"]):
            kw["dtype"] = ttnn.bfloat16
        if setting.get("compute") is not None and (not a.prefill_only or x.shape[-2] > 1):
            kw["compute_kernel_config"] = setting["compute"]
        if setting.get("input8"):
            x = ttnn.typecast(x, ttnn.bfloat8_b)
            kw["dtype"] = kw.get("dtype") or ttnn.bfloat16
        decode = x.shape[-2] == 1 and math.prod(tuple(x.shape)[:-2]) == 1
        if decode and "dram" in setting:
            wd, imem, pc = setting["dram"]
            shape = tuple(x.shape)
            y = linear(
                ttnn.to_memory_config(ttnn.reshape(x, (1, 1, 1, shape[-1])), imem),
                wd,
                program_config=pc,
                compute_kernel_config=kw.get("compute_kernel_config"),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=kw.get("dtype") or ttnn.bfloat16,
            )
            return ttnn.reshape(ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG), (*shape[:-1], w.shape[-1]))
        if decode and setting.get("l1"):
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            kw["memory_config"] = ttnn.L1_MEMORY_CONFIG
        if decode and "program" in setting:
            kw["program_config"] = setting["program"]
        return linear(x, w, **kw)

    target.MultichipDecoder.from_state_dict = classmethod(configure)
    ttnn.linear = tuned
    bench.CANDIDATE = dict(
        advice=vars(a),
        roles=records,
        runner=str(Path(__file__)),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if a.validate or a.long_context or a.stack:
        import json

        if a.stack:
            from models.autoports.qwen_qwen3_8_flash_next.tests import multichip_stack_contract as validation
        elif a.long_context:
            from models.autoports.qwen_qwen3_8_flash_next.tests import multichip_long_context as validation
        else:
            from models.autoports.qwen_qwen3_8_flash_next.tests import validate_multichip_decoder as validation

        tag = rest[rest.index("--tag") + 1]
        directory = Path(__file__).resolve().parents[1] / "doc/multichip_decoder" / tag
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "advice_runner.py.txt").write_text(Path(__file__).read_text())
        try:
            validation.main()
        finally:
            (directory / "advice.json").write_text(json.dumps(bench.CANDIDATE, indent=2) + "\n")
    else:
        bench.main()


if __name__ == "__main__":
    main()
