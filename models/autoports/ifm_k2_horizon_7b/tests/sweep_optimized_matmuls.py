"""Precision-locked per-role geometry sweep on recorded real layer activations."""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from tracy import signpost

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder
from .run_functional import load_reference, pcc, to_device
from .sweep_optimized import policy_for, real_activations


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--roles", nargs="+", default=["qkv", "o", "gate", "up", "gateup", "down"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--shortlist", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--dtype", default="bfloat4_b")
    parser.add_argument("--fidelity", default="LoFi")
    parser.add_argument("--cores", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--readers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--multichip", action="store_true")
    parser.add_argument("--blocks", nargs="+", type=int)
    parser.add_argument("--bf16-dest", action="store_true")
    parser.add_argument("--fp32-only", action="store_true")
    parser.add_argument("--full-model-policy", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(16)
    config, state, hf, rope_fn = load_reference()
    acts = real_activations(4097)[None]
    from ..tt.multichip_decoder import MultichipDecoder
    from .run_multichip import read, upload

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if args.multichip else ttnn.FabricConfig.DISABLED)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4) if args.multichip else ttnn.MeshShape(1, 1),
        **({} if args.multichip else {"physical_device_ids": [0]}),
        trace_region_size=0,
    )

    def device(x, integer=False):
        return upload(x, mesh, integer=integer) if args.multichip else to_device(x, mesh, integer)

    def host(x):
        return read(x, mesh, -1 if args.multichip else None)

    rows = []
    try:
        layer_options = {} if args.multichip else {"policy": policy_for("dram_16_8_2_split_b4")}
        if args.full_model_policy:
            assert args.multichip and args.fp32_only
            from ..tt.full_model_policy import stage6_precision_policy

            layer_options = {"policy": stage6_precision_policy(0 if args.dtype == "bfloat8_b" else 1)}
        layer = (MultichipDecoder if args.multichip else OptimizedDecoder).from_state_dict(
            state,
            hf_config=config,
            layer_idx=0,
            mesh_device=mesh,
            **layer_options,
        )
        caches = tuple(
            ttnn.zeros(
                (132, 2 if args.multichip else 8, 32, 128), dtype=layer.kv_dtype, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            for _ in range(2)
        )
        table = device(torch.arange(132, dtype=torch.int32)[None], True)
        r = rope_fn(acts[:, :4096], torch.arange(4096)[None])
        out = layer.prefill_forward(
            upload(acts[:, :4096][None], mesh, shard=-1) if args.multichip else device(acts[:, :4096][None]),
            rope=tuple(device(t[:, None]) for t in r),
            kv_cache=caches,
            page_table=table,
            plan=layer.prepare_prefill(seq_len=4096),
        )
        out.deallocate(True)
        inputs = {}
        original = layer._decode_linear
        names = {id(getattr(layer, "w" + role)): role for role in ["qkv", "o", "gate", "up", "gateup", "down"]}

        def record(x, w):
            inputs[names[id(w)]] = host(x)
            return original(x, w)

        layer._decode_linear = record
        r = rope_fn(acts[:, -1:], torch.tensor([[4096]]))
        out = layer.decode_forward(
            upload(acts[:, -1:][None], mesh, shard=-1) if args.multichip else device(acts[:, -1:][None]),
            rope=tuple(device(t[None].repeat(1, 1, 32, 1)) for t in r),
            kv_cache=caches,
            page_table=table,
            current_pos=device(torch.tensor([4096], dtype=torch.int32), True),
        )
        out.deallocate(True)
        inputs["gateup"] = inputs["gate"]
        gamma1 = state["model.layers.0.input_layernorm.weight"].bfloat16().float()
        gamma2 = state["model.layers.0.post_attention_layernorm.weight"].bfloat16().float()

        def fold(name, gamma):
            return (state["model.layers.0." + name].bfloat16().float().T * gamma[:, None]).bfloat16()

        weights = {
            "qkv": torch.cat([fold("self_attn." + p + "_proj.weight", gamma1) for p in "qkv"], -1),
            "o": state["model.layers.0.self_attn.o_proj.weight"].T.bfloat16(),
            "gate": fold("mlp.gate_proj.weight", gamma2),
            "up": fold("mlp.up_proj.weight", gamma2),
            "down": state["model.layers.0.mlp.down_proj.weight"].T.bfloat16(),
        }
        weights["gateup"] = torch.cat([weights["gate"], weights["up"]], -1)
        if args.multichip:
            weights["qkv"] = torch.stack(
                [
                    torch.cat([fold("self_attn." + p + "_proj.weight", gamma1).chunk(4, -1)[rank] for p in "qkv"], -1)
                    for rank in range(4)
                ]
            )
            for role in ("gate", "up"):
                weights[role] = torch.stack(weights[role].chunk(4, -1))
            for role in ("o", "down"):
                weights[role] = torch.stack(weights[role].chunk(4, -2))
            weights["gateup"] = torch.cat([weights["gate"], weights["up"]], -1)
        # Drop setup layer and its L1 tensors before testing each independent geometry.
        layer._decode_linear = None
        original = None
        layer = None
        caches = None
        gc.collect()
        ttnn.synchronize_device(mesh)
        banks = mesh.dram_grid_size()
        for role in list(reversed(args.roles)) if args.reverse else args.roles:
            weight = weights[role]
            k, n = weight.shape[-2:]
            reference = (
                torch.cat([a.float() @ w.float() for a, w in zip(inputs[role].chunk(4, -1), weight)], -1)
                if args.multichip
                else inputs[role].float() @ weight.float()
            )
            selected = {
                "qkv": (16, 32),
                "o": (32, 4),
                "gate": (32, 8),
                "up": (32, 8),
                "gateup": (32, 8),
                "down": (32, 12),
            }
            role_cores = [selected[role][0]] if args.multichip and args.shortlist else args.cores
            for cores in role_cores:
                if k % (32 * cores):
                    rows.append(
                        dict(
                            role=role,
                            cores=cores,
                            passed=False,
                            error="K must divide into whole tiles on each activation storage core",
                        )
                    )
                    continue
                shard_tiles = k // 32 // cores
                blocks = [4, 8, 16, 32] if k == 4096 else [3, 6, 8, 12, 16, 24, 32, 48]
                if args.shortlist:
                    blocks = [
                        (
                            selected[role][1]
                            if args.multichip
                            else 8
                            if role in ("qkv", "o")
                            else 12
                            if role == "down"
                            else 4
                        )
                    ]
                if args.blocks:
                    blocks = args.blocks
                for block in blocks:
                    if k // 32 % block or (shard_tiles % block and block % shard_tiles):
                        continue
                    for readers in list(reversed(args.readers)) if args.reverse else args.readers:
                        # Explicit inert padding, including the logical shape, fixes writer partitions.
                        nphys = math.ceil(n / math.lcm(32 * cores, 32 * banks.x * readers)) * math.lcm(
                            32 * cores, 32 * banks.x * readers
                        )
                        for fp32 in (
                            [True] if args.fp32_only else [False] if args.shortlist or args.bf16_dest else [True, False]
                        ):
                            row = dict(
                                role=role,
                                k=k,
                                n=n,
                                nphysical=nphys,
                                dtype=args.dtype,
                                fidelity=args.fidelity,
                                cores=cores,
                                input_shard_tiles=shard_tiles,
                                block_w=block,
                                readers=readers,
                                fp32=fp32,
                                per_core_N=nphys // 32 // cores,
                                reader_row_bytes=nphys
                                // 32
                                // banks.x
                                // readers
                                * (576 if args.dtype == "bfloat4_b" else 1088),
                            )
                            name = f"{role}_c{cores}_k{block}_r{readers}_f{int(fp32)}"
                            trace = None
                            tensors = []
                            try:
                                bg = ttnn.CoreRangeSet(
                                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks.x - 1, banks.y - 1))}
                                )
                                wm = ttnn.MemoryConfig(
                                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                                    ttnn.BufferType.DRAM,
                                    ttnn.ShardSpec(bg, (k, nphys // banks.x), ttnn.ShardOrientation.ROW_MAJOR),
                                )
                                padded_weight = torch.nn.functional.pad(weight, (0, nphys - n))
                                wt = ttnn.from_torch(
                                    (
                                        torch.cat(list(padded_weight), -1) if args.multichip else padded_weight
                                    ).contiguous(),
                                    device=mesh,
                                    dtype=getattr(ttnn, args.dtype),
                                    layout=ttnn.TILE_LAYOUT,
                                    memory_config=wm,
                                    **({"mesh_mapper": ttnn.ShardTensorToMesh(mesh, dim=-1)} if args.multichip else {}),
                                )
                                tensors.append(wt)
                                grid = ttnn.num_cores_to_corerangeset(
                                    cores, mesh.compute_with_storage_grid_size(), row_wise=True
                                )
                                im = ttnn.create_sharded_memory_config(
                                    (32, k // cores),
                                    core_grid=grid,
                                    strategy=ttnn.ShardStrategy.WIDTH,
                                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                    use_height_and_width_as_shard_shape=True,
                                )
                                at = ttnn.from_torch(
                                    inputs[role].contiguous(),
                                    device=mesh,
                                    dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT,
                                    memory_config=im,
                                    **({"mesh_mapper": ttnn.ShardTensorToMesh(mesh, dim=-1)} if args.multichip else {}),
                                )
                                tensors.append(at)
                                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                                    in0_block_w=block,
                                    per_core_M=1,
                                    per_core_N=nphys // 32 // cores,
                                    num_workers_per_dram_bank=readers,
                                )
                                ck = ttnn.init_device_compute_kernel_config(
                                    mesh.arch(),
                                    math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
                                    math_approx_mode=False,
                                    fp32_dest_acc_en=fp32,
                                    packer_l1_acc=True,
                                )

                                def forward():
                                    return ttnn.linear(
                                        at,
                                        wt,
                                        dtype=ttnn.bfloat16,
                                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                        program_config=pc,
                                        compute_kernel_config=ck,
                                    )

                                y = forward()
                                result = host(y)
                                unpadded = (
                                    torch.cat([v[..., :n] for v in result.chunk(4, -1)], -1)
                                    if args.multichip
                                    else result[..., :n]
                                )
                                row["pcc"] = pcc(unpadded, reference)
                                y.deallocate(True)
                                ttnn.synchronize_device(mesh)
                                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                                y = forward()
                                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                                tensors.append(y)
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                                first = host(y)
                                if args.profile:
                                    signpost(name)
                                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                                    signpost(name + "_END")
                                else:
                                    times = []
                                    for _ in range(3):
                                        start = time.perf_counter()
                                        for _ in range(30):
                                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                                        ttnn.synchronize_device(mesh)
                                        times.append((time.perf_counter() - start) * 1e6 / 30)
                                    row["traced_host_us"] = statistics.median(times)
                                    row["samples_us"] = times
                                row["deterministic"] = torch.equal(first, host(y))
                                row["passed"] = row["deterministic"]
                            except Exception as exc:
                                row.update(passed=False, error=str(exc).split("backtrace:")[0])
                            finally:
                                if trace is not None:
                                    ttnn.release_trace(mesh, trace)
                                for t in tensors:
                                    t.deallocate(True)
                            rows.append(row)
                            Path(args.output).write_text(
                                json.dumps(
                                    {
                                        "invocation": sys.argv,
                                        "activation_source": "recorded real-weight layer on checkpoint token embeddings",
                                        "records": rows,
                                    },
                                    indent=2,
                                )
                                + "\n"
                            )
                            print(name, row.get("traced_host_us"), row.get("pcc"), row.get("error", ""), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
