# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Role-specific traced DRAM matmul search on recorded official-HF activations."""

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.prepare_optimized_inputs import load_state
from models.autoports.meta_models_muse_glimmer_30b.tests.run_fused import (
    CONFIG,
    ROOT,
    AutoConfig,
    LayerCache,
    MuseGlimmerTextRotaryEmbedding,
    pcc,
    read,
    reference,
)


def width_mem(k, grid):
    tiles = k // 32
    count = max(c for c in range(1, math.prod(grid) + 1) if tiles % c == 0)
    cores = ttnn.num_cores_to_corerangeset(count, ttnn.CoreCoord(*grid), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [32, k // count], ttnn.ShardOrientation.ROW_MAJOR),
    )


def main(args):
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(CONFIG).text_config
    cfg._attn_implementation = "sdpa"
    state = load_state(args.layer)
    with torch.device("meta"):
        ref = MuseGlimmerTextDecoderLayer(cfg, args.layer)
    ref.load_state_dict(state, assign=True)
    activations = {}
    names = [
        "self_attn.q_proj",
        "self_attn.gate_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    hooks = []
    for name in names:

        def save(module, inputs, name=name):
            activations[name] = inputs[0][:, -1:].detach().clone()

        hooks.append(ref.get_submodule(name).register_forward_pre_hook(save))
    x = torch.load(ROOT / f"doc/optimized_decoder/inputs/layer_{args.layer}.pt", weights_only=True)[:, :1025]
    reference(ref, x, 0, LayerCache(), MuseGlimmerTextRotaryEmbedding(cfg))
    for h in hooks:
        h.remove()
    del ref, x
    activations["qkv"] = activations["self_attn.q_proj"]
    state["qkv.weight"] = torch.cat([state[f"self_attn.{r}_proj.weight"] for r in ("q", "k", "v")])
    activations["packed_attention"] = activations["qkv"]
    state["packed_attention.weight"] = torch.cat([state["qkv.weight"], state["self_attn.gate_proj.weight"]])
    activations["mlp.gate_up"] = activations["mlp.gate_proj"]
    state["mlp.gate_up.weight"] = torch.cat([state["mlp.gate_proj.weight"], state["mlp.up_proj.weight"]])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    dg = mesh.dram_grid_size()
    banks = dg.x * dg.y
    rows = []
    try:
        for role in args.roles:
            wt = state[role + ".weight"].t().contiguous()
            k, n = wt.shape
            act = activations[role].reshape(1, 1, 1, k)
            wanted = torch.nn.functional.linear(act, wt.t())
            for dtype, fidelity in args.precision:
                ck = ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                    math_approx_mode=False,
                    fp32_dest_acc_en=args.fp32,
                    packer_l1_acc=True,
                )
                for order, reader_group in enumerate(args.reader_order or ([1, 2, 3] if args.readers else [1])):
                    # Exact per-bank padding is a tensor contract, zeros are sliced in the measured path.
                    npad = math.ceil(n / (banks * reader_group * 32)) * banks * reader_group * 32
                    padded = torch.nn.functional.pad(wt, (0, npad - n))
                    wm = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(
                            ttnn.CoreRangeSet(
                                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))}
                            ),
                            [k, npad // banks],
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    )
                    w = ttnn.from_torch(
                        padded, device=mesh, dtype=getattr(ttnn, dtype), layout=ttnn.TILE_LAYOUT, memory_config=wm
                    )
                    grids = [tuple(map(int, g.split("x"))) for g in args.grids]
                    for grid in grids:
                        mem = width_mem(k, grid)
                        adtype = ttnn.bfloat8_b if args.selected and role == "mlp.down_proj" else ttnn.bfloat16
                        odtype = ttnn.bfloat8_b if args.selected and role.startswith("mlp") else ttnn.bfloat16
                        a = ttnn.from_torch(act, device=mesh, dtype=adtype, layout=ttnn.TILE_LAYOUT, memory_config=mem)
                        shard = mem.shard_spec.shape[1] // 32
                        blocks = args.blocks or [
                            b
                            for b in range(1, min(k // 32, 104) + 1)
                            if k // 32 % b == 0 and (shard % b == 0 or b % shard == 0)
                        ]
                        if args.selected and not args.blocks:
                            blocks = [
                                {"packed_attention": 4, "self_attn.o_proj": 8, "mlp.gate_up": 4, "mlp.down_proj": 6}[
                                    role
                                ]
                            ]
                        for block in blocks:
                            row = dict(
                                role=role,
                                order=order,
                                shape=[1, k, n],
                                physical_n=npad,
                                dtype=dtype,
                                fidelity=fidelity,
                                fp32=args.fp32,
                                activation_dtype=str(adtype),
                                output_dtype=str(odtype),
                                grid=grid,
                                shard_k_tiles=shard,
                                block=block,
                                readers=reader_group,
                                dram_banks=banks,
                                per_core_N=math.ceil(npad / (32 * mem.shard_spec.grid.num_cores())),
                                storage_cores=mem.shard_spec.grid.num_cores(),
                                per_bank_tiles=npad // (32 * banks),
                                per_reader_physical_row_bytes=npad
                                // (32 * banks * reader_group)
                                * (576 if dtype == "bfloat4_b" else 1088),
                            )
                            tid = out = None
                            try:
                                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                                    in0_block_w=block,
                                    per_core_M=1,
                                    per_core_N=row["per_core_N"],
                                    num_workers_per_dram_bank=reader_group,
                                )

                                def forward():
                                    y = ttnn.linear(
                                        a,
                                        w,
                                        dtype=odtype,
                                        compute_kernel_config=ck,
                                        program_config=pc,
                                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                    )
                                    return ttnn.slice(y, [0, 0, 0, 0], [1, 1, 1, n]) if npad != n else y

                                y = forward()
                                score = pcc(wanted, read(y))
                                del y
                                ttnn.synchronize_device(mesh)
                                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                                out = forward()
                                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                                from tracy import signpost

                                tag = f"READER_{role}_{order}_{reader_group}"
                                signpost(tag)
                                samples = []
                                for _ in range(3):
                                    ttnn.synchronize_device(mesh)
                                    t0 = time.perf_counter()
                                    for __ in range(30):
                                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                                    ttnn.synchronize_device(mesh)
                                    samples.append((time.perf_counter() - t0) * 1000 / 30)
                                row.update(
                                    status="pass",
                                    pcc=score,
                                    trace_pcc=pcc(wanted, read(out)),
                                    ms=statistics.median(samples),
                                    samples_ms=samples,
                                )
                                signpost(tag + "_END")
                            except Exception as e:
                                row.update(status="error", error=str(e))
                            finally:
                                if tid is not None:
                                    ttnn.release_trace(mesh, tid)
                                del out
                                gc.collect()
                            rows.append(row)
                            args.output.write_text(
                                json.dumps(
                                    dict(
                                        command=sys.argv,
                                        layer=args.layer,
                                        rows=rows,
                                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    ),
                                    indent=2,
                                )
                            )
                            print(json.dumps({key: value for key, value in row.items() if key != "error"}), flush=True)
                            if row["status"] == "error":
                                print(row["error"].split("backtrace:")[0], flush=True)
                        del a
                    del w
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument(
        "--roles",
        nargs="+",
        default=["qkv", "self_attn.gate_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    )
    p.add_argument("--precision", nargs="+", default=["bfloat4_b:LoFi"])
    p.add_argument("--grids", nargs="+", default=["4x1", "8x1", "8x2", "8x4", "8x8"])
    p.add_argument("--blocks", nargs="+", type=int)
    p.add_argument("--readers", action="store_true")
    p.add_argument("--reader-order", nargs="+", type=int)
    p.add_argument("--selected", action="store_true", help="Final role blocks and BF8 MLP intermediates")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.precision = [item.split(":") for item in args.precision]
    main(args)
