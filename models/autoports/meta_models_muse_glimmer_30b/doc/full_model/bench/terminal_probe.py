# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shape/contract/geometry probe for the three full-model-only pieces.

Runs on the real 1x4 mesh at the real widths, and answers the questions the
wrapper has to get right before 52 real layers are worth loading:

* ``ttnn.embedding`` over a **hidden-fractured** table: what input shape it takes,
  what it returns, and whether the fractured lookup plus one ``all_gather`` equals
  the replicated lookup;
* the terminal RMSNorm on the decoder's width-sharded L1 boundary layout;
* the column-parallel LM head, swept over both candidate matmul contracts and
  their legal geometries at the real 32-row decode payload.

The LM head sweep is the point of the file.  Two contracts, and the vocab padding
each one forces:

* **DRAM width-sharded weight + DRAM-sharded matmul** -- the shape every decode
  projection in the layer already uses.  ``dram_sharded_weight_memcfg`` pads the
  per-device width to ``32 * dram_banks = 256`` so there is one shard per bank,
  which the op requires of ``input_tensor_b``, so the vocab has to be padded to
  ``4 * 50688 = 202752``.  The op additionally requires ``K_tiles % cores == 0``,
  and K is ``6656/32 = 208`` tiles, so the legal core counts are the divisors of
  208 that fit an 11x10 grid: 8, 13, 16, 26, 52, 104;
* **DRAM-interleaved weight + 1D-mcast matmul** -- only needs tile alignment, so
  the minimum legal padding is ``4 * 50528 = 202112``.

Usage::

    python doc/full_model/bench/terminal_probe.py [--skip-embedding] [--dtype bfloat8_b,bfloat4_b]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (  # noqa: E402
    dram_sharded_weight_memcfg,
    width_sharded_l1,
)

HIDDEN = 6656
VOCAB = 202048
TP = 4
TILE = 32
K_TILES = HIDDEN // TILE  # 208
DRAM_SHARDED_WIDTH = 202752  # 4 * 50688; 50688 = 256 * 198
MCAST_WIDTH = 202112  # 4 * 50528; 50528 = 32 * 1579
DTYPES = {"bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b, "bfloat16": ttnn.bfloat16}


def say(*args) -> None:
    print(*args, flush=True)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def probe_embedding(mesh) -> None:
    table = (torch.randn(1024, HIDDEN, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    w_frac = ttnn.from_torch(
        table,
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    for label, ids in (
        ("decode b=1", torch.tensor([[7]], dtype=torch.int32)),
        ("decode b=32", torch.arange(32, dtype=torch.int32).reshape(1, 32)),
        ("prefill s=100", torch.arange(100, dtype=torch.int32).reshape(1, 100)),
    ):
        tt_ids = ttnn.from_torch(
            ids,
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        emb = ttnn.embedding(tt_ids, w_frac, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        emb4 = ttnn.unsqueeze_to_4D(emb)
        # No ``topology=`` here, matching ``MultichipDecoder._all_reduce``'s
        # ``rs_ag`` arm: the composite wrapper picks its own, and the args this op
        # takes are deprecated for removal.
        gathered = ttnn.all_gather(emb4, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        got = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).reshape(-1, HIDDEN)
        want = table[ids.flatten().tolist()].to(torch.float32)
        say(
            f"PROBE embedding[{label}] in={tuple(ids.shape)} out={tuple(emb.shape)} "
            f"gathered={tuple(gathered.shape)} pcc={pcc(got, want):.9f}"
        )
        ttnn.deallocate(gathered)
        ttnn.deallocate(emb4)
        ttnn.deallocate(tt_ids)
    ttnn.deallocate(w_frac)


def probe_final_norm(mesh, grid) -> None:
    cores = 16
    memcfg = width_sharded_l1(TILE, HIDDEN, cores, grid)
    x = torch.randn(1, 1, TILE, HIDDEN, dtype=torch.float32)
    gamma = torch.randn(HIDDEN, dtype=torch.float32) * 0.1 + 1.0
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    tt_x_sharded = ttnn.interleaved_to_sharded(tt_x, memcfg)
    tt_gamma_rm = ttnn.from_torch(
        gamma.to(torch.bfloat16).reshape(1, 1, HIDDEN // TILE, TILE),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    prg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[min(cores, grid.x), (cores + grid.x - 1) // grid.x],
        subblock_w=1,
        block_h=1,
        block_w=HIDDEN // cores // TILE,
        inplace=False,
    )
    normed = ttnn.rms_norm(tt_x_sharded, weight=tt_gamma_rm, epsilon=1e-5, program_config=prg, memory_config=memcfg)
    got = ttnn.to_torch(ttnn.get_device_tensors(ttnn.sharded_to_interleaved(normed, ttnn.DRAM_MEMORY_CONFIG))[0])
    want = (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)) * gamma
    say(f"PROBE final_norm sharded pcc={pcc(got, want):.9f}")
    ttnn.deallocate(tt_x)


def probe_lm_head(mesh, grid, dtype_names) -> None:
    torch.manual_seed(1)
    head = (torch.randn(HIDDEN, VOCAB, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    act = torch.randn(1, 1, TILE, HIDDEN, dtype=torch.float32) * 0.5
    want = (act.to(torch.float32) @ head.to(torch.float32)).reshape(TILE, VOCAB)
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def padded_head(width: int) -> torch.Tensor:
        out = torch.zeros(1, 1, HIDDEN, width, dtype=torch.bfloat16)
        out[0, 0, :, :VOCAB] = head
        return out

    def act_tensor(memcfg=None):
        return ttnn.from_torch(
            act.to(torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=memcfg or ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def read(logits, width):
        shards = [ttnn.to_torch(t).reshape(TILE, -1)[:, : width // TP] for t in ttnn.get_device_tensors(logits)]
        return torch.cat(shards, dim=-1)[:, :VOCAB]

    def timed(fn, rounds: int = 40) -> float:
        out = fn()
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(out)
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(rounds):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh)
            best = min(best, (time.perf_counter() - start) / rounds * 1e3)
        return best

    for name in dtype_names:
        dtype = DTYPES[name]

        # ------------------- (a) DRAM width-sharded weight, DRAM-sharded matmul
        started = time.perf_counter()
        per_dev = DRAM_SHARDED_WIDTH // TP
        w = ttnn.from_torch(
            padded_head(DRAM_SHARDED_WIDTH),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=dram_sharded_weight_memcfg(HIDDEN, per_dev, mesh),
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        say(f"PROBE lm_head weight[{name} dram_sharded n={per_dev}] uploaded in {time.perf_counter()-started:.1f}s")
        for cores in (8, 13, 16, 26, 52, 104):
            if K_TILES % cores:
                continue
            in0_max = K_TILES // cores
            for in0_block_w in sorted({d for d in range(1, in0_max + 1) if in0_max % d == 0}):
                tt_act = act_tensor(width_sharded_l1(TILE, HIDDEN, cores, grid))
                prg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=in0_block_w,
                    per_core_M=1,
                    per_core_N=(per_dev + TILE * cores - 1) // (TILE * cores),
                )
                out_memcfg = width_sharded_l1(TILE, per_dev, cores, grid)

                def run(tt_act=tt_act, prg=prg, out_memcfg=out_memcfg):
                    return ttnn.linear(
                        tt_act,
                        w,
                        dtype=ttnn.bfloat16,
                        memory_config=out_memcfg,
                        program_config=prg,
                        compute_kernel_config=ck,
                    )

                try:
                    logits = run()
                    p = pcc(
                        read(ttnn.sharded_to_interleaved(logits, ttnn.DRAM_MEMORY_CONFIG), DRAM_SHARDED_WIDTH), want
                    )
                    ttnn.deallocate(logits)
                    say(
                        f"PROBE lm_head dram_sharded {name} cores={cores} in0={in0_block_w} "
                        f"ms={timed(run):.4f} pcc={p:.6f}"
                    )
                except Exception as exc:  # noqa: BLE001
                    say(
                        f"PROBE lm_head dram_sharded {name} cores={cores} in0={in0_block_w} "
                        f"FAIL {str(exc).splitlines()[-1][:120]}"
                    )
                ttnn.deallocate(tt_act)
        ttnn.deallocate(w)

        # ------------------------- (b) interleaved weight, 1D-mcast matmul
        started = time.perf_counter()
        per_dev = MCAST_WIDTH // TP
        w2 = ttnn.from_torch(
            padded_head(MCAST_WIDTH),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        say(f"PROBE lm_head weight[{name} interleaved n={per_dev}] uploaded in {time.perf_counter()-started:.1f}s")
        n_tiles = per_dev // TILE
        num_cores = grid.x * grid.y
        tt_act = act_tensor()
        for in0_block_w in (1, 2, 4, 8, 13, 16, 26):
            if K_TILES % in0_block_w:
                continue
            per_core_n = (n_tiles + num_cores - 1) // num_cores
            out_subblock_w = min(per_core_n, 4)
            while out_subblock_w > 1 and per_core_n % out_subblock_w:
                out_subblock_w -= 1
            prg2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

            def run2(prg2=prg2):
                return ttnn.linear(
                    tt_act,
                    w2,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=prg2,
                    compute_kernel_config=ck,
                )

            try:
                logits = run2()
                p = pcc(read(logits, MCAST_WIDTH), want)
                ttnn.deallocate(logits)
                say(
                    f"PROBE lm_head mcast1d {name} in0={in0_block_w} per_core_N={per_core_n} "
                    f"ms={timed(run2):.4f} pcc={p:.6f}"
                )
            except Exception as exc:  # noqa: BLE001
                say(f"PROBE lm_head mcast1d {name} in0={in0_block_w} FAIL {str(exc).splitlines()[-1][:120]}")

        def run3():
            return ttnn.linear(tt_act, w2, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        try:
            logits = run3()
            p = pcc(read(logits, MCAST_WIDTH), want)
            ttnn.deallocate(logits)
            say(f"PROBE lm_head op_default {name} ms={timed(run3):.4f} pcc={p:.6f}")
        except Exception as exc:  # noqa: BLE001
            say(f"PROBE lm_head op_default {name} FAIL {str(exc).splitlines()[-1][:120]}")
        ttnn.deallocate(tt_act)
        ttnn.deallocate(w2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument("--dtype", default="bfloat8_b")
    args = parser.parse_args()

    torch.manual_seed(0)
    mesh = open_multichip_mesh(trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        say(f"PROBE grid={grid.x}x{grid.y} dram_banks={mesh.dram_grid_size().x} devices={mesh.get_num_devices()}")
        if not args.skip_embedding:
            probe_embedding(mesh)
            probe_final_norm(mesh, grid)
        if not args.skip_lm_head:
            probe_lm_head(mesh, grid, [n.strip() for n in args.dtype.split(",") if n.strip()])
        say("PROBE_OK")
        return 0
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
