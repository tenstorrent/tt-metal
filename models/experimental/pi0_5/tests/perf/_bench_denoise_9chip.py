# SPDX-License-Identifier: Apache-2.0
"""9-chip pipelined denoise: 2 layers per chip, 18 layers total.
Chips 22-30 (snake order on BH 8×4 galaxy), all weights in L1, d2d sockets.

Topology: chips 22->23->...->30, all adjacent in snake order.
Each chip owns 2 consecutive layers and its own L1-resident weights.
Socket transfers the [1,1,M,HIDDEN] hidden state between chips.

Run:
  tt-smi -glx_reset
  python _bench_denoise_9chip.py --device-start 22
  python -m tracy -p -r -n denoise_9chip _bench_denoise_9chip.py
"""
import argparse
import os
import statistics
import time

# Production env flags before model imports.
_PROD_DEFAULTS = {
    "PI0_EXPERT_MM_LOFI": "1",
    "PI0_ROPE_TABLES_L1": "1",
    "PI0_MM_SWEEP_V2": "1",
    "PI0_DENOISE_MM_TUNE": "1",
    "PI0_UPSTREAM_MASKS": "1",
    "QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT": "1",
    "QWEN_NLP_CREATE_HEADS_HEAD_SPLIT": "1",
    "PI0_MQA_HEAD_SPLIT": "1",
    "PI0_SDPA_DENOISE_K_FORCE": "96",
    "PI0_MD_DENOISE": "1",
}
for _k, _v in _PROD_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.tt_bh_glx.expert_slice import ExpertChunkSlice

N_WARMUP = 3
N_ITER = 10
N_CHIPS = 9
N_PER = 2  # layers per chip


def _snake_order(rows, cols):
    order = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in cs:
            order.append((r, c))
    return order


def _synthetic_weights(config):
    W, M = config.width, config.mlp_dim
    H = config.num_heads * config.head_dim
    KVH = config.num_kv_heads * config.head_dim
    w = {}
    for i in range(config.depth):
        p = f"model.layers.{i}."
        w[f"{p}self_attn.q_proj.weight"] = torch.randn(H, W) * 0.02
        w[f"{p}self_attn.k_proj.weight"] = torch.randn(KVH, W) * 0.02
        w[f"{p}self_attn.v_proj.weight"] = torch.randn(KVH, W) * 0.02
        w[f"{p}self_attn.o_proj.weight"] = torch.randn(W, H) * 0.02
        w[f"{p}mlp.gate_proj.weight"] = torch.randn(M, W) * 0.02
        w[f"{p}mlp.up_proj.weight"] = torch.randn(M, W) * 0.02
        w[f"{p}mlp.down_proj.weight"] = torch.randn(W, M) * 0.02
        w[f"{p}input_layernorm.weight"] = torch.ones(W)
        w[f"{p}post_attention_layernorm.weight"] = torch.ones(W)
        w[f"{p}input_layernorm.dense.weight"] = torch.randn(3 * W, W) * 0.02
        w[f"{p}post_attention_layernorm.dense.weight"] = torch.randn(3 * W, W) * 0.02
    return w


def _move_weights_to_l1(chunk):
    L1 = ttnn.L1_MEMORY_CONFIG
    moved = []

    def move_obj(obj, label, skip=()):
        if obj is None:
            return
        for k, v in vars(obj).items():
            if k in skip:
                continue
            if (
                isinstance(v, ttnn.Tensor)
                and v.storage_type() == ttnn.StorageType.DEVICE
                and v.memory_config().buffer_type != ttnn.BufferType.L1
            ):
                setattr(obj, k, ttnn.to_memory_config(v, L1))
                moved.append(f"{label}.{k}")

    move_obj(chunk, "chunk")
    for i, blk in enumerate(chunk.blocks):
        move_obj(blk, f"blk{i}")
        move_obj(blk.attention, f"blk{i}.attn")
        mlp_skip = ("gate_proj", "up_proj", "down_proj") if getattr(blk.mlp, "_md_denoise", False) else ()
        move_obj(blk.mlp, f"blk{i}.mlp", skip=mlp_skip)
    return moved


def _mk_precomputed_mods(config, weights, layer_range, device):
    """Precompute the 6 per-layer modulation vectors (TIER A) for layers in layer_range."""
    W = config.width
    cond = torch.randn(1, W).to(torch.bfloat16)
    mods = []
    for i in range(*layer_range):
        p = f"model.layers.{i}."
        fw = torch.cat(
            [weights[f"{p}input_layernorm.dense.weight"], weights[f"{p}post_attention_layernorm.dense.weight"]], dim=0
        ).to(torch.bfloat16)
        mod = torch.nn.functional.linear(cond, fw)  # (1, 6W)
        parts = [mod[:, k * W : (k + 1) * W] for k in range(6)]
        parts[0] = parts[0] + 1.0  # sa1
        parts[3] = parts[3] + 1.0  # sf1
        tup = tuple(
            ttnn.from_torch(
                t.unsqueeze(1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for t in parts
        )
        mods.append(tup)
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-start", type=int, default=22, help="first chip in chain (22-30 = 9 chips)")
    ap.add_argument("--m-pad", type=int, default=32)
    ap.add_argument(
        "--prefix", type=int, default=768, help="prefix KV length: 2-cam=768 (2×256 img + 256 lang), 3-cam=1024"
    )
    args = ap.parse_args()

    CHIP_IDS = list(range(args.device_start, args.device_start + N_CHIPS))
    print(f"[9chip] chips={CHIP_IDS}  {N_CHIPS}×{N_PER} layers  M_pad={args.m_pad}  PREFIX={args.prefix}")

    coords = _snake_order(8, 4)
    HIDDEN = None  # set after config load

    config = GemmaConfig.gemma_300m()
    HIDDEN = config.width
    HEAD_D = config.head_dim

    # Socket buffer: one hidden-state tile-array [1,1,M,HIDDEN] bf16 = M*HIDDEN*2 bytes.
    SOCK_BYTES = args.m_pad * HIDDEN * 2

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(8, 4),
        l1_small_size=24576,
        trace_region_size=134_217_728,
    )
    submeshes = []
    try:
        # Carve one 1×1 submesh per chip.
        for chip_id in CHIP_IDS:
            r, c = coords[chip_id]
            sm = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))
            submeshes.append(sm)
        print(f"[9chip] carved {N_CHIPS} submeshes")

        # Per-chip weights, chunk, and precomputed mods.
        weights_list = [_synthetic_weights(config) for _ in range(N_CHIPS)]
        chunks = []
        mods_list = []
        for i, sm in enumerate(submeshes):
            lo = i * N_PER
            hi = lo + N_PER
            chunk = ExpertChunkSlice(
                config,
                weights_list[i],
                sm,
                layer_range=(lo, hi),
                max_seq_len=args.m_pad + args.prefix + 64,
            )
            moved = _move_weights_to_l1(chunk)
            chunks.append(chunk)
            mods_list.append(_mk_precomputed_mods(config, weights_list[i], (lo, hi), sm))
            print(f"  chip {CHIP_IDS[i]}: layers {lo}-{hi-1}, {len(moved)} tensors -> L1")

        # Per-chip prefix KV (synthetic, one (K,V) per local layer).
        _k_dt = ttnn.bfloat8_b
        prefix_kv_list = []
        for i, sm in enumerate(submeshes):
            pkv = [
                (
                    ttnn.from_torch(
                        torch.randn(1, 1, args.prefix, HEAD_D).bfloat16(),
                        dtype=_k_dt,
                        layout=ttnn.TILE_LAYOUT,
                        device=sm,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    ttnn.from_torch(
                        torch.randn(1, 1, args.prefix, HEAD_D).bfloat16(),
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=sm,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                )
                for _ in range(N_PER)
            ]
            prefix_kv_list.append(pkv)

        # Per-chip adarms_cond.
        adarms_conds = [
            ttnn.from_torch(
                torch.randn(1, 1, 1, HIDDEN).bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=sm,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for sm in submeshes
        ]

        # Input hidden on chip 0.
        hidden0 = ttnn.from_torch(
            torch.randn(1, 1, args.m_pad, HIDDEN).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submeshes[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Receive buffers for chips 1-8 (DRAM, pre-allocated, same shape as hidden).
        recv_bufs = [None] + [
            ttnn.from_torch(
                torch.zeros(1, 1, args.m_pad, HIDDEN).bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=submeshes[i],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for i in range(1, N_CHIPS)
        ]

        # Socket pairs: chips[i] -> chips[i+1].
        socks_send = []
        socks_recv = []
        for i in range(N_CHIPS - 1):
            conn = ttnn.SocketConnection(
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
            )
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, SOCK_BYTES)
            s, r = ttnn.create_socket_pair(submeshes[i], submeshes[i + 1], ttnn.SocketConfig([conn], mem))
            socks_send.append(s)
            socks_recv.append(r)
        print(f"[9chip] created {len(socks_send)} socket pairs")

        def run_chip(i):
            """One chip's forward: (optional recv) + 2-layer forward + (optional send)."""
            inp = hidden0 if i == 0 else recv_bufs[i]
            if i > 0:
                ttnn.experimental.recv_direct_async(inp, socks_recv[i - 1])
            out = chunks[i].forward(inp, adarms_conds[i], prefix_kv_list[i], precomputed_mods=mods_list[i])
            # Normalize to DRAM-interleaved TILE before socket send (required by fabric).
            if i < N_CHIPS - 1:
                out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
                out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.experimental.send_direct_async(out, socks_send[i])

        # Warmup (JIT compile all kernels before trace capture).
        print("[9chip] warmup (eager) ...")
        for i in range(N_CHIPS):
            run_chip(i)
        ttnn.synchronize_device(submeshes[-1])
        print("[9chip] warmup OK")

        # Per-submesh trace capture.
        tids = []
        for i in range(N_CHIPS):
            tid = ttnn.begin_trace_capture(submeshes[i], cq_id=0)
            run_chip(i)
            ttnn.end_trace_capture(submeshes[i], tid, cq_id=0)
            tids.append(tid)
        print(f"[9chip] captured {N_CHIPS} per-submesh traces")

        # Warmup replays.
        for _ in range(N_WARMUP):
            for i in range(N_CHIPS):
                ttnn.execute_trace(submeshes[i], tids[i], cq_id=0, blocking=False)
        ttnn.synchronize_device(submeshes[-1])

        # Timed replays: dispatch all 9 chip traces, sync on last chip.
        times = []
        for it in range(N_ITER):
            t0 = time.perf_counter()
            for i in range(N_CHIPS):
                ttnn.execute_trace(submeshes[i], tids[i], cq_id=0, blocking=False)
            ttnn.synchronize_device(submeshes[-1])
            ms = (time.perf_counter() - t0) * 1000.0
            times.append(ms)
            print(f"  iter {it+1:2d}: {ms:.2f} ms")

        for sm in submeshes:
            ttnn.ReadDeviceProfiler(sm)

        for i, tid in enumerate(tids):
            ttnn.release_trace(submeshes[i], tid)

        print("=" * 60)
        avg = statistics.mean(times)
        mn = min(times)
        print(
            f"9-chip pipeline ({N_CHIPS}×{N_PER} layers): "
            f"avg={avg:.2f}ms  min={mn:.2f}ms  per-layer={avg/(N_CHIPS*N_PER)*1000:.1f}µs"
        )
        print("=" * 60)

    finally:
        for sm in reversed(submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
