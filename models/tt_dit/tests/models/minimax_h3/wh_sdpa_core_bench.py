# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-core bench of the WH SDPA core loop (tt-lab-private h3-wh wh_sdpa_trisc.h) as a tt-metal compute kernel.

One 64-query block of a head of 128 against BLOCKS 64-key blocks resident in L1. Prints cycles per block and the
accuracy of O against a float32 torch reference. Usage: python wh_sdpa_core_bench.py [BLOCKS] [LAST_VALID] [DEBUG]
"""

import sys

import torch

import ttnn

KERNEL = "ttnn/cpp/ttnn/operations/experimental/minimax_h3_wh_sdpa/kernels/wh_sdpa_bench.cpp"
Q_BYTES, P_BYTES, KV_BYTES, REGION = 8 * 2048, 8 * 2048, 16 * 2048, 8192


def rand_score(shape, gen):
    # |x| in [2^-3, 1), random sign, as the release bench's random_score_bf16
    mag = torch.exp2(-3 * torch.rand(shape, generator=gen))
    sign = torch.where(torch.rand(shape, generator=gen) < 0.5, -1.0, 1.0)
    return (mag * sign).to(torch.bfloat16)


def q_chunks(q):  # q [64, 128] -> [8 * 1024]: chunk f = [query face c][16 d][16 queries]
    return q.reshape(4, 16, 8, 16).permute(2, 0, 3, 1).reshape(-1)


def kv_block(k, v):  # k, v [64, 128] -> [16 * 1024]
    kc = k.reshape(64, 8, 16).permute(1, 0, 2).reshape(8, 1024)  # chunk f: [key][16 d]
    # chunk 8 + 2 s + h: [64 d][16 keys] for keys 16 s.., d 64 h..
    vt = v.reshape(4, 16, 2, 64).permute(0, 2, 3, 1).reshape(8, 1024)
    return torch.cat([kc, vt]).reshape(-1)


def o_from_region(region):  # region [256 rows][16] slice major -> O half [64 queries][64 d]
    r = region.reshape(4, 4, 16, 16)  # [i // 16][c][i % 16][col]
    return r.permute(1, 3, 0, 2).reshape(64, 64)  # [c][col][i//16][i%16]


def l1_tensor(t, dev, dtype=ttnn.bfloat16):
    rows, cols = t.shape
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        [rows, cols],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)


def main():
    control = "--control" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--control"]
    blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    last_valid = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    debug = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    gen = torch.Generator().manual_seed(1)
    q = rand_score((64, 128), gen)
    k = rand_score((64 * blocks, 128), gen)
    v = torch.randn((64 * blocks, 128), generator=gen).to(torch.bfloat16)
    pad = 64 * blocks - (64 - last_valid)
    k[pad:] = 0
    v[pad:] = 0

    qp = torch.cat([q_chunks(q), torch.zeros(P_BYTES // 2, dtype=torch.bfloat16)]).reshape(16, 1024)
    kv = torch.cat([kv_block(k[64 * j : 64 * j + 64], v[64 * j : 64 * j + 64]) for j in range(blocks)]).reshape(
        16 * blocks, 1024
    )
    out = torch.full((12, 1024), float("nan"), dtype=torch.bfloat16)
    sync = torch.zeros((16, 64), dtype=torch.int32)

    dev = ttnn.open_device(device_id=0)
    try:
        t_qp, t_kv, t_out = l1_tensor(qp, dev), l1_tensor(kv, dev), l1_tensor(out, dev)
        t_sync = l1_tensor(sync, dev, ttnn.int32)
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        results = []
        for go in () if control else (1, 2, 3):
            rt = ttnn.RuntimeArgs()
            rt[0][0] = [
                t_qp.buffer_address(),
                t_kv.buffer_address(),
                t_out.buffer_address(),
                blocks,
                last_valid,
                debug,
                t_sync.buffer_address(),
                go,
            ]
            kd = ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.ComputeConfigDescriptor(),
            )
            pd = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])
            ttnn.generic_op([t_qp, t_kv, t_sync, t_out], pd)
            ttnn.synchronize_device(dev)
            s = ttnn.to_torch(t_sync).to(torch.int64).flatten() & 0xFFFFFFFF
            start = [s[12 + 2 * i] | (s[13 + 2 * i] << 32) for i in range(3)]
            end = [s[18 + 2 * i] | (s[19 + 2 * i] << 32) for i in range(3)]
            cycles = int(max(end) - min(start))
            o_dev = ttnn.to_torch(t_out).reshape(-1)
            results.append((cycles, o_dev.clone()))
            print(f"run {go}: {cycles} cycles, {cycles / blocks:.0f} a block; done {s[8:11].tolist()}")
        if results:
            o_dev = results[-1][1]
            o = torch.cat([o_from_region(o_dev[h * 4096 : (h + 1) * 4096]) for h in range(2)], dim=1).float()
            valid = 64 * blocks - (64 - last_valid)
            ref = torch.softmax((q.float() @ k[:valid].float().T) / 128**0.5, dim=-1) @ v[:valid].float()
            pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
            rel = ((o - ref).norm() / ref.norm()).item()
            same = all(torch.equal(results[0][1], r[1]) for r in results[1:])
            print(
                f"blocks {blocks} valid {last_valid}: PCC {pcc:.6f}, rel err {rel:.3e}, "
                f"max abs {(o - ref).abs().max():.3e}, runs identical {same}"
            )
        # Standard ops after the custom kernel on the same core, to check it leaves the core usable.
        a = torch.randn(256, 256) / 16
        ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        mm = ttnn.to_torch(ttnn.matmul(ta, ta)).float()
        ex = ttnn.to_torch(ttnn.exp(ta)).float()
        sm = ttnn.to_torch(ttnn.softmax(ta, dim=-1)).float()
        ab = a.bfloat16().float()
        for name, got, want in (
            ("matmul", mm, ab @ ab),
            ("exp", ex, torch.exp(ab)),
            ("softmax", sm, torch.softmax(ab, -1)),
        ):
            pcc_f = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
            print(f"follow-up {name}: PCC {pcc_f:.6f}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
