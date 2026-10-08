# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip correctness + device-time harness for ``ttnn.experimental.indexer_score_msa`` (+ the
``topk_large_indices`` selection it feeds) at the M3 4x4 prefill shape: one SP-rank shard of S = chunk/4 query
rows, one index head per device (TP=4), T = 56 320 cached keys, causal, block-max-pool over 128-token blocks,
q_chunk 64 / k_chunk 1024 / head_group_size 0 (the model's program config, tt/attention/msa.py).

Prints:
  IDX_PCC min=<over rows> mean=<..>   (block scores vs fp32 torch on the device's bf8-rounded K; -inf masked)
  IDX_TOPK_RECALL mean=<..>           (|device top-16 ∩ reference top-16| / 16 over the valid rows)
  IDX_OP_US median=<device kernel us> min= max= n=
Device time from the profiler's C++ post-process (env set below, before ttnn is imported).
"""

import argparse
import os
import statistics
import sys

for _k, _v in {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
}.items():
    os.environ.setdefault(_k, _v)

import torch  # noqa: E402

import ttnn  # noqa: E402

BLK = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=1280)
    ap.add_argument("--T", type=int, default=56320, help="valid keys (kv_len)")
    ap.add_argument(
        "--T-alloc",
        type=int,
        default=None,
        help="allocated K rows (the model passes the capacity-sized gather buffer: 1044480 at 1M capacity)",
    )
    ap.add_argument("--chunk-start", type=int, default=51200)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--q-dtype", default="bf16", choices=["bf16", "bf8"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    S, T, d = a.S, a.T, 128
    TA = a.T_alloc or T
    assert TA >= T and TA % BLK == 0
    q0 = a.chunk_start + a.rank * S
    gen = torch.Generator().manual_seed(a.seed)
    # correlated keys (a slow drift + noise) so block maxima are not all ties
    q = torch.randn(1, 1, S, d, generator=gen)
    base = torch.randn(1, 1, T // BLK, 1, d, generator=gen).repeat(1, 1, 1, BLK, 1).view(1, 1, T, d)
    k = 0.6 * base + 0.8 * torch.randn(1, 1, T, d, generator=gen)
    scale = d**-0.5
    dev = ttnn.open_device(device_id=0)
    try:
        qdt = ttnn.bfloat16 if a.q_dtype == "bf16" else ttnn.bfloat8_b
        tq = ttnn.from_torch(q.bfloat16(), dtype=qdt, layout=ttnn.TILE_LAYOUT, device=dev)
        k_alloc = torch.cat([k, torch.zeros(1, 1, TA - T, d)], dim=2) if TA > T else k
        tk = ttnn.from_torch(k_alloc.bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0)

        def run():
            return ttnn.experimental.indexer_score_msa(
                tq, tk, num_groups=1, chunk_start_idx=q0, scale=scale, block_size=BLK, program_config=cfg, kv_len=T
            )

        out = run()
        ids = ttnn.experimental.topk_large_indices(out, k=16, valid_length=T // BLK)
        sc = ttnn.to_torch(out).float()[0, 0][:, : T // BLK]  # [S, nblk] (valid blocks only)
        dev_ids = ttnn.to_torch(ids).to(torch.int64)[0, 0] & 0xFFFFFFFF  # [S, 16]
        qq = ttnn.to_torch(tq).float()[0, 0]
        kq = ttnn.to_torch(tk).float()[0, 0][:T]
        full = (qq @ kq.T) * scale  # [S, T]
        pos = torch.arange(S)[:, None] + q0
        full = full.masked_fill(torch.arange(T)[None, :] > pos, float("-inf"))
        ref = full.view(S, T // BLK, BLK).amax(-1)
        pccs, rec, mismatch = [], [], 0
        for s in range(0, S, max(1, S // 64)):
            fr, fs = torch.isfinite(ref[s]), torch.isfinite(sc[s])
            mismatch += int((fr != fs).sum())  # masked (-inf) pattern must match exactly
            m = fr & fs
            x, y = sc[s][m], ref[s][m]
            pccs.append(torch.corrcoef(torch.stack([x, y]))[0, 1].item())
            r = set(ref[s].topk(16).indices.tolist())
            got = set(dev_ids[s].tolist())
            rec.append(len(r & got) / 16)
        print(
            f"IDX_PCC min={min(pccs):.6f} mean={statistics.mean(pccs):.6f} rows={len(pccs)} mask_mismatch={mismatch}",
            flush=True,
        )
        print(f"IDX_TOPK_RECALL mean={statistics.mean(rec):.4f} min={min(rec):.4f}", flush=True)

        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        durs = []
        for _ in range(a.iters):
            o = run()
            ttnn.synchronize_device(dev)
            ttnn.deallocate(o)
            ttnn.ReadDeviceProfiler(dev)
            best = None
            for lst in ttnn.get_latest_programs_perf_data().values():
                for p in lst:
                    r = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
                    if r is not None and (best is None or r.duration > best):
                        best = r.duration
            if best is not None:
                durs.append(best / 1e3)
        if durs:
            print(
                f"IDX_OP_US median={statistics.median(durs):.2f} min={min(durs):.2f} max={max(durs):.2f} n={len(durs)}",
                flush=True,
            )
        else:
            print("IDX_OP_US unavailable", flush=True)
    finally:
        ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
