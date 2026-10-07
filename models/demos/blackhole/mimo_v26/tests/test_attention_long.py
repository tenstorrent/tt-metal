# SPDX-License-Identifier: Apache-2.0
"""64k-context attention on the 4x8 mesh: chunked prefill (chunks of 2048, traced, chunk_start_idx_tensor) of three users with TWO different context lengths in one
process (64k, 32k, 64k again), per-chunk time, DRAM allocated/free after each user (leak check), then decode at positions 64k / 32k / 64k.
Reference: torch fp32 on sampled query rows (all key/value rows they need). Random weights/data of real shapes. ATTN_KINDS=ga,swa ATTN_CTX=65536 ATTN_CTX2=32768."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_ref import pcc, ref_attention_rows  # noqa: E402
from harness import Traced, close_mesh, dev_rows, dram, gather_all, gather_col, host_rows, open_mesh, rec  # noqa: E402

from models.demos.blackhole.mimo_v26.tt.attention import (  # noqa: E402
    BLOCK,
    DQK,
    DV,
    GA,
    HIDDEN,
    NQ,
    SWA,
    MiMoAttention,
    convert_weights,
)

E = os.environ.get
CTX, CTX2, CH, ND = int(E("ATTN_CTX", "65536")), int(E("ATTN_CTX2", "32768")), 2048, int(E("ATTN_ND", "2"))
MAXPOS = CTX + ND + 256 - (CTX + ND) % 256
ROWS = 4
NS = int(E("ATTN_SAMPLES", "6"))


def main():
    kinds = E("ATTN_KINDS", "ga,swa").split(",")
    mesh = open_mesh(trace_mb=512)
    try:
        g = torch.Generator().manual_seed(0)
        for kind in kinds:
            cfg = SWA if kind == "swa" else GA
            q_w = torch.randn(NQ * DQK, HIDDEN, generator=g) / 64
            k_w = torch.randn(cfg.nkv * DQK, HIDDEN, generator=g) / 64
            v_w = torch.randn(cfg.nkv * DV, HIDDEN, generator=g) / 64
            o_w = torch.randn(HIDDEN, NQ * DV, generator=g) / 90
            sink = ((torch.rand(NQ, generator=g) * 0.58 + 0.56) * 3).bfloat16() if cfg.sink else None
            hw = convert_weights(cfg, q_w, k_w, v_w, o_w, sink)
            ring = CH // BLOCK + 8 if cfg.window else 0
            nb = ring if cfg.window else MAXPOS // BLOCK
            perm = torch.randperm(32 * nb, generator=g).reshape(32, nb).to(torch.int32)
            d0 = dram(mesh)
            m = MiMoAttention(mesh, cfg, hw, perm, MAXPOS, ring_blocks=ring)
            rec("build", kind=kind, maxpos=MAXPOS, dram_before=d0, dram_after_build=dram(mesh))
            x0 = torch.randn(ROWS, CTX + ND, HIDDEN, generator=g).bfloat16()
            x1 = torch.randn(ROWS, CTX2 + ND, HIDDEN, generator=g).bfloat16()
            scen = [(0, CTX, x0), (1, CTX2, x1), (2, CTX, x0)]
            io = m.prefill_io(CH)
            xdev = dev_rows(mesh, torch.zeros(ROWS, 1, CH, HIDDEN).bfloat16())
            tr = None
            for user, ctx, x in scen:
                samples = {}  # chunk -> (offsets, out[rows, n, 4096])
                times = []
                for c in range(ctx // CH):
                    xin = x[:, c * CH : (c + 1) * CH].reshape(ROWS, 1, CH, HIDDEN)
                    ttnn.copy_host_to_device_tensor(host_rows(mesh, xin), xdev)
                    m.prefill_set(io, c * CH, user)
                    if tr is None:
                        tr = Traced(mesh, lambda: m.prefill_chunk(xdev, io))
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    o = tr.replay(block=True)
                    times.append((time.perf_counter() - t0) * 1e3)
                    offs = sorted(set(torch.randint(0, CH, (NS,), generator=g).tolist() + [0, CH - 1]))
                    h = gather_col(mesh, o)  # [rows, CH, 4096] (column 0 of each row)
                    samples[c] = (offs, h.reshape(ROWS, CH, HIDDEN)[:, offs].clone())
                scen_res = {}
                # reference on sampled rows
                allpos = sorted(c * CH + off for c, (offs, _) in samples.items() for off in offs)
                per_row = []
                for r in range(ROWS):
                    refr = ref_attention_rows(x[r], allpos, q_w, k_w, v_w, o_w, sink, cfg.nkv, cfg.theta, cfg.window)
                    ref_map = {p: refr[i] for i, p in enumerate(allpos)}
                    dev = torch.cat([samples[c][1][r] for c in sorted(samples)], 0)
                    rf = torch.stack([ref_map[c * CH + off] for c in sorted(samples) for off in samples[c][0]], 0)
                    chunk_pcc = []
                    i = 0
                    for c in sorted(samples):
                        n = len(samples[c][0])
                        chunk_pcc.append(pcc(dev[i : i + n], rf[i : i + n]))
                        i += n
                    per_row.append(
                        dict(
                            all=round(pcc(dev, rf), 5),
                            min_chunk=round(min(chunk_pcc), 5),
                            last_chunk=round(chunk_pcc[-1], 5),
                        )
                    )
                n = len(times)
                rec(
                    "prefill_long",
                    kind=kind,
                    user=user,
                    ctx=ctx,
                    chunks=n,
                    pcc_per_row=per_row,
                    chunk_ms_first=round(times[0], 2),
                    chunk_ms_median=round(sorted(times)[n // 2], 2),
                    chunk_ms_last=round(times[-1], 2),
                    total_s=round(sum(times) / 1e3, 2),
                    chunk_ms_every4=[round(t, 1) for t in times[::4]],
                    dram_after=dram(mesh),
                )
            # ---- decode at 64k / 32k / 64k (slots 0,1,2)
            dio = m.decode_io()
            dx = dev_rows(mesh, torch.zeros(ROWS, 1, 32, HIDDEN).bfloat16())
            dtr = None
            res = []
            for step in range(ND):
                pos = [CTX + step, CTX2 + step, CTX + step] + [0] * 29
                xin = torch.randn(ROWS, 32, HIDDEN, generator=g).bfloat16()
                xin[:, 0], xin[:, 1], xin[:, 2] = x0[:, CTX + step], x1[:, CTX2 + step], x0[:, CTX + step]
                m.decode_set(dio, pos)
                ttnn.copy_host_to_device_tensor(host_rows(mesh, xin.reshape(ROWS, 1, 32, HIDDEN)), dx)
                if dtr is None:
                    dtr = Traced(mesh, lambda: m.decode(dx, dio["cur"], dio["idx"]))
                h = gather_all(mesh, dtr.replay())
                out = {}
                for slot, (x, p) in enumerate([(x0, CTX + step), (x1, CTX2 + step), (x0, CTX + step)]):
                    out[slot] = min(
                        pcc(
                            h[r, 0, slot],
                            ref_attention_rows(x[r], [p], q_w, k_w, v_w, o_w, sink, cfg.nkv, cfg.theta, cfg.window)[0],
                        )
                        for r in range(ROWS)
                    )
                res.append(
                    dict(
                        step=step,
                        pcc_min_per_slot={k: round(v, 5) for k, v in out.items()},
                        col_spread=float((h - h[:, :1]).abs().max()),
                    )
                )
            same = bool(torch.equal(gather_all(mesh, dtr.replay()), h))
            t0 = time.perf_counter()
            for _ in range(20):
                dtr.replay(block=False)
            ttnn.synchronize_device(mesh)
            rec(
                "decode_long",
                kind=kind,
                steps=res,
                replay_bit_identical=same,
                decode_ms_traced_32users=round((time.perf_counter() - t0) / 20 * 1e3, 3),
                dram_after=dram(mesh),
            )
            tr.release()
            dtr.release()
            del m
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
