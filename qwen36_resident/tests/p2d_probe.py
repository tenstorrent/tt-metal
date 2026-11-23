# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""batch-dflash2 promotions from the prefill's taps (P2d), measured before building it: the drafter's acceptance and the
promotion's cost when its window comes from
  exact    the whole prompt's prefill taps (the original spec start; the reference);
  refill   option B: the last `window` tokens re-prefilled as a fresh prompt (what promotions do now);
  p2d      the prefill's exact taps kept in host memory (copied out right after the prefill), uploaded at the promotion,
           and the newest block (the positions a served prompt's first-token launch / first plain step leave without
           taps) filled from a short fresh pass of the last `--tail` tokens.
Spec decode loop as tests/drafter_refill.py. Reported: tokens per user-step, the promotion time per user (refill vs
p2d's upload + fills + tail pass) and p2d's copy-out time at the prefill.
usage: python -m qwen36_resident.tests.p2d_probe [--users 4] [--prompt-len 6145] [--new 128] [--tail 256]
"""
import argparse
import faulthandler
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows, tap_buffers
from qwen36_resident.tests.drafter_refill import CORPUS, run
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh


def taps_host(pp, s0, n):
    """the last prefill's taps of positions s0 .. n - 1 as host [taps, n - s0, H] bf16 (each tap from its chip)"""
    rows = -(-(n - s0) // pp.block) * pp.block
    out = []
    for L, x in zip(pp.taps, pp.tap_rows(s0, rows)):
        out.append(ttnn.to_torch(ttnn.get_device_tensors(x)[L // pp.Ls]).reshape(rows, -1)[: n - s0])
        ttnn.deallocate(x)
    return torch.stack(out)


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=6145)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--real-only", action="store_true")
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"]
    stride = (len(ids) - a.prompt_len - 101) // a.users
    prompts = [ids[stride * u + 101 : stride * u + 101 + a.prompt_len].tolist() for u in range(a.users)]
    mesh = open_mesh()
    try:
        eng = ResidentDFlashRows(
            mesh, Q.Checkpoint(Q.checkpoint_dir()), users=a.users, max_pos=a.prompt_len + a.new + 256, R=4, prefill=True
        )
        W, pp, mesh_ = eng.window, eng.pp, eng.mesh
        exact_fill = eng._fill_context
        res, times = {}, {"refill": [], "p2d": [], "copy-out": []}
        g_ref, res["exact"] = run(eng, prompts, a.new)
        cur = dict(enumerate(prompts))

        def refill(n_tokens, phys=0, offset=0, **kw):
            ttnn.synchronize_device(mesh_)
            t0 = time.perf_counter()
            s0 = max(0, n_tokens - W) // 64 * 64
            pp.reset()
            pp.run(torch.as_tensor(cur[phys][s0:n_tokens]))
            exact_fill(n_tokens - s0, phys=phys, offset=s0)
            ttnn.synchronize_device(mesh_)
            times["refill"].append(time.perf_counter() - t0)

        def p2d(n_tokens, phys=0, offset=0, **kw):
            ttnn.synchronize_device(mesh_)
            s0 = max(0, n_tokens - W) // 64 * 64
            t0 = time.perf_counter()
            host = taps_host(pp, s0, n_tokens)  # (at the prefill)
            times["copy-out"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()  # (the promotion)
            hb = (n_tokens - 2) // 64 * 64  # (the newest block: the served prompt's positions without taps)
            if hb > s0:
                full = torch.zeros(host.shape[0], eng.fill_rows, host.shape[-1])
                full[:, : hb - s0] = host[:, : hb - s0].float()
                dev = tap_buffers(mesh_, host.shape[0], eng.fill_rows, full)
                eng.drafter.fill_context(dev, s0, phys=phys, valid_len=hb - s0)
                for t in dev:
                    ttnn.deallocate(t)
            ts = max(0, hb - (a.tail - (n_tokens - hb)) // 64 * 64)
            pp.reset()
            pp.run(torch.as_tensor(cur[phys][ts:n_tokens]))
            exact_fill(n_tokens - ts, phys=phys, offset=ts, s0=hb - ts)
            ttnn.synchronize_device(mesh_)
            times["p2d"].append(time.perf_counter() - t0)

        def holes(mode):
            # the served promotion's taps: exact for every prefilled position but the last two (the prompt's last
            # token through the first-token launch, the first plain step), which get `mode`: the last exact row
            # repeated (dup) or zeros — no PP pass at the promotion
            def fill(n_tokens, phys=0, offset=0, **kw):
                s0 = max(0, n_tokens - W) // 64 * 64
                host = taps_host(pp, s0, n_tokens).float()
                host[:, -2:] = host[:, -3:-2] if mode == "dup" else 0.0
                full = torch.zeros(host.shape[0], eng.fill_rows, host.shape[-1])
                full[:, : n_tokens - s0] = host
                dev = tap_buffers(mesh_, host.shape[0], eng.fill_rows, full)
                eng.drafter.fill_context(dev, s0, phys=phys, valid_len=n_tokens - s0)
                for x in dev:
                    ttnn.deallocate(x)

            return fill

        def real(n_tokens, phys=0, offset=0, **kw):  # (ResidentDFlashRows.capture_taps / fill_from_taps, 2 holes)
            ttnn.synchronize_device(mesh_)
            t0 = time.perf_counter()
            rec = eng.capture_taps(n_tokens - 2)
            ttnn.synchronize_device(mesh_)
            times.setdefault("real copy-out", []).append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            eng.fill_from_taps(phys, rec, n_tokens)
            ttnn.synchronize_device(mesh_)
            times.setdefault("real promotion", []).append(time.perf_counter() - t0)

        variants = [("refill", refill), ("p2d", p2d), ("holes-dup", holes("dup")), ("holes-zero", holes("zero"))]
        if a.real_only:
            variants = [("refill", refill)]
        for name, fn in variants + [("real", real), ("real (warm)", real)]:
            if name == "real (warm)":
                for k in ("real promotion", "real copy-out"):
                    times[k] = []
            eng._fill_context = fn
            g, res[name] = run(eng, prompts, a.new)
            first = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(g_ref, g)]
            print(
                f"{name}: {res[name]:.2f} tokens per user-step (exact {res['exact']:.2f}); tokens vs exact, first "
                f"difference per user: {first}",
                flush=True,
            )
        ms = lambda k: f"{1e3 * sum(times[k]) / max(1, len(times[k])):.0f} ms"  # noqa: E731
        print(
            f"per user ({a.prompt_len}-token prompts, window {W}): refill {ms('refill')}, p2d promotion "
            f"{ms('p2d')} (+ copy-out at the prefill {ms('copy-out')}); real: promotion {ms('real promotion')}, "
            f"copy-out {ms('real copy-out')}",
            flush=True,
        )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
