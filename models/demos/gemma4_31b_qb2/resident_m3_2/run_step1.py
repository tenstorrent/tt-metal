# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Step-1 driver of the resident Gemma prototype: kernel build on a mock mesh, device correctness and perf runs,
the torch reference, and their comparison.

  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step1 mock --layers 6
  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step1 device --layers 60 --tokens 2,818 --chain 32 --out dev.pt
  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step1 perf --replays 200 --timeline --out perf.json
  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step1 ref --layers 60 --tokens 2,818 --chain 32 --out ref.pt
  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step1 compare dev.pt ref.pt
"""
import argparse
import json
import os
import sys
import time

MOCK_DESC = "tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_qb2_p300_cluster_desc/bh_qb2_p300_cluster_desc.yaml"
HOST_DESC = "tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml"
CACHE = os.environ.get("GRES_CACHE", "/home/ttuser/experiments/gemma4/resident_cache")

if len(sys.argv) > 1 and sys.argv[1] in ("mock", "ref", "compare", "timeline"):
    # no real chips: kernels build against a mock QB2 mesh; the reference only dequantizes host tensors
    desc = MOCK_DESC if sys.argv[1] == "mock" else HOST_DESC
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = os.path.join(os.environ["TT_METAL_HOME"], desc)

import torch  # noqa: E402

import ttnn  # noqa: E402


def open_mesh(trace=64 << 20):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def tokens_arg(s):
    return [int(t) for t in s.split(",")] if s else []


def cmd_mock(a):
    from models.demos.gemma4_31b_qb2.resident_m3_2.model import ResidentGemma

    mesh = open_mesh()
    try:
        m = ResidentGemma(mesh, CACHE, layers=a.layers, rows=a.rows, timeline=True, dump_rounds=True, embed_rows=1024)
        m.set_rows(list(range(2, 2 + a.rows)))
        t0 = time.time()
        m.launch()
        ttnn.synchronize_device(mesh)
        print(f"mock launch (kernel build) ok: {time.time() - t0:.1f} s")
    finally:
        close_mesh(mesh)


def cmd_device(a):
    from models.demos.gemma4_31b_qb2.resident_m3_2.model import ResidentGemma

    mesh = open_mesh()
    res = {"layers": a.layers, "rows": a.rows, "tokens": {}, "chain": None}
    try:
        t0 = time.time()
        m = ResidentGemma(
            mesh,
            CACHE,
            layers=a.layers,
            rows=a.rows,
            slices=a.slices,
            timeline=a.timeline,
            dump_rounds=a.dump_rounds,
            embed_rows=a.embed_rows,
            fidelity=getattr(ttnn.MathFidelity, a.fidelity),
        )
        print(f"model built: {time.time() - t0:.0f} s", flush=True)
        m.reset()
        toks = tokens_arg(a.tokens)
        for g in range(0, len(toks) - a.rows + 1, a.rows):
            group = toks[g : g + a.rows]
            m.set_rows(group, 0)
            m.launch()
            ttnn.synchronize_device(mesh)
            outs = m.out()
            fh, lg = m.final_h(), m.logits()
            rh = m.round_h() if a.dump_rounds else None
            for r, t in enumerate(group):
                entry = {
                    "out": [(o[0][r], o[1][r]) for o in outs],
                    "final_h": fh[:, r],
                    "logits": lg[r],
                    "meta": outs[0][2],
                }
                if rh is not None:
                    entry["round_h"] = rh[:, r]
                res["tokens"][t] = entry
            print(f"rows {group}: tokens per chip {[o[0] for o in outs]}, meta {outs[0][2]}", flush=True)
        if a.chain:
            assert a.rows == 1, "the device feed chains plain decode (M = 1)"
            start = toks[0]
            m.set_rows([start], 0)
            chain = []
            for _ in range(a.chain):
                m.launch()  # each step feeds the next on device
                ttnn.synchronize_device(mesh)
                chain.append(m.tokens()[0])
            res["chain"] = {"start": start, "tokens": chain}
            print(f"chain from {start}: {chain}", flush=True)
    finally:
        close_mesh(mesh)
    torch.save(res, a.out)


def read_aiclk():
    """per chip AICLK (MHz) from the driver's sysfs, where present"""
    import glob

    out = {}
    for d in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*")):
        row = {}
        p = os.path.join(d, "tt_aiclk")
        if os.path.exists(p):
            row["aiclk_mhz"] = open(p).read().strip()
        for hw in glob.glob(os.path.join(d, "device/hwmon/*")):
            for name, key, scale in (("temp1_input", "temp_c", 1e-3), ("power1_input", "power_w", 1e-6)):
                q = os.path.join(hw, name)
                if os.path.exists(q):
                    row[key] = round(int(open(q).read().strip()) * scale, 1)
        out[os.path.basename(d)] = row
    return out


def cmd_perf(a):
    from models.demos.gemma4_31b_qb2.resident_m3_2.model import ResidentGemma

    mesh = open_mesh()
    res = {"layers": a.layers}
    try:
        t0 = time.time()
        m = ResidentGemma(
            mesh,
            CACHE,
            layers=a.layers,
            rows=a.rows,
            slices=a.slices,
            timeline=a.timeline,
            fidelity=getattr(ttnn.MathFidelity, a.fidelity),
        )
        print(f"model built: {time.time() - t0:.0f} s", flush=True)
        res["rows"], res["slices"], res["ring_bytes"] = a.rows, m.K, m.ring_bytes
        m.reset()
        m.set_rows(list(range(2, 2 + a.rows)), 0)
        m.launch()  # compile + warm
        ttnn.synchronize_device(mesh)
        m.capture_trace()
        res["aiclk_before"] = read_aiclk()
        # per-replay device time: one replay, synchronized, timed on the host
        per = []
        for _ in range(a.replays):
            ts = time.perf_counter()
            m.launch()
            ttnn.synchronize_device(mesh)
            per.append(1e3 * (time.perf_counter() - ts))
        per.sort()
        res["per_replay_ms"] = {"median": per[len(per) // 2], "min": per[0], "p90": per[int(0.9 * len(per))]}
        # back-to-back replays (burst): the host only enqueues, the hubs feed every next step on device
        ts = time.perf_counter()
        for _ in range(a.replays):
            m.launch()
        ttnn.synchronize_device(mesh)
        res["burst_ms_per_token"] = 1e3 * (time.perf_counter() - ts) / a.replays
        res["weight_bytes_per_chip"] = m.weight_bytes
        res["GBps_per_chip_burst"] = m.weight_bytes / (res["burst_ms_per_token"] * 1e6)
        res["GBps_per_chip_median"] = m.weight_bytes / (res["per_replay_ms"]["median"] * 1e6)
        if a.timeline:
            st, hub, sl = m.timelines()
            torch.save({"streamer": st, "hub": hub, "slices": sl}, a.out.replace(".json", "_timeline.pt"))
        if a.sustain:
            ts = time.perf_counter()
            n, samples, chunks = 0, [], []
            while time.perf_counter() - ts < a.sustain:
                tc = time.perf_counter()
                for _ in range(100):
                    m.launch()
                if len(samples) < 1 or time.perf_counter() - ts > len(samples) * 10:
                    samples.append((round(time.perf_counter() - ts, 1), read_aiclk()))  # device busy
                ttnn.synchronize_device(mesh)
                chunks.append(1e3 * (time.perf_counter() - tc) / 100)
                n += 100
            res["sustained"] = {
                "seconds": time.perf_counter() - ts,
                "tokens": n,
                "ms_per_token": 1e3 * (time.perf_counter() - ts) / n,
                "first_chunk_ms": chunks[0],
                "last_chunk_ms": chunks[-1],
                "telemetry_during": samples,
            }
            res["aiclk_after"] = read_aiclk()
        res["out"] = m.out()[0]
        m.release_trace()
    finally:
        close_mesh(mesh)
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


def cmd_ref(a):
    from models.demos.gemma4_31b_qb2.resident_m3_2.reference import Reference

    ref = Reference(CACHE, layers=a.layers)
    res = {"layers": a.layers, "tokens": {}, "chain": None}
    for t in tokens_arg(a.tokens):
        rounds = []
        logits, h = ref.step(t, rounds_out=rounds)
        res["tokens"][t] = {"token": ref.argmax(logits), "final_h": h, "logits": logits, "round_h": torch.stack(rounds)}
        print(f"token {t} -> {res['tokens'][t]['token']}", flush=True)
    if a.chain:
        start = tokens_arg(a.tokens)[0]
        res["chain"] = {"start": start, "tokens": ref.chain(start, a.chain)}
        print(f"chain from {start}: {res['chain']['tokens']}", flush=True)
    torch.save(res, a.out)


def cmd_timeline(a):
    """phase breakdown of one step from the hub, slice and streamer timelines (wall-clock cycles at AICLK)"""
    tl = torch.load(a.file)
    st, hub, sl = tl["streamer"], tl["hub"], tl["slices"]
    n, S, L, _ = st.shape
    rounds = 2 * L
    out = {}
    for c in range(n):
        h, s0 = hub[c], sl[c, 0]
        t0, t1 = int(h[0]), int(h[1])
        sr = lambda ar, k: int(s0[2 + 4 * ar + k])
        stream = post = pre = mcast = fabric = 0
        prev = int(s0[2 + 4 * 0]) if rounds == 0 else None
        for ar in range(rounds):
            if ar > 0:
                stream += sr(ar, 0) - sr(ar - 1, 3)
            post += sr(ar, 1) - sr(ar, 0)
            pre += sr(ar, 2) - sr(ar, 1)
            mcast += sr(ar, 3) - sr(ar, 2)
            fabric += int(h[3 + 2 * ar]) - int(h[2 + 2 * ar])
        last = sr(rounds - 1, 3)
        stalls = st[c, :, :, 4:8].sum(dim=(1, 2)).double()
        out[c] = {
            "total_cycles": t1 - t0,
            "seed_to_first_slots": sr(0, 0) - t0,
            "streaming (h mcast -> all slots in)": stream,
            "slice post norm + exchange": post,
            "slice pre norm + exchange": pre,
            "slice h multicast": mcast,
            "hub fabric sends (overlapped)": fabric,
            "lm_head + records (last h -> records in)": int(h[2 + 2 * rounds]) - last,
            "argmax exchange": int(h[3 + 2 * rounds]) - int(h[2 + 2 * rounds]),
            "tail": t1 - int(h[3 + 2 * rounds]),
            "reader_ring_stall_cycles_mean": float(stalls.mean()),
            "reader_ring_stall_cycles_max": float(stalls.max()),
        }
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def cmd_compare(a):
    dev, ref = torch.load(a.dev), torch.load(a.ref)
    same = dev["layers"] == ref["layers"]
    report = {"layers": dev["layers"], "ref_layers": ref["layers"]}
    for t, d in dev["tokens"].items():
        r = ref["tokens"][t]
        row = {"device_token": d["out"][0][0], "chips_agree": len({o[0] for o in d["out"]}) == 1}
        if "meta" in d:
            row["meta"] = list(d["meta"])
        if same:
            row.update(
                {
                    "ref_token": r["token"],
                    "final_h_pcc": pcc(d["final_h"][0], r["final_h"]),
                    "logits_pcc": pcc(d["logits"], r["logits"]),
                    "ref_top2_gap": float(torch.topk(r["logits"], 2).values.diff().abs()),
                    "device_logit_of_ref_token_minus_max": float(d["logits"][r["token"]] - d["logits"].max()),
                }
            )
        if "round_h" in d:
            dh, rh = d["round_h"][0], r["round_h"]
            k = min(dh.shape[0], rh.shape[0])
            pc = [pcc(dh[i], rh[i]) for i in range(k)]
            row["round_pcc_min"] = min(pc)
            row["round_pcc_last"] = pc[-1]
            row["round_pcc"] = [round(v, 5) for v in pc]
            row["chips_round_h_identical"] = bool((d["round_h"] == d["round_h"][0:1]).all())
        report[t] = row
    if same and dev.get("chain") and ref.get("chain"):
        dc, rc = dev["chain"]["tokens"], ref["chain"]["tokens"]
        first = next((i for i, (x, y) in enumerate(zip(dc, rc)) if x != y), None)
        report["chain"] = {"device": dc, "ref": rc, "match": sum(x == y for x, y in zip(dc, rc)), "first_diff": first}
    print(json.dumps(report, indent=1))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("mock")
    s.add_argument("--layers", type=int, default=6)
    s.add_argument("--rows", type=int, default=1)
    for name in ("device", "ref"):
        s = sub.add_parser(name)
        s.add_argument("--layers", type=int, default=60)
        s.add_argument("--tokens", default="2")
        s.add_argument("--chain", type=int, default=0)
        s.add_argument("--out", required=True)
        if name == "device":
            s.add_argument("--rows", type=int, default=1)
            s.add_argument("--slices", type=int, default=None)
            s.add_argument("--timeline", action="store_true")
            s.add_argument("--dump-rounds", action="store_true")
            s.add_argument("--embed-rows", type=int, default=None)
            s.add_argument("--fidelity", default="LoFi", help="matmul fidelity of the M > 8 path (custom_mm is LoFi)")
    s = sub.add_parser("perf")
    s.add_argument("--layers", type=int, default=60)
    s.add_argument("--rows", type=int, default=1)
    s.add_argument("--slices", type=int, default=None)
    s.add_argument("--fidelity", default="LoFi")
    s.add_argument("--replays", type=int, default=200)
    s.add_argument("--timeline", action="store_true")
    s.add_argument("--sustain", type=float, default=0.0)
    s.add_argument("--out", required=True)
    s = sub.add_parser("compare")
    s.add_argument("dev")
    s.add_argument("ref")
    s.add_argument("--out")
    s = sub.add_parser("timeline")
    s.add_argument("file")
    s.add_argument("--out")
    a = p.parse_args()
    cmds = {"mock": cmd_mock, "device": cmd_device, "perf": cmd_perf, "ref": cmd_ref, "compare": cmd_compare}
    cmds["timeline"] = cmd_timeline
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
