# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Step-2 driver of the resident Gemma engine (real attention, one user): kernel build on a mock mesh, the ttnn
prefill -> resident decode handoff checks and the perf runs.

  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step2 mock --rows 1
  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step2 tf --ctx 128 --out tf_128.json
  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step2 e2e --tokens 256 --out e2e.json
  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step2 m8 --ctx 128 --out m8.json
  python -m models.demos.gemma4_31b_qb2.resident_m3_2.run_step2 perf --rows 1 --ctx 128,1024,8192,32768 --out perf.json
"""
import argparse
import json
import os
import sys
import time

MOCK_DESC = "tt_metal/third_party/tt-cluster-descriptors/blackhole/bh_qb2_p300_cluster_desc/bh_qb2_p300_cluster_desc.yaml"
CACHE = os.environ.get("GRES_CACHE", "/home/ttuser/experiments/gemma4/resident_cache")

if len(sys.argv) > 1 and sys.argv[1] == "mock":
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = os.path.join(os.environ["TT_METAL_HOME"], MOCK_DESC)

import torch  # noqa: E402

import ttnn  # noqa: E402


def open_mesh(trace=256 << 20):
    """the ttnn QB2 path's device setup (1D ring fabric, 8 KB packets), so both engines share one process"""
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=16384, trace_region_size=trace)


def read_aiclk():
    """per chip AICLK (MHz) from the driver's sysfs, where present"""
    import glob

    out = {}
    for d in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*")):
        p = os.path.join(d, "tt_aiclk")
        if os.path.exists(p):
            out[os.path.basename(d)] = open(p).read().strip()
    return out


def time_engine(m, mesh, replays):
    """(per-replay median ms, burst ms per launch) of the captured trace"""
    m.capture_trace()
    per = []
    for _ in range(replays):
        ts = time.perf_counter()
        m.launch()
        ttnn.synchronize_device(mesh)
        per.append(1e3 * (time.perf_counter() - ts))
    per.sort()
    ts = time.perf_counter()
    for _ in range(replays):
        m.launch()
    ttnn.synchronize_device(mesh)
    burst = 1e3 * (time.perf_counter() - ts) / replays
    m.release_trace()
    return {"median": per[len(per) // 2], "min": per[0], "p90": per[int(0.9 * len(per))]}, burst


def cmd_perf(a):
    """synthetic (zero) caches at each context; M = 1 steps feed themselves on device (the position advances)"""
    from models.demos.gemma4_31b_qb2.resident_m3_2.attention import AttnInputs
    from models.demos.gemma4_31b_qb2.resident_m3_2.model import ResidentGemma

    ctxs = [int(c) for c in a.ctxs.split(",")]
    mesh = open_mesh()
    res = {"rows": a.rows, "layers": a.layers, "runs": []}
    first = None
    try:
        for ctx in ctxs:
            attn = AttnInputs.synthetic(mesh, ctx + 2 * a.replays + 128, layers=a.layers, random=False)
            t0 = time.time()
            m = ResidentGemma(mesh, CACHE, attn, layers=a.layers, rows=a.rows, shared=first, timeline=bool(a.timeline),
                              acc_fp32=bool(a.acc_fp32),
                              attn_fidelity=(ttnn.MathFidelity.HiFi4, getattr(ttnn.MathFidelity, a.attn_fidelity)))
            first = first or m
            m.reset()
            m.set_rows(list(range(2, 2 + a.rows)), ctx)
            m.launch()
            ttnn.synchronize_device(mesh)
            per, burst = time_engine(m, mesh, a.replays)
            row = {"ctx": ctx, "per_replay_ms": per, "burst_ms": burst, "build_s": round(time.time() - t0),
                   "out_meta": m.out()[0][2], "aiclk": read_aiclk()}
            if a.timeline:
                # streamer writer marks (chip 0, every streamer): 8 h of the mixer in, 9 o_in in (QKV + attention + O),
                # 11 h of the MLP in; per layer the slowest streamer, microseconds at AICLK 1350 MHz
                st = m.timelines()[0][0].double()  # [S, layers, 16]
                mixer = (st[:, :, 9] - st[:, :, 8]).max(0).values / 1350
                layer = torch.cat([(st[:, 1:, 8] - st[:, :-1, 8]).max(0).values / 1350])
                full = torch.tensor([l % 6 == 5 for l in range(a.layers)])
                row["timeline_us"] = {
                    "mixer_sliding": float(mixer[~full].mean()), "mixer_full": float(mixer[full].mean()),
                    "layer_mean": float(layer.mean()),
                }
            print(json.dumps(row), flush=True)
            res["runs"].append(row)
            if m is not first:
                del m
            del attn
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def cmd_mock(a):
    from models.demos.gemma4_31b_qb2.resident_m3_2.attention import AttnInputs
    from models.demos.gemma4_31b_qb2.resident_m3_2.model import ResidentGemma

    mesh = open_mesh()
    try:
        attn = AttnInputs.synthetic(mesh, a.ctx + 64, layers=a.layers)
        m = ResidentGemma(mesh, CACHE, attn, layers=a.layers, rows=a.rows, embed_rows=1024, taps=[0])
        m.set_rows(list(range(2, 2 + a.rows)), a.ctx)
        t0 = time.time()
        m.launch()
        ttnn.synchronize_device(mesh)
        print(f"mock launch (kernel build) ok: {time.time() - t0:.1f} s")
    finally:
        close_mesh(mesh)


def per_chip(t):
    return [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]


def read_caches(attn, layers):
    """[layer][chip] (K, V) float [pages, H, 128, D]"""
    return [list(zip(per_chip(k), per_chip(v))) for k, v in attn.kv[:layers]]


def rope_host(attn):
    return {kind: tuple(per_chip(t)[0] for t in pair) for kind, pair in attn.rope.items()}


def cmd_unit(a):
    """synthetic (random) caches, one resident step at position ctx with every layer tapped, then a torch reference
    per layer from the device's own layer inputs: attention output, new K / V rows, layer output"""
    from models.demos.gemma4_31b_qb2.resident_m3_2.attention import AttnInputs
    from models.demos.gemma4_31b_qb2.resident_m3_2.model import ResidentGemma
    from models.demos.gemma4_31b_qb2.resident_m3_2.reference_s2 import Reference2, pcc
    from models.demos.gemma4_31b_qb2.resident_m3_2.weights import Checkpoint

    toks = list(range(1000, 1000 + a.rows))
    mesh = open_mesh()
    try:
        attn = AttnInputs.synthetic(mesh, a.ctx + 256, layers=a.layers)
        m = ResidentGemma(mesh, CACHE, attn, layers=a.layers, rows=a.rows, dump_rounds=True, taps=range(a.layers),
                          embed_rows=4096, fidelity=getattr(ttnn.MathFidelity, a.fidelity), acc_fp32=bool(a.acc_fp32))
        m.reset()
        m.set_rows(toks, a.ctx)
        t0 = time.time()
        m.launch()
        ttnn.synchronize_device(mesh)
        print(f"step done {time.time() - t0:.1f} s, out {m.out()[0]}", flush=True)
        taps, o_dev = m.taps(), m.attn_out()
        logits_dev, fh = m.logits(), m.final_h()[0]
        caches = read_caches(attn, a.layers)
        tables, ropes = attn.tables, rope_host(attn)
        emb = m.emb_host
    finally:
        close_mesh(mesh)
    ck = Checkpoint()
    norms = [tuple(ck.layer(l, f"self_attn.{n}_norm.weight").bfloat16() for n in ("q", "k")) for l in range(a.layers)]
    ref = Reference2(CACHE, a.layers, ropes, norms)
    res = []
    x = torch.stack([emb[t].float() for t in toks])
    for l in range(a.layers):
        xr, info = ref.layer(l, x, a.ctx, caches[l], tables)
        D = 512 if l % 6 == 5 else 256
        row = {
            "layer": l,
            "o": [pcc(o_dev[c, l, :, : 8 * D], info["o"][c]) for c in range(4)],
            "k": [pcc(info["k_dev"][c], info["k"][c]) for c in range(4)],
            "v": [pcc(info["v_dev"][c], info["v"][c]) for c in range(4)],
            "x": pcc(taps[l][0], xr),
            "x_chips": max(float((taps[l][c] - taps[l][0]).abs().max()) for c in range(4)),
        }
        if a.detail:
            for c in range(1 if a.detail == 1 else 4):
                od = o_dev[c, l, :, : 8 * D].reshape(a.rows, 8, D)
                orf = info["o"][c].reshape(a.rows, 8, D)
                row[f"heads_c{c}"] = [[round(pcc(od[r, h], orf[r, h]), 4) for h in range(8)] for r in range(a.rows)]
                row[f"scale_c{c}"] = [
                    [round(float(od[r, h] @ orf[r, h] / (orf[r, h] @ orf[r, h] + 1e-30)), 3) for h in range(8)]
                    for r in range(a.rows)
                ]
        print(json.dumps(row), flush=True)
        res.append(row)
        x = taps[l][0]
    # LM head on the device's own final hidden rows: device logits vs fp32 logits of the same BF8 head weights
    head = torch.cat([fh.double() @ w.double() for w in ref.head], dim=-1)
    rel = [float((logits_dev[r].double() - head[r]).norm() / head[r].norm()) for r in range(a.rows)]
    agree = [int(logits_dev[r].argmax()) == int(head[r].argmax()) for r in range(a.rows)]
    print(json.dumps({"head_fidelity": a.fidelity, "head_rel_l2": rel, "head_argmax_agree": agree}), flush=True)
    res.append({"head_fidelity": a.fidelity, "head_rel_l2": rel, "head_argmax_agree": agree})
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


PROMPTS = {
    "code": "Write a Python function that parses a log file of HTTP requests, counts the requests per status code and "
    "per hour, and prints a small report. Include type hints, docstrings and a short usage example.",
    "prose": "Write a short story (about 300 words) about a lighthouse keeper who discovers that the light has been "
    "guiding something other than ships. Use vivid but plain language.",
    "math": "A train leaves city A at 9:00 traveling at 80 km/h toward city B, 420 km away. A second train leaves B at "
    "10:30 toward A at 100 km/h. At what time and how far from A do they meet? Show the steps.",
}

FILLER = (
    "The archive held letters from every decade of the town's history: shipping manifests, school reports, recipes "
    "copied by hand, quarrels about fences and water rights, and long descriptions of weather that nobody else had "
    "bothered to record. Section {i} of the catalogue lists the boxes by year, the condition of each item and the "
    "names of the families who donated them. "
)


def chat_ids(tok, text):
    msgs = [{"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok.encode(prompt, add_special_tokens=False)


def context_ids(tok, n):
    """n tokens of a chat prompt over a long filler document (the teacher-forced checks' context)"""
    body, i = "", 0
    while True:
        ids = chat_ids(tok, "Summarize the following catalogue.\n\n" + body)
        if len(ids) >= n + 16:
            return ids[:n], ids[n : n + 16]
        body += FILLER.format(i=i) * 8
        i += 1


class Handoff:
    """the ttnn QB2 generator (prefill, the reference decoder) and resident engines on its KV cache"""

    def __init__(self, mesh, capacity, attn_fidelity="HiFi2", acc_fp32=True):
        from models.demos.gemma4_31b_qb2.resident_m3_2.attention import AttnInputs
        from models.demos.gemma4_31b_qb2.resident_m3_2.model import ResidentGemma
        from models.demos.gemma4_31b_qb2.tt.generator import build_generator

        t0 = time.time()
        self.mesh = mesh
        self.gen = build_generator(mesh, max_seq_len=capacity)
        print(f"ttnn generator built: {time.time() - t0:.0f} s", flush=True)
        self.state = self.gen._owned(1, capacity)
        self.gen._tables(self.state, self.state.page_tables)
        self.attn = AttnInputs.from_generator(self.gen, self.state)
        t0 = time.time()
        # the ttnn model keeps L1 buffers near the top of L1 (seen at 1491904): the streamers' CBs end below 1.33 MB
        top = 1_330_000
        fid = (ttnn.MathFidelity.HiFi4, getattr(ttnn.MathFidelity, attn_fidelity))
        self.m1 = ResidentGemma(mesh, CACHE, self.attn, rows=1, dump_rounds=True, taps=range(60), l1_top=top,
                                attn_fidelity=fid, acc_fp32=acc_fp32)
        self.m8 = ResidentGemma(mesh, CACHE, self.attn, rows=8, dump_rounds=True, taps=range(60), shared=self.m1,
                                l1_top=top, attn_fidelity=fid, acc_fp32=acc_fp32)
        logger_cfg = {"cache": CACHE, "attn_fidelity": attn_fidelity, "acc_fp32": acc_fp32}
        print(f"handoff config {logger_cfg}", flush=True)
        print(f"resident engines built: {time.time() - t0:.0f} s", flush=True)
        self.tok = self.gen.tokenizer

    def prefill(self, ids):
        logits = self.gen.prefill_forward(
            torch.tensor([ids]), page_table=self.state.page_tables, kv_cache=self.state, prompt_lens=[len(ids)]
        )
        return logits[0, -1].float()

    def kv_history(self, l, pos):
        """(K, V) [kv heads of all chips, n, D] bf16 at the positions row pos attends to before itself"""
        full = l % 6 == 5
        kind = "full_attention" if full else "sliding_attention"
        tab = self.attn.tables[kind]
        ps = torch.arange(0 if full else max(0, pos - 1023), pos)
        pages = tab[ps // 128] if full else tab[(ps % 1152) // 128]
        out = []
        for t in self.state.kv[l]:
            chips = per_chip(t)
            out.append(torch.cat([c[pages, :, ps % 128].transpose(0, 1) for c in chips]).bfloat16())
        return out

    def kv_row(self, l, pos):
        """per chip (K, V) rows [H, D] at position pos of layer l's cache"""
        full = l % 6 == 5
        kind = "full_attention" if full else "sliding_attention"
        tab = self.attn.tables[kind]
        page = int(tab[pos // 128]) if full else int(tab[(pos % 1152) // 128])
        out = []
        for t in self.state.kv[l]:
            sl = ttnn.slice(t, [page, 0, 0, 0], [page + 1, t.shape[1], 128, t.shape[3]])
            out.append(torch.stack([x[0, :, pos % 128].float() for x in per_chip(sl)]))
            ttnn.deallocate(sl)
        return out

    def ttnn_layer(self, l, x, pos):
        """the ttnn decoder layer l on x [HIDDEN] (row 0) at position pos -> [HIDDEN]"""
        layer = self.gen.model.layers[l]
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        xt = ttnn.from_torch(x.reshape(1, 1, 1, -1).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=self.mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xt = ttnn.to_memory_config(xt, layer._mem(5376, 28))
        positions = self.gen._upload(torch.tensor([pos], dtype=torch.int32))
        y = layer.decode_forward(xt, positions=positions, page_table=self.state.device_tables[layer.kind],
                                 kv_cache=self.state.kv[l], rope_positions=positions)
        return per_chip(y)[0].reshape(-1, 5376)[0]


def cmd_handoff(a):
    """ttnn prefill -> resident decode: teacher-forced per-layer checks at each context, M = 8 vs 8 x M = 1, greedy
    end-to-end against the ttnn B = 1 decode"""
    from models.demos.gemma4_31b_qb2.resident_m3_2.reference_s2 import pcc

    ctxs = [int(c) for c in a.ctxs.split(",")]
    mesh = open_mesh()
    res = {"tf": {}, "m8": None, "e2e": {}}
    hf_data = {}
    save = lambda: json.dump(res, open(a.out, "w"), indent=1)  # noqa: E731
    try:
        h = Handoff(mesh, max(ctxs) + 512, a.attn_fidelity, bool(a.acc_fp32))
        res["config"] = {"cache": CACHE, "attn_fidelity": a.attn_fidelity, "acc_fp32": bool(a.acc_fp32)}
        m1, m8 = h.m1, h.m8
        m1.reset()
        m8.reset()
        for ctx in ctxs if "tf" in a.parts else []:
            ids, nxt = context_ids(h.tok, ctx)
            t0 = time.time()
            logits = h.prefill(ids)
            tok = int(logits.argmax())
            m1.set_rows([tok], ctx)
            m1.launch()
            ttnn.synchronize_device(mesh)
            taps, out = m1.taps(), m1.out()
            o_dev, h_dev = m1.attn_out(), m1.round_h()
            rows = []
            x = m1.emb_host[tok].float()
            rope_rows = {
                k: tuple(per_chip(ttnn.slice(t, [ctx, 0], [ctx + 1, t.shape[1]]))[0][0].float() for t in pair)
                for k, pair in h.attn.rope.items()
            }
            hist = []
            for l in range(60):
                hist.append(h.kv_history(l, ctx) if a.hf else None)
                kv_res = h.kv_row(l, ctx)
                y = h.ttnn_layer(l, x, ctx)
                kv_tt = h.kv_row(l, ctx)
                xr = taps[l][0, 0]
                if a.hf:
                    D = 512 if l % 6 == 5 else 256
                    o_l = torch.cat([o_dev[c, l, 0, : 8 * D] for c in range(4)])
                    hist[l] = (x.clone(), xr.clone(), y.clone()) + tuple(hist[l]) + (o_l, h_dev[0, 0, 2 * l + 1])
                rows.append({
                    "layer": l,
                    "x_pcc": pcc(xr, y),
                    "x_maxabs": float((xr - y).abs().max()),
                    "x_norm": float(y.norm()),
                    "k_pcc": pcc(kv_res[0], kv_tt[0]),
                    "v_pcc": pcc(kv_res[1], kv_tt[1]),
                })
                x = xr
            worst = min(rows, key=lambda r: r["x_pcc"])
            if a.hf:
                hf_data[ctx] = (hist, rope_rows)
            res["tf"][ctx] = {"prefill_token": tok, "resident_next": out[0][0], "layers": rows,
                              "min_x_pcc": worst["x_pcc"], "min_k_pcc": min(r["k_pcc"] for r in rows),
                              "min_v_pcc": min(r["v_pcc"] for r in rows), "seconds": round(time.time() - t0)}
            print(f"tf ctx {ctx}: min x pcc {worst['x_pcc']:.5f} (layer {worst['layer']}), "
                  f"min k {res['tf'][ctx]['min_k_pcc']:.5f}, v {res['tf'][ctx]['min_v_pcc']:.5f}", flush=True)
            save()
        # M = 8 rows vs 8 sequential M = 1 steps (teacher forced, the same tokens) on the shortest context
        if "m8" in a.parts:
            ctx = ctxs[0]
            ids, nxt = context_ids(h.tok, ctx)
            h.prefill(ids)
            toks = nxt[:8]
            m8.set_rows(toks, ctx)
            m8.launch()
            ttnn.synchronize_device(mesh)
            l8, t8, f8 = m8.logits(), m8.out()[0][0], m8.final_h()[0]
            l1s, t1s, f1s = [], [], []
            for r in range(8):
                m1.set_rows([toks[r]], ctx + r)
                m1.launch()
                ttnn.synchronize_device(mesh)
                l1s.append(m1.logits()[0])
                t1s.append(m1.out()[0][0][0])
                f1s.append(m1.final_h()[0, 0])
            res["m8"] = {
                "ctx": ctx,
                "tokens_in": toks,
                "argmax_m8": t8,
                "argmax_m1": t1s,
                "logits_pcc": [pcc(l8[r], l1s[r]) for r in range(8)],
                "final_h_pcc": [pcc(f8[r], f1s[r]) for r in range(8)],
                "logits_maxabs": [float((l8[r] - l1s[r]).abs().max()) for r in range(8)],
            }
            print(f"m8: argmax equal {t8 == t1s}, logits pcc min {min(res['m8']['logits_pcc']):.5f}", flush=True)
            save()
        # greedy end to end
        for name, text in PROMPTS.items() if "e2e" in a.parts else []:
            ids = chat_ids(h.tok, text)
            tops = []
            orig = h.gen._host_logits

            def rec(logits, orig=orig):
                out = orig(logits)
                v, i = out.reshape(-1, out.shape[-1])[0].topk(2)
                tops.append((int(i[0]), int(i[1]), float(v[0] - v[1]), float(v[0])))
                return out

            h.gen._host_logits = rec
            t0 = time.time()
            ref = h.gen.generate(ids, a.tokens, top_k=1, stop_on_eos=False, host_sampling=True)
            h.gen._host_logits = orig
            t_tt = time.time() - t0
            first = int(h.prefill(ids).argmax())
            m1.set_rows([first], len(ids))
            got, rtops = [first], []
            t0 = time.time()
            for _ in range(a.tokens - 1):
                m1.launch()
                ttnn.synchronize_device(mesh)
                # the final soft-cap (monotonic, so the device argmax skips it), for gaps on ttnn's and HF's scale
                lg = 30.0 * torch.tanh(m1.logits()[0] / 30.0)
                v, i = lg.topk(2)
                rtops.append((int(i[0]), int(i[1]), float(v[0] - v[1]), float(v[0])))
                got.append(m1.out()[0][0][0])
            t_res = time.time() - t0
            div = next((k for k in range(min(len(ref), len(got))) if ref[k] != got[k]), None)
            row = {"prompt_tokens": len(ids), "match": div is None, "first_divergence": div,
                   "matched_tokens": div if div is not None else len(ref), "ttnn": ref, "resident": got,
                   "seconds": {"ttnn": round(t_tt, 1), "resident": round(t_res, 1)}}
            if div is not None:
                row["ttnn_top2_at_div"] = tops[div] if div < len(tops) else None
                row["resident_top2_at_div"] = rtops[div - 1] if div >= 1 else None
                row["text_common"] = h.tok.decode(ref[:div])
            row["text_ttnn"] = h.tok.decode(ref)
            row["text_resident"] = h.tok.decode(got)
            res["e2e"][name] = row
            print(f"e2e {name}: {len(ids)} prompt tokens, match {row['match']}, first divergence {div}, "
                  f"ttnn top2 {row.get('ttnn_top2_at_div')}, resident top2 {row.get('resident_top2_at_div')}", flush=True)
            save()
    finally:
        close_mesh(mesh)
    if hf_data:
        # both engines against the unquantized model, per layer from the same inputs (the resident's x_l)
        from models.demos.gemma4_31b_qb2.resident_m3_2.reference_s2 import HFLayer
        from models.demos.gemma4_31b_qb2.resident_m3_2.weights import Checkpoint

        ck = Checkpoint()
        for l in range(60):
            layer = HFLayer(ck, l)
            kind = "full_attention" if l % 6 == 5 else "sliding_attention"
            for ctx, (hist, rope_rows) in hf_data.items():
                x, xr, y, kh, vh, o_l, h_mid = hist[l]
                parts = {}
                ref = layer(x, kh, vh, *rope_rows[kind], parts=parts)
                row = res["tf"][ctx]["layers"][l]
                row["hf_pcc_resident"] = pcc(xr, ref)
                row["hf_pcc_ttnn"] = pcc(y, ref)
                D = 512 if l % 6 == 5 else 256
                oh, orf = o_l.reshape(32, D), parts["o"].reshape(32, D)
                row["hf_o_pcc"] = pcc(o_l, parts["o"])
                row["hf_o_head_pcc_min"] = min(pcc(oh[i], orf[i]) for i in range(32))
                from models.demos.gemma4_31b_qb2.resident_m3_2.reference import rms as _rms

                row["hf_hmid_pcc"] = pcc(h_mid, _rms(parts["x_mid"]))
            print(f"hf layer {l}: " + ", ".join(
                f"ctx {c}: res {res['tf'][c]['layers'][l]['hf_pcc_resident']:.5f} "
                f"ttnn {res['tf'][c]['layers'][l]['hf_pcc_ttnn']:.5f} o {res['tf'][c]['layers'][l]['hf_o_pcc']:.5f} "
                f"(head min {res['tf'][c]['layers'][l]['hf_o_head_pcc_min']:.4f}) "
                f"h_mid {res['tf'][c]['layers'][l]['hf_hmid_pcc']:.5f}" for c in hf_data), flush=True)
        for ctx in hf_data:
            L = res["tf"][ctx]["layers"]
            res["tf"][ctx]["min_hf_pcc_resident"] = min(r["hf_pcc_resident"] for r in L)
            res["tf"][ctx]["min_hf_pcc_ttnn"] = min(r["hf_pcc_ttnn"] for r in L)
        save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--layers", type=int, default=60)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--out", default=None)
    p.add_argument("--detail", type=int, default=0)
    p.add_argument("--ctxs", default="128,1024,8192,32768")
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--attn-fidelity", default="HiFi2")
    p.add_argument("--fidelity", default="HiFi2")
    p.add_argument("--acc-fp32", type=int, default=1)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--timeline", type=int, default=0)
    p.add_argument("--hf", type=int, default=1)
    p.add_argument("--parts", default="tf,m8,e2e")
    a = p.parse_args()
    {"mock": cmd_mock, "unit": cmd_unit, "perf": cmd_perf, "handoff": cmd_handoff}[a.cmd](a)


if __name__ == "__main__":
    main()
