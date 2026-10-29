# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bring-up checks of the resident decode on a 1 x 4 mesh (P300x2 / QB2).

random: a few layers with random weights at batch 1, 2, 4, 8 against torch_step for several steps
    (the per-user x after every step: relative L2 error and PCC).
real:   the Qwen3.8 checkpoint, teacher-forced against a Hugging Face reference (hf_teacher_ref.py):
    top-1 / top-5 agreement of the device logits, then the traced greedy decode TPOT and the
    pipeline-parallel prefill TTFT at a few prompt lengths.

usage: python -m qwen36_resident.tests.e0_bringup random [--layers 4] [--steps 4] [--batches 1,2,4,8]
       python -m qwen36_resident.tests.e0_bringup real --ref hf_ref_qwen38.pt [--max-len 32768]
"""
import argparse
import copy
import faulthandler
import sys
import json
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State, is_attn, random_weights, torch_step
from qwen36_resident.prefill.pp_prefill import PPPrefill

PREFILL_CHUNKS = (128, 256, 512, 1024)


def open_mesh(trace_region=134217728):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_region)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


def run_random(mesh, layers, steps, batches, interval=4, max_pos=256, start_pos=37):
    d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x)
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    n_gdn = layers - n_attn
    w = random_weights(d, n_gdn, n_attn, layers)
    rows = []
    for B in batches:
        sts = [State(d, n_gdn, n_attn, start_pos + 3 * u, max_pos, seed=11 + u) for u in range(B)]
        ref = copy.deepcopy(sts)
        t0 = time.time()
        model = ResidentModel(mesh, d, w, sts if B > 1 else sts[0], layers, interval)
        build_s = time.time() - t0
        g = torch.Generator().manual_seed(100 + B)
        worst = (0.0, 1.0)
        for s in range(steps):
            x0 = torch.randn(B, 5120, generator=g)
            pos = [r.pos for r in ref]
            tok = model.token_state(x0[0] if B == 1 else x0, pos[0] if B == 1 else pos, ring=None if B == 1 else s)
            model.step(tok)
            ttnn.synchronize_device(mesh)
            got = model.x().reshape(B, -1)
            for u in range(B):
                want = torch_step(x0[u], d, w, ref[u], layers, interval)
                e, p = rel(got[u], want), pcc(got[u], want)
                worst = (max(worst[0], e), min(worst[1], p))
                rows.append(dict(B=B, step=s, user=u, rel=e, pcc=p))
        print(
            f"random B={B} G={model.head_groups} layers={layers} steps={steps}: worst rel {worst[0]:.4f} "
            f"pcc {worst[1]:.5f} (build {build_s:.1f} s)",
            flush=True,
        )
        del model
    return rows


def run_real(mesh, ref_path, max_seq_len, ttft_lens, tpot_steps):
    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    cfg = ck.config
    layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
    n = mesh.get_num_devices()
    d = Dims(n, mesh.dram_grid_size().x, vocab=vocab)
    emb = Q.embedding(ck)
    C = max(PREFILL_CHUNKS)
    max_len = -(-max_seq_len // C) * C
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    t0 = time.time()

    def phase(name):
        print(f"[{time.time() - t0:7.1f} s] {name}", flush=True)

    st = State(d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True, kv=False)
    phase("state")
    pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
    phase("prefill model")
    w = Q.load(ck, d, layers, interval)
    phase("decode weights (host)")
    model = ResidentModel(mesh, d, w, st, layers, interval, lm_head=True, embed=pp.embed)
    del w
    phase("decode model")
    model.step(model.token_state(emb[0].float(), 0))
    ttnn.synchronize_device(mesh)
    phase("first step")
    model.reset()
    model.capture_trace()
    phase("decode trace")
    pp.capture()
    phase("prefill trace")
    pp.handoff(model, 1)
    ttnn.synchronize_device(mesh)
    phase("handoff")
    print(f"real: built in {time.time() - t0:.0f} s, grid G={model.head_groups}", flush=True)

    def logits_at(token, pos):
        model.step(model.token_state(emb[int(token)].float(), int(pos)))
        ttnn.synchronize_device(mesh)
        return model.logits()[:, : d.vocab_chip].reshape(-1)[:vocab].float()

    def prefill(ids):
        model.reset()
        pp.reset()
        T = len(ids)
        if T > 1:
            pp.run(ids[: T - 1])
            pp.handoff(model, T - 1)
        return logits_at(ids[T - 1], T - 1)

    out = dict(teacher=[], mismatch=[], tpot={}, ttft={})
    ref = torch.load(ref_path)
    P, N = ref["prefix"], ref["scored"]
    top1 = top5 = total = 0
    for i, s in enumerate(ref["seqs"]):
        ids = s["ids"]
        lg = prefill(ids[:P])
        h1 = h5 = 0
        for k in range(N):
            p = P - 1 + k
            if k:
                lg = logits_at(ids[p], p)
            want = s["top_i"][p]
            dev1 = int(lg.argmax())
            h1 += dev1 == int(want[0])
            h5 += int(want[0]) in lg.topk(5).indices.tolist()
            if dev1 != int(want[0]):
                # HF's margin between its top-1 and the device's pick (inf when the pick is outside HF's top 20)
                hit = (s["top_i"][p] == dev1).nonzero()
                hf_gap = float(s["top_v"][p, 0] - s["top_v"][p, hit[0, 0]]) if len(hit) else float("inf")
                out["mismatch"].append(
                    dict(
                        seq=i,
                        pos=p,
                        dev=dev1,
                        hf=int(want[0]),
                        hf_gap=hf_gap,
                        dev_gap=float(lg[dev1] - lg[int(want[0])]),
                    )
                )
        top1, top5, total = top1 + h1, top5 + h5, total + N
        out["teacher"].append(dict(seq=i, top1=h1 / N, top5=h5 / N))
        print(f"teacher seq {i}: top-1 {h1 / N:.4f} top-5 {h5 / N:.4f}", flush=True)
    out["top1"], out["top5"], out["scored"] = top1 / total, top5 / total, total
    print(f"teacher all {total}: top-1 {top1 / total:.4f} top-5 {top5 / total:.4f}", flush=True)

    # greedy traced decode with the hub feeding itself, and prefill TTFT
    ids = ref["seqs"][0]["ids"].tolist()
    for T in ttft_lens:
        prompt = torch.tensor((ids * (T // len(ids) + 1))[:T])
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        prefill(prompt)
        ttft = time.time() - t0
        model.set_sampling(0.0, 1, 1.0, 0)
        model.step(model.token_state(emb[int(prompt[-1])].float(), T - 1))
        for _ in range(8):
            model.step(None)
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(tpot_steps):
            model.step(None)
        ttnn.synchronize_device(mesh)
        tpot = (time.time() - t0) / tpot_steps * 1e3
        out["ttft"][T], out["tpot"][T] = ttft, tpot
        print(f"T={T}: TTFT {ttft:.3f} s, TPOT {tpot:.2f} ms ({1e3 / tpot:.1f} tok/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["random", "real"])
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--ref")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--ttft", default="128,2048,8192")
    ap.add_argument("--tpot-steps", type=int, default=128)
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    mesh = open_mesh()
    try:
        if a.mode == "random":
            res = run_random(mesh, a.layers, a.steps, [int(b) for b in a.batches.split(",")])
        else:
            res = run_real(mesh, a.ref, a.max_len, [int(t) for t in a.ttft.split(",")], a.tpot_steps)
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
