"""Per-layer teacher-forced gate for layers beyond L5: run the CPU oracle (fp32 and bf16 weights) on the DEVICE h_in of each layer and compare with the
device captures (attn_out, ffn_out, h_out, router sets), next to the oracle's own bf16-vs-fp32 floor on the same input.

    python -m ...probes.device_vs_oracle <device_dir> <out.jsonl> [--layers 6-47]

Per layer row (jsonl): type (swa/global); for t in attn_out, ffn_out, h_out: pcc and rel L2 of device vs oracle_fp32, device vs oracle_bf16, and the floor
oracle_bf16 vs oracle_fp32; router: frac of tokens with identical 8-expert set device vs oracle_fp32 (oracle router on oracle ffn_in), the same oracle_bf16 vs
oracle_fp32 (floor), device vs router(device ffn_in) in fp32 (isolates the device router op), slot overlap, and near-tie share of flips (fp32 oracle route_gap <
tau for tau 0.01/0.02); weight rel error on identical sets.
"""
import argparse
import json
import os
import time

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch

from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.reference import ref_layer as R
from models.demos.blackhole.mimo_v26.reference.ref_prefill_dump import parse_layers, pcc

torch.set_num_threads(32)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def cmp(a, b):
    return {"pcc": pcc(a, b), "rel": rel(a, b)}


def sets_equal(i1, i2):
    return (i1.sort(-1).values == i2.sort(-1).values).all(-1)


def overlap(i1, i2):
    return (sum((i1[:, k : k + 1] == i2).any(-1).float() for k in range(i1.shape[1])) / i1.shape[1]).mean().item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev")
    ap.add_argument("out")
    ap.add_argument("--layers", default="6-47")
    a = ap.parse_args()
    ck = Ld.Ckpt()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["L"] for l in open(a.out)}
    for L in parse_layers(a.layers):
        if L in done:
            continue
        t0 = time.time()
        d = torch.load(f"{a.dev}/layer_{L}.pt")["prefill"]
        sd32 = Ld.load_layer(L, torch.float32, ck)
        sd16 = {
            k: (v.to(torch.bfloat16) if v.is_floating_point() and "e_score" not in k else v) for k, v in sd32.items()
        }
        h = d["h_in"]
        r32 = R.layer_forward(h.float(), sd32, L)
        r16 = R.layer_forward(h.to(torch.bfloat16), sd16, L)
        row = {"L": L, "type": "swa" if R.is_swa(L) else "global"}
        for t in ("attn_out", "ffn_out", "h_out"):
            row[t] = {
                "dev_vs_fp32": cmp(d[t], r32[t]),
                "dev_vs_bf16": cmp(d[t], r16[t]),
                "floor_bf16_vs_fp32": cmp(r16[t], r32[t]),
            }
        di, dw = d["routing_idx"], d["routing_w"].float()
        f32i, f32w, gap = r32["routing_idx"], r32["routing_w"], r32["route_gap"]
        rt = {}
        eq = sets_equal(di, f32i)
        rt["set_equal_dev_vs_fp32"] = eq.float().mean().item()
        rt["set_equal_dev_vs_bf16"] = sets_equal(di, r16["routing_idx"]).float().mean().item()
        rt["set_equal_floor_bf16_vs_fp32"] = sets_equal(r16["routing_idx"], f32i).float().mean().item()
        rt["slot_overlap_dev_vs_fp32"] = overlap(di, f32i)
        rt["slot_overlap_floor"] = overlap(r16["routing_idx"], f32i)
        idx_d, _, gap_d = R.router(
            d["ffn_in"].reshape(-1, 4096).float(), sd32
        )  # oracle router (fp32) on the DEVICE ffn_in
        rt["set_equal_dev_vs_router_on_dev_ffn_in"] = sets_equal(di, idx_d).float().mean().item()
        flips = ~eq
        for tau in (0.01, 0.02):
            rt[f"near_tie_share_of_flips_tau{tau}"] = (gap[flips] < tau).float().mean().item() if flips.any() else None
            rt[f"frac_tokens_gap_lt_{tau}"] = (gap < tau).float().mean().item()
        # weights on identical sets
        o_d, o_f = di.argsort(-1), f32i.argsort(-1)
        wd, wf = dw.gather(1, o_d), f32w.gather(1, o_f)
        rt["w_rel_err_on_equal_sets"] = rel(wd[eq], wf[eq]) if eq.any() else None
        row["router"] = rt
        row["secs"] = round(time.time() - t0)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            L,
            row["type"],
            "h_out dev/fp32 pcc %.6f rel %.4f | floor rel %.4f | set_eq dev %.3f floor %.3f"
            % (
                row["h_out"]["dev_vs_fp32"]["pcc"],
                row["h_out"]["dev_vs_fp32"]["rel"],
                row["h_out"]["floor_bf16_vs_fp32"]["rel"],
                rt["set_equal_dev_vs_fp32"],
                rt["set_equal_floor_bf16_vs_fp32"],
            ),
            flush=True,
        )
        del sd32, sd16, r32, r16


if __name__ == "__main__":
    main()
