"""t457: per-tile numerics sweep of the real vsa_sdpa calls (sparsity 0), then stop.

At the 9472-row vsa_sdpa calls with indices in T457_AT (default "0,49": layer 0 and layer 49 of step 1), for
devices T457_DEVS (default 0,24) and heads 0..T457_HEADS-1 (default 14), every 128-row q tile:
  vsa  = PCC(vsa_sdpa out, fp32 softmax over the T456_SEQ real keys)
  sdpa = PCC(plain bf16 ttnn SDPA on the same q/k/v, fp32 softmax over all gathered keys)
The sdpa column is the bf16 floor of the dense kernel family (HiFi2, fp32_dest_acc off, exact exp, as the dense
ring path) on the same data. Prints per-head min/median and the 12 worst vsa tiles with their sdpa value, then
raises T457_SWEEP_DONE after the last index.
"""
import math
import os
import time

import torch
import ttnn

_orig = ttnn.transformer.vsa_sdpa
_state = {"n": 0}
torch.set_num_threads(int(os.environ.get("T457_THREADS", "16")))


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


def _dense_sdpa(q, k, v):
    dev = q.device()
    g = dev.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y),
        q_chunk_size=256,
        k_chunk_size=512,
        exp_approx_mode=False,
    )
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    return ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, program_config=pc, compute_kernel_config=ck
    )


def _sweep(call, q, k, v, out, kwargs):
    seq = int(os.environ.get("T456_SEQ", "75449"))
    devs = [int(x) for x in os.environ.get("T457_DEVS", "0,24").split(",")]
    heads = int(os.environ.get("T457_HEADS", "14"))
    t0 = time.time()
    try:
        dense = _dense_sdpa(q, k, v)
        dts = ttnn.get_device_tensors(dense)
    except Exception as e:  # keep the vsa sweep even if the comparator cannot build here
        print(f"[t457] call {call}: dense sdpa failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
        dts = None
    qs, ks, vs, os_ = (ttnn.get_device_tensors(x) for x in (q, k, v, out))
    print(f"[t457] call {call}: kwargs {kwargs} skipped {_state['n']} prep {time.time() - t0:.1f}s", flush=True)
    worst = []
    for d in devs:
        qd, kd, vd, od = (ttnn.to_torch(x)[0].float() for x in (qs[d], ks[d], vs[d], os_[d]))
        dd = ttnn.to_torch(dts[d])[0].float() if dts is not None else None
        scale = 1.0 / math.sqrt(qd.shape[-1])
        for h in range(min(heads, qd.shape[0])):
            th = time.time()
            vsa_p, sd_p = [], []
            for c0 in range(0, qd.shape[1], 1024):
                qc = qd[h, c0 : c0 + 1024]
                s = (qc @ kd[h].T) * scale
                e = torch.exp(s - s.max(-1, keepdim=True).values)
                del s
                num_r = e[:, :seq] @ vd[h, :seq]
                sum_r = e[:, :seq].sum(-1, keepdim=True)
                num_a = num_r + e[:, seq:] @ vd[h, seq:]
                sum_a = sum_r + e[:, seq:].sum(-1, keepdim=True)
                del e
                ref_r, ref_a = num_r / sum_r, num_a / sum_a
                for t in range(qc.shape[0] // 128):
                    r = slice(t * 128, (t + 1) * 128)
                    tile = (c0 + t * 128) // 128
                    if float(qc[r].abs().max()) == 0:
                        continue
                    o = od[h, c0 + t * 128 : c0 + (t + 1) * 128]
                    pv = _pcc(o, ref_r[r])
                    rv = _rel(o, ref_r[r])
                    ps = rs = float("nan")
                    if dd is not None:
                        ds = dd[h, c0 + t * 128 : c0 + (t + 1) * 128]
                        ps, rs = _pcc(ds, ref_a[r]), _rel(ds, ref_a[r])
                    vsa_p.append(pv)
                    sd_p.append(ps)
                    worst.append((pv, d, h, tile, rv, ps, rs))
            vp = torch.tensor(vsa_p)
            sp = torch.tensor(sd_p)
            print(
                f"[t457] call {call} dev {d} h{h}: tiles {len(vsa_p)} vsa min {vp.min():.5f} med {vp.median():.5f} "
                f"sdpa min {sp.min():.5f} med {sp.median():.5f} ({time.time() - th:.1f}s)",
                flush=True,
            )
    worst.sort()
    for pv, d, h, tile, rv, ps, rs in worst[:12]:
        print(
            f"[t457] call {call} worst: dev {d} h{h} tile {tile}: vsa pcc {pv:.5f} rel {rv:.4f} | "
            f"sdpa pcc {ps:.5f} rel {rs:.4f}",
            flush=True,
        )
    allv = torch.tensor([w[0] for w in worst])
    alls = torch.tensor([w[5] for w in worst])
    print(
        f"[t457] call {call} summary: {len(worst)} tiles, vsa<0.999: {int((allv < 0.999).sum())}, "
        f"sdpa<0.999: {int((alls < 0.999).sum())}, vsa mean {allv.mean():.5f} sdpa mean {alls.nanmean():.5f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    if dts is not None:
        ttnn.deallocate(dense)


def _wrapped(q, k, v, ids, counts, *args, **kwargs):
    out = _orig(q, k, v, ids, counts, *args, **kwargs)
    rows = int(os.environ.get("T456_ROWS", "9472"))
    if q.shape[2] != rows:
        _state["n"] += 1
        return out
    at = [int(x) for x in os.environ.get("T457_AT", "0,49").split(",")]
    call = _state.setdefault("rows", 0)
    _state["rows"] = call + 1
    if call in at:
        kw = {a: b for a, b in kwargs.items() if isinstance(b, (int, float, bool, str, type(None)))}
        _sweep(call, q, k, v, out, kw)
        if call == max(at):
            raise RuntimeError("T457_SWEEP_DONE")
    return out


ttnn.transformer.vsa_sdpa = _wrapped
