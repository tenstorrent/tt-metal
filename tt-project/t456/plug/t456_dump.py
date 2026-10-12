"""t456: dump one real SLA layer's vsa_sdpa inputs and output (device 0, first untraced call at the 75776 rung), then stop.

Saves q, gathered k/v (first T456_HEADS heads), raw ids, block counts, output and the scalar kwargs to
$T456_DUMP/layer0.pt and raises so pytest tears the mesh down. Calls whose local q rows are not
T456_ROWS (the 76800 warm rung) are passed through.
"""
import math
import os

import torch
import ttnn

_orig = ttnn.transformer.vsa_sdpa
_state = {"n": 0}


def _dev0(x):
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


def _wrapped(q, k, v, ids, counts, *args, **kwargs):
    out = _orig(q, k, v, ids, counts, *args, **kwargs)
    rows = int(os.environ.get("T456_ROWS", "9472"))
    if q.shape[2] != rows:
        _state["n"] += 1
        return out
    if os.environ.get("T456_CHECK"):
        _check_all(q, k, v, out)
        raise RuntimeError("T456_DUMP_DONE")
    heads = int(os.environ.get("T456_HEADS", "4"))
    d = {
        "q": _dev0(q)[:, :heads].to(torch.bfloat16),
        "k": _dev0(k)[:, :heads].to(torch.bfloat16),
        "v": _dev0(v)[:, :heads].to(torch.bfloat16),
        "ids": _dev0(ids)[:, :heads],
        "counts": _dev0(counts),
        "out": _dev0(out)[:, :heads].to(torch.bfloat16),
        "kwargs": {a: b for a, b in kwargs.items() if isinstance(b, (int, float, bool, str, type(None)))},
        "shapes": {n: list(t.shape) for n, t in (("q", q), ("k", k), ("ids", ids), ("counts", counts), ("out", out))},
        "skipped_calls": _state["n"],
    }
    path = os.path.join(os.environ["T456_DUMP"], "layer0.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(d, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[t456] dumped {path}: {d['shapes']} kwargs {d['kwargs']} skipped {_state['n']}", flush=True)
    raise RuntimeError("T456_DUMP_DONE")


def _check_all(q, k, v, out):
    """Sparsity-0 check on every device: head 0, first/middle/last q tile vs dense attention over the
    T456_SEQ real keys and over all gathered keys (pad leak). Prints one line per device."""
    seq = int(os.environ.get("T456_SEQ", "75449"))
    qs, ks, vs, os_ = (ttnn.get_device_tensors(x) for x in (q, k, v, out))
    pcc = lambda a, b: torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    for i in range(len(qs)):
        qd, kd, vd, od = (ttnn.to_torch(x)[0, 0].float() for x in (qs[i], ks[i], vs[i], os_[i]))
        res = []
        for t in (0, qd.shape[0] // 256, qd.shape[0] // 128 - 1):
            r = slice(t * 128, (t + 1) * 128)
            if float(qd[r].abs().max()) == 0:
                res.append(f"t{t}:zero-q")
                continue
            s = qd[r] @ kd.T / math.sqrt(qd.shape[-1])
            real = torch.softmax(s[:, :seq], -1) @ vd[:seq]
            full = torch.softmax(s, -1) @ vd
            res.append(f"t{t}:real {pcc(od[r], real):.4f} all {pcc(od[r], full):.4f}")
        print(
            f"[t456] dev {i} q{list(qd.shape)} k{list(kd.shape)} kpad|max| {float(kd[seq:].abs().max()):.2f} "
            + " ".join(res),
            flush=True,
        )


ttnn.transformer.vsa_sdpa = _wrapped
