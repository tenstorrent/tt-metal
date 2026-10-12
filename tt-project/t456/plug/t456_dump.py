"""t456: dump one real SLA layer's vsa_sdpa inputs and output (device 0, first untraced call at the 75776 rung), then stop.

Saves q, gathered k/v (first T456_HEADS heads), raw ids, block counts, output and the scalar kwargs to
$T456_DUMP/layer0.pt and raises so pytest tears the mesh down. Calls whose local q rows are not
T456_ROWS (the 76800 warm rung) are passed through.
"""
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


ttnn.transformer.vsa_sdpa = _wrapped
