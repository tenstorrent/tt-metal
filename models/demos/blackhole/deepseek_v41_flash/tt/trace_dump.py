# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_PF_LAYER_DUMP=1 (diagnostics): named intermediate tensors of the traced prefill chunk are copied into PERSISTENT buffers (cloned in the first eager pass, before any capture; copied
in place in every later pass / replay) so that a host can read them after each replay. ``report`` prints one TRACESUM line per buffer (sum / abs sum / nan count of device 0 of every mesh row)."""

import os

import ttnn

ON = os.environ.get("DSV41_PF_LAYER_DUMP") == "1"
D = {}
_CNT = [0]


def begin():
    _CNT[0] = 0


def put(tag, t):
    """record ``t`` as '<running index>.<tag>' (the running index counts the put() calls of one forward: the same in every pass)"""
    if not ON:
        return
    name = f"{_CNT[0]:03d}.{tag}"
    _CNT[0] += 1
    if name not in D:
        D[name] = ttnn.clone(t)
    else:
        ttnn.copy(t, D[name])


def report(md, rows, cols, win):
    if not ON or not D:
        return
    ttnn.synchronize_device(md)
    for name, t in sorted(D.items()):
        devs = ttnn.get_device_tensors(t)
        sums = []
        for r in range(rows):
            x = ttnn.to_torch(devs[r * cols]).double()
            sums.append(f"{x.sum().item():.9e}/{x.abs().sum().item():.9e}/{int(x.isnan().sum())}")
        print(f"TRACESUM {win} {name} " + " ".join(sums), flush=True)


_KEEP = [False]


def _walk(o, path, out, depth, seen):
    if depth > 4 or id(o) in seen:
        return
    if isinstance(o, ttnn.Tensor):
        out[path] = o if _KEEP[0] else o.buffer_address()
        return
    seen.add(id(o))
    if isinstance(o, dict):
        for k, v in o.items():
            _walk(v, f"{path}[{k}]", out, depth + 1, seen)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            _walk(v, f"{path}[{i}]", out, depth + 1, seen)
    elif hasattr(o, "__dict__") and not isinstance(o, type):
        for k, v in list(vars(o).items()):
            if k in ("md", "mesh_device", "dec", "a", "head", "mesh_config", "ccl"):
                continue
            _walk(v, f"{path}.{k}", out, depth + 1, seen)


def addr_snapshot(roots):
    """{path: device buffer address} of every tensor reachable (depth <= 4) from the named roots {name: object}"""
    out, seen = {}, set()
    for n, o in roots.items():
        _walk(o, n, out, 0, seen)
    return out


def tensor_map(roots):
    _KEEP[0] = True
    try:
        return addr_snapshot(roots)
    finally:
        _KEEP[0] = False
