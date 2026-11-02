# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BONSAI_PF_TIME=1 (debug): synchronized wall time per prefill projection call, by site and row count.

Only the first two prefill passes are timed (LIMIT calls per site). Those are eager in text_demo:
_warmup_prefill (traced_4k: the 2048 chunk + the tail) or the eager chunk + masked-bucket warmups
(traced_128). That keeps the synchronize out of any trace capture. Medians print at exit.
"""

from __future__ import annotations

import atexit
import os
import time

import ttnn

ON = bool(os.environ.get("BONSAI_PF_TIME"))
LIMIT = {"gdn_in": 96, "gdn_out": 96, "attn_qkv": 32, "attn_o": 32, "mlp": 128}  # 2 passes x layers of that kind
_t: dict = {}
_n: dict = {}


def timed(site: str, rows: int, fn, *args, **kwargs):
    if not ON or rows <= 32 or _n.get(site, 0) >= LIMIT[site]:
        return fn(*args, **kwargs)
    dev = args[0].device()  # x is always the first argument
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    _t.setdefault((site, rows), []).append((time.perf_counter() - t0) * 1e3)
    _n[site] = _n.get(site, 0) + 1
    return out


@atexit.register
def _report():
    native = os.environ.get("BONSAI_NATIVE", "")
    for (site, rows), v in sorted(_t.items()):
        v = sorted(v)
        print(f"PFTIME native={native or 'dense'} {site} S={rows} n={len(v)} median {v[len(v) // 2]:.3f} ms "
              f"sum {sum(v):.1f} ms", flush=True)
