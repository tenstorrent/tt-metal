# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device time of each op of one CG iteration against its speed of light (reference/speed_of_light.py).

usage: python models/experimental/hpcg/demo/op_perf.py nx
Each op runs 10 times back to back in a trace, best of 3 replays.
"""

import json
import sys
import time

import ttnn

from models.experimental.hpcg.reference import speed_of_light
from models.experimental.hpcg.tt.hpcg import HPCG


def main(nx):
    device = ttnn.open_device(device_id=0, trace_region_size=512 << 20)

    def timed(fn, k=10, reps=3):
        fn()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(k):
            fn()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            best = min(best, (time.perf_counter() - t0) / k)
        ttnn.release_trace(device, tid)
        return best * 1e6

    S = {o["op"].replace(" ", ""): o["sol_us"] for o in speed_of_light.ops(nx)}
    try:
        h = HPCG(device, nx)
        h.reset()
        h.start()
        res = {}
        for l in range(len(h.lv)):
            res[f"gs_pre_L{l}"] = (
                timed(h.ops[("gs_pre", l)]),
                S[f"SYMGSpreL{l}"] + S[f"SpMVresidL{l}"] if l < 3 else S["SYMGSL3"],
            )
            if l < 3:
                res[f"gs_post_L{l}"] = (timed(h.ops[("gs", l)]), S[f"SYMGSpostL{l}"])
                res[f"restrict_L{l}"] = (timed(h.ops[("restrict", l)]), S[f"restrictL{l}"])
                res[f"prolong_L{l}"] = (timed(h.ops[("prolong", l)]), S[f"prolongL{l}"])
        res["spmv_cg"] = (timed(h.ops["spmv_cg"]), S["SpMVAp"])
        v = h.vops
        # fused vector kernels against the limits of the HPCG operations they replace
        res["dot_r_z"] = (timed(lambda: (v["rz"](), v["rz_fin"]())), S["DDOT"])
        res["p_update"] = (timed(v["p"]), S["WAXPBY"])
        res["dot_p_Ap"] = (timed(lambda: (v["pAp"](), v["pAp_fin"]())), S["DDOT"])
        res["x_r_update_norm"] = (timed(lambda: (v["xr"](), v["nr_fin"]())), 2 * S["WAXPBY"] + S["DDOT"])
        tot = sum(us for us, _ in res.values())
        start_us = timed(lambda: (h.reset(), h.start())) - timed(h.reset)  # once per CG set, not per iteration
        print(
            json.dumps(
                dict(
                    nx=nx,
                    sum_ms=round(tot / 1e3, 3),
                    sol_ms=round(sum(s for _, s in res.values()) / 1e3, 3),
                    cg_start_us=round(start_us, 1),
                    ops={
                        k: dict(us=round(us, 1), sol_us=round(s, 2), pct_of_sol=round(100 * s / us, 1))
                        for k, (us, s) in res.items()
                    },
                )
            ),
            flush=True,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main(int(sys.argv[1]))
