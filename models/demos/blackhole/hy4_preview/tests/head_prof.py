"""prefix timing of Hy4DeviceHead.hidden (stage deltas)"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("STAGES", "none")
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])
head = Hy4DeviceHead(md, mc, ccl, W, head_dtype="bf16", out_dtype="fp32")
for T in TS:
    s = streams_tt(X[:T])
    prev = 0
    for st in (
        ["flat", "ms", "mixes", "pre", "y", None]
        if os.environ.get("HV") in ("3", "4")
        else (
            ["flat", "rms", "mixes", "pre", "y", None]
            if os.environ.get("HV") == "2"
            else ["round", "flat", "ms", "mixes", "pre", "y", None]
        )
    ):
        head.stop = st
        us = timed_trace(
            lambda: (
                head.hidden4(s, os.environ.get("HY_Y", "ew"), os.environ.get("HY_M", "loop"))
                if os.environ.get("HV") == "4"
                else (
                    head.hidden3(s)
                    if os.environ.get("HV") == "3"
                    else head.hidden2(s)
                    if os.environ.get("HV") == "2"
                    else head.hidden(s)
                )
            )
        )
        res(event="prof", T=T, stop=st, cum_us=round(us, 1), delta_us=round(us - prev, 1))
        prev = us
ttnn.close_mesh_device(md)
