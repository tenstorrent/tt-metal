"""greedy alternatives: max on RM / gather of the argmax value / argmax secondary DM / single all_gather of (value, idx)"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])
head = Hy4DeviceHead(md, mc, ccl, W, head_dtype="bf16", out_dtype="fp32", fidelity="HiFi4")
cols = 8


def tryt(name, fn, T):
    try:
        res(event="g3", name=name, T=T, us=round(timed_trace(fn), 1))
    except Exception as e:
        res(event="g3", name=name, T=T, err=str(e)[:160].replace("\n", " "))


for T in TS:
    lg = head.forward(streams_tt(X[:T]))
    rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
    tryt("max_tile", lambda: ttnn.max(lg, dim=-1, keepdim=True), T)
    tryt("max_rm", lambda: ttnn.max(rm, dim=-1, keepdim=True), T)
    tryt("argmax_rm", lambda: ttnn.argmax(rm, dim=-1, keepdim=True), T)
    tryt("argmax_rm_sdm_off", lambda: ttnn.argmax(rm, dim=-1, keepdim=True, enable_secondary_dm=False), T)
    tryt("argmax_rm_sdm_on", lambda: ttnn.argmax(rm, dim=-1, keepdim=True, enable_secondary_dm=True), T)
    idx = ttnn.argmax(rm, dim=-1, keepdim=True)
    tryt("gather_rm", lambda: ttnn.gather(rm, -1, index=idx), T)
    tryt("gather_tile", lambda: ttnn.gather(lg, -1, index=ttnn.to_layout(idx, ttnn.TILE_LAYOUT)), T)

    # two-stage max on a tall view
    def tall_max():
        t = ttnn.to_layout(ttnn.reshape(rm, [T, 1, 472, 32]), ttnn.TILE_LAYOUT)
        return ttnn.max(ttnn.max(t, dim=-1, keepdim=True), dim=2, keepdim=True)

    tryt("max_tall2", tall_max, T)

    # single-allgather greedy using max_tall2 or gather
    def greedy2():
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)
        mx = ttnn.max(rm, dim=-1, keepdim=True)
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        mxt = ttnn.to_layout(mx, ttnn.TILE_LAYOUT) if mx.layout != ttnn.TILE_LAYOUT else mx
        g = mc.allgather(ttnn.concat([mxt, idxf], dim=-1), ccl, axis=1, dim=3)
        return g

    tryt("greedy2_concat_ag", greedy2, T)
    tryt(
        "ag_only",
        lambda: mc.allgather(
            ttnn.concat([ttnn.max(lg, dim=-1, keepdim=True), ttnn.max(lg, dim=-1, keepdim=True)], dim=-1),
            ccl,
            axis=1,
            dim=3,
        ),
        T,
    )
ttnn.close_mesh_device(md)
