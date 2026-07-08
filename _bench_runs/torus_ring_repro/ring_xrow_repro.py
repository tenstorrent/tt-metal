# Ring + CROSS-ROW KV sockets, both datelines active under torus_xy: create 8 cross-row
# sockets chip(0,c)->chip(1,c) (Y dateline), do a KV handoff, keep them OPEN, then run
# the denoise ring on row 1 (X dateline). Tests the both-datelines deadlock hypothesis.
import os

os.environ["TT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31"
import torch, ttnn
from models.experimental.pi0_5.tt.tt_pipeline._transport import SplitSocketTransport


def log(m):
    print(f"[xr] {m}", flush=True)


FAB = os.environ.get("FAB", "torus_xy").lower()
W = 1024
fabmap = {
    "2d": ttnn.FabricConfig.FABRIC_2D,
    "torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "torus_x": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
}
log(f"FAB={FAB}")
ttnn.set_fabric_config(fabmap[FAB])
parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 8))


def mk(dev, w=W):
    return ttnn.from_torch(
        torch.randn(1, 1, 32, w, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


try:
    row0 = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, c)) for c in range(8)]
    row1 = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, c)) for c in range(8)]
    # cross-row KV sockets (Y dateline under torus_xy): chip(0,c)->chip(1,c)
    xr_tps = [SplitSocketTransport() for _ in range(8)]
    xr = []
    xsrc = [mk(row0[c]) for c in range(8)]
    for c in range(8):
        log(f"CREATE xrow{c} chip(0,{c})->chip(1,{c})")
        ss, rs, buf = xr_tps[c].prepare(xsrc[c], row1[c], tag=f"xr{c}")
        xr.append((ss, rs, buf))
    log("xrow sockets created; doing KV handoff (send+recv), keeping OPEN")
    for c in range(8):
        xr_tps[c].send_only(xsrc[c], xr[c][0])
    for c in range(8):
        xr_tps[c].recv_only(xr[c][2], xr[c][1])
    for ch in row1:
        ttnn.synchronize_device(ch)
    log("KV handoff done; building denoise ring (X dateline) with xrow sockets still open")
    hop_tps = [SplitSocketTransport() for _ in range(7)]
    wrap_tp = SplitSocketTransport()
    x0 = mk(row1[0])
    hop = []
    cur = x0
    for c in range(7):
        ss, rs, buf = hop_tps[c].prepare(cur, row1[c + 1], tag=f"hop{c}")
        hop.append((ss, rs, buf))
        cur = buf
    wss, wrs, wbuf = wrap_tp.prepare(cur, row1[0], tag="wrap")
    log("ring sockets created; running ring (both datelines now in play)")
    for step in range(3):
        cur = x0
        for c in range(7):
            log(f"s{step} send hop{c}")
            hop_tps[c].send_only(cur, hop[c][0])
            log(f"s{step} recv hop{c}")
            hop_tps[c].recv_only(hop[c][2], hop[c][1])
            cur = hop[c][2]
        log(f"s{step} send wrap")
        wrap_tp.send_only(cur, wss)
        log(f"s{step} recv wrap")
        wrap_tp.recv_only(wbuf, wrs)
        for ch in row1:
            ttnn.synchronize_device(ch)
        log(f"s{step} DONE")
    log("XROW+RING PASS")
except Exception as e:
    log(f"XROW+RING FAIL {type(e).__name__}: {str(e)[:160]}")
finally:
    ttnn.close_mesh_device(parent)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
log("DONE")
