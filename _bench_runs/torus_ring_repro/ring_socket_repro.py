# Minimal reproducer of the denoise socket RING under a chosen fabric. Builds the 8
# forward hops chip(1,c)->chip(1,c+1) + wrap chip(1,7)->chip(1,0), then relays a
# tensor around the ring. Logs before every CREATE/send/recv so a hang pinpoints the
# culprit. FAB=2d (control, should PASS) | torus_xy | torus_x | torus_y.
import os

os.environ["TT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31"
import torch, ttnn
from models.experimental.pi0_5.tt.tt_pipeline._transport import SplitSocketTransport


def log(m):
    print(f"[ring] {m}", flush=True)


FAB = os.environ.get("FAB", "torus_xy").lower()
fabmap = {
    "2d": ttnn.FabricConfig.FABRIC_2D,
    "torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "torus_x": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    "torus_y": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
}
log(f"FAB={FAB}")
ttnn.set_fabric_config(fabmap[FAB])
parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 8))


def mk(dev):
    return ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


try:
    chips = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, c)) for c in range(8)]
    log(f"opened; denoise-row chip ids={[list(ch.get_device_ids())[0] for ch in chips]}")
    srcs = [mk(chips[c]) for c in range(8)]
    hop_tps = [SplitSocketTransport() for _ in range(7)]
    wrap_tp = SplitSocketTransport()
    hop = []
    for c in range(7):
        log(f"CREATE hop {c}: chip(1,{c})->chip(1,{c+1})")
        ss, rs, buf = hop_tps[c].prepare(srcs[c], chips[c + 1], tag=f"hop{c}")
        hop.append((ss, rs, buf))
    log("CREATE wrap: chip(1,7)->chip(1,0)")
    wss, wrs, wbuf = wrap_tp.prepare(srcs[7], chips[0], tag="wrap")
    log("ALL SOCKETS CREATED")
    for step in range(3):
        for c in range(7):
            log(f"step{step} send hop{c}")
            hop_tps[c].send_only(srcs[c], hop[c][0])
            log(f"step{step} recv hop{c}")
            hop_tps[c].recv_only(hop[c][2], hop[c][1])
        log(f"step{step} send wrap")
        wrap_tp.send_only(srcs[7], wss)
        log(f"step{step} recv wrap")
        wrap_tp.recv_only(wbuf, wrs)
        for ch in chips:
            ttnn.synchronize_device(ch)
        log(f"step{step} DONE")
    log("RING PASS")
except Exception as e:
    log(f"RING FAIL {type(e).__name__}: {str(e)[:160]}")
finally:
    ttnn.close_mesh_device(parent)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
log("DONE")
