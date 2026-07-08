# CHAINED denoise-ring reproducer: data flows chip(1,0)->..->chip(1,7)->wrap->chip(1,0),
# each send depends on the prior recv (true circular dependency), wrap feeds next step.
# Realistic payload (1,1,32,1024)=expert_width. Logs before each op to pinpoint a hang.
import os

os.environ["TT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31"
import torch, ttnn
from models.experimental.pi0_5.tt.tt_pipeline._transport import SplitSocketTransport


def log(m):
    print(f"[cring] {m}", flush=True)


FAB = os.environ.get("FAB", "torus_xy").lower()
W = int(os.environ.get("PAYLOAD_W", "1024"))
fabmap = {
    "2d": ttnn.FabricConfig.FABRIC_2D,
    "torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "torus_x": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    "torus_y": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
}
log(f"FAB={FAB} payload=(1,1,32,{W})")
ttnn.set_fabric_config(fabmap[FAB])
parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 8))


def mk(dev):
    return ttnn.from_torch(
        torch.randn(1, 1, 32, W, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


try:
    chips = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, c)) for c in range(8)]
    x0 = mk(chips[0])
    hop_tps = [SplitSocketTransport() for _ in range(7)]
    wrap_tp = SplitSocketTransport()
    # prepare (chained buffers): hop c template = the buffer being sent from chip c
    hop = []
    cur = x0
    for c in range(7):
        log(f"CREATE hop{c} chip(1,{c})->chip(1,{c+1})")
        ss, rs, buf = hop_tps[c].prepare(cur, chips[c + 1], tag=f"hop{c}")
        hop.append((ss, rs, buf))
        cur = buf
    log("CREATE wrap chip(1,7)->chip(1,0)")
    wss, wrs, wbuf = wrap_tp.prepare(cur, chips[0], tag="wrap")
    log("ALL CREATED")
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
        for ch in chips:
            ttnn.synchronize_device(ch)
        x0 = ttnn.clone(wbuf, memory_config=ttnn.L1_MEMORY_CONFIG) if hasattr(ttnn, "clone") else wbuf
        log(f"s{step} DONE")
    log("CHAINED RING PASS")
except Exception as e:
    log(f"CHAINED RING FAIL {type(e).__name__}: {str(e)[:160]}")
finally:
    ttnn.close_mesh_device(parent)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
log("DONE")
