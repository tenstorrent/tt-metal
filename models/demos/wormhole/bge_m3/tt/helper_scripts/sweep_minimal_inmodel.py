"""In-model sweep of minimal_matmul configs for the MLP wi or wo at S512.

Builds the model once. For each candidate, clears the 2D program config and sets
a MinimalMatmulConfig on all 24 layers (BgeM3MLP takes minimal_matmul when the
program config is None), then times the traced forward, nomask. The incumbent
(2D program config) is measured first.

Usage: python sweep_minimal_inmodel.py <batch> <wi|wo>
"""

import statistics
import sys
import time

import ttnn
from models.demos.wormhole.bge_m3.tests.perf import perf
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

batch = int(sys.argv[1])
op = sys.argv[2]
SEQ = 512
REPLAYS = 20
prg_field, min_field = "%s_prg_config" % op, "%s_minimal_config" % op

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000, num_command_queues=1)
args, model, _ = create_tt_model(mesh_device=mesh, max_batch_size=batch, max_seq_len=SEQ, dtype=ttnn.bfloat8_b)
inputs = perf._n300_dp_inputs(args.pad_token_id, batch, SEQ, SEQ)
dev = perf._n300_dp_batchshard(inputs, mesh, on_device=True)
layers = getattr(model, "encoder", model).layers
g = mesh.compute_with_storage_grid_size()
grid = ttnn.CoreCoord(int(g.x), int(g.y))


def set_cfg(prg, mm):
    for layer in layers:
        c = layer.feed_forward.config
        setattr(c, prg_field, prg)
        setattr(c, min_field, mm)


def measure():
    out = model.forward(**dev, no_padding=True)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
    model.capture_trace(**dev, mesh_device=mesh, cq_id=0, no_padding=True)
    for _ in range(3):
        model.execute_trace(blocking=True)
    t = []
    for _ in range(REPLAYS):
        s = time.perf_counter()
        model.execute_trace(blocking=True)
        t.append((time.perf_counter() - s) * 1000)
    model.release_trace()
    return statistics.mean(t)


def candidates():
    for m in (4, 8, 16):
        for k in (4, 8, 16):
            for n in (4, 8):
                for sh, sw in ((1, 4), (2, 2), (2, 4), (4, 2), (1, 8), (8, 1)):
                    if m % sh or n % sw or sh * sw > 8:
                        continue
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=m,
                        K_block_size=k,
                        N_block_size=n,
                        subblock_h=sh,
                        subblock_w=sw,
                        compute_with_storage_grid_size=grid,
                    )
                    yield "m%-2d k%-2d n%d sb%dx%d %dx%d" % (m, k, n, sh, sw, grid.x, grid.y), cfg


c0 = layers[0].feed_forward.config
incumbent = (getattr(c0, prg_field), getattr(c0, min_field))
results = [("incumbent (2D)", measure())]
print("RESULT %-34s %.3f" % results[-1], flush=True)
for name, cfg in candidates():
    set_cfg(None, cfg)
    try:
        ms = measure()
    except Exception as exc:
        print("RESULT %-34s FAIL %s" % (name, str(exc).split("\n")[0][:80]), flush=True)
        continue
    results.append((name, ms))
    print("RESULT %-34s %.3f" % (name, ms), flush=True)
set_cfg(*incumbent)
ttnn.close_mesh_device(mesh)

print("SUMMARY op=%s batch=%d" % (op, batch))
base = results[0][1]
for name, ms in sorted(results, key=lambda r: r[1])[:6]:
    print("  %-34s %.3f  %+.3f" % (name, ms, ms - base))
