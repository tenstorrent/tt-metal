"""In-model sweep of one matmul's program config at S512.

Builds the model once, then for each candidate sets the program config on all
24 layers, warms up with no_padding=True, captures the trace, and times 20
replays. The metric is the traced forward wall, not the op alone: isolated
sweeps have picked configs that lost in the model.

Usage: python sweep_inmodel.py <batch> <qkv|ao|wo>
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

# (K tiles, N tiles, owner, field)
SHAPES = {
    "qkv": (32, 96, "attention", "qkv_prg_config"),
    "ao": (32, 32, "attention", "output_prg_config"),
    "wo": (128, 32, "feed_forward", "wo_prg_config"),
    "wi": (32, 128, "feed_forward", "wi_prg_config"),
}
FUSED = {"wi": (ttnn.UnaryOpType.GELU, True)}
k_tiles, n_tiles, owner, field = SHAPES[op]
m_tiles = batch * SEQ // 32


def ceil_div(a, b):
    return (a + b - 1) // b


def subblocks(pm, pn, cap=8):
    out = [(h, w) for h in range(1, pm + 1) for w in range(1, pn + 1) if pm % h == 0 and pn % w == 0 and h * w <= cap]
    out.sort(key=lambda hw: (-hw[0] * hw[1], -hw[1]))
    return out[:2]


def candidates():
    seen = set()
    for gx, gy in ((11, 10), (12, 10), (13, 10), (8, 10), (8, 8)):
        pm, pn = ceil_div(m_tiles, gy), ceil_div(n_tiles, gx)
        for ibw in (2, 4, 8, 16, 32):
            if k_tiles % ibw:
                continue
            # obh splits the M block; at B16 an L1 output only fits with a split block.
            for obh in sorted({pm} | {pm // d for d in (2, 4) if pm % d == 0}, reverse=True):
                for h, w in subblocks(obh, pn):
                    key = (gx, gy, ibw, h, w, obh)
                    if key in seen:
                        continue
                    seen.add(key)
                    kw = {} if obh == pm else {"out_block_h": obh}
                    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=ibw,
                        out_subblock_h=h,
                        out_subblock_w=w,
                        per_core_M=pm,
                        per_core_N=pn,
                        transpose_mcast=False,
                        fused_activation=FUSED.get(op),
                        **kw,
                    )
                    yield "%dx%d ibw%-2d sb%dx%d pm%d pn%d obh%d" % (gx, gy, ibw, h, w, pm, pn, obh), cfg


mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000, num_command_queues=1)
args, model, _ = create_tt_model(mesh_device=mesh, max_batch_size=batch, max_seq_len=SEQ, dtype=ttnn.bfloat8_b)
inputs = perf._n300_dp_inputs(args.pad_token_id, batch, SEQ, SEQ)
dev = perf._n300_dp_batchshard(inputs, mesh, on_device=True)
layers = getattr(model, "encoder", model).layers


def set_cfg(cfg):
    for layer in layers:
        setattr(getattr(layer, owner).config, field, cfg)


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


results = []
incumbent = getattr(getattr(layers[0], owner).config, field)
results.append(("incumbent (%s)" % ("auto" if incumbent is None else "tuned"), measure()))
print("RESULT %-34s %.3f" % results[-1], flush=True)
for name, cfg in candidates():
    set_cfg(cfg)
    try:
        ms = measure()
    except Exception as exc:
        print("RESULT %-34s FAIL %s" % (name, str(exc).split("\n")[0][:80]), flush=True)
        continue
    results.append((name, ms))
    print("RESULT %-34s %.3f" % (name, ms), flush=True)
set_cfg(incumbent)
ttnn.close_mesh_device(mesh)

print("SUMMARY op=%s batch=%d" % (op, batch))
base = results[0][1]
for name, ms in sorted(results, key=lambda r: r[1])[:6]:
    print("  %-34s %.3f  %+.3f" % (name, ms, ms - base))
