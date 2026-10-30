"""Sweep of the reduce-scatter / all-gather parameters (links, workers per link, chunks per sync, buffers per
channel) at the TP=4 shapes of the Mistral step, through ttml's wrappers (env knobs TTML_CCL_*, read per call)."""
import os
import time

import numpy as np
import ttnn
import ttml

mesh = ttml.Mesh((1, 4), ("dp", "tp"))
ttml.open_device_mesh(mesh)
dev = ttml.autograd.AutoContext.get_instance().get_device()
rng = np.random.default_rng(0)
KNOBS = [
    "TTML_CCL_NUM_LINKS",
    "TTML_CCL_RS_CHUNKS_PER_SYNC",
    "TTML_CCL_RS_WORKERS_PER_LINK",
    "TTML_CCL_RS_BUFFERS_PER_CHANNEL",
    "TTML_CCL_AG_CHUNKS_PER_SYNC",
    "TTML_CCL_AG_WORKERS_PER_LINK",
    "TTML_CCL_AG_BUFFERS_PER_CHANNEL",
]


def dev_t(shape):
    return ttml.autograd.Tensor.from_numpy(
        rng.normal(0, 1, shape).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


def setenv(**kw):
    for k in KNOBS:
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ[k] = str(v)


x_rs = dev_t((1, 1, 1024, 5120))  # all-reduce input of every parallel linear (reduce-scatter on dim 3)
x_ag = dev_t((1, 1, 1024, 1280))  # its all-gather half
configs = [dict()]
configs += [dict(TTML_CCL_NUM_LINKS=n) for n in (1, 2)]
configs += [dict(TTML_CCL_RS_WORKERS_PER_LINK=w, TTML_CCL_AG_WORKERS_PER_LINK=w) for w in (1, 2, 4, 8)]
configs += [dict(TTML_CCL_RS_CHUNKS_PER_SYNC=c, TTML_CCL_AG_CHUNKS_PER_SYNC=c) for c in (1, 2, 4, 8, 16, 32)]
configs += [dict(TTML_CCL_RS_BUFFERS_PER_CHANNEL=b, TTML_CCL_AG_BUFFERS_PER_CHANNEL=b) for b in (2, 4, 8)]
for cfg in configs:
    setenv(**cfg)
    try:
        t_rs = bench(lambda: ttml.ops.distributed.reduce_scatter(x_rs, 3))
    except Exception as e:  # noqa: BLE001
        t_rs = float("nan")
        print(f"  reduce_scatter failed: {str(e).splitlines()[0][:160]}", flush=True)
    try:
        t_ag = bench(lambda: ttml.ops.distributed.all_gather(x_ag, 3))
    except Exception as e:  # noqa: BLE001
        t_ag = float("nan")
        print(f"  all_gather failed: {str(e).splitlines()[0][:160]}", flush=True)
    print(
        f"{cfg or 'default'}: reduce_scatter {t_rs:.0f} us, all_gather {t_ag:.0f} us (wall per op, 20 iters)",
        flush=True,
    )
ttml.close_device_mesh()
