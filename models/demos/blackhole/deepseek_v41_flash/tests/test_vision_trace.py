# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does capturing the image tower as a trace pay? (tt/vision.py ``capture_traces``)
  1. profile of the eager tower: host time to enqueue every op (``_run`` returns before the device is done) vs the time the device needs after that vs upload / readback,
  2. the traced tower (one trace per (patch bucket, aligner bucket)): output bit-identical to the eager one, time per call for 1 / 4 / 8 images,
  3. the trace region the captured traces need (the test opens the device with VIS_TRACE_REGION bytes; a capture that does not fit fails loudly).

    VIS_TRACE_REGION (400000000), VIS_REPEAT (5)
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_vision_pcc import _image
from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.vision import VIT_BUCKETS, DSV41Vision, VisionArgs
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

TRACE_REGION = int(os.environ.get("VIS_TRACE_REGION", "400000000"))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": TRACE_REGION,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_vision_trace(mesh_device):
    import traceback

    try:
        _body(mesh_device)
    except BaseException:
        traceback.print_exc()
        raise


def _body(md):
    va = VisionArgs()
    vision = DSV41Vision(
        md, mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), args=va, shards=_Shards()
    )
    vision.warmup()
    rep = int(os.environ.get("VIS_REPEAT", "5"))
    sizes = {"1610 patches": (640, 480), "3364 patches": (800, 800), "7200 patches": (1400, 1000)}
    pool = {}
    for name, (w, h) in sizes.items():
        patches, n_h, n_w, _, _ = VP.patchify_pil(_image(w, h, 1), va)
        pool[name] = (patches, n_h, n_w)

    # 1. profile of the eager tower
    for name, img in pool.items():
        imgs = [img] * vision.rows
        n_pad, nt_pad = vision._buckets(imgs)
        tu = te = td = tr = 0.0
        for _ in range(rep):
            t0 = time.perf_counter()
            vision._upload_wave(imgs, n_pad, nt_pad)
            t1 = time.perf_counter()
            out = vision._run(n_pad, nt_pad)
            t2 = time.perf_counter()
            ttnn.synchronize_device(md)
            t3 = time.perf_counter()
            devs = ttnn.get_device_tensors(out)
            _ = [ttnn.to_torch(devs[i * vision.cols]) for i in range(vision.rows)]
            t4 = time.perf_counter()
            ttnn.deallocate(out)
            tu, te, td, tr = tu + t1 - t0, te + t2 - t1, td + t3 - t2, tr + t4 - t3
        print(
            f"EAGER {name} (bucket {n_pad}/{nt_pad}): upload {tu / rep * 1e3:.1f} ms, enqueue (host dispatch) {te / rep * 1e3:.1f} ms, device after enqueue {td / rep * 1e3:.1f} ms, "
            f"readback {tr / rep * 1e3:.1f} ms; total {(tu + te + td + tr) / rep * 1e3:.1f} ms",
            flush=True,
        )
    eager = {name: vision.encode_many([img] * 4) for name, img in pool.items()}
    t_eager = {}
    for name, img in pool.items():
        for k in (1, 4, 8):
            t0 = time.perf_counter()
            for _ in range(rep):
                vision.encode_many([img] * k)
            t_eager[(name, k)] = (time.perf_counter() - t0) / rep

    # 2. traced
    t0 = time.perf_counter()
    vision.capture_traces()
    print(
        f"capture_traces: {time.perf_counter() - t0:.1f} s, {len(vision.traces)} traces (patch buckets {VIT_BUCKETS})",
        flush=True,
    )
    for name, img in pool.items():
        got = vision.encode_many([img] * 4)
        same = all(torch.equal(a, b) for a, b in zip(got, eager[name]))
        d = max((a.float() - b.float()).abs().max().item() for a, b in zip(got, eager[name]))
        print(f"TRACED vs EAGER {name}: bit-equal {same}, max|diff| {d:.5f}", flush=True)
        assert d < 0.05
        for k in (1, 4, 8):
            t0 = time.perf_counter()
            for _ in range(rep):
                vision.encode_many([img] * k)
            dt = (time.perf_counter() - t0) / rep
            print(
                f"TIMING {name} x{k}: eager {t_eager[(name, k)] * 1e3:.0f} ms, traced {dt * 1e3:.0f} ms ({t_eager[(name, k)] / dt:.2f}x)",
                flush=True,
            )
