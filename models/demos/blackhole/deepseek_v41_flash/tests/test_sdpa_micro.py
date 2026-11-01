# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""sparse_sdpa micro benchmark on ONE chip with the shapes of the prefill (q [1,32,512,512] bf16 RM = 8 real + 24 zero heads, kv table [1,1,T,512], indices [1,1,512,640]):
traced replay, per-op kernel time from the device profiler (TT_METAL_DEVICE_PROFILER=1 ...). Env: SD_FID (HiFi4|HiFi2|LoFi), SD_FP32 (1|0), SD_KC (k chunk, default 128),
SD_FP8 (1: fp8_e4m3 kv + q), SD_T (kv rows, default 65536), SD_OVERLAP (fraction of the 512 selected ids shared between adjacent queries, default 0 = random),
SD_REPS (calls per trace, default 4), SD_HEADS (real heads with non-zero q, default 8)."""
import os

import pytest
import torch

import ttnn

E = os.environ.get


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [pytest.param({"l1_small_size": 16384, "trace_region_size": 200000000}, id="p")], indirect=True
)
@torch.no_grad()
def test_sdpa_micro(mesh_device):
    md = mesh_device
    C, T, H = 512, int(E("SD_T", "65536")), 32
    reps = int(E("SD_REPS", "4"))
    fp8 = E("SD_FP8", "0") == "1"
    rep = ttnn.ReplicateTensorToMesh(md)
    dt = ttnn.fp8_e4m3 if fp8 else ttnn.bfloat16
    torch.manual_seed(0)
    q = torch.zeros(1, H, C, 512)
    q[:, : int(E("SD_HEADS", "8"))] = torch.randn(1, int(E("SD_HEADS", "8")), C, 512)
    kv = torch.randn(1, 1, T, 512)
    pos = torch.arange(C).view(-1, 1) + (T - C)  # the chunk is the last C rows of the table
    win = (pos - 127 + torch.arange(128).view(1, -1)).clamp(min=0)
    ov = float(E("SD_OVERLAP", "0"))
    if (
        ov > 0
    ):  # adjacent queries share ov of their selected ids: base set per block of 8 queries, a fraction re-drawn per query
        base = torch.randint(0, T - C, (C // 8, 512))
        sel = base.repeat_interleave(8, 0)
        redraw = torch.rand(C, 512) > ov
        sel = torch.where(redraw, torch.randint(0, T - C, (C, 512)), sel)
    else:
        sel = torch.randint(0, T - C, (C, 512))
    idx = torch.cat([win, sel], dim=1).reshape(1, 1, C, 640).to(torch.int32)
    up = lambda t, d, lay: ttnn.from_torch(
        t, device=md, dtype=d, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    tq = up(q, dt if fp8 else ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    tkv = up(kv, dt, ttnn.ROW_MAJOR_LAYOUT)
    tidx = up(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    sink = up(torch.randn(1, 1, 1, H), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, E("SD_FID", "HiFi4")),
        math_approx_mode=False,
        fp32_dest_acc_en=E("SD_FP32", "1") == "1",
        packer_l1_acc=False,
    )
    fmt = ttnn.transformer.SparseKVFormat.FP8_E4M3 if fp8 else ttnn.transformer.SparseKVFormat.BF16

    def run():
        return ttnn.transformer.sparse_sdpa(
            tq,
            tkv,
            tidx,
            512,
            kv_format=fmt,
            scale=512**-0.5,
            k_chunk_size=int(E("SD_KC", "128")),
            compute_kernel_config=ckc,
            attention_sink=sink,
        )

    o = run()
    ttnn.synchronize_device(md)
    ref = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float()
    torch.save(ref[:, :8], E("SD_OUT", "/tmp/sd_out.pt"))
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        run()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    for _ in range(4):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ttnn.release_trace(md, tid)
    print("SD_DONE", flush=True)
