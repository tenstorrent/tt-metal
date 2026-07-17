# SPDX-License-Identifier: Apache-2.0
"""Run ONE encoder-SDPA chunk config in isolation (no pytest mesh fixture, no
fabric). Usage: python encoder_sdpa_single_config.py <q_chunk> <k_chunk>

Standalone so a device hang only kills this child process (parent can timeout it
and reset), and so the conftest FABRIC_1D mesh fixture is not involved. Opens a
plain (2,1) mesh, runs stock + experimental once, prints PCC + traced wall.
"""
import sys
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import EncoderSDPAConfig
from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa.op import (
    bge_encoder_sdpa_stock,
    build_encoder_sdpa_descriptor,
)

B, HQ, HKV, SQ, SK, DH = 6, 32, 16, 4096, 8192, 64


def main():
    qc, kc = int(sys.argv[1]), int(sys.argv[2])
    torch.manual_seed(0)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(2, 1), trace_region_size=40_000_000)
    try:

        def mk(h, s, dt):
            return ttnn.from_torch(
                torch.randn(B, h, s, DH, dtype=torch.bfloat16),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        q, k, v = mk(HQ, SQ, ttnn.bfloat8_b), mk(HKV, SK, ttnn.bfloat4_b), mk(HKV, SK, ttnn.bfloat8_b)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True
        )
        stock = bge_encoder_sdpa_stock(q, k, v, compute_kernel_config=ckc)
        ttnn.synchronize_device(dev)
        stock_t = ttnn.to_torch(stock, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:B]

        import os as _os0

        _stream = _os0.environ.get("BGE_SDPA_STREAMING", "0") == "1"
        _fp32 = _os0.environ.get("BGE_SDPA_FP32DEST", "1") == "1"  # default True (parity)
        _fullsync = _os0.environ.get("BGE_SDPA_FULLSYNC", "0") == "1"
        _bf8score = _os0.environ.get("BGE_SDPA_BF8SCORE", "0") == "1"
        _kd = int(_os0.environ.get("BGE_SDPA_KDEPTH", "2"))
        _vd = int(_os0.environ.get("BGE_SDPA_VDEPTH", "2"))
        _qd = int(_os0.environ.get("BGE_SDPA_QDEPTH", "2"))
        _kvalias = _os0.environ.get("BGE_SDPA_KVALIAS", "0") == "1"
        cfg = EncoderSDPAConfig(
            q_chunk_size=qc,
            k_chunk_size=kc,
            use_streaming=_stream,
            fp32_dest_acc_en=_fp32,
            dst_full_sync_en=_fullsync,
            score_cb_bf8=_bf8score,
            k_buffer_depth=_kd,
            v_buffer_depth=_vd,
            q_buffer_depth=_qd,
            kv_alias=_kvalias,
        )
        print(f"[q{qc}/k{kc}] stream={_stream} fp32={_fp32} fullsync={_fullsync} bf8score={_bf8score} kvq_depth={_kd}/{_vd}/{_qd} kv_alias={_kvalias}", flush=True)
        build = build_encoder_sdpa_descriptor(q, k, v, config=cfg)
        print(f"[q{qc}/k{kc}] descriptor built, launching...", flush=True)
        # Warm once (compile + program cache), then ONE clean signposted launch for
        # the bounded internal profile (reviewer step 5: profile ONE warm launch).
        ttnn.generic_op(build.io_tensors, build.descriptor)
        ttnn.synchronize_device(dev)
        import os as _os

        if _os.environ.get("BGE_SDPA_SIGNPOST", "0") == "1":
            ttnn.synchronize_device(dev)
            from tracy import signpost

            signpost("BGE_SDPA_PROFILE_START")
            ttnn.generic_op(build.io_tensors, build.descriptor)
            ttnn.synchronize_device(dev)
            signpost("BGE_SDPA_PROFILE_STOP")
        print(f"[q{qc}/k{kc}] launch OK", flush=True)
        got = ttnn.to_torch(build.output, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:B]
        _, msg = comp_pcc(stock_t, got, 0.99)

        if _os.environ.get("BGE_SDPA_SIGNPOST", "0") == "1":
            print(f"RESULT q={qc} k={kc} pcc={msg} (signpost-profile mode, wall skipped)", flush=True)
            return
        # traced wall
        ttnn.generic_op(build.io_tensors, build.descriptor)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        ttnn.generic_op(build.io_tensors, build.descriptor)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.release_trace(dev, tid)
        ts.sort()
        print(f"RESULT q={qc} k={kc} traced_wall_min={ts[0]:.3f} med={ts[len(ts)//2]:.3f} ms pcc={msg}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
