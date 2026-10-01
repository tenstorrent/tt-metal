# Fused heads op at the model's batched placement: v3 compute, QKV input in L1, Q out DRAM (HQ_L1=1: L1), K / V out
# L1, preallocated outputs (bs32: a quarter-batch chunk writing into bs32 Q / K / V, B_in 8 B_out 32). Ablations reuse
# bench_heads_bs16_ablate's scratch kernel variants; HV_ONLY=full,"compute only" picks variants.
# HB1=1: the bs1 call instead (resident cos / sin / rotation / scaler / eps shard, Q in L1, outputs allocated per call).
# Usage: bench_heads_placement_ablate.py <B_in> <B_out>
import os
import statistics
import sys
import time

import torch

import ttnn

B_IN, B_OUT = int(sys.argv[1]), int(sys.argv[2])
sys.argv = [sys.argv[0], str(B_IN)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_heads_bs16_ablate as A

from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import op as hop
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import make_norm_constants
from models.tt_transformers.tt.common import get_rot_transformation_mat

NH, NKV, DH, EPS, S = 32, 8, 128, 1e-6, 512
reader, writer, compute = hop.READER_KERNEL, hop.WRITER_KERNEL, hop.COMPUTE_KERNEL_BS1
pass_c = os.path.join(A.SCRATCH, "compute_pass.cpp")
open(pass_c, "w").write(A.PASS_COMPUTE)
r0, w0 = A.variant(reader, "r_noread.cpp", read=0), A.variant(writer, "w_nowrite.cpp", write=0)
ONLY = os.getenv("HV_ONLY")
variants = {
    "full": (reader, compute, writer),
    "compute only": (r0, compute, w0),
    "DM only (copy compute)": (reader, pass_c, writer),
    "read only": (reader, pass_c, w0),
    "write only": (r0, pass_c, writer),
    "handshakes only": (r0, pass_c, w0),
}
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
B8 = ttnn.bfloat8_b
try:
    GQ, GK, SC, EP = make_norm_constants(torch.rand(DH) + 0.5, torch.rand(DH) + 0.5, EPS, D)
    ang = torch.rand(1, 1, S, DH) * 6.28
    tl = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    cos, sin, T = tl(torch.cos(ang)), tl(torch.sin(ang)), tl(get_rot_transformation_mat(32))
    x = ttnn.from_torch(
        torch.randn(B_IN, 1, S, (NH + 2 * NKV) * DH), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1
    )
    HB1 = os.getenv("HB1", "0") == "1"
    qmc = L1 if HB1 or os.getenv("HQ_L1", "0") == "1" else DR
    alloc = lambda shp, mc: ttnn.allocate_tensor_on_device(ttnn.Shape(shp), B8, ttnn.TILE_LAYOUT, D, mc)
    outs = (
        None
        if HB1
        else (alloc([B_OUT, NH, S, DH], qmc), alloc([B_OUT, NKV, S, DH], L1), alloc([B_OUT, NKV, S, DH], L1))
    )
    for vname, (r, c, w) in variants.items():
        if ONLY and vname not in ONLY.split(","):
            continue
        hop.READER_KERNEL, hop.WRITER_KERNEL, hop.COMPUTE_KERNEL_BS1 = r, w, c
        fn = lambda: hop.nlp_create_qkv_heads_norm_headsplit(
            x,
            GQ,
            GK,
            SC,
            EP,
            num_heads=NH,
            num_kv_heads=NKV,
            memory_config=qmc,
            kv_memory_config=L1,
            rot_cos=cos,
            rot_sin=sin,
            trans_mat=T,
            q_dtype=B8,
            kv_dtype=B8,
            norm_eps=EPS,
            use_v3=True,
            **(dict(resident=True) if HB1 else dict(out_tensors=outs, batch_offset=0)),
        )
        free = lambda r: [ttnn.deallocate(t) for t in r] if HB1 else None
        for _ in range(2):
            free(fn())
        ttnn.synchronize_device(D)
        n = 8
        tid = ttnn.begin_trace_capture(D, cq_id=0)
        for _ in range(n):
            free(fn())
        ttnn.end_trace_capture(D, tid, cq_id=0)
        ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(D, tid)
        print(
            f"RES heads Bin={B_IN} Bout={B_OUT} Q={'L1' if qmc is L1 else 'DRAM'} rotdb={os.getenv('QWEN_HEADS_ROT_DB','1')} {vname:24s} {statistics.median(ts):7.1f} us/call",
            flush=True,
        )
finally:
    hop.READER_KERNEL, hop.WRITER_KERNEL, hop.COMPUTE_KERNEL_BS1 = reader, writer, compute
    ttnn.close_device(D)
