# Bit-identity of an env switch of the v3 heads op: Q / K / V with <ENV>=0 vs <ENV>=1 at bs1 (resident constants, Q in
# L1), bs8 and bs16 (Q in DRAM, K / V in L1), same inputs, one process (generic_op hashes compile-time args, so each
# setting builds its own program). Usage: check_heads_env_bitid.py <ENV>
import os
import sys

import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import op as hop
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import make_norm_constants
from models.tt_transformers.tt.common import get_rot_transformation_mat

NH, NKV, DH, EPS, S = 32, 8, 128, 1e-6, 512
D = ttnn.open_device(device_id=0, l1_small_size=32768)
L1, DR, B8 = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat8_b
try:
    torch.manual_seed(0)
    GQ, GK, SC, EP = make_norm_constants(torch.rand(DH) + 0.5, torch.rand(DH) + 0.5, EPS, D)
    ang = torch.rand(1, 1, S, DH) * 6.28
    tl = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    cos, sin, T = tl(torch.cos(ang)), tl(torch.sin(ang)), tl(get_rot_transformation_mat(32))
    for B in (1, 8, 16):
        x = ttnn.from_torch(
            torch.randn(B, 1, S, (NH + 2 * NKV) * DH) * 3, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1
        )
        res = {}
        for col in ("0", "1"):  # QWEN_HEADS_ONE_RSQRT
            os.environ[sys.argv[1]] = col
            kw = dict(resident=True) if B == 1 else {}
            outs = hop.nlp_create_qkv_heads_norm_headsplit(
                x,
                GQ,
                GK,
                SC,
                EP,
                num_heads=NH,
                num_kv_heads=NKV,
                memory_config=L1 if B == 1 else DR,
                kv_memory_config=L1,
                rot_cos=cos,
                rot_sin=sin,
                trans_mat=T,
                q_dtype=B8,
                kv_dtype=B8,
                norm_eps=EPS,
                use_v3=True,
                **kw,
            )
            res[col] = [ttnn.to_torch(t).float() for t in outs]
            [ttnn.deallocate(t) for t in outs]
        same = [torch.equal(a, b) for a, b in zip(res["0"], res["1"])]
        pcc = [float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]) for a, b in zip(res["0"], res["1"])]
        print(f"RES B{B} pcc q/k/v {pcc}", flush=True)
        print(
            f"RES B{B} q/k/v bit-identical {same} max|diff| {[float((a - b).abs().max()) for a, b in zip(res['0'], res['1'])]}",
            flush=True,
        )
        ttnn.deallocate(x)
finally:
    ttnn.close_device(D)
