# Bit-identity of the two ops behind the chunked bs32 QKV path (tt/qkv_chunks.py), at bs32 ISL 512 shapes:
# 1. the row-split fused add+RMSNorm writing its normalised output as N batch-chunk tensors vs one tensor;
# 2. the fused heads op (v3) run per batch chunk into full-batch Q / K / V at a batch offset vs one full-batch call.
# Usage: TT_VISIBLE_DEVICES=<chip> test_qkv_chunks.py [N=2 | 4]
import sys

import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_add_rmsnorm import (
    fused_add_rmsnorm_split,
    make_add_norm_constants,
)
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import op as hop
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import make_norm_constants
from models.tt_transformers.tt.common import get_rot_transformation_mat

B, S, W = 32, 512, 2560
NH, NKV, DH, EPS = 32, 8, 128, 1e-6
L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def add_norm_split(D, N):
    M = B * S
    consts = make_add_norm_constants(torch.rand(W) + 0.5, EPS, D)
    a = ttnn.from_torch(
        torch.randn(1, 1, M, W), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DR
    )
    b = ttnn.from_torch(
        torch.randn(1, 1, M, W), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1
    )
    kw = dict(R=4, sum_dtype=ttnn.bfloat8_b, memory_config=DR, out_memory_config=L1)
    s1, n1 = fused_add_rmsnorm_split(a, b, *consts, **kw)
    ref_s, ref_n = ttnn.to_torch(s1), ttnn.to_torch(n1)
    ttnn.deallocate(s1)
    ttnn.deallocate(n1)
    parts = tuple(
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M // N, W]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, D, L1)
        for _ in range(N)
    )
    s2, got = fused_add_rmsnorm_split(a, b, *consts, **kw, out_tensor=parts)
    got_n = torch.cat([ttnn.to_torch(h) for h in got], dim=2)
    return torch.equal(ttnn.to_torch(s2), ref_s) and torch.equal(got_n, ref_n)


def heads_chunks(D, N):
    GQ, GK, SC, EP = make_norm_constants(torch.rand(DH) + 0.5, torch.rand(DH) + 0.5, EPS, D)
    ang = torch.rand(1, 1, S, DH) * 6.28
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    cos, sin, T = mk(torch.cos(ang)), mk(torch.sin(ang)), mk(get_rot_transformation_mat(32))
    xt = torch.randn(B, 1, S, (NH + 2 * NKV) * DH)
    kw = dict(
        num_heads=NH, num_kv_heads=NKV, memory_config=DR, rot_cos=cos, rot_sin=sin, trans_mat=T,
        q_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat8_b, norm_eps=EPS,
    )  # fmt: skip
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DR)
    ref = [ttnn.to_torch(t) for t in hop.nlp_create_qkv_heads_norm_headsplit(x, GQ, GK, SC, EP, **kw, use_v3=False)]
    ttnn.deallocate(x)
    full = tuple(
        ttnn.allocate_tensor_on_device(ttnn.Shape(list(r.shape)), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, D, DR) for r in ref
    )
    for c in range(N):
        xh = ttnn.from_torch(
            xt[c * B // N : (c + 1) * B // N], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1
        )
        hop.nlp_create_qkv_heads_norm_headsplit(
            xh, GQ, GK, SC, EP, **kw, out_tensors=full, batch_offset=c * B // N, use_v3=True
        )
        ttnn.deallocate(xh)
    return all(torch.equal(ttnn.to_torch(g), r) for g, r in zip(full, ref))


def main():
    torch.manual_seed(0)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    D = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        print(f"RES add+norm split output in {N} parts bit-identical: {add_norm_split(D, N)}", flush=True)
        print(
            f"RES heads op in {N} batch chunks (v3) vs full batch (v1) bit-identical: {heads_chunks(D, N)}", flush=True
        )
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
