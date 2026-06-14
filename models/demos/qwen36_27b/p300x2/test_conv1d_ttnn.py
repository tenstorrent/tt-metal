# SPDX-License-Identifier: Apache-2.0
"""Validate + time a VECTORIZED ttnn conv1d+SiLU (the reader's scalar 1ms hotspot).
Depthwise causal conv, K=4: window=[s1,s2,s3,new]; out=silu(sum_k window*w); new_state=window.
If this is correct vs torch AND ~µs (not ~1ms), it justifies moving conv off the dataflow RISC."""
import time, statistics
import torch
import ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig

TP = 4


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def silu(x):
    return x * torch.sigmoid(x)


def main():
    cfg = Qwen36ModelConfig()
    dk, nk, dv, nv = cfg.linear_key_head_dim, cfg.linear_num_key_heads, cfg.linear_value_head_dim, cfg.linear_num_value_heads
    K = cfg.linear_conv_kernel_dim  # 4
    conv_dim = (dk * nk * 2 + dv * nv) // TP  # per-chip
    torch.manual_seed(0)
    cs = torch.randn(conv_dim, K) * 0.2          # conv_state window [x_{t-3..t-1}, x_{?}]
    cw = torch.randn(conv_dim, K) * 0.2          # conv weights
    qkv = torch.randn(conv_dim) * 0.3            # new token's qkv_proj

    # torch reference (matches reader): window = [cs[1],cs[2],cs[3],qkv]; out=silu(sum window*cw)
    window = torch.cat([cs[:, 1:K], qkv[:, None]], dim=1)   # [conv_dim, K]
    ref_out = silu((window * cw).sum(dim=1))                 # [conv_dim]
    ref_state = window

    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))  # single-row; conv is per-chip elementwise (no CCL)
    try:
        # pad K to 32 (tile width). place window in cols 0..K-1, zeros elsewhere.
        def to_tt(x2d):  # [conv_dim, K] -> [1,1,conv_dim,32] padded
            p = torch.zeros(conv_dim, 32); p[:, :x2d.shape[1]] = x2d
            return ttnn.from_torch(p.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=ttnn.ReplicateTensorToMesh(md))
        cs_t = to_tt(cs); cw_t = to_tt(cw)
        qkv_t = ttnn.from_torch(qkv.view(1, 1, conv_dim, 1).repeat(1, 1, 1, 1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=ttnn.ReplicateTensorToMesh(md))

        def conv_step():
            # shift: window = [cs[:,1:4], qkv]   (cols)
            sh = ttnn.slice(cs_t, [0, 0, 0, 1], [1, 1, conv_dim, K])      # [.,K-1]
            win = ttnn.concat([sh, qkv_t], dim=3)                        # [.,K]
            # pad to 32 by computing dot only over K cols: mul then sum over last dim
            prod = ttnn.mul(win, ttnn.slice(cw_t, [0, 0, 0, 0], [1, 1, conv_dim, K]))
            dot = ttnn.sum(prod, dim=3)                                  # [1,1,conv_dim,1] (or [..conv_dim])
            return ttnn.silu(dot), win

        out_t, win_t = conv_step()
        out_h = ttnn.to_torch(out_t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[0:1].float().reshape(-1)[:conv_dim]
        print(f"[shapes] ref={tuple(ref_out.shape)} ttnn_out={tuple(out_h.shape)}", flush=True)
        print(f"[PCC] vectorized conv1d+silu vs torch = {pcc(ref_out, out_h):.5f}", flush=True)

        # time it
        for _ in range(10): conv_step()
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        for _ in range(200): conv_step()
        ttnn.synchronize_device(md)
        ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"[time] vectorized conv1d ttnn = {ms:.3f} ms/call (reader scalar was 1.012 ms)", flush=True)
        print("PASS" if pcc(ref_out, out_h) > 0.97 else "FAIL", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
