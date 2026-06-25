# SPDX-License-Identifier: Apache-2.0
"""A/B RoPE at pi0.5 denoise shapes: split-half (current) vs rotary_embedding_llama.
Both run in ONE trace so tracy reports all 4 rotary ops in the same replay session.
Q=[1,8,32,256], K=[1,1,32,256], head_dim=256, seq=32 (prefill/multi-position)."""
import argparse
import torch
import ttnn

from models.tt_transformers.tt.common import precompute_freqs, gather_cos_sin, get_rot_transformation_mat

HEADS, KVH, SEQ, HD = 8, 1, 32, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=1)
    args = ap.parse_args()
    dev = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=24576, trace_region_size=134_217_728)

    def t(shape):
        return ttnn.from_torch(
            torch.randn(*shape).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    q = t([1, HEADS, SEQ, HD])
    k = t([1, KVH, SEQ, HD])

    # --- split-half cos/sin (current pi0.5 convention) ---
    half = HD // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, HD, 2).float() / HD))
    ang = torch.outer(torch.arange(SEQ).float(), inv)  # [SEQ, half]
    cos_sh = torch.cat([ang.cos(), ang.cos()], dim=-1)[None, None]  # [1,1,SEQ,HD]
    sin_sh = torch.cat([ang.sin(), ang.sin()], dim=-1)[None, None]
    cos_sh = ttnn.from_torch(cos_sh.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    sin_sh = ttnn.from_torch(sin_sh.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    # --- llama cos/sin (interleaved) + 32x32 transformation matrix ---
    cosf, sinf = precompute_freqs(HD, SEQ * 2, theta=10000.0, scale_factor=None, orig_context_len=None)
    cos_ll, sin_ll = gather_cos_sin(torch.arange(SEQ), cosf, sinf)  # [1,1,SEQ,HD]
    cos_ll = ttnn.from_torch(cos_ll.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    sin_ll = ttnn.from_torch(sin_ll.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tm = get_rot_transformation_mat(32)
    print("trans_mat torch shape:", tuple(tm.shape))
    trans_mat = ttnn.from_torch(tm.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def run():
        a = ttnn.experimental.rotary_embedding(q, cos_sh, sin_sh)
        b = ttnn.experimental.rotary_embedding(k, cos_sh, sin_sh)
        c = ttnn.experimental.rotary_embedding_llama(q, cos_ll, sin_ll, trans_mat, is_decode_mode=False)
        d = ttnn.experimental.rotary_embedding_llama(k, cos_ll, sin_ll, trans_mat, is_decode_mode=False)
        return a, b, c, d

    print("[warmup]")
    run()
    ttnn.synchronize_device(dev)

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    run()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    ttnn.release_trace(dev, tid)
    ttnn.CloseDevice(dev)
    print("[done] order in trace: 0=splithalf-Q 1=splithalf-K 2=llama-Q 3=llama-K")


if __name__ == "__main__":
    main()
