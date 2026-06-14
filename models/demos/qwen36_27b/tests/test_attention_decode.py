# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Fast single-layer correctness test for the gated GQA attention decode path,
against the PyTorch reference (reference/deltanet_reference.py:GatedAttentionLayer).

Runs T sequential decode steps (prefill the first token, then decode), comparing
the TT attention output to the reference at every step. Validates BOTH the CPU
fallback path and the on-device path (USE_DEVICE_ATTENTION=1), so it is the
iteration vehicle for the device-attention work — no 364s full-model build.

Run (device 1, healthy):
  TT_DEVICE=1 ./run_qwen36.sh bash -lc \
    'python -u -m pytest models/demos/qwen36_27b/tests/test_attention_decode.py -s'
or directly:
  TT_DEVICE=1 ./run_qwen36.sh bash -lc \
    'python -u -m models.demos.qwen36_27b.tests.test_attention_decode'
"""
import os
import torch
import ttnn

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.attention import TtGatedAttention
from models.demos.qwen36_27b.reference.deltanet_reference import Qwen36Config, GatedAttentionLayer


def build_rope(rotary_dim, max_seq, theta=10_000_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().reshape(1, 1, max_seq, rotary_dim // 2).repeat(1, 1, 1, 2)
    sin = freqs.sin().reshape(1, 1, max_seq, rotary_dim // 2).repeat(1, 1, 1, 2)
    return cos, sin


def _run(device, use_device_attn):
    torch.manual_seed(0)
    # Small but architecturally faithful config (head_dim multiple of 32, partial rope).
    # head_dim=128, partial 0.5 -> rotary_dim=64, half=32 (tile-aligned, like the
    # real model's 256*0.25=64). Keep dims multiples of 32 for tile-layout slicing.
    cfg = Qwen36ModelConfig(
        hidden_size=512, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, partial_rotary_factor=0.5, rope_theta=10_000_000.0,
        max_seq_len=64, weights_dtype=ttnn.bfloat16,
    )
    ref_cfg = Qwen36Config(
        hidden_size=cfg.hidden_size, num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
    )
    ref = GatedAttentionLayer(ref_cfg, layer_idx=0).eval()

    # Map reference weights -> tt state_dict keys.
    sd = {}
    p = "model.layers.0.self_attn"
    rsd = ref.state_dict()
    sd[f"{p}.q_proj.weight"] = rsd["q_proj.weight"]
    sd[f"{p}.k_proj.weight"] = rsd["k_proj.weight"]
    sd[f"{p}.v_proj.weight"] = rsd["v_proj.weight"]
    sd[f"{p}.o_proj.weight"] = rsd["o_proj.weight"]
    sd[f"{p}.q_norm.weight"] = rsd["q_norm.weight"]
    sd[f"{p}.k_norm.weight"] = rsd["k_norm.weight"]

    os.environ["USE_DEVICE_ATTENTION"] = "1" if use_device_attn else "0"
    tt = TtGatedAttention(device, sd, layer_idx=0, config=cfg, dtype=ttnn.bfloat16)
    if hasattr(tt, "reset"):
        tt.reset()

    cos_c, sin_c = build_rope(cfg.rotary_dim, cfg.max_seq_len, cfg.rope_theta)

    # Fixed inputs so CPU and device runs see identical sequences.
    torch.manual_seed(1234)
    xs = [torch.randn(1, 1, cfg.hidden_size) for _ in range(6)]

    ref_kv = None
    tt_kv = None
    max_diffs = []
    outs = []
    for t, x in enumerate(xs):
        cos_t = cos_c[:, :, t:t + 1, :]
        sin_t = sin_c[:, :, t:t + 1, :]
        with torch.no_grad():
            ref_out, ref_kv = ref(x, cos_t, sin_t, kv_cache=ref_kv)

        x_tt = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt, tt_kv = tt(x_tt, cos_t, sin_t, kv_cache=tt_kv, mode="decode", current_pos=t)
        out_cpu = ttnn.to_torch(out_tt).float().reshape_as(ref_out)
        outs.append(out_cpu)

        md = (out_cpu - ref_out.float()).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_cpu.flatten().unsqueeze(0), ref_out.float().flatten().unsqueeze(0)
        ).item()
        print(f"  step {t}: vs-ref max_diff={md:.4e} cos={cos_sim:.5f}", flush=True)
        max_diffs.append(md)
    return max(max_diffs), outs


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        print("=== CPU-fallback attention vs reference ===", flush=True)
        d_cpu, outs_cpu = _run(dev, use_device_attn=False)
        print(f"CPU path worst vs-ref max_diff={d_cpu:.4e}", flush=True)
        if os.environ.get("TEST_DEVICE_ATTN", "1") == "1":
            print("\n=== device attention vs reference ===", flush=True)
            try:
                d_dev, outs_dev = _run(dev, use_device_attn=True)
                print(f"DEVICE path worst vs-ref max_diff={d_dev:.4e}", flush=True)
                # The real correctness criterion: device path must match the known-good
                # CPU path (both are TT bf16).
                print("\n=== device vs CPU path (should be ~0) ===", flush=True)
                for t, (a, b) in enumerate(zip(outs_dev, outs_cpu)):
                    md = (a - b).abs().max().item()
                    cs = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
                    print(f"  step {t}: dev-vs-cpu max_diff={md:.4e} cos={cs:.5f}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"DEVICE path ERROR: {e}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
