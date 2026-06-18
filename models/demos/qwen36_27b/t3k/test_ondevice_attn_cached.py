# SPDX-License-Identifier: Apache-2.0
"""PCC: trace-friendly cached on-device decode (paged_update_cache + sdpa_decode) vs CPU."""
import torch
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.attention import TtGatedAttention


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def build_cos_sin(cfg, pos):
    dim = cfg.rotary_dim
    freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    f = torch.outer(torch.arange(cfg.max_seq_len).float(), freqs)
    cos = f.cos().reshape(1, 1, -1, dim // 2).repeat(1, 1, 1, 2)
    sin = f.sin().reshape(1, 1, -1, dim // 2).repeat(1, 1, 1, 2)
    return cos[:, :, pos:pos + 1, :], sin[:, :, pos:pos + 1, :]


def main():
    cfg = Qwen36ModelConfig()
    H, nh, nkv, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    li = 3
    max_seq = 256
    torch.manual_seed(0)
    sd = {
        f"model.layers.{li}.self_attn.q_proj.weight": torch.randn(nh * hd * 2, H) * 0.02,
        f"model.layers.{li}.self_attn.k_proj.weight": torch.randn(nkv * hd, H) * 0.02,
        f"model.layers.{li}.self_attn.v_proj.weight": torch.randn(nkv * hd, H) * 0.02,
        f"model.layers.{li}.self_attn.o_proj.weight": torch.randn(H, nh * hd) * 0.02,
        f"model.layers.{li}.self_attn.q_norm.weight": torch.randn(hd) * 0.1,
        f"model.layers.{li}.self_attn.k_norm.weight": torch.randn(hd) * 0.1,
    }

    dev = ttnn.open_device(device_id=0)
    try:
        attn = TtGatedAttention(dev, sd, li, cfg, weights_dtype=ttnn.bfloat16)
        past_len = 8
        hidden = torch.randn(1, 1, 1, H) * 0.5
        past_k = torch.randn(1, nkv, past_len, hd) * 0.3
        past_v = torch.randn(1, nkv, past_len, hd) * 0.3
        cos_p, sin_p = build_cos_sin(cfg, past_len)
        hidden_tt = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        # CPU reference
        out_cpu, _ = attn._decode(hidden_tt, cos_p, sin_p, (past_k.clone(), past_v.clone()))
        out_cpu = ttnn.to_torch(out_cpu).float()

        # cached on-device: preload caches with past at positions [0, past_len)
        kc = torch.zeros(1, nkv, max_seq, hd); kc[:, :, :past_len, :] = past_k
        vc = torch.zeros(1, nkv, max_seq, hd); vc[:, :, :past_len, :] = past_v
        k_cache = ttnn.from_torch(kc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        v_cache = ttnn.from_torch(vc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cos_tt = ttnn.from_torch(cos_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        sin_tt = ttnn.from_torch(sin_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cur_pos = ttnn.from_torch(torch.tensor([past_len], dtype=torch.int32), device=dev)

        out_dev, = (attn._decode_ondevice_cached(hidden_tt, cos_tt, sin_tt, k_cache, v_cache, cur_pos),)
        out_dev = ttnn.to_torch(out_dev).float()

        p = pcc(out_cpu, out_dev)
        print(f"[shapes] cpu={tuple(out_cpu.shape)} dev={tuple(out_dev.shape)}", flush=True)
        print(f"[PCC] cached on-device vs CPU = {p:.5f}", flush=True)
        print("PASS" if p > 0.98 else "FAIL", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
