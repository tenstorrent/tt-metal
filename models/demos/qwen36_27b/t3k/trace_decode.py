# SPDX-License-Identifier: Apache-2.0
"""
Trace-based single-chip decode for Qwen3.6-27B.

Captures the all-device decode forward (on-device attention cached path + DeltaNet
full-fused with in-place state) into a trace and replays it, so per-token host
dispatch across all 64 layers is eliminated. Measures TPOT vs the eager baseline.

  QWEN_ONDEVICE_ATTN=1 python3 trace_decode.py --dummy --layers 8 --steps 16
  QWEN_ONDEVICE_ATTN=1 python3 trace_decode.py --model-path /home/ttuser/ttwork/qwen36-weights --steps 32
"""
import argparse, time, statistics
import torch
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict, create_dummy_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator


def rope_at(cfg, pos):
    dim = cfg.rotary_dim
    freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    f = torch.outer(torch.arange(cfg.max_seq_len).float(), freqs)
    cos = f.cos().reshape(1, 1, -1, dim // 2).repeat(1, 1, 1, 2)
    sin = f.sin().reshape(1, 1, -1, dim // 2).repeat(1, 1, 1, 2)
    return cos[:, :, pos:pos + 1, :].to(torch.bfloat16), sin[:, :, pos:pos + 1, :].to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--model-path", type=str, default="/home/ttuser/ttwork/qwen36-weights")
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    cfg = Qwen36ModelConfig()
    if args.layers is not None:
        cfg.num_hidden_layers = args.layers
    nkv, hd = cfg.num_key_value_heads, cfg.head_dim

    if args.dummy:
        sd = create_dummy_state_dict(cfg, num_layers=cfg.num_hidden_layers)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    else:
        sd = load_state_dict(cfg, max_layers=args.layers, model_path=args.model_path)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    S = input_ids.shape[1]

    dev = ttnn.open_device(device_id=0)
    try:
        model = TtQwen36Model(dev, sd, cfg)
        gen = Qwen36Generator(model, cfg)
        del sd

        # --- prefill (CPU attention prefill -> torch KV; DeltaNet state on device) ---
        last_logits = gen.prefill(input_ids)
        nxt = int(torch.argmax(ttnn.to_torch(last_logits).float().reshape(-1)[: cfg.vocab_size]))

        # --- build preallocated device KV caches from prefill torch KV ---
        attn_layers = [i for i, t in enumerate(cfg.layer_types[: cfg.num_hidden_layers]) if t == "full_attention"]
        dev_kv = {}
        for i in attn_layers:
            k_t, v_t = gen.kv_caches[i]  # [1, nkv, S, hd] torch (post-rope)
            kc = torch.zeros(1, nkv, args.max_seq, hd); kc[:, :, :S, :] = k_t
            vc = torch.zeros(1, nkv, args.max_seq, hd); vc[:, :, :S, :] = v_t
            dev_kv[i] = (
                ttnn.from_torch(kc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(vc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            )

        # --- device buffers: cur_pos, cos/sin ---
        cur_pos = ttnn.from_torch(torch.tensor([S], dtype=torch.int32), device=dev)
        cos0, sin0 = rope_at(cfg, S)
        cos_buf = ttnn.from_torch(cos0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        sin_buf = ttnn.from_torch(sin0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        # --- enable trace-decode mode ---
        gen.deltanet_state.trace_mode = True
        model.trace_decode = True
        model.trace_cos, model.trace_sin = cos_buf, sin_buf
        for i in attn_layers:
            model.layers[i].token_mixer.trace_decode = True
            model.layers[i].token_mixer.cur_pos_tt = cur_pos

        def embed_host(tok_id):
            # host-side embedding [1,1,1,H] to copy into the persistent trace input buffer
            e = model.embedding_weight[torch.tensor([[tok_id]])]  # [1,1,H]
            return ttnn.from_torch(e.unsqueeze(0), dtype=model.dtype, layout=ttnn.TILE_LAYOUT)

        def set_pos(p):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor([p], dtype=torch.int32)), cur_pos)
            cphost, sphost = rope_at(cfg, p)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(cphost, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), cos_buf)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(sphost, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sin_buf)

        # --- warmup eager decode (compiles kernels) ---
        pos = S
        for _ in range(3):
            set_pos(pos)
            emb = model.embed(torch.tensor([[nxt]]))
            logits, _ = model.forward_from_embedding(emb, pos, gen.deltanet_state, dev_kv)
            nxt = int(torch.argmax(ttnn.to_torch(logits).float().reshape(-1)[: cfg.vocab_size]))
            pos += 1
        ttnn.synchronize_device(dev)

        # --- capture trace ---
        set_pos(pos)
        emb_buf = model.embed(torch.tensor([[nxt]]))
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        trace_logits, _ = model.forward_from_embedding(emb_buf, pos, gen.deltanet_state, dev_kv)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        print(f"[trace] captured at pos={pos}", flush=True)

        # --- replay ---
        step_ms = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(embed_host(nxt), emb_buf)
            set_pos(pos)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            logits_cpu = ttnn.to_torch(trace_logits).float().reshape(-1)
            nxt = int(torch.argmax(logits_cpu[: cfg.vocab_size]))
            step_ms.append((time.perf_counter() - t0) * 1000)
            pos += 1

        med = statistics.median(step_ms)
        print(f"[trace decode] {len(step_ms)} steps, {cfg.num_hidden_layers} layers", flush=True)
        print(f"  median {med:.1f} ms/tok ({1000/med:.2f} tok/s), min {min(step_ms):.1f} ms", flush=True)
        print(f"  (baseline eager batch=1 full model was 171.8 ms = 5.82 tok/s)", flush=True)
        ttnn.release_trace(dev, tid)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
