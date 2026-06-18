# SPDX-License-Identifier: Apache-2.0
"""Standalone HW validation of the qwen36_27b vLLM+TP=8 model (no vLLM runtime).

Loads the REAL Qwen3.6-27B weights, builds TtQwen36VllmModel (TP=8, 1x8 line),
and drives generator_vllm's prefill/decode directly to check the port runs and
produces sane tokens on T3K.

  MESH_DEVICE=T3K python3 models/demos/qwen36_27b/t3k/validate_vllm_model.py \
      --prompt "The capital of France is" --gen 8 --layers 64
"""
import argparse, os
from types import SimpleNamespace
import torch
import ttnn

CKPT = "/home/yito/work/qwen36_27b_hf"


def build_hf_config():
    tc = SimpleNamespace(
        hidden_size=5120, num_hidden_layers=64, vocab_size=248320,
        num_attention_heads=24, num_key_value_heads=4, head_dim=256,
        full_attention_interval=4, intermediate_size=17408,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25, rms_norm_eps=1e-6,
        rope_parameters=SimpleNamespace(rope_theta=1.0e7),
    )
    return SimpleNamespace(text_config=tc, _name_or_path=CKPT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=8, help="decode steps")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    from models.demos.qwen36_27b.tt.generator_vllm import Qwen3_5ForConditionalGeneration

    # tokenizer from the HF checkpoint dir
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    print(f"[val] prompt={args.prompt!r} -> {len(ids)} tokens: {ids}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
    try:
        hf = build_hf_config()
        print("[val] initialize_vllm_model (load 52GB weights, TP=8 shard) ...", flush=True)
        gen = Qwen3_5ForConditionalGeneration.initialize_vllm_model(
            hf, md, max_batch_size=1, max_seq_len=args.max_seq_len, n_layers=args.layers,
        )
        print("[val] model built. allocating kv cache pool ...", flush=True)
        # minimal paged pool (model self-manages the contiguous cache): (blocks, nkv, block, hd)
        kv = gen.allocate_kv_cache((8, 4, 64, 256), ttnn.bfloat16, num_layers=args.layers // 4)

        L = len(ids)
        nb = max(1, (args.max_seq_len + 63) // 64)
        page_table = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
        toks = torch.tensor(ids, dtype=torch.long).reshape(1, L)

        print("[val] prefill ...", flush=True)
        logits = gen.prefill_forward(toks, page_table=page_table, kv_cache=kv,
                                     start_pos=[0], prompt_lens=[L])
        lv = logits[0, 0]
        nxt = int(torch.argmax(lv).item())
        top5 = torch.topk(lv, 5).indices.tolist()
        print(f"[val] prefill OK. finite={bool(torch.isfinite(lv).all())} "
              f"next={nxt} ({tok.decode([nxt])!r}) top5={[tok.decode([t]) for t in top5]}", flush=True)

        out = [nxt]
        for s in range(args.gen):
            pos = L + s
            dl = gen.decode_forward(torch.tensor([[nxt]]), page_table=page_table, kv_cache=kv,
                                    start_pos=torch.tensor([pos]))
            host = gen.process_decode_output_host(gen.read_decode_output(dl))
            lv = host[0, 0]
            nxt = int(torch.argmax(lv).item())
            out.append(nxt)
        print(f"[val] decode {args.gen} steps -> ids {out}", flush=True)
        print(f"[val] GENERATION: {args.prompt!r} + {tok.decode(out)!r}", flush=True)
        print("VAL_DONE", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
