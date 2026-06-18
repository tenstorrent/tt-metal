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
    ap.add_argument("--trace", action="store_true", help="enable_trace=True for decode (trace capture/replay)")
    ap.add_argument("--b2", action="store_true", help="batched num_seq=2 test: prefill 2 prompts, decode both")
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
            hf, md, max_batch_size=(2 if args.b2 else 1), max_seq_len=args.max_seq_len, n_layers=args.layers,
        )
        print("[val] model built. allocating kv cache pool ...", flush=True)
        # minimal paged pool (model self-manages the contiguous cache): (blocks, nkv, block, hd)
        kv = gen.allocate_kv_cache((8, 4, 64, 256), ttnn.bfloat16, num_layers=args.layers // 4)
        nb = max(1, (args.max_seq_len + 63) // 64)

        if args.b2:
            prompts = [args.prompt, "The largest planet in our solar system is"]
            ids2 = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]
            Ls = [len(x) for x in ids2]
            maxL = max(Ls)
            toks2 = torch.zeros(2, maxL, dtype=torch.long)
            for r, x in enumerate(ids2):
                toks2[r, :len(x)] = torch.tensor(x)
            pt2 = torch.stack([torch.arange(r * nb, r * nb + nb, dtype=torch.int32) for r in range(2)])  # distinct blocks
            print("[val] prefill 2 prompts ...", flush=True)
            logits = gen.prefill_forward(toks2, page_table=pt2, kv_cache=kv, start_pos=[0, 0], prompt_lens=Ls)
            nxt = [int(torch.argmax(logits[r, 0]).item()) for r in range(2)]
            print(f"[val] B2 prefill next: {[(n, tok.decode([n])) for n in nxt]}", flush=True)
            outs = [[n] for n in nxt]
            import time as _time
            step_ms = []
            for s in range(args.gen):
                pos = torch.tensor([Ls[0] + s, Ls[1] + s])
                _t0 = _time.perf_counter()
                dl = gen.decode_forward(torch.tensor([[outs[0][-1]], [outs[1][-1]]]),
                                        page_table=pt2, kv_cache=kv, start_pos=pos, enable_trace=args.trace)
                host = gen.process_decode_output_host(gen.read_decode_output(dl))
                step_ms.append((_time.perf_counter() - _t0) * 1000)
                for r in range(2):
                    outs[r].append(int(torch.argmax(host[r, 0]).item()))
            import statistics as _st
            med = _st.median(step_ms[2:] if len(step_ms) > 3 else step_ms)
            print(f"[val] B2 decode step ms: {[round(x) for x in step_ms]} | median={med:.0f} ms/step "
                  f"=> {2000.0/med:.2f} tok/s aggregate (B=2)", flush=True)
            for r in range(2):
                print(f"[val] B2 GEN[{r}]: {prompts[r]!r} + {tok.decode(outs[r])!r}", flush=True)
            print("VAL_DONE", flush=True)
            return

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
        import time as _time
        step_ms = []
        for s in range(args.gen):
            pos = L + s
            _t0 = _time.perf_counter()
            dl = gen.decode_forward(torch.tensor([[nxt]]), page_table=page_table, kv_cache=kv,
                                    start_pos=torch.tensor([pos]), enable_trace=args.trace)
            host = gen.process_decode_output_host(gen.read_decode_output(dl))
            step_ms.append((_time.perf_counter() - _t0) * 1000)
            lv = host[0, 0]
            nxt = int(torch.argmax(lv).item())
            out.append(nxt)
        # first 1-2 steps include trace capture; report steady-state median of the rest
        warm = step_ms[2:] if len(step_ms) > 3 else step_ms
        import statistics as _st
        print(f"[val] decode step ms: {[round(x) for x in step_ms]} | steady median={_st.median(warm):.0f} ms/tok "
              f"(trace={args.trace})", flush=True)
        print(f"[val] decode {args.gen} steps -> ids {out}", flush=True)
        print(f"[val] GENERATION: {args.prompt!r} + {tok.decode(out)!r}", flush=True)
        print("VAL_DONE", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
