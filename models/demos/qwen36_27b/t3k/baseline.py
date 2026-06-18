# SPDX-License-Identifier: Apache-2.0
"""
Single-chip baseline for Qwen3.6-27B on one P300 chip (fabric disabled).

Measures prefill (TTFT) and decode TPOT for batch=1. This is the reference point
for the P300x2 / batch=8 work.

  python3 baseline.py --dummy --layers 4 --max-tokens 8
  python3 baseline.py --model-path /home/ttuser/ttwork/qwen36-weights --max-tokens 32 --prompt "..."

Run inside the qwen36-test container with ONE chip exposed and:
  TT_MESH_GRAPH_DESC_PATH=<repo>/.../p150_mesh_graph_descriptor.textproto
  TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1
"""
import argparse, sys, time
import torch
import ttnn

# Single chip on a P300 board: disable fabric so open_device doesn't try to bring up
# the (uncabled / unexposed-partner) ethernet links.
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict, create_dummy_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="dummy random weights")
    ap.add_argument("--layers", type=int, default=None, help="limit num layers")
    ap.add_argument("--model-path", type=str, default="/home/ttuser/ttwork/qwen36-weights")
    ap.add_argument("--prompt", type=str, default="Explain what a transformer is in two sentences.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    config = Qwen36ModelConfig()
    if args.layers is not None:
        config.num_hidden_layers = args.layers

    # tokenizer + input
    if args.dummy:
        tokenizer = None
        input_ids = torch.tensor([[1, 42, 100, 7, 88, 9, 14, 3]])
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        messages = [{"role": "user", "content": args.prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(formatted, return_tensors="pt")
    prompt_len = input_ids.shape[1]

    # weights
    if args.dummy:
        print(f"[weights] dummy, {config.num_hidden_layers} layers", flush=True)
        state_dict = create_dummy_state_dict(config, num_layers=config.num_hidden_layers)
    else:
        print(f"[weights] loading from {args.model_path} ...", flush=True)
        t0 = time.time()
        state_dict = load_state_dict(config, max_layers=args.layers, model_path=args.model_path)
        print(f"[weights] loaded {len(state_dict)} tensors in {time.time()-t0:.1f}s", flush=True)

    print(f"[device] opening chip {args.device_id} ...", flush=True)
    dev = ttnn.open_device(device_id=args.device_id)
    try:
        t0 = time.time()
        model = TtQwen36Model(dev, state_dict, config)
        gen = Qwen36Generator(model, config, tokenizer=tokenizer)
        print(f"[model] built in {time.time()-t0:.1f}s", flush=True)
        del state_dict

        # prefill
        t0 = time.time()
        last_logits = gen.prefill(input_ids)
        ttnn.synchronize_device(dev)
        prefill_s = time.time() - t0
        logits_cpu = ttnn.to_torch(last_logits).float().reshape(-1)
        next_token = int(torch.argmax(logits_cpu[: config.vocab_size]))
        print(f"[prefill] {prompt_len} tok in {prefill_s:.2f}s -> {prompt_len/prefill_s:.1f} tok/s (TTFT)", flush=True)

        # decode loop with per-token timing
        gen_ids = [next_token]
        step_times = []
        for i in range(args.max_tokens - 1):
            tok = torch.tensor([[next_token]], dtype=torch.long)
            t0 = time.time()
            _, nt = gen.decode_one_token(tok)
            ttnn.synchronize_device(dev)
            step_times.append(time.time() - t0)
            next_token = int(nt.item())
            gen_ids.append(next_token)
            if tokenizer is not None and next_token == tokenizer.eos_token_id:
                break

        if step_times:
            import statistics
            avg = sum(step_times) / len(step_times)
            med = statistics.median(step_times)
            print(f"[decode] {len(step_times)} steps: avg {avg*1000:.1f}ms/tok ({1/avg:.2f} tok/s), "
                  f"median {med*1000:.1f}ms ({1/med:.2f} tok/s)", flush=True)
            print(f"[decode] TPOT target 20 tok/s = 50ms/step; baseline batch=1 = {1/avg:.2f} tok/s", flush=True)
        if tokenizer is not None:
            print("[output]", tokenizer.decode(gen_ids, skip_special_tokens=True)[:300], flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
