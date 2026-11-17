"""WS8: prefix-cache continuation speed (full 30-layer model, one user).

  python ws8_bench.py <out.json> [--prefix 8192] [--suffixes 512,2048] [--serial-tokens 32]

Prefills `prefix` tokens, then continues with each suffix through Gemma4Generator.prefill_forward(start_pos=...) as
the vLLM prefix-cache path does: rows (WS8) and, for `serial-tokens` tokens only, the token-at-a-time path it
replaces. Also times a cold prefill of prefix+suffix for comparison. Each timing repeats once after a warm-up run
(the first run compiles programs)."""
import argparse
import json
import time

import torch

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt import generator as generator_module
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator
from models.demos.gemma4_26b_a4b_qb2.tt.generator_vllm import Gemma4ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--prefix", type=int, default=8192)
ap.add_argument("--suffixes", default="512,2048")
ap.add_argument("--serial-tokens", type=int, default=32)
ap.add_argument("--text", default="", help="real-text token source (a text file) instead of random ids")
ap.add_argument("--check-only", action="store_true", help="only the rows / serial / cold logit agreement")
args = ap.parse_args()
suffixes = [int(s) for s in args.suffixes.split(",")]

ttnn.set_fabric_config(**Gemma4ForCausalLM.model_capabilities["fabric_config"])
# The serving mesh parameters (as decode_timing.py / vLLM): no L1_SMALL region. With l1_small_size the full model's
# persistent L1 (WS5 boundary buffers) sits 16 KB lower and long prefills clash with it.
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_000_000_000)
results = {"prefix": args.prefix}
try:
    total = args.prefix + max(suffixes) + 64
    gen = Gemma4Generator(mesh, max_seq_len=total)
    model = gen.model
    torch.manual_seed(0)
    tokens = torch.randint(10, 100000, (1, total))
    if args.text:
        from transformers import AutoTokenizer

        from models.demos.gemma4_26b_a4b_qb2.tt.model import MODEL_ID, REVISION

        ids = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)(open(args.text).read())["input_ids"]
        assert len(ids) >= total, f"text has {len(ids)} tokens, need {total}"
        tokens = torch.tensor(ids[:total]).reshape(1, total)
    caches, table = model.allocate_cache(slots=1, context=total)

    def sync():
        ttnn.synchronize_device(mesh)

    def timed(fn):
        sync()
        t = time.perf_counter()
        out = fn()
        gen._read_logits(out)
        return time.perf_counter() - t

    def prefill(end):
        return gen.prefill_forward(tokens[:, :end], page_table=table, kv_cache=caches, prompt_lens=[end])

    def cont(start, end):
        return gen.prefill_forward(
            tokens[:, :end], page_table=table, kv_cache=caches, prompt_lens=[end], start_pos=[start]
        )

    p = args.prefix
    # Same suffix both ways: full-model last-logit agreement of the rows and serial continuation.
    last = {}
    for rows in (True, False):
        generator_module.WS8_ROWS = rows
        prefill(p)
        last[rows] = gen._read_logits(cont(p, p + args.serial_tokens)).reshape(-1).clone()
    prefill(p + args.serial_tokens)
    last["cold"] = gen._read_logits(prefill(p + args.serial_tokens)).reshape(-1).clone()

    def agree(x, y):
        x, y = x.double(), y.double()
        return {
            "pcc": round(torch.corrcoef(torch.stack([x, y]))[0, 1].item(), 6),
            "top1_equal": int(x.argmax()) == int(y.argmax()),
            "top5_overlap": len(set(x.topk(5).indices.tolist()) & set(y.topk(5).indices.tolist())),
        }

    results["last_logits"] = {
        "suffix": args.serial_tokens,
        "text": bool(args.text),
        "rows_vs_serial": agree(last[True], last[False]),
        "rows_vs_cold": agree(last[True], last["cold"]),
        "serial_vs_cold": agree(last[False], last["cold"]),
    }
    print("last_logits", results["last_logits"], flush=True)
    if args.check_only:
        raise SystemExit
    for rows in (True, False):
        generator_module.WS8_ROWS = rows
        for s in suffixes if rows else [args.serial_tokens]:
            prefill(p)
            timed(lambda: cont(p, p + s))  # warm-up / compile
            prefill(p)
            dt = timed(lambda: cont(p, p + s))
            key = f"{'rows' if rows else 'serial'}_{s}"
            results[key] = {"seconds": round(dt, 3), "tok_s": round(s / dt, 1)}
            print(key, results[key], flush=True)
    generator_module.WS8_ROWS = True
    for s in suffixes:
        prefill(p + s)
        dt = timed(lambda: prefill(p + s))
        results[f"cold_prefill_{p + s}"] = {"seconds": round(dt, 3), "tok_s": round((p + s) / dt, 1)}
        print(f"cold_prefill_{p + s}", results[f"cold_prefill_{p + s}"], flush=True)
    gen._release_trace()
finally:
    json.dump(results, open(args.out, "w"), indent=1)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
