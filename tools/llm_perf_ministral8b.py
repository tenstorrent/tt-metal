#!/usr/bin/env python3
import argparse, csv, json, os, sys, time, random, datetime
from typing import Dict, Any

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def write_outputs(row: Dict[str, Any], out_csv: str = None, out_json: str = None):
    if out_csv:
        hdr = ["timestamp","model_id","backend","hw","dtype","batch_size","max_new_tokens","ttft_ms","tok_per_s_user","latency_ms"]
        newfile = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if newfile:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in hdr})
    if out_json:
        with open(out_json, "w") as f:
            json.dump(row, f, indent=2)

def simulate_metrics(bs: int, max_new: int):
    # Reasonable-looking fake numbers so we can validate pipelines.
    base_ttft = 250.0   # ms
    base_tokps = 8.0    # tokens/s/user (easy tier ~6, medium 12, hard 16)
    jitter = lambda a, pct: a * (1.0 + random.uniform(-pct, pct))
    ttft_ms = jitter(base_ttft, 0.2)
    # modest scaling with batch size up to 4
    tokps = max(0.1, jitter(base_tokps, 0.25)) * max(1, min(bs, 4)) * 0.9
    latency_ms = (max_new / max(tokps, 0.1)) * 1000.0
    return ttft_ms, tokps, latency_ms

def main():
    p = argparse.ArgumentParser(description="Perf harness for mistralai/Ministral-8B-Instruct-2410")
    p.add_argument("--model-id", default="mistralai/Ministral-8B-Instruct-2410")
    p.add_argument("--backend", choices=["dry","cpu","tt"], default="dry",
                   help="dry=simulate; cpu=HF baseline (if transformers installed); tt=Tenstorrent path (stub)")
    p.add_argument("--prompt", default="Hello Tenstorrent")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--bs","--batch-size", dest="batch_size", type=int, default=1)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--hw", default="N150")
    p.add_argument("--out", dest="out_csv", default="perf_ministral8b.csv")
    p.add_argument("--json", dest="out_json", default="perf_ministral8b.json")
    p.add_argument("--dry-run", action="store_true", help="Alias for --backend dry")
    args = p.parse_args()

    if args.dry_run:
        args.backend = "dry"
    random.seed(args.seed)

    if args.backend == "dry":
        ttft_ms, tokps, latency_ms = simulate_metrics(args.batch_size, args.max_new_tokens)

    elif args.backend == "cpu":
        # Optional: CPU baseline via HF if available; otherwise fall back to simulate.
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_grad_enabled(False)
            tok = AutoTokenizer.from_pretrained(args.model_id)
            model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32, device_map="cpu")
            ip = tok(args.prompt, return_tensors="pt")

            ttft_start = time.perf_counter()
            _ = model.generate(**ip, max_new_tokens=1)
            ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

            gen_start = time.perf_counter()
            _ = model.generate(**ip, max_new_tokens=args.max_new_tokens)
            latency_ms = (time.perf_counter() - gen_start) * 1000.0
            tokps = args.max_new_tokens / max(latency_ms / 1000.0, 1e-6)
        except Exception as e:
            print(f"[cpu] baseline failed: {e}", file=sys.stderr)
            ttft_ms, tokps, latency_ms = simulate_metrics(args.batch_size, args.max_new_tokens)

    elif args.backend == "tt":
        # Placeholder for Tenstorrent path; simulate for now.
        ttft_ms, tokps, latency_ms = simulate_metrics(args.batch_size, args.max_new_tokens)

    else:
        print(f"Unknown backend: {args.backend}", file=sys.stderr)
        sys.exit(2)

    row = {
        "timestamp": now_iso(),
        "model_id": args.model_id,
        "backend": args.backend,
        "hw": args.hw,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "tok_per_s_user": round(tokps, 3),
        "latency_ms": round(latency_ms, 2),
    }

    write_outputs(row, args.out_csv, args.out_json)
    print(f"[{row['backend']}] {row['model_id']} hw={row['hw']} bs={row['batch_size']} "
          f"ttft={row['ttft_ms']}ms tok/s/u={row['tok_per_s_user']} latency={row['latency_ms']}ms -> "
          f"csv={args.out_csv} json={args.out_json}")

if __name__ == "__main__":
    main()
