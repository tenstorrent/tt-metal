"""Inventory of method calls on untyped receivers.

Reads `py_index.json` and reports every ref where:
  - The call is method-style (chain has at least 2 components)
  - The receiver (chain[0]) is NOT a known module or builtin
  - The receiver is NOT `self` / `cls`
  - No `receiver_type` is set (i.e. the propagator either didn't run or
    couldn't infer a type for this receiver)

This is the test surface for the type propagator: every entry here is a
call site where the graph DOESN'T currently have an outgoing edge for the
method call, and where a propagator MIGHT add one if it can infer the
receiver type.

Output:
  - Total count of untyped method calls
  - Breakdown by (receiver_name, method_name) — to see hot patterns
  - Per-method tally — to see which methods are called the most without types
  - Sample call sites (file:line) so you can spot-check

Run:
  python find_untyped_method_calls.py /workspace/dep-graph/cache/py_index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# Receivers we deliberately ignore: stdlib, external libs, builtins,
# anything that's not a candidate for type propagation.
SKIP_RECEIVERS = {
    # stdlib
    "os", "sys", "re", "time", "json", "random", "math", "collections",
    "itertools", "functools", "tempfile", "pathlib", "unittest", "typing",
    "subprocess", "enum", "warnings", "datetime", "argparse", "shutil",
    "csv", "pickle", "copy", "string", "io", "glob", "hashlib", "logging",
    "inspect", "traceback", "gc", "dataclasses", "signal", "asyncio",
    "queue", "threading", "urllib", "base64", "contextlib", "dis",
    "tokenize", "weakref", "ssl", "socket", "operator", "struct",
    # external
    "pytest", "logger", "torch", "numpy", "np", "pandas", "pd", "matplotlib",
    "plt", "PIL", "Image", "scipy", "sklearn", "requests", "tqdm", "tabulate",
    "HfApi", "huggingface_hub", "torchvision", "torchaudio", "librosa",
    "torchtune", "transformers", "diffusers", "tokenizers", "wandb",
    "lightning", "accelerate", "onnx", "onnxruntime", "sentencepiece",
    "safetensors", "rich", "timm", "sentence_transformers", "tracy", "loguru",
    # tt-metal modules (already handled by other paths in the stitcher)
    "ttnn", "tt_metal",
    # self/cls and common patterns we already handle
    "self", "cls",
    # call results (these are inline-chain patterns, separate issue)
    # — these don't appear as chain[0] anyway, just skip
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("py_index", type=Path,
                    help="Path to py_index.json from py_index.py")
    ap.add_argument("--samples", type=int, default=5,
                    help="Number of sample call sites to show per pattern")
    ap.add_argument("--top", type=int, default=20,
                    help="Show top N patterns by frequency")
    ap.add_argument("--receiver", type=str, default=None,
                    help="Filter to a specific receiver name (e.g. 'device')")
    ap.add_argument("--method", type=str, default=None,
                    help="Filter to a specific method name (e.g. 'compute_with_storage_grid_size')")
    args = ap.parse_args()

    data = json.loads(args.py_index.read_text())
    refs = data.get("refs", [])

    untyped: list[dict] = []
    for r in refs:
        chain = r.get("target_chain") or []
        if len(chain) < 2:
            continue
        receiver = chain[0]
        if receiver in SKIP_RECEIVERS:
            continue
        if r.get("receiver_type"):
            continue
        if args.receiver and receiver != args.receiver:
            continue
        if args.method and chain[1] != args.method:
            continue
        untyped.append(r)

    print(f"Found {len(untyped)} untyped method-style calls (after filters)")
    print()

    # Pattern breakdown
    by_pair = Counter()
    by_method = Counter()
    by_receiver = Counter()
    samples_by_pair: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for r in untyped:
        chain = r["target_chain"]
        recv, method = chain[0], chain[1]
        by_pair[(recv, method)] += 1
        by_method[method] += 1
        by_receiver[recv] += 1
        key = (recv, method)
        if key not in samples_by_pair:
            samples_by_pair[key] = []
        if len(samples_by_pair[key]) < args.samples:
            samples_by_pair[key].append((r["site_file"], r["site_line"]))

    print(f"=== Top {args.top} (receiver, method) pairs ===")
    for (recv, method), n in by_pair.most_common(args.top):
        print(f"  {n:>5d}  {recv}.{method}")
        for site_file, site_line in samples_by_pair[(recv, method)]:
            short = site_file.split("/")[-1]
            print(f"           {short}:{site_line}")

    print()
    print(f"=== Top {args.top} receivers (just by name) ===")
    for recv, n in by_receiver.most_common(args.top):
        print(f"  {n:>5d}  {recv}")

    print()
    print(f"=== Top {args.top} methods (just by name) ===")
    for method, n in by_method.most_common(args.top):
        print(f"  {n:>5d}  .{method}")

    # Summary suitable for diffing before/after a propagator change
    print()
    print(f"=== Summary ===")
    print(f"  Total untyped method calls: {len(untyped)}")
    print(f"  Distinct (receiver, method) pairs: {len(by_pair)}")
    print(f"  Distinct methods: {len(by_method)}")
    print(f"  Distinct receiver names: {len(by_receiver)}")


if __name__ == "__main__":
    main()
