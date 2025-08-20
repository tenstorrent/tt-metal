#!/usr/bin/env python3
import sys, json

PREFIXES = (
    "/home/ubuntu/tt-metal/tt_metal",
    "/home/ubuntu/tt-metal/ttnn",
)


def main(src, dst):
    with open(src, "r") as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_val = obj.get("file")
            if isinstance(file_val, str) and any(p in file_val for p in PREFIXES):
                fout.write(line)


if __name__ == "__main__":
    main("execution_trace_clang.jsonl", "execution_trace_clang_filtered.jsonl")
