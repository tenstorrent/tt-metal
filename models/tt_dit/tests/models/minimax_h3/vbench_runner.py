# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Run VBench on an mp4 in a separate interpreter, and report the scores as JSON on stdout.

VBench cannot share `python_env`. Installing it there would downgrade **numpy 2.2.6 -> 1.26.4** and
**transformers 5.12.1 -> 4.33.2** (plus huggingface-hub, tokenizers and timm), which breaks `ttnn`'s
numpy ABI and destroys the Qwen3-VL reference the text-encoder gate depends on.

That is not a compromise, because VBench evaluates a *file*: it needs no mesh, no ttnn, and nothing
the pipeline produced in memory. So generation runs in `python_env` and evaluation runs in an
interpreter that owns its own numpy -- which is the right split regardless of the conflict.

Usage (from the eval venv):
    python vbench_runner.py <video.mp4> <dimension>[,<dimension>...] [--prompt "..."]

Create the venv with:
    uv venv --python 3.10 /data/kevinmi/vbench_env
    uv pip install --python /data/kevinmi/vbench_env/bin/python vbench decord
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("dimensions", help="comma-separated VBench dimension names")
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    # VBench 0.1.5 checkpoints hold typing.OrderedDict, which torch.load's weights_only=True
    # default (PyTorch 2.6+) rejects.
    import typing

    import torch
    from vbench import VBench

    torch.serialization.add_safe_globals([typing.OrderedDict])

    dimensions = [d for d in args.dimensions.split(",") if d]
    if not dimensions:
        print(json.dumps({"error": "no dimensions requested"}), flush=True)
        return 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        name = "eval"
        bench = VBench(device="cpu", full_info_dir="", output_path=tmp_dir)
        bench.evaluate(
            videos_path=args.video,
            name=name,
            dimension_list=dimensions,
            prompt_list=[args.prompt] if args.prompt is not None else [],
            mode="custom_input",
        )
        with open(os.path.join(tmp_dir, f"{name}_eval_results.json")) as handle:
            raw = json.load(handle)

    scores = {metric: value[0] for metric, value in raw.items()}
    # The only thing on stdout that the caller parses. VBench itself is chatty on stderr.
    print("VBENCH_JSON " + json.dumps(scores), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
