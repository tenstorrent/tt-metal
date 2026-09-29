"""Add explicit request identity to the unmodified upstream vLLM 0.26 client."""

import os
import sys
from pathlib import Path

args = sys.argv[1:]
name = args[args.index("--result-filename") + 1].removesuffix(".json")
root = Path(__file__).resolve().parents[5]
exe = root / "benchmark-env/bin/vllm"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_TARGET_DEVICE"] = "empty"
os.environ.pop("PYTHONPATH", None)
os.execv(
    str(exe),
    [
        str(exe),
        "bench",
        "serve",
        *args,
        "--request-id-prefix",
        name + "-",
        "--tokenizer",
        "/mnt/models/huggingface/hub/models--IFM--K2-Horizon-7B/snapshots/036114ce8d46c32b24c15423211069abb9c5d25e",
        "--trust-remote-code",
    ],
)
