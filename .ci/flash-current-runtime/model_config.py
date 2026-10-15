"""CPU-only admission of the pinned model configuration and required Transformers release."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import urllib.request

import transformers
from transformers import AutoConfig

assert transformers.__version__ == "5.16.1"
assert sys.version_info[:2] == (3, 12)
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
url = (
    "https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/2741eec155d03a8ce151b993ccce1a7b1e398d6b/config.json"
)
with urllib.request.urlopen(url, timeout=60) as response:
    raw = response.read()
assert len(raw) == 4745
assert hashlib.sha256(raw).hexdigest() == "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
(out / "config.json").write_bytes(raw)
config = AutoConfig.from_pretrained(out, local_files_only=True, trust_remote_code=False)
assert config.model_type == "qwen4_exp"
origin = Path(inspect.getfile(type(config))).resolve(strict=True)
assert "transformers/models/qwen4_exp" in str(origin)
fds = {}
for p in Path("/proc/self/fd").iterdir():
    try:
        value = os.readlink(p)
    except FileNotFoundError:
        continue
    if value.startswith("/dev/tenstorrent/"):
        fds[p.name] = value
assert not fds
(out / "proof.json").write_text(
    json.dumps(
        {
            "python": sys.version,
            "transformers": transformers.__version__,
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "class": type(config).__name__,
            "origin": str(origin),
            "origin_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
            "tt_fds": fds,
            "scope": "config only; no model weights or accelerator construction",
        },
        indent=2,
    )
)
