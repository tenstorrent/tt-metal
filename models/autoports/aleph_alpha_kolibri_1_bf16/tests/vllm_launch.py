# SPDX-License-Identifier: Apache-2.0
"""Record and invoke the shared runner without shell-dependent JSON quoting."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--reduced", action="store_true")
p.add_argument("--tracking", action="store_true")
p.add_argument("--host-compat", action="store_true")
p.add_argument("--startup-control", action="store_true")
p.add_argument("--chunk-control", action="store_true")
p.add_argument("--stages", default="serve")
p.add_argument("--max-num-seqs", type=int, default=32)
p.add_argument("--label", required=True)
a = p.parse_args()
root = Path("models/autoports/aleph_alpha_kolibri_1_bf16")
out = root / "readiness_vllm"
env = os.environ.copy()
env["TT_METAL_TRACE_ALLOC_TRACKING"] = "1" if a.tracking else "0"
env["KOLIBRI_VLLM_EVENT_DETAIL"] = "1" if a.tracking else "0"
env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] = "0"
env["KOLIBRI_VLLM_HOST_COMPAT"] = "1" if a.host_compat else "0"
env["KOLIBRI_VLLM_EVENTS"] = str((out / (a.label + "-events.jsonl")).resolve())
if a.startup_control:
    env["KOLIBRI_VLLM_STARTUP_CONTROL"] = str((out / (a.label + "-standalone-logits.json")).resolve())
else:
    env.pop("KOLIBRI_VLLM_STARTUP_CONTROL", None)
env["KOLIBRI_VLLM_CHUNK_CONTROL"] = "1" if a.chunk_control else "0"
if a.reduced:
    env["KOLIBRI_VLLM_LAYERS"] = "[0,4]"
else:
    env.pop("KOLIBRI_VLLM_LAYERS", None)
length = 4096 if a.reduced else 262144
config = dict(
    trace_region_size=200000000,
    fabric_config="FABRIC_1D",
    trace_mode="all",
    decode_interleave_enabled=True,
    decode_interleave_prefill_steps=1,
    decode_interleave_decode_steps=1,
)
extra = [
    "--max-num-batched-tokens",
    "512",
    "--enable-chunked-prefill",
    "--dtype",
    "bfloat16",
    "--shutdown-timeout",
    "10",
    "--override-generation-config",
    json.dumps({"top_k": 32}),
]
cmd = [
    sys.executable,
    "-m",
    "readiness_check.run_vllm_server",
    "--stages",
    a.stages,
    "--model-dir",
    str(root),
    "--hf-model",
    env["KOLIBRI_CHECKPOINT_DIR"],
    "--mesh-device",
    "P300x2",
    "--max-num-seqs",
    str(a.max_num_seqs),
    "--max-model-len",
    str(length),
    "--block-size",
    "32",
    "--sampling-profile",
    "full",
    "--tt-config",
    json.dumps(config),
    "--additional-server-args",
    shlex.join(extra),
]
(out / (a.label + "-command.json")).write_text(
    json.dumps(
        dict(
            command=cmd,
            environment={
                k: v
                for k, v in env.items()
                if k.startswith(("KOLIBRI_", "TT_METAL_TRACE", "TT_MODEL_CLASS", "VLLM_TT"))
            },
        ),
        indent=2,
    )
    + "\n"
)
snapshot_dir = out / (a.label + "-source")
snapshot_dir.mkdir(exist_ok=True)
source_files = [
    root / "tt" / name
    for name in ("generator.py", "generator_vllm.py", "model.py", "precision.py", "vllm_registration.py")
]
source_files += [
    Path(env["VLLM_TT_PLUGIN_ROOT"]) / "src/vllm_tt_plugin" / name for name in ("model_runner.py", "worker.py")
]
hashes = {}
for path in source_files:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    (snapshot_dir / (digest + ".py.txt")).write_bytes(data)
    hashes[str(path.resolve())] = digest
(snapshot_dir / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
print(shlex.join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, env=env))
