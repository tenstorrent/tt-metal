"""Capture measured source identity and whitelisted live-server settings."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    model = Path(__file__).resolve().parents[1]
    plugin = repo.parent / "vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin"
    paths = list((model / "tt").glob("*.py")) + list(plugin.glob("*.py"))
    paths += [model / "doc/datatype_sweep/selected_precision_config.json"]
    result = dict(captured_at=time.time(), servers=[], engine_core_pids=[], code_sha256={})
    for path in paths:
        result["code_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    keys = (
        "HF_HUB_OFFLINE",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "OMP_NUM_THREADS",
        "K2_VLLM_ALLOW_HOST_SAMPLING",
        "K2_VLLM_FORCE_HOST_SAMPLING",
        "K2_VLLM_AUDIT_PATH",
        "MESH_DEVICE",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_WATCHER",
    )
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = proc.joinpath("cmdline").read_bytes().decode().split("\0")[:-1]
            if "vllm.entrypoints.openai.api_server" in argv:
                env = dict(
                    field.split("=", 1)
                    for field in proc.joinpath("environ").read_bytes().decode().split("\0")
                    if "=" in field
                )
                result["servers"].append(
                    dict(pid=int(proc.name), argv=argv, environment={key: env.get(key) for key in keys})
                )
            elif argv and argv[0].startswith("VLLM::EngineCore"):
                result["engine_core_pids"].append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
