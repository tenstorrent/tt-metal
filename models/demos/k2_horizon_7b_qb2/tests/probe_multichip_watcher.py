"""Observe completed watcher polls after mesh/router close, before global exit."""
import argparse
import hashlib
import json
import os
import re
import runpy
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mesh", "decoder"], default="decoder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--stack", type=int, choices=[1, 2], default=1)
    parser.add_argument("--split", type=int)
    parser.add_argument("--remap", action="store_true")
    parser.add_argument("--heterogeneous", action="store_true")
    args = parser.parse_args()
    core_path = Path("tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp")
    core_sha256 = hashlib.sha256(core_path.read_bytes()).hexdigest()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not os.environ.get("TT_METAL_WATCHER"):
        raise ValueError("This probe requires watcher with all checks enabled")
    disabled = [key for key in os.environ if key.startswith("TT_METAL_WATCHER_DISABLE")]
    if disabled:
        raise ValueError(disabled)
    if args.mode == "decoder":
        sys.argv = ["run_multichip", "--output", args.output]
        for key in ("seq", "steps", "batch", "stack", "split"):
            value = getattr(args, key)
            if value is not None:
                sys.argv.extend(["--" + key, str(value)])
        for key in ("remap", "heterogeneous"):
            if getattr(args, key):
                sys.argv.append("--" + key)
        runpy.run_module("models.demos.k2_horizon_7b_qb2.tests.run_multichip", run_name="__main__")
    else:
        import ttnn

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
        ttnn.close_mesh_device(mesh)
    log = Path("generated/watcher/watcher.log")

    def completed():
        return len(re.findall(r"Dump #\d+ completed", log.read_text()))

    before = completed()
    print("POST_CLOSE_WATCHER_OBSERVATION", before, flush=True)
    deadline = time.monotonic() + 45
    while completed() < before + 2:
        if time.monotonic() >= deadline:
            raise TimeoutError("No two completed watcher polls after close")
        time.sleep(0.1)
    evidence = {
        "mode": args.mode,
        "command_args": vars(args),
        "core_path": str(core_path),
        "core_sha256": core_sha256,
        "runner_sha256": runner_sha256,
        "post_close_completed_polls": completed() - before,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("TT_METAL_WATCHER")
            or k in ("TT_METAL_DISABLE_MULTI_AERISC", "TT_METAL_DISABLE_FABRIC_TWO_ERISC")
        },
        "scope": "post-router observation; complete process log must also pass firmware teardown",
    }
    Path(args.output + ".watcher.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("POST_CLOSE_WATCHER_POLLS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
