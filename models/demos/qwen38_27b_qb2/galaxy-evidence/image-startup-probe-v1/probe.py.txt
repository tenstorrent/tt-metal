# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Run the image's TTIS setup, stopping immediately before vLLM server startup.

Run in a disposable container with no TT devices or network, read-only weights,
and writable temporary cache/log directories. This checks packaging, not model
loading, hardware, HTTP responses, or accuracy. The image is left unchanged.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def argument_values(argv):
    result = {}
    index = 1
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional startup argument: {token}")
        name, separator, value = token[2:].partition("=")
        name = name.replace("_", "-")
        if name in result:
            raise ValueError(f"Duplicate startup argument: {name}")
        if not separator:
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                index += 1
                value = argv[index]
            else:
                value = True
        result[name] = value
        index += 1
    return result


def verify_handoff(spec, argv, environment):
    """Check the actual handoff against every declared argument and env value."""
    actual = argument_values(argv)
    expected = spec["device_model_spec"]["vllm_args"]
    if set(actual) - {name.replace("_", "-") for name in expected}:
        raise ValueError("Unexpected additional startup arguments")
    for name, value in expected.items():
        name = name.replace("_", "-")
        observed = actual.get(name)
        if value is False or value is None:
            if name in actual:
                raise ValueError(f"Unexpected disabled argument: {name}")
        elif isinstance(value, (dict, list)):
            if not isinstance(observed, str) or json.loads(observed) != value:
                raise ValueError(f"Startup argument differs from bundle: {name}")
        elif observed != (True if value is True else str(value)):
            raise ValueError(f"Startup argument differs from bundle: {name}")
    declared = {
        **spec["device_model_spec"].get("env_vars", {}),
        **spec.get("env_vars", {}),
    }
    for name, value in declared.items():
        if environment.get(name) != str(value):
            raise ValueError(f"Startup environment differs from bundle: {name}")
    if "TT_METAL_OPERATION_TIMEOUT_SECONDS" in environment:
        raise ValueError("Unexpected operation timeout in the qualified launch")
    return {
        "arguments": actual,
        "declared_environment": {name: environment[name] for name in declared},
    }


def probe(entrypoint, spec_path):
    if Path("/dev/tenstorrent").exists():
        raise RuntimeError("Run this probe in a container without TT devices")
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    if spec.get("impl", {}).get("impl_id") != "qwen38_27b_qb2":
        raise ValueError("Expected the Qwen Galaxy runtime bundle")
    os.environ["RUNTIME_MODEL_SPEC_JSON_PATH"] = str(spec_path)
    sys.path.insert(0, str(entrypoint.parent))
    module_spec = importlib.util.spec_from_file_location(
        "qwen_startup_probe", entrypoint
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    handoffs = []

    def capture(name, *, run_name):
        if name != "vllm.entrypoints.openai.api_server" or run_name != "__main__":
            raise RuntimeError(f"Unexpected server entrypoint: {name}")
        handoffs.append(verify_handoff(spec, sys.argv, os.environ))

    # Replace only this wrapper's final call, leaving imports, cache setup,
    # model registry, logging, secrets handling and argument merging real.
    module.runpy = SimpleNamespace(run_module=capture)
    old_argv = sys.argv
    try:
        sys.argv = [str(entrypoint)]
        module.main()
    finally:
        sys.argv = old_argv
    if len(handoffs) != 1:
        raise RuntimeError("Wrapper did not reach exactly one server handoff")
    weights = Path(os.environ["MODEL_WEIGHTS_DIR"]).resolve(strict=True)
    if Path(os.environ["HF_MODEL"]).resolve(strict=True) != weights:
        raise ValueError("Wrapper changed the mounted checkpoint path")
    return {
        "state": "startup_handoff_passed_unqualified",
        "entrypoint_sha256": hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
        "runtime_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "hardware_opened": False,
        "server_started": False,
        "weights_contents_verified": False,
        "checkpoint_path": str(weights),
        **handoffs[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entrypoint",
        type=Path,
        default=Path("/home/container_app_user/app/src/run_vllm_api_server.py"),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("/home/container_app_user/qwen38-bundle/runtime-model-spec.json"),
    )
    args = parser.parse_args()
    print(json.dumps(probe(args.entrypoint, args.spec), indent=2))
