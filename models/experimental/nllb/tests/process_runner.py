# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Fixed subprocess entry point: validated JSON is data, never a command or module path."""

import json
import math
from pathlib import Path
import re
import subprocess
import sys

MAX_REQUEST_BYTES = 65536
SCHEMAS = {
    "component": ({"component", "device"}, set()),
    "translate": (
        {
            "checkpoint",
            "config",
            "device",
            "source_language",
            "target_language",
            "text",
            "max_new_tokens",
            "precision",
            "output",
        },
        {"tokenizer_directory"},
    ),
    "reusable": ({"checkpoint", "device", "precision"}, {"config", "tokenizer_directory"}),
    "packed": ({"checkpoint", "config", "tokenizer_directory", "device", "output"}, set()),
    "envelope": (
        {"checkpoint", "device", "output", "precision", "timeout"},
        {"config", "tokenizer_directory", "oracle", "oracle_sha256"},
    ),
    "trained": ({"checkpoint", "device", "precision", "mode"}, {"config", "tokenizer_directory"}),
}
COMPONENTS = {"fused_attention", "generation_projection", "precision_storage"}


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"task", "options"}:
        raise ValueError("request requires task and options")
    task, options = request["task"], request["options"]
    if not isinstance(task, str) or task not in SCHEMAS or not isinstance(options, dict):
        raise ValueError("unknown task or invalid options")
    required, optional = SCHEMAS[task]
    if not required <= options.keys() or not options.keys() <= required | optional:
        raise ValueError("missing or unknown task options")
    for key, value in options.items():
        if key in {"device", "max_new_tokens", "timeout"}:
            lower, upper = {"device": (0, 255), "max_new_tokens": (1, 256), "timeout": (1, 1180)}[key]
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid " + key)
        elif not isinstance(value, str) or not value or "\0" in value or len(value) > 16384:
            raise ValueError("invalid " + key)
    if "precision" in options and options["precision"] not in {"bf16", "bfp8_b"}:
        raise ValueError("invalid precision")
    if "mode" in options and options["mode"] not in {"masks", "recovery"}:
        raise ValueError("invalid mode")
    if "component" in options and options["component"] not in COMPONENTS:
        raise ValueError("unknown component")
    if "oracle_sha256" in options and re.fullmatch(r"[0-9a-fA-F]{64}", options["oracle_sha256"]) is None:
        raise ValueError("invalid oracle SHA256")
    return task, options


def run_task(task, options, *, timeout, check=False):
    """Use the current interpreter and one fixed module; send user values only on stdin."""
    request = {"task": task, "options": options}
    validate_request(request)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 1200:
        raise ValueError("invalid process timeout")
    payload = json.dumps(request)
    if len(payload.encode()) > MAX_REQUEST_BYTES:
        raise ValueError("request too large")
    return subprocess.run(
        [sys.executable, "-m", "models.experimental.nllb.tests.process_runner"],
        input=payload,
        cwd=Path(__file__).resolve().parents[4],
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate request field")
        result[key] = value
    return result


def dispatch(request):
    task, options = validate_request(request)
    # --key=value binds even leading-dash strings as values, not additional options.
    argv = ["--" + key.replace("_", "-") + "=" + str(value) for key, value in options.items()]
    from models.experimental.nllb.tt import runtime_setup

    runtime_setup.configure_tracking()
    if task == "translate":
        from models.experimental.nllb.demo.translate import cli as main
    elif task == "reusable":
        from models.experimental.nllb.tests.test_reusable_port import main
    elif task == "packed":
        from models.experimental.nllb.tests.probe_packed_integration import main
    elif task == "envelope":
        from models.experimental.nllb.reference.envelope_regression import main
    elif task == "trained":
        from models.experimental.nllb.tests.test_trained_masks_recovery import main
    else:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if options["component"] == "fused_attention":
            from models.experimental.nllb.tests.test_fused_attention import check
        elif options["component"] == "generation_projection":
            from models.experimental.nllb.tests.test_generation_projection import check
        else:
            from models.experimental.nllb.tests.test_precision_storage import check
        with runtime_setup.RuntimeOwner() as owner:
            check(owner.open(options["device"]))
        return 0
    return main(argv)


def main():
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("request too large")
    return dispatch(json.loads(payload, object_pairs_hook=unique_object))


if __name__ == "__main__":
    raise SystemExit(main())
