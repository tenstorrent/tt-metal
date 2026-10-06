# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned profile argv and validate observed /server_info configuration.

This does not launch or stop a server and is not the runner's server-control hook.
It deliberately separates requested argv from a running server's evidence.
"""

import argparse
import json
import shlex
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def profile_plan(slots):
    if slots not in (1, 32):
        raise ValueError("Required profiles have one or 32 server slots")
    inherited = json.loads((MODEL_DIR / "doc/optimized_vllm/server_command.json").read_text())
    args = inherited["args"]

    def option(name):
        return args[args.index(name) + 1]

    if option("--hf-model") != MODEL or option("--max-model-len") != "262144":
        raise ValueError("Stage 10 handoff model/context changed; re-audit the launch plan")
    tt_config = {"sample_on_device_mode": "all", **json.loads(option("--tt-config"))}
    command = [
        args[0],
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--block-size",
        option("--block-size"),
        "--max-num-seqs",
        str(slots),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        option("--max-model-len"),
        "--additional-config",
        json.dumps({"tt": tt_config}),
        *shlex.split(option("--additional-server-args")),
    ]
    return {
        "status": "unlaunched_plan_not_running_identity",
        "server_slots": slots,
        "concurrent_requests": slots,
        "command": command,
        "environment_overrides": {
            **inherited["environment"],
            "MESH_DEVICE": option("--mesh-device"),
            "HF_MODEL": MODEL,
            "VLLM_SERVER_DEV_MODE": "1",
        },
        "inherited_precision": "doc/datatype_sweep/selected_precision_config.json",
        "required_before_launch": [
            "Complete and validate phase instrumentation/collector and install or locate authorized evaluation client.",
            "Replace inherited log/cache output paths with owned stage paths, retain existing runtime library paths.",
            "Check reservation ownership, process/port ownership and reject any unrelated live server.",
            "Start launch inside benchmark_stage run; record observed startup and imported model identity.",
            "Validate accuracy sampling/response parsing separately; native top_k64 currently clamps to32 on device.",
        ],
    }


def validate_observed_config(server_info, slots):
    """Reject drift using actual server_info JSON, never requested argv alone."""
    if slots not in (1, 32):
        raise ValueError("Invalid profile slots")
    config = server_info["vllm_config"]
    model, cache, scheduler = (config[key] for key in ("model_config", "cache_config", "scheduler_config"))
    expected = {
        "model": (model["model"], MODEL),
        "revision": (model["revision"], REVISION),
        "tokenizer_revision": (model["tokenizer_revision"], REVISION),
        "max_model_len": (model["max_model_len"], 262144),
        "block_size": (cache["block_size"], 32),
        "enable_prefix_caching": (cache["enable_prefix_caching"], False),
        "max_num_seqs": (scheduler["max_num_seqs"], slots),
        "enable_chunked_prefill": (scheduler["enable_chunked_prefill"], False),
        "async_scheduling": (scheduler["async_scheduling"], True),
    }
    plan = profile_plan(slots)["command"]
    expected_tt = json.loads(plan[plan.index("--additional-config") + 1])["tt"]
    for key, wanted in expected_tt.items():
        expected["tt." + key] = (config["additional_config"]["tt"][key], wanted)
    differences = {
        key: {"observed": got, "expected": wanted} for key, (got, wanted) in expected.items() if got != wanted
    }
    if differences:
        raise ValueError("Observed serving configuration mismatch: " + json.dumps(differences, sort_keys=True))
    return {key: got for key, (got, _) in expected.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-num-seqs", type=int, choices=(1, 32), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed-server-info", type=Path)
    args = parser.parse_args()
    result = profile_plan(args.max_num_seqs)
    if args.observed_server_info:
        result["observed_configuration"] = validate_observed_config(
            json.loads(args.observed_server_info.read_text()), args.max_num_seqs
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
