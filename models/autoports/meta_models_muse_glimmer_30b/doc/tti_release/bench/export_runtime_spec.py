#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Export the autoport's tt-inference-server runtime model spec JSON.

The spec is *derived* from the catalog entry this stage added to
``workflows/model_specs/prod/llm.yaml`` rather than hand-written, so the JSON
that ``run.py --runtime-model-spec-json`` loads and the entry that
``EVAL_CONFIGS`` is keyed from cannot drift apart.

The embedded ``cli_args`` are filled in for the external-server topology
(``docker_server=false``, ``local_server=false``, the running autoport server's
port, and the workflow being run) so the loaded JSON is already correct rather
than relying on command-line flags to override it.  ``run.py`` additionally
back-fills ``cli_args`` from its own RuntimeConfig
(``run.py::populate_model_spec_cli_args``); setting them here means the two
agree instead of one silently winning.

Usage::

    python3 export_runtime_spec.py --workflow release --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_ID = "id_muse-glimmer-30b-autoport_Muse-Glimmer-30B_p300x2"
AUTOPORT_CODE_PATH = "models/autoports/meta_models_muse_glimmer_30b"
CONTEXT_CONTRACT = Path(__file__).resolve().parents[2] / "context_contract.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tti-root", required=True, help="tt-inference-server checkout")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--service-port", default="8000")
    parser.add_argument("--device", default="p300x2")
    parser.add_argument("--limit-samples-mode", default=None)
    parser.add_argument("--disable-trace-capture", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.tti_root)
    from workflows.model_spec import MODEL_SPECS  # noqa: E402

    spec = MODEL_SPECS[MODEL_ID]
    data = spec.get_serialized_dict()

    assert data["impl"]["code_path"] == AUTOPORT_CODE_PATH, data["impl"]
    contract = json.loads(CONTEXT_CONTRACT.read_text())
    supported = contract["current_supported_context"]
    assert data["device_model_spec"]["max_context"] == supported, (
        f"spec max_context={data['device_model_spec']['max_context']} does not match "
        f"doc/context_contract.json current_supported_context={supported}"
    )

    data["cli_args"] = {
        "model": data["model_name"],
        "workflow": args.workflow,
        "device": args.device,
        "tt_device": args.device,
        "impl": data["impl"]["impl_id"],
        "engine": data["inference_engine"],
        # External-server topology: the autoport vLLM server is already running
        # from the tt-metal checkout, so TTI is a pure client.
        "docker_server": False,
        "local_server": False,
        "service_port": str(args.service_port),
        "server_url": None,
        "no_auth": True,
        "disable_trace_capture": bool(args.disable_trace_capture),
        "limit_samples_mode": args.limit_samples_mode,
        "runtime_model_spec_json": str(Path(args.out).resolve()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    print(f"  model_id      {data['model_id']}")
    print(f"  impl.code_path {data['impl']['code_path']}")
    print(f"  hf_model_repo {data['hf_model_repo']}")
    print(f"  device_type   {data['device_type']}")
    print(f"  status        {data['status']}")
    print(f"  max_context   {data['device_model_spec']['max_context']}")
    print(f"  cli_args      {json.dumps(data['cli_args'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
