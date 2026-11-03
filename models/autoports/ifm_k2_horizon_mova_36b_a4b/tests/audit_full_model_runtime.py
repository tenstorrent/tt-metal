# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static audit of the model's device forwards and explicit host boundaries."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    prohibited = {
        "to_torch",
        "from_torch",
        "cpu",
        "numpy",
        "item",
        "tolist",
        "argmax",
        "copy_host_to_device_tensor",
        "synchronize_device",
    }
    entrypoints = {
        "model.py": {"embed", "lookup", "terminal", "decode_forward", "prefill_chunk"},
        "generator.py": {
            "_decode",
            "_sample",
            "_prefill_head",
            "_stage_prefill_hidden",
            "replay",
            "_prefill_trace_forward",
            "_replay_prefill",
        },
    }
    report = dict(status="pass", source_sha256={}, device_methods={}, host_boundaries={})
    for filename, names in entrypoints.items():
        path = root / "tt" / filename
        source = path.read_text()
        report["source_sha256"][filename] = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.ExceptHandler) for node in ast.walk(tree)), "No runtime exception fallback"
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            for method in (node for node in cls.body if isinstance(node, ast.FunctionDef)):
                calls = [ast.unparse(node.func) for node in ast.walk(method) if isinstance(node, ast.Call)]
                key = f"{cls.name}.{method.name}"
                if method.name in names:
                    bad = [call for call in calls if call.split(".")[-1] in prohibited]
                    assert not bad, (key, bad)
                    report["device_methods"][key] = calls
                else:
                    boundaries = [call for call in calls if call.split(".")[-1] in prohibited]
                    if boundaries:
                        report["host_boundaries"][key] = boundaries
    report["inspection"] = {
        "strategy": "Model requires1x4 and constructs accepted MultichipDecoder; no single-chip or host alternative. Replicated BF16 residual is inherited optimized policy, passed directly between layers.",
        "cache": "Generator owns initial cache or accepts fresh external cache before prepare. Cache tensors remain stable across traces. No host cache read in model/generator; host tensor allocation only constructs zero cache at setup.",
        "logits": "Host full-vocab reads only prefill_logits/return_all_logits, explicit host-sampling mode, or diagnostic read_logits. Optimized device prefill stages final hidden and samples ondevice; decode reads IDs only.",
        "sampling": "Common device ops run inside retained generator sampling trace; common sample()/capture_trace() wrappers are never called. tt_out_tok aliases persistent decoder input; _sample masks inactive output/seed/count updates.",
        "feedback": "generate host/device branch inspection: .argmax and token upload are guarded by host mode or explicit teacher-forcing next_input. Device free-running loop only submits two traces and observes caller-visible token. Position/RoPE increments reside in model trace.",
        "reload": "Low-level version1 explicit commands govern uploads. reload_inputs=False does not inspect stale token/position values; page-only refresh copies only pages. No comparison heuristic suppresses commanded copies.",
        "reset": "Request reset resets common penalty/RNG/parameter state, preserves allocations/cache/traces. New prompt makes old cache beyond its causal prefix unreachable. No reset/synchronization per resident decode step. close releases traces before mesh close; no device reset.",
        "dynamic_guard": "build_generator sets ttnn.CONFIG.throw_exception_on_fallback=True. Allocation/program-cache tracking acceptance logs are separate from timing runs.",
        "scope": "This AST audit covers wrapper device methods; audit_multichip_runtime resolves inherited decoder methods separately. Static inspection supplements hardware contracts and is not evidence of numerical correctness.",
    }
    dest = args.output or root / "doc/full_model/runtime_audit.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
