"""Reviewable setup/request/output boundaries versus model and sampling kernels."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = {"from_state_dict", "make_cache", "prepare", "_configure_dram"}
MESH_SETUP = {"__init__", "upload", "shards", "_hidden_parts", "build"}
GENERATOR_BOUNDARIES = {
    "__init__": "persistent setup and sampler parameter allocation",
    "build_generator": "checkpoint/tokenizer/model setup",
    "_upload": "setup-only tensor upload",
    "_copy": "explicit request/scheduler input copy",
    "_positions": "request-boundary positions; device advance handles steady state",
    "_slots": "scheduler-owned request-slot update",
    "_request_pages": "retained logical scheduler page mappings, separate from inactive scratch pages",
    "_pages": "copy only when scheduler changes page table",
    "_active_rows": "request/scheduler active-row mask update",
    "_read_tokens": "explicit caller output; first-token TTFT or compatibility/teacher-forcing",
    "_read_logits": "explicit full-logit diagnostic/host-sampling boundary",
    "_single_prompt": "single logical request mapped to fixed batch rows",
    "set_sampling_params": "request-boundary persistent sampler parameter/seed update",
    "prefill_forward": "logical prompt scheduling/chunk upload and optional explicit outputs",
    "decode_forward": "optional scheduler updates and optional caller-visible token read",
    "prefill_logits": "readiness-only full-logit boundary",
    "generate": "request setup, final history output, explicit host compatibility and teacher forcing",
}
BOUNDARIES = {"from_torch", "to_torch", "as_tensor", "numpy", "cpu", "tolist", "item", "argmax"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full_model")
    args = ap.parse_args()
    hashes = {}
    pure = []
    exclusions = []
    violations = []
    for path in sorted((ROOT / "tt").glob("*.py")):
        source = path.read_text()
        relative = str(path.relative_to(ROOT))
        hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
        if path.name == "checkpoint.py":
            exclusions.append(dict(file=relative, reason="setup-only lazy checkpoint loading"))
            continue
        for owner in ast.walk(ast.parse(source)):
            if not isinstance(owner, (ast.Module, ast.ClassDef)):
                continue
            for fn in owner.body:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                identity = dict(file=relative, owner=getattr(owner, "name", "module"), method=fn.name, line=fn.lineno)
                reason = None
                if fn.name in SETUP:
                    reason = "explicit setup preparation"
                if path.name == "multichip_decoder.py" and fn.name in MESH_SETUP:
                    reason = "mesh weight/setup boundary"
                if path.name == "model.py" and fn.name == "__init__":
                    reason = "real weight loading and persistent state setup"
                if path.name == "generator.py":
                    reason = GENERATOR_BOUNDARIES.get(fn.name, reason)
                if getattr(owner, "name", None) == "_PrefixView":
                    reason = "setup state-dict mapping"
                if reason:
                    exclusions.append(dict(**identity, reason=reason))
                    continue
                pure.append(identity)
                for node in ast.walk(fn):
                    if isinstance(node, ast.Call):
                        name = ast.unparse(node.func)
                        # Native TTNN argmax remains on device; host argmax is
                        # allowed only in the explicit compatibility boundary.
                        host = name.startswith(("torch.", "np.", "numpy.")) or (
                            name.split(".")[-1] in BOUNDARIES and not name.startswith("ttnn.argmax")
                        )
                        if host:
                            violations.append(dict(**identity, call=name, call_line=node.lineno))
                    if isinstance(node, ast.Import) and any(a.name in ("torch", "numpy") for a in node.names):
                        violations.append(dict(**identity, call=ast.unparse(node), call_line=node.lineno))
    report = dict(
        status="pass" if not violations else "fail",
        source_sha256=hashes,
        runtime_methods=pure,
        explicit_boundaries=exclusions,
        violations=violations,
        limitations="Static call-boundary audit; dynamic guards, trace counters, cache ownership and reset evidence are also required.",
    )
    (ROOT / "doc" / args.stage / "runtime_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], methods=len(pure), violations=violations)))
    assert not violations


if __name__ == "__main__":
    main()
