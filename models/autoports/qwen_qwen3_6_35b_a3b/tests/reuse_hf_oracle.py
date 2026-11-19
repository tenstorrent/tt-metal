# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reuse a canonical CPU oracle only after proving its computation is unchanged."""

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
BASE = "2b0d17f5ddec89e91b5e2176e58e503a0dc15b66"


def prefix(source):
    tree = ast.parse(source)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    before_device = []
    for node in run.body:
        if isinstance(node, ast.Import) and any(a.name == "ttnn" for a in node.names):
            break
        before_device.append(node)
    return ast.dump(ast.Module(body=before_device, type_ignores=[]))


def main():
    source = ROOT / "tests/long_context.py"
    old_source = subprocess.check_output(["git", "show", f"{BASE}:{source.relative_to(Path.cwd())}"], text=True)
    assert prefix(old_source) == prefix(source.read_text()), "HF oracle computation changed"
    old = ROOT / "doc/functional_decoder/oracle_cache/full_262144.pt"
    new = ROOT / "doc/fused_decoder/oracle_cache/full_262144.pt"
    data = torch.load(old, map_location="cpu", weights_only=True)
    old_identity = copy.deepcopy(data["identity"])
    assert old_identity["reference_source"]["long_context.py"] == hashlib.sha256(old_source.encode()).hexdigest()
    assert (
        old_identity["reference_source"]["reference.py"]
        == hashlib.sha256((ROOT / "tests/reference.py").read_bytes()).hexdigest()
    )
    data["identity"]["reference_source"]["long_context.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    new.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, new)
    proof = dict(
        source=str(old),
        destination=str(new),
        base_commit=BASE,
        cpu_prefix_ast_equal=True,
        cpu_prefix_sha256=hashlib.sha256(prefix(old_source).encode()).hexdigest(),
        old_identity=old_identity,
        new_identity=data["identity"],
        note="Only downstream TT decoder selection/reset code changed; canonical CPU oracle is unchanged.",
    )
    (ROOT / "doc/fused_decoder/oracle_reuse_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print("HF_ORACLE_REUSE_PROVEN", new)
    del data

    # The linear oracle was generated in this stage before the reset became
    # layout-independent. Recover that exact source and verify its hash before
    # migrating the metadata. Preserve the original cache as separate evidence.
    linear = ROOT / "doc/fused_decoder/oracle_cache/linear_262144.pt"
    original = linear.with_name("linear_262144_original.pt")
    prior_source = source.read_text().replace(
        "ttnn.copy(ttnn.zeros_like(state[k]), state[k])",
        "ttnn.multiply(state[k], 0, output_tensor=state[k])",
    )
    assert prefix(prior_source) == prefix(source.read_text())
    data = torch.load(original if original.exists() else linear, map_location="cpu", weights_only=True)
    old_identity = copy.deepcopy(data["identity"])
    assert old_identity["reference_source"]["long_context.py"] == hashlib.sha256(prior_source.encode()).hexdigest()
    assert (
        old_identity["reference_source"]["reference.py"]
        == hashlib.sha256((ROOT / "tests/reference.py").read_bytes()).hexdigest()
    )
    if not original.exists():
        linear.rename(original)
    data["identity"]["reference_source"]["long_context.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    temporary = linear.with_suffix(".pending")
    torch.save(data, temporary)
    temporary.replace(linear)
    (ROOT / "doc/fused_decoder/linear_oracle_source.py.snapshot").write_text(prior_source)
    proof = dict(
        source=str(original),
        destination=str(linear),
        cpu_prefix_ast_equal=True,
        old_identity=old_identity,
        new_identity=data["identity"],
        old_source_snapshot="linear_oracle_source.py.snapshot",
        note="Only downstream device-state reset changed; cached CPU outputs are preserved exactly.",
    )
    (ROOT / "doc/fused_decoder/linear_oracle_reuse_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print("HF_ORACLE_REUSE_PROVEN", linear)


if __name__ == "__main__":
    main()
