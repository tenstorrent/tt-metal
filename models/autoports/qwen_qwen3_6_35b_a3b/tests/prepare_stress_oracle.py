# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Execute the frozen stress harness's CPU prefix without importing TTNN."""

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, hf_forward, layer, weights


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    target = Path(args.oracle_cache).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    stress = Path(__file__).with_name("stress_fused_decoder.py").resolve()
    text = stress.read_text()
    source_hash = digest(stress)
    tree = ast.parse(text)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    start = next(
        i
        for i, node in enumerate(run.body)
        if isinstance(node, ast.Expr) and ast.unparse(node.value).startswith("torch.set_num_threads(")
    )
    stop = next(
        i
        for i, node in enumerate(run.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "mesh" for target in node.targets)
    )
    cpu_nodes = run.body[start:stop]
    # No textual rewriting of reference generation, source keys, RNG or tensor operations.
    cpu_program = ast.Module(body=cpu_nodes, type_ignores=[])
    imports = [node for node in ast.walk(cpu_program) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("ttnn" not in ast.unparse(node) for node in imports)
    assert "ttnn." not in ast.unparse(cpu_program)
    staging = target if target.exists() else target.with_name(target.name + ".preparing")
    args.oracle_cache = str(staging)
    scope = dict(
        torch=torch,
        DynamicCache=DynamicCache,
        config=config,
        hf_forward=hf_forward,
        layer=layer,
        weights=weights,
        Path=Path,
        hashlib=hashlib,
        args=args,
        __file__=str(stress),
        __name__="__cpu_stress_oracle__",
    )
    begin = time.monotonic()
    exec(compile(ast.fix_missing_locations(cpu_program), str(stress), "exec"), scope)
    assert digest(stress) == source_hash, "Stress source changed while the CPU oracle was generated"
    assert not any(name == "ttnn" or name.startswith("ttnn.") for name in sys.modules), "TTNN unexpectedly imported"
    if staging != target:
        staging.replace(target)
    proof = dict(
        execution="CPU only, 8 Torch threads; no TTNN import or device access",
        method="Unmodified AST statements from torch.set_num_threads through CPU oracle save, stopping before open_mesh_device",
        stress_source=str(stress),
        stress_source_sha256=source_hash,
        executed_cpu_ast_sha256=hashlib.sha256(ast.dump(cpu_program, include_attributes=False).encode()).hexdigest(),
        helper_source=str(Path(__file__).resolve()),
        helper_source_sha256=digest(__file__),
        oracle=str(target),
        oracle_sha256=digest(target),
        key=scope["oracle_key"],
        elapsed_seconds=time.monotonic() - begin,
        decode_references=len(scope["refs"]),
        prefix_shape=list(scope["prefix"].shape),
        continuation_shape=list(scope["continuation"].shape),
        all_outputs_finite=all(
            bool(torch.isfinite(t).all()) for t in [scope["ref_p"], *scope["refs"], scope["ref_continuation"]]
        ),
    )
    assert proof["all_outputs_finite"]
    Path(str(target) + ".proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print("STRESS_ORACLE_READY", json.dumps(proof), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-length", type=int, default=8192)
    parser.add_argument("--continuation-length", type=int, default=33)
    parser.add_argument("--oracle-cache", required=True)
    main(parser.parse_args())
