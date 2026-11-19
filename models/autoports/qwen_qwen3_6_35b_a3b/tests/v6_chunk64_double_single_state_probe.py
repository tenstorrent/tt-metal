# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Combined diagnostic: canonical triangular sum plus double-single cumsum.

The base and ordered wrappers remain unchanged. Only the already-ordered call's
cumulative-g assignment is replaced, with full AST restoration proof.
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_ordered_state_probe as ordered
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_double_single_cumsum_probe as helper

base = ordered.base
ORIGINAL_ATTACH_CALL = ordered.attach_call


def attach_call():
    ordered_hash = ORIGINAL_ATTACH_CALL()
    tree = ast.parse(Path(base.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Chunk64StateGraph")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    count = 0
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "correction":
                assert isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "t.sum"
                assert ast.unparse(node.value.args[0]) == "product"
                node.value = ast.parse("self.triangular_sum(product)", mode="eval").body
                count += 1
    assert count == 1
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    assert hashlib.sha256(ast.dump(module).encode()).hexdigest() == ordered_hash
    before = copy.deepcopy(module)
    count = 0
    target = None
    original_value = None
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "cumulative":
                assert isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "t.cumsum"
                assert ast.unparse(node.value.args[0]) == "gg"
                original_value = copy.deepcopy(node.value)
                target = node
                node.value = ast.parse(
                    "ttnn_double_single_chunk_cumsum(t, gg, mesh_device=self.mesh, memory_config=self.mem)",
                    mode="eval",
                ).body
                count += 1
    assert count == 1
    ast.fix_missing_locations(module)
    patched_hash = hashlib.sha256(ast.dump(module).encode()).hexdigest()
    scope = dict(vars(base), ttnn_double_single_chunk_cumsum=helper.ttnn_double_single_chunk_cumsum)
    exec(compile(module, str(Path(__file__).resolve()) + ":ordered_call_with_double_single_cumsum", "exec"), scope)
    ordered.OrderedChunk64StateGraph.__call__ = scope["__call__"]
    target.value = original_value
    assert ast.dump(module) == ast.dump(before), "Only the cumulative assignment may differ from ordered wrapper"
    attach_call.proof = dict(
        unchanged_ordered_call_ast_sha256=ordered_hash,
        patched_call_ast_sha256=patched_hash,
        full_ordered_call_ast_restoration=True,
        changed_assignment="cumulative only",
    )
    return patched_hash


def metadata():
    helper_tree = ast.parse(Path(helper.__file__).read_text())
    method = next(
        n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == "ttnn_double_single_chunk_cumsum"
    )
    calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
    assert not any(call.startswith("torch.") or call.endswith((".from_torch", ".to_torch")) for call in calls)
    assert not any(isinstance(n, ast.keyword) and n.arg == "fast_and_approximate_mode" for n in ast.walk(method))
    return dict(
        double_single_wrapper=str(Path(__file__).resolve()),
        double_single_wrapper_sha256=base.sha(__file__),
        double_single_helper=str(Path(helper.__file__).resolve()),
        double_single_helper_sha256=base.sha(helper.__file__),
        unchanged_ordered_wrapper_sha256=base.sha(ordered.__file__),
        unchanged_base_source_sha256=base.sha(base.__file__),
        double_single_runtime_ast_audit="no Torch or host conversion; no fast_and_approximate_mode flags",
        double_single_patch_proof=attach_call.proof,
        acceptance_eligible=False,
    )


def run(args):
    original_setup = base.cpu_setup
    original_attach = ordered.attach_call
    source_hashes = {p: base.sha(p) for p in (base.__file__, ordered.__file__, helper.__file__, __file__)}

    def setup(options):
        result, payload = original_setup(options)
        result.update(metadata())
        result.update(
            cumsum="double_single",
            base_cumsum_option_ignored=args.cumsum,
            cumsum_precision_claim="Double-single matches1193984 tested CPU prefixes; device parity still requires measurement",
        )
        for label, source in (("double_single_wrapper", __file__), ("double_single_helper", helper.__file__)):
            snapshot = Path(str(args.output) + f".{label}.py.snapshot")
            snapshot.write_bytes(Path(source).read_bytes())
            result[label + "_snapshot"] = str(snapshot)
            result[label + "_snapshot_sha256"] = base.sha(snapshot)
        return result, payload

    ordered.attach_call = attach_call
    base.cpu_setup = setup
    try:
        ordered.run(args)
    finally:
        ordered.attach_call = original_attach
        base.cpu_setup = original_setup
        assert all(base.sha(p) == digest for p, digest in source_hashes.items())
        provenance = Path(str(args.output) + ".provenance.json")
        if provenance.exists():
            data = json.loads(provenance.read_text())
            data.update(metadata())
            data["cumsum"] = "double_single"
            provenance.write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[128])
    p.add_argument("--initial-state", choices=["zero", "prefix64"], default="zero")
    p.add_argument("--variant", choices=["native", "sfpu_reduce", "sfpu_fma"], default="sfpu_fma")
    p.add_argument(
        "--cumsum",
        choices=["plain", "compensated"],
        default="compensated",
        help="Compatibility only; replaced by double-single",
    )
    p.add_argument("--sum-order", choices=["native", "canonical"], default="canonical")
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
