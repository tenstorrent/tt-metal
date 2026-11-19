# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full chunk64 output/state diagnostic with the frozen scaled MKL exp helper.

Changes only graph setup and exp dispatch in process; existing sources and the
output/state arithmetic remain unchanged. --audit-only needs no model or device.
"""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_output_probe as previous
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_mkl_exp_scaled_ttnn_probe as helper
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import verify_source

base = previous.base
EXPECTED = {
    "v6_chunk64_output_probe.py": "4c17aca9c01f3e19ed60b99a5f4880bf7a83053713778316c19cba781737b5d6",
    "v6_mkl_exp_scaled_ttnn_probe.py": "c55f1249fae3671bdf98496d582346a947f5d27163ac2bc8a43b224346738748",
    "v6_mkl_exp_ttnn_probe.py": "8b262be19cf72ca69d553a65de656398a3efa8388f5e3ed45e9a1df5113787fa",
    "v6_chunk64_mkl_exp_scaled_state_probe.py": "d5f32a7252784083440452a4cb124359f29d92c81d3539ab8ddd539d28190dec",
    "v6_chunk64_mkl_exp_state_probe.py": "c4004cd0992298ba33c5ab21b9ecb8cb5e1d44dea6c790630643fe3a68ec27eb",
}


def audit():
    root = Path(__file__).parent
    for name, expected in EXPECTED.items():
        assert verify_source(root / name, expected), name
    # Class-source extraction avoids dedenting nested method bodies incorrectly.
    tree = ast.parse(Path(previous.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OutputChunk64Graph")
    method_ast = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    return dict(
        exp_candidate="scaled MKL normal-range exp; explicit rare-negative flush",
        mkl_exp_helper=helper.audit(),
        output_call_ast_sha256=hashlib.sha256(ast.dump(method_ast).encode()).hexdigest(),
        patch_scope="Only OutputChunk64Graph.__init__ and exp replaced; full output __call__ and attached state_call unchanged",
        mathematical_tables_only=True,
        oracle_assisted=False,
        acceptance_eligible=False,
    )


def run(args):
    assert not args.output.exists(), "Preserve evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = audit()
    graph = previous.OutputChunk64Graph
    original_init = graph.__init__
    original_call = graph.__call__
    original_attributes = {name: graph.__dict__.get(name) for name in ("__init__", "exp")}
    original_metadata = previous.dependency_metadata
    source_paths = [Path(__file__)] + [Path(__file__).parent / name for name in EXPECTED]
    source_hashes = {str(path.resolve()): base.sha(path) for path in source_paths}

    def initialize(self, ttnn, mesh, upload, **kwargs):
        original_init(self, ttnn, mesh, upload, **kwargs)
        self.mkl_negative_exp = helper.MKLNegativeExp(ttnn, mesh, upload, self.mem)

    def exp(self, value):
        return self.mkl_negative_exp(value)

    def metadata(output):
        data = original_metadata(output)
        already = {item["path"] for item in data["dependencies"]}
        for path in source_paths:
            absolute = str(path.resolve())
            if absolute in already:
                continue
            snapshot = Path(str(output) + f".dependency{len(data['dependencies'])}.{path.name}.snapshot")
            snapshot.write_bytes(path.read_bytes())
            data["dependencies"].append(
                dict(path=absolute, sha256=base.sha(path), snapshot=str(snapshot), snapshot_sha256=base.sha(snapshot))
            )
            already.add(absolute)
        data.update(details, wrapper_source=str(Path(__file__).resolve()), wrapper_sha256=base.sha(__file__))
        assert graph.__call__ is original_call
        data["output_call_identity_unchanged"] = True
        data["state_call_identity_unchanged"] = graph.state_call is previous.ordered.OrderedChunk64StateGraph.__call__
        assert data["state_call_identity_unchanged"]
        return data

    graph.__init__, graph.exp = initialize, exp
    previous.dependency_metadata = metadata
    try:
        if args.audit_only:
            result = metadata(args.output)
            result.update(status="ast_setup_pass", device_executed=False, cpu_reconstruction_repeated=False)
            assert "ttnn" not in sys.modules
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("SCALED_MKL_OUTPUT_AST_PASS", base.sha(__file__))
        else:
            previous.run(args)
    finally:
        previous.dependency_metadata = original_metadata
        for name, value in original_attributes.items():
            if value is None:
                delattr(graph, name)
            else:
                setattr(graph, name, value)
        assert graph.__call__ is original_call
        assert all(base.sha(path) == digest for path, digest in source_hashes.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--initial-state", choices=["zero", "prefix64"], default="zero")
    parser.add_argument("--variant", choices=["native", "sfpu_reduce", "sfpu_fma"], default="sfpu_fma")
    parser.add_argument("--sum-order", choices=["native", "canonical"], default="canonical")
    parser.add_argument("--cumsum", choices=["plain", "compensated"], default="compensated")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
