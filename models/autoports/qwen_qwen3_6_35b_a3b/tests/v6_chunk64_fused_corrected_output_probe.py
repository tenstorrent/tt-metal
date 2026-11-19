# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full output/state diagnostic replacing only corrected GEMM construction.

Reuses the frozen full-graph wrapper and its mm dispatch verbatim. Hardware is
parent-owned; --audit-only performs dependency/source checks without TTNN.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_corrected_fma_output_probe as previous
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_fused_corrected_matmul_probe as fused
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import verify_source

EXPECTED = {
    "v6_chunk64_corrected_fma_output_probe.py": "8568bee017b255a738b1034bb97e1fa4ff975be734307c4a1e6a93c7a594906c",
    "v6_fused_corrected_matmul_probe.py": "845c959110a713f808d1475eb22ded5ffa4e6ddd8209f02170fb90ed02b353dc",
}


def run(args):
    assert not args.output.exists(), "Preserve evidence"
    root = Path(__file__).parent
    assert all(verify_source(root / name, digest) for name, digest in EXPECTED.items())
    graph = previous.output_probe.OutputChunk64Graph
    original_call = graph.__call__
    original_audit = previous.audit
    original_factory = previous.corrected.CorrectedMatmul
    original_metadata = previous.output_probe.dependency_metadata
    details = original_audit()
    details.update(
        fused_matmul_source_checks=fused.source_checks(),
        variant="single_dispatch_corrected_fma",
        patch_scope="Only CorrectedMatmul construction replaced with FusedCorrectedMatmul; frozen corrected wrapper mm dispatch, output/state equations and all other helpers unchanged",
        performance_claim="None in full-output correctness diagnostic; separate warmed adapter timing required",
        source_generator_policy="Diagnostic setup only; eventual decoder builder must embed literal sources",
        per_k_boundary="Frozen corrected arithmetic and FP32 pack/unpack via local feedback CB24",
    )
    evidence = fused.E / "v6_fused_corrected_matmul_all_device.json"
    measured = json.loads(evidence.read_text())
    assert measured["status"] == "component_adapter_exact" and measured["device_closed"]
    assert len(measured["cases"]) == 11 and all(
        c["vs_frozen_adapter"]["fp32_bits_different"] == 0 for c in measured["cases"]
    )
    details["matrix_device_evidence"] = dict(path=str(evidence), sha256=fused.sha(evidence), cases=11)

    def construct(fma):
        return fused.FusedCorrectedMatmul(fma.t, fma.mesh, max_cores=64)

    tree = ast.parse(Path(__file__).read_text())
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "construct")
    assert [ast.unparse(n.func) for n in ast.walk(factory) if isinstance(n, ast.Call)] == ["fused.FusedCorrectedMatmul"]
    details["factory_ast_sha256"] = hashlib.sha256(ast.dump(factory).encode()).hexdigest()

    def metadata(output):
        data = original_metadata(output)
        existing = {d["path"] for d in data["dependencies"]}
        for path in [Path(__file__), root / "v6_fused_corrected_matmul_probe.py"]:
            absolute = str(path.resolve())
            if absolute in existing:
                continue
            snapshot = Path(str(output) + f'.dependency{len(data["dependencies"])}.{path.name}.snapshot')
            assert not snapshot.exists()
            snapshot.write_bytes(path.read_bytes())
            data["dependencies"].append(
                dict(path=absolute, sha256=fused.sha(path), snapshot=str(snapshot), snapshot_sha256=fused.sha(snapshot))
            )
        sources = {"reader": fused.READER, "compute": fused.compute_source(), "writer": fused.WRITER}
        kernels = {}
        for name, source in sources.items():
            path = Path(str(output) + f".fused_matmul_{name}.cpp.snapshot")
            assert not path.exists()
            path.write_text(source)
            kernels[name] = dict(path=str(path), sha256=fused.sha(path))
        data.update(details, fused_matmul_kernels=kernels, fused_wrapper_sha256=fused.sha(__file__))
        assert (
            graph.__call__ is original_call
            and graph.state_call is previous.output_probe.ordered.OrderedChunk64StateGraph.__call__
        )
        assert previous.corrected.CorrectedMatmul is construct
        data["output_and_state_call_identity_unchanged"] = True
        data["fused_factory_installed"] = True
        return data

    previous.corrected.CorrectedMatmul = construct
    previous.audit = lambda: details
    previous.output_probe.dependency_metadata = metadata
    try:
        previous.run(args)
    finally:
        previous.output_probe.dependency_metadata = original_metadata
        previous.audit = original_audit
        previous.corrected.CorrectedMatmul = original_factory
        assert (
            graph.__call__ is original_call
            and graph.state_call is previous.output_probe.ordered.OrderedChunk64StateGraph.__call__
        )
        assert all(verify_source(root / name, digest) for name, digest in EXPECTED.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[128])
    p.add_argument("--initial-state", choices=["zero", "prefix64"], default="zero")
    p.add_argument("--variant", choices=["sfpu_fma"], default="sfpu_fma")
    p.add_argument("--sum-order", choices=["canonical"], default="canonical")
    p.add_argument("--cumsum", choices=["compensated"], default="compensated")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--oracle-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
