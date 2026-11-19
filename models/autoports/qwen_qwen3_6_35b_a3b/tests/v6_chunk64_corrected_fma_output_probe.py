# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full chunk64 output/state diagnostic with the frozen corrected-FMA GEMM.

Wraps the unchanged scaled-MKL output graph. Only setup and mm dispatch change;
CPU oracles, nonaligned padding, output equations, and state equations do not.
"""

import argparse
import ast
import hashlib
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_mkl_exp_scaled_output_probe as previous
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_corrected_fma_matmul_probe as corrected
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import verify_source

output_probe = previous.previous
base = output_probe.base
EXPECTED = {
    **previous.EXPECTED,
    "v6_chunk64_mkl_exp_scaled_output_probe.py": "5b75128be0dda666823c012ad28c6aa5e8853929111e57628a1f89f452576477",
    "v6_chunk64_double_single_state_probe.py": "0724002182bde728747eb7464f557f22eb00b198dffdd6fef3f3586536b0240f",
    "v6_chunk64_ordered_state_probe.py": "9e4e4d864ad235a0e1bb318d79f77c2a47e22aa3a06635b082c3d027b95d97cc",
    "v6_chunk64_state_probe.py": "b5179c6d3a311172bcb3c2f6f008109ab926b112f47a7baa25dcca9642fbfbc3",
    "v6_double_single_cumsum_probe.py": "fc5b300fa4e1fea9023cd6eb8e6123dea394b1cdb89a6fdf2c16e96a537cb419",
    "v6_corrected_fma_matmul_probe.py": "facef89f3f760fcd86c54e31e63ac938657f2ce02c639715699eb3c94ab90442",
    "v6_fma_sfpu_corrected_probe.py": "185217ed00091eddcdf15db9494071901edaa472d3935b986b4b512834f6832f",
    "v6_fma_sfpu_primitive_probe.py": "8938937a8bdf385a2c908242f296a745bd0ce498dad5b834a93759cd092ca3bc",
}


def audit():
    root = Path(__file__).parent
    for name, expected in EXPECTED.items():
        assert verify_source(root / name, expected), name
    details = previous.audit()
    details.update(
        corrected_fma_runtime_audit=corrected.runtime_audit(),
        corrected_fma_arithmetic_cpp_sha256=hashlib.sha256(corrected.Graph().build().cpp().encode()).hexdigest(),
        corrected_fma_compute_sha256=hashlib.sha256(corrected.adapter_source().encode()).hexdigest(),
        variant="corrected_fma",
        legacy_variant_argument="sfpu_fma is ignored by overridden mm; no native-addcmul GEMM fallback",
        patch_scope="Setup creates CorrectedFma/CorrectedMatmul; only mm dispatch changes relative to scaled-MKL output/state graph",
        precision_claim="Measured bounded corrected FMA, increasing-K matrix recurrence; actual tiny transformed-K/state-contribution differences remain, so whole-graph parity is measured, not assumed",
        runtime_host_policy="No runtime Torch/from_torch/to_torch in corrected adapter or matrix wrapper; optional existing observer reads diagnostic intermediates outside the mathematical graph",
        cumsum="double_single",
        sum_order="canonical",
        exp_candidate="scaled MKL normal-range exp with existing explicit rare-negative flush",
        positive_logical_length_required=True,
        public_alignment_restriction=False,
        oracle_assisted=False,
        acceptance_eligible=False,
        performance_claim="None; per-K corrected-FMA dispatch and broadcasting are diagnostic",
    )
    for label, filename in (
        ("corrected_fma_primitive_device_evidence", "v6_fma_sfpu_corrected_device_v2.json"),
        ("corrected_fma_adapter_device_evidence", "v6_corrected_fma_adapter_device_v2.json"),
        ("corrected_fma_matrix_device_evidence", "v6_corrected_fma_matrices_device.json"),
    ):
        path = corrected.DOC / filename
        assert path.exists(), path
        details[label] = dict(path=str(path.resolve()), sha256=base.sha(path))
    return details


def run(args):
    assert not args.output.exists(), "Preserve previous evidence"
    assert args.variant == "sfpu_fma" and args.sum_order == "canonical"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = audit()
    graph = output_probe.OutputChunk64Graph
    original_init = graph.__init__
    original_call = graph.__call__
    original_dict = {name: graph.__dict__.get(name) for name in ("__init__", "mm")}
    original_metadata = output_probe.dependency_metadata
    original_scaled_audit = previous.audit
    paths = [Path(__file__)] + [Path(__file__).parent / name for name in EXPECTED]
    source_hashes = {path: base.sha(path) for path in paths}

    def initialize(self, ttnn, mesh, upload, **kwargs):
        original_init(self, ttnn, mesh, upload, **kwargs)
        self.corrected_fma = corrected.CorrectedFma(ttnn, mesh)
        self.corrected_mm = corrected.CorrectedMatmul(self.corrected_fma)

    def mm(self, a, b):
        return self.corrected_mm(a, b)

    # Explicitly check this wrapper's runtime dispatch before installing it.
    tree = ast.parse(Path(__file__).read_text())
    dispatch = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "mm")
    calls = [ast.unparse(node.func) for node in ast.walk(dispatch) if isinstance(node, ast.Call)]
    assert calls == ["self.corrected_mm"], calls
    details["mm_dispatch_ast_sha256"] = hashlib.sha256(ast.dump(dispatch).encode()).hexdigest()

    def metadata(output):
        data = original_metadata(output)
        already = {item["path"] for item in data["dependencies"]}
        for path in paths:
            absolute = str(path.resolve())
            if absolute in already:
                continue
            snapshot = Path(str(output) + f".dependency{len(data['dependencies'])}.{path.name}.snapshot")
            assert not snapshot.exists(), snapshot
            snapshot.write_bytes(path.read_bytes())
            data["dependencies"].append(
                dict(path=absolute, sha256=base.sha(path), snapshot=str(snapshot), snapshot_sha256=base.sha(snapshot))
            )
            already.add(absolute)
        data.update(
            details,
            corrected_fma_wrapper_source=str(Path(__file__).resolve()),
            corrected_fma_wrapper_sha256=base.sha(__file__),
        )
        assert graph.__call__ is original_call
        assert graph.mm is mm
        data["output_call_identity_unchanged"] = True
        data["state_call_identity_unchanged"] = (
            graph.state_call is output_probe.ordered.OrderedChunk64StateGraph.__call__
        )
        data["corrected_mm_dispatch_installed"] = True
        assert data["state_call_identity_unchanged"]
        return data

    graph.__init__, graph.mm = initialize, mm
    output_probe.dependency_metadata = metadata
    # The inner wrapper merges its audit metadata last; provide the combined
    # description so it cannot overwrite the corrected-GEMM scope/precision claim.
    previous.audit = lambda: details
    try:
        previous.run(args)
    finally:
        previous.audit = original_scaled_audit
        output_probe.dependency_metadata = original_metadata
        for name, value in original_dict.items():
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
    parser.add_argument(
        "--variant",
        choices=["sfpu_fma"],
        default="sfpu_fma",
        help="Compatibility field only; mm always uses corrected FMA",
    )
    parser.add_argument("--sum-order", choices=["canonical"], default="canonical")
    parser.add_argument(
        "--cumsum",
        choices=["compensated"],
        default="compensated",
        help="Compatibility field only; graph uses double-single cumsum",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--oracle-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
