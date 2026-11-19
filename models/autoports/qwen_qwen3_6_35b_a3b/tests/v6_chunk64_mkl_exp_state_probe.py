# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Separate chunk64 diagnostic adding MKL normal-range exp to double-single.

The existing base, canonical-sum and double-single sources are never edited.
Only the graph exp method and its setup are replaced in this process. Generic
mathematical constants are uploaded once; no oracle state or outputs are used.
"""

import argparse
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_double_single_state_probe as previous
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_mkl_exp_ttnn_probe as helper

base = previous.base


def run(args):
    graph = previous.ordered.OrderedChunk64StateGraph
    original_setup = base.cpu_setup
    original_init = graph.__init__
    original_attributes = {name: graph.__dict__.get(name) for name in ("__init__", "exp")}
    source_paths = (
        base.__file__,
        previous.ordered.__file__,
        previous.__file__,
        previous.helper.__file__,
        helper.__file__,
        __file__,
    )
    source_hashes = {str(Path(p).resolve()): base.sha(p) for p in source_paths}

    def initialize(self, ttnn, mesh, upload, **kwargs):
        original_init(self, ttnn, mesh, upload, **kwargs)
        self.mkl_negative_exp = helper.MKLNegativeExp(ttnn, mesh, upload, self.mem)

    def exp(self, value):
        return self.mkl_negative_exp(value)

    def metadata():
        return dict(
            exp_candidate="MKL EXHAynn normal-range table128; explicit rare-negative flush",
            mkl_exp_wrapper=str(Path(__file__).resolve()),
            mkl_exp_wrapper_sha256=base.sha(__file__),
            mkl_exp_helper=helper.audit(),
            mkl_exp_patch_proof="Only OrderedChunk64StateGraph.__init__ and exp are replaced; inherited double-single __call__ remains unchanged",
            unchanged_source_hashes=source_hashes,
            oracle_assisted=False,
            acceptance_eligible=False,
        )

    def setup(options):
        result, payload = original_setup(options)
        result.update(metadata())
        result["mkl_exp_cpu_adapter_checks"] = helper.cpu_check(payload)
        for label, path in (("mkl_exp_wrapper", __file__), ("mkl_exp_helper", helper.__file__)):
            snapshot = Path(str(args.output) + f".{label}.py.snapshot")
            snapshot.write_bytes(Path(path).read_bytes())
            result[label + "_snapshot"] = str(snapshot)
            result[label + "_snapshot_sha256"] = base.sha(snapshot)
        return result, payload

    graph.__init__, graph.exp, base.cpu_setup = initialize, exp, setup
    try:
        previous.run(args)
    finally:
        base.cpu_setup = original_setup
        for name, value in original_attributes.items():
            if value is None:
                delattr(graph, name)
            else:
                setattr(graph, name, value)
        assert all(base.sha(p) == digest for p, digest in source_hashes.items())
        provenance = Path(str(args.output) + ".provenance.json")
        if provenance.exists():
            data = json.loads(provenance.read_text())
            data.update(metadata())
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
        help="Compatibility; double-single replaces this option",
    )
    p.add_argument("--sum-order", choices=["native", "canonical"], default="canonical")
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
