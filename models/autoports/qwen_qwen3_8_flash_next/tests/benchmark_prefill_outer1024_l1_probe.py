# SPDX-License-Identifier: Apache-2.0
"""Run the pinned outer1024 decoder test with the isolated L1/DRAM candidate.

The original probe remains immutable. Only its candidate dependency and report
provenance are substituted in a private function namespace, never module globals.
"""

import ast
import hashlib
import inspect
import types
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_outer1024_decoder_probe as baseline
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_outer1024_l1_candidate as candidate

BASELINE_PROBE_SHA256 = "95598d39793b92a35f51e7c36f553c52d9727b337c4eed7afec5009f89bea8f5"


def main():
    assert hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest() == BASELINE_PROBE_SHA256
    namespace = dict(baseline.__dict__, candidate=candidate, baseline_probe=baseline, __file__=__file__)
    # run_layer is the original bytecode: same baseline512, thirteen lengths,
    # selected policy, shared-public-state checks and original-owner handoff.
    namespace["run_layer"] = types.FunctionType(
        baseline.run_layer.__code__, namespace, baseline.run_layer.__name__, baseline.run_layer.__defaults__
    )
    tree = ast.parse(inspect.getsource(baseline.main))
    function = tree.body[0]
    sources = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "sources" for x in node.targets)
    )
    assert isinstance(sources.value, ast.List)
    sources.value.elts.extend(
        ast.parse("[Path(baseline_probe.__file__), Path(candidate.base.__file__)]", mode="eval").body.elts
    )
    report = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "report" for x in node.targets)
    )
    keyword = next(item for item in report.value.keywords if item.arg == "candidate")
    keyword.value = ast.Constant(
        "isolated GDN outer1024/group1024: L1 projection/residual family only at1024; padded HC norm/restored streams DRAM; QSA internal512 and <=512 unchanged"
    )
    ast.fix_missing_locations(tree)
    exec(compile(tree, __file__ + ":pinned_probe_candidate_dependency", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
