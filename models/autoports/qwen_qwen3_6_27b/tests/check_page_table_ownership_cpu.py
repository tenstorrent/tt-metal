# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU lease semantics only; no claim about device fill or full-layer correctness."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PagedState:
    key: object
    value: object
    page_table: object


class AliasingTensor:
    shape = (3, 8)

    def __deepcopy__(self, memo):
        # Conservative legal device-wrapper behavior: Python copy may alias storage.
        return self

    def buffer_address(self):
        return 1234


def check(source):
    namespace = {"PagedState": PagedState}
    exec(source.replace("from .optimized_decoder import PagedState", ""), namespace)
    cls = namespace["CertifiedPagedState"]
    certified = namespace["page_table_certified"]
    internal = namespace["page_table_for_attention"]
    certify = namespace["certify_host_page_rows"]
    tests = []
    for copier in (copy.copy, copy.deepcopy):
        state = cls(object(), object(), AliasingTensor(), certified=True)
        alias = copier(state)
        assert internal(alias) is internal(state)
        assert certified(state) and certified(alias)
        assert state.page_table_shape == (3, 8) and state.page_table_address == 1234
        assert certified(state), "Metadata access unnecessarily revokes eligibility"
        exposed = alias.page_table
        assert exposed is internal(state)
        assert not certified(state) and not certified(alias), "Copied state retained stale disjointness certificate"
        alias.page_table = AliasingTensor()
        assert not certified(alias) and not certified(state)
        state._replace_with_fresh_private_table(AliasingTensor(), certified=True)
        assert certified(state) and not certified(alias)
        tests.append(copier.__name__)
    state = cls(None, None, AliasingTensor(), certified=True)
    alias = copy.copy(state)
    old = internal(state)
    state._replace_with_fresh_private_table(AliasingTensor(), certified=True)
    assert certified(state) and not certified(alias) and internal(alias) is old
    assert certify([[0, 1, -1], [2, 3, -1]])
    assert not certify([[0, 1], [1, 2]])
    assert not certified(PagedState(None, None, AliasingTensor()))
    return tests + [
        "fresh replacement revokes old aliases",
        "skip entries",
        "overlap rejection",
        "external state fallback",
    ]


if __name__ == "__main__":
    doc = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"
    original_patch = (doc / "batched_cache_fill_integration.patch").read_text()
    added = original_patch.split("@@ -0,0 +1,61 @@\n", 1)[1].split("--- a/", 1)[0]
    original = "\n".join(line[1:] for line in added.splitlines() if line.startswith("+"))
    try:
        check(original)
    except AssertionError as exc:
        negative_control = str(exc)
    else:
        raise AssertionError("Negative control did not reproduce deepcopy alias flaw")
    production_source = doc.parents[1] / "tt/page_table_ownership.py"
    tests = check(production_source.read_text())
    result = {
        "status": "pass",
        "scope": "CPU lease semantics with aliasing tensor stand-in",
        "negative_control": negative_control,
        "checks": tests,
        "device_evidence": False,
        "production_source": str(production_source),
    }
    (doc / "page_table_ownership_cpu_production.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
