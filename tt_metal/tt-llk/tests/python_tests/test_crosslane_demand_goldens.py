# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-lane arsenal: demand-golden fixture gate (lane FB).

Regenerates every fixture from the oracle and requires byte-identity with
the committed crosslane_fixtures/*.json (drift = either an oracle change or
a tampered fixture -- both must be deliberate), plus internal consistency
checks a consumer relies on.
"""

import json
import os

from helpers import crosslane_demand_goldens as dg
from helpers import crosslane_oracle as co


def test_fixtures_match_generator():
    gen = dg.generate_all()
    for name, data in gen.items():
        path = dg.fixture_path(name)
        assert os.path.exists(path), (
            f"fixture {name}.json missing -- run "
            "python -m helpers.crosslane_demand_goldens")
        with open(path) as f:
            committed = json.load(f)
        assert committed == json.loads(json.dumps(data)), (
            f"fixture {name}.json does not match the oracle regeneration "
            "(oracle drift or fixture tamper -- adjudicate before updating)")


def test_bitonic_fixture_consistency():
    with open(dg.fixture_path("bitonic_stages")) as f:
        fx = json.load(f)
    for case in fx["cases"]:
        if case["kv"]:
            ks = [int(x, 16) for x in case["sorted_keys"]]
            inp = [int(x, 16) for x in case["keys"]]
        else:
            ks = [int(x, 16) for x in case["sorted"]]
            inp = [int(x, 16) for x in case["input"]]
        assert sorted(ks) == sorted(inp), "not a permutation"
        keys_m = [co._smkey(v) for v in ks]
        assert keys_m == sorted(keys_m, reverse=(case["order"] == "desc"))
        n = case["n"]
        assert len(case["stages"]) == len(co.bitonic_network_stages(n))


def test_topk_fixture_consistency():
    with open(dg.fixture_path("moe_gate_topk")) as f:
        fx = json.load(f)
    for case in fx["cases"]:
        keys = [int(x, 16) for x in case["keys"]]
        vals = [int(x, 16) for x in case["topk_values_desc"]]
        idxs = case["topk_indices_desc"]
        assert len(vals) == case["k"]
        assert all(keys[i] == v for i, v in zip(idxs, vals))
        full = sorted(keys, key=co._smkey, reverse=True)
        assert vals == full[:case["k"]]


def test_ema_fixture_contracts_present_and_distinct():
    with open(dg.fixture_path("ema")) as f:
        fx = json.load(f)
    for case in fx["cases"]:
        assert case["out_rows_fma"] != case["out_rows_mul_add"], (
            "fma and mul_add contracts identical on this stimulus -- "
            "witness lost, sharpen the stimulus")


def test_tie_fixture_documents_divergence():
    with open(dg.fixture_path("tie_behavior")) as f:
        fx = json.load(f)
    v = fx["variants"]
    # the doc and sim variants must differ somewhere (that IS the finding)
    assert any(v[k] != v[k.replace("_doc", "_sim")]
               for k in v if k.endswith("_doc")), "divergence witness lost"
