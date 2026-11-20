# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""F278: an unfinished perf task names the prior work on its subject (brief.prior_work) before it runs."""

from models.demos.common.bringup.core.prior_work import _hits, needs_prior_work, terms_of


def test_perf_task_needs_prior_work():
    t = {"id": "P.9", "step": "perf", "role": "perf", "title": "expert replication", "brief": {"details": "x"}}
    assert needs_prior_work(t, None) and needs_prior_work(t, "TODO")
    assert needs_prior_work(t, "PASS") == [] and needs_prior_work(t, "REJECTED") == []
    t["brief"]["prior_work"] = "P.46 / P.48 (branch moe_replication): static in-sample -8 %; frozen = off"
    assert needs_prior_work(t, None) == []
    assert needs_prior_work({"id": "X.1", "step": "perf", "title": "a measurement"}, None) == []  # no role: no agent


def test_hits_find_related_tasks():
    tasks = [
        {"id": "P.46", "title": "MOE_REPLICAS static table of hot experts", "brief": {}},
        {"id": "P.1", "title": "rope"},
    ]
    rows = _hits(tasks, ["replicas", "experts"], lambda i: "PASS")
    assert len(rows) == 1 and rows[0].startswith("P.46") and "PASS" in rows[0]
    assert "replication" in terms_of({"title": "OWNER 2026-11-19 perf: expert replication"})
