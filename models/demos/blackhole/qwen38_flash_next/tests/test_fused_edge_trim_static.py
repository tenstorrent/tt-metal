# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_EDGE_TRIM without a device: off by default, the option plumbing, the registry switch, and the call sites."""

from __future__ import annotations

import inspect

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import embedding, ple
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import edge_trim, registry


def test_off_by_default_and_options():
    assert (
        edge_trim.options({}) == edge_trim.DEFAULT_OPTS and edge_trim.options({edge_trim.ENV: "0"}) == ()
    )  # default on
    assert edge_trim.options({edge_trim.ENV: "1"}) == edge_trim.DEFAULT_OPTS == ("embed_reduce", "ple_kv")
    assert "candidate_row" not in edge_trim.DEFAULT_OPTS and set(edge_trim.DEFAULT_OPTS) < set(edge_trim.ALL_OPTS)
    assert edge_trim.options({edge_trim.ENV: "1", edge_trim.OPTS_ENV: "candidate_row"}) == ("candidate_row",)
    assert edge_trim.options({edge_trim.OPTS_ENV: "candidate_row"}) == (
        "candidate_row",
    )  # default on: the list applies
    assert edge_trim.options({edge_trim.ENV: "0", edge_trim.OPTS_ENV: "candidate_row"}) == ()
    with pytest.raises(ValueError):  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        edge_trim.options({edge_trim.ENV: "1", edge_trim.OPTS_ENV: "bogus"})


def test_registry_switch_adds_candidate_row_only_when_asked():
    assert registry.edge_trim_names({}) == frozenset()
    assert registry.edge_trim_names({edge_trim.ENV: "1"}) == frozenset()
    assert registry.edge_trim_names({edge_trim.ENV: "1", edge_trim.OPTS_ENV: "candidate_row"}) == {"candidate_row"}
    source = inspect.getsource(registry.enabled_names)
    assert "names & (TP8_READY | edge)" in source and "edge_trim_names(environ)" in source
    assert "candidate_row" not in registry.TP8_READY  # not served at TP=8 without the switch


def test_call_sites_are_behind_the_switch_with_the_chain_kept():
    reduce_src = inspect.getsource(embedding._all_reduce_owner_select_hidden)
    assert (
        'edge_trim.on("embed_reduce")' in reduce_src
        and "ttnn.all_reduce(" in reduce_src
        and "ttnn.mesh_partition(" in reduce_src
    )
    project = inspect.getsource(ple.Qwen38TTNNPLE._project)
    assert (
        'edge_trim.on("ple_kv")' in project and "edge_trim.hidden_gather(" in project and "self.weights.key" in project
    )
    assert 'edge_trim.on("ple_kv")' in inspect.getsource(ple.Qwen38TTNNPLE.__init__)
    gather = inspect.getsource(edge_trim.hidden_gather)
    assert "int(x.shape[2]) == 1" in gather and "ttnn.all_gather(" in gather
