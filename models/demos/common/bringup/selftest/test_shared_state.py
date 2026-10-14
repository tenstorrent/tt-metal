# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F65: layers that read other layers' state (hooks state_sources). A layer attending over another layer's KV cache,
or reusing what another layer computed for the chunk (top-k indices, a candidate pool), keeps no state of its own; its
tests load its source layers' state as of the end of the chunk. CPU only, on the fixture (fixture.shared)."""

import pytest

from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.reference import check_hf, check_reference, generate_golden, prompt
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.selftest import fixture_model
from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.testing.component import run_component_test, run_swap_test
from models.demos.common.bringup.testing.ladder import run_ladder

KV = {  # layer 2 attends over layer 0's keys and values
    "block_types": {"attn": {"layers": [0, 1]}, "reuse": {"layers": [2]}},
    "state": {"kind": "kv", "tensors": ["key", "value"], "by_block_type": {"attn": ["key", "value"], "reuse": []}},
    "fixture": {"shared": {2: 0}},
}
REC = {  # layer 2 reads layer 1's recurrent state after the chunk (a fixed-size tensor: snapshots at chunk ends)
    "block_types": {"attn": {"layers": [0]}, "rec": {"layers": [1]}, "reuse": {"layers": [2]}},
    "state": {
        "kind": "mixed",
        "tensors": ["key", "value", "recurrent"],
        "by_block_type": {"attn": ["key", "value"], "rec": ["recurrent"], "reuse": []},
        "fixed": ["recurrent"],
    },
    "fixture": {"recurrent_layers": [1], "shared": {2: 1}},
}


@pytest.fixture
def sspec(fx, monkeypatch):
    def make(cfg, **over):
        p = fx(**{**cfg, **over})
        monkeypatch.setenv("BRINGUP_SPEC", p)
        prompt.build(Spec.load(p))
        generate_golden.main(["--spec", p, "--rung", "s256"])
        return Spec.load(p)

    return make


def test_sources_from_the_hooks(fx):
    s = Spec.load(fx(**KV))
    assert s.validate() == [] and s.state_sources(2) == [0] and s.state_sources(1) == [] and s.state_names(2) == []
    assert Spec.load(fx()).state_sources(2) == []


def test_hf_parity_and_graph_replay_with_a_shared_kv(fx):
    p = fx(**KV)
    check_hf.main(["--spec", p, "--seq", "128"])
    assert min(v for k, v in got().items() if k.startswith("pcc_")) > 0.99999
    check_reference.main(["--spec", p, "--seq", "256", "--chunk", "64"])
    m = got()
    assert m["pcc_hidden"] > 0.999999 and m["graph_errors"] == 0 and m["graph_replay_maxabs"] == 0.0


def test_golden_snapshots_the_source_at_chunk_ends(sspec):
    s = sspec(REC)
    g = Golden.for_rung(s, "s256")
    assert g.state(2) == {} and set(g.state(1)) == {"recurrent"}
    # full dumps: before every chunk, after every chunk (layer 2 reads layer 1 after it), and the contract tail
    assert sorted(g.meta["state_snapshots"]) == [0, 64, 128, 192, 224]
    assert g.state(1, at=128)["recurrent"].abs().sum() > 0


def test_golden_refuses_a_missing_source(fx, monkeypatch):
    p = fx(**KV, layers="1-2")
    prompt.build(Spec.load(p))
    with pytest.raises(SystemExit, match="select layer 0"):
        generate_golden.main(["--spec", p, "--rung", "s256"])


@pytest.mark.parametrize("cfg", [KV, REC], ids=["kv", "recurrent"])
@pytest.mark.parametrize("impl,noise_level,passes", [("device", 1e-3, True), ("stub", 0.0, False)])
def test_component_of_a_reader_gets_its_sources(sspec, monkeypatch, cfg, impl, noise_level, passes):
    from models.demos.common.bringup.core.runs import IMPL_ENV

    s = sspec(cfg)
    monkeypatch.setitem(fixture_model.NOISE, "value", noise_level)
    monkeypatch.setenv(IMPL_ENV, impl)
    assert run_component_test(s, "attention", layer=2) is passes
    if passes:
        assert got()["pcc_attention_L02"] > 0.999


@pytest.mark.parametrize("cfg", [KV, REC], ids=["kv", "recurrent"])
def test_swap_and_ladder_with_readers(sspec, monkeypatch, cfg):
    s = sspec(cfg)
    monkeypatch.setitem(fixture_model.NOISE, "value", 0.0)
    assert run_swap_test(s, "reuse", ["attn_norm", "attention", "attn_residual", "ffn_norm", "mlp"])
    assert got()["pcc_swap_out"] > 0.99999
    out = run_ladder(s, "s256", None)
    assert not out["failed"], out
    assert got()["pcc_layer_L02"] > 0.9999 and "pcc_state_key_L02" not in got()
