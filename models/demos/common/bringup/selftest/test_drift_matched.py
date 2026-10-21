# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F134: the end-to-end margin is measured on the comparison it gates: the official reference fed as the device is
on each rung (golden prefix, gapped subsets, the contract's length), and pinned once its G task passes. CPU only."""

import json

import pytest
import torch

from models.demos.common.bringup.core import freeze
from models.demos.common.bringup.reference import drift as D
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.selftest import fixture_model
from models.demos.common.bringup.selftest.test_device_side import gspec  # noqa: F401 (fixture)
from models.demos.common.bringup.testing import e2e_limits as E


class _Official(fixture_model.Reference):
    """The fixture's 'official' arithmetic: every step's output rounded to bf16; records the prefixes it is given."""

    loads: list = []

    def component(self, layer, name):
        fn = super().component(layer, name)
        return lambda ctx, *x: fn(ctx, *x).to(torch.bfloat16).float()

    def load_state(self, state, layer, tensors, length):
        _Official.loads.append((layer, length))
        super().load_state(state, layer, tensors, length)


class _Cloning(_Official):
    """A forward that hands its steps a copy of the block input it records: a feed cannot reach them."""

    def forward_chunk(self, tokens, start, state, rec=fixture_model.noop, logits_last_n=0):
        return super().forward_chunk(
            tokens, start, state, lambda n, t: rec(n, t.clone() if n.endswith(".in") else t), logits_last_n
        )


@pytest.fixture
def official(monkeypatch):
    def use(cls=_Official):
        _Official.loads = []
        native = fixture_model.reference

        def reference(spec, layers=None, dtype=torch.float32, numerics="native"):
            return cls(spec, layers, dtype) if numerics == "official" else native(spec, layers, dtype, numerics)

        monkeypatch.setattr(fixture_model, "reference", reference)

    return use


def test_a_prefix_rung_replays_the_last_chunk_from_the_golden_prefix(gspec, official):  # noqa: F811
    official()
    s = gspec()
    _Official.loads = []
    d = D.compute(s, "last")
    assert sorted(_Official.loads) == [(i, 384) for i in range(3)]  # seeded with the native prefix at the last chunk
    assert d["pcc_state_key_L00"]["region"] == [0, 384, 512]  # compared over the positions the device computed
    assert d["pcc_layer_L00"]["drift"] < 1.0
    full = D.compute(s, "s512")
    assert full["pcc_state_key_L00"]["region"] is None


def test_a_gapped_subset_starts_each_run_from_the_golden_input(gspec, official):  # noqa: F811
    """Layer 2 after a gap runs from the golden's block input: its drift is one layer's, not three layers'."""
    from models.demos.common.bringup.core.metrics import pcc
    from models.demos.common.bringup.reference.golden import reference_at

    official()
    s = gspec(layers=[0, 2])
    gapped = D.compute(s, "s256")
    g = Golden.for_rung(s, "s256")
    ref, outs = reference_at(s, "official"), {}
    state = ref.new_state(g.seq)
    for c in range(g.n_chunks):  # the official run propagated through layer 1, as the -official golden had it
        ref.forward_chunk(
            g.tokens()[c * g.chunk : (c + 1) * g.chunk], c * g.chunk, state, lambda n, x: outs.update({n: x})
        )
    unmatched = pcc(outs["L2.out"], g.layer(g.n_chunks - 1, 2)["out"])
    assert 1 - gapped["pcc_layer_L02"]["drift"] < 1 - unmatched


def test_a_forward_that_ignores_the_fed_input_stops(gspec, official):  # noqa: F811
    official(_Cloning)
    with pytest.raises(SystemExit, match="run_block"):
        D.compute(gspec(layers=[0, 2]), "s256")


def test_the_contract_rung_carries_the_read_back_states(gspec, official):  # noqa: F811
    official()
    s = gspec()
    d = D.compute(s, "s256")  # the contract rung (ladder[0]): every chunk, the last ending tail_pad early
    length = 256 - int(s.get("tests.contract_tail_pad"))
    assert d["contract_state_key_L00"]["region"] == [0, 0, length - length % D.BLOCK]
    assert E.state_limit(d, "key", "contract_state") < 1.0
    assert not any(k.startswith("contract_") for k in D.compute(s, "s512"))


def test_the_g_task_writes_each_readers_drift_and_a_changed_drift_fails_the_pinned_gates(gspec):  # noqa: F811
    from models.demos.common.bringup.core.ledger import Ledger

    s = gspec()  # generate_golden --set: drift for s512 and for "last", which reads s512's golden
    d = Golden.for_rung(s, "s512").dir
    assert D.drift_path(d, "s512").exists() and D.drift_path(d, "last").exists()
    assert json.loads(D.drift_path(d, "last").read_text())["pcc_state_key_L00"]["region"] == [0, 384, 512]
    led = Ledger(s.bringup_dir)
    e2e = {"pcc_layer_L*": ">= limit"}
    led.write_tasks(
        {
            "tasks": [
                {"id": "L.last", "title": "l", "deps": [], "device": True, "gate": {"cmd": "x", "metrics": e2e}},
                {
                    "id": "C.x",
                    "title": "c",
                    "deps": [],
                    "device": True,
                    "gate": {"cmd": "x", "metrics": {"err_x": "<= limit"}},
                },
            ]
        }
    )
    assert D.pin(s, led) == ["L.last"]
    task = led.task("L.last")
    assert not freeze.verify(s.repo, task) and "frozen" not in led.task("C.x")
    D.drift_path(d, "last").write_text("{}")
    assert any("drift-last.json" in e for e in freeze.verify(s.repo, task))
