# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F264: the golden state at a serving handoff point (a request stopped before the engine's actual_end, DeepSeek-V4.1
P.49) comes from the generator's own pipeline, stays outside the pinned content hash, and has its own drift for the
contract's limits. CPU only, on the fixture with layer 1 as a linear recurrence (a fixed-size state)."""

import json

import pytest
import torch

from models.demos.common.bringup.reference import handoff as H
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.selftest import fixture_model
from models.demos.common.bringup.selftest.test_mixed_state import mspec, reference_state_at  # noqa: F401 (fixture)


@pytest.fixture
def hspec(mspec, monkeypatch):  # noqa: F811
    # (the default text moved with models/tt_transformers -> models/ttt_compat, #62523)
    s = mspec(text={"source": "models/ttt_compat/tests/tale-of-two-cities.txt.bz2"})
    # the contract request ends at 224 (s256, tail 32); the model hands off 16 earlier
    monkeypatch.setattr(fixture_model, "contract_handoffs", lambda spec, n: [n - 16], raising=False)
    return s


def test_handoff_snapshot_is_the_generators_state_and_keeps_the_hash(hspec):
    s = hspec
    g = Golden.for_rung(s, "s256")
    h = g.manifest["content_hash"]
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        g.state(1, at=208)
    rec = H.golden(s)
    assert rec["points"] == [208] and rec["checked_at"] == 192
    g = Golden.for_rung(s, "s256")
    assert g.verify() and g.manifest["content_hash"] == h  # derived: the pinned golden is unchanged
    want = reference_state_at(s, 1, 208)["recurrent"]
    torch.testing.assert_close(g.state(1, at=208)["recurrent"], want.float(), rtol=1e-4, atol=1e-4)
    assert not torch.equal(g.state(1, at=208)["recurrent"], g.state(1, at=224)["recurrent"])
    assert H.golden(s) == {}  # nothing left to make
    f = g.dir / H.snap_name(1, 208)
    f.write_bytes(f.read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(ValueError, match="sha256"):
        g.state(1, at=208)


def test_handoff_drift_gives_the_contract_limits_at_the_point(hspec):
    from models.demos.common.bringup.testing import e2e_limits as E

    s = hspec
    H.golden(s)
    d = H.drift(s)
    assert set(d) == {"208"}
    e = H.load_drift(Golden.for_rung(s, "s256").dir, "s256", 208)
    assert "contract_state_recurrent_L01" in e and e["contract_state_key_L00"]["region"] == [0, 0, 192]
    assert 0 < E.state_limit(e, "recurrent") <= 1
    with pytest.raises(FileNotFoundError, match="handoff point 192"):
        H.load_drift(Golden.for_rung(s, "s256").dir, "s256", 192)


def test_a_changed_reference_writes_nothing(hspec, monkeypatch):
    s = hspec
    native = fixture_model.reference

    class Moved(fixture_model.Reference):
        def component(self, layer, name):
            fn = super().component(layer, name)
            return lambda ctx, *x: fn(ctx, *x) * 1.001

    monkeypatch.setattr(
        fixture_model,
        "reference",
        lambda spec, layers=None, dtype=torch.float32, numerics="native": Moved(spec, layers, dtype),
    )
    with pytest.raises(SystemExit, match="no longer reproduces"):
        H.golden(s)
    monkeypatch.setattr(fixture_model, "reference", native)
    assert H.index(Golden.for_rung(s, "s256").dir) is None
    assert json.loads((Golden.for_rung(s, "s256").dir / "manifest.json").read_text())["content_hash"]
