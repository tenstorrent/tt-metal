"""F202: a mesh a test opens itself carries every device param of the spec (the fabric router config included), and no
model test opens one by hand."""

import re
from pathlib import Path

import ttnn
from models.demos.common.bringup.testing import harness

REPO = Path(__file__).resolve().parents[5]
HAND_ROLLED = re.compile(r"\bttnn\.(set_fabric_config|open_mesh_device)\(")


class _Spec:
    mesh = (8, 4)

    def __init__(self, params):
        self.params = params

    def hooks(self):
        spec = self

        class H:
            @staticmethod
            def device_params(_):
                return dict(spec.params)

        return H


def test_open_mesh_applies_every_device_param(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "set_fabric_config", lambda *a: calls.append(("fabric", a)))
    monkeypatch.setattr(ttnn, "open_mesh_device", lambda shape, **kw: calls.append(("open", kw)) or "MESH")
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda m: calls.append(("close", m)))
    rc = object()
    s = _Spec({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": rc, "l1_small_size": 24576})
    mesh, fab = harness.open_mesh(s)
    assert mesh == "MESH"
    (kind, args), (kind2, kw) = calls[:2]
    assert kind == "fabric" and args[0] == ttnn.FabricConfig.FABRIC_2D and args[-1] is rc, args
    assert kind2 == "open" and kw == {"l1_small_size": 24576}, kw  # no fabric key leaks into the open
    harness.close_mesh(mesh, fab)
    assert calls[2] == ("close", "MESH") and calls[3][1] == (ttnn.FabricConfig.DISABLED,)


def test_no_fabric_no_fabric_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "set_fabric_config", lambda *a: calls.append(a))
    monkeypatch.setattr(ttnn, "open_mesh_device", lambda shape, **kw: "MESH")
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda m: None)
    mesh, fab = harness.open_mesh(_Spec({"l1_small_size": 0}))
    harness.close_mesh(mesh, fab)
    assert calls == []


def test_no_model_test_opens_a_mesh_by_hand():
    """Model tests open a mesh only through the repo fixture (mesh_parametrize) or harness.open_mesh."""
    bad = []
    for root in sorted((REPO / "models/demos").glob("*/tests/bringup")):
        for f in root.rglob("*.py"):
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if HAND_ROLLED.search(line) and not line.lstrip().startswith("#"):
                    bad.append(f"{f.relative_to(REPO)}:{i}: {line.strip()}")
    assert not bad, "open the mesh with harness.open_mesh (F202):\n" + "\n".join(bad)
