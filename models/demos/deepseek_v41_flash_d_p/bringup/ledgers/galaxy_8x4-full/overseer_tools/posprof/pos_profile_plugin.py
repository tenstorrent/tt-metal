"""Overseer measurement (not in the tree): one 5120-token chunk of the whole model (40 layers) at start position
BRINGUP_POS (owner 2026-10-27: 0, 50k, 100k, 200k, 500k). Timing only, as testing/positions.py: the KV prefix is the
fresh state's zeros, the tokens random ids (the work does not depend on the values). run_profile otherwise unchanged:
warm, the plain wall run (host wall, no profiler), then one real-time-profiler run (device time per section/chip)."""
import os

import torch

POS = int(os.environ["BRINGUP_POS"])
CHUNK = 5120


class _FakeGolden:
    def __init__(self, s):
        self.layers = list(s.layers())
        self.n_chunks = (POS + CHUNK) // CHUNK
        vocab = int(s.get("checkpoint.config.vocab_size", 32000) or 32000)
        g = torch.Generator().manual_seed(0)
        self._tokens = torch.randint(0, vocab, (POS + CHUNK,), generator=g)

    def tokens(self):
        return self._tokens

    def state(self, i, at):
        return None

    def layer(self, c, i):
        return {"out": torch.zeros(1), "in": torch.zeros(1)}


def pytest_configure(config):
    if os.path.exists("/scratch/dnijemcevic/posprof/STOP"):  # stop a series before the mesh opens
        import pytest

        pytest.exit("posprof STOP file present", returncode=0)
    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.core.spec import Spec
    from models.demos.common.bringup.testing import profile

    name = f"pos{POS}"
    orig_rung = Spec.rung
    Spec.rung = lambda self, n: {"name": name, "seq": POS + CHUNK, "chunk": CHUNK} if n == name else orig_rung(self, n)
    profile.profile_rung = lambda s: name
    profile.Golden = type("G", (), {"for_rung": staticmethod(lambda s, n: _FakeGolden(s))})
    orig_pcc = metrics.pcc
    metrics.pcc = lambda a, b: float("nan") if a.shape != b.shape else orig_pcc(a, b)

    orig_load = Spec.load if hasattr(Spec, "load") else None
    orig_hooks = Spec.hooks

    def hooks(self):
        h = orig_hooks(self)
        if not getattr(h, "_pos_wrapped", False):
            dm = h.device_model

            def device_model(mesh, s, layers, **kw):
                s.data["target"]["seq"] = max(int(s.data["target"]["seq"]), POS + CHUNK)  # position tables, in memory
                m = dm(mesh, s, layers, **kw)
                ns = m.new_state

                def new_state(seq):
                    st = ns(seq)
                    st.load_prefix = lambda *a, **k: None  # zero prefix (timing only)
                    return st

                m.new_state = new_state
                return m

            h.device_model = device_model
            h._pos_wrapped = True
        return h

    Spec.hooks = hooks
    print(f"POS PROFILE: chunk [{POS},{POS + CHUNK}), all layers, zero prefix", flush=True)
