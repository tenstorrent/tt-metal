# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only: which of the Sol pre-pass gathers wait on a ring barrier (no device).

Run:
  pytest models/tt_dit/tests/models/minimax_h3/test_sol_prepass_gather_minimax_h3.py
"""

from types import SimpleNamespace

import ttnn

from ....models.transformers.minimax_h3 import sol_prepass
from ....models.transformers.minimax_h3.sol_prepass import SolPrepass


def test_k_gather_waits_on_a_ring_barrier(monkeypatch):
    """The K gather is the first cross-chip write of a pre-pass call and lands in a fresh buffer: it must carry a
    barrier semaphore (two runs of one seed differed at 10 s without it), and consecutive calls must not share one.
    The V gather that follows needs none."""
    gathers = []
    fake_ttnn = SimpleNamespace(
        bfloat16=ttnn.bfloat16,
        create_global_semaphore=lambda *a: object(),
        experimental=SimpleNamespace(all_gather_async=lambda t, **kw: gathers.append(kw) or object()),
    )
    monkeypatch.setattr(sol_prepass, "ttnn", fake_ttnn)
    barriers = [object(), object()]
    handed_out = []
    ccl = SimpleNamespace(
        ccl_cores=None,
        num_links=1,
        topology=None,
        get_barrier_semaphore=lambda axis: handed_out.append(axis) or barriers[len(handed_out) % 2],
    )
    prepass = SolPrepass(mesh_device=None, ccl_manager=ccl, sp_axis=1, tau=1.0)
    prepass._state = {"scale": 0.5}
    prepass.geo = None
    monkeypatch.setattr(prepass, "_run_pool", lambda q, k, v: (object(), object(), object()))
    monkeypatch.setattr(prepass, "_run_route", lambda q_bar, kc: object())
    x = SimpleNamespace(dtype=ttnn.bfloat16)

    for _ in range(2):
        prepass.forward(x, x, x, scale=0.5)

    k1, v1, k2, v2 = (g["barrier_semaphore"] for g in gathers)
    assert k1 is not None and k2 is not None and k1 is not k2
    assert v1 is None and v2 is None
    assert handed_out == [1, 1]  # the barrier of the SP ring
