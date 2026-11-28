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
from ....parallel import manager
from ....parallel.manager import CCLManager


def _prepass(monkeypatch):
    """A pre-pass on fakes: its gathers' keyword arguments, the tensors it frees and the barriers' ring axes."""
    gathers, freed = [], []
    fake_ttnn = SimpleNamespace(
        bfloat16=ttnn.bfloat16,
        deallocate=freed.append,
        experimental=SimpleNamespace(all_gather_async=lambda t, **kw: gathers.append(kw) or ("gathered", t)),
    )
    monkeypatch.setattr(sol_prepass, "ttnn", fake_ttnn)
    barriers = [object(), object()]
    handed_out = []
    ccl = SimpleNamespace(
        ccl_cores=None,
        num_links=1,
        topology=None,
        get_barrier_semaphore=lambda axis: handed_out.append(axis) or barriers[len(handed_out) % 2],
        get_sol_gather_semaphores=lambda: [object(), object()],
    )
    prepass = SolPrepass(mesh_device=None, ccl_manager=ccl, sp_axis=1, tau=1.0)
    prepass._state = {"scale": 0.5}
    prepass.geo = None
    monkeypatch.setattr(prepass, "_run_pool", lambda q, k, v: ("q_bar", "k_local", "v_local"))
    monkeypatch.setattr(prepass, "_run_route", lambda q_bar, kc: ("mask", kc))
    return prepass, gathers, freed, handed_out


def test_k_gather_waits_on_a_ring_barrier(monkeypatch):
    """The K gather is the first cross-chip write of a pre-pass call and lands in a fresh buffer: it must carry a
    barrier semaphore (two runs of one seed differed at 10 s without it), and consecutive calls must not share one.
    The V gather that follows needs none."""
    prepass, gathers, _, handed_out = _prepass(monkeypatch)
    x = SimpleNamespace(dtype=ttnn.bfloat16)

    for _ in range(2):
        prepass.forward(x, x, x, scale=0.5)

    k1, v1, k2, v2 = (g["barrier_semaphore"] for g in gathers)
    assert k1 is not None and k2 is not None and k1 is not k2
    assert v1 is None and v2 is None
    assert handed_out == [1, 1]  # the barrier of the SP ring


def test_pooled_off_gathers_k_for_the_routing_only(monkeypatch):
    """Default: the mask with the gathered pooled K and V, nothing freed. `pooled=False`: the mask alone; K is still
    gathered behind its barrier (the route program scores against it) and freed after the route, V is not gathered."""
    prepass, gathers, freed, _ = _prepass(monkeypatch)
    x = SimpleNamespace(dtype=ttnn.bfloat16)
    k, v = ("gathered", "k_local"), ("gathered", "v_local")

    assert prepass.forward(x, x, x, scale=0.5) == {"block_mask": ("mask", k), "pooled_k": k, "pooled_v": v}
    assert len(gathers) == 2 and freed == []

    assert prepass.forward(x, x, x, scale=0.5, pooled=False) == {"block_mask": ("mask", k)}
    assert len(gathers) == 3 and gathers[2]["barrier_semaphore"] is not None
    assert freed == [k]


def test_every_prepass_gathers_on_the_managers_one_semaphore_pool(monkeypatch):
    """A worker keeps a pre-pass per layer and per padded length, and a global semaphore holds L1 for good: with
    four of its own per pre-pass, the 4th length of one process ran into the circular buffers of cached programs.
    They all draw on one pool of the CCL manager: two sets, K and V of a call on different ones."""
    made, gathers = [], []
    monkeypatch.setattr(manager.ttnn, "create_global_semaphore", lambda *a: made.append(object()) or made[-1])
    monkeypatch.setattr(
        sol_prepass.ttnn.experimental,
        "all_gather_async",
        lambda t, **kw: gathers.append(kw["multi_device_global_semaphore"]),
    )
    ccl = SimpleNamespace(
        mesh_device=None, ccl_cores=None, num_links=1, topology=None, get_barrier_semaphore=lambda axis: object()
    )
    ccl._sol_gather_semaphores, ccl._sol_gather_idx = None, 0
    ccl.get_sol_gather_semaphores = lambda: CCLManager.get_sol_gather_semaphores(ccl)

    for _ in range(3):  # three pre-passes, as for three layers or lengths
        prepass = SolPrepass(mesh_device=None, ccl_manager=ccl, sp_axis=1, tau=1.0)
        prepass._gather(object(), barrier=True)
        prepass._gather(object(), barrier=False)

    assert len(made) == 4
    k, v = gathers[:2]
    assert gathers == [k, v] * 3 and not set(k) & set(v)
