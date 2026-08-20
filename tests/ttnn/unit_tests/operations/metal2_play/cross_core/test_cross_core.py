# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-core probes on the pybound Metal 2.0 ProgramSpec surface.

Ordering matters: everything that could hang lives at the top so a hang does not cost the
host-side negative results below it.
"""

import sys
from pathlib import Path

import pytest
import torch

import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import xc_specs as xc  # noqa: E402

TILE = 32


@pytest.fixture(scope="module")
def device():
    ttnn.CONFIG.validate_program_args = True
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _input(device, num_tiles, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(1, 1, TILE, TILE * num_tiles, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _tile(t, idx, width=1):
    return t[..., idx * TILE : (idx + width) * TILE]


# ============================================================ probe 1: unicast core -> core (ring)


@pytest.mark.parametrize("cols,tiles_per_core,num_entries", [(4, 1, 1), (4, 4, 2), (8, 2, 2), (8, 8, 4)])
def test_ring_unicast_rotates(device, cols, tiles_per_core, num_entries):
    """Each core NoC-writes its tiles into its successor's recv DFB. Output must be rotated by one
    core -- a local read would produce the identity instead."""
    t, inp = _input(device, cols * tiles_per_core, seed=cols)
    out = xc.ring_rotate(inp, cols, tiles_per_core, num_entries)
    got = ttnn.to_torch(out).float()
    for i in range(cols):
        prev = (i - 1) % cols
        assert torch.equal(
            _tile(got, i * tiles_per_core, tiles_per_core), _tile(t.float(), prev * tiles_per_core, tiles_per_core)
        ), f"core {i} did not receive core {prev}'s tiles"


def test_ring_without_reverse_credit(device):
    """Same ring with the hand-rolled reverse credit compiled out. The DFB's own reserve/push flow
    control is node-local, so nothing stops a fast core from overwriting a slow one's ring slot.
    Reported, not asserted: whether it actually corrupts is a timing race."""
    cols, tpc, entries = 8, 16, 4
    t, inp = _input(device, cols * tpc, seed=99)
    out = xc.ring_rotate(inp, cols, tpc, entries, use_credit=False)
    got = ttnn.to_torch(out).float()
    bad = [
        i
        for i in range(cols)
        if not torch.equal(_tile(got, i * tpc, tpc), _tile(t.float(), ((i - 1) % cols) * tpc, tpc))
    ]
    print(f"\n[no reverse credit] cols={cols} tiles/core={tpc} entries={entries} -> corrupted cores: {bad}")


# ================================================ probe 2: raw multicast, role-specialized kernels


@pytest.mark.parametrize("cols", [4, 8])
def test_raw_mcast_broadcasts(device, cols):
    """Sender KernelSpec on core 0, receiver KernelSpec on cores 1..N-1: two PRODUCER bindings on
    one DFB over disjoint node sets, plus `writer` listed in both work units."""
    t, inp = _input(device, cols, seed=100 + cols)
    out = xc.raw_mcast(inp, cols, shared_writer=True)
    got = ttnn.to_torch(out).float()
    for i in range(cols):
        assert torch.equal(_tile(got, i), _tile(t.float(), 0)), f"core {i} did not get the broadcast tile"


def test_work_units_may_not_overlap(device):
    """The obvious factoring -- one work unit per role plus a third grid-wide one for the shared
    writer -- is illegal: WorkUnitSpec target_nodes must be pairwise disjoint. So a kernel that runs
    everywhere has to be re-listed in every role's work unit."""
    cols = 4
    t, inp = _input(device, cols, seed=7)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        xc.raw_mcast_variant(inp, cols, shared_writer=False, dfb_producers="both")
    print("\n[work unit overlap] " + str(e.value)[:900])
    assert "overlap in target nodes" in str(e.value)


# ==================================================== probe 3: same mcast via McastFamily helper


@pytest.mark.parametrize("cols", [4, 8])
def test_family_mcast_broadcasts(device, cols):
    t, inp = _input(device, cols, seed=200 + cols)
    out = xc.family_mcast(inp, cols)
    got = ttnn.to_torch(out).float()
    for i in range(cols):
        assert torch.equal(_tile(got, i), _tile(t.float(), 0)), f"core {i} did not get the broadcast tile"


# ======================================================= probe 4: the producer/consumer rule bites


def test_dfb_with_no_producer_is_rejected(device):
    t, inp = _input(device, 4, seed=1)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        xc.raw_mcast_variant(inp, 4, shared_writer=True, dfb_producers="none")
    print("\n[no-producer] " + str(e.value)[:1200])
    assert "no producer" in str(e.value)


def test_honest_cross_node_dfb_is_rejected(device):
    """The wiring you actually want: producer bound only where the data is read (core 0), consumer
    on every core. This is a cross-node DFB, and Metal 2.0 has no local DFB that can express it."""
    t, inp = _input(device, 4, seed=2)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        xc.raw_mcast_variant(inp, 4, shared_writer=False, dfb_producers="sender_only")
    print("\n[cross-node] " + str(e.value)[:1500])
    assert "malformed at node" in str(e.value)


def test_pure_cross_node_dfb_is_rejected(device):
    """Producer on core 0, consumer on core 1 only -- the textbook cross-node FIFO."""
    cols = 4
    one = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))])
    t, inp = _input(device, cols, seed=3)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        xc.raw_mcast_variant(inp, cols, shared_writer=False, dfb_producers="sender_only", writer_nodes=one)
    print("\n[pure cross-node] " + str(e.value)[:1500])
    assert "malformed at node" in str(e.value)


# ================================================================== probe 5: semaphore plumbing


def test_semaphore_initial_value_is_unreachable():
    """SemaphoreSpec's non-zero initial value exists in C++ (SemaphoreAdvancedOptions) but is not
    pybound, so a Python-authored op cannot start a credit semaphore at N."""
    assert not hasattr(ttnn.SemaphoreSpec, "advanced_options")
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        TypeError
    ):  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.SemaphoreSpec(unique_id="s", target_nodes=ttnn.CoreCoord(0, 0), initial_value=4)


def test_semaphore_placement_disjoint_from_binding_kernels(device):
    """The semaphore is placed on core (7,7); the kernels that bind it run on cores (0..3, 0).
    Does the host care?"""
    cols = 4
    far = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(7, 7), ttnn.CoreCoord(7, 7))])
    t, inp = _input(device, cols, seed=4)
    try:
        out = xc.sem_placement_probe(inp, cols, sem_nodes=far)
    except RuntimeError as e:
        print("\n[sem disjoint placement REJECTED] " + str(e)[:1200])
        return
    got = ttnn.to_torch(out).float()
    assert torch.equal(got, t.float()), "passthrough itself broke"
    print("\n[sem disjoint placement ACCEPTED SILENTLY] program ran and produced correct data")


def test_binding_an_undeclared_semaphore(device):
    cols = 4
    t, inp = _input(device, cols, seed=5)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        xc.sem_placement_probe(inp, cols, sem_nodes=xc._row(cols), declare_sem=False)
    print("\n[undeclared sem] " + str(e.value)[:1200])


# ==================================== probe 6: is a DFB at the same L1 address on every node?


def test_dfb_address_is_uniform_when_node_sets_match(device):
    """The premise every kernel here relies on: my own DFB address is also my peer's."""
    cols = 4
    rep = ttnn.to_torch(xc.address_report(device, cols, asymmetric=False))
    recv = [int(rep[0, 0, i, 0]) for i in range(cols)]
    stage = [int(rep[0, 0, i, 1]) for i in range(cols)]
    print(f"\n[uniform node sets] recv={[hex(a) for a in recv]} stage={[hex(a) for a in stage]}")
    assert len(set(recv)) == 1, "recv DFB is NOT at the same address on every node"
    assert len(set(stage)) == 1, "scratchpad is NOT at the same address on every node"


def test_dfb_address_when_node_sets_differ(device):
    """Node 0 holds {recv, stage}; nodes 1.. hold {pad, recv, stage}. Does `recv` move?"""
    cols = 4
    rep = ttnn.to_torch(xc.address_report(device, cols, asymmetric=True))
    recv = [int(rep[0, 0, i, 0]) for i in range(cols)]
    stage = [int(rep[0, 0, i, 1]) for i in range(cols)]
    marker = [hex(int(rep[0, 0, i, 2])) for i in range(cols)]
    print(f"\n[differing node sets] recv={[hex(a) for a in recv]} stage={[hex(a) for a in stage]} marker={marker}")
    if len(set(recv)) == 1:
        print("[differing node sets] recv address is GLOBAL (same on lean and fat nodes)")
    else:
        print("[differing node sets] recv address is PER-NODE -- reusing a local address as a peer's is WRONG here")


# ======================================= probe 7: what the named bindings make impossible


def test_dfb_accessor_name_mismatch_is_a_compile_error(device):
    """The ProgramDescriptor equivalent of this bug -- two kernels handed different CB indices for
    the same buffer -- is silent. Here the writer is promised the accessor name "out_dfb" while its
    source says `dfb::recv`, and the kernel does not compile."""
    t, inp = _input(device, 4, seed=11)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        xc.passthrough(inp, 4, writer_accessor="out_dfb")
    msg = str(e.value)
    print("\n[accessor name mismatch] " + "\n".join(l for l in msg.splitlines() if "error:" in l)[:600])
    assert "recv" in msg


def test_binding_an_undeclared_dfb(device):
    t, inp = _input(device, 4, seed=12)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        xc.passthrough(inp, 4, declare_dfb=False)
    print("\n[undeclared dfb] " + str(e.value)[:600])


def test_passthrough_baseline_is_green(device):
    t, inp = _input(device, 4, seed=13)
    got = ttnn.to_torch(xc.passthrough(inp, 4)).float()
    assert torch.equal(got, t.float())
