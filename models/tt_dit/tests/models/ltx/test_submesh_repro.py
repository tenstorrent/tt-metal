# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.tt_dit.layers.audio_ops import _t_neighbor_pad
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor


def _run_ag(ccl, mesh, label):
    """Run a barrier-semaphore all-gather on ``mesh`` and report whether it completes."""
    t = bf16_tensor(torch.randn(1, 1, 32, 256 * mesh.shape[1]), device=mesh, mesh_axis=1, shard_dim=3)
    out = ccl.all_gather(t, dim=3, mesh_axis=1, use_hyperparams=False)
    ttnn.synchronize_device(mesh)
    print(f"[REPRO] all_gather OK ({label}): {tuple(out.shape)}", flush=True)


def _run_ag_persistent(ccl, mesh, label):
    """Run the vocoder's exact all-gather: dim=1 over mesh_axis=1, persistent ping-pong buffer,
    NO barrier semaphore (the persistent-buffer path passes barrier_semaphore=None). This is the
    op the E2E hangs on; the plain barrier-sem all_gather (_run_ag) is a different code path."""
    t = bf16_tensor(torch.randn(1, 32 * mesh.shape[1], 32, 64), device=mesh, mesh_axis=1, shard_dim=1)
    out = ccl.all_gather_persistent_buffer(t, dim=1, mesh_axis=1)
    ttnn.synchronize_device(mesh)
    print(f"[REPRO] all_gather(persistent) OK ({label}): {tuple(out.shape)}", flush=True)


def _run_neighbor_pad(ccl, mesh, label, padding_mode="zeros"):
    """Run the vocoder conv's actual CCL: a T-axis halo exchange (neighbor_pad_async) on a
    T-sharded BTC tensor, via the same _t_neighbor_pad the vocoder uses. This — not all_gather —
    is the op the E2E hangs on (manager.py:310, get_np_ping_pong_buffer, first neighbor_pad)."""
    pcfg = ParallelFactor(factor=mesh.shape[1], mesh_axis=1)
    t_total = 128
    x = bf16_tensor(torch.randn(1, t_total, 256), device=mesh, mesh_axis=1, shard_dim=1, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = _t_neighbor_pad(x, pad_left=3, pad_right=0, parallel_config=pcfg, ccl_manager=ccl, padding_mode=padding_mode)
    ttnn.synchronize_device(mesh)
    print(f"[REPRO] neighbor_pad({padding_mode}) OK ({label}): {tuple(out.shape)}", flush=True)


_PARAMS = [
    [
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 300000000,
            "num_command_queues": 2,
        },
    ]
]


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_subset_submesh_standalone(mesh_device):
    """H1: a 1x4 strict-subset submesh all_gather (Linear, cq0), with NO parent op first.

    Mirrors the CI-passing test_all_gather_async_2x4 geometry. If this hangs, subset-submesh
    CCL is itself broken on this build/topology. If it passes, the failure is in the overlap.
    """
    parent = mesh_device
    audio = parent.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    _run_ag(audio_ccl, audio, "subset 1x4 standalone cq0")
    print("[REPRO] PASS: subset-submesh standalone all_gather completed", flush=True)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_subset_submesh_neighbor_pad_standalone(mesh_device):
    """H8: a 1x4 strict-subset submesh NEIGHBOR_PAD (cq0), with NO parent op first.

    Control for H7: if neighbor_pad works standalone but H7 (after a heavy parent) hangs, the
    failure is the overlap-with-prior-parent-CCL, not neighbor_pad on a subset submesh per se.
    """
    parent = mesh_device
    audio = parent.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    for i in range(8):
        mode = "replicate" if i % 2 == 0 else "zeros"
        _run_neighbor_pad(audio_ccl, audio, f"subset 1x4 standalone npad {i}", padding_mode=mode)
    print("[REPRO] PASS: subset-submesh standalone neighbor_pad completed", flush=True)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_parent_then_child_cq0(mesh_device):
    """H2: parent 4x8 ring all_gather (cq0), then child 1x4 subset all_gather (cq0).

    No cq split. Isolates whether the parent-then-child shared-router overlap deadlocks
    independent of the command queue.
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)
    _run_ag(ccl, mesh, "parent 4x8 cq0")

    audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(parent)
    _run_ag(audio_ccl, audio, "child 1x4 cq0 after parent")

    # Both meshes hold cq 0 in_use; clear it so the per-cq close guard does not throw at teardown.
    ttnn.reset_cq_in_use(audio)
    ttnn.reset_cq_in_use(mesh)
    print("[REPRO] PASS: parent-then-child both cq0 completed", flush=True)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_child_cq0_repeated(mesh_device):
    """H4: parent 4x8 ring (cq0), then the SAME 1x4 child runs all_gather REPEATEDLY (cq0).

    Models repeated decode_audio calls on the overlapping submesh. The full audio-only test
    showed decode #1 completing then decode #2 hanging; this isolates whether repeated child CCL
    on cq0 is the cause, with and without a between-iteration parent synchronize_device.
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)
    _run_ag(ccl, mesh, "parent 4x8 cq0")

    audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(parent)

    for i in range(4):
        _run_ag(audio_ccl, audio, f"child 1x4 cq0 iter {i}")

    # Both meshes hold cq 0 in_use; clear it so the per-cq close guard does not throw at teardown.
    ttnn.reset_cq_in_use(audio)
    ttnn.reset_cq_in_use(mesh)
    print("[REPRO] PASS: repeated child cq0 all_gather completed", flush=True)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_parent_then_child_cq1(mesh_device):
    """H3: parent 4x8 ring all_gather (cq0), then child 1x4 subset all_gather (cq1).

    This is the LTX_AUDIO_SUBMESH=1x4 pattern. Device-confirmed to deadlock at the child's
    CCL (workers stuck at NSMW on the 4 child chips, dispatched on cq1).
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)
    _run_ag(ccl, mesh, "parent 4x8 cq0")

    audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(parent)
    with ttnn.command_queue(1):
        _run_ag(audio_ccl, audio, "child 1x4 cq1 after parent")
    ttnn.synchronize_device(parent)
    _run_ag(ccl, mesh, "parent 4x8 cq0 after child")

    # The child dirtied cq 0 building its global semaphores even though its CCL ran on cq 1, so the
    # per-cq close guard would throw at teardown for every mesh in the chain sharing cq 0. Finish +
    # reset in_use on the child and the parent submesh so conftest teardown closes them cleanly.
    ttnn.reset_cq_in_use(audio)
    ttnn.reset_cq_in_use(mesh)
    print("[REPRO] PASS: parent cq0 + child cq1 + parent cq0 completed", flush=True)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_heavy_parent_then_child_multiop_cq0(mesh_device):
    """H5: HEAVY parent CCL volume (cq0), quiesce, then child runs a MULTI-OP CCL chain (cq0).

    The light-parent overlap (H2/H4: one parent all_gather, then child) passes; the full E2E
    (video DiT ring-attention + VAE decode, then the child vocoder CCL) hangs even with the
    parent quiesced. This probes whether the VOLUME of prior parent CCL — not the act of
    overlapping — is what leaves the shared-chip EDM/erisc router state that a host synchronize
    does not reset, so the child can no longer re-handshake through it. The parent runs many ring
    all_gathers (approximating the video pipeline's CCL load); after a quiesce the child runs a
    multi-op chain on cq0 (approximating the vocoder resblocks' repeated all_gather). If this
    hangs where H2/H4 passed, the accumulated-shared-chip-router-state hypothesis is confirmed.
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)

    HEAVY_REPS = 64
    for i in range(HEAVY_REPS):
        _run_ag(ccl, mesh, f"parent 4x8 cq0 heavy {i}")
    print(f"[REPRO] parent heavy CCL volume done ({HEAVY_REPS} all_gathers)", flush=True)

    audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(parent)

    # Multi-op child chain on the shared chips (vocoder-like): each op opens/closes its own
    # worker->EDM connection on chips the heavy parent just drove.
    CHILD_OPS = 8
    for i in range(CHILD_OPS):
        _run_ag(audio_ccl, audio, f"child 1x4 cq0 multiop {i}")

    ttnn.reset_cq_in_use(audio)
    ttnn.reset_cq_in_use(mesh)
    print("[REPRO] PASS: heavy parent + child multi-op cq0 completed", flush=True)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_heavy_parent_then_child_persistent_cq0(mesh_device):
    """H6: heavy parent CCL (cq0), quiesce, then a child multi-op chain using the PERSISTENT-buffer
    all_gather — the exact code path the vocoder uses (dim=1, mesh_axis=1, no barrier semaphore).

    H5 (plain barrier-sem all_gather child) passes, so if H6 hangs the difference is the
    persistent-buffer / no-barrier-semaphore CCL path on the overlapping child, not the overlap or
    the parent CCL volume. This is the closest cheap repro of the failing E2E vocoder all_gather.
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)

    HEAVY_REPS = 64
    for i in range(HEAVY_REPS):
        _run_ag(ccl, mesh, f"parent 4x8 cq0 heavy {i}")
    print(f"[REPRO] parent heavy CCL volume done ({HEAVY_REPS} all_gathers)", flush=True)

    audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(parent)

    CHILD_OPS = 8
    for i in range(CHILD_OPS):
        _run_ag_persistent(audio_ccl, audio, f"child 1x4 cq0 persistent {i}")

    ttnn.reset_cq_in_use(audio)
    ttnn.reset_cq_in_use(mesh)
    print("[REPRO] PASS: heavy parent + child persistent-buffer cq0 completed", flush=True)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_heavy_parent_then_child_neighbor_pad_cq0(mesh_device):
    """H7: heavy parent CCL (cq0), quiesce, then a child NEIGHBOR_PAD chain (cq0) — the actual op
    the E2E hangs on (vocoder conv T-causal halo exchange, not all_gather).

    The E2E traceback hangs at manager.py:310 (get_np_ping_pong_buffer) on the first neighbor_pad
    of the overlapping child. H5/H6 (all_gather child) pass; if H7 hangs the failing primitive is
    neighbor_pad_async on the overlapping child, isolated from the rest of the vocoder.
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Ring)

    HEAVY_REPS = 64
    for i in range(HEAVY_REPS):
        _run_ag(ccl, mesh, f"parent 4x8 cq0 heavy {i}")
    print(f"[REPRO] parent heavy CCL volume done ({HEAVY_REPS} all_gathers)", flush=True)

    audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(parent)

    CHILD_OPS = 8
    for i in range(CHILD_OPS):
        mode = "replicate" if i % 2 == 0 else "zeros"
        _run_neighbor_pad(audio_ccl, audio, f"child 1x4 cq0 npad {i}", padding_mode=mode)

    ttnn.reset_cq_in_use(audio)
    ttnn.reset_cq_in_use(mesh)
    print("[REPRO] PASS: heavy parent + child neighbor_pad cq0 completed", flush=True)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_parent_npad_then_child_npad_cq0(mesh_device):
    """H9: parent runs NEIGHBOR_PAD on 4x8 (multicast barrier over the full mesh), quiesce, then the
    overlapping 1x4 child runs neighbor_pad (multicast barrier over its 4 shared chips).

    neighbor_pad's Phase-0 startup barrier is a MULTICAST across the cluster axis (unlike all_gather's
    unicast). The video VAE decode runs neighbor_pad on the full parent mesh before audio; the child's
    neighbor_pad multicast then targets chips the parent's neighbor_pad multicast just drove. This
    tests whether parent-then-child OVERLAPPING NEIGHBOR_PAD (the real multicast-on-multicast case)
    deadlocks where heavy parent all_gather + child neighbor_pad (H7) did not.
    """
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(4, 8))
    ccl = CCLManager(mesh, num_links=2, topology=ttnn.Topology.Linear)

    # Parent neighbor_pad on the full mesh, T-sharded over mesh_axis=1 (the 8-wide axis).
    for i in range(8):
        mode = "replicate" if i % 2 == 0 else "zeros"
        _run_neighbor_pad(ccl, mesh, f"parent 4x8 cq0 npad {i}", padding_mode=mode)
    print("[REPRO] parent 4x8 neighbor_pad done", flush=True)

    audio = mesh.create_submesh(ttnn.MeshShape(1, 4))
    audio_ccl = CCLManager(audio, num_links=2, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(parent)

    for i in range(8):
        mode = "replicate" if i % 2 == 0 else "zeros"
        _run_neighbor_pad(audio_ccl, audio, f"child 1x4 cq0 npad {i}", padding_mode=mode)

    ttnn.reset_cq_in_use(audio)
    ttnn.reset_cq_in_use(mesh)
    print("[REPRO] PASS: parent npad + child npad cq0 completed", flush=True)
