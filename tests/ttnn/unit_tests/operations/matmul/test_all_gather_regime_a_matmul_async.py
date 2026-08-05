# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Correctness tests for the fused Regime-A all-gather matmul described in
# tools/mm_sweep/REGIME_A_AGMM_DESIGN_SPEC.md.
#
# THE OP DOES NOT EXIST YET -- these are the bring-up target. Every test skips
# cleanly until `ttnn.experimental.all_gather_regime_a_matmul_async` is bound, so
# the file is runnable today and turns green incrementally as the op lands.
#
# Expected signature (this file IS the interface contract; update both together):
#
#   ttnn.experimental.all_gather_regime_a_matmul_async(
#       input_tensor,                    # in0 [M, K/TP], K-sharded over cluster_axis
#       weight_tensor,                   # in1 [K, N], FULL K, device-local, DRAM width-sharded
#       config=None,                     # RegimeAMatmulConfig, None => picker
#       *,
#       bias_tensor=None,                # [1, N]
#       fused_activation=None,           # UnaryWithParam, applied after bias
#       fused_ternary_scalar=None,       # addcmul; exclusive with fused_activation
#       fused_ternary_input_a=None,      # residual [M, N]
#       fused_ternary_input_b=None,      # gate [1, N] or [M, N]
#       persistent_output_buffer=None,   # gathered-A scratch [M, K]
#       multi_device_global_semaphore=None,
#       barrier_semaphore=None,
#       num_links=1,
#       topology=ttnn.Topology.Ring,
#       cluster_axis=0,
#   ) -> ttnn.Tensor                     # [M, N], REPLICATED across the TP group
#
# Numerics follow regime_a_matmul and are fixed: BF16 in/out, HiFi2, FP32 dest acc.
#
# WHICH GATHER PATH RUNS is an environment decision, so this one file covers all three:
#
#   (unset)                                        Phase 0: all_gather_async + matmul, the correctness oracle
#   TT_AGMM_FUSED_GATHER=1                         Phase 1: fused gather via a DRAM staging buffer -- 40/40
#   TT_AGMM_FUSED_GATHER=1 TT_AGMM_DIRECT_L1=1     Phase 2: fabric writes straight into cb0 -- 28/40, with
#                                                  the other 12 REFUSED at program creation (Ns>1, and >64
#                                                  mux channels on a LINE at num_links=1), not failing.
#                                                  See tools/mm_sweep/AGMM_DIRECT_L1_DESIGN.md, "Scope
#                                                  limits of the implementation".
#
# A refusal is a TT_FATAL, never a silent fallback -- so a Phase-2 run can never be reported as Phase 2 when
# it actually took the staged path. Note that ONE refusal shows up as many failures: the fixture leaks
# fabric config on a failed open (see open_cluster), so every later test dies on a bogus
# "Tried to override previous value of fabric config".

import contextlib

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

OP_NAME = "all_gather_regime_a_matmul_async"

# ---------------------------------------------------------------------------
# Shapes. Regime-A structural requirements (see the HeyGen filter in
# tools/mm_sweep/picker_gen/P150_VS_GLX.md): M < N, Nt wide enough to width-shard
# over 8 banks, and Kt >= 8. Here Kt is the PER-DEVICE count, so K/TP must still
# clear 8 tiles at TP=8 -- the smallest shape below gives exactly 8.
# ---------------------------------------------------------------------------
# (label, M, K, N)   K is the GLOBAL K; each device holds K/TP.
SHAPES = [
    ("small", 32, 2048, 2048),  # Kt=64  -> 8/dev at TP=8, 16/dev at TP=4
    ("medium", 256, 5120, 2560),  # Kt=160 -> 20/dev at TP=8, 40/dev at TP=4
    ("large", 512, 6144, 4608),  # Kt=192 -> 24/dev at TP=8, 48/dev at TP=4
]
SHAPE_IDS = [s[0] for s in SHAPES]

TOPOLOGIES = [("ring", ttnn.Topology.Ring), ("line", ttnn.Topology.Linear)]
TOPOLOGY_IDS = [t[0] for t in TOPOLOGIES]

PCC = 0.999


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _require_op():
    if not hasattr(ttnn.experimental, OP_NAME):
        pytest.skip(f"ttnn.experimental.{OP_NAME} not implemented yet")


def _mesh_geometry(tp):
    """Return (parent_shape, cluster_axis) for a TP group, or None if unavailable.

    On a 4x8 Galaxy, TP=4 runs down a column (axis 0) and TP=8 across a row (axis 1),
    matching the bh_4x8 config in models/tt_dit/utils/sweep_mm_block_sizes.py.
    """
    n = ttnn.get_num_devices()
    if n >= 32:
        return ((4, 8), 0 if tp == 4 else 1)
    if n >= tp:
        return ((1, tp), 1)
    return None


@contextlib.contextmanager
def open_cluster(tp, topology):
    """Open a TP-sized submesh with fabric configured for `topology`.

    Fabric must be set before the mesh is opened and reset after it closes, so this
    owns the whole lifecycle. Mirrors open_mesh() in sweep_mm_block_sizes.py.
    """
    geom = _mesh_geometry(tp)
    if geom is None:
        pytest.skip(f"need >= {tp} devices for TP={tp}, have {ttnn.get_num_devices()}")
    (rows, cols), cluster_axis = geom

    fabric = ttnn.FabricConfig.FABRIC_1D_RING if topology == ttnn.Topology.Ring else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(
        fabric,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    try:
        sub_shape = [1, 1]
        sub_shape[cluster_axis] = tp
        submesh = parent.create_submesh(ttnn.MeshShape(tuple(sub_shape)))
        # `parent` is yielded so tests can carve a UNIT submesh for single-chip reference runs:
        # regime_a_matmul queries get_worker_noc_hop_distance(), which hard-asserts num_devices()==1.
        yield parent, submesh, cluster_axis
    finally:
        for s in parent.get_submeshes():
            ttnn.close_mesh_device(s)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


class CclPool:
    """Ping-ponged CCL resources for repeated invocations, mirroring CCLManager in
    models/tt_dit/parallel/manager.py. Three rules, all of which matter:

    1. DOUBLE-BUFFER the persistent output buffer and the semaphore sets. A single set handed to
       back-to-back calls is still receiving the previous invocation's in-flight fabric traffic.
    2. synchronize_device() AFTER allocating them, BEFORE first use. Global semaphores and buffers
       are created with per-device work; without the barrier a fast device can launch the op and
       fire a cross-device atomic-inc at a peer that hasn't allocated/zeroed its copy yet, and the
       increment is silently lost.
    3. Allocate ONCE per mesh and rotate -- not per call.

    Getting this wrong does not crash. It yields intermittent, per-device PARTIAL corruption
    (measured: PCC 0.967-0.984 vs a 0.999 target on a subset of devices, clean on the next
    iteration), which is easy to misread as a program-cache bug.
    """

    def __init__(self, mesh, M, K, depth=2):
        self.mesh = mesh
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

        ttnn.synchronize_device(mesh)  # rule 2: everyone ready to allocate
        self.sem_sets = [[ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)] for _ in range(depth)]
        self.barriers = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(depth)]
        self.buffers = [
            ttnn.from_torch(
                torch.zeros((M, K), dtype=torch.float32),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=mesh,
            )
            for _ in range(depth)
        ]
        ttnn.synchronize_device(mesh)  # rule 2: allocated+zeroed everywhere before any op launches

        self.depth = depth
        self.idx = 0

    def next(self):
        """Return (semaphores, barrier, persistent_buffer) for this call and rotate."""
        i = self.idx
        self.idx = (self.idx + 1) % self.depth
        # CCLManager passes barrier_semaphore only when NOT using a persistent buffer
        # (manager.py all_gather); mirror that so the two paths don't both try to order the gather.
        return self.sem_sets[i], None, self.buffers[i]


def _shard_in0(mesh, cluster_axis, torch_in0):
    """in0 [M, K] -> [M, K/TP] per device, sharded along K over the cluster axis."""
    dims = [None, None]
    dims[cluster_axis] = 1  # tensor dim 1 == K
    return ttnn.from_torch(
        torch_in0,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
    )


def _replicate_weight(mesh, torch_in1):
    """in1 [K, N] full-K on every device, in the DRAM width-sharded layout regime-A requires."""
    mem_cfg = ttnn.create_regime_a_weight_memory_config(list(torch_in1.shape), ttnn.bfloat16, mesh)
    return ttnn.from_torch(
        torch_in1,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh,
        memory_config=mem_cfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _read_replicas(mesh, cluster_axis, tt_out, tp):
    """Return the per-device [M, N] replicas as a list of torch tensors.

    The output is replicated, so concatenating over the cluster axis yields TP stacked
    copies; splitting them back apart lets each device be checked independently, which
    is what makes "every device got the full global-K result" an actual assertion.
    """
    concat_dims = (0, 1)
    stacked = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=concat_dims),
    )
    while stacked.dim() > 2:
        stacked = stacked.squeeze(0)
    return list(torch.chunk(stacked, tp, dim=concat_dims[cluster_axis]))


def _single_chip_reference(a, b):
    """Run regime_a_matmul on a UNIT mesh and return the result as a torch tensor.

    Opened and closed on its own. Creating a second submesh alongside a live TP submesh on the same
    parent HANGS (observed: test wedged indefinitely after fabric init, pytest-timeout never fired),
    so the two meshes must be sequential, never concurrent.
    """
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    unit = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        in0 = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=unit)
        mem_cfg = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, unit)
        in1 = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=unit, memory_config=mem_cfg)
        out = ttnn.experimental.regime_a_matmul(in0, in1, config=None)
        ttnn.synchronize_device(unit)
        got = ttnn.to_torch(ttnn.from_device(out))
        while got.dim() > 2:
            got = got.squeeze(0)
        return got.float()
    finally:
        ttnn.close_mesh_device(unit)


def _reference(a, b, bias=None, act=None, scalar=None, resid=None, gate=None):
    """Torch reference: y = act(a@b + bias) or resid + scalar*(a@b + bias)*gate."""
    y = a.float() @ b.float()
    if bias is not None:
        y = y + bias.float()
    if act == "gelu":
        y = torch.nn.functional.gelu(y)
    elif act == "relu":
        y = torch.relu(y)
    elif act is not None:
        raise ValueError(f"unsupported activation {act}")
    if scalar is not None:
        y = resid.float() + scalar * y * gate.float()
    return y


def _run_agmm(
    tp,
    topology,
    M,
    K,
    N,
    config=None,
    bias=False,
    act=None,
    scalar=None,
    gate_full=False,
    num_iters=1,
    pcc=PCC,
):
    """Build inputs, run the fused op `num_iters` times, PCC-check every device replica."""
    _require_op()
    assert K % (tp * 32) == 0, f"K={K} must shard over TP={tp} on tile boundaries"

    with open_cluster(tp, topology) as (_parent, mesh, cluster_axis):
        op = getattr(ttnn.experimental, OP_NAME)

        # Allocated ONCE for the whole mesh and rotated per iteration -- see CclPool.
        ccl = CclPool(mesh, M, K)

        for it in range(num_iters):
            # Fresh tensors each iteration: different contents AND different buffer
            # addresses, so a cached program must pick the new addresses up.
            torch.manual_seed(it)
            a = torch.randn(M, K, dtype=torch.bfloat16)
            b = torch.randn(K, N, dtype=torch.bfloat16)

            in0 = _shard_in0(mesh, cluster_axis, a)
            in1 = _replicate_weight(mesh, b)
            semaphores, barrier, persistent = ccl.next()

            kw = dict(config=config)
            bias_t = None
            if bias:
                bias_t = torch.randn(1, N, dtype=torch.bfloat16)
                kw["bias_tensor"] = ttnn.from_torch(
                    bias_t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            if act is not None:
                op_type = ttnn.UnaryOpType.RELU if act == "relu" else ttnn.UnaryOpType.GELU
                kw["fused_activation"] = ttnn.UnaryWithParam(op_type)

            resid_t = gate_t = None
            if scalar is not None:
                resid_t = torch.randn(M, N, dtype=torch.bfloat16)
                gate_t = torch.randn(M if gate_full else 1, N, dtype=torch.bfloat16)
                kw["fused_ternary_scalar"] = scalar
                kw["fused_ternary_input_a"] = ttnn.from_torch(
                    resid_t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                kw["fused_ternary_input_b"] = ttnn.from_torch(
                    gate_t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )

            out = op(
                in0,
                in1,
                persistent_output_buffer=persistent,
                multi_device_global_semaphore=semaphores,
                barrier_semaphore=barrier,
                num_links=1,
                topology=topology,
                cluster_axis=cluster_axis,
                **kw,
            )
            ttnn.synchronize_device(mesh)

            ref = _reference(a, b, bias_t, act, scalar, resid_t, gate_t)
            replicas = _read_replicas(mesh, cluster_axis, out, tp)
            assert len(replicas) == tp, f"expected {tp} replicas, got {len(replicas)}"
            for dev, got in enumerate(replicas):
                assert tuple(got.shape) == tuple(ref.shape), f"dev{dev}: {tuple(got.shape)} != {tuple(ref.shape)}"
                assert torch.isfinite(got.float()).all(), f"dev{dev}: non-finite output"
                assert_with_pcc(ref, got.float(), pcc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")


@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize("topo_label,topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
@pytest.mark.parametrize("label,M,K,N", SHAPES, ids=SHAPE_IDS)
def test_agmm_regime_a_correctness(tp, topo_label, topology, label, M, K, N):
    """Core gate: every device must end up with the full global-K result.

    This is the test that catches the failure mode the spec warns about most --
    a K tile consumed twice or never, which shows up as a wrong result on some
    subset of devices rather than a crash.
    """
    _run_agmm(tp, topology, M, K, N)


@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize(
    "cfg_label,Ns,Pk,Sm,kb,nsb",
    [
        # NOTE on Pk: the matmul runs on the GATHERED K (full 5120 = 160 tiles), not the per-device
        # shard. Pk=1 puts all 160 K tiles in one slice and overflows L1 ("planner rejected config:
        # L1 over budget"), so every case here needs Pk >= 2. This is the main way the multi-device
        # config space differs from the single-chip one.
        ("pk2", 1, 2, 1, 2, 1),  # split-K reduction across the gathered K
        ("pk4", 1, 4, 1, 2, 1),  # deeper split-K
        ("ns2", 2, 2, 1, 2, 1),  # Ns>1: groups need identical A, must not duplicate fabric copies
        ("sm2", 1, 2, 2, 2, 1),  # Sm>1: groups own distinct M rows
    ],
    ids=["pk2", "pk4", "ns2", "sm2"],
)
def test_agmm_regime_a_parallel_config(tp, cfg_label, Ns, Pk, Sm, kb, nsb):
    """Pinned Pk/Ns/Sm. The spec calls out Ns>1 and Sm>1 as distinct fabric-ownership
    cases, so they need coverage independent of whatever the picker happens to choose."""
    config = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    _run_agmm(tp, ttnn.Topology.Ring, 256, 5120, 2560, config=config)


@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize(
    "fuse_label,kwargs",
    [
        ("bias", dict(bias=True)),
        ("gelu", dict(act="gelu")),
        ("bias_relu", dict(bias=True, act="relu")),
        ("addcmul_bcast", dict(scalar=1.0)),
        ("addcmul_full", dict(scalar=0.5, gate_full=True)),
    ],
    ids=["bias", "gelu", "bias_relu", "addcmul_bcast", "addcmul_full"],
)
def test_agmm_regime_a_fused_epilogue(tp, fuse_label, kwargs):
    """Epilogues must fire exactly once, after the complete global-K result reaches the
    reduction endpoint -- applying them per-shard would silently multiply the bias by TP."""
    _run_agmm(tp, ttnn.Topology.Ring, 256, 5120, 2560, **kwargs)


@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize("topo_label,topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
def test_agmm_regime_a_cache_replay(tp, topo_label, topology):
    """Three back-to-back runs on fresh tensors and fresh semaphores.

    Run 2+ hit the program cache and must pick up new buffer addresses; stale semaphore
    state across runs is the classic CCL bring-up bug and would show here as run 1
    passing and later runs hanging or returning garbage.
    """
    _run_agmm(tp, topology, 256, 5120, 2560, num_iters=3)


@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize("label,M,K,N", SHAPES, ids=SHAPE_IDS)
def test_agmm_regime_a_matches_single_chip(tp, label, M, K, N):
    """Op-vs-op parity: fused multi-device result == single-chip regime_a_matmul on the
    same full-K problem.

    Baseline 1 of the four the spec requires. Comparing against the proven single-chip op
    rather than torch isolates *distribution* bugs from numerics -- if both drift from
    torch identically, the gather is fine and the tolerance is the issue.
    """
    _require_op()

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)

    # Reference FIRST, on its own unit mesh that is fully closed before the TP mesh opens.
    # Two concurrent submeshes on one parent HANGS (wedges after fabric init; pytest-timeout
    # does not fire), so these must be sequential.
    single_ref = _single_chip_reference(a, b)

    with open_cluster(tp, topology=ttnn.Topology.Ring) as (parent, mesh, cluster_axis):
        # Fused, K-sharded across the TP group.
        in0 = _shard_in0(mesh, cluster_axis, a)
        in1 = _replicate_weight(mesh, b)
        semaphores, barrier, persistent = CclPool(mesh, M, K).next()
        fused = getattr(ttnn.experimental, OP_NAME)(
            in0,
            in1,
            config=None,
            persistent_output_buffer=persistent,
            multi_device_global_semaphore=semaphores,
            barrier_semaphore=barrier,
            num_links=1,
            topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis,
        )
        ttnn.synchronize_device(mesh)
        fused_replicas = _read_replicas(mesh, cluster_axis, fused, tp)

        for dev, got in enumerate(fused_replicas):
            assert_with_pcc(single_ref, got.float(), 0.9999)
