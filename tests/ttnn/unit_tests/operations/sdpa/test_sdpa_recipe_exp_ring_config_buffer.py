# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exp ring recipes at the default worker L1 (kernel config buffer regression).

The Tensix kernel config buffer holds every program's kernel binaries; it is the L1 between the kernel
config base and the allocator's unreserved base, so it shrinks as worker_l1_size grows: 70656 B at the
default Blackhole worker L1, ~187 KB at MiniMax H3's worker_l1_size=1344544 (the exp ring suites' default).
At the previously qualified geometries the exp ring COMPENSATED / LOW_PRECISION programs (pack-only -Os
for odd Q, all -O2 for even Q, plus the MUX writer) measure 71.3-76.0 KB, so at the default L1 they threw
``Program size (...) too large for kernel config buffer (70656)`` (e.g. the DiT parity cases
wan_exp_720p_4x32 and h3_exp_4x32). Such builds now take the generic-geometry size flags (pack and unpack
at -Os); builds with a large kernel config buffer keep their qualified flags.

Host tests check the rule and the chooser's accounting; device tests run on a default-worker-L1 1x2 mesh:
the exp ring recipe suite's bit-exact gate (each chip equals the dense recipe in visiting order) at the
qualified Q256/Q128/Q224 K512 geometries, and op-selected blocking on the DiT shapes.
"""

import math
import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from . import test_sdpa_recipe_exp_ring as exp_ring
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, prepare

T = ttnn._ttnn.operations.transformer
KV_DTYPE = {"E_bf16": ttnn.bfloat16, "E_bfp8": ttnn.bfloat8_b, "E_bfp4": ttnn.bfloat4_b}
DEFAULT_CONFIG_BUFFER = 70656  # Blackhole, default worker L1
PIPELINE_CONFIG_BUFFER = 187 * 1024  # approximately, worker_l1_size=1344544
PIPELINE_L1 = 1_344_512
BF16_DEST = ("B", "E_bf16", "E_bfp8", "E_bfp4")


def precision_of(variant):
    return getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION"))


def size_optimized(variant, q, k, d, config_bytes):
    return T._sdpa_recipe_exp_ring_size_optimized(
        precision_of(variant), KV_DTYPE.get(variant, ttnn.bfloat16), q, k, d, config_bytes
    )


# ------------------------------------------------------------------------------------------ host


@pytest.mark.parametrize("variant", VARIANTS)
def test_size_flags_rule(variant):
    qualified = [(q, k, d) for q in (128, 224, 256, 320) for k in (256, 384, 512) for d in (64, 128, 256)]
    for q, k, d in qualified:
        expected = variant in BF16_DEST
        assert size_optimized(variant, q, k, d, DEFAULT_CONFIG_BUFFER) == expected, (q, k, d)
        # Large buffers (pipeline L1) and an unknown buffer keep the qualified flags.
        assert not size_optimized(variant, q, k, d, PIPELINE_CONFIG_BUFFER)
        assert not size_optimized(variant, q, k, d, 0)
    # Outside the qualified geometries the recipe program already builds the generic-geometry flags.
    for q, k, d in [(384, 512, 128), (256, 128, 128), (96, 160, 96), (256, 512, 96)]:
        assert not size_optimized(variant, q, k, d, DEFAULT_CONFIG_BUFFER)


def choose(variant, heads, local, config_bytes):
    return T._sdpa_recipe_blocking_candidates(
        "exp_ring",
        precision_of(variant),
        KV_DTYPE.get(variant, ttnn.bfloat16),
        1,
        heads,
        local,
        local,
        128,
        ttnn.CoreCoord(11, 10),
        PIPELINE_L1,
        ring_size=2,
        kernel_config_bytes=config_bytes,
    )


@pytest.mark.parametrize("variant", BF16_DEST)
@pytest.mark.parametrize("heads, local", [(10, 2368), (14, 1216)], ids=["wan720_4x32", "h3_4x32"])
def test_chooser_costs_size_flags(variant, heads, local):
    # A small buffer costs the qualified geometries like the generic ones (size-optimized builds); a large
    # or unknown buffer leaves the cost model unchanged.
    small = {(c[0], c[1], c[2]): c[4] for c in choose(variant, heads, local, DEFAULT_CONFIG_BUFFER)}
    large = {(c[0], c[1], c[2]): c[4] for c in choose(variant, heads, local, PIPELINE_CONFIG_BUFFER)}
    unknown = {(c[0], c[1], c[2]): c[4] for c in choose(variant, heads, local, 0)}
    assert small.keys() == large.keys() == unknown.keys() and large == unknown
    for key, cost in small.items():
        if size_optimized(variant, key[0], key[1], 128, DEFAULT_CONFIG_BUFFER):
            assert cost > large[key]
        else:
            assert cost == large[key]


# ---------------------------------------------------------------------------------------- device


@pytest.fixture(scope="module")
def exp_ring_mesh():
    """The exp ring recipe suite's 1x2 FABRIC_1D_RING mesh, at the default worker L1."""
    if not is_blackhole() or ttnn.GetNumAvailableDevices() != 2:
        pytest.skip("requires two connected Blackholes")
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router,
    )
    mesh = manager = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), trace_region_size=16777216)
        mesh.enable_program_cache()
        hardware = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hardware.x - 1, hardware.y - 1))})
        subdevice = ttnn.SubDeviceId(0)
        manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([subdevice])
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        yield mesh, subdevice, semaphores
    finally:
        if mesh is not None:
            if manager is not None:
                mesh.reset_sub_device_stall_group()
                mesh.clear_loaded_sub_device_manager()
                mesh.remove_sub_device_manager(manager)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_default_kernel_config_buffer(exp_ring_mesh):
    mesh, *_ = exp_ring_mesh
    assert T._sdpa_recipe_kernel_config_bytes(mesh) == DEFAULT_CONFIG_BUFFER


# The exp ring suite's gate (bit-exact against the dense recipe in visiting order where chunk-aligned) on the
# qualified geometries whose builds did not fit, one and several passes, joint KV and padded tails.
QUALIFIED_CASES = [
    ("aligned", 256),
    ("joint", 256),
    ("padded-tails", 256),
    ("two-pass", 256),
    ("three-pass-joint-skip", 256),
    ("aligned", 128),
    ("two-pass", 128),
    ("aligned", 224),
    ("two-pass", 224),
]


@pytest.mark.parametrize("variant", VARIANTS[1:])
@pytest.mark.parametrize("case, q_chunk", QUALIFIED_CASES, ids=[f"{c}-q{q}" for c, q in QUALIFIED_CASES])
def test_qualified_geometry_default_l1(exp_ring_mesh, case, q_chunk, variant, record_property):
    exp_ring.test_recipe_exp_ring(exp_ring_mesh, case, variant, q_chunk, record_property)


@pytest.mark.parametrize("variant", VARIANTS[1:])
@pytest.mark.parametrize("heads, local", [(10, 2368), (14, 1216)], ids=["wan720_4x32", "h3_4x32"])
def test_dit_auto_blocking_default_l1(exp_ring_mesh, heads, local, variant, record_property):
    """The DiT parity shapes with op-selected blocking on the full grid: they build, and match the explicit call."""
    mesh, subdevice, semaphores = exp_ring_mesh
    full = mesh.compute_with_storage_grid_size()
    host = [torch.randn((1, heads, 2 * local, 128), generator=torch.Generator().manual_seed(7 + i)).bfloat16() for i in range(3)]
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    inputs = [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard) for x in host]
    if variant.startswith("E_"):
        inputs = prepare(inputs, variant)
    backing = [
        ttnn.allocate_tensor_on_device([1, heads, 2 * local, 128], x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    options = dict(precision=precision_of(variant), inputs_prepared=variant.startswith("E_"))
    auto_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=full)
    resolved = T._sdpa_recipe_resolved_program_config(
        "exp_ring", options["precision"], inputs[0], inputs[1], program_config=auto_config, ring_size=2
    )
    record_property("chosen", f"Q{resolved.q_chunk_size}/K{resolved.k_chunk_size}/x{resolved.compute_with_storage_grid_size.x}")

    def run(config):
        out, _joint, _lse = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
            *inputs,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=2 * local,
            program_config=config,
            dim=2,
            multi_device_global_semaphore=semaphores,
            num_links=2,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=subdevice,
            num_workers_per_link=full.y // 2,
            num_buffers_per_channel=16,
            **options,
        )
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    auto = run(auto_config)
    assert torch.isfinite(auto.float()).all()
    assert torch.equal(auto, run(resolved))
    # Sanity: a head/row sample against FP32 attention (catches a broken build, not a precision gate).
    rows = torch.linspace(0, local - 1, 64).long()
    q = host[0][:, :1, rows].float()
    ref = torch.softmax(q @ host[1][:, :1].float().transpose(-1, -2) / math.sqrt(128), dim=-1) @ host[2][:, :1].float()
    got = auto[:1, :1, rows].float()
    rel = ((got - ref).norm() / ref.norm()).item()
    record_property("sample_rel_l2", rel)
    assert rel < 0.06
