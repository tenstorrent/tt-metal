# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed-KV CP=8 coverage for ttnn.experimental.gemma4.rjsdpa; no model weights."""

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    close_ring_joint_sdpa_runtime,
    nd_sharded_dram_memory_config,
    open_ring_joint_sdpa_runtime,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

CP = 8


@pytest.fixture(scope="module")
def runtime():
    if MESH_CONFIG.sp_size != CP:
        pytest.skip(f"Requires CP=8, detected CP={MESH_CONFIG.sp_size}")
    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG, topology=ttnn.Topology.Linear, trace_region_size=4 * 1024 * 1024)
    rt.mesh_device.enable_program_cache()
    try:
        yield rt
    finally:
        close_ring_joint_sdpa_runtime(rt)


class PackedCase:
    def __init__(self, runtime, *, tokens=512, stripe=32, capacity=1536, head_dim=64, packed_dim=128, nd_sharded=False):
        self.rt = runtime
        self.mesh = runtime.mesh_device
        self.tokens, self.stripe, self.capacity = tokens, stripe, capacity
        self.head_dim, self.packed_dim = head_dim, packed_dim
        self.tp = MESH_CONFIG.tp_size
        self.allocated = []
        self.trace_id = None
        # Two users with two layers each. Every slot differs, detecting stale trace metadata.
        gen = torch.Generator().manual_seed(731)
        self.q = torch.randn(1, 2 * self.tp, capacity, head_dim, generator=gen)
        self.kv = torch.randn(4, self.tp, capacity, packed_dim, generator=gen)
        self.cache_positions = torch.cat(self.positions(0, capacity))
        memory_config = nd_sharded_dram_memory_config(self.mesh, packed_dim) if nd_sharded else None
        self.tt_kv = self.upload(self.kv[:, :, self.cache_positions], ttnn.bfloat8_b, memory_config=memory_config)
        self.buffer = self.upload(torch.zeros_like(self.kv), ttnn.bfloat8_b, gathered=True)
        self.tt_q = self.upload(torch.zeros(1, 2 * self.tp, tokens, head_dim), ttnn.bfloat16)
        self.slot = self.scalar(0)
        self.prefix = self.scalar(0)
        self.composer = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(1, 2))
        # Compare against the actual quantized inputs, independently restoring natural token order.
        quantized = ttnn.to_torch(self.tt_kv, mesh_composer=self.composer).float()
        self.kv = quantized[:, :, torch.argsort(self.cache_positions)]

    def positions(self, start, end):
        positions = torch.arange(start, end)
        owners = positions.div(self.stripe, rounding_mode="floor") % CP
        return [positions[owners == rank] for rank in range(CP)]

    def upload(self, value, dtype, *, gathered=False, memory_config=None):
        kwargs = {} if memory_config is None else {"memory_config": memory_config}
        tensor = ttnn.from_torch(
            value.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(1, None if gathered else 2)
            ),
            **kwargs,
        )
        self.allocated.append(tensor)
        return tensor

    def scalar(self, value):
        tensor = ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self.allocated.append(tensor)
        return tensor

    def stage(self, start, *, valid_tokens=None, slot=0):
        self.start, self.user = start, slot
        self.end = min(start + (self.tokens if valid_tokens is None else valid_tokens), self.capacity)
        rank_positions = self.positions(start, self.end)
        local_q = self.tokens // CP
        # Pack by rank, padding only each rank's trailing invalid rows.
        packed_q = torch.zeros_like(self.q[:, :, : self.tokens])
        self.output_indices = []
        self.query_positions = []
        for rank, positions in enumerate(rank_positions):
            begin = rank * local_q
            packed_q[:, :, begin : begin + len(positions)] = self.q[:, :, positions]
            self.output_indices.append(torch.arange(begin, begin + len(positions)))
            self.query_positions.append(positions)
        self.output_indices = torch.cat(self.output_indices)
        self.query_positions = torch.cat(self.query_positions)
        host_q = ttnn.from_torch(
            packed_q,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(1, 2)),
        )
        ttnn.copy_host_to_device_tensor(host_q, self.tt_q)
        for device, value in ((self.slot, slot), (self.prefix, start)):
            host = ttnn.from_torch(
                torch.tensor([value], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            ttnn.copy_host_to_device_tensor(host, device)

    def call(self, *, metadata=False, legacy=False, explicit_stripe=True, q_chunk=32, k_chunk=64, **overrides):
        rt = self.rt
        kwargs = dict(
            persistent_output_buffer_kv=self.buffer,
            head_dim_v=self.head_dim,
            logical_n=self.capacity if metadata else self.end,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=rt.sdpa_compute_grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            ),
            compute_kernel_config=rt.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=rt.ccl_semaphore_handles,
            num_links=rt.num_links,
            cluster_axis=rt.sp_axis,
            mesh_device=self.mesh,
            topology=rt.topology,
            subdevice_id=rt.worker_sub_device_id,
            ccl_core_grid_offset=(rt.ccl_column, 0),
            use_column_major_ccl=True,
            kv_cache_num_layers=2,
            kv_cache_layer_idx=1,
            **(
                {"slot_id": self.slot, "kv_actual_isl_tensor": self.prefix}
                if metadata
                else {"kv_cache_batch_idx": self.user, "kv_actual_isl": self.start}
            ),
        )
        if not legacy and explicit_stripe:
            kwargs["stripe_size"] = self.stripe
        kwargs.update(overrides)
        op = ttnn.transformer.ring_mla if legacy else ttnn.experimental.gemma4.rjsdpa
        output, stats = op(self.tt_q, self.tt_kv, **kwargs)
        self.allocated.extend((output, stats))
        return output, stats

    def check(self, output):
        got = ttnn.to_torch(output, mesh_composer=self.composer).float()[:, :, self.output_indices]
        q = self.q[:, :, self.query_positions].bfloat16().float()
        kv = self.kv[self.user * 2 + 1 : self.user * 2 + 2, :, : self.end].repeat_interleave(2, dim=1)
        k = kv[..., : self.head_dim]
        v = kv[..., -self.head_dim :]
        mask = torch.arange(self.end)[None, :] <= self.query_positions[:, None]
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        passed, pcc = comp_pcc(expected, got, pcc=0.99)
        rmse = (expected - got).square().mean().sqrt().item()
        logger.info("stripe={} tokens={} prefix={} PCC={} RMSE={}", self.stripe, self.tokens, self.start, pcc, rmse)
        assert passed, pcc
        assert rmse < 0.025
        return got

    def close(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
        for tensor in reversed(self.allocated):
            ttnn.deallocate(tensor)


@pytest.fixture
def case_factory(runtime):
    cases = []

    def make(**kwargs):
        case = PackedCase(runtime, **kwargs)
        cases.append(case)
        return case

    yield make
    for case in reversed(cases):
        case.close()


@pytest.mark.parametrize("metadata", [False, True], ids=["host", "device_metadata"])
@pytest.mark.parametrize("q_chunk,k_chunk", [(32, 32), (64, 64), (96, 128)])
def test_disjoint_stripes(case_factory, metadata, q_chunk, k_chunk):
    case = case_factory()
    for prefix in (0, 512, 32):
        case.stage(prefix, slot=prefix % 3 % 2)
        output, _ = case.call(metadata=metadata, q_chunk=q_chunk, k_chunk=k_chunk)
        case.check(output)


@pytest.mark.parametrize("metadata", [False, True])
def test_legacy_equivalence(case_factory, metadata):
    case = case_factory(stripe=64)
    for prefix in (0, 32, 512):
        case.stage(prefix, slot=1)
        legacy, legacy_stats = case.call(metadata=metadata, legacy=True)
        same, same_stats = case.call(metadata=metadata, explicit_stripe=False)
        striped, _ = case.call(metadata=metadata)
        case.check(legacy)
        case.check(striped)
        torch.testing.assert_close(
            ttnn.to_torch(legacy, mesh_composer=case.composer),
            ttnn.to_torch(same, mesh_composer=case.composer),
            rtol=0,
            atol=0,
        )
        # Statistics is scratch: the single-Q-per-core schedule can leave it unwritten.
        assert same_stats.shape == legacy_stats.shape
        assert same_stats.dtype == legacy_stats.dtype


@pytest.mark.parametrize(
    "tokens,stripe,capacity,head_dim,packed_dim,nd_sharded",
    [
        (512, 32, 512, 64, 128, False),  # Q and KV have the same physical length.
        (768, 32, 2304, 64, 128, False),  # Three stripes, including non-power-of-two Q extent.
        (1024, 64, 3072, 64, 128, False),
        (512, 32, 1536, 512, 640, True),  # Gemma4 packed K/V overlap, in ND-sharded DRAM.
        (8192, 32, 16384, 64, 128, False),  # 32 ownership stripes in one call.
    ],
)
def test_stripe_geometries(case_factory, tokens, stripe, capacity, head_dim, packed_dim, nd_sharded):
    case = case_factory(
        tokens=tokens, stripe=stripe, capacity=capacity, head_dim=head_dim, packed_dim=packed_dim, nd_sharded=nd_sharded
    )
    case.stage(0)
    output, _ = case.call(q_chunk=64, k_chunk=128, kv_actual_isl=None)  # Infer the prefix from Q and logical_n.
    case.check(output)


def test_partial_queries(case_factory):
    case = case_factory(stripe=64, capacity=1536)
    case.stage(32, valid_tokens=288)
    case.check(case.call(q_chunk=64)[0])
    # The final device-metadata query is clipped by the physical cache boundary.
    case.stage(1344)
    case.check(case.call(metadata=True, q_chunk=64)[0])
    # KV capacity need only be a multiple of the stripe, not of the query size.
    # The gather's conservative Q-sized prefix must be clipped to this odd tail.
    tail = case_factory(capacity=768)
    tail.stage(512)
    tail.check(tail.call(metadata=True, q_chunk=64)[0])


def test_trace_replay(case_factory):
    case = case_factory(nd_sharded=True)
    case.stage(0)
    case.check(case.call(metadata=True, q_chunk=64)[0])
    entries = case.mesh.num_program_cache_entries()
    case.trace_id = ttnn.begin_trace_capture(case.mesh, cq_id=0)
    output, _ = case.call(metadata=True, q_chunk=64)
    ttnn.end_trace_capture(case.mesh, case.trace_id, cq_id=0)
    for prefix, slot in ((512, 1), (32, 0), (1024, 1), (0, 0), (1280, 1)):
        case.stage(prefix, slot=slot)
        ttnn.execute_trace(case.mesh, case.trace_id, cq_id=0, blocking=True)
        case.check(output)
    assert case.mesh.num_program_cache_entries() == entries


@pytest.mark.parametrize("max_k_splits,segmented", [(2, False), (5, False), (1, True)])
def test_compute_options(case_factory, max_k_splits, segmented):
    case = case_factory(capacity=16384)
    case.stage(8192, slot=1)
    output, _ = case.call(
        metadata=True,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=case.rt.sdpa_compute_grid,
            q_chunk_size=64,
            k_chunk_size=64,
            exp_approx_mode=False,
            max_k_splits=max_k_splits,
            segmented_accumulation=segmented,
            matmul_math_fidelity=ttnn.MathFidelity.LoFi,
        ),
    )
    case.check(output)


@pytest.mark.parametrize("stripe", [0, 16, 96])
def test_invalid_stripes(case_factory, expect_error, stripe):
    case = case_factory()
    case.stage(0)
    with expect_error(RuntimeError, "stripe_size must"):
        case.call(stripe_size=stripe)
