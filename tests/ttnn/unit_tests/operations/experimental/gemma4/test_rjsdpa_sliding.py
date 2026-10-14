# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CP=8 striped sliding attention, compared with an absolute-position PyTorch oracle."""

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.unit_tests.operations.experimental.gemma4.test_rjsdpa import (
    CP,
    MESH_CONFIG,
    PackedCase,
    nd_sharded_dram_memory_config,
    runtime as runtime,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


class SlidingCase(PackedCase):
    def __init__(
        self,
        runtime,
        *,
        tokens=256,
        stripe=32,
        capacity=8192,
        head_dim=64,
        kv_heads=1,
        window=1024,
        nd_sharded=True,
        cache_slots=4,
    ):
        self.rt, self.mesh = runtime, runtime.mesh_device
        self.tokens, self.stripe, self.capacity = tokens, stripe, capacity
        self.head_dim, self.tp, self.window = head_dim, MESH_CONFIG.tp_size, window
        self.cache_slots = cache_slots
        self.layers, self.layer = (2, 1) if cache_slots == 4 else (1, 0)
        self.allocated, self.trace_id = [], None
        gen = torch.Generator().manual_seed(931)
        self.q = torch.randn(1, 2 * kv_heads * self.tp, capacity, head_dim, generator=gen)
        # Exactly BFP8-representable values make the oracle independent of device cache reads.
        shape = (cache_slots, kv_heads * self.tp, capacity, head_dim)
        self.k = torch.randint(-7, 8, shape, generator=gen).float() / 8
        self.v = torch.randint(-7, 8, shape, generator=gen).float() / 8
        self.cache_positions = torch.cat(self.positions(0, capacity))
        memory = nd_sharded_dram_memory_config(self.mesh, head_dim) if nd_sharded else None
        self.tt_k = self.upload(self.k[:, :, self.cache_positions], ttnn.bfloat8_b, memory_config=memory)
        self.tt_v = self.upload(self.v[:, :, self.cache_positions], ttnn.bfloat8_b, memory_config=memory)
        # One cache slot is gathered even when input caches contain several users/layers.
        self.buffer_k = self.upload(torch.zeros_like(self.k[:1]), ttnn.bfloat8_b, gathered=True)
        self.buffer_v = self.upload(torch.zeros_like(self.v[:1]), ttnn.bfloat8_b, gathered=True)
        halo = ((window - 1 + 127) // 128) * 128 * 2
        self.halo_k = self.upload(torch.zeros_like(self.k[:1, :, :halo]), ttnn.bfloat8_b, gathered=True)
        self.halo_v = self.upload(torch.zeros_like(self.v[:1, :, :halo]), ttnn.bfloat8_b, gathered=True)
        self.tt_q = self.upload(torch.zeros_like(self.q[:, :, :tokens]), ttnn.bfloat16)
        self.slot, self.prefix = self.scalar(0), self.scalar(0)
        self.composer = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(1, 2))

    def call(
        self,
        *,
        metadata=False,
        legacy=False,
        q_chunk=32,
        k_chunk=128,
        max_k_splits=1,
        scale=None,
        fidelity=None,
        **overrides,
    ):
        rt = self.rt
        kwargs = dict(
            logical_n=self.capacity if metadata else self.end,
            sliding_window_size=self.window,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=rt.sdpa_compute_grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                max_k_splits=max_k_splits,
                exp_approx_mode=False,
                matmul_math_fidelity=fidelity,
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
            kv_cache_num_layers=self.layers,
            kv_cache_layer_idx=self.layer,
            scale=scale,
            **(
                {"slot_id": self.slot, "kv_actual_isl_tensor": self.prefix}
                if metadata
                else {"kv_cache_batch_idx": self.user, "kv_actual_isl": self.start}
            ),
        )
        kwargs.update(overrides)
        if legacy:
            results = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                self.tt_q,
                self.tt_k,
                self.tt_v,
                None,
                None,
                None,
                persistent_output_buffer_k=self.halo_k,
                persistent_output_buffer_v=self.halo_v,
                joint_strategy="rear",
                is_causal=True,
                is_balanced=False,
                **kwargs,
            )
        else:
            results = ttnn.experimental.gemma4.rjsdpa(
                self.tt_q,
                self.tt_k,
                input_tensor_v=self.tt_v,
                persistent_output_buffer_kv=self.buffer_k,
                persistent_output_buffer_v=self.buffer_v,
                head_dim_v=self.head_dim,
                stripe_size=self.stripe,
                **kwargs,
            )
        self.allocated.extend(t for t in results if t is not None)
        return results[:2]

    def check(self, output, *, scale=None, sample_count=None):
        indices = torch.arange(len(self.query_positions))
        if sample_count is not None and len(indices) > sample_count:
            indices = torch.linspace(0, len(indices) - 1, sample_count).long().unique()
        pos = self.query_positions[indices]
        got = ttnn.to_torch(output, mesh_composer=self.composer).float()[:, :, self.output_indices[indices]]
        q = self.q[:, :, pos].bfloat16().float()
        slot = self.user * self.layers + self.layer
        expected = torch.empty_like(q)
        for start in range(0, len(pos), 32):
            qp = pos[start : start + 32]
            low, high = max(0, int(qp.min()) - self.window + 1), int(qp.max()) + 1
            kp = torch.arange(low, high)
            mask = (kp[None, :] <= qp[:, None]) & (kp[None, :] > qp[:, None] - self.window)
            expected[:, :, start : start + len(qp)] = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, start : start + len(qp)],
                self.k[slot : slot + 1, :, low:high].repeat_interleave(2, dim=1),
                self.v[slot : slot + 1, :, low:high].repeat_interleave(2, dim=1),
                attn_mask=mask,
                scale=scale,
            )
        passed, pcc = comp_pcc(expected, got, pcc=0.99)
        rmse = (expected - got).square().mean().sqrt().item()
        assert torch.isfinite(got).all()
        logger.info(
            "sliding C={} stripe={} W={} prefix={} PCC={} RMSE={}",
            self.tokens,
            self.stripe,
            self.window,
            self.start,
            pcc,
            rmse,
        )
        assert passed, pcc
        assert rmse < 0.025
        return {"pcc": float(str(pcc).rsplit("PCC:", 1)[-1]), "rmse": rmse, "queries_checked": len(pos)}


@pytest.mark.parametrize("tokens", [256, 512, 768, 1024, 1280, 3072, 4096, 6144, 8192])
def test_sliding_striped_chunks(runtime, tokens):
    case = SlidingCase(runtime, tokens=tokens, capacity=16384)
    try:
        case.stage(4096 + 32)
        case.check(case.call()[0])
    finally:
        case.close()


@pytest.mark.parametrize(
    "window,stripe,q_chunk,k_chunk",
    [
        (33, 32, 32, 32),
        (257, 32, 32, 64),
        (1023, 32, 64, 128),
        (1024, 64, 32, 256),
        (2048, 32, 128, 128),
    ],
)
def test_sliding_window_edges(runtime, window, stripe, q_chunk, k_chunk):
    case = SlidingCase(runtime, tokens=1024, window=window, stripe=stripe)
    try:
        for prefix in (0, 32, 6144):
            case.stage(prefix)
            case.check(case.call(q_chunk=q_chunk, k_chunk=k_chunk)[0])
    finally:
        case.close()


def test_sliding_partial_and_trace(runtime):
    case = SlidingCase(runtime, tokens=768, capacity=8192)
    try:
        case.stage(32, valid_tokens=256)
        case.check(case.call()[0])
        case.stage(0)
        case.call(metadata=True)
        case.trace_id = ttnn.begin_trace_capture(case.mesh, cq_id=0)
        output = case.call(metadata=True)[0]
        ttnn.end_trace_capture(case.mesh, case.trace_id, cq_id=0)
        entries = case.mesh.num_program_cache_entries()
        for prefix, slot in ((0, 0), (4128, 1), (8000, 0), (256, 1)):
            case.stage(prefix, slot=slot)
            ttnn.execute_trace(case.mesh, case.trace_id, cq_id=0, blocking=True)
            case.check(output)
            assert case.mesh.num_program_cache_entries() == entries
    finally:
        case.close()


@pytest.mark.parametrize("tokens", [1024, 4096])
def test_sliding_matches_legacy_geometry(runtime, tokens):
    case = SlidingCase(runtime, tokens=tokens, stripe=tokens // CP, capacity=16384)
    try:
        case.stage(8192)
        case.check(case.call(q_chunk=64)[0])
        case.check(case.call(legacy=True, q_chunk=64)[0])
    finally:
        case.close()


@pytest.mark.parametrize("splits", [1, 3])
def test_sliding_gemma_shape(runtime, splits):
    case = SlidingCase(runtime, tokens=256, head_dim=256, kv_heads=4)
    try:
        case.stage(4096)
        case.check(case.call(max_k_splits=splits, fidelity=ttnn.MathFidelity.LoFi, scale=1.0)[0], scale=1.0)
    finally:
        case.close()


@pytest.mark.parametrize("q_chunk", [128, 256])
def test_sliding_split_queries_across_stripes(runtime, q_chunk):
    # Early rows have no visible keys in some split bands at the start of a request.
    # Check every row, including those whose diagonal is on a different worker.
    case = SlidingCase(runtime, tokens=CP * q_chunk, head_dim=256, kv_heads=4)
    try:
        for prefix in (0, 32, 4096):
            case.stage(prefix)
            output = case.call(q_chunk=q_chunk, k_chunk=32, max_k_splits=3, fidelity=ttnn.MathFidelity.LoFi, scale=1.0)[
                0
            ]
            case.check(output, scale=1.0)
    finally:
        case.close()


@pytest.mark.parametrize("tokens", [256, 512, 768, 1280])
def test_legacy_rejects_small_or_nonmultiple_chunks(runtime, tokens, expect_error):
    case = SlidingCase(runtime, tokens=tokens)
    try:
        case.stage(0)
        with expect_error(RuntimeError, "chunk_size must divide"):
            case.call(legacy=True, q_chunk=64)
    finally:
        case.close()
