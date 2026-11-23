# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Distributed prefill RMSNorm: all mesh shards and repeated shared-resource reuse."""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.ci_failure import assert_perf_gate
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    assert_requested_tp_wrap_was_realized,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler


@pytest.mark.skipif(not is_blackhole(), reason="Kimi fused prefill targets Blackhole")
@pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4")
@pytest.mark.parametrize(
    "mesh_device,device_params,topology",
    [
        pytest.param(
            (8, 4),
            dict(torus_xy_device_params(fabric_payload_size=7168, l1_small_size=768), trace_region_size=4194304),
            ttnn.Topology.Ring,
            id="torus-xy",
        ),
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 4194304},
            ttnn.Topology.Linear,
            id="linear-1d",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "seq_len,emb_dim,output_memcfg",
    [
        pytest.param(5120, 7168, ttnn.DRAM_MEMORY_CONFIG, id="kimi"),
        pytest.param(1024, 6144, ttnn.L1_MEMORY_CONFIG, id="glm-width-short-sequence-l1"),
        # GLM-5.3 chunked prefill: 5k chunks (640 rows per device); the attention norm writes L1, the FFN norm DRAM.
        pytest.param(5120, 6144, ttnn.L1_MEMORY_CONFIG, id="glm-chunk-attn-norm-l1"),
        pytest.param(5120, 6144, ttnn.DRAM_MEMORY_CONFIG, id="glm-chunk-ffn-norm"),
    ],
)
def test_kimi_fused_rmsnorm(mesh_device, device_params, topology, seq_len, emb_dim, output_memcfg, expect_error):
    assert_requested_tp_wrap_was_realized(mesh_device)
    clear_tt_ccl_cache()
    torch.manual_seed(50932)
    inputs = [
        torch.randn(1, 1, seq_len, emb_dim).to(torch.bfloat16),
        (torch.randn(1, 1, seq_len, emb_dim) + 20).to(torch.bfloat16),
    ]
    weights = [(torch.randn(emb_dim) * 0.2 + 1).to(torch.bfloat16) for _ in inputs]
    xs = [
        ttnn.from_torch(
            x,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(8, 4), dims=(2, 3)),
        )
        for x in inputs
    ]
    norms = [
        TtDistributedRmsNorm(
            mesh_device,
            emb_dim=emb_dim,
            epsilon=1e-5,
            torch_weight=weight,
            cluster_axis=1,
            num_links=2,
            topology=topology,
            use_fused=True,
            output_memcfg=output_memcfg,
        )
        for weight in weights
    ]
    # Hold outputs until the whole mixed sequence is enqueued. This catches stale
    # stats, missed semaphore resets and cache-hit weight-address mistakes.
    outputs = [norms[i % 2](xs[i % 2]) for i in range(16)]
    ttnn.synchronize_device(mesh_device)
    logger.info("Fused RMSNorm mixed-input reuse completed")
    ccl = norms[0].tt_ccl
    assert norms[1].tt_ccl is ccl
    assert len(ccl.fused_rmsnorm_resources) == 1
    resources = next(iter(ccl.fused_rmsnorm_resources.values()))
    assert len(resources["pairs"]) == 2
    pair_ids = [(id(semaphores), id(stats)) for semaphores, stats in resources["pairs"]]
    # An explicit opt-out and re-enable must reuse the initialized resources.
    for enabled in (False, True):
        for norm, x in zip(norms, xs):
            norm.set_fused_enabled(enabled)
            assert norm.use_fused is enabled
            outputs.append(norm(x))
    assert len(ccl.fused_rmsnorm_resources) == 1
    assert next(iter(ccl.fused_rmsnorm_resources.values())) is resources
    assert [(id(semaphores), id(stats)) for semaphores, stats in resources["pairs"]] == pair_ids

    unfused_norms = [
        TtDistributedRmsNorm(
            mesh_device,
            emb_dim=emb_dim,
            epsilon=1e-5,
            torch_weight=weight,
            cluster_axis=1,
            num_links=2,
            topology=topology,
            use_fused=False,
            output_memcfg=output_memcfg,
        )
        for weight in weights
    ]
    with expect_error(ValueError, "initialized with use_fused=True"):
        unfused_norms[0].set_fused_enabled(True)
    assert not unfused_norms[0].use_fused
    unfused_outputs = [norm(x) for norm, x in zip(unfused_norms, xs)]
    composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(2, 3))
    try:
        for i, output in enumerate(outputs):
            x = inputs[i % 2].float()
            reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weights[i % 2].float()
            assert output.memory_config() == output_memcfg
            actual = ttnn.to_torch(output, mesh_composer=composer).float()
            assert actual.shape == reference.shape
            assert torch.isfinite(actual).all()
            if i in (16, 17):
                # The shifted input stresses BF16 statistics in the existing path.
                # Disabling fusion must match an independently initialized fallback
                # exactly; the fused results retain their stricter CPU accuracy gate.
                baseline = ttnn.to_torch(unfused_outputs[i - 16], mesh_composer=composer).float()
                assert torch.equal(actual, baseline)
                ttnn.deallocate(unfused_outputs[i - 16])
            else:
                relative_error = ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
                assert relative_error < 0.01, (i, relative_error)
            ttnn.deallocate(output)
        # Odd trace lengths exercise reuse across replay boundaries, including
        # the same scratch pair at the end of one replay and start of the next.
        trace_input = ttnn.clone(xs[0])
        for captured_calls in (1, 3):
            ttnn.synchronize_device(mesh_device)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            traced_outputs = [norms[i % 2](trace_input) for i in range(captured_calls)]
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            try:
                for iteration in range(4):
                    ttnn.copy(xs[iteration % 2], trace_input)
                    eager_outputs = [norms[i % 2](trace_input) for i in range(captured_calls)]
                    # Consecutive asynchronous replays must not accumulate stale
                    # semaphore counts or consume statistics from an earlier input.
                    for replay in range(3):
                        ttnn.copy(xs[(iteration + replay) % 2], trace_input)
                        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    for eager, traced in zip(eager_outputs, traced_outputs):
                        expected = ttnn.to_torch(eager, mesh_composer=composer)
                        actual = ttnn.to_torch(traced, mesh_composer=composer)
                        assert torch.equal(actual, expected)
                        ttnn.deallocate(eager)
            finally:
                ttnn.release_trace(mesh_device, trace_id)
                for output in traced_outputs:
                    ttnn.deallocate(output)
        ttnn.deallocate(trace_input)
        assert len(ccl.fused_rmsnorm_resources) == 1
        assert next(iter(ccl.fused_rmsnorm_resources.values())) is resources
        # A requested fused path must raise in the op rather than silently falling
        # back when the input width no longer matches the initialized affine weight.
        bad_x = ttnn.from_torch(
            inputs[0][..., :-128].contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(8, 4), dims=(2, 3)),
        )
        try:
            with expect_error(RuntimeError, "Weight last dim"):
                norms[0](bad_x)
        finally:
            ttnn.deallocate(bad_x)
    finally:
        clear_tt_ccl_cache()


# GLM-5.3's two fused norms on the [RING, RING] 4x4 stage, one 2,560-token chunk (640 rows a SP chip on mesh axis 1, the
# hidden width over the TP ring on axis 0): the attention norm writes L1, the FFN norm DRAM. Device time of one call
# (every program's slowest chip, summed). Ceilings in us: the slowest median over six runs on two 8 kW Galaxies (43-45 and
# 47-49 us) plus 10%; provisional until recalibrated on the 14 kW runners. Override with TT_METAL_RMSNORM_RING_RING_4X4_MAX_US
# (every case).
RMSNORM_PERF_SAMPLES = 7
RMSNORM_RING_RING_4X4_CEILING_US = {"attn-norm-l1": 50, "ffn-norm-dram": 55}


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.requires_mesh_topology(mesh_shape=(4, 4), topology="mesh-4x4")
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (4, 4),
            dict(torus_xy_device_params(fabric_payload_size=6144, l1_small_size=768), trace_region_size=4194304),
            id="torus-xy-4x4-tp0",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "case, output_memcfg",
    [("attn-norm-l1", ttnn.L1_MEMORY_CONFIG), ("ffn-norm-dram", ttnn.DRAM_MEMORY_CONFIG)],
)
def test_fused_rmsnorm_ring_ring_4x4_perf(mesh_device, device_params, case, output_memcfg):
    require_high_power("[RING, RING] 4x4 fused RMSNorm perf")
    require_realtime_profiler("fused RMSNorm perf")
    clear_tt_ccl_cache()
    seq_len, emb_dim = 2560, 6144
    torch.manual_seed(50932)
    x = ttnn.from_torch(
        torch.randn(1, 1, seq_len, emb_dim).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(4, 4), dims=(3, 2)),
    )
    norm = TtDistributedRmsNorm(
        mesh_device,
        emb_dim=emb_dim,
        epsilon=1e-5,
        torch_weight=(torch.randn(emb_dim) * 0.2 + 1).to(torch.bfloat16),
        cluster_axis=0,
        num_links=2,
        topology=ttnn.Topology.Ring,
        use_fused=True,
        output_memcfg=output_memcfg,
    )

    def run():
        ttnn.deallocate(norm(x))

    run()
    run()
    ttnn.synchronize_device(mesh_device)
    samples = []
    for _ in range(RMSNORM_PERF_SAMPLES):
        _, per_program = profile_realtime_program_merged(mesh_device, run)
        samples.append(sum(entry["duration_ns"] for entry in per_program.values()))
    median_us = statistics.median(samples) / 1e3
    print(
        f"RMSNORM_PERF case={case} mesh={tuple(mesh_device.shape)} median={median_us:.1f}us min={min(samples) / 1e3:.1f}us "
        f"max={max(samples) / 1e3:.1f}us samples_us={[round(v / 1e3, 1) for v in samples]}"
    )
    ceiling_us = float(
        os.environ.get("TT_METAL_RMSNORM_RING_RING_4X4_MAX_US", RMSNORM_RING_RING_4X4_CEILING_US[case] or 0)
    )
    assert ceiling_us > 0, f"no ceiling yet; measured median {median_us:.1f} us"
    assert_perf_gate(median_us <= ceiling_us, f"{case} median {median_us:.1f} us over its ceiling {ceiling_us} us")
