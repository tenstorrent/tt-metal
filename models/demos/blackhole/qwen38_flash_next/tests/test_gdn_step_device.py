# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused GDN decode step on the line's head shards versus tt/gdn.py's arithmetic, with in-place state and replay."""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.helpers.tp_harness import (
    DEVICE_PARAMS,
    assert_program_cache_stable,
    assert_shards_equal,
    assert_tensors_equal,
    download_shards,
    pcc,
    upload,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import dram_sharded_matmul_configs
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as gs
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")

DIES = MESH_SHAPE[1]
HIDDEN = 2560
# COMPONENT: the program keeps tt/gdn.py's rounding points except the q scale (applied to the read-out in fp32), so it
# is held to the oracle's arithmetic, at about twice a measured 64-step component bound.
STATE_RELATIVE, GATED_RELATIVE, GATED_PCC = 0.01, 0.02, 0.999


def _relative(actual, expected):
    actual, expected = actual.float(), expected.float()
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
@pytest.mark.parametrize("rows", [1, 4])
def test_gdn_step_against_oracle_arithmetic(mesh_device, rows, record_property):
    entry = fused.kernel(gs.NAME)
    assert entry.tolerance == fused.COMPONENT, f"{entry.name}: tolerance {entry.tolerance}"
    out_config, _ = dram_sharded_matmul_configs(mesh_device, gs.VALUE_WIDTH, HIDDEN, num_cores=16)
    state_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 1))
    cached = None
    for seed in (811, 812):
        rng = torch.Generator().manual_seed(seed)
        owned = []

        def upload_owned(host, dtype, *, shard_dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG):
            # the per-die values on the last axis, die-major: the head shards of the model's GDN tensors
            owned.append(upload(mesh_device, host, dtype, memory_config=memory_config, shard_dim=shard_dim))
            return owned[-1]

        def bf16(*shape, scale=0.2):
            return (torch.randn(shape, generator=rng) * scale).to(torch.bfloat16)

        try:
            projected_host = bf16(rows, DIES, gs.PROJECTION_WIDTH)
            older_host = [bf16(rows, DIES, gs.QKV_WIDTH) for _ in range(3)]
            taps_host = [bf16(DIES, gs.QKV_WIDTH) for _ in range(4)]
            dt_host = torch.randn(DIES, gs.HEADS, generator=rng) * 0.2
            neg_a_host = -torch.exp(torch.randn(DIES, gs.HEADS, generator=rng) * 0.2)
            norm_host = (1.0 + torch.randn(gs.HEAD_DIM, generator=rng) * 0.1).to(torch.bfloat16)
            state_host = torch.randn(rows, DIES * gs.HEADS, gs.HEAD_DIM, gs.HEAD_DIM, generator=rng) * 0.05

            projected = upload_owned(
                projected_host.reshape(1, 1, rows, -1), ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            older = [upload_owned(slot.reshape(1, 1, rows, -1), ttnn.bfloat16) for slot in older_host]
            newest = upload_owned(bf16(1, 1, rows, DIES * gs.QKV_WIDTH), ttnn.bfloat16)  # overwritten by the step
            taps = [upload_owned(tap.reshape(1, 1, 1, -1), ttnn.bfloat16) for tap in taps_host]
            dt_bias = upload_owned(dt_host.reshape(1, 1, 1, -1), ttnn.float32)
            neg_exp_a = upload_owned(neg_a_host.reshape(1, 1, 1, -1), ttnn.float32)
            constants = gs.constant_tiles(mesh_device, dt_bias, neg_exp_a)
            owned.append(constants)
            norm = upload_owned(norm_host.reshape(1, 1, 1, -1), ttnn.bfloat16, shard_dim=None)
            recurrent = upload_owned(state_host, ttnn.float32, shard_dim=1)
            out = fp.allocate((1, 1, rows, gs.VALUE_WIDTH), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, out_config)
            owned.append(out)
            assert entry.admits(None, projected, (*older, newest), SimpleNamespace(recurrent=recurrent)), (
                f"{entry.name} does not admit rows={rows}: projected {tuple(projected.shape)} {projected.layout}, "
                f"{len(older) + 1} window slots, state {tuple(recurrent.shape)} {recurrent.dtype}"
            )

            def step():
                gs.run(projected, older, newest, taps, constants, norm, recurrent, out)

            step()
            results = [download_shards(tensor) for tensor in (newest, recurrent, out)]
            for die in range(DIES):
                heads = slice(die * gs.HEADS, (die + 1) * gs.HEADS)
                new_state, gated, *_ = gs.reference_step(
                    projected_host[:, die],
                    [slot[:, die] for slot in older_host],
                    [tap[die] for tap in taps_host],
                    dt_host[die],
                    neg_a_host[die],
                    norm_host,
                    state_host[:, heads],
                )
                landed, state, output = (result[die] for result in results)
                assert_tensors_equal(
                    landed.reshape(rows, -1),
                    projected_host[:, die, : gs.QKV_WIDTH],
                    f"ring slot, die {die}, seed {seed}",
                )
                state_error = _relative(state, new_state)
                gated_error = _relative(output.reshape(rows, -1), gated)
                gated_pcc = pcc(output.reshape(rows, -1), gated)
                record_property(
                    f"seed{seed}_die{die}",
                    f"state_rel={state_error:.6f}; gated_rel={gated_error:.6f}; PCC={gated_pcc:.6f}",
                )
                assert (
                    state_error <= STATE_RELATIVE
                ), f"state, die {die}, seed {seed}: relative error {state_error:.6f} > {STATE_RELATIVE}"
                assert gated_error <= GATED_RELATIVE and gated_pcc >= GATED_PCC, (
                    f"gated output, die {die}, seed {seed}: relative error {gated_error:.6f} (limit {GATED_RELATIVE}), "
                    f"PCC {gated_pcc:.6f} (floor {GATED_PCC})"
                )
            cached = assert_program_cache_stable(mesh_device, cached)
            # The step updates the state in place: every replay starts from the same state image, so it must
            # reproduce the eager step bit for bit at the captured addresses.
            image = ttnn.from_torch(state_host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=state_mapper)
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                step()
            finally:
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            try:
                for replay in range(3):
                    ttnn.copy_host_to_device_tensor(image, recurrent)
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    for name, tensor, eager in zip(
                        ("ring slot", "state", "gated output"), (newest, recurrent, out), results, strict=True
                    ):
                        assert_shards_equal(tensor, eager, f"replay {replay}, {name}, seed {seed}")
                count = mesh_device.num_program_cache_entries()
                assert count == cached, f"program cache {count} entries after the replays, {cached} before the capture"
            finally:
                ttnn.release_trace(mesh_device, trace)
        finally:
            for tensor in owned:
                ttnn.deallocate(tensor)
