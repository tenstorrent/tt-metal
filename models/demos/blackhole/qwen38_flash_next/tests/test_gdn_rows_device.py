# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GDN row SFPU calls versus their composed TTNN operations, including trace/address reuse."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_post_rows as post
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_pre_rows as pre

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")


def _host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], value) for value in replicas[1:])
    return replicas[0]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
@pytest.mark.parametrize("rows", [32, 128])
def test_gdn_rows_sfpu_and_reuse(mesh_device, rows):
    cached = None
    for seed in (711, 712):
        rng = torch.Generator().manual_seed(seed)
        owned = []

        def upload(value, dtype=ttnn.bfloat16):
            tensor = ttnn.from_torch(
                value,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            owned.append(tensor)
            return tensor

        def random(shape, scale=0.2):
            return torch.randn(shape, generator=rng) * scale

        try:
            projected = upload(random((1, 1, rows, pre.PROJECTION_WIDTH)))
            history = upload(random((1, 1, 32, pre.QKV_WIDTH)))
            taps = tuple(upload(random((1, 1, 1, pre.QKV_WIDTH))) for _ in range(4))
            dt = upload(random((1, 1, 1, pre.HEADS)), ttnn.float32)
            neg_a = upload(-torch.exp(random((1, 1, 1, pre.HEADS))), ttnn.float32)
            constants = pre.upload_constants(mesh_device, dt, neg_a)
            outputs = pre.allocate_outputs(mesh_device, rows)
            owned.extend((*constants, *outputs))
            reference = pre.chain_on_device(mesh_device, projected, history, taps, dt, neg_a)
            owned.extend(reference)
            o = upload(random((post.HEADS, rows, post.HEAD_DIM)), ttnn.float32)
            z = upload(random((1, 1, rows, post.VALUE_WIDTH)))
            norm = upload(torch.ones((1, 1, 1, post.HEAD_DIM)) + random((1, 1, 1, post.HEAD_DIM)))
            post_reference = post.chain_on_device(mesh_device, o, z, norm, projected)
            scalars = post.scalar_tensor(mesh_device)
            post_outputs = post.allocate_outputs(mesh_device, rows // 32)
            owned.extend((*post_reference, scalars, *post_outputs.values()))

            def execute():
                pre.run(projected, history, taps, *constants, *outputs, rows=rows)
                post.post_cast(o, post_outputs["o16"])
                post.post_norm(
                    post_outputs["o16"],
                    post_reference[2],
                    norm,
                    scalars,
                    projected,
                    post_outputs["gated"],
                    post_outputs["history_next"],
                    rows=rows,
                )

            def compare():
                for label, actual, expected in zip(("q", "k", "v", "beta", "g", "sig"), outputs, reference):
                    assert torch.equal(_host(actual), _host(expected)), f"{label}, rows={rows}, seed={seed}"
                for label, expected in zip(("gated", "history_next"), post_reference[:2]):
                    assert torch.equal(
                        _host(post_outputs[label]), _host(expected)
                    ), f"{label}, rows={rows}, seed={seed}"

            execute()
            compare()
            count = mesh_device.num_program_cache_entries()
            if cached is not None:
                assert count == cached
            cached = count
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                execute()
            finally:
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            try:
                for _ in range(3):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    compare()
                assert mesh_device.num_program_cache_entries() == cached
            finally:
                ttnn.release_trace(mesh_device, trace)
        finally:
            for tensor in owned:
                ttnn.deallocate(tensor)
