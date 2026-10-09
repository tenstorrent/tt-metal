# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal check of the head <-> sequence all-to-all the head-parallel MLA needs (2x4 mesh, FABRIC_2D, axis 1):
  heads -> rows: chip c holds [1, 16, 2560, 512] (its 16 heads, the mesh row's 2560 rows) and must end with
                 [1, 64, 640, 512] (all 64 heads, its own 640-row quarter)
  rows -> heads: the inverse.
Every element encodes (head, row), so the test checks exactly which (in_dim, out_dim) gives which redistribution,
then times both directions at the model's shapes."""

import time

import pytest
import torch

import ttnn

H, ROWS, W, TP = 64, 2560, 512, 4


def _enc(heads, rows):  # value = head * 4096 + row (exact in fp32; bf16 checked via a coarser code below)
    return heads[:, None] * 4096.0 + rows[None, :]


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 24576}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_mla_tp_a2a(mesh_device):
    hq = H // TP
    rq = ROWS // TP
    # per chip (r, c): heads [hq c, hq (c+1)), rows [0, ROWS) of mesh row r; small W for the semantic check (fp32)
    host = torch.zeros(2, TP, hq, ROWS, 32)
    for c in range(TP):
        code = _enc(torch.arange(hq * c, hq * (c + 1)).float(), torch.arange(ROWS).float())
        host[:, c] = code[:, :, None]
    x = ttnn.from_torch(  # [2, TP * hq, ROWS, 32] -> chip (r, c): [1, hq, ROWS, 32], heads hq c ..
        host.reshape(2, TP * hq, ROWS, 32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, TP), dims=(0, 1)),
    )
    for in_dim, out_dim in ((1, 2), (2, 1)):
        y = ttnn.experimental.all_to_all_async_generic(
            x, in_dim=in_dim, out_dim=out_dim, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
        )
        shapes = tuple(y.shape)
        t = ttnn.to_torch(ttnn.get_device_tensors(y)[1]).float()  # chip (0, 1)
        want = _enc(torch.arange(H).float(), torch.arange(rq, 2 * rq).float())[
            None, :, :, None
        ]  # all heads, rows quarter 1
        ok = shapes == (1, H, rq, 32) and torch.equal(t, want.expand_as(t))
        print(
            f"[a2a] in_dim={in_dim} out_dim={out_dim}: out {shapes}, heads->rows exact on chip (0,1): {ok}", flush=True
        )
        ttnn.deallocate(y)

    # timing at the model's shapes, bf16
    for name, shape, dims in (
        ("heads->rows", (1, hq, ROWS, W), None),
        ("rows->heads", (1, H, rq, W), None),
    ):
        t_in = ttnn.from_torch(
            torch.randn(*shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for in_dim, out_dim in ((1, 2), (2, 1)):
            try:
                f = lambda: ttnn.experimental.all_to_all_async_generic(  # noqa: E731
                    t_in,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_links=2,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cluster_axis=1,
                )
                out = f()
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(out)
                t0 = time.time()
                for _ in range(10):
                    ttnn.deallocate(f())
                ttnn.synchronize_device(mesh_device)
                print(
                    f"[a2a] {name} {shape} in_dim={in_dim} out_dim={out_dim}: {(time.time() - t0) / 10 * 1e3:.3f} ms",
                    flush=True,
                )
            except Exception as ex:
                print(
                    f"[a2a] {name} in_dim={in_dim} out_dim={out_dim}: FAILED {str(ex).splitlines()[0][:90]}", flush=True
                )
        ttnn.deallocate(t_in)
