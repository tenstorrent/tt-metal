# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein's ring joint SDPA at every per-device length of its chunk tables, on the streaming compute path with
fp32 dest: SP8 x TP4 on one Blackhole Galaxy (4 x 8 mesh) on the 12 x 9 grid and with the CCL row's cores added
(116 cores), SP32 x TP4 on a quad of Galaxies under tt-run (4 x 32 mesh, one rank per host) on the 12 x 9 grid.
Checks: finite and deterministic output, error and PCC against an fp64 reference on sampled query rows with the K tail
masked at logical_n, and byte-identical outputs on both grids. Each rank checks the shards it holds."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import (
    RING_SDPA_CCL_ROW_START_X,
    RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
    RING_SDPA_CHUNKS_SP32_BY_LOCAL_ROWS,
)
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor
from ....utils.test import ring_params_8k

HEADS, HEAD_DIM, TP = 24, 128, 4
VALID_TEXT = 320  # valid caption rows; the rest of the text bucket is the masked K tail
Q_GAIN = 3.0  # peaked scores (std ~3), so the max / exp numerics matter
SAMPLE_ROWS = 48
MAX_REL_ERR = 0.045
MIN_PCC = 0.998

# (sp, per-device rows): text bucket (0: no text rows)
TEXT_BUCKET_BY_ROWS = {
    **{(8, r): b for r, b in {2560: 4096, 2688: 1024, 2816: 2048, 3072: 4096, 8192: 0, 8320: 1024, 8960: 2048}.items()},
    **{
        (32, r): b
        for r, b in {2048: 0, 2080: 1024, 2112: 2048, 2208: 1024, 2240: 2048, 128: 0, 64: 2048, 32: 1024}.items()
    },
    **{(32, r): b for r, b in {2176: 4096, 2304: 4096}.items()},
    # 4096x2304 and 2304x4096: no reference, a 1 MP and a 4 MP reference
    **{(32, r): b for r, b in {1152: 0, 1184: 1024, 1216: 2048, 1280: 4096}.items()},
    **{(32, r): b for r, b in {1312: 1024, 1344: 2048, 1408: 4096, 1696: 1024, 1728: 2048, 1792: 4096}.items()},
}
CHUNKS = {8: RING_SDPA_CHUNKS_BY_LOCAL_ROWS, 32: RING_SDPA_CHUNKS_SP32_BY_LOCAL_ROWS}


def _reference(q, k, v, logical_n, heads, rows_idx):
    qs = q[0][heads][:, rows_idx].double()
    kk, vv = k[0][heads][:, :logical_n].double(), v[0][heads][:, :logical_n].double()
    s = torch.einsum("hrd,hkd->hrk", qs, kk) / HEAD_DIM**0.5
    return torch.einsum("hrk,hkd->hrd", torch.softmax(s, dim=-1), vv)


def _local_shards(out: ttnn.Tensor) -> list[tuple[tuple[int, int], torch.Tensor]]:
    """((tp, sp) coordinate, [heads_per_device, rows, head_dim] bf16) of every shard this host holds."""
    view = out.device().get_view() if ttnn.using_distributed_env() else None
    shards = []
    for coord, shard in zip(out.tensor_topology().mesh_coords(), ttnn.get_device_tensors(out), strict=True):
        if view is None or view.is_local(coord):
            shards.append(((int(coord[0]), int(coord[1])), ttnn.to_torch(shard)[0]))
    return shards


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device, rows",
    [((4, sp), rows) for sp, table in CHUNKS.items() for rows in table],
    ids=[f"sp{sp}-{rows}" for sp, table in CHUNKS.items() for rows in table],
    indirect=["mesh_device"],
)
def test_ring_sdpa_flux3_klein(*, mesh_device: ttnn.MeshDevice, rows: int) -> None:
    sp = mesh_device.shape[1]
    q_chunk, k_chunk = CHUNKS[sp][rows]
    bucket = TEXT_BUCKET_BY_ROWS[(sp, rows)]
    logical_n = rows * sp - (bucket - VALID_TEXT if bucket else 0)

    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    full = mesh_device.compute_with_storage_grid_size()
    grid = (full.x, full.y - 1)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(TP, sp), dims=(1, 2))

    gen = torch.Generator().manual_seed(0)
    host = [torch.randn((1, HEADS, rows * sp, HEAD_DIM), generator=gen) for _ in range(3)]
    host[0] *= Q_GAIN
    host = [t.to(torch.bfloat16) for t in host]
    q, k, v = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
        for t in host
    )
    empty = bf16_tensor(torch.zeros((1, HEADS // TP, 0, HEAD_DIM)), device=mesh_device)
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )

    def call(ccl_row_start_x: int | None) -> list[tuple[tuple[int, int], torch.Tensor]]:
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=True,
            streaming_fp32_dest=True,
            ccl_row_sdpa_start_x=ccl_row_start_x,
        )
        out = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            empty,
            empty,
            empty,
            persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(k.shape, 2, 1),
            persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(v.shape, 2, 1),
            joint_strategy="rear",
            logical_n=logical_n,
            logical_l=0,
            program_config=program_config,
            compute_kernel_config=compute_config,
            dim=2,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(1),
            num_links=2,
            cluster_axis=1,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Ring,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(0, grid[1]),
        )
        result = _local_shards(out[0])
        for t in out:
            ttnn.deallocate(t)
        return result

    # the model adds the CCL row's cores at SP8 only
    grids = (("108", None), ("116", RING_SDPA_CCL_ROW_START_X)) if sp == 8 else (("108", None),)
    heads_per_device = HEADS // TP
    outputs = {}
    for label, start_x in grids:
        first, second = call(start_x), call(start_x)
        assert first, "this host holds no shard of the output"
        samples = max(1, SAMPLE_ROWS // len(first))
        local_idx = torch.linspace(0, rows - 1, samples).long()
        got, ref = [], []
        for ((tp, s), shard), (_, again) in zip(first, second, strict=True):
            assert torch.isfinite(shard.float()).all(), f"{label}: non-finite attention output on ({tp}, {s})"
            assert torch.equal(shard.view(torch.int16), again.view(torch.int16)), f"{label}: two calls differ"
            heads = slice(tp * heads_per_device, (tp + 1) * heads_per_device)
            got.append(shard[:, local_idx].double())
            ref.append(_reference(host[0], host[1], host[2], logical_n, heads, local_idx + s * rows))
        got, ref = torch.cat(got, dim=1), torch.cat(ref, dim=1)
        rel = ((got - ref).norm() / ref.norm()).item()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        logger.info(
            f"sp {sp} rows {rows} chunks ({q_chunk}, {k_chunk}) {label} cores: rel err {rel:.3e}, PCC {pcc:.6f}"
        )
        assert rel <= MAX_REL_ERR, f"{label}: rel err {rel:.3e} > {MAX_REL_ERR}"
        assert pcc >= MIN_PCC, f"{label}: PCC {pcc:.6f} < {MIN_PCC}"
        outputs[label] = first
    if "116" in outputs:
        for (_, a), (_, b) in zip(outputs["108"], outputs["116"], strict=True):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), "116 != 108 cores"
