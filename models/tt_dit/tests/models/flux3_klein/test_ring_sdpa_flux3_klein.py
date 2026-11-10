# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein's ring joint SDPA at every per-device length of its chunk table (Blackhole Galaxy, SP8 x TP4), on the
streaming compute path with fp32 dest, on the 12 x 9 grid and with the CCL row's cores added (116 cores). Checks: finite
and deterministic output, error and PCC against an fp64 reference on sampled query rows with the K tail masked at
logical_n, and byte-identical outputs on both grids."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import (
    RING_SDPA_CCL_ROW_START_X,
    RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
    RING_SDPA_CHUNKS_STREAMING_BY_LOCAL_ROWS,
)
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor
from ....utils.test import ring_params_8k

HEADS, HEAD_DIM, SP, TP = 24, 128, 8, 4
VALID_TEXT = 320  # valid caption rows; the rest of the text bucket is the masked K tail
Q_GAIN = 3.0  # peaked scores (std ~3), so the max / exp numerics matter
SAMPLE_ROWS = 48
MAX_REL_ERR = 0.045
MIN_PCC = 0.998

# per-device rows: text bucket (0: no text rows)
TEXT_BUCKET_BY_ROWS = {2560: 4096, 2688: 1024, 2816: 2048, 3072: 4096, 8192: 0, 8320: 1024, 8960: 2048}
# the chunks Klein's attention resolves on the streaming path: the streaming entry, else the shared table's
CHUNKS = {**RING_SDPA_CHUNKS_BY_LOCAL_ROWS, **RING_SDPA_CHUNKS_STREAMING_BY_LOCAL_ROWS}


def _reference(q, k, v, logical_n, rows_idx):
    qs = q[0][:, rows_idx].double()
    kk, vv = k[0][:, :logical_n].double(), v[0][:, :logical_n].double()
    s = torch.einsum("hrd,hkd->hrk", qs, kk) / HEAD_DIM**0.5
    return torch.einsum("hrk,hkd->hrd", torch.softmax(s, dim=-1), vv)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("rows", sorted(CHUNKS))
def test_ring_sdpa_flux3_klein(*, mesh_device: ttnn.MeshDevice, rows: int) -> None:
    q_chunk, k_chunk = CHUNKS[rows]
    bucket = TEXT_BUCKET_BY_ROWS[rows]
    logical_n = rows * SP - (bucket - VALID_TEXT if bucket else 0)

    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    full = mesh_device.compute_with_storage_grid_size()
    grid = (full.x, full.y - 1)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(TP, SP), dims=(1, 2))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(TP, SP), dims=(1, 2))

    gen = torch.Generator().manual_seed(0)
    host = [torch.randn((1, HEADS, rows * SP, HEAD_DIM), generator=gen) for _ in range(3)]
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

    def call(ccl_row_start_x: int | None) -> torch.Tensor:
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
        result = ttnn.to_torch(out[0], mesh_composer=composer)
        for t in out:
            ttnn.deallocate(t)
        return result

    rows_idx = torch.linspace(0, rows * SP - 1, SAMPLE_ROWS).long()
    ref = _reference(host[0], host[1], host[2], logical_n, rows_idx)
    outputs = {}
    for label, start_x in (("108", None), ("116", RING_SDPA_CCL_ROW_START_X)):
        first, second = call(start_x), call(start_x)
        assert torch.isfinite(first.float()).all(), f"{label}: non-finite attention output"
        assert torch.equal(first.view(torch.int16), second.view(torch.int16)), f"{label}: two identical calls differ"
        got = first[0][:, rows_idx].double()
        rel = ((got - ref).norm() / ref.norm()).item()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        logger.info(f"rows {rows} chunks ({q_chunk}, {k_chunk}) {label} cores: rel err {rel:.3e}, PCC {pcc:.6f}")
        assert rel <= MAX_REL_ERR, f"{label}: rel err {rel:.3e} > {MAX_REL_ERR}"
        assert pcc >= MIN_PCC, f"{label}: PCC {pcc:.6f} < {MIN_PCC}"
        outputs[label] = first
    assert torch.equal(outputs["108"].view(torch.int16), outputs["116"].view(torch.int16)), "116 != 108 cores"
