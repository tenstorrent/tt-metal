# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein's ring joint SDPA at its served per-device shapes (Blackhole Galaxy, SP8 x TP4), on both compute paths:
the compute_common.hpp path (fp32 dest, fp32 score block) and the streaming path with fp32 dest
(SDPAProgramConfig.streaming_fp32_dest, bf16 score block), which Klein's attention selects. Checks: finite output,
deterministic output (two calls, identical bytes), relative error and PCC against an fp64 reference on sampled query
rows with the K tail masked at logical_n."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor
from ....utils.test import ring_params_8k

HEADS, HEAD_DIM, SP, TP = 24, 128, 8, 4
TEXT_PAD = 1024 - 320  # 1024-row text bucket with 320 valid caption rows: the masked K tail
Q_GAIN = 3.0  # peaked scores (std ~3), so the max / exp numerics matter
SAMPLE_ROWS = 48

# per-device rows -> (q, k) chunks, as Klein's Flux3KleinAttention resolves them
CASES = {
    "2048": (2176, (128, 512)),
    "2048_ref": (2688, (160, 512)),
    "4096": (8320, (128, 640)),
    "4096_ref": (8832, (128, 512)),
}
# measured (exp 11): compute_common.hpp path 4.0e-2 to 4.7e-2, streaming path 3.0e-2 to 3.1e-2
MAX_REL_ERR = {False: 0.06, True: 0.045}
MIN_PCC = 0.998


def _reference(q, k, v, logical_n, rows_idx):
    qs = q[0][:, rows_idx].double()
    kk, vv = k[0][:, :logical_n].double(), v[0][:, :logical_n].double()
    s = torch.einsum("hrd,hkd->hrk", qs, kk) / HEAD_DIM**0.5
    return torch.einsum("hrk,hkd->hrd", torch.softmax(s, dim=-1), vv)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("streaming_fp32_dest", [False, True], ids=["legacy", "streaming"])
@pytest.mark.parametrize("case", list(CASES))
def test_ring_sdpa_flux3_klein(*, mesh_device: ttnn.MeshDevice, case: str, streaming_fp32_dest: bool) -> None:
    rows, (q_chunk, k_chunk) = CASES[case]
    logical_n = rows * SP - TEXT_PAD

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

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=True,
        streaming_fp32_dest=streaming_fp32_dest,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )

    def call() -> torch.Tensor:
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

    first = call()
    second = call()

    assert torch.isfinite(first.float()).all(), "non-finite attention output"
    assert torch.equal(first.view(torch.int16), second.view(torch.int16)), "two identical calls differ"

    rows_idx = torch.linspace(0, rows * SP - 1, SAMPLE_ROWS).long()
    ref = _reference(host[0], host[1], host[2], logical_n, rows_idx)
    got = first[0][:, rows_idx].double()
    rel = ((got - ref).norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    logger.info(
        f"klein ring SDPA {case} rows {rows} logical_n {logical_n} chunks ({q_chunk}, {k_chunk}) "
        f"streaming_fp32_dest={streaming_fp32_dest}: rel err vs fp64 {rel:.3e}, PCC {pcc:.6f}"
    )
    assert rel <= MAX_REL_ERR[streaming_fp32_dest], f"rel err {rel:.3e} > {MAX_REL_ERR[streaming_fp32_dest]}"
    assert pcc >= MIN_PCC, f"PCC {pcc:.6f} < {MIN_PCC}"
