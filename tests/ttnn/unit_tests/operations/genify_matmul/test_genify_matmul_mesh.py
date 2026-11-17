# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""genify_matmul on a multi-chip mesh, the way the TP models call it (#65150).

Qwen3.8 runs on a (1, 4) mesh and FLUX.2 on a (2, 2) mesh. Each chip holds the replicated
activation and its own column slice of the weight (tensor parallel), so every chip runs the same
program on different weight data. genify_matmul dispatches one ProgramDescriptor to every chip
of the mesh (ttnn.generic_op SPMD); this checks every chip's output against torch.
"""

import pytest
import torch

import ttnn
from eval.golden_tests.genify_matmul.helpers import (
    activation_memory_config,
    pytorch_genify_matmul,
    quantize,
    weight_memory_config,
)
from models.common.utility_functions import comp_pcc
from ttnn.operations.genify_matmul import genify_matmul

TILE = 32

# name, M, K, N per chip, activation dtype, weight dtype, placement, epilogue, fidelity, PCC
CASES = [
    ("qwen_decode_dram_sharded", 32, 5120, 3584, ttnn.bfloat16, ttnn.bfloat4_b, "dram_sharded", "none", "LoFi", 0.99),
    ("qwen_prefill_interleaved", 512, 5120, 3584, ttnn.bfloat16, ttnn.bfloat4_b, "interleaved", "none", "LoFi", 0.99),
    ("qwen_gate_up_swiglu", 128, 5120, 4352, ttnn.bfloat16, ttnn.bfloat4_b, "interleaved", "swiglu", "LoFi", 0.99),
    ("flux_attn_out_addcmul", 256, 6144, 1536, ttnn.bfloat16, ttnn.bfloat16, "interleaved", "addcmul", "HiFi2", 0.999),
]


def _upload(t, mesh_device, dtype, memory_config, mapper):
    out = ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    if memory_config is not None and memory_config != ttnn.DRAM_MEMORY_CONFIG:
        out = ttnn.to_memory_config(out, memory_config)
    return out


@pytest.mark.parametrize("mesh_device", [(1, 4), (2, 2)], ids=["mesh1x4", "mesh2x2"], indirect=True)
@pytest.mark.parametrize(
    "name, M, K, N, a_dtype, w_dtype, placement, epilogue, fidelity, pcc", CASES, ids=[c[0] for c in CASES]
)
def test_genify_matmul_mesh(mesh_device, name, M, K, N, a_dtype, w_dtype, placement, epilogue, fidelity, pcc):
    torch.manual_seed(0)
    n_dev = mesh_device.get_num_devices()
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    by_column = ttnn.ShardTensorToMesh(mesh_device, dim=-1)

    a = torch.randn(1, 1, M, K)
    w = torch.randn(K, N * n_dev) / K**0.5  # each chip gets its own N-column slice
    a_t = _upload(
        a, mesh_device, a_dtype, activation_memory_config(placement, a.shape, a_dtype, mesh_device, {}), replicate
    )
    granule = 2 * TILE if epilogue == "swiglu" else TILE
    w_mc = weight_memory_config(placement, (K, N), mesh_device, {}, granule=granule)
    w_t = _upload(w, mesh_device, w_dtype, w_mc, by_column)

    kwargs = {}
    n_out = N // 2 if epilogue == "swiglu" else N
    residual = gate = None
    if epilogue == "swiglu":
        kwargs["activation"] = "swiglu"
    elif epilogue == "addcmul":
        residual = torch.randn(1, 1, M, n_out * n_dev)
        gate = torch.randn(1, 1, 1, n_out * n_dev)
        kwargs["residual"] = _upload(residual, mesh_device, a_dtype, None, by_column)
        kwargs["gate"] = _upload(gate, mesh_device, a_dtype, None, by_column)
    if placement == "dram_sharded":
        kwargs["memory_config"] = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    out = genify_matmul(a_t, w_t, compute_kernel_config=compute, **kwargs)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()

    # Reference per chip on the values each chip stores, then concatenated like the output.
    a_q = quantize(a, a_dtype)
    refs = []
    for d in range(n_dev):
        w_d = quantize(w[:, d * N : (d + 1) * N], w_dtype)
        r = residual[..., d * n_out : (d + 1) * n_out] if residual is not None else None
        g = gate[..., d * n_out : (d + 1) * n_out] if gate is not None else None
        refs.append(
            pytorch_genify_matmul(
                a_q,
                w_d,
                activation=kwargs.get("activation"),
                residual=quantize(r, a_dtype) if r is not None else None,
                gate=quantize(g, a_dtype) if g is not None else None,
            )
        )
    ref = torch.cat(refs, dim=-1)
    assert got.shape[-2:] == ref.shape[-2:], f"{name}: shape {got.shape} vs {ref.shape}"
    for d in range(n_dev):
        sl = slice(d * n_out, (d + 1) * n_out)
        ok, value = comp_pcc(ref[..., sl], got[..., sl], pcc)
        assert ok, f"{name}: chip {d} PCC {value} < {pcc}"
