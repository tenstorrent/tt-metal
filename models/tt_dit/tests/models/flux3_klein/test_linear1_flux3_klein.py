# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein's linear1 (QKV + MLP-in as one all-gather-matmul, swiglu_chunk + grouped N) at the 4096 row
counts: its q / k / v / MLP outputs must be byte-identical to the separate to_qkv and proj_mlp calls, and a whole
block with linear1 and the one-tile TP barrier before proj_out must be byte-identical to the block without them,
call after call."""

from __future__ import annotations

import pytest
import torch

import ttnn

from ....blocks.attention_opt import Attention
from ....models.transformers.flux3_klein import transformer_flux3_klein as klein
from ....models.transformers.flux3_klein.transformer_block_flux3_klein import Flux3KleinTransformerBlock
from ....parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.test import ring_params_8k

HIDDEN, MLP_HIDDEN, HEAD_DIM, SP, TP = 3072, 9216, 128, 8, 4
ROWS = (8192, 8320, 8960)
CALLS = 3


def _random_state(gen: torch.Generator) -> dict[str, torch.Tensor]:
    w = lambda *shape: (torch.randn(shape, generator=gen) * shape[-1] ** -0.5).to(torch.bfloat16)
    return {
        "q_proj.weight": w(HIDDEN, HIDDEN),
        "k_proj.weight": w(HIDDEN, HIDDEN),
        "v_proj.weight": w(HIDDEN, HIDDEN),
        "norm.query_norm.scale": 1 + 0.1 * torch.randn(HEAD_DIM, generator=gen),
        "norm.key_norm.scale": 1 + 0.1 * torch.randn(HEAD_DIM, generator=gen),
        "mlp_in.weight": w(2 * MLP_HIDDEN, HIDDEN),
        "attn_out.weight": w(HIDDEN, HIDDEN),
        "mlp_out.weight": w(HIDDEN, MLP_HIDDEN),
    }


def _bytes(t: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).view(torch.int16) for d in ttnn.get_device_tensors(t)]


def _same(a: ttnn.Tensor | list[torch.Tensor], b: ttnn.Tensor | list[torch.Tensor]) -> bool:
    a, b = (_bytes(t) if isinstance(t, ttnn.Tensor) else t for t in (a, b))
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(TP, SP)], indirect=True)
def test_linear1(*, mesh_device: ttnn.MeshDevice) -> None:
    tp_axis, sp_axis = 0, 1
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    parallel = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=TP, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=SP, mesh_axis=sp_axis),
    )
    block = Flux3KleinTransformerBlock(device=mesh_device, ccl_manager=ccl, parallel_config=parallel)
    gen = torch.Generator().manual_seed(0)
    missing = block.load_torch_state_dict(_random_state(gen), strict=False).missing_keys
    assert not missing, missing
    attn = block.attn
    full = mesh_device.compute_with_storage_grid_size()

    def rows_tensor(t: torch.Tensor) -> ttnn.Tensor:
        return tensor.from_torch(t, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None, sp_axis, tp_axis])

    for rows in ROWS:
        total = rows * SP
        ids = torch.randint(0, 256, (total, len(klein.ROPE_AXES_DIM)), generator=gen)
        rope = tuple(
            tensor.from_torch(t, device=mesh_device, dtype=ttnn.float32, mesh_axes=[None, None, sp_axis, None])
            for t in klein.rope_cos_sin(ids, total)
        )
        x_host = torch.randn(1, total, HIDDEN, generator=gen)
        shift_host, gate_host = (0.1 * torch.randn(1, total, HIDDEN, generator=gen) for _ in range(2))
        scale_host = 1 + 0.1 * torch.randn(1, total, HIDDEN, generator=gen)
        x = rows_tensor(x_host)

        grid = Attention.get_core_grid(rows, full)
        q, k, v = attn.to_qkv(
            x,
            compute_kernel_config=attn.mm_compute_kernel_config,
            parallel_config=attn.parallel_config,
            core_grid=grid,
            use_heuristic_mmcfg=True,
        )
        mlp = block.proj_mlp(x, parallel_config=parallel, use_heuristic_mmcfg=True, core_grid=grid)
        fused = block._linear1(x)
        assert fused is not None, f"linear1 fell back at {rows} rows"
        for name, a, b in zip(("q", "k", "v", "mlp"), (q, k, v, mlp), fused):
            assert _same(a, b), f"linear1 {name} differs from the separate op at {rows} rows"

        def run_block(linear1: bool, rope=rope, total=total, hosts=(x_host, shift_host, scale_host, gate_host)):
            block._l1_disabled = not linear1
            x_in, shift, scale_plus_one, gate = (rows_tensor(t) for t in hosts)
            return _bytes(block(x_in, rope=rope, mod=(shift, scale_plus_one, gate), logical_length=total))

        reference = run_block(linear1=False)
        for i in range(CALLS):
            out = run_block(linear1=True)
            assert _same(reference, out), f"block with linear1 differs from the plain block at {rows} rows (call {i})"
        assert _same(reference, run_block(linear1=False)), f"plain block not deterministic at {rows} rows"
