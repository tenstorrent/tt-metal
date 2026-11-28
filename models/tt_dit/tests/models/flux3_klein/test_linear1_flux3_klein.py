# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein's linear1 (QKV + MLP-in as one all-gather matmul, swiglu_last_chunk + grouped N) at the 4096 row counts:
its MLP output byte-identical to the separate proj_mlp call and q / k / v equal to to_qkv but for one rounding of the
last K block on a few elements (to_qkv packs it straight from the fp32 dest, linear1 accumulates it in L1; measured:
at most 1.4e-6 of the elements, by at most 2^-5), and a block with linear1 (and the TP barrier before proj_out)
byte-identical call after call and matching the same block without it."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....blocks.attention_opt import Attention
from ....models.transformers.flux3_klein import transformer_flux3_klein as klein
from ....models.transformers.flux3_klein.transformer_block_flux3_klein import Flux3KleinTransformerBlock
from ....parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
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


def _diff(a: ttnn.Tensor, b: ttnn.Tensor) -> tuple[float, float]:
    """The largest |a - b| and the largest fraction of differing elements over the devices."""
    worst, frac = 0.0, 0.0
    for x, y in zip(_bytes(a), _bytes(b), strict=True):
        diff = (x.view(torch.bfloat16).float() - y.view(torch.bfloat16).float()).abs()
        worst, frac = max(worst, float(diff.max())), max(frac, float((diff > 0).sum()) / diff.numel())
    return worst, frac


@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(TP, SP)], indirect=True)
def test_linear1(*, mesh_device: ttnn.MeshDevice) -> None:
    tp_axis, sp_axis = 0, 1
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    parallel = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=TP, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=SP, mesh_axis=sp_axis),
    )
    state = _random_state(torch.Generator().manual_seed(0))
    blocks = {}
    for linear1 in (False, True):
        block = Flux3KleinTransformerBlock(
            device=mesh_device, ccl_manager=ccl, parallel_config=parallel, linear1=linear1
        )
        missing = block.load_torch_state_dict(dict(state), strict=False).missing_keys
        assert not missing, missing
        blocks[linear1] = block
    plain, fused_block = blocks[False], blocks[True]
    full = mesh_device.compute_with_storage_grid_size()
    gen = torch.Generator().manual_seed(1)

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
        q, k, v = plain.attn.to_qkv(
            x,
            compute_kernel_config=plain.attn.mm_compute_kernel_config,
            parallel_config=plain.attn.parallel_config,
            core_grid=grid,
            use_heuristic_mmcfg=True,
        )
        mlp = plain.proj_mlp(x, parallel_config=parallel, use_heuristic_mmcfg=True, core_grid=grid)
        k_block = fused_block._linear1_k_block(rows)
        assert k_block is not None, f"QKV and MLP-in resolve different K blocks at {rows} rows"
        fused = fused_block._linear1_qkv_mlp(x, k_block)
        diffs = {name: _diff(a, b) for name, a, b in zip(("q", "k", "v"), (q, k, v), fused)}
        logger.info(f"linear1 vs separate at {rows} rows (max |diff|, differing fraction): {diffs}")
        for name, (worst, frac) in diffs.items():
            assert worst <= 2**-3 and frac <= 1e-5, f"linear1 {name} differs beyond rounding at {rows} rows"
        assert _same(mlp, fused[3]), f"linear1 mlp differs from the separate op at {rows} rows"

        def run_block(block, rope=rope, total=total, hosts=(x_host, shift_host, scale_host, gate_host)):
            # fresh inputs per call; the output lives in proj_out's persistent RS buffer, so copy it out right away
            x_in, shift, scale_plus_one, gate = (rows_tensor(t) for t in hosts)
            return _bytes(block(x_in, rope=rope, mod=(shift, scale_plus_one, gate), logical_length=total))

        reference = run_block(plain)
        first = run_block(fused_block)
        values = lambda out: torch.cat([d.view(torch.bfloat16) for d in out]).float()
        assert_quality(values(reference), values(first), pcc=0.9999)
        for i in range(CALLS):
            assert _same(first, run_block(fused_block)), f"linear1 block not deterministic at {rows} rows (call {i})"
        assert _same(reference, run_block(plain)), f"plain block not deterministic at {rows} rows"
