# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GegluRoutedExpert (custom op) vs the torch golden, on one chip and on the 2x4 block, at small shapes and at
the 26B-A4B dispatch-prefill shapes (weights and index table from TtRoutedExpert, which only loads them here).

Every chip of the mesh holds its own local experts (``TtRoutedExpert``'s placement, the index table sharded as
``dispatch_prefill.py`` shards it), its own dispatched buffer, counts and region offsets; the outputs are compared
per chip over the expert regions. Weights are bf4 and activations bf8 as in the model, so the golden is computed on
the bf4-rounded weights and the rounded input (the kernel's own LoFi / bf16-accumulator error is what remains).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.gemma4.custom_ops.geglu_routed_expert import GegluRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc

MESHES = [
    pytest.param((1, 1), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="1x1"),
    pytest.param((2, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="2x4"),
]
NUM_GLOBAL_EXPERTS = 128  # the 26B-A4B count/offset vector length (the Galaxy table indexes 128 experts)
GEMMA_26B = dict(emb=2816, hidden=704, E=4, max_tokens=4096)  # dispatch_prefill on the 8x4 Galaxy (compute_constants)


def _round(t: torch.Tensor, dtype) -> torch.Tensor:
    return ttnn.to_torch(ttnn.from_torch(t.float(), dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


class _Fixture:
    """The per-chip inputs of one op call on the whole mesh plus the per-chip torch golden."""

    def __init__(self, mesh_device, emb, hidden, E, max_tokens, counts_per_chip, x_layout, seed=0):
        torch.manual_seed(seed)
        rows, cols = mesh_device.shape
        self.D = D = rows * cols
        self.E, self.max_tokens, self.emb, self.hidden = E, max_tokens, emb, hidden
        self.M_buf = E * max_tokens
        self.x_layout = x_layout
        # HF (out, in) weights of every global expert; TtRoutedExpert transposes and distributes them.
        self.torch_weights = [
            {
                "gate_proj": torch.randn(hidden, emb) * 0.08,
                "up_proj": torch.randn(hidden, emb) * 0.08,
                "down_proj": torch.randn(emb, hidden) * 0.05,
            }
            for _ in range(D * E)
        ]
        table = ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=E, dispatch_group_size=rows, num_dispatch_groups=cols
        )
        global_idx = ttnn.from_torch(
            table,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(1, 0)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        self.global_idx = ttnn.squeeze(ttnn.squeeze(global_idx, 0), 0)
        self.idx_per_chip = [ttnn.to_torch(t).flatten().tolist() for t in ttnn.get_device_tensors(self.global_idx)]
        self.routed_expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=E,
            global_expert_idx_table=self.global_idx,
            emb_dim=emb,
            hidden_dim=hidden,
            max_tokens=max_tokens,
            torch_weights=self.torch_weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat4_b,
            activation=ttnn.RoutedExpertActivation.Silu,  # loader only; its fused op is never run
            hybrid_token_threshold=None,
        )
        # Per chip: counts and region offsets indexed by GLOBAL id (expert e's region at e * max_tokens).
        G = max(NUM_GLOBAL_EXPERTS, D * E)
        self.counts = torch.zeros(D, G, dtype=torch.int32)
        self.starts = torch.zeros(D, G, dtype=torch.int32)
        for d in range(D):
            for e, gid in enumerate(self.idx_per_chip[d]):
                self.counts[d, gid] = counts_per_chip[d][e]
                self.starts[d, gid] = e * max_tokens
        self.x = torch.randn(D, self.M_buf, emb)
        shard0 = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        self.tt_counts = ttnn.from_torch(
            self.counts, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=shard0
        )
        self.tt_starts = ttnn.from_torch(
            self.starts, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=shard0
        )
        if x_layout == "row_major":
            self.tt_x = ttnn.from_torch(
                self.x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=shard0
            )
            self.x_rounded = self.x.bfloat16().float()
        else:
            self.tt_x = ttnn.from_torch(
                self.x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=shard0
            )
            self.x_rounded = torch.stack([_round(self.x[d], ttnn.bfloat8_b) for d in range(D)])
        self.mesh_device = mesh_device

    def golden(self, d):
        """Chip d's expected output and the mask of the rows the op writes."""
        w = lambda name, e: _round(
            self.torch_weights[self.idx_per_chip[d][e]][name].T.contiguous(), ttnn.bfloat4_b
        )  # noqa: E731
        out = GegluRoutedExpert.golden(
            self.x_rounded[d],
            self.counts[d],
            self.starts[d],
            self.idx_per_chip[d],
            [w("gate_proj", e) for e in range(self.E)],
            [w("up_proj", e) for e in range(self.E)],
            [w("down_proj", e) for e in range(self.E)],
        )
        mask = torch.zeros(self.M_buf, dtype=torch.bool)
        for e, gid in enumerate(self.idx_per_chip[d]):
            n, s = int(self.counts[d, gid]), int(self.starts[d, gid])
            mask[s : s + n] = True
        return out, mask

    def custom_op(self):
        return GegluRoutedExpert(
            self.mesh_device,
            self.routed_expert.gate_projs,
            self.routed_expert.up_projs,
            self.routed_expert.down_projs,
            self.global_idx,
            max_tokens=self.max_tokens,
        )

    def run_custom(self, op):
        output = None
        if self.tt_x.layout == ttnn.TILE_LAYOUT:  # keep x pristine (the op would write in place)
            output = ttnn.allocate_tensor_on_device(
                self.tt_x.shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
        return op(self.tt_x, self.tt_counts, self.tt_starts, output=output)

    def per_chip(self, tt_out):
        return [ttnn.to_torch(t).float().reshape(self.M_buf, self.emb) for t in ttnn.get_device_tensors(tt_out)]


def _check(fx, custom, tag, golden_pcc=0.98):
    worst_golden = 1.0
    for d in range(fx.D):
        golden, mask = fx.golden(d)
        g, c = golden[mask], custom[d][mask]
        assert torch.isfinite(c).all(), f"chip {d}: non-finite custom output"
        _, pcc_g = comp_pcc(g, c, golden_pcc)
        logger.info(f"[{tag}] chip {d}: PCC custom/golden {pcc_g:.6f}")
        worst_golden = min(worst_golden, float(pcc_g))
    assert worst_golden >= golden_pcc, f"custom op vs golden PCC {worst_golden} < {golden_pcc}"
    return worst_golden


def _time_ms(mesh_device, fn, iters):
    for _ in range(2):  # compile, then the program-cache-hit path
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e3


@pytest.mark.skipif(not is_blackhole(), reason="GegluRoutedExpert assumes the Blackhole compute grid")
@pytest.mark.parametrize("x_layout", ["row_major", "tile"])
@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
def test_geglu_routed_expert_small(mesh_device, device_params, x_layout):
    """emb 256 / hidden 512, 2 local experts of 256 rows: one full region, one 100-row tail (4 tile rows)."""
    D = mesh_device.get_num_devices()
    counts = [[256 if (d + e) % 2 == 0 else 100 for e in range(2)] for d in range(D)]
    fx = _Fixture(mesh_device, 256, 512, 2, 256, counts, x_layout, seed=1)
    custom = fx.per_chip(fx.run_custom(fx.custom_op()))
    _check(fx, custom, f"small/{x_layout}")


@pytest.mark.skipif(not is_blackhole(), reason="GegluRoutedExpert assumes the Blackhole compute grid")
@pytest.mark.parametrize("x_layout", ["row_major", "tile"])
@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
def test_geglu_routed_expert_gemma_shapes(mesh_device, device_params, x_layout):
    """The 26B-A4B dispatch-prefill shapes (hidden 2816, intermediate 704, 4 local experts, 4096 max tokens): a
    full region, a multi-chunk tail (1500 rows = 46 full + 1 partial tile rows), one partial tile (37 rows) and an
    empty expert, rotated per chip; then the warm per-call time of the op."""
    D = mesh_device.get_num_devices()
    pattern = [4096, 1500, 37, 0]
    counts = [pattern[d % 4 :] + pattern[: d % 4] for d in range(D)]
    fx = _Fixture(mesh_device, **GEMMA_26B, counts_per_chip=counts, x_layout=x_layout, seed=2)
    op = fx.custom_op()
    custom = fx.per_chip(fx.run_custom(op))
    plan = next(iter(op._plans.values()))
    logger.info(
        f"plan: per_core_M {plan['per_core_M']} in0_block_w_gu {plan['in0_block_w_gu']} chunk {plan['chunk_M_tiles']} "
        f"per_core_N gu/d {plan['per_core_N_gu']}/{plan['per_core_N_d']} down subblock {plan['d_out_subblock_w']} "
        f"(tail {plan['d_out_subblock_w_tail']}) shard widths gu/d {plan['gu_shard_w']}/{plan['d_shard_w']} "
        f"CB bytes {plan['cb_bytes']} of {plan['l1_budget']}"
    )
    _check(fx, custom, f"gemma/{x_layout}")
    iters = int(os.environ.get("GEMMA4_GEGLU_TIMING_ITERS", "5"))
    t_custom = _time_ms(mesh_device, lambda: fx.run_custom(op), iters)
    logger.info(
        f"[gemma/{x_layout}] warm per-call wall time over {iters} iterations ({D} chips, sum of counts "
        f"{sum(pattern)} rows/chip): {t_custom:.3f} ms"
    )
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        ttnn.ReadDeviceProfiler(mesh_device)
