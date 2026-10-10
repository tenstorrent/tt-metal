# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic routed experts that cost a pool, not 896 experts (plan §5.4, option c).

Converting 896 distinct experts to bf4 on host is what makes a K3 MoE layer slow to build
(~30 G parameters tilized and packed per layer). Here ``P`` pool weight sets are converted
once and replicated to every chip; each (chip, local slot) expert is then made distinct ON
DEVICE by flipping the sign of whole intermediate channels of its ``down`` weight:

    expert e = pool p(e), with  down_e^T[k, :] = sign_e[k] * down_p^T[k, :]

A whole row of ``down^T`` spans whole bf4 shared-exponent blocks, so the flip commutes with
the quantization exactly: the device expert is ``sign_e ⊙ Q(down_p)``, and the reference uses
the dequantized device pool with the same signs (``PooledExperts``). Gate and up are the
pool's own tensors, shared by every slot that uses that pool.

Every slot on every chip gets its own random signs, so a token dispatched to the wrong chip
or slot meets a different expert -- the property the MoE routing oracle needs.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import PooledExperts
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping


class PooledRoutedExperts:
    """Weight provider for ``TtRoutedExpert`` (``torch_weights=`` accepts it in place of a list).

    Args:
        num_experts: routed experts in the model (896 for K3).
        pools: distinct weight sets behind them.
        hidden: the experts' input width (K3 LatentMoE: 3584).
        intermediate: the experts' intermediate width (K3: 3072).
        seed: drives both the pool weights and the per-expert signs.
        std: pool weight scale.
    """

    def __init__(self, num_experts: int, pools: int, hidden: int, intermediate: int, *, seed: int, std: float = 0.02):
        if pools < 1 or pools > num_experts:
            raise ValueError(f"need 1 <= pools <= num_experts, got pools={pools}, num_experts={num_experts}")
        self.num_experts, self.pools, self.hidden, self.intermediate = num_experts, pools, hidden, intermediate
        generator = torch.Generator().manual_seed(seed)
        normal = lambda *shape: (torch.randn(*shape, generator=generator) * std).bfloat16()
        # HF layout, [out, in].
        self.gate = normal(pools, intermediate, hidden)
        self.up = normal(pools, intermediate, hidden)
        self.down = normal(pools, hidden, intermediate)
        self.sign = torch.where(torch.rand(num_experts, intermediate, generator=generator) < 0.5, -1.0, 1.0)
        self._experts_per_chip = None
        self._dequantized = None

    def __len__(self) -> int:
        return self.num_experts

    def pool_of(self, experts_per_chip: int) -> torch.Tensor:
        """Pool behind each global expert. Slot ``i`` of every chip uses pool ``i % P``; the global id
        of slot ``i`` is ``device * experts_per_chip + i`` (``ExpertMapping``, either major order)."""
        return (torch.arange(self.num_experts) % experts_per_chip) % self.pools

    def build_device_weights(
        self,
        mesh_device,
        experts_per_chip: int,
        weights_dtype,
        memory_config_for: Callable[[int], ttnn.MemoryConfig],
    ):
        """``(gate, up, down)`` lists, one device tensor per local slot, as ``TtRoutedExpert`` holds them.

        ``memory_config_for(n)`` is the placement ``TtRoutedExpert`` uses for a weight whose free
        dim is ``n``. Also records the dequantized pools for ``reference_experts``.
        """
        rows, cols = mesh_device.shape
        if rows * cols * experts_per_chip != self.num_experts:
            raise ValueError(f"{rows}x{cols} mesh x {experts_per_chip} slots != {self.num_experts} experts")
        table = ExpertMapping.create_global_expert_idx_table(
            experts_per_chip, rows, cols
        )  # [group=col, chip=row, slot]
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def place(kn: torch.Tensor, memory_config) -> ttnn.Tensor:
            return ttnn.from_torch(
                kn.contiguous(),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=replicate,
                memory_config=memory_config,
            )

        # TTNN matmul layout is [K, N] = HF transposed.
        gate_pool = [place(self.gate[p].T, memory_config_for(self.intermediate)) for p in range(self.pools)]
        up_pool = [place(self.up[p].T, memory_config_for(self.intermediate)) for p in range(self.pools)]
        down_pool = [place(self.down[p].T, ttnn.DRAM_MEMORY_CONFIG) for p in range(self.pools)]

        downs = []
        for slot in range(experts_per_chip):
            signs = torch.empty(rows, cols, self.intermediate, 1)
            for row in range(rows):
                for col in range(cols):
                    signs[row, col, :, 0] = self.sign[int(table[col, row, slot])]
            sign_tt = ttnn.from_torch(
                signs,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, 1)),
            )
            sign_tt = ttnn.squeeze(ttnn.squeeze(sign_tt, 0), 0)  # per chip [I, 1]
            flipped = ttnn.multiply(down_pool[slot % self.pools], sign_tt, dtype=weights_dtype)
            ttnn.deallocate(sign_tt)
            placed = ttnn.to_memory_config(flipped, memory_config_for(self.hidden))
            if placed is not flipped:
                ttnn.deallocate(flipped)
            downs.append(placed)

        first = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        # Back to HF layout for the reference: the dequantized values the device multiplies by.
        self._dequantized = (
            torch.stack([first(t).T for t in gate_pool]),
            torch.stack([first(t).T for t in up_pool]),
            torch.stack([first(t).T for t in down_pool]),
        )
        for tensor in down_pool:
            ttnn.deallocate(tensor)
        self._experts_per_chip = experts_per_chip
        return (
            [gate_pool[slot % self.pools] for slot in range(experts_per_chip)],
            [up_pool[slot % self.pools] for slot in range(experts_per_chip)],
            downs,
        )

    def reference_experts(self) -> PooledExperts:
        """The experts the device holds, for ``reference_ops.routed_experts``. Needs a prior build."""
        if self._dequantized is None:
            raise RuntimeError("build_device_weights has not run; the dequantized pools come from the device")
        gate, up, down = self._dequantized
        return PooledExperts(gate, up, down, self.pool_of(self._experts_per_chip), self.sign)
