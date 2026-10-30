# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash ``lm_head``, vocabulary-parallel over a 1xTP stage.

A replicated ``[D, V]`` head has every rank of the output stage read the whole ~370 MB weight
(bfloat4_b at ``V = 129280``) each step. Here each rank holds and multiplies ``V / TP`` columns, and
an all-gather along the TP axis puts the whole ``[..., V]`` logits row back on every rank, the
output a replicated :class:`~models.experimental.deepseek_v4_flash.tt.layers.Linear` gives.
"""

import torch
import ttnn

from models.experimental.deepseek_v41_flash.tt.decode.dspark import host_to_device
from models.experimental.deepseek_v4_flash.tt.common import _MASK_NEG, CCL_TOPOLOGY, DeepSeekV4Module
from models.experimental.deepseek_v4_flash.tt.decode.attention import _tp_cluster_axis
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.weight_cache import _CachePath

# Candidates each rank keeps for its draw: one tile of ttnn.sampling's top-k.
SAMPLE_K = 32


class DeepSeekV41LMHead(DeepSeekV4Module):
    """``x [A, B, T, D]`` (TILE) -> logits ``[A, B, T, V]`` (TILE), replicated over ``device``.

    ``weight`` is the checkpoint's ``[V, D]`` head or a thunk. On a multi-device ``device`` (a 1xTP
    mesh with fabric) the weight is split along ``V``, so ``V / TP`` has to be whole tiles; its cache
    file then carries the split (``<name>.tp<TP>``), since a cache key does not record the mesh mapping.
    """

    def __init__(self, weight, device: ttnn.MeshDevice, cache_file_name=None, dtype: ttnn.DataType = ttnn.bfloat16):
        self.device = device
        self.tp_size = device.get_num_devices()
        mesh_mapper = None
        if self.tp_size > 1:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=-1)
            if cache_file_name is not None:
                cache_file_name = _CachePath(
                    f"{cache_file_name}.tp{self.tp_size}", getattr(cache_file_name, "require_cache", False)
                )
        self.linear = Linear(weight, device, cache_file_name, dtype=dtype, mesh_mapper=mesh_mapper)
        self.weight = self.linear.weight
        local_vocab = self.weight.shape[-1]
        if local_vocab % ttnn.TILE_SIZE:
            raise ValueError(f"lm_head's {local_vocab} columns per rank are not whole tiles over TP{self.tp_size}")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        logits = self.linear(x)
        if self.tp_size == 1:
            return logits
        gathered = ttnn.all_gather(
            logits,
            dim=3,
            cluster_axis=_tp_cluster_axis(self.device),
            num_links=1,
            topology=CCL_TOPOLOGY,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(logits)
        return gathered


class DeepSeekV41DeviceSampler(DeepSeekV4Module):
    """The next token, picked on the output stage from the ranks' ``lm_head`` shards, so only an int leaves it.

    ``logits [1, 1, 1, V / TP]`` (TILE) is this rank's column shard of :class:`DeepSeekV41LMHead`'s ``linear``.
    Each rank draws a token from its top-:data:`SAMPLE_K` at temperature ``T`` and reduces them to their
    log-partition ``L_r = log sum exp(top / T)``. Only ``(L_r, token_r)`` is all-gathered, and a second draw
    picks rank ``r`` with probability ``exp(L_r) / sum exp(L)``. The token is a draw from ``softmax(logits / T)``
    over the union of the ranks' top-:data:`SAMPLE_K` -- what gathering every rank's candidates and sampling
    once gives, with ``TP`` pairs crossing the ring instead of ``V`` logits. Greedy (``T = 0``) takes ``k = 1``
    in both draws and ``L_r`` = the shard's max logit: an exact argmax.

    ``ttnn.sampling`` with no seed continues its core's PRNG, which :meth:`set_sampling` seeds and nothing in a
    trace re-seeds, so every replay draws fresh numbers. Its values are bf16, so the gathered ``L_r`` are taken
    relative to their max first: bf16 cannot place a log-partition of ~30 closer than 0.125.

    Returns ``[1, 1, 1, out_words]`` int32 ROW_MAJOR, the token first: one PCIe-aligned D2H page.
    """

    def __init__(self, device: ttnn.MeshDevice, local_vocab: int, pcie_alignment: int):
        self.device = device
        self.tp_size = device.get_num_devices()
        self.out_words = pcie_alignment // 4
        replicate = ttnn.ReplicateTensorToMesh(device)
        offsets = torch.arange(self.tp_size, dtype=torch.int32).reshape(-1, 1, 1, 1) * local_vocab
        self.offsets = ttnn.from_torch(
            offsets.expand(-1, 1, 1, SAMPLE_K).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
        )
        lead = torch.full((1, 1, 1, SAMPLE_K), _MASK_NEG)
        lead[..., 0] = 0.0
        consts = {"lead": lead, "zeros": torch.zeros(1, 1, 1, SAMPLE_K), "gate": torch.zeros(1, 1, 1, 1)}
        for name, value in consts.items():
            setattr(
                self,
                name,
                ttnn.from_torch(
                    value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=replicate
                ),
            )
        self.inv_temp = ttnn.from_torch(
            torch.ones(1, 1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=replicate
        )
        # ttnn.sampling's local sort consumes candidate tiles in pairs and hangs on a single one, so the rank's
        # top-SAMPLE_K row is widened by a tile of never-picked candidates.
        self.fill_vals = ttnn.from_torch(
            torch.full((1, 1, 1, SAMPLE_K), _MASK_NEG),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=replicate,
        )
        self.fill_idx = ttnn.from_torch(
            torch.zeros(1, 1, 1, SAMPLE_K, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=replicate,
        )
        # ttnn.sampling's per-user k, p and temp; temp multiplies, and the values arrive already scaled.
        self.k = ttnn.from_torch(
            torch.ones(1, dtype=torch.int32), dtype=ttnn.int32, device=device, mesh_mapper=replicate
        )
        self.p = ttnn.from_torch(torch.ones(1), dtype=ttnn.bfloat16, device=device, mesh_mapper=replicate)
        self.one = ttnn.from_torch(torch.ones(1), dtype=ttnn.bfloat16, device=device, mesh_mapper=replicate)

    def set_sampling(self, temperature: float, seed: int) -> None:
        """Sample at ``temperature`` (0, the initial state, is greedy) from PRNGs seeded with ``seed``.

        Queued on command queue 0, between the steps already posted there and the ones after."""
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        greedy = temperature == 0
        host_to_device(torch.full((1,), 1 if greedy else SAMPLE_K, dtype=torch.int32), self.k)
        host_to_device(torch.full((1, 1, 1, 1), 1.0 if greedy else 1.0 / temperature), self.inv_temp)
        host_to_device(torch.full((1, 1, 1, 1), 0.0 if greedy else 1.0), self.gate)
        ttnn.manual_seed(seed, device=self.device)

    def forward(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        vals, idx = ttnn.topk(logits, SAMPLE_K, dim=-1, largest=True, sorted=True)
        idx = ttnn.add(ttnn.typecast(idx, ttnn.int32), self.offsets, dtype=ttnn.int32)
        idx = ttnn.to_layout(ttnn.concat([idx, self.fill_idx], dim=-1), ttnn.ROW_MAJOR_LAYOUT)
        vals = ttnn.multiply(vals, self.inv_temp)
        token = ttnn.sampling(ttnn.concat([vals, self.fill_vals], dim=-1), idx, k=self.k, p=self.p, temp=self.one)
        if self.tp_size > 1:
            scaled = ttnn.typecast(vals, ttnn.float32)
            top = ttnn.max(scaled, dim=-1, keepdim=True)
            spread = ttnn.log(ttnn.sum(ttnn.exp(ttnn.subtract(scaled, top)), dim=-1, keepdim=True))
            weight = ttnn.add(top, ttnn.multiply(spread, self.gate))
            # Each rank's row: its weight in column 0 behind no-candidate fill, and its token in every column.
            weights = ttnn.add(self.lead, weight)
            tokens = ttnn.add(self.zeros, ttnn.typecast(ttnn.to_layout(token, ttnn.TILE_LAYOUT), ttnn.float32))
            weights, tokens = (
                ttnn.all_gather(
                    row,
                    dim=3,
                    cluster_axis=_tp_cluster_axis(self.device),
                    num_links=1,
                    topology=CCL_TOPOLOGY,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for row in (weights, tokens)
            )
            weights = ttnn.typecast(ttnn.subtract(weights, ttnn.max(weights, dim=-1, keepdim=True)), ttnn.bfloat16)
            tokens = ttnn.to_layout(ttnn.typecast(tokens, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT)
            token = ttnn.sampling(weights, tokens, k=self.k, p=self.p, temp=self.one)
        return ttnn.pad(token, [(0, 0), (0, 0), (0, 0), (0, self.out_words - 1)], 0)
