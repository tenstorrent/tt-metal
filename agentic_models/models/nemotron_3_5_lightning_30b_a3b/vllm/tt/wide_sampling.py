# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-owned, traced 64-candidate sampler; common <=32 sampling is unchanged.

Construct and reset before capture; call ``sample(logits, seeds, output)`` inside
sampling warmup/capture. The caller owns active-slot masking, seed advancement,
and merging narrow outputs for rows whose actual k <= 32. This module NEVER
advances seeds. Inputs/parameters/output retain their addresses across replay.
The draw is a deterministic counter hash, not the common sampler's RNG stream.
"""

import torch

import ttnn


class WideSampling:
    width = 64
    rows = 32

    def __init__(self, sampling, mesh_device):
        self.common = sampling.tt_sampling
        self.mesh = mesh_device
        self.mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        shards = self.common._get_num_sampling_shards()
        if self.common.multi_step_reduction or tuple(mesh_device.shape) != (1, 4):
            raise ValueError("WideSampling currently implements the Nemotron TP4 vocabulary contract")
        offsets = torch.arange(shards, dtype=torch.int64).repeat_interleave(64)
        offsets = offsets * (self.common.padded_vocab_size // shards)
        self.offsets = self._tensor(offsets.reshape(1, 1, 1, -1).expand(1, 1, 32, -1), ttnn.uint32)
        self.ranks = self._tensor(torch.arange(64).float().reshape(1, 1, 1, 64).expand(1, 1, 32, 64))
        self.zero_column = self._tensor(torch.zeros(1, 1, 32, 1))
        self.multiplier1 = self._tensor(torch.full((1, 1, 32, 1), 0x7FEB352D, dtype=torch.int64), ttnn.uint32)
        self.multiplier2 = self._tensor(torch.full((1, 1, 32, 1), 0x846CA68B, dtype=torch.int64), ttnn.uint32)
        self.k = self._tensor(torch.ones(1, 1, 32, 64))
        self.p = self._tensor(torch.zeros(1, 1, 32, 64))
        self.inv_temperature = self._tensor(torch.ones(1, 1, 32, 64))
        self.wide_mask = self._tensor(torch.zeros(1, 32))

    def _tensor(self, value, dtype=ttnn.float32):
        return ttnn.from_torch(
            value.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mapper,
        )

    def _write(self, target, value):
        host = ttnn.from_torch(value.contiguous(), dtype=target.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.mapper)
        ttnn.copy_host_to_device_tensor(host, target)

    def reset(self, *, top_k, top_p, temperature):
        """Refresh scheduler-owned parameters outside capture, only on changes.

        Inputs are scalars or exactly 32 slot-indexed values. Inactive rows must
        have safe values (k=1,p=0,temp=1). Greedy temp=0 is mapped to k=1.
        Unbounded k must be routed elsewhere, never silently clipped here.
        """

        def rows(value):
            result = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            if result.numel() == 1:
                result = result.expand(32)
            if result.numel() != 32 or not torch.isfinite(result).all():
                raise ValueError("Expected finite scalar or 32 slot parameters")
            return result.clone()

        k, p, temp = rows(top_k), rows(top_p), rows(temperature)
        k[temp == 0] = 1
        if ((k < 1) | (k > 64) | (k != k.floor()) | (p < 0) | (p > 1) | (temp < 0)).any():
            raise ValueError("Wide sampling requires integer k in 1..64, p in [0,1], temperature >=0")
        inv_temp = torch.where(temp == 0, 1.0, temp.reciprocal())
        for target, value in ((self.k, k), (self.p, p), (self.inv_temperature, inv_temp)):
            self._write(target, value.reshape(1, 1, 32, 1).expand(1, 1, 32, 64))
        self._write(self.wide_mask, (k > 32).float().reshape(1, 32))

    def uniform(self, seeds):
        """Read persistent 32-row UINT32 seeds without updating them.

        lowbias32 mixer (0x7feb352d, 0x846ca68b); the uniform draw is its high 24 bits / 2**24.
        """
        x = ttnn.reshape(seeds, (1, 1, 32, 1))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.mul(ttnn.bitwise_xor(x, ttnn.bitwise_right_shift(x, 16)), self.multiplier1)
        x = ttnn.mul(ttnn.bitwise_xor(x, ttnn.bitwise_right_shift(x, 15)), self.multiplier2)
        x = ttnn.bitwise_xor(x, ttnn.bitwise_right_shift(x, 16))
        return ttnn.mul(ttnn.typecast(ttnn.bitwise_right_shift(x, 8), ttnn.float32), 2.0**-24)

    def candidates(self, logits):
        logits = self.common._mask_invalid_vocab_logits(logits)
        values, labels = ttnn.topk(
            logits, k=64, dim=-1, sorted=True, stable=False, sub_core_grids=self.common.sub_core_grid_topk
        )
        gather_args = dict(
            dim=3,
            cluster_axis=self.common._get_sampling_cluster_axis(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=self.common.num_gather_links,
        )
        values = self.common._perform_all_gather(values, buffer_key="NEMOTRON_WIDE_VALUES", **gather_args)
        labels = self.common._perform_all_gather(labels, buffer_key="NEMOTRON_WIDE_LABELS", **gather_args)
        labels = ttnn.add(ttnn.typecast(labels, ttnn.uint32), self.offsets, dtype=ttnn.uint32)
        return ttnn.topk(
            values,
            k=64,
            dim=-1,
            sorted=True,
            stable=False,
            indices_tensor=labels,
            sub_core_grids=self.common.sub_core_grid_topk,
        )

    def draw(self, values, labels, uniform, output):
        """Inverse CDF from sorted global64, with injectable device uniforms.

        Match upstream vLLM's ascending-CDF <= (1-p) removal: retain
        descending candidates with prior mass < p, always keeping rank zero.
        Draw at
        the first prefix strictly above u*mass. If FP32 rounds u*mass to mass,
        select the last positive candidate. At least one finite logit per row
        is a precondition shared with the model sampling path.
        """
        values = ttnn.mul(ttnn.typecast(values, ttnn.float32), self.inv_temperature)
        values = ttnn.where(ttnn.lt(self.ranks, self.k), values, float("-inf"))
        # Explicit FP32 exp and scans avoid BF16 accumulation in probabilities.
        largest = ttnn.slice(values, (0, 0, 0, 0), (1, 1, 32, 1))
        weights = ttnn.exp(ttnn.subtract(values, largest), fast_and_approximate_mode=False)
        prefix = ttnn.cumsum(weights, dim=-1, dtype=ttnn.float32)
        mass = ttnn.slice(prefix, (0, 0, 0, 63), (1, 1, 32, 64))
        prior = ttnn.concat((self.zero_column, ttnn.slice(prefix, (0, 0, 0, 0), (1, 1, 32, 63))), dim=-1)
        keep = ttnn.logical_or(
            ttnn.ge(self.p, 1.0),
            ttnn.logical_or(ttnn.eq(self.ranks, 0.0), ttnn.lt(prior, ttnn.mul(self.p, mass))),
        )
        kept = ttnn.where(keep, weights, 0.0)
        cdf = ttnn.cumsum(kept, dim=-1, dtype=ttnn.float32)
        mass = ttnn.slice(cdf, (0, 0, 0, 63), (1, 1, 32, 64))
        hits = ttnn.gt(cdf, ttnn.mul(uniform, mass))
        selected = ttnn.min(ttnn.where(hits, self.ranks, 64.0), dim=-1, keepdim=True)
        last_positive = ttnn.max(ttnn.where(ttnn.gt(kept, 0.0), self.ranks, 0.0), dim=-1, keepdim=True)
        selected = ttnn.typecast(ttnn.minimum(selected, last_positive), ttnn.uint32)
        tokens = ttnn.gather(labels, dim=-1, index=selected)
        tokens = ttnn.reshape(ttnn.to_layout(tokens, ttnn.ROW_MAJOR_LAYOUT), output.shape)
        ttnn.copy(tokens, output)
        return output

    def sample(self, logits, seeds, output, *, uniform=None):
        values, labels = self.candidates(logits)
        return self.draw(values, labels, self.uniform(seeds) if uniform is None else uniform, output)
