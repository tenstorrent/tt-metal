# SPDX-License-Identifier: Apache-2.0
"""Move decode top-k padding into immutable router weights and expert bias."""

import argparse
import os

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from . import multichip_checks


class PaddedRouter(MultichipDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        weights = {k.removeprefix(f"model.layers.{self.layer_idx}."): v for k, v in state_dict.items()}
        self.router_padded = ttnn.from_torch(
            torch.nn.functional.pad(weights["mlp.gate.weight"].T, (0, 640)).contiguous(),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=getattr(ttnn, self.policy.router_dtype),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias_padded = ttnn.from_torch(
            torch.nn.functional.pad(
                weights["moe.router.expert_bias"].reshape(1, 1, 1, 384), (0, 640), value=float("-inf")
            ),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cores = int(os.environ.get("MC_ROUTER_CORES", "32"))
        self.padded_router_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, cores // 8),
            in0_block_w=80,
            out_subblock_h=1,
            out_subblock_w=32 // cores,
            out_block_h=1,
            out_block_w=32 // cores,
            per_core_M=1,
            per_core_N=32 // cores,
            fuse_batch=False,
            mcast_in0=True,
        )
        return self

    def _moe(self, x, *, prefill=False):
        if x.shape[-2] != 1:
            return super()._moe(x, prefill=prefill)
        x = self._gather_input(x)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        logits = ttnn.linear(
            x,
            self.router_padded,
            dtype=ttnn.float32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self.padded_router_program,
            compute_kernel_config=self.router_compute,
        )
        choice = ttnn.typecast(ttnn.add(logits, self.bias_padded), ttnn.bfloat16)
        _, ids = ttnn.topk(choice, k=6, dim=-1)
        return self._indexed_moe(x, logits, ids)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repetitions", type=int, default=100)
    a = p.parse_args()
    a.baseline = False
    multichip_checks.MultichipDecoder = PaddedRouter
    multichip_checks.run(a)
