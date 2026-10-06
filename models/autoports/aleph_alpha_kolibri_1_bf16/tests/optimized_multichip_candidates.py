# SPDX-License-Identifier: Apache-2.0
"""Stage5 whole-layer topology probes on the real TP4 contract."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from . import multichip_checks
from .optimized_multichip_split import MeshSplitCandidate


class Candidate(MultichipDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        self.options = json.loads(os.environ.get("MC_OPTIONS", "{}"))
        if self.options.get("wo_ag"):
            weights = {k.removeprefix(f"model.layers.{self.layer_idx}."): v for k, v in state_dict.items()}
            w = weights["self_attn.o_proj.weight"].T
            self.wo_column = ttnn.from_torch(
                w[None, None].contiguous(),
                device=self.device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=3),
                dtype=getattr(ttnn, self.policy.attention_dtype),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.wo_gather = ttnn.empty(
                (1, 1, 1, 6144),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=getattr(ttnn, self.policy.ccl_dtype),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        return self

    def decode_forward(self, hidden_states, **kwargs):
        if self.options.get("sdpa_short_only") and kwargs["page_table"].shape[-1] * 32 > 8192:
            from dataclasses import replace

            original = self.policy
            self.policy = replace(original, decode_cores_per_head=64)
            try:
                return super().decode_forward(hidden_states, **kwargs)
            finally:
                self.policy = original
        return super().decode_forward(hidden_states, **kwargs)

    def _linear(self, x, w, *, dtype=ttnn.bfloat16, activation=None):
        info = self.projection_info.get(id(w))
        role = info["role"] if info else "router"
        if self.options.get("wo_ag") and role == "o_proj" and x.shape[-2] == 1:
            a = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            if a.dtype != getattr(ttnn, self.policy.ccl_dtype):
                a = ttnn.typecast(a, getattr(ttnn, self.policy.ccl_dtype))
            if self.options["wo_ag"] == "separate":
                gathered = self._gather(a)
                return ttnn.experimental.minimal_matmul(
                    gathered,
                    self.wo_column,
                    config=self._wo_config(),
                    dtype=dtype,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=info["compute"],
                )
            return ttnn.experimental.all_gather_minimal_matmul_async(
                a,
                self.wo_column,
                persistent_output_buffer=self.wo_gather,
                config=self._wo_config(),
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                topology=self.ccl.topology,
                cluster_axis=1,
                num_links=self.ccl.num_links,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=dtype,
                compute_kernel_config=info["compute"],
                force_transpose=True,
                num_workers_per_link=self.options.get("workers", 4),
            )[0]
        selected = self.options.get("activation")
        if x.shape[-2] <= 32 and (
            (selected == "attention" and role in ("qkv", "o_proj"))
            or (selected == "moe" and role in ("gate_up", "down_proj"))
        ):
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        return super()._linear(x, w, dtype=dtype, activation=activation)

    def _moe(self, x, *, prefill=False):
        width = self.options.get("topk_width")
        if not width or x.shape[-2] != 1:
            return super()._moe(x, prefill=prefill)
        original = ttnn.topk

        def padded(a, *args, **kwargs):
            a = ttnn.pad(a, ((0, 0), (0, 0), (0, 0), (0, width - a.shape[-1])), float("-inf"))
            return original(a, *args, **kwargs)

        ttnn.topk = padded
        try:
            return super()._moe(x, prefill=prefill)
        finally:
            ttnn.topk = original

    def _prefill_linear(self, x, w, dtype, compute):
        if self.options.get("prefill_minimal") and x.shape[-2] > 1024:
            opts = self.options["prefill_minimal"]
            kt = w.shape[-2] // 32
            k = max(v for v in range(1, min(kt, opts.get("k", 4)) + 1) if kt % v == 0)
            n = opts.get("n", 4)
            return ttnn.experimental.minimal_matmul(
                x,
                w,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=opts.get("m", 4),
                    K_block_size=k,
                    N_block_size=n,
                    subblock_h=1,
                    subblock_w=min(n, 4),
                    compute_with_storage_grid_size=tuple(opts.get("grid", [10, 8])),
                ),
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute,
            )
        return super()._prefill_linear(x, w, dtype, compute)

    def _grouped_prefill_moe(self, x, logits, ids):
        if not self.options.get("prefix_cores"):
            return super()._grouped_prefill_moe(x, logits, ids)
        cores = self.options["prefix_cores"]
        pn = 12 // cores
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cores, 1),
            in0_block_w=12,
            out_subblock_h=1,
            out_subblock_w=pn,
            per_core_M=1,
            per_core_N=pn,
            fuse_batch=False,
            mcast_in0=True,
        )
        original = ttnn.matmul

        def configured(a, w, **kwargs):
            if w is self.expert_prefix_sum:
                kwargs["program_config"] = program
            return original(a, w, **kwargs)

        ttnn.matmul = configured
        try:
            return super()._grouped_prefill_moe(x, logits, ids)
        finally:
            ttnn.matmul = original

    def _sparse_config(self, m, n):
        grid = self.options.get("sparse_down_grid" if n == 2560 else "sparse_gate_grid")
        if not grid:
            return super()._sparse_config(m, n)
        pn = (n // 32 + grid[0] * grid[1] - 1) // (grid[0] * grid[1])
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=tuple(grid),
            in0_block_w=self.policy.sparse_down_k if n == 2560 else self.policy.sparse_gate_k,
            out_subblock_h=1,
            out_subblock_w=min(8, pn),
            out_block_h=1,
            out_block_w=pn,
            per_core_M=(m + 31) // 32,
            per_core_N=pn,
            fuse_batch=False,
            mcast_in0=True,
        )

    def _wo_config(self):
        return ttnn.MinimalMatmulConfig(
            M_block_size=1,
            K_block_size=self.options.get("kblock", 4),
            N_block_size=self.options.get("nblock", 1),
            subblock_h=1,
            subblock_w=self.options.get("nblock", 1),
            compute_with_storage_grid_size=tuple(self.options.get("grid", [4, 4])),
        )

    def _indexed_moe(self, x, logits, ids):
        if self.options.get("activation") != "moe":
            return super()._indexed_moe(x, logits, ids)
        original = ttnn.sparse_matmul

        def reduced(a, *args, **kwargs):
            return original(ttnn.typecast(a, ttnn.bfloat8_b), *args, **kwargs)

        ttnn.sparse_matmul = reduced
        try:
            return super()._indexed_moe(x, logits, ids)
        finally:
            ttnn.sparse_matmul = original


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repetitions", type=int, default=100)
    a = p.parse_args()
    a.baseline = False
    multichip_checks.MultichipDecoder = MeshSplitCandidate if os.environ.get("MC_SPLIT") else Candidate
    multichip_checks.run(a)
    path = multichip_checks.OUT / f"{a.tag}_{a.layer}_{a.tokens}.json"
    data = json.loads(path.read_text())
    data["options"] = json.loads(os.environ.get("MC_OPTIONS", "{}"))
    data["split"] = json.loads(os.environ.get("MC_SPLIT", "{}"))
    data["source_files"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path(__file__).with_name("optimized_multichip_split.py"),
            multichip_checks.ROOT / "tt/multichip_decoder.py",
        ]
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
