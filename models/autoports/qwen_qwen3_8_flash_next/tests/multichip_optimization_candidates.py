# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit candidate policies for the unchanged multichip evidence harness.

Policies are setup-time experiment choices, never runtime fallbacks. Invoke
with --candidate NAME followed by the ordinary probe arguments.
"""
import argparse
import sys

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as m


def apply(candidate):
    if "+" in candidate:
        for policy in candidate.split("+"):
            apply(policy)
        return
    if candidate in ("parallel-indexer", "parallel-indexer-rows"):
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder, ordered_projection
        from models.autoports.qwen_qwen3_8_flash_next.tt.parallel_ordered_projection import (
            parallel_ordered_indexer_projection,
        )

        selected = parallel_ordered_indexer_projection
        if candidate == "parallel-indexer-rows":
            # MeshQSA's existing FusedOrderedIndexerProjection explicitly
            # passes False for both flags. Override those caller values so
            # this candidate actually exercises the optimized row/copy path.
            def selected(hidden_states, transposed_weight, **kwargs):
                kwargs["valid_row_compute"] = True
                kwargs["copy_once"] = True
                return parallel_ordered_indexer_projection(hidden_states, transposed_weight, **kwargs)

        ordered_projection.ordered_indexer_projection = selected
        fused_decoder.fused_ordered_indexer_projection = selected
        return
    if candidate.startswith("lookup-"):
        from models.autoports.qwen_qwen3_8_flash_next.tt import striped_gather
        from models.autoports.qwen_qwen3_8_flash_next.tt.pipelined_striped_gather import kernel_source

        batch = int(candidate.split("-")[1])
        striped_gather.kernel_source = lambda: kernel_source(batch)
        return
    if candidate.startswith("attention-"):
        original = m.MeshQSA.build.__func__
        fidelity = candidate.split("-")[1]
        dtype = candidate.split("-")[2] if len(candidate.split("-")) > 2 else "bfloat8_b"

        def build(cls, loader, config, ccl, workspace):
            import torch

            result = original(cls, loader, config, ccl, workspace)
            a = result.attention
            a.compute = ttnn.init_device_compute_kernel_config(
                loader.mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
                math_approx_mode=False,
            )
            if dtype != "bfloat8_b":
                w = loader.state_dict
                q, k, v = (
                    w["self_attn.q_proj.weight"].chunk(4),
                    w["self_attn.k_proj.weight"].chunk(2),
                    w["self_attn.v_proj.weight"].chunk(2),
                )
                result.packed_qkv = loader.shards(
                    [torch.cat([q[r], k[r // 2], v[r // 2]], dim=0).T.reshape(1, 1, 2560, 3584) for r in range(4)],
                    dim=3,
                    dtype=getattr(ttnn, dtype),
                )
                a.out = loader.shards(
                    [part.T.reshape(1, 1, 1536, 2560) for part in w["self_attn.o_proj.weight"].chunk(4, dim=1)],
                    dim=2,
                    dtype=getattr(ttnn, dtype),
                )
            return result

        m.MeshQSA.build = classmethod(build)
        return
    if candidate.startswith("ccl8-"):
        original_sum = m.Collectives.sum
        original_call = m.MeshMoe.__call__ if candidate == "ccl8-moe" else m.MeshQSA.forward_chunk

        def total(self, x):
            if getattr(self, "quantized_sum", False):
                return ttnn.sum(
                    ttnn.typecast(self.gather(ttnn.typecast(x, ttnn.bfloat8_b), dim=1), ttnn.float32),
                    dim=1,
                    keepdim=True,
                )
            return original_sum(self, x)

        def call(self, *args, **kwargs):
            self.ccl.quantized_sum = True
            try:
                return original_call(self, *args, **kwargs)
            finally:
                self.ccl.quantized_sum = False

        m.Collectives.sum = total
        if candidate == "ccl8-moe":
            m.MeshMoe.__call__ = call
        else:
            m.MeshQSA.forward_chunk = call
        return
    if candidate.startswith("async"):
        original = m.Collectives.__init__

        def setup(self, device):
            original(self, device)
            grid = device.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
            self.buffers = {}
            self.semaphores = [
                ttnn.create_global_semaphore(device, cores, 0, ttnn.BufferType.L1_SMALL) for _ in range(3)
            ]

        def gather(self, value, dim=-1):
            dtype = value.dtype
            if candidate == "async-bfp8" and dtype == ttnn.bfloat16:
                value = ttnn.typecast(value, ttnn.bfloat8_b)
            buffer = None
            if candidate == "async-persistent" and value.shape[-2] == 1:
                key = (tuple(value.shape), value.dtype, dim % len(value.shape))
                if key not in self.buffers:
                    shape = list(value.shape)
                    shape[key[2]] *= 4
                    self.buffers[key] = ttnn.empty(
                        shape, dtype=value.dtype, layout=value.layout, device=self.device, memory_config=m.DRAM
                    )
                buffer = self.buffers[key]
            result = ttnn.experimental.all_gather_async(
                value,
                persistent_output_buffer=buffer,
                dim=dim % len(value.shape),
                multi_device_global_semaphore=self.semaphores[:2],
                barrier_semaphore=self.semaphores[2],
                num_links=2,
                topology=ttnn.Topology.Ring,
                memory_config=m.DRAM,
            )
            return ttnn.typecast(result, dtype) if result.dtype != dtype else result

        m.Collectives.__init__, m.Collectives.gather = setup, gather
        return
    if candidate.startswith("shared"):
        original = m.MeshMoe.__init__
        packed = "packed" in candidate
        fidelity = candidate.split("-")[-1]

        def setup(self, loader, ccl, expert_dtype, workspace):
            import torch

            original(self, loader, ccl, expert_dtype, workspace)
            self.shared_compute = ttnn.init_device_compute_kernel_config(
                loader.mesh_device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            if hasattr(self, "dram_shared"):
                del self.dram_shared
            self.shared = [
                loader.shards(
                    [
                        part.T
                        for part in loader.state_dict["mlp.shared_expert." + role + "_proj.weight"].chunk(4, dim=0)
                    ],
                    dim=1,
                    dtype=ttnn.bfloat8_b,
                )
                for role in ("gate", "up")
            ]
            if packed:
                parts = []
                for rank in range(4):
                    parts.append(
                        torch.cat(
                            [
                                loader.state_dict["mlp.shared_expert." + role + "_proj.weight"].chunk(4, dim=0)[rank]
                                for role in ("gate", "up")
                            ],
                            dim=0,
                        ).T
                    )
                self.shared_packed = loader.shards(parts, dim=1, dtype=ttnn.bfloat8_b)
                del self.shared

        def forward(self, x):
            if packed:
                projected = m.linear(x, self.shared_packed, self.shared_compute)
                gate, up = projected[..., :160], projected[..., 160:]
            else:
                gate, up = [m.linear(x, w, self.shared_compute) for w in self.shared]
            middle = ttnn.multiply(ttnn.silu(gate), up)
            shared = m.linear(middle, self.shared_down, self.shared_compute)
            return ttnn.multiply(shared, ttnn.sigmoid(m.linear(x, self.shared_route, self.shared_compute)))

        m.MeshMoe.__init__, m.MeshMoe.shared_forward = setup, forward
        return
    if candidate.startswith("inject"):
        splits = int(candidate.split("-")[1])
        original = m.MeshResidual.__init__

        class Partitioned(m.FusedPreciseProjection):
            def __init__(self, loader, host):
                self.compute = loader.compute
                self.parts = splits
                self.k = 10240 // splits
                self.weight = loader.shards(
                    [
                        __import__("torch").nn.functional.pad(p.T, (0, 31)).reshape(1, splits, self.k, 32)
                        for p in host.chunk(4, dim=0)
                    ],
                    dim=3,
                )

            def __call__(self, x):
                n = x.shape[-2]
                reshaped = ttnn.permute(ttnn.reshape(x, (1, n, self.parts, self.k)), (0, 2, 1, 3))
                partials = ttnn.matmul(
                    reshaped, self.weight, dtype=ttnn.float32, compute_kernel_config=self.compute, memory_config=m.DRAM
                )
                return ttnn.typecast(ttnn.sum(partials, dim=1, keepdim=True), ttnn.bfloat16)

        def setup(self, loader, prefix, config, ccl):
            original(self, loader, prefix, config, ccl)
            self.inject = Partitioned(loader, loader.state_dict[prefix + "block_inject_weight.weight"])

        m.MeshResidual.__init__ = setup
        return
    raise ValueError(candidate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    args, rest = parser.parse_known_args()
    apply(args.candidate)
    sys.argv = [sys.argv[0], *rest]
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_decoder import main as probe

    probe()


if __name__ == "__main__":
    main()
