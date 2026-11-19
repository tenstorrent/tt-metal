# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch dense projections while retaining independent per-request attention state."""

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_optimization import Candidate


class BatchedCandidate(Candidate):
    input_l1 = True

    def linear(self, x, name):
        if getattr(self, "_attention_only", False):
            if name == "packed_input":
                return self._packed_row
            if name in ("linear_attn.out_proj", "self_attn.o_proj"):
                return x
        if getattr(self, "_grouped", False):
            im, om, cfg = self.dram_layouts[name]
            out = ttnn.linear(
                ttnn.to_memory_config(x, im),
                self.dram_weights[name],
                program_config=cfg,
                compute_kernel_config=self.matmul_compute(name),
                memory_config=om,
            )
            return ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG) if name == "packed_input" else out
        return super().linear(x, name)

    def reduce(self, x):
        if getattr(self, "_attention_only", False):
            return x
        if getattr(self, "_grouped", False):
            i = self.ar_index
            self.ar_index = 1 - i
            return ttnn.experimental.all_reduce_async(
                ttnn.to_memory_config(x, self.ar_mem),
                self.ar_buffers[i],
                cluster_axis=1,
                mesh_device=self.device,
                multi_device_global_semaphore=self.ar_semaphores[i],
                memory_config=self.ar_mem,
                dtype=ttnn.bfloat16,
                topology=self.ccl.topology,
                num_links=self.ccl.num_links,
            )
        return super().reduce(x)

    def norm(self, x, name):
        if getattr(self, "_grouped", False) and name in ("input_layernorm", "post_attention_layernorm"):
            return ttnn.rms_norm(
                ttnn.to_memory_config(x, self.ar_mem),
                weight=self.weights[name + ".weight"],
                epsilon=self.config.rms_norm_eps,
                program_config=self.norm_program,
                compute_kernel_config=self.compute,
                memory_config=self.ar_mem,
            )
        return super().norm(x, name)

    def finish(self, x, mixed):
        if not getattr(self, "_grouped", False):
            return super().finish(x, mixed)
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        gate, up = self.linear(h, "mlp.gate_proj"), self.linear(h, "mlp.up_proj")
        h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(residual, self.reduce(self.linear(h, "mlp.down_proj")))

    def decode_batch_packed(self, x, *, states, current_positions, page_tables):
        n = x.shape[2]
        assert 1 <= n <= 32 and n == len(states) == len(current_positions) == len(page_tables)
        assert self.native_local
        self._grouped = True
        try:
            x = ttnn.to_memory_config(x, self.ar_mem)
            h = self.norm(x, "input_layernorm")
            packed = self.linear(h, "packed_input")
            local = []
            self._attention_only = True
            try:
                for i, (st, pos, pt) in enumerate(zip(states, current_positions, page_tables)):
                    self._packed_row = packed[:, :, i : i + 1, :]
                    dummy = self._packed_row
                    local.append(
                        self.linear_attention(dummy, st, decode=True)
                        if self.kind == "linear_attention"
                        else self.full_attention(dummy, st, page_table=pt, position=pos, decode=True)
                    )
            finally:
                self._attention_only = False
                self._packed_row = None
            local = ttnn.concat(local, dim=2) if n > 1 else local[0]
            name = "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj"
            mixed = self.reduce(self.linear(local, name))
            return self.finish(x, mixed)
        finally:
            self._grouped = False

    def decode_forward(self, x, *, state, page_table=None, current_position=None):
        return ttnn.to_memory_config(
            self.decode_batch_packed(x, states=[state], current_positions=[current_position], page_tables=[page_table]),
            ttnn.DRAM_MEMORY_CONFIG,
        )
