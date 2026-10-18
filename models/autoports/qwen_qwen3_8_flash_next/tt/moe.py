"""Top-10 routed SiLU experts plus a sigmoid-gated shared expert.

Sparse matmul skips experts unused by each physical group of 32 tokens. Routing
weights remain per-token; union sparsity never changes mixture semantics.
"""
import ttnn


class MoE:
    @classmethod
    def from_state_dict(cls, state, *, hf_config, mesh_device, expert_dtype=ttnn.bfloat8_b):
        import torch

        obj = cls()
        obj.hidden = hf_config.hidden_size
        obj.intermediate = hf_config.moe_intermediate_size
        obj.experts = hf_config.num_experts
        obj.topk = hf_config.num_experts_per_tok
        obj.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        obj.router = upload(state["mlp.gate.weight"].T)
        gate_up = state["mlp.experts.gate_up_proj"]
        obj.gate = upload(gate_up[:, : obj.intermediate].transpose(-2, -1).unsqueeze(0), expert_dtype)
        obj.up = upload(gate_up[:, obj.intermediate :].transpose(-2, -1).unsqueeze(0), expert_dtype)
        del gate_up
        obj.down = upload(state["mlp.experts.down_proj"].transpose(-2, -1).unsqueeze(0), expert_dtype)
        obj.shared = {name: upload(state[f"mlp.shared_expert.{name}_proj.weight"].T) for name in ("gate", "up", "down")}
        obj.shared_gate = upload(state["mlp.shared_expert_gate.weight"].T)
        mask = torch.full((1, 1, 1, 32), float("-inf"))
        mask[..., : obj.topk] = 0
        obj.top_mask = upload(mask)
        obj.programs = {}
        for width in (obj.intermediate, obj.hidden):
            # A simple legal geometry, not a performance sweep.
            ntiles = width // 32
            obj.programs[width] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=1,
                per_core_N=ntiles // 20,
                fuse_batch=False,
                mcast_in0=True,
            )
        return obj

    def _sparse(self, x, w, sparsity, width, *, input_sparse=False):
        return ttnn.sparse_matmul(
            x,
            w,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=input_sparse,
            is_input_b_sparse=True,
            program_config=self.programs[width],
            compute_kernel_config=self.compute,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _group(self, x):
        logits = ttnn.linear(x, self.router, compute_kernel_config=self.compute)
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        sparsity = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        gate = ttnn.reshape(
            self._sparse(x, self.gate, sparsity, self.intermediate), (1, self.experts, 32, self.intermediate)
        )
        up = ttnn.reshape(
            self._sparse(x, self.up, sparsity, self.intermediate), (1, self.experts, 32, self.intermediate)
        )
        hidden = ttnn.multiply(ttnn.silu(gate), up)
        down = self._sparse(hidden, self.down, sparsity, self.hidden, input_sparse=True)
        routing = ttnn.permute(routing, (0, 3, 2, 1))
        result = ttnn.sum(ttnn.multiply(down, routing), dim=1, keepdim=True)
        sg = ttnn.silu(ttnn.linear(x, self.shared["gate"], compute_kernel_config=self.compute))
        su = ttnn.linear(x, self.shared["up"], compute_kernel_config=self.compute)
        shared = ttnn.linear(ttnn.multiply(sg, su), self.shared["down"], compute_kernel_config=self.compute)
        scale = ttnn.sigmoid(ttnn.linear(x, self.shared_gate, compute_kernel_config=self.compute))
        return ttnn.add(result, ttnn.multiply(shared, scale))

    def __call__(self, x):
        # Caller owns padding/masking; keep batching explicit and preserve all logical rows.
        shape = tuple(x.shape)
        batch, seq = shape[1:3]
        outputs = []
        for b in range(batch):
            parts = []
            for start in range(0, seq, 32):
                valid = min(32, seq - start)
                group = x[:, b : b + 1, start : start + valid, :]
                if valid < 32:
                    group = ttnn.pad(group, [(0, 0), (0, 0), (0, 32 - valid), (0, 0)], 0)
                part = self._group(group)
                parts.append(part[:, :, :valid, :])
            outputs.append(ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0])
        return ttnn.concat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
