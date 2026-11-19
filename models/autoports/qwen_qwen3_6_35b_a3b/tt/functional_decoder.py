# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B decoder; setup owns all host tensor conversion.

Inputs/outputs are device tensors [1, S, 2048]. Each request owns its recurrent
state or paged KV pool. Runtime methods perform only TTNN tensor operations.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

DRAM = ttnn.DRAM_MEMORY_CONFIG


def reshape(x, shape):
    """Repack logical rows explicitly across padded tile boundaries."""
    return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT)


class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        import torch

        self = cls()
        self.config, self.mesh_device = hf_config, mesh_device
        self.layer_idx, self.kind = layer_idx, hf_config.layer_types[layer_idx]
        expected = dict(
            hidden_size=2048,
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        for name, value in expected.items():
            if getattr(hf_config, name) != value:
                raise ValueError(f"target config requires {name}={value}")
        if self.kind not in ("linear_attention", "full_attention"):
            raise ValueError(f"unsupported layer kind {self.kind}")
        if (
            hf_config.rope_parameters["rope_type"] != "default"
            or hf_config.rope_parameters["partial_rotary_factor"] != 0.25
        ):
            raise ValueError("target text contract requires default 64-dimensional partial RoPE")
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.w = {}

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
            )

        for name, value in state_dict.items():
            if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                continue
            if name.endswith("conv1d.weight"):
                self.taps = [upload(value[:, 0, j].reshape(1, 1, -1)) for j in range(4)]
            elif name.endswith("A_log"):
                self.w[name] = upload(-value.float().exp().reshape(1, 1, -1), ttnn.float32)
            elif name.endswith("dt_bias"):
                self.w[name] = upload(value.reshape(1, 1, -1), ttnn.float32)
            elif name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                v = value.float() if name == "linear_attn.norm.weight" else value.float() + 1
                self.w[name] = upload(v.reshape(1, 1, -1), ttnn.float32)
            else:
                self.w[name] = upload(value.T, ttnn.float32 if name == "mlp.gate.weight" else ttnn.bfloat16)
        gate, up = state_dict["mlp.experts.gate_up_proj"].chunk(2, dim=1)
        self.eg = upload(gate.transpose(-1, -2).unsqueeze(0))
        self.eu = upload(up.transpose(-1, -2).unsqueeze(0))
        self.ed = upload(state_dict["mlp.experts.down_proj"].transpose(-1, -2).unsqueeze(0))
        eye, ones = torch.eye(32), torch.ones(32, 32)
        i, j = torch.arange(32)[:, None], torch.arange(32)[None, :]
        masks = torch.cat(
            [((i < 16) & (j < 16)).float(), ((i >= 16) & (j >= 16)).float(), ((i >= 16) & (j < 16)).float()], dim=1
        )
        self.constants = dict(
            zip(
                ("eye", "tril", "ones", "masks"),
                [upload(t.reshape(1, 1, *t.shape), ttnn.float32) for t in (eye, ones.tril(), ones, masks)],
            )
        )
        return self

    def linear(self, x, key, dtype=ttnn.bfloat16):
        return ttnn.linear(x, self.w[key], dtype=dtype, compute_kernel_config=self.compute, memory_config=DRAM)

    def norm(self, x, key):
        return ttnn.typecast(
            ttnn.rms_norm(
                ttnn.typecast(x, ttnn.float32),
                weight=self.w[key],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.compute,
            ),
            ttnn.bfloat16,
        )

    def new_recurrent_state(self):
        """Setup boundary, called before capture."""
        import torch

        def zeros(shape, dtype):
            return ttnn.from_torch(
                torch.zeros(shape), device=self.mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
            )

        return {"conv": zeros((1, 3, 8192), ttnn.bfloat16), "recurrent": zeros((1, 32, 128, 128), ttnn.float32)}

    def _linear_attention(self, x, state, *, decode):
        s = x.shape[1]
        qkv = self.linear(x, "linear_attn.in_proj_qkv.weight")
        history = ttnn.concat([state["conv"], qkv], dim=1)
        conv = ttnn.multiply(history[:, 0:s, :], self.taps[0])
        for j in range(1, 4):
            conv = ttnn.add(conv, ttnn.multiply(history[:, j : j + s, :], self.taps[j]))
        ttnn.copy(history[:, s : s + 3, :], state["conv"])
        conv = ttnn.silu(conv)
        q = reshape(conv[:, :, :2048], (1, s, 16, 128))
        k = reshape(conv[:, :, 2048:4096], (1, s, 16, 128))
        v = reshape(conv[:, :, 4096:], (1, s, 32, 128))

        # HF l2norm uses sum + epsilon, rather than mean + epsilon.
        def l2(a):
            a = ttnn.typecast(a, ttnn.float32)
            return ttnn.multiply(a, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.square(a), dim=-1, keepdim=True), 1e-6)))

        q, k = l2(q), l2(k)
        a = self.linear(x, "linear_attn.in_proj_a.weight", ttnn.float32)
        b = self.linear(x, "linear_attn.in_proj_b.weight", ttnn.float32)
        g = ttnn.multiply(self.w["linear_attn.A_log"], ttnn.softplus(ttnn.add(a, self.w["linear_attn.dt_bias"])))
        beta = ttnn.sigmoid(b)
        if decode:
            o, new = ttnn.transformer.fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=state["recurrent"],
                output_final_state=True,
                compute_kernel_config=self.compute,
            )
        else:
            o, new = ttnn.transformer.chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=state["recurrent"],
                output_final_state=True,
                chunk_size=32,
                compute_kernel_config=self.compute,
                **self.constants,
            )
        ttnn.copy(reshape(new, (1, 32, 128, 128)), state["recurrent"])
        o = reshape(o, (1, s * 32, 128))
        z = reshape(self.linear(x, "linear_attn.in_proj_z.weight"), (1, s * 32, 128))
        o = ttnn.multiply(self.norm(o, "linear_attn.norm.weight"), ttnn.silu(z))
        return self.linear(reshape(o, (1, s, 4096)), "linear_attn.out_proj.weight")

    def _moe(self, x):
        s = x.shape[1]
        logits = reshape(self.linear(x, "mlp.gate.weight", ttnn.float32), (s, 256))
        values, indices = ttnn.topk(logits, k=8, dim=-1, sorted=True)
        scores = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        routing = ttnn.scatter(
            ttnn.zeros_like(ttnn.typecast(logits, ttnn.bfloat16)),
            dim=1,
            index=indices,
            src=ttnn.typecast(scores, ttnn.bfloat16),
        )
        xx = reshape(x, (1, 1, s, 2048))

        def mm(a, w):
            return ttnn.matmul(a, w, compute_kernel_config=self.compute, memory_config=DRAM)

        gate, up = mm(xx, self.eg), mm(xx, self.eu)
        down = mm(ttnn.multiply(ttnn.silu(gate), up), self.ed)
        routing = reshape(ttnn.transpose(routing, 0, 1), (1, 256, s, 1))
        routed = reshape(ttnn.sum(ttnn.multiply(down, routing), dim=1), (1, s, 2048))
        shared = ttnn.multiply(
            ttnn.silu(self.linear(x, "mlp.shared_expert.gate_proj.weight")),
            self.linear(x, "mlp.shared_expert.up_proj.weight"),
        )
        shared = self.linear(shared, "mlp.shared_expert.down_proj.weight")
        shared = ttnn.multiply(shared, ttnn.sigmoid(self.linear(x, "mlp.shared_expert_gate.weight")))
        return ttnn.add(routed, shared)

    def new_kv_state(self, *, context_length=262144, page_order=None, num_physical_pages=None, kv_cache=None):
        """Allocate finite paged storage. Page size is 32; order names owned pages.

        State allocation and page-table upload are setup, before any live trace.
        kv_cache can share a physical pool across rows with disjoint owned pages.
        The table is padded to eight entries by aliasing its final owned page.
        These entries are read-only causal padding, never writable logical pages.
        """
        import torch

        if not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("context_length is outside the HF contract")
        blocks = (context_length + 31) // 32
        padded_blocks = (blocks + 7) // 8 * 8
        pages = torch.arange(blocks, dtype=torch.int32) if page_order is None else page_order.to(torch.int32)
        physical = padded_blocks if num_physical_pages is None else num_physical_pages
        if physical < padded_blocks:
            raise ValueError(
                "physical pool must contain at least round_up(logical_pages, 8) pages for TTNN table validation"
            )
        if pages.shape != (blocks,) or len(pages.unique()) != blocks or pages.min() < 0 or pages.max() >= physical:
            raise ValueError("page_order must name distinct owned pages within the physical pool")
        pages = torch.cat((pages, pages[-1:].expand(padded_blocks - blocks)))
        table = ttnn.from_torch(pages.reshape(1, -1), dtype=ttnn.int32, device=self.mesh_device)
        shape = (physical, 2, 32, 256)
        if kv_cache is None:
            kv_cache = tuple(
                ttnn.zeros(shape, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                for _ in range(2)
            )
        if len(kv_cache) != 2 or any(
            tuple(t.shape) != shape or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT for t in kv_cache
        ):
            raise ValueError("shared KV pool must contain two BF16 tiled tensors with the declared geometry")
        return {"k": kv_cache[0], "v": kv_cache[1], "page_table": table, "context_length": context_length}

    def prepare_positions(self, *, start_pos, seq_len):
        """Host setup boundary: absolute positions and text-only partial RoPE.

        Caller refreshes these same device tensors before trace replay. For text,
        the three HF mRoPE axes have identical positions.
        """
        import torch

        if not 0 <= start_pos < start_pos + seq_len <= self.config.max_position_embeddings:
            raise ValueError("positions are outside the HF contract")
        inv = 1.0 / (self.config.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
        angles = torch.arange(start_pos, start_pos + seq_len).float()[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), dim=-1).reshape(1, seq_len, 1, 64)

        def upload(x, dtype, layout):
            return ttnn.from_torch(x, device=self.mesh_device, dtype=dtype, layout=layout)

        return {
            "cos": upload(angles.cos(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "sin": upload(angles.sin(), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "current_pos": upload(torch.tensor([start_pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        }

    def _full_attention(self, x, state, positions, *, decode, start_pos):
        n = x.shape[1]
        qg = reshape(self.linear(x, "self_attn.q_proj.weight"), (1, n, 16, 512))
        q, gate = qg[:, :, :, :256], qg[:, :, :, 256:]
        k = reshape(self.linear(x, "self_attn.k_proj.weight"), (1, n, 2, 256))
        v = reshape(self.linear(x, "self_attn.v_proj.weight"), (1, n, 2, 256))
        q, k = self.norm(q, "self_attn.q_norm.weight"), self.norm(k, "self_attn.k_norm.weight")

        def rope(a):
            rotated = ttnn.concat([ttnn.neg(a[:, :, :, 32:64]), a[:, :, :, :32]], dim=-1)
            r = ttnn.add(ttnn.multiply(a[:, :, :, :64], positions["cos"]), ttnn.multiply(rotated, positions["sin"]))
            return ttnn.concat([r, a[:, :, :, 64:]], dim=-1)

        q, k = rope(q), rope(k)
        if decode:
            shard = ttnn.create_sharded_memory_config(
                (32, 256),
                ttnn.CoreGrid(y=1, x=1),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            k, v = [ttnn.to_memory_config(a, shard) for a in (k, v)]
            for dest, src in ((state["k"], k), (state["v"], v)):
                ttnn.experimental.paged_update_cache(
                    dest, src, update_idxs_tensor=positions["current_pos"], page_table=state["page_table"]
                )
            q = ttnn.to_memory_config(q, shard)
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                state["k"],
                state["v"],
                cur_pos_tensor=positions["current_pos"],
                page_table_tensor=state["page_table"],
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    # Native decode spills running statistics to BF16. Split the
                    # context across cores to avoid thousands of serial rounded
                    # updates at the advertised context (see real_tail_probe).
                    # The sole request's Q owner remains core (0, 0).
                    compute_with_storage_grid_size=(8, 8),
                    q_chunk_size=32,
                    k_chunk_size=256,
                ),
                compute_kernel_config=self.compute,
                memory_config=DRAM,
            )
        else:
            q, k, v = [ttnn.permute(a, (0, 2, 1, 3)) for a in (q, k, v)]
            pages = state["page_table"][:, start_pos // 32 :]
            for dest, src in ((state["k"], k), (state["v"], v)):
                ttnn.experimental.paged_fill_cache(dest, src, pages, batch_idx=0)
            o = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                state["k"],
                state["v"],
                state["page_table"],
                chunk_start_idx=start_pos,
                scale=1 / 16,
                program_config=ttnn.SDPAProgramConfig(
                    # Bound BF16 numerator accumulation depth at long contexts.
                    compute_with_storage_grid_size=(8, 8),
                    q_chunk_size=32,
                    k_chunk_size=256,
                ),
                compute_kernel_config=self.compute,
            )
            o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.multiply(ttnn.to_memory_config(o, DRAM), ttnn.sigmoid(gate))
        return self.linear(reshape(o, (1, n, 4096)), "self_attn.o_proj.weight")

    def _forward(self, hidden_states, *, state, decode, positions=None, start_pos=0):
        x = self.norm(hidden_states, "input_layernorm.weight")
        if self.kind == "linear_attention":
            x = self._linear_attention(x, state, decode=decode)
        else:
            x = self._full_attention(x, state, positions, decode=decode, start_pos=start_pos)
        x = ttnn.add(hidden_states, x)
        return ttnn.add(x, self._moe(self.norm(x, "post_attention_layernorm.weight")))

    def prepare_prefill(self, *, seq_len, start_pos=0, chunk_size=256):
        """Setup a bounded host plan; boundary tokens use the exact decode kernel.

        Arbitrary logical starts/lengths are accepted. Aligned interiors use chunks
        of at most chunk_size. A tail never writes padded cache or recurrent state.
        Prepare plans and warm all their kernels before recording a live trace.
        """
        if chunk_size < 32 or chunk_size % 32:
            raise ValueError("physical chunk size must be a positive multiple of 32")
        if not 0 <= start_pos < start_pos + seq_len <= self.config.max_position_embeddings:
            raise ValueError("logical sequence is outside HF context")
        plan, offset = [], 0
        while offset < seq_len:
            absolute = start_pos + offset
            remaining = seq_len - offset
            alignment = 256 if self.kind == "full_attention" else 32
            physical_chunk = min(chunk_size, 256) if self.kind == "full_attention" else chunk_size
            n = min(physical_chunk, remaining // 32 * 32) if absolute % alignment == 0 and remaining >= 32 else 1
            positions = self.prepare_positions(start_pos=absolute, seq_len=n) if self.kind == "full_attention" else None
            plan.append((offset, n, absolute, positions))
            offset += n
        return plan

    def prefill_forward(self, hidden_states, *, state, plan):
        """Run one row [1,S,2048], preserving all logical outputs and prior state.

        Plan is produced by prepare_prefill outside the measured pass.
        For heterogeneous rows use prefill_batch.
        """
        if plan[-1][0] + plan[-1][1] != hidden_states.shape[1]:
            raise ValueError("plan length does not match input")
        if self.kind == "full_attention" and plan[-1][2] + plan[-1][1] > state["context_length"]:
            raise ValueError("prefill exceeds the allocated context")
        outputs = []
        for offset, n, absolute, positions in plan:
            outputs.append(
                self._forward(
                    hidden_states[:, offset : offset + n, :],
                    state=state,
                    decode=n == 1,
                    positions=positions,
                    start_pos=absolute,
                )
            )
        return ttnn.concat(outputs, dim=1) if len(outputs) > 1 else outputs[0]

    def prefill_batch(self, hidden_states, *, states, plans):
        """Heterogeneous rows; row-owned state and plans, shared layer weights."""
        if not len(hidden_states) == len(states) == len(plans):
            raise ValueError("row counts differ")
        return tuple(self.prefill_forward(x, state=s, plan=p) for x, s, p in zip(hidden_states, states, plans))

    def decode_forward(self, hidden_states, *, state, positions=None):
        """Single-row decode [1,1,2048]; tensor current_pos and stable state.

        Every operation is on device and this entire call can be captured.
        """
        if tuple(hidden_states.shape) != (1, 1, 2048):
            raise ValueError("decode row must have shape [1, 1, 2048]; use decode_batch for several rows")
        return self._forward(hidden_states, state=state, decode=True, positions=positions)

    def decode_batch(self, hidden_states, *, states, positions):
        """Row-independent device decode; the complete batch loop is traceable."""
        if not len(hidden_states) == len(states) == len(positions):
            raise ValueError("row counts differ")
        return tuple(self.decode_forward(x, state=s, positions=p) for x, s, p in zip(hidden_states, states, positions))
