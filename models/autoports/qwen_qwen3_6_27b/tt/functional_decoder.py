# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-device Qwen3.6 text decoder.

Weights and request buffers are prepared outside forward. Activations use
[batch, sequence, 5120]; each request owns its recurrent state or paged KV.
Text positions use the same position on all three MRoPE axes. Vision encoding
is outside this decoder's contract.
"""

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class RecurrentState:
    conv: ttnn.Tensor
    recurrent: ttnn.Tensor


@dataclass
class PagedState:
    key: ttnn.Tensor
    value: ttnn.Tensor
    page_table: ttnn.Tensor


class FunctionalDecoder(LightweightModule):
    """Real HF layer weights; both linear_attention and full_attention kinds.

    ``prefill_forward(x, *, state, ...)`` accepts device activations and mutates
    request state. ``decode_forward`` consumes one token per row. The caller
    owns state restoration and stable inputs when capturing decode.
    """

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        self = cls()
        self.config = getattr(hf_config, "text_config", hf_config)
        c = self.config
        assert (c.hidden_size, c.intermediate_size) == (5120, 17408)
        assert mesh_device.get_num_devices() == 1
        self.mesh = mesh_device
        self.kind = c.layer_types[layer_idx]
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.weights = {}
        folded_norms = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        )
        for name, value in state_dict.items():
            # The loading boundary accepts a layer-local HF state dict.
            value = value.detach().float()
            if name.endswith(".weight") and value.ndim == 2:
                value = value.T.contiguous()
            elif name in folded_norms:
                value = 1.0 + value
            if value.ndim == 1:
                value = value.reshape(1, 1, -1)
            if "conv1d" not in name:
                self.weights[name] = self._upload(
                    value,
                    dtype=ttnn.float32
                    if name in folded_norms or name.endswith(("A_log", "dt_bias"))
                    else ttnn.bfloat16,
                )
        if self.kind == "linear_attention":
            import torch

            assert (
                c.linear_num_key_heads,
                c.linear_num_value_heads,
                c.linear_key_head_dim,
                c.linear_value_head_dim,
                c.linear_conv_kernel_dim,
            ) == (16, 48, 128, 128, 4)
            self.eye = self._upload(torch.eye(32).reshape(1, 1, 32, 32), dtype=ttnn.float32)
            self.lower = self._upload(torch.ones(32, 32).tril().reshape(1, 1, 32, 32), dtype=ttnn.float32)
            self.strict_lower = self._upload(torch.ones(32, 32).tril(-1).reshape(1, 1, 32, 32), dtype=ttnn.float32)
            self.upper = self._upload(torch.ones(32, 32).triu().reshape(1, 1, 32, 32), dtype=ttnn.float32)
            conv = state_dict["linear_attn.conv1d.weight"].float()
            self.taps = [self._upload(conv[:, 0, k].reshape(1, 1, -1)) for k in range(4)]
            self.neg_a = self._upload(
                -state_dict["linear_attn.A_log"].float().exp().reshape(1, 1, -1), dtype=ttnn.float32
            )
        else:
            assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (24, 4, 256)
            # HF interleaves Q and output gate per head in q_proj.
            w = state_dict["self_attn.q_proj.weight"].reshape(24, 512, 5120)
            self.weights["query"] = self._upload(w[:, :256].reshape(6144, 5120).T.contiguous())
            self.weights["gate"] = self._upload(w[:, 256:].reshape(6144, 5120).T.contiguous())
        return self

    def _upload(self, value, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            value, device=self.mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def create_recurrent_state(self, batch=1):
        import torch

        return RecurrentState(
            self._upload(torch.zeros(batch, 3, 10240)),
            self._upload(torch.zeros(batch, 48, 128, 128), dtype=ttnn.float32),
        )

    def _linear(self, x, key):
        return ttnn.linear(x, self.weights[key], compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _norm(self, x, key):
        # HF accumulates normalization and applies (1+weight) in float32.
        y = ttnn.rms_norm(
            ttnn.typecast(x, ttnn.float32),
            weight=self.weights[key],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.ckc,
        )
        return ttnn.typecast(y, ttnn.bfloat16)

    def _finish(self, x, attention):
        x = ttnn.add(x, attention)
        h = self._norm(x, "post_attention_layernorm.weight")
        gate = ttnn.silu(self._linear(h, "mlp.gate_proj.weight"))
        up = self._linear(h, "mlp.up_proj.weight")
        return ttnn.add(x, self._linear(ttnn.multiply(gate, up), "mlp.down_proj.weight"))

    @staticmethod
    def _l2(x):
        x = ttnn.typecast(x, ttnn.float32)
        return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True), 1e-6)))

    def _delta_step(self, q, k, v, beta, g, state):
        b = q.shape[0]
        q = ttnn.repeat_interleave(self._l2(q), 3, dim=2)
        k = ttnn.repeat_interleave(self._l2(k), 3, dim=2)
        q = ttnn.permute(q, (0, 2, 3, 1))
        k = ttnn.permute(k, (0, 2, 3, 1))
        v = ttnn.permute(ttnn.typecast(v, ttnn.float32), (0, 2, 1, 3))
        beta = ttnn.reshape(ttnn.typecast(beta, ttnn.float32), (b, 48, 1, 1))
        g = ttnn.reshape(g, (b, 48, 1, 1))
        h = ttnn.multiply(state, ttnn.exp(g))
        prediction = ttnn.sum(ttnn.multiply(h, k), dim=-2, keepdim=True)
        delta = ttnn.multiply(ttnn.subtract(v, prediction), beta)
        h = ttnn.add(h, ttnn.multiply(k, delta))
        out = ttnn.sum(ttnn.multiply(h, ttnn.multiply(q, 128**-0.5)), dim=-2, keepdim=True)
        ttnn.copy(h, state)
        return ttnn.permute(out, (0, 2, 1, 3))

    def _gdn(self, x, state):
        b, s, _ = x.shape
        qkv = self._linear(x, "linear_attn.in_proj_qkv.weight")
        context = ttnn.concat([state.conv, qkv], dim=1)
        conv = ttnn.multiply(context[:, :s, :], self.taps[0])
        for j in range(1, 4):
            conv = ttnn.add(conv, ttnn.multiply(context[:, j : j + s, :], self.taps[j]))
        ttnn.copy(context[:, s : s + 3, :], state.conv)
        conv = ttnn.silu(conv)
        q = ttnn.reshape(conv[:, :, :2048], (b, s, 16, 128))
        k = ttnn.reshape(conv[:, :, 2048:4096], (b, s, 16, 128))
        v = ttnn.reshape(conv[:, :, 4096:], (b, s, 48, 128))
        beta = ttnn.sigmoid(self._linear(x, "linear_attn.in_proj_b.weight"))
        a = ttnn.typecast(self._linear(x, "linear_attn.in_proj_a.weight"), ttnn.float32)
        g = ttnn.multiply(self.neg_a, ttnn.softplus(ttnn.add(a, self.weights["linear_attn.dt_bias"])))
        if s == 1:
            out = self._delta_step(q, k, v, beta, g, state.recurrent)
        else:
            out = self._delta_chunks(q, k, v, beta, g, state.recurrent)
        out = ttnn.typecast(out, ttnn.bfloat16)
        out = ttnn.rms_norm(
            out,
            weight=self.weights["linear_attn.norm.weight"],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.ckc,
        )
        z = ttnn.reshape(self._linear(x, "linear_attn.in_proj_z.weight"), (b, s, 48, 128))
        out = ttnn.multiply(out, ttnn.silu(z))
        return self._linear(ttnn.reshape(out, (b, s, 6144)), "linear_attn.out_proj.weight")

    def decode_forward(self, x, *, state, current_pos=None, rope=None):
        """One device token per row. Inputs/state must remain stable across replay."""
        assert x.shape[1] == 1
        h = self._norm(x, "input_layernorm.weight")
        attention = (
            self._gdn(h, state)
            if self.kind == "linear_attention"
            else self._attention_decode(h, state, current_pos, rope)
        )
        return self._finish(x, attention)

    def prefill_forward(self, x, *, state, start_pos=0, rope=None):
        """Chunked device prefill with arbitrary logical lengths and starts.

        A list of tensors/states/starts/rope tuples describes independent rows
        with different lengths. Recurrent continuation requires the exact state
        snapshot at start_pos; full attention requires all earlier KV pages.
        The caller must not share writable request state between rows.
        """
        if isinstance(x, (list, tuple)):
            starts = [start_pos] * len(x) if isinstance(start_pos, int) else start_pos
            rotations = [None] * len(x) if rope is None else rope
            if not x or not (len(x) == len(state) == len(starts) == len(rotations)):
                raise ValueError("Each prefill row needs its own state, start and rotations")
            return [
                self.prefill_forward(t, state=s, start_pos=p, rope=r) for t, s, p, r in zip(x, state, starts, rotations)
            ]
        length = x.shape[1]
        if not (
            0 <= start_pos < self.config.max_position_embeddings
            and 1 <= length <= self.config.max_position_embeddings - start_pos
        ):
            raise ValueError("Prefill is outside the HF context contract")
        outputs = []
        offset = 0
        alignment = self._attention_chunk(state) if self.kind == "full_attention" else 1
        while offset < length:
            absolute = start_pos + offset
            # SDPA-block-unaligned continuation uses exact token updates until
            # aligned; no padded write can overwrite an existing prefix row.
            n = 1 if absolute % alignment else min(512, length - offset)
            chunk = x[:, offset : offset + n]
            h = self._norm(chunk, "input_layernorm.weight")
            if self.kind == "linear_attention":
                attention = self._gdn(h, state)
            else:
                rotations = tuple(t[:, :, offset : offset + n, :] for t in rope)
                if absolute % alignment:
                    # Creation.full uploads small int32 tensors from the host.
                    # moreh_full dispatches a device fill, including ROW_MAJOR.
                    # A full 32-int physical row also covers its aligned NoC
                    # write; the native fill's tiny-row CB is undersized.
                    position = ttnn.reshape(
                        ttnn.moreh_full((1, 32), absolute, self.mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)[
                            :, : x.shape[0]
                        ],
                        (x.shape[0],),
                    )
                    attention = self._attention_decode(h, state, position, rotations)
                else:
                    attention = self._attention_prefill(h, state, start_pos=absolute, rope=rotations)
            outputs.append(self._finish(chunk, attention))
            offset += n
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)

    def create_paged_state(self, page_table):
        """Setup: 32-token pages; append finite owned pages for 512-key SDPA blocks."""
        import torch

        assert page_table.ndim == 2 and 1 <= page_table.shape[0] <= 32
        blocks = int(page_table.max()) + 1
        if page_table.shape[1] >= 16:
            extra = (-page_table.shape[1]) % 16
            if extra:
                padding = torch.arange(blocks, blocks + page_table.shape[0] * extra, dtype=page_table.dtype)
                page_table = torch.cat((page_table, padding.reshape(page_table.shape[0], extra)), dim=1)
                blocks += padding.numel()
        shape = (blocks, 4, 32, 256)
        table = ttnn.from_torch(
            page_table.to(torch.int32), device=self.mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        return PagedState(
            self._upload(torch.zeros(shape, dtype=torch.bfloat16)),
            self._upload(torch.zeros(shape, dtype=torch.bfloat16)),
            table,
        )

    @staticmethod
    def _attention_chunk(state):
        # Native SDPA repacks running state at each key-block boundary. A
        # 512-key block bounds long-context rounding; FP32 dest alone does not.
        pages = state.page_table.shape[1]
        if pages < 16:
            return 32
        assert pages % 16 == 0, "Large page tables require finite padding to 512-token extent"
        return 512

    def _heads(self, x, count):
        b, s, _ = x.shape
        return ttnn.permute(ttnn.reshape(x, (b, s, count, 256)), (0, 2, 1, 3))

    @staticmethod
    def _rope(x, rope):
        cos, sin = rope
        rot, tail = x[:, :, :, :64], x[:, :, :, 64:]
        half = ttnn.concat([ttnn.neg(rot[:, :, :, 32:]), rot[:, :, :, :32]], dim=-1)
        rot = ttnn.add(ttnn.multiply(rot, cos), ttnn.multiply(half, sin))
        return ttnn.concat([rot, tail], dim=-1)

    def _qkv(self, x, rope):
        q = self._norm(self._heads(self._linear(x, "query"), 24), "self_attn.q_norm.weight")
        k = self._norm(self._heads(self._linear(x, "self_attn.k_proj.weight"), 4), "self_attn.k_norm.weight")
        v = self._heads(self._linear(x, "self_attn.v_proj.weight"), 4)
        return self._rope(q, rope), self._rope(k, rope), v

    def _attention_out(self, x, out):
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (x.shape[0], x.shape[1], 6144))
        out = ttnn.multiply(out, ttnn.sigmoid(self._linear(x, "gate")))
        return self._linear(out, "self_attn.o_proj.weight")

    def _attention_decode(self, x, state, current_pos, rope):
        b = x.shape[0]
        q, k, v = self._qkv(x, rope)
        # Cache update owns one core per request; SDPA reads interleaved Q.
        shard_grid = ttnn.num_cores_to_corerangeset(b, ttnn.CoreCoord(8, 4), True)
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard_grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
        )
        for tensor, cache in ((k, state.key), (v, state.value)):
            tensor = ttnn.permute(tensor, (2, 0, 1, 3))
            tensor = ttnn.to_memory_config(tensor, memory)
            ttnn.experimental.paged_update_cache(
                cache, tensor, update_idxs_tensor=current_pos, page_table=state.page_table
            )
        q = ttnn.permute(q, (2, 0, 1, 3))
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            state.key,
            state.value,
            page_table_tensor=state.page_table,
            cur_pos_tensor=current_pos,
            scale=256**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
                q_chunk_size=32,
                k_chunk_size=self._attention_chunk(state),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.ckc,
        )
        out = ttnn.permute(out[:, :, :24, :], (1, 2, 0, 3))
        return self._attention_out(x, out)

    def _attention_prefill(self, x, state, *, start_pos, rope):
        assert start_pos % self._attention_chunk(state) == 0
        q, k, v = self._qkv(x, rope)
        s = x.shape[1]
        padded = (s + 31) // 32 * 32
        if s != padded:
            q, k, v = (ttnn.pad(t, ((0, 0), (0, 0), (0, padded - s), (0, 0)), value=0.0) for t in (q, k, v))
        table = state.page_table[:, start_pos // 32 :]
        for row in range(x.shape[0]):
            ttnn.experimental.paged_fill_cache(state.key, k[row : row + 1], table, batch_idx=row)
            ttnn.experimental.paged_fill_cache(state.value, v[row : row + 1], table, batch_idx=row)
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            state.key,
            state.value,
            state.page_table,
            chunk_start_idx=start_pos,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
                q_chunk_size=32,
                k_chunk_size=self._attention_chunk(state),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.ckc,
        )
        return self._attention_out(x, out[:, :, :s, :])

    def _delta_chunks(self, q, k, v, beta, g, state):
        """Exact unit-diagonal delta-rule chunks; padding has beta=g=0.

        For a strictly lower triangular 32-row matrix N, N**32=0, so
        (I-N)^-1 = (I+N)(I+N**2)(I+N**4)(I+N**8)(I+N**16).
        No diagonal damping or state clipping changes the HF recurrence.
        """
        b, s, _, _ = q.shape
        q, k = (ttnn.permute(ttnn.repeat_interleave(self._l2(t), 3, dim=2), (0, 2, 1, 3)) for t in (q, k))
        v = ttnn.permute(ttnn.typecast(v, ttnn.float32), (0, 2, 1, 3))
        beta = ttnn.reshape(ttnn.permute(ttnn.typecast(beta, ttnn.float32), (0, 2, 1)), (b, 48, s, 1))
        g = ttnn.reshape(ttnn.permute(g, (0, 2, 1)), (b, 48, s, 1))
        padded = (s + 31) // 32 * 32
        if s != padded:
            q, k, v, beta, g = (
                ttnn.pad(t, ((0, 0), (0, 0), (0, padded - s), (0, 0)), value=0.0) for t in (q, k, v, beta, g)
            )

        def mm(a, c):
            return ttnn.matmul(a, c, compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        outputs = []
        h = state
        for start in range(0, padded, 32):
            qi, ki, vi, bi, gi = (t[:, :, start : start + 32, :] for t in (q, k, v, beta, g))
            qi = ttnn.multiply(qi, 128**-0.5)
            cumulative = ttnn.transpose(mm(ttnn.transpose(gi, -1, -2), self.upper), -1, -2)
            diff = ttnn.subtract(cumulative, ttnn.transpose(cumulative, -1, -2))
            decay_mask = ttnn.multiply(ttnn.exp(ttnn.multiply(diff, self.lower)), self.lower)
            kb = ttnn.multiply(ki, bi)
            n = ttnn.neg(
                ttnn.multiply(ttnn.multiply(mm(kb, ttnn.transpose(ki, -1, -2)), decay_mask), self.strict_lower)
            )
            inv = ttnn.add(self.eye, n)
            for _ in range(4):
                n = mm(n, n)
                inv = mm(inv, ttnn.add(self.eye, n))
            vb = mm(inv, ttnn.multiply(vi, bi))
            exponential = ttnn.exp(cumulative)
            kd = mm(inv, ttnn.multiply(kb, exponential))
            delta = ttnn.subtract(vb, mm(kd, h))
            scores = ttnn.multiply(mm(qi, ttnn.transpose(ki, -1, -2)), decay_mask)
            outputs.append(ttnn.add(mm(ttnn.multiply(qi, exponential), h), mm(scores, delta)))
            last = cumulative[:, :, 31:32, :]
            decayed_key = ttnn.multiply(ki, ttnn.exp(ttnn.subtract(last, cumulative)))
            h = ttnn.add(ttnn.multiply(h, ttnn.exp(last)), mm(ttnn.transpose(decayed_key, -1, -2), delta))
        ttnn.copy(h, state)
        output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        return ttnn.permute(output[:, :, :s, :], (0, 2, 1, 3))
