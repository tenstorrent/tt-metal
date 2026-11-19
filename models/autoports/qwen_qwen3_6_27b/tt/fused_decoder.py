# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused single-device Qwen3.6 text decoder.

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


class FusedDecoder(LightweightModule):
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
            packed = torch.cat(
                [
                    state_dict["linear_attn.in_proj_qkv.weight"],
                    state_dict["linear_attn.in_proj_z.weight"],
                    torch.nn.functional.pad(state_dict["linear_attn.in_proj_b.weight"], (0, 0, 0, 16)),
                    torch.nn.functional.pad(state_dict["linear_attn.in_proj_a.weight"], (0, 0, 0, 16)),
                ],
                dim=0,
            )
            self.weights["packed_gdn"] = self._upload(packed.T.contiguous())
            self.conv_start = ttnn.from_torch(
                torch.tensor([0], dtype=torch.int64), device=self.mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            exchange_core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
                self.mesh.compute_with_storage_grid_size()
            )
            exchange_grid = ttnn.CoreRangeSet({ttnn.CoreRange(exchange_core, exchange_core)})
            self.exchange = ttnn.from_torch(
                torch.zeros(1, 12, 10240), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh
            )
            self.conv_semaphores = [ttnn.create_global_semaphore(self.mesh, exchange_grid, 0) for _ in range(3)]
            self.conv_ends = {
                n: ttnn.from_torch(
                    torch.tensor([n], dtype=torch.int64),
                    device=self.mesh,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                for n in range(1, 513)
            }
            self.eye = self._upload(torch.eye(32).reshape(1, 1, 32, 32), dtype=ttnn.float32)
            self.lower = self._upload(torch.ones(32, 32).tril().reshape(1, 1, 32, 32), dtype=ttnn.float32)
            self.ones = self._upload(torch.ones(1, 1, 32, 32), dtype=ttnn.float32)
            ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
            lo_i, lo_j = ii < 16, jj < 16
            masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)
            self.masks = self._upload(masks.reshape(1, 1, 32, 96), dtype=ttnn.float32)
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
        if self.kind == "full_attention":
            import torch

            qw = state_dict["self_attn.q_proj.weight"].reshape(24, 512, 5120)[:, :256].reshape(6144, 5120)
            self.weights["packed_qkv"] = self._upload(
                torch.cat(
                    [qw, state_dict["self_attn.k_proj.weight"], state_dict["self_attn.v_proj.weight"]], dim=0
                ).T.contiguous()
            )
        obsolete = (
            [
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
            ]
            if self.kind == "linear_attention"
            else ["query", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]
        )
        for name in obsolete:
            self.weights.pop(name, None)
        return self

    def _upload(self, value, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            value, device=self.mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def create_recurrent_state(self, batch=1):
        import torch

        return RecurrentState(
            ttnn.from_torch(
                torch.zeros(batch, 3, 10240), device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            self._upload(torch.zeros(batch, 48, 128, 128), dtype=ttnn.float32),
        )

    def _linear(self, x, key):
        b, seq, width = x.shape
        if b > 1:
            x = ttnn.reshape(x, (1, b * seq, width))
        y = ttnn.linear(x, self.weights[key], compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(y, (b, seq, y.shape[-1])) if b > 1 else y

    def _norm(self, x, key):
        # Single-row RMSNorm shares width across eight cores; the two required
        # transfers also widen/narrow, replacing separate dtype conversions.
        if len(x.shape) == 3 and x.shape[0] == 1 and x.shape[1] == 1 and x.shape[-1] == 5120:
            grid = ttnn.CoreGrid(x=8, y=1)
            memory = ttnn.create_sharded_memory_config(
                (32, 640),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            y = ttnn.interleaved_to_sharded(x, memory, output_dtype=ttnn.float32)
            y = ttnn.rms_norm(
                y,
                weight=self.weights[key],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.ckc,
                program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(8, 1),
                    subblock_w=4,
                    block_h=1,
                    block_w=20,
                    inplace=False,
                ),
            )
            return ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        # Native BF16 intake keeps FP32 intermediates/affine under self.ckc,
        # so widening/narrowing is absorbed into the norm's unpack/pack.
        return ttnn.rms_norm(
            x,
            weight=self.weights[key],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.ckc,
        )

    def _finish(self, x, attention):
        x = ttnn.add(x, attention)
        h = self._norm(x, "post_attention_layernorm.weight")
        gate = self._linear(h, "mlp.gate_proj.weight")
        up = self._linear(h, "mlp.up_proj.weight")
        return ttnn.add(
            x,
            self._linear(
                ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), "mlp.down_proj.weight"
            ),
        )

    def _gdn(self, x, state):
        b, s, _ = x.shape
        packed = self._linear(x, "packed_gdn")
        qkv = packed[:, :, :10240]
        padded = (s + 31) // 32 * 32
        conv_input = qkv if s == padded else ttnn.pad(qkv, ((0, 0), (0, padded - s), (0, 0)), value=0.0)
        outputs = []
        for row in range(b):
            history = state.conv[row : row + 1]
            outputs.append(
                ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
                    conv_input[row : row + 1],
                    history,
                    *self.taps,
                    2048,
                    2048,
                    6144,
                    actual_start=self.conv_start,
                    actual_end=self.conv_ends[s],
                    exchange_buffer=self.exchange,
                    exchange_semaphores=self.conv_semaphores,
                    program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        packer_l1_acc=False,
                    ),
                )
            )
        q, k, v = (ttnn.concat([o[i] for o in outputs], dim=0) if b > 1 else outputs[0][i] for i in range(3))
        ttnn.copy(ttnn.concat([o[3] for o in outputs], dim=0) if b > 1 else outputs[0][3], state.conv)
        beta = ttnn.sigmoid(packed[:, :, 16384:16432])
        a = packed[:, :, 16448:16496]
        g = ttnn.multiply(
            self.neg_a,
            ttnn.add(a, self.weights["linear_attn.dt_bias"], dtype=ttnn.float32),
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        out = self._delta_chunks(q, k, v, beta, g, state.recurrent)
        out = ttnn.typecast(out, ttnn.bfloat16)
        out = ttnn.rms_norm(
            out,
            weight=self.weights["linear_attn.norm.weight"],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.ckc,
        )
        out = ttnn.transformer.concatenate_heads(out)
        out = ttnn.multiply(packed[:, :, 10240:16384], out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return self._linear(out, "linear_attn.out_proj.weight")

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

    def _rope(self, x, rope):
        cos, sin = rope
        rotated = [
            ttnn.experimental.rotary_embedding_hf(
                x[row : row + 1, :, :, :64],
                cos[row : row + 1],
                sin[row : row + 1],
                compute_kernel_config=self.ckc,
            )
            for row in range(x.shape[0])
        ]
        rot = rotated[0] if len(rotated) == 1 else ttnn.concat(rotated, dim=0)
        return ttnn.concat([rot, x[:, :, :, 64:]], dim=-1)

    def _qkv(self, x, rope):
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            self._linear(x, "packed_qkv"), num_heads=24, num_kv_heads=4, transpose_key=False
        )
        q = self._norm(q, "self_attn.q_norm.weight")
        k = self._norm(k, "self_attn.k_norm.weight")
        return self._rope(q, rope), self._rope(k, rope), v

    def _attention_out(self, x, out):
        out = ttnn.transformer.concatenate_heads(out)
        out = ttnn.multiply(self._linear(x, "gate"), out, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return self._linear(out, "self_attn.o_proj.weight")

    def _attention_decode(self, x, state, current_pos, rope):
        b = x.shape[0]
        q, k, v = self._qkv(x, rope)
        inputs = []
        for offset, tensor in ((0, k), (b, v)):
            cores = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(i % 8, i // 8), ttnn.CoreCoord(i % 8, i // 8))
                    for i in range(offset, offset + b)
                }
            )
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
            )
            inputs.append(ttnn.to_memory_config(ttnn.permute(tensor, (2, 0, 1, 3)), memory))
        ttnn.experimental.paged_fused_update_cache(
            state.key,
            inputs[0],
            state.value,
            inputs[1],
            update_idxs_tensor=current_pos,
            page_table=state.page_table,
        )
        # Update and attention share the ordered queue. Release the temporary
        # L1 inputs before SDPA reserves its large-context circular buffers.
        del inputs
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
        b, s = q.shape[0], beta.shape[1]
        padded = (s + 31) // 32 * 32
        if b == 1 and s == 1:
            # Same phased topology as the public op at this shape. The prep
            # reader masks logical gate tails, removing two explicit pads.
            # Larger batches retain the public wave-capable native path.
            native = ttnn._ttnn.operations.transformer
            prep = native.chunk_gdn_prep(
                q,
                k,
                v,
                g,
                beta,
                self.eye,
                self.lower,
                self.ones,
                self.masks,
                chunk_size=32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ckc,
                v_flat=True,
                HV=48,
                qk_norm=True,
                scale=128**-0.5,
                qk_flat=True,
                Hk=16,
                prep_serial=False,
                wy_inverse=ttnn.ChunkGdnWyInverse.AUTO,
            )
            out, updated = native.chunk_gdn_scan(
                *prep,
                initial_state=ttnn.reshape(state, (48, 128, 128)),
                chunk_size=32,
                output_final_state=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ckc,
                use_mcast=True,
                force_serial=False,
            )
            ttnn.copy(ttnn.reshape(updated, (1, 48, 128, 128)), state)
            return ttnn.reshape(out, ttnn.Shape((1, 48, 1, 128)), ttnn.Shape((1, 48, 32, 128)))
        if s != padded:
            beta, g = (ttnn.pad(t, ((0, 0), (0, padded - s), (0, 0)), value=0.0) for t in (beta, g))
        out, updated = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            output_final_state=True,
            chunk_size=32,
            num_key_heads=16,
            output_head_major=True,
            eye=self.eye,
            tril=self.lower,
            ones=self.ones,
            masks=self.masks,
            compute_kernel_config=self.ckc,
        )
        ttnn.copy(updated, state)
        return ttnn.reshape(out, ttnn.Shape((b, 48, s, 128)), ttnn.Shape((b, 48, padded, 128)))
