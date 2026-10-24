# SPDX-License-Identifier: Apache-2.0
"""Fused single-device Qwen3.8 text decoder; weights and request state are explicit.

Runtime tensors stay on device. Setup and reference comparisons belong to callers.
Prefill accepts [B,S,5120], decode [B,1,5120]. Full-attention callers supply
RoPE cos/sin for the absolute positions and a paged cache; linear attention
callers supply convolution and FP32 recurrent state. State is updated in place.
Validation evidence and limitations are in doc/fused_decoder/README.md.
"""

from dataclasses import dataclass

import ttnn

from models.common.lightweightmodule import LightweightModule
from .fp32_recurrence import _KERNEL, _UPSTREAM


@dataclass
class LinearState:
    conv: object  # [B,3,10240], BF16 row-major
    recurrent: object  # [B,48,128,128], FP32


@dataclass
class PagedState:
    key: object  # [physical_pages,4,32,256], BF16
    value: object


def reshape(x, shape):
    """Preserve logical ordering across tile boundaries."""
    return ttnn.reshape(x, shape)


class FusedDecoder(LightweightModule):
    chunk_size = 128
    page_size = 32

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device):
        import torch

        config = getattr(hf_config, "text_config", hf_config)
        assert config.hidden_size == 5120 and config.intermediate_size == 17408
        assert config.num_attention_heads == 24 and config.num_key_value_heads == 4
        assert config.linear_num_key_heads == 16 and config.linear_num_value_heads == 48
        assert mesh_device.get_num_devices() == 1
        self = cls()
        self.config, self.mesh_device = config, mesh_device
        self.layer_idx, self.layer_type = layer_idx, config.layer_types[layer_idx]
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        prefix = f"model.language_model.layers.{layer_idx}."
        if not any(k.startswith(prefix) for k in state_dict):
            prefix = ""  # layer-only HF state dict

        def weight(name):
            return state_dict[prefix + name].detach().float()

        def put(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                   device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def linear(name):
            return put(weight(name + ".weight").T)

        self.in_norm = put(1 + weight("input_layernorm.weight"))
        self.post_norm = put(1 + weight("post_attention_layernorm.weight"))
        self.gate, self.up = linear("mlp.gate_proj"), linear("mlp.up_proj")
        self.down = linear("mlp.down_proj")
        if self.layer_type == "linear_attention":
            eye = torch.eye(32).reshape(1, 1, 32, 32)
            masks = torch.zeros(1, 1, 32, 96)
            masks[..., :16, :16] = 1
            masks[..., 16:, 48:64] = 1
            masks[..., 16:, 64:80] = 1
            self.gdn_constants = dict(eye=put(eye, ttnn.float32),
                tril=put(torch.ones_like(eye).tril(), ttnn.float32),
                ones=put(torch.ones_like(eye), ttnn.float32), masks=put(masks, ttnn.float32))
            self.l2_weight = put(torch.full((128,), 128**-0.5), ttnn.float32)
            self.conv_start = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
            self.mix = put(torch.cat([weight("linear_attn.in_proj_qkv.weight").T,
                weight("linear_attn.in_proj_z.weight").T,
                torch.nn.functional.pad(weight("linear_attn.in_proj_a.weight").T, (0, 16)),
                torch.nn.functional.pad(weight("linear_attn.in_proj_b.weight").T, (0, 16))], dim=-1))
            self.out = linear("linear_attn.out_proj")
            conv = weight("linear_attn.conv1d.weight")[:, 0, :]
            self.taps = [put(conv[:, i].reshape(1, 1, -1)) for i in range(4)]
            self.neg_A = put(-weight("linear_attn.A_log").exp().reshape(1, 1, 48), ttnn.float32)
            self.dt = put(weight("linear_attn.dt_bias").reshape(1, 1, 48), ttnn.float32)
            self.out_norm = put(weight("linear_attn.norm.weight"))
        else:
            qg = weight("self_attn.q_proj.weight").reshape(24, 2, 256, 5120)
            self.mix = put(torch.cat([qg[:, 0].reshape(6144, 5120).T,
                weight("self_attn.k_proj.weight").T, weight("self_attn.v_proj.weight").T,
                qg[:, 1].reshape(6144, 5120).T], dim=-1))
            self.out = linear("self_attn.o_proj")
            self.q_norm = put(1 + weight("self_attn.q_norm.weight"))
            self.k_norm = put(1 + weight("self_attn.k_norm.weight"))
            self.chunk_positions = [
                ttnn.from_torch(torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
                for position in range(0, config.max_position_embeddings, self.chunk_size)
            ]
        return self

    def allocate_state(self, *, batch_size, context_length=None):
        """Setup only. Owners release state after all traces referencing it retire."""
        def zeros(shape, dtype):
            return ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                              device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.layer_type == "linear_attention":
            return LinearState(ttnn.zeros([batch_size, 3, 10240], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                               zeros([batch_size, 48, 128, 128], ttnn.float32))
        if context_length is None or not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("Supply a valid context_length for the paged pool")
        # Flexible chunked attention reads page-table rows in 32-byte sticks:
        # round physical capacity to eight INT32 page IDs, not logical context.
        pages = batch_size * ((context_length + 255) // 256) * 8
        return PagedState(zeros([pages, 4, 32, 256], ttnn.bfloat16),
                          zeros([pages, 4, 32, 256], ttnn.bfloat16))

    def snapshot(self, state):
        """Setup/boundary operation: retain an independent resume point on device."""
        if isinstance(state, LinearState):
            return LinearState(ttnn.clone(state.conv), ttnn.clone(state.recurrent))
        return PagedState(ttnn.clone(state.key), ttnn.clone(state.value))

    def _linear(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _norm(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=self.config.rms_norm_eps,
                             compute_kernel_config=self.ckc)

    def _finish(self, residual, mixer):
        x = ttnn.add(residual, mixer)
        h = self._norm(x, self.post_norm)
        gate = ttnn.linear(h, self.gate, activation="silu", compute_kernel_config=self.ckc,
            core_grid=ttnn.CoreGrid(x=11, y=10), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mlp = self._linear(ttnn.multiply(gate, self._linear(h, self.up)), self.down)
        return ttnn.add(x, mlp)

    def _delta(self, x, state):
        batch, length, _ = x.shape
        projected = self._linear(x, self.mix)
        physical_conv = ((length + 31) // 32) * 32
        padded_projected = ttnn.pad(projected, [(0, 0), (0, physical_conv-length), (0, 0)], value=0) if physical_conv != length else projected
        qs, ks, vs = [], [], []
        for bidx in range(batch):
            hist = ttnn.to_layout(state.conv[bidx:bidx+1], ttnn.ROW_MAJOR_LAYOUT)
            cq, ck, cv = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                padded_projected[bidx:bidx+1], hist, *self.taps, 2048, 2048, 6144,
                actual_start=self.conv_start, predecessor_carry=hist,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
            qs.append(cq[:, :length]); ks.append(ck[:, :length]); vs.append(cv[:, :length])
        qflat, kflat, vflat = [parts[0] if batch == 1 else ttnn.concat(parts, dim=0) for parts in (qs, ks, vs)]
        next_conv = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)[:, length-3:length, :10240] if length >= 3 else ttnn.concat(
            [state.conv[:, length:, :], ttnn.to_layout(projected[..., :10240], ttnn.ROW_MAJOR_LAYOUT)], dim=1)
        ttnn.copy(next_conv, state.conv)

        def l2(t):
            t = ttnn.typecast(t, ttnn.float32)
            return ttnn.rms_norm(t, weight=self.l2_weight, epsilon=1e-6/128, compute_kernel_config=self.ckc)

        a = ttnn.typecast(projected[..., 16384:16432], ttnn.float32)
        decay = ttnn.multiply(self.neg_A, ttnn.softplus(ttnn.add(a, self.dt)))
        beta = ttnn.sigmoid(ttnn.typecast(projected[..., 16448:16496], ttnn.float32))
        if length > 1:
            physical = ((length + 31) // 32) * 32
            def padded(t):
                return ttnn.pad(t, [(0, 0), (0, physical-length), (0, 0)], value=0) if physical != length else t
            chunk_outputs, chunk_states = [], []
            kernel_batch = (self.mesh_device.compute_with_storage_grid_size().x * self.mesh_device.compute_with_storage_grid_size().y) // 48
            pq, pk, pv, pg, pb = map(padded, (qflat, kflat, vflat, decay, beta))
            for begin in range(0, batch, kernel_batch):
                end = min(begin + kernel_batch, batch)
                out, final = ttnn.transformer.chunk_gated_delta_rule(
                    pq[begin:end], pk[begin:end], pv[begin:end], pg[begin:end], pb[begin:end],
                    initial_state=state.recurrent[begin:end], output_final_state=True,
                    chunk_size=32, output_head_major=True, compute_kernel_config=self.ckc, **self.gdn_constants)
                chunk_outputs.append(out); chunk_states.append(final)
            y = chunk_outputs[0] if len(chunk_outputs) == 1 else ttnn.concat(chunk_outputs, dim=0)
            new_state = chunk_states[0] if len(chunk_states) == 1 else ttnn.concat(chunk_states, dim=0)
            norm_gate = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                ttnn.typecast(y, ttnn.bfloat16), padded_projected, self.out_norm, 48,
                gate_column_offset=10240, epsilon=self.config.rms_norm_eps, output_dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
            ttnn.copy(new_state, state.recurrent)
            return self._linear(ttnn.multiply(norm_gate[:, :length], projected[..., 10240:16384]), self.out)
        else:
            q = l2(reshape(qflat, [batch, length, 16, 128]))
            k = l2(reshape(kflat, [batch, length, 16, 128]))
            v = ttnn.typecast(reshape(vflat, [batch, length, 48, 128]), ttnn.float32)
            # This kernel assigns one value head to each core. Split its batch axis,
            # retaining batched projections and an explicit state for every request.
            grid = self.mesh_device.compute_with_storage_grid_size()
            kernel_batch = (grid.x * grid.y) // 48
            ys, states = [], []
            for begin in range(0, batch, kernel_batch):
                end = min(begin + kernel_batch, batch)
                y_part, state_part = fp32_recurrence(
                    q[begin:end], k[begin:end], v[begin:end], decay[begin:end], beta[begin:end],
                    initial_state=state.recurrent[begin:end], output_final_state=True, compute_kernel_config=self.ckc,
                )
                ys.append(y_part)
                states.append(state_part)
            y = ttnn.concat(ys, dim=0) if len(ys) > 1 else ys[0]
            new_state = ttnn.concat(states, dim=0) if len(states) > 1 else states[0]
        ttnn.copy(new_state, state.recurrent)
        y = self._norm(ttnn.typecast(y, ttnn.bfloat16), self.out_norm)
        z = reshape(projected[..., 10240:16384], [batch, length, 48, 128])
        return self._linear(reshape(ttnn.multiply(z, y, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), [batch, length, 6144]), self.out)

    def _rope(self, x, cos, sin):
        rot, rest = x[..., :64], x[..., 64:]
        pieces = [ttnn.experimental.rotary_embedding(rot[b:b+1], cos[b:b+1], sin[b:b+1],
                    compute_kernel_config=self.ckc) for b in range(x.shape[0])]
        rotated = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=0)
        return ttnn.concat([rotated[..., :x.shape[-2], :], rest], dim=-1)

    def _attention(self, x, state, *, page_table, cos, sin, current_pos=None, chunk_start=0):
        batch, length, _ = x.shape
        projected = self._linear(x, self.mix)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            projected[..., :8192], num_heads=24, num_kv_heads=4, transpose_key=False)
        q = self._rope(self._norm(q, self.q_norm), cos, sin)
        k = self._rope(self._norm(k, self.k_norm), cos, sin)
        if current_pos is not None:
            # Decode kernel layout is [1,B,H,D]. Cache update requires height sharding.
            q = ttnn.permute(q, (2, 0, 1, 3))
            k, v = ttnn.permute(k, (2, 0, 1, 3)), ttnn.permute(v, (2, 0, 1, 3))
            mem = ttnn.create_sharded_memory_config(
                [32, 256], core_grid=ttnn.num_cores_to_corerangeset(batch, self.mesh_device.compute_with_storage_grid_size(), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            for cache, update in ((state.key, k), (state.value, v)):
                ttnn.experimental.paged_update_cache(cache, ttnn.to_memory_config(update, mem),
                                                     update_idxs_tensor=current_pos, page_table=page_table)
            y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, state.key, state.value, page_table_tensor=page_table, cur_pos_tensor=current_pos,
                scale=256**-0.5, compute_kernel_config=self.ckc,
                # The default is 32 tokens and one core/head: 8192 serial BF16
                # softmax-stat updates at full context. Bound the accumulation
                # chain with larger chunks and sequence-parallel cores.
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32, k_chunk_size=256, max_cores_per_head_batch=16,
                    exp_approx_mode=False,
                ),
            )
            y = reshape(y, [batch, 1, 6144])
        else:
            fill_table = page_table[:, chunk_start // 32:]
            for user in range(batch):
                ttnn.experimental.paged_fill_cache(state.key, k[user:user + 1], fill_table, batch_idx=user)
                ttnn.experimental.paged_fill_cache(state.value, v[user:user + 1], fill_table, batch_idx=user)
            y = ttnn.transformer.chunked_scaled_dot_product_attention(
                q, state.key, state.value, page_table_tensor=page_table, chunk_start_idx=None,
                chunk_start_idx_tensor=self.chunk_positions[chunk_start // self.chunk_size],
                compute_kernel_config=self.ckc,
            )
            y = ttnn.transformer.concatenate_heads(y)
        # Matches the installed HF reference. Gate semantics are independently audited.
        y = ttnn.multiply(projected[..., 8192:], y, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return self._linear(y, self.out)

    def decode_forward(self, hidden_states, *, state, page_table=None, current_pos=None, cos=None, sin=None):
        """One token per request; current_pos is a device INT32 tensor, never a host list."""
        assert hidden_states.shape[1] == 1
        h = self._norm(hidden_states, self.in_norm)
        if self.layer_type == "linear_attention":
            y = self._delta(h, state)
        else:
            y = self._attention(h, state, page_table=page_table, current_pos=current_pos, cos=cos, sin=sin)
        return self._finish(hidden_states, y)

    def prefill_forward(self, hidden_states, *, state, start_position=0, page_table=None, cos=None, sin=None):
        """Chunked [B,S,H] prefill, all logical outputs returned; resume at a chunk boundary.

        cos/sin: [B,1,S,64] for absolute positions start_position..start_position+S.
        Linear state summarizes precisely the prefix; paged state owns the prefix pages.
        """
        batch, length, _ = hidden_states.shape
        if not 0 < length or start_position + length > self.config.max_position_embeddings:
            raise ValueError("Invalid logical sequence extent")
        if start_position % self.chunk_size:
            raise ValueError("Prefill continuation starts at a chunk boundary; decode handles individual tokens")
        outputs = []
        for offset in range(0, length, self.chunk_size):
            n = min(self.chunk_size, length - offset)
            x = ttnn.to_layout(hidden_states[:, offset:offset + n, :], ttnn.TILE_LAYOUT)
            h = self._norm(x, self.in_norm)
            if self.layer_type == "linear_attention":
                y = self._delta(h, state)
            else:
                c, s = cos[:, :, offset:offset + n, :], sin[:, :, offset:offset + n, :]
                physical = ((n + 31) // 32) * 32
                if physical != n:
                    h = ttnn.pad(h, [(0, 0), (0, physical - n), (0, 0)], value=0)
                    c = ttnn.pad(c, [(0, 0), (0, 0), (0, physical - n), (0, 0)], value=0)
                    s = ttnn.pad(s, [(0, 0), (0, 0), (0, physical - n), (0, 0)], value=0)
                y = self._attention(h, state, page_table=page_table, cos=c, sin=s, chunk_start=start_position + offset)
                y = y[:, :n, :]
            outputs.append(self._finish(x, y))
        # Bound each concatenation's fan-in for advertised-context prompts.
        while len(outputs) > 1:
            outputs = [ttnn.concat(outputs[i:i + 16], dim=1) if len(outputs[i:i + 16]) > 1 else outputs[i]
                       for i in range(0, len(outputs), 16)]
        return outputs[0]


def fp32_recurrence(q, k, v, g, beta, *, initial_state, scale=None, output_final_state=True,
                    compute_kernel_config=None):
    """[B,T,Hq,128], [B,T,48,128], log g/beta [B,T,48] -> output, FP32 state.

    Only final-state mode is supported. The compute policy is fixed HiFi4/FP32;
    the optional compute_kernel_config argument preserves call-site compatibility.
    """
    batch, length, qheads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if key_dim != 128 or value_dim != 128 or value_heads != 48 or qheads not in (16, 48):
        raise ValueError("FP32 recurrence supports the Qwen3.8-27B head dimensions only")
    if not output_final_state or initial_state is None:
        raise ValueError("An explicit initial state and final state output are required")
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    heads = batch * value_heads
    if heads > grid.x * grid.y:
        raise ValueError("Split recurrence batches to at most the available cores / 48")
    for x in (q, k, v, g, beta, initial_state):
        if x.dtype != ttnn.float32:
            raise ValueError("Recurrence inputs and state must be FP32")

    def split(x, h, width):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), [batch * h, length, width])
        if h != value_heads:
            x = ttnn.repeat_interleave(x, value_heads // h, dim=0)
        return reshape(x, [heads * length, 1, width]) if length > 1 else x

    q = split(ttnn.multiply(q, key_dim ** -0.5 if scale is None else scale), qheads, key_dim)
    k = split(k, qheads, key_dim)
    v = split(v, value_heads, value_dim)

    def split_scalar(x):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1)), [heads, length])
        return reshape(x, [heads * length, 1, 1])

    # Exponentiation remains the public FP32 TTNN op, as in the upstream wrapper.
    decay = split_scalar(ttnn.exp(g))
    beta = split_scalar(beta)
    state = ttnn.reshape(initial_state, [heads, key_dim, value_dim])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([heads * length, 1, value_dim]), ttnn.float32,
                                        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    final = ttnn.allocate_tensor_on_device(ttnn.Shape([heads, key_dim, value_dim]), ttnn.float32,
                                          ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    coords = [ttnn.CoreCoord(h // grid.y, h % grid.y) for h in range(heads)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    kt, vt, tile_bytes = 4, 4, 4096
    cb_sizes = [kt, kt, vt, 2 if length > 1 else 1, 2 if length > 1 else 1,
                kt * vt, 2 * vt, 2 * kt * vt, kt * vt, kt * vt, kt * vt, vt, vt, kt, kt * vt, kt * vt]
    cbs = [ttnn.CBDescriptor(total_size=tiles * tile_bytes, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index,
                                     data_format=ttnn.float32, page_size=tile_bytes)])
           for index, tiles in enumerate(cb_sizes)]
    reader_ct = [kt, vt, 0]
    # Ring indexing is disabled. A valid dummy accessor keeps the unused template
    # argument well-formed without any scratch buffer or new persistent state.
    for tensor in (q, k, v, decay, beta, state, state):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_ct = [kt, vt, 0]
    for tensor in (out, final):
        writer_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for h, core in enumerate(coords):
        rr[core.x][core.y] = [h, length, q.buffer_address(), k.buffer_address(), v.buffer_address(),
                              decay.buffer_address(), beta.buffer_address(), state.buffer_address(), 0, 0, heads]
        wr[core.x][core.y] = [h, length, out.buffer_address(), final.buffer_address(), heads]
        cr[core.x][core.y] = [length]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # Only copy/SFPU consumers get direct-to-DST FP32. CB 14/15 are separate
    # packed copies of state/sd for matmul; q/k/u/kcol remain matrix operands.
    for index in (2, 3, 4, 5, 8, 9, 10, 11):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    descriptor = ttnn.ProgramDescriptor(kernels=[
        ttnn.KernelDescriptor(kernel_source=_UPSTREAM + "reader_fused_recurrent_gated_delta_rule.cpp",
                             core_ranges=cores, compile_time_args=reader_ct, runtime_args=rr,
                             config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_UPSTREAM + "writer_fused_recurrent_gated_delta_rule.cpp",
                             core_ranges=cores, compile_time_args=writer_ct, runtime_args=wr,
                             config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_KERNEL, core_ranges=cores, compile_time_args=[kt, vt, 0],
                             runtime_args=cr, config=config),
    ], cbs=cbs, semaphores=[])
    ttnn.generic_op([q, k, v, decay, beta, state, out, final], descriptor)
    out = ttnn.permute(ttnn.reshape(out, [batch, value_heads, length, value_dim]), (0, 2, 1, 3))
    return out, ttnn.reshape(final, [batch, value_heads, key_dim, value_dim])
