"""Experimental alternatives; never imported by the delivered runtime path."""

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import _PackedMoE as FusedMoE


class DedicatedMoE(FusedMoE):
    pad_capacity = True
    retain_intermediates = False
    dedicated_weight_dtype = ttnn.bfloat8_b

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        import torch

        obj = super().from_state_dict(state, **kwargs)
        del obj.gate_up, obj.down
        mesh = kwargs["mesh_device"]

        def upload(x, dtype=None, layout=ttnn.TILE_LAYOUT):
            dtype = cls.dedicated_weight_dtype if dtype is None else dtype
            return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=layout, device=mesh)

        gu = state["mlp.experts.gate_up_proj"]
        down = state["mlp.experts.down_proj"]
        obj.expert_gates = [upload(gu[e, : obj.intermediate].T) for e in range(obj.experts)]
        obj.expert_ups = [upload(gu[e, obj.intermediate :].T) for e in range(obj.experts)]
        obj.expert_downs = [upload(down[e].T) for e in range(obj.experts)]
        obj.dedicated_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
        )
        obj.expert_ids = upload(torch.arange(obj.experts, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        obj.expert_offsets = upload(
            torch.arange(obj.experts, dtype=torch.int32) * 32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        obj.output_template = upload(torch.zeros(1, 1, obj.experts * 32, obj.hidden), ttnn.bfloat16)
        return obj

    def _group(self, x):
        logits = ttnn.linear(x, self.router, compute_kernel_config=self.compute)
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        counts = ttnn.typecast(ttnn.where(ttnn.max(routing, dim=2, keepdim=True) > 0, 32, 0), ttnn.uint32)
        counts = ttnn.reshape(ttnn.to_layout(counts, ttnn.ROW_MAJOR_LAYOUT), (self.experts,))
        output = ttnn.zeros_like(self.output_template)
        capacity_x = ttnn.pad(x, [(0, 0), (0, 0), (0, self.experts * 32 - 32), (0, 0)], 0) if self.pad_capacity else x
        down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            ttnn.to_layout(capacity_x, ttnn.ROW_MAJOR_LAYOUT),
            self.expert_gates,
            self.expert_ups,
            self.expert_downs,
            counts,
            self.expert_ids,
            input_m_tiles=1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.dedicated_compute,
            core_grid=ttnn.CoreCoord(5, 4),
            output=output,
            expert_region_offsets=self.expert_offsets,
            read_x_at_offset=False,
        )
        down = ttnn.reshape(down, (1, self.experts, 32, self.hidden))
        if self.retain_intermediates:
            self.intermediates = dict(routing=routing, counts=counts, down=down)
        result = ttnn.sum(ttnn.multiply(down, ttnn.permute(routing, (0, 3, 2, 1))), dim=1, keepdim=True)
        shared_packed = ttnn.linear(x, self.shared_packed, compute_kernel_config=self.compute)
        sg, su = shared_packed[..., :640], shared_packed[..., 640:1280]
        scale = shared_packed[..., 1280:1281]
        shared = ttnn.linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.compute,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))


from models.autoports.qwen_qwen3_8_flash_next.tt.gdn import GatedDeltaNet


class ChunkGDN(GatedDeltaNet):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        import torch

        obj = super().from_state_dict(*args, **kwargs)
        c = 32
        ii, jj = torch.arange(c)[:, None], torch.arange(c)[None, :]
        masks = torch.cat(((ii < 16) & (jj < 16), (ii >= 16) & (jj >= 16), (ii >= 16) & (jj < 16)), dim=1).float()
        obj.chunk_constants = tuple(
            ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=obj.device)
            for t in (torch.eye(c), torch.tril(torch.ones(c, c)), torch.ones(c, c), masks)
        )
        return obj

    def _recurrence(self, q, k, v, g, beta):
        length = q.shape[1]
        padded = (length + 31) // 32 * 32
        if padded != length:
            q, k, v = [ttnn.pad(t, [(0, 0), (0, padded - length), (0, 0)], 0) for t in (q, k, v)]
            g, beta = [ttnn.pad(t, [(0, 0), (0, padded - length), (0, 0)], 0) for t in (g, beta)]
        output, state = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=self.recurrent_state,
            output_final_state=True,
            chunk_size=32,
            eye=self.chunk_constants[0],
            tril=self.chunk_constants[1],
            ones=self.chunk_constants[2],
            masks=self.chunk_constants[3],
            compute_kernel_config=self.compute,
        )
        return ttnn.to_layout(output[:, :length, :, :], ttnn.TILE_LAYOUT), state

    def _chunk(self, x):
        b, s, _ = x.shape
        projected = self._linear(x, self.qkv_weight)
        history = ttnn.concat([self.conv_state, projected], dim=1)
        conv = None
        for tap in range(4):
            shifted = ttnn.to_layout(ttnn.slice(history, [0, tap, 0], [b, tap + s, 10240]), ttnn.TILE_LAYOUT)
            term = ttnn.multiply(shifted, self.conv_taps[tap])
            conv = term if conv is None else ttnn.add(conv, term)
        # The retained history is the un-convolved projection, including prefix
        # rows when a chunk has fewer than three logical tokens.
        new_conv = ttnn.to_layout(ttnn.slice(history, [0, s, 0], [b, s + 3, 10240]), ttnn.TILE_LAYOUT)
        ttnn.copy(new_conv, self.conv_state)
        conv = ttnn.silu(conv)
        q = ttnn.slice(conv, [0, 0, 0], [b, s, 2048])
        k = ttnn.slice(conv, [0, 0, 2048], [b, s, 4096])
        v = ttnn.slice(conv, [0, 0, 4096], [b, s, 10240])
        a = ttnn.typecast(self._linear(x, self.a_weight), ttnn.float32)
        g = ttnn.multiply(self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)))
        beta = ttnn.typecast(ttnn.sigmoid(self._linear(x, self.b_weight)), ttnn.float32)
        o, new_state = self._recurrence(q, k, v, g, beta)
        ttnn.copy(new_state, self.recurrent_state)
        # HF returns core-attention output in the original projection dtype,
        # then computes RMS statistics in FP32, rounds norm*weight to BF16,
        # multiplies the FP32 sigmoid gate, and rounds back to BF16.
        o = ttnn.typecast(o, ttnn.bfloat16)
        normed = ttnn.rms_norm(ttnn.typecast(o, ttnn.float32), epsilon=self.norm_eps)
        normed = ttnn.multiply(ttnn.typecast(normed, ttnn.bfloat16), self.norm_weight)
        gate = self._heads(self._linear(x, self.z_weight), b, s, 48)
        gated = ttnn.multiply(ttnn.typecast(normed, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32)))
        gated = ttnn.to_layout(ttnn.typecast(gated, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        gated = ttnn.to_layout(ttnn.reshape(gated, [b, s, 6144]), ttnn.TILE_LAYOUT)
        return self._linear(gated, self.out_weight)


class NativeFrontEndGDN(ChunkGDN):
    native_conv = True
    native_norm = True

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        import torch

        obj = super().from_state_dict(*args, **kwargs)
        obj.conv_zero = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=obj.device
        )
        obj.norm_weight_flat = ttnn.reshape(obj.norm_weight, (128,))
        return obj

    def _chunk(self, x):
        b, s, _ = x.shape
        padded = (s + 31) // 32 * 32
        projected = self._linear(x, self.qkv_weight)
        history = ttnn.concat([self.conv_state, projected], dim=1)
        if self.native_conv:
            proj = ttnn.pad(projected, [(0, 0), (0, padded - s), (0, 0)], 0) if padded != s else projected
            lanes = []
            for lane in range(b):
                incoming = ttnn.to_layout(self.conv_state[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT)
                lanes.append(
                    ttnn.experimental.kda.qkv_causal_conv1d_silu(
                        ttnn.to_layout(proj[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT),
                        incoming,
                        *self.conv_taps,
                        2048,
                        2048,
                        6144,
                        actual_start=self.conv_zero,
                        predecessor_carry=incoming,
                        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                        compute_kernel_config=self.compute,
                    )
                )
            q, k, v = [ttnn.concat([row[i] for row in lanes], dim=0) if b > 1 else lanes[0][i] for i in range(3)]
            q, k, v = [ttnn.to_layout(t[:, :s, :], ttnn.TILE_LAYOUT) for t in (q, k, v)]
        else:
            conv = None
            for tap in range(4):
                shifted = ttnn.to_layout(history[:, tap : tap + s, :], ttnn.TILE_LAYOUT)
                term = ttnn.multiply(shifted, self.conv_taps[tap])
                conv = term if conv is None else ttnn.add(conv, term)
            conv = ttnn.silu(conv)
            q, k, v = conv[..., :2048], conv[..., 2048:4096], conv[..., 4096:]
        ttnn.copy(ttnn.to_layout(history[:, s : s + 3, :], ttnn.TILE_LAYOUT), self.conv_state)
        a = ttnn.typecast(self._linear(x, self.a_weight), ttnn.float32)
        g = ttnn.multiply(self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)))
        beta = ttnn.typecast(ttnn.sigmoid(self._linear(x, self.b_weight)), ttnn.float32)
        o, new_state = self._recurrence(q, k, v, g, beta)
        ttnn.copy(new_state, self.recurrent_state)
        o = ttnn.typecast(o, ttnn.bfloat16)
        gate = self._linear(x, self.z_weight)
        if self.native_norm:
            o = ttnn.permute(o, (0, 2, 1, 3))
            if padded != s:
                o = ttnn.pad(o, [(0, 0), (0, 0), (0, padded - s), (0, 0)], 0)
                gate = ttnn.pad(gate, [(0, 0), (0, padded - s), (0, 0)], 0)
            o = ttnn.reshape(o, (b * 48, padded, 128))
            gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                o,
                gate,
                self.norm_weight_flat,
                48,
                epsilon=self.norm_eps,
                compute_kernel_config=self.compute,
                output_dtype=ttnn.bfloat16,
            )
            gated = ttnn.to_layout(gated[:, :s, :], ttnn.TILE_LAYOUT)
        else:
            normed = ttnn.rms_norm(ttnn.typecast(o, ttnn.float32), epsilon=self.norm_eps)
            normed = ttnn.multiply(ttnn.typecast(normed, ttnn.bfloat16), self.norm_weight)
            gate = self._heads(gate, b, s, 48)
            gated = ttnn.multiply(ttnn.typecast(normed, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32)))
            gated = ttnn.to_layout(ttnn.typecast(gated, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
            gated = ttnn.to_layout(ttnn.reshape(gated, [b, s, 6144]), ttnn.TILE_LAYOUT)
        return self._linear(gated, self.out_weight)


class NativeConvGDN(NativeFrontEndGDN):
    native_norm = False


class NativeNormGDN(NativeFrontEndGDN):
    native_conv = False


from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedGatedDeltaNet, FusedHyperConnection
from models.autoports.qwen_qwen3_8_flash_next.tt.hyperconnection import HyperConnection


class SplitPrefillHC(FusedHyperConnection):
    @classmethod
    def from_state_dict(cls, state, *, prefix, hf_config, mesh_device):
        obj = super().from_state_dict(state, prefix=prefix, hf_config=hf_config, mesh_device=mesh_device)
        weight = 1 + state[prefix + ".hc_norm.weight"].float()
        obj.norm = [
            ttnn.from_torch(
                weight[i * obj.width : (i + 1) * obj.width].reshape(1, 1, 1, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            for i in range(obj.count)
        ]
        return obj

    def read(self, x):
        if x.shape[2] == 1:
            return super().read(x)
        streams = [
            ttnn.rms_norm(self._part(x, i), weight=self.norm[i], epsilon=self.eps, compute_kernel_config=self.compute)
            for i in range(self.count)
        ]
        normalized = ttnn.concat(streams, dim=-1)
        packed = ttnn.linear(normalized, self.read_packed, compute_kernel_config=self.compute)
        mix = ttnn.silu(packed[..., :320])
        mix = ttnn.sigmoid(ttnn.linear(mix, self.up, compute_kernel_config=self.compute))
        mixed = ttnn.multiply(self._part(mix, 0), streams[0])
        for i in range(1, self.count):
            mixed = ttnn.add(mixed, ttnn.multiply(self._part(mix, i), streams[i]))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return ttnn.multiply(mixed, 1 / self.count), inject

    def write(self, residual, update, inject):
        if residual.shape[2] == 1:
            return super().write(residual, update, inject)
        return HyperConnection.write(self, residual, update, inject)


class PackedGDN(FusedGatedDeltaNet):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        if hasattr(obj, "packed_weight"):
            return obj
        obj.packed_weight = ttnn.concat(
            [
                obj.qkv_weight,
                obj.z_weight,
                ttnn.pad(obj.a_weight, [(0, 0), (0, 16)], 0),
                ttnn.pad(obj.b_weight, [(0, 0), (0, 16)], 0),
            ],
            dim=-1,
        )
        del obj.qkv_weight, obj.z_weight, obj.a_weight, obj.b_weight
        return obj

    # Packing is now the production path; inherit its head-major recurrence consumer.


from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedQSA


class NativeQSAHeads(FusedQSA):
    def _chunk_forward(self, hidden_states, *, current_pos, page_table):
        """Vectorized projections/selection with bounded selected-KV attention."""
        batch, seq = self.batch_size, hidden_states.shape[-2]
        x = self._reshape_logical(hidden_states, (batch, 1, seq, 2560))
        positions_packed = self._packed_position(current_pos)
        start = ttnn.permute(self._position_float(self._canonical_position(current_pos)), (1, 0, 2, 3))
        positions = ttnn.add(start, self.chunk_offsets[:, :, :seq, :])

        def heads(value, count, width):
            value = self._reshape_logical(value, (batch, seq, count, width))
            return ttnn.permute(value, (0, 2, 1, 3))

        projections = self._linear(x, self.packed_projection)
        qkv = ttnn.reshape(projections[..., :7168], (batch, seq, 7168))
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=24, num_kv_heads=2, transpose_key=False
        )
        q = self._chunk_rope(self._norm(q, self.nq), positions)
        k = self._chunk_rope(self._norm(k, self.nk), positions)
        gate = projections[..., 7168:13312]
        index = projections[..., 13312:]
        iq = self._chunk_rope(self._norm(heads(index[..., :512], 4, 128), self.niq), positions)
        raw = index[..., 512:640]
        for value, cache, count in ((k, self.k_cache, 2), (v, self.v_cache, 2), (raw, self.raw_key_cache, 1)):
            self._transfer(value, cache, positions_packed, page_table, mode=0, sequence=seq, heads=count)
        groups = self._transfer(
            self.raw_key_cache,
            self._allocate((batch * seq, 1, 4, 128)),
            positions_packed,
            page_table,
            mode=1,
            sequence=seq,
            heads=1,
        )
        pooled = ttnn.typecast(ttnn.mean(ttnn.typecast(groups, ttnn.float32), dim=2, keepdim=True), ttnn.bfloat16)
        pooled = self._reshape_logical(pooled, (batch, 1, seq, 128))
        block_start = ttnn.multiply(ttnn.floor(ttnn.multiply(positions, 0.25)), 4.0)
        pooled = self._chunk_rope(self._norm(pooled, self.nik), block_start)
        self._transfer(pooled, self.pooled_keys, positions_packed, page_table, mode=2, sequence=seq, heads=1)
        # TTNN batched matmul cannot broadcast only the head axis when B>1.
        # Fold the four query heads into M; keep each request's K bank separate.
        flat_iq = self._reshape_logical(iq, (batch, 1, 4 * seq, 128))
        scores = ttnn.matmul(flat_iq, self.pooled_keys, transpose_b=True, compute_kernel_config=self.compute)
        scores = self._reshape_logical(scores, (batch, 4, seq, self.blocks))
        scores = ttnn.sum(ttnn.relu(scores), dim=1, keepdim=True)
        complete = ttnn.floor(ttnn.multiply(ttnn.add(positions, 1.0), 0.25))
        scores = ttnn.where(ttnn.typecast(ttnn.lt(self.block_ids, complete), ttnn.bfloat16), scores, float("-inf"))
        _, selected = ttnn.topk(ttnn.typecast(scores, ttnn.bfloat16), k=512, dim=-1)
        valid = ttnn.lt(ttnn.typecast(selected, ttnn.float32), complete)
        selected_tokens = ttnn.repeat_interleave(valid, 4, dim=3)
        tail_count = ttnn.subtract(ttnn.add(positions, 1.0), ttnn.multiply(complete, 4.0))
        tail = ttnn.lt(self.tail_offsets, tail_count)
        mask = ttnn.typecast(ttnn.where(ttnn.concat((selected_tokens, tail), dim=3), 0.0, float("-inf")), ttnn.bfloat16)
        mask = self._reshape_logical(mask, (batch * seq, 1, 1, 2080))
        gathered = []
        for cache in (self.k_cache, self.v_cache):
            gathered.append(
                self._transfer(
                    cache,
                    self._allocate((batch * seq, 2, 2080, 256)),
                    positions_packed,
                    page_table,
                    mode=3,
                    sequence=seq,
                    heads=2,
                    indices=selected,
                )
            )
        q = self._reshape_logical(ttnn.permute(q, (0, 2, 1, 3)), (batch * seq, 24, 1, 256))
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            gathered[0],
            gathered[1],
            is_causal=False,
            attn_mask=mask,
            scale=0.0625,
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute,
        )
        out = self._reshape_logical(out, (batch, 1, seq, 6144))
        out = self._linear(ttnn.multiply(out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]), self.wo)
        return self._reshape_logical(out, (1, batch, seq, 2560))


_MOE_CAPACITY_COPY = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto ya = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto y = TensorAccessor(ya, get_arg_val<uint32_t>(1));
    const uint32_t row = get_arg_val<uint32_t>(2);
    CircularBuffer cb(0); cb.reserve_back(1);
    Noc noc;
    noc.async_read(x, CoreLocalMem<uint32_t>(cb.get_write_ptr()), 5120, {.page_id=row}, {});
    noc.async_read_barrier();
    noc.async_write(CoreLocalMem<uint32_t>(cb.get_write_ptr()), y, 5120, {}, {.page_id=row});
    noc.async_write_barrier();
}
"""


class FastCapacityMoE(DedicatedMoE):
    @classmethod
    def from_state_dict(cls, state, **kwargs):
        obj = super().from_state_dict(state, **kwargs)
        obj.capacity_input = ttnn.zeros(
            (1, 1, obj.experts * 32, obj.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=kwargs["mesh_device"],
        )
        return obj

    def _capacity_input(self, x):
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(32)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        args = (
            ttnn.TensorAccessorArgs(source).get_compile_time_args()
            + ttnn.TensorAccessorArgs(self.capacity_input).get_compile_time_args()
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=_MOE_CAPACITY_COPY,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=[
                ((c.x, c.y), [source.buffer_address(), self.capacity_input.buffer_address(), i])
                for i, c in enumerate(coords)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = ttnn.CBDescriptor(
            total_size=5120,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=5120)],
        )
        ttnn.generic_op([source, self.capacity_input], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
        return self.capacity_input

    def _group(self, x):
        logits = ttnn.linear(x, self.router, compute_kernel_config=self.compute)
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        counts = ttnn.typecast(ttnn.where(ttnn.max(routing, dim=2, keepdim=True) > 0, 32, 0), ttnn.uint32)
        counts = ttnn.reshape(ttnn.to_layout(counts, ttnn.ROW_MAJOR_LAYOUT), (self.experts,))
        output = ttnn.zeros_like(self.output_template)
        capacity_x = self._capacity_input(x)
        down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            ttnn.to_layout(capacity_x, ttnn.ROW_MAJOR_LAYOUT),
            self.expert_gates,
            self.expert_ups,
            self.expert_downs,
            counts,
            self.expert_ids,
            input_m_tiles=1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.dedicated_compute,
            core_grid=ttnn.CoreCoord(5, 4),
            output=output,
            expert_region_offsets=self.expert_offsets,
            read_x_at_offset=False,
        )
        down = ttnn.reshape(down, (1, self.experts, 32, self.hidden))
        if self.retain_intermediates:
            self.intermediates = dict(routing=routing, counts=counts, down=down)
        result = ttnn.sum(ttnn.multiply(down, ttnn.permute(routing, (0, 3, 2, 1))), dim=1, keepdim=True)
        shared_packed = ttnn.linear(x, self.shared_packed, compute_kernel_config=self.compute)
        sg, su = shared_packed[..., :640], shared_packed[..., 640:1280]
        scale = shared_packed[..., 1280:1281]
        shared = ttnn.linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.compute,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))


class ConcatQSAHeads(FusedQSA):
    @staticmethod
    def _reshape_logical(x, shape):
        if len(x.shape) == 4 and tuple(x.shape)[1:] == (24, 1, 256) and shape[-1] == 6144:
            return ttnn.reshape(ttnn.transformer.concatenate_heads(x), shape)
        return FusedQSA._reshape_logical(x, shape)


class MergedHC(FusedHyperConnection):
    def read(self, x):
        if x.shape[2] > 1:
            return self._read_prefill(x)
        b, seq = tuple(x.shape)[1:3]
        streams = ttnn.reshape(x, (1, b * seq, self.count, self.width))
        streams = ttnn.multiply(
            ttnn.rms_norm(streams, epsilon=self.eps, compute_kernel_config=self.compute), self.norm_packed
        )
        normalized = ttnn.reshape(streams, (1, b, seq, self.count * self.width))
        packed = ttnn.linear(normalized, self.read_packed, compute_kernel_config=self.compute)
        mix = ttnn.silu(packed[..., :320])
        mix = ttnn.linear(mix, self.up, activation="sigmoid", compute_kernel_config=self.compute)
        mix = ttnn.reshape(mix, (1, b * seq, self.count, self.width))
        mixed = ttnn.multiply(mix, streams)
        mixed = ttnn.mean(mixed, dim=2, keepdim=True)
        mixed = ttnn.reshape(mixed, (1, b, seq, self.width))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return mixed, inject


from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedPLE
from models.autoports.qwen_qwen3_8_flash_next.tt.ple import _conv_taps, _mask_padding, lookup_rows, ngram_indices


class PackedPLE(FusedPLE):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        if hasattr(obj, "packed_projection"):
            return obj
        obj.packed_projection = ttnn.concat([obj.key, obj.value], dim=-1)
        del obj.key, obj.value
        return obj

    def forward(self, x, *, tokens, positions, slots, valid_lengths, cache):
        batch, seq = tokens.shape
        if tuple(x.shape) != (1, batch, seq, 10240):
            raise ValueError("PLE x must have shape [1,B,S,10240]")
        ws = self.workspaces[(batch, seq)]
        ngram_indices(
            tokens,
            positions,
            slots,
            cache.token_history,
            valid_lengths=valid_lengths,
            output=ws.indices,
            multipliers=self.multipliers,
            primes=self.primes,
            offsets=self.offsets,
            eos=self.eos,
        )
        lookup_rows(self.tables, ws.indices, output=ws.embeddings)
        n = batch * seq
        emb = ttnn.reshape(ttnn.to_layout(ws.embeddings, ttnn.TILE_LAYOUT), (1, 1, n, 2560))
        projections = ttnn.linear(emb, self.packed_projection, compute_kernel_config=self.compute)
        key = self._norm(projections[..., :10240], self.norm_key, n)
        query = self._norm(x, self.norm_query, n)
        # HF reduces a BF16 product, then rounds each gate transform to BF16.
        dot = ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True)
        gate = ttnn.multiply(dot, 2560**-0.5)
        gate = ttnn.multiply(ttnn.sqrt(ttnn.maximum(ttnn.abs(gate), 1e-6)), ttnn.sign(gate))
        value = projections[..., 10240:]
        value = ttnn.reshape(value, (1, n, 1, 2560))
        gated = ttnn.reshape(
            ttnn.multiply(gate, value, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]), (1, 1, n, 10240)
        )
        normalized = self._norm(gated, self.norm_conv, n)
        normalized = ttnn.to_layout(ttnn.reshape(normalized, (n, 10240)), ttnn.ROW_MAJOR_LAYOUT)
        # The gather zeros padding, resets only position-zero requests, and
        # updates all nine historical vectors after reading all old taps.
        _conv_taps(
            normalized, positions, slots, valid_lengths, cache.conv_history, output=ws.taps, batch=batch, seq=seq
        )
        taps = ttnn.to_layout(ws.taps, ttnn.TILE_LAYOUT)
        convolved = ttnn.sum(ttnn.multiply(ttnn.typecast(taps, ttnn.float32), self.conv), dim=0, keepdim=True)
        convolved = ttnn.typecast(convolved, ttnn.bfloat16)
        # Mask gated values via the same device controls; tap index 3 is the
        # normalized current token, so cannot be used as a numerical mask.
        gated_rm = ttnn.to_layout(ttnn.reshape(gated, (n, 10240)), ttnn.ROW_MAJOR_LAYOUT)
        _mask_padding(gated_rm, valid_lengths, batch=batch, seq=seq)
        result = ttnn.add(
            ttnn.to_layout(gated_rm, ttnn.TILE_LAYOUT),
            ttnn.reshape(convolved, (n, 10240)),
            input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
        )
        return ttnn.reshape(result, (1, batch, seq, 10240))


class DecodeRoPE(FusedQSA):
    def _chunk_rope(self, x, positions):
        if x.shape[2] != 1:
            return super()._chunk_rope(x, positions)
        batch, heads = x.shape[0], x.shape[1]
        indices = ttnn.reshape(ttnn.typecast(ttnn.to_layout(positions, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32), (batch, 1))
        cos = ttnn.reshape(ttnn.embedding(indices, self.cos, layout=ttnn.TILE_LAYOUT), (batch, 1, 1, 64))
        sin = ttnn.reshape(ttnn.embedding(indices, self.sin, layout=ttnn.TILE_LAYOUT), (batch, 1, 1, 64))
        lanes = []
        for b in range(batch):
            lane = ttnn.permute(x[b : b + 1, :, :, :64], (2, 0, 1, 3))
            lane = ttnn.experimental.rotary_embedding(
                lane, cos[b : b + 1], sin[b : b + 1], token_index=0, compute_kernel_config=self.compute
            )
            lanes.append(ttnn.permute(lane[:, :, :heads, :], (1, 2, 0, 3)))
        rotary = lanes[0] if batch == 1 else ttnn.concat(lanes, dim=0)
        return ttnn.concat((ttnn.to_layout(rotary, ttnn.TILE_LAYOUT), x[..., 64:]), dim=-1)


class RMSL2GDN(FusedGatedDeltaNet):
    @staticmethod
    def _l2norm(x):
        return ttnn.multiply(ttnn.rms_norm(ttnn.typecast(x, ttnn.float32), epsilon=1e-6 / 128), 128**-0.5)


class PreciseRMSL2GDN(FusedGatedDeltaNet):
    def _l2norm(self, x):
        return ttnn.multiply(
            ttnn.rms_norm(ttnn.typecast(x, ttnn.float32), epsilon=1e-6 / 128, compute_kernel_config=self.compute),
            128**-0.5,
        )


class FoldQueryScaleGDN(PreciseRMSL2GDN):
    def _recurrence(self, q, k, v, g, beta):
        from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import _precise_recurrence

        b, seq, _ = q.shape
        q = ttnn.rms_norm(
            ttnn.typecast(self._heads(q, b, seq, 16), ttnn.float32),
            epsilon=1e-6 / 128,
            compute_kernel_config=self.compute,
        )
        k = self._l2norm(self._heads(k, b, seq, 16))
        v = ttnn.typecast(self._heads(v, b, seq, 48), ttnn.float32)
        return _precise_recurrence(self, q, k, v, g, beta, query_scale=1 / 128)
