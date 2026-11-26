# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized single-device Qwen3.6 text decoder.

Weights and request buffers are prepared outside forward. Activations use
[batch, sequence, 5120]; each request owns its recurrent state or paged KV.
Text positions use the same position on all three MRoPE axes. Vision encoding
is outside this decoder's contract.
"""

import math
import struct
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


class OptimizedDecoder(LightweightModule):
    """Real HF layer weights; both linear_attention and full_attention kinds.

    ``prefill_forward(x, *, state, ...)`` accepts device activations and mutates
    request state. ``decode_forward`` consumes one token per row. The caller
    owns state restoration and stable inputs when capturing decode.
    """

    def _combine_correction(self, primary, correction, key):
        if not (self.narrow_gdn_correction and key == "packed_gdn"):
            return ttnn.add(primary, correction, dtype=ttnn.bfloat16)
        # First legal movement control. Keep FP32 add -> BF16 for every region,
        # including primary Z + scalar0; do not substitute a standalone cast.
        primary = ttnn.to_memory_config(primary, ttnn.DRAM_MEMORY_CONFIG)
        correction = ttnn.to_memory_config(correction, ttnn.DRAM_MEMORY_CONFIG)
        assert primary.dtype == correction.dtype == ttnn.float32
        b, s, _ = tuple(primary.shape)
        assert tuple(correction.shape)[:2] == (b, s)

        def cut(t, start, stop):
            return ttnn.slice(t, (0, 0, start), (b, s, stop), (1, 1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        qkv = ttnn.add(cut(primary, 0, 10240), cut(correction, 0, 10240), dtype=ttnn.bfloat16)
        z = ttnn.add(cut(primary, 10240, 16384), 0.0, dtype=ttnn.bfloat16)
        ba = ttnn.add(cut(primary, 16384, 16512), cut(correction, 10240, 10368), dtype=ttnn.bfloat16)
        return ttnn.concat([qkv, z, ba], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _linear(self, x, key):
        if key not in self.weight_corrections:
            return self._linear_raw(x, key)
        if self.policy.get("flat_prefill_correction", False) and x.shape[0] > 1:
            return corrected_projection_flat_prefill(self, x, key)
        if (
            self.policy.get("correction_l1", False)
            and x.shape[0] == 1
            and x.shape[1] <= 32
            and key in self.projection_info
        ):
            correction_key = self.weight_corrections[key]
            info = self.projection_info[key]
            assert info["input_mem"] == self.projection_info[correction_key]["input_mem"]
            activation = ttnn.to_memory_config(x, info["input_mem"])
            partials = [
                ttnn.linear(
                    activation,
                    self.weights[k],
                    program_config=self.projection_info[k]["program"],
                    compute_kernel_config=self.matmul_configs["attention"],
                    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
                    dtype=ttnn.float32,
                )
                for k in (key, correction_key)
            ]
            result = self._combine_correction(*partials, key)
            result = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
            return result[:, :, : info["logical_n"]]
        primary = self._linear_raw(x, key)
        correction = self._linear_raw(x, self.weight_corrections[key])
        return self._combine_correction(primary, correction, key)

    def _precision_group(self, key):
        from .precision import group_for_weight

        return group_for_weight(key)

    def _linear_raw(self, x, key):
        output_dtype = (
            ttnn.float32
            if key in self.weight_corrections or key.endswith("_residual")
            else getattr(ttnn, self.policy.get("activation_dtype", "bfloat16"))
        )
        b, seq, width = x.shape
        if b > 1:
            x = ttnn.reshape(x, (1, b * seq, width))
        group = "mlp" if key.startswith("mlp.") else "attention"
        target_dtype = getattr(ttnn, self.policy.get(group + "_activation_dtype", "bfloat16"))
        if x.dtype != target_dtype:
            x = ttnn.typecast(x, target_dtype)
        weight = self.weights[key]
        cap = 256 if weight.shape[-1] > 20000 else self.policy.get("matmul_row_cap", 512)
        if key in self.projection_info and b * seq > cap:
            parts = [self._linear_raw(x[:, start : min(start + cap, b * seq)], key) for start in range(0, b * seq, cap)]
            y = ttnn.concat([ttnn.to_memory_config(part, ttnn.DRAM_MEMORY_CONFIG) for part in parts], dim=1)
            return ttnn.reshape(y, (b, seq, y.shape[-1])) if b > 1 else y
        if key in self.projection_info and b * seq <= 32:
            info = self.projection_info[key]
            x = ttnn.to_memory_config(x, info["input_mem"])
            y = ttnn.linear(
                x,
                weight,
                program_config=info["program"],
                compute_kernel_config=self.matmul_configs.get(self._precision_group(key), self.matmul_configs[group]),
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
                dtype=output_dtype,
            )
            keep_l1 = (
                self.policy.get("l1_mlp", False)
                and b == 1
                and (key.startswith("mlp.") or "out_proj" in key or "o_proj" in key)
            )
            if not keep_l1:
                y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
        else:
            program = None
            prefill_configs = self.policy.get("prefill_projection_configs", {})
            prefill_options = prefill_configs.get(key, prefill_configs.get(key.removesuffix("_residual"), {}))
            if key in self.projection_info or self.policy.get("prefill_2d", False):
                assert set(prefill_options) <= {
                    "grid",
                    "in0_block_w",
                    "subblock_h",
                    "subblock_w",
                    "input_l1",
                    "out_block_h",
                }
                gx, gy = prefill_options.get("grid", self.policy.get("prefill_grid", (8, 8)))
                pm = math.ceil(b * seq / 32 / gy)
                pn = math.ceil(weight.shape[-1] / 32 / gx)
                sw = prefill_options.get("subblock_w", max(v for v in (1, 2, 4) if pn % v == 0))
                requested_bh = prefill_options.get(
                    "out_block_h", 1 if self.policy.get("prefill_height_block", False) else pm
                )
                assert requested_bh >= 1
                # Height overrides are ceilings, preserving arbitrary short tails.
                bh = max(v for v in range(1, pm + 1) if v <= requested_bh and pm % v == 0)
                requested_h = min(
                    bh, prefill_options.get("subblock_h", 4 if self.policy.get("prefill_subblock_h", False) else 1)
                )
                sh = max(v for v in (1, 2, 4) if v <= requested_h and bh % v == 0 and v * sw <= 4)
                kb = prefill_options.get(
                    "in0_block_w", 2 if weight.shape[-1] > 20000 else self.policy.get("prefill_k", 4)
                )
                assert kb > 0 and (width // 32) % kb == 0
                assert sh > 0 and sw > 0 and pm % sh == 0 and pn % sw == 0 and sh * sw <= 4
                program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=kb,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    per_core_M=pm,
                    per_core_N=pn,
                    out_block_h=bh,
                    out_block_w=pn,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                )
            if prefill_options.get("input_l1", self.policy.get("prefill_input_l1", False)):
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            y = ttnn.linear(
                x,
                weight,
                program_config=program,
                compute_kernel_config=self.matmul_configs.get(self._precision_group(key), self.matmul_configs[group]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=output_dtype,
            )
        if key in self.projection_info and y.shape[-1] != self.projection_info[key]["logical_n"]:
            y = y[..., : self.projection_info[key]["logical_n"]]
        return ttnn.reshape(y, (b, seq, y.shape[-1])) if b > 1 else y

    def _norm(self, x, key):
        if (
            self.policy.get("sfpu_hidden_norm", False)
            and tuple(x.shape)[0] == 1
            and x.shape[1] <= 32
            and x.shape[-1] == 5120
        ):
            return _norm_zero_meanhidden_rmsnorm_sfpu(
                ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), self.weights[key], epsilon=self.config.rms_norm_eps
            )
        if self.policy.get("explicit_sum_norm", False) and x.shape[-1] == 5120:
            xf = ttnn.typecast(x, ttnn.float32)
            variance = ttnn.multiply(ttnn.sum(ttnn.multiply(xf, xf), dim=-1, keepdim=True), 1.0 / 5120)
            norm = ttnn.multiply(xf, ttnn.rsqrt(ttnn.add(variance, self.config.rms_norm_eps)))
            return ttnn.multiply(norm, self.weights[key], dtype=ttnn.bfloat16)
        if self.policy.get("explicit_fp32_norm", False) and not (
            self.policy.get("accurate_sharded_norm", False)
            and tuple(x.shape)[0] == 1
            and x.shape[1] <= 32
            and x.shape[-1] == 5120
        ):
            y = ttnn.rms_norm(
                ttnn.typecast(x, ttnn.float32),
                weight=self.weights[key],
                epsilon=self.config.rms_norm_eps,
                compute_kernel_config=self.ckc,
            )
            return ttnn.typecast(y, ttnn.bfloat16)
        # Single-row RMSNorm shares width across eight cores; the two required
        # transfers also widen/narrow, replacing separate dtype conversions.
        if (
            len(x.shape) == 3
            and x.shape[0] == 1
            and (
                x.shape[1] == 1
                or (self.policy.get("l1_mlp", False) or self.policy.get("shard_hidden_norm", False))
                and x.shape[1] <= 32
            )
            and x.shape[-1] == 5120
        ):
            grid = ttnn.CoreGrid(x=8, y=1)
            memory = ttnn.create_sharded_memory_config(
                (32, 640),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            y = (
                ttnn.typecast(x, ttnn.float32)
                if x.is_sharded()
                else ttnn.interleaved_to_sharded(x, memory, output_dtype=ttnn.float32)
            )
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
            if self.policy.get("l1_mlp", False) and key == "post_attention_layernorm.weight":
                return ttnn.typecast(y, ttnn.bfloat16)
            return ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        # Native BF16 intake keeps FP32 intermediates/affine under self.ckc,
        # so widening/narrowing is absorbed into the norm's unpack/pack.
        return ttnn.rms_norm(
            x,
            weight=self.weights[key],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.ckc,
        )

    def _decay(self, a):
        value = ttnn.add(a, self.weights["linear_attn.dt_bias"], dtype=ttnn.float32)
        if self.policy.get("unfused_g", False):
            return ttnn.multiply(self.neg_a, ttnn.softplus(value))
        return ttnn.multiply(
            self.neg_a, value, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]
        )

    def _round_recurrent_state(self, state):
        if (
            self.kind == "linear_attention"
            and self.policy.get("round_recurrent_state", False)
            and state.recurrent.dtype == ttnn.float32
        ):
            rounded = ttnn.typecast(state.recurrent, ttnn.bfloat16)
            ttnn.typecast(rounded, ttnn.float32, output_tensor=state.recurrent)

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
        published_state = state
        if self.kind == "linear_attention" and state.recurrent.dtype == ttnn.bfloat16:
            state = RecurrentState(state.conv, ttnn.typecast(state.recurrent, ttnn.float32))
        outputs = []
        offset = 0
        alignment = self._attention_chunk(state) if self.kind == "full_attention" else 1
        while offset < length:
            absolute = start_pos + offset
            # SDPA-block-unaligned continuation uses exact token updates until
            # aligned; no padded write can overwrite an existing prefix row.
            n = 1 if absolute % alignment else min(self.policy.get("prefill_chunk", 512), length - offset)
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
            if self.kind == "full_attention" and self.policy.get("norm_projection_l1", True) and h.is_sharded():
                # Covers compact small-prefill chunks and continuations too.
                ttnn.deallocate(h)
            outputs.append(self._finish(chunk, attention, prefill=True))
            offset += n
        if state is not published_state:
            ttnn.typecast(state.recurrent, ttnn.bfloat16, output_tensor=published_state.recurrent)
        else:
            self._round_recurrent_state(state)
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)

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

    def _attention_out(self, x, out, gate=None):
        out = ttnn.transformer.concatenate_heads(out)
        out = ttnn.multiply(
            self._linear(x, "gate") if gate is None else gate,
            out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        return self._linear(out, "self_attn.o_proj.weight")

    def _decode_attention_chunk(self, state, *, decode=False):
        baseline = self._attention_chunk(state)
        if decode and baseline == 512 and state.page_table.shape[1] <= 64:
            chunk = self.policy.get("short_capacity_decode_k_chunk", 512)
            assert chunk in (32, 64, 128, 256, 512)
            return chunk
        return baseline

    def _attention_prefill(self, x, state, *, start_pos, rope):
        assert start_pos % self._attention_chunk(state) == 0
        q, k, v, gate = self._qkv(x, rope, prefill=True)
        s = x.shape[1]
        padded = (s + 31) // 32 * 32
        if s != padded:
            q, k, v = (ttnn.pad(t, ((0, 0), (0, 0), (0, padded - s), (0, 0)), value=0.0) for t in (q, k, v))
        k, v = (ttnn.typecast(t, state.key.dtype) if t.dtype != state.key.dtype else t for t in (k, v))
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
                compute_with_storage_grid_size=ttnn.CoreCoord(*self.policy.get("sdpa_grid", [8, 4])),
                q_chunk_size=32,
                k_chunk_size=self._attention_chunk(state),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.ckc,
        )
        return self._attention_out(x, out[:, :, :s, :], gate)

    def _compact_cores(self, count, offset=0):
        ranges = []
        while count:
            x, y = offset % 8, offset // 8
            rows = count // 8 if x == 0 else 0
            if rows:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(7, y + rows - 1)))
                step = rows * 8
            else:
                step = min(count, 8 - x)
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + step - 1, y)))
            count -= step
            offset += step
        return ttnn.CoreRangeSet(ranges)

    def _compact_height_mem(self, cores, height, width):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [height, width], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _compact_rope_mem(self, batch, grid):
        # The rotary factory iterates a complete bounding rectangle, not sparse cores.
        # Choose an exact divisor of logical B: no fabricated users or mixed positions.
        choices = [
            (x * y, x, y)
            for x in range(1, grid.x + 1)
            for y in range(1, grid.y + 1)
            if x * y <= batch and batch % (x * y) == 0
        ]
        n, x, y = max(choices)
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})
        return self._compact_height_mem(cores, (batch // n) * 32, 64)


def corrected_projection_flat_prefill(layer, x, key):
    assert key in layer.weight_corrections and len(x.shape) == 3
    batch, seq, width = x.shape
    flat = ttnn.reshape(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), (1, batch * seq, width))
    primary = layer._linear_raw(flat, key)
    correction = layer._linear_raw(flat, layer.weight_corrections[key])
    # Keep the original FP32-partial -> BF16-add precision boundary.
    result = layer._combine_correction(primary, correction, key)
    return ttnn.reshape(result, (batch, seq, result.shape[-1]))


_conv107__CONV_RM_PIPE2_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ha=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto t0a=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto t1a=TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    constexpr auto t2a=TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    constexpr auto t3a=TensorAccessorArgs<t2a.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),20480);
    const auto history=TensorAccessor(ha,get_arg_val<uint32_t>(1),20480);
    const auto t0=TensorAccessor(t0a,get_arg_val<uint32_t>(2),20480);
    const auto t1=TensorAccessor(t1a,get_arg_val<uint32_t>(3),20480);
    const auto t2=TensorAccessor(t2a,get_arg_val<uint32_t>(4),20480);
    const auto t3=TensorAccessor(t3a,get_arg_val<uint32_t>(5),20480);
    const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);
    Noc noc;
    for(uint32_t task=start;task<start+count;task++) {
        const uint32_t user=task/10,chunk=task%10;
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.reserve_back(1); }
        for(uint32_t row=0;row<3;row++) {
            CircularBuffer cb(row);
            noc.async_read(history,cb,2048,{.page_id=user*3+row,.offset_bytes=chunk*2048},{.offset_bytes=0});
        }
        CircularBuffer cb4(4),cb5(5),cb6(6),cb7(7);
        noc.async_read(t0,cb4,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t1,cb5,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t2,cb6,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read(t3,cb7,2048,{.page_id=0,.offset_bytes=chunk*2048},{.offset_bytes=0});
        // Native untilize has already placed each logical user's channels
        // contiguously. This is the identical physical CB3 sequence produced
        // by the old 64-small-read gather and RISC-V packed-word copy loop.
        CircularBuffer token(3);
        noc.async_read(x,token,2048,{.page_id=user,.offset_bytes=chunk*2048},{.offset_bytes=0});
        noc.async_read_barrier();
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.push_back(1); }
        // Two input slots allow this reader to fill the next task while
        // compute/writer use the first. Compute alone pops the completed pair.
    }
}
"""

_conv107__CONV_RM_PIPE2_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ha=TensorAccessorArgs<0>();
    constexpr auto oa=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto history=TensorAccessor(ha,get_arg_val<uint32_t>(0),20480);
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(1),20480);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer out(9),done(10);
    for(uint32_t task=start;task<start+count;task++) {
        const uint32_t user=task/10,chunk=task%10,slot=(task-start)%2;
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {.offset_bytes=0},{.page_id=user,.offset_bytes=chunk*2048});
        for(uint32_t row=0;row<3;row++) {
            // Output publication proves compute saw all inputs for this task.
            // This writer never pops input CBs, so its read pointer stays at
            // the ring base; choose the physical slot explicitly.
            CircularBuffer carry(row+1);
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(carry),history,2048,
                {.offset_bytes=slot*2048},{.page_id=user*3+row,.offset_bytes=chunk*2048});
        }
        noc.async_write_barrier();
        out.pop_front(1);
        done.reserve_back(1); done.push_back(1);
    }
}
"""

_conv107__CONV_RM_PIPE2_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
// Native ttnn.silu(BF16) chooses calculate_silu<false>, including its BF16
// approximation/rounding. Keep that arithmetic while addressing our FP32 DST.
// silu_tile<false> would also select BF16 DST addressing, so do not use it here.
inline void conv_bf16_silu(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_silu,(false,8),idst,VectorMode::RC));
}
void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0,9); copy_init(0);
    CircularBuffer out(9),done(10);
    // Pop inputs only as complete groups. The unpack read pointer is at
    // the ring base for every group, so copy_tile(...,slot,...) never crosses
    // the physical CB end. A final singleton is safe because no task follows.
    for(uint32_t first=0;first<count;first+=2) {
        const uint32_t group=(count-first>=2)?2:1;
        for(uint32_t slot=0;slot<group;slot++) {
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.wait_front(slot+1); }
        out.reserve_back(1); tile_regs_acquire();
        for(uint32_t tap=0;tap<4;tap++) {
            const uint32_t work=tap?1:0;
            copy_tile(tap,slot,work); copy_tile(tap+4,slot,2);
            mul_binary_tile_init(); mul_binary_tile(work,2,work);
            if(tap) { add_binary_tile_init(); add_binary_tile(0,1,0); }
        }
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        silu_tile_init(); conv_bf16_silu(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,9); tile_regs_release();
        out.push_back(1);
        }
        // Each token follows that task's complete NoC write barrier. Preserve
        // input storage through both consumers; compute is the sole pop owner.
        done.wait_front(group);
        for(uint32_t id=0;id<8;id++) { CircularBuffer cb(id); cb.pop_front(group); }
        done.pop_front(group);
    }
}
"""


# Exact measured helper from narrow_gdn_combine.py
_NARROW_GDN_COMBINE_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto pa=TensorAccessorArgs<0>();
    constexpr auto ra=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto primary=TensorAccessor(pa,get_arg_val<uint32_t>(0),4096);
    const auto residual=TensorAccessor(ra,get_arg_val<uint32_t>(1),4096);
    const uint32_t pn=get_arg_val<uint32_t>(2),rn=get_arg_val<uint32_t>(3);
    const uint32_t start=get_arg_val<uint32_t>(4),count=get_arg_val<uint32_t>(5);
    Noc noc; CircularBuffer p(0),r(1),zero(3);
    zero.reserve_back(1); noc.async_write_zeros(zero,4096); noc.write_zeros_l1_barrier();zero.push_back(1);
    for(uint32_t tile=start;tile<start+count;tile++) {
        const uint32_t row=tile/516,col=tile%516;
        const bool has_residual=col<320 || col>=512;
        p.reserve_back(1);
        if(has_residual) r.reserve_back(1);
        noc.async_read(primary,p,4096,{.page_id=row*pn+col},{.offset_bytes=0});
        if(has_residual) {
            const uint32_t rc=col<320?col:col-192;
            noc.async_read(residual,r,4096,{.page_id=row*rn+rc},{.offset_bytes=0});
        }
        noc.async_read_barrier(); p.push_back(1);
        if(has_residual) r.push_back(1);
    }
}
"""
_NARROW_GDN_COMBINE_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer out(2);
    for(uint32_t tile=start;tile<start+count;tile++) {
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {.offset_bytes=0},{.page_id=tile});
        noc.async_write_barrier();out.pop_front(1);
    }
}
"""
_NARROW_GDN_COMBINE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t start=get_arg_val<uint32_t>(0),count=get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0,2);copy_init(0);
    CircularBuffer p(0),r(1),out(2),zero(3);zero.wait_front(1);
    for(uint32_t tile=start;tile<start+count;tile++) {
        const uint32_t col=tile%516;const bool has_residual=col<320 || col>=512;
        p.wait_front(1);if(has_residual)r.wait_front(1);
        out.reserve_back(1);tile_regs_acquire();
        copy_tile(0,0,0);
        if(has_residual) copy_tile(1,0,1);else copy_tile(3,0,1);
        add_binary_tile_init();add_binary_tile(0,1,0);
        // Preserve native FP32 add -> RNE BF16 boundary, including Z +0.
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,2);tile_regs_release();
        out.push_back(1);p.pop_front(1);if(has_residual)r.pop_front(1);
    }
    zero.pop_front(1);
}
"""


# Exact measured helper from prefill_unflatten_bf16_v2.py


_fused_delta_row_GDN_FUSE_ACCUMULATOR = True
_fused_delta_row_GDN_HOIST_DELTA = True
_fused_delta_row_GDN_HOIST_SCALAR_INIT = True
_fused_delta_row_GDN_HOIST_COL_INIT = True
_fused_delta_row_GDN_FUSE_DECAY_PREDICT = True


def _fused_delta_row_compact_fp32_compute_config(state_bf16=False):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    # FP32 DEST alone still allows Float32 -> TF32 through source registers.
    # Explicit direct unpack is required for every FP32 CB copied into DST.
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for index in range(19):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if state_bf16:
        modes[5] = modes[11] = ttnn.UnpackToDestMode.Default
    config.unpack_to_dest_mode = modes
    return config


_fused_delta_row_GDN_COMPACT_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t STATE_BYTES=get_compile_time_arg_val(0);
    constexpr auto sa=TensorAccessorArgs<1>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(0),STATE_BYTES);
    const auto o=TensorAccessor(oa,get_arg_val<uint32_t>(1),4096);
    constexpr uint32_t BH=get_compile_time_arg_val(oa.next_compile_time_args_offset());
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer updated(11),out(12);
    for(uint32_t h=start;h<start+count;h++) {
        updated.wait_front(16); out.wait_front(4);
        // Waiting for output guarantees compute finished reading updated CB.
        for(uint32_t i=0;i<16;i++)
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,STATE_BYTES,{.offset_bytes=i*STATE_BYTES},{.page_id=h*16+i});
        for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
            const uint32_t row=h%32;
            const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                {.offset_bytes=i*4096+face*1024},
                {.page_id=(h/32)*4+i,.offset_bytes=dest});
        }
        if(h+1==BH && BH%32) {
            // Preserve deterministic zero padding for non-tile-aligned head count.
            // Row1 in the reduced output tile is zero by the reduction pack mask.
            for(uint32_t row=BH%32;row<32;row++) for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
                const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                    {.offset_bytes=i*4096+64+face*1024},
                    {.page_id=(h/32)*4+i,.offset_bytes=dest});
            }
        }
        noc.async_write_barrier(); updated.pop_front(16); out.pop_front(4);
    }
}
"""

_fused_delta_row_GDN_COMPACT_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"


// Stage-local precise FP32 specialization of native scalar normalization:
// mean is known zero, so eliminate its unpack/broadcast and32 scalar adds.
namespace ckernel {
#ifdef TRISC_MATH
namespace sfpu {
inline void _gdn_mul_scalar_(uint32_t data, uint32_t scalar, uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    _broadcast_scalar_from_dest_(scalar*DEST_TILE_SIZE_RAW);
    TTI_SFPMOV(0,LREG_BCAST,LREG_SCALAR_INV_STD,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        const uint32_t db=data*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

// Replicate row0 without loading a tile of ones and multiplying by it.
// Native row-broadcast's SFPTRANSP mapping is retained exactly.
inline void _gdn_copy_row_(uint32_t data,uint32_t row,uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t rb=row*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
    TT_SFPLOAD(p_sfpu::LREG4,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7,IM,ADDR_MOD_7,rb+COL_GROUP_OFFSETS[3]);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0,0,0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG1,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG2,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG3,0);
    TTI_SFPTRANSP(0,0,0,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t base=ob+half*FACE2_BASE+band*ROW_BAND_STRIDE;
        TT_SFPSTORE(p_sfpu::LREG0,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[0]);
        TT_SFPSTORE(p_sfpu::LREG1,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[1]);
        TT_SFPSTORE(p_sfpu::LREG2,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[2]);
        TT_SFPSTORE(p_sfpu::LREG3,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[3]);
    }
}


// Only row0 survives the incumbent subtract/multiply/row-broadcast sequence.
// Transpose two four-row bands so LREG0 and LREG4 hold the useful V/pred rows.
// Keep subtraction and beta multiplication as distinct FP32 operations.
inline void _gdn_delta_row_(uint32_t value,uint32_t prediction,uint32_t output,uint32_t beta) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t vb=value*DEST_TILE_SIZE_RAW,pb=prediction*DEST_TILE_SIZE_RAW;
    const uint32_t ob=output*DEST_TILE_SIZE_RAW;
    TT_SFPLOAD(p_sfpu::LREG0,IM,ADDR_MOD_7,vb+COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG1,IM,ADDR_MOD_7,vb+COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG2,IM,ADDR_MOD_7,vb+COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG3,IM,ADDR_MOD_7,vb+COL_GROUP_OFFSETS[3]);
    TT_SFPLOAD(p_sfpu::LREG4,IM,ADDR_MOD_7,pb+COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5,IM,ADDR_MOD_7,pb+COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6,IM,ADDR_MOD_7,pb+COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7,IM,ADDR_MOD_7,pb+COL_GROUP_OFFSETS[3]);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0,0,0,0);
    TTI_SFPMOV(0,p_sfpu::LREG4,p_sfpu::LREG4,1 /* negate */);
    TTI_SFPADD(p_sfpu::LREG0,p_sfpu::LCONST_1,p_sfpu::LREG4,p_sfpu::LREG1,0);
    // Native scalar broadcast touches only LREG0/2/14, preserving delta in1.
    _broadcast_scalar_from_dest_(beta*DEST_TILE_SIZE_RAW);
    TTI_SFPMUL(p_sfpu::LREG1,p_sfpu::LREG0,p_sfpu::LCONST_0,p_sfpu::LREG1,0);
    TTI_SFPNOP;
    // As in _gdn_copy_row_: four identical row vectors, inverse transpose,
    // then the same four slots written to every four-row band.
    TTI_SFPMOV(0,p_sfpu::LREG1,p_sfpu::LREG0,0);
    TTI_SFPMOV(0,p_sfpu::LREG1,p_sfpu::LREG2,0);
    TTI_SFPMOV(0,p_sfpu::LREG1,p_sfpu::LREG3,0);
    TTI_SFPTRANSP(0,0,0,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t base=ob+half*FACE2_BASE+band*ROW_BAND_STRIDE;
        TT_SFPSTORE(p_sfpu::LREG0,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[0]);
        TT_SFPSTORE(p_sfpu::LREG1,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[1]);
        TT_SFPSTORE(p_sfpu::LREG2,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[2]);
        TT_SFPSTORE(p_sfpu::LREG3,IM,ADDR_MOD_7,base+COL_GROUP_OFFSETS[3]);
    }
}

// Preserve TWO rounding steps: product=delta*K, then updated=product+state.
// This deliberately uses SFPMUL followed by SFPADD, never SFPMAD/FMA.
inline void _gdn_col_mul_add_(uint32_t data,uint32_t column,uint32_t output,uint32_t residual) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t db=data*DEST_TILE_SIZE_RAW, cb=column*DEST_TILE_SIZE_RAW;
    const uint32_t ob=output*DEST_TILE_SIZE_RAW, rb=residual*DEST_TILE_SIZE_RAW;
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,cb+left);
        lltt::replay(REPLAY_SLOT_BROADCAST,REPLAY_LEN_BROADCAST);
        _broadcast_stage3_with_data_prefetch_(db+left,db+left+ODD_COLS_OFFSET,db+right,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        // Broadcast is complete; its two scratch LREGs can carry residuals.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+left);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+left+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
        TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+right);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+right+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
        TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

// Decay product, K product and accumulation remain separate FP32 operations.
template<bool ACCUMULATE>
inline void _gdn_decay_predict_(uint32_t data,uint32_t column,uint32_t output,uint32_t scalar,uint32_t segment) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t slot=(segment/2)*FACE1_BASE+(segment%2)*ODD_COLS_OFFSET;
    const uint32_t db=data*DEST_TILE_SIZE_RAW, cb=column*DEST_TILE_SIZE_RAW+slot;
    const uint32_t ob=output*DEST_TILE_SIZE_RAW, rb=ob;
    // First K tile loads the decay; later K tiles retain LREG7 across copy_tile.
    // Column broadcast uses LREG0..6 and leaves this scalar untouched.
    if constexpr(!ACCUMULATE) {
        _broadcast_scalar_from_dest_(scalar*DEST_TILE_SIZE_RAW);
        TTI_SFPMOV(0,LREG_BCAST,LREG_SCALAR_INV_STD,0);
    }
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        // Prepared K is already replicated; all SFPU lanes contain this row's K.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,cb+left);
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        // Preserve rounded decay in state DST, then reuse those FP32 LREGs.
        TTI_SFPMUL(LREG_DATA0,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_SCALAR_INV_STD,p_sfpu::LCONST_0,LREG_DATA3,0);
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        if constexpr(ACCUMULATE) {
        // Broadcast is complete; its two scratch LREGs can carry residuals.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+left);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+left+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
        TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,rb+right);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,rb+right+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
        TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        }
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}


// Match native calculate_typecast_fp32_to_fp16b's integer sequence exactly.
// DATA stays unrounded for the current Q readout. TMP and MASK are dead scratch;
// packed Q/K direct loads do not use the old column mask in LREG_MASK.
template<uint32_t DATA>
inline void _gdn_store_bf16_rne_(uint32_t address) {
    TTI_SFPMOV(0,DATA,LREG_MASK,0);
    TTI_SFPSHFT((-16)&0xFFF,DATA,LREG_TMP,5);
    TTI_SFPAND(0,p_sfpu::LREG12,LREG_TMP,0);
    TTI_SFPIADD(0,p_sfpu::LREG13,LREG_MASK,sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0,LREG_TMP,LREG_MASK,sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPAND(0,p_sfpu::LREG14,LREG_MASK,0);
    TT_SFPSTORE(LREG_MASK,InstrModLoadStore::FP32,ADDR_MOD_7,address);
}

// Preserve update=(delta*K)+decayed and readout=(update*Q)+accumulator.
// Rounded updated values remain live across Q broadcast; persistence stores
// fill that broadcast's four otherwise-idle dependent-shift latency slots.
template<bool ACCUMULATE>
inline void _gdn_update_readout_(uint32_t state,uint32_t key,uint32_t output,uint32_t delta,uint32_t query,uint32_t segment) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    constexpr bool STATE_BF16=get_compile_time_arg_val(1);
    const uint32_t slot=(segment/2)*FACE1_BASE+(segment%2)*ODD_COLS_OFFSET;
    const uint32_t db=delta*DEST_TILE_SIZE_RAW, kb=key*DEST_TILE_SIZE_RAW+slot;
    const uint32_t sb=state*DEST_TILE_SIZE_RAW, qb=query*DEST_TILE_SIZE_RAW+slot, ob=output*DEST_TILE_SIZE_RAW;
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        // Prepared K is already replicated; all SFPU lanes contain this row's K.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,kb+left);
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,sb+left);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,sb+left+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
        TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,sb+right);
        TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,sb+right+ODD_COLS_OFFSET);
        TTI_SFPNOP;
        TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
        TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        // Q is also pre-expanded. Four independent state stores preserve
        // updated FP32 values and cover the loaded-Q dependency latency.
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,qb+left);
        if constexpr(STATE_BF16) {
            _gdn_store_bf16_rne_<LREG_DATA0>(sb+left);
            _gdn_store_bf16_rne_<LREG_DATA1>(sb+left+ODD_COLS_OFFSET);
            _gdn_store_bf16_rne_<LREG_DATA2>(sb+right);
            _gdn_store_bf16_rne_<LREG_DATA3>(sb+right+ODD_COLS_OFFSET);
        } else {
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,sb+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,sb+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,sb+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,sb+right+ODD_COLS_OFFSET);
        }
        TTI_SFPMUL(LREG_DATA0,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA0,0);
        TTI_SFPMUL(LREG_DATA1,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA1,0);
        TTI_SFPMUL(LREG_DATA2,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA2,0);
        TTI_SFPMUL(LREG_DATA3,LREG_BCAST,p_sfpu::LCONST_0,LREG_DATA3,0);
        if constexpr(ACCUMULATE) {
            TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,ob+left);
            TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
            TTI_SFPNOP;
            TTI_SFPADD(LREG_DATA0,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA0,0);
            TTI_SFPADD(LREG_DATA1,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA1,0);
            TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,ob+right);
            TT_SFPLOAD(LREG_TMP,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
            TTI_SFPNOP;
            TTI_SFPADD(LREG_DATA2,p_sfpu::LCONST_1,LREG_BCAST,LREG_DATA2,0);
            TTI_SFPADD(LREG_DATA3,p_sfpu::LCONST_1,LREG_TMP,LREG_DATA3,0);
        }
        TT_SFPSTORE(LREG_DATA0,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(LREG_DATA1,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(LREG_DATA2,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(LREG_DATA3,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}


// Replicate each prepared column-zero value across its tile with the same
// mask/rotate-add broadcast used by the incumbent's per-state-tile path.
// Each segment occupies one eight-column slot; four segments fill one tile.
inline void _gdn_expand_column_(uint32_t data,uint32_t unused,uint32_t output,uint32_t segment) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t slot=(segment/2)*FACE1_BASE+(segment%2)*ODD_COLS_OFFSET;
    const uint32_t db=data*DEST_TILE_SIZE_RAW,ob=output*DEST_TILE_SIZE_RAW+slot;
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        TT_SFPLOAD(LREG_BCAST,IM,ADDR_MOD_7,db+left);
        lltt::replay(REPLAY_SLOT_BROADCAST,REPLAY_LEN_BROADCAST);
        _broadcast_stage3_preserve_data_();
        TT_SFPSTORE(LREG_BCAST,IM,ADDR_MOD_7,ob+left);
    }
}

} // namespace sfpu
#endif
template<bool INIT=true>
ALWI void gdn_mul_scalar(uint32_t data,uint32_t scalar) {
    if constexpr(INIT) sfpu_normalize_bcast_scalar_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_mul_scalar_,data,scalar,data,VectorMode::None)));
}

ALWI void gdn_copy_row(uint32_t data) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_copy_row_,data,data,data,VectorMode::None)));
}


ALWI void gdn_delta_row(uint32_t value,uint32_t prediction,uint32_t output,uint32_t beta) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_delta_row_,value,prediction,output,VectorMode::None,beta)));
}

ALWI void gdn_col_mul_add(uint32_t data,uint32_t column,uint32_t residual) {
    sfpu_bcast_col_init();
    MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(data,residual,data,VectorMode::None),
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_col_mul_add_,data,column,data,VectorMode::None,residual)));
}

template<bool INIT=true>
ALWI void gdn_col_mul_add_to(uint32_t data,uint32_t column,uint32_t residual,uint32_t output) {
    if constexpr(INIT) sfpu_bcast_col_init();
    MATH((::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(data,residual,output,VectorMode::None),
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_col_mul_add_,data,column,output,VectorMode::None,residual)));
}


// Native broadcast implementation supports distinct input/output registers;
// use that directly instead of copying a retained state before multiplying it.
template<bool INIT=true>
ALWI void gdn_col_mul_to(uint32_t data,uint32_t column,uint32_t output) {
    if constexpr(INIT) sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        _calculate_sfpu_binary_bcast_full_tile_,(ckernel::BinaryOp::MUL,ckernel::BroadcastType::COL),
        data,column,output,VectorMode::None)));
}
ALWI void gdn_copy_row_to(uint32_t row,uint32_t output) {
    sfpu_bcast_row_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_copy_row_,row,row,output,VectorMode::None)));
}

template<bool ACCUMULATE, bool INIT=true>
ALWI void gdn_decay_predict(uint32_t data,uint32_t column,uint32_t output,uint32_t scalar,uint32_t segment) {
    if constexpr(INIT) sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_decay_predict_,(ACCUMULATE),data,column,output,VectorMode::None,scalar,segment)));
}


template<bool ACCUMULATE>
ALWI void gdn_update_readout(uint32_t state,uint32_t key,uint32_t output,uint32_t delta,uint32_t query,uint32_t segment) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_update_readout_,(ACCUMULATE),state,key,output,VectorMode::None,delta,query,segment)));
}


ALWI void gdn_expand_column(uint32_t tile,uint32_t output,uint32_t segment) {
    sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_expand_column_,tile,tile,output,VectorMode::None,segment)));
}

} // namespace ckernel

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    copy_tile(cb,tile,dst);
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,cb,tile); tile_regs_release();
}
// Exactly the accurate FP32 reduction topology: fold the four H tiles with
// SFPU addition, then SFPU reduce their32 rows into row0. No TF32 FPU path.
void contract(uint32_t matrix,uint32_t column,uint32_t result) {
    CircularBuffer dst(result); dst.reserve_back(4);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(result)));
    for(uint32_t n=0;n<4;n++) {
        tile_regs_acquire();
        for(uint32_t m=0;m<4;m++) {
            const uint32_t work=m==0 ? 0 : 2;
            load(matrix,m*4+n,work); load(column,m,1);
            sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(work,1);
            if(m) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
        finish(result,n);
    }
    PACK((llk_pack_reduce_mask_clear())); dst.push_back(4);
}

// Raw mode matches external FP32 square/sum(+epsilon)/rsqrt/multiply, then
// separate Q scaling. Normalized mode is a pure transpose control.
void prepare_vector(uint32_t input,uint32_t output,bool is_q) {
    constexpr bool NORMALIZED=get_compile_time_arg_val(0);
    CircularBuffer in(input),col(output),norm(16),rows(17);
    in.wait_front(4);
    copy_init(input);
    uint32_t transposed_source=input;
    if constexpr(!NORMALIZED) {
        norm.reserve_back(1); tile_regs_acquire();
        for(uint32_t i=0;i<4;i++) {
            const uint32_t work=i?2:0;
            load(input,i,work); load(input,i,1);
            mul_binary_tile_init(); mul_binary_tile(work,1,work);
            if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
        // Float32 encodings of1e-6 and128**-0.5. Scaling remains separate.
        binop_with_scalar_tile_init(); add_unary_tile(0,0x358637bdu);
        rsqrt_tile_init(); rsqrt_tile(0);
        finish(16,0); norm.push_back(1); norm.wait_front(1);
        rows.reserve_back(4);
        for(uint32_t i=0;i<4;i++) {
            tile_regs_acquire(); load(input,i,0); load(16,0,1);
            gdn_mul_scalar(0,1);
            if(is_q) { binop_with_scalar_tile_init(); mul_unary_tile(0,0x3db504f3u); }
            finish(17,i);
        }
        rows.push_back(4); rows.wait_front(4); norm.pop_front(1);
        transposed_source=17;
    }
    // Four independent 32-row segments fill four eight-column slots of DST1.
    // Preserve this output across transposes; each transpose writes only DST0.
    col.reserve_back(1); transpose_init(transposed_source); tile_regs_acquire();
    for(uint32_t i=0;i<4;i++) {
        transpose_tile(transposed_source,i,0); gdn_expand_column(0,1,i);
    }
    tile_regs_commit(); tile_regs_wait(); pack_tile<true>(1,output,0); tile_regs_release();
    col.push_back(1); in.pop_front(4);
    if constexpr(!NORMALIZED) rows.pop_front(4);
}


// Per value-column full-DST allocation: state0..3, work4, Q/K5,
// accumulator/delta6, decay/beta7. No retained state crosses a release.
void kernel_main() {
#ifdef TRISC_MATH
    static_assert(get_dest_max_tiles<DST_SYNC_MODE,true,DstTileShape::Tile32x32>()>=8,
                  "Retained GDN needs eight FP32 DST tiles and full synchronization");
#endif
    const uint32_t count=get_arg_val<uint32_t>(0);
    const uint32_t start_head=get_arg_val<uint32_t>(1);
    constexpr bool STATE_BF16=get_compile_time_arg_val(1);
    constexpr bool GROUP_QK=get_compile_time_arg_val(2);
    constexpr bool FUSE_ACCUMULATOR=get_compile_time_arg_val(3);
    constexpr bool HOIST_DELTA=get_compile_time_arg_val(4);
    constexpr bool HOIST_SCALAR_INIT=get_compile_time_arg_val(5);
    constexpr bool HOIST_COL_INIT=get_compile_time_arg_val(6);
    constexpr bool FUSE_DECAY_PREDICT=get_compile_time_arg_val(7);
    static_assert(!FUSE_DECAY_PREDICT || FUSE_ACCUMULATOR,
                  "Decay prediction fusion requires ordered fused accumulation");
    static_assert(FUSE_DECAY_PREDICT && FUSE_ACCUMULATOR && HOIST_DELTA,
                  "Replicated Q/K candidate requires the active fused recurrence path");
    compute_kernel_hw_startup(STATE_BF16?3:5,6,12);
    copy_init(STATE_BF16?3:5);
    CircularBuffer q(13),k(14),v(2),raw_g(3),decay(18),beta(4),state(5),zero(6),one(7);
    CircularBuffer updated(11),out(12),raw_q(0),raw_k(1);
    zero.wait_front(1); one.wait_front(1);
    for(uint32_t head=0;head<count;head++) {
        state.wait_front(16); raw_g.wait_front(1);
        decay.reserve_back(1); tile_regs_acquire(); load(3,0,0);
        exp_tile_init<false>(); exp_tile<false>(0); finish(18,0);
        decay.push_back(1); decay.wait_front(1); raw_g.pop_front(1);
        // Core boundaries may split triples. Prepare at the local boundary or
        // global group start, and retain only while the same Q/K group continues.
        if (!GROUP_QK || head==0 || (start_head+head)%3==0) {
            prepare_vector(0,13,true); prepare_vector(1,14,false);
        } else {
            // Reader's rings remain one head deep: consume every duplicate input
            // to permit it to fetch the next state/value/gate without stalling.
            raw_q.wait_front(4); raw_k.wait_front(4);
            raw_q.pop_front(4); raw_k.pop_front(4);
        }
        copy_init(STATE_BF16?3:5);
        q.wait_front(1); k.wait_front(1); v.wait_front(4); beta.wait_front(1);
        updated.reserve_back(16); out.reserve_back(4);
        for(uint32_t n=0;n<4;n++) {
            tile_regs_acquire();
            load(18,0,7);
            if constexpr(STATE_BF16) {
                reconfig_data_format_srca(5); copy_init(5);
            }
            // copy_tile does not alter SFPU config, MOD7, LREGs or replay.
            if constexpr(HOIST_SCALAR_INIT && !FUSE_DECAY_PREDICT) sfpu_normalize_bcast_scalar_init();
            for(uint32_t m=0;m<4;m++) {
                load(5,m*4+n,m);
                if constexpr(!FUSE_DECAY_PREDICT) gdn_mul_scalar<!HOIST_SCALAR_INIT>(m,7);
            }
            if constexpr(STATE_BF16) {
                reconfig_data_format_srca(3); copy_init(3);
            }
            // One packed K tile survives the four ordered prediction/update segments.
            load(14,0,5);
            // Preserve the baseline four-K-tile accumulation order exactly.
            if constexpr(HOIST_COL_INIT && FUSE_ACCUMULATOR) sfpu_bcast_col_init();
            for(uint32_t m=0;m<4;m++) {
                if constexpr(FUSE_DECAY_PREDICT) {
                    if(m) gdn_decay_predict<true,!HOIST_COL_INIT>(m,5,6,7,m);
                    else gdn_decay_predict<false,!HOIST_COL_INIT>(m,5,6,7,m);
                } else if constexpr(FUSE_ACCUMULATOR) {
                    if(m) {
                        // Exact separate FP32 product then add into accumulator.
                        // Skip the full temporary-tile store/reload/add pass.
                        gdn_col_mul_add_to<!HOIST_COL_INIT>(m,5,6,6);
                    } else { gdn_col_mul_to<!HOIST_COL_INIT>(m,5,6); }
                } else {
                    const uint32_t work=m?4:6;
                    gdn_col_mul_to(m,5,work);
                    if(m) { add_binary_tile_init(); add_binary_tile(6,4,6); }
                }
            }
            sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(6,1,1);
            load(2,n,4); load(4,0,7);
            gdn_delta_row(4,6,4,7);
            // DST7's beta is dead after delta; reuse it for Q beside K in DST5.
            // Each m keeps exactly the old update and readout arithmetic, while
            // eliminating an intervening updated-state store/reload traversal.
            static_assert(HOIST_DELTA && FUSE_ACCUMULATOR,
                          "Fused update/readout requires hoisted delta and ordered accumulation");
            load(13,0,7);
            if constexpr(STATE_BF16) {
                // Establish exactly the native RNE integer constants. Column
                // init below restores the zero-stride MOD7 without changing them.
                typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
            }
            sfpu_bcast_col_init();
            for(uint32_t m=0;m<4;m++) {
                if(m) gdn_update_readout<true>(m,5,6,4,7,m);
                else gdn_update_readout<false>(m,5,6,4,7,m);
            }
            sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(6,1,1);
            // Persistent state was rounded from a copy of each live update.
            tile_regs_commit(); tile_regs_wait();
            PACK((llk_pack_reduce_mask_clear()));
            // Q readout is already complete in FP32 DST6. Only persistence rounds.
            if constexpr(STATE_BF16) pack_reconfig_data_format(12,11);
            for(uint32_t m=0;m<4;m++) pack_tile<true>(m,11,m*4+n);
            if constexpr(STATE_BF16) pack_reconfig_data_format(11,12);
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL,PackMode::Default>(12)));
            pack_tile<true>(6,12,n);
            PACK((llk_pack_reduce_mask_clear()));
            tile_regs_release();
        }
        updated.push_back(16); out.push_back(4);
        state.pop_front(16); decay.pop_front(1); v.pop_front(4); beta.pop_front(1);
        if (!GROUP_QK || head+1==count || (start_head+head)%3==2) {
            q.pop_front(1); k.pop_front(1);
        }
    }
    zero.pop_front(1); one.pop_front(1);
}
"""


_affinev3__GDN_AFFINE_GATE_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
void kernel_main() {
    constexpr auto na=TensorAccessorArgs<0>();
    constexpr auto wa=TensorAccessorArgs<na.next_compile_time_args_offset()>();
    constexpr auto pa=TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto norm=TensorAccessor(na,get_arg_val<uint32_t>(0),2048);
    const auto weight=TensorAccessor(wa,get_arg_val<uint32_t>(1),2048);
    const auto packed=TensorAccessor(pa,get_arg_val<uint32_t>(2),2048);
    const uint32_t batch=get_arg_val<uint32_t>(3),start=get_arg_val<uint32_t>(4),count=get_arg_val<uint32_t>(5);
    Noc noc; CircularBuffer n(0),w(1),gate(2),scratch(3);
    scratch.reserve_back(2);
    const uint32_t sb=(scratch.get_write_ptr()+63u)&~63u;
    for(uint32_t tile=start;tile<start+count;tile++) {
        n.reserve_back(1); w.reserve_back(1); gate.reserve_back(1);
        const uint32_t head=tile/4,column=tile%4,np=n.get_write_ptr();
        for(uint32_t user=0;user<batch;user++) {
            const uint32_t source_head=user*48+head,row=source_head%32;
            const uint32_t source_tile=(source_head/32)*4+column;
            for(uint32_t face=0;face<2;face++) {
                const uint32_t src=((row/16)*512+face*256+(row%16)*16)*2;
                const uint32_t dst=(user/16)*1024+face*512+(user%16)*32;
                // A32B DRAM read is legal when endpoint modulo64 agrees.
                // Source-head parity is constant, destination-user parity is
                // not, so only the matching half can bypass staging.
                if((src&63u)==((np+dst)&63u)) {
                    noc.async_read(norm,n,32,{.page_id=source_tile,.offset_bytes=src},{.offset_bytes=dst});
                } else {
                    noc.async_read(norm,CoreLocalMem<uint32_t>(sb+user*128+face*64),64,
                        {.page_id=source_tile,.offset_bytes=src&~63u},{});
                }
            }
        }
        for(uint32_t face=0;face<2;face++)
            noc.async_read(weight,CoreLocalMem<uint32_t>(sb+4096+face*64),64,
                {.page_id=column,.offset_bytes=face*512},{});
        noc.async_read(packed,gate,2048,{.page_id=320+tile},{.offset_bytes=0});
        noc.async_read_barrier();
        // Local NoC copies replace all baby-RISC per-element loops.
        for(uint32_t user=0;user<batch;user++) for(uint32_t face=0;face<2;face++) {
            const uint32_t row=(user*48+head)%32;
            const uint32_t src=((row/16)*512+face*256+(row%16)*16)*2;
            const uint32_t dst=(user/16)*1024+face*512+(user%16)*32;
            if((src&63u)!=((np+dst)&63u))
                tt::data_movement::common::tt_memmove<false,true,true,32>(
                    noc,np+dst,sb+user*128+face*64+(src&63u),32);
        }
        // Only gamma row0 is needed: compute broadcasts it in FP32 DST.
        for(uint32_t face=0;face<2;face++)
            tt::data_movement::common::tt_memmove<false,true,true,32>(
                noc,w.get_write_ptr()+face*512,sb+4096+face*64,32);
        for(uint32_t half=0;half<2;half++) {
            const uint32_t first=batch>half*16?batch:half*16,end=(half+1)*16;
            if(first<end) for(uint32_t face=0;face<2;face++) {
                const uint32_t dst=half*1024+face*512+(first%16)*32;
                noc.async_write_zeros(n,(end-first)*32,{.offset_bytes=dst});
                noc.async_write_zeros(gate,(end-first)*32,{.offset_bytes=dst});
            }
        }
        // BH local zero uses read-loopback; this completes copies and pad zeros.
        noc.write_zeros_l1_barrier();
        n.push_back(1); w.push_back(1); gate.push_back(1);
    }
}
"""

_affinev3__GDN_AFFINE_GATE_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer out(4);
    for(uint32_t tile=start;tile<start+count;tile++) {
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {.offset_bytes=0},{.page_id=tile});
        noc.async_write_barrier(); out.pop_front(1);
    }
}
"""

_affinev3__GDN_AFFINE_GATE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0,4); copy_init(0);
    CircularBuffer norm(0),weight(1),gate(2),out(4);
    for(uint32_t tile=0;tile<count;tile++) {
        norm.wait_front(1); weight.wait_front(1); gate.wait_front(1);
        out.reserve_back(1); tile_regs_acquire();
        copy_tile(0,0,0); copy_tile(1,0,1);
        sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row<true>(0,1);
        // Match the standalone BF16 norm-weight multiplication boundary.
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        copy_tile(2,0,2);
        // Original gate is cast toFP32 before fused SILU/multiply.
        silu_tile_init(); silu_tile<true>(2);
        mul_binary_tile_init(); mul_binary_tile(2,0,0);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,4); tile_regs_release();
        out.push_back(1); norm.pop_front(1); weight.pop_front(1); gate.pop_front(1);
    }
}
"""


_gate178_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto pa=TensorAccessorArgs<0>();
    constexpr auto da=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto na=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    auto packed=TensorAccessor(pa,get_arg_val<uint32_t>(0),2048);
    auto bias=TensorAccessor(da,get_arg_val<uint32_t>(1),4096);
    auto neg=TensorAccessor(na,get_arg_val<uint32_t>(2),4096);
    const uint32_t task=get_arg_val<uint32_t>(3),user=task/2,col=task%2;
    Noc noc; CircularBuffer a(0),b(1),d(2),n(3),scratch(6);
    a.reserve_back(1); b.reserve_back(1); d.reserve_back(1); n.reserve_back(1); scratch.reserve_back(1);
    noc.async_write_zeros(a,2048);noc.async_write_zeros(b,4096);
    noc.async_write_zeros(d,4096);noc.async_write_zeros(n,4096);
    noc.write_zeros_l1_barrier();
    const uint32_t sb=(scratch.get_write_ptr()+63u)&~63u;
    for(uint32_t face=0;face<2;face++) {
        const uint32_t src=(user/16)*1024+face*512+(user%16)*32;
        noc.async_read(packed,CoreLocalMem<uint32_t>(sb+face*64),64,
            {.page_id=514+col,.offset_bytes=src&~63u},{});
        noc.async_read(packed,CoreLocalMem<uint32_t>(sb+128+face*64),64,
            {.page_id=512+col,.offset_bytes=src&~63u},{});
        noc.async_read(bias,d,64,{.page_id=col,.offset_bytes=face*1024},{.offset_bytes=face*1024});
        noc.async_read(neg,n,64,{.page_id=col,.offset_bytes=face*1024},{.offset_bytes=face*1024});
    }
    noc.async_read_barrier();
    // Exact bitwise BF16 widening, only32 values per tile and no floating math.
    auto ap=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a.get_write_ptr());
    auto bp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b.get_write_ptr());
    auto src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sb);
    for(uint32_t face=0;face<2;face++) for(uint32_t i=0;i<16;i++) {
        ap[face*256+i]=src[face*32+(user%2)*16+i];
        bp[face*256+i]=uint32_t(src[64+face*32+(user%2)*16+i])<<16;
    }
    a.push_back(1);b.push_back(1);d.push_back(1);n.push_back(1);
}
"""
_gate178_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ba=TensorAccessorArgs<0>();
    constexpr auto ga=TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    auto beta=TensorAccessor(ba,get_arg_val<uint32_t>(0),2048);
    auto decay=TensorAccessor(ga,get_arg_val<uint32_t>(1),4096);
    const uint32_t task=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer b(4),g(5);b.wait_front(1);g.wait_front(1);
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(b),beta,2048,{}, {.page_id=task});
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(g),decay,4096,{}, {.page_id=task});
    noc.async_write_barrier();b.pop_front(1);g.pop_front(1);
}
"""
_gate178_COMPUTE = r"""
// Required by native softplus: otherwise it selects the BF16 polynomial.
#define INP_FLOAT32 1
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
namespace ckernel {
ALWI void native_bf16_sigmoid_in_fp32_dst(uint32_t idst) {
    // Native BF16 sigmoid math/RNE, with FP32 DST storage.
    // Each SFPU vector covers4rows x8columns. Two iterations cover16
    // face columns; VectorMode::R visits both top faces (logical32columns).
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_sigmoid,(false,false,2),idst,VectorMode::R));
}
ALWI void native_fp32_softplus_row(uint32_t idst) {
    // Same native polynomial and constants; Face0/1, both 8-column groups in the first 4-row band.
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_softplus,(false,true,2),
        idst,VectorMode::R,0x3f800000u,0x3f800000u,0x41a00000u));
}
}
void kernel_main() {
    compute_kernel_hw_startup(0,2,5);
    constexpr uint32_t stage=get_compile_time_arg_val(0);
    add_init(0,2);
    CircularBuffer a(0),b(1),d(2),n(3),beta(4),g(5);
    a.wait_front(1);b.wait_front(1);d.wait_front(1);n.wait_front(1);
    beta.reserve_back(1);g.reserve_back(1);tile_regs_acquire();
    // Native ADD defaults to FPU for mixed BF16 / FP32 input. Preserve its
    // source-register conversion, instead of silently upgrading this boundary.
    add_tiles(0,2,0,0,0);
    if constexpr(stage!=1) {
        softplus_tile_init();native_fp32_softplus_row(0);
    }
    reconfig_data_format_srca(0,3);copy_init(3);
    if constexpr(stage==0) {
        copy_tile(3,0,1);mul_binary_tile_init();mul_binary_tile(1,0,0);
    }
    copy_tile(1,0,1);sigmoid_tile_init<false>();native_bf16_sigmoid_in_fp32_dst(1);
    tile_regs_commit();tile_regs_wait();
    pack_tile(0,5);pack_reconfig_data_format(5,4);pack_tile(1,4);tile_regs_release();
    g.push_back(1);beta.push_back(1);a.pop_front(1);b.pop_front(1);d.pop_front(1);n.pop_front(1);
}
"""


_norm_zero_meanFOLD_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const uint32_t block=get_arg_val<uint32_t>(1);
    Noc noc; CircularBuffer data(0); data.reserve_back(5);
    for(uint32_t chunk=0;chunk<5;chunk++)
        noc.async_read(x,data,2048,{.page_id=chunk*32+block},{.offset_bytes=chunk*2048});
    noc.async_read_barrier(); data.push_back(5);
}
"""
_norm_zero_meanFOLD_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto fa=TensorAccessorArgs<0>();
    const auto folded=TensorAccessor(fa,get_arg_val<uint32_t>(0),4096);
    Noc noc; CircularBuffer data(3); data.wait_front(1);
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(data),folded,4096,
        {.offset_bytes=0},{.page_id=get_arg_val<uint32_t>(1)});
    noc.async_write_barrier(); data.pop_front(1);
}
"""
_norm_zero_meanFOLD_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    compute_kernel_hw_startup(0,0,3); copy_init(0);
    CircularBuffer x(0),out(3); x.wait_front(5); out.reserve_back(1);
    tile_regs_acquire();
    // Identical scalar arithmetic/order to the incumbent's five-tile fold.
    for(uint32_t i=0;i<5;i++) {
        const uint32_t work=i?2:0;
        copy_tile(0,i,work); copy_tile(0,i,1);
        mul_binary_tile_init(); mul_binary_tile(work,1,work);
        if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
    }
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,3); tile_regs_release();
    out.push_back(1); x.pop_front(5);
}
"""
_norm_zero_meanINVERSE_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto fa=TensorAccessorArgs<0>();
    const auto folded=TensorAccessor(fa,get_arg_val<uint32_t>(0),4096);
    const uint32_t user=get_arg_val<uint32_t>(1);
    Noc noc; CircularBuffer dense(0); dense.reserve_back(1);
    for(uint32_t block=0;block<32;block++) for(uint32_t face=0;face<2;face++) {
        // Folded channel d=block*32+face*16 becomes row=block,col=face*16
        // in the same dense face-major tile the old reader constructed.
        const uint32_t source=((user/16)*512+face*256+(user%16)*16)*4;
        const uint32_t target=((block/16)*512+face*256+(block%16)*16)*4;
        noc.async_read(folded,dense,64,{.page_id=block,.offset_bytes=source},{.offset_bytes=target});
    }
    noc.async_read_barrier(); dense.push_back(1);
}
"""
_norm_zero_meanINVERSE_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ia=TensorAccessorArgs<0>();
    const auto inverse=TensorAccessor(ia,get_arg_val<uint32_t>(0),4096);
    const uint32_t user=get_arg_val<uint32_t>(1),batch=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer data(2),zero(4);
    if(user+1==batch && batch<32) {
        zero.reserve_back(1);
        volatile tt_l1_ptr uint32_t* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
        for(uint32_t i=0;i<1024;i++) z[i]=0;
        zero.push_back(1); zero.wait_front(1);
    }
    data.wait_front(1);
    for(uint32_t face=0;face<2;face++) {
        const uint32_t target=((user/16)*512+face*256+(user%16)*16)*4;
        // Only col0 is consumed, but initialize the entire row without
        // read-modify-write or overlap with another user's output.
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(data),inverse,64,
            {.offset_bytes=face*1024},{.page_id=0,.offset_bytes=target});
    }
    if(user+1==batch && batch<32) {
        for(uint32_t row=batch;row<32;) {
            const uint32_t end=(row/16+1)*16;
            for(uint32_t face=0;face<2;face++) {
                const uint32_t target=((row/16)*512+face*256+(row%16)*16)*4;
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(zero),inverse,(end-row)*64,
                    {.offset_bytes=0},{.page_id=0,.offset_bytes=target});
            }
            row=end;
        }
    }
    noc.async_write_barrier(); data.pop_front(1);
    if(user+1==batch && batch<32) zero.pop_front(1);
}
"""
_norm_zero_meanINVERSE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t EPS_BITS=get_compile_time_arg_val(0), RECIPROCAL_BITS=get_compile_time_arg_val(1);
    compute_kernel_hw_startup(0,0,2); copy_init(0);
    CircularBuffer dense(0),inverse(2); dense.wait_front(1); inverse.reserve_back(1);
    tile_regs_acquire(); copy_tile(0,0,0);
    // The tile is bit-identical to DST0 after the incumbent's five-tile fold.
    // Keep both axis order and all scalar operations exactly as in that helper.
    sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_COL>(0,1,1);
    sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
    binop_with_scalar_tile_init(); mul_unary_tile(0,RECIPROCAL_BITS);
    binop_with_scalar_tile_init(); add_unary_tile(0,EPS_BITS);
    rsqrt_tile_init(); rsqrt_tile(0);
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,2); tile_regs_release();
    inverse.push_back(1); dense.pop_front(1);
}
"""
_norm_zero_meanOUTPUT_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ga=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ia=TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto gamma=TensorAccessor(ga,get_arg_val<uint32_t>(1),4096);
    const auto inverse=TensorAccessor(ia,get_arg_val<uint32_t>(2),4096);
    const uint32_t block=get_arg_val<uint32_t>(3), count=get_arg_val<uint32_t>(4);
    Noc noc; CircularBuffer data(0),weight(1),inv(2);
    data.reserve_back(count); weight.reserve_back(count); inv.reserve_back(1);
    // Only col0 of left faces0/2 participates in column normalization.
    // Keep their original offsets; the two right faces are never consumed.
    noc.async_read(inverse,inv,1024,{.page_id=0,.offset_bytes=0},{.offset_bytes=0});
    noc.async_read(inverse,inv,1024,{.page_id=0,.offset_bytes=2048},{.offset_bytes=2048});
    for(uint32_t chunk=0;chunk<count;chunk++) {
        const uint32_t page=block+chunk;
        noc.async_read(x,data,2048,{.page_id=page},{.offset_bytes=chunk*2048});
        noc.async_read(gamma,weight,4096,{.page_id=page},{.offset_bytes=chunk*4096});
    }
    noc.async_read_barrier(); inv.push_back(1); data.push_back(count); weight.push_back(count);
}
"""
_norm_zero_meanOUTPUT_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t block=get_arg_val<uint32_t>(1),batch=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer data(3); data.wait_front(count);
    volatile tt_l1_ptr uint32_t* raw=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data.get_read_ptr());
    for(uint32_t chunk=0;chunk<count;chunk++) {
        for(uint32_t row=batch;row<32;row++) for(uint32_t face=0;face<2;face++) {
            const uint32_t word=chunk*512+(row/16)*256+face*128+(row%16)*8;
            for(uint32_t j=0;j<8;j++) raw[word+j]=0;
        }
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(data),output,2048,
            {.offset_bytes=chunk*2048},{.page_id=block+chunk});
    }
    noc.async_write_barrier(); data.pop_front(count);
}
"""
_norm_zero_meanOUTPUT_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
// Specialize a known +0 mean without changing the native centering/multiply.
namespace ckernel {
#ifdef TRISC_MATH
namespace sfpu {
inline void _hidden_normalize_zero_(uint32_t data,uint32_t inverse,uint32_t output) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t db=data*DEST_TILE_SIZE_RAW,ib=inverse*DEST_TILE_SIZE_RAW,ob=output*DEST_TILE_SIZE_RAW;
    // The incumbent broadcasts a +0 tile, then negates it into LREG7.
    TTI_SFPMOV(0,p_sfpu::LCONST_0,p_sfpu::LREG7,1 /* negate */);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        TT_SFPLOAD(LREG_DATA0,IM,ADDR_MOD_7,db+left);
        TT_SFPLOAD(LREG_DATA1,IM,ADDR_MOD_7,db+left+ODD_COLS_OFFSET);
        TT_SFPLOAD(LREG_DATA2,IM,ADDR_MOD_7,db+right);
        TT_SFPLOAD(LREG_DATA3,IM,ADDR_MOD_7,db+right+ODD_COLS_OFFSET);
        // Identical native inverse broadcast, data+(-0), multiply and stores.
        _finish_col_normalize_row_band_(ib+left,ob+left,ob+left+ODD_COLS_OFFSET,
                                        ob+right,ob+right+ODD_COLS_OFFSET);
    }
}
} // namespace sfpu
#endif
ALWI void hidden_normalize_zero(uint32_t data,uint32_t inverse,uint32_t output) {
    sfpu_bcast_col_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _hidden_normalize_zero_,data,inverse,output,VectorMode::None)));
}
} // namespace ckernel

void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(2,2,3);
    CircularBuffer x(0),gamma(1),inverse(2),out(3);
    x.wait_front(count); gamma.wait_front(count); inverse.wait_front(1); out.reserve_back(count);
    for(uint32_t chunk=0;chunk<count;chunk++) {
        tile_regs_acquire();
        reconfig_data_format_srca(0); copy_init(0); copy_tile(0,chunk,0);
        reconfig_data_format_srca(2); copy_init(2); copy_tile(2,0,2);
        // Preserve even the incumbent zero-subtract before inverse multiply.
        hidden_normalize_zero(0,2,0);
        copy_tile(1,chunk,1);
        sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row(0,1);
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,3,chunk); tile_regs_release();
    }
    out.push_back(count); x.pop_front(count); gamma.pop_front(count); inverse.pop_front(1);
}
"""


def _norm_zero_meanhidden_rmsnorm_sfpu(x, gamma, *, epsilon=1e-6, output=None, scratch=None):
    """Optional scratch=(folded, inverse), FP32 interleaved TILE [1,32,1024]/[1,32,32].
    Default scratch lives in L1; explicitly supplied DRAM scratch remains supported.

    Supply output and both scratch buffers to avoid tensor allocation. Their
    addresses are registered and rebound in each generic-op invocation.
    """
    assert len(x.shape) == 3 and x.shape[0] == 1 and 1 <= x.shape[1] <= 32 and x.shape[2] == 5120
    assert x.dtype == ttnn.bfloat16 and tuple(gamma.shape) == (1, 1, 5120) and gamma.dtype == ttnn.float32
    assert epsilon > 0

    def allocate(shape, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.empty(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=x.device(), memory_config=memory)

    if output is None:
        output = allocate(tuple(x.shape), ttnn.bfloat16)
    if scratch is None:
        scratch = (
            allocate((1, 32, 1024), ttnn.float32, ttnn.L1_MEMORY_CONFIG),
            allocate((1, 32, 32), ttnn.float32, ttnn.L1_MEMORY_CONFIG),
        )
    assert len(scratch) == 2
    folded, inverse = scratch
    assert tuple(output.shape) == tuple(x.shape) and output.dtype == ttnn.bfloat16
    assert tuple(folded.shape) == (1, 32, 1024) and folded.dtype == ttnn.float32
    assert tuple(inverse.shape) == (1, 32, 32) and inverse.dtype == ttnn.float32
    tensors = (x, gamma, output, folded, inverse)
    for tensor in tensors:
        assert tensor.layout == ttnn.TILE_LAYOUT
    for tensor in (x, gamma, output):
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for tensor in (folded, inverse):
        assert tensor.memory_config() in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
    assert len({t.buffer_address() for t in tensors}) == len(tensors)
    grid = x.device().compute_with_storage_grid_size()
    assert grid.x * grid.y >= 32
    output_cores = min(80, grid.x * grid.y)
    output_tiles = (160 + output_cores - 1) // output_cores

    def output_count(c):
        return 160 // output_cores + (c < 160 % output_cores)

    def output_start(c):
        return c * (160 // output_cores) + min(c, 160 % output_cores)

    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(output_cores)]
    bits = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]

    def accessor_args(*tensors):
        args = []
        for t in tensors:
            args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return args

    def launch(sources, runtime, ct, sizes, active, tensors):
        selected = coords[:active]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in selected])
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for index, count, dtype in sizes:
            if dtype == ttnn.float32:
                modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        configs = [ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor(), config]
        kernels = []
        for source, args, compile_args, kernel_config in zip(sources, runtime, ct, configs):
            rt = ttnn.RuntimeArgs()
            for index, c in enumerate(selected):
                rt[c.x][c.y] = args(index)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=source,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=rt,
                    config=kernel_config,
                )
            )
        cbs = []
        for index, count, dtype in sizes:
            page = 2048 if dtype == ttnn.bfloat16 else 4096
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=count * page,
                    core_ranges=ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
                )
            )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    launch(
        [_norm_zero_meanFOLD_READER, _norm_zero_meanFOLD_WRITER, _norm_zero_meanFOLD_COMPUTE],
        [lambda c: [x.buffer_address(), c], lambda c: [folded.buffer_address(), c], lambda c: []],
        [accessor_args(x), accessor_args(folded), []],
        [(0, 5, ttnn.bfloat16), (3, 1, ttnn.float32)],
        32,
        [x, folded],
    )
    launch(
        [_norm_zero_meanINVERSE_READER, _norm_zero_meanINVERSE_WRITER, _norm_zero_meanINVERSE_COMPUTE],
        [lambda u: [folded.buffer_address(), u], lambda u: [inverse.buffer_address(), u, x.shape[1]], lambda u: []],
        [accessor_args(folded), accessor_args(inverse), [bits(epsilon), bits(1 / 5120)]],
        [(0, 1, ttnn.float32), (2, 1, ttnn.float32), (4, 1, ttnn.float32)],
        x.shape[1],
        [folded, inverse],
    )
    launch(
        [_norm_zero_meanOUTPUT_READER, _norm_zero_meanOUTPUT_WRITER, _norm_zero_meanOUTPUT_COMPUTE],
        [
            lambda c: [
                x.buffer_address(),
                gamma.buffer_address(),
                inverse.buffer_address(),
                output_start(c),
                output_count(c),
            ],
            lambda c: [output.buffer_address(), output_start(c), x.shape[1], output_count(c)],
            lambda c: [output_count(c)],
        ],
        [accessor_args(x, gamma, inverse), accessor_args(output), []],
        [
            (0, output_tiles, ttnn.bfloat16),
            (1, output_tiles, ttnn.float32),
            (2, 1, ttnn.float32),
            (3, output_tiles, ttnn.bfloat16),
        ],
        output_cores,
        [x, gamma, inverse, output],
    )
    return output
