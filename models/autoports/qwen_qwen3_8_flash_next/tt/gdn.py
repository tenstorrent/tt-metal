# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next GDN, with device-only convolution and FP32 recurrence.

Inputs/outputs are BF16 [B,S,2560] or [1,B,S,2560]. A loaded instance owns B
independent request lanes. ``current_positions`` is an optional device tensor
[B,32] (first column used): a zero position resets that lane before consuming
input; nonzero positions continue it. Positions are not rotary inputs to GDN.
Optional UINT32 RM ``slots`` [B,32] maps lanes to persistent state-bank slots;
the first column holds a unique request ID below setup-time ``max_requests``.
Without positions, callers explicitly reset before fresh requests. All rows in
one call have the same logical S. There is no runtime host tensor conversion.

Prefill splits arbitrary logical lengths into bounded chunks; padding is solely
TT tensor storage padding and is never processed as a token. Decode consumes one
token per lane, writes both states into persistent buffers, and can be traced.
Compile all intended chunk/tail paths before capturing any live decode trace.
Slot IDs may change between trace replays. A paged-attention page table is not
interpreted as a recurrent-state slot table. Do not mix fixed-lane and slotted
continuation for one request: these use distinct state-ownership contracts.
"""

from pathlib import Path

import ttnn
from models.common.lightweightmodule import LightweightModule


class GatedDeltaNet(LightweightModule):
    @classmethod
    def from_state_dict(
        cls, state_dict, *, hf_config, layer_idx, mesh_device, batch_size=1, chunk_size=128, max_requests=None
    ):
        """Load canonical ``linear_attn.*`` HF weights (or a stripped substate).

        Torch operations and upload occur only here. No random/synthetic weight
        substitutions are made. BF16 projections retain original checkpoint
        precision; decay parameters and recurrent state use FP32.
        """
        import torch

        cfg = getattr(hf_config, "text_config", hf_config)
        actual = (
            cfg.hidden_size,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            cfg.linear_conv_kernel_dim,
        )
        if actual != (2560, 16, 48, 128, 128, 4):
            raise ValueError(f"Unexpected target GDN dimensions: {actual}")
        if cfg.hidden_act != "silu" or cfg.output_gate_type != "sigmoid":
            raise ValueError("This target requires SiLU convolution and sigmoid output gate")
        if batch_size < 1 or chunk_size < 1:
            raise ValueError("batch_size and chunk_size must be positive")
        self = cls()
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.norm_eps = cfg.rms_norm_eps
        self.device = mesh_device
        self.compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        prefixes = (
            "",
            f"model.language_model.layers.{layer_idx}.linear_attn.",
            f"model.layers.{layer_idx}.linear_attn.",
            f"layers.{layer_idx}.linear_attn.",
        )
        prefix = next((p for p in prefixes if p + "in_proj_qkv.weight" in state_dict), None)
        if prefix is None:
            raise KeyError(f"No GDN weights for layer {layer_idx}")

        def get(name):
            return state_dict[prefix + name]

        def upload(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.qkv_weight = upload(get("in_proj_qkv.weight").T)
        self.z_weight = upload(get("in_proj_z.weight").T)
        self.a_weight = upload(get("in_proj_a.weight").T)
        self.b_weight = upload(get("in_proj_b.weight").T)
        self.out_weight = upload(get("out_proj.weight").T)
        self.conv_taps = [upload(get("conv1d.weight")[:, 0, tap].reshape(1, 1, 10240)) for tap in range(4)]
        self.a_neg = upload(-get("A_log").float().exp().reshape(1, 1, 48), ttnn.float32)
        self.dt_bias = upload(get("dt_bias").float().reshape(1, 1, 48), ttnn.float32)
        self.norm_weight = upload(get("norm.weight").reshape(1, 1, 1, 128))
        self.recurrent_state = upload(torch.zeros(batch_size, 48, 128, 128), ttnn.float32)
        self.conv_state = upload(torch.zeros(batch_size, 3, 10240))
        self.zero_recurrent = upload(torch.zeros(batch_size, 48, 128, 128), ttnn.float32)
        self.zero_conv = upload(torch.zeros(batch_size, 3, 10240))
        self.max_requests = max_requests or batch_size
        self.recurrent_bank = upload(torch.zeros(self.max_requests, 48, 128, 128), ttnn.float32)
        self.conv_bank = upload(torch.zeros(self.max_requests, 3, 10240))
        return self

    def _route_state(self, slots, positions, *, store):
        """Gather/scatter complete raw tiles using device request IDs.

        Unique slot IDs within each batch are a caller precondition. Position
        zero initializes the selected lane on gather; scatter updates only the
        selected slot. Banks and work buffers are persistent across replay.
        """
        for bank, work, tiles, tile_bytes in (
            (self.recurrent_bank, self.recurrent_state, 48 * 4 * 4, 4096),
            (self.conv_bank, self.conv_state, 320, 2048),
        ):
            grid = self.device.compute_with_storage_grid_size()
            coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(self.batch_size)]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            tensors = [bank, work, slots, positions]
            args = [tiles, tile_bytes, self.max_requests, int(store)]
            for tensor in tensors:
                args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
            cb = ttnn.CBDescriptor(
                total_size=tile_bytes + 256,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=tile_bytes + 256)
                ],
            )
            kernel = ttnn.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name("kernels") / "gdn_state.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=args,
                runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
                config=ttnn.ReaderConfigDescriptor(),
            )
            ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))

    def reset_state_inplace(self):
        """Reset all fixed lanes using device copies; state addresses stay stable."""
        ttnn.copy(self.zero_recurrent, self.recurrent_state)
        ttnn.copy(self.zero_conv, self.conv_state)

    def _reset_fresh_lanes(self, current_positions):
        if current_positions is None:
            return
        positions = ttnn.slice(current_positions, [0, 0], [self.batch_size, 1])
        positions = ttnn.to_layout(ttnn.typecast(positions, ttnn.float32), ttnn.TILE_LAYOUT)
        keep = ttnn.ne(positions, 0.0)
        rec_keep = ttnn.reshape(keep, [self.batch_size, 1, 1, 1])
        conv_keep = ttnn.reshape(keep, [self.batch_size, 1, 1])
        ttnn.copy(ttnn.multiply(self.recurrent_state, rec_keep), self.recurrent_state)
        ttnn.copy(ttnn.multiply(self.conv_state, conv_keep), self.conv_state)

    def _linear(self, x, weight):
        return ttnn.linear(x, weight, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    @staticmethod
    def _heads(x, batch, seq, heads):
        # Explicit RM reshape prevents tile-padding from becoming head data.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.reshape(x, [batch, seq, heads, 128]), ttnn.TILE_LAYOUT)

    @staticmethod
    def _l2norm(x):
        x = ttnn.typecast(x, ttnn.float32)
        squared_norm = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(squared_norm, 1e-6)))

    def _recurrence(self, q, k, v, g, beta):
        """FP32 state carry without the native fused op's TF32 source reloads.

        Float32 storage alone is insufficient for this target's long-lived
        state: the native fused recurrence narrows state during every decay
        and update seed. SFPU binary operations and accurate reductions keep
        all recurrent arithmetic in FP32. All slicing/loops depend only on
        static tensor shapes, so one-token execution is trace compatible.
        """
        b, length, _, d = q.shape
        q = ttnn.multiply(ttnn.repeat_interleave(q, 3, dim=2), 128**-0.5)
        k = ttnn.repeat_interleave(k, 3, dim=2)
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        decay = ttnn.exp(ttnn.permute(g, (0, 2, 1)))
        beta = ttnn.permute(beta, (0, 2, 1))
        state = self.recurrent_state
        outputs = []
        for token in range(length):

            def row(tensor):
                return ttnn.slice(tensor, [0, 0, token, 0], [b, 48, token + 1, d])

            qt, kt, vt = row(q), row(k), row(v)

            def scalar(tensor):
                value = ttnn.slice(tensor, [0, 0, token], [b, 48, token + 1])
                return ttnn.to_layout(
                    ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), [b, 48, 1, 1]), ttnn.TILE_LAYOUT
                )

            state = ttnn.multiply(state, scalar(decay))
            kcol = ttnn.transpose(kt, -1, -2)
            read = ttnn.sum(
                ttnn.multiply(state, kcol),
                dim=-2,
                keepdim=True,
                compute_kernel_config=self.compute,
                fast_and_approximate_mode=False,
            )
            delta = ttnn.multiply(ttnn.subtract(vt, read), scalar(beta))
            state = ttnn.add(state, ttnn.multiply(kcol, delta))
            out = ttnn.sum(
                ttnn.multiply(state, ttnn.transpose(qt, -1, -2)),
                dim=-2,
                keepdim=True,
                compute_kernel_config=self.compute,
                fast_and_approximate_mode=False,
            )
            outputs.append(out)
        output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        return ttnn.permute(output, (0, 2, 1, 3)), state

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
        q = self._heads(ttnn.slice(conv, [0, 0, 0], [b, s, 2048]), b, s, 16)
        k = self._heads(ttnn.slice(conv, [0, 0, 2048], [b, s, 4096]), b, s, 16)
        v = self._heads(ttnn.slice(conv, [0, 0, 4096], [b, s, 10240]), b, s, 48)
        a = ttnn.typecast(self._linear(x, self.a_weight), ttnn.float32)
        g = ttnn.multiply(self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)))
        beta = ttnn.typecast(ttnn.sigmoid(self._linear(x, self.b_weight)), ttnn.float32)
        q, k, v = self._l2norm(q), self._l2norm(k), ttnn.typecast(v, ttnn.float32)
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

    def prefill_forward(self, hidden_states, *, current_positions=None, slots=None):
        rank4 = len(hidden_states.shape) == 4
        x = hidden_states
        if rank4:
            if x.shape[0] != 1:
                raise ValueError("Rank-4 input must have shape [1,B,S,2560]")
            x = ttnn.reshape(x, list(x.shape)[1:])
        if len(x.shape) != 3 or x.shape[0] != self.batch_size or x.shape[2] != 2560 or x.shape[1] < 1:
            raise ValueError("Expected [B,S,2560] with fixed B and positive logical S")
        if slots is not None:
            if current_positions is None:
                raise ValueError("Slot routing requires device current_positions")
            for tensor in (slots, current_positions):
                if (
                    tensor.dtype != ttnn.uint32
                    or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
                    or list(tensor.shape) != [self.batch_size, 32]
                ):
                    raise ValueError("Slot IDs and positions must be UINT32 ROW_MAJOR [B,32]")
            self._route_state(slots, current_positions, store=False)
        else:
            self._reset_fresh_lanes(current_positions)
        b, length, _ = x.shape
        outputs = []
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            chunk = ttnn.to_layout(ttnn.slice(x, [0, start, 0], [b, stop, 2560]), ttnn.TILE_LAYOUT)
            outputs.append(self._chunk(chunk))
        output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)
        if slots is not None:
            self._route_state(slots, current_positions, store=True)
        return ttnn.reshape(output, [1, b, length, 2560]) if rank4 else output

    def decode_forward(self, hidden_states, *, current_positions=None, slots=None):
        if hidden_states.shape[-2] != 1:
            raise ValueError("decode_forward requires exactly one token per lane")
        return self.prefill_forward(hidden_states, current_positions=current_positions, slots=slots)
