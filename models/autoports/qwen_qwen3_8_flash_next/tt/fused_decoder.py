# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device Qwen3.8-Flash-Next decoder, including all three layer kinds.

Inputs/outputs are BF16 TILE [1,B,S,10240] (four residual streams). Decode
requires S=1. Positions and request slots are UINT32 ROW_MAJOR [B,32], column
zero carrying the absolute first-token position and state-bank slot. Position
zero starts a fresh request. All lanes have the same logical S; padding is an
internal TT tile property, never a token. PLE additionally takes UINT32 RM token
IDs [B,S]. Indexed attention takes an INT32 RM page table [B,num_pages].

``prepare(sequence_lengths)`` allocates PLE workspaces before warmup/capture.
Prefill splits arbitrary logical S into bounded chunks and preserves every
output. A caller may stream chunks with absolute positions to bound input and
output storage at the full advertised context. Runtime never reads host data.
"""

from collections.abc import Mapping

import ttnn
from models.common.lightweightmodule import LightweightModule

from .gdn import GatedDeltaNet
from .hyperconnection import HyperConnection
from .moe import MoE
from .ple import PLE, _conv_taps, _mask_padding, lookup_rows, ngram_indices
from .qsa import QSA


class _PrefixView(Mapping):
    def __init__(self, state, prefix):
        self.state, self.prefix = state, prefix

    def __getitem__(self, key):
        return self.state[self.prefix + key]

    def __iter__(self):
        return (k[len(self.prefix) :] for k in self.state if k.startswith(self.prefix))

    def __len__(self):
        return sum(1 for _ in self)


class FusedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch_size=1,
        max_requests=None,
        chunk_size=128,
        cache_dir=None,
    ):
        import torch  # Setup only.

        obj = cls()
        cfg = getattr(hf_config, "text_config", hf_config)
        if (cfg.hidden_size, cfg.hc_count) != (2560, 4):
            raise ValueError("Expected the target Qwen3.8-Flash-Next configuration")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        prefix = f"model.language_model.layers.{layer_idx}."
        state = _PrefixView(state_dict, prefix) if prefix + "mlp.gate.weight" in state_dict else state_dict
        obj.batch_size, obj.chunk_size = batch_size, chunk_size
        obj.context = cfg.max_position_embeddings
        obj.mesh = mesh_device
        obj.kind = cfg.layer_types[layer_idx]
        obj.attn_hyper = FusedHyperConnection.from_state_dict(
            state, prefix="attn_hyper_connection", hf_config=cfg, mesh_device=mesh_device
        )
        obj.mlp_hyper = FusedHyperConnection.from_state_dict(
            state, prefix="mlp_hyper_connection", hf_config=cfg, mesh_device=mesh_device
        )
        obj.mlp = FusedMoE.from_state_dict(state, hf_config=cfg, mesh_device=mesh_device)
        requests = max_requests or batch_size
        if obj.kind == "linear_attention":
            obj.attention = FusedGatedDeltaNet.from_state_dict(
                _PrefixView(state, "linear_attn."),
                hf_config=cfg,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=requests,
                chunk_size=chunk_size,
            )
        else:
            obj.attention = FusedQSA.from_state_dict(
                _PrefixView(state, "self_attn."),
                hf_config=cfg,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=requests,
            )
        obj.ple = None
        if layer_idx + 1 in cfg.ple_layer_ids:
            obj.ple = FusedPLE.from_state_dict(
                state, hf_config=cfg, layer_idx=layer_idx, prefix="ple.", mesh_device=mesh_device, cache_dir=cache_dir
            )
            obj.ple_cache = obj.ple.make_cache(max_requests=requests)
        obj.lengths = {}
        obj.prepare([1, chunk_size])
        return obj

    def prepare(self, sequence_lengths):
        """Prepare every chunk/tail shape before a live trace is captured."""
        import torch  # Explicit setup boundary.

        for length in sequence_lengths:
            if not 1 <= length <= self.context:
                raise ValueError("Sequence length is outside the advertised context")
            for start in range(0, length, self.chunk_size):
                size = min(self.chunk_size, length - start)
                if size not in self.lengths:
                    host = torch.zeros(self.batch_size, 32, dtype=torch.int32)
                    host[:, 0] = size
                    self.lengths[size] = ttnn.from_torch(
                        host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh
                    )
                    if self.ple is not None:
                        self.ple.prepare(batch=self.batch_size, physical_seq_len=size)

    def _chunk(self, x, *, positions, slots, tokens, page_table, decode):
        size = x.shape[2]
        if self.ple is not None:
            x = ttnn.add(
                x,
                self.ple.forward(
                    x,
                    tokens=tokens,
                    positions=positions,
                    slots=slots,
                    valid_lengths=self.lengths[size],
                    cache=self.ple_cache,
                ),
            )
        mixed, inject = self.attn_hyper.read(x)
        forward = self.attention.decode_forward if decode else self.attention.prefill_forward
        if self.kind == "linear_attention":
            update = forward(mixed, current_positions=positions, slots=slots)
        else:
            update = forward(mixed, current_pos=positions, page_table=page_table, slots=slots)
        x = self.attn_hyper.write(x, update, inject)
        mixed, inject = self.mlp_hyper.read(x)
        return self.mlp_hyper.write(x, self.mlp(mixed), inject)

    def prefill_forward(self, hidden_states, *, current_positions, slots=None, tokens=None, page_table=None):
        if tuple(hidden_states.shape)[:2] != (1, self.batch_size) or hidden_states.shape[-1] != 10240:
            raise ValueError("Expected [1,B,S,10240] residual streams")
        length = hidden_states.shape[2]
        if not 1 <= length <= self.context:
            raise ValueError("Sequence length is outside the advertised context")
        if self.ple is not None and (tokens is None or slots is None):
            raise ValueError("PLE requires token IDs and request slots")
        # Reject missing setup before the first chunk can mutate request state.
        if self.ple is not None:
            sizes = {min(self.chunk_size, length - start) for start in range(0, length, self.chunk_size)}
            if not sizes.issubset(self.lengths):
                raise ValueError("Call prepare([sequence_length]) before executing this PLE prefill shape")
        outputs = []
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            position = current_positions if start == 0 else ttnn.add(current_positions, start)
            x = hidden_states[:, :, start:stop, :]
            ids = None if tokens is None else tokens[:, start:stop]
            outputs.append(
                self._chunk(x, positions=position, slots=slots, tokens=ids, page_table=page_table, decode=False)
            )
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def decode_forward(self, hidden_states, *, current_positions, slots=None, tokens=None, page_table=None):
        if tuple(hidden_states.shape) != (1, self.batch_size, 1, 10240):
            raise ValueError("Decode requires [1,B,1,10240]")
        return self._chunk(
            hidden_states, positions=current_positions, slots=slots, tokens=tokens, page_table=page_table, decode=True
        )


class FusedHyperConnection(HyperConnection):
    @classmethod
    def from_state_dict(cls, state, *, prefix, hf_config, mesh_device):
        import torch

        obj = super().from_state_dict(state, prefix=prefix, hf_config=hf_config, mesh_device=mesh_device)

        def upload(x):
            return ttnn.from_torch(x.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        obj.norm_packed = upload((1 + state[prefix + ".hc_norm.weight"].float()).reshape(1, 1, 4, obj.width))
        obj.read_packed = upload(
            torch.cat(
                (
                    state[prefix + ".input_mix_weight_down.weight"].T / obj.count,
                    torch.nn.functional.pad(state[prefix + ".block_inject_weight.weight"].T / obj.count, (0, 28)),
                ),
                dim=-1,
            )
        )
        del obj.down, obj.inject
        return obj

    def _read_prefill(self, x):
        streams = [
            ttnn.rms_norm(self._part(x, i), weight=self.norm[i], epsilon=self.eps, compute_kernel_config=self.compute)
            for i in range(self.count)
        ]
        normalized = ttnn.concat(streams, dim=-1)
        packed = ttnn.linear(normalized, self.read_packed, compute_kernel_config=self.compute)
        mix = ttnn.silu(packed[..., :320])
        mix = ttnn.linear(mix, self.up, activation="sigmoid", compute_kernel_config=self.compute)
        mixed = ttnn.multiply(self._part(mix, 0), streams[0])
        for i in range(1, self.count):
            mixed = ttnn.add(mixed, ttnn.multiply(self._part(mix, i), streams[i]))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return ttnn.multiply(mixed, 1 / self.count), inject

    def read(self, x):
        # Up to four tokens use no more padded rows than four split streams.
        # This geometry also preserves the short-tail routing accuracy.
        if x.shape[2] > 4:
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

    def write(self, residual, update, inject):
        if residual.shape[2] > 1:
            return HyperConnection.write(self, residual, update, inject)
        b, seq = tuple(residual.shape)[1:3]
        streams = ttnn.reshape(residual, (1, b * seq, self.count, self.width))
        update = ttnn.reshape(update, (1, b * seq, 1, self.width))
        inject = ttnn.reshape(inject, (1, b * seq, self.count, 1))
        result = ttnn.add(streams, ttnn.multiply(update, inject))
        return ttnn.reshape(result, residual.shape)


class _PackedMoE(MoE):
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

        gate_up = state["mlp.experts.gate_up_proj"]
        obj.gate_up = upload(gate_up.transpose(-2, -1).unsqueeze(0), expert_dtype)
        del gate_up
        obj.down = upload(state["mlp.experts.down_proj"].transpose(-2, -1).unsqueeze(0), expert_dtype)
        obj.shared_down = upload(state["mlp.shared_expert.down_proj.weight"].T)
        import torch

        obj.router_shared_packed = upload(
            torch.cat(
                [
                    state["mlp.gate.weight"].T,
                    state["mlp.shared_expert.gate_proj.weight"].T,
                    state["mlp.shared_expert.up_proj.weight"].T,
                    torch.nn.functional.pad(state["mlp.shared_expert_gate.weight"].T, (0, 31)),
                ],
                dim=-1,
            )
        )
        mask = torch.full((1, 1, 1, 32), float("-inf"))
        mask[..., : obj.topk] = 0
        obj.top_mask = upload(mask)
        obj.programs = {}
        for width in (obj.intermediate * 2, obj.hidden):
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

    def _group(self, x):
        router_shared = ttnn.linear(x, self.router_shared_packed, compute_kernel_config=self.compute)
        logits = router_shared[..., :512]
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        sparsity = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        packed = ttnn.reshape(
            self._sparse(x, self.gate_up, sparsity, self.intermediate * 2), (1, self.experts, 32, self.intermediate * 2)
        )
        gate, up = packed[..., : self.intermediate], packed[..., self.intermediate :]
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        down = self._sparse(hidden, self.down, sparsity, self.hidden, input_sparse=True)
        routing = ttnn.permute(routing, (0, 3, 2, 1))
        result = ttnn.sum(ttnn.multiply(down, routing), dim=1, keepdim=True)
        sg, su = router_shared[..., 512:1152], router_shared[..., 1152:1792]
        scale = router_shared[..., 1792:1793]
        shared = ttnn.linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.compute,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))


class FusedMoE(_PackedMoE):
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
        obj.capacity_input = ttnn.zeros(
            (1, 1, obj.experts * 32, obj.hidden), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        obj._dispatch_coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(32)]
        obj._dispatch_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in obj._dispatch_coords])
        if mesh.get_num_devices() != 1 or (obj.experts, obj.hidden) != (512, 2560):
            raise ValueError("Weighted expert reduction is qualified for single-device E512/H2560")
        indices = torch.arange(obj.experts, dtype=torch.int32).reshape(1, 1, 1, obj.experts)
        obj._reduce_indices = upload(
            indices.expand(32, 1, 1, obj.experts).contiguous(), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
        )
        obj._reduce_mapping = upload(
            torch.zeros((1, obj.experts), dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
        )
        return obj

    def __call__(self, x):
        # Caller owns padding/masking; keep batching explicit and preserve all logical rows.
        shape = tuple(x.shape)
        batch, seq = shape[1:3]
        group_fn = self._group_unified if seq == 1 else self._group
        outputs = []
        for b in range(batch):
            parts = []
            for start in range(0, seq, 32):
                valid = min(32, seq - start)
                group = x[:, b : b + 1, start : start + valid, :]
                if valid < 32:
                    group = ttnn.pad(group, [(0, 0), (0, 0), (0, 32 - valid), (0, 0)], 0)
                part = group_fn(group)
                parts.append(part[:, :, :valid, :])
            outputs.append(ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0])
        return ttnn.concat(outputs, dim=1) if len(outputs) > 1 else outputs[0]

    def _weighted_reduce(self, down, routing):
        if tuple(down.shape) != (1, self.experts, 32, self.hidden):
            raise ValueError("Expected selected expert output [1,512,32,2560]")
        if tuple(routing.shape) != (1, 1, 32, self.experts):
            raise ValueError("Expected selected routing [1,1,32,512]")
        scores = ttnn.reshape(ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT), (32, 1, 1, self.experts))
        outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            down,
            self._reduce_indices,
            self._reduce_mapping,
            reduce_dim=1,
            split_size=self.hidden,
            cluster_axis=0,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            scores_tensor=scores,
            compute_kernel_config=self.compute,
        )
        if len(outputs) != 1:
            raise RuntimeError("The whole hidden dimension must remain one output tensor")
        return outputs[0]

    def _group(self, x):
        router_shared = ttnn.linear(x, self.router_shared_packed, compute_kernel_config=self.compute)
        logits = router_shared[..., :512]
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
        result = self._weighted_reduce(down, routing)
        sg, su = router_shared[..., 512:1152], router_shared[..., 1152:1792]
        scale = router_shared[..., 1792:1793]
        shared = ttnn.linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.compute,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))

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

    def _dispatch_input(self, x, counts):
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        args = (
            ttnn.TensorAccessorArgs(source).get_compile_time_args()
            + ttnn.TensorAccessorArgs(self.capacity_input).get_compile_time_args()
            + ttnn.TensorAccessorArgs(counts).get_compile_time_args()
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=_MOE_DISPATCH_COPY,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self._dispatch_cores,
            compile_time_args=args,
            runtime_args=[
                (
                    (c.x, c.y),
                    [source.buffer_address(), self.capacity_input.buffer_address(), counts.buffer_address(), i],
                )
                for i, c in enumerate(self._dispatch_coords)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=self._dispatch_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
            for index, size, dtype in ((0, 5120, ttnn.bfloat16), (1, 2048, ttnn.uint32))
        ]
        ttnn.generic_op([source, self.capacity_input, counts], ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs))
        return self.capacity_input

    def _group_unified(self, x):
        if tuple(x.shape) != (1, 1, 32, self.hidden):
            raise ValueError("The inherited MoE caller must supply one padded 32-token group")
        router_shared = ttnn.linear(x, self.router_shared_packed, compute_kernel_config=self.compute)
        logits = router_shared[..., :512]
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        active = ttnn.max(routing, dim=2, keepdim=True) > 0
        counts = ttnn.typecast(ttnn.where(active, 32, 0), ttnn.uint32)
        counts = ttnn.reshape(ttnn.to_layout(counts, ttnn.ROW_MAJOR_LAYOUT), (self.experts,))
        dispatched = self._dispatch_input(x, counts)
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            dispatched,
            self.expert_offsets,
            counts,
            self.expert_ids,
            self.expert_gates,
            self.expert_ups,
            self.expert_downs,
            32,
        )
        down = ttnn.reshape(ttnn.typecast(down, ttnn.bfloat16), (1, self.experts, 32, self.hidden))
        # ROW_MAJOR input makes the native wrapper allocate uninitialized
        # BFP8 output. It never writes inactive regions. True conditional
        # selection clears those regions, including NaNs; multiplication by
        # zero is insufficient. Active-region non-finites remain visible to
        # correctness checks instead of being silently sanitized.
        down = ttnn.where(ttnn.permute(active, (0, 3, 2, 1)), down, 0.0)
        if self.retain_intermediates:
            self.intermediates = dict(routing=routing, counts=counts, down=down)
        result = self._weighted_reduce(down, routing)
        sg, su = router_shared[..., 512:1152], router_shared[..., 1152:1792]
        scale = router_shared[..., 1792:1793]
        shared = ttnn.linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.compute,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))


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


class FusedGatedDeltaNet(GatedDeltaNet):
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

    def _chunk(self, x):
        b, s, _ = x.shape
        padded = (s + 31) // 32 * 32
        projections = self._linear(x, self.packed_weight)
        projected = projections[..., :10240]
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
        a = ttnn.typecast(ttnn.to_layout(projections[..., 16384:16432], ttnn.TILE_LAYOUT), ttnn.float32)
        g = ttnn.multiply(
            self.a_neg,
            ttnn.softplus(ttnn.add(a, self.dt_bias)),
            activations=[ttnn.UnaryOpType.EXP],
        )  # g carries exp(log_decay).
        beta = ttnn.typecast(
            ttnn.sigmoid(ttnn.to_layout(projections[..., 16448:16496], ttnn.TILE_LAYOUT)), ttnn.float32
        )
        o, new_state = self._recurrence(q, k, v, g, beta)
        ttnn.copy(new_state, self.recurrent_state)
        o = ttnn.typecast(o, ttnn.bfloat16)
        gate = projections[..., 10240:16384]
        if self.native_norm:
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
            o = ttnn.permute(o, (0, 2, 1, 3))
            normed = ttnn.rms_norm(ttnn.typecast(o, ttnn.float32), epsilon=self.norm_eps)
            normed = ttnn.multiply(ttnn.typecast(normed, ttnn.bfloat16), self.norm_weight)
            gate = self._heads(gate, b, s, 48)
            gated = ttnn.multiply(ttnn.typecast(normed, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32)))
            gated = ttnn.to_layout(ttnn.typecast(gated, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
            gated = ttnn.to_layout(ttnn.reshape(gated, [b, s, 6144]), ttnn.TILE_LAYOUT)
        return self._linear(gated, self.out_weight)

    @staticmethod
    def _heads(x, batch, seq, heads):
        return ttnn.reshape(x, (batch, seq, heads, 128))

    def _recurrence(self, q, k, v, g, beta):
        b, seq, _ = q.shape
        q = _scaled_query_l2norm(self._heads(q, b, seq, 16))
        k = self._l2norm(self._heads(k, b, seq, 16))
        v = ttnn.typecast(self._heads(v, b, seq, 48), ttnn.float32)
        return _precise_recurrence(self, q, k, v, g, beta)


_PHYSICAL_INDICES = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t seq = get_compile_time_arg_val(0);
    constexpr uint32_t page_rows = get_compile_time_arg_val(1);
    constexpr uint32_t capacity = get_compile_time_arg_val(2);
    constexpr uint32_t cache_pages = get_compile_time_arg_val(3);
    constexpr uint32_t output_offset = 192; // 64-byte position + at most 128-byte page window.
    constexpr auto pa = TensorAccessorArgs<4>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(0));
    const auto table = TensorAccessor(ta, get_arg_val<uint32_t>(1));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(2));
    CircularBuffer scratch(0);
    scratch.reserve_back(1);
    // Blackhole DRAM reads require matching source/destination residues mod64.
    // Reserve alignment slack even if the generic CB allocator returns mod32.
    const uint32_t base = (scratch.get_write_ptr() + 63) & ~63u;
    auto pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto pages = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + 64);
    auto physical = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + output_offset);
    Noc noc;
    noc.async_read(positions, CoreLocalMem<uint32_t>(base), 32, {.page_id=0}, {});
    noc.async_read_barrier();
    const uint32_t start = pos[0];
    ASSERT(start <= capacity && seq <= capacity - start);
    for (uint32_t token = 0; token < 128; ++token) physical[token] = -1;
    if (start <= capacity && seq <= capacity - start) {
        // At most five page IDs are needed for 128 rows with page_rows>=32.
        // Align the NoC source/destination offsets to 64 bytes. The final
        // aligned window can include row-major page padding but never crosses
        // the allocated aligned page; only valid logical entries are indexed.
        const uint32_t first_page = start / page_rows;
        const uint32_t last_page = (start + seq - 1) / page_rows;
        const uint32_t first_entry = first_page & ~15u;
        const uint32_t entries_end = (last_page + 16) & ~15u;
        const uint32_t read_bytes = (entries_end - first_entry) * 4;
        ASSERT(read_bytes <= 128);
        noc.async_read(table, CoreLocalMem<uint32_t>(base + 64), read_bytes,
                       {.page_id=0, .offset_bytes=first_entry * 4}, {});
        noc.async_read_barrier();
        for (uint32_t token = 0; token < seq; ++token) {
            const uint32_t absolute = start + token;
            const int32_t page = pages[absolute / page_rows - first_entry];
            ASSERT(page >= 0 && static_cast<uint32_t>(page) < cache_pages);
            if (page >= 0 && static_cast<uint32_t>(page) < cache_pages)
                physical[token] = static_cast<int32_t>(page * page_rows + absolute % page_rows);
        }
    }
    noc.async_write(CoreLocalMem<uint32_t>(base + output_offset), output, 512, {}, {.page_id=0});
    noc.async_write_barrier();
}
"""


class FusedQSA(QSA):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj.packed_projection = ttnn.concat([obj.wq, obj.wk, obj.wv, obj.wgate, obj.windex], dim=-1)
        del obj.wq, obj.wk, obj.wv, obj.wgate, obj.windex
        if obj.batch_size == 1:
            obj._physical_update_indices = ttnn.empty(
                (1, 128),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=obj.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            core = ttnn.CoreCoord(0, 0)
            obj._index_core = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        return obj

    def prefill_forward(self, hidden_states, **kwargs):
        self.route_sequence = hidden_states.shape[-2]
        return super().prefill_forward(hidden_states, **kwargs)

    def decode_forward(self, hidden_states, **kwargs):
        # Each invocation binds its own logical write extent; a live decode
        # trace must not inherit the most recent prefill length.
        self.route_sequence = 1
        return super().decode_forward(hidden_states, **kwargs)

    def _route_pooled(self, slots, position, *, store):
        if slots is None:
            return
        if self.pooled_bank is None:
            raise ValueError("Set max_requests to use request slots")
        if tuple(slots.shape) != (self.batch_size, 32) or slots.dtype != ttnn.uint32:
            raise ValueError("slots must be UINT32 [B,32]")
        position = self._packed_position(position)
        grid = self.device.compute_with_storage_grid_size()
        workers = max(1, grid.x * grid.y // self.batch_size)
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers * self.batch_size)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        tensors = [self.pooled_bank, self.pooled_keys, slots, position]
        args = [self.blocks // 32 * 4, 2048, self.max_requests, int(store), self.route_sequence, workers]
        for tensor in tensors:
            args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        cb = ttnn.CBDescriptor(
            total_size=2304,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=2304)],
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=_POOLED_ROUTE_SOURCE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=[
                ((c.x, c.y), [*[t.buffer_address() for t in tensors], i // workers, i % workers])
                for i, c in enumerate(coords)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))

    def _update_kv_indexed(self, k, v, positions, page_table):
        seq = k.shape[2]
        if tuple(k.shape) != (1, 2, seq, 256) or tuple(v.shape) != tuple(k.shape) or not 1 <= seq <= 128:
            raise ValueError("Indexed-cache candidate requires B1/H2/D256 with 1..128 logical rows")
        if tuple(positions.shape) != (1, 32) or tuple(page_table.shape) != (1, self.max_context // self.page_size):
            raise ValueError("Expected packed B1 positions and full request page table")
        output = self._physical_update_indices
        size = 768  # alignment slack64 + position64 + page window128 + indices512.
        tensors = [positions, page_table, output]
        args = [seq, self.page_size, self.max_context, self.k_cache.shape[0]]
        for tensor in tensors:
            args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        kernel = ttnn.KernelDescriptor(
            kernel_source=_PHYSICAL_INDICES,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self._index_core,
            compile_time_args=args,
            runtime_args=[((0, 0), [tensor.buffer_address() for tensor in tensors])],
            config=ttnn.ReaderConfigDescriptor(),
        )
        scratch = ttnn.CBDescriptor(
            total_size=size,
            core_ranges=self._index_core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)],
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[scratch]))
        # Cache and input tensors are distinct BF16 TILE interleaved allocations.
        # The operation is in-place, takes no program config, and reads the live
        # device indices on every replay. No layout conversion is needed here.
        ttnn.experimental.indexed_fused_update_cache(self.k_cache, k, self.v_cache, v, output)

    def _norm(self, x, weight):
        return ttnn.typecast(
            ttnn.rms_norm(
                ttnn.typecast(x, ttnn.float32), weight=weight, epsilon=self.eps, compute_kernel_config=self.compute
            ),
            ttnn.bfloat16,
        )

    @staticmethod
    def _reshape_logical(x, shape):
        return ttnn.to_layout(ttnn.reshape(x, shape), ttnn.TILE_LAYOUT)

    def _chunk_rope(self, x, positions):
        batch, _, seq, _ = x.shape
        indices = ttnn.reshape(
            ttnn.typecast(ttnn.to_layout(positions, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32), (batch, seq)
        )
        cos = ttnn.reshape(ttnn.embedding(indices, self.cos, layout=ttnn.TILE_LAYOUT), (batch, 1, seq, 64))
        sin = ttnn.reshape(ttnn.embedding(indices, self.sin, layout=ttnn.TILE_LAYOUT), (batch, 1, seq, 64))
        lanes = [
            ttnn.experimental.rotary_embedding(
                x[b : b + 1, :, :, :64], cos[b : b + 1], sin[b : b + 1], compute_kernel_config=self.compute
            )
            for b in range(batch)
        ]
        rotary = lanes[0] if batch == 1 else ttnn.concat(lanes, dim=0)
        rotary = ttnn.to_layout(rotary[:, :, :seq, :], ttnn.TILE_LAYOUT)
        return ttnn.concat((rotary, x[..., 64:]), dim=-1)

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
        # These three consumers use exactly the same absolute positions.
        rope_cos, rope_sin = _shared_rope_tables(self, positions, batch, seq)
        q = _shared_apply_rope(self, self._norm(q, self.nq), rope_cos, rope_sin)
        k = _shared_apply_rope(self, self._norm(k, self.nk), rope_cos, rope_sin)
        gate = projections[..., 7168:13312]
        index = projections[..., 13312:]
        iq = _shared_apply_rope(self, self._norm(heads(index[..., :512], 4, 128), self.niq), rope_cos, rope_sin)
        raw = index[..., 512:640]
        if batch == 1 and seq == 1:
            self._update_kv_indexed(k, v, positions_packed, page_table)
            self._transfer(raw, self.raw_key_cache, positions_packed, page_table, mode=0, sequence=seq, heads=1)
        else:
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
        scores = ttnn.matmul(
            flat_iq, self.pooled_keys, transpose_b=True, activation="relu", compute_kernel_config=self.compute
        )
        scores = self._reshape_logical(scores, (batch, 4, seq, self.blocks))
        scores = ttnn.sum(scores, dim=1, keepdim=True)
        next_position = ttnn.add(positions, 1.0)
        complete = ttnn.floor(ttnn.multiply(next_position, 0.25))
        scores = ttnn.where(ttnn.lt(self.block_ids, complete, dtype=ttnn.bfloat16), scores, float("-inf"))
        _, selected = ttnn.topk(ttnn.typecast(scores, ttnn.bfloat16), k=512, dim=-1)
        valid = ttnn.lt(ttnn.typecast(selected, ttnn.float32), complete, dtype=ttnn.bfloat16)
        selected_tokens = ttnn.repeat_interleave(valid, 4, dim=3)
        tail_count = ttnn.mac(complete, -4.0, next_position)
        tail = ttnn.lt(self.tail_offsets, tail_count, dtype=ttnn.bfloat16)
        mask = ttnn.where(ttnn.concat((selected_tokens, tail), dim=3), 0.0, float("-inf"))
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


_POOLED_ROUTE_SOURCE = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Request-slot gather/scatter of complete interleaved state tiles. No numeric
// conversion: FP32 recurrence and BF16 convolution state retain their bits.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t pages = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr bool store = get_compile_time_arg_val(3);
    constexpr uint32_t sequence = get_compile_time_arg_val(4);
    constexpr uint32_t workers = get_compile_time_arg_val(5);
    constexpr auto ba = TensorAccessorArgs<6>();
    constexpr auto wa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto bank = TensorAccessor(ba, get_arg_val<uint32_t>(0));
    const auto work = TensorAccessor(wa, get_arg_val<uint32_t>(1));
    const auto slot = TensorAccessor(sa, get_arg_val<uint32_t>(2));
    const auto pos = TensorAccessor(pa, get_arg_val<uint32_t>(3));
    const uint32_t lane = get_arg_val<uint32_t>(4);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Noc noc;
    noc.async_read(slot, CoreLocalMem<uint32_t>(base + bytes), 128, {.page_id = lane}, {});
    noc.async_read(pos, CoreLocalMem<uint32_t>(base + bytes + 128), 128, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t request = words[bytes / 4];
    const uint32_t position = words[bytes / 4 + 32];
    ASSERT(request < slots);
    // Only complete prefix rows can be read by masked index selection. A
    // zero-position request needs no gather. Store only tiles touched by this
    // call; preserve every other request-owned bank tile bit for bit.
    const uint32_t first = store ? (position / 128) * 4 : 0;
    const uint32_t last = store ? ((position + sequence + 127) / 128) * 4 : ((position + 127) / 128) * 4;
    ASSERT(last <= pages);
    for (uint32_t tile = first + get_arg_val<uint32_t>(5); tile < last; tile += workers) {
        if constexpr (store) {
            noc.async_read(work, CoreLocalMem<uint32_t>(base), bytes, {.page_id = lane * pages + tile}, {});
            noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base), bank, bytes, {}, {.page_id = request * pages + tile});
        } else {
            if (position != 0) {
                noc.async_read(bank, CoreLocalMem<uint32_t>(base), bytes, {.page_id = request * pages + tile}, {});
                noc.async_read_barrier();
            }
            noc.async_write(CoreLocalMem<uint32_t>(base), work, bytes, {}, {.page_id = lane * pages + tile});
        }
        noc.async_write_barrier();
    }
}

"""


class FusedPLE(PLE):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
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
        # Gate definition: sign(g) * sqrt(max(abs(g), 1e-6)).
        # The fused unary chain removes intermediate BF16 packs; matched
        # correctness checks qualify this rounding change.
        dot = ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True)
        gate = ttnn.multiply(dot, 2560**-0.5)
        gate = ttnn.multiply(
            gate,
            gate,
            input_tensor_a_activations=[
                ttnn.UnaryOpType.ABS,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 1e-6),
                ttnn.UnaryOpType.SQRT,
            ],
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGN],
        )
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

    def _norm(self, x, weight, n):
        x = ttnn.typecast(ttnn.reshape(x, (1, n, 4, 2560)), ttnn.float32)
        normed = ttnn.rms_norm(x, epsilon=self.eps, compute_kernel_config=self.compute)
        return ttnn.multiply(normed, weight, dtype=ttnn.bfloat16)


_PRECISE_GDN_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t b = job/48, h = job%48;
        cb_reserve_back(0,16);
        for (uint32_t i=0;i<16;++i)
            noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=job*16+i},{});
        noc.async_read_barrier();
        cb_push_back(0,16);
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for (uint32_t tile=0;tile<4;++tile) {
                uint32_t page = which==3 ? (b*2+h/32)*4+tile : b*4+tile;
                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[off(h/3,r)];
                        for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }
                    }
                }
            }
            cb_push_back(which,4);
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=b*2+h/32},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=b*2+h/32},{});
            noc.async_read_barrier();
            const uint32_t value=src[off(0,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for(uint32_t i=0;i<1024;++i) dst[i]=value;
            cb_push_back(which,1);
        }
        cb_wait_front(12,16);
        for(uint32_t i=0;i<16;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=job*16+i});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,4);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        for(uint32_t t=0;t<4;++t) {
            for(uint32_t r=1;r<32;++r) for(uint32_t c=0;c<32;++c) out[t*1024+off(r,c)]=0;
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(13)+t*4096),output,4096,{},
                {.page_id=job*4+t});
        }
        noc.async_write_barrier();
        cb_pop_front(13,4);
        cb_pop_front(12,16);
    }
}
"""


_PRECISE_GDN_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb,tile,dst);
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,cb,tile); tile_regs_release();
}
template<uint32_t Out> void reduce_product() {
    cb_wait_front(8,16);
    compute_kernel_lib::reduce<PoolType::SUM,ReduceDim::REDUCE_COL,8,6,Out,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::of(4,4));
    cb_pop_front(8,16);
}
void product(uint32_t state,uint32_t column) {
    cb_wait_front(state,16); cb_wait_front(column,4); cb_reserve_back(8,16);
    pack_reconfig_data_format(8);
    for(uint32_t i=0;i<16;++i) {
        tile_regs_acquire(); load(state,i,0); load(column,i/4,1);
        mul_binary_tile_init(); mul_binary_tile(0,1,0); finish(8,i);
    }
    cb_push_back(8,16);
}
void kernel_main() {
    compute_kernel_hw_startup(0,4,7);
    const uint32_t count=get_arg_val<uint32_t>(0);
    for(uint32_t job=0;job<count;++job) {
        cb_wait_front(0,16); cb_wait_front(4,1); cb_reserve_back(7,16);
        pack_reconfig_data_format(7);
        for(uint32_t i=0;i<16;++i) {
            tile_regs_acquire(); load(0,i,0); load(4,0,1);
            mul_binary_tile_init(); mul_binary_tile(0,1,0); finish(7,i);
        }
        cb_push_back(7,16); cb_pop_front(0,16); cb_pop_front(4,1);
        product(7,2); reduce_product<9>();
        cb_wait_front(9,4); cb_wait_front(3,4); cb_wait_front(5,1);
        cb_reserve_back(11,4); pack_reconfig_data_format(11);
        for(uint32_t i=0;i<4;++i) {
            tile_regs_acquire(); load(3,i,0); load(9,i,1); load(5,0,2);
            sub_binary_tile_init(); sub_binary_tile(0,1,0);
            mul_binary_tile_init(); mul_binary_tile(0,2,0); finish(11,i);
        }
        cb_push_back(11,4); cb_pop_front(9,4); cb_pop_front(3,4); cb_pop_front(5,1);
        cb_wait_front(11,4); cb_reserve_back(12,16); pack_reconfig_data_format(12);
        for(uint32_t i=0;i<16;++i) {
            tile_regs_acquire(); load(2,i/4,0); load(11,i%4,1); load(7,i,2);
            sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row(0,1);
            add_binary_tile_init(); add_binary_tile(0,2,0); finish(12,i);
        }
        cb_push_back(12,16); cb_pop_front(11,4); cb_pop_front(7,16); cb_pop_front(2,4);
        product(12,1); reduce_product<13>(); cb_pop_front(1,4);
    }
    cb_pop_front(6,1);
}
"""


_PRECISE_PREFILL_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;
    cb_reserve_back(15,4);
    auto assembled=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(15));
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t b = job/48, h = job%48;
        for(uint32_t token=0;token<length;++token) {
        cb_reserve_back(0,16);
        for (uint32_t i=0;i<16;++i) {
            if(token==0) noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=job*16+i},{});
            // Later tokens reuse the FP32 bytes through aliased CB0/CB12.

        }
        noc.async_read_barrier();
        cb_push_back(0,16);
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for (uint32_t tile=0;tile<4;++tile) {
                uint32_t page = which==3 ? ((b*length+token)*2+h/32)*4+tile : (b*length+token)*4+tile;
                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[off(h/3,r)];
                        for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }
                    }
                }
            }
            cb_push_back(which,4);
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)*2+h/32},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)*2+h/32},{});
            noc.async_read_barrier();
            const uint32_t value=src[off(token%32,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for(uint32_t i=0;i<1024;++i) dst[i]=value;
            cb_push_back(which,1);
        }
        cb_wait_front(12,16);
        if(token+1==length) for(uint32_t i=0;i<16;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=job*16+i});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,4);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        if(token%32==0) for(uint32_t i=0;i<4096;++i) assembled[i]=0;
        for(uint32_t t=0;t<4;++t) for(uint32_t c=0;c<32;++c)
            assembled[t*1024+off(token%32,c)]=out[t*1024+off(0,c)];
        if(token%32==31 || token+1==length) {
            for(uint32_t t=0;t<4;++t)
                noc.async_write(CoreLocalMem<uint32_t>(get_write_ptr(15)+t*4096),output,4096,{},
                    {.page_id=(job*st+token/32)*4+t});
            noc.async_write_barrier();
        }
        cb_pop_front(13,4);
        cb_pop_front(12,16);
        }
    }
}
"""


def _precise_recurrence(self, q, k, v, g, beta):
    """Exact token recurrence; prefill keeps FP32 state in aliased local buffers."""
    batch, length = q.shape[0], q.shape[1]
    # Query normalization already applied the scale as its output activation.
    decay = g  # Already exponentiated by the multiply output activation.
    output = ttnn.empty(
        (batch, 48, length, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final = ttnn.empty(
        (batch, 48, 128, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tensors = [q, k, v, decay, beta, self.recurrent_state, output, final]
    grid = self.device.compute_with_storage_grid_size()
    workers = min(batch * 48, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    assignments = [
        (i * (batch * 48) // workers, (i + 1) * (batch * 48) // workers - i * (batch * 48) // workers)
        for i in range(workers)
    ]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_PRECISE_GDN_READER if length == 1 else _PRECISE_PREFILL_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            config=ttnn.ReaderConfigDescriptor(),
            runtime_args=[
                ((c.x, c.y), [*[t.buffer_address() for t in tensors], *job, length])
                for c, job in zip(coords, assignments)
            ],
        ),
        ttnn.KernelDescriptor(
            kernel_source=_PRECISE_GDN_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            runtime_args=[((c.x, c.y), [job[1] * length]) for c, job in zip(coords, assignments)],
        ),
    ]
    sizes = [16, 4, 4, 4, 1, 1, 1, 16, 16, 4, 1, 4, 16, 4, 1]
    if length > 1:
        sizes.append(4)
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i, count in enumerate(sizes)
        if not (length > 1 and i in (0, 12))
    ]
    if length > 1:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=16 * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096) for i in (0, 12)
                ],
            )
        )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return output, final


_MOE_DISPATCH_COPY = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto ya = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ya.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto y = TensorAccessor(ya, get_arg_val<uint32_t>(1));
    const auto counts = TensorAccessor(ca, get_arg_val<uint32_t>(2));
    const uint32_t row = get_arg_val<uint32_t>(3);
    CircularBuffer row_cb(0), count_cb(1);
    row_cb.reserve_back(1);
    count_cb.reserve_back(1);
    Noc noc;
    noc.async_read(x, CoreLocalMem<uint32_t>(row_cb.get_write_ptr()), 5120, {.page_id=row}, {});
    noc.async_read(counts, CoreLocalMem<uint32_t>(count_cb.get_write_ptr()), 2048, {.page_id=0}, {});
    noc.async_read_barrier();
    const volatile uint32_t* active = reinterpret_cast<volatile uint32_t*>(count_cb.get_write_ptr());
    for (uint32_t expert = 0; expert < 512; ++expert) {
        if (active[expert] == 0) continue;
        // Every active expert receives the complete 32-row union group. Each
        // core owns one row, and expert regions are disjoint. The row scratch
        // remains unchanged until all writes have completed.
        noc.async_write(CoreLocalMem<uint32_t>(row_cb.get_write_ptr()), y, 5120,
                        {}, {.page_id=expert * 32 + row});
    }
    noc.async_write_barrier();
}
"""


def _shared_rope_tables(self, positions, batch, seq):
    indices = ttnn.reshape(ttnn.typecast(ttnn.to_layout(positions, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32), (batch, seq))
    cos = ttnn.reshape(ttnn.embedding(indices, self.cos, layout=ttnn.TILE_LAYOUT), (batch, 1, seq, 64))
    sin = ttnn.reshape(ttnn.embedding(indices, self.sin, layout=ttnn.TILE_LAYOUT), (batch, 1, seq, 64))
    return cos, sin


def _shared_apply_rope(self, x, cos, sin):
    batch, _, seq, _ = x.shape
    lanes = [
        ttnn.experimental.rotary_embedding(
            x[b : b + 1, :, :, :64], cos[b : b + 1], sin[b : b + 1], compute_kernel_config=self.compute
        )
        for b in range(batch)
    ]
    rotary = lanes[0] if batch == 1 else ttnn.concat(lanes, dim=0)
    rotary = ttnn.to_layout(rotary[:, :, :seq, :], ttnn.TILE_LAYOUT)
    return ttnn.concat((rotary, x[..., 64:]), dim=-1)


def _scaled_query_l2norm(x):
    x = ttnn.typecast(x, ttnn.float32)
    squared_norm = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
    return ttnn.multiply(
        x,
        ttnn.rsqrt(ttnn.add(squared_norm, 1e-6)),
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 128**-0.5)],
    )
