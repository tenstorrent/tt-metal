# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen4-Exp decoder components. Weight conversion is setup-only.

Runtime tensors use [1,1,tokens,width] for one request. State is explicitly
owned by the request; components do not use host token positions or torch.
These components are under bringup and are not yet acceptance-validated.
"""

import ttnn
from models.demos.gpt_oss.tt.experts.config import ProgramConfig

from .gdn_recurrence import fp32_recurrent_gated_delta_rule
from .precise_gdn_projection import PreciseGDNProjection
from .precise_projection import PreciseRouterProjection
from .precise_residual_projection import PreciseResidualProjection
from .router_topk import aten_router_topk


def linear(x, weight, compute):
    if isinstance(weight, PreciseGDNProjection):
        return weight(x)
    return ttnn.linear(x, weight, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def rms(x, weight, eps):
    value = ttnn.typecast(x, ttnn.float32)
    scale = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(value, value), dim=-1, keepdim=True), eps))
    return ttnn.typecast(ttnn.multiply(ttnn.multiply(value, scale), weight), x.dtype)


class WeightLoader:
    def __init__(self, state_dict, mesh_device):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def tensor(self, value, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def matrix(self, name, dtype=ttnn.bfloat16):
        return self.tensor(self.state_dict[name].transpose(-1, -2), dtype)

    def norm(self, name, *, groups=1, offset=True):
        value = self.state_dict[name].float()
        if offset:
            value = value + 1
        return self.tensor(value.reshape(1, groups, 1, -1), ttnn.float32)


class GatedResidual:
    def __init__(self, loader, prefix, config):
        self.compute = loader.compute
        self.width = config.hidden_size
        self.streams = config.hc_count
        self.eps = config.rms_norm_eps
        self.weight = loader.norm(prefix + "hc_norm.weight", groups=self.streams)
        # These projections feed discrete routing. Exact-operand controls
        # identify native matmul rounding as a separate upstream error.
        self.precise_down = PreciseResidualProjection(loader, prefix + "input_mix_weight_down.weight")
        self.precise_up = PreciseResidualProjection(loader, prefix + "input_mix_weight_up.weight")
        self.inject = loader.matrix(prefix + "block_inject_weight.weight")

    def read(self, x):
        tokens = x.shape[-2]
        grouped = ttnn.permute(ttnn.reshape(x, (1, tokens, self.streams, self.width)), (0, 2, 1, 3))
        normalized = rms(grouped, self.weight, self.eps)
        flattened = ttnn.reshape(ttnn.permute(normalized, (0, 2, 1, 3)), (1, 1, tokens, self.streams * self.width))
        # BF16 SFPU nonlinear/reduction intermediates lose bits before the
        # final rounding. Evaluate in FP32, preserving each HF BF16 boundary.
        scaled = ttnn.multiply(self.precise_down(flattened), 1 / self.streams)
        mixing = ttnn.typecast(ttnn.silu(ttnn.typecast(scaled, ttnn.float32)), ttnn.bfloat16)
        mixing = ttnn.typecast(ttnn.sigmoid(ttnn.typecast(self.precise_up(mixing), ttnn.float32)), ttnn.bfloat16)
        mixing = ttnn.permute(ttnn.reshape(mixing, (1, tokens, self.streams, self.width)), (0, 2, 1, 3))
        mixed = ttnn.typecast(
            ttnn.mean(ttnn.typecast(ttnn.multiply(mixing, normalized), ttnn.float32), dim=1, keepdim=True),
            ttnn.bfloat16,
        )
        scaled_injection = ttnn.multiply(linear(flattened, self.inject, self.compute), 1 / self.streams)
        injection = ttnn.multiply(
            ttnn.typecast(ttnn.sigmoid(ttnn.typecast(scaled_injection, ttnn.float32)), ttnn.bfloat16), 2
        )
        injection = ttnn.permute(ttnn.reshape(injection, (1, 1, tokens, self.streams)), (0, 3, 2, 1))
        return mixed, injection

    def write(self, residual, block, injection):
        tokens = residual.shape[-2]
        # Match HF's two separate BF16 rounding boundaries. Native BF16
        # elementwise arithmetic differs even when both operands are exact.
        update = ttnn.typecast(
            ttnn.multiply(ttnn.typecast(block, ttnn.float32), ttnn.typecast(injection, ttnn.float32)),
            ttnn.bfloat16,
        )
        update = ttnn.reshape(ttnn.permute(update, (0, 2, 1, 3)), (1, 1, tokens, self.streams * self.width))
        return ttnn.typecast(
            ttnn.add(ttnn.typecast(residual, ttnn.float32), ttnn.typecast(update, ttnn.float32)), ttnn.bfloat16
        )


class SparseMoe:
    def __init__(self, loader, config, *, expert_dtype=ttnn.bfloat16, reuse_private_buffers=False):
        self.compute = loader.compute
        self.reuse_private_buffers = reuse_private_buffers
        self.width = config.hidden_size
        self.intermediate = config.moe_intermediate_size
        self.experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.precise_router = PreciseRouterProjection(loader)
        gate_up = loader.state_dict["mlp.experts.gate_up_proj"]
        gate, up = gate_up.split(self.intermediate, dim=1)
        self.gate = loader.tensor(gate.transpose(-1, -2).unsqueeze(0), expert_dtype)
        self.up = loader.tensor(up.transpose(-1, -2).unsqueeze(0), expert_dtype)
        self.down = loader.tensor(
            loader.state_dict["mlp.experts.down_proj"].transpose(-1, -2).unsqueeze(0), expert_dtype
        )
        self.shared_gate = loader.matrix("mlp.shared_expert.gate_proj.weight")
        self.shared_up = loader.matrix("mlp.shared_expert.up_proj.weight")
        self.shared_down = loader.matrix("mlp.shared_expert.down_proj.weight")
        self.shared_route = loader.matrix("mlp.shared_expert_gate.weight")
        self.program = ProgramConfig(decode_gate_up_cores=(5, 1), decode_down_cores=(8, 1))

    def router_logits(self, x):
        """FP32 accumulation, HF BF16 output boundary, then FP32 routing."""
        return ttnn.typecast(self.precise_router(x), ttnn.float32)

    def __call__(self, x):
        tokens = x.shape[-2]
        if tokens > 32:
            raise ValueError("SparseMoe physical chunk is at most one tile; caller must chunk")
        logits = self.router_logits(x)
        scores, indices = aten_router_topk(logits)
        scores = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        dense = ttnn.scatter(
            ttnn.zeros_like(ttnn.typecast(logits, ttnn.bfloat16)),
            dim=3,
            index=indices,
            src=ttnn.typecast(scores, ttnn.bfloat16),
        )
        # Sparse matmul chooses experts per token tile. Union the selected experts
        # across rows, retaining the independent per-token scores for reduction.
        active = ttnn.to_layout(ttnn.max(dense, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        kwargs = dict(
            sparsity=active,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute,
        )
        program = self.program.get_decode_gate_up_config(tokens, self.intermediate, k=self.width)
        gate = ttnn.sparse_matmul(x, self.gate, program_config=program, **kwargs)
        up = ttnn.sparse_matmul(x, self.up, program_config=program, **kwargs)
        if self.reuse_private_buffers:
            # The expert gate has no other consumer. Preserve both BF16
            # boundaries while avoiding two private 20 MiB temporaries.
            activated = ttnn.multiply(ttnn.silu(gate, output_tensor=gate), up, output_tensor=gate)
        else:
            activated = ttnn.multiply(ttnn.silu(gate), up)
        # These private intermediates have no further consumers. Release their
        # references before the 80 MiB per-expert output allocation on PLE.
        del gate, up
        down = ttnn.sparse_matmul(
            activated,
            self.down,
            is_input_a_sparse=True,
            program_config=self.program.get_decode_down_config(tokens, self.width, k=self.intermediate),
            **kwargs,
        )
        del activated
        weights = ttnn.permute(ttnn.typecast(dense, ttnn.bfloat16), (0, 3, 2, 1))
        # The private per-expert output has no other consumer. Weight it in
        # place so the resident PLE table need not coexist with an 80 MiB copy.
        ttnn.multiply(down, weights, output_tensor=down)
        if self.reuse_private_buffers and tokens < 32:
            # Generic sum clones a padded input before zeroing its padding.
            # This weighted down buffer is private, so zero it in place and
            # expose its existing physical height to the same reduction.
            # Explicit padded shapes make both reshapes metadata-only views.
            ttnn.fill_implicit_tile_padding(down, 0.0)
            down = ttnn.reshape(down, (1, self.experts, 32, self.width), (1, self.experts, 32, self.width))
            routed = ttnn.sum(down, dim=1, keepdim=True)
            routed = ttnn.reshape(routed, (1, 1, tokens, self.width), (1, 1, 32, self.width))
        else:
            routed = ttnn.sum(down, dim=1, keepdim=True)
        del down
        shared = ttnn.multiply(
            ttnn.silu(linear(x, self.shared_gate, self.compute)), linear(x, self.shared_up, self.compute)
        )
        shared = linear(shared, self.shared_down, self.compute)
        shared = ttnn.multiply(shared, ttnn.sigmoid(linear(x, self.shared_route, self.compute)))
        return ttnn.add(routed, shared)


class GatedDeltaNet:
    def __init__(self, loader, config):
        self.compute = loader.compute
        self.eps = config.rms_norm_eps
        self.hkey = config.linear_num_key_heads
        self.hvalue = config.linear_num_value_heads
        self.dkey = config.linear_key_head_dim
        self.dvalue = config.linear_value_head_dim
        self.key_width = self.hkey * self.dkey
        self.value_width = self.hvalue * self.dvalue
        self.conv_width = 2 * self.key_width + self.value_width
        self.qkv = PreciseGDNProjection(loader, "linear_attn.in_proj_qkv.weight")
        self.z = loader.matrix("linear_attn.in_proj_z.weight")
        self.a = loader.matrix("linear_attn.in_proj_a.weight")
        self.b = loader.matrix("linear_attn.in_proj_b.weight")
        self.out = PreciseGDNProjection(loader, "linear_attn.out_proj.weight")
        self.A = loader.tensor(-loader.state_dict["linear_attn.A_log"].float().exp().reshape(1, 1, 1, -1), ttnn.float32)
        self.dt = loader.tensor(loader.state_dict["linear_attn.dt_bias"].float().reshape(1, 1, 1, -1), ttnn.float32)
        # HF RMSNormGated rounds normalization and the learned-weight product
        # separately to BF16, then applies its gate in FP32.
        self.norm = loader.tensor(loader.state_dict["linear_attn.norm.weight"].reshape(1, 1, 1, -1))
        conv = loader.state_dict["linear_attn.conv1d.weight"]
        self.conv = [loader.tensor(conv[:, 0, lane].reshape(1, 1, 1, -1), ttnn.float32) for lane in range(4)]

    def __call__(self, x, *, conv_state, recurrent_state):
        tokens = x.shape[-2]
        projected = linear(x, self.qkv, self.compute)
        history = ttnn.concat((conv_state, projected), dim=2)
        # HF Conv1d accumulates all four BF16 products before its BF16 output
        # rounding. Intermediate BF16 products/sums perturb the downstream gate.
        conv_input = ttnn.typecast(history, ttnn.float32)
        parts = [ttnn.multiply(conv_input[:, :, lane : lane + tokens, :], self.conv[lane]) for lane in range(4)]
        convolved = ttnn.typecast(ttnn.add(ttnn.add(parts[0], parts[1]), ttnn.add(parts[2], parts[3])), ttnn.bfloat16)
        # HF evaluates these nonlinear functions before rounding their BF16
        # outputs. Exact-operand controls show native BF16 kernels lose bits.
        convolved = ttnn.typecast(ttnn.silu(ttnn.typecast(convolved, ttnn.float32)), ttnn.bfloat16)
        ttnn.copy(history[:, :, -3:, :], conv_state)
        q = ttnn.reshape(convolved[:, :, :, : self.key_width], (1, tokens, self.hkey, self.dkey))
        k = ttnn.reshape(convolved[:, :, :, self.key_width : 2 * self.key_width], (1, tokens, self.hkey, self.dkey))
        v = ttnn.reshape(convolved[:, :, :, 2 * self.key_width :], (1, tokens, self.hvalue, self.dvalue))

        def normalize(value):
            value = ttnn.typecast(value, ttnn.float32)
            return ttnn.multiply(
                value, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(value, value), dim=-1, keepdim=True), 1e-6))
            )

        q, k = normalize(q), normalize(k)
        decay = ttnn.multiply(
            self.A, ttnn.softplus(ttnn.add(ttnn.typecast(linear(x, self.a, self.compute), ttnn.float32), self.dt))
        )
        beta = ttnn.typecast(
            ttnn.typecast(ttnn.sigmoid(ttnn.typecast(linear(x, self.b, self.compute), ttnn.float32)), ttnn.bfloat16),
            ttnn.float32,
        )
        output, state = fp32_recurrent_gated_delta_rule(
            q,
            k,
            ttnn.typecast(v, ttnn.float32),
            ttnn.reshape(decay, (1, tokens, self.hvalue)),
            ttnn.reshape(beta, (1, tokens, self.hvalue)),
            initial_state=recurrent_state,
            output_final_state=True,
            compute_kernel_config=self.compute,
        )
        ttnn.copy(state, recurrent_state)
        output = ttnn.typecast(output, ttnn.bfloat16)
        value = ttnn.typecast(output, ttnn.float32)
        scale = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(value, value), dim=-1, keepdim=True), self.eps))
        output = ttnn.multiply(ttnn.typecast(ttnn.multiply(value, scale), ttnn.bfloat16), self.norm)
        gate = ttnn.reshape(linear(x, self.z, self.compute), (1, tokens, self.hvalue, self.dvalue))
        output = ttnn.typecast(
            ttnn.multiply(ttnn.typecast(output, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32))),
            ttnn.bfloat16,
        )
        return linear(ttnn.reshape(output, (1, 1, tokens, self.value_width)), self.out, self.compute)
