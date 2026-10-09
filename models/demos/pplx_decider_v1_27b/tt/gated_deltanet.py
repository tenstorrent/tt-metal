# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet (Qwen3.5 ``linear_attn``), causal prefill.

Packed in-projection [qkv | z | b | a] -> causal depthwise conv (k=4, no bias) + SiLU on qkv ->
q/k/v split (16 k-heads x128, 48 v-heads x128; the native scan L2-normalises q/k, scales q by
128^-0.5 and maps k-head h//3 to v-head h) -> chunked gated delta rule with fp32 state ->
gated RMSNorm ``w * norm(o) * silu(z)`` -> out_proj.

Prefill runs in bounded chunks; the recurrent state [1, 48, 128, 128] fp32 and the last three
conv inputs [1, 3, 10240] are carried between chunks as device tensors. A fresh request starts
from persistent zero tensors that are never written, matching HF's zero initial state.

Adapted from models/demos/qwen38_27b_qb2/tt/decoder.py (_delta), single device, functional
state carry instead of in-place state copies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.common import prefill_linear, resolve
from models.demos.pplx_decider_v1_27b.tt.model_config import PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.optimizations import Optimizations
from models.demos.pplx_decider_v1_27b.tt.weight_adapter import TILE, GatedDeltaNetWeights
from models.demos.qwen38_27b_qb2.tt.decode_conv import make_actual_start


@dataclass
class DeltaState:
    recurrent: ttnn.Tensor  # [1, 48, 128, 128] fp32
    conv: ttnn.Tensor  # [1, 3, 10240] ROW_MAJOR, the last three conv inputs


@dataclass
class GatedDeltaNetConfig:
    weights: GatedDeltaNetWeights
    args: PplxDeciderArgs
    optimizations: Optimizations
    mesh_device: object | None = None


class PplxGatedDeltaNet(LightweightModule):
    def __init__(self, weights: GatedDeltaNetWeights, args: PplxDeciderArgs, optimizations: Optimizations):
        super().__init__()
        self.config = _resolve(GatedDeltaNetConfig(weights=weights, args=args, optimizations=optimizations))
        self._setup()

    @classmethod
    def from_config(cls, config: GatedDeltaNetConfig) -> "PplxGatedDeltaNet":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = _resolve(config)
        instance._setup()
        return instance

    def _setup(self) -> None:
        import torch

        c, d = self.config, self.config.optimizations.delta
        device = c.mesh_device

        def upload(t, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        a = c.args
        self.zero_state = DeltaState(
            recurrent=upload(
                torch.zeros(1, a.linear_num_value_heads, a.linear_key_head_dim, a.linear_value_head_dim),
                d.recurrent_dtype,
            ),
            conv=upload(
                torch.zeros(1, a.linear_conv_kernel_dim - 1, a.conv_width), d.conv_state_dtype, ttnn.ROW_MAJOR_LAYOUT
            ),
        )
        self.conv_actual_start = make_actual_start(device)
        # Scan constants supplied explicitly; the op's default builds them on host per call.
        c32 = d.scan_chunk
        masks = torch.zeros(1, 1, 32, 96)
        masks[:, :, :16, :16] = 1
        masks[:, :, 16:, 48:64] = 1
        masks[:, :, 16:, 64:80] = 1
        self.scan_constants = {
            "eye": upload(torch.eye(c32).reshape(1, 1, c32, c32), ttnn.float32),
            "tril": upload(torch.ones(c32, c32).tril().reshape(1, 1, c32, c32), ttnn.float32),
            "ones": upload(torch.ones(1, 1, c32, c32), ttnn.float32),
            "masks": upload(masks, ttnn.float32),
        }
        self._loaded = False

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        w = self.config.weights
        self.in_proj = w.in_proj.get_device_weight()
        self.out_proj = w.out_proj.get_device_weight()
        self.conv_taps = [tap.get_device_weight() for tap in w.conv_taps]
        self.a_neg = w.a_neg.get_device_weight()
        self.dt_bias = w.dt_bias.get_device_weight()
        self.norm = w.norm.get_device_weight()
        self._loaded = True

    def forward(self, x: ttnn.Tensor, state: DeltaState | None = None) -> tuple[ttnn.Tensor, DeltaState]:
        """One prefill chunk. x: [1, t, 5120] (input-normed). ``state=None`` starts a fresh request."""
        self.load_device_weights()
        state = state or self.zero_state
        a, opts = self.config.args, self.config.optimizations
        b, t, _ = x.shape
        if b != 1:
            raise ValueError("Prefill GDN takes one request at a time")
        h, hv, dk = a.linear_num_key_heads, a.linear_num_value_heads, a.linear_key_head_dim
        conv_width, z_width = a.conv_width, a.linear_value_dim
        gate_width = (hv + TILE - 1) // TILE * TILE

        packed = prefill_linear(x, self.in_proj, "delta_in", opts.linear)
        qkv = packed[:, :, :conv_width]
        z = packed[:, :, conv_width : conv_width + z_width]
        beta = ttnn.sigmoid(packed[:, :, conv_width + z_width : conv_width + z_width + hv])
        a_raw = ttnn.typecast(
            packed[:, :, conv_width + z_width + gate_width : conv_width + z_width + gate_width + hv], ttnn.float32
        )

        padded_t = (t + TILE - 1) // TILE * TILE

        def pad_time(tensor):
            return tensor if padded_t == t else ttnn.pad(tensor, [(0, 0), (0, padded_t - t), (0, 0)], 0.0)

        row_qkv = ttnn.to_layout(pad_time(qkv), ttnn.ROW_MAJOR_LAYOUT)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            row_qkv,
            state.conv,
            *self.conv_taps,
            h * dk,
            h * dk,
            hv * dk,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=opts.delta.conv_channel_chunk),
            actual_start=self.conv_actual_start,
            predecessor_carry=state.conv,
        )
        # Next chunk's left context: the last three real (unpadded) conv inputs.
        if t >= 3:
            conv_tail = row_qkv[:, t - 3 : t, :]
        else:
            conv_tail = ttnn.concat([state.conv[:, t:, :], row_qkv[:, :t, :]], dim=1)

        # g = -exp(A_log) * softplus(a + dt_bias), softplus threshold 20 as torch.
        g = ttnn.mul(
            self.a_neg,
            ttnn.add(a_raw, self.dt_bias),
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        # Zero beta and zero log-decay make every padded step an identity on the state.
        g, beta = pad_time(g), pad_time(beta)
        output, new_recurrent = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state.recurrent,
            output_final_state=True,
            output_head_major=True,
            chunk_size=opts.delta.scan_chunk,
            **self.scan_constants,
        )
        z = pad_time(z)
        output = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            output, z, self.norm, hv, epsilon=a.rms_norm_eps, output_dtype=ttnn.bfloat16
        )
        output = ttnn.mul(output, z)  # norm(o) * w * sigmoid(z) * z == w * norm(o) * silu(z)
        # Hide trailing padded rows; projections and norms act per row.
        output = ttnn.reshape(output, [b, t, hv * dk], output.padded_shape)
        return prefill_linear(output, self.out_proj, "delta_out", opts.linear), DeltaState(new_recurrent, conv_tail)


def _resolve(config: GatedDeltaNetConfig) -> GatedDeltaNetConfig:
    device = config.mesh_device or config.optimizations.mesh_device
    w = config.weights
    return replace(
        config,
        mesh_device=device,
        weights=GatedDeltaNetWeights(
            in_proj=resolve(w.in_proj, device),
            out_proj=resolve(w.out_proj, device),
            conv_taps=tuple(resolve(tap, device) for tap in w.conv_taps),
            a_neg=resolve(w.a_neg, device),
            dt_bias=resolve(w.dt_bias, device),
            norm=resolve(w.norm, device),
        ),
    )
