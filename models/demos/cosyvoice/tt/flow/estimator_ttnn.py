"""CausalConditionalDecoder UNet1D estimator — native TTNN implementation (Stage 2.3).

Ports the flow estimator from host CPU to Tenstorrent N300 via TTNN.
Architecture: 1 down + 12 mid + 1 up blocks, each = CausalResnetBlock1D + 4× BasicTransformerBlock.
Batch=2 (CFG), channels=256, 8 heads × 64 head_dim, T=variable (padded to tile boundary).

Reference: cosyvoice/flow/decoder.py::CausalConditionalDecoder
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

import ttnn


def _to_tile(t: torch.Tensor, device: ttnn.MeshDevice, dtype=ttnn.DataType.BFLOAT16) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _to_host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t)


class CausalConv1dTtnn:
    """CausalConv1d: left-pad(k-1) + Conv1d(stride=1, pad=0)."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        device: ttnn.MeshDevice,
    ):
        self.device = device
        self.kernel_size = weight.shape[2]
        self.causal_padding = self.kernel_size - 1
        self.in_channels = weight.shape[1]
        self.out_channels = weight.shape[0]
        self.weight_tt = ttnn.from_torch(
            weight.unsqueeze(2), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self.bias_tt = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._pad_cache: Dict[Tuple[int, int], ttnn.Tensor] = {}
        self._preprocessed_weights = None
        self._preprocessed_bias = None
        self._preprocessed_input_length = None

    def _get_pad_tensor(self, batch_size: int) -> ttnn.Tensor:
        key = (batch_size, self.in_channels)
        if key not in self._pad_cache:
            self._pad_cache[key] = ttnn.zeros(
                (batch_size, 1, self.causal_padding, self.in_channels),
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        return self._pad_cache[key]

    def __call__(self, x: ttnn.Tensor, batch_size: int, input_length: int) -> ttnn.Tensor:
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (batch_size, 1, input_length, self.in_channels))

        if self.causal_padding > 0:
            pad_tensor = self._get_pad_tensor(batch_size)
            x_rm = ttnn.concat([pad_tensor, x_rm], dim=2)
            padded_length = input_length + self.causal_padding
        else:
            padded_length = input_length

        use_cached = self._preprocessed_weights is not None and self._preprocessed_input_length == padded_length

        if use_cached:
            out = ttnn.conv1d(
                input_tensor=x_rm,
                weight_tensor=self._preprocessed_weights,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                device=self.device,
                bias_tensor=self._preprocessed_bias,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                batch_size=batch_size,
                input_length=padded_length,
                dtype=ttnn.DataType.BFLOAT16,
                return_output_dim=True,
            )
        else:
            out = ttnn.conv1d(
                input_tensor=x_rm,
                weight_tensor=self.weight_tt,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                device=self.device,
                bias_tensor=self.bias_tt,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                batch_size=batch_size,
                input_length=padded_length,
                dtype=ttnn.DataType.BFLOAT16,
                return_output_dim=True,
                return_weights_and_bias=True,
            )

        if isinstance(out, tuple):
            out_tensor = out[0]
            out_length = out[1]
            if len(out) == 3 and not use_cached:
                wb = out[2]
                self._preprocessed_weights = wb[0]
                self._preprocessed_bias = wb[1]
                self._preprocessed_input_length = padded_length
        else:
            out_tensor = out
            out_length = input_length

        out_tensor = ttnn.reshape(out_tensor, (batch_size, out_length, self.out_channels))
        out_tensor = ttnn.to_layout(out_tensor, ttnn.TILE_LAYOUT)
        return out_tensor


class Conv1x1Ttnn:
    """Conv1d with kernel_size=1 (equivalent to a linear projection along channel dim)."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        device: ttnn.MeshDevice,
    ):
        self.device = device
        self.in_channels = weight.shape[1]
        self.out_channels = weight.shape[0]
        self.weight_tt = ttnn.from_torch(
            weight.squeeze(2).T.contiguous(), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias_tt = ttnn.from_torch(
            bias.reshape(1, 1, -1), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(ttnn.matmul(x, self.weight_tt), self.bias_tt)


class LinearTtnn:
    """Linear layer: y = x @ W^T + b."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        device: ttnn.MeshDevice,
    ):
        self.device = device
        self.weight_tt = ttnn.from_torch(
            weight.T.contiguous(), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device
        )
        if bias is not None:
            self.bias_tt = ttnn.from_torch(
                bias.reshape(1, 1, -1), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device
            )
        else:
            self.bias_tt = None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        out = ttnn.matmul(x, self.weight_tt)
        if self.bias_tt is not None:
            out = ttnn.add(out, self.bias_tt)
        return out


class LayerNormTtnn:
    """LayerNorm with weight and bias."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        device: ttnn.MeshDevice,
    ):
        self.weight_tt = ttnn.from_torch(
            weight.reshape(1, 1, -1), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias_tt = ttnn.from_torch(
            bias.reshape(1, 1, -1), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, weight=self.weight_tt, bias=self.bias_tt, epsilon=1e-5)


class CausalBlock1DTtnn:
    """CausalBlock1D: CausalConv1d(k=3) + LayerNorm + Mish."""

    def __init__(self, prefix: str, weights: Dict[str, torch.Tensor], device: ttnn.MeshDevice):
        self.conv = CausalConv1dTtnn(
            weights[f"{prefix}.block.0.weight"],
            weights[f"{prefix}.block.0.bias"],
            device,
        )
        self.ln = LayerNormTtnn(
            weights[f"{prefix}.block.2.weight"],
            weights[f"{prefix}.block.2.bias"],
            device,
        )

    def __call__(self, x: ttnn.Tensor, mask: ttnn.Tensor, batch_size: int, T: int) -> ttnn.Tensor:
        x = ttnn.multiply(x, mask)
        x = self.conv(x, batch_size, T)
        x = self.ln(x)
        x = ttnn.mish(x)
        x = ttnn.multiply(x, mask)
        return x


class CausalResnetBlock1DTtnn:
    """CausalResnetBlock1D: block1 + time_emb + block2 + residual."""

    def __init__(self, prefix: str, weights: Dict[str, torch.Tensor], device: ttnn.MeshDevice):
        self.block1 = CausalBlock1DTtnn(f"{prefix}.block1", weights, device)
        self.block2 = CausalBlock1DTtnn(f"{prefix}.block2", weights, device)
        self.time_proj = LinearTtnn(
            weights[f"{prefix}.mlp.1.weight"],
            weights[f"{prefix}.mlp.1.bias"],
            device,
        )
        self.res_conv = Conv1x1Ttnn(
            weights[f"{prefix}.res_conv.weight"],
            weights[f"{prefix}.res_conv.bias"],
            device,
        )

    def __call__(
        self, x: ttnn.Tensor, mask: ttnn.Tensor, time_emb: ttnn.Tensor, batch_size: int, T: int
    ) -> ttnn.Tensor:
        h = self.block1(x, mask, batch_size, T)
        t_proj = ttnn.mish(time_emb)
        t_proj = self.time_proj(t_proj)
        t_proj = ttnn.reshape(t_proj, (batch_size, 1, -1))
        h = ttnn.add(h, t_proj)
        h = self.block2(h, mask, batch_size, T)
        res = ttnn.multiply(x, mask)
        res = self.res_conv(res)
        return ttnn.add(h, res)


class BasicTransformerBlockTtnn:
    """BasicTransformerBlock: LN + self-attention + residual + LN + FF + residual."""

    def __init__(self, prefix: str, weights: Dict[str, torch.Tensor], device: ttnn.MeshDevice):
        self.num_heads = 8
        self.head_dim = 64
        self.inner_dim = 512
        scale = 1.0 / math.sqrt(64)

        self.norm1 = LayerNormTtnn(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"], device)
        self.norm3 = LayerNormTtnn(weights[f"{prefix}.norm3.weight"], weights[f"{prefix}.norm3.bias"], device)

        q_weight = weights[f"{prefix}.attn1.to_q.weight"] * scale
        self.to_q = LinearTtnn(q_weight, None, device)
        self.to_k = LinearTtnn(weights[f"{prefix}.attn1.to_k.weight"], None, device)
        self.to_v = LinearTtnn(weights[f"{prefix}.attn1.to_v.weight"], None, device)
        self.to_out = LinearTtnn(
            weights[f"{prefix}.attn1.to_out.0.weight"], weights[f"{prefix}.attn1.to_out.0.bias"], device
        )

        self.ff_proj = LinearTtnn(
            weights[f"{prefix}.ff.net.0.proj.weight"], weights[f"{prefix}.ff.net.0.proj.bias"], device
        )
        self.ff_out = LinearTtnn(weights[f"{prefix}.ff.net.2.weight"], weights[f"{prefix}.ff.net.2.bias"], device)

    def __call__(self, x: ttnn.Tensor, attn_mask: ttnn.Tensor, batch_size: int, T: int) -> ttnn.Tensor:
        residual = x
        norm_x = self.norm1(x)

        q = self.to_q(norm_x)
        k = self.to_k(norm_x)
        v = self.to_v(norm_x)

        q = ttnn.reshape(q, (batch_size, T, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (batch_size, T, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (batch_size, T, self.num_heads, self.head_dim))

        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        k_t = ttnn.transpose(k, -2, -1)
        attn_weights = ttnn.matmul(q, k_t)

        attn_weights = ttnn.add(attn_weights, attn_mask)
        attn_weights = ttnn.softmax(attn_weights, dim=-1)

        attn_output = ttnn.matmul(attn_weights, v)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, T, self.inner_dim))

        attn_output = self.to_out(attn_output)
        x = ttnn.add(attn_output, residual)

        residual = x
        norm_x = self.norm3(x)
        ff_h = self.ff_proj(norm_x)
        ff_h = ttnn.gelu(ff_h)
        ff_out = self.ff_out(ff_h)
        x = ttnn.add(ff_out, residual)

        return x


class UNetEstimatorTtnn:
    """Native TTNN CausalConditionalDecoder for the flow estimator.

    Replaces the host-side UNetEstimator (Stage 1) with device-accelerated TTNN ops.
    """

    def __init__(self, decoder_weights: Dict[str, torch.Tensor], device: ttnn.MeshDevice):
        self.device = device
        self.in_channels = 320
        self.out_channels = 80
        self.channels = 256
        self.n_blocks = 4
        self.num_mid_blocks = 12
        self.num_heads = 8
        self.head_dim = 64

        est_sd = {
            k.replace("decoder.estimator.", ""): v
            for k, v in decoder_weights.items()
            if k.startswith("decoder.estimator.")
        }

        self._build_time_embedding(est_sd)
        self._build_down_blocks(est_sd)
        self._build_mid_blocks(est_sd)
        self._build_up_blocks(est_sd)
        self._build_final(est_sd)

    def _build_time_embedding(self, sd: Dict[str, torch.Tensor]):
        half_dim = self.in_channels // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        self.emb_freqs = torch.exp(torch.arange(half_dim).float() * -emb_scale)
        self.time_linear1 = LinearTtnn(sd["time_mlp.linear_1.weight"], sd["time_mlp.linear_1.bias"], self.device)
        self.time_linear2 = LinearTtnn(sd["time_mlp.linear_2.weight"], sd["time_mlp.linear_2.bias"], self.device)

    def _build_down_blocks(self, sd: Dict[str, torch.Tensor]):
        self.down_resnet = CausalResnetBlock1DTtnn("down_blocks.0.0", sd, self.device)
        self.down_transformers = []
        for i in range(self.n_blocks):
            self.down_transformers.append(BasicTransformerBlockTtnn(f"down_blocks.0.1.{i}", sd, self.device))
        self.down_downsample = CausalConv1dTtnn(sd["down_blocks.0.2.weight"], sd["down_blocks.0.2.bias"], self.device)

    def _build_mid_blocks(self, sd: Dict[str, torch.Tensor]):
        self.mid_resnets = []
        self.mid_transformers = []
        for m in range(self.num_mid_blocks):
            self.mid_resnets.append(CausalResnetBlock1DTtnn(f"mid_blocks.{m}.0", sd, self.device))
            blocks = []
            for i in range(self.n_blocks):
                blocks.append(BasicTransformerBlockTtnn(f"mid_blocks.{m}.1.{i}", sd, self.device))
            self.mid_transformers.append(blocks)

    def _build_up_blocks(self, sd: Dict[str, torch.Tensor]):
        self.up_resnet = CausalResnetBlock1DTtnn("up_blocks.0.0", sd, self.device)
        self.up_transformers = []
        for i in range(self.n_blocks):
            self.up_transformers.append(BasicTransformerBlockTtnn(f"up_blocks.0.1.{i}", sd, self.device))
        self.up_upsample = CausalConv1dTtnn(sd["up_blocks.0.2.weight"], sd["up_blocks.0.2.bias"], self.device)

    def _build_final(self, sd: Dict[str, torch.Tensor]):
        self.final_conv = CausalConv1dTtnn(
            sd["final_block.block.0.weight"], sd["final_block.block.0.bias"], self.device
        )
        self.final_ln = LayerNormTtnn(sd["final_block.block.2.weight"], sd["final_block.block.2.bias"], self.device)
        self.final_proj = Conv1x1Ttnn(sd["final_proj.weight"], sd["final_proj.bias"], self.device)

    def _sinusoidal_pos_emb(self, t: torch.Tensor) -> torch.Tensor:
        emb = 1000.0 * t.unsqueeze(1) * self.emb_freqs.unsqueeze(0).to(t.device)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def _build_attn_mask(self, mask: torch.Tensor, T: int, device: ttnn.MeshDevice) -> ttnn.Tensor:
        mask_2d = mask.squeeze(1).unsqueeze(1).expand(-1, T, -1)
        bias = (1.0 - mask_2d.float()) * -1.0e10
        bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        return _to_tile(bias, device)

    def _device_forward(
        self, x_tt: ttnn.Tensor, mask_tt: ttnn.Tensor, attn_mask_tt: ttnn.Tensor, t_emb_tt: ttnn.Tensor, B: int, T: int
    ) -> ttnn.Tensor:
        x_tt = self.down_resnet(x_tt, mask_tt, t_emb_tt, B, T)
        for tb in self.down_transformers:
            x_tt = tb(x_tt, attn_mask_tt, B, T)
        hidden = x_tt
        x_tt = ttnn.multiply(x_tt, mask_tt)
        x_tt = self.down_downsample(x_tt, B, T)

        for m in range(self.num_mid_blocks):
            x_tt = self.mid_resnets[m](x_tt, mask_tt, t_emb_tt, B, T)
            for tb in self.mid_transformers[m]:
                x_tt = tb(x_tt, attn_mask_tt, B, T)

        x_tt = ttnn.concat([x_tt, hidden], dim=-1)
        x_tt = self.up_resnet(x_tt, mask_tt, t_emb_tt, B, T)
        for tb in self.up_transformers:
            x_tt = tb(x_tt, attn_mask_tt, B, T)
        x_tt = ttnn.multiply(x_tt, mask_tt)
        x_tt = self.up_upsample(x_tt, B, T)

        x_tt = ttnn.multiply(x_tt, mask_tt)
        x_tt = self.final_conv(x_tt, B, T)
        x_tt = self.final_ln(x_tt)
        x_tt = ttnn.mish(x_tt)
        x_tt = ttnn.multiply(x_tt, mask_tt)
        x_tt = self.final_proj(x_tt)
        x_tt = ttnn.multiply(x_tt, mask_tt)
        return x_tt

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        B = x.shape[0]
        T_orig = x.shape[2]
        T_pad = ((T_orig + 31) // 32) * 32

        if T_pad != T_orig:
            pad_len = T_pad - T_orig
            x = torch.nn.functional.pad(x, (0, pad_len))
            mu = torch.nn.functional.pad(mu, (0, pad_len))
            mask = torch.nn.functional.pad(mask, (0, pad_len))
            cond = torch.nn.functional.pad(cond, (0, pad_len))

        T = T_pad

        t_emb = self._sinusoidal_pos_emb(t)
        t_emb_tt = _to_tile(t_emb, self.device)
        t_emb_tt = self.time_linear1(t_emb_tt)
        t_emb_tt = ttnn.silu(t_emb_tt)
        t_emb_tt = self.time_linear2(t_emb_tt)

        x_btc = x.permute(0, 2, 1).contiguous()
        mu_btc = mu.permute(0, 2, 1).contiguous()
        spks_exp = spks.unsqueeze(2).expand(-1, -1, T).permute(0, 2, 1).contiguous()
        cond_btc = cond.permute(0, 2, 1).contiguous()
        x_cat = torch.cat([x_btc, mu_btc, spks_exp, cond_btc], dim=-1)

        mask_btc = mask.permute(0, 2, 1).contiguous()

        x_tt = _to_tile(x_cat, self.device)
        mask_tt = _to_tile(mask_btc, self.device)
        attn_mask_tt = self._build_attn_mask(mask, T, self.device)

        x_tt = self._device_forward(x_tt, mask_tt, attn_mask_tt, t_emb_tt, B, T)

        out = _to_host(x_tt)
        out = out[:, :T_orig, :].permute(0, 2, 1).contiguous()
        return out

    def init_trace(self, B: int, T_orig: int, mask: torch.Tensor) -> bool:
        """Capture a device trace for repeated NFE steps with fixed shapes.

        Args:
            B: batch size (2 for CFG)
            T_orig: original mel length (before padding)
            mask: [B, 1, T_orig] mask tensor

        Returns:
            True if trace was captured successfully, False if trace region too small.
        """
        T_pad = ((T_orig + 31) // 32) * 32
        if T_pad != T_orig:
            mask = torch.nn.functional.pad(mask, (0, T_pad - T_orig))
        T = T_pad

        self._trace_B = B
        self._trace_T_orig = T_orig
        self._trace_T = T

        mask_btc = mask.permute(0, 2, 1).contiguous()
        self._trace_mask_tt = _to_tile(mask_btc, self.device)
        self._trace_attn_mask_tt = self._build_attn_mask(mask, T, self.device)

        self._trace_x_buf = ttnn.zeros(
            (B, T, self.in_channels), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self._trace_t_emb_buf = ttnn.zeros(
            (B, 1024), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        dummy_x = torch.zeros(B, T, self.in_channels)
        dummy_t_emb = torch.zeros(B, 1024)
        x_host = ttnn.from_torch(dummy_x, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT)
        t_host = ttnn.from_torch(dummy_t_emb, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT)

        for _ in range(2):
            ttnn.copy_host_to_device_tensor(x_host, self._trace_x_buf)
            ttnn.copy_host_to_device_tensor(t_host, self._trace_t_emb_buf)
            self._device_forward(
                self._trace_x_buf, self._trace_mask_tt, self._trace_attn_mask_tt, self._trace_t_emb_buf, B, T
            )

        ttnn.copy_host_to_device_tensor(x_host, self._trace_x_buf)
        ttnn.copy_host_to_device_tensor(t_host, self._trace_t_emb_buf)

        try:
            self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._trace_output = self._device_forward(
                self._trace_x_buf, self._trace_mask_tt, self._trace_attn_mask_tt, self._trace_t_emb_buf, B, T
            )
            ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
            return True
        except RuntimeError:
            self._trace_id = None
            return False

    def forward_traced(self, x_cat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Replay the captured trace with updated inputs.

        Args:
            x_cat: [B, T_orig, 320] assembled input (x, mu, spks, cond concatenated)
            t: [B] timestep values

        Returns:
            [B, 80, T_orig] predicted velocity
        """
        B = self._trace_B
        T = self._trace_T
        T_orig = self._trace_T_orig

        t_emb = self._sinusoidal_pos_emb(t)
        t_emb_tt = _to_tile(t_emb, self.device)
        t_emb_tt = self.time_linear1(t_emb_tt)
        t_emb_tt = ttnn.silu(t_emb_tt)
        t_emb_tt = self.time_linear2(t_emb_tt)

        if x_cat.shape[1] < T:
            pad_len = T - x_cat.shape[1]
            x_cat = torch.nn.functional.pad(x_cat, (0, 0, 0, pad_len))

        x_host = ttnn.from_torch(x_cat, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT)
        t_emb_host = ttnn.to_torch(t_emb_tt)
        t_emb_host_tt = ttnn.from_torch(t_emb_host, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT)

        ttnn.copy_host_to_device_tensor(x_host, self._trace_x_buf)
        ttnn.copy_host_to_device_tensor(t_emb_host_tt, self._trace_t_emb_buf)

        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)

        out = _to_host(self._trace_output)
        out = out[:, :T_orig, :].permute(0, 2, 1).contiguous()
        return out

    def release_trace(self):
        if hasattr(self, "_trace_id") and self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
