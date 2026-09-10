# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""CosyVoice2's flow-matching estimator: `CausalConditionalDecoder` (the U-Net that
predicts the ODE's vector field) and `CausalConditionalCFM` (the fixed-noise Euler
solver that drives it), confirmed against real upstream source directly --
`cosyvoice/flow/decoder.py`, `cosyvoice/flow/flow_matching.py`, and the `matcha-tts`
components CosyVoice imports rather than reimplements
(`matcha/models/components/{decoder,transformer,flow_matching}.py`) -- not assumed
from CosyVoice1's non-causal `ConditionalDecoder`/`ConditionalCFM`.

**Causality is real new work here, not a pass-through** (unlike the vocoder, where
the causal/non-causal split turned out not to matter structurally):

* `CausalBlock1D`/`CausalResnetBlock1D` don't just left-pad their convs -- they swap
  `GroupNorm` for `LayerNorm`. GroupNorm's statistics span the whole sequence per
  sample; a streaming decoder needs a per-timestep norm so a longer prefix's early
  output doesn't change. This is a real output-value difference from CosyVoice1's
  decoder, not just an added mask.
* `CausalConditionalCFM` draws its ODE's initial noise from a **fixed, pre-seeded
  buffer** (`set_all_random_seed(0)`; `torch.randn([1, 80, 50*300])`, sliced to
  length) instead of a fresh `torch.randn_like(mu)` per call, and it drops the
  `cache`/`flow_cache` overlap-splicing that CosyVoice1's `ConditionalCFM.forward`
  has entirely. The fixed buffer is what makes a longer chunk's noise a genuine
  continuation of a shorter chunk's -- both facts are essential to replicate exactly
  for a correctness claim, not simplifications to make on our own.
* `CausalMaskedDiffWithXvec` (the caller, a separate, later module -- see
  `tt/flow/flow.py`) has no `length_regulator` at all: the token-rate -> mel-rate
  upsampling moves inside the encoder (`UpsampleConformerEncoder`'s own `Upsample1D`),
  which is a real architecture-shape difference, not addressed here.

**What genuinely transfers unchanged** (the vocoder experience repeats for these
specific pieces): `BasicTransformerBlock` itself is identical causal or not -- only
the `attention_mask` argument differs; it is plain LayerNorm + self-attention + GELU
FFN, no AdaLN (CosyVoice never sets `norm_type="ada_norm"`, so the `timestep` kwarg
each block receives is accepted but unused -- time conditioning only enters through
`CausalResnetBlock1D`'s additive FiLM-style bias). `SinusoidalPosEmb`/
`TimestepEmbedding` are untouched. The 1x1 `res_conv` inside each resnet block is
trivially causal already (kernel=1).

**Scope of this module, verified from the real checkpoint's own config**
(`cosyvoice2.yaml`'s `flow.decoder.estimator`, not the class's generic default):
`channels=[256]` -- a SINGLE down/mid/up stage, not the `CausalConditionalDecoder`
class's own two-stage default `(256, 256)`. Tracing `CausalConditionalDecoder.__init__`'s
arithmetic at this config shows `is_last` is always `True` in both the down-block and
up-block loops (each loop runs exactly once), so the non-causal `Downsample1D`/
`matcha.Upsample1D(use_conv_transpose=True)` classes are **never instantiated** for
this checkpoint -- only `CausalConv1d` (kernel=3, stride=1, shape-preserving) is ever
used for the down/up legs. There is no real spatial down/up-sampling anywhere in this
U-Net: it is a flat-resolution stack (1 causal down-block + 12 mid-blocks + 1 causal
up-block, all at the input's own temporal resolution, `channels=[256]` throughout,
`n_blocks=4` transformer blocks per stage). This module hardcodes that verified
1-stage topology rather than the class's general N-stage form -- if a future
checkpoint ships a different `channels` list, this module needs extending, not
silently mis-porting the wrong shape.

`streaming=False` only (the non-streaming, single-shot path): `add_optional_chunk_mask`
with `static_chunk_size=0` and `use_dynamic_chunk=False` (confirmed from
`cosyvoice/utils/mask.py`) collapses to the plain padding mask with no chunk-block
restriction -- so this is exactly the same attention pattern a non-causal decoder
would use, and the streaming chunk-mask path is not built here. See tt/flow/flow.py's
module docstring for why the streaming path's absence of any cross-call cache (unlike
the "chunk-seam corruption" bug documented for a *different* port's own carried-state
design) makes this a safe place to stop for this bring-up phase.

Tensors are `[N, L, C]` throughout, per this package's existing channels-last
convention (conv.py). Upstream's `[B, C, T]` channel-dim concatenations
(`einops.pack([x, mu], "b * t")`) become last-axis concatenations here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

from ..hifigan.conv import accurate_compute_config, safe_compute_config

# The real checkpoint's verified estimator config (cosyvoice2.yaml's
# flow.decoder.estimator) -- see module docstring for why this is hardcoded rather
# than generic.
IN_CHANNELS = 320  # x(80) + mu(80) + spks(80, broadcast) + cond(80)
OUT_CHANNELS = 80
CHANNELS = 256
N_BLOCKS = 4  # transformer blocks per resnet stage
NUM_MID_BLOCKS = 12
NUM_HEADS = 8
ATTENTION_HEAD_DIM = 64
INNER_DIM = NUM_HEADS * ATTENTION_HEAD_DIM  # 512 -- an expand-then-project attention
TIME_EMBED_DIM = CHANNELS * 4  # 1024


# ---------------------------------------------------------------------------
# torch reference -- real nn.Module submodules (nn.Conv1d/nn.Linear/nn.LayerNorm),
# real torch.nn.functional ops for every primitive (mish, gelu,
# scaled_dot_product_attention) rather than a hand-derived reimplementation, so a
# comparison against this reference has no "shared bug" risk for those pieces --
# matching this package's existing TorchHiFTDecodeRef pattern (matcha-tts/diffusers
# are not installed in this environment, so importing the real classes directly is
# not available; this is a faithful line-by-line transcription of the real source
# read for this port, not a redesign).
# ---------------------------------------------------------------------------


def sinusoidal_pos_emb_torch(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """`matcha.models.components.decoder.SinusoidalPosEmb.forward`, verbatim."""
    if t.ndim < 1:
        t = t.unsqueeze(0)
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
    emb = scale * t.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbeddingRef(nn.Module):
    """`matcha.models.components.decoder.TimestepEmbedding`, the `act_fn="silu"`
    path only (the only one CosyVoice uses): Linear -> SiLU -> Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class CausalConv1dRef(nn.Conv1d):
    """`cosyvoice.flow.decoder.CausalConv1d`: left-pad by `kernel_size - 1`, then a
    plain `padding=0` conv -- so position `i` of the output only ever depends on
    input positions `<= i`."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=bias)
        self.causal_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.causal_padding, 0), value=0.0))


class CausalBlock1DRef(nn.Module):
    """`cosyvoice.flow.decoder.CausalBlock1D`: CausalConv1d -> LayerNorm(over
    channels) -> Mish. GroupNorm -> LayerNorm is the real, causality-motivated swap
    from matcha's plain `Block1D` (see module docstring)."""

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = CausalConv1dRef(dim, dim_out, 3)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.conv(x * mask)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.mish(x)
        return x * mask


class CausalResnetBlock1DRef(nn.Module):
    """`cosyvoice.flow.decoder.CausalResnetBlock1D` (via matcha's `ResnetBlock1D`
    structure, unchanged -- only its two `Block1D`s become causal)."""

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = CausalBlock1DRef(dim, dim_out)
        self.block2 = CausalBlock1DRef(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class BasicTransformerBlockRef(nn.Module):
    """`matcha.models.components.transformer.BasicTransformerBlock` at the config
    CosyVoice actually instantiates it with: no cross-attention
    (`cross_attention_dim=None`, `double_self_attention=False`), no AdaLN
    (`norm_type` defaults to `"layer_norm"`, so `timestep` is accepted but never
    used), `activation_fn="gelu"` (plain GELU FFN, not GEGLU). Attention is
    `diffusers.models.attention_processor.Attention`'s standard scaled dot-product
    form: `inner_dim = num_heads * attention_head_dim` (512), which need not equal
    `dim` (256) -- an expand-then-project-back attention, not equal-width. Q/K/V
    have no bias (`attention_bias=False`, diffusers' default); the output
    projection does.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads, self.head_dim = num_heads, head_dim
        self.norm1 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, dim * 4)
        self.ff_out = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """x: [B, T, dim]. attn_bias: additive, broadcastable to [B, 1, T, T]."""
        b, t, _ = x.shape
        h = self.norm1(x)

        def _heads(m):
            return m.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _heads(self.to_q(h)), _heads(self.to_k(h)), _heads(self.to_v(h))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        attn = attn.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        x = x + self.to_out(attn)

        h = self.norm3(x)
        h = self.ff_out(F.gelu(self.ff_in(h)))
        return x + h


class CausalConditionalDecoderRef(nn.Module):
    """`cosyvoice.flow.decoder.CausalConditionalDecoder` at the real checkpoint's
    1-stage config (see module docstring). `streaming=False` only.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        channels: int = CHANNELS,
        n_blocks: int = N_BLOCKS,
        num_mid_blocks: int = NUM_MID_BLOCKS,
        num_heads: int = NUM_HEADS,
        head_dim: int = ATTENTION_HEAD_DIM,
    ):
        super().__init__()
        self.channels = channels
        time_embed_dim = channels * 4
        self.time_embeddings_dim = in_channels
        self.time_mlp = TimestepEmbeddingRef(in_channels, time_embed_dim)

        self.down_resnet = CausalResnetBlock1DRef(in_channels, channels, time_embed_dim)
        self.down_tbs = nn.ModuleList(
            [BasicTransformerBlockRef(channels, num_heads, head_dim) for _ in range(n_blocks)]
        )
        self.down_conv = CausalConv1dRef(channels, channels, 3)

        self.mid_resnets = nn.ModuleList(
            [CausalResnetBlock1DRef(channels, channels, time_embed_dim) for _ in range(num_mid_blocks)]
        )
        self.mid_tbs = nn.ModuleList(
            [
                nn.ModuleList([BasicTransformerBlockRef(channels, num_heads, head_dim) for _ in range(n_blocks)])
                for _ in range(num_mid_blocks)
            ]
        )

        self.up_resnet = CausalResnetBlock1DRef(channels * 2, channels, time_embed_dim)
        self.up_tbs = nn.ModuleList([BasicTransformerBlockRef(channels, num_heads, head_dim) for _ in range(n_blocks)])
        self.up_conv = CausalConv1dRef(channels, channels, 3)

        self.final_block = CausalBlock1DRef(channels, channels)
        self.final_proj = nn.Conv1d(channels, out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        """`cosyvoice.flow.decoder.ConditionalDecoder.initialize_weights`, verbatim."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """x/mu/cond: [B, T, 80]. mask: [B, T, 1] (1.0 = valid). spks: [B, 80]. t: [B]."""
        temb = sinusoidal_pos_emb_torch(t, self.time_embeddings_dim)
        temb = self.time_mlp(temb)

        h = torch.cat([x, mu], dim=-1)
        h = torch.cat([h, spks.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)
        h = torch.cat([h, cond], dim=-1)

        # channel-first for the resnet/conv legs (matches nn.Conv1d's own layout);
        # channel-last for LayerNorm/attention. Transposed at each boundary, exactly
        # like the real source's own `rearrange` calls.
        mask_cl = mask  # [B, T, 1]
        attn_bias = (1.0 - mask_cl.transpose(1, 2)).float() * -1.0e10  # [B, 1, T] -> broadcasts to [B,1,Tq,Tk]
        attn_bias = attn_bias.unsqueeze(1)

        h = h.transpose(1, 2)  # [B, in_channels, T]
        mask_cf = mask.transpose(1, 2)  # [B, 1, T]
        h = self.down_resnet(h, mask_cf, temb)
        h = h.transpose(1, 2)
        for tb in self.down_tbs:
            h = tb(h, attn_bias)
        skip = h
        h = h.transpose(1, 2)
        h = self.down_conv(h * mask_cf)

        for resnet, tbs in zip(self.mid_resnets, self.mid_tbs):
            h = resnet(h, mask_cf, temb)
            h = h.transpose(1, 2)
            for tb in tbs:
                h = tb(h, attn_bias)
            h = h.transpose(1, 2)

        h = h.transpose(1, 2)  # [B, T, channels]
        h = torch.cat([h, skip], dim=-1)
        h = h.transpose(1, 2)
        h = self.up_resnet(h, mask_cf, temb)
        h = h.transpose(1, 2)
        for tb in self.up_tbs:
            h = tb(h, attn_bias)
        h = h.transpose(1, 2)
        h = self.up_conv(h * mask_cf)

        h = self.final_block(h, mask_cf)
        out = self.final_proj(h * mask_cf)
        out = out.transpose(1, 2)  # [B, T, 80]
        return out * mask


class CausalConditionalCFMRef:
    """`cosyvoice.flow.flow_matching.CausalConditionalCFM`: the fixed-noise-buffer
    Euler solver wrapped around `CausalConditionalDecoderRef`. `streaming=False`
    only (see module docstring).
    """

    def __init__(
        self,
        estimator: CausalConditionalDecoderRef,
        sigma_min: float = 1e-6,
        t_scheduler: str = "cosine",
        inference_cfg_rate: float = 0.7,
        seed: int = 0,
    ):
        self.estimator = estimator
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.inference_cfg_rate = inference_cfg_rate
        # `set_all_random_seed(0)` then `torch.randn([1, 80, 50*300])`, verbatim --
        # a fixed buffer, not fresh-per-call noise. Stored channel-last here
        # ([1, 15000, 80]) to match this module's convention.
        torch.manual_seed(seed)
        self.rand_noise = torch.randn(1, 80, 50 * 300).transpose(1, 2).contiguous()

    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        spks: torch.Tensor,
        cond: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        t_len = mu.shape[1]
        z = self.rand_noise[:, :t_len, :].to(mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond)

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        b, t_len, c = x.shape
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]

        x_in = torch.zeros(2 * b, t_len, c, dtype=x.dtype)
        mask_in = torch.zeros(2 * b, t_len, 1, dtype=x.dtype)
        mu_in = torch.zeros(2 * b, t_len, c, dtype=x.dtype)
        t_in = torch.zeros(2 * b, dtype=x.dtype)
        spks_in = torch.zeros(2 * b, spks.shape[-1], dtype=x.dtype)
        cond_in = torch.zeros(2 * b, t_len, c, dtype=x.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[:b] = mu
            t_in[:] = t
            spks_in[:b] = spks
            cond_in[:b] = cond
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = dphi_dt[:b], dphi_dt[b:]
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x


# ---------------------------------------------------------------------------
# TTNN port. Channels-last ([N, L, C]) throughout, per this package's convention --
# unlike the torch reference above, which mirrors upstream's own [B, C, T] storage
# internally (with transposes at the attention/LayerNorm boundaries), the TT port
# never needs those transposes: `ttnn.conv1d` and `ttnn.layer_norm` both already
# operate channels-last natively.
# ---------------------------------------------------------------------------


def _linear_weight(device, weight: torch.Tensor, dtype):
    """torch Linear weight [out, in] -> ttnn [in, out], matching ttnn.linear's
    expected orientation (see TtLinearHead in tt/llm/qwen2lm.py, the same
    convention used there)."""
    return ttnn.from_torch(
        weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


def _bias(device, bias: torch.Tensor, dtype):
    return ttnn.from_torch(bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class TtCausalConv1d:
    """`CausalConv1dRef` on device: `ttnn.conv1d` with asymmetric `padding=(k-1, 0)`
    -- confirmed from `ttnn.conv1d`'s own docstring that `padding` accepts a
    `[pad_left, pad_right]` tuple directly, so this needs no manual pad step (which
    would hit `ttnn.pad`'s documented "front padding on device not supported in
    tile layout" restriction). Not `prepare_conv_weights`-cached (unlike
    `hifigan.conv.TtConv1d`) -- correctness first, matching this bring-up's
    consistent priority; still verified against `safe_compute_config` per geometry
    (see `accurate_compute_config`'s docstring for the measured, silent-corruption
    reason this matters), reusing the exact same two configs the vocoder already
    validated rather than re-deriving them.
    """

    def __init__(
        self, device, weight: torch.Tensor, bias: torch.Tensor, dtype=ttnn.bfloat16, weights_dtype=ttnn.bfloat16
    ):
        assert weight.dim() == 3
        self.device = device
        self.out_channels, self.in_channels, self.kernel_size = weight.shape
        self.dtype = dtype
        self._weight_4d = ttnn.from_torch(
            weight.detach().float().unsqueeze(2), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self._bias = ttnn.from_torch(
            bias.detach().float().reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=weights_dtype, deallocate_activation=False)
        self._accurate = accurate_compute_config(device)
        self._safe = safe_compute_config(device)
        self._verified: dict = {}

    @classmethod
    def from_module(cls, device, module: CausalConv1dRef, dtype=ttnn.bfloat16):
        return cls(device, module.weight, module.bias, dtype=dtype)

    def _conv(self, x, input_length: int, batch_size: int, compute_config):
        return ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self._weight_4d,
            bias_tensor=self._bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_length=input_length,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size - 1, 0),
            dilation=1,
            groups=1,
            conv_config=self.conv_config,
            compute_config=compute_config,
            dtype=self.dtype,
            return_output_dim=True,
        )

    def __call__(self, x, input_length: int, batch_size: int = 1):
        key = (input_length, batch_size)
        cfg = self._verified.get(key, self._accurate)
        out, out_length = self._conv(x, input_length, batch_size, cfg)
        if key not in self._verified:
            ref, _ = self._conv(x, input_length, batch_size, self._safe)
            a = float(ttnn.to_torch(out).float().abs().max())
            b = float(ttnn.to_torch(ref).float().abs().max())
            if a == a and abs(a - b) <= 0.02 * max(b, 1e-9):
                self._verified[key] = self._accurate
                ttnn.deallocate(ref)
            else:
                self._verified[key] = self._safe
                ttnn.deallocate(out)
                out = ref
        out = ttnn.reshape(out, (batch_size, out_length, self.out_channels))
        return out


class TtCausalBlock1D:
    def __init__(self, device, module: CausalBlock1DRef, dtype=ttnn.bfloat16):
        self.conv = TtCausalConv1d.from_module(device, module.conv, dtype=dtype)
        self.norm_weight = ttnn.from_torch(
            module.norm.weight.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm_bias = ttnn.from_torch(
            module.norm.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, x, mask, length: int, batch_size: int = 1):
        h = ttnn.multiply(x, mask)
        h = self.conv(h, length, batch_size)
        h = ttnn.layer_norm(h, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5)
        h = ttnn.mish(h)
        return ttnn.multiply(h, mask)


class TtCausalResnetBlock1D:
    def __init__(self, device, module: CausalResnetBlock1DRef, dtype=ttnn.bfloat16):
        self.block1 = TtCausalBlock1D(device, module.block1, dtype=dtype)
        self.block2 = TtCausalBlock1D(device, module.block2, dtype=dtype)
        mlp_linear = module.mlp[1]  # nn.Sequential(Mish, Linear)
        self.mlp_weight = _linear_weight(device, mlp_linear.weight, dtype)
        self.mlp_bias = _bias(device, mlp_linear.bias, dtype)
        self.res_weight = _linear_weight(device, module.res_conv.weight.squeeze(-1), dtype)
        self.res_bias = _bias(device, module.res_conv.bias, dtype)

    def __call__(self, x, mask, time_emb, length: int, batch_size: int = 1):
        """time_emb: `[2B, 1, time_embed_dim]` -- already shaped to broadcast
        against `h`'s `[2B, T, dim_out]` with no reshape needed."""
        h = self.block1(x, mask, length, batch_size)
        temb = ttnn.mish(time_emb)
        temb = ttnn.linear(temb, self.mlp_weight, bias=self.mlp_bias)
        h = ttnn.add(h, temb)
        h = self.block2(h, mask, length, batch_size)
        skip = ttnn.multiply(x, mask)
        skip = ttnn.linear(skip, self.res_weight, bias=self.res_bias)
        return ttnn.add(h, skip)


class TtBasicTransformerBlock:
    def __init__(self, device, module: BasicTransformerBlockRef, dtype=ttnn.bfloat16):
        self.num_heads, self.head_dim = module.num_heads, module.head_dim
        self.norm1_w = ttnn.from_torch(
            module.norm1.weight.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm1_b = ttnn.from_torch(
            module.norm1.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm3_w = ttnn.from_torch(
            module.norm3.weight.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm3_b = ttnn.from_torch(
            module.norm3.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.wq = _linear_weight(device, module.to_q.weight, dtype)
        self.wk = _linear_weight(device, module.to_k.weight, dtype)
        self.wv = _linear_weight(device, module.to_v.weight, dtype)
        self.wo = _linear_weight(device, module.to_out.weight, dtype)
        self.bo = _bias(device, module.to_out.bias, dtype)
        self.w_ff_in = _linear_weight(device, module.ff_in.weight, dtype)
        self.b_ff_in = _bias(device, module.ff_in.bias, dtype)
        self.w_ff_out = _linear_weight(device, module.ff_out.weight, dtype)
        self.b_ff_out = _bias(device, module.ff_out.bias, dtype)
        self.scale = module.head_dim**-0.5

    def _heads(self, x, b, t):
        x = ttnn.reshape(x, (b, t, self.num_heads, self.head_dim))
        return ttnn.transpose(x, 1, 2)  # [B, heads, T, head_dim]

    def __call__(self, x, attn_bias):
        b, t, _ = x.shape
        h = ttnn.layer_norm(x, weight=self.norm1_w, bias=self.norm1_b, epsilon=1e-5)
        q = self._heads(ttnn.linear(h, self.wq), b, t)
        k = self._heads(ttnn.linear(h, self.wk), b, t)
        v = self._heads(ttnn.linear(h, self.wv), b, t)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        scores = ttnn.multiply(scores, self.scale)
        scores = ttnn.add(scores, attn_bias)
        attn = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn, v)  # [B, heads, T, head_dim]
        out = ttnn.transpose(out, 1, 2)
        out = ttnn.reshape(out, (b, t, self.num_heads * self.head_dim))
        out = ttnn.linear(out, self.wo, bias=self.bo)
        x = ttnn.add(x, out)

        h = ttnn.layer_norm(x, weight=self.norm3_w, bias=self.norm3_b, epsilon=1e-5)
        h = ttnn.linear(h, self.w_ff_in, bias=self.b_ff_in)
        h = ttnn.gelu(h)
        h = ttnn.linear(h, self.w_ff_out, bias=self.b_ff_out)
        return ttnn.add(x, h)


class TtCausalConditionalDecoder:
    """`CausalConditionalDecoderRef` on device -- see that class and the module
    docstring for the verified 1-stage topology this hardcodes."""

    def __init__(self, device, module: CausalConditionalDecoderRef, dtype=ttnn.bfloat16):
        self.device = device
        self.time_embeddings_dim = module.time_embeddings_dim
        self.channels = module.channels
        self.time_mlp_w1 = _linear_weight(device, module.time_mlp.linear_1.weight, dtype)
        self.time_mlp_b1 = _bias(device, module.time_mlp.linear_1.bias, dtype)
        self.time_mlp_w2 = _linear_weight(device, module.time_mlp.linear_2.weight, dtype)
        self.time_mlp_b2 = _bias(device, module.time_mlp.linear_2.bias, dtype)

        self.down_resnet = TtCausalResnetBlock1D(device, module.down_resnet, dtype=dtype)
        self.down_tbs = [TtBasicTransformerBlock(device, tb, dtype=dtype) for tb in module.down_tbs]
        self.down_conv = TtCausalConv1d.from_module(device, module.down_conv, dtype=dtype)

        self.mid_resnets = [TtCausalResnetBlock1D(device, m, dtype=dtype) for m in module.mid_resnets]
        self.mid_tbs = [[TtBasicTransformerBlock(device, tb, dtype=dtype) for tb in tbs] for tbs in module.mid_tbs]

        self.up_resnet = TtCausalResnetBlock1D(device, module.up_resnet, dtype=dtype)
        self.up_tbs = [TtBasicTransformerBlock(device, tb, dtype=dtype) for tb in module.up_tbs]
        self.up_conv = TtCausalConv1d.from_module(device, module.up_conv, dtype=dtype)

        self.final_block = TtCausalBlock1D(device, module.final_block, dtype=dtype)
        self.final_weight = _linear_weight(device, module.final_proj.weight.squeeze(-1), dtype)
        self.final_bias = _bias(device, module.final_proj.bias, dtype)

    def _sinusoidal_pos_emb(self, t: torch.Tensor) -> ttnn.Tensor:
        """Computed on host (cheap: t is a length-2*B vector, not a mel-length
        tensor) and uploaded directly at `[2B, 1, dim]` -- the shape every
        downstream use (broadcast-add against `[2B, T, dim_out]`) actually needs,
        so no on-device reshape ever has to touch this tensor's tile-critical last
        two dims (a `[N, 1, dim]` reshape of an already-tiled `[N, dim]` tensor hit
        `TT_FATAL: Invalid arguments to reshape` -- inserting a non-tile-aligned
        size-1 dim into an already-tiled tensor is not a valid in-place reshape;
        building the right shape on the host side, where it is free, avoids it
        entirely).
        """
        emb = sinusoidal_pos_emb_torch(t, self.time_embeddings_dim)
        return ttnn.from_torch(
            emb.reshape(emb.shape[0], 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def __call__(self, x, mask, mu, t: torch.Tensor, spks, cond, length: int, batch_size: int):
        temb = self._sinusoidal_pos_emb(t)  # [2B, 1, time_embeddings_dim]
        temb = ttnn.linear(temb, self.time_mlp_w1, bias=self.time_mlp_b1)
        temb = ttnn.silu(temb)
        temb = ttnn.linear(temb, self.time_mlp_w2, bias=self.time_mlp_b2)  # [2B, 1, time_embed_dim]

        h = ttnn.concat([x, mu], dim=-1)
        # spks arrives pre-shaped [2B, 1, spk_dim] (see TtCausalConditionalCFM --
        # uploaded that way on the host to avoid the same tile-reshape pitfall
        # `_sinusoidal_pos_emb` documents).
        h = ttnn.concat([h, ttnn.repeat(spks, ttnn.Shape((1, h.shape[1], 1)))], dim=-1)
        h = ttnn.concat([h, cond], dim=-1)

        attn_bias = ttnn.multiply(ttnn.subtract(mask, 1.0), 1.0e10)  # (mask-1)*1e10 == (1-mask)*-1e10
        attn_bias = ttnn.transpose(attn_bias, 1, 2)  # [B, 1, T]
        attn_bias = ttnn.reshape(attn_bias, (attn_bias.shape[0], 1, 1, attn_bias.shape[-1]))  # [B,1,1,T]

        h = self.down_resnet(h, mask, temb, length, batch_size)
        for tb in self.down_tbs:
            h = tb(h, attn_bias)
        skip = h
        h = self.down_conv(ttnn.multiply(h, mask), length, batch_size)

        for resnet, tbs in zip(self.mid_resnets, self.mid_tbs):
            h = resnet(h, mask, temb, length, batch_size)
            for tb in tbs:
                h = tb(h, attn_bias)

        h = ttnn.concat([h, skip], dim=-1)
        h = self.up_resnet(h, mask, temb, length, batch_size)
        for tb in self.up_tbs:
            h = tb(h, attn_bias)
        h = self.up_conv(ttnn.multiply(h, mask), length, batch_size)

        h = self.final_block(h, mask, length, batch_size)
        out = ttnn.linear(ttnn.multiply(h, mask), self.final_weight, bias=self.final_bias)
        return ttnn.multiply(out, mask)


class TtCausalConditionalCFM:
    """`CausalConditionalCFMRef` on device -- same fixed-noise-buffer, same Euler
    solver, same CFG-doubling batch-of-2 trick, built directly against the ttnn
    estimator above rather than round-tripping through torch for the solve."""

    def __init__(
        self, device, estimator: TtCausalConditionalDecoder, rand_noise: torch.Tensor, cfm_ref: CausalConditionalCFMRef
    ):
        self.device = device
        self.estimator = estimator
        self.rand_noise = rand_noise  # host tensor, [1, 15000, 80] -- see CausalConditionalCFMRef
        self.t_scheduler = cfm_ref.t_scheduler
        self.inference_cfg_rate = cfm_ref.inference_cfg_rate

    def forward(
        self, mu_t: torch.Tensor, mask_t: torch.Tensor, n_timesteps: int, spks_t: torch.Tensor, cond_t: torch.Tensor
    ):
        """All *_t args are torch tensors (host), batch=1: mu/cond [1,T,80], mask
        [1,T,1], spks [1,80]. Returns a torch tensor [1,T,80] -- the solve loop's
        conditioning tensors (mu/spks/cond/mask) are uploaded once, CFG-doubled, and
        reused every Euler step; only x and t change per step.
        """
        t_len = mu_t.shape[1]
        z = self.rand_noise[:, :t_len, :].to(mu_t.dtype)
        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=mu_t.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        zero_mu = torch.zeros_like(mu_t)
        zero_spks = torch.zeros_like(spks_t)
        zero_cond = torch.zeros_like(cond_t)
        mu_in = torch.cat([mu_t, zero_mu], dim=0)
        spks_in = torch.cat([spks_t, zero_spks], dim=0)
        cond_in = torch.cat([cond_t, zero_cond], dim=0)
        mask_in = torch.cat([mask_t, mask_t], dim=0)

        mu_dev = ttnn.from_torch(mu_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        # [2B, 1, spk_dim], not [2B, spk_dim] -- see TtCausalConditionalDecoder's
        # spks handling / _sinusoidal_pos_emb's docstring for why.
        spks_dev = ttnn.from_torch(
            spks_in.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        cond_dev = ttnn.from_torch(cond_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        mask_dev = ttnn.from_torch(mask_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]
        x = z
        for step in range(1, len(t_span)):
            x_in = torch.cat([x, x], dim=0)
            t_in = t.repeat(2)
            x_dev = ttnn.from_torch(x_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            dphi_dev = self.estimator(x_dev, mask_dev, mu_dev, t_in, spks_dev, cond_dev, t_len, batch_size=2)
            dphi = ttnn.to_torch(dphi_dev).float()
            dphi_dt, cfg_dphi_dt = dphi[:1], dphi[1:2]
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt.to(x.dtype)
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x
