# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the LLVC voice-conversion generator.

The model is constructed directly from a PyTorch reference ``Net`` instance
(see ``models/demos/llvc/reference/llvc_reference.py``): weights are read by
attribute and materialised as TTNN tensors. This keeps weight plumbing local
and avoids a fragile string-keyed state dict.

Layout convention: activations flow as ``[B, T, C]`` (channels-last), which is
the transpose of the reference's ``[B, C, T]``. Streaming state (all the ring
buffers the reference threads through ``forward``) is held in ``LLVCState``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn
from models.demos.llvc.reference.llvc_reference import Net, build_reference_model
from models.demos.llvc.tt import ops
from models.demos.llvc.tt.config import LLVCConfig, get_math_fidelity, get_ttnn_dtype


@dataclass
class LLVCState:
    """Per-stream causal buffers, mirroring the reference's threaded state."""

    convnet_ctx: list[ttnn.Tensor]
    enc_ctx: list[ttnn.Tensor]
    dec_mem_ctx: ttnn.Tensor
    dec_tgt_ctx: list[ttnn.Tensor]
    out_buf: ttnn.Tensor


def _to_device(t: torch.Tensor, *, device, dtype, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t.contiguous(), device=device, dtype=dtype, layout=layout)


class LLVCModel:
    """LLVC generator on Tenstorrent hardware via TTNN."""

    def __init__(self, config: LLVCConfig, reference: Net, *, device):
        self.config = config
        self.device = device
        self.dtype = get_ttnn_dtype(config.dtype)
        self.weights_dtype = get_ttnn_dtype(config.dtype)
        self.math_fidelity = get_math_fidelity(config.math_fidelity)
        self.mem_cfg = ttnn.L1_MEMORY_CONFIG if config.use_l1 else ttnn.DRAM_MEMORY_CONFIG

        self._conv_weight_cache: dict[str, ttnn.Tensor] = {}
        self._prepare_weights(reference)

        if config.use_program_cache:
            device.enable_program_cache()

    # ------------------------------------------------------------------ weights
    def _host_conv_weight(self, w: torch.Tensor) -> ttnn.Tensor:
        wd = ttnn.float32 if self.weights_dtype == ttnn.bfloat8_b else self.weights_dtype
        return ttnn.from_torch(w.contiguous(), wd)

    def _host_conv_bias(self, b: Optional[torch.Tensor], out_ch: int) -> Optional[ttnn.Tensor]:
        if b is None:
            return None
        wd = ttnn.float32 if self.weights_dtype == ttnn.bfloat8_b else self.weights_dtype
        return ttnn.from_torch(b.reshape(1, 1, 1, out_ch).contiguous(), wd)

    def _tap_weights(self, conv_weight: torch.Tensor) -> list[ttnn.Tensor]:
        """Split a ``[Cout, Cin, K]`` conv weight into ``K`` linear taps ``[Cout, Cin]``."""
        k = conv_weight.shape[-1]
        return [_to_device(conv_weight[:, :, j], device=self.device, dtype=self.dtype) for j in range(k)]

    def _depthwise_taps(self, conv_weight: torch.Tensor) -> list[ttnn.Tensor]:
        """Split a depthwise ``[C, 1, K]`` conv weight into ``K`` per-channel vectors ``[1, 1, C]``."""
        k = conv_weight.shape[-1]
        return [
            _to_device(conv_weight[:, 0, j].reshape(1, 1, -1), device=self.device, dtype=self.dtype) for j in range(k)
        ]

    def _depthwise_bias(self, bias: Optional[torch.Tensor]) -> Optional[ttnn.Tensor]:
        if bias is None:
            return None
        return _to_device(bias.reshape(1, 1, -1), device=self.device, dtype=self.dtype)

    def _tap_biases(self, bias: Optional[torch.Tensor], kernel: int) -> Optional[list[ttnn.Tensor]]:
        # Conv bias is added once, not per-tap; fold it onto the first tap only.
        if bias is None:
            return None
        biases = [_to_device(bias.reshape(1, 1, -1), device=self.device, dtype=self.dtype)]
        for _ in range(kernel - 1):
            biases.append(None)
        return biases

    def _prepare_weights(self, ref: Net):
        cfg = self.config
        dev = self.device

        # --- cached conv prenet (12 gated residual blocks, in=out=1, k=3) ---
        self.convnet_taps = []  # list of dicts {filter, gate, filter_bias, gate_bias}
        if hasattr(ref, "convnet_pre"):
            for block in ref.convnet_pre.down_convs:
                self.convnet_taps.append(
                    {
                        "filter": self._tap_weights(block.filter.weight.detach().float()),
                        "filter_bias": self._tap_biases(
                            block.filter.bias.detach().float() if block.filter.bias is not None else None, 3
                        ),
                        "gate": self._tap_weights(block.gate.weight.detach().float()),
                        "gate_bias": self._tap_biases(
                            block.gate.bias.detach().float() if block.gate.bias is not None else None, 3
                        ),
                        "buf_len": (block.filter.kernel_size[0] - 1) * block.filter.dilation[0],
                    }
                )

        # --- in_conv: Conv1d(1, enc_dim, k=3L, stride=L), no bias ---
        self.in_conv_w = self._host_conv_weight(ref.in_conv[0].weight.detach().float())

        # --- constant label embedding (label is always zeros) ---
        with torch.no_grad():
            l_const = ref.label_embedding(torch.zeros(1, 1))  # [1, enc_dim]
        self.l_const = _to_device(l_const.reshape(1, 1, cfg.enc_dim), device=dev, dtype=self.dtype)

        # --- dilated causal encoder (8 depthwise-separable layers) ---
        enc = ref.mask_gen.encoder
        self.enc_layers = []
        for i, dcc in enumerate(enc.dcc_layers):
            layers = dcc.layers
            depthwise = layers[0]
            ln1 = layers[1]
            pointwise = layers[3]
            ln2 = layers[4]
            self.enc_layers.append(
                {
                    "dw_taps": self._depthwise_taps(depthwise.weight.detach().float()),
                    "dw_bias": self._depthwise_bias(
                        depthwise.bias.detach().float() if depthwise.bias is not None else None
                    ),
                    "dilation": 2**i,
                    "buf_len": (depthwise.kernel_size[0] - 1) * (2**i),
                    "ln1_w": _to_device(ln1.weight.detach().float(), device=dev, dtype=self.dtype),
                    "ln1_b": _to_device(ln1.bias.detach().float(), device=dev, dtype=self.dtype),
                    "pw_w": _to_device(pointwise.weight.detach().float().squeeze(-1), device=dev, dtype=self.dtype),
                    "pw_b": _to_device(pointwise.bias.detach().float(), device=dev, dtype=self.dtype),
                    "ln2_w": _to_device(ln2.weight.detach().float(), device=dev, dtype=self.dtype),
                    "ln2_b": _to_device(ln2.bias.detach().float(), device=dev, dtype=self.dtype),
                }
            )

        # --- e2d/d2e grouped 1x1 projections ---
        mg = ref.mask_gen
        self.proj_e2d_e_w = self._host_conv_weight(mg.proj_e2d_e[0].weight.detach().float())
        self.proj_e2d_e_b = self._host_conv_bias(mg.proj_e2d_e[0].bias.detach().float(), cfg.dec_dim)
        self.proj_e2d_l_w = self._host_conv_weight(mg.proj_e2d_l[0].weight.detach().float())
        self.proj_e2d_l_b = self._host_conv_bias(mg.proj_e2d_l[0].bias.detach().float(), cfg.dec_dim)
        self.proj_d2e_w = self._host_conv_weight(mg.proj_d2e[0].weight.detach().float())
        self.proj_d2e_b = self._host_conv_bias(mg.proj_d2e[0].bias.detach().float(), cfg.enc_dim)
        self.proj_e2d_groups = mg.proj_e2d_e[0].groups
        self.proj_d2e_groups = mg.proj_d2e[0].groups

        # --- transformer decoder layers ---
        self.dec_layers = []
        for layer in mg.decoder.tf_dec_layers:
            self.dec_layers.append(self._prepare_decoder_layer(layer))
        self.dec_pos_enc = ops.sinusoidal_position_encoding(
            cfg.dec_buf_len + cfg.dec_chunk_size, cfg.dec_dim, device=dev, dtype=self.dtype
        )

        # --- out_conv: ConvTranspose1d(enc_dim, 1, k=(out_buf_len+1)L, stride=L), no bias ---
        # ttnn.conv_transpose2d weight layout is [Cin, Cout, kH, kW]; torch ConvTranspose1d
        # weight is [Cin, Cout, k] -> add singleton kW.
        w = ref.out_conv[0].weight.detach().float().unsqueeze(-1)  # [enc_dim, 1, k, 1]
        self.out_conv_w = self._host_conv_weight(w)

    def _prepare_decoder_layer(self, layer) -> dict:
        dev = self.device
        d = self.config.dec_dim

        def split_in_proj(mha):
            w = mha.in_proj_weight.detach().float()  # [3D, D]
            b = mha.in_proj_bias.detach().float()  # [3D]
            qw, kw, vw = w[:d], w[d : 2 * d], w[2 * d :]
            qb, kb, vb = b[:d], b[d : 2 * d], b[2 * d :]
            return {
                "qw": _to_device(qw, device=dev, dtype=self.dtype),
                "qb": _to_device(qb, device=dev, dtype=self.dtype),
                "kw": _to_device(kw, device=dev, dtype=self.dtype),
                "kb": _to_device(kb, device=dev, dtype=self.dtype),
                "vw": _to_device(vw, device=dev, dtype=self.dtype),
                "vb": _to_device(vb, device=dev, dtype=self.dtype),
                "ow": _to_device(mha.out_proj.weight.detach().float(), device=dev, dtype=self.dtype),
                "ob": _to_device(mha.out_proj.bias.detach().float(), device=dev, dtype=self.dtype),
            }

        return {
            "self_attn": split_in_proj(layer.self_attn),
            "cross_attn": split_in_proj(layer.multihead_attn),
            "l1_w": _to_device(layer.linear1.weight.detach().float(), device=dev, dtype=self.dtype),
            "l1_b": _to_device(layer.linear1.bias.detach().float(), device=dev, dtype=self.dtype),
            "l2_w": _to_device(layer.linear2.weight.detach().float(), device=dev, dtype=self.dtype),
            "l2_b": _to_device(layer.linear2.bias.detach().float(), device=dev, dtype=self.dtype),
            "n1_w": _to_device(layer.norm1.weight.detach().float(), device=dev, dtype=self.dtype),
            "n1_b": _to_device(layer.norm1.bias.detach().float(), device=dev, dtype=self.dtype),
            "n2_w": _to_device(layer.norm2.weight.detach().float(), device=dev, dtype=self.dtype),
            "n2_b": _to_device(layer.norm2.bias.detach().float(), device=dev, dtype=self.dtype),
            "n3_w": _to_device(layer.norm3.weight.detach().float(), device=dev, dtype=self.dtype),
            "n3_b": _to_device(layer.norm3.bias.detach().float(), device=dev, dtype=self.dtype),
        }

    # ------------------------------------------------------------------ state
    def init_state(self, batch_size: int) -> LLVCState:
        cfg = self.config
        dev = self.device

        # Persistent ring buffers held in ROW_MAJOR/DRAM so their addresses stay
        # stable across chunks: streaming updates them in-place via ``ttnn.copy``
        # (see ``_carry``), which is what lets the whole ``forward_chunk`` be
        # captured once as a device trace and replayed per chunk.
        def zeros(t: int, c: int) -> ttnn.Tensor:
            return ttnn.zeros(
                (batch_size, t, c),
                dtype=self.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        convnet_ctx = [zeros(blk["buf_len"], 1) for blk in self.convnet_taps]
        enc_ctx = [zeros(layer["buf_len"], cfg.enc_dim) for layer in self.enc_layers]
        dec_mem_ctx = zeros(cfg.dec_buf_len, cfg.dec_dim)
        dec_tgt_ctx = [zeros(cfg.dec_buf_len, cfg.dec_dim) for _ in self.dec_layers]
        out_buf = zeros(cfg.out_buf_len, cfg.enc_dim)
        return LLVCState(convnet_ctx, enc_ctx, dec_mem_ctx, dec_tgt_ctx, out_buf)

    @staticmethod
    def _carry(buf: ttnn.Tensor, new_ctx: ttnn.Tensor) -> None:
        """Write the next chunk's context into the persistent ring buffer in place.

        Reassigning ``state.x = slice(...)`` would bind a fresh device tensor each
        chunk (new address), which breaks trace replay. Copying into the existing
        buffer keeps the address stable so ``forward_chunk`` can be traced once.
        The old context has already been consumed (concat'd) before this runs, so
        overwriting the buffer here is safe.
        """
        ttnn.copy(new_ctx, buf)

    # ------------------------------------------------------------------ blocks
    def _run_conv1d(self, cache_key, x, weight, bias, **kw):
        # Reuse the device-prepared weight *and* bias once cached: passing the
        # already-prepared tensors back means ttnn.conv1d does no host->device
        # upload, which is required for the call to be captured inside a trace.
        cached = self._conv_weight_cache.get(cache_key)
        if cached is not None:
            weight, bias = cached
        out, weight_dev, bias_dev = ops.conv1d(
            x,
            weight,
            bias,
            device=self.device,
            weights_dtype=self.weights_dtype,
            output_dtype=self.dtype,
            math_fidelity=self.math_fidelity,
            **kw,
        )
        self._conv_weight_cache[cache_key] = (weight_dev, bias_dev)
        return out

    def _cached_convnet(self, x, state: LLVCState):
        """12 gated residual blocks with per-block causal context; top-level 'add' skip."""
        residual_input = x
        for i, blk in enumerate(self.convnet_taps):
            # Filter and gate are two convs over the *same* causal window, so build
            # the window (concat + slice + tilize) once and share it across both.
            T = x.shape[1]
            x_ext_tile, ctx_new = ops.causal_window(x, state.convnet_ctx[i], buf_len=blk["buf_len"])
            filt = ops.apply_taps(x_ext_tile, T, blk["filter"], blk["filter_bias"], dilation=1)
            gate = ops.apply_taps(x_ext_tile, T, blk["gate"], blk["gate_bias"], dilation=1)
            self._carry(state.convnet_ctx[i], ctx_new)
            residual = ttnn.mul(ops.tanh(filt), ops.sigmoid(gate))
            x = ttnn.add(x, residual)  # ResidualBlock internal residual (crop == prepend)
        if self.config.convnet.skip_connection == "add":
            return ttnn.add(residual_input, x)
        if self.config.convnet.skip_connection == "multiply":
            return ttnn.mul(residual_input, x)
        return x

    def _encoder(self, x, state: LLVCState):
        cfg = self.config
        for i, layer in enumerate(self.enc_layers):
            ctx = state.enc_ctx[i]
            dcc_in = ops.concat_time(ctx, x)  # [B, buf_len + T, C]
            self._carry(state.enc_ctx[i], ops.slice_time(dcc_in, dcc_in.shape[1] - layer["buf_len"], dcc_in.shape[1]))

            # depthwise dilated conv (padding=0 consumes the prepended context).
            # Expressed as a shifted per-channel MAC: ttnn.conv1d cannot find a
            # valid shard/slice config for these depthwise layers.
            dw = ops.depthwise_causal_conv1d(
                dcc_in,
                layer["dw_taps"],
                layer["dw_bias"],
                dilation=layer["dilation"],
            )
            h = ops.layernorm_channels(dw, layer["ln1_w"], layer["ln1_b"])
            h = ops.relu(h)
            # pointwise 1x1 conv == linear over channels
            h = ops.linear(h, layer["pw_w"], layer["pw_b"], dtype=self.dtype)
            h = ops.layernorm_channels(h, layer["ln2_w"], layer["ln2_b"])
            h = ops.relu(h)
            x = ttnn.add(x, h)  # residual connection
        return x

    def _grouped_conv1x1(self, cache_key, x, weight, bias, in_ch, out_ch, groups):
        return self._run_conv1d(
            cache_key,
            x,
            weight,
            bias,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
        )

    def _mha(self, spec, query, key, value):
        cfg = self.config
        head_dim = cfg.dec_dim // cfg.nhead
        q = ops.linear(query, spec["qw"], spec["qb"], dtype=self.dtype)
        k = ops.linear(key, spec["kw"], spec["kb"], dtype=self.dtype)
        v = ops.linear(value, spec["vw"], spec["vb"], dtype=self.dtype)
        ctx = ops.scaled_dot_product_attention(q, k, v, n_heads=cfg.nhead, head_dim=head_dim)
        return ops.linear(ctx, spec["ow"], spec["ob"], dtype=self.dtype)

    def _build_windows(self, seq, num_windows):
        """Unfold ``[B, ctx_len + T, D]`` into ``[B * num_windows, ctx_len + chunk, D]``."""
        cfg = self.config
        win = cfg.dec_buf_len + cfg.dec_chunk_size
        stride = cfg.dec_chunk_size
        windows = []
        for i in range(num_windows):
            windows.append(ops.slice_time(seq, i * stride, i * stride + win))
        if len(windows) == 1:
            return windows[0]
        return ttnn.concat([ops.as_row_major(w) for w in windows], dim=0)

    def _decoder(self, m, e, state: LLVCState):
        cfg = self.config
        cs = cfg.dec_chunk_size
        B = m.shape[0]
        orig_T = m.shape[1]
        # Reference mod_pads the encoder-frame sequence up to a multiple of the
        # chunk size, then crops the output back (streaming T is always aligned).
        if orig_T % cs != 0:
            m, _ = ops.pad_time_to_multiple(m, cs)
            e, _ = ops.pad_time_to_multiple(e, cs)
        T = m.shape[1]
        num_windows = T // cs

        # memory (cross-attention keys/values)
        mem = ops.concat_time(state.dec_mem_ctx, e)
        self._carry(state.dec_mem_ctx, ops.slice_time(mem, mem.shape[1] - cfg.dec_buf_len, mem.shape[1]))
        mem_ctx = ops.as_tile(self._build_windows(mem, num_windows))
        if cfg.use_pos_enc:
            mem_ctx = ttnn.add(mem_ctx, self.dec_pos_enc)

        tgt = m
        for li, spec in enumerate(self.dec_layers):
            seq = ops.concat_time(state.dec_tgt_ctx[li], tgt)
            self._carry(state.dec_tgt_ctx[li], ops.slice_time(seq, seq.shape[1] - cfg.dec_buf_len, seq.shape[1]))
            tgt_ctx = ops.as_tile(self._build_windows(seq, num_windows))
            if cfg.use_pos_enc and li == 0:
                tgt_ctx = ttnn.add(tgt_ctx, self.dec_pos_enc)

            last = ops.slice_time(tgt_ctx, tgt_ctx.shape[1] - cs, tgt_ctx.shape[1])  # query = last chunk
            # self-attention
            sa = self._mha(spec["self_attn"], last, tgt_ctx, tgt_ctx)
            x = ops.layernorm_channels(ttnn.add(last, sa), spec["n1_w"], spec["n1_b"])
            # cross-attention
            ca = self._mha(spec["cross_attn"], x, mem_ctx, mem_ctx)
            x = ops.layernorm_channels(ttnn.add(x, ca), spec["n2_w"], spec["n2_b"])
            # feed-forward
            ff = ops.linear(x, spec["l1_w"], spec["l1_b"], dtype=self.dtype, activation="relu")
            ff = ops.linear(ff, spec["l2_w"], spec["l2_b"], dtype=self.dtype)
            x = ops.layernorm_channels(ttnn.add(x, ff), spec["n3_w"], spec["n3_b"])

            # [B*nw, cs, D] -> [B, T, D]
            tgt = ttnn.reshape(ops.as_row_major(x), (B, T, cfg.dec_dim))
        if T != orig_T:
            tgt = ops.slice_time(tgt, 0, orig_T)
        return tgt

    def _mask_gen(self, x, state: LLVCState):
        cfg = self.config
        e = self._encoder(x, state)  # [B, T, enc_dim]
        l = ttnn.mul(e, self.l_const)  # label integration (broadcast)

        if cfg.proj:
            e_proj = self._grouped_conv1x1(
                "proj_e2d_e", e, self.proj_e2d_e_w, self.proj_e2d_e_b, cfg.enc_dim, cfg.dec_dim, self.proj_e2d_groups
            )
            e_proj = ops.relu(e_proj)
            m = self._grouped_conv1x1(
                "proj_e2d_l", l, self.proj_e2d_l_w, self.proj_e2d_l_b, cfg.enc_dim, cfg.dec_dim, self.proj_e2d_groups
            )
            m = ops.relu(m)
            m = self._decoder(m, e_proj, state)
            m = self._grouped_conv1x1(
                "proj_d2e", m, self.proj_d2e_w, self.proj_d2e_b, cfg.dec_dim, cfg.enc_dim, self.proj_d2e_groups
            )
            m = ops.relu(m)
        else:
            m = self._decoder(l, e, state)

        if cfg.skip_connection:
            m = ttnn.add(l, m)
        return m

    # ------------------------------------------------------------------ forward
    def forward_chunk(self, x_btc: ttnn.Tensor, state: LLVCState) -> ttnn.Tensor:
        """Convert one (already padded) waveform chunk. ``x_btc`` is ``[B, T_samples, 1]``."""
        cfg = self.config

        if self.convnet_taps:
            x_btc = self._cached_convnet(x_btc, state)

        # in_conv (strided) downsamples to encoder frames, fused relu
        x = self._run_conv1d(
            "in_conv",
            x_btc,
            self.in_conv_w,
            None,
            in_channels=1,
            out_channels=cfg.enc_dim,
            kernel_size=cfg.kernel_size_in_conv,
            stride=cfg.L,
            padding=0,
            groups=1,
            activation="relu",
        )

        m = self._mask_gen(x, state)
        x = ttnn.mul(x, m)  # apply mask

        # prepend output buffer, update it, synthesise waveform
        x = ops.concat_time(state.out_buf, x)
        self._carry(state.out_buf, ops.slice_time(x, x.shape[1] - cfg.out_buf_len, x.shape[1]))
        wav, self.out_conv_w, _ = ops.conv_transpose1d(
            x,
            self._conv_weight_cache.get("out_conv", self.out_conv_w),
            None,
            in_channels=cfg.enc_dim,
            out_channels=1,
            kernel_size=(cfg.out_buf_len + 1) * cfg.L,
            stride=cfg.L,
            padding=cfg.out_buf_len * cfg.L,
            device=self.device,
            weights_dtype=self.weights_dtype,
            output_dtype=self.dtype,
            math_fidelity=self.math_fidelity,
        )
        self._conv_weight_cache["out_conv"] = self.out_conv_w
        return ops.tanh(wav)  # [B, T_out, 1]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Non-streaming conversion of ``[B, 1, T]`` (or ``[T]``) torch waveform -> ``[B, 1, T]``."""
        x, mod, original_len = self._pad_input(waveform)
        x_btc = _to_device(x.transpose(1, 2), device=self.device, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        state = self.init_state(x.shape[0])
        wav = self.forward_chunk(x_btc, state)
        out = ops.to_torch(wav).float().transpose(1, 2)  # [B, 1, T_out]
        if mod != 0:
            out = out[:, :, :-mod]
        return out[:, :, :original_len]

    def stream(self, waveform: torch.Tensor, *, chunk_factor: int = 1) -> tuple[torch.Tensor, float, float]:
        """Streaming conversion mirroring KoeAI ``infer_stream``.

        Splits ``waveform`` ([T] or [1, T]) into chunks of
        ``dec_chunk_size * L * chunk_factor`` samples, prepends a 2*L lookahead
        context from the previous chunk, and threads ``LLVCState`` across chunks.

        Returns ``(converted [1, 1, T], rtf, per_chunk_latency_ms)``.
        """
        cfg = self.config
        L = cfg.L
        if waveform.dim() == 2:
            waveform = waveform.reshape(-1)
        chunk_len = cfg.dec_chunk_size * L * chunk_factor
        original_len = waveform.shape[0]
        if original_len % chunk_len != 0:
            waveform = torch.nn.functional.pad(waveform, (0, chunk_len - (original_len % chunk_len)))

        # scoot down by L (matches reference lookahead alignment)
        waveform = torch.cat((waveform[L:], torch.zeros(L)))
        chunks = list(torch.split(waveform, chunk_len))
        prepped = []
        for i, c in enumerate(chunks):
            front = torch.zeros(L * 2) if i == 0 else chunks[i - 1][-L * 2 :]
            prepped.append(torch.cat([front, c]))

        state = self.init_state(1)
        # Trace needs at least one warmup + one capture chunk before it can replay.
        if cfg.use_trace and len(prepped) >= 3:
            outputs, times = self._stream_traced(prepped, state)
        else:
            outputs, times = self._stream_eager(prepped, state)

        out = torch.cat(outputs, dim=2)[:, :, :original_len]
        avg_time = float(sum(times) / max(1, len(times)))
        # Standard RTF = processing_time / audio_duration (lower is faster; the
        # bounty target is RTF < 0.3), so it drops as larger chunks amortise
        # the fixed per-chunk dispatch overhead.
        chunk_audio_s = chunk_len / cfg.sample_rate
        rtf = avg_time / max(chunk_audio_s, 1e-9)
        e2e_latency_ms = ((2 * L + chunk_len) / cfg.sample_rate + avg_time) * 1000.0
        return out, rtf, e2e_latency_ms

    def _host_chunk(self, chunk: torch.Tensor) -> ttnn.Tensor:
        """A single prepped chunk as a host ``[1, T, 1]`` ROW_MAJOR tensor."""
        x = chunk.reshape(1, 1, -1).transpose(1, 2)  # [1, T, 1]
        return ttnn.from_torch(x, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _stream_eager(self, prepped: list[torch.Tensor], state: LLVCState):
        """One host-dispatched ``forward_chunk`` per chunk (no trace)."""
        outputs, times = [], []
        for chunk in prepped:
            x_btc = _to_device(
                chunk.reshape(1, 1, -1).transpose(1, 2),
                device=self.device,
                dtype=self.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.synchronize_device(self.device)
            start = time.time()
            wav = self.forward_chunk(x_btc, state)
            ttnn.synchronize_device(self.device)
            times.append(time.time() - start)
            outputs.append(ops.to_torch(wav).float().transpose(1, 2))
        return outputs, times

    def _stream_traced(self, prepped: list[torch.Tensor], state: LLVCState):
        """Capture ``forward_chunk`` once, then replay it per chunk with no host dispatch.

        The streaming state is threaded naturally: chunk 0 runs eager (compiles the
        kernels and pins the conv weights on device), chunk 1 is captured (advancing
        the in-place ring buffers), and every later chunk copies its samples into the
        persistent input tensor and replays the trace. Because the ring buffers live
        at fixed addresses and are updated in place, replay reproduces the exact same
        recurrence as the eager path. Only the replayed chunks are timed (steady state).
        """
        dev = self.device
        seq_len = int(prepped[0].shape[0])
        in_dev = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, seq_len, 1]), self.dtype, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        outputs, times = [], []

        # chunk 0 — eager warmup (JIT + conv-weight prep so capture has no host->device moves)
        ttnn.copy_host_to_device_tensor(self._host_chunk(prepped[0]), in_dev)
        wav0 = self.forward_chunk(in_dev, state)
        outputs.append(ops.to_torch(wav0).float().transpose(1, 2))

        # chunk 1 — capture the trace (ring buffers advance in place as usual).
        # end_trace_capture must run even if forward_chunk raises, otherwise the
        # device is left mid-capture and every later synchronize/close fails.
        ttnn.copy_host_to_device_tensor(self._host_chunk(prepped[1]), in_dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            traced_out = self.forward_chunk(in_dev, state)
        finally:
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        outputs.append(ops.to_torch(traced_out).float().transpose(1, 2))

        # chunks 2..N — replay
        try:
            for chunk in prepped[2:]:
                ttnn.copy_host_to_device_tensor(self._host_chunk(chunk), in_dev)
                ttnn.synchronize_device(dev)
                start = time.time()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                ttnn.synchronize_device(dev)
                times.append(time.time() - start)
                outputs.append(ops.to_torch(traced_out).float().transpose(1, 2))
        finally:
            ttnn.release_trace(dev, tid)

        if not times:
            times = [0.0]
        return outputs, times

    def _pad_input(self, waveform: torch.Tensor):
        cfg = self.config
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        original_len = waveform.shape[-1]
        mod = 0
        if waveform.shape[-1] % cfg.L != 0:
            mod = cfg.L - (waveform.shape[-1] % cfg.L)
        waveform = torch.nn.functional.pad(waveform, (0, mod))
        if cfg.lookahead:
            waveform = torch.nn.functional.pad(waveform, (cfg.L, cfg.L))
        return waveform, mod, original_len


def create_llvc(
    config: LLVCConfig | None = None,
    *,
    device,
    checkpoint_path: str | None = None,
    reference: Net | None = None,
) -> LLVCModel:
    """Build a TTNN LLVC model.

    * ``reference`` — use an existing torch ``Net`` (e.g. with random weights for
      PCC tests).
    * ``checkpoint_path`` — load official LLVC weights into a fresh reference.
    * neither — random-initialised reference (shape/smoke tests only).
    """
    if config is None:
        config = LLVCConfig()
    if reference is None:
        reference = build_reference_model()
        if checkpoint_path is not None:
            from models.demos.llvc.tt.state_io import load_llvc_checkpoint

            load_llvc_checkpoint(reference, checkpoint_path)
    return LLVCModel(config, reference, device=device)
