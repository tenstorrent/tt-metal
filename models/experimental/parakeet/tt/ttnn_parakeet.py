# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""TTNN backend for nvidia/parakeet-tdt-0.6b-v3.

FastConformer encoder and TDT greedy decode on TT.
All dimensions come from the pinned config.json; nothing model-specific is hardcoded.
Mel preprocessing is host CPU policy (the harness supplies log-mel frames).

Precision: `precision` selects the matmul weight format ("bf16" or "fp32"). Matmul inputs and
elementwise constants default to fp32 in both modes (bf16 mode = bf16 weights, fp32 activations,
declared in precision_policy); act_dtype="bf16" selects all-bf16 activations for diagnosis.
Accumulation is fp32 everywhere; the encoder residual
stream, LayerNorm (params and I/O) and attention scores/softmax are fp32 in every mode.
The subsampling stack (conv2d, pointwise and output linear weights, ~2% of parameters) keeps fp32
weights in every mode: the pure-tone input amplifies its rounding error (tone encoder NRMSE .078 with
bf16 subsampling weights vs .031 with fp32, tests/diag_tone.py).
The prediction network, encoder/decoder projectors and joint head (~2% of parameters) also keep fp32
weights in every mode: TDT duration argmaxes can be near-ties (silence step 19: CPU FP32 margin .003,
tests/diag_silence.py), and bf16 decoder weights flipped one there.
Weight rounding (weight_round_bits, default FP32_WEIGHT_DROP_BITS; PARAKEET_WEIGHT_ROUND_BITS=0 disables):
the device truncates fp32 weight operands of matmuls to 9 mantissa bits (tests/diag_matmul_format.py),
a biased (toward zero) rounding. Every fp32 weight is therefore rounded to nearest-even at that width on
the host before upload, so the device keeps exactly the RNE value (bf16 uploads are already RNE).
This halved the subsampling output error on hard inputs (tests/diag_hard_speech.py) at zero runtime cost.
Targeted widening (fp32_layers, default none; PARAKEET_FP32_LAYERS="9,23" selects layers): the listed
encoder layers get fp32 (host-rounded) matmul weights in bf16 mode; activations/accumulation are fp32
already. Diagnostic option, off by default (see docs/PRECISION.md for the measured trade-off).
LayerNorm and softmax are composed from primitives (mean / centered variance, max / exp / sum):
the fused ttnn.layer_norm and ttnn.softmax kernels measured ~1.5e-3 NRMSE on fp32 inputs versus
<1e-7 for the compositions (tests/diag_ops.py).

Encoder layout conventions:
- Time is padded on the host with zero frames to a multiple of 8*32 mel frames, so the
  subsampled length Tp is tile aligned. Every mask is derived from the true lengths, and
  relative attention only depends on i-j, so frames < T' are unaffected by the padding.
- Padded frames < T' follow the reference exactly: subsampling masks, attention output zero for
  fully masked query rows (torch SDPA safe-softmax semantics), conv-module input zeroed.
- Subsampling convs run in NHWC (H=time, W=mel) with ttnn.conv2d; the 1x1 pointwise convs
  are ttnn.linear. The flatten to [B, T', C*F] is absorbed into a host permutation of
  the subsampling linear weight.
- The depthwise time conv (kernel 9) is a matmul with a stacked 0/1 shift matrix
  (exact copies with fp32 accumulation) followed by per-channel scale/add; BatchNorm
  (eval) is folded into those scales and a bias.

Encoder trace (use_trace, default on; PARAKEET_TRACE=0 or use_trace=False selects the untraced path):
the conformer blocks are fixed-shape per (batch, Tp), so the first call for a shape runs untraced
(compiling kernels and filling the per-Tp caches) and then captures a trace whose inputs (subsampling
output, key bias, time mask) are persistent device tensors; later calls with the same shape copy
their inputs in and replay it. The same ops run on the same values, so precision is unchanged.
Only one trace is live: buffers allocated after a capture can alias the trace's freed intermediates,
so a shape change releases the trace before any new per-Tp tensor is allocated. The subsampling
stack stays untraced (ttnn.conv2d prepares its host weights on every call).

TDT decode: 2-layer LSTM (one [x,h] matmul per layer), projector, frame gather (row lookup into
the projected encoder table), joint ReLU + head and both argmaxes run on device. The two argmax
results are read back together, so each step costs one readback. The host keeps per-row frame
pointers / finished flags (mirrors transformers ParakeetTDTGenerationMixin semantics).
fast_decode (default on; PARAKEET_FAST_DECODE=0 or fast_decode=False selects the previous path):
the token embedding lookup is an exact fp32 row slice of a device-resident table (no per-step host
upload), and the LSTM gate nonlinearities run as one sigmoid and one tanh over the whole
[1,B,4Hd] gate tensor before slicing (elementwise, so values are identical). Tokens were
bit-identical to the previous path on all bringup cases (long 1.66 -> 1.54 ms per step).
"""
import glob
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

SUPPORTED_PRECISIONS = ("bf16", "fp32")
DEFAULT_ACT_DTYPE = "fp32"
DEVICE_OPTIONS = {"l1_small_size": 32768, "trace_region_size": 96 * 1024 * 1024}
TIME_ALIGN = 32  # subsampled frames per tile row
TILE = 32
FP32_WEIGHT_DROP_BITS = 14  # fp32 weight operands keep 23 - 14 = 9 mantissa bits on device


@dataclass
class ParakeetConfig:
    hidden: int
    layers: int
    heads: int
    ffn: int
    conv_kernel: int
    mel_bins: int
    sub_channels: int
    sub_kernel: int
    sub_stride: int
    sub_factor: int
    scale_input: bool
    dec_hidden: int
    dec_layers: int
    vocab: int
    blank: int
    pad: int
    durations: list = field(default_factory=list)
    max_symbols: int = 10
    start: int = None
    eos: int = None
    ln_eps: float = 1e-5
    bn_eps: float = 1e-5

    @classmethod
    def from_dict(cls, cfg, gen=None):
        e = cfg["encoder_config"]
        gen = gen or {}
        return cls(hidden=e["hidden_size"], layers=e["num_hidden_layers"], heads=e["num_attention_heads"],
                   ffn=e["intermediate_size"], conv_kernel=e["conv_kernel_size"], mel_bins=e["num_mel_bins"],
                   sub_channels=e["subsampling_conv_channels"], sub_kernel=e["subsampling_conv_kernel_size"],
                   sub_stride=e["subsampling_conv_stride"], sub_factor=e["subsampling_factor"],
                   scale_input=bool(e.get("scale_input", False)), dec_hidden=cfg["decoder_hidden_size"],
                   dec_layers=cfg["num_decoder_layers"], vocab=cfg["vocab_size"], blank=cfg["blank_token_id"],
                   pad=gen.get("pad_token_id", cfg["pad_token_id"]), durations=list(cfg["durations"]),
                   max_symbols=cfg.get("max_symbols_per_step", 10),
                   start=gen.get("decoder_start_token_id", cfg["blank_token_id"]),
                   eos=gen.get("eos_token_id", cfg.get("eos_token_id")))

    @property
    def n_sub_convs(self):
        return int(round(math.log2(self.sub_factor)))

    @property
    def head_dim(self):
        return self.hidden // self.heads

    def sub_length(self, n):
        pad = (self.sub_kernel - 1) // 2
        for _ in range(self.n_sub_convs):
            n = (n + 2 * pad - self.sub_kernel) // self.sub_stride + 1
        return n


def _load_state_dict(weights_path):
    from safetensors.torch import load_file
    files = sorted(glob.glob(os.path.join(weights_path, "**", "*.safetensors"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no safetensors under {weights_path}")
    sd = {}
    for f in files:
        sd.update(load_file(f))
    return sd


def rel_positional_encoding(length, hidden):
    """Relative sinusoid table for positions length-1 .. -(length-1): [2*length-1, hidden] fp32."""
    import torch
    pos = torch.arange(length - 1, -length, -1, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hidden, 2, dtype=torch.float32) / hidden))
    ang = pos[:, None] * inv_freq[None, :]
    return torch.stack([ang.sin(), ang.cos()], dim=-1).reshape(pos.shape[0], hidden)


def rne_drop_bits(t, bits):
    """fp32 torch tensor rounded to nearest-even with the low `bits` mantissa bits cleared (bits <= 0: unchanged)."""
    import torch
    t = t.contiguous().float()
    if bits <= 0:
        return t
    b = t.view(torch.int32)
    half = (1 << (bits - 1)) - 1
    return ((b + half + ((b >> bits) & 1)) & ~((1 << bits) - 1)).view(torch.float32)


def _round_up(n, m):
    return -(-n // m) * m


def _parse_layers(s):
    return [int(v) for v in str(s).replace("+", ",").split(",") if v.strip()]


class Backend:
    def __init__(self, weights_path, config, device, precision, generation_config=None, act_dtype=None,
                 use_trace=None, fast_decode=None, weight_round_bits=None, fp32_layers=None):
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported precision {precision!r}; supported: {SUPPORTED_PRECISIONS}")
        act = act_dtype or DEFAULT_ACT_DTYPE
        if act not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported act_dtype {act!r}; supported: {SUPPORTED_PRECISIONS}")
        import ttnn
        self.ttnn = ttnn
        fmt = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}
        self.cfg = ParakeetConfig.from_dict(config, generation_config)
        self.device = device
        self.precision = precision
        self.wdtype = fmt[precision]  # matmul weights, linear biases
        self.sub_wdtype = ttnn.float32  # subsampling conv/pointwise/linear weights in every mode
        self.dec_wdtype = ttnn.float32  # LSTM, projectors and joint head weights in every mode
        self.adtype = fmt[act]  # matmul inputs and elementwise constants
        self.residual_dtype = ttnn.float32
        if weight_round_bits is None:
            weight_round_bits = int(os.environ.get("PARAKEET_WEIGHT_ROUND_BITS", FP32_WEIGHT_DROP_BITS))
        self.weight_round_bits = max(0, int(weight_round_bits))
        if fp32_layers is None:
            fp32_layers = _parse_layers(os.environ.get("PARAKEET_FP32_LAYERS", ""))
        elif isinstance(fp32_layers, str):
            fp32_layers = _parse_layers(fp32_layers)
        self.fp32_layers = sorted({int(i) for i in fp32_layers if 0 <= int(i) < self.cfg.layers})
        exceptions = ["subsampling conv2d/pointwise/linear weights fp32",
                      "prediction network, encoder/decoder projector and joint head weights fp32",
                      "encoder residual stream and LayerNorm params/I/O fp32",
                      "attention scores and softmax fp32",
                      "decoder LSTM cell/hidden state and joint logits fp32",
                      "relative positional table computed fp32 on host"]
        if self.fp32_layers and precision != "fp32":
            exceptions.append(f"encoder layers {','.join(map(str, self.fp32_layers))} matmul weights fp32")
        if self.weight_round_bits:
            exceptions.append(f"fp32 weights host-rounded to nearest-even, {23 - self.weight_round_bits} "
                              "mantissa bits (device operand width)")
        if act != precision:
            exceptions.insert(0, f"matmul activations {act}")
        self.precision_policy = {"mode": precision, "weights": precision, "activations": act,
                                 "accumulation": "fp32", "exceptions": exceptions}
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        if use_trace is None:
            use_trace = os.environ.get("PARAKEET_TRACE", "1") != "0"
        self.use_trace = bool(use_trace)
        if fast_decode is None:
            fast_decode = os.environ.get("PARAKEET_FAST_DECODE", "1") != "0"
        self.fast_decode = bool(fast_decode)
        self._trace = None  # live encoder trace: {"key", "id", "x", "key_bias", "tmask", "out"}
        self._pos_cache = {}
        self._shift_cache = {}
        sd = _load_state_dict(weights_path)
        self._prepare_weights(sd)
        self._prepare_decoder(sd)

    # ------------------------------------------------------------------ weights
    def _dev(self, t, dtype, layout=None):
        ttnn = self.ttnn
        return ttnn.from_torch(t.contiguous().float(), dtype=dtype, layout=layout or ttnn.TILE_LAYOUT,
                               device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _rw(self, t):
        """fp32 weight rounded to the device operand width (identity when weight_round_bits == 0)."""
        return rne_drop_bits(t, self.weight_round_bits)

    def _w(self, t, dtype=None):
        dtype = dtype or self.wdtype
        return self._dev(self._rw(t) if dtype == self.ttnn.float32 else t, dtype)

    def _sw(self, t):
        return self._dev(self._rw(t), self.sub_wdtype)

    def _dw(self, t):
        return self._dev(self._rw(t), self.dec_wdtype)

    def _a(self, t):
        return self._dev(t, self.adtype)

    def _f32(self, t):
        return self._dev(t, self.ttnn.float32)

    def _host(self, t):
        return self.ttnn.from_torch(self._rw(t), dtype=self.ttnn.float32)

    def _prepare_weights(self, sd):
        import torch
        c = self.cfg
        H, hd, D = c.heads, c.head_dim, c.hidden
        pre = "encoder.subsampling."
        convs = sorted({int(k.split(".")[3]) for k in sd if k.startswith(pre + "layers.")})
        # conv0 (Cin=1) is zero-padded to CIN0 input channels; the input is padded to match.
        self.cin0 = 16
        w0 = sd[pre + f"layers.{convs[0]}.weight"]
        w0p = torch.zeros(w0.shape[0], self.cin0, *w0.shape[2:])
        w0p[:, : w0.shape[1]] = w0
        self.sub_conv0 = (self._host(w0p), self._host(sd[pre + f"layers.{convs[0]}.bias"].reshape(1, 1, 1, -1)))
        self.sub_stages = []
        for dw, pw in zip(convs[1::2], convs[2::2]):
            wdw = self._host(sd[pre + f"layers.{dw}.weight"])
            bdw = self._host(sd[pre + f"layers.{dw}.bias"].reshape(1, 1, 1, -1))
            wpw = self._sw(sd[pre + f"layers.{pw}.weight"][:, :, 0, 0].t())
            bpw = self._sw(sd[pre + f"layers.{pw}.bias"].reshape(1, -1))
            self.sub_stages.append((wdw, bdw, wpw, bpw))
        lw = sd[pre + "linear.weight"]  # [D, C*F], columns c*F+f -> reorder to f*C+c (NHWC flatten)
        C = c.sub_channels
        F = lw.shape[1] // C
        self.sub_lin_w = self._sw(lw.reshape(D, C, F).permute(0, 2, 1).reshape(D, F * C).t())
        self.sub_lin_b = self._sw(sd[pre + "linear.bias"].reshape(1, -1))

        scale = hd ** -0.5
        self.blocks = []
        for i in range(c.layers):
            p = f"encoder.layers.{i}."
            g = lambda n: sd[p + n]
            ln = lambda n: (self._f32(g(n + ".weight").reshape(1, -1)), self._f32(g(n + ".bias").reshape(1, -1)))
            # widened layers (fp32_layers) take fp32 matmul weights; the rest use the mode's weight dtype
            w = lambda t, d=self.ttnn.float32 if i in self.fp32_layers else None: self._w(t, d)
            # softmax scale folded into q (and its biases) in fp32 before the device cast
            wq = g("self_attn.q_proj.weight") * scale
            wqkv = torch.cat([wq, g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")], 0)
            bn_s = g("conv.norm.weight") / torch.sqrt(g("conv.norm.running_var") + c.bn_eps)
            bn_b = g("conv.norm.bias") - g("conv.norm.running_mean") * bn_s
            dw = g("conv.depthwise_conv.weight")[:, 0, :] * bn_s[:, None]  # [D, K]
            self.blocks.append({
                "ln_ff1": ln("norm_feed_forward1"),
                "ff1_w1": w(g("feed_forward1.linear1.weight").t()),
                "ff1_w2": w(0.5 * g("feed_forward1.linear2.weight").t()),
                "ln_att": ln("norm_self_att"),
                "wqkv": w(wqkv.t()),
                "bias_u": self._a((g("self_attn.bias_u") * scale).reshape(1, H, 1, hd)),
                "bias_v": self._a((g("self_attn.bias_v") * scale).reshape(1, H, 1, hd)),
                "wpos": w(g("self_attn.relative_k_proj.weight").t()),
                "wo": w(g("self_attn.o_proj.weight").t()),
                "ln_conv": ln("norm_conv"),
                "pw1": w(g("conv.pointwise_conv1.weight")[:, :, 0].t()),
                "dw": [self._a(dw[:, k].reshape(1, 1, -1)) for k in range(c.conv_kernel)],
                "dw_b": self._a(bn_b.reshape(1, 1, -1)),
                "pw2": w(g("conv.pointwise_conv2.weight")[:, :, 0].t()),
                "ln_ff2": ln("norm_feed_forward2"),
                "ff2_w1": w(g("feed_forward2.linear1.weight").t()),
                "ff2_w2": w(0.5 * g("feed_forward2.linear2.weight").t()),
                "ln_out": ln("norm_out"),
            })

    def _prepare_decoder(self, sd):
        import torch
        ttnn, c = self.ttnn, self.cfg
        Hd = c.dec_hidden
        # host fp32 copy: the per-step row lookup is an exact host gather + one upload (fast_decode off)
        self.dec_emb = sd["decoder.embedding.weight"].float().contiguous()
        # device fp32 row-major copy: exact per-step row slices (fast_decode on)
        self.dec_emb_dev = self._dev(self.dec_emb, ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT) \
            if self.fast_decode else None
        self.lstm = []
        for l in range(c.dec_layers):
            w = torch.cat([sd[f"decoder.lstm.weight_ih_l{l}"], sd[f"decoder.lstm.weight_hh_l{l}"]], 1)
            b = sd[f"decoder.lstm.bias_ih_l{l}"] + sd[f"decoder.lstm.bias_hh_l{l}"]
            self.lstm.append((self._dw(w.t()), self._dw(b.reshape(1, -1))))
        self.dec_proj = (self._dw(sd["decoder.decoder_projector.weight"].t()),
                         self._dw(sd["decoder.decoder_projector.bias"].reshape(1, -1)))
        self.enc_proj = (self._dw(sd["encoder_projector.weight"].t()),
                         self._dw(sd["encoder_projector.bias"].reshape(1, -1)))
        # Joint head laid out as [tokens | pad | durations | pad] with tile-aligned regions;
        # padded columns get a very negative bias so they never win an argmax.
        V, ND = c.vocab, len(c.durations)
        self.vpad, dpad = _round_up(V, TILE), _round_up(ND, TILE)
        hw, hb = sd["joint.head.weight"], sd["joint.head.bias"]
        W = torch.zeros(Hd, self.vpad + dpad)
        bias = torch.full((self.vpad + dpad,), -1e30)
        W[:, :V], bias[:V] = hw[:V].t(), hb[:V]
        W[:, self.vpad:self.vpad + ND], bias[self.vpad:self.vpad + ND] = hw[V:].t(), hb[V:]
        self.head = (self._dw(W), self._dw(bias.reshape(1, -1)))

    # ------------------------------------------------------------------ cached per-length tensors
    def _pos(self, tp):
        """Relative table with a leading zero row: [1, 2*tp, D]."""
        if tp not in self._pos_cache:
            import torch
            pe = rel_positional_encoding(tp, self.cfg.hidden)
            pe = torch.cat([torch.zeros(1, pe.shape[1]), pe], 0)
            self._pos_cache[tp] = self._a(pe.unsqueeze(0))
        return self._pos_cache[tp]

    def _shift(self, tp, batch):
        """Stacked shift matrices S[k*tp + t, s] = 1 iff s == t + k - K//2: [B, K*tp, tp]."""
        key = (tp, batch)
        if key not in self._shift_cache:
            import torch
            K = self.cfg.conv_kernel
            S = torch.zeros(K, tp, tp)
            t = torch.arange(tp)
            for k in range(K):
                s = t + k - K // 2
                ok = (s >= 0) & (s < tp)
                S[k, t[ok], s[ok]] = 1.0
            self._shift_cache[key] = self._a(S.reshape(1, K * tp, tp).expand(batch, -1, -1))
        return self._shift_cache[key]

    def _mask_values(self, lengths, tp):
        """Host torch (key_bias [B,1,1,tp] additive, tmask [B,tp,1] 0/1)."""
        import torch
        B = len(lengths)
        valid = (np.arange(tp)[None, :] < np.asarray(lengths)[:, None])
        key_bias = torch.from_numpy(np.where(valid, 0.0, -1e9).astype(np.float32)).reshape(B, 1, 1, tp)
        tmask = torch.from_numpy(valid.astype(np.float32)).reshape(B, tp, 1)
        return key_bias, tmask

    def masks(self, lengths, tp):
        """(key_bias [B,1,1,tp] fp32 additive, tmask [B,tp,1] 0/1 in the activation dtype)."""
        key_bias, tmask = self._mask_values(lengths, tp)
        return self._f32(key_bias), self._a(tmask)

    # ------------------------------------------------------------------ encoder pieces
    def _linear(self, x, w, bias=None, activation=None, dtype=None):
        return self.ttnn.linear(x, w, bias=bias, activation=activation, compute_kernel_config=self.compute_cfg,
                                memory_config=self.ttnn.DRAM_MEMORY_CONFIG, dtype=dtype)

    def _ln(self, x, wb):
        """LayerNorm over the last dim with centered (two-pass) variance; x and params fp32."""
        ttnn = self.ttnn
        xc = ttnn.subtract(x, ttnn.mean(x, dim=-1, keepdim=True))
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True)
        y = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, self.cfg.ln_eps)))
        return ttnn.add(ttnn.multiply(y, wb[0]), wb[1])

    def _softmax(self, s):
        """Numerically stable softmax over the last dim (fp32 in/out)."""
        ttnn = self.ttnn
        e = ttnn.exp(ttnn.subtract(s, ttnn.max(s, dim=-1, keepdim=True)))
        return ttnn.multiply(e, ttnn.reciprocal(ttnn.sum(e, dim=-1, keepdim=True)))

    def _cast(self, x, dtype):
        return x if x.dtype == dtype else self.ttnn.typecast(x, dtype)

    def _conv2d(self, x, w, b, batch, h, wdt, cin, cout, groups):
        ttnn = self.ttnn
        k = self.cfg.sub_kernel
        conv_cfg = ttnn.Conv2dConfig(weights_dtype=self.sub_wdtype, output_layout=ttnn.TILE_LAYOUT)
        out, (ho, wo) = ttnn.conv2d(
            input_tensor=x, weight_tensor=w, bias_tensor=b, device=self.device, in_channels=cin,
            out_channels=cout, batch_size=batch, input_height=h, input_width=wdt, kernel_size=(k, k),
            stride=(self.cfg.sub_stride, self.cfg.sub_stride), padding=((k - 1) // 2, (k - 1) // 2),
            groups=groups, conv_config=conv_cfg, compute_config=self.compute_cfg, dtype=self.adtype,
            return_output_dim=True, return_weights_and_bias=False)
        return out, ho, wo

    def _time_mask(self, lengths, t, width):
        """[B, t*width, 1] 0/1 mask (row index = time*width + w)."""
        import torch
        m = (np.arange(t)[None, :] < np.asarray(lengths)[:, None]).astype(np.float32)
        return self._a(torch.from_numpy(np.repeat(m, width, axis=1)).unsqueeze(-1))

    def _subsample(self, mel_pad, lengths):
        ttnn, c = self.ttnn, self.cfg
        import torch
        B, T, F = mel_pad.shape
        C = c.sub_channels
        x = torch.zeros(B, T, F, self.cin0)
        x[..., 0] = torch.from_numpy(mel_pad)
        x = ttnn.from_torch(x, dtype=self.adtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        lens = np.asarray(lengths, dtype=np.int64)
        pad = (c.sub_kernel - 1) // 2
        nxt = lambda n: (n + 2 * pad - c.sub_kernel) // c.sub_stride + 1
        x, h, w = self._conv2d(x, *self.sub_conv0, B, T, F, self.cin0, C, 1)
        lens = nxt(lens)
        x = ttnn.reshape(x, (B, h * w, C))
        x = ttnn.relu(ttnn.multiply(x, self._time_mask(lens, h, w)))
        for wdw, bdw, wpw, bpw in self.sub_stages:
            x = ttnn.reshape(x, (1, 1, B * h * w, C))
            x, h, w = self._conv2d(x, wdw, bdw, B, h, w, C, C, C)
            lens = nxt(lens)
            x = ttnn.reshape(x, (B, h * w, C))
            x = self._linear(x, wpw, bpw, activation="relu")
            x = ttnn.multiply(x, self._time_mask(lens, h, w))
        x = ttnn.reshape(x, (B, h, w * C))
        return self._linear(x, self.sub_lin_w, self.sub_lin_b), lens

    def _rel_shift(self, bd, B, tp):
        """bd [B,H,tp,2tp] (col 0 = zero pad) -> out[i,j] = bd_ref[i, j + tp-1 - i], [B,H,tp,tp]."""
        ttnn = self.ttnn
        H = self.cfg.heads
        x = ttnn.to_layout(bd, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, H, 2 * tp, tp))
        x = ttnn.slice(x, (0, 0, 1, 0), (B, H, 2 * tp, tp))
        x = ttnn.reshape(x, (B, H, tp, 2 * tp - 1))
        x = ttnn.slice(x, (0, 0, 0, 0), (B, H, tp, tp))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _attention(self, x, blk, B, tp, key_bias, tmask):
        ttnn, c = self.ttnn, self.cfg
        H, hd = c.heads, c.head_dim
        f32 = ttnn.float32
        qkv = self._linear(x, blk["wqkv"])
        q, kT, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=H, transpose_key=True)
        p = self._linear(self._pos(tp), blk["wpos"])  # [1, 2tp, D]
        p = ttnn.permute(ttnn.reshape(p, (1, 2 * tp, H, hd)), (0, 2, 3, 1))  # [1,H,hd,2tp]
        if B > 1:
            p = ttnn.repeat(p, (B, 1, 1, 1))
        ac = ttnn.matmul(ttnn.add(q, blk["bias_u"]), kT, compute_kernel_config=self.compute_cfg, dtype=f32)
        bd = ttnn.matmul(ttnn.add(q, blk["bias_v"]), p, compute_kernel_config=self.compute_cfg, dtype=f32)
        s = ttnn.add(ttnn.add(ac, self._rel_shift(bd, B, tp)), self._cast(key_bias, f32))
        s = self._cast(self._softmax(s), self.adtype)
        ctx = ttnn.matmul(s, v, compute_kernel_config=self.compute_cfg)
        # fully masked query rows produce zero (o_proj has no bias), as in the reference
        return ttnn.multiply(self._linear(ttnn.transformer.concatenate_heads(ctx), blk["wo"]), tmask)

    def _conv_module(self, x, blk, B, tp, tmask):
        ttnn, c = self.ttnn, self.cfg
        D = c.hidden
        g = self._linear(x, blk["pw1"])
        a = ttnn.slice(g, (0, 0, 0), (B, tp, D))
        b = ttnn.slice(g, (0, 0, D), (B, tp, 2 * D))
        g = ttnn.multiply(ttnn.multiply(a, ttnn.sigmoid(b)), tmask)
        xs = ttnn.matmul(self._shift(tp, B), g, compute_kernel_config=self.compute_cfg)  # [B, K*tp, D]
        acc = blk["dw_b"]
        for k in range(c.conv_kernel):
            sk = ttnn.slice(xs, (0, k * tp, 0), (B, (k + 1) * tp, D))
            acc = ttnn.add(acc, ttnn.multiply(sk, blk["dw"][k]))
        return self._linear(ttnn.silu(acc), blk["pw2"])

    def _ffn(self, h, w1, w2):
        return self._linear(self._linear(h, w1, activation="silu"), w2)

    def _block(self, x, blk, B, tp, key_bias, tmask):
        ttnn, rd = self.ttnn, self.residual_dtype
        ln = lambda t, wb: self._cast(self._ln(t, wb), self.adtype)
        res = lambda t, y: ttnn.add(t, self._cast(y, rd))
        x = res(x, self._ffn(ln(x, blk["ln_ff1"]), blk["ff1_w1"], blk["ff1_w2"]))
        x = res(x, self._attention(ln(x, blk["ln_att"]), blk, B, tp, key_bias, tmask))
        x = res(x, self._conv_module(ln(x, blk["ln_conv"]), blk, B, tp, tmask))
        x = res(x, self._ffn(ln(x, blk["ln_ff2"]), blk["ff2_w1"], blk["ff2_w2"]))
        return self._cast(self._ln(x, blk["ln_out"]), rd)

    def _blocks(self, x, B, tp, key_bias, tmask):
        for blk in self.blocks:
            x = self._block(x, blk, B, tp, key_bias, tmask)
        return x

    def release_trace(self):
        if self._trace is not None:
            self.ttnn.release_trace(self.device, self._trace["id"])
            self._trace = None

    def _blocks_traced(self, x, lens, B, tp):
        """Conformer blocks via the live trace for (B, tp); captures one after an untraced run."""
        ttnn = self.ttnn
        key = (B, tp)
        tr = self._trace
        if tr is not None and tr["key"] == key:
            kb, tm = self._mask_values(lens, tp)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(kb, dtype=tr["key_bias"].dtype, layout=ttnn.TILE_LAYOUT), tr["key_bias"])
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(tm, dtype=tr["tmask"].dtype, layout=ttnn.TILE_LAYOUT), tr["tmask"])
            ttnn.copy(x, tr["x"])
            ttnn.deallocate(x)
            ttnn.execute_trace(self.device, tr["id"], cq_id=0, blocking=False)
            return tr["out"]
        self.release_trace()  # before any new per-(B, tp) allocation
        key_bias, tmask = self.masks(lens, tp)
        out = self._blocks(x, B, tp, key_bias, tmask)  # compiles kernels, fills caches
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            tr_out = self._blocks(x, B, tp, key_bias, tmask)
        finally:
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
        self._trace = {"key": key, "id": tid, "x": x, "key_bias": key_bias, "tmask": tmask, "out": tr_out}
        return out

    def encode_device(self, mel, mel_lengths, taps=None):
        """Run the encoder; returns (device tensor [B, Tp, D], valid sub lengths, T' of unpadded input).

        taps: optional dict filled with host float32 copies {"subsampling", "layer{i}"} for checks
        (always untraced). With use_trace the returned tensor is the trace output, valid until the
        next encoder call.
        """
        ttnn, c = self.ttnn, self.cfg
        mel = np.asarray(mel, dtype=np.float32)
        B, T, _ = mel.shape
        t_out = c.sub_length(T)
        tp = _round_up(t_out, TIME_ALIGN)
        t_pad = tp * c.sub_factor
        assert c.sub_length(t_pad) == tp
        mel_pad = np.zeros((B, t_pad, mel.shape[2]), np.float32)
        mel_pad[:, :T] = mel
        if taps is not None or not self.use_trace:
            self.release_trace()
        x, lens = self._subsample(mel_pad, mel_lengths)
        x = self._cast(x, self.residual_dtype)
        if taps is None and self.use_trace:
            return self._blocks_traced(x, lens, B, tp), lens, t_out
        if taps is not None:
            taps["subsampling"] = ttnn.to_torch(x).float()[:, :t_out]
        key_bias, tmask = self.masks(lens, tp)
        for i, blk in enumerate(self.blocks):
            x = self._block(x, blk, B, tp, key_bias, tmask)
            if taps is not None:
                taps[f"layer{i}"] = ttnn.to_torch(x).float()[:, :t_out]
        return x, lens, t_out

    def encode(self, mel, mel_lengths):
        x, _, t_out = self.encode_device(mel, mel_lengths)
        out = self.ttnn.to_torch(x).float().numpy()[:, :t_out]
        return {"encoder": np.ascontiguousarray(out, dtype=np.float32)}

    # ------------------------------------------------------------------ TDT decode
    def _ids(self, vals):
        import torch
        return self.ttnn.from_torch(torch.tensor(np.asarray(vals, dtype=np.int32)[None, :]), dtype=self.ttnn.uint32,
                                    layout=self.ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def _gather(self, table, rows):
        """Rows of a row-major [N, W] device table -> [1, len(rows), W] tile tensor."""
        ttnn = self.ttnn
        if table.dtype == ttnn.bfloat16:
            return ttnn.embedding(self._ids(rows), table, layout=ttnn.TILE_LAYOUT)
        # ttnn.embedding is bf16-only; fp32 tables use exact row slices
        width = table.shape[-1]
        parts = [ttnn.slice(table, (int(r), 0), (int(r) + 1, width)) for r in rows]
        x = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=0)
        return ttnn.to_layout(ttnn.reshape(x, (1, len(rows), width)), ttnn.TILE_LAYOUT)

    def _decoder_step(self, tokens, h, c):
        """One prediction-network step for all rows. Returns (dec_out [1,B,Hd], h list, c list)."""
        ttnn = self.ttnn
        Hd = self.cfg.dec_hidden
        B = len(tokens)
        if self.fast_decode:
            x = self._cast(self._gather(self.dec_emb_dev, tokens), self.adtype)  # [1,B,Hd]
        else:
            x = self._dev(self.dec_emb[np.asarray(tokens, dtype=np.int64)].unsqueeze(0), self.adtype)
        hn, cn = [], []
        for l, (w, b) in enumerate(self.lstm):
            xh = ttnn.concat([x, self._cast(h[l], self.adtype)], dim=-1)
            gates = self._linear(xh, w, b, dtype=ttnn.float32)  # [1,B,4Hd], torch order i,f,g,o
            sl = lambda t, k: ttnn.slice(t, (0, 0, k * Hd), (1, B, (k + 1) * Hd))
            if self.fast_decode:
                sg, tg = ttnn.sigmoid(gates), ttnn.tanh(gates)
                i_g, f_g, g_g, o_g = sl(sg, 0), sl(sg, 1), sl(tg, 2), sl(sg, 3)
            else:
                i_g, f_g = ttnn.sigmoid(sl(gates, 0)), ttnn.sigmoid(sl(gates, 1))
                g_g, o_g = ttnn.tanh(sl(gates, 2)), ttnn.sigmoid(sl(gates, 3))
            c_new = ttnn.add(ttnn.multiply(f_g, c[l]), ttnn.multiply(i_g, g_g))
            h_new = ttnn.multiply(o_g, ttnn.tanh(c_new))
            hn.append(h_new)
            cn.append(c_new)
            x = self._cast(h_new, self.adtype)
        return self._linear(x, *self.dec_proj, dtype=self.adtype), hn, cn

    def _joint(self, table, frame_rows, dec_out):
        """Returns host (token ids, duration indices) for each row, from a single readback."""
        ttnn = self.ttnn
        B = len(frame_rows)
        e = self._gather(table, frame_rows)  # [1,B,Hd]
        z = ttnn.relu(ttnn.add(e, dec_out))
        logits = self._linear(z, *self.head, dtype=ttnn.float32)  # [1,B,vpad+dpad]
        logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        tok = ttnn.argmax(ttnn.slice(logits, (0, 0, 0), (1, B, self.vpad)), dim=-1)
        dur = ttnn.argmax(ttnn.slice(logits, (0, 0, self.vpad), (1, B, logits.shape[-1])), dim=-1)
        both = ttnn.to_torch(ttnn.concat([tok, dur], dim=0)).reshape(2, -1).numpy().astype(np.int64)
        return both[0, :B], both[1, :B]

    def _masked(self, mask_np, new, old):
        import torch
        m = self._dev(torch.from_numpy(np.repeat(mask_np.astype(np.float32)[None, :, None], new.shape[-1], 2)),
                      new.dtype)
        return self.ttnn.where(m, new, old)

    def transcribe(self, mel, mel_lengths):
        """Greedy TDT decode. Returns {"tokens": int64 [B, L]} (start token + one symbol per step,
        finished rows padded with pad_token_id), matching transformers generate().sequences."""
        import torch
        ttnn, c = self.ttnn, self.cfg
        x, lens, t_out = self.encode_device(mel, mel_lengths)
        B, tp = x.shape[0], x.shape[1]
        Hd = c.dec_hidden
        ep = self._linear(self._cast(x, self.adtype), *self.enc_proj, dtype=self.adtype)  # [B,tp,Hd]
        table = ttnn.reshape(ttnn.to_layout(ep, ttnn.ROW_MAJOR_LAYOUT), (B * tp, Hd))
        zeros = lambda: self._dev(torch.zeros(1, B, Hd), ttnn.float32)
        h = [zeros() for _ in range(c.dec_layers)]
        cs = [zeros() for _ in range(c.dec_layers)]
        durations = np.asarray(c.durations, dtype=np.int64)
        valid = np.asarray(lens, dtype=np.int64)
        last = np.full(B, c.start, dtype=np.int64)
        seqs = [last.copy()]
        frames = np.zeros(B, dtype=np.int64)
        finished = np.zeros(B, dtype=bool)
        max_len = c.max_symbols * t_out
        dec_out = None
        while True:
            if dec_out is None:
                dec_out, h, cs = self._decoder_step(last, h, cs)
            else:
                upd = last != c.blank
                if upd.all():
                    dec_out, h, cs = self._decoder_step(last, h, cs)
                elif upd.any():
                    d_n, h_n, c_n = self._decoder_step(last, h, cs)
                    dec_out = self._masked(upd, d_n, dec_out)
                    h = [self._masked(upd, a, b) for a, b in zip(h_n, h)]
                    cs = [self._masked(upd, a, b) for a, b in zip(c_n, cs)]
            rows = np.arange(B) * tp + np.minimum(frames, t_out - 1)
            tok, di = self._joint(table, rows, dec_out)
            dur = durations[di]
            dur = np.where((tok == c.blank) & (dur == 0), 1, dur)
            frames = frames + dur
            nt = np.where(finished, c.pad, tok)
            seqs.append(nt)
            finished |= (frames >= valid)
            if c.eos is not None:
                finished |= nt == c.eos
            last = nt
            if finished.all() or len(seqs) >= max_len:
                break
        return {"tokens": np.stack(seqs, axis=1).astype(np.int64)}


def create_backend(weights_path, config, device, *, precision="bf16", **options):
    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)
    gen = None
    gen_path = os.path.join(weights_path, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            gen = json.load(f)
    return Backend(weights_path, config, device, precision, generation_config=gen, **options)
