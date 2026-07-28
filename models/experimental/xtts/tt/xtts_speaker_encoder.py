# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 speaker encoder (``ResNetSpeakerEncoder``): log-mel -> 512-d ``g``.

Mirrors ``reference/xtts_speaker_encoder.py``. Everything below is shaped by one fact about
this network: the SE-ResNet-34's 16 blocks move a lot of activation for very little math, so
the convolutions were never the cost — layout, per-channel eltwise and pooling were. Four
choices follow from that, each measured (device time for one mel_len=801 pass, ~8 s of
reference audio, via ``tests/test_speaker_encoder_profile.py``):

* **The body never leaves the flat channels-last TILE form** ``[1, 1, H*W, C]`` (H=freq,
  W=time) in **L1** — exactly what ``ttnn.conv2d`` emits and consumes. Untilizing to a
  ``[N, H, W, C]`` ROW_MAJOR view per conv cost ~200us each, the 4D TILE form pads W to a
  tile *per freq row* (inflating every eltwise op that follows), and a DRAM activation puts
  ttnn.conv2d on its DRAM path, which brackets each conv with a 4D unflatten + re-flatten.
  The spatial extent travels alongside as Python ints, so it stays static/trace-safe.
* **BatchNorm never runs as a BatchNorm.** It is a per-channel affine at inference
  (``scale = gamma/sqrt(var+eps)``, ``shift = beta - mean*scale``); where a conv feeds BN
  directly (``conv2 -> bn2``, ``downsample``) it folds into that conv's weight and bias and
  costs nothing. Only ``bn1`` cannot fold — coqui's block order puts a relu between conv1
  and bn1 — so the relu rides on conv1 as a fused activation and bn1 becomes a diagonal
  matmul (:func:`_scale_channels`), as does the SE's channel scaling.
* **A stage stops being sharded once its shard is mostly tile padding**
  (:func:`_stage_memory_config`), which is what the narrowest stage was paying for.
* **bfloat16 body, float32 ASP tail** — the body is bandwidth-bound so the narrow dtype is
  nearly free, but the tail's ``E[x^2] - mu^2`` would lose too much to bf16 cancellation.
  Math fidelity stays HiFi4: see ``BODY_FIDELITY``.

Together: 26.9 ms -> 2.9 ms per pass, at PCC 0.9990 against the torch reference (the
fp32 implementation this replaces scored 0.9994). Only the ASP reshape needs a permute, into
a ``[C=2048, T']`` column layout where the attention softmax and the ASP reductions are over
the last dim.

Weights are read from the folded/eval reference module.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.tt.xtts_conv import TtConv2d

TILE = 32
BN_EPS = 1e-5
INSTANCENORM_EPS = 1e-5
ASP_EPS = 1e-5

BODY_DTYPE = ttnn.bfloat16
# HiFi4, not HiFi2: dropping fidelity costs far more accuracy than it buys time here (the
# body is bandwidth-bound, not math-bound). Measured end-to-end PCC at mel_len=200 —
# convs/affines both HiFi2: 0.967, only one of them HiFi4: 0.986-0.991, both HiFi4: 0.998.
BODY_FIDELITY = ttnn.MathFidelity.HiFi4
RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
SIGMOID = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)


def _to_tile(t: torch.Tensor, device, dtype=ttnn.float32) -> ttnn.Tensor:
    return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)


def _to_body(t: torch.Tensor, device) -> ttnn.Tensor:
    """A body-side constant: bfloat16, and left in DRAM.

    These are all matmul operands (see :func:`_scale_channels`), so each core reads one
    once and DRAM residency costs nothing measurable. Keeping them in L1 *does* cost:
    they are resident for the model's lifetime, and the ~1.5 MB they take is enough to
    push a later op's circular buffers out of L1 in the full-model trace."""
    return _to_tile(t, device, BODY_DTYPE)


def _bn_scale_shift(bn, eps=BN_EPS):
    """Fold a BatchNorm into an inference-time per-channel affine (scale, shift)."""
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + eps)
    shift = bn.bias.detach() - bn.running_mean.detach() * scale
    return scale, shift


def _body_compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=BODY_FIDELITY, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _scale_channels(x, diag, compute_config, core_grid, bias=None):
    """Per-channel scale (and optional shift) of a height-sharded activation, as a
    **diagonal matmul** ``x @ diag(scale) [+ shift]``.

    The obvious spellings — ``ttnn.addcmul``, or ``mul``/``add`` against a ``[1, 1, 1, C]``
    operand — broadcast that one operand tile over the sharded height, and each core
    re-reads it per output tile: 153us, or 43us per op, on the first stage, against 1.5us
    for the same-shape (non-broadcast) residual add. As a matmul the ``[C, C]`` weight is
    read once per core instead — 6.6us including the bias — and it is bit-identical
    (PCC 1.0), the off-diagonal terms being exact zeros.

    ``core_grid`` is what gets the bias *fused*: ttnn.linear post-processes a bias as a
    separate (43us broadcast) add whenever it has to guess a program config for a non-DRAM
    output, and passing a core grid is what makes it stop guessing. The output reuses the
    input's shard spec, so the next conv still sees an L1 input."""
    return ttnn.linear(
        x,
        diag,
        bias=bias,
        compute_kernel_config=compute_config,
        memory_config=x.memory_config(),
        core_grid=core_grid,
    )


class _ChannelAffine(LightweightModule):
    """``x*scale + shift`` for a folded BatchNorm, as a diagonal matmul (see
    :func:`_scale_channels`)."""

    def __init__(self, device, scale, shift):
        super().__init__()
        self.diag = _to_body(torch.diag(scale), device)  # [C, C]
        self.bias = _to_body(shift.reshape(1, -1), device)  # [1, C]
        self.compute_config = _body_compute_config(device)
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

    def forward(self, x):
        return _scale_channels(x, self.diag, self.compute_config, self.core_grid, bias=self.bias)


def _fold_bn(weight, bias, bn):
    """Fold a BatchNorm that *directly* follows a conv into the conv's weight and bias.
    ``bn(conv(x)) == conv_folded(x)`` exactly: the affine is per-output-channel, and the
    conv's own bias is affine in the same channel, so both collapse into the conv."""
    scale, shift = _bn_scale_shift(bn)
    folded_bias = shift if bias is None else bias * scale + shift
    return weight * scale.reshape(-1, 1, 1, 1), folded_bias


def _stage_memory_config(ncores, hw):
    """Where a stage's activations live: L1-sharded (``None``, i.e. as the conv produces it)
    or L1-interleaved.

    Sharded is what keeps the conv chain on ttnn.conv2d's relayout-free L1 path, and it is
    the right choice while a shard is mostly real data. But a shard's height is padded up to
    a whole tile per core, so once ``hw/ncores`` falls below half a tile the padding
    dominates *and* the conv inherits a grid chosen for the previous stage's much larger
    shape: at the last stage (8x101) that cost 48.5us of halo + 45us of conv per layer,
    against 2.4us + 26.6us for the same conv handed an interleaved input to shard itself.
    The de-shard/re-shard pair it costs instead is ~3us at those sizes."""
    return None if hw >= ncores * (TILE // 2) else ttnn.L1_MEMORY_CONFIG


def _global_mean(x, hw):
    """Mean over the flat spatial dim of ``[1, 1, H*W, C]`` -> ``[1, 1, 1, C]`` (interleaved,
    so the SE's matmuls below stay on the plain — non-sharded — matmul path).

    ``ttnn.mean`` over a single tall dim runs on ~1 core (0.74 ms on the first stage's
    51264 rows). Splitting the height into whole tiles first — ``[1, H*W/32, 32, C]``, a
    free view since the split is tile-aligned — gives the reduce a batch dim to spread over
    cores, for the same result ~12x faster. Only possible when H*W is a multiple of a tile
    (true for every stage when the mel length is even; an odd length leaves the two
    narrowest stages, ~1/8 the elements, on the plain path)."""
    x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)  # de-shard: reduce needs a plain view
    if hw % TILE == 0:
        return ttnn.mean(ttnn.reshape(x, [1, hw // TILE, TILE, x.shape[-1]]), dim=[1, 2], keepdim=True)
    return ttnn.mean(x, dim=2, keepdim=True)


class TtSELayer(LightweightModule):
    """Squeeze-excite: global avg-pool -> Linear(C->C/8) -> relu -> Linear -> sigmoid -> scale.

    ``forward`` returns the excitation already as the ``[C, C]`` **diagonal matrix** the
    caller multiplies by, because scaling the activation's channels is 6x cheaper as a
    matmul than as a broadcast multiply (see :func:`_scale_channels`). Building the diagonal
    is one 2us op on the small ``[C, C]`` tile, and the sigmoid rides on it."""

    def __init__(self, device, se):
        super().__init__()
        # torch Linear weight is [out, in]; ttnn.linear wants [in, out].
        self.w1 = _to_body(se.fc[0].weight.t(), device)
        self.b1 = _to_body(se.fc[0].bias.reshape(1, -1), device)
        self.w2 = _to_body(se.fc[2].weight.t(), device)
        self.b2 = _to_body(se.fc[2].bias.reshape(1, -1), device)
        channels = se.fc[0].weight.shape[1]
        self.eye = _to_body(torch.eye(channels).reshape(1, 1, channels, channels), device)

    def forward(self, x, hw):  # x: [1, 1, H*W, C] -> diag(sigmoid(excitation)) [1, 1, C, C]
        y = _global_mean(x, hw)
        y = ttnn.linear(y, self.w1, bias=self.b1, activation="relu", memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.linear(y, self.w2, bias=self.b2, memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.mul(self.eye, y, input_tensor_b_activations=[SIGMOID], memory_config=ttnn.L1_MEMORY_CONFIG)


class TtSEBasicBlock(LightweightModule):
    """conv1 -> relu -> bn1 -> conv2 -> bn2 -> SE -> (+downsample) -> relu. The relus ride on
    the conv / the residual add, bn2 (and the downsample's BN) are folded into their conv's
    weights, and bn1 and the SE scaling are diagonal matmuls."""

    def __init__(self, device, block, **conv_kwargs):
        super().__init__()
        self.stride = stride = block.stride[0] if isinstance(block.stride, tuple) else block.stride
        self.conv1 = TtConv2d(
            device, block.conv1.weight.detach(), None, stride=stride, padding=1, activation=RELU, **conv_kwargs
        )
        self.bn1 = _ChannelAffine(device, *_bn_scale_shift(block.bn1))
        w2, b2 = _fold_bn(block.conv2.weight.detach(), None, block.bn2)
        self.conv2 = TtConv2d(device, w2, b2, stride=1, padding=1, **conv_kwargs)
        self.se = TtSELayer(device, block.se)
        self.compute_config = _body_compute_config(device)
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.ncores = grid.x * grid.y
        self.downsample_conv = None
        if block.downsample is not None:
            wd, bd = _fold_bn(block.downsample[0].weight.detach(), None, block.downsample[1])
            self.downsample_conv = TtConv2d(device, wd, bd, stride=stride, padding=0, **conv_kwargs)

    def forward(self, x, h, w):
        # conv2d's output size for kernel 3 / padding 1, so the stage's memory config is
        # known before the conv runs (all Python ints — nothing reads back from device).
        oh, ow = (h - 1) // self.stride + 1, (w - 1) // self.stride + 1
        mem = _stage_memory_config(self.ncores, oh * ow)
        out, oh, ow = self.conv1(x, h, w, mem)  # relu fused
        out = self.bn1(out)
        out, _, _ = self.conv2(out, oh, ow, mem)  # bn2 folded into the weights
        se_diag = self.se(out, oh * ow)
        residual = x if self.downsample_conv is None else self.downsample_conv(x, h, w, mem)[0]
        out = _scale_channels(out, se_diag, self.compute_config, self.core_grid)  # SE scale
        return ttnn.add(out, residual, activations=[RELU]), oh, ow


class TtResNetSpeakerEncoder(LightweightModule):
    """log-mel ``[1, 64, T]`` -> speaker embedding ``[1, 512]`` (L2-normalized)."""

    def __init__(self, device, ref):
        super().__init__()
        self.device = device
        body = {"activations_dtype": BODY_DTYPE, "math_fidelity": BODY_FIDELITY}
        self.conv1 = TtConv2d(
            device,
            ref.conv1.weight.detach(),
            ref.conv1.bias.detach(),
            stride=1,
            padding=1,
            activation=RELU,
            **body,
        )
        self.bn1 = _ChannelAffine(device, *_bn_scale_shift(ref.bn1))
        self.layers = [
            [TtSEBasicBlock(device, blk, **body) for blk in layer]
            for layer in (ref.layer1, ref.layer2, ref.layer3, ref.layer4)
        ]

        # Attention (ASP) in [C, T'] column layout: y = W @ x + b.
        att = ref.attention
        self.att_w1 = _to_tile(att[0].weight.detach().squeeze(-1), device)  # [128, 2048]
        self.att_b1 = _to_tile(att[0].bias.detach().reshape(-1, 1), device)  # [128, 1]
        att_scale, att_shift = _bn_scale_shift(att[2])  # BatchNorm1d(128) -> [128, 1]
        self.att_scale = _to_tile(att_scale.reshape(-1, 1), device)
        self.att_shift = _to_tile(att_shift.reshape(-1, 1), device)
        self.att_w2 = _to_tile(att[3].weight.detach().squeeze(-1), device)  # [2048, 128]
        self.att_b2 = _to_tile(att[3].bias.detach().reshape(-1, 1), device)  # [2048, 1]

        self.fc_w = _to_tile(ref.fc.weight.detach(), device)  # [512, 4096]
        self.fc_b = _to_tile(ref.fc.bias.detach().reshape(-1, 1), device)  # [512, 1]

    def forward(self, mel):  # mel: ttnn [1, 64, T] TILE
        _, freq, time = mel.shape
        # log(mel + 1e-6) then InstanceNorm1d over time (per freq) == a plain layer_norm:
        # both normalize the last dim, with no affine. One op instead of a mean/var chain,
        # and it handles the tile padding of a non-tile-aligned T.
        x = ttnn.layer_norm(ttnn.log(ttnn.add(mel, 1e-6)), epsilon=INSTANCENORM_EPS)
        # -> the conv's flat channels-last form [1, 1, H*W, C] with H=freq, W=time, C=1.
        # Typecast first (cheap while still tiled and 32-wide), and reshape in TILE: the
        # ROW_MAJOR route (untilize -> reshape -> retilize) costs ~5x more, because a flat
        # single-column ROW_MAJOR tensor is one page per element.
        x = ttnn.reshape(ttnn.typecast(x, BODY_DTYPE), [1, 1, freq * time, 1])
        # Into L1 for the whole body: every conv then takes ttnn.conv2d's L1 path, which
        # (unlike the DRAM path) doesn't bracket each conv with a 4D unflatten/re-flatten.
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        x, h, w = self.conv1(x, freq, time)  # relu fused
        x = self.bn1(x)
        for layer in self.layers:
            for block in layer:
                x, h, w = block(x, h, w)  # -> [1, 1, 8*T', 256]

        # ASP reshape: flat [1, 1, H*W, C] -> [N, H, W, C] -> [N, C, H, W] -> [C*H, W].
        c = x.shape[-1]
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)  # gather the body's last L1 shards
        x = ttnn.typecast(x, ttnn.float32)  # the ASP variance below needs fp32
        x = ttnn.reshape(x, [1, h, w, c])
        x = ttnn.permute(x, (0, 3, 1, 2))  # [1, 256, 8, T']
        x = ttnn.reshape(x, [c * h, w])  # [2048, T']

        # Attention weights over time. The tail's per-channel affine stays an addcmul rather
        # than the body's diagonal matmul: it runs on a small interleaved [128, T'] tensor,
        # where the broadcast that made the body's version expensive costs ~11us.
        a = ttnn.relu(ttnn.add(ttnn.matmul(self.att_w1, x), self.att_b1))  # [128, T']
        a = ttnn.addcmul(self.att_shift, a, self.att_scale, value=1.0)
        a = ttnn.add(ttnn.matmul(self.att_w2, a), self.att_b2)  # [2048, T']
        wgt = ttnn.softmax(a, dim=-1)  # over time

        # Attentive statistics pooling.
        mu = ttnn.sum(ttnn.mul(x, wgt), dim=-1, keepdim=True)  # [2048, 1]
        e2 = ttnn.sum(ttnn.mul(ttnn.mul(x, x), wgt), dim=-1, keepdim=True)
        var = ttnn.sub(e2, ttnn.mul(mu, mu))
        sg = ttnn.sqrt(ttnn.clamp(var, min=ASP_EPS))
        feat = ttnn.concat([mu, sg], dim=0)  # [4096, 1]

        g = ttnn.add(ttnn.matmul(self.fc_w, feat), self.fc_b)  # [512, 1]
        # L2 normalize over the 512 dim.
        norm = ttnn.sqrt(ttnn.sum(ttnn.mul(g, g), dim=0, keepdim=True))
        g = ttnn.div(g, norm)
        return ttnn.reshape(g, [1, 512])
