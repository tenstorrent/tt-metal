# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident TT pipeline for the FLUX.2 VAE (`AutoencoderKLFlux2`, 84M params).

This is the one place the bring-up's graduated TTNN stubs are wired into a
runnable model. Nothing here re-implements a layer: every op on the hot path
comes out of `models/tt_dit/pipelines/flux_2_klein_9b_vae/_stubs/`, which is
read-only graduated output (each live `_stubs/<name>.py` is byte-identical to
its `.last_good_sharded` snapshot, and a gate test checks that).

What the model is
-----------------
An image autoencoder, not a generative transformer. There is no `generate()`,
no KV cache and no sampling on the deterministic path:

    encode(x)      = DiagonalGaussian(quant_conv(encoder(x))).mode()
                   = first 32 channels of quant_conv(encoder(x))
    decode(z)      = decoder(post_quant_conv(z))
    reconstruct(x) = decode(encode(x).mode())          # == model(x).sample

At the pinned capacity an image is `[1, 3, 224, 224]` and a latent is
`[1, 32, 28, 28]`; the encoder itself emits `[1, 64, 28, 28]` (32 latent means
plus 32 log-variances, and the mode keeps only the means). The top-level `bn`
BatchNorm that the checkpoint registers is NEVER called by encode/decode/forward
in diffusers 0.38 — patchify/unpatchify live in the FLUX.2 *pipeline*, not in
the VAE — so it is deliberately absent from this chain. Putting it in would
diverge from the golden.

Three heads, four stacks, and why
---------------------------------
Twelve modules graduated, but they only cover TEN distinct positions in the
graph, because two of them are alias ports:

    encoder_stack -> `encoder`   (`_captured/encoder_stack/manifest.json`
                                  binds it to submodule_path "encoder";
                                  `TtEncoderStack` subclasses `TtEncoder`)
    decoder_head  -> `decoder`   (same story, `TtDecoderHead(TtDecoder)`)

A single chain cannot invoke both members of an alias pair without running the
same stack twice, so instead of dropping two graduated work products the alias
pair gets the head that actually corresponds to it: the whole-stack image ->
image forward (`run_reconstruct`). That gives four stacks:

    E   encoder      + graduated children     -> `run_encode`,      trace "encode"
    D   decoder      + graduated children     -> `run_decode`,      trace "decode"
    RE  encoder_stack, monolithic             -\
    RD  decoder_head,  monolithic             -/  `run_reconstruct`

Why the child stubs are PLUGGED INTO the monolith, not run alongside it
-----------------------------------------------------------------------
`encoder` and `down_encoder_block2_d` are ports of a parent and its own child:
`TtEncoder` already builds its down blocks out of the same `_vae_blocks`
pieces. Running the child stub *next to* the encoder — feeding it the same
input and throwing its output away — would be a coverage sweep, not a forward
path: the child's output would reach no task output and a mis-wired child would
not change a single number. So the child stubs REPLACE the parent's own
children. `E.down_blocks[i]` IS the graduated `down_encoder_block2_d` object;
its output is what the next down block consumes. Break it and `run_encode`
breaks. That is what makes the invocation ledger mean something.

The flat <-> NCHW seam, and what it costs
-----------------------------------------
`_vae_blocks` moves activations as **flat NHWC** — `[1, 1, N*H*W, C]` in
`TILE_LAYOUT`, with `(batch, height, width)` carried alongside as plain ints —
because that is what `ttnn.conv2d` consumes and produces. A parent therefore
calls its children with the flat signature `(x, batch, h, w) -> (x, h, w)`.

The graduated child stubs, however, were PCC-gated as standalone components
against `[N, C, H, W]` goldens, so each is an NCHW-in / NCHW-out wrapper: it
calls `nchw_to_flat_nhwc` on the way in and `flat_nhwc_to_nchw` on the way out.
Plugging one into a flat parent means the adapter has to hand it NCHW, so each
seam costs a `reshape + permute` down and the stub's own `permute + reshape`
back up — two permutes that a fused build would not pay. They are pure data
movement, they are exact, and they buy the thing the gate is about: the object
in the parent's list is the graduated stub itself, invoked through its real
`__call__`, not a re-implementation that merely looks like it.

`attention` is the exception: `TtVaeAttention.__call__` already folds a rank-4
`[B, ?, N, C]` activation into `[B, N, C]` itself, so its adapter passes the
flat tensor straight through with no conversion at all.

Tensor parallelism
------------------
TP is entirely the stubs' business and is not re-derived here; see
`_stubs/_vae_blocks.py`. In one line: every conv is COLUMN-parallel over its
output channels (`ShardTensorToMesh(dim=0)` on the weight, `dim=3` on the bias
so a bias stays with its own columns) closed by an `all_gather` on the channel
dim, and the mid-block attention splits qkv on the channel axis with a
row-parallel `to_out` closed by an `all_reduce`. Every stage output is
therefore REPLICATED across the mesh, which is why the readback concatenates
the mesh and keeps only the first `1/n_devices` of dim 0.

The two 1x1 convs this file owns directly — `quant_conv` (64 -> 64) and
`post_quant_conv` (32 -> 32) — are built as plain `_vae_blocks.Conv2d`, so they
inherit exactly that scheme; both out-channel counts divide 8, so both are
column-parallel like every other conv in the model.

Trace stages
------------
`PIPELINE_STAGES = ["encode", "decode"]`, and that is read straight off the
diffusers config: `down_block_types` is the compression phase and
`up_block_types` the expansion phase. This is a feed-forward autoencoder: no
`generate()`, no KV cache, no token-by-token phase, so there is no
prefill/single-token pair and no vocode stage. Inventing one would be a lie
about the model, and the trace unit is therefore the steady-state FORWARD pass
of each stage.

The variable axis is SPATIAL, not sequence — the config bound is `sample_size`
and the compression factor is 8 — so each stage pins a spatial capacity
(`image_size` for encode, `latent_size` for decode) at build time and traces at
exactly that shape. Every shape-dependent constant (GroupNorm one-hot
membership matrices, gamma/beta, conv weights) is staged by the stubs at build
time, and `<stage>_trace_setup` additionally warms the stage once so
`ttnn.conv2d` caches its device-prepared weights on the `Conv2d` objects — the
stubs write them back to `self.weight/self.bias` for exactly this reason.
Otherwise conv weight preparation, a host op, would run inside the capture.

Zero-padding a short input is supported but is NOT free and says so out loud:
this VAE's GroupNorm reduces over all H*W positions, so a zero-padded tail
moves the statistics of the real region. There is no mask that fixes that. The
supported path is that `VaeImageProcessor.preprocess(height=C, width=C)` has
already resized the input to exactly the pinned capacity, so `real == C`.

The `layers` knob
-----------------
`layers` caps the depth of every repeated block (None = every layer, 0 clamps
to 1). The genuine repeats here are the `resnets` INSIDE each down/up block
(2 per encoder block, 3 per decoder block) — NOT the four `down_blocks` /
`up_blocks`, which each change channel width and spatial resolution, so
dropping one would change the output shape and the stage could not run at all.
The mid block is held at 2 resnets whatever the cap, because
`UNetMidBlock2D` runs `resnets[0]` then `zip(attentions, resnets[1:])`: capping
it to 1 would make the ATTENTION structurally absent. That is the documented
"cap to the smallest depth that keeps every stage able to run" case, and it is
printed when it bites.

Capping happens at BUILD time, through read-only proxies over the HF module, so
a capped build never stages weights it then throws away — and `pipeline.hf`
stays pristine, because the goldens come from it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

import ttnn
from models.demos.flux_2_klein_9b.vae._stubs import _vae_blocks
from models.demos.flux_2_klein_9b.vae._stubs import attention as _s_attention
from models.demos.flux_2_klein_9b.vae._stubs import decoder as _s_decoder
from models.demos.flux_2_klein_9b.vae._stubs import decoder_head as _s_decoder_head
from models.demos.flux_2_klein_9b.vae._stubs import down_encoder_block2_d as _s_down_encoder_block2_d
from models.demos.flux_2_klein_9b.vae._stubs import downsample2_d as _s_downsample2_d
from models.demos.flux_2_klein_9b.vae._stubs import encoder as _s_encoder
from models.demos.flux_2_klein_9b.vae._stubs import encoder_stack as _s_encoder_stack
from models.demos.flux_2_klein_9b.vae._stubs import resnet_block2_d as _s_resnet_block2_d
from models.demos.flux_2_klein_9b.vae._stubs import self_attention as _s_self_attention
from models.demos.flux_2_klein_9b.vae._stubs import u_net_mid_block2_d as _s_u_net_mid_block2_d
from models.demos.flux_2_klein_9b.vae._stubs import up_decoder_block2_d as _s_up_decoder_block2_d
from models.demos.flux_2_klein_9b.vae._stubs import upsample2_d as _s_upsample2_d

__all__ = [
    "PIPELINE_STAGES",
    "GRADUATED_MODULES",
    "RECOMMENDED_L1_SMALL_SIZE",
    "TtFluxVaePipeline",
    "build_pipeline",
    "recommended_trace_region_size",
    "host_op_selftest",
    "trace_capture_selftest",
    "selftest_config",
]


# --------------------------------------------------------------------------
# Frozen public constants
# --------------------------------------------------------------------------

#: The trace stages this model has. Derived from the diffusers config, which
#: states the phases directly (`down_block_types` / `up_block_types`). Both are
#: one-shot forward passes; neither of the single-token generative hooks exists
#: on this pipeline, and `tests/e2e/test_trace_contract.py` asserts their
#: ABSENCE, because this model has no token-by-token phase to hang them on.
PIPELINE_STAGES = ["encode", "decode"]

#: The twelve modules that graduated in the bring-up (each has a
#: `_stubs/<name>.py.last_good_sharded` snapshot). Every one of them is routed
#: into a real forward path below; see `_ROUTING` for the table.
GRADUATED_MODULES = (
    "attention",
    "decoder",
    "decoder_head",
    "down_encoder_block2_d",
    "downsample2_d",
    "encoder",
    "encoder_stack",
    "resnet_block2_d",
    "self_attention",
    "u_net_mid_block2_d",
    "up_decoder_block2_d",
    "upsample2_d",
)

#: `ttnn.conv2d`'s sliding-window/halo helper allocates out of the L1_SMALL
#: bank. With the historical default of 0 the first conv raises
#: `TT_FATAL: bank size is 0 B` and silently degrades to a host implementation,
#: which would break both Gate 1 and the host-op selftest. Open the device with
#: `device_params={"l1_small_size": RECOMMENDED_L1_SMALL_SIZE}`.
RECOMMENDED_L1_SMALL_SIZE = 24576

#: Image side the bring-up capture used; `_captured/encoder/args.pt` is
#: `[1, 3, 224, 224]` and `_captured/decoder/args.pt` is `[1, 32, 28, 28]`.
_DEFAULT_IMAGE_SIZE = 224

#: Spatial compression of this VAE (`spatial_compression: 8` in config.json).
_VAE_SCALE_FACTOR = 8

#: .../models/demos/flux_2_klein_9b_vae/tt/pipeline.py -> .../models
_MODELS_ROOT = Path(__file__).resolve().parents[3]
_BRINGUP_DIR = _MODELS_ROOT / "tt_dit" / "pipelines" / "flux_2_klein_9b_vae"

#: Which captured golden supplies each stage's trace input. `encoder`/`decoder`
#: are the components whose captured args ARE the stage inputs.
_STAGE_CAPTURE = {"encode": "encoder", "decode": "decoder"}


# --------------------------------------------------------------------------
# The routing table (Gate 2). Kept as data so a test can read it, and so the
# wiring below can be checked against a single statement of intent.
#
#   component            -> (stack, position it occupies, what it replaces)
# --------------------------------------------------------------------------
_ROUTING = {
    "encoder": ("E", "encode stack, top level", "the whole diffusers Encoder"),
    "down_encoder_block2_d": ("E", "E.down_blocks[0..3]", "TtEncoder's own down blocks"),
    "resnet_block2_d": ("E", "E.down_blocks[0].resnets[0]", "that block's first resnet"),
    "downsample2_d": ("E", "E.down_blocks[0].downsamplers[0]", "that block's downsampler"),
    "u_net_mid_block2_d": ("E", "E.mid_block", "TtEncoder's own mid block"),
    "attention": ("E", "E.mid_block.attentions[0]", "the encoder mid-block attention"),
    "decoder": ("D", "decode stack, top level", "the whole diffusers Decoder"),
    "up_decoder_block2_d": ("D", "D.up_blocks[0..3]", "TtDecoder's own up blocks"),
    "upsample2_d": ("D", "D.up_blocks[0].upsamplers[0]", "that block's upsampler"),
    "self_attention": ("D", "D.mid_block.attentions[0]", "the decoder mid-block attention"),
    "encoder_stack": ("RE", "reconstruct stack, encode leg", "the alias port of `encoder`"),
    "decoder_head": ("RD", "reconstruct stack, decode leg", "the alias port of `decoder`"),
}


# ==========================================================================
# Recording adapters
#
# A parent from `_vae_blocks` calls its children with the FLAT signature
# `(x, batch, h, w) -> (x, h, w)`. The graduated child stubs speak NCHW. Each
# adapter therefore: converts down to NCHW, calls the stub's REAL `__call__`
# (this is what "invoked" means for Gate 2), converts back up to flat, notes
# the component name in the ledger, and returns the flat tuple the parent
# expects. The tensor it returns is the stub's own output, unchanged.
#
# There is ONE base class and the concrete adapters differ only in the calling
# convention they present, so every element of a replaced list can be wrapped
# in the SAME class -- `down_blocks`, `resnets`, `attentions`, `upsamplers` and
# `downsamplers` all stay plain Python lists of same-typed elements, which is
# what keeps the built model discoverable.
#
# `component=None` means "record nothing": it is how an element that has no
# graduated stub bound to its position (e.g. `resnets[1]`) is wrapped in the
# same class as its sibling without inventing a fake invocation.
# `flat=True` means the wrapped object already speaks the flat protocol, so no
# NCHW round trip is inserted.
# ==========================================================================


class _Adapter:
    """Common base: holds the ledger and the recording rule."""

    def __init__(self, component, inner, ledger, *, flat=False):
        self.component = component
        self.inner = inner
        self._ledger = ledger
        self._flat = flat

    def _record(self):
        if self.component:
            self._ledger[self.component] = self._ledger.get(self.component, 0) + 1

    def __getattr__(self, name):
        # Delegate anything a parent might read (`.out_channels`,
        # `.out_channels_full`, ...) to the wrapped object. The guard keeps the
        # lookup of our own state out of the delegation path, so a half-built
        # adapter raises AttributeError instead of recursing forever.
        if name in ("component", "inner", "_ledger", "_flat"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.component!r}, {type(self.inner).__name__})"

    # -- the flat <-> NCHW seam -------------------------------------------
    def _call_spatial(self, x, batch, height, width):
        """Run a `(x, batch, h, w) -> (x, h, w)` child, recording the call."""
        self._record()
        if self._flat:
            # Already a `_vae_blocks.*` object: it IS the flat protocol.
            return self.inner(x, batch, height, width)

        # The incoming activation is `[1, 1, N*H*W, C]`; C is exact from the
        # tensor itself, so the NCHW view never has to be guessed from the
        # module (a conv's in_channels would be wrong for a resnet, and a
        # block's out_channels would be wrong for its input).
        channels = int(x.shape[-1])
        nchw = _vae_blocks.flat_nhwc_to_nchw(x, batch, channels, height, width)
        out = self.inner(nchw)
        flat, _batch, _channels, out_h, out_w = _vae_blocks.nchw_to_flat_nhwc(out)
        return flat, out_h, out_w


class _BlockAdapter(_Adapter):
    """A down/up/mid block sitting in a parent's block list."""

    def __call__(self, x, batch, height, width):
        return self._call_spatial(x, batch, height, width)


class _ResnetAdapter(_Adapter):
    """One element of a block's `resnets` list."""

    def __call__(self, x, batch, height, width):
        return self._call_spatial(x, batch, height, width)


class _SpatialAdapter(_Adapter):
    """One element of a block's `downsamplers` / `upsamplers` list."""

    def __call__(self, x, batch, height, width):
        return self._call_spatial(x, batch, height, width)


class _AttentionAdapter(_Adapter):
    """One element of a mid block's `attentions` list.

    No conversion: `TtVaeAttention.__call__` takes the flat rank-4 activation
    directly (it folds `[B, ?, N, C]` down to `[B, N, C]` itself) and returns
    it in the same shape, so the seam cost here is zero.
    """

    def __call__(self, x, *args, **kwargs):
        self._record()
        return self.inner(x, *args, **kwargs)


class _StackAdapter(_Adapter):
    """A whole stack (`TtEncoder` / `TtDecoder` and their alias ports).

    NCHW in, NCHW out -- the stacks are the model boundary, so there is no
    conversion to do; the adapter exists only to put the stack's own name in
    the ledger when its real `__call__` runs.
    """

    def __call__(self, sample, *args, **kwargs):
        self._record()
        return self.inner(sample, *args, **kwargs)


# ==========================================================================
# Build-time depth caps (the `layers` knob)
#
# Read-only proxies over the HF module: they expose a TRUNCATED `resnets` /
# `down_blocks` / `up_blocks` list and delegate everything else. Nothing is
# mutated, so `pipeline.hf` stays pristine for the goldens -- and because the
# cap is applied BEFORE the stub constructors read the weights, a capped build
# never stages a weight it then throws away.
# ==========================================================================


class _CappedResnets:
    """A down/up block with its `resnets` list truncated to `n`."""

    def __init__(self, block, n):
        self._b = block
        self.resnets = list(block.resnets)[: max(1, int(n))]

    def __getattr__(self, k):
        return getattr(self._b, k)


class _CappedEncoder:
    """`encoder` with every down block's resnet list capped.

    `mid_block` is deliberately delegated UNCAPPED: `UNetMidBlock2D` runs
    `resnets[0]` and then `zip(attentions, resnets[1:])`, so a mid block with
    one resnet has no attention at all.
    """

    def __init__(self, module, n):
        self._m = module
        self.down_blocks = [_CappedResnets(b, n) for b in module.down_blocks]

    def __getattr__(self, k):
        return getattr(self._m, k)


class _CappedDecoder:
    """`decoder` with every up block's resnet list capped (mid block intact)."""

    def __init__(self, module, n):
        self._m = module
        self.up_blocks = [_CappedResnets(b, n) for b in module.up_blocks]

    def __getattr__(self, k):
        return getattr(self._m, k)


def _resolve_cap(layers, stage_layers):
    """`stage_layers` wins, then `layers`; None means every layer, 0 clamps to 1."""
    value = stage_layers if stage_layers is not None else layers
    if value is None:
        return None
    return max(1, int(value))


# ==========================================================================
# Mesh-safe host <-> device transfer
#
# Lifted from the bring-up's own PCC harness
# (`_stubs/../tests/pcc/test_encoder.py::_ttnn_to_torch_mesh_safe`), which is
# the readback path every graduated component was gated with. Bare `to_torch`
# and `get_device_tensors()[0]` can busy-loop in the C extension with the GIL
# held on a MeshDevice, so the composer form is the only stable one.
# ==========================================================================


def _is_mesh_device(device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


def _n_devices(device):
    try:
        ids = device.get_device_ids() if hasattr(device, "get_device_ids") else []
        return len(ids) or 1
    except Exception:
        return 1


def _ttnn_to_torch_mesh_safe(tensor, device):
    """Read a (replicated) device tensor back as a float32 torch tensor."""
    if isinstance(tensor, torch.Tensor):
        # Should never happen on the TT path; a torch tensor here would mean a
        # stub fell back to host compute. The stage asserts catch that first,
        # but do not crash inside the readback if it ever does.
        return tensor.to(torch.float32)

    try:
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(device)
        elif hasattr(device, "synchronize"):
            device.synchronize()
    except Exception:
        pass

    if _is_mesh_device(device):
        for make_composer in (
            lambda: ttnn.concat_mesh_to_tensor_composer(device, 0),
            lambda: ttnn.ConcatMeshToTensor(device, dim=0),
        ):
            try:
                composer = make_composer()
            except (AttributeError, TypeError):
                continue
            try:
                out = ttnn.to_torch(tensor, mesh_composer=composer)
                if out is None:
                    continue
                # Every stage output is REPLICATED across the mesh, so the
                # concatenation is `n_devices` identical copies stacked on dim
                # 0. Keep the first one.
                if out.ndim >= 1 and out.shape[0] > 1:
                    n = _n_devices(device)
                    if n > 1 and out.shape[0] % n == 0:
                        out = out[: out.shape[0] // n]
                return out.to(torch.float32)
            except Exception:
                continue
    return ttnn.to_torch(tensor).to(torch.float32)


# ==========================================================================
# Structural walkers (used by `repeated_block_counts` and `<stage>_trace_items`)
# ==========================================================================


def _native_block(obj):
    """Peel adapters and graduated wrappers down to the `_vae_blocks` object."""
    for _ in range(8):
        if isinstance(obj, _Adapter):
            obj = obj.inner
            continue
        # `__dict__` rather than getattr, so an adapter's delegation cannot
        # smuggle in an attribute the object does not really own.
        inner = obj.__dict__.get("block")
        if inner is not None:
            obj = inner
            continue
        break
    return obj


def _resnets_of(obj):
    return list(getattr(_native_block(obj), "resnets", []))


def _conv_out_hw(conv, height, width, pad_h=None, pad_w=None):
    """Output spatial size of a `Conv2d` reference module, padding overridable."""
    kh, kw = int(conv.kernel_size[0]), int(conv.kernel_size[1])
    sh, sw = int(conv.stride[0]), int(conv.stride[1])
    dh, dw = int(conv.dilation[0]), int(conv.dilation[1])
    ph = 2 * int(conv.padding[0]) if pad_h is None else int(pad_h)
    pw = 2 * int(conv.padding[1]) if pad_w is None else int(pad_w)
    return (
        (height + ph - dh * (kh - 1) - 1) // sh + 1,
        (width + pw - dw * (kw - 1) - 1) // sw + 1,
    )


def _conv_params(conv):
    n = int(conv.weight.numel())
    if conv.bias is not None:
        n += int(conv.bias.numel())
    return n


def _resnet_convs(resnet, height, width, out):
    """Append `(conv, out_h, out_w)` for a VAE resnet; return its output size."""
    h1, w1 = _conv_out_hw(resnet.conv1, height, width)
    out.append((resnet.conv1, h1, w1))
    h2, w2 = _conv_out_hw(resnet.conv2, h1, w1)
    out.append((resnet.conv2, h2, w2))
    if getattr(resnet, "conv_shortcut", None) is not None:
        # The shortcut sees the block INPUT, not conv1's output.
        hs, ws = _conv_out_hw(resnet.conv_shortcut, height, width)
        out.append((resnet.conv_shortcut, hs, ws))
    return h2, w2


def _capped(resnets, cap):
    return list(resnets)[: max(1, int(cap))] if cap is not None else list(resnets)


def _encode_conv_ladder(hf, size, cap):
    """Every `nn.Conv2d` the encode stage runs, with its OUTPUT spatial size."""
    enc = hf.encoder
    height = width = int(size)
    ladder = []

    height, width = _conv_out_hw(enc.conv_in, height, width)
    ladder.append((enc.conv_in, height, width))

    for block in enc.down_blocks:
        for resnet in _capped(block.resnets, cap):
            height, width = _resnet_convs(resnet, height, width, ladder)
        for down in block.downsamplers or []:
            # `Downsample2D(padding=0)` is diffusers' hand-rolled asymmetric
            # pad -- bottom and right only, so ONE extra row and column, not
            # two, which is what makes 224 -> 112 rather than 224 -> 111.
            pad = 1 if int(down.padding) == 0 else None
            height, width = _conv_out_hw(down.conv, height, width, pad, pad)
            ladder.append((down.conv, height, width))

    mid = enc.mid_block
    height, width = _resnet_convs(mid.resnets[0], height, width, ladder)
    for _attn, resnet in zip(mid.attentions, mid.resnets[1:]):
        height, width = _resnet_convs(resnet, height, width, ladder)

    height, width = _conv_out_hw(enc.conv_out, height, width)
    ladder.append((enc.conv_out, height, width))

    if getattr(hf, "quant_conv", None) is not None:
        qh, qw = _conv_out_hw(hf.quant_conv, height, width)
        ladder.append((hf.quant_conv, qh, qw))
    return ladder


def _decode_conv_ladder(hf, size, cap):
    """Every `nn.Conv2d` the decode stage runs, with its OUTPUT spatial size."""
    dec = hf.decoder
    height = width = int(size)
    ladder = []

    if getattr(hf, "post_quant_conv", None) is not None:
        height, width = _conv_out_hw(hf.post_quant_conv, height, width)
        ladder.append((hf.post_quant_conv, height, width))

    height, width = _conv_out_hw(dec.conv_in, height, width)
    ladder.append((dec.conv_in, height, width))

    mid = dec.mid_block
    height, width = _resnet_convs(mid.resnets[0], height, width, ladder)
    for _attn, resnet in zip(mid.attentions, mid.resnets[1:]):
        height, width = _resnet_convs(resnet, height, width, ladder)

    for block in dec.up_blocks:
        for resnet in _capped(block.resnets, cap):
            height, width = _resnet_convs(resnet, height, width, ladder)
        for up in block.upsamplers or []:
            # Nearest 2x interpolation happens BEFORE the conv, so the conv
            # already sees the doubled feature map.
            height, width = height * 2, width * 2
            height, width = _conv_out_hw(up.conv, height, width)
            ladder.append((up.conv, height, width))

    height, width = _conv_out_hw(dec.conv_out, height, width)
    ladder.append((dec.conv_out, height, width))
    return ladder


# ==========================================================================
# Trace region sizing
# ==========================================================================

# Per-op dispatch-command budget inside the trace buffer. A conv2d expands to
# a halo/sliding-window op plus its matmul, and its command stream is the
# widest of anything here; 24 KiB per dispatched op is the conservative figure
# this repo's CNN demos land on.
_TRACE_BYTES_PER_DISPATCH = 24 * 1024
_TRACE_HEADROOM = 1.25
_TRACE_FLOOR = 8 * 1024 * 1024
_TRACE_CEILING = 128 * 1024 * 1024

# Op counts per structural unit, read off `_vae_blocks`:
#   conv        : ttnn.conv2d (+ its halo helper) + the channel all_gather
#   group_norm  : 3 matmuls, 2 sums, sub/mul/mul/mul/add/rsqrt/add, 2 reshapes
#   resnet      : 2 convs + 2 group norms + 2 silu + add + scale
#   attention   : 1 group norm + 3 linears + 2 all_gathers + transpose +
#                 2 matmuls + scale + softmax + matmul + all_reduce + 2 adds
#   resampler   : 1 conv + the layout/reshape pair around ttnn.upsample
_OPS_PER_CONV = 3
_OPS_PER_GROUPNORM = 12
_OPS_PER_RESNET = 2 * _OPS_PER_CONV + 2 * _OPS_PER_GROUPNORM + 4
_OPS_PER_ATTENTION = _OPS_PER_GROUPNORM + 14
_OPS_PER_RESAMPLER = _OPS_PER_CONV + 5

# Structure of this checkpoint, from config.json (`block_out_channels` has 4
# entries; `layers_per_block=2` in the encoder, +1 in the decoder as diffusers
# builds `Decoder`; 3 of the 4 blocks resample; the mid block is 2 resnets and
# 1 attention on both sides).
_STRUCTURE = {
    "encode": {"resnets": 4 * 2 + 2, "resamplers": 3, "attentions": 1, "loose_convs": 3, "loose_norms": 1},
    "decode": {"resnets": 4 * 3 + 2, "resamplers": 3, "attentions": 1, "loose_convs": 3, "loose_norms": 1},
}


def _stage_op_estimate(stage):
    s = _STRUCTURE[stage]
    return (
        s["resnets"] * _OPS_PER_RESNET
        + s["resamplers"] * _OPS_PER_RESAMPLER
        + s["attentions"] * _OPS_PER_ATTENTION
        + s["loose_convs"] * _OPS_PER_CONV
        + s["loose_norms"] * _OPS_PER_GROUPNORM
        + 4  # silu, slice, and the NCHW seam at the model boundary
    )


def recommended_trace_region_size() -> int:
    """Bytes to reserve for the trace region, sized from the LARGEST stage.

    Formula, and why each term is what it is:

        ops   = the dispatched-op count of the deepest stage (`decode`: it has
                3 resnets per up block against the encoder's 2), derived from
                the config's block ladder rather than measured, so this can be
                answered before a device exists;
        bytes = ops * 24 KiB * 1.25, clamped to [8 MiB, 128 MiB].

    Only ONE stage's trace is ever resident at a time -- `trace_capture_selftest`
    releases each stage before capturing the next -- so the region is sized for
    the maximum, not the sum.
    """
    ops = max(_stage_op_estimate(stage) for stage in PIPELINE_STAGES)
    raw = int(ops * _TRACE_BYTES_PER_DISPATCH * _TRACE_HEADROOM)
    raw = max(_TRACE_FLOOR, min(_TRACE_CEILING, raw))
    mib = 1024 * 1024
    return ((raw + mib - 1) // mib) * mib


# ==========================================================================
# Captured trace inputs
# ==========================================================================


def _captured_stage_input(stage):
    """The stage's real captured input tensor, float32 on CPU.

    Goes through `tt/reference.py::captured_tensor` when that module is
    importable (it is the owner of the capture contract); falls back to reading
    `_captured/<component>/args.pt` directly so this module keeps working if
    the reference module is momentarily absent.
    """
    component = _STAGE_CAPTURE[stage]
    try:
        from models.demos.flux_2_klein_9b.vae.tt.reference import captured_tensor

        return captured_tensor(component, "args", 0)
    except Exception:
        path = _BRINGUP_DIR / "_captured" / component / "args.pt"
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, (tuple, list)):
            obj = obj[0]
        elif isinstance(obj, dict):
            obj = list(obj.values())[0]
        return obj.detach().to(device="cpu", dtype=torch.float32)


# ==========================================================================
# The pipeline
# ==========================================================================


class TtFluxVaePipeline:
    """The resident TT model: four stacks, three heads, two trace stages.

    Built by `build_pipeline`; never build this directly unless you already
    hold the HF reference module.
    """

    def __init__(self, device, hf, *, layers=None, encode_layers=None, decode_layers=None, image_size=None):
        self.device = device
        #: The HF reference, kept reachable and PRISTINE -- it is the ground
        #: truth for how many sections the model has and how deep each is, and
        #: the goldens come from it.
        self.hf = hf
        self.tp = _vae_blocks.mesh_width(device)

        # ---- pinned capacities -------------------------------------------
        self.image_size = int(image_size or _DEFAULT_IMAGE_SIZE)
        self.latent_size = self.image_size // _VAE_SCALE_FACTOR
        self.latent_channels = int(getattr(hf.config, "latent_channels", 32))

        # ---- depth caps ---------------------------------------------------
        self._cap = {
            "encode": _resolve_cap(layers, encode_layers),
            "decode": _resolve_cap(layers, decode_layers),
        }
        if self._cap["encode"] is not None or self._cap["decode"] is not None:
            # The mid block is the documented floor case: capping its resnets
            # to 1 would delete the attention, so it is held at 2 and said so.
            print(
                "[flux2-vae] layers cap active "
                f"(encode={self._cap['encode']}, decode={self._cap['decode']}); "
                "the mid block is held at 2 resnets because UNetMidBlock2D runs "
                "resnets[0] then zip(attentions, resnets[1:]) -- capping it to 1 "
                "would make the attention structurally absent.",
                flush=True,
            )

        # ---- the invocation ledger (Gate 2) -------------------------------
        # A passive recorder: an adapter bumps its component's count when the
        # graduated stub's real `__call__` runs. Nothing sweeps it.
        self._ledger = {}

        # ---- the four stacks ----------------------------------------------
        self.E = self._build_encode_stack()
        self.D = self._build_decode_stack()
        self.RE, self.RD = self._build_reconstruct_stacks()

        # ---- the two 1x1 convs this file owns -----------------------------
        # 64 -> 64 and 32 -> 32; both divide the 8-wide mesh, so both are
        # column-parallel with an all_gather, exactly like every other conv.
        self.quant_conv = _vae_blocks.Conv2d(device, hf.quant_conv, self.tp)
        self.post_quant_conv = _vae_blocks.Conv2d(device, hf.post_quant_conv, self.tp)

        # ---- trace state ---------------------------------------------------
        self._trace_in = {}  # stage -> persistent device input buffer
        self._trace_out = {}  # stage -> last device output of `<stage>_trace_step`
        self._stage_device = {"encode": self._encode_device, "decode": self._decode_device}

    # ----------------------------------------------------------------------
    # Stack construction
    # ----------------------------------------------------------------------

    def _adapt(self, cls, component, inner, *, flat=False):
        return cls(component, inner, self._ledger, flat=flat)

    def _build_encode_stack(self):
        """`encoder` with its children replaced by their own graduated stubs.

        `TtEncoder` builds a complete set of native children of its own; those
        are dropped on the floor here and replaced. The waste is small and
        deliberate: the conv weights the discarded children hold are still HOST
        tensors at this point (`_vae_blocks.Conv2d` stages them without
        `device=` and only `ttnn.conv2d`'s first call moves them), so all that
        is freed is a handful of GroupNorm membership matrices, which Python
        drops as soon as the lists are reassigned.
        """
        device, hf, cap = self.device, self.hf, self._cap["encode"]
        src = _CappedEncoder(hf.encoder, cap) if cap is not None else hf.encoder

        encoder_obj = _s_encoder.build(device, src)

        # -- down_blocks[0..3] -> the graduated `down_encoder_block2_d` ------
        # Every element of the list is wrapped in the SAME adapter class, so
        # the list stays homogeneous and a test can walk it.
        # `src.down_blocks` is the capped proxy list when a cap is in force and
        # the HF list otherwise -- either way it is what `TtEncoder` itself was
        # built from, so parent and children see the same depth.
        down_src = src.down_blocks
        encoder_obj.down_blocks = [
            self._adapt(
                _BlockAdapter,
                "down_encoder_block2_d",
                _s_down_encoder_block2_d.build(device, block),
            )
            for block in down_src
        ]

        # -- inside down_blocks[0]: resnets[0] and the downsampler -----------
        # `resnets[0]` is the position `resnet_block2_d` was captured at
        # (`encoder.down_blocks.0.resnets.0`) and `downsamplers[0]` the one
        # `downsample2_d` was captured at.
        block0_src = down_src[0]
        inner0 = encoder_obj.down_blocks[0].inner.block  # _vae_blocks.DownEncoderBlock2D
        native_resnets = list(inner0.resnets)

        resnets = [
            self._adapt(
                _ResnetAdapter,
                "resnet_block2_d",
                _s_resnet_block2_d.build(device, block0_src.resnets[0]),
            )
        ]
        # No stub is bound to `resnets[1]`, so it keeps the natively built
        # object -- wrapped in the same class with `component=None` (record
        # nothing) and `flat=True` (it already speaks the flat protocol), so
        # the list is still same-typed end to end.
        for native in native_resnets[1:]:
            resnets.append(self._adapt(_ResnetAdapter, None, native, flat=True))
        inner0.resnets = resnets

        if inner0.downsamplers:
            inner0.downsamplers = [
                self._adapt(
                    _SpatialAdapter,
                    "downsample2_d",
                    _s_downsample2_d.build(device, down_src),
                )
                for down_src in block0_src.downsamplers
            ]

        # -- mid_block -> the graduated `u_net_mid_block2_d` ------------------
        encoder_obj.mid_block = self._adapt(
            _BlockAdapter,
            "u_net_mid_block2_d",
            _s_u_net_mid_block2_d.build(device, hf.encoder.mid_block),
        )
        # ...and its attention -> the graduated `attention` (captured at
        # `encoder.mid_block.attentions.0`).
        mid_inner = encoder_obj.mid_block.inner.block  # _vae_blocks.UNetMidBlock2D
        mid_inner.attentions = [
            self._adapt(_AttentionAdapter, "attention", _s_attention.build(device, attn_src))
            for attn_src in hf.encoder.mid_block.attentions
        ]

        return self._adapt(_StackAdapter, "encoder", encoder_obj)

    def _build_decode_stack(self):
        """`decoder` with its children replaced by their own graduated stubs."""
        device, hf, cap = self.device, self.hf, self._cap["decode"]
        src = _CappedDecoder(hf.decoder, cap) if cap is not None else hf.decoder

        decoder_obj = _s_decoder.build(device, src)

        up_src = src.up_blocks
        decoder_obj.up_blocks = [
            self._adapt(
                _BlockAdapter,
                "up_decoder_block2_d",
                _s_up_decoder_block2_d.build(device, block),
            )
            for block in up_src
        ]

        # -- inside up_blocks[0]: the upsampler ------------------------------
        # `upsample2_d` was captured at `decoder.up_blocks.0.upsamplers.0`.
        inner0 = decoder_obj.up_blocks[0].inner.block  # _vae_blocks.UpDecoderBlock2D
        if inner0.upsamplers:
            inner0.upsamplers = [
                self._adapt(_SpatialAdapter, "upsample2_d", _s_upsample2_d.build(device, up))
                for up in up_src[0].upsamplers
            ]

        # -- the mid-block attention -> the graduated `self_attention` -------
        # The decode mid block stays the natively built `_vae_blocks`
        # UNetMidBlock2D (no `u_net_mid_block2_d` stub is bound to the decoder
        # side; that component was captured on the encoder). Only its
        # attention list is replaced.
        decoder_obj.mid_block.attentions = [
            self._adapt(
                _AttentionAdapter,
                "self_attention",
                _s_self_attention.build(device, attn_src),
            )
            for attn_src in hf.decoder.mid_block.attentions
        ]

        return self._adapt(_StackAdapter, "decoder", decoder_obj)

    def _build_reconstruct_stacks(self):
        """The alias ports, built monolithically -- no child replacement.

        `encoder_stack`/`decoder_head` are graduated ports of the SAME modules
        as `encoder`/`decoder`. Replacing their children too would just run the
        child stubs a second time and prove nothing extra; what these two are
        for is the head that actually corresponds to them, the whole-stack
        image -> image forward.
        """
        device, hf = self.device, self.hf
        enc_cap, dec_cap = self._cap["encode"], self._cap["decode"]
        enc_src = _CappedEncoder(hf.encoder, enc_cap) if enc_cap is not None else hf.encoder
        dec_src = _CappedDecoder(hf.decoder, dec_cap) if dec_cap is not None else hf.decoder

        recon_encode = self._adapt(_StackAdapter, "encoder_stack", _s_encoder_stack.build(device, enc_src))
        recon_decode = self._adapt(_StackAdapter, "decoder_head", _s_decoder_head.build(device, dec_src))
        return recon_encode, recon_decode

    # ----------------------------------------------------------------------
    # Host <-> device
    # ----------------------------------------------------------------------

    def _upload(self, tensor):
        """torch -> replicated ttnn on the mesh."""
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.tp > 1 else None
        return ttnn.from_torch(
            tensor.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mapper,
        )

    def _upload_host(self, tensor):
        """torch -> a HOST-side ttnn tensor shaped for `copy_host_to_device_tensor`."""
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.tp > 1 else None
        return ttnn.from_torch(
            tensor.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )

    def _download(self, tensor):
        return _ttnn_to_torch_mesh_safe(tensor, self.device)

    # ----------------------------------------------------------------------
    # The device-side chains. `run_*`, the trace steps and the host-op
    # selftest ALL go through these -- there is exactly one copy of the wiring.
    # ----------------------------------------------------------------------

    def _posterior_mode(self, latent_full):
        """`encoder(x)` -> the posterior MODE, still flat and still on device.

        `DiagonalGaussianDistribution.mode()` is its mean, which for this VAE
        is simply the first `latent_channels` channels of `quant_conv`'s
        output -- so the "sampling" step is a `ttnn.slice` on the channel dim.
        32 channels is exactly one tile, so the slice is tile-aligned.
        """
        flat, batch, _channels, height, width = _vae_blocks.nchw_to_flat_nhwc(latent_full)
        flat, height, width = self.quant_conv(flat, batch, height, width)
        positions = int(flat.shape[-2])
        mode = ttnn.slice(flat, (0, 0, 0, 0), (1, 1, positions, self.latent_channels))
        return mode, batch, height, width

    def _apply_post_quant(self, flat, batch, height, width):
        """The flat latent -> `post_quant_conv` -> NCHW, ready for a decode stack."""
        flat, height, width = self.post_quant_conv(flat, batch, height, width)
        return _vae_blocks.flat_nhwc_to_nchw(flat, batch, self.post_quant_conv.out_channels_full, height, width)

    def _encode_device(self, x_tt, stack=None):
        """`[1, 3, H, W]` ttnn -> `[1, 32, H/8, W/8]` ttnn. Nothing touches host."""
        stack = self.E if stack is None else stack
        latent_full = stack(x_tt)
        assert isinstance(latent_full, ttnn.Tensor), (
            "stage `encode`: the encode stack returned a "
            f"{type(latent_full).__name__}, not a ttnn.Tensor -- a graduated stub "
            "fell back to torch on host"
        )
        mode, batch, height, width = self._posterior_mode(latent_full)
        out = _vae_blocks.flat_nhwc_to_nchw(mode, batch, self.latent_channels, height, width)
        assert isinstance(out, ttnn.Tensor), "stage `encode`: output is not a ttnn.Tensor"
        return out

    def _decode_device(self, z_tt, stack=None):
        """`[1, 32, h, w]` ttnn -> `[1, 3, 8h, 8w]` ttnn. Nothing touches host."""
        stack = self.D if stack is None else stack
        flat, batch, _channels, height, width = _vae_blocks.nchw_to_flat_nhwc(z_tt)
        z_nchw = self._apply_post_quant(flat, batch, height, width)
        out = stack(z_nchw)
        assert isinstance(out, ttnn.Tensor), (
            "stage `decode`: the decode stack returned a "
            f"{type(out).__name__}, not a ttnn.Tensor -- a graduated stub fell "
            "back to torch on host"
        )
        return out

    def _reconstruct_device(self, x_tt):
        """`[1, 3, H, W]` ttnn -> `[1, 3, H, W]` ttnn, through the alias stacks.

        The latent NEVER leaves the device and is never substituted: the tensor
        `RE` produced is the tensor `RD` consumes.
        """
        z_tt = self._encode_device(x_tt, stack=self.RE)
        assert isinstance(z_tt, ttnn.Tensor), "stage `reconstruct`: the latent is not a ttnn.Tensor"
        out = self._decode_device(z_tt, stack=self.RD)
        assert isinstance(out, ttnn.Tensor), "stage `reconstruct`: output is not a ttnn.Tensor"
        return out

    # ----------------------------------------------------------------------
    # The three task heads
    # ----------------------------------------------------------------------

    def run_encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`[1, 3, H, W]` -> `[1, 32, H/8, W/8]`, float32 on CPU."""
        return self._download(self._encode_device(self._upload(pixel_values)))

    def run_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """`[1, 32, h, w]` -> `[1, 3, 8h, 8w]`, float32 on CPU."""
        return self._download(self._decode_device(self._upload(latent)))

    def run_reconstruct(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`[1, 3, H, W]` -> `[1, 3, H, W]`, float32 on CPU. The model's own forward."""
        return self._download(self._reconstruct_device(self._upload(pixel_values)))

    # ----------------------------------------------------------------------
    # Gate 2 surface
    # ----------------------------------------------------------------------

    def invoked_modules(self) -> dict:
        """`{component: times its graduated `__call__` ran}` since the last reset."""
        return dict(self._ledger)

    def reset_invocations(self) -> None:
        self._ledger.clear()

    def repeated_block_counts(self) -> dict:
        """Structural proof that the `layers` knob is not inert.

        `*_resnets` counts every resnet actually BUILT in that stack (down/up
        blocks plus the mid block, which is never capped). `*_down_blocks` /
        `*_up_blocks` are the block ladder, which `layers` deliberately does
        not touch -- dropping one would change the output shape.
        """
        encoder_obj = self.E.inner
        decoder_obj = self.D.inner
        encode_resnets = sum(len(_resnets_of(b)) for b in encoder_obj.down_blocks)
        encode_resnets += len(_resnets_of(encoder_obj.mid_block))
        decode_resnets = sum(len(_resnets_of(b)) for b in decoder_obj.up_blocks)
        decode_resnets += len(_resnets_of(decoder_obj.mid_block))
        return {
            "encode_resnets": encode_resnets,
            "decode_resnets": decode_resnets,
            "encode_down_blocks": len(encoder_obj.down_blocks),
            "decode_up_blocks": len(decoder_obj.up_blocks),
        }

    # ----------------------------------------------------------------------
    # Trace contract
    # ----------------------------------------------------------------------

    def _stage_capacity(self, stage):
        return self.image_size if stage == "encode" else self.latent_size

    def _repin(self, stage, size):
        """Re-pin both capacities from one stage's spatial size."""
        if stage == "encode":
            self.image_size = int(size)
            self.latent_size = self.image_size // _VAE_SCALE_FACTOR
        else:
            self.latent_size = int(size)
            self.image_size = self.latent_size * _VAE_SCALE_FACTOR

    def _trace_setup(self, stage, inputs):
        """Pin the stage's shape, stage its persistent input buffer, warm up."""
        if not torch.is_tensor(inputs):
            raise TypeError(
                f"{stage}_trace_setup expects the stage's torch input tensor, got " f"{type(inputs).__name__}"
            )
        if inputs.ndim != 4:
            raise ValueError(f"{stage}_trace_setup expects a rank-4 NCHW tensor, got {tuple(inputs.shape)}")

        real_h, real_w = int(inputs.shape[-2]), int(inputs.shape[-1])
        real = max(real_h, real_w)
        capacity = self._stage_capacity(stage)

        if real > capacity:
            # Honest re-pin rather than a silent crop: the caller wants a
            # bigger picture than the build was sized for.
            print(
                f"[flux2-vae] trace {stage}: input {real_h}x{real_w} exceeds the pinned "
                f"capacity {capacity}; RE-PINNING the capacity to {real}. Every later "
                f"setup/step for this pipeline now runs at that size.",
                flush=True,
            )
            self._repin(stage, real)
            capacity = self._stage_capacity(stage)

        if real_h < capacity or real_w < capacity:
            print(
                f"[flux2-vae] trace {stage}: FALLBACK -- input {real_h}x{real_w} is smaller "
                f"than the pinned capacity {capacity}, so the spatial tail is ZERO-PADDED. "
                f"Be aware this is not free: this VAE's GroupNorm reduces over ALL H*W "
                f"positions, so the padded tail DOES move the statistics of the real "
                f"region, and no mask can make it free. The supported path is that "
                f"VaeImageProcessor.preprocess(height={capacity}, width={capacity}) has "
                f"already resized the input to exactly the capacity, so real == capacity.",
                flush=True,
            )
            padded = torch.zeros(
                (int(inputs.shape[0]), int(inputs.shape[1]), capacity, capacity),
                dtype=inputs.dtype,
            )
            padded[:, :, :real_h, :real_w] = inputs
            inputs = padded

        target_shape = (int(inputs.shape[0]), int(inputs.shape[1]), capacity, capacity)

        # ---- the persistent device buffer, created ONCE -------------------
        buffer = self._trace_in.get(stage)
        if buffer is not None and tuple(int(v) for v in buffer.shape) == target_shape:
            try:
                ttnn.copy_host_to_device_tensor(self._upload_host(inputs), buffer)
            except Exception as exc:  # pragma: no cover - depends on the build
                print(
                    f"[flux2-vae] trace {stage}: FALLBACK -- copy_host_to_device_tensor "
                    f"refused this tensor ({exc}); re-allocating the persistent input "
                    f"buffer instead. Any trace captured against the old buffer must be "
                    f"released and re-captured.",
                    flush=True,
                )
                buffer = None
        else:
            buffer = None

        if buffer is None:
            buffer = self._upload(inputs)
            self._trace_in[stage] = buffer

        # ---- warm up OUTSIDE any capture ----------------------------------
        # The first `ttnn.conv2d` call prepares (and re-lays-out) its weights on
        # host and writes them back to the Conv2d object. That is a host op; if
        # it happened inside the capture the trace would be neither replayable
        # nor host-op-free. One eager pass gets every conv past it.
        warm = self._stage_device[stage](buffer)
        assert isinstance(warm, ttnn.Tensor), f"stage `{stage}`: warm-up output is not a ttnn.Tensor"
        return warm

    def encode_trace_setup(self, inputs):
        return self._trace_setup("encode", inputs)

    def decode_trace_setup(self, inputs):
        return self._trace_setup("decode", inputs)

    def _trace_step(self, stage):
        """One host-op-free forward at the pinned shape, reading only the buffer."""
        buffer = self._trace_in.get(stage)
        if buffer is None:
            raise RuntimeError(f"{stage}_trace_setup(...) must run before {stage}_trace_step()")
        out = self._stage_device[stage](buffer)
        assert isinstance(out, ttnn.Tensor), f"stage `{stage}`: trace step output is not a ttnn.Tensor"
        self._trace_out[stage] = out
        return out

    def encode_trace_step(self):
        return self._trace_step("encode")

    def decode_trace_step(self):
        return self._trace_step("decode")

    def encode_trace_inputs(self):
        """Zero-arg. Exactly what `encode_trace_setup` takes: `[1, 3, 224, 224]`."""
        return _captured_stage_input("encode")

    def decode_trace_inputs(self):
        """Zero-arg. Exactly what `decode_trace_setup` takes: `[1, 32, 28, 28]`."""
        return _captured_stage_input("decode")

    def _trace_items(self, stage):
        """Params-weighted mean conv output area for the stage.

        A conv stack retires SPATIAL POSITIONS, not tokens, and different blocks
        see wildly different areas (224^2 at the top of the encoder, 28^2 at the
        bottleneck). Pricing the whole stage at ONE item would understate it by
        four orders of magnitude, and a plain mean over convs would ignore that
        the 28x28 convs carry most of the parameters. So:

            items = sum_c(params_c * H_out_c * W_out_c) / sum_c(params_c)

        which is exactly the number that makes `2 * params * items` equal the
        stage's true MAC count. Derived from `self.hf` at the pinned capacity,
        walking the same capped block ladder that was actually built.
        """
        cap = self._cap[stage]
        if stage == "encode":
            ladder = _encode_conv_ladder(self.hf, self.image_size, cap)
        else:
            ladder = _decode_conv_ladder(self.hf, self.latent_size, cap)

        weighted = 0
        total = 0
        for conv, out_h, out_w in ladder:
            params = _conv_params(conv)
            weighted += params * out_h * out_w
            total += params
        if total == 0:  # pragma: no cover - a stage with no convs cannot happen
            return 1
        return int(round(weighted / total))

    def encode_trace_items(self):
        """Zero-arg."""
        return self._trace_items("encode")

    def decode_trace_items(self):
        """Zero-arg."""
        return self._trace_items("decode")

    # ----------------------------------------------------------------------
    # Selftests
    # ----------------------------------------------------------------------

    def trace_capture_selftest(self, device=None) -> bool:
        """Capture, replay and PCC-check each stage, one at a time.

        Stage traces must not co-reside, so each stage's trace is RELEASED
        before the next is captured -- which is also why
        `recommended_trace_region_size()` sizes for the largest stage rather
        than the sum. On a trace-region overflow the pinned capacity is halved
        (224 -> 112 -> 56), loudly, and the stage is retried; a stage is never
        silently dropped.
        """
        from models.common.utility_functions import comp_pcc

        dev = device if device is not None else self.device
        target = 0.99
        all_ok = True

        for stage in PIPELINE_STAGES:
            inputs = getattr(self, f"{stage}_trace_inputs")()
            floor = 56 if stage == "encode" else 7
            captured = False
            pcc_value = None

            while True:
                capacity = self._stage_capacity(stage)
                staged = inputs
                if int(inputs.shape[-1]) != capacity or int(inputs.shape[-2]) != capacity:
                    # Crop rather than resize: an interpolation would be a torch
                    # compute op, which the TT-only contract forbids here.
                    staged = inputs[:, :, :capacity, :capacity].contiguous()

                trace_id = None
                try:
                    # Setup also warms every conv past its host-side weight prep.
                    self._trace_setup(stage, staged)

                    # The eager reference, read back BEFORE the capture so the
                    # comparison cannot be contaminated by trace buffers.
                    eager = self._stage_device[stage](self._trace_in[stage])
                    eager_torch = self._download(eager)

                    trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
                    self._trace_step(stage)
                    ttnn.end_trace_capture(dev, trace_id, cq_id=0)

                    ttnn.execute_trace(dev, trace_id, cq_id=0, blocking=True)
                    traced_torch = self._download(self._trace_out[stage])

                    ok, pcc_value = comp_pcc(eager_torch, traced_torch, target)
                    captured = True
                    all_ok = all_ok and bool(ok)
                    print(f"trace {stage}: captured=True pcc={pcc_value}", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001 - classify, then decide
                    message = str(exc)
                    overflow = any(
                        marker in message
                        for marker in (
                            "trace_region",
                            "Trace region",
                            "trace region",
                            "TRACE",
                            "Out of Memory",
                            "out of memory",
                            "OOM",
                            "not enough space",
                            "Not enough space",
                        )
                    )
                    if overflow and capacity // 2 >= floor:
                        print(
                            f"[flux2-vae] trace {stage}: FALLBACK -- capture overflowed the "
                            f"trace region at capacity {capacity} ({message.splitlines()[0]}). "
                            f"Shrinking the pinned capacity to {capacity // 2} and retrying. "
                            f"The stage is NOT being dropped.",
                            flush=True,
                        )
                        self._repin(stage, capacity // 2)
                        # The buffer is sized for the old capacity; drop it so
                        # setup allocates a fresh one.
                        self._trace_in.pop(stage, None)
                        continue
                    print(f"trace {stage}: captured=False pcc=None", flush=True)
                    print(f"[flux2-vae] trace {stage}: capture failed:\n{traceback.format_exc()}", flush=True)
                    all_ok = False
                    break
                finally:
                    if trace_id is not None:
                        # Release before the next stage: stage traces must not
                        # co-reside in the region.
                        try:
                            ttnn.release_trace(dev, trace_id)
                        except Exception:  # pragma: no cover
                            print(
                                f"[flux2-vae] trace {stage}: release_trace failed:\n" f"{traceback.format_exc()}",
                                flush=True,
                            )

            if not captured:
                all_ok = False

        return all_ok

    def host_op_selftest(self) -> dict:
        """The authoritative fully-on-device check for all three heads.

        Input ENCODING (building the pixel/latent tensors), the upload and the
        one-time conv warm-up all happen OUTSIDE the observed region --
        `ttnn.from_torch`/`ttnn.to_torch` legitimately fire aten ops, and so
        does `ttnn.conv2d`'s first-call weight preparation. What is observed is
        the model math: `_encode_device`, `_decode_device` and the chained
        `_reconstruct_device`. Any aten op that fires in there is host compute.
        """
        try:
            from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict
        except ImportError as exc:  # pragma: no cover - bring-up-only hook
            raise RuntimeError(
                "host_op_selftest() needs scripts/tt_hw_planner, which ships "
                "with the bring-up tool rather than with this model. It is a "
                "bring-up verification hook, not part of the model path."
            ) from exc

        # ---- input encoding, OUTSIDE the observed region -------------------
        try:
            pixels = self.encode_trace_inputs()
            latent = self.decode_trace_inputs()
        except Exception:
            pixels = torch.zeros(1, 3, self.image_size, self.image_size)
            latent = torch.zeros(1, self.latent_channels, self.latent_size, self.latent_size)
        capacity, latent_capacity = self.image_size, self.latent_size
        pixels = pixels[:, :, :capacity, :capacity].contiguous()
        latent = latent[:, :, :latent_capacity, :latent_capacity].contiguous()

        x_tt = self._upload(pixels)
        z_tt = self._upload(latent)

        heads = (
            ("encode", self._encode_device, x_tt),
            ("decode", self._decode_device, z_tt),
            ("reconstruct", self._reconstruct_device, x_tt),
        )

        # ---- weight build + conv warm-up, also OUTSIDE ---------------------
        for name, fn, arg in heads:
            warm = fn(arg)
            assert isinstance(warm, ttnn.Tensor), f"head `{name}`: warm-up output is not a ttnn.Tensor"

        # ---- the model math, INSIDE -----------------------------------------
        all_ops = []
        per_head = {}
        for name, fn, arg in heads:
            with observe_host_ops() as ops:
                out = fn(arg)
            assert isinstance(out, ttnn.Tensor), f"head `{name}`: output is not a ttnn.Tensor"
            head_ops = list(ops)
            per_head[name] = verdict(head_ops)
            all_ops.extend(head_ops)

        result = verdict(all_ops)
        result["per_head"] = per_head
        result["heads_on_device"] = {k: v["on_device"] for k, v in per_head.items()}
        return result


# ==========================================================================
# Entry point
# ==========================================================================


def build_pipeline(
    device,
    model=None,
    layers=None,
    encode_layers=None,
    decode_layers=None,
    **kwargs,
) -> TtFluxVaePipeline:
    """CONSTRUCT and return the resident pipeline. It is never run here.

    Args:
        device: an open `ttnn.MeshDevice` (1x8 on T3K). Open it with
            `device_params={"l1_small_size": RECOMMENDED_L1_SMALL_SIZE}` and,
            if you intend to trace, `"trace_region_size":
            recommended_trace_region_size()`.
        model: the HF reference `AutoencoderKLFlux2`. `None` means "load it
            yourself" via `tt/reference.py::load_reference_model()`.
        layers: cap on the depth of every repeated block (the `resnets` inside
            each down/up block). `None` = every layer; `0` clamps to 1.
        encode_layers / decode_layers: per-stack overrides, named after the
            `PIPELINE_STAGES` entries that own each stack. `None` falls back to
            `layers`.
        **kwargs: unrelated demo keywords (`text=`, `prompt=`, `language=`, ...)
            are accepted and ignored, so one demo driver can call every model's
            `build_pipeline` with the same argument bag. `image_size=` IS
            honoured -- it pins the trace capacity at build time.
    """
    if model is None:
        # Imported lazily so this module still imports if `tt/reference.py`
        # is momentarily absent (it is another agent's file).
        from models.demos.flux_2_klein_9b.vae.tt.reference import load_reference_model

        model = load_reference_model()

    image_size = kwargs.pop("image_size", None)
    return TtFluxVaePipeline(
        device,
        model,
        layers=layers,
        encode_layers=encode_layers,
        decode_layers=decode_layers,
        image_size=image_size,
    )


# ==========================================================================
# Standalone selftest hooks -- MODULE level, zero-arg
#
# `scripts/tt_hw_planner/_host_op_probe.py` and `_trace_capture_probe.py`
# import THIS MODULE and call `host_op_selftest()` / `trace_capture_selftest()`
# with no arguments and no device. The methods of the same name on
# `TtFluxVaePipeline` are the real implementations; these two are the
# no-argument front doors to them.
#
# Neither front door may open a mesh in this process. `tt/pipeline.py` runs on
# the `device` handed to `build_pipeline`, and inside pytest that device comes
# from the `mesh_device` fixture, which is the SOLE opener -- a second open
# with a different command-queue count is exactly what breaks trace capture. So
# each hook runs the work in a CHILD PROCESS whose `__main__` owns the mesh:
#
#     python -m models.demos.flux_2_klein_9b.vae.tt.pipeline --selftest host_ops
#     python -m models.demos.flux_2_klein_9b.vae.tt.pipeline --selftest trace
#
# The child prints one `FLUX2_VAE_SELFTEST=<json>` line; the parent parses it.
# `tests/e2e/test_trace_contract.py` does NOT go through this path at all: it
# already owns a device from the fixture and calls the METHODS directly, so the
# two entry points share one implementation and cannot drift.
# ==========================================================================

MODULE_PATH = "models.demos.flux_2_klein_9b.vae.tt.pipeline"
SELFTEST_MARKER = "FLUX2_VAE_SELFTEST="

#: Mesh and capacity the standalone selftests run at. `tp` mirrors the gate
#: tests' `TT_HW_PLANNER_SHARD_TP`; `image_size` is the pinned capture capacity
#: (`_captured/encoder/args.pt` is `[1, 3, 224, 224]`); `timeout` bounds the
#: child so a device hang surfaces as a verdict rather than as a wedged probe.
SELFTEST_DEFAULTS = {
    "tp": 8,
    "image_size": _DEFAULT_IMAGE_SIZE,
    "timeout": 1500,
}

#: The repo root, i.e. the directory that holds `models/`. `_MODELS_ROOT` is
#: `.../models`, so its parent is what has to be importable in the child.
_REPO_ROOT = _MODELS_ROOT.parent


def selftest_config() -> dict:
    """`SELFTEST_DEFAULTS` with env overrides applied.

    `TT_HW_PLANNER_SHARD_TP` is honoured first so a standalone selftest lands on
    the same mesh width the gate tests use, then any
    `TT_FLUX2_VAE_SELFTEST_<KEY>` override wins. `layers` is separate because
    its meaningful default is `None` ("build every layer"), which is not an int:
    the selftests run at FULL depth, since a capped build would answer the
    residency and capture questions about a model that is not the shipped one.
    """
    cfg = dict(SELFTEST_DEFAULTS)
    shard_tp = os.environ.get("TT_HW_PLANNER_SHARD_TP", "").strip()
    if shard_tp:
        cfg["tp"] = int(shard_tp)
    for key in tuple(SELFTEST_DEFAULTS):
        raw = os.environ.get(f"TT_FLUX2_VAE_SELFTEST_{key.upper()}", "").strip()
        if raw:
            cfg[key] = int(raw)
    raw_layers = os.environ.get("TT_FLUX2_VAE_SELFTEST_LAYERS", "").strip()
    cfg["layers"] = int(raw_layers) if raw_layers else None
    return cfg


def _selftest_python() -> str:
    """The interpreter the child runs under: the project venv when there is one."""
    candidate = _REPO_ROOT / "python_env" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _selftest_child(kind: str) -> dict:
    """Run `--selftest <kind>` in a child that owns its own mesh; return its json."""
    cfg = selftest_config()
    env = dict(os.environ)
    env["TT_METAL_HOME"] = env.get("TT_METAL_HOME") or str(_REPO_ROOT)
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        proc = subprocess.run(
            [_selftest_python(), "-m", MODULE_PATH, "--selftest", kind],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=env,
            timeout=cfg["timeout"],
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{kind} selftest exceeded {cfg['timeout']}s"}

    payload = None
    for line in ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines():
        if line.startswith(SELFTEST_MARKER):
            try:
                payload = json.loads(line[len(SELFTEST_MARKER) :])
            except json.JSONDecodeError:
                payload = None
    if payload is None:
        tail = "\n".join(((proc.stdout or "") + (proc.stderr or "")).splitlines()[-12:])
        return {"ok": False, "error": f"{kind} selftest produced no verdict (rc={proc.returncode}):\n{tail}"}
    return payload


def host_op_selftest() -> dict:
    """MODULE-LEVEL, zero-arg. One `host_op_observer.verdict`-shaped dict.

    The hook `scripts/tt_hw_planner/_host_op_probe.py` calls. All three heads --
    `encode`, `decode` and the chained `reconstruct` -- are observed in the child
    and their host-op lists merged, so a host aten op in any one of them fails
    the whole verdict.
    """
    payload = _selftest_child("host_ops")
    result = payload.get("verdict")
    if result is None:
        return {
            "on_device": False,
            "host_ops": [],
            "n_host_ops": 0,
            "reason": payload.get("error", "host-op selftest produced no verdict"),
        }
    return result


def trace_capture_selftest() -> bool:
    """MODULE-LEVEL, zero-arg. True iff every stage captures, replays and PCCs.

    The hook `scripts/tt_hw_planner/_trace_capture_probe.py` calls. Shadows
    nothing: `TtFluxVaePipeline.trace_capture_selftest(device)` is the method on
    the class, this is the module-level function that gets a device for it.
    """
    payload = _selftest_child("trace")
    if not payload.get("ok"):
        print(f"[flux2-vae] trace_capture_selftest: {payload.get('error') or payload}", flush=True)
    return bool(payload.get("ok"))


def _selftest_body(kind, device, cfg, model) -> dict:
    """The child's work, on a device the caller owns and closes."""
    pipe = build_pipeline(device, model=model, layers=cfg["layers"], image_size=cfg["image_size"])
    if kind == "host_ops":
        result = pipe.host_op_selftest()
        return {"ok": bool(result.get("on_device")), "verdict": result}
    if kind == "trace":
        return {"ok": bool(pipe.trace_capture_selftest(device))}
    return {"ok": False, "error": f"unknown selftest kind {kind!r}"}


if __name__ == "__main__":
    # The ONLY device open in this package outside a test fixture or a demo, and
    # it lives HERE, under the `__main__` guard, on purpose: nothing importable
    # in `tt/` may open a device, because the pipeline runs on the `device`
    # passed into `build_pipeline`.
    _argv = sys.argv[1:]
    if len(_argv) != 2 or _argv[0] != "--selftest":
        print(f"usage: python -m {MODULE_PATH} --selftest {{host_ops|trace}}", file=sys.stderr)
        sys.exit(2)

    from models.demos.flux_2_klein_9b.vae.tt.reference import load_reference_model

    _kind = _argv[1]
    _cfg = selftest_config()
    _region = recommended_trace_region_size()
    print(f"[flux2-vae] selftest kind={_kind} cfg={_cfg} trace_region_size={_region}", flush=True)

    _model = load_reference_model()

    # Fabric BEFORE the open, mirroring the `mesh_device` fixture: the stubs'
    # column-parallel all_gathers and the attention all_reduce need it.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    _device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, _cfg["tp"]),
        l1_small_size=RECOMMENDED_L1_SMALL_SIZE,
        trace_region_size=_region,
        num_command_queues=1,
    )
    try:
        _payload = _selftest_body(_kind, _device, _cfg, _model)
    except Exception:  # noqa: BLE001 - the parent gets the traceback as the verdict
        _payload = {"ok": False, "error": traceback.format_exc()}
    finally:
        try:
            for _submesh in _device.get_submeshes():
                ttnn.close_mesh_device(_submesh)
        except Exception:  # pragma: no cover - not every build has submeshes
            pass
        ttnn.close_mesh_device(_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print(SELFTEST_MARKER + json.dumps(_payload), flush=True)
    sys.exit(0 if _payload.get("ok") else 1)
