# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step 1.5 Oobleck (Stable-Audio) VAE decoder — Block 3.

Maps a 25 Hz latent ``[1, 64, T]`` to 48 kHz stereo audio ``[1, 2, 1920*T]``.
84,414,082 parameters, 37 convs, 36 Snakes, no normalisation, no attention,
``groups=1`` everywhere.

Architecture (measured from ``diffusers.OobleckDecoder(channels=128,
input_channels=64, audio_channels=2, upsampling_ratios=[10,6,4,4,2],
channel_multiples=[1,2,4,8,16])``)::

    conv1              Conv1d   64 -> 2048  k=7  pad=3
    block.0  conv_t1   ConvT  2048 -> 1024  k=20 s=10 pad=5   + 3x res_unit @1024
    block.1  conv_t1   ConvT  1024 ->  512  k=12 s=6  pad=3   + 3x res_unit @512
    block.2  conv_t1   ConvT   512 ->  256  k=8  s=4  pad=2   + 3x res_unit @256
    block.3  conv_t1   ConvT   256 ->  128  k=8  s=4  pad=2   + 3x res_unit @128
    block.4  conv_t1   ConvT   128 ->  128  k=4  s=2  pad=1   + 3x res_unit @128
    snake1 + conv2     Conv1d  128 ->    2  k=7  pad=3  (bias=False)

    res_unit_j = Snake -> Conv1d(k=7, dilation=d_j, pad=3*d_j)
                       -> Snake -> Conv1d(k=1) -> + residual,   d = 1, 3, 9

Layout
------
``tt_dit.layers.audio_ops`` works on channels-**last** ``(B, T, C)`` ROW_MAJOR
(asserted inside the primitives); the torch reference is channels-first
``(B, C, T)``. The transpose happens exactly twice, at the host/device boundary
(``_host_to_device`` / ``_device_to_host``). Everything between is ``(B, T, C)``.
``SnakeBeta`` consumes/returns TILE, so ``_snake()`` converts back to ROW_MAJOR
before the next conv (same pattern as ``audio_resample.Activation1d.forward``).

Precision — fp32 activations, non-negotiable
--------------------------------------------
Three independent findings converge:

* upstream ACE-Step hit **fp16 overflow inside Snake** for ``alpha > ~11`` and
  defaults its MLX VAE to fp32;
* ``tt_dit/models/audio_vae/vocoder_ltx.py`` states fp32 is mandatory because
  bf16 accumulation degrades spectral metrics through a long conv chain;
* the prior XTTS-v2 bringup measured a stacked bf16 1-D conv vocoder capping
  waveform PCC at ~0.96, needing fp32 + ``l1_small_size=65536`` to reach 0.998.

Open the device with ``l1_small_size=65536``.

Reuse
-----
Every primitive is taken from ``tt_dit`` (hand-tuned, verified standalone on a
1x1 mesh with ``parallel_config=None, ccl_manager=None``):
``_AlignedOutConv1d`` (dilation *is* supported: ``eff_k = (k-1)*d + 1`` and
``internal_padding = eff_k // 2`` gives exactly ``3*d`` for ``k=7``),
``ConvTranspose1dViaConv3d``, ``SnakeBeta`` (fused ``ttnn.snake_beta``), and the
``forward`` / ``_host_to_device`` / ``_forward_device`` / ``_device_to_host``
split plus ``@traced_function`` pattern from ``vocoder_ltx.py``.

Traps handled — see ``ACE_STEP_1_5_BUGS.md``
--------------------------------------------
**TRAP-3** ``tt_dit.utils.conv3d._FP32_BLOCKINGS`` is a closed LTX-only table;
28 of our 37 convs miss it and a miss on the fp32 path silently falls back to
``T_out_block=1`` with no warning. ``register_conv3d_configs()`` cannot fix this
(it only writes ``_DEFAULT_BLOCKINGS``, which the fp32 branch never reads). We
therefore overwrite ``conv_config`` on every conv module after construction from
``CONV3D_FP32_BLOCKINGS`` below. Must happen *before* ``load_torch_state_dict``,
because ``prepare_conv3d_weights`` reshapes the weight by ``C_in_block``.

**TRAP-4** ``ConvTranspose1dViaConv3d`` hardcodes ``padding = floor(stride/2)``
with no kwarg; ACE-Step wants ``ceil(stride/2)``. Our strides are
``[10, 6, 4, 4, 2]`` — all even — so ``floor == ceil`` and the module is correct
*by luck*. ``assert all(s % 2 == 0)`` in ``OobleckDecoder.__init__`` makes a
future odd stride fail loudly. (Measured in float64: even strides match
``ConvTranspose1d`` to 2.2e-16; ``s=3`` returns ``T*s + 1`` samples where the
reference returns ``T*s - 1`` — two samples too long, silently.)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch
from loguru import logger

import ttnn

from models.tt_dit.layers.audio_ops import (
    ConvTranspose1dViaConv3d,
    SnakeBeta,
    _AlignedOutConv1d,
    _pick_c_out_block_shard,
)
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.utils.conv3d import _walk_conv3d_modules
from models.tt_dit.utils.tracing import traced_function

# ---------------------------------------------------------------------------
# Configuration — the deployed ACE-Step 1.5 audio VAE (verified against the
# live reference; do not "simplify" these into a loop over a shorter list).
# ---------------------------------------------------------------------------

DECODER_CHANNELS = 128
DECODER_INPUT_CHANNELS = 64  # latent channels
AUDIO_CHANNELS = 2  # stereo
CHANNEL_MULTIPLES = (1, 2, 4, 8, 16)
UPSAMPLING_RATIOS = (10, 6, 4, 4, 2)
RES_DILATIONS = (1, 3, 9)
RES_KERNEL = 7
CONV_KERNEL = 7
TOTAL_UPSAMPLE = 1920  # prod(UPSAMPLING_RATIOS); 25 Hz latent -> 48 kHz audio
SAMPLING_RATE = 48000

NUM_CONVS = 37
NUM_SNAKES = 36
NUM_PARAMS = 84_414_082
NUM_FOLDED_STATE_TENSORS = 145  # 37 weights + 36 biases (conv2 has none) + 72 Snake alpha/beta

# Chunked decode defaults (overlap-DISCARD, not cross-fade; chunks are fully
# independent and carry no state). Receptive field per block is 3+9+27 = 39
# latent taps each side, so 64 frames of context is generous.
CHUNK_FRAMES = 512
CHUNK_OVERLAP = 64
CHUNK_STRIDE = CHUNK_FRAMES - 2 * CHUNK_OVERLAP  # 384
MIN_CHUNK_OVERLAP = 4  # adaptive-shrink floor
_WINDOW_ALIGN = 32  # keep every decode window a multiple of T_out_block

# ---------------------------------------------------------------------------
# TRAP-3: hand-built fp32 conv3d blockings, one entry per distinct conv shape.
#
# Key is ``(C_in_aligned, C_out_aligned, (k, 1, 1))`` — identical to
# ``_FP32_BLOCKINGS``'s key, so the 9 shapes that *do* hit the upstream table
# are overridden too. That is deliberate: the two upstream hits
# ``(256,256,(7,1,1)) -> T_out_block=3`` and ``(128,128,(7,1,1)) -> T_out_block=2``
# were swept for the LTX BWE vocoder's short T; our block.2/3/4 residual units
# run at ``240*T``/``960*T``/``1920*T`` frames, where a 2-3 frame M dim is as bad
# as the 1-frame default.
#
# Value is ``(C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)``.
# H/W are always 1 for a 1-D conv (kernel ``(k,1,1)`` on ``(B,T,1,1,C)``).
#
# Sizing rules (documented so the table stays tunable):
#
# 1. ``T_out_block = 32`` everywhere. The conv3d matmul's M dim is
#    ``T_out_block * H_out_block * W_out_block``; 32 fills exactly one 32-row
#    tile. This is the specific failure the ``T_out_block=1`` fallback causes —
#    upstream's own comment: "collapses the matmul M dim to a single 1/32-full
#    tile and reloads the weight once per output frame, so the long-T ups cost
#    ~130/66/15 ms". 32 is also what all five BWE ``ups`` entries use, and those
#    are the closest analogue to our ``conv_t1`` inner convs (same k = 2*stride
#    long-kernel shape on a long T). Going to 64 would double weight reuse but
#    also doubles the input/output L1 blocks; 32 is the safe first value.
#    Every conv here sees ``T_out >= 32`` because decode windows are forced to a
#    multiple of 32 latent frames (``_WINDOW_ALIGN``) and every conv after
#    ``conv1`` runs at >= 10x that.
# 2. ``C_in_block`` divides ``C_in`` and is a multiple of 32; 128 for the wide /
#    long-kernel ``conv_t1`` convs, 256 for the k=7 residual convs, full ``C_in``
#    for the k=1 pointwise convs (whose weight is tiny).
# 3. ``C_out_block`` satisfies conv3d's ``matmul_N_t`` rule (``N_t <= 4`` or
#    ``N_t % 4 == 0``), i.e. one of 32/64/96/128/256/384.
# 4. Budget: ``k * C_in_block * C_out_block * 4 B`` (the fp32 weight block, which
#    is double-buffered) is kept <= 524 KB. Wormhole L1 is 1.44 MB/core and
#    ``l1_small_size`` takes 64 KB of it; the accepted upstream fp32 entries go
#    as high as 786 KB (``(512,256,(12,1,1)) -> (128,128,32,1,1)``), so 524 KB
#    leaves headroom for the input block ``(T_out_block + (eff_k-1)) * C_in_block``
#    and the output block ``T_out_block * C_out_block``.
#
# The trailing comment on each row is the resulting fp32 weight-block size.
# ---------------------------------------------------------------------------

CONV3D_FP32_BLOCKINGS: dict[tuple[int, int, tuple[int, int, int]], tuple[int, int, int, int, int]] = {
    (64, 2048, (7, 1, 1)): (64, 128, 32, 1, 1),  # conv1                  229 KB
    (2048, 1024, (20, 1, 1)): (128, 32, 32, 1, 1),  # block.0.conv_t1     328 KB
    (1024, 1024, (7, 1, 1)): (256, 64, 32, 1, 1),  # block.0 res k7  x3   459 KB
    (1024, 1024, (1, 1, 1)): (1024, 128, 32, 1, 1),  # block.0 res k1 x3  524 KB
    (1024, 512, (12, 1, 1)): (128, 64, 32, 1, 1),  # block.1.conv_t1      393 KB
    (512, 512, (7, 1, 1)): (256, 64, 32, 1, 1),  # block.1 res k7  x3     459 KB
    (512, 512, (1, 1, 1)): (512, 128, 32, 1, 1),  # block.1 res k1  x3    262 KB
    (512, 256, (8, 1, 1)): (128, 128, 32, 1, 1),  # block.2.conv_t1       524 KB
    (256, 256, (7, 1, 1)): (256, 64, 32, 1, 1),  # block.2 res k7  x3     459 KB
    (256, 256, (1, 1, 1)): (256, 128, 32, 1, 1),  # block.2 res k1  x3    131 KB
    (256, 128, (8, 1, 1)): (128, 128, 32, 1, 1),  # block.3.conv_t1       524 KB
    (128, 128, (7, 1, 1)): (128, 128, 32, 1, 1),  # block.3/4 res k7 x6   459 KB
    (128, 128, (1, 1, 1)): (128, 128, 32, 1, 1),  # block.3/4 res k1 x6    65 KB
    (128, 128, (4, 1, 1)): (128, 128, 32, 1, 1),  # block.4.conv_t1       262 KB
    (128, 32, (7, 1, 1)): (128, 32, 32, 1, 1),  # conv2 (out 2 -> 32)     115 KB
}


def _rebuild_conv3d_config(
    base: ttnn.Conv3dConfig,
    blocking: Sequence[int],
    grid_size: Any,
) -> ttnn.Conv3dConfig:
    """Clone ``base`` with only the blocking replaced (``ttnn.Conv3dConfig`` has no setters).

    ``dilation`` and ``alignment`` are carried over rather than defaulted: today
    ``Conv1dViaConv3d`` passes dilation as a ``ttnn.experimental.conv3d`` argument and
    leaves the config field at ``(1,1,1)``, but silently resetting a field we did not
    intend to touch is exactly the class of bug TRAP-3 is.
    """
    c_in_block, c_out_block, t_out_block, h_out_block, w_out_block = blocking
    return ttnn.Conv3dConfig(
        weights_dtype=base.weights_dtype,
        output_layout=base.output_layout,
        T_out_block=t_out_block,
        W_out_block=w_out_block,
        H_out_block=h_out_block,
        C_out_block=c_out_block,
        C_in_block=c_in_block,
        dilation=tuple(base.dilation),
        alignment=base.alignment,
        compute_with_storage_grid_size=grid_size,
    )


def apply_conv3d_blockings(
    root: Module,
    *,
    table: dict = CONV3D_FP32_BLOCKINGS,
    strict: bool = True,
) -> list[tuple[int, int, tuple[int, int, int]]]:
    """TRAP-3 workaround: overwrite every conv's ``conv_config`` from ``table``.

    Walks the whole module tree (``ConvTranspose1dViaConv3d`` holds its conv as
    a ``.conv`` child, so the walk must recurse) and replaces
    ``conv_config`` — and ``conv_config_shard`` where it exists — with a
    hand-built ``ttnn.Conv3dConfig``.

    MUST be called before ``load_torch_state_dict``: ``prepare_conv3d_weight_state``
    reads ``conv_config.C_in_block`` to reshape the weight, so patching after the
    load would leave a weight prepared for the wrong blocking.

    Returns the list of patched keys. Raises on an unknown shape when ``strict``
    (silently accepting the ``T_out_block=1`` fallback is the entire bug).
    """
    if root.is_loaded():
        msg = (
            "apply_conv3d_blockings() must run before load_torch_state_dict(): "
            "prepare_conv3d_weights reshapes the weight by conv_config.C_in_block"
        )
        raise RuntimeError(msg)

    patched: list[tuple[int, int, tuple[int, int, int]]] = []
    missing: list[tuple[int, int, tuple[int, int, int]]] = []

    for module in _walk_conv3d_modules(root):
        key = (int(module.in_channels), int(module.out_channels), tuple(int(k) for k in module.kernel_size))
        blocking = table.get(key)
        if blocking is None:
            missing.append(key)
            continue
        grid_size = module.mesh_device.compute_with_storage_grid_size()
        shard_cfg = getattr(module, "conv_config_shard", None)
        module.conv_config = _rebuild_conv3d_config(module.conv_config, blocking, grid_size)
        if shard_cfg is not None:
            # Channel-TP path: C_out_block must divide the per-chip shard.
            shard_blocking = list(blocking)
            shard_blocking[1] = _pick_c_out_block_shard(full=blocking[1], shard=module.out_channels_shard)
            module.conv_config_shard = _rebuild_conv3d_config(shard_cfg, shard_blocking, grid_size)
        patched.append(key)

    if missing:
        msg = (
            f"conv3d fp32 blocking table is missing {len(missing)} shape(s): {sorted(set(missing))}. "
            "A miss falls back to T_out_block=1 with NO warning (TRAP-3) — add an entry to "
            "CONV3D_FP32_BLOCKINGS instead of letting it through."
        )
        if strict:
            raise KeyError(msg)
        logger.warning(msg)

    logger.debug(f"patched conv3d fp32 blockings on {len(patched)} convs")
    return patched


# ---------------------------------------------------------------------------
# Weight preparation — weight_norm folding is Block 0's job.
# ---------------------------------------------------------------------------

_BLOCK0_WEIGHTS_MODULE = "models.experimental.ace_step_v15.tt.ttnn_ace_step_weights"
# Accept any of these names so a naming choice in Block 0 does not break us; the
# error below lists what was actually found.
_FOLD_CANDIDATES = (
    "fold_vae_weight_norm",
    "fold_weight_norm",
    "fold_decoder_weight_norm",
    "prepare_vae_state_dict",
    "fold_weight_norms",
)
_WEIGHT_NORM_MARKERS = ("weight_g", "weight_v", "parametrizations.weight.original")


def _resolve_block0_fold() -> Callable[[dict], dict]:
    """Import Block 0's ``weight_norm`` folder. We deliberately do not reimplement it."""
    import importlib

    try:
        mod = importlib.import_module(_BLOCK0_WEIGHTS_MODULE)
    except ImportError as err:  # pragma: no cover - depends on Block 0 landing
        msg = (
            f"weight_norm folding lives in Block 0 ({_BLOCK0_WEIGHTS_MODULE}) and is not importable: {err}. "
            "Either land Block 0 or pass an already-folded state dict."
        )
        raise ImportError(msg) from err

    for name in _FOLD_CANDIDATES:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn

    public = sorted(n for n in vars(mod) if not n.startswith("_") and callable(getattr(mod, n)))
    msg = (
        f"{_BLOCK0_WEIGHTS_MODULE} exposes none of {_FOLD_CANDIDATES}; it has {public}. "
        "Add one of those names (or extend _FOLD_CANDIDATES)."
    )
    raise ImportError(msg)


def prepare_decoder_state_dict(state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, torch.Tensor]:
    """Normalise a checkpoint slice into the state dict ``OobleckDecoder`` expects.

    * strips a leading ``vae.decoder.`` / ``decoder.`` / ``model.decoder.`` prefix;
    * folds ``weight_norm`` via Block 0 when the raw ``weight_g``/``weight_v`` form
      is present (already-folded input passes straight through);
    * asserts the expected tensor count (145 for the decoder — 37 conv weights,
      36 conv biases since ``conv2`` has none, and 72 Snake ``alpha``/``beta``).

    The keys are exactly the reference module paths (``conv1.weight``,
    ``block.0.conv_t1.weight``, ``block.0.res_unit2.snake1.alpha``, ...), which
    the ``tt_dit`` ``_prepare_torch_state`` hooks consume directly:
    ``Conv1dViaConv3d`` wants ``weight``/``bias``, ``ConvTranspose1dViaConv3d``
    wants the ``[in, out, K]`` weight (note the swap vs Conv1d's ``[out, in, K]``)
    and flips/permutes it itself, ``SnakeBeta`` wants ``alpha``/``beta`` and
    exponentiates them (``alpha_logscale=True``) — a ``[1, C, 1]`` or ``[C]``
    shape both work, it reshapes to ``(1, 1, -1)``.
    """
    for prefix in ("vae.decoder.", "model.decoder.", "decoder."):
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            break

    if any(m in k for k in state_dict for m in _WEIGHT_NORM_MARKERS):
        if any("parametrizations.weight.original" in k for k in state_dict):
            logger.warning(
                "state dict uses the NEW torch parametrizations weight_norm form "
                "(parametrizations.weight.original0/1). Upstream ACE-Step converters only recognise the "
                "legacy weight_g/weight_v form and would silently load UNFUSED weights — verify Block 0 "
                "handles this form."
            )
        state_dict = _resolve_block0_fold()(state_dict)
        # Re-strip: a folder may return the full-VAE dict.
        for prefix in ("vae.decoder.", "model.decoder.", "decoder."):
            if any(k.startswith(prefix) for k in state_dict):
                state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
                break

    leftovers = sorted(k for k in state_dict for m in _WEIGHT_NORM_MARKERS if m in k)
    if leftovers:
        msg = f"weight_norm was not folded; {len(leftovers)} raw keys remain, e.g. {leftovers[:4]}"
        raise ValueError(msg)

    if strict and len(state_dict) != NUM_FOLDED_STATE_TENSORS:
        msg = (
            f"expected {NUM_FOLDED_STATE_TENSORS} folded decoder tensors, got {len(state_dict)}. "
            "The full VAE folds 365 -> 291 tensors; the decoder half is 145."
        )
        raise ValueError(msg)

    return state_dict


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def _snake(act: SnakeBeta, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
    """``SnakeBeta`` on ROW_MAJOR ``(B, T, C)``, returning ROW_MAJOR.

    ``SnakeBeta.forward`` tilizes internally and returns TILE; the convs assert
    ROW_MAJOR, so convert back (same as ``audio_resample.Activation1d.forward``).
    We use ``SnakeBeta`` rather than ``Snake`` even though the maths is Snake's,
    because ACE-Step's ``Snake1d`` *does* carry independent ``alpha`` and ``beta``
    (both log-scale) — it is literally SnakeBeta — and ``ttnn.snake_beta`` is one
    fused op against ``Snake``'s six eltwise ops. ``Snake`` would additionally
    fail outright on the ROW_MAJOR tensors the convs emit (it does no layout
    conversion).
    """
    y = act(x_BTC)
    if y.layout != ttnn.ROW_MAJOR_LAYOUT:
        y_rm = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(y)
        return y_rm
    return y


def _noop_record(name: str, tensor: ttnn.Tensor) -> None:
    pass


class OobleckResidualUnit(Module):
    """``Snake -> Conv1d(k=7, dilation=d, pad=3d) -> Snake -> Conv1d(k=1) -> + residual``."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dilation = dilation

        # _AlignedOutConv1d computes internal_padding = eff_k // 2 with
        # eff_k = (k-1)*d + 1 = 6d+1 (odd), so it pads exactly 3d per side —
        # bit-identical to OobleckResidualUnit's pad=((7-1)*d)//2 = 3d. That
        # makes T_out == T_in, so the reference's residual crop
        # (padding = (T_in - T_out)//2) is a no-op and we can add directly.
        eff_k = (RES_KERNEL - 1) * dilation + 1
        assert eff_k // 2 == 3 * dilation, f"dilated 'same' padding mismatch for d={dilation}"

        self.snake1 = SnakeBeta(channels, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype)
        self.conv1 = _AlignedOutConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=RES_KERNEL,
            stride=1,
            dilation=dilation,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=None,
            ccl_manager=None,
        )
        self.snake2 = SnakeBeta(channels, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype)
        self.conv2 = _AlignedOutConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=None,
            ccl_manager=None,
        )

    def forward(self, x_BTC: ttnn.Tensor, record: Callable = _noop_record, tag: str = "") -> ttnn.Tensor:
        """``x_BTC`` stays owned by the caller; the returned tensor is new."""
        h = _snake(self.snake1, x_BTC)
        record(f"{tag}snake1", h)
        c = self.conv1(h)
        ttnn.deallocate(h)
        record(f"{tag}conv1", c)

        h = _snake(self.snake2, c)
        ttnn.deallocate(c)
        record(f"{tag}snake2", h)
        c = self.conv2(h)
        ttnn.deallocate(h)
        record(f"{tag}conv2", c)

        out = ttnn.add(x_BTC, c)
        ttnn.deallocate(c)
        return out


class OobleckDecoderBlock(Module):
    """``Snake -> ConvTranspose1d(k=2s, stride=s) -> 3x OobleckResidualUnit``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # TRAP-4: ConvTranspose1dViaConv3d hardcodes padding = floor(stride/2)
        # (no kwarg); ACE-Step's OobleckDecoderBlock uses ceil(stride/2). Equal
        # only for even strides. Guarded once more here so a per-block override
        # cannot slip past the decoder-level assert.
        assert stride % 2 == 0, (
            f"stride {stride} is odd — ConvTranspose1dViaConv3d would use padding=floor({stride}/2)="
            f"{stride // 2} where ACE-Step wants ceil={math.ceil(stride / 2)}, silently emitting "
            f"stride*T + 1 samples instead of stride*T - 1. See TRAP-4."
        )

        self.snake1 = SnakeBeta(in_channels, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype)
        self.conv_t1 = ConvTranspose1dViaConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=None,
            ccl_manager=None,
        )
        self.res_unit1 = OobleckResidualUnit(out_channels, RES_DILATIONS[0], mesh_device=mesh_device, dtype=dtype)
        self.res_unit2 = OobleckResidualUnit(out_channels, RES_DILATIONS[1], mesh_device=mesh_device, dtype=dtype)
        self.res_unit3 = OobleckResidualUnit(out_channels, RES_DILATIONS[2], mesh_device=mesh_device, dtype=dtype)

    def forward(self, x_BTC: ttnn.Tensor, record: Callable = _noop_record, tag: str = "") -> ttnn.Tensor:
        h = _snake(self.snake1, x_BTC)
        record(f"{tag}snake1", h)
        u = self.conv_t1(h)
        ttnn.deallocate(h)
        record(f"{tag}conv_t1", u)

        for i, unit in enumerate((self.res_unit1, self.res_unit2, self.res_unit3), start=1):
            nxt = unit(u, record, f"{tag}res_unit{i}.")
            ttnn.deallocate(u)
            u = nxt
            record(f"{tag}res_unit{i}", u)
        return u


class OobleckDecoder(Module):
    """ACE-Step 1.5 Oobleck VAE decoder: latent ``[1, 64, T]`` -> audio ``[1, 2, 1920*T]``.

    ``forward()`` takes and returns host ``torch`` tensors (channels-first) and
    owns the two transposes to/from the device's channels-last layout.
    ``decode()`` adds chunking. ``forward_traced()`` / ``decode(traced=True)``
    capture and replay the device graph per input shape.
    """

    def __init__(
        self,
        *,
        channels: int = DECODER_CHANNELS,
        input_channels: int = DECODER_INPUT_CHANNELS,
        audio_channels: int = AUDIO_CHANNELS,
        upsampling_ratios: Sequence[int] = UPSAMPLING_RATIOS,
        channel_multiples: Sequence[int] = CHANNEL_MULTIPLES,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()

        strides = list(upsampling_ratios)
        # TRAP-4 — the load-bearing assert. Everything below is only correct
        # because floor(s/2) == ceil(s/2) for even s.
        assert all(s % 2 == 0 for s in strides), (
            f"upsampling_ratios {strides} contain an odd stride. ConvTranspose1dViaConv3d hardcodes "
            "padding = floor(stride/2) with no kwarg, but ACE-Step's OobleckDecoderBlock uses "
            "ceil(stride/2); they agree only for even strides. An odd stride silently produces an "
            "output two samples too long. See TRAP-4 in ACE_STEP_1_5_BUGS.md."
        )
        if dtype != ttnn.float32:
            logger.warning(
                f"OobleckDecoder built with dtype={dtype}; fp32 is mandatory here (Snake overflows fp16 "
                "for alpha > ~11 and bf16 accumulation caps waveform PCC at ~0.96 through this conv chain)."
            )

        self.mesh_device = mesh_device
        self.dtype = dtype
        self.channels = channels
        self.input_channels = input_channels
        self.audio_channels = audio_channels
        self.upsampling_ratios = strides
        self.total_upsample = math.prod(strides)
        self._record: Callable | None = None

        mults = [1, *list(channel_multiples)]

        self.conv1 = _AlignedOutConv1d(
            in_channels=input_channels,
            out_channels=channels * mults[-1],
            kernel_size=CONV_KERNEL,
            stride=1,
            dilation=1,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=None,
            ccl_manager=None,
        )

        self.block = ModuleList(
            [
                OobleckDecoderBlock(
                    in_channels=channels * mults[len(strides) - i],
                    out_channels=channels * mults[len(strides) - i - 1],
                    stride=stride,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
                for i, stride in enumerate(strides)
            ]
        )

        self.snake1 = SnakeBeta(channels, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype)
        self.conv2 = _AlignedOutConv1d(
            in_channels=channels,
            out_channels=audio_channels,
            kernel_size=CONV_KERNEL,
            stride=1,
            dilation=1,
            padding_mode="zeros",
            bias=False,  # reference: nn.Conv1d(..., bias=False)
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=None,
            ccl_manager=None,
            # audio_channels=2 is 1/16 of a tile; _AlignedOutConv1d pads C_out to
            # 32 and slices [..., :2] back. Too narrow to channel-shard.
            channel_shard_output=False,
        )

    # -- loading ------------------------------------------------------------

    def load_decoder_state_dict(self, state_dict: dict[str, torch.Tensor], *, fold: bool = True) -> None:
        """Patch conv blockings (TRAP-3), then load. Order matters — see ``apply_conv3d_blockings``."""
        apply_conv3d_blockings(self)
        prepared = prepare_decoder_state_dict(state_dict) if fold else dict(state_dict)
        self.load_torch_state_dict(prepared)

    # -- host/device boundary ----------------------------------------------

    def _host_to_device(self, latents: torch.Tensor) -> ttnn.Tensor:
        """``[B, 64, T]`` channels-first torch -> ``(B, T, 64)`` fp32 ROW_MAJOR device tensor."""
        assert latents.dim() == 3, f"expected [B, C, T] latents, got {tuple(latents.shape)}"
        b, c, t = latents.shape
        assert b == 1, f"batch {b} != 1"  # BATCH-1 ASSUMPTION
        assert c == self.input_channels, f"expected {self.input_channels} latent channels, got {c}"
        if t < _WINDOW_ALIGN:
            # conv1 is the only conv running at the latent rate, and the fp32 blocking table
            # fixes T_out_block=32 for it (TRAP-3). Below 32 frames it would see less than one
            # full output block. decode() never does this; a direct forward() can.
            logger.warning(
                f"latent T={t} is below the {_WINDOW_ALIGN}-frame window floor implied by "
                f"CONV3D_FP32_BLOCKINGS' T_out_block=32; conv1 gets a partial output block."
            )
        x_btc = latents.transpose(1, 2).float().contiguous()  # (B, T, C)
        return ttnn.from_torch(x_btc, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)

    def _device_to_host(self, y_dev: ttnn.Tensor) -> torch.Tensor:
        """``(B, T_out, 2)`` ROW_MAJOR device tensor -> ``[B, 2, T_out]`` channels-first torch."""
        y = ttnn.to_torch(ttnn.get_device_tensors(y_dev)[0]).to(torch.float32)
        y = y[..., : self.audio_channels]  # no-op: _AlignedOutConv1d already trimmed
        return y.transpose(-1, -2).contiguous()

    # -- device graph -------------------------------------------------------

    @traced_function(device=lambda self: self.mesh_device, prep_run=True, clone_prep_inputs=True)
    def _forward_device(self, x_dev: ttnn.Tensor) -> ttnn.Tensor:
        """Pure-device graph on ``(B, T, C)`` ROW_MAJOR — fixed-shape in/out, trace-capturable.

        Recording (``self._record``) is only ever set on the untraced debug path;
        it reads intermediates back to host, which a trace cannot contain.
        """
        record = self._record or _noop_record

        x = self.conv1(x_dev)
        record("conv1", x)

        # Index rather than iterate: tt_dit's ModuleList defines __len__/__getitem__ but no
        # __iter__ (vocoder_ltx.py indexes too).
        for i in range(len(self.block)):
            nxt = self.block[i](x, record, f"block.{i}.")
            ttnn.deallocate(x)
            x = nxt
            record(f"block.{i}", x)

        h = _snake(self.snake1, x)
        ttnn.deallocate(x)
        record("snake1", h)

        y = self.conv2(h)
        ttnn.deallocate(h)
        record("conv2", y)
        return y

    def release_trace(self) -> None:
        """Free every captured decode trace (shutdown, or before re-warming)."""
        for tracer in type(self)._forward_device._tracers_keyed.get(self, {}).values():
            tracer.release_trace()

    # -- public API ---------------------------------------------------------

    def forward(
        self,
        latents: torch.Tensor,
        *,
        traced: bool = False,
        record: Callable[[str, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Decode one window in a single pass. ``[B, 64, T]`` -> ``[B, 2, 1920*T]``.

        ``record(name, tensor_BCT)`` receives every stage output (37 convs,
        36 Snakes, and the per-block / per-residual-unit sums) as a host
        ``[B, C, T]`` fp32 torch tensor. Incompatible with ``traced=True``.
        """
        assert not (traced and record is not None), "recording reads back to host; a trace cannot contain that"
        x_dev = self._host_to_device(latents)
        self._record = self._make_device_recorder(record) if record is not None else None
        try:
            if traced:
                y_dev = self._forward_device(x_dev, traced=True, tracer_trace_key=tuple(x_dev.shape))
            else:
                y_dev = self._forward_device(x_dev)
        finally:
            self._record = None
        return self._device_to_host(y_dev)

    def forward_traced(self, latents: torch.Tensor) -> torch.Tensor:
        """``forward`` with trace capture/replay, keyed on the input shape."""
        return self.forward(latents, traced=True)

    def _make_device_recorder(self, sink: Callable[[str, torch.Tensor], None]) -> Callable:
        """Wrap a host sink so ``_forward_device`` can hand it device tensors.

        Converts ``(B, T, C)`` ROW_MAJOR -> host ``[B, C, T]`` immediately, so the
        caller can compute a PCC and drop it instead of holding ~850 MB of
        intermediates (the last blocks are 503 MB fp32 each at a 512-frame chunk).
        """

        def record(name: str, tensor: ttnn.Tensor) -> None:
            host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).to(torch.float32)
            sink(name, host.transpose(-1, -2).contiguous())

        return record

    # -- chunked decode -----------------------------------------------------

    def decode(
        self,
        latents: torch.Tensor,
        *,
        chunked: bool | None = None,
        chunk: int = CHUNK_FRAMES,
        overlap: int = CHUNK_OVERLAP,
        min_overlap: int = MIN_CHUNK_OVERLAP,
        traced: bool = False,
    ) -> torch.Tensor:
        """Chunked decode by **overlap-discard** (no cross-fade, no window, no averaging).

        Each chunk decodes completely independently; the ``overlap`` latent frames
        of context on either side are decoded and then thrown away, and the
        surviving cores are concatenated. This exists purely for memory: the
        largest intermediates are ``[1, 128, 983040]`` = 503 MB fp32 each at
        ``chunk=512``, ~1.5 GB live.

        Overlap adapts down to ``min_overlap`` (floor 4 frames) if a core plus
        both contexts would exceed ``chunk``. Windows are forced to a multiple of
        ``_WINDOW_ALIGN`` = 32 latent frames — extending backwards first, then
        forwards — so ``conv1`` always sees ``T_out >= T_out_block`` and so the
        tail chunk reuses one of at most two trace shapes. Extending a window is
        free: the extra frames are context and get discarded, exactly like the
        nominal overlap.
        """
        b, c, t = latents.shape
        assert b == 1, f"batch {b} != 1"  # BATCH-1 ASSUMPTION
        assert overlap >= min_overlap >= 0
        assert chunk > 2 * overlap, f"chunk {chunk} must exceed 2*overlap {2 * overlap}"

        if chunked is None:
            chunked = t > chunk
        if not chunked:
            return self.forward(latents, traced=traced)

        if t < _WINDOW_ALIGN:
            msg = (
                f"latent T={t} is below the {_WINDOW_ALIGN}-frame window floor; the fp32 conv blocking "
                f"table fixes T_out_block=32, so conv1 would see fewer output frames than one block."
            )
            raise ValueError(msg)

        stride = chunk - 2 * overlap
        pieces: list[torch.Tensor] = []

        for core_start in range(0, t, stride):
            core_end = min(core_start + stride, t)
            ov_l = overlap if core_start > 0 else 0
            ov_r = overlap if core_end < t else 0

            # Adaptive shrink: keep the decoded window inside `chunk`, floor at min_overlap.
            while (core_end - core_start) + ov_l + ov_r > chunk and max(ov_l, ov_r) > min_overlap:
                if ov_l > min_overlap:
                    ov_l -= 1
                if ov_r > min_overlap and (core_end - core_start) + ov_l + ov_r > chunk:
                    ov_r -= 1

            win_start = max(0, core_start - ov_l)
            win_end = min(t, core_end + ov_r)

            # Align the window length to _WINDOW_ALIGN: grow left, then right.
            rem = (win_end - win_start) % _WINDOW_ALIGN
            if rem:
                need = _WINDOW_ALIGN - rem
                take = min(need, win_start)
                win_start -= take
                need -= take
                if need:
                    win_end = min(t, win_end + need)

            wav = self.forward(latents[:, :, win_start:win_end], traced=traced)
            drop_l = (core_start - win_start) * self.total_upsample
            drop_r = (win_end - core_end) * self.total_upsample
            pieces.append(wav[:, :, drop_l : wav.shape[-1] - drop_r] if drop_r else wav[:, :, drop_l:])

        out = torch.cat(pieces, dim=-1)
        expected = t * self.total_upsample
        assert out.shape[-1] == expected, f"chunked decode produced {out.shape[-1]} samples, expected {expected}"
        return out


def build_decoder(
    mesh_device: ttnn.MeshDevice,
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: ttnn.DataType = ttnn.float32,
    fold: bool = True,
) -> OobleckDecoder:
    """Construct, patch conv blockings (TRAP-3) and load in the one correct order."""
    model = OobleckDecoder(mesh_device=mesh_device, dtype=dtype)
    model.load_decoder_state_dict(state_dict, fold=fold)
    return model
