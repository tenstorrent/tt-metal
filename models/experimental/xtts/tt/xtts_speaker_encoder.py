# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 speaker encoder (``ResNetSpeakerEncoder``): log-mel -> 512-d ``g``.

Mirrors ``reference/xtts_speaker_encoder.py``. Everything below is shaped by one fact about
this network: the SE-ResNet-34's 16 blocks move a lot of activation for very little math, so
the convolutions were never the cost — layout, per-channel eltwise and pooling were. Four
choices follow from that, each measured (device time for one mel_len=801 pass, ~8 s of
reference audio, via ``tests/test_speaker_encoder_profile.py``):

* **The body never leaves the flat channels-last TILE form** ``[1, 1, H*W, C]`` in **L1** —
  exactly what ``ttnn.conv2d`` emits and consumes. Untilizing to a ``[N, H, W, C]`` ROW_MAJOR
  view per conv cost ~200us each, the 4D TILE form pads W to a tile *per row* (inflating every
  eltwise op that follows), and a DRAM activation puts ttnn.conv2d on its DRAM path, which
  brackets each conv with a 4D unflatten + re-flatten. The spatial extent travels alongside as
  Python ints, so it stays static/trace-safe.
* **The image is convolved transposed** — ``H=time, W=freq``, not the other way round. In the
  flat form the row above is ``W`` positions away, so a 3x3 conv's halo is ``shard_h + 2*(W+1)``
  rows per core: with W=time the stages duplicate their activation 4.4x/7.8x/14.6x, and worse as
  they downsample. Putting the *short* axis in W costs one transpose of the input mel and
  :func:`_time_major` on every kernel, and is worth ~326us. See
  :meth:`TtResNetSpeakerEncoder.forward`.
* **BatchNorm never runs as a BatchNorm.** It is a per-channel affine at inference
  (``scale = gamma/sqrt(var+eps)``, ``shift = beta - mean*scale``); where a conv feeds BN
  directly (``conv2 -> bn2``, ``downsample``) it folds into that conv's weight and bias and
  costs nothing. Only ``bn1`` cannot fold — coqui's block order puts a relu between conv1
  and bn1 — so the relu rides on conv1 as a fused activation and bn1 becomes a diagonal
  matmul (:func:`_scale_channels`). The SE's channel scaling is one too, but only on the tall
  early stages — past ~200 tile rows a broadcast multiply is cheaper, and the SE runs after
  its stage's downsampling, so it crosses over where the other per-channel affines do not.
  See :meth:`TtSELayer.forward`.
* **A stage stops being sharded once its shard is mostly tile padding**
  (:func:`_stage_memory_config`), which is what the narrowest stage was paying for.
* **bfloat16 throughout; fp32 only for what leaves.** The body is bandwidth-bound, so the
  narrow dtype is nearly free there. The ASP tail was fp32 on the expectation that its
  ``E[x^2] - mu^2`` would lose too much to bf16 cancellation — measured, it does not. The
  cancellation is mild (the variance keeps >=1% of ``E[x^2]`` in the worst channel), and bf16
  costs at most 0.00009 PCC over 12 (mel_len, seed) pairs, and 0.000007 on a real 3.1 s
  reference (cosine similarity 0.9999873 against the fp32 embedding), for 55us. So
  ``TAIL_DTYPE`` is bf16 and only ``fc`` onward stays fp32 (``OUT_DTYPE``) — which is what keeps
  the returned embedding the dtype the vocoder's conditioning matmuls were written against.
  Narrower does not pay, and mostly is not reachable: a bfp8 tail is *slower* (block-float
  conversions at op boundaries) and less accurate; a bfp8 body measures 2478us against 2072 at
  PCC 0.972; bfp8 conv weights are impossible via ttnn.conv2d, whose host weight path requires
  ROW_MAJOR and block-float cannot be; and FP8_E4M3 is ROW_MAJOR-only with no ``pad`` support.
  Math fidelity stays HiFi4: see ``BODY_FIDELITY``.
* **No matmul is left for ttnn to configure.** Every matmul here is small, oddly shaped, or
  both, and ttnn's guess for such a shape is routinely 2-3x off — and worse, a guessed config
  arrives with a guessed *fidelity*, and post-processes fused bias/activation as separate
  110-core ops. Spelling the configs out is worth 271us of the 2887us per pass (161us of it
  inside the matmuls, 110us in bias/relu ops that stop existing), and 46 of the 329 device
  ops. See :func:`_matmul_1d`, :func:`_se_core_grid`, and
  ``tests/test_speaker_encoder_matmul_sweep.py``, which measures the alternatives per site.

Together: 26.9 ms -> 1.79 ms per pass by the profiler's summed DEVICE FW DURATION, or 1.93 ms
as a replayed trace, both at mel_len=801. At PCC 0.9990 against the torch reference (0.9998 on
a real reference clip; the fp32 implementation this replaces scored 0.9994), and >= 0.9985
across mel_len 32..1601. The two device-time metrics do not always rank a change the same way —
see :func:`_global_mean` for one where they disagree outright, and which to believe.

Weights are read from the folded/eval reference module.
"""

from functools import lru_cache

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.tt.xtts_conv import TtConv2d

TILE = 32
BN_EPS = 1e-5
INSTANCENORM_EPS = 1e-5
ASP_EPS = 1e-5

BODY_DTYPE = ttnn.bfloat16
# The attention/ASP working dtype: the [2048, T'] tensors the attention and the statistics
# pooling run on, and the attention weights. bfloat16 -- see TtResNetSpeakerEncoder.forward for
# what that was expected to cost and what it measures. ``fc`` and everything after it stays fp32
# so the embedding this returns keeps the dtype its consumers were written against.
TAIL_DTYPE = ttnn.bfloat16
OUT_DTYPE = ttnn.float32
# Keep the three ASP tail weights (att_w1, att_w2, fc_w) resident in L1 rather than DRAM --
# the profiler's "place input 0 in L1" advice on the tail matmuls, which read their *weight*
# as in0. Worth -0.2% (~4us) and it costs 5.37 MB of permanent residency; see
# TtResNetSpeakerEncoder.__init__ for the measurement and the reason that is affordable.
TAIL_WEIGHTS_L1 = True
# HiFi4, not HiFi2: dropping fidelity costs far more accuracy than it buys time here (the
# body is bandwidth-bound, not math-bound). Measured end-to-end PCC at mel_len=200 —
# convs/affines both HiFi2: 0.967, only one of them HiFi4: 0.986-0.991, both HiFi4: 0.998.
#
# The device-perf report asks for HiFi2 on every matmul in the body ("2x the throughput of
# HiFi4"), and asks for a bigger ``out_subblock``, which means turning ``fp32_dest_acc_en``
# off (fp32 DST holds 4 tiles, not 8 — see :func:`_max_subblock`). Neither pays, because
# fidelity is not what these ops are waiting on. Sweeping both, whole-forward traced replay
# at mel_len=801 against this baseline's 2035us:
#
#     HiFi3 + fp32acc      -1.5us   PCC 0.9989   HiFi3 no fp32acc  -1.5us   PCC 0.9808
#     HiFi2 + fp32acc      -9.0us   PCC 0.9736   HiFi2 no fp32acc -18.7us   PCC 0.9957
#     HiFi4 no fp32acc    -10.2us   PCC 0.9796
#
# Halving the math passes moves the total by 0.9%. That is the measurement that settles it:
# every one of these ops is small enough to be dominated by its fixed cost, not by the FPU
# (the report's own columns agree — the body's matmuls run at 2-38% of DRAM peak and 2-56% of
# FLOP peak, saturating neither). Time here comes from op *count*, not op efficiency.
BODY_FIDELITY = ttnn.MathFidelity.HiFi4
# Height (in tiles) above which :func:`_global_mean` switches to its batched form. Measured
# crossover is ~50 tiles; 64 keeps the cheaper path on ties.
_MEAN_BATCH_MIN_TILES = 64
# Height (in tiles) up to which the SE applies its excitation as a plain broadcast multiply
# rather than a diagonal matmul -- see :meth:`TtSELayer.forward`. Swept per stage: the
# broadcast wins at every C at 192 tile rows (by 5.1/4.3/0.4/0.9us at C=32/64/128/256) and
# has lost at every C by 384, with the per-C crossover between 200 and 380. 192 is the
# largest height that is on the right side of all four.
_SE_BROADCAST_MAX_TILES = 192
RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
SIGMOID = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
LOG = ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)
SQRT = ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT)
CLAMP_ASP_EPS = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, ASP_EPS)  # relu_min(x, e) == clamp(x, min=e)


def _to_tile(t: torch.Tensor, device, dtype=None, memory_config=None) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.float(),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=dtype or TAIL_DTYPE,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


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


def _max_subblock(per_core_M, per_core_N, max_dst=4):
    """Largest ``(out_subblock_h, out_subblock_w)`` that the matmul will accept.

    Each must divide its per_core dim, ``h*w`` must fit DST -- 4 tiles, not 8, because
    fp32 dest accumulate halves it -- and a subblock narrower than the full output row is
    only legal one row tall (``w == per_core_N or h == 1``).

    Raising the cap to 8 by turning ``fp32_dest_acc_en`` off is the profiler's "output
    subblock is small" advice, and it does widen these (layer1 3x1 -> 5x1, layer2 2x2 -> 4x2).
    It is not worth it: see ``BODY_FIDELITY`` for the measurement."""
    cands = [(1, w) for w in range(1, per_core_N + 1) if per_core_N % w == 0]
    cands += [(h, per_core_N) for h in range(2, per_core_M + 1) if per_core_M % h == 0]
    return max((c for c in cands if c[0] * c[1] <= max_dst), key=lambda c: c[0] * c[1], default=(1, 1))


def _largest_divisor(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


@lru_cache(maxsize=None)
def _matmul_1d(per_core_M, per_core_N, in0_block_w, grid_x, grid_y):
    """1D matmul, in0 split over cores by height and in1 multicast to all of them."""
    sub_h, sub_w = _max_subblock(per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        mcast_in0=False,
        gather_in0=False,
        fuse_batch=True,  # required whenever in0 is sharded
        fused_activation=None,
    )


@lru_cache(maxsize=None)
def _matmul_2d(per_core_M, per_core_N, in0_block_w, grid_x, grid_y):
    """2D matmul: ``grid_y`` splits M, ``grid_x`` splits N."""
    sub_h, sub_w = _max_subblock(per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
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

    Note **on the first stage**. That 43us is the broadcast at its worst, 1602 tile rows
    tall; the cost is roughly linear in height and the matmul's is flat, so the two cross at
    ~200-380 tile rows and below that the broadcast wins. Every caller here is above it —
    ``bn1`` and the folded BatchNorms run at full stage height — except the SE scaling, which
    runs after the same stage's downsampling and so switches; see :meth:`TtSELayer.forward`.

    ``core_grid`` is what gets the bias *fused*: ttnn.linear post-processes a bias as a
    separate (43us broadcast) add whenever it has to guess a program config for a non-DRAM
    output, and passing a core grid is what makes it stop guessing. The output reuses the
    input's shard spec, so the next conv still sees an L1 input.

    Better still is not to make it guess at all. While the input is sharded the whole config
    is determined -- ``per_core_M`` is the shard's tile height, and K == N == C for a diagonal
    -- so :func:`_matmul_1d` spells it out. Per op: layer3 9.24us -> 5.52us, layer2
    6.64us -> 4.99us, layer1 unchanged at 5.27us (it is one K-tile wide, so there was nothing
    to choose). ``per_core_M`` is *not* ours to pick: a config wanting a different split needs
    a reshard first, which costs more than the ~2us it would save. Layer4 is left to ttnn --
    it arrives interleaved, and the guess is already the best of the measured configs.
    Measured in ``tests/test_speaker_encoder_matmul_sweep.py``."""
    mem = x.memory_config()
    program_config = None
    if mem.shard_spec is not None:
        channel_tiles = x.shape[-1] // TILE
        program_config = _matmul_1d(
            mem.shard_spec.shape[0] // TILE, channel_tiles, channel_tiles, core_grid.x, core_grid.y
        )
    return ttnn.linear(
        x,
        diag,
        bias=bias,
        compute_kernel_config=compute_config,
        memory_config=mem,
        # ttnn takes one or the other, and an explicit config fuses the bias just as well.
        core_grid=None if program_config is not None else core_grid,
        program_config=program_config,
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


def _time_major(weight):
    """Swap a Conv2d kernel's two spatial axes.

    The body convolves the *transposed* spectrogram — ``[H=time, W=freq]``, not
    ``[H=freq, W=time]`` — and this is the entirety of what that costs on the weight side:
    ``conv2d(x.T, w.T) == conv2d(x, w).T`` for a symmetric padding and equal strides, both true
    of every conv here. See :meth:`TtResNetSpeakerEncoder.forward` for why the transpose is worth
    it. A 1x1 downsample kernel is unchanged by this."""
    return weight.transpose(-1, -2).contiguous()


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


@lru_cache(maxsize=None)
def _mean_block(tiles, ncores):
    """Rows per block for :func:`_global_mean`'s batched form: pick the block count ``N``, then
    ``B = tiles/N`` rows.

    Only exact divisors of ``tiles`` are candidates -- any other B needs a zero-pad, and that
    costs a flat ~12us however few rows it adds (measured over B in 32..2048 on all four stage
    shapes: layer1's exact splits run 18.2/19.6/20.0/22.9/32.8us at N=89/267/534/801/1602,
    while *every* inexact B lands at 30-33us).

    Among those, the two reduces pull opposite ways, so N is a balance and both terms have to
    be in the model:

    * the first reduce runs ``N`` blocks over the grid, so each core sees
      ``ceil(N/ncores) * (tiles/N)`` tiles of height -- this is what collapses when N is small,
      and it is *not* symmetric about ``ncores``: a tile count of ``2*p`` for prime p offers
      N=2, which is 2 cores doing half the stage each;
    * the second gathers those N rows back on the C/32 cores a narrow output gets (one core at
      C=32), so it costs ``ceil(N/TILE)`` tiles -- this is what rules out just maximising N.

    Minimising the sum reproduces the measured optimum at every stage (layer1 N=89/B=576,
    layer2 N=401/B=32, layer3 N=101/B=32) and stays sane where the tile count is prime or
    twice a prime, which "nearest ``ncores``" does not: at mel_len=401 that picked N=2 and
    cost 550us on the pass."""

    def cost(n):
        return -(-n // ncores) * (tiles // n) + -(-n // TILE)

    n = min((d for d in range(1, tiles + 1) if tiles % d == 0), key=lambda d: (cost(d), d))
    return (tiles // n) * TILE


def _global_mean(x, hw, ncores):
    """Mean over the flat spatial dim of ``[1, 1, H*W, C]`` -> ``[1, 1, 1, C]`` (interleaved,
    so the SE's matmuls below stay on the plain — non-sharded — matmul path).

    ``ttnn.mean`` over a single tall dim parallelizes only over the *output* width, i.e. the
    C tile-columns — 1 core at 32 channels, 4 at 128 — so it costs ~0.32us per tile of height
    almost regardless of C (491us on the first stage's 51264 rows). Splitting the height into
    whole blocks first — ``[1, N, B, C]``, a free view when the split is tile-aligned — gives
    the reduce a batch dim to spread over the whole grid instead, for a near-flat 16-23us at
    any height. It pays only above ``_MEAN_BATCH_MIN_TILES`` tiles of height (at 25 tiles plain
    is 9.3-10.2us against 15.9-19.2us batched; at 50 they tie; at 100 it is 32-34us against
    ~19us), and ``_mean_block`` picks B.

    The two reduces are **staged by hand rather than asked for as** ``dim=[1, 2]``. A multi-axis
    ttnn.mean reduces axis-by-axis downwards and wraps any axis above rank-2 in a transpose
    *pair* (generic_reductions.cpp:182-221), but by the time the second transpose runs both the
    axes it swaps are size 1 — it is a no-op that still costs a device op, and a 110-core one.
    Staging skips it and puts each reduce on ttnn's accurate (single-axis SFPU) path instead of
    the fast/approximate one the multi-axis form forces, which is why this is not free: the
    accurate path's premium scales with N, so staging *loses* where N is large (layer1 at B=64,
    N=801: 25.7us against 22.9us) and wins where ``_mean_block`` has kept N near the core count
    (layer1 at B=576, N=89: 16.3us against 18.2us; layer3 at N=101: 23.4 against 26.8). The two
    changes only work together — the block rule on its own measures -0.1%, i.e. nothing.

    **The two device-time metrics disagree about this, and the replayed trace is the one that
    matters.** Staging trades 13 transposes on 110 cores for 13 FillPads on 1-4 cores (the
    accurate path needs the pad region of its non-tile-aligned ``[1, 1, N, C]`` input zeroed,
    which the approximate path folds into its scaler). By the profiler's summed
    DEVICE FW DURATION that is -31.2us of transpose against +59.3us of FillPad, so the pass
    reads 1809.5us -> 1815.8us, +0.35%. By captured-trace replay of the whole forward — no host
    in the loop, which is how this model is actually run — it is *-2.9%*, holding at every one
    of five mel lengths (200/401/512/801/1024) and reproducing to within 1.5us with the variant
    order reversed, so it is not measurement drift. The FW column sums per-op firmware duration
    and evidently does not capture what a grid-wide op costs to start and drain; the honest
    statement is that the two metrics rank this change differently and only the replay figure
    was cross-checked. Do not "fix" a FW-column regression here by reverting to ``dim=[1, 2]``.

    A height that is not a whole number of tiles (layer3's 16x201 at mel_len=801, and whichever
    stages a given mel length lands unaligned) is zero-padded up to one first. That divides by
    the padded height, so the result is scaled back by ``hw_pad/hw``.

    **Tried and rejected: ``ttnn.avg_pool2d``** in place of this whole function. Its global-pool
    fast path (kernel == input spatial -> one ``pool_sum``, generic_pools.cpp:1031-1150) looks
    like a strict win -- five ops become one, and the pad/rescale stop existing. Measured, the
    naive ``batch_size=1`` form is **1781 -> 3846us/pass (2.16x)**: ``pool_sum`` has exactly the
    un-batched parallelization problem described above, landing on ``C/32`` cores (1 at layer1).
    ``batch_size=N`` does recover the batching -- the fast path's own canonicalization *is* the
    ``[1, N, B, C]`` reshape (generic_pools.cpp:1104-1121), and ``ReduceDeviceOperation`` then
    costs **171.0us either way, identical** -- but the whole pass is still **2139us/pass (+20%),
    305 ops against 288**, because ``avg_pool2d`` cannot fuse away the two support jobs this
    chain does and re-implements both on worse primitives (per layer1 block): the *gather* of the
    N partials, ``ttnn.transpose`` 2.1us on 89 cores against its ``reshape`` 15.6us on 3
    (batch_size != 1 forces ``[N,1,1,C] -> [1,1,N,C]``, generic_pools.cpp:1140, on the generic
    tile-reshape path pinned to the 3 *output* tiles); and the *pad-zeroing*, one 4.5us FillPad
    against a 12.3us ROW_MAJOR untilize/tilize round trip. PCC is unaffected throughout
    (0.9987-0.9993 across mel_len 32-1601), so this is purely a speed verdict.

    **Tried and rejected: ``ttnn.experimental.fast_reduce_nc(y, dims=[1])``** for the second
    reduce, which consumes ``[1, N, 1, C]`` directly and so appears to delete the transpose *and*
    the FillPad in one op (a tile-wise add leaves each tile's garbage rows 1..31 in the output's
    pad rows, and logical row 0 stays exact). It is **3-8x slower**: 8.66 -> 27.03us at layer1,
    14.32 -> 119.80us at layer2, 10.48 -> 30.78us at layer3, i.e. 146 -> 745us a pass. The
    transpose above is not bookkeeping, it is a **compaction** -- stage 1 leaves each partial
    mean alone in row 0 of its own tile, and the transpose packs N sparse tiles into
    ``ceil(N/32)`` dense ones (89->3, 401->13, 101->4), so the reduce that follows reads 32x
    less. ``fast_reduce_nc`` skips that and reads all N *full* tiles, on the ``C/32`` = 1/2/4
    cores its output tile count buys. Skipping the FillPad is real but worth 4.5us, against a
    32x read amplification. (It also returns logical ``[1,1,32,C]``, not ``[1,1,1,C]`` -- its
    output spec comes from ``padded_shape``, fast_reduce_nc_device_operation.cpp:84 -- which
    breaks the broadcast/diag multiply below; the preallocated ``output=`` arg is the way round
    that, since ``compute_output_specs`` then returns the preallocated spec.)

    **Also tried and reverted: replacing the second reduce (transpose+[FillPad]+reduce) with a
    matmul** against a constant ``[1, N]`` averaging vector (mirroring :func:`_scale_channels`'s
    diagonal-matmul trick). Per-call it looked reasonable and PCC was unaffected, but
    whole-forward traced replay was not monotonic in mel_len: every point <= ~875 won (-0.9% to
    -3.6%) and every point >= ~880 lost (+2.1% to +6.3%), and neither ``N`` nor any per-op FW
    duration explains the flip (a profiler diff at a win and a loss mel_len shows this path's
    own ops -- the new matmul, the reshape it needs -- costing about the same, or less, at the
    losing mel_len; only the replayed metric moves). That means the regression is a scheduling/
    dispatch interaction with the rest of the forward pass, not a cost in the ops this touches,
    and it was not root-caused before deciding it was not worth the added state (a per-shape
    weight cache) and the conditional-correctness risk of shipping something with an unexplained
    failure mode. If revisited, a mel_len-gated version (matmul below ~875, this chain above)
    reproduced the original's exact numbers on the fallback side -- the sweep to redo that:
    ``for mel_len in range(700, 1200, 25): compare traced replay, both variants``."""
    x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)  # de-shard: reduce needs a plain view
    tiles = -(-hw // TILE)
    if tiles < _MEAN_BATCH_MIN_TILES:
        return ttnn.mean(x, dim=2, keepdim=True)
    hw_pad = tiles * TILE
    if hw_pad != hw:
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, hw_pad - hw), (0, 0)], value=0.0)
    block = _mean_block(tiles, ncores)
    mean = ttnn.mean(ttnn.reshape(x, [1, hw_pad // block, block, x.shape[-1]]), dim=2, keepdim=True)
    # ``scalar`` rides the ``hw_pad/hw`` correction on the reduce that already runs, rather than
    # spending a separate op on it: ttnn.mean's scaler is ``scalar / reduced_volume``, and
    # reduced_volume here is exactly the ``N`` this stage divides by, so the two compose to the
    # same arithmetic. A standalone ttnn.multiply cost 3.05us on 110 cores for a one-tile
    # operand, 18.3us a pass over layer3's six blocks -- the only stage where hw is not
    # tile-aligned at mel_len=801. At an aligned hw the ratio is 1.0 and this is a no-op.
    return ttnn.mean(ttnn.transpose(mean, 1, -2), dim=2, keepdim=True, scalar=hw_pad / hw)


def _se_core_grid(out_channels, grid):
    """One core per output tile, for a squeeze-excite bottleneck linear.

    What matters is passing a grid *at all*. Without one ttnn.linear has to guess a program
    config, and then it post-processes the bias and the relu as separate ops on all 110
    cores: three device ops per ``fc[0]`` instead of one fused matmul. Over the 16 SE blocks
    that is 32 bias adds and 16 relus, 110us per pass, gone. Fusing them does make the matmul
    itself slower (an extra CB and a tile read: ``fc[2]`` at C=256 goes 1.93us -> 2.72us), but
    only by a fifth of what the separate ops cost.

    One core per output tile is what the guessed config was already choosing, and it is the
    right amount of hardware: ``fc[0]`` narrows to C/8 <= 32, one tile, so one core, and
    ``fc[2]`` widens back to C. Handing ``fc[2]`` fewer cores than tiles measurably hurts
    (C=256 on 4 cores is 2.94us against 2.72us on 8).

    Pinning the fidelity alongside is not optional: the guessed config came with its own
    default, and inheriting it drops these matmuls to LoFi, which costs ~1% on every
    excitation -- 0.9989 -> 0.9706 end to end, once each has scaled a whole stage's
    channels. Fidelity is free at this size (HiFi4 measures the same as LoFi), so the caller
    pins HiFi4, above even the HiFi2 the guessed path used. Measured in
    ``tests/test_speaker_encoder_matmul_sweep.py``."""
    tiles = max(1, out_channels // TILE)
    x = _largest_divisor(tiles, grid.x)
    return ttnn.CoreGrid(y=min(tiles // x, grid.y), x=x)


class TtSELayer(LightweightModule):
    """Squeeze-excite: global avg-pool -> Linear(C->C/8) -> relu -> Linear -> sigmoid -> scale.

    ``forward`` applies the scaling itself (as the torch reference does) because *how* it is
    applied depends on the activation's height -- see there."""

    def __init__(self, device, se):
        super().__init__()
        # torch Linear weight is [out, in]; ttnn.linear wants [in, out].
        self.w1 = _to_body(se.fc[0].weight.t(), device)
        self.b1 = _to_body(se.fc[0].bias.reshape(1, -1), device)
        self.w2 = _to_body(se.fc[2].weight.t(), device)
        self.b2 = _to_body(se.fc[2].bias.reshape(1, -1), device)
        channels = se.fc[0].weight.shape[1]
        self.eye = _to_body(torch.eye(channels).reshape(1, 1, channels, channels), device)
        grid = device.compute_with_storage_grid_size()
        self.grid1 = _se_core_grid(se.fc[0].weight.shape[0], grid)
        self.grid2 = _se_core_grid(channels, grid)
        self.ncores = grid.x * grid.y  # what _global_mean sizes its block count against
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.compute_config = _body_compute_config(device)

    def forward(self, x, hw):  # x: [1, 1, H*W, C] -> x with its channels scaled
        """Two ways to apply the excitation, and which one wins depends on ``hw``.

        As a **broadcast multiply** ``x * sigmoid(y)``, the ``[1, C]`` operand is re-read once
        per output tile row, so the cost grows with the activation's height: measured
        5.8us at 32 tile rows, 9.4 at 256, 22.2 at 768 (C=32).

        As a **diagonal matmul** ``x @ diag(sigmoid(y))`` (:func:`_scale_channels`) it is flat
        at ~12.7us, and flat across *stages* too -- each stage quarters the spatial extent
        while doubling C, so ``hw * C^2`` is the same at all four and every stage does the
        same work. It costs one extra op to build the diagonal (``eye * y``, ~3.8us, itself
        fixed-cost: the same time for C=32's 2 KB matrix as for C=256's 128 KB).

        So the matmul wins on tall activations and loses on short ones, and the model has
        both: at mel_len=801 the four stages are 1602 / 401 / 100 / 25 tile rows. The
        docstring on :func:`_scale_channels` recorded the broadcast at 43us against 6.6us,
        which is true -- of the *first* stage, the tallest, where it is worst by far. On the
        last two it is the other way round, worth 4.0us at layer3 and 6.4us at layer4 per
        block, and it drops the diagonal-build op entirely (the sigmoid moves onto the linear
        below, where it is free -- fusing it onto the broadcast instead costs 0.8-2.9us,
        presumably re-evaluated per tile row, and a standalone ttnn.sigmoid costs 2.8-5.7us).
        """
        broadcast = hw <= _SE_BROADCAST_MAX_TILES * TILE
        y = _global_mean(x, hw, self.ncores)
        y = ttnn.linear(
            y,
            self.w1,
            bias=self.b1,
            activation="relu",
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.grid1,
            compute_kernel_config=self.compute_config,
        )
        y = ttnn.linear(
            y,
            self.w2,
            bias=self.b2,
            activation="sigmoid" if broadcast else None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.grid2,
            compute_kernel_config=self.compute_config,
        )
        if broadcast:
            return ttnn.mul(x, y, memory_config=x.memory_config())
        diag = ttnn.mul(self.eye, y, input_tensor_b_activations=[SIGMOID], memory_config=ttnn.L1_MEMORY_CONFIG)
        return _scale_channels(x, diag, self.compute_config, self.core_grid)


class TtSEBasicBlock(LightweightModule):
    """conv1 -> relu -> bn1 -> conv2 -> bn2 -> SE -> (+downsample) -> relu. The relus ride on
    the conv / the residual add, bn2 (and the downsample's BN) are folded into their conv's
    weights, bn1 is a diagonal matmul, and the SE applies its own scaling."""

    def __init__(self, device, block, **conv_kwargs):
        super().__init__()
        self.stride = stride = block.stride[0] if isinstance(block.stride, tuple) else block.stride
        self.conv1 = TtConv2d(
            device,
            _time_major(block.conv1.weight.detach()),
            None,
            stride=stride,
            padding=1,
            activation=RELU,
            **conv_kwargs,
        )
        self.bn1 = _ChannelAffine(device, *_bn_scale_shift(block.bn1))
        w2, b2 = _fold_bn(block.conv2.weight.detach(), None, block.bn2)
        self.conv2 = TtConv2d(device, _time_major(w2), b2, stride=1, padding=1, **conv_kwargs)
        self.se = TtSELayer(device, block.se)
        self.compute_config = _body_compute_config(device)
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.ncores = grid.x * grid.y
        self.downsample_conv = None
        if block.downsample is not None:
            wd, bd = _fold_bn(block.downsample[0].weight.detach(), None, block.downsample[1])
            self.downsample_conv = TtConv2d(device, _time_major(wd), bd, stride=stride, padding=0, **conv_kwargs)

    def forward(self, x, h, w):  # h is the TIME extent, w the freq extent -- see _time_major
        # conv2d's output size for kernel 3 / padding 1, so the stage's memory config is
        # known before the conv runs (all Python ints — nothing reads back from device).
        oh, ow = (h - 1) // self.stride + 1, (w - 1) // self.stride + 1
        mem = _stage_memory_config(self.ncores, oh * ow)
        out, oh, ow = self.conv1(x, h, w, mem)  # relu fused
        out = self.bn1(out)
        out, _, _ = self.conv2(out, oh, ow, mem)  # bn2 folded into the weights
        out = self.se(out, oh * ow)  # SE scale, applied inside the SE
        residual = x if self.downsample_conv is None else self.downsample_conv(x, h, w, mem)[0]
        return ttnn.add(out, residual, activations=[RELU]), oh, ow


class TtResNetSpeakerEncoder(LightweightModule):
    """log-mel ``[1, 64, T]`` -> speaker embedding ``[1, 512]`` (L2-normalized)."""

    def __init__(self, device, ref):
        super().__init__()
        self.device = device
        body = {"activations_dtype": BODY_DTYPE, "math_fidelity": BODY_FIDELITY}
        self.conv1 = TtConv2d(
            device,
            _time_major(ref.conv1.weight.detach()),
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
        # The 2048 ASP rows come out ordered ``f*C + c``, not coqui's ``c*F' + f`` -- see
        # ``forward``'s relayout, which is 4x cheaper for it. The attention cannot tell the
        # difference as long as everything that indexes those 2048 rows is reordered to match,
        # which is this fixed permutation: att_w1's columns, att_w2's rows and bias, and (via
        # ``feat``) both halves of fc's columns. All host-side, all exact.
        asp_channels = ref.layer4[-1].conv2.weight.shape[0]  # C after layer4 (256)
        asp_dim = att[0].weight.shape[1]  # C * F' (2048)
        asp_freq = asp_dim // asp_channels  # F' (8)
        asp_perm = torch.tensor([(k % asp_channels) * asp_freq + (k // asp_channels) for k in range(asp_dim)])
        # ``TAIL_WEIGHTS_L1``: the tail computes ``W @ x``, so the *weight* is in0, and the
        # profiler asks for in0 in L1 on all three of these matmuls. Moving them there is worth
        # -1.42us on att_w1, -0.19us on att_w2 and -3.24us on fc, i.e. -0.2% of the pass --
        # small, and confirmed small on both metrics (a traced-replay A/B over four mel lengths
        # gives -0.21%, and -0.17% with the order reversed) at bit-identical PCC.
        #
        # What had to be checked was not the speed but the residency: 5.37 MB is 50 KB per core,
        # and ``xtts_conv._SHARD_L1_BUDGET_BYTES`` (48 KB/core) is a *hardcoded* budget, so
        # ``sharded_chain_fits_l1`` cannot see this and will still claim the vocoder's resblock
        # chain fits. Against per-core L1 it is 3.6% (50 KB of 1395 KB), and the whole traced
        # pipeline -- conditioning, speaker encoder, GPT prefill+decode, HiFi-GAN vocoder, all
        # resident together -- passes ``tests/test_tt_trace.py`` with it on. If L1 ever does get
        # tight, drop fc_w first: it is 4.0 MB of the 5.37 for -3.24us, while att_w1 returns
        # 2.8us/MB against fc_w's 0.81.
        wmem = ttnn.L1_MEMORY_CONFIG if TAIL_WEIGHTS_L1 else None
        self.att_w1 = _to_tile(
            att[0].weight.detach().squeeze(-1)[:, asp_perm], device, memory_config=wmem
        )  # [128, 2048]
        self.att_b1 = _to_tile(att[0].bias.detach().reshape(-1, 1), device)  # [128, 1]
        # The BatchNorm1d between the two attention convs folds into the second one, exactly:
        # it is a per-channel affine and conv2 is linear in that channel, so
        # ``W2 @ (scale*a + shift) + b2 == (W2*scale) @ a + (W2 @ shift + b2)``. (The relu sits
        # *before* the BN here, not between it and conv2, so nothing blocks the fold.) That is
        # one 11us addcmul over the whole [128, T'] attention map that stops existing.
        att_scale, att_shift = _bn_scale_shift(att[2])  # BatchNorm1d(128)
        w2 = att[3].weight.detach().squeeze(-1)[asp_perm]  # [2048, 128], ASP rows reordered
        self.att_w2 = _to_tile(
            w2 * att_scale.reshape(1, -1), device, memory_config=wmem
        )  # scale is per *input* channel
        att_b2 = w2 @ att_shift + att[3].bias.detach()[asp_perm]
        self.att_b2 = _to_tile(att_b2.reshape(-1, 1), device)  # [2048, 1]

        # The one tail weight that is bfloat16, and the only one worth narrowing: at
        # [512, 4096] it is 8 MB of fp32 to read for a matrix-*vector* product, so this matmul
        # is bound by nothing but that read -- 40.3us of the tail's 64.5us once configured.
        # bf16 takes it to 25.9us. It is also the only tail matmul *downstream* of the ASP
        # variance, so narrowing it does not touch the cancellation that wants fp32: 0.9991488
        # -> 0.9991443 end to end. The attention pair stays fp32 -- bf16 there is worth only
        # 5.6us and it feeds the softmax the variance depends on.
        # ``feat`` is concat([mu, sg]), so both halves of fc's 4096 columns take the ASP reorder.
        fc_perm = torch.cat([asp_perm, asp_perm + asp_dim])
        self.fc_w = _to_tile(ref.fc.weight.detach()[:, fc_perm], device, ttnn.bfloat16, wmem)  # [512, 4096]
        self.fc_b = _to_tile(ref.fc.bias.detach().reshape(-1, 1), device, OUT_DTYPE)  # [512, 1]

        # The tail's three matmuls got no program config at all, which left them on ttnn's
        # fallback: 49.5us, 13.8us and 72.0us for att_w1/att_w2/fc, a third of all the matmul
        # time in the model. Spelling the configs out takes them to 15.7us, 8.5us and 40.3us
        # at PCC 1.000000, since none of the fp32 numerics change; fc then goes on to 25.9us
        # on a bf16 weight (see self.fc_w). Measured in
        # ``tests/test_speaker_encoder_matmul_sweep.py``.
        self.tail_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        grid = device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        # fc is fully static -- [512, 4096] @ [4096, 1], so 16 M-tiles split down grid_y and a
        # single output tile wide. grid_y is 10, so this lands on 8 cores at 2 M-tiles each;
        # handing the same tiles to 16 cores one apiece (1D, an 8x2 rectangle, as att_w2 does)
        # is 32.8us against 25.9us -- the extra cores each pay the in1 multicast for a single
        # output tile. The attention pair depends on T' and is built in forward.
        fc_mt, fc_kt = ref.fc.weight.shape[0] // TILE, ref.fc.weight.shape[1] // TILE
        fc_gy = _largest_divisor(fc_mt, grid.y)
        self.fc_config = _matmul_2d(fc_mt // fc_gy, 1, _largest_divisor(fc_kt, 8), 1, fc_gy)

    def forward(self, mel):  # mel: ttnn [1, 64, T] TILE
        _, freq, time = mel.shape
        # log(mel + 1e-6) then InstanceNorm1d over time (per freq) == a plain layer_norm:
        # both normalize the last dim, with no affine. One op instead of a mean/var chain,
        # and it handles the tile padding of a non-tile-aligned T.
        # The log rides on the +1e-6 as a fused activation, so the pair costs one op.
        x = ttnn.layer_norm(ttnn.add(mel, 1e-6, activations=[LOG], dtype=TAIL_DTYPE), epsilon=INSTANCENORM_EPS)
        # -> the conv's flat channels-last form [1, 1, H*W, C] with C=1, straight into L1 for
        # the whole body: every conv then takes ttnn.conv2d's L1 path, which (unlike the DRAM
        # path) doesn't bracket each conv with a 4D unflatten/re-flatten. Asking reshape for that
        # placement is what makes the move free -- landing in DRAM and copying to L1 after costs
        # a separate 17.8us op on this 32x-tile-padded C=1 tensor. Typecast first (cheap while
        # still tiled and 32-wide), and reshape in TILE: the ROW_MAJOR route (untilize -> reshape
        # -> retilize) costs ~5x more, because a flat single-column ROW_MAJOR tensor is one page
        # per element.
        #
        # H=**time**, W=freq -- the image is convolved transposed, and this transpose (plus
        # :func:`_time_major` on every kernel) is all that takes. It is worth ~326us, a fifth of
        # all conv+halo time, and the reason is the halo. ttnn.conv2d gathers, per core, the input
        # window its output rows read from; in this flat layout the row above is W positions away,
        # so a 3x3 kernel makes each core's halo ``shard_h + 2*(W+1)`` rows deep. That predicts
        # the measured halo size to within 1% at every stage, and the ``2*W`` term does not care
        # how much real data there is: with W=time the stages duplicate their activation 4.4x
        # (layer1), 7.8x (layer2), 14.6x (layer3) -- worse as they downsample, because shard_h
        # collapses (480 -> 128 -> 32) while W only halves. With W=freq it is 64/32/16 instead of
        # 801/401/201, so the same halos shrink 3.4x/4.8x/6.7x. Measured conv2d end to end
        # (halo+conv, one stride-1 3x3 per stage): layer1 31.8->28.1us, layer2 28.9->22.9us,
        # layer3 58.1->27.1us, layer4 46.0->46.3us (block-sharded, so its halo was already 2.5us).
        x = ttnn.transpose(x, -2, -1)  # [1, T, 64]
        x = ttnn.reshape(x, [1, 1, time * freq, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

        x, h, w = self.conv1(x, time, freq)  # relu fused; h is the time extent from here on
        x = self.bn1(x)
        for layer in self.layers:
            for block in layer:
                x, h, w = block(x, h, w)  # -> [1, 1, T'*8, 256], row index t*8 + f

        # ASP relayout. The body's flat row index is ``t*F' + f``, so merging the freq extent into
        # the channel dim -- [1, 1, T'*F', C] -> [1, 1, T', F'*C] -> transpose -> [F'*C, T'] --
        # gets there in a reshape and one transpose. Going through coqui's 4D order instead
        # (reshape to [1, T', F', C], permute, re-flatten) costs 122us against 29us, because F'
        # is 8 and the 4D TILE form pads it to a whole tile per time step: 3.2x the bytes through
        # every one of those ops. What it leaves behind is a 2048 dim ordered ``f*C + c``, which
        # __init__ absorbs into the weights that index it.
        c = x.shape[-1]
        # Gather the body's last L1 shards, and keep the whole ASP chain in L1: the three tail
        # matmuls read both operands from DRAM otherwise, and this side of it is transient --
        # it costs no residency and measures -18.8us. Their *weights* are a further 5.37 MB and
        # are L1-resident too, but that one is a residency decision rather than a free one --
        # see ``TAIL_WEIGHTS_L1``.
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, [1, 1, h, w * c])  # [1, 1, T', 2048]
        x = ttnn.transpose(x, -2, -1)  # [1, 1, 2048, T']
        x = ttnn.reshape(x, [w * c, h])  # [2048, T'] -- rank 2, as the matmuls below want
        # fp32 *after* the relayout, not before: the ASP variance below needs it, but widening
        # first doubles the bytes the reshape/permute/reshape trio moves (63us of them) for a
        # cast that is lossless whenever it runs.

        # Attention weights over time. The BatchNorm between the two matmuls is folded into
        # the second one's weight and bias (see __init__), so nothing here applies it.
        #
        # Both matmuls are the same weight size but want opposite configs, because what is
        # scarce differs. att_w1 is [128, 2048] @ [2048, T']: only 4x4 output tiles to hand
        # out, so 2D over exactly those 16 cores. att_w2 is [2048, 128] @ [128, T']: 64
        # M-tiles, so 1D gives every core one of them and the whole of N (2D over the same
        # cores measures 13.3us against 7.7us).
        gx, gy = self.grid
        n_tiles = -(-h // TILE)  # tiles of T'; h is the time extent
        w1_mt, w1_kt = self.att_w1.shape[0] // TILE, self.att_w1.shape[1] // TILE
        w1_gx = _largest_divisor(n_tiles, gx)
        w1_gy = _largest_divisor(w1_mt, gy)
        a = ttnn.matmul(
            self.att_w1,
            x,
            program_config=_matmul_2d(w1_mt // w1_gy, n_tiles // w1_gx, _largest_divisor(w1_kt, 8), w1_gx, w1_gy),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.tail_compute_config,
        )
        a = ttnn.add(a, self.att_b1, activations=[RELU])  # [128, T'], relu fused onto the add
        w2_mt, w2_kt = self.att_w2.shape[0] // TILE, self.att_w2.shape[1] // TILE
        w2_cores = _largest_divisor(w2_mt, gx * gy)
        w2_gx = _largest_divisor(w2_cores, gx)
        a = ttnn.matmul(
            self.att_w2,
            a,
            program_config=_matmul_1d(w2_mt // w2_cores, n_tiles, _largest_divisor(w2_kt, 8), w2_gx, w2_cores // w2_gx),
            # Its [2048, T'] output is 1 MB; keeping it off DRAM is worth 3.6us of the 5.3us.
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.tail_compute_config,
        )
        a = ttnn.add(a, self.att_b2)  # [2048, T']
        wgt = ttnn.softmax(a, dim=-1)  # over time

        # Attentive statistics pooling. ``x^2 * w`` is spelled ``x * (x*w)`` so it reuses the
        # weighted product mu already needs — one full-size [2048, T'] multiply instead of two —
        # and the clamp/sqrt ride on the subtraction as post-activations.
        xw = ttnn.mul(x, wgt)
        mu = ttnn.sum(xw, dim=-1, keepdim=True)  # [2048, 1]
        e2 = ttnn.sum(ttnn.mul(x, xw), dim=-1, keepdim=True)
        sg = ttnn.sub(e2, ttnn.mul(mu, mu), activations=[CLAMP_ASP_EPS, SQRT])
        feat = ttnn.concat([mu, sg], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)  # [4096, 1]

        # ``feat`` stays fp32 and the matmul takes the mixed pair as it is: casting it down to
        # match the weight buys 3.1us inside the matmul and costs 4.9us to do. The output
        # comes back fp32 so the bias and the L2 norm stay there.
        g = ttnn.matmul(
            self.fc_w,
            feat,
            program_config=self.fc_config,
            dtype=OUT_DTYPE,
            # L1 for the output, so the bias add, the reshape and the L2 norm below run there too
            # -- 27us of small ops that were all reading and writing DRAM.
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.tail_compute_config,
        )
        g = ttnn.add(g, self.fc_b)  # [512, 1]
        # Take the [1, 512] row shape *before* the L2 norm, not after. The reshape costs the
        # same either way, but on the column form the reduction is over a dim that is 1 element
        # in a 32-wide tile, so ttnn has to zero the tile padding first (a 9us FillPad) and then
        # reduce on one core; on the row form the 512 dim is tile-aligned and neither happens.
        g = ttnn.reshape(g, [1, 512])
        norm = ttnn.sqrt(ttnn.sum(ttnn.mul(g, g), dim=-1, keepdim=True))
        return ttnn.div(g, norm)
