# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
import re
from typing import List

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import _nearest_y, is_blackhole, is_quasar, is_wormhole_b0, nearest_32
from models.demos.vision.classification.resnet50.quasar.tt.ttnn_functional_resnet50_model_utils import is_blackhole_p100

# --- Per-op fingerprint logging (WH vs Quasar divergence pinpointing) ---------------------------
# Enable with RESNET_PCC_LOG=1. After every value-producing op we log a numeric fingerprint of the
# output tensor, tagged with a stable op NAME (e.g. "layer2_module1.conv2"). Run the model on WH and
# on the Quasar emulator, then diff the two logs BY NAME: the first op whose fingerprint differs is
# where the numerics diverge. Names are used (not the running index) because arch-gated ops -- e.g.
# the WH-only stem tilize/reshards -- shift the index between arches; the name is the stable key.
#
# Optionally set RESNET_PCC_DUMP=<dir> to also save each op's torch output to
# <dir>/op<NNN>_<name>.pt, so exact PCC can be computed offline (dump on WH, load+compare on Quasar).
#
# NOTE: enabling this forces a device->host readback per op (implicit sync), so it perturbs timing --
# use it for numeric comparison, not perf. Disabled by default => zero overhead.
_PCC_OP_IDX = 0

# Optional golden intermediates keyed by op NAME (e.g. "layer2_module1.add" -> torch [NHW, C]).
# When populated (see set_golden_intermediates), _log_op computes and logs the exact PCC of each
# device op against its golden, so the FIRST op whose PCC drops localizes the numeric break --
# device-vs-golden, on a single arch (the fingerprint diff above is arch-vs-arch). Populated by the
# test infra from torch forward hooks; empty => no golden compare (zero overhead).
_GOLDEN = {}


def set_golden_intermediates(d):
    """Install per-op golden torch tensors (name -> torch.Tensor) for _log_op to PCC against."""
    global _GOLDEN
    _GOLDEN = d or {}


def _pcc(dev, gold):
    """Inf/NaN-robust Pearson PCC, sliced to golden's logical [rows, cols].

    Device conv outputs are [1, 1, NHW_pad, C_pad] (NHW/C padded up to tile / channel alignment); the
    golden is the logical [NHW, C]. Reshape both to 2D and slice the device to the golden's extent so
    padding lanes don't dilute the correlation. Non-finite device elements (inf/nan from uninitialized
    L1 or overflow) are MASKED OUT of the correlation and reported separately as `finite_frac` -- so a
    sparse-inf op still yields a meaningful PCC over its real values. Returns (pcc, finite_frac).
    """
    # float64: device garbage often lands at finite-but-huge magnitudes (~1e37) whose square (1e74)
    # overflows float32 in the norm/dot -> spurious nan PCC. double() keeps the reduction finite.
    d = dev.reshape(-1, dev.shape[-1]).double()
    g = gold.reshape(-1, gold.shape[-1]).double()
    r = min(d.shape[0], g.shape[0])
    c = min(d.shape[1], g.shape[1])
    d = d[:r, :c].reshape(-1)
    g = g[:r, :c].reshape(-1)
    # Mask inf/nan AND garbage-magnitude: real resnet activations are << 1e30, so anything larger is
    # same-bug garbage (uninitialized-tile inf/1e37). finite_frac = fraction of REAL (usable) values;
    # the PCC is then computed over just those, so a sparse-garbage op still shows if its math is right.
    good = torch.isfinite(d) & torch.isfinite(g) & (d.abs() < 1e30)
    finite_frac = float(good.double().mean()) if good.numel() else 0.0
    d = d[good]
    g = g[good]
    if d.numel() == 0:
        return float("nan"), finite_frac
    d = d - d.mean()
    g = g - g.mean()
    denom = d.norm() * g.norm()
    if denom == 0:
        return (1.0 if (d.norm() == 0 and g.norm() == 0) else 0.0), finite_frac
    return float((d @ g) / denom), finite_frac


def _reset_op_log():
    global _PCC_OP_IDX
    _PCC_OP_IDX = 0


def _log_op(name, t):
    if os.environ.get("RESNET_PCC_LOG") != "1":
        return t
    global _PCC_OP_IDX
    _PCC_OP_IDX += 1
    idx = _PCC_OP_IDX
    try:
        tt = ttnn.to_torch(t).float()
        f = tt.flatten()
        logger.info(
            f"[PCCLOG] op{idx:03d} {name} shape={tuple(t.shape)} dtype={t.dtype} layout={t.layout} "
            f"mem={t.memory_config().memory_layout} "
            f"mean={f.mean().item():.6f} std={f.std().item():.6f} "
            f"min={f.min().item():.6f} max={f.max().item():.6f} absmean={f.abs().mean().item():.6f} "
            f"nan={int(torch.isnan(f).sum().item())} "
            f"inf={int(torch.isinf(f).sum().item())} posinf={int(torch.isposinf(f).sum().item())} "
            f"first8={[round(v, 4) for v in f[:8].tolist()]}"
        )
        g = _GOLDEN.get(name)
        if g is not None:
            pcc, finite_frac = _pcc(tt, g)
            logger.info(
                f"[GOLDENPCC] op{idx:03d} {name} pcc={pcc:.6f} finite={finite_frac:.6f} "
                f"dev={tuple(tt.shape)} gold={tuple(g.shape)}" + ("  <<< DIVERGES" if not (pcc >= 0.98) else "")
            )
        dump = os.environ.get("RESNET_PCC_DUMP")
        if dump:
            # Sanitize path components before touching the filesystem (path-traversal / SAST): the op
            # `name` becomes a filename, so strip anything that isn't a safe filename char (no path
            # separators, no ".."), and confine the write to the RESNET_PCC_DUMP dir via a realpath
            # containment check so neither the env value nor the name can escape it.
            dump_dir = os.path.realpath(dump)
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            out_path = os.path.realpath(os.path.join(dump_dir, f"op{idx:03d}_{safe_name}.pt"))
            if os.path.commonpath([dump_dir, out_path]) == dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                torch.save(tt, out_path)
    except Exception as e:
        logger.info(f"[PCCLOG] op{idx:03d} {name} <to_torch failed: {type(e).__name__}: {e}>")
    return t


def fit_width_sharded_cores(width_elems, desired_cores, device):
    """Tie a WIDTH_SHARDED core count to the device.

    The model's per-batch grids target a full silicon part; Quasar has at most 32 Tensix neo
    clusters and the emulator 1-2, so a hardcoded grid (e.g. 8x8=64) requests more shards than
    there are L1 banks. Return (num_cores, core_range_set) where num_cores is the largest count
    <= min(desired, device cores) that divides the width into tile-aligned (multiple of 32)
    shards, so the shard width (width_elems // num_cores) stays exact and tile-aligned. On a full
    part where the desired grid already fits this is a no-op.
    """
    grid = device.compute_with_storage_grid_size()
    cap = min(desired_cores, grid.x * grid.y)
    width_tiles = max(1, width_elems // 32)
    num_cores = cap
    while num_cores > 1 and width_tiles % num_cores != 0:
        num_cores -= 1
    return num_cores, ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)


# --- Bring-up host-bypass for convs -------------------------------------------------------------
# The sliced 3x3 convs on the Quasar emulator hit a MATH_PACK program-boundary deadlock in the
# multi-core matmul (LLK-team WIP, ~/conv_stem_sliced.md). To keep exercising the rest of the model,
# any conv can be computed on HOST from its real device input + weights and re-uploaded, so every
# downstream op still runs on device AND stays numerically correct. Flip an entry to True to run that
# conv on the device once the LLK fence lands. (The stem conv1 has its own `on_device` toggle in run().)
# Only the bottleneck conv2 (3x3) slices/deadlocks; conv1/conv3/downsample are 1x1 (mm_conv) and run on
# device. conv1 and conv2 are wired through _conv2d_or_host below (they share the 3-tuple return); to bypass
# the 1x1s too, wrap their calls the same way (conv3 uses return_output_dim=False -> a 2-tuple return).
# Full-model triage: host-bypass ONLY the SLICING convs — the stem (4x4) and bottleneck conv2 (3x3) are the
# ones that route through the split-conv/DRAM-slicing path and hit the tile-counter reuse limitation (#48552).
# All convs run ON DEVICE now that the split-conv path works end-to-end (the reshape_tiled stale-L2 fix
# unblocked the DRAM-slicing 3x3s). Keeping every conv on device also keeps outputs height/block-sharded L1,
# so the residual `add` stays on the ported Metal-V2 binary_ng factory (a host-bypassed conv uploads
# INTERLEAVED DRAM -> the add/next-conv falls to the unported legacy ProgramFactory ->
# "DataMovementKernel not supported on Quasar"). Flip any entry to host-bypass a single conv for debug.
_CONV_ON_DEVICE = {"stem": True, "conv1": True, "conv2": True, "conv3": True, "downsample": True}
# [#48552] The Quasar stem max_pool2d DEADLOCKS at the real stem size (112x112=12544) in compute_pool_2d
# (pool-reduce dest-sync; smsg G semaphore-wait) -- a pre-existing LLK item (repro: test_stem_maxpool.py
# -k 112x112). _MAXPOOL_ON_DEVICE=False computes the 3x3/s2/p1 maxpool on HOST from the device input and
# re-uploads height-sharded ROW_MAJOR (same style as the stem conv1 host fallback), so maxpool doesn't block
# layer1..4 from running on device (lets us validate the layer3 height split end-to-end). Flip back to True
# once the LLK pool-reduce dest-sync hang is fixed.
_MAXPOOL_ON_DEVICE = False


def _host_conv2d(
    input_tensor,
    weight_tensor,
    bias_tensor,
    device,
    conv_config,
    *,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation=(1, 1),
    groups=1,
):
    """Compute a conv on HOST from the real device tensors; return (out, [oh, ow], [w, b]) exactly like
    ttnn.experimental.quasar.conv2d(..., return_output_dim=True, return_weights_and_bias=True) so it is a
    drop-in bypass. Output is uploaded height-sharded ROW_MAJOR (downstream convs reshard/tilize it; the
    quasar RELU activation is folded in when the conv_config sets one)."""
    inp = (
        ttnn.to_torch(input_tensor)
        .float()
        .reshape(batch_size, input_height, input_width, in_channels)
        .permute(0, 3, 1, 2)  # NHWC -> NCHW
    )
    w = ttnn.to_torch(weight_tensor).float().reshape(out_channels, in_channels // groups, *kernel_size)
    b = ttnn.to_torch(bias_tensor).float().reshape(-1)[:out_channels] if bias_tensor is not None else None
    g = torch.nn.functional.conv2d(
        inp, w, bias=b, stride=tuple(stride), padding=tuple(padding), dilation=tuple(dilation), groups=groups
    )
    if getattr(conv_config, "activation", None) is not None:
        g = torch.relu(g)  # model convs use RELU; extend here if other activations are ever configured
    if getattr(conv_config, "deallocate_activation", False):
        ttnn.deallocate(input_tensor)  # mirror the on-device conv's input dealloc so L1 doesn't leak
    oh, ow = int(g.shape[2]), int(g.shape[3])
    flat = g.permute(0, 2, 3, 1).reshape(1, 1, batch_size * oh * ow, out_channels).contiguous()
    # Upload TILE interleaved (DRAM): the downstream conv reshards/consumes it directly. TILE carries the
    # tile-grid padding in padded_shape, so a non-tile-aligned N*oH*oW (e.g. layer2's 28x28=784) does NOT
    # trigger a separate ROW_MAJOR pad op -- that pad routes to the core PadRm* factory whose legacy
    # DataMovementKernel is unported on Quasar ("Use QuasarDataMovementKernel"). (The stem conv1 bypass in
    # run() keeps ROW_MAJOR sharded because its consumer is max_pool2d, which wants that layout; here the
    # consumer is a conv, which prefers TILE and reshards an interleaved input via reshard_if_not_optimal.)
    out = ttnn.from_torch(
        flat.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return out, [oh, ow], [weight_tensor, bias_tensor]


def _conv2d_or_host(
    on_device, name, *, input_tensor, weight_tensor, bias_tensor, compute_config, dtype, **conv2d_kwargs
):
    """Drop-in for the model's ttnn.experimental.quasar.conv2d(...) calls (return_output_dim=True,
    return_weights_and_bias=True). Runs on device when `on_device` else computes on host. `conv2d_kwargs`
    is the per-conv conv_kwargs dict (in_channels/out_channels/batch_size/input_height/input_width/
    kernel_size/stride/padding/dilation/groups/device/conv_config)."""
    if on_device:
        return ttnn.experimental.quasar.conv2d(
            input_tensor=input_tensor,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            compute_config=compute_config,
            dtype=dtype,
            return_output_dim=True,
            return_weights_and_bias=True,
            **conv2d_kwargs,
        )
    logger.warning(f"[QSR] conv '{name}' HOST bypass (on_device=False)")
    return _host_conv2d(
        input_tensor,
        weight_tensor,
        bias_tensor,
        conv2d_kwargs["device"],
        conv2d_kwargs["conv_config"],
        in_channels=conv2d_kwargs["in_channels"],
        out_channels=conv2d_kwargs["out_channels"],
        batch_size=conv2d_kwargs["batch_size"],
        input_height=conv2d_kwargs["input_height"],
        input_width=conv2d_kwargs["input_width"],
        kernel_size=conv2d_kwargs["kernel_size"],
        stride=conv2d_kwargs["stride"],
        padding=conv2d_kwargs["padding"],
        dilation=conv2d_kwargs.get("dilation", (1, 1)),
        groups=conv2d_kwargs.get("groups", 1),
    )


# uint16 DFB ring-extent limit for a bf16 tile (2048B = 128 x 16B units; entry*cap < 65536 -> <512 tiles).
_DFB_RING_LIMIT_TILES = 511


def _no_spill_out_block(per_core_N, in0_block_w):
    """Largest out_block_w that (a) divides per_core_N and (b) keeps the in1 DFB ring
    (out_block_w * in0_block_w tiles, no mcast-depth x2 since num_blocks==1) within the uint16 limit,
    plus a matching out_subblock_w (divides out_block_w, dest holds <=8 bf16 tiles with per_core_M==1)."""
    max_obw = max(1, _DFB_RING_LIMIT_TILES // in0_block_w)
    out_block_w = 1
    for cand in range(min(per_core_N, max_obw), 0, -1):
        if per_core_N % cand == 0:
            out_block_w = cand
            break
    out_subblock_w = out_block_w
    while out_subblock_w > 1 and (out_block_w % out_subblock_w != 0 or out_subblock_w > 8):
        out_subblock_w -= 1
    return out_block_w, out_subblock_w


def fit_fc_grid(device, n_tiles, k_tiles):
    """Pick a rectangular core grid for the resnet fc 1D-mcast matmul that fits the device and
    evenly tiles the N output dimension.

    Returns (grid_x, grid_y, num_cores, per_core_N, in0_block_w). The stock config is an 8x4=32
    grid with per_core_N=1 (N=1024/32=32 tiles, one tile/core) and in0_block_w=2 (K=2048/32=64
    tiles -> 2 tiles/core). On Quasar (32 cores) this is unchanged; on a smaller part (emulator)
    we pick the largest rectangle that fits the device AND divides n_tiles, then raise per_core_N
    so every N tile is still covered (num_cores * per_core_N == n_tiles). A rectangle (not a
    row-wise core set) is required because both the matmul config and the activation width-shard
    feeding it take a (grid_x, grid_y) and must agree.
    """
    if is_quasar():
        # Quasar no-spill fc (Option 1). mcast_in0 width-shards K across the grid, so
        # num_blocks == num_cores; ANY multi-core grid forces num_blocks > 1 -> the interm0/mm_partials
        # K-spill accumulate, which hits the intra-tensix TILE_COUNTERS fault on Quasar (no compute-side
        # implicit-sync opt-out exists). Run the fc on a SINGLE core so the whole K sits on that core and
        # in0_block_w == full K (num_blocks == 1, no spill, interm0 never touched). Shrink out_block_w so
        # the in1 DFB ring (out_block_w * in0_block_w tiles) fits the uint16 ring-extent limit.
        per_core_N = n_tiles
        in0_block_w = k_tiles
        out_block_w, out_subblock_w = _no_spill_out_block(per_core_N, in0_block_w)
        return 1, 1, 1, per_core_N, in0_block_w, out_block_w, out_subblock_w

    grid = device.compute_with_storage_grid_size()
    best_gx, best_gy, best_nc = 1, 1, 1
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            nc = gx * gy
            if n_tiles % nc == 0 and nc > best_nc:
                best_gx, best_gy, best_nc = gx, gy, nc
    per_core_N = n_tiles // best_nc
    kt_per_core = k_tiles // best_nc  # best_nc | n_tiles | k_tiles, so this is exact
    in0_block_w = 2 if kt_per_core % 2 == 0 else kt_per_core
    # WH/BH keep the full per-core N as one output block (out_block_w=None -> ResnetLinear leaves the
    # config's out_block_* to normalize_program_config, i.e. unchanged from before).
    return best_gx, best_gy, best_nc, per_core_N, in0_block_w, None, 1


def ResnetLinear(
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    output_mem_config,
    model_config,
    compute_kernel_config,
    matmul_grid=(8, 4),
    per_core_N=1,
    in0_block_w=2,
    out_block_w=None,
    out_subblock_w=1,
):
    """
    Returns a function for linear operation in resnet with bias.
    """

    if out_block_w is not None:
        # Quasar no-spill config: explicit out_block_h/out_block_w so in0_block_w==full K gives
        # num_blocks==1 (no interm0/mm_partials K-spill) while the in1 ring stays within the uint16 limit.
        matmul_config = ttnn._ttnn.operations.experimental.quasar.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=matmul_grid,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=out_block_w,
            per_core_M=1,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    else:
        matmul_config = ttnn._ttnn.operations.experimental.quasar.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=matmul_grid,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    weight = weight.reshape(weight.shape.to_rank(4))
    bias = bias.reshape(bias.shape.to_rank(4))

    def linear_(act):
        output = ttnn.experimental.quasar.linear(
            act,
            weight,
            bias=bias,
            program_config=matmul_config,
            memory_config=output_mem_config,
            dtype=model_config["ACTIVATIONS_DTYPE"],
            compute_kernel_config=compute_kernel_config,
        )
        return output

    return linear_


class resnet50Bottleneck:
    expansion: int = 4

    def __init__(self, parameters, downsample, stride, model_config) -> None:
        # init is just to pre-process pytorch weights and bias tensors
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 1

        self.conv2_weight_tensor = parameters.conv2.weight
        self.conv2_bias_tensor = parameters.conv2.bias
        self.conv2_input_channels = self.conv2_weight_tensor.shape[1]
        self.conv2_output_channels = self.conv2_weight_tensor.shape[0]
        self.conv2_stride = 2 if downsample else 1
        assert self.conv2_weight_tensor.shape[2] == 3

        self.conv3_weight_tensor = parameters.conv3.weight
        self.conv3_bias_tensor = parameters.conv3.bias
        self.conv3_input_channels = self.conv3_weight_tensor.shape[1]
        self.conv3_output_channels = self.conv3_weight_tensor.shape[0]
        assert self.conv3_weight_tensor.shape[2] == 1

        self.downsample = downsample
        self.stride = stride
        if downsample:
            self.ds_conv_weight_tensor = parameters.downsample.weight
            self.ds_conv_bias_tensor = parameters.downsample.bias
            self.ds_conv_input_channels = self.ds_conv_weight_tensor.shape[1]
            self.ds_conv_output_channels = self.ds_conv_weight_tensor.shape[0]
            assert self.ds_conv_weight_tensor.shape[2] == 1
        self.model_config = model_config
        return

    def run_downsample_if_req(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_accum_enabled=True,
    ):
        if self.downsample:
            logger.debug(f"Running downsample")
            conv_kwargs = {
                "in_channels": self.ds_conv_input_channels,
                "out_channels": self.ds_conv_output_channels,
                "batch_size": batch_size,
                "input_height": input_height,
                "input_width": input_width,
                "kernel_size": (1, 1),
                "stride": (self.stride, self.stride),
                "padding": (0, 0),
                "dilation": (1, 1),
                "groups": 1,
                "device": device,
                "conv_config": ttnn.Conv2dConfig(
                    weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                    shard_layout=(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                        if height_sharding and input_height != 28
                        else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                    ),
                    deallocate_activation=True,
                    # bfloat16 doubles every tensor; mirror the large variant's minimal
                    # downsample config (no double buffering / activation reuse / full
                    # inner dim) and cap the activation block height at one tile so the
                    # CBs fit alongside the pinned residual + the wide projection output.
                    reallocate_halo_output=True,
                    act_block_h_override=32,
                    reshard_if_not_optimal=reshard_if_not_optimal,
                ),
            }

            ds_out, _, [self.ds_conv_weight_tensor, self.ds_conv_bias_tensor] = _conv2d_or_host(
                _CONV_ON_DEVICE["downsample"],
                "downsample",
                input_tensor=x,
                weight_tensor=self.ds_conv_weight_tensor,
                bias_tensor=self.ds_conv_bias_tensor,
                compute_config=ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=self.model_config["MATH_FIDELITY"],
                    packer_l1_acc=packer_l1_accum_enabled,
                ),
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
                **conv_kwargs,
            )
            # Mirror the large variant: free the residual input and defragment the
            # downsample output so the following convs have contiguous L1.
            ttnn.deallocate(x)
            ds_out = ttnn.experimental.quasar.reallocate(ds_out)
        else:
            ds_out = x
        return ds_out

    def __call__(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_acc=True,
        layer_module=None,
    ):
        logger.debug(
            f"==== Running {batch_size}, {input_height}, {input_width}, {self.conv1_input_channels}, {self.conv1_output_channels}"
        )

        ds_input_height = input_height
        ds_input_width = input_width

        # conv1 is 1x1 conv
        logger.debug(f"Running conv1")
        conv_kwargs_1 = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
            ),
        }

        (
            out,
            [input_height, input_width],
            [self.conv1_weight_tensor, self.conv1_bias_tensor],
        ) = _conv2d_or_host(
            _CONV_ON_DEVICE["conv1"],
            f"{layer_module}.conv1",
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            **conv_kwargs_1,
        )
        out = _log_op(f"{layer_module}.conv1", out)

        # bfloat16 doubles every tensor and the residual is pinned through conv2, so the
        # bfloat8_b-tuned act_block_h overflows L1. Cap conv2 at one tile on every arch
        # (one tile divides any per-core height); throughput is not a concern here.
        act_block_h_override = 32

        # Mirror the large resnet50 variant: run the downsample before conv2 for the
        # projection/strided modules. bfloat16 doubles every tensor, so the pinned
        # residual input can no longer co-reside in L1 with conv2's circular buffers.
        # Running the downsample first lets the residual be consumed/freed before
        # conv2. layer1_module1 (input 56, 64 in-channels) keeps the original order.
        run_downsample_before_conv2 = not (ds_input_height == 56 and self.conv1_input_channels == 64)
        ds_out = None
        if run_downsample_before_conv2:
            if ds_input_height == 56 and self.conv1_input_channels == 256:
                # [#48552] Defragment/relocate the pinned residual x before the downsample + conv2 on
                # the 256-channel/56x56 layer1/2 modules, where bf16 doubles every tensor and x's L1
                # shard otherwise clashes with conv2's static DFB region (dataflow_buffer.cpp:1919:
                # L1 buffer @327680 vs DFB region end @404672). Use quasar.reallocate (the move op),
                # NOT a to_layout(RM) round-trip: move deallocates x first via a ghost tensor then
                # reallocates it (move.cpp:42/129), so it compacts in ~1x space; the RM round-trip
                # allocates a full second copy while x is still live and OOMs at layer2_module1
                # (x 448KB + conv1 out 224KB + RM copy 448KB > fragmented free). Must run BEFORE
                # run_downsample_if_req so that for identity blocks (ds_out = x) the residual the
                # end-of-block add consumes aliases the relocated buffer, not the freed one.
                # Numerics-neutral relocate.
                x = ttnn.experimental.quasar.reallocate(x)
            ds_out = self.run_downsample_if_req(
                x,
                device,
                batch_size,
                ds_input_height,
                ds_input_width,
                reshard_if_not_optimal,
                height_sharding,
                packer_l1_accum_enabled=packer_l1_acc,
            )
            ds_out = _log_op(f"{layer_module}.downsample", ds_out)

        logger.debug(f"Running conv2")

        conv_kwargs_2 = {
            "in_channels": self.conv2_input_channels,
            "out_channels": self.conv2_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (3, 3),
            "stride": (self.stride, self.stride),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                deallocate_activation=True,
                reallocate_halo_output=False,
                act_block_h_override=act_block_h_override,
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
                # [#48552] NOTE: the height-sharded 3x "fully buffered weights" inflation (num_blocks_act_h>1,
                # 28 here from act_block_h_override=32, tripling the WEIGHTS CB ~72KB and overrunning L1 for this
                # 3x3 conv2) is now suppressed INSIDE the experimental/quasar conv2d factory itself, so no
                # ttnn.Conv2dConfig flag is needed (and the shared conv2d is left unmodified). It is a
                # DRAM-bandwidth perf opt only; disabling it is numerically identical.
                # bfloat16 doubles every tensor; mirror the large variant's minimal
                # conv2 config (no double buffering / activation reuse / full inner
                # dim) so the CBs fit in L1.
            ),
        }

        (
            out,
            [input_height, input_width],
            [self.conv2_weight_tensor, self.conv2_bias_tensor],
        ) = _conv2d_or_host(
            _CONV_ON_DEVICE["conv2"],
            f"{layer_module}.conv2",
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            bias_tensor=self.conv2_bias_tensor,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            **conv_kwargs_2,
        )
        out = _log_op(f"{layer_module}.conv2", out)

        # conv3 is 1x1 conv
        logger.debug(f"Running conv3")
        conv_kwargs_3 = {
            "in_channels": self.conv3_input_channels,
            "out_channels": self.conv3_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
                deallocate_activation=True,
            ),
        }

        out, _, [self.conv3_weight_tensor, self.conv3_bias_tensor] = _conv2d_or_host(
            _CONV_ON_DEVICE["conv3"],
            f"{layer_module}.conv3",
            input_tensor=out,
            weight_tensor=self.conv3_weight_tensor,
            bias_tensor=self.conv3_bias_tensor,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
            **conv_kwargs_3,
        )
        out = _log_op(f"{layer_module}.conv3", out)

        if not run_downsample_before_conv2:
            ds_out = self.run_downsample_if_req(
                x,
                device,
                batch_size,
                ds_input_height,
                ds_input_width,
                reshard_if_not_optimal,
                height_sharding,
                packer_l1_accum_enabled=packer_l1_acc,
            )
            ds_out = _log_op(f"{layer_module}.downsample", ds_out)

        if ds_out.memory_config() != out.memory_config():
            ds_out = ttnn.experimental.quasar.to_memory_config(ds_out, out.memory_config())

        # underscore version is in_place = True
        out = ttnn.experimental.quasar.add_(
            out,
            ds_out,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        )
        out = _log_op(f"{layer_module}.add", out)
        ttnn.deallocate(ds_out)
        return out, input_height, input_width


class resnet50:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
        input_shape,
        kernel_size,
        stride,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    ) -> None:
        super().__init__()
        layers = [3, 4, 6, 3]
        conv_input_face_shape_hw = [224, 224]
        self.device = device
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.model_config = model_config
        self.inplanes = 64
        self.final_output_mem_config = final_output_mem_config
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=model_config["MATH_FIDELITY"],
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 4

        self.layer1 = self._make_layer(
            parameters=parameters.layer1,
            planes=64,
            blocks=layers[0],
            stride=1,
            model_config=model_config,
        )
        self.layer2 = self._make_layer(
            parameters=parameters.layer2,
            planes=128,
            blocks=layers[1],
            stride=2,
            model_config=model_config,
        )
        self.layer3 = self._make_layer(
            parameters=parameters.layer3,
            planes=256,
            blocks=layers[2],
            stride=2,
            model_config=model_config,
        )
        self.layer4 = self._make_layer(
            parameters=parameters.layer4,
            planes=512,
            blocks=layers[3],
            stride=2,
            model_config=model_config,
        )

        # All modules in RN50 are unrolled here. One variable for each module. Only specific number of modules supported - layers MUST equal to [3, 4, 6, 3]
        assert layers == [3, 4, 6, 3]
        self.layer1_module1 = self.layer1[0]
        self.layer1_module2 = self.layer1[1]
        self.layer1_module3 = self.layer1[2]

        self.layer2_module1 = self.layer2[0]
        self.layer2_module2 = self.layer2[1]
        self.layer2_module3 = self.layer2[2]
        self.layer2_module4 = self.layer2[3]

        self.layer3_module1 = self.layer3[0]
        self.layer3_module2 = self.layer3[1]
        self.layer3_module3 = self.layer3[2]
        self.layer3_module4 = self.layer3[3]
        self.layer3_module5 = self.layer3[4]
        self.layer3_module6 = self.layer3[5]

        self.layer4_module1 = self.layer4[0]
        self.layer4_module2 = self.layer4[1]
        self.layer4_module3 = self.layer4[2]

        # Tie the fc 1D-mcast matmul grid to the device. resnet50 fc: N=1000 -> padded 1024 = 32
        # tiles, K=2048 = 64 tiles. On Quasar (32 cores) this stays the stock 8x4 grid /
        # per_core_N=1; on a smaller part it shrinks the grid and raises per_core_N so all N tiles
        # are covered. The same (grid_x, grid_y) is reused for the activation width-shard feeding
        # fc (see run()), since mcast_in0 requires the input sharding to match the matmul grid.
        fc_gx, fc_gy, self.fc_num_cores, fc_per_core_N, fc_in0_block_w, fc_out_block_w, fc_out_subblock_w = fit_fc_grid(
            device, n_tiles=32, k_tiles=64
        )
        self.fc_matmul_grid = (fc_gx, fc_gy)
        self.fc = ResnetLinear(
            weight=ttnn.experimental.quasar.to_device(parameters.fc.weight, device),
            bias=ttnn.experimental.quasar.to_device(parameters.fc.bias, device),
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config=model_config,
            compute_kernel_config=compute_kernel_config,
            matmul_grid=self.fc_matmul_grid,
            per_core_N=fc_per_core_N,
            in0_block_w=fc_in0_block_w,
            out_block_w=fc_out_block_w,
            out_subblock_w=fc_out_subblock_w,
        )  # num_classes = 1000

        act_block_h_override = 0

        if is_wormhole_b0():
            act_block_h_override = 1568

        if is_blackhole() and self.batch_size == 32:
            act_block_h_override = 32 * 32 if is_blackhole_p100(device) else 49 * 32

        # Mirror the large resnet50 variant's first-conv config: bfloat16 doubles the
        # activation footprint, so activation reuse + double buffering no longer fit in
        # L1. The large variant omits both and relies on reallocate_halo_output instead.
        self.conv1_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=dealloc_input,
            reallocate_halo_output=True,
            act_block_h_override=act_block_h_override,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
        )
        self.conv1_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            packer_l1_acc=True,
        )
        if is_wormhole_b0():
            # Issue #13145: Temp workaround for Galaxy to avoid hangs
            if device.get_num_devices() > 8:
                self.conv1_config.act_block_h_override = 64
            else:
                self.conv1_config.act_block_h_override = 49 * 32

        self.conv1_kernel_size = (4, 4)
        self.conv1_stride = (1, 1)
        self.conv1_padding = (0, 0)
        self.conv1_input_height = 115
        self.conv1_input_width = 115
        self.conv1_output_height = (
            (self.conv1_input_height - self.conv1_kernel_size[0] + 2 * self.conv1_padding[0]) // self.conv1_stride[0]
        ) + 1
        self.conv1_output_width = (
            (self.conv1_input_width - self.conv1_kernel_size[1] + 2 * self.conv1_padding[1]) // self.conv1_stride[1]
        ) + 1

        # fold params
        self.fold_stride_h = stride
        self.fold_stride_w = stride
        _, c, h, w = input_shape
        n = batch_size
        h += kernel_size * 2
        w += kernel_size * 2
        # Quasar aligns fold channels to 8 (bf16 row-major 16B shard-width); the first-conv weights are
        # folded to groups*8 input channels (see custom_preprocessing), so the direct fold's aligned
        # groups*8 output feeds conv1 with no per-group padding strip. WH/BH keep alignment 4.
        C = _nearest_y(c, 8 if is_quasar() else 4)
        self.fold_pad_c = C - c
        self.fold_pad_h = kernel_size
        self.fold_pad_w = kernel_size
        self.fold_output_shape = (
            n,
            h // self.fold_stride_h,
            w // self.fold_stride_w,
            C * (self.fold_stride_h * self.fold_stride_w),
        )
        num_cores_x = 8
        num_cores_y = 8
        # Default grid, used for batch 16 and for any batch not explicitly handled below (e.g. the
        # small batches used on the 2x3 emulator / craq-sim grid). The device-cap clamp further down
        # reduces this to the device's real core count, so leaving fold_compute_grid_size always-set
        # here is what lets resnet run on tiny grids instead of hitting an undefined-attribute error.
        self.fold_compute_grid_size = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
        )
        if self.batch_size == 20:
            if is_wormhole_b0():
                num_cores_x = 8
                num_cores_y = 5
            elif is_blackhole():
                num_cores_x = 10
                num_cores_y = 8
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
            )
        elif self.batch_size == 32:
            core_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 8)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(10, 9)),
                }
            )
            if is_blackhole_p100(device):
                core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
            self.fold_compute_grid_size = core_grid

        # Cap the fold compute grid to the device's real core count. The per-batch grids above target
        # a full silicon part; Quasar has at most 32 Tensix neo clusters and the emulator 1-2, so an
        # 8x8 (=64) fold grid would request more shards than there are L1 banks. Clamp to the device
        # grid (no-op when it already fits) so this matches the (also-capped) input sharding.
        _fold_compute_grid = device.compute_with_storage_grid_size()
        _fold_max_cores = _fold_compute_grid.x * _fold_compute_grid.y
        if self.fold_compute_grid_size.num_cores() > _fold_max_cores:
            self.fold_compute_grid_size = ttnn.num_cores_to_corerangeset(
                _fold_max_cores, _fold_compute_grid, row_wise=True
            )

        conv_dummy_tensor = torch.rand((self.fold_output_shape), dtype=torch.bfloat16)
        conv_dummy_tensor = ttnn.from_torch(conv_dummy_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Create sharded memory config for fold operation
        compute_grid = device.compute_with_storage_grid_size()

        # Calculate core grid
        if is_blackhole():
            # Override num cores to avoid padding issues
            nhw_ntiles = math.ceil(self.batch_size * self.conv1_output_height * self.conv1_output_width / 32)
            # Find closest largest divisor
            num_cores_target = compute_grid.x * compute_grid.y
            while nhw_ntiles % num_cores_target != 0:
                num_cores_target -= 1
            core_grid = ttnn.num_cores_to_corerangeset(num_cores_target, compute_grid, row_wise=True)
        else:
            # Use full grid
            core_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
            )

        # Calculate shard dimensions
        input_channels_padded = (
            nearest_32(self.conv1_input_channels) if self.conv1_input_channels % 8 != 0 else self.conv1_input_channels
        )
        if input_channels_padded % 8 != 0:
            input_channels_padded = ((input_channels_padded + 7) // 8) * 8

        tensor_height = self.conv1_input_width * self.conv1_input_height * self.batch_size
        tensor_width = input_channels_padded

        # Calculate shard shape for HEIGHT sharding
        num_cores = core_grid.num_cores()
        shard_height = math.ceil(tensor_height / num_cores)
        shard_width = tensor_width

        self.override_fold_mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, shard_height, shard_width),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def __del__(self):
        # Nothing to do
        pass

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        model_config=None,
    ) -> List[resnet50Bottleneck]:
        layers = []
        layers.append(
            resnet50Bottleneck(
                parameters=parameters[0],
                downsample=stride != 1 or self.inplanes != planes * resnet50Bottleneck.expansion,
                stride=stride,
                model_config=model_config,
            )
        )
        self.inplanes = planes * resnet50Bottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                resnet50Bottleneck(
                    parameters=parameters[block_num],
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                )
            )
        return layers

    def __call__(self, input_tensor, device, ops_parallel_config) -> ttnn.Tensor:
        return self.run(
            input_tensor,
            device,
        )

    ## merged runs (first and optimized)
    def run(self, input_tensor, device) -> ttnn.Tensor:
        _reset_op_log()
        logger.debug(f"==== fold on device")

        # run fold
        if is_quasar():
            # Direct data-movement fold. Input arrives channels-last (NHWC), host-padded to the aligned
            # width (see setup_l1_sharded_input); the transpose-chain fold has no Quasar kernel. output_shape
            # C == groups*C_aligned (== fold_output_shape[3]) so c_keep == c_aligned -> the fold skips the
            # per-group padding strip and returns the aligned groups*C_aligned width directly, which conv1
            # consumes (its weights are folded to groups*C_aligned input channels with zero pad channels).
            fold_output_tensor = ttnn.experimental.quasar.fold(
                input_tensor,
                self.fold_stride_h,
                self.fold_stride_w,
                use_transpose_as_fold=False,
                padding=[self.fold_pad_h, self.fold_pad_h, self.fold_pad_w, self.fold_pad_w, 0, self.fold_pad_c],
                grid_size=self.fold_compute_grid_size,
                input_is_nhwc=True,
                output_shape=ttnn.Shape(list(self.fold_output_shape)),
            )
        else:
            fold_output_tensor = ttnn.experimental.quasar.fold(
                input_tensor,
                self.fold_stride_h,
                self.fold_stride_w,
                use_transpose_as_fold=True,
                padding=[self.fold_pad_h, self.fold_pad_h, self.fold_pad_w, self.fold_pad_w, 0, self.fold_pad_c],
                grid_size=self.fold_compute_grid_size,
                override_memory_config=self.override_fold_mem_config,
            )
        n, c, h, w = fold_output_tensor.shape
        fold_output_tensor = ttnn.experimental.quasar.reshape(fold_output_tensor, (1, 1, n * c * h, w))
        fold_output_tensor = _log_op("fold", fold_output_tensor)

        ttnn.deallocate(input_tensor)

        logger.debug(f"==== first conv")

        # first conv
        conv_kwargs = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": self.batch_size,
            "input_height": self.conv1_input_height,
            "input_width": self.conv1_input_width,
            "kernel_size": self.conv1_kernel_size,
            "stride": self.conv1_stride,
            "padding": self.conv1_padding,
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": self.conv1_config,
        }

        # Toggle for the stem conv1. on_device=True runs it on the device (currently deadlocks on the Quasar
        # emulator — sliced split-conv, LLK-team WIP, ~/conv_stem_sliced.md). on_device=False computes the
        # folded 4x4/s1/p0 conv on HOST from the REAL device input + weights so maxpool and all downstream
        # layers still run and stay numerically correct. Flip this bool to go back and forth. Restore to True
        # once the LLK MATH_PACK program-boundary fence lands.
        on_device = _CONV_ON_DEVICE["stem"]
        # [#48552] stem_conv1 emits inf in some outputs (poisons maxpool + all downstream -> PCC 0).
        # Log the raw folded weight/bias finiteness so a rerun tells us whether the inf is in the
        # WEIGHTS (preprocessing padding garbage) or the device conv COMPUTE (uninitialized output
        # tiles). Complements the on_device=False host A/B (same fold output + weights, no device conv).
        if os.environ.get("RESNET_PCC_LOG") == "1":
            try:
                _wt = ttnn.to_torch(self.conv1_weight_tensor).float()
                _msg = (
                    f"[STEMW] weight shape={tuple(self.conv1_weight_tensor.shape)} "
                    f"nan={int(torch.isnan(_wt).sum())} inf={int(torch.isinf(_wt).sum())} "
                    f"min={_wt.min().item():.4f} max={_wt.max().item():.4f} absmean={_wt.abs().mean().item():.6f}"
                )
                if self.conv1_bias_tensor is not None:
                    _bt = ttnn.to_torch(self.conv1_bias_tensor).float()
                    _msg += (
                        f" | bias nan={int(torch.isnan(_bt).sum())} inf={int(torch.isinf(_bt).sum())} "
                        f"min={_bt.min().item():.4f} max={_bt.max().item():.4f}"
                    )
                logger.info(_msg)
            except Exception as e:
                logger.info(f"[STEMW] <weight introspection failed: {type(e).__name__}: {e}>")
        if on_device:
            (
                x,
                [x_height, x_width],
                [self.conv1_weight_tensor, self.conv1_bias_tensor],
            ) = ttnn.experimental.quasar.conv2d(
                input_tensor=fold_output_tensor,
                weight_tensor=self.conv1_weight_tensor,
                bias_tensor=self.conv1_bias_tensor,
                **conv_kwargs,
                compute_config=self.conv1_compute_config,
                return_output_dim=True,
                return_weights_and_bias=True,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
        else:
            logger.warning("[QSR] stem conv1 HOST bypass active (on_device=False)")
            _inp = (
                ttnn.to_torch(fold_output_tensor)
                .float()
                .reshape(self.batch_size, self.conv1_input_height, self.conv1_input_width, self.conv1_input_channels)
                .permute(0, 3, 1, 2)  # NHWC -> NCHW [batch, in_ch, 115, 115]
            )
            _w = (
                ttnn.to_torch(self.conv1_weight_tensor)
                .float()
                .reshape(self.conv1_output_channels, self.conv1_input_channels, *self.conv1_kernel_size)
            )
            _b = None
            if self.conv1_bias_tensor is not None:
                _b = ttnn.to_torch(self.conv1_bias_tensor).float().reshape(-1)[: self.conv1_output_channels]
            _g = torch.nn.functional.conv2d(_inp, _w, bias=_b, stride=self.conv1_stride, padding=self.conv1_padding)
            _g = torch.relu(_g)  # conv1_config fuses RELU
            x_height, x_width = int(_g.shape[2]), int(_g.shape[3])
            _nhw = self.batch_size * x_height * x_width
            _flat = _g.permute(0, 2, 3, 1).reshape(1, 1, _nhw, self.conv1_output_channels).contiguous()
            # Upload height-sharded ROW_MAJOR (the layout quasar max_pool2d consumes — mirrors test_stem_maxpool).
            _grid = device.compute_with_storage_grid_size()
            _maxc = _grid.x * _grid.y
            _nc = max(c for c in range(1, _maxc + 1) if _nhw % c == 0)
            _mem = ttnn.create_sharded_memory_config(
                shape=(1, 1, _nhw // _nc, self.conv1_output_channels),
                core_grid=ttnn.num_cores_to_corerangeset(_nc, _grid, True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            x = ttnn.from_torch(
                _flat.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=_mem,
            )
            ttnn.deallocate(fold_output_tensor)
        x = _log_op("stem_conv1", x)

        if _MAXPOOL_ON_DEVICE:
            x = ttnn.experimental.quasar.max_pool2d(
                input_tensor=x,
                batch_size=self.batch_size,
                input_h=x_height,
                input_w=x_width,
                channels=self.conv1_output_channels,
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
                dilation=[1, 1],
            )
        else:
            # [#48552] HOST maxpool bypass (see _MAXPOOL_ON_DEVICE): the Quasar max_pool2d deadlocks at the
            # real stem size (compute_pool_2d pool-reduce dest-sync -- LLK item). Read the device input back,
            # run the 3x3/s2/p1 maxpool on host, and re-upload height-sharded ROW_MAJOR (same style as the
            # stem conv1 host fallback) so layer1..4 still run on device.
            logger.warning("[QSR] stem max_pool2d HOST bypass active (_MAXPOOL_ON_DEVICE=False)")
            _mp = (
                ttnn.to_torch(ttnn.from_device(x))
                .float()
                .reshape(self.batch_size, x_height, x_width, self.conv1_output_channels)
                .permute(0, 3, 1, 2)  # NHWC -> NCHW
            )
            _mp = torch.nn.functional.max_pool2d(_mp, kernel_size=3, stride=2, padding=1)
            _oh, _ow = int(_mp.shape[2]), int(_mp.shape[3])
            _flat = (
                _mp.permute(0, 2, 3, 1)
                .reshape(1, 1, self.batch_size * _oh * _ow, self.conv1_output_channels)
                .contiguous()
            )
            _nhw = self.batch_size * _oh * _ow
            ttnn.deallocate(x)
            if is_wormhole_b0():
                # [#48552] Upload the host-maxpool output ALREADY TILE-tiled, directly into layer1's 8x7
                # height-sharded config -- the exact tensor the WH-only stem_tilize block used to produce.
                # This lets us skip that block (quasar.to_memory_config + quasar.tilize), whose on-device
                # ROW_MAJOR->tiled conversion routes to transpose_wh_rm_sharded, which DEADLOCKS on WH (MATH
                # frozen in dest_section_flip TTI_STALLWAIT @ cmath_common.h:280 -- LLK item, handed off).
                # Tilizing on host during from_torch is numerically identical to the device tilize.
                _mem = ttnn.create_sharded_memory_config_(
                    (1, 1, _nhw, self.conv1_output_channels),
                    ttnn.CoreGrid(x=8, y=7),
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    tile_layout=True,
                )
                x = ttnn.from_torch(
                    _flat.to(torch.bfloat16),
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=_mem,
                )
            else:
                _grid = device.compute_with_storage_grid_size()
                _maxc = _grid.x * _grid.y
                _nc = max(c for c in range(1, _maxc + 1) if _nhw % c == 0)
                _mem = ttnn.create_sharded_memory_config(
                    shape=(1, 1, _nhw // _nc, self.conv1_output_channels),
                    core_grid=ttnn.num_cores_to_corerangeset(_nc, _grid, True),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                x = ttnn.from_torch(
                    _flat.to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=_mem,
                )
        x = _log_op("stem_maxpool", x)

        x_height = 56
        x_width = 56

        # [#48552] Only the on-device maxpool path needs the WH stem tilize here. The host bypass
        # (_MAXPOOL_ON_DEVICE=False) already uploaded the tiled 8x7 height-sharded tensor above, so it skips
        # this block -- avoiding the transpose_wh_rm_sharded WH deadlock (see the bypass comment).
        if is_wormhole_b0() and _MAXPOOL_ON_DEVICE:
            core_range_set = ttnn.CoreGrid(x=8, y=7)
            mem_config = ttnn.create_sharded_memory_config_(
                x.shape,
                core_range_set,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
            )
            x = ttnn.experimental.quasar.to_memory_config(x, mem_config)
            x = ttnn.experimental.quasar.tilize(x, dtype=self.model_config["ACTIVATIONS_DTYPE"])
            x = _log_op("stem_tilize", x)

        logger.debug(f"==== Running layer 1 module 1")

        reshard = is_blackhole()
        height_shard = True

        x, x_height, x_width = self.layer1_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module="layer1_module1",
        )

        logger.debug(f"==== Running layer 1 module 2")
        x, x_height, x_width = self.layer1_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded -> split path (not fused conv_bmm_tilize)
            layer_module="layer1_module2",
        )

        logger.debug(f"==== Running layer 1 module 3")
        x, x_height, x_width = self.layer1_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded -> split path
            layer_module="layer1_module3",
        )

        reshard = False
        height_shard = True

        logger.debug(f"==== Running layer 2 module 1")
        x, x_height, x_width = self.layer2_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module="layer2_module1",
        )

        logger.debug(f"==== Running layer 2 module 2")
        x, x_height, x_width = self.layer2_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded -> split path
            layer_module="layer2_module2",
        )

        logger.debug(f"==== Running layer 2 module 3")
        x, x_height, x_width = self.layer2_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded -> split path
            layer_module="layer2_module3",
        )

        logger.debug(f"==== Running layer 2 module 4")
        x, x_height, x_width = self.layer2_module4(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded -> split path
            layer_module="layer2_module4",
        )

        # [#48552] Quasar: layer3 conv2 uses the HEIGHT-sharded split (Program A tilize + Program B matmul) --
        # the same path layer1/2 pass on -- instead of the numerically-broken fused block conv. The old reason
        # to block-shard here was the ~512-tile uint16 DFB ring overflowing on the full per-core weights; the
        # 16-bit ring widen (f6b15a: compute-DFB ring_size -> uint32) removes that, so the full weights fit
        # resident. WH/BH keep their block-shard path; reshard reshards the input to the optimal height layout.
        reshard = is_blackhole() or is_quasar()
        height_shard = is_blackhole() or is_quasar()
        if is_wormhole_b0():
            x = ttnn.experimental.quasar.to_memory_config(
                x, ttnn.create_sharded_memory_config(x.shape, ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.BLOCK)
            )

        logger.debug(f"==== Running layer 3 module 1")
        x, x_height, x_width = self.layer3_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module="layer3_module1",
        )

        logger.debug(f"==== Running layer 3 module 2")
        x, x_height, x_width = self.layer3_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded split
            layer_module="layer3_module2",
        )

        logger.debug(f"==== Running layer 3 module 3")
        x, x_height, x_width = self.layer3_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded split
            layer_module="layer3_module3",
        )

        logger.debug(f"==== Running layer 3 module 4")
        x, x_height, x_width = self.layer3_module4(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded split
            layer_module="layer3_module4",
        )

        logger.debug(f"==== Running layer 3 module 5")
        x, x_height, x_width = self.layer3_module5(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded split
            layer_module="layer3_module5",
        )

        logger.debug(f"==== Running layer 3 module 6")
        x, x_height, x_width = self.layer3_module6(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            height_sharding=height_shard,  # [#48552] match module1 -> height-sharded split
            layer_module="layer3_module6",
        )

        # [#48552] Quasar: layer4 conv2 STAYS block-sharded (fused conv / LLK path). Unlike layer3, layer4's
        # full per-core weights (K=144 x N=16 = 2304 tiles ~= 4.7 MB) EXCEED Quasar's ~4 MB unreserved L1, so
        # even with f6b15a's uint32 ring they don't physically fit the single-K-block height split
        # (dataflow_buffer.cpp:812 FATAL ring_bytes <= unreserved_l1_size). Needs N-split (block) to fit.
        reshard = is_quasar()
        height_shard = False

        if is_wormhole_b0():
            block_mem_config = ttnn.create_sharded_memory_config(
                x.shape,
                ttnn.CoreGrid(x=8, y=7),
                ttnn.ShardStrategy.BLOCK,
            )
            x = ttnn.experimental.quasar.to_memory_config(x, block_mem_config)
        if is_blackhole():
            grid_size = (8, 10)
            block_mem_config = ttnn.create_sharded_memory_config_(
                [nearest_32(x.shape[2] // grid_size[1]), x.shape[3] // grid_size[0]],
                ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
                use_height_and_width_as_shard_shape=True,
            )
            x = ttnn.experimental.quasar.to_memory_config(x, block_mem_config)

        logger.debug(f"==== Running layer 4 module 1")
        x, x_height, x_width = self.layer4_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module="layer4_module1",
        )

        logger.debug(f"==== Running layer 4 module 2")
        x, x_height, x_width = self.layer4_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer4_module2",
        )

        logger.debug(f"==== Running layer 4 module 3")
        x, x_height, x_width = self.layer4_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer4_module3",
        )

        # WIDTH_SHARDED grid tied to device core count.
        num_cores, core_grid = fit_width_sharded_cores(x.shape[3], 8 * 8, device)
        width_mem_config = ttnn.create_sharded_memory_config_(
            [nearest_32(x.shape[2]), x.shape[3] // num_cores],
            core_grid,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.experimental.quasar.to_memory_config(x, width_mem_config)

        x = ttnn.experimental.quasar.avg_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=x_height,
            input_w=x_width,
            channels=x.shape[3],
            kernel_size=[x_height, x_width],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
            ),
        )
        x = _log_op("avgpool", x)

        # WIDTH_SHARDED activation for fc, on the SAME rectangular grid as the fc
        # 1D-mcast matmul (mcast_in0 requires the input sharding to match the matmul grid). Both
        # were derived together from the device in __init__ (fit_fc_grid), so this is the stock
        # 8x4=32 layout on Quasar and a smaller rectangle on the emulator.
        fc_core_grid = ttnn.CoreGrid(x=self.fc_matmul_grid[0], y=self.fc_matmul_grid[1])
        width_mem_config = ttnn.create_sharded_memory_config_(
            [nearest_32(x.shape[2]), x.shape[3] // self.fc_num_cores],
            fc_core_grid,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.experimental.quasar.to_memory_config(x, width_mem_config)

        x = self.fc(x)
        x = _log_op("fc", x)
        desired_shape = list(x.shape)
        desired_shape[-1] = 1000
        x = ttnn.experimental.quasar.untilize_with_unpadding(
            x,
            output_tensor_end=(desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1),
            memory_config=self.final_output_mem_config,
        )
        x = ttnn.experimental.quasar.reshape(
            x,
            (
                self.batch_size,
                x.shape[1],
                x.shape[2] // self.batch_size,
                x.shape[3],
            ),
        )
        x = _log_op("output", x)

        return x
