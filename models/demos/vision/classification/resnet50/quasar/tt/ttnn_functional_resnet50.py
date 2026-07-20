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
            f"nan={int(torch.isnan(f).sum().item())} first8={[round(v, 4) for v in f[:8].tolist()]}"
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

            ds_out, [self.ds_conv_weight_tensor, self.ds_conv_bias_tensor] = ttnn.experimental.quasar.conv2d(
                input_tensor=x,
                weight_tensor=self.ds_conv_weight_tensor,
                bias_tensor=self.ds_conv_bias_tensor,
                **conv_kwargs,
                compute_config=ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=self.model_config["MATH_FIDELITY"],
                    packer_l1_acc=packer_l1_accum_enabled,
                ),
                return_output_dim=False,
                return_weights_and_bias=True,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
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
        ) = ttnn.experimental.quasar.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs_1,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
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
            if ds_input_height == 56 and self.conv1_input_channels == 256 and self.downsample:
                # Defragment L1 before the projection conv so it fits alongside conv2.
                x_rm = ttnn.experimental.quasar.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(x)
                x = ttnn.experimental.quasar.reallocate(x_rm)
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
                # bfloat16 doubles every tensor; mirror the large variant's minimal
                # conv2 config (no double buffering / activation reuse / full inner
                # dim) so the CBs fit in L1.
            ),
        }

        (
            out,
            [input_height, input_width],
            [self.conv2_weight_tensor, self.conv2_bias_tensor],
        ) = ttnn.experimental.quasar.conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            bias_tensor=self.conv2_bias_tensor,
            **conv_kwargs_2,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
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

        out, [self.conv3_weight_tensor, self.conv3_bias_tensor] = ttnn.experimental.quasar.conv2d(
            input_tensor=out,
            weight_tensor=self.conv3_weight_tensor,
            bias_tensor=self.conv3_bias_tensor,
            **conv_kwargs_3,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=False,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
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

        x, [x_height, x_width], [self.conv1_weight_tensor, self.conv1_bias_tensor] = ttnn.experimental.quasar.conv2d(
            input_tensor=fold_output_tensor,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs,
            compute_config=self.conv1_compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )
        x = _log_op("stem_conv1", x)

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
        x = _log_op("stem_maxpool", x)

        x_height = 56
        x_width = 56

        if is_wormhole_b0():
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
            layer_module="layer1_module2",
        )

        logger.debug(f"==== Running layer 1 module 3")
        x, x_height, x_width = self.layer1_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
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
            layer_module="layer2_module2",
        )

        logger.debug(f"==== Running layer 2 module 3")
        x, x_height, x_width = self.layer2_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer2_module3",
        )

        logger.debug(f"==== Running layer 2 module 4")
        x, x_height, x_width = self.layer2_module4(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer2_module4",
        )

        # Quasar: block-shard layer3 like WH does. height_shard is already False here (block config, via
        # is_blackhole()==False), but the WH block-input reshard below is gated on is_wormhole_b0() and
        # skips Quasar, leaving the input height-sharded -> the 512->1024 conv keeps the full 1024-ch
        # weight matrix per core (1 MB) and overflows the uint16_t DFB ring. Enable reshard_if_not_optimal
        # so the op reshards the height-sharded input to the optimal block layout (grid-agnostic; the
        # reshard_if_not_optimal path is the one BH uses here). Splitting output channels across the grid
        # keeps the per-core weights DFB well under 1 MB.
        reshard = is_blackhole() or is_quasar()
        height_shard = is_blackhole()
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
            layer_module="layer3_module2",
        )

        logger.debug(f"==== Running layer 3 module 3")
        x, x_height, x_width = self.layer3_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module3",
        )

        logger.debug(f"==== Running layer 3 module 4")
        x, x_height, x_width = self.layer3_module4(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module4",
        )

        logger.debug(f"==== Running layer 3 module 5")
        x, x_height, x_width = self.layer3_module5(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module5",
        )

        logger.debug(f"==== Running layer 3 module 6")
        x, x_height, x_width = self.layer3_module6(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module6",
        )

        # Quasar: same block-shard gap as layer3 (the WH/BH block-input reshards below skip Quasar).
        # height_shard is already False (block config); enable reshard_if_not_optimal so the op reshards
        # the input to the optimal block layout, keeping the 1024->2048 conv's per-core weights DFB
        # under the uint16_t ring limit.
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
