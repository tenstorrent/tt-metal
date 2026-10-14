# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ResNet101-DCN backbone for BEVFormer.

Bottleneck stages. layer3 and layer4 use a modulated deformable conv for conv2.
Each BatchNorm is already folded into the conv before it.
"""

from types import SimpleNamespace

import torch

import ttnn
from models.experimental.bevformer.tt.tt_common import TtnnConv2D
from models.experimental.bevformer.tt.tt_modulated_deform_conv import TtModulatedDeformConv2dDevice


class TtModulatedDeformConv2dPack:
    """DCNv2 conv together with the conv that predicts its offsets and mask.

    The geometry (stride, padding, dilation, input shape) comes from ``conv_args.conv_offset``,
    which shares it with the deformable conv.
    """

    def __init__(self, conv_args, conv_pth, device, input_dtype=ttnn.bfloat16, relu=False):
        offset_args = conv_args.conv_offset
        # __call__ reshapes the input to the output's (B, H, W), which holds only at stride 1.
        # BEVFormer's caffe-style ResNet puts every stride on conv1 or the downsample shortcut.
        assert tuple(offset_args.stride) == (1, 1), f"DCN supports stride 1 only, got {offset_args.stride}"
        self.batch_size = offset_args.batch_size

        weight = conv_pth.weight  # torch (C_out, C_in / groups, K, K)
        kernel_positions = weight.shape[2] * weight.shape[3]
        self.device_dcn = TtModulatedDeformConv2dDevice(
            weight=weight,
            bias=conv_pth.bias,
            stride=tuple(offset_args.stride),
            padding=tuple(offset_args.padding),
            dilation=tuple(offset_args.dilation),
            groups=offset_args.in_channels // weight.shape[1],
            deform_groups=offset_args.out_channels // (3 * kernel_positions),
            device=device,
            input_shape=(offset_args.batch_size, offset_args.input_height, offset_args.input_width),
            relu=relu,
        )

        # conv_offset emits the offsets in grid units, in the device DCN's (x, y) order (see
        # create_resnet_parameters and _grid_scaled_offsets), then the mask logits. Its output
        # is bfloat16 whatever the input dtype, since the offsets set the sampling positions.
        # Trained offsets reach tens of pixels, and accumulating in a bfloat16 destination
        # register errs by up to a few of them, so the register is fp32. The partial sums conv2d
        # writes back between reduction blocks stay bfloat16 (packer_l1_acc is off); keeping
        # those in fp32 too measured no better.
        self.num_offset_channels = 2 * kernel_positions
        self.conv_offset = TtnnConv2D(
            offset_args,
            self._grid_scaled_offsets(conv_pth.conv_offset, offset_args),
            device=device,
            fp32_dest_acc_en=True,
            input_dtype=input_dtype,
            output_dtype=ttnn.bfloat16,
        )

    def _grid_scaled_offsets(self, conv_offset, offset_args):
        """conv_offset's parameters with the offset rows pre-scaled by ``2 / [W, H]``.

        grid_sample takes offsets in grid units, a pixel offset times ``2 / [W, H]``; scaling
        the conv's (x, y) rows is exact and drops that multiply from every forward. Copies,
        so parameters shared by several modules are scaled only here.
        """
        scale = torch.ones(conv_offset.weight.shape[0])
        scale[: self.num_offset_channels : 2] = 2.0 / offset_args.input_width
        scale[1 : self.num_offset_channels : 2] = 2.0 / offset_args.input_height
        weight = ttnn.to_torch(conv_offset.weight) * scale.view(-1, 1, 1, 1)
        bias = ttnn.to_torch(conv_offset.bias) * scale.view(1, 1, 1, -1)
        return SimpleNamespace(
            weight=ttnn.from_torch(weight, dtype=conv_offset.weight.dtype),
            bias=ttnn.from_torch(bias, dtype=conv_offset.bias.dtype),
        )

    def __call__(self, x):
        """``x`` is a (1, 1, B*H*W, C_in) tensor in conv2d's layout. Returns the
        (1, 1, B*H_out*W_out, C_out) output, also in conv2d's layout, and its height and width."""
        out, out_h, out_w = self.conv_offset(x)
        out = ttnn.reshape(out, (self.batch_size, out_h, out_w, out.shape[-1]))
        grid_offset = out[:, :, :, : self.num_offset_channels]
        mask = ttnn.sigmoid(out[:, :, :, self.num_offset_channels :])
        ttnn.deallocate(out)

        # At stride 1 the input has the output's height and width. To ROW_MAJOR first, the layout
        # grid_sample reads, where the reshape to NHWC moves no data.
        x_nhwc = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (self.batch_size, out_h, out_w, x.shape[-1]))
        out = self.device_dcn(x_nhwc, grid_offset, mask)
        ttnn.deallocate(grid_offset)
        ttnn.deallocate(mask)
        return out, out_h, out_w


class TtResLayer:
    """One ResNet layer (layer1 .. layer4), one bottleneck per block in ``conv_pth``.

    ``input_dtype``, ``input_layout`` and ``dram_input`` describe the layer input, which
    only the first block reads. The other arguments go to every block.
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        dram_activation=False,
        dram_input=False,
        dram_conv_slices=None,
        block_sharded_downsample=False,
        fp32_acc=False,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
    ):
        num_blocks = len(conv_pth)
        self.layer = []
        for j in range(num_blocks):
            first = j == 0
            self.layer.append(
                TtBottleneck(
                    conv_args[j],
                    conv_pth[j],
                    device,
                    dram_activation=dram_activation,
                    dram_input=dram_input and first,
                    dram_conv_slices=dram_conv_slices,
                    block_sharded_downsample=block_sharded_downsample,
                    fp32_acc=fp32_acc,
                    input_dtype=input_dtype if first else self.layer[-1].output_dtype,
                    input_layout=input_layout if first else ttnn.TILE_LAYOUT,
                )
            )
        self.output_dtype = self.layer[-1].output_dtype

    def __call__(self, x):
        for block in self.layer:
            x = block(x)
        return x


class TtBottleneck:
    """One bottleneck block.

    Every conv's shape and stride come from ``conv_args``, recorded from one forward of
    the reference model. The block's conv2 is DCNv2 when ``conv_pth.conv2`` has an offset
    conv, and the block has a downsample shortcut when ``conv_pth`` has one.

    ``input_dtype`` and ``input_layout`` describe the block input; the dtype and layout
    each conv sees follow from them. ``dram_activation`` keeps the activations of conv1,
    a non-DCN conv2, conv3 and the downsample in DRAM, slicing the spatial convs into
    ``dram_conv_slices`` width slices; a DCN conv2 is unaffected. ``dram_input`` only
    covers the convs that read the block input, for a block fed by a DRAM layer whose own
    activations fit in L1. ``block_sharded_downsample`` block-shards the downsample conv.
    ``fp32_acc`` turns on fp32 destination accumulation (``fp32_dest_acc_en``) for conv1, a
    non-DCN conv2, conv3 and the downsample; a DCN conv2 always has it
    (TtModulatedDeformConv2dPack).
    """

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        dram_activation=False,
        dram_input=False,
        dram_conv_slices=None,
        block_sharded_downsample=False,
        fp32_acc=False,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
    ):
        self.with_dcn = "conv_offset" in conv_pth.conv2
        self.is_downsample = "downsample" in conv_pth

        # conv2d and ttnn.linear keep their input's dtype and emit TILE, and the DCN branch
        # emits bfloat16.
        conv2_dtype = input_dtype
        conv3_dtype = ttnn.bfloat16 if self.with_dcn else input_dtype
        self.output_dtype = conv3_dtype

        self.conv1 = TtnnConv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            fp32_dest_acc_en=fp32_acc,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            dram_activation=dram_activation or dram_input,
            dram_conv_slices=dram_conv_slices,
            input_dtype=input_dtype,
            input_layout=input_layout,
        )

        if not self.with_dcn:
            self.conv2 = TtnnConv2D(
                conv_args.conv2,
                conv_pth.conv2,
                device=device,
                fp32_dest_acc_en=fp32_acc,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                # act_block_h here and in the stem comes from the UniAD port and is not
                # re-tuned for 928x1600.
                act_block_h=32,
                dealloc_act=True,
                dram_activation=dram_activation,
                dram_conv_slices=dram_conv_slices,
                input_dtype=conv2_dtype,
            )
        else:
            # The BatchNorm is folded into the DCN weights (create_resnet_parameters); the ReLU
            # runs in the DCN's bias add.
            self.conv2 = TtModulatedDeformConv2dPack(
                conv_args.conv2, conv_pth.conv2, device, input_dtype=conv2_dtype, relu=True
            )

        self.conv3 = TtnnConv2D(
            conv_args.conv3,
            conv_pth.conv3,
            device=device,
            fp32_dest_acc_en=fp32_acc,
            activation=None,
            dealloc_act=True,
            dram_activation=dram_activation,
            dram_conv_slices=dram_conv_slices,
            input_dtype=conv3_dtype,
        )

        if self.is_downsample:
            self.downsample = TtnnConv2D(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                fp32_dest_acc_en=fp32_acc,
                activation=None,
                is_blk=block_sharded_downsample,
                dram_activation=dram_activation or dram_input,
                dram_conv_slices=dram_conv_slices,
                input_dtype=input_dtype,
                input_layout=input_layout,
            )

    def __call__(self, x_identity):
        x, _, _ = self.conv1(x_identity)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, _, _ = self.conv2(x)
        x, _, _ = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self.is_downsample:
            x_identity, _, _ = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.add(x, x_identity, dtype=self.output_dtype, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])

        ttnn.deallocate(x_identity)
        return x


class TtResNet:
    """Bottleneck ResNet built from ``conv_args`` and ``conv_pth``.

    ``create_resnet_parameters`` records those from one forward of the reference model: the conv
    shapes and strides, the max pool, each layer's block count and which blocks are DCNv2.

    ``out_indices``, ``dram_activation_stages`` and ``block_sharded_downsample_stages``
    index the four ResNet layers (0 is layer1). ``dram_activation_stages`` lists the layers
    whose activations are kept in DRAM, for layers whose convs do not fit in L1: their
    spatial convs run in ``dram_conv_slices`` width slices and their 1x1 convs as a DRAM
    matmul. A layer that follows one of them and is not listed itself reads its input from
    DRAM the same way. ``block_sharded_downsample_stages`` lists the layers whose downsample
    conv is block sharded, and ``fp32_acc_stages`` the layers whose convs accumulate in an fp32
    destination register (see TtBottleneck). The defaults keep everything in L1 at bfloat16
    accumulation; ``config/backbone_config.tt_resnet_kwargs`` gives BEVFormer-base's values.
    """

    num_layers = 4

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        out_indices=(0, 1, 2, 3),
        dram_activation_stages=(),
        dram_conv_slices=None,
        block_sharded_downsample_stages=(),
        fp32_acc_stages=(),
    ):
        self.out_indices = out_indices
        self.maxpool_args = conv_args.maxpool
        stage_config = dict(
            dram_activation_stages=dram_activation_stages,
            dram_conv_slices=dram_conv_slices,
            block_sharded_downsample_stages=block_sharded_downsample_stages,
            fp32_acc_stages=fp32_acc_stages,
        )

        self.conv1 = TtnnConv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            act_block_h=64,
            dealloc_act=True,
            input_dtype=ttnn.bfloat16,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # The max pool emits a bfloat16 ROW_MAJOR tensor and every layer emits bfloat16 TILE.
        # Activations stay bfloat16: the trained backbone has outlier channels that bfloat8_b's
        # shared block exponent flattens.
        layer_input_dtype = ttnn.bfloat16
        layer_input_layout = ttnn.ROW_MAJOR_LAYOUT
        self.output_dtypes = []
        self.res_layers = []
        for i in range(self.num_layers):
            res_layer = TtResLayer(
                conv_args=conv_args[f"layer{i+1}"],
                conv_pth=conv_pth[f"layer{i+1}"],
                device=device,
                **self.layer_kwargs(i, **stage_config),
                input_dtype=layer_input_dtype,
                input_layout=layer_input_layout,
            )
            self.res_layers.append(res_layer)
            layer_input_dtype = res_layer.output_dtype
            layer_input_layout = ttnn.TILE_LAYOUT
            if i in out_indices:
                self.output_dtypes.append(layer_input_dtype)

    @staticmethod
    def layer_kwargs(
        i, dram_activation_stages=(), dram_conv_slices=None, block_sharded_downsample_stages=(), fp32_acc_stages=()
    ):
        """The per-layer arguments of layer ``i`` (0 is layer1); a layer after a DRAM layer reads
        its input from DRAM."""
        return dict(
            dram_activation=i in dram_activation_stages,
            dram_input=i not in dram_activation_stages and i - 1 in dram_activation_stages,
            dram_conv_slices=dram_conv_slices,
            block_sharded_downsample=i in block_sharded_downsample_stages,
            fp32_acc=i in fp32_acc_stages,
        )

    def __call__(self, x):
        x, out_h, out_w = self.conv1(x)
        pool = self.maxpool_args
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv1.batch_size,
            input_h=out_h,
            input_w=out_w,
            channels=x.shape[3],
            kernel_size=[pool.kernel_size, pool.kernel_size],
            stride=[pool.stride, pool.stride],
            padding=[pool.padding, pool.padding],
            dilation=[pool.dilation, pool.dilation],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
        )

        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
