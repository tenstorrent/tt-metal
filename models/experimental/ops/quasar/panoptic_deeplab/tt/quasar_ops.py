# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The model's ops on Quasar.

Several ops the model uses have no Quasar kernels in their base implementation (it builds
``ComputeKernel`` / ``DataMovementKernel``, which Quasar rejects). On Quasar they run as their
copy under ``ttnn.experimental.quasar``, which takes the same arguments. Wormhole and Blackhole
keep the base ops.

``TtConv2d`` and ``TtMaxPool2d`` are the shared ``models/tt_cnn`` builder layers with their op
calls routed the same way. The builder itself is unchanged.
"""

import ttnn
from models.experimental.ops.quasar.panoptic_deeplab.tt.arch import is_quasar
from models.tt_cnn.tt import builder

# Ops with a Quasar copy that the model calls.
FORKED_OPS = ("conv2d", "max_pool2d", "avg_pool2d", "matmul", "move", "add", "mul")


def op(device, name):
    """``ttnn.experimental.quasar.<name>`` on Quasar, else ``ttnn.<name>``."""
    assert name in FORKED_OPS, f"{name} has no Quasar routing here; call ttnn.{name} directly"
    if is_quasar(device):
        return getattr(ttnn.experimental.quasar, "multiply" if name == "mul" else name)
    return getattr(ttnn, name)


def matmul(device, *args, program_config=None, **kwargs):
    """``ttnn.matmul``, routed. On Quasar only auto-config is supported: the Quasar matmul takes
    its own program-config classes, and grids without a tuned profile pass no config anyway."""
    if is_quasar(device):
        assert program_config is None, f"Quasar matmul needs its own program-config class, got {program_config}"
    return op(device, "matmul")(*args, program_config=program_config, **kwargs)


class TtConv2d(builder.TtConv2d):
    """``builder.TtConv2d`` with conv2d, and the move / add of its channel-sliced path, routed."""

    def _apply_channel_slicing(self, x):
        cfg = self.configuration
        num_slices = cfg.slice_strategy.get_num_slices()
        channels_per_slice = cfg.in_channels // num_slices
        is_flattened = (
            x.shape[0] == 1 and x.shape[1] == 1 and x.shape[2] == cfg.batch_size * cfg.input_height * cfg.input_width
        )
        conv2d, move, add = (op(self.device, name) for name in ("conv2d", "move", "add"))

        accumulated_output = None
        for i in range(num_slices):
            input_slice = ttnn.slice(
                x,
                [0, 0, 0, i * channels_per_slice],
                [
                    1 if is_flattened else cfg.batch_size,
                    1 if is_flattened else cfg.input_height,
                    cfg.batch_size * cfg.input_height * cfg.input_width if is_flattened else cfg.input_width,
                    (i + 1) * channels_per_slice,
                ],
            )
            slice_kwargs = self.get_conv2d_kwargs()
            slice_kwargs["in_channels"] = channels_per_slice
            output_slice, (h_out, w_out), (weight, _) = conv2d(
                input_tensor=input_slice,
                weight_tensor=self.weight_slices[i],
                bias_tensor=None,
                return_output_dim=True,
                return_weights_and_bias=True,
                compute_config=self.compute_config,
                **slice_kwargs,
            )
            self.weight_slices[i] = weight
            output_slice = move(output_slice)  # without this, some edge-case convs OOM
            if i == 0:
                accumulated_output = ttnn.to_memory_config(output_slice, ttnn.DRAM_MEMORY_CONFIG)
            else:
                accumulated_output = add(output_slice, accumulated_output, output_tensor=accumulated_output)
            output_slice.deallocate(True)

        if self.bias is not None:
            if self.bias.layout != ttnn.TILE_LAYOUT or self.bias.dtype != accumulated_output.dtype:
                self.bias = ttnn.to_layout(self.bias, ttnn.TILE_LAYOUT, dtype=accumulated_output.dtype)
            accumulated_output = add(accumulated_output, self.bias, output_tensor=accumulated_output)

        return accumulated_output, (h_out, w_out)

    def __call__(self, x, return_output_dim: bool = False):
        if self.weight_slices:
            x, (h_out, w_out) = self._apply_channel_slicing(x)
        else:
            x, [h_out, w_out], [self.weight, self.bias] = op(self.device, "conv2d")(
                input_tensor=x,
                weight_tensor=self.weight,
                bias_tensor=self.bias,
                return_output_dim=True,
                return_weights_and_bias=True,
                compute_config=self.compute_config,
                **self.get_conv2d_kwargs(),
            )
        return (x, (h_out, w_out)) if return_output_dim else x


class TtMaxPool2d(builder.TtMaxPool2d):
    """``builder.TtMaxPool2d`` with max_pool2d routed."""

    def __call__(self, x):
        max_pool2d = op(self.device, "max_pool2d")
        if not self.use_channel_slicing:
            return max_pool2d(input_tensor=x, channels=self.configuration.channels, **self.get_maxpool2d_kwargs())

        cfg = self.configuration
        output_slices = []
        for i in range(self.num_slices):
            input_slice = ttnn.slice(
                x,
                [0, 0, 0, i * self.channels_per_slice],
                [1, 1, cfg.batch_size * cfg.input_height * cfg.input_width, (i + 1) * self.channels_per_slice],
            )
            output_slice = max_pool2d(
                input_tensor=input_slice, channels=self.channels_per_slice, **self.get_maxpool2d_kwargs()
            )
            output_slices.append(ttnn.to_memory_config(output_slice, ttnn.DRAM_MEMORY_CONFIG))
            input_slice.deallocate(True)
        x = ttnn.concat(output_slices, dim=3)
        for output_slice in output_slices:
            output_slice.deallocate(True)
        return x
