# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blocks shared across the port: the LayerNorm helper, the linear into L1 with its program
config, the decoder's FFN, layer stacking and device constants, and the conv the ResNet101-DCN backbone and the FPN
share.

conv2d lays the conv's weights out once, in the constructor, for one interleaved DRAM input dtype
and layout. A later call with a different input fails. A 1x1 conv whose
activation lives in DRAM runs as a matmul, over a subsample of its input when strided; a spatial
conv there runs in a fixed number of width or height slices.
"""

import functools
import math

import ttnn

try:
    from tracy import signpost as _tracy_signpost

    use_signpost = True
except ModuleNotFoundError:  # Tracy is only installed for profiling.
    _tracy_signpost = None
    use_signpost = False


def signpost(header, message=None):
    """Tracy stage marker. No-op unless ``use_signpost`` is on and Tracy is installed."""
    if not use_signpost or _tracy_signpost is None:
        return
    _tracy_signpost(header, message)


def layer_norm(x, params, residual=None, memory_config=None):
    """``ttnn.layer_norm(x + residual)`` with the eps of the module ``params`` were preprocessed from,
    in ``memory_config`` (None: the input's).

    ttnn defaults to epsilon=1e-12; ``preprocess_layer_norm_parameters`` records the
    module's own (1e-5 for nn.LayerNorm). The residual add runs inside the norm's kernel.
    """
    return ttnn.layer_norm(
        x,
        weight=params.weight,
        bias=params.bias,
        epsilon=params.eps,
        residual_input_tensor=residual,
        memory_config=memory_config,
    )


def device_constant(values, shape, dtype, device):
    """``values``, a flat list, as a ``shape`` TILE tensor on ``device``, uploaded once so a trace
    may capture the ops that read it."""
    return ttnn.to_device(ttnn.Tensor(values, shape, dtype, ttnn.TILE_LAYOUT), device)


def stack_layers(tensors):
    """``tensors`` stacked along a new leading axis in L1, where ttnn.stack lands in DRAM. A
    single tensor gains the axis as a view."""
    leading = [ttnn.unsqueeze(t, 0) for t in tensors]
    return leading[0] if len(leading) == 1 else ttnn.concat(leading, dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)


def linear_program_config(device, rows, in_features, out_features, activation=None):
    """A matmul over ``rows`` rows on the whole grid, the bias and ``activation`` inside it:
    ttnn.linear without a program config adds the bias in a separate op when the output goes to
    L1, and applies ``activation`` in a separate op always.

    An output of 4 tiles' width or more splits both ways, its columns over up to 8 grid columns,
    which divide the decoder's widths of 8 and 16 tiles, and its rows over the grid's rows: a few
    hundred rows split by rows alone leave most cores idle. A narrower one has too few columns for
    that to pay off and splits its rows alone, in blocks of up to 4 tile rows. The inner
    dimension goes in blocks of up to 8 tiles, which bounds the input's circular buffer in L1."""
    grid = device.compute_with_storage_grid_size()
    return _linear_program_config((grid.x, grid.y), rows, in_features, out_features, activation)


@functools.lru_cache
def _linear_program_config(grid, rows, in_features, out_features, activation):
    grid = ttnn.CoreCoord(*grid)
    tile = ttnn.TILE_SIZE
    rows_t = -(-rows // tile)
    in_t, out_t = -(-in_features // tile), -(-out_features // tile)
    in0_block_w = max(w for w in range(1, 9) if in_t % w == 0)
    fused_activation = None if activation is None else ttnn.UnaryWithParam(activation)
    # A subblock of out_subblock_w tiles, one tile high, sits in DST, which holds 4 tiles with
    # fp32 accumulation.
    if out_t >= 4:
        columns = max(c for c in range(1, min(8, grid.x) + 1) if out_t % c == 0)
        per_core_m, per_core_n = -(-rows_t // grid.y), out_t // columns
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(columns, grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=max(w for w in range(1, 5) if per_core_n % w == 0),
            out_block_h=per_core_m,
            out_block_w=per_core_n,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=True,
        )
    per_core_m = -(-rows_t // (grid.x * grid.y))
    block_h = min(4, per_core_m)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=max(w for w in range(1, 5) if out_t % w == 0),
        out_block_h=block_h,
        out_block_w=out_t,
        per_core_M=-(-per_core_m // block_h) * block_h,
        per_core_N=out_t,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=False,
    )


def l1_linear(x, linear, activation=None, dtype=None):
    """``linear`` (a preprocessed Linear's ``weight`` and optional ``bias``) applied to ``x`` into L1,
    with the bias and ``activation``, a ``ttnn.UnaryOpType``, inside the matmul.

    A matmul with a program config defaults to LoFi; the compute config here is the one ttnn.linear
    picks for bfloat16 operands without a program config: HiFi2, fp32 accumulation for an fp32
    output only.
    """
    fp32_output = dtype == ttnn.float32
    rows = math.prod(list(x.shape)[:-1])
    return ttnn.linear(
        x,
        linear.weight,
        bias=getattr(linear, "bias", None),
        dtype=dtype,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=linear_program_config(x.device(), rows, x.shape[-1], linear.weight.shape[-1], activation),
        compute_kernel_config=_linear_compute_config(x.device().arch(), fp32_output),
    )


@functools.lru_cache
def _linear_compute_config(arch, fp32_output):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_output,
        packer_l1_acc=not fp32_output,
    )


# Share of the device's L1 a group of intermediates that live at once may take; the rest holds
# the programs' circular buffers, which sit below the lowest L1 buffer.
L1_INTERMEDIATE_SHARE = 0.75


def l1_if_fits(device, nbytes):
    """``ttnn.L1_MEMORY_CONFIG`` for ``nbytes`` of intermediates that live at once when they fit
    ``L1_INTERMEDIATE_SHARE`` of the device's L1, else ``ttnn.DRAM_MEMORY_CONFIG``.

    A static budget against the L1's capacity, not its free space: the caller counts everything
    of its own that lives at once, and the order of the groups keeps others' buffers out of the
    way (``tt_encoder.TTBEVFormerLayer``)."""
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    capacity = view.num_banks * view.total_bytes_per_bank
    return ttnn.L1_MEMORY_CONFIG if nbytes <= L1_INTERMEDIATE_SHARE * capacity else ttnn.DRAM_MEMORY_CONFIG


def grid_linear_kwargs(device):
    """``ttnn.linear`` keyword arguments that keep the gridless default's HiFi2 on a full core grid.

    Without a core grid, an activation runs as a separate unary op over the whole output and an
    L1 output picks a config that post-processes the bias. A core grid fuses both, but defaults
    to LoFi.
    """
    grid = device.compute_with_storage_grid_size()
    return dict(
        core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
    )


class TTFFN:
    """The decoder's FFN, Linear-ReLU-Linear, into L1, without the residual: the layer adds it
    inside the LayerNorm that follows. ``params`` from ``model_preprocessing.preprocess_ffn_parameters``."""

    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        p = self.params
        return l1_linear(l1_linear(x, p.linear1, activation=ttnn.UnaryOpType.RELU), p.linear2)


def largest_divisor(n, limit):
    """The largest divisor of ``n`` that is at most ``limit``."""
    return max(d for d in range(1, min(n, limit) + 1) if n % d == 0)


def dst_tiles(fp32_dest_acc_en):
    """The tiles a matmul's output subblock may hold: the destination register's half that one
    subblock gets holds 8 tiles, 4 when they are fp32."""
    return 4 if fp32_dest_acc_en else 8


def grid_shard_tiles(device, rows):
    """Tile rows per core for ``rows`` rows height-sharded over ``device``'s whole grid."""
    grid = device.compute_with_storage_grid_size()
    return math.ceil(math.ceil(rows / ttnn.TILE_SIZE) / (grid.x * grid.y))


def grid_height_sharded(device, shard_tiles, channels):
    """L1 height sharding over ``device``'s whole grid, ``shard_tiles`` tile rows of ``channels``
    per core in row-major core order. Every per-core tensor of the backbone takes this layout, and
    the code compares layouts to skip a reshard, so they all come from here."""
    grid = device.compute_with_storage_grid_size()
    return ttnn.create_sharded_memory_config(
        shape=(shard_tiles * ttnn.TILE_SIZE, channels),
        core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def conv_output_size(height, width, kernel_size, stride, padding):
    """The output height and width of a conv on a ``height x width`` input, at dilation (1, 1), the
    only one TTConv2D and TTCameraBottleneck take."""
    return tuple(
        (size + 2 * pad - kernel) // step + 1
        for size, kernel, step, pad in zip((height, width), kernel_size, stride, padding)
    )


def is_pointwise(conv):
    """Whether ``conv``, anything with a conv's geometry, is a 1x1 conv without padding or groups: a
    matmul over channels of its input, subsampled at its stride."""
    return tuple(conv.kernel_size) == (1, 1) and tuple(conv.padding) == (0, 0) and conv.groups == 1


class TTConv2D:
    """``conv`` carries the conv's geometry and its input's batch, height and width.

    Most convs run through conv2d. ``input_dtype``, ``input_layout`` and
    ``input_memory_config`` describe the tensor every call receives, interleaved in DRAM unless
    given; conv2d lays its weights out for that input, so they are prepared here against it and a
    forward does no host work, and a call with a different input fails instead of running on
    mismatched weights.
    With ``dram_activation``, a 1x1 conv runs as ``ttnn.linear`` instead, over its input
    subsampled at its stride, and takes any layout, sharding or dtype, ignoring ``input_dtype`` and
    ``input_layout``; ``__call__(x, subsampled=True)`` takes ``x`` already at the conv's stride, as
    ``subsample`` returns it. Its
    ``conv_pth`` is then a Linear's ``(in, out)`` weight and ``(1, out)`` bias, which
    ``model_preprocessing`` emits for such convs. A spatial conv keeps conv2d and runs in
    ``dram_conv_slices`` width slices.

    ``dealloc_act`` lets conv2d free its L1 input, or the L1-sharded copy it makes of a DRAM
    input, once the halo has read it; a DRAM input itself is never freed.

    ``output_layout`` is the layout conv2d emits, None for its TILE default; a 1x1 conv run as
    ``ttnn.linear`` emits TILE and takes no other. conv2d's DRAM path runs every slice through its
    L1 conv with a TILE output and, with a single slice, returns that tensor as it is, whatever
    layout was asked for; so a ROW_MAJOR conv whose slice count comes to one, and one without
    ``dram_activation``, runs unsliced on the L1 path instead, and its activation and output must
    fit in L1. Its output is then sharded in L1.
    ``reshard_input`` lets conv2d reshard an L1-sharded input to the cores it would pick itself;
    without it, the conv runs on the input's own shards.
    ``l1_output`` lets a 1x1 conv run as ``ttnn.linear`` write its output height-sharded in L1
    when each core's shard fits ``L1_OUTPUT_MAX_SHARD_BYTES``; ``output_in_l1`` says whether it does.
    ``output_dtype`` is the dtype the conv emits, None for its input's. A spatial conv, on
    conv2d's own path with packer_l1_acc off, keeps its partial sums between reduction blocks
    in the output dtype; a 1x1 conv runs as a matmul (conv2d lowers a stride-1 one, and
    ``ttnn.linear`` above runs any), which keeps them in fp32 with ``fp32_dest_acc_en``.
    """

    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        activation=None,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk=False,
        dealloc_act=False,
        act_block_h=None,
        fp32_dest_acc_en=False,
        dram_activation=False,
        dram_conv_slices=None,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_dtype=None,
        output_layout=None,
        l1_output=False,
        reshard_input=True,
    ):
        self.dram_activation = dram_activation
        self.output_in_l1 = False
        self.input_dtype = input_dtype
        self.input_layout = input_layout
        self.input_memory_config = input_memory_config
        self.output_dtype = output_dtype
        if is_blk:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        # conv2d gets no dilation, nor does the slice count below account for one.
        assert tuple(conv.dilation) == (1, 1), f"TTConv2D supports dilation (1, 1) only, got {conv.dilation}"
        self.groups = conv.groups
        math_fidelity = ttnn.MathFidelity.HiFi2 if ttnn.get_arch_name() == "blackhole" else ttnn.MathFidelity.LoFi
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=dealloc_act,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=reshard_input,
            activation=activation,
        )
        if act_block_h is not None:
            self.conv_config.act_block_h_override = act_block_h
        if output_layout is not None:
            self.conv_config.output_layout = output_layout

        self.bias = conv_pth.bias

        self.weight = conv_pth.weight

        self.weights_dtype = weights_dtype
        self.is_pointwise = is_pointwise(conv)
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.batch_size = conv.batch_size
        self.output_height, self.output_width = conv_output_size(
            self.input_height, self.input_width, self.kernel_size, self.stride, self.padding
        )

        assert not (
            dram_activation and self.is_pointwise and output_layout == ttnn.ROW_MAJOR_LAYOUT
        ), "a 1x1 conv run as ttnn.linear emits TILE"
        self.slice_config = None
        num_slices = None
        if self.dram_activation and not self.is_pointwise:
            # conv2d's automatic slice count can overflow L1 on large activations, so the
            # count is fixed here. Width slices of a TILE output are whole tiles wide, so a
            # conv whose output is W wide takes at most ceil(W / TILE_SIZE) of them.
            assert dram_conv_slices is not None, "a spatial conv with dram_activation needs dram_conv_slices"
            num_slices = min(dram_conv_slices, math.ceil(self.output_width / ttnn.TILE_SIZE))
            self.slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=num_slices)
        if output_layout == ttnn.ROW_MAJOR_LAYOUT and num_slices in (None, 1):
            self.slice_config = ttnn.Conv2dL1FullSliceConfig

        self.linear_weight = None
        self.linear_bias = None
        self.linear_activation = None
        # ttnn.linear fuses its activation into the matmul only when it is given a core grid;
        # otherwise it runs it as a separate unary pass over the output.
        grid = device.compute_with_storage_grid_size()
        self.linear_core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        if self.dram_activation and self.is_pointwise and activation is not None:
            assert (
                activation.op_type == ttnn.UnaryOpType.RELU
            ), f"only RELU is supported after the pointwise linear, got {activation.op_type}"
            self.linear_activation = "relu"
        if not (self.dram_activation and self.is_pointwise):
            self._prepare_conv_weights()
        else:
            linear_shape = (self.in_channels, self.out_channels)
            assert tuple(self.weight.shape) == linear_shape and self.weight.dtype == weights_dtype, (
                f"a DRAM 1x1 conv takes a {weights_dtype} {linear_shape} Linear weight, got "
                f"{self.weight.dtype} {tuple(self.weight.shape)}"
            )
            self.linear_weight = ttnn.to_device(self.weight, self.device)
            self.linear_bias = None if self.bias is None else ttnn.to_device(self.bias, self.device)
            self.linear_program_config = self._linear_program_config(grid, fp32_dest_acc_en)
            if l1_output:
                # bfloat8_b is counted as 2 bytes too, which only overstates its shard.
                self.output_element_bytes = 4 if output_dtype == ttnn.float32 else 2
                self._l1_output_config(grid, fp32_dest_acc_en)

    # A pointwise linear's output shard per core above this stays in DRAM: layer3's 1024-channel
    # output takes about 650 KB per core next to the matmul's own buffers.
    L1_OUTPUT_MAX_SHARD_BYTES = 768 * 1024

    def _l1_output_config(self, grid, fp32_dest_acc_en):
        """Rows over every core, each core's output rows kept in its own L1 as a height shard, when
        that shard fits L1_OUTPUT_MAX_SHARD_BYTES. The shard then takes much of the core's L1, so
        the matmul works in output blocks of one tile row and inner blocks of at most 4 tiles,
        whose buffers stay small beside it."""
        per_core_m = grid_shard_tiles(self.device, self.batch_size * self.output_height * self.output_width)
        if per_core_m * ttnn.TILE_SIZE * self.out_channels * self.output_element_bytes > self.L1_OUTPUT_MAX_SHARD_BYTES:
            return
        in_t, out_t = self.in_channels // ttnn.TILE_SIZE, self.out_channels // ttnn.TILE_SIZE
        self.linear_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=largest_divisor(in_t, 4),
            out_subblock_h=1,
            out_subblock_w=largest_divisor(out_t, dst_tiles(fp32_dest_acc_en)),
            out_block_h=1,
            out_block_w=out_t,
            per_core_M=per_core_m,
            per_core_N=out_t,
            fuse_batch=True,
            fused_activation=None if self.linear_activation is None else ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            mcast_in0=False,
        )
        self.linear_output_memory_config = grid_height_sharded(self.device, per_core_m, self.out_channels)
        self.output_in_l1 = True

    # A widening linear whose activation gives each grid row fewer tile rows than this keeps
    # ttnn.linear's own split: layer4's 512 -> 2048 runs faster on it than on the 2D split.
    LINEAR_2D_MIN_ROWS_PER_GRID_ROW = 64

    def _linear_program_config(self, grid, fp32_dest_acc_en):
        """The pointwise linear's program config. It is None, for ttnn.linear's own, when the rows
        are fewer tile rows than cores, or when a widening linear gives each grid row fewer than
        LINEAR_2D_MIN_ROWS_PER_GRID_ROW; elsewhere ttnn.linear's own split runs the backbone's
        activations far from DRAM bandwidth. A linear that narrows to at most 8 tiles
        splits the rows over every core with the weights multicast; a wider output also splits its
        columns over up to 8 grid columns, which keeps a write-heavy linear off a few cores."""
        num_cores = grid.x * grid.y
        rows_t = math.ceil(self.batch_size * self.output_height * self.output_width / ttnn.TILE_SIZE)
        if rows_t < num_cores:
            return None
        in_t, out_t = self.in_channels // ttnn.TILE_SIZE, self.out_channels // ttnn.TILE_SIZE
        max_subblock = dst_tiles(fp32_dest_acc_en)
        relu = None if self.linear_activation is None else ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

        if out_t <= in_t and out_t <= 8:
            per_core_m = grid_shard_tiles(self.device, self.batch_size * self.output_height * self.output_width)
            # Blocks of 4 tile rows where each core holds at least 16, one row otherwise.
            block_h = 4 if per_core_m >= 16 else 1
            per_core_m = math.ceil(per_core_m / block_h) * block_h
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=in_t if in_t <= 16 else largest_divisor(in_t, 8),
                out_subblock_h=1,
                out_subblock_w=largest_divisor(out_t, max_subblock),
                out_block_h=block_h,
                out_block_w=out_t,
                per_core_M=per_core_m,
                per_core_N=out_t,
                fuse_batch=True,
                fused_activation=relu,
                mcast_in0=False,
            )
        if out_t > in_t and math.ceil(rows_t / grid.y) < self.LINEAR_2D_MIN_ROWS_PER_GRID_ROW:
            return None
        cols = largest_divisor(out_t, min(grid.x, 8))
        per_core_n = out_t // cols
        block_h = 16
        subblock_w = largest_divisor(per_core_n, max_subblock)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(cols, grid.y),
            in0_block_w=largest_divisor(in_t, 4),
            out_subblock_h=2 if 2 * subblock_w <= max_subblock else 1,
            out_subblock_w=subblock_w,
            out_block_h=block_h,
            out_block_w=per_core_n,
            per_core_M=math.ceil(math.ceil(rows_t / grid.y) / block_h) * block_h,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=relu,
            fuse_batch=True,
        )

    def _prepare_conv_weights(self):
        # conv2d lays its weights out for the input it sees and prepares them on the host on
        # its first call. Making that call here, on zeros shaped like every later input, puts
        # the host work in the constructor and uses conv2d's own preparation, which
        # ttnn.prepare_conv_weights does not reproduce for interleaved DRAM inputs.
        warmup_input = ttnn.zeros(
            (1, 1, self.batch_size * self.input_height * self.input_width, self.in_channels),
            dtype=self.input_dtype,
            layout=self.input_layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        warmup_input = ttnn.to_memory_config(warmup_input, self.input_memory_config)
        warmup_output, _, _ = self._conv2d(warmup_input)
        ttnn.deallocate(warmup_output)

    def subsample(self, x):
        """``x``, the ``(1, 1, N*H*W, C)`` input, at this conv's stride: ``(1, 1, N*H'*W', C)``
        ROW_MAJOR in DRAM. ROW_MAJOR, where the strided slice copies whole channel rows."""
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        nhwc = (self.batch_size, self.input_height, self.input_width, self.in_channels)
        x = ttnn.slice(
            ttnn.reshape(x, nhwc), [0, 0, 0, 0], list(nhwc), [1, *self.stride, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.reshape(x, (1, 1, self.batch_size * self.output_height * self.output_width, self.in_channels))

    def _pointwise_as_linear(self, x, subsampled):
        # conv2d lowers a 1x1 conv to a matmul that always shards its activation in L1 and
        # never slices it, so an activation larger than L1 cannot run as a conv; strided, it
        # also runs width-sliced with a halo. A 1x1 conv over NHWC is a matmul over the
        # channels of the pixels its stride keeps, and ttnn.linear streams interleaved DRAM
        # operands of any height.
        if tuple(self.stride) != (1, 1) and not subsampled:
            x = self.subsample(x)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        # conv2d returns (1, 1, N*H*W, C) whatever the input rank; match it so the
        # residual add sees the same shape on both branches.
        rows = math.prod(list(x.shape)[:-1])
        x = ttnn.reshape(x, (1, 1, rows, x.shape[-1]))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if self.linear_program_config is not None:
            # The ReLU runs as the program config's fused_activation.
            return ttnn.linear(
                x,
                self.linear_weight,
                bias=self.linear_bias,
                dtype=self.output_dtype,
                memory_config=self.linear_output_memory_config if self.output_in_l1 else ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.linear_program_config,
                compute_kernel_config=self.compute_config,
            )
        return ttnn.linear(
            x,
            self.linear_weight,
            bias=self.linear_bias,
            activation=self.linear_activation,
            dtype=self.output_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            core_grid=self.linear_core_grid,
        )

    def __call__(self, x, subsampled=False):
        if self.dram_activation and self.is_pointwise:
            return self._pointwise_as_linear(x, subsampled), self.output_height, self.output_width
        assert not subsampled, "only a 1x1 conv run as ttnn.linear takes a subsampled input"

        assert (
            x.memory_config() == self.input_memory_config
            and x.dtype == self.input_dtype
            and x.layout == self.input_layout
        ), (
            f"conv {self.in_channels}->{self.out_channels} k={tuple(self.kernel_size)}: weights prepared for a "
            f"{self.input_dtype} {self.input_layout} input in {self.input_memory_config}, got {x.dtype} "
            f"{x.layout} in {x.memory_config()}"
        )
        return self._conv2d(x)

    def _conv2d(self, x):
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            batch_size=self.batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            groups=self.groups,
            dtype=self.output_dtype,
            compute_config=self.compute_config,
            slice_config=self.slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x, output_height, output_width
