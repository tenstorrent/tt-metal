# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device quantize-dequantize (QDQ) of DeepSeek-V4.1 activations (dev-spec D-H, D-I; beads F6.1a, 8y7.9.4).

The reference quantizes activations in groups along the last dimension and immediately dequantizes
them back to bf16 (``kernel_cpu.act_quant`` / ``fp4_act_quant`` with ``inplace=True``). These functions
return the same bf16 values, bit for bit, given the same bf16 input:

* :func:`fp8_qdq`        -- FP8 e4m3, group 32, power-of-two (ue8m0) scale: window KV, DSpark rings and
  the activations entering quantized GEMMs.
* :func:`fp4_ue8m0_qdq`  -- FP4 E2M1, group 32, power-of-two scale: index keys and indexer queries.
* :func:`fp4_e4m3_qdq`   -- FP4 E2M1, group 16, e4m3 scale: compressed KV.

Each is one fused device op (``kernels/qdq_reader.cpp`` + ``kernels/qdq_compute.cpp`` + the stock interleaved
tile writer, launched through ``ttnn.generic_op``): a group of 32 is one tile row, a group of 16 one face row
(tile columns 0-15 or 16-31). Per tile the compute kernel takes the row max of ``|x|`` on the FPU (for group 16
twice, with reduce scaler tiles that mask the other half's columns), turns the group amax into the scale and
quantize-dequantizes every element on the SFPU in fp32. Every op is local to a row (a token) and to its device:
no collectives, so replicated or sequence-sharded mesh tensors work unchanged.

Exactness argument (all SFPU intermediates fp32; SFPU fp32 mul/add/compare are exact whenever the true result is
representable, which is all this relies on except where noted; every value leaving DST -- ``|x|``, the amax, the
scale, the output -- is exact in bf16, so packing to bf16 never rounds):

* Group amax is a max of bf16 magnitudes, hence exact (the scaler entries are 1.0 or 0.0).
* Power-of-two scales are built from the amax bit pattern. ``2^ceil(log2(fl(amax * fp32(1/M))))``
  (``fast_round_scale``) equals ``2^(e - k + [mantissa(amax) > m*])`` for bf16 amax, where
  ``M = 448`` (``k = 8``, ``m* = 1.75``) or ``M = 6`` (``k = 2``, ``m* = 1.5``), clamped from below by the
  scale of the amax floor (monotone). Checked on CPU for every positive finite bf16.
  Dividing by a power of two is exact, so the reference's ``x / s`` equals ``x * (1/s)`` here.
* e4m3 round-to-nearest-even (RNE) of a value ``v``: values below 2^-6 (the e4m3 subnormals, quantum 2^-9) are
  offset by 2^-6 onto the binade [2^-6, 2^-5), which has the same quantum and code parity, so clearing the low 20
  mantissa bits floors both ranges onto the grid. The decision compares the exact value against the half-way
  point above that floor (a <= 5-bit number) and resolves exact ties to the even code; an inexact estimate of
  ``v`` only selects the floor, and an estimate on the wrong side of a grid point still rounds to that point.
  Saturation (clamp to 448 before the cast) equals rounding then ``min(., 448)``.
* The e4m3 scale ``e4m3(amax_f / 6)`` of the group-16 FP4 path starts from the estimate ``amax_f * fp32(1/6)``
  (within an ulp of ``amax_f / 6``) and decides by ``amax_f`` vs ``6 * midpoint`` (<= 7-bit product, exact).
* E2M1 rounding of ``v = x / s`` compares ``v`` (power-of-two ``s``) or ``|x|`` against ``midpoint * s``
  (e4m3 ``s``: products of <= 3 and <= 4 significant bits, exact) with each boundary's tie direction: for the
  non-power-of-two scale ``x / s`` is inexact, but it lands on a midpoint exactly when ``x == midpoint * s`` and
  otherwise stays far from one, so the comparison reproduces the reference's rounding, ties included; ``> 5``
  gives the clamp to 6.
* Dequantized values have <= 6 significant bits and a normal exponent, so they are exact in bf16.

Checked on device against the reference for every finite normal bf16 value (tests/v41/test_qdq.py), with one
exception: :func:`fp4_ue8m0_qdq` groups whose amax is below ``FP4_UE8M0_MIN_AMAX`` (2^-115, ~2.4e-35) are not
reproduced -- their scale is within 2^-117 of the fp32 normal floor, where the device results differ (observed,
cause not isolated; far below any activation; accepted by the user 2026-09-30, bead 8y7.9.4). bf16 subnormal inputs are flushed to zero by the device (the reference
keeps them); that changes a result only inside the same tiny-amax FP4 groups. Non-finite inputs are outside the
contract. Zeros are compared by value (signed zeros are not distinguished).
"""

import ttnn

FP8_GROUP = 32
FP4_UE8M0_GROUP = 32
FP4_E4M3_GROUP = 16
FP4_UE8M0_MIN_AMAX = 2.0**-115  # fp4_ue8m0_qdq is bit-exact for groups with amax >= this (module docstring)

_KERNEL_DIR = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
_BLOCK = 4  # tiles per compute block (<= 4: one fp32 half-sync DST)
_TILE_BYTES = 32 * 32 * 2  # bf16 tile
_CB_IN, _CB_SCALER, _CB_ABS, _CB_SCALE, _CB_SCALE_BCAST, _CB_OUT = 0, 1, 2, 3, 4, 16

# qdq_compute.cpp QDQ_FORMAT ids
_FP8_E4M3_UE8M0 = 0
_FP4_E2M1_UE8M0 = 1
_FP4_E2M1_E4M3 = 2


def fp8_qdq(x: ttnn.Tensor) -> ttnn.Tensor:
    """``act_quant(x, 32, "ue8m0", inplace=True)``: FP8 e4m3 QDQ, group 32, power-of-two scale."""
    return _qdq(x, _FP8_E4M3_UE8M0, FP8_GROUP)


def fp4_ue8m0_qdq(x: ttnn.Tensor) -> ttnn.Tensor:
    """``fp4_act_quant(x, 32, inplace=True)``: FP4 E2M1 QDQ, group 32, power-of-two (E8M0) scale."""
    return _qdq(x, _FP4_E2M1_UE8M0, FP4_UE8M0_GROUP)


def fp4_e4m3_qdq(x: ttnn.Tensor) -> ttnn.Tensor:
    """``fp4_act_quant(x, 16, inplace=True, scale_dtype=float8_e4m3fn)``: FP4 E2M1 QDQ, group 16,
    scale ``e4m3_satfinite(max(amax, 6 * 2^-9) / 6)``."""
    return _qdq(x, _FP4_E2M1_E4M3, FP4_E4M3_GROUP)


def _qdq(x: ttnn.Tensor, fmt: int, group: int) -> ttnn.Tensor:
    """bf16 TILE interleaved ``x`` [..., W] -> a new tensor of its shape and memory config (one fused op)."""
    assert x.dtype == ttnn.bfloat16, f"QDQ input must be bf16 (the reference kernels' input dtype), got {x.dtype}"
    assert x.layout == ttnn.TILE_LAYOUT, "QDQ input must be TILE layout"
    width = x.shape[-1]
    assert width % group == 0, f"last dim {width} is not a multiple of the group size {group}"
    memory_config = x.memory_config()
    assert not memory_config.is_sharded(), "QDQ input must be interleaved"
    halves = 2 if group == 16 else 1
    device = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )

    # contiguous runs of tiles, one per core (tile padding is quantized too: groups never straddle tiles)
    num_tiles = x.buffer_num_pages()
    grid = device.compute_with_storage_grid_size()
    num_cores = min(grid.x * grid.y, num_tiles)
    base, extra = divmod(num_tiles, num_cores)
    last = ttnn.CoreCoord((num_cores - 1) % grid.x, (num_cores - 1) // grid.x)
    ranges = []
    if last.y > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, last.y - 1)))
    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, last.y), last))
    cores = ttnn.CoreRangeSet(ranges)
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    src, dst = x.buffer_address(), out.buffer_address()
    start = 0
    for i in range(num_cores):
        cx, cy = i % grid.x, i // grid.x
        count = base + (i < extra)
        reader_args[cx][cy] = [src, count, start]
        writer_args[cx][cy] = [dst, count, start]
        compute_args[cx][cy] = [count]
        start += count

    def cb(index: int, tiles: int) -> ttnn.CBDescriptor:
        page = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=_TILE_BYTES)
        return ttnn.CBDescriptor(total_size=tiles * _TILE_BYTES, core_ranges=cores, format_descriptors=[page])

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/qdq_reader.cpp",
            core_ranges=cores,
            compile_time_args=[_CB_IN, _CB_SCALER, halves, _BLOCK] + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            core_ranges=cores,
            compile_time_args=[_CB_OUT] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/qdq_compute.cpp",
            core_ranges=cores,
            compile_time_args=[fmt, _BLOCK],
            runtime_args=compute_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        ),
    ]
    cbs = [
        cb(_CB_IN, 2 * _BLOCK),
        cb(_CB_SCALER, halves),
        cb(_CB_ABS, _BLOCK),
        cb(_CB_SCALE, _BLOCK * halves),
        cb(_CB_SCALE_BCAST, _BLOCK),
        cb(_CB_OUT, 2 * _BLOCK),
    ]
    return ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
