# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
``matmul_decode`` operation descriptor.

Creates an ``OpDescriptor`` for ``ttnn.experimental.matmul_decode`` (the decode-optimized
matmul used by ``LinearDecode`` / ``BatchedLinearDecode``, see
``models/experimental/deepseek_v4_flash/tt/layers.py``), so decode projections can be combined
with :class:`~models.experimental.ops.descriptors.fusion.fusion.Parallel` /
:class:`~models.experimental.ops.descriptors.fusion.fusion.Sequential` the same way plain
``matmul`` can via :mod:`~models.experimental.ops.descriptors.matmul`.

Unlike plain ``matmul``, this op takes no ``core_range_set`` override: the factory derives every
core placement from the shard specs already on ``input_tensor_a`` / ``input_tensor_b`` / the
output tensor (or, on the tensor-prefetcher path, from the ``global_cb``'s receiver set). So to
put two ``matmul_decode`` branches on disjoint cores for :class:`Parallel`, shard their
activations/weights onto disjoint ``CoreRangeSet``s *before* building the descriptor -- there is
nothing else for this module to place.

The nanobind types this binds against (``MatmulDecodeParams``, ``MatmulDecodeInputs``,
``MatmulDecodeDeviceOperation``, the per-factory ``create_descriptor``s, and
``matmul_decode_select_program_factory``) live under
``ttnn._ttnn.operations.experimental`` -- see
``ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/matmul_decode_descriptor.hpp`` and
its nanobind bindings in ``matmul_decode_nanobind.cpp``. They mirror the plain-matmul descriptor
bindings in ``ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp``.
"""

from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import (
    OpDescriptor,
    LazyOutputList,
    extend_branch_program_cache_key,
)

_prim = ttnn._ttnn.operations.experimental


@OpDescriptor.create(name="matmul_decode")
def matmul_decode(
    input_tensor_a: "ttnn.Tensor",
    input_tensor_b: "ttnn.Tensor",
    *,
    K: int,
    N: int,
    partial_width_sharded: bool = False,
    batch: int = 1,
    b_blocks: int = 1,
    n_blocks: int = 1,
    output_mem_config: Optional["ttnn.MemoryConfig"] = None,
    output_dtype: Optional["ttnn.DataType"] = None,
    global_cb=None,
    global_cb_k_blocks: int = 1,
    all_gather: bool = False,
    ring_size: int = 1,
    ring_gather: bool = False,
) -> "OpDescriptor":
    """Create a ``matmul_decode`` op descriptor.

    Args:
        input_tensor_a: Activation, width(K)-sharded L1 (the ``LinearDecode`` input layout).
        input_tensor_b: Weight. Either L1 width-sharded (matches ``partial_width_sharded`` /
            ``batch`` folding) or, when ``global_cb`` is given, DRAM ND-sharded (one slab per
            GCB receiver) -- the two ``LinearDecode`` weight-residency paths.
        K: Reduction dimension. Must match ``input_tensor_a``'s last dim.
        N: Output width (unfolded -- ``input_tensor_b``'s folded logical width divides out any
            ``k_blocks`` / ``b_blocks`` grouping the same way ``LinearDecode`` derives it).
        partial_width_sharded: Cut B along both K and N (the K-partials are reduced across
            cores). Ignored when ``batch > 1`` selects the batched factory.
        batch: > 1 selects the batched (block-diagonal) factory over a rank-4 activation
            (``BatchedLinearDecode``'s layout).
        b_blocks: Batch-axis fold count for the batched factory (defaults to 1; pass ``batch`` to
            match ``BatchedLinearDecode``'s default of one batch entry per core row).
        n_blocks: N-axis fold count for the partial / batched factories.
        output_mem_config: Output memory config override.
        output_dtype: Output dtype (defaults to ``input_tensor_a``'s).
        global_cb: ``ttnn.GlobalCircularBuffer`` supplying ``input_tensor_b`` from the tensor
            prefetcher (see ``LinearDecode``'s ``use_prefetcher`` path).
        global_cb_k_blocks: GCB pages per receiver slab (see ``ttnn.experimental.matmul_decode``).
        all_gather: Fuse a fabric all-gather of the local N-shard. ``ring_size`` must match
            the input mesh when this is set.
        ring_gather: Gather in0 over a pipelined closed ring instead of the two-hub gather.
            Full- and partial-width L1-resident paths only. Defaults to False.

    Returns:
        OpDescriptor with the matmul_decode program descriptor and IO tensors.
    """
    device = input_tensor_a.device()
    M = input_tensor_a.shape[-2]

    attrs = _prim.MatmulDecodeParams()
    attrs.M = M
    attrs.N = N
    attrs.K = K
    attrs.partial_width_sharded = partial_width_sharded
    attrs.batch = batch
    attrs.b_blocks = b_blocks
    attrs.n_blocks = n_blocks
    if output_mem_config is not None:
        attrs.output_mem_config = output_mem_config
    if output_dtype is not None:
        attrs.output_dtype = output_dtype
    if global_cb is not None:
        attrs.global_cb = global_cb
    attrs.global_cb_k_blocks = global_cb_k_blocks
    attrs.all_gather = all_gather
    attrs.ring_size = ring_size
    attrs.ring_gather = ring_gather

    tensor_args = _prim.MatmulDecodeInputs(input_tensor_a, input_tensor_b)

    factory = _prim.matmul_decode_select_program_factory(attrs, tensor_args)

    h = _prim.MatmulDecodeDeviceOperation.compute_program_hash(attrs, tensor_args)
    # global_cb identity (address, not just shape) already folds into the device hash (see
    # MatmulDecodeDeviceOperation::compute_program_hash); batch/b_blocks/n_blocks/K/N/
    # partial_width_sharded are all plain attribute fields the default hash already covers too,
    # so no extra fusion-cache key material is needed here (unlike plain matmul's core_range_set,
    # which the device hash omits).
    program_cache_key = extend_branch_program_cache_key(h)

    inputs = {"input_tensor_a": input_tensor_a, "input_tensor_b": input_tensor_b}

    def _alloc_outputs(slots):
        spec = _prim.MatmulDecodeDeviceOperation.compute_output_specs(attrs, tensor_args)
        slots[0] = ttnn.allocate_tensor_on_device(spec, device)

    outputs = LazyOutputList([None], _alloc_outputs)

    def _run_factory():
        out = outputs[0]
        return factory.create_descriptor(attrs, tensor_args, out)

    return OpDescriptor(
        factory_fn=_run_factory,
        input_tensors=inputs,
        output_tensors=outputs,
        program_cache_key=program_cache_key,
    )
