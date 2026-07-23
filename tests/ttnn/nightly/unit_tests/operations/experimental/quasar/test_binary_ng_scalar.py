# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-SCALAR for the Quasar binary_ng DFB path (writer-fill of the RHS scalar tile).

Black-box, same op/golden/PCC as the no-broadcast suite, but the RHS is a Python number instead of a
tensor. On the DFB path the writer becomes the producer of the RHS input DFB (in1) and fills it ONCE
with the packed scalar via a coherent store (the non-cacheable L1 alias on Quasar DM cores, since the
DM write-back D$ is incoherent with the TL1 the compute consumer reads); the reader produces in0 only;
the compute waits on in1 once and reuses tile index 0. This is the make-or-break coherence proof: a
plain cacheable fill would leave the consumer reading zeros (out ~ a / ~ 0) or corrupt the neighbor DFB.

add/subtract are bf16-FPU. The metal_v2 / DFB tensor-scalar path is arch-portable -- CB-backed on
Wormhole/Blackhole, overlay-backed on Quasar -- and the is_scalar gate admits interleaved + sharded
scalar ops to it with no arch check, so these tests run on all three. On Quasar the descriptor path
throws (a pass implicitly proves v2 routing); on WH/BH the descriptor path also runs, so
test_scalar_v2_routing_distinct_cache_entries asserts v2 routing EXPLICITLY via the scalar-in-hash cache
signature (the metal_v2 compute_program_hash folds the packed scalar; the descriptor path and
operation_attributes_t::to_hash() exclude it).

Run on the Quasar simulator:
    unset TT_METAL_DISABLE_SFPLOADMACRO
    TT_METAL_SIMULATOR=<path>/libttsim.so TT_SIMULATOR_LOCALHOST=1 ARCH_NAME=quasar CHIP_ARCH=quasar \
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_scalar.py

Run on real Wormhole / Blackhole (first-silicon validation of the DFB path):
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_scalar.py
"""
import pytest

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.binary_ng_quasar_test_utils import (
    _run_scalar,
    _height_sharded_config,
)


@pytest.mark.parametrize("op_name", ["add", "subtract"])
@pytest.mark.parametrize("scalar", [3.5, -2.0])
@pytest.mark.parametrize("shape", [[2, 1, 64, 128], [1, 1, 32, 32]])
def test_scalar_fpu_bf16_interleaved(device, op_name, scalar, shape):
    # DRAM-interleaved bf16 tensor-scalar add/subtract. subtract (non-commutative) guards against an
    # a/scalar operand swap. Two scalars per shape also exercise the program-hash scalar fold: same
    # shape + different scalar must not false-hit the cache (the metal_v2 cache-hit adapter refreshes
    # only tensor bindings, not the baked packed_scalar runtime arg).
    _run_scalar(device, op_name, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, shape, scalar)


@pytest.mark.parametrize("op_name", ["multiply", "divide"])
@pytest.mark.parametrize("scalar", [3.5, -2.0])
@pytest.mark.parametrize("shape", [[2, 1, 64, 128]])
def test_scalar_sfpu_bf16_interleaved(device, op_name, scalar, shape):
    # DRAM-interleaved bf16 tensor-scalar multiply/divide: the SFPU sibling of
    # test_scalar_fpu_bf16_interleaved above, exercising eltwise_binary_sfpu_scalar_dfb.cpp (the
    # copy_tile-to-DST / BINARY_SFPU_OP path with the RHS scalar tile waited once and reused at index 0)
    # instead of the FPU BINARY_OP path. maximum/minimum tensor-scalar are intentionally excluded: their
    # ttnn.experimental.quasar overloads route through the unary clamp path (UnaryOpType::MAXIMUM/MINIMUM),
    # not invoke_binary_ng, so they never reach this DFB factory -- a separate, documented Quasar gap.
    _run_scalar(device, op_name, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, shape, scalar)


@pytest.mark.parametrize("op_name", ["add", "subtract", "multiply", "divide"])
@pytest.mark.parametrize("scalar", [2.0, -3.5])
def test_scalar_fp32_interleaved(device, op_name, scalar):
    # fp32 tensor-scalar, DRAM-interleaved. The gate's scalar RHS format is derived (no b tensor): the
    # DFB path is taken only when the derived b_dtype == a. is_binary_sfpu_op is dtype-aware, so ALL of
    # add/subtract/multiply/divide are SFPU for fp32 (a==b==FLOAT32) -- integer/float add & subtract route
    # the SFPU on Quasar (#49883), not just multiply/divide -- hence b_dtype = fp32 = a for every op here
    # and all four are admitted (the tensor-tensor no-bcast suite validates the same fp32 set). subtract
    # (non-commutative) guards an a/scalar operand swap; two scalars exercise the program-hash scalar fold.
    # Reuses _run_scalar (fp32 is exact enough for the file's _PCC[float32] = 0.9999).
    #
    # int32 tensor-scalar is intentionally NOT covered here: although the gate's format invariant would
    # admit int32 add/multiply (their RHS derives to int32 == a, and add_int_sfpu / mul_int_sfpu are in
    # the SFPU scalar kernel), the int32 SFPU tile ops do NOT produce correct results on the Quasar DFB
    # compute path (empirically: scalar add/mul return all-zero tiles, int32 tensor-tensor returns
    # garbage) -- a compute/sim-level gap, not a format issue. The gate keeps every int32 op on the
    # descriptor, so this stays out of the DFB suite until the int32 DFB compute path is fixed.
    _run_scalar(device, op_name, ttnn.DRAM_MEMORY_CONFIG, ttnn.float32, [2, 1, 64, 128], scalar)


# --- Layout + activation generality: sharded LHS, fused LHS activation ---------------------------------
# A scalar has no `b` tensor, so the DFB factory's all-or-nothing borrow_shards (= a_sharded && b_sharded
# && c_sharded, binary_ng_metal_v2_factory.cpp) can NEVER be true for a scalar op: b_sharded is derived
# from get_shard_volumes(a, is_scalar ? nullopt : b, c), and a nullopt b always yields an empty
# b_shard_volume, so b_sharded is unconditionally false. A sharded `a` (and, since _run_scalar applies one
# mem_config to both the input and the output, a sharded `c`) therefore never take the borrowed-shard
# fast path -- they are read/written over the NoC via their own sharding-aware TensorAccessor, exactly the
# reader_no_bcast_dfb.cpp in0 path reader_scalar_op_dfb.cpp preserves untouched (the #if SRC_SHARDED
# borrow branch simply never compiles in for this reader; only the else/TensorAccessor branch is live).
# Shape [2, 1, 64, 128] flattens (N*C*H, W) to a physical (128, 128) = 4 tile-rows x 4 tile-cols (16
# tiles); height-shard across 4 cores, one tile-row (32 rows) per core -- the same physical layout
# test_binary_ng_bcast.py's _ROW_B_FULL_HEIGHT uses for this exact shape.
_SCALAR_LHS_SHAPE = [2, 1, 64, 128]
_SCALAR_LHS_HEIGHT = _height_sharded_config([1 * 32, 4 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}))


@pytest.mark.parametrize("op_name", ["add", "multiply"])
def test_scalar_sharded_lhs(device, op_name):
    # Sharded LHS (+ sharded output, via _run_scalar's single mem_config) with a Python scalar RHS: proves
    # the in0 sharding-aware NoC path reader_scalar_op_dfb.cpp inherited from reader_no_bcast_dfb.cpp
    # handles a sharded operand correctly even though nothing is borrowed. add takes the FPU compute
    # kernel, multiply the SFPU kernel -- both exercised sharded.
    _run_scalar(device, op_name, _SCALAR_LHS_HEIGHT, ttnn.bfloat16, _SCALAR_LHS_SHAPE, 3.5)


@pytest.mark.parametrize("op_name", ["add", "multiply"])
@pytest.mark.parametrize("lhs_act", [ttnn.UnaryOpType.RELU])
def test_scalar_lhs_activation(device, op_name, lhs_act):
    # Fused LHS (pre) activation on the tensor operand, then the binary op against the scalar --
    # relu(a) op scalar. The FPU/SFPU scalar compute kernels carry the same PREPROCESS(LHS, ...) /
    # add_activation_defines machinery as the no-bcast kernels (test_no_bcast_lhs_activation in
    # test_binary_ng_no_bcast.py), and _run_scalar already threads lhs_act into both the ttnn call kwarg
    # (input_tensor_a_activations) and the torch golden (_ACT_GOLDEN[lhs_act](a)), mirroring _run/_run_mixed.
    _run_scalar(device, op_name, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, _SCALAR_LHS_SHAPE, 3.5, lhs_act=lhs_act)


# --- Explicit metal_v2 / DFB routing proof (replaces the Quasar-only "descriptor throws" implicit proof) --
def test_scalar_v2_routing_distinct_cache_entries(device):
    # On WH/BH the descriptor path also runs, so a passing PCC no longer implies the op took the DFB
    # (metal_v2) factory. Assert routing EXPLICITLY via the metal_v2 tensor-scalar cache signature: the v2
    # compute_program_hash folds the packed scalar (binary_ng_device_operation.cpp, the `if (metal_v2)`
    # branch), while operation_attributes_t::to_hash() and the descriptor path deliberately EXCLUDE it.
    # So, on the SAME shape:
    #   - repeating the SAME scalar   -> program-cache HIT   (+0 entries)
    #   - a DISTINCT scalar           -> a distinct program  (+1 entry)
    # A distinct scalar adding 0 entries would mean the op routed to the DESCRIPTOR path (scalar not
    # hashed), not the DFB factory. Mirrors test_binary_ng_descriptor_cache_hit.py's cross-shape check
    # and doubles as a regression guard on the scalar-in-hash fold. Requires the program cache (the
    # `device` fixture enables it, as in test_binary_ng_resnet_add.py / test_binary_ng_descriptor_cache_hit.py).
    shape = [1, 1, 32, 32]
    mem = ttnn.DRAM_MEMORY_CONFIG

    # Warm up the add-for-3.5 program (and any from_torch/to_torch machinery) so the deltas below isolate
    # the binary program.
    _run_scalar(device, "add", mem, ttnn.bfloat16, shape, 3.5)

    n0 = device.num_program_cache_entries()
    _run_scalar(device, "add", mem, ttnn.bfloat16, shape, 3.5)  # same shape + SAME scalar
    n1 = device.num_program_cache_entries()
    assert n1 - n0 == 0, f"same shape + same scalar must hit the program cache, got {n1 - n0} new entries"

    _run_scalar(device, "add", mem, ttnn.bfloat16, shape, -2.0)  # same shape, DISTINCT scalar
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 1, (
        f"a distinct scalar must create a distinct metal_v2 cache entry (the packed scalar is folded into "
        f"the v2 program hash) -- got {n2 - n1} new; 0 would mean the op routed to the descriptor path, "
        f"not the DFB/metal_v2 factory"
    )
