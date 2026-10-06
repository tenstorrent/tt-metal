# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""llk_analysis stage 2 (Blackhole eltwise binary): measurement variants that perf_eltwise_binary.py does not offer.

Drives the shadow kernel eltwise_binary_fpu_perf_x.cpp (a copy of eltwise_binary_fpu_perf.cpp with a broadcast
type, a dest-reuse type, a per-tile-init switch and a block-size override). Lives in the shadow tree only.
Variants are an explicit list so the node ids stay short; the sweep columns come from the template parameters.
"""

from collections import OrderedDict
from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    REUSE_DEST_TYPE,
    TILE_COUNT,
    TemplateParameter,
)

KERNEL = "sources/eltwise_binary_fpu_perf_x.cpp"


@dataclass
class PER_TILE_INIT(TemplateParameter):
    per_tile_init: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool PER_TILE_INIT = {str(self.per_tile_init).lower()};"


@dataclass
class HANDOFF(TemplateParameter):
    handoff: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t HANDOFF = {self.handoff};"


@dataclass
class UNPACK_BLOCK(TemplateParameter):
    unpack_block: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool UNPACK_BLOCK = {str(self.unpack_block).lower()};"


@dataclass
class BLOCK_TILES(TemplateParameter):
    block_tiles: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t BLOCK_TILES = {self.block_tiles};"


F = DataFormat
BF16, F32, BFP8 = F.Float16_b, F.Float32, F.Bfp8_b
ADD, SUB, MUL = MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul
LO, H2, H4 = MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4
NO, YES = DestAccumulation.No, DestAccumulation.Yes
BC = {"none": BroadcastType.None_, "row": BroadcastType.Row, "col": BroadcastType.Column, "scalar": BroadcastType.Scalar}
RD = {"none": EltwiseBinaryReuseDestType.NONE, "d2b": EltwiseBinaryReuseDestType.DEST_TO_SRCB, "d2a": EltwiseBinaryReuseDestType.DEST_TO_SRCA}

VARIANTS = OrderedDict()


def add(name, fin, fout, op, fid, acc, bcast="none", reuse="none", per_tile_init=False, block_tiles=0, handoff=0, unpack_block=False):
    assert name not in VARIANTS, name
    VARIANTS[name] = dict(formats=InputOutputFormat(fin, fout), op=op, fid=fid, acc=acc, bcast=bcast, reuse=reuse,
                          per_tile_init=per_tile_init, block_tiles=block_tiles, handoff=handoff,
                          unpack_block=unpack_block)


OPS3 = [("add", ADD, LO), ("mullofi", MUL, LO), ("mulhifi4", MUL, H4)]
OPS4 = [("add", ADD, LO), ("mullofi", MUL, LO), ("mulhifi2", MUL, H2), ("mulhifi4", MUL, H4)]
# fp32 inputs / outputs (H10): the standard test has no Float32
for accn, acc in (("noacc", NO), ("acc", YES)):
    for opn, op, fid in OPS4:
        add(f"f32_f32_{accn}_{opn}", F32, F32, op, fid, acc)
    for opn, op, fid in OPS3:
        add(f"f32_bf16_{accn}_{opn}", F32, BF16, op, fid, acc)
        add(f"bf16_f32_{accn}_{opn}", BF16, F32, op, fid, acc)
# bf16 reference rows (cross-check against perf_eltwise_binary.py, same kernel code path)
for opn, op, fid in OPS3:
    add(f"bf16_ref_{opn}", BF16, BF16, op, fid, NO)
# broadcast (H5 / E4)
for bc in ("row", "col", "scalar"):
    for opn, op, fid in OPS3:
        add(f"bf16_{bc}_{opn}", BF16, BF16, op, fid, NO, bcast=bc)
    add(f"bfp8_{bc}_add", BFP8, BFP8, ADD, LO, NO, bcast=bc)
    add(f"f32_acc_{bc}_add", F32, F32, ADD, LO, YES, bcast=bc)
# dest reuse (H6 / E6)
for rd in ("d2b", "d2a"):
    for opn, op, fid in (("add", ADD, LO), ("mullofi", MUL, LO), ("mulhifi2", MUL, H2)):
        add(f"bf16_{rd}_{opn}", BF16, BF16, op, fid, NO, reuse=rd)
add("f32_noacc_d2b_add", F32, F32, ADD, LO, NO, reuse="d2b")
# per-tile init (H11)
add("bf16_ptinit_add", BF16, BF16, ADD, LO, NO, per_tile_init=True)
add("bf16_ptinit_mulhifi4", BF16, BF16, MUL, H4, NO, per_tile_init=True)
add("f32_acc_ptinit_add", F32, F32, ADD, LO, YES, per_tile_init=True)
# one tile per DEST section (H12) and a 4-tile block
for opn, op, fid in OPS3:
    add(f"bf16_block1_{opn}", BF16, BF16, op, fid, NO, block_tiles=1)
add("f32_acc_block1_add", F32, F32, ADD, LO, YES, block_tiles=1)
add("bf16_acc_block1_add", BF16, BF16, ADD, LO, YES, block_tiles=1)
add("bf16_block4_add", BF16, BF16, ADD, LO, NO, block_tiles=4)
# round 2: what the binary_ng kernels run at one tile per DEST section (interleaved), with their hand-off switch;
# binary_ng compiles at HiFi4 (add and sub run at LoFi, the wrapper's effective fidelity) and turns on fp32 DEST
# when an operand or the output is Float32
for opn, op, fid in (("add", ADD, LO), ("sub", SUB, LO), ("mulhifi4", MUL, H4)):
    add(f"bng_bf16_block1_{opn}", BF16, BF16, op, fid, NO, block_tiles=1, handoff=1)
    add(f"bng_f32_acc_block1_{opn}", F32, F32, op, fid, YES, block_tiles=1, handoff=1)
for opn, op, fid in (("add", ADD, LO), ("mulhifi4", MUL, H4)):
    add(f"bng_bfp8_block1_{opn}", BFP8, BFP8, op, fid, NO, block_tiles=1, handoff=1)
    add(f"bng_bf16_f32_acc_block1_{opn}", BF16, F32, op, fid, YES, block_tiles=1, handoff=1)
    add(f"bng_f32_bf16_acc_block1_{opn}", F32, BF16, op, fid, YES, block_tiles=1, handoff=1)
    add(f"bng_bf16_block1_ptinit_{opn}", BF16, BF16, op, fid, NO, per_tile_init=True, block_tiles=1, handoff=1)
    add(f"bng_f32_acc_block1_ptinit_{opn}", F32, F32, op, fid, YES, per_tile_init=True, block_tiles=1, handoff=1)
# the sharded layouts: 8 tiles per DEST section (4 with fp32 DEST)
for opn, op, fid in (("add", ADD, LO), ("sub", SUB, LO), ("mulhifi4", MUL, H4)):
    add(f"bng_bf16_{opn}", BF16, BF16, op, fid, NO, handoff=1)
    add(f"bng_f32_acc_{opn}", F32, F32, op, fid, YES, handoff=1)
for opn, op, fid in (("add", ADD, LO), ("mulhifi4", MUL, H4)):
    add(f"bng_bfp8_{opn}", BFP8, BFP8, op, fid, NO, handoff=1)
    add(f"bng_bf16_bfp8_{opn}", BF16, BFP8, op, fid, NO, handoff=1)
    add(f"bng_bf16_f32_acc_{opn}", BF16, F32, op, fid, YES, handoff=1)
    add(f"bng_f32_bf16_acc_{opn}", F32, BF16, op, fid, YES, handoff=1)

# final round (r3): HANDOFF 0 (every row above without handoff) is the compute API default, the per-face hand-off;
# HANDOFF 1 is binary_ng's opt-in; HANDOFF 2 a kernel that opts in (per-tile on the standard and dest-reuse paths)
for rd in ("d2b", "d2a"):
    for opn, op, fid in (("add", ADD, LO), ("mullofi", MUL, LO), ("mulhifi2", MUL, H2)):
        add(f"opt_bf16_{rd}_{opn}", BF16, BF16, op, fid, NO, reuse=rd, handoff=2)
add("opt_f32_noacc_d2b_add", F32, F32, ADD, LO, NO, reuse="d2b", handoff=2)
# the prod kernels: dest-reuse multiply DEST_TO_SRCA at HiFi4 with fp32 DEST, default and opted in
for h in (0, 2):
    pre = "opt_" if h else ""
    add(f"{pre}bf16_f32acc_d2a_mulhifi4", BF16, BF16, MUL, H4, YES, reuse="d2a", handoff=h)
    add(f"{pre}bf16_f32acc_d2a_mulhifi4_block1", BF16, BF16, MUL, H4, YES, reuse="d2a", block_tiles=1, handoff=h)
# tt-blaze layernorm at width 1024 (one 32x32 tile per DEST section, LoFi): direct LLK inits with the default hand-off
# and the compute API execute, which is the per-face program unless the kernel opts in
add("blaze_ln_bf16_d2a_mullofi_block1", BF16, BF16, MUL, LO, NO, reuse="d2a", block_tiles=1)
add("blaze_ln_bf16_d2a_add_block1", BF16, BF16, ADD, LO, NO, reuse="d2a", block_tiles=1)

# round 3: the two-operand block unpack (one context acquire per DEST section), default and opted-in hand-off
for h in (0, 2):
    for blk in (0, 2, 4, 8):
        sfx = (f"_block{blk}" if blk else "") + ("_opt" if h else "")
        for opn, op, fid in (("add", ADD, LO), ("mulhifi4", MUL, H4)):
            add(f"blk_bf16_{opn}{sfx}", BF16, BF16, op, fid, NO, block_tiles=blk, handoff=h, unpack_block=True)
        add(f"blk_bfp8_add{sfx}", BFP8, BFP8, ADD, LO, NO, block_tiles=blk, handoff=h, unpack_block=True)
        add(f"blk_f32_acc_add{sfx}", F32, F32, ADD, LO, YES, block_tiles=blk, handoff=h, unpack_block=True)
        add(f"blk_bf16_f32_acc_add{sfx}", BF16, F32, ADD, LO, YES, block_tiles=blk, handoff=h, unpack_block=True)

# round 3 (#58723): the broadcast multiplies, per-face (HANDOFF 0) and opted in (HANDOFF 3, per-tile with the broadcast)
for h in (0, 3):
    sfx = "_opt" if h else ""
    for bc in ("row", "col", "scalar"):
        for opn, op, fid in (("mullofi", MUL, LO), ("mulhifi2", MUL, H2), ("mulhifi4", MUL, H4)):
            add(f"bcm_bf16_{bc}_{opn}{sfx}", BF16, BF16, op, fid, NO, bcast=bc, handoff=h)
        add(f"bcm_bfp8_{bc}_mulhifi4{sfx}", BFP8, BFP8, MUL, H4, NO, bcast=bc, handoff=h)
        add(f"bcm_f32_acc_{bc}_mulhifi4{sfx}", F32, F32, MUL, H4, YES, bcast=bc, handoff=h)
        add(f"bcm_bf16_{bc}_mulhifi4_block1{sfx}", BF16, BF16, MUL, H4, NO, bcast=bc, block_tiles=1, handoff=h)
        add(f"bcm_bf16_f32acc_{bc}_mulhifi2{sfx}", BF16, BF16, MUL, H2, YES, bcast=bc, handoff=h)


@pytest.mark.perf
@parametrize(variant=list(VARIANTS.keys()))
def test_perf_eltwise_binary_x(perf_report, variant):
    if isinstance(variant, tuple):  # the single-axis parametrize hands the value over as a 1-tuple
        variant = variant[0]
    v = VARIANTS[variant]
    formats = v["formats"]
    tile_count = 16

    configuration = PerfConfig(
        KERNEL,
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_FIDELITY(v["fid"]),
            MATH_OP(mathop=v["op"]),
            BROADCAST_TYPE(BC[v["bcast"]]),
            REUSE_DEST_TYPE(reuse_dest_type=RD[v["reuse"]]),
            PER_TILE_INIT(v["per_tile_init"]),
            BLOCK_TILES(v["block_tiles"]),
            HANDOFF(v["handoff"]),
            UNPACK_BLOCK(v["unpack_block"]),
        ],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(8)],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        dest_acc=v["acc"],
    )

    configuration.run(perf_report)
