# SPDX-License-Identifier: Apache-2.0
#
# llk_analysis, ops/blackhole/sfpu_unary second-review rework, #58739: every ordered pair of the chain table's ops through
# ttnn.unary_chain (the chain forms and the op-major blocks for the combinations no model builds). Saves the output bits of
# every bf16 pattern as bfloat16 and as float32 to SFPU_FIX_DUMP_DIR; R3_PAIR_OPS selects the ops.
import os
import pathlib

import pytest
import torch
import ttnn
import sys

if not os.environ.get("HWLOCK_HELD") and not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
    sys.exit("not under hwlock")

R3_LAYOUT = os.environ.get("R3_DUMP_LAYOUT", "dram")


def _r3_put(x, dtype, device):
    # dram: DRAM interleaved, a 256 x 256 input is one tile per core; sh4: height-sharded on 4 cores, 16 tiles per core;
    # dram16: the input repeated 16 times along the height, DRAM interleaved, 9 or 10 tiles per core
    if R3_LAYOUT.startswith("sh"):
        # shN: height-sharded on N cores (sh4: 16 tiles per core; sh2: 32, past the program factory's 17-tile block threshold)
        cores = int(R3_LAYOUT[2:])
        mem = ttnn.create_sharded_memory_config(
            shape=(x.shape[-2] // cores, x.shape[-1]),
            core_grid=ttnn.CoreGrid(y=1, x=cores),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.from_torch(x, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem)
    if R3_LAYOUT == "dram16":
        x = x.repeat(1, 1, 16, 1)
    return ttnn.from_torch(x, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

pytestmark = pytest.mark.use_module_device

U = ttnn.UnaryOpType
W = ttnn.UnaryWithParam
TABLE = {
    "neg": W(U.NEG),
    "abs": W(U.ABS),
    "add": W(U.ADD_UNARY_SFPU, 1.0),
    "sub": W(U.SUB_UNARY_SFPU, 1.0),
    "mul": W(U.MUL_UNARY_SFPU, 2.5),
    "div": W(U.DIV_UNARY_SFPU, 3.0),
    "square": W(U.SQUARE),
    "rsqrt": W(U.RSQRT),
    "tanh": W(U.TANH, 0),
    "exp1": W(U.EXP, 1.0),
    "recip": W(U.RECIP),
}
SEL = [o for o in os.environ.get("R3_PAIR_OPS", ",".join(TABLE)).split(",") if o]
OUT = pathlib.Path(os.environ.get("SFPU_FIX_DUMP_DIR", "/tmp/sfpu_chain_pairs_r3"))


@pytest.mark.parametrize("first", SEL)
def test_r3_chain_pairs(device, first):
    OUT.mkdir(parents=True, exist_ok=True)
    bits = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    inputs = {"bf16": (bits, ttnn.bfloat16, torch.int16), "fp32widened": (bits.to(torch.float32), ttnn.float32, torch.int32)}
    for name, (x, dtype, view) in inputs.items():
        tt_x = _r3_put(x, dtype, device)
        for second in SEL:
            out = ttnn.to_torch(ttnn.unary_chain(tt_x, [TABLE[first], TABLE[second]])).contiguous()
            torch.save(out.view(view), OUT / f"{first}__{second}__{name}.pt")
