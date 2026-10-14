# SPDX-License-Identifier: Apache-2.0
#
# llk_analysis, ops/blackhole/sfpu_unary round 3, #58739: the multi-op unary chains ttnn and the models build, run
# through ttnn.unary_chain (the kernel and defines the composite ops use). R3_CHAIN_MODE=dump saves the output bits
# (every bf16 pattern as bfloat16; the same patterns widened to float32 and 65536 seeded float32 values in [-16, 16]),
# R3_CHAIN_MODE=perf runs five launches after a warm-up per case for tracy (DRAM interleaved at R3_SHAPES, or L1
# height-sharded with 8 x 8 tiles per core on the whole grid).
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
CHAINS = {
    # ttnn.sigmoid(mode=AccurateWithFastExp), unary.cpp
    "sig_fastexp": [W(U.NEG), W(U.EXP, 1.0), W(U.ADD_UNARY_SFPU, 1.0), W(U.RECIP)],
    # models/demos/gemma4_31b_qb2/tt/model.py final logit softcapping
    "softcap30": [W(U.MUL_UNARY_SFPU, 1.0 / 30.0), W(U.TANH, 0), W(U.MUL_UNARY_SFPU, 30.0)],
    # unary_backward.cpp and binary_backward.cpp chains
    "atan_bw": [W(U.SQUARE), W(U.ADD_UNARY_SFPU, 1.0), W(U.RECIP)],
    "atanh_bw": [W(U.SQUARE), W(U.SUB_UNARY_SFPU, 1.0), W(U.NEG), W(U.RECIP)],
    "asin_bw": [W(U.SQUARE), W(U.NEG), W(U.ADD_UNARY_SFPU, 1.0), W(U.RSQRT)],
    "asinh_bw": [W(U.SQUARE), W(U.ADD_UNARY_SFPU, 1.0), W(U.RSQRT)],
    "softsign_bw": [W(U.ABS), W(U.ADD_UNARY_SFPU, 1.0), W(U.SQUARE), W(U.RECIP)],
    "hypot_bw": [W(U.SQUARE), W(U.RECIP)],
    "neg_mul": [W(U.NEG), W(U.MUL_UNARY_SFPU, 2.5)],
    # controls: an op outside the table keeps every init per tile; one op takes the single-op path
    "ctl_neg_exp_acc": [W(U.NEG), W(U.EXP, 0.0)],
    "ctl_neg": [W(U.NEG)],
}
MODE = os.environ.get("R3_CHAIN_MODE", "dump")
SEL = [c for c in os.environ.get("R3_CHAINS", ",".join(CHAINS)).split(",") if c]
OUT = pathlib.Path(os.environ.get("SFPU_FIX_DUMP_DIR", "/tmp/sfpu_chain_dump_r3"))


def _dump_inputs():
    bits = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    g = torch.Generator().manual_seed(1234)
    rnd = torch.empty(1, 1, 256, 256, dtype=torch.float32).uniform_(-16.0, 16.0, generator=g)
    return {
        "bf16": (bits, ttnn.bfloat16, torch.int16),
        "fp32widened": (bits.to(torch.float32), ttnn.float32, torch.int32),
        "fp32random": (rnd, ttnn.float32, torch.int32),
    }


@pytest.mark.parametrize("chain", SEL)
def test_r3_chains(device, chain):
    ops = CHAINS[chain]
    if MODE == "dump":
        OUT.mkdir(parents=True, exist_ok=True)
        for name, (x, dtype, view) in _dump_inputs().items():
            tt_x = _r3_put(x, dtype, device)
            out = ttnn.to_torch(ttnn.unary_chain(tt_x, ops)).contiguous()
            torch.save(out.view(view), OUT / f"{chain}__{name}.pt")
        return
    torch.manual_seed(0)
    for dt in [d for d in os.environ.get("R3_DTYPES", "bf16,fp32").split(",") if d]:
        dtype = ttnn.bfloat16 if dt == "bf16" else ttnn.float32
        tdt = torch.bfloat16 if dt == "bf16" else torch.float32
        cases = []
        for layout in [l for l in os.environ.get("R3_LAYOUTS", "l1,dram").split(",") if l]:
            if layout == "l1":
                grid = device.compute_with_storage_grid_size()
                shard = (8 * 32, 8 * 32)
                shape = (1, 1, shard[0] * grid.x * grid.y, shard[1])
                mem = ttnn.create_sharded_memory_config(
                    shape=shard,
                    core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                cases.append((shape, mem))
            else:
                for s in os.environ.get("R3_SHAPES", "1x1x1024x1024").split(","):
                    cases.append((tuple(int(v) for v in s.split("x")), ttnn.DRAM_MEMORY_CONFIG))
        for shape, mem in cases:
            x = torch.empty(shape, dtype=tdt).uniform_(0.25, 0.75)
            tt_x = ttnn.from_torch(x, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem)
            out = ttnn.unary_chain(tt_x, ops)
            ttnn.synchronize_device(device)
            for _ in range(5):
                out = ttnn.unary_chain(tt_x, ops)
            ttnn.synchronize_device(device)
            assert ttnn.to_torch(out).shape == x.shape
