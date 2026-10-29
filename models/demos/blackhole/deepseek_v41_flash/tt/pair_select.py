# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JIT sampler tail of the drafter (``ttnn.generic_op``): one scalar core per user row reduces the 8 gathered (max, index) column pairs to the global token and fetches the
markov-embedding row of that token (see mhc_kernels/pair_select.cpp). Bit-identical to the slice / argmax / gather / arithmetic chain of ``DSV41DeviceHead.sample_global_rm``."""

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel


def pair_select(pair, shard, embed, tile_pairs=False):
    """pair: fp32 TILE [1,1,U,2*cols] (the all-gathered (max_c, idx_c) pairs, U <= 32; tile_pairs=True: [1,1,U,64*cols], the max of column c in col 0 of tile 2c and its index in col 0
    of tile 2c+1: the layout of the gather of a [U,64] = two-tile row, which is ONE program where the 2-wide gather is two), embed: bf16 ROW_MAJOR [V,E] (E * 2 B per row).
    -> (tok uint32 RM [U,1], emb bf16 RM [1,1,U,E]) in DRAM."""
    U, W = int(pair.shape[2]), int(pair.shape[3])
    E = int(embed.shape[-1])
    cols = W // 64 if tile_pairs else W // 2
    assert U <= 32 and (W <= 16 or tile_pairs) and pair.dtype == ttnn.float32 and pair.layout == ttnn.TILE_LAYOUT
    assert embed.layout == ttnn.ROW_MAJOR_LAYOUT and embed.dtype == ttnn.bfloat16
    mesh = pair.device()
    tok = ttnn.allocate_tensor_on_device(
        ttnn.Shape([U, 1]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    emb = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, U, E]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = mesh.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][:U]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    rt = ttnn.RuntimeArgs()
    for u, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [u]
    emb_page = E * 2
    cbs = [_cb(core_set, 0, 1, 2048, ttnn.float32)]
    ct = [0, int(shard), cols, emb_page, int(tile_pairs)] + _acc(pair) + _acc(tok) + _acc(embed) + _acc(emb)
    kern = _kernel(
        "pair_select.cpp",
        core_set,
        ct,
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[pair.buffer_address(), tok.buffer_address(), embed.buffer_address(), emb.buffer_address()],
    )
    prog = ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x5C3,
        U,
        W,
        int(shard),
        E,
        int(tile_pairs),
        tuple(_acc(pair)),
        tuple(_acc(tok)),
        tuple(_acc(embed)),
        tuple(_acc(emb)),
    )
    ttnn.generic_op([pair, embed, tok, emb], prog)
    return tok, emb
