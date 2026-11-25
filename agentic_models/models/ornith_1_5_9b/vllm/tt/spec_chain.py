# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chained multi-token GDN programs for speculative verify (one request, T candidates; prototype).

Plain decode advances one token per launch: conv_gates (4-tap conv over the slot's 3-row history) and
flat_ordered_conv_gates (one core per value head: exact FP32 recurrence, norm, gate). A verify needs the T
candidates' outputs AND the state after every candidate (to roll back a partial accept). These programs do all T
candidates in one launch each and leave the live state untouched:

- conv_gates_chain: same compute kernel; the reader hands row t the operands of candidate t's plain step (history
  rows from the slot and from earlier candidates); the writer's history output becomes the chain history (row t =
  history after candidate t).
- gdn_chain: 8 cores (one per value head) each walk the T candidates; candidate t reads the live slot state (t = 0)
  or chain row t-1 written by its own writer (L1 semaphore hand-off) and writes chain row t.
- gdn_commit: one core per GDN layer copies chain row a (a = accepted candidate, read from a device tensor, so it
  can be traced) into the live slot: recurrent heads and history row.
Arithmetic per candidate is the plain step's, so results are bit-identical to T plain decode steps.
"""
import struct
from pathlib import Path

import ttnn
from agentic_models.models.ornith_1_5_9b.vllm.tt.flat_ordered_conv_gates import MATH_FIDELITY

KERNELS = Path(__file__).parent / "kernels"


def conv_gates_chain(raw, history, chain_history, taps, bias, decay, T):
    """raw rows r = u*T + t (candidate t of user u, whose live history is pool row u)."""
    R = raw.shape[2]
    assert raw.dtype == history.dtype == chain_history.dtype == ttnn.bfloat16 and raw.shape[-1] == 3136
    # the slot pool has max_num_seqs rows: one physical 32-row tile per tap and column either way
    assert tuple(history.shape) == tuple(chain_history.shape) and tuple(history.shape)[:2] == (3, 1)
    assert history.shape[2] <= 32 and history.shape[3] == 2048 and 1 <= R <= 32 and R % T == 0
    dev = raw.device()
    out = ttnn.empty(
        (1, 1, R, 2048), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    gates = [
        ttnn.empty((1, 1, R, 8), device=dev, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        for dtype in (ttnn.float32, ttnn.bfloat16)
    ]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    cbs = []
    for idx in (*range(14), 16, 17, 18, 19):
        dtype = ttnn.float32 if idx in (4, 5, 6, 7, 10, 11, 12, 13, 17) else ttnn.bfloat16
        size = 4096 if dtype == ttnn.float32 else 2048
        pages = 3 if idx == 19 else 1
        cbs.append(
            ttnn.CBDescriptor(
                total_size=pages * size,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)],
            )
        )
    rct = []
    for tensor in (raw, history, *taps, bias, decay):
        rct += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    wct = (
        ttnn.TensorAccessorArgs(out).get_compile_time_args()
        + ttnn.TensorAccessorArgs(chain_history).get_compile_time_args()
    )
    wct += (
        ttnn.TensorAccessorArgs(gates[0]).get_compile_time_args()
        + ttnn.TensorAccessorArgs(gates[1]).get_compile_time_args()
    )
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i in range(64):
        x, y = i % 8, i // 8
        rr[x][y] = [
            i,
            raw.buffer_address(),
            history.buffer_address(),
            *[t.buffer_address() for t in taps],
            bias.buffer_address(),
            decay.buffer_address(),
            T,
            R,
        ]
        wr[x][y] = [
            i,
            out.buffer_address(),
            chain_history.buffer_address(),
            gates[0].buffer_address(),
            gates[1].buffer_address(),
        ]
        cr[x][y] = [i]
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
    for index in (10, 11, 12, 13):
        unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    compute_config.unpack_to_dest_mode = unpack_modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "conv_gates_chain/reader.cpp"),
            core_ranges=grid,
            compile_time_args=rct,
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "conv_gates/writer.cpp"),
            core_ranges=grid,
            compile_time_args=wct,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "conv_gates/compute.cpp"),
            core_ranges=grid,
            runtime_args=cr,
            config=compute_config,
            defines=[("INP_FLOAT32", "1")],
        ),
    ]
    ttnn.generic_op(
        [raw, history, *taps, bias, decay, chain_history, out, *gates],
        ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]),
    )
    return out, *gates


# CB page counts of flat_ordered_conv_gates (kept identical: the compute kernel is shared)
_GDN_CBS = {
    0: 4,
    1: 4,
    2: 4,
    3: 1,
    4: 1,
    5: 16,
    6: 4,
    7: 16,
    8: 4,
    9: 4,
    10: 4,
    11: 4,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 4,
    17: 4,
    18: 4,
    19: 4,
    20: 4,
    21: 4,
    22: 4,
    23: 1,
    24: 1,
    25: 4,
    26: 1,
    27: 1,
    28: 1,
    29: 4,
    30: 4,
    31: 4,
    32: 16,
    33: 16,
    34: 1,
}


def gdn_chain(conv, g, beta, state, chain_state, gate, gamma, epsilon, T):
    """B users x T candidates (rows u*T + t). state: the live pool [P*8,128,128] FP32 (user u = pool row u);
    chain_state: [>=8*B*T,128,128] FP32, head (u*T + t)*8 + hv = state after candidate t of user u."""
    R = conv.shape[2]
    users = R // T
    items = 8 * users
    assert R % T == 0 and tuple(gate.shape) == (1, 1, R, 3136) and tuple(conv.shape) == (1, 1, R, 2048)
    assert chain_state.shape[0] >= 8 * R and tuple(chain_state.shape)[1:] == (128, 128)
    assert g.dtype == state.dtype == chain_state.dtype == ttnn.float32
    out = ttnn.empty(
        (1, 1, R, 1024),
        device=state.device(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    count = min(items, 104)  # one (user, value head) item per core; more users walk several items per core
    cores = [ttnn.CoreCoord(h // 10, h % 10) for h in range(count)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    cbs = []
    for i, n in _GDN_CBS.items():
        dtype = ttnn.bfloat16 if i in (20, 21, 22, 24, 27, 29, 30, 31) else ttnn.float32
        size = 2048 if dtype == ttnn.bfloat16 else 4096
        cbs.append(
            ttnn.CBDescriptor(
                total_size=n * size,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=size)],
            )
        )
    rct = []
    for x in (conv, g, beta, state, gate, gamma):
        rct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
    wct = (
        [4, 4, 0]
        + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        + ttnn.TensorAccessorArgs(chain_state).get_compile_time_args()
    )
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for h, c in enumerate(cores):
        rr[c.x][c.y] = [
            h,
            conv.buffer_address(),
            g.buffer_address(),
            beta.buffer_address(),
            state.buffer_address(),
            gate.buffer_address(),
            gamma.buffer_address(),
            count,
            items,
            chain_state.buffer_address(),
            T,
        ]
        wr[c.x][c.y] = [h, 1, out.buffer_address(), chain_state.buffer_address(), items, count, T]
        cr[c.x][c.y] = [len(range(h, items, count)) * T]
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=MATH_FIDELITY, fp32_dest_acc_en=True, math_approx_mode=False
    )
    unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
    for index in (23, 32, 33, 34):
        unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = unpack_modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "gdn_chain/reader.cpp"),
            core_ranges=grid,
            compile_time_args=rct,
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "gdn_chain/writer.cpp"),
            core_ranges=grid,
            compile_time_args=wct,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "flat_ordered_conv_gates/compute.cpp"),
            core_ranges=grid,
            compile_time_args=[4, 4, 0, struct.unpack("<I", struct.pack("<f", epsilon))[0]],
            runtime_args=cr,
            config=compute_config,
        ),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=grid, initial_value=0)]
    ttnn.generic_op(
        [conv, g, beta, state, gate, gamma, chain_state, out],
        ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems),
    )
    return out


def gdn_commit(index, layers, T, users):
    """index: device int32 ROW_MAJOR [1, 32]: accepted candidate a_u of user u (pool row u), -1 = untouched.
    layers: (chain_state, live_state_pool, chain_history, live_history) per GDN layer."""
    n_layers = len(layers)
    rec = [(layer, u) for layer in range(n_layers) for u in range(users)]
    n_cores = 104  # fixed core set (one program for every user count)
    cores = [ttnn.CoreCoord(i // 10, i % 10) for i in range(n_cores)]  # grid is 10 rows tall
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    cbs = [
        ttnn.CBDescriptor(
            total_size=16 * 4096,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4096)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * 2048,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=2048)],
        ),
    ]
    cs, ls, ch, lh = layers[0]
    ct = (
        ttnn.TensorAccessorArgs(index).get_compile_time_args()
        + ttnn.TensorAccessorArgs(ls).get_compile_time_args()
        + ttnn.TensorAccessorArgs(lh).get_compile_time_args()
    )
    rt = ttnn.RuntimeArgs()
    for k, c in enumerate(cores):
        mine = rec[k::n_cores]
        hist = [layers[k][2:]] if k < n_layers else []  # history tiles of layer k: one owner core
        args = [index.buffer_address(), len(mine), len(hist), T]
        for layer, u in mine:
            args += [layers[layer][0].buffer_address(), layers[layer][1].buffer_address(), u]
        for chain_hist, live_hist in hist:
            args += [chain_hist.buffer_address(), live_hist.buffer_address()]
        # generic_op caches the program by kernels and cores, not by runtime-argument count: pad every core's list to
        # the largest (32 users) so all user counts share one program (counts above say what is real)
        max_rec = -(-n_layers * 32 // n_cores)
        args += [0] * (4 + 3 * max_rec + 2 - len(args))
        rt[c.x][c.y] = args
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "gdn_commit/copy.cpp"),
        core_ranges=grid,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    tensors = [index] + [t for layer in layers for t in layer]
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs, semaphores=[]))
