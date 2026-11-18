# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B40 (HY4_B40_DERIVE=1, default OFF): the integer half of ``SpecDerive.derive`` (write rows, key counts, col info, per-row key-append indices, rope hi/lo indices) as ONE one-core program,
bit-identical to the stock ~50 tiny elementwise ops (kernel hy4_derive_kernels/derive_geom.cpp)."""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_derive_kernels")


def enabled():
    return os.environ.get("HY4_B40_DERIVE", "0") == "1"


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def rope_enabled():
    return os.environ.get("HY4_B47D_ROPE", "0") == "1"


def derive_geom(p, own_col, wr, kvcnt, colinfo, ih, il, upd_rows, S, CH, SL, B2, rope=None):
    """rope (B47d): None or (cos_tab, sin_tab, cos_out, sin_out, c128_out, s128_out, first, pmax): the same program also gathers the bf16 cos / sin rows of
    positions p + j from the final-value tables ([P, 64] bf16 RM, replicated) and writes the four persistent rope tabs (TILE bf16 [S, 64] / [S, 128]).
    """
    assert 1 <= S <= 4 and len(upd_rows) >= S
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cbsz = 64 * 16 if rope is None else 4096
    cb = ttnn.CBDescriptor(
        total_size=cbsz,
        core_ranges=core,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=cbsz)],
    )
    ups = [upd_rows[j] for j in range(S)]
    ct = (
        [0, S, CH, SL, B2]
        + _acc(p)
        + _acc(own_col)
        + _acc(wr)
        + _acc(kvcnt)
        + _acc(colinfo)
        + _acc(ih)
        + _acc(il)
        + _acc(ups[0])
    )
    common = [t.buffer_address() for t in (p, own_col, wr, kvcnt, colinfo, ih, il)] + [t.buffer_address() for t in ups]
    rts = []
    if rope is not None:
        rts = list(rope[:6])
        ct = ct + [1, int(bool(rope[6])), int(rope[7])] + sum((_acc(t) for t in rts), [])
        common = common + [0] * (11 - len(common)) + [t.buffer_address() for t in rts]
    else:
        ct = ct + [0, 0, 0] + _acc(p) * 6  # unused accessor slots (never constructed at ROPE=0)
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/derive_geom.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=ct,
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (0x4B40 << 40) | (hash(tuple(ct)) & ((1 << 40) - 1))
    ttnn.generic_op([p, own_col, wr, kvcnt, colinfo, ih, il] + ups + rts, prog)


def tail_enabled():
    return os.environ.get("HY4_B40_TAIL", "0") == "1"


def _prog(kernel, ct, common, core, cbbytes, tag):
    cb = ttnn.CBDescriptor(
        total_size=cbbytes,
        core_ranges=core,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=cbbytes)],
    )
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{kernel}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=ct,
        runtime_args=ttnn.RuntimeArgs(),
        common_runtime_args=common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (tag << 40) | (hash(tuple(ct)) & ((1 << 40) - 1))
    return prog


def spec_accept(a_tok, d_row, cap, pos, a_out, m_out, oh_col, p_new, nxt, S, k):
    """one-core greedy accept (see spec_accept.cpp): writes a_out, m_out, oh_col (bf16 [1,1,S,1]), p_new = pos + m + 1, nxt = a_m."""
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    ts = [a_tok, d_row, cap, pos, a_out, m_out, oh_col, p_new, nxt]
    ct = [0, S, k] + sum((_acc(t) for t in ts), [])
    prog = _prog("spec_accept.cpp", ct, [t.buffer_address() for t in ts], core, 64 * 24, 0x4B41)
    ttnn.generic_op(ts, prog)


def spec_state(drafts, nxt, p_new, park, parkvec, d_row, tok, pos, S, k, pad=120002):
    """one-core next-round state (see spec_state.cpp): writes d_row, tok (uint32 RM [1,S]) and pos."""
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    haspv = parkvec is not None
    pv = parkvec if haspv else park
    ins = list(drafts) + [nxt, p_new, park] + ([parkvec] if haspv else [])
    ct = (
        [0, S, k, pad, int(haspv)]
        + _acc(drafts[0])
        + sum((_acc(t) for t in (nxt, p_new, park, pv, d_row, tok, pos)), [])
    )
    common = [t.buffer_address() for t in ins] + [t.buffer_address() for t in (d_row, tok, pos)]
    prog = _prog("spec_state.cpp", ct, common, core, 64 * 12, 0x4B42)
    ttnn.generic_op(ins + [d_row, tok, pos], prog)
