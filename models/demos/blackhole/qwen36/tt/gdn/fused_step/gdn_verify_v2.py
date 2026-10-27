# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused Gated DeltaNet spec-VERIFY step over T rows, bit-exact to T fused v2 decode steps (QWEN36_GDN_FUSED_VERIFY=1).

One program between the GDN in-projection (the decode qkvzab matmul over the T candidate rows, unsliced) and the
out-projection, batch 1, fp32 state, on the v2 core grid (Nv * Vt cores; 16 on the TP8 Qwen3.5-9B pup). Row t of the
verify runs EXACTLY the v2 decode step's arithmetic (gdn_step_v2.py, kernels/gdn_step_v2_*.cpp) for token t from the
state token t-1 left: the compute kernel is the v2 compute body in a token loop, with the same dense tile layout,
the same constants (shared with the layer's GDNFusedDecodeStepV2 object), the same op order and the same fp32 math.

Differences from T calls of v2, none of which changes a bit of the result:
* State carry: token t's new state is packed (fp32, exact) both to the writer (-> ring) and to the next token's input
  CB, instead of a DRAM round trip through rec_state. The initial state is the ring block state_blk_idx names.
* Conv history: the reader loads every channel tile it needs ONCE (E_prev = _verify_win_buf, the new rows = proj),
  full 2 KB tiles, and builds token t's dense tiles in L1 (RISC-V copies of 32-byte rows): history slots H1, H2, H3
  are rows t, t+1, t+2 of the virtual window E_new = [E_prev rows mi+1 .. mi+3 ; proj rows 0 .. T-1], X is proj
  row t. The dense tiles' odd rows are zero here (v2: the decode tile's finite padding row 1); their taps are 0 and
  nothing reads them, so every even (used) row is bit-identical.
* mi (the last accepted row) comes from state_blk_idx itself: for B = 1, state_blk_idx[h] = mi * Nv + h, so
  mi = state_blk_idx[0] / Nv -- the same host value spec_conv_sel(mi) encodes (conv_sel is not read).
* Side effects match the composite verify: E_new (the full [K-1+T] window, zero padding rows) is written into
  _verify_win_buf in place by the channel owners after every reader of the key group has loaded E_prev (v2's
  ownership + semaphore); token t's state goes to ring block t * Nv + h (spec_state_blk_idx layout, B = 1).
* The gated RMSNorm partial sums are exchanged per token through two alternating gather tiles and a cumulative
  semaphore count ((t + 1) * (Vt - 1)), so a fast peer one token ahead cannot clobber a slot still being read.
* The output is the gated (normed, silu(z)-multiplied) bf16 [1, T, value_dim_tp]: row t = v2's row 0 for token t.
"""
from __future__ import annotations

import os
from pathlib import Path

import ttnn

from .gdn_step_v2 import KERNELS as _V2_KERNELS
from .gdn_step_v2 import TILE, GDNFusedDecodeStepV2, f32_bits

KERNELS = Path(os.environ.get("QWEN36_GDN_FUSED_VERIFY_KERNELS") or _V2_KERNELS)
# CB ids shared with kernels/gdn_verify_v2_*.cpp (v2 ids + the verify's own)
CB = dict(
    h=0,
    x=1,
    tap=2,
    abraw=3,
    wscr=4,
    out=5,  # bf16
    const=6,
    c=7,
    t1=8,
    t2=9,
    n=10,
    aq=11,
    ak=12,
    kcol=13,
    v=14,
    sz=15,
    a=16,
    b=17,
    na=18,
    dt=19,
    decay=20,
    beta=21,
    S=22,
    sd=23,
    vread=24,
    u=25,
    snew=26,
    state=27,
    o=28,
    sq=29,
    ps=30,
    gather=31,
    r=32,
    w=33,
    ep=34,
    pj=35,
    oasm=36,
    idx=37,  # bf16 / raw
    Sn=38,  # fp32: state carry between tokens
)
SEM_RD, SEM_X = 0, 1
MAX_T = 29  # E_prev rows K-1+T must fit one tile row


def v2_step_of(gdn):
    """The layer's GDNFusedDecodeStepV2 (shared constants); built (host writes, so eagerly) on first use."""
    st = getattr(gdn, "_fused_step", None)
    if isinstance(st, GDNFusedDecodeStepV2):
        return st
    st = getattr(gdn, "_fused_verify_v2step", None)
    if st is None:
        st = GDNFusedDecodeStepV2(gdn)
        if getattr(gdn, "_fused_step", None) is None and os.environ.get("QWEN36_GDN_FUSED_STEP_V2") == "1":
            gdn._fused_step = st  # the decode step would build the same object
        gdn._fused_verify_v2step = st
    return st


class GDNFusedVerifyV2:
    def __init__(self, step: GDNFusedDecodeStepV2):
        self.s = step
        self.gdn = step.gdn
        self.mesh = step.mesh
        g = self.gdn
        assert g.B == 1, "the fused verify is batch 1"
        self.NS = 2 * step.Kt + 1

    def _cbs(self, T):
        s = self.s
        Kt, NS = s.Kt, self.NS
        nown = s.n_own_qk_max + 1
        bf = {"h": 3, "x": 1, "tap": 4, "abraw": 1, "wscr": nown, "out": 1, "ep": NS, "pj": NS + 1, "oasm": 1, "idx": 1}
        f32 = {
            "const": 6,
            "c": 1,
            "t1": 1,
            "t2": 1,
            "n": 1,
            "aq": Kt,
            "ak": Kt,
            "kcol": Kt,
            "v": 1,
            "sz": 1,
            "a": 1,
            "b": 1,
            "na": 1,
            "dt": 1,
            "decay": 1,
            "beta": 1,
            "S": Kt,
            "sd": Kt,
            "vread": 1,
            "u": 1,
            "snew": Kt,
            "state": Kt,
            "o": 1,
            "sq": 1,
            "ps": 1,
            "gather": 2,
            "r": 1,
            "w": 1,
            "Sn": Kt,
        }
        cbs = []
        for names, dtype, page in ((bf, ttnn.bfloat16, 2048), (f32, ttnn.float32, 4096)):
            for name, n in names.items():
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=n * page,
                        core_ranges=s.core_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=CB[name], data_format=dtype, page_size=page)
                        ],
                    )
                )
        return cbs

    def __call__(self, proj, state_blk_idx, T, out=None):
        """proj: [1, T, qkvz_dim_tp + 2 Nv] bf16 TILE (the decode in-projection of the T rows, unsliced);
        state_blk_idx: uint32 ROW_MAJOR [Nv] (spec_state_blk_idx, B = 1). Reads/writes gdn._verify_win_buf and
        gdn._spec_ring in place. Returns the gated bf16 [1, T, value_dim_tp]."""
        s, g = self.s, self.gdn
        ring, win = g._spec_ring, g._verify_win_buf
        assert ring is not None and g._spec_shape == (1, T), f"spec buffers {g._spec_shape} vs (1, {T})"
        assert ring.dtype == ttnn.float32
        assert 1 <= T <= MAX_T and g.K - 1 + T <= TILE
        assert proj.shape[-1] == g.qkvz_dim_tp + 2 * s.Nv, f"in-projection width {proj.shape[-1]}"
        assert proj.shape[-2] == T, f"proj rows {proj.shape[-2]} != T {T}"
        if out is None:
            # per call, like the v2 decode step's output (a trace-captured device allocation, no host write). Not a
            # persistent buffer: the decode out-projection (tpc.matmul_1d_decode) force-deallocates an input that is
            # already L1-interleaved, which would free a persistent buffer after its first use.
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, T, g.value_dim_tp]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, ttnn.L1_MEMORY_CONFIG
            )
        Kt, Vt, NS = s.Kt, s.Vt, self.NS
        reader_ct = [Kt, Vt, s.z_off, s.ab_tile, s.a_col, s.Nv, SEM_RD, s.G, T]
        for t in (proj, win, s.taps_dense, s.gconst, s.normw, s.consts, ring, state_blk_idx):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [Kt, Vt, SEM_RD, SEM_X, s.G, T, s.Nv]
        for t in (out, ring, win):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [
            Kt,
            f32_bits(1e-6),
            f32_bits(1.0 / s.Dv),
            f32_bits(1e-6),
            f32_bits(1.0),
            f32_bits(20.0),
            T,
        ]
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        a = lambda t: t.buffer_address()  # noqa: E731
        for h, j, core in s.cores:
            kh = h // s.rf
            chans = s._channels(h, kh, j)
            group = [s.phys[(hh, jj)] for hh in range(kh * s.rf, (kh + 1) * s.rf) for jj in range(Vt)]
            peers = [s.phys[(h, jj)] for jj in range(Vt) if jj != j]
            rd[core.x][core.y] = (
                [
                    a(proj),
                    a(win),
                    a(s.taps_dense),
                    a(s.gconst),
                    a(s.normw),
                    a(s.consts),
                    a(ring),
                    a(state_blk_idx),
                    h,
                    j,
                    (h * Vt + j),
                ]
                + chans
                + [c for p in group for c in (p.x, p.y)]
            )
            member = (h - kh * s.rf) * Vt + j
            own_slots = [c for c in range(2 * Kt) if c % s.G == member] + [2 * Kt]
            wr[core.x][core.y] = (
                [a(out), a(ring), a(win), h, j, len(own_slots)]
                + [x for sl in own_slots for x in (sl, chans[sl])]
                + [c for p in peers for c in (p.x, p.y)]
            )
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_verify_v2_reader.cpp"),
                core_ranges=s.core_set,
                compile_time_args=reader_ct,
                runtime_args=rd,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_verify_v2_writer.cpp"),
                core_ranges=s.core_set,
                compile_time_args=writer_ct,
                runtime_args=wr,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "gdn_verify_v2_compute.cpp"),
                core_ranges=s.core_set,
                compile_time_args=compute_ct,
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=s.math_fidelity, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=[
                ttnn.SemaphoreDescriptor(SEM_RD, ttnn.CoreType.WORKER, s.core_set, 0),
                ttnn.SemaphoreDescriptor(SEM_X, ttnn.CoreType.WORKER, s.core_set, 0),
            ],
            cbs=self._cbs(T),
        )
        mesh_program = ttnn.MeshProgramDescriptor()
        mesh_program[s.mesh_range] = program
        io = [proj, win, s.taps_dense, s.gconst, s.normw, s.consts, ring, state_blk_idx, out]
        ttnn.generic_op(io, mesh_program)
        return out


# --------------------------------------------------------------------------------------------------------------- #
# tp.py hooks (QWEN36_GDN_FUSED_VERIFY=1): the verify and the seed with the fused v2 decode arithmetic
# --------------------------------------------------------------------------------------------------------------- #
def _project_rows(gdn, x, valid_len, pre_gathered):
    """x -> the unsliced decode in-projection [1, R, qkvz + 2 Nv] of the R = valid_len real rows (the matmul
    _forward_decode_fused runs; row-independent, so row t equals a one-row decode projection). Returns (proj, bucket).
    """
    import models.demos.blackhole.qwen36.tt.tp_common as tpc
    from models.tt_transformers.tt.ccl import tt_all_gather

    mc, _L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    if gdn.conv_states is None:
        gdn.reset_state()
    x_owned = False
    if pre_gathered and x.is_sharded():
        x = ttnn.to_memory_config(x, mc)
        x_owned = True
    if len(x.shape) == 4:
        x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
    bucket = x.shape[-2]
    R = valid_len
    xv = x if R == bucket else ttnn.slice(x, (0, 0, 0), (1, R, x.shape[-1]))
    if pre_gathered:
        xg = xv
    else:
        xg = tt_all_gather(
            xv, gdn.mesh, gdn.tt_ccl, cluster_axis=None, dim=-1, topology=gdn.args.ccl_topology(), memory_config=mc
        )
        if xv is not x:
            ttnn.deallocate(xv)
    if getattr(gdn.args, "proj_1d_decode", False):
        proj = tpc.matmul_1d_decode(
            xg, gdn.tw["qkvz"], gdn.args.gdn_qkvz_decode_1d_progcfg, gdn.cfg, out_memory_config=_L1
        )
    else:
        proj = gdn._col_proj(xg, gdn.tw["qkvz"], gdn.args.gdn_qkvzab_progcfg, out_memory_config=_L1)
    if xg is not x:
        ttnn.deallocate(xg)
    if x_owned:
        ttnn.deallocate(x)
    return proj, bucket


def _tail(gdn, gated, R, bucket, free_gated):
    """Out-projection + the DECODE reduce (tpc.reduce_partials, as _forward_decode_fused), zero-padded to bucket."""
    import models.demos.blackhole.qwen36.tt.tp_common as tpc

    mc, rm = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
    partial = gdn._row_proj(gated, gdn.tw["out"])
    if free_gated:
        ttnn.deallocate(gated)
    partial = ttnn.reshape(partial, (1, 1, R, partial.shape[-1]))
    o_red = tpc.reduce_partials(partial, gdn.mesh, gdn.tt_ccl, gdn.args.ccl_topology())
    if R < bucket:
        o_rm = ttnn.to_layout(o_red, rm)
        ttnn.deallocate(o_red)
        pad = gdn._verify_pad_buf(bucket - R, o_rm.shape[-1], o_rm.dtype, rm, mc)
        o_full = ttnn.concat([o_rm, pad], dim=2, memory_config=mc)
        ttnn.deallocate(o_rm)
        o_red = ttnn.to_memory_config(ttnn.to_layout(o_full, ttnn.TILE_LAYOUT), mc)
        ttnn.deallocate(o_full)
    return o_red


def fused_verify_gated(gdn, proj, state_blk_idx, T):
    """The verify program alone (proj rows -> gated rows; ring and window side effects)."""
    fv = getattr(gdn, "_fused_verify", None)
    if fv is None:
        fv = gdn._fused_verify = GDNFusedVerifyV2(v2_step_of(gdn))
    return fv(proj, state_blk_idx, T)


def fused_verify_forward(gdn, x, valid_len, pre_gathered, T, state_blk_idx):
    """forward_verify_recurrent with QWEN36_GDN_FUSED_VERIFY=1 (B = 1): decode in-projection of the T rows, the fused
    verify program, out-projection + decode reduce. Same side effects as _verify_fullbatch (ring, _verify_win_buf)."""
    assert gdn.B == 1, "QWEN36_GDN_FUSED_VERIFY is batch 1"
    assert state_blk_idx is not None, "the fused verify needs state_blk_idx"
    proj, bucket = _project_rows(gdn, x, valid_len, pre_gathered)
    gated = fused_verify_gated(gdn, proj, state_blk_idx, T)
    ttnn.deallocate(proj)
    return _tail(gdn, gated, T, bucket, free_gated=True)


def fused_seed_forward(gdn, x, valid_len, pre_gathered):
    """forward_seed_recurrent with QWEN36_GDN_FUSED_VERIFY=1: ONE fused v2 decode step on the durable state (the
    verify rows are v2 steps, so the seed is too), then both conv mirrors made current, as _seed_fullbatch leaves them.
    """
    assert gdn.B == 1 and valid_len == 1
    gdn._ensure_conv_win()
    gdn.sync_conv_taps()
    proj, bucket = _project_rows(gdn, x, valid_len, pre_gathered)
    gated = v2_step_of(gdn)(proj)
    ttnn.deallocate(proj)
    gdn._conv_win_stale = True  # the step shifted the taps
    gdn.sync_conv_win()
    gdn._conv_taps_stale, gdn._conv_win_stale = False, False
    return _tail(gdn, gated, 1, bucket, free_gated=True)
