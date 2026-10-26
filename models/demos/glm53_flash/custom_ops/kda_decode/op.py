# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``KdaDecode``: the fused KDA (Kimi Delta Attention) single-token decode block as a model-local
``ttnn.generic_op`` (kernels in ``kernels/``; contract in README.md).

One program does everything between the packed projection matmuls and o_proj: the causal conv (kernel 4)
+ SiLU on the device-owned bf16 history (tap-major tiles, shifted in place, raw tiles appended), the
per-head L2 norms, the per-key-channel forget gate ``g = lb * sigmoid(exp(A_log) (f + dt_bias))``,
``beta = sigmoid(b)``, the gated delta-rule step per user on the recurrent state (read once, written once,
in place) and the gated RMSNorm ``rmsnorm(o) * w * sigmoid(og)``, packed as bf16 TILE output. Work items
are (head, 32-user tile row, user part); the parts of a (head, row) slot sit in one column of cores: the
part-0 core reads the slot's inputs once and multicasts them, and sums the parts' output rows for the norm.

Options (``tt/kda.py`` reads them from ``GLM53_KDA_DECODE_*``; every default is the path above). With every option at
its default the op runs today's kernels (``kernels/{reader,writer,compute}_kda_decode.cpp``) with today's argument
layouts, byte for byte; any non-default option selects the variant kernels ``reader_kda_decode_v2.cpp`` /
``compute_kda_decode_v2.cpp`` (compile-time arg 11 ``GATES_ACT``, reader runtime args 15..19 for the second multicast
column) so a process that imported an older ``op.py`` keeps compiling the unchanged default sources:

* ``gates="act"``: the ``gates`` input already holds ``[sigmoid(a (f + dt)) | sigmoid(og) | sigmoid(b)]`` (the
  channel constants folded into the g2 projection by ``prearrange_gate_weights`` and the sigmoid fused into that
  matmul); the program then only forms ``exp(lb * s)`` and ``s_og * w`` and never reads ``chan_consts``.
* ``placement="rect"``: a slot's parts are a 2-column rectangle (``2 x h`` cores, the part-0 core top-left,
  multicast as two single-column rectangles) when that gives more parts than the column layout (16 slots on a
  12x10 grid: 6 parts / 96 cores instead of 5 / 80, i.e. 6 users per part at B = 32 instead of 7).
* ``gated=True`` (the speculative commit, ``tt/spec_decode.py`` under ``GLM53_SPEC_COMMIT=gated``): a per-user write
  gate ``gate [Bt, 32]`` int32 ROW_MAJOR (page ``tb`` = the 32 users of tile row ``tb``, non-zero = write) read by the
  kernels at run time: a user whose gate is 0 gets its recurrent state tiles and its conv-history rows left byte for
  byte as they were (the writer skips the state write, the part-0 reader merges the history shift row-wise in a scratch
  region of ``cb_onw``), a user whose gate is 1 gets exactly the plain update. Sources ``reader_kda_decode_gated.cpp``
  / ``writer_kda_decode_gated.cpp`` + the ``_v2`` compute kernel (compile-time arg 11 ``GATES_ACT`` as v2, one more
  accessor block for the gate, runtime arg 20 (reader) / 7 (writer) = the gate address).
* ``fused_steps=T`` (SP2, the speculative loop under ``GLM53_SPEC_COMMIT=fused``): ONE launch on one tile row of
  ``R = USERS x T`` candidate-major rows (row ``j USERS + u`` = user ``u``, candidate ``j``) applies, per user, its
  candidates in order with the recurrent state kept in L1 across the steps (read once, written once at most): the
  reader assembles the three conv tap blocks of the R rows (history for ``j + t < 3``, the earlier raw rows otherwise)
  so the per-item phases run once on the R-row block, the compute loops (user, step) with the state ping-ponged
  through a two-slot loop CB (index 28, the v1 / v2 ``oq``; ``qd . S`` rides ``cb_vread``), the output block holds
  row ``j USERS + u`` of every step. ``fused_dry=True`` (the verify): ``T`` steps per user, the state and history
  are never written; ``fused_dry=False`` (the commit): a step page ``steps [Bt, 32]`` int32 ROW_MAJOR (word ``u`` =
  ``n_u``, the steps user ``u`` takes, 0 = untouched) read by the kernels at run time: user ``u`` ends at the plain
  state after its ``n_u`` candidates (bit for bit the sequential launches' result), the history shifted by ``n_u``.
  Sources ``{reader,writer,compute}_kda_decode_fused.cpp``; compile-time ``USERS, STEPS, DRY`` at 12 / 13 / 14
  (reader, compute; the reader's accessors from 15) and 7 / 8 / 9 (writer; accessors from 10), one more accessor
  block (the step page), runtime arg 20 (reader) / 7 (writer) = the page address; ``golden_fused`` is the torch
  reference.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

import ttnn
from models.demos.glm53_flash.custom_ops.core_ranges import rect_core_ranges

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
GATES_MODES = ("pre", "act")
PLACEMENTS = ("col", "rect")
GATE_WORDS = 32  # a gate page: one int32 per user row of a 32-row tile row ([Bt, 32] int32 ROW_MAJOR, page = tile row)
GATE_BYTES = GATE_WORDS * 4


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def selector_tiles() -> torch.Tensor:
    """[33*32, 32] constant (used by the DSA state op): tile r < 32 has a 1 at (0, r) (E_r lifts row r of a block
    into row 0; its transpose places row 0 at row r), tile 32 is all ones (row sums by matmul)."""
    sel = torch.zeros(33, TILE, TILE)
    sel[torch.arange(TILE), 0, torch.arange(TILE)] = 1.0
    sel[32] = 1.0
    return sel.reshape(33 * TILE, TILE)


def kda_selector_tiles() -> torch.Tensor:
    """[34*32, 32] constant of ``KdaDecode``: tile r < 32 is E_r^T (a 1 at (r, 0): e_r in column 0, i.e. a row mask
    under a column broadcast and a column picker under a matmul), tile 32 is all ones (row sums by matmul), tile
    33 is the identity (diagonal tiles by a column broadcast)."""
    sel = torch.zeros(34, TILE, TILE)
    sel[torch.arange(TILE), torch.arange(TILE), 0] = 1.0
    sel[32] = 1.0
    sel[33] = torch.eye(TILE)
    return sel.reshape(34 * TILE, TILE)


ONES_PAGE = 32  # selector page of the all-ones tile (identity at 33)


def prearrange_gate_weights(
    w_g2: torch.Tensor, a_exp: torch.Tensor, dt_bias: torch.Tensor, hk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``gates="act"`` pre-arrangement of one chip's g2 block ``w_g2 [d, N]`` (columns ``[f_b | g_b | I]``, the
    first ``hk`` = this chip's key channels): ``a (f + dt) = (a w)^T g1 + a dt`` per key channel, so the f columns
    are scaled by ``a_exp [hk]`` and a bias row ``[a dt_bias | 0 ...] [N]`` is added before the fused sigmoid.
    Both returned in fp32 (``w`` exact up to fp32 products; the caller rounds ``w`` to the matmul dtype)."""
    w = w_g2.float().clone()
    assert a_exp.numel() == hk and dt_bias.numel() == hk and w.shape[1] >= hk, (w.shape, a_exp.shape, hk)
    a, dt = a_exp.float().reshape(-1), dt_bias.float().reshape(-1)
    w[:, :hk] = w[:, :hk] * a[None, :]
    bias = torch.zeros(w.shape[1], dtype=torch.float32)
    bias[:hk] = a * dt
    return w, bias


class KdaDecode:
    """Builds and launches the fused KDA decode program. All tensors interleaved (DRAM or L1); K = V = 128,
    conv kernel 4 (3 history taps). Call ``__call__`` with a pre-allocated output."""

    # CB plan (kept in sync with the kernels)
    CB = dict(
        hist=0, taps=1, gf=2, gog=3, gb=4, cc=5, onw=6, sel=7, selT=8, S=9, out=10, state_out=11, qn=12, kn=13,
        vc=14, dec=15, bcol=16, wg=17, tmp=18, col=19, kd=20, qd=21, kcol=22, decT=23, gather=24, dcol=25, pq=26,
        vread=27, oq=28, u=29, oacc=30, ofin=31,
    )  # fmt: skip
    # per-core semaphores: parts -> part-0 "ready", part-0 -> parts "valid" (multicast of the constant "one"),
    # parts -> part-0 "gather" (output blocks delivered)
    SEM_READY, SEM_VALID, SEM_ONE, SEM_GATHER = 0, 1, 2, 3
    # runtime-arg layout (indices; the kernels read the same): the v1 (default) reader takes the first 15 entries,
    # the v2 reader all 20
    READER_RT = dict(
        item0=0, n_items=1, stride=2, qkv=3, hist=4, taps=5, gates=6, cc=7, onw=8, sel=9, st=10, p0_x=11, p0_y=12,
        pe_x=13, pe_y=14, rb_x0=15, rb_y0=16, rb_x1=17, rb_y1=18, n_b=19,
    )  # fmt: skip
    READER_RT_LEN = {"v1": 15, "v2": 20, "gated": 21, "fused": 21}
    READER_RT_GATED = dict(READER_RT, gate=20)  # the gated reader: the v2 vector + the gate address
    READER_RT_FUSED = dict(READER_RT, steps=20)  # the fused reader: the v2 vector + the step page address
    # compile-time scalar counts before the accessor block (reader / writer) / in total (compute) per variant
    READER_CT_SCALARS = {"v1": 11, "v2": 12, "gated": 12, "fused": 15}
    WRITER_CT_SCALARS = {"v1": 7, "v2": 7, "gated": 7, "fused": 10}
    COMPUTE_CT_LEN = {"v1": 11, "v2": 12, "gated": 12, "fused": 15}
    WRITER_RT = dict(item0=0, n_items=1, stride=2, out=3, st=4, p0_x=5, p0_y=6)
    WRITER_RT_GATED = dict(WRITER_RT, gate=7)  # the gated writer: today's vector + the gate address
    WRITER_RT_FUSED = dict(WRITER_RT, steps=7)  # the fused writer: today's vector + the step page address
    WRITER_RT_LEN = {"v1": 7, "v2": 7, "gated": 8, "fused": 8}
    COMPUTE_RT = dict(item0=0, n_items=1, stride=2)
    # accessor blocks per kernel: (reader, writer); the gated / fused kernels take one more (the gate / step page
    # tensor, appended)
    READER_ACCESSORS = ("qkv", "conv_hist", "taps", "gates", "chan_consts", "o_norm_w", "selectors", "recurrent")
    WRITER_ACCESSORS = ("out", "recurrent")
    # the fused variant's CB plan: index 28 is the two-slot loop state (the v1 / v2 "oq": qd . S rides cb_vread)
    CB_FUSED = {("loop" if n == "oq" else n): i for n, i in CB.items()}
    FUSED_SELT_PAGES = 3  # [the step page (commit mode),] E_c^T, E_r^T per step
    MAX_FUSED_ROWS = TILE  # USERS x STEPS <= 32: one tile row of candidate rows
    # the gated variant's scratch region: fp32 pages appended to cb_onw after its Vt weight tiles (the CB protocol never
    # reserves them: one reserve / push of Vt pages per kernel run) = 1 page holding the gate page (the reader's copy at
    # +0, the writer's at +2048) + the 9 Kt merged bf16 history tiles (3 branches x 3 destination taps x Kt)
    GATED_SCRATCH_PAGES = 1 + (9 * (128 // TILE) * TILE * TILE * 2) // (TILE * TILE * 4)
    # the fused variant's scratch region (the same layout): the step page (reader +0, writer +2048) + 9 Kt bf16 tiles
    # = the live history tiles of the item (the tap-block assembly source), then the merged new history tiles
    FUSED_SCRATCH_PAGES = GATED_SCRATCH_PAGES

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int = 128,
        gate_lower_bound: float = -5.0,
        l2_eps: float = 1e-6,
        norm_eps: float = 1e-5,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        gates: str = "pre",
        placement: str = "col",
        gated: bool = False,
        fused_steps: int = 0,
        fused_dry: bool = False,
    ):
        self.H, self.K = num_heads, head_dim
        self.V = head_dim
        assert head_dim == 128, "the kernels are written for K = V = 128"
        assert gates in GATES_MODES and placement in PLACEMENTS, (gates, placement)
        self.lb, self.l2_eps, self.norm_eps = gate_lower_bound, l2_eps, norm_eps
        self.fidelity = math_fidelity
        self.gates, self.placement_mode = gates, placement
        self.gated = bool(gated)
        self.fused_steps = int(fused_steps)  # T = K + 1 candidates per user (0: not the fused variant)
        self.fused_dry = bool(fused_dry)
        assert self.fused_steps >= 0 and not (self.gated and self.fused_steps), "gated and fused are exclusive"
        assert self.fused_steps <= self.MAX_FUSED_ROWS, self.fused_steps

    @property
    def gates_activated(self) -> bool:
        return self.gates == "act"

    @property
    def variant(self) -> str:
        """``"v1"``: every option at its default -> today's kernel sources and argument layouts; ``"v2"``: the knob
        variant sources (``*_v2.cpp``) with compile-time arg 11 ``GATES_ACT`` and the reader's runtime args 15..19;
        ``"gated"``: the per-user write-gate sources (``*_gated.cpp`` + the v2 compute) with the gate accessor / address;
        ``"fused"``: the T = K + 1 steps-per-launch sources (``*_fused.cpp``) with the step-page accessor / address.
        """
        if self.gated:
            return "gated"
        if self.fused_steps:
            return "fused"
        return "v1" if (self.gates == "pre" and self.placement_mode == "col") else "v2"

    @property
    def fused(self) -> bool:
        return self.fused_steps > 0

    def kernel_sources(self) -> tuple[str, str, str]:
        """``(reader, writer, compute)`` kernel file names of this variant (the writer is shared by v1 / v2)."""
        if self.gated:
            return "reader_kda_decode_gated.cpp", "writer_kda_decode_gated.cpp", "compute_kda_decode_v2.cpp"
        if self.fused:
            return "reader_kda_decode_fused.cpp", "writer_kda_decode_fused.cpp", "compute_kda_decode_fused.cpp"
        sfx = "" if self.variant == "v1" else "_v2"
        return f"reader_kda_decode{sfx}.cpp", "writer_kda_decode.cpp", f"compute_kda_decode{sfx}.cpp"

    def gated_twin(self) -> "KdaDecode":
        """The same program options with the per-user write gate (the speculative commit's launches)."""
        return KdaDecode(
            num_heads=self.H,
            head_dim=self.K,
            gate_lower_bound=self.lb,
            l2_eps=self.l2_eps,
            norm_eps=self.norm_eps,
            math_fidelity=self.fidelity,
            gates=self.gates,
            placement=self.placement_mode,
            gated=True,
        )

    def fused_twin(self, steps: int, dry: bool) -> "KdaDecode":
        """The same program options as the fused T = ``steps`` launch (``dry``: the verify's read-only launch, else the
        commit's step-page launch)."""
        return KdaDecode(
            num_heads=self.H,
            head_dim=self.K,
            gate_lower_bound=self.lb,
            l2_eps=self.l2_eps,
            norm_eps=self.norm_eps,
            math_fidelity=self.fidelity,
            gates=self.gates,
            placement=self.placement_mode,
            fused_steps=int(steps),
            fused_dry=bool(dry),
        )

    # ------------------------------------------------------------------ reference
    @staticmethod
    def golden(
        qkv,
        gates,
        conv,
        S,
        taps,
        dt_bias,
        a_exp,
        onw,
        *,
        num_heads: int,
        gate_lower_bound=-5.0,
        l2_eps=1e-6,
        norm_eps=1e-5,
        gates_activated: bool = False,
    ):
        """Torch reference of one step. qkv [B, 3*H*K] (bf16-valued), gates [B, H*K + H*V + Hb] fp32, conv [B, 3, 3*H*K],
        S [B, H, K, V], taps [4, 3*H*K], dt_bias / a_exp [H*K], onw [V]. Returns (o [B, H*V], new state, new conv).
        ``gates_activated``: ``gates`` already holds ``[sigmoid(a (f + dt)) | sigmoid(og) | sigmoid(b)]`` (the
        ``gates="act"`` contract; ``dt_bias`` / ``a_exp`` are then ignored)."""
        B, H = qkv.shape[0], num_heads
        K = V = taps.shape[-1] // (3 * H)
        rows = torch.cat([conv, qkv[:, None, :]], dim=1)  # [B, 4, W], oldest first
        cv = torch.nn.functional.silu((rows * taps[None]).sum(1))
        q, k, v = (t.reshape(B, H, K) for t in cv.split(H * K, dim=-1))
        l2 = lambda t: t / torch.sqrt((t * t).sum(-1, keepdim=True) + l2_eps)
        q = l2(q) * K**-0.5
        k = l2(k)
        f = gates[:, : H * K]
        og = gates[:, H * K : H * K + H * V].reshape(B, H, V)
        b = gates[:, H * K + H * V : H * K + H * V + H]
        if gates_activated:
            decay = torch.exp(gate_lower_bound * f).reshape(B, H, K)
            beta, gate = b, og
        else:
            decay = torch.exp(gate_lower_bound * torch.sigmoid(a_exp * (f + dt_bias))).reshape(B, H, K)
            beta = torch.sigmoid(b)
            gate = torch.sigmoid(og)
        Sd = S * decay[..., None]
        vread = (Sd * k[..., None]).sum(-2)
        u = beta[..., None] * (v - vread)
        Sn = Sd + k[..., None] * u[:, :, None, :]
        o = (Sn * q[..., None]).sum(-2)
        o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + norm_eps) * onw * gate
        new_conv = torch.cat([conv[:, 1:], qkv[:, None, :]], dim=1)
        return o.reshape(B, H * V), Sn, new_conv

    @staticmethod
    def golden_gated(qkv, gates, conv, S, taps, dt_bias, a_exp, onw, gate, **kw):
        """``golden`` with the per-user write gate ``gate [B]`` (non-zero = write): the returned state / conv history of a
        user whose gate is 0 is its input, bit for bit; a user whose gate is 1 gets the plain update. The output rows of
        gated-off users are the plain launch's (the kernel computes them too; the commit discards every output)."""
        o, Sn, conv_n = KdaDecode.golden(qkv, gates, conv, S, taps, dt_bias, a_exp, onw, **kw)
        on = torch.as_tensor(gate).reshape(-1).bool()
        assert on.numel() == S.shape[0], (on.numel(), S.shape)
        Sg = torch.where(on[:, None, None, None], Sn, S)
        convg = torch.where(on[:, None, None], conv_n, conv)
        return o, Sg, convg

    @staticmethod
    def golden_fused(qkv, gates, conv, S, taps, dt_bias, a_exp, onw, *, steps, dry: bool, steps_per_user: int, **kw):
        """Torch reference of one fused launch on ``R = B x T`` candidate-major rows (``T = steps_per_user``): ``qkv``
        ``[R, 3 H K]``, ``gates`` ``[R, Wg]`` (row ``j B + u`` = user ``u``'s candidate ``j``), ``conv`` ``[B, 3, W]``,
        ``S`` ``[B, H, K, V]``, ``steps`` ``[B]`` int (``n_u``; ``None`` = ``T`` for every user). Candidate ``j`` is
        applied to user ``u`` iff ``j < n_u``: the sequential golden launches on the per-candidate slabs with the
        per-user select of ``golden_gated`` (so the committed state equals the plain state after ``n_u`` steps, bit for
        bit). Returns ``(o [R, H V], state, conv)``: output row ``j B + u`` is the launch-``j`` output of user ``u`` when
        applied, zero otherwise; ``dry`` applies ``T`` steps to every user on a scratch and returns the INPUT state /
        history (the verify)."""
        T = int(steps_per_user)
        R = qkv.shape[0]
        B = S.shape[0]
        assert R == B * T and T >= 1, (R, B, T)
        n = (
            torch.full((B,), T, dtype=torch.long)
            if (steps is None or dry)
            else torch.as_tensor(steps).long().reshape(B)
        )
        assert int(n.min()) >= 0 and int(n.max()) <= T, n
        S_run, conv_run = S, conv
        outs = []
        for j in range(T):
            gate = (n > j).to(torch.int32)
            o_j, S_run, conv_run = KdaDecode.golden_gated(
                qkv[j * B : (j + 1) * B],
                gates[j * B : (j + 1) * B],
                conv_run,
                S_run,
                taps,
                dt_bias,
                a_exp,
                onw,
                gate,
                **kw,
            )
            outs.append(o_j * gate[:, None].to(o_j.dtype))
        o = torch.cat(outs, dim=0)  # candidate-major
        if dry:
            return o, S, conv
        return o, S_run, conv_run

    @staticmethod
    def activate_gates(gates: torch.Tensor, dt_bias: torch.Tensor, a_exp: torch.Tensor, hk: int) -> torch.Tensor:
        """The ``gates="pre"`` -> ``"act"`` input map in fp32 (what the pre-arranged g2 matmul + fused sigmoid
        produce): ``[sigmoid(a (f + dt)) | sigmoid(og) | sigmoid(b)]``."""
        g = gates.float().clone()
        g[..., :hk] = a_exp * (g[..., :hk] + dt_bias)
        return torch.sigmoid(g)

    # ------------------------------------------------------------------ placement
    @staticmethod
    def placement_col(grid, slots: int):
        """Cores for ``slots`` (head, tile row) items: ``parts`` cores per slot, a vertical run in one column
        (NoC multicast rectangle); the largest ``parts`` that fits, else one core per item round-robin."""
        gx, gy = grid.x, grid.y
        parts = 1
        for ry in range(min(TILE, gy), 1, -1):
            if gx * (gy // ry) >= slots:
                parts = ry
                break
        if parts == 1:
            NC = min(slots, gx * gy)
            return 1, [ttnn.CoreCoord(c // gy, c % gy) for c in range(NC)]
        per_col = gy // parts
        cores = []
        for s in range(slots):
            x, y0 = s // per_col, (s % per_col) * parts
            cores += [ttnn.CoreCoord(x, y0 + p) for p in range(parts)]
        return parts, cores

    @staticmethod
    def placement_rect(grid, slots: int):
        """``2 x h`` rectangles per slot (parts ``0 .. h-1`` down the left column from the part-0 core, ``h .. 2h-1``
        down the right column), the largest ``h`` whose ``(gx // 2) * (gy // h)`` rectangles cover ``slots``.
        Returns ``(parts, cores, h)`` or ``None`` when no rectangle layout fits (``parts >= 2`` needed)."""
        gx, gy = grid.x, grid.y
        per_row = gx // 2
        for h in range(min(TILE // 2, gy), 0, -1):
            if per_row * (gy // h) >= slots:
                cores = []
                for s in range(slots):
                    x0, y0 = 2 * (s % per_row), h * (s // per_row)
                    cores += [ttnn.CoreCoord(x0, y0 + p) for p in range(h)]
                    cores += [ttnn.CoreCoord(x0 + 1, y0 + p) for p in range(h)]
                return 2 * h, cores, h
        return None

    @classmethod
    def placement(cls, grid, slots: int, mode: str = "col"):
        """``(parts, cores, rect_h)``: the column layout, or (``mode="rect"``) the rectangle layout when it hands every
        slot more parts than the column layout; ``rect_h = 0`` for a column layout (no second multicast column)."""
        parts, cores = cls.placement_col(grid, slots)
        if mode == "rect":
            rect = cls.placement_rect(grid, slots)
            if rect is not None and rect[0] > parts:
                return rect
        return parts, cores, 0

    @staticmethod
    def slot_geometry(core_of, parts: int, rect_h: int, s: int):
        """Logical cores of slot ``s``: ``(p0, pe, rb0, rb1)`` = the part-0 core, the last core of its column (the
        first multicast rectangle is ``p0 + 1 .. pe`` down that column), and the second column's top / bottom
        (``None`` for a column layout)."""
        base = s * parts
        p0 = core_of[base]
        if rect_h:
            pe = core_of[base + rect_h - 1]
            rb0, rb1 = core_of[base + rect_h], core_of[base + 2 * rect_h - 1]
        else:
            pe = core_of[base + parts - 1]
            rb0 = rb1 = None
        return p0, pe, rb0, rb1

    @staticmethod
    def item_of(core_index: int, parts: int, Bt: int):
        """The kernels' decode of work item ``core_index`` (one item per core when ``parts > 1``):
        ``(head, tile row, part)``."""
        h = core_index // (Bt * parts)
        rem = core_index % (Bt * parts)
        return h, rem // parts, rem % parts

    # ------------------------------------------------------------------ program arguments (pure: host-testable)
    def compile_time_args(self, B: int, parts: int, reader_acc: list, writer_acc: list):
        """``(reader, writer, compute)`` compile-time vectors; ``reader_acc`` = the concatenated
        ``TensorAccessorArgs`` words of (qkv, conv_hist, taps, gates, chan_consts, o_norm_w, selectors, recurrent),
        ``writer_acc`` of (out, recurrent)."""
        H, K, V = self.H, self.K, self.V
        Kt, Vt = K // TILE, V // TILE
        Bt = -(-B // TILE)
        HK_t, HV_t = H * Kt, H * Vt
        Wq_t, Wg_t = 3 * HK_t, HK_t + HV_t + (-(-H // TILE))
        it_half = max(2, -(-min(B, 16) // 2))  # SFPU iterations per face (2 rows each) for tile rows with <= 16 users
        compute = [Kt, Vt, Bt, B, parts, f32_bits(self.lb), f32_bits(self.l2_eps), f32_bits(self.norm_eps),
                   f32_bits(K**-0.5), f32_bits(1.0 / V), it_half]  # fmt: skip
        reader = [Kt, Vt, Bt, B, H, Wq_t, Wg_t, HK_t, HV_t, parts, ONES_PAGE]
        if self.variant != "v1":  # the v2 / gated / fused kernels read GATES_ACT at index 11 (reader accessors from 12)
            act = 1 if self.gates_activated else 0
            compute.append(act)
            reader.append(act)
        writer = [Kt, Vt, Bt, B, H, parts, HV_t]
        if (
            self.fused
        ):  # USERS, STEPS, DRY: reader / compute 12..14 (accessors from 15), writer 7..9 (accessors from 10)
            T = self.fused_steps
            assert (
                B % T == 0 and B <= self.MAX_FUSED_ROWS
            ), f"fused rows {B} must be USERS x {T} <= {self.MAX_FUSED_ROWS}"
            tail = [B // T, T, 1 if self.fused_dry else 0]
            compute += tail
            reader += tail
            writer += tail
        reader += list(reader_acc)
        writer += list(writer_acc)
        return reader, writer, compute

    def runtime_args_for_core(self, c: int, n_items: int, NC: int, addrs: dict, noc, core_of, parts: int, rect_h: int):
        """``(reader, writer, compute)`` runtime vectors of core ``c``; ``addrs`` = buffer addresses keyed as
        ``READER_RT`` / ``WRITER_RT`` name them, ``noc(core) -> (x, y)`` the NoC coordinates of a logical core.
        The v1 reader vector is today's 15 entries; the v2 one appends the second multicast column (5 entries); the
        gated reader / writer append the gate page address (``addrs["gate"]``: reader index 20, writer index 7), the
        fused ones the step page address (``addrs["steps"]``: the same indices)."""
        s = c // parts
        p0, pe, rb0, rb1 = self.slot_geometry(core_of, parts, rect_h, s)
        p0x, p0y = noc(p0)
        pex, pey = noc(pe)
        reader = [
            c, n_items, NC, addrs["qkv"], addrs["hist"], addrs["taps"], addrs["gates"], addrs["cc"], addrs["onw"],
            addrs["sel"], addrs["st"], p0x, p0y, pex, pey,
        ]  # fmt: skip
        if self.variant != "v1":
            if rb0 is not None:
                bx0, by0 = noc(rb0)
                bx1, by1 = noc(rb1)
                reader += [bx0, by0, bx1, by1, rect_h]
            else:
                reader += [0, 0, 0, 0, 0]
        else:
            assert rect_h == 0, "the v1 kernels know no second multicast column"
        writer = [c, n_items, NC, addrs["out"], addrs["st"], p0x, p0y]
        if self.gated:
            reader.append(addrs["gate"])
            writer.append(addrs["gate"])
        elif self.fused:
            reader.append(addrs["steps"])
            writer.append(addrs["steps"])
        compute = [c, n_items, NC]
        return reader, writer, compute

    def cb_plan(self, parts: int):
        """``[(name, dtype, pages)]`` of the per-core circular buffers."""
        Kt, Vt = self.K // TILE, self.V // TILE
        kv = Kt * Vt
        f32, f16 = ttnn.float32, ttnn.bfloat16
        # gated: + the gate page + merge scratch; fused: + the step page + history tiles / merge scratch
        onw_pages = Vt + (self.GATED_SCRATCH_PAGES if self.gated else self.FUSED_SCRATCH_PAGES if self.fused else 0)
        selT_pages = self.FUSED_SELT_PAGES if self.fused else 2
        # index 28: v1 / v2 / gated "oq" (qd . S, Vt); fused "loop" (the per-user state between steps, two kv slots)
        slot28 = ("loop", f32, 2 * kv) if self.fused else ("oq", f32, Vt)
        return [
            ("hist", f16, 12 * Kt), ("taps", f16, 3 * Kt), ("gf", f32, Kt), ("gog", f32, Vt), ("gb", f32, 1),
            ("cc", f32, Kt), ("onw", f32, onw_pages), ("sel", f16, 2), ("selT", f16, selT_pages), ("S", f32, 2 * kv),
            ("out", f16, Vt), ("state_out", f32, 2 * kv), ("qn", f32, Kt), ("kn", f32, Kt), ("vc", f32, Vt),
            ("dec", f32, Kt), ("bcol", f32, 1), ("wg", f32, Vt), ("tmp", f32, 2 * Kt), ("col", f32, 2),
            ("kd", f32, Kt), ("qd", f32, Kt), ("kcol", f32, Kt), ("decT", f32, Kt),
            ("gather", f32, max(1, parts - 1) * Vt), ("dcol", f32, Kt), ("pq", f32, 4), ("vread", f32, Vt),
            slot28, ("u", f32, Vt), ("oacc", f32, 2 * Vt), ("ofin", f32, Vt),
        ]  # fmt: skip

    def l1_bytes(self, parts: int) -> int:
        t32, t16 = TILE * TILE * 4, TILE * TILE * 2
        return sum((t32 if dt == ttnn.float32 else t16) * pages for _, dt, pages in self.cb_plan(parts))

    # ------------------------------------------------------------------ program
    def _cb(self, idx, dtype, page, pages, cores):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )

    def __call__(
        self, qkv, gates, conv_hist, recurrent, taps, chan_consts, o_norm_w, selectors, out, gate=None, steps=None
    ):
        """qkv [1,1,B,3*H*K] bf16 TILE; gates [1,1,B,H*K+H*V+Hb] fp32 TILE; conv_hist [3,B,3*H*K] bf16 TILE
        (tap-major history, tap 0 the oldest; updated in place); recurrent [B,H,K,V] fp32 TILE (updated in place);
        taps [1,1,4,3*H*K] bf16 TILE; chan_consts [1,1,2,H*K] fp32 TILE (row 0 dt_bias, row 1 exp(A_log); not read
        with ``gates="act"``); o_norm_w [1,1,1,V] fp32 TILE; selectors [34*32,32] bf16 TILE (``kda_selector_tiles``);
        out [1,1,B,H*V] bf16 TILE (pre-allocated); ``gate`` (``gated=True`` only) [Bt, 32] int32 ROW_MAJOR, row ``tb``
        = the write gates of tile row ``tb``'s users (non-zero = the state / history of that user is written);
        ``steps`` (``fused_steps=T`` only) [Bt, 32] int32 ROW_MAJOR, word ``u`` = the candidates user ``u`` applies (the
        verify's dry launch never reads it) -- the fused launch takes ``qkv`` / ``gates`` / ``out`` with ``R = B x T``
        candidate-major rows (one tile row) against the ``B``-slot state. Returns ``out``."""
        H = self.H
        B = recurrent.shape[0]
        if self.fused:
            R = int(qkv.shape[-2])
            assert R == B * self.fused_steps and R <= self.MAX_FUSED_ROWS, (R, B, self.fused_steps)
            assert steps is not None, "the fused variant needs the per-user step page tensor"
            assert tuple(steps.shape)[-1] == GATE_WORDS and int(steps.shape[-2]) >= 1, tuple(steps.shape)
            assert gate is None, "the fused variant takes a step page, not a write gate"
            B = R  # the program's rows: one tile row of USERS x STEPS candidates
        else:
            assert steps is None, "a step page needs KdaDecode(fused_steps=T)"
        Bt = -(-B // TILE)
        if self.gated:
            assert gate is not None, "the gated variant needs the per-user write gate tensor"
            assert tuple(gate.shape)[-1] == GATE_WORDS and int(gate.shape[-2]) >= Bt, (tuple(gate.shape), Bt)
        else:
            assert gate is None, "a write gate needs KdaDecode(gated=True)"
        device = qkv.device()
        grid = device.compute_with_storage_grid_size()
        parts, core_of, rect_h = self.placement(grid, H * Bt, self.placement_mode)
        NI = H * Bt * parts
        NC = len(core_of)
        cores = rect_core_ranges(core_of)  # 1-3 rectangles instead of one range per core: < 1 us of dispatch

        t32, t16 = TILE * TILE * 4, TILE * TILE * 2
        cb = self.CB
        cbs = [
            self._cb(cb[name], dt, t32 if dt == ttnn.float32 else t16, pages, cores)
            for name, dt, pages in self.cb_plan(parts)
        ]

        reader_acc, writer_acc = [], []
        reader_tensors = [qkv, conv_hist, taps, gates, chan_consts, o_norm_w, selectors, recurrent]
        writer_tensors = [out, recurrent]
        if self.gated:  # the gate accessor is the last block of both kernels
            reader_tensors.append(gate)
            writer_tensors.append(gate)
        elif self.fused:  # the step page accessor is the last block of both kernels
            reader_tensors.append(steps)
            writer_tensors.append(steps)
        for t in reader_tensors:
            reader_acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        for t in writer_tensors:
            writer_acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_ct, writer_ct, compute_ct = self.compile_time_args(B, parts, reader_acc, writer_acc)

        addrs = dict(
            qkv=qkv.buffer_address(), hist=conv_hist.buffer_address(), taps=taps.buffer_address(),
            gates=gates.buffer_address(), cc=chan_consts.buffer_address(), onw=o_norm_w.buffer_address(),
            sel=selectors.buffer_address(), st=recurrent.buffer_address(), out=out.buffer_address(),
        )  # fmt: skip
        if self.gated:
            addrs["gate"] = gate.buffer_address()
        elif self.fused:
            addrs["steps"] = steps.buffer_address()

        def noc(core):
            p = device.worker_core_from_logical_core(core)
            return p.x, p.y

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, core in enumerate(core_of):
            n_items = (NI - c + NC - 1) // NC if parts == 1 else 1
            r, w, k = self.runtime_args_for_core(c, n_items, NC, addrs, noc, core_of, parts, rect_h)
            reader_rt[core.x][core.y] = r
            writer_rt[core.x][core.y] = w
            compute_rt[core.x][core.y] = k

        reader_src, writer_src, compute_src = self.kernel_sources()
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / reader_src),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / writer_src),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / compute_src),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.fidelity, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
                ),
            ),
        ]
        sems = [
            ttnn.SemaphoreDescriptor(sid, ttnn.CoreType.WORKER, cores, init)
            for sid, init in ((self.SEM_READY, 0), (self.SEM_VALID, 0), (self.SEM_ONE, 1), (self.SEM_GATHER, 0))
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        io = [qkv, gates, conv_hist, recurrent, taps, chan_consts, o_norm_w, selectors, out]
        if self.gated:
            io.append(gate)
        elif self.fused:
            io.append(steps)
        return ttnn.generic_op(io, program)
