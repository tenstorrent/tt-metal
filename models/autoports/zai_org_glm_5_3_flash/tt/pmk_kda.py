# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill megakernel (Stage 16, doc/ttnn/16-prefill-production/prefill_megakernel.md): the KDA phase = P4 (conv +
gates) + P5 (chunk prepare) + P6 (chunk scan + gated RMSNorm) + P9 (state write), inside the persistent layer program.
The q / k / v, f_a | g_a | b, f_b and g_b projections (P3 / the P4 "up" matmuls) are PfStreamPhase instances (§4c) of
the family's streaming readers: this module only DECLARES them (`KdaPlan.stream_phases`) with their activation source
and H3 destinations; it owns the kernels of everything after them (kernels/pmk/kda_*.cpp).

Geometry (`KdaGeometry`, the coordinator's TP2 / TP8 requirement): `rows` = the gathered rows of the bucket on every
chip (the KDA recurrence is head-parallel over ALL rows, §1), `heads` = the chip's head shard (TP8: 8, TP2: 32),
NCH = rows / 32 chunks per head. The q/k/v/g/β store (H3) is in DRAM form in both geometries (`store="dram"`: u
[rows, 3 W] bf16, lr [rows, 2 d + H] bf16, f / gate_out [rows, W] fp32, written by the stream phases per 32-row block
and counted on SEM_QKV); the L1 form does not fit next to the item CBs at TP8 (census in `KdaPlan`) and is not built.

Cores (time-multiplexed with the family's roles, §2):
  item cores  H x CPH (64 at TP8: CPH = 8; at TP2 32 heads x 2): core (head, slot) owns chunks slot, slot + CPH, ... of
              the head, in order; per item: the conv (shift-matrix matmuls on this and the previous chunk's u tiles +
              the carry rows for chunk 0), silu, the decay gate from the raw f_b output, beta, then the chunk terms of
              prepare_chunk_recurrence -> TERMS DRAM scratch [(head, chunk), 22 tiles fp32]; CBs 32..54 (§4b's
              "KDA prepare" range extended by 5: the item and scan core sets are disjoint).
  scan cores  H x SPREAD (32 at TP8: 8 heads x 4 value-column tiles; TP2: 32 x 1): core (head, v-block) scans its
              Vt = 4 / SPREAD value tiles of the head's state over the valid chunks, exchanges the gated-norm partial
              sums with the head's other cores (kda_rec's REC_SPREAD protocol), writes `on` (H4) and the final state
              (P9); CBs 50..63 (§4b's "KDA scan" range).
Hand-offs (§4b counters, all wait_min on monotonic counters reset once per layer by the family):
  H3  SEM_QKV (5)     P3 / P4 stream phases -> item cores: +1 per 32-row block per producer (NPROD producers)
  --  SEM_KDA (6)     item cores -> the head's scan cores: +1 per finished chunk, raised IN CHUNK ORDER (the item
                      writers pass a baton, id 6 on the item cores, so a scan core's wait_min(>= c + 1) is exact)
  --  SEM_RMS (0)     scan peers' partial sums (SPREAD - 1 per chunk); SEM_ONAGG (1): every scan core -> the leader,
                      per chunk. Ids are 0..15 (NUM_SEMAPHORES = 16): 0 / 1 are the mHC pre ids of members / roots, free
                      on MM-only cores, so the scan cores must be MM-only cores (or the family passes PK_SEM_RMS /
                      PK_SEM_ONAGG defines)
  H4  SEM_ON (7)      the leader -> every P7 consumer, +1 per chunk once all scan cores finished it (NSCAN x NCONS
                      atomics per layer would be 213k at TP8; the leader aggregation makes it NSCAN + NCONS per chunk)
valid_end (device tensor, request-agnostic trace): chunks past ceil(valid_end / 32) are dead (no item work, the state
unchanged, zero output rows); the partial chunk's rows past valid_end are identity steps (k, v, g, beta masked to 0,
as the op's zeroed rows). The carry rows (state C) = rows valid_end - 3 .. valid_end - 1 of u (previous carry rows
for rows before 0), written by the carry core (prefill_tp's select_local_final_history).
"""

from __future__ import annotations

import dataclasses
import math
import os
import struct

KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "pmk")
TILE = 32
T32, T16 = 4096, 2048
ITEM_CB0, SCAN_CB0 = 32, 50
SEM_QKV, SEM_KDA, SEM_ON, SEM_RMS, SEM_ONAGG = 5, 6, 7, 0, 1
# CBs the compute reads ONLY through copy_tile / SFPU paths: they get UnpackToDestFp32 (exact fp32 into DST). Every
# other CB is an FPU operand (matmul_block / add_block / mul_tiles_bcast / reduce) and stays Default = unpack to SrcA/B:
# the mode on an FPU operand diverts the unpacker from the source registers and the FPU multiplies zeros (silicon run
# r3-probe: zero U / Y / S with nonzero inputs; the op sets UnpackToSrc on all its buffers, mhc_pre / kda_pre / ds_mm
# set the mode on copy-only CBs only)
ITEM_COPY_ONLY_CBS = ("F", "DTR", "BETA", "MASK")
SCAN_COPY_ONLY_CBS = ("O", "PIN")
L1_CB_BUDGET = 1026 * 1024  # §4: about 1,026 KB of CB room per core under Stage 14's reader CB end
PEAK_TFLOPS = 297.0  # §3: the HiFi2 grid peak the MATH % is measured against


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


@dataclasses.dataclass(frozen=True)
class KdaGeometry:
    """rows = gathered rows per chip (the bucket), heads = heads per chip; item / scan core budgets."""

    rows: int
    heads: int
    head_dim: int = 128
    hidden: int = 4096
    item_cores: int = 64
    scan_budget: int = 32  # scan cores available: spread = the largest of 4 / 2 / 1 with heads * spread <= budget
    store: str = "dram"  # the H3 q/k/v/g/β store form ("l1" is declared, not built: see KdaPlan.l1_store_fits)
    # the chunk-term knob (risk: the value-split scan re-reads the terms from DRAM): "fp32" = the op path's
    # output_bf16_mask 0; "bf16" = terms and gate_out as bf16 tiles (half the scan's DRAM; the op's bf16 mask option)
    terms_dtype: str = "fp32"
    mhz: float = 1350.0

    def __post_init__(self):
        if self.rows % TILE or self.rows < TILE:
            raise ValueError(f"KDA geometry: rows {self.rows} (a multiple of 32)")
        if self.head_dim != 128:
            raise ValueError("KDA head dim 128")
        if self.item_cores % self.heads or self.item_cores // self.heads > 8:
            raise ValueError(f"KDA geometry: {self.item_cores} item cores for {self.heads} heads (1..8 per head)")
        if self.spread == 0:
            raise ValueError(f"KDA geometry: {self.scan_budget} scan cores for {self.heads} heads")
        if self.store not in ("dram", "l1"):
            raise ValueError(self.store)
        if self.terms_dtype not in ("fp32", "bf16"):
            raise ValueError(self.terms_dtype)

    @property
    def n_chunks(self) -> int:
        return self.rows // TILE

    @property
    def width(self) -> int:
        return self.heads * self.head_dim

    @property
    def kt(self) -> int:
        return self.head_dim // TILE

    @property
    def cph(self) -> int:
        return self.item_cores // self.heads

    @property
    def spread(self) -> int:
        for s in (4, 2, 1):
            if self.heads * s <= self.scan_budget:
                return s
        return 0

    @property
    def vt(self) -> int:
        return self.kt // self.spread

    @property
    def scan_cores(self) -> int:
        return self.heads * self.spread

    @property
    def term_bytes(self) -> int:
        return T32 if self.terms_dtype == "fp32" else T16


@dataclasses.dataclass
class CBSpec:
    index: int
    name: str
    fmt: str  # "fp32" / "bf16"
    pages: int

    @property
    def page_bytes(self) -> int:
        return T32 if self.fmt == "fp32" else T16

    @property
    def bytes(self) -> int:
        return self.pages * self.page_bytes


def out_pages(kt: int, vtf: int) -> int:
    """TERMS pages per (head, chunk) item: v_beta VTF | q_decay Kt | kd Kt | intra | t_inv | final_decay Kt | kdt Kt"""
    return vtf + 4 * kt + 2


def in_pages(kt: int, vt: int) -> int:
    """scan IN ring pages per chunk: v_beta Vt | kd | q_decay | intra | t_inv | fd | kdt | gate_out Vt"""
    return 2 * vt + 4 * kt + 2


def kda_rne() -> str:
    """GLM53_PMK_KDA_RNE=1 (default off; coordinator 2026-10-25 17:31:59Z, 14's KDA y deficit root cause): the FPU reads
    fp32 CB tiles into srcA / srcB by truncating them to tf32 (a -2^-11 mean magnitude bias per operand); the item and
    scan computes round every fp32 tile a later FPU op reads to tf32 RNE in DST before packing it
    (tt/kernels/pmk/rne_tf32.hpp, 14's header) so that read is exact. Skipped: y (the epilogue reads it by copy / SFPU)
    and x * 0.5 of a rounded tile (exact). realgate10's accuracy arm decides whether it ships.
    GLM53_PMK_KDA_RNE=pack: the same tf32 words from the packer gasket's 10-bit mantissa rounding (Round_10b_mant, armed
    on the item / scan pack threads, every fp32 pack incl. y) -- no SFPU pass; only once pack_rnd10_probe shows the
    gasket rounds to nearest. Returns "0" | "1" | "pack"."""
    v = os.environ.get("GLM53_PMK_KDA_RNE", "0")
    if v not in ("0", "1", "pack"):
        raise ValueError(f"GLM53_PMK_KDA_RNE={v!r}: 0 | 1 | pack")
    return v


def kda_kdec() -> str:
    """GLM53_PMK_KDA_KDEC=dest (default anchor; coordinator 2026-10-26 03:55:13Z, Stage 14 kprec3's root cause of record):
    the item compute forms k_dec_t = (k * exp(G_last - G))^T from the EXACT exponent -- G_last (ones @ g) and G (tril @ g)
    as two fp32 matmul accumulations in DEST, the SFPU subtract and the precise exp in the same DEST pass (14's op fix
    fc999c8e7f9, exact_key_decay) -- instead of the anchored factors exp(anchor - G) * exp(G_last / 2), whose ~ +-72
    exponents in fast channels reach the SFPU exp through SrcA tf32 (k_dec_t biased low). Kernel-only (a define on the
    item kernels, no CB / constant change); 16 gates it. Returns "anchor" | "dest". (14b 5c78a35f1d8, applied on mb2)"""
    v = os.environ.get("GLM53_PMK_KDA_KDEC", "anchor")
    if v not in ("anchor", "dest"):
        raise ValueError(f"GLM53_PMK_KDA_KDEC={v!r}: anchor | dest")
    return v


def kda_carry() -> str:
    """GLM53_PMK_KDA_CARRY=fp32 (default tf32; coordinator 2026-10-26 04:57:37Z, Stage 14 kprec7's op fix b0a41e568b7): the
    scan forms the new state S * decay + U in DEST on the SFPU (S, the decay column and U copied in; sfpu_mul_bcast_col +
    add_binary_tile; ONE rounded pack) instead of the FPU eltwise mul_tiles_bcast_cols, which returns S x decay truncated
    to tf32 every chunk (~ -2^-12 per chunk, compounding on the long-memory rows). Kernel-only (a define on the scan
    kernels). Use with GLM53_PMK_KDA_RNE=pack (or 1): the copies are exact only for tf32-rounded operands. Returns
    "tf32" | "fp32"."""
    v = os.environ.get("GLM53_PMK_KDA_CARRY", "tf32")
    if v not in ("tf32", "fp32"):
        raise ValueError(f"GLM53_PMK_KDA_CARRY={v!r}: tf32 | fp32")
    return v


def item_conv_fpu() -> bool:
    """GLM53_PMK_ITEM_CONV=fpu (Stage 16 W2, a15prof3: conv = 41 pct of P4): the item reader builds the three row-shifted
    u tiles per column (local NoC copies of contiguous face rows) and the compute multiplies them by the taps on the FPU
    (bf16 x bf16 at HiFi4: exact into fp32 DST, as the SFPU products) before the same three SFPU adds -- bitwise the
    shift-matmul conv. CB U holds [previous | this] per item (2 x CT pages, two items deep), CB SH the reader's S1..S3 (3 x CT, one item deep: -72 KB vs the 5 x CT two-deep ring)
    """
    return os.environ.get("GLM53_PMK_ITEM_CONV", "") == "fpu"


def item_cbs(g: KdaGeometry) -> list:
    kt, ct = g.kt, 3 * g.kt
    n = lambda i, name, fmt, pages: CBSpec(ITEM_CB0 + i, name, fmt, pages)  # noqa: E731
    return [
        n(0, "U", "bf16", 2 * 2 * ct),
        n(1, "F", "fp32", 2 * kt),
        n(2, "BETA", "fp32", 2),
        n(3, "MASK", "fp32", 2),
        n(4, "CONST", "fp32", 12),
        n(5, "TAPS", "bf16", 4 * ct),
        n(6, "DTR", "fp32", 2 * kt),
        n(7, "CONV", "bf16", ct),
        n(8, "G", "fp32", kt),
        n(9, "NQ", "fp32", kt),
        n(10, "NK", "fp32", kt),
        n(11, "W0", "fp32", kt),
        n(12, "W1", "fp32", kt),
        n(13, "W2", "fp32", kt),
        n(14, "W3", "fp32", kt),
        n(15, "TW0", "fp32", 1),
        n(16, "TW1", "fp32", 2),
        n(17, "TW2", "fp32", 1),
        n(18, "SD", "fp32", kt),
        n(19, "CID", "fp32", kt),
        n(20, "ANCH", "fp32", kt),
        n(21, "AKK", "fp32", 1),
        n(22, "OUT", g.terms_dtype, 2 * out_pages(kt, kt)),
    ] + (
        [n(23, "SH", "bf16", 3 * ct)] if item_conv_fpu() else []
    )  # fmt: skip  (fpu conv: S1..S3, one item deep)


def scan_cbs(g: KdaGeometry) -> list:
    kt, vt, kv = g.kt, g.vt, g.kt * g.vt
    n = lambda i, name, fmt, pages: CBSpec(SCAN_CB0 + i, name, fmt, pages)  # noqa: E731
    return [
        n(0, "S", "fp32", kv), n(1, "IN", g.terms_dtype, 2 * in_pages(kt, vt)), n(2, "W", "fp32", vt + 1), n(3, "SR", "fp32", 2 * kv),
        n(4, "VN", "fp32", vt), n(5, "OI", "fp32", vt), n(6, "SCR", "fp32", kv), n(7, "SU", "fp32", kv), n(8, "ST", "fp32", kv),
        n(9, "FS", "fp32", kv), n(10, "O", "fp32", vt), n(11, "PSUM", "fp32", 1), n(12, "PIN", "fp32", max(1, 2 * (g.spread - 1))),
        n(13, "ON", "bf16", 2 * vt),
    ]  # fmt: skip


@dataclasses.dataclass
class KdaPlan:
    """Pure-python plan: CB tables, L1 bytes, DRAM scratch bytes, the §4c stream-phase declarations, the estimates."""

    geo: KdaGeometry
    item: list
    scan: list

    @property
    def item_l1(self) -> int:
        return sum(c.bytes for c in self.item)

    @property
    def scan_l1(self) -> int:
        return sum(c.bytes for c in self.scan)

    def scratch_bytes(self) -> dict:
        g = self.geo
        return dict(
            u=g.rows * 3 * g.width * 2, lr=g.rows * (2 * g.head_dim + g.heads) * 2, f=g.rows * g.width * 4,
            gate_out=g.rows * g.width * (4 if g.terms_dtype == "fp32" else 2),
            terms=g.heads * g.n_chunks * out_pages(g.kt, g.kt) * g.term_bytes,
            on=g.rows * g.width * 2, state_s=g.heads * g.head_dim * g.head_dim * 4, state_c=3 * 3 * g.width * 2,
        )  # fmt: skip

    def l1_store_fits(self, resident_bytes: int = 0) -> bool:
        """would the H3 store (u, lr, f, gate_out) fit in L1 spread over the item cores next to their CBs?"""
        s = self.scratch_bytes()
        per_core = (s["u"] + s["lr"] + s["f"] + s["gate_out"]) // self.geo.item_cores
        return per_core + self.item_l1 + resident_bytes <= L1_CB_BUDGET

    def stream_phases(self) -> list:
        """The P3 / P4 PfStreamPhase instances (§4c) this phase needs, in program order: ds role, K / N, the activation
        source, the H3 destination, the counter. The family instantiates them on its streaming readers."""
        g = self.geo
        m_tiles = g.rows // TILE
        return [
            dict(role="kda_qkv", k=g.hidden, n=3 * g.width, m_tiles=m_tiles, in0="H2 x-hat store", out="u (H3, bf16 TILE)",
                 counter="SEM_QKV +1 per 32-row block"),
            dict(role="kda_lowrank", k=g.hidden, n=2 * g.head_dim + g.heads, m_tiles=m_tiles, in0="H2 x-hat store",
                 out="lr (H3, bf16 TILE: f_a | g_a | b)", counter="SEM_QKV +1 per 32-row block"),
            dict(role="kda_fb", k=g.head_dim, n=g.width, m_tiles=m_tiles, in0="lr tiles 0..3 (f_a)", out="f (H3, fp32 TILE, raw)",
                 counter="SEM_QKV +1 per 32-row block", math="HiFi4 fp32 (ds_fb.ckc)"),
            dict(role="kda_gb", k=g.head_dim, n=g.width, m_tiles=m_tiles, in0="lr tiles 4..7 (g_a)", out="gate_out (H3, fp32 TILE, raw)",
                 counter="SEM_QKV +1 per 32-row block", math="HiFi4 fp32 (ds_gb.ckc)"),
        ]  # fmt: skip

    def estimate(
        self, dram_gbs: float = 480.0, cycles_per_op: float = 60.0, term_tile_bytes: int | None = None
    ) -> dict:
        """Busy-time arithmetic (README): tile-op counts per item / chunk at ~cycles_per_op FPU cycles each (fp32 HiFi4
        matmul_block / SFPU op incl. its pack), DRAM at the §3 rate; compute-core busy fraction of the phase."""
        g = self.geo
        if term_tile_bytes is None:
            term_tile_bytes = g.term_bytes
        kt, ct = g.kt, 3 * g.kt
        conv_ops = ct * (6 + 4 + 3 + 2)  # shifts (2 matmuls x 3), tap broadcasts, adds, silu + load
        gate_ops = kt * 5 + 3
        prep_ops = 24 + kt + 36 + 20 + kt + kt + 9 + 19 + 12  # the op's helpers at Kt = 4 (see README)
        item_cycles = (conv_ops + gate_ops + prep_ops) * cycles_per_op
        items_per_core = g.heads * g.n_chunks / g.item_cores
        item_span_us = items_per_core * item_cycles / g.mhz
        item_in_bytes = 2 * ct * T16 + kt * T32 + T16  # u (2 chunks), f, the lr tile
        item_dram_us = items_per_core * item_in_bytes * g.item_cores / (dram_gbs * 1e3)
        scan_ops = 14 * (4 // g.vt if g.vt else 1) + 10 + 12  # matmuls scale with the value tiles; elementwise; norm
        scan_cycles = (14 * g.vt + 10 * g.vt + 12) * cycles_per_op + 400
        scan_compute_us = g.n_chunks * scan_cycles / g.mhz
        scan_bytes = g.scan_cores * g.n_chunks * (in_pages(kt, g.vt) - g.vt) * term_tile_bytes
        scan_dram_us = scan_bytes / (dram_gbs * 1e3)
        scan_span_us = max(scan_compute_us, scan_dram_us)
        item_busy = min(1.0, item_span_us / max(item_span_us, item_dram_us)) * 0.95
        scan_busy = scan_compute_us / scan_span_us
        core_busy = (g.item_cores * item_span_us * item_busy + g.scan_cores * scan_span_us * scan_busy) / (
            g.item_cores * item_span_us + g.scan_cores * scan_span_us
        )
        flop = g.rows * g.heads * (g.head_dim * g.head_dim * 2 * 5 + 32 * g.head_dim * 2 * 4) * 1.0  # scan + conv-ish
        return dict(
            item_ops_per_item=conv_ops + gate_ops + prep_ops, items_per_core=items_per_core, item_span_us=item_span_us,
            item_dram_us=item_dram_us, scan_ops_per_chunk=scan_ops, scan_compute_us=scan_compute_us,
            scan_dram_mb=scan_bytes / 1e6, scan_dram_us=scan_dram_us, scan_span_us=scan_span_us,
            phase_span_us=item_span_us + scan_span_us, compute_core_busy=core_busy, scan_busy=scan_busy,
            scan_dram_mb_bf16_terms=scan_bytes / 2e6, gflop_lower_bound=flop / 1e9,
        )  # fmt: skip

    def check(self, resident_bytes: int = 0) -> list:
        """problems (empty = fine): CB budgets, index ranges, counter ids"""
        g = self.geo
        out = []
        if self.item_l1 + resident_bytes > L1_CB_BUDGET:
            out.append(f"item cores: {self.item_l1 + resident_bytes} B of CBs over the {L1_CB_BUDGET} budget")
        if self.scan_l1 > L1_CB_BUDGET:
            out.append(f"scan cores: {self.scan_l1} B over budget")
        idx = [c.index for c in self.item]
        if idx != list(range(ITEM_CB0, ITEM_CB0 + len(idx))) or max(idx) > 63:
            out.append("item CB indices")
        sdx = [c.index for c in self.scan]
        if sdx != list(range(SCAN_CB0, SCAN_CB0 + len(sdx))) or max(sdx) > 63:
            out.append("scan CB indices")
        if (
            len({SEM_QKV, SEM_KDA, SEM_ON, SEM_RMS, SEM_ONAGG}) != 5
            or max(SEM_QKV, SEM_KDA, SEM_ON, SEM_RMS, SEM_ONAGG) > 15
        ):
            out.append("semaphore ids (0..15, distinct)")
        if g.store == "l1" and not self.l1_store_fits(resident_bytes):
            out.append("the L1 store form does not fit")
        return out


def plan(geo: KdaGeometry) -> KdaPlan:
    return KdaPlan(geo, item_cbs(geo), scan_cbs(geo))


# ---------------------------------------------------------------------------------------------------- ttnn builder
class KdaPhase:
    """The phase's kernels / CBs / semaphores / runtime args for a persistent program (`parts`), and a standalone
    one-phase program (`program` / `__call__`: the mock unit and the device bench)."""

    def __init__(self, device, geo: KdaGeometry, *, item_cores: list, scan_cores: list, taps: list, dt_bias, rate,
                 o_norm_row, lower_bound: float, eps: float, consumers: list | None = None, nprod: int = 1,
                 h3_preraised: bool = False, item_map: list | None = None, scan_map: list | None = None):  # fmt: skip
        """item_cores / scan_cores: logical CoreCoords (len H x CPH / H x SPREAD; scan core h * SPREAD + b = (head h,
        v-block b)); taps: the 4 [1, 1, 3 W] bf16 TILE tap rows (OptimizedKDA.taps_prefill), dt_bias / rate [1,1,1,W]
        fp32 TILE, o_norm_row [1,1,32,128] fp32 TILE (row 0 = o_norm); consumers: NoC (x, y) of the P7 consumers
        signalled on SEM_ON; nprod: H3 producers per 32-row block; h3_preraised: standalone (the projections ran as
        ops before): SEM_QKV starts at nprod x NCH."""
        import ttnn
        from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

        self.ttnn, self.device, self.geo = ttnn, device, geo
        self.plan = plan(geo)
        # the integrator's CB index maps (Stage 16: the member / MM-only composites host other pieces' CBs too): the
        # kernels take them as PK_ITEM_MAP / PK_SCAN_MAP (kda_common.hpp) = self.kdefines; the specs (UnpackToDest
        # modes, CB descriptors) follow
        import dataclasses

        self.item_map, self.scan_map = (list(item_map) if item_map else None), (list(scan_map) if scan_map else None)
        if self.item_map:
            assert len(self.item_map) == len(self.plan.item), (len(self.item_map), len(self.plan.item))
            self.plan.item = [dataclasses.replace(sp_, index=int(i)) for sp_, i in zip(self.plan.item, self.item_map)]
        if self.scan_map:
            assert len(self.scan_map) == len(self.plan.scan), (len(self.scan_map), len(self.plan.scan))
            self.plan.scan = [dataclasses.replace(sp_, index=int(i)) for sp_, i in zip(self.plan.scan, self.scan_map)]
        self.kdefines = dict(
            item=[("PK_ITEM_MAP", ",".join(str(i) for i in self.item_map))] if self.item_map else [],
            scan=[("PK_SCAN_MAP", ",".join(str(i) for i in self.scan_map))] if self.scan_map else [],
        )
        if item_conv_fpu():
            self.kdefines["item"].append(("PMK_ITEM_CONV_FPU", "1"))
        if os.environ.get("GLM53_PMK_ITEM_GATEB", "0") == "1":  # the gate step's Kt tiles in one DST pass (bitwise)
            self.kdefines["item"].append(("PMK_ITEM_GATE_BATCH", "1"))
        if (
            kda_rne() == "1"
        ):  # tf32 RNE pre-round of every packed fp32 FPU operand on the SFPU (items + scan; 14's rne_tf32.hpp)
            self.kdefines["item"].append(("PMK_KDA_RNE", "1"))
            self.kdefines["scan"].append(("PMK_KDA_RNE", "1"))
        elif kda_rne() == "pack":  # the same rounding by the packer gasket (Round_10b_mant): no SFPU pass
            self.kdefines["item"].append(("PMK_KDA_RNE_PACK", "1"))
            self.kdefines["scan"].append(("PMK_KDA_RNE_PACK", "1"))
            if geo.terms_dtype == "fp32":
                self.kdefines["item"].append(("PMK_KDA_RNE_OUT", "1"))
        if kda_kdec() == "dest":  # k_dec_t from the exact exponent G_last - G in DEST (14's kprec3 fix)
            self.kdefines["item"].append(("PMK_KDA_KDEC_DEST", "1"))
        if kda_carry() == "fp32":  # the scan's state chain S * d + U on the SFPU in fp32 (14's kprec7 fix)
            self.kdefines["scan"].append(("PMK_KDA_CARRY_FP32", "1"))
        if os.environ.get("GLM53_PMK_ITEM_ZONES", "0") == "1":  # standalone builds: the item step / reader zones
            self.kdefines["item"].append(("PMK_ITEM_ZONES_SA", "1"))  # (PMK composites ignore it: MK_PHASED)
        if (
            os.environ.get("GLM53_PMK_SCAN_ZONES", "0") == "1"
        ):  # (3''): the scan's per-chunk step zones (standalone builds)
            self.kdefines["scan"].append(("PMK_SCAN_ZONES_SA", "1"))
        g = geo
        if len(item_cores) != g.item_cores or len(scan_cores) != g.scan_cores:
            raise ValueError(f"KDA phase: {len(item_cores)} item / {len(scan_cores)} scan cores for {g}")
        self.item_cores, self.scan_cores = list(item_cores), list(scan_cores)
        self.item_crs = merged_ranges([(int(c.x), int(c.y)) for c in item_cores])
        self.scan_crs = merged_ranges([(int(c.x), int(c.y)) for c in scan_cores])
        self.taps, self.dt_bias, self.rate, self.w = list(taps), dt_bias, rate, o_norm_row
        self.lower_bound, self.eps = float(lower_bound), float(eps)
        self.consumers = list(consumers or [])
        self.nprod, self.h3_preraised = int(nprod), bool(h3_preraised)
        self.noc = lambda c: (lambda w: (int(w.x), int(w.y)))(device.worker_core_from_logical_core(c))
        self.calls = 0

    @classmethod
    def from_kda(cls, kda, geo: KdaGeometry, **kw):
        """from an OptimizedKDA / KDA (its taps_prefill, dt_bias, decay_rate, o_norm, lower_bound, eps)"""
        import torch

        import ttnn

        dev = kda.device
        w = torch.zeros(1, 1, TILE, geo.head_dim)
        w[0, 0, 0] = ttnn.to_torch(ttnn.get_device_tensors(kda.o_norm)[0]).float().reshape(-1)
        n_dev = dev.get_num_devices() if hasattr(dev, "get_num_devices") else 1
        wt = ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if n_dev > 1 else None)  # fmt: skip
        return cls(dev, geo, taps=kda.taps_prefill, dt_bias=kda.dt_bias, rate=kda.decay_rate, o_norm_row=wt,
                   lower_bound=kda.lower_bound, eps=kda.eps, **kw)  # fmt: skip

    # ------------------------------------------------------------------ scratch (persistent, made outside traces)
    def scratch(self, memory_config=None) -> dict:
        """u [1,1,rows,3W] bf16 TILE, lr [1,1,rows,2d+H] bf16, f / gate_out [1,1,rows,W] fp32, terms, on [1,1,rows,W]
        bf16: the H3 store (DRAM form) + the items' chunk terms + the H4 output. Padded widths are the stream phases'
        n_pad (the family passes the DSLinears' n_pad); here the natural widths."""
        ttnn, g = self.ttnn, self.geo
        mem = memory_config or ttnn.DRAM_MEMORY_CONFIG
        mk = lambda shape, dt: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, self.device, mem)  # noqa: E731
        np_ = out_pages(g.kt, g.kt)
        tdt = ttnn.float32 if g.terms_dtype == "fp32" else ttnn.bfloat16
        n_cores = g.item_cores + g.scan_cores  # the probe rows (kda_common.hpp): one 128-B row per core
        probe = ttnn.zeros((n_cores, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.device, mem)
        return dict(
            u=mk((1, 1, g.rows, 3 * g.width), ttnn.bfloat16), lr=mk((1, 1, g.rows, 2 * g.head_dim + g.heads), ttnn.bfloat16),
            f=mk((1, 1, g.rows, g.width), ttnn.float32), gate_out=mk((1, 1, g.rows, g.width), tdt),
            terms=mk((1, 1, g.heads * g.n_chunks * TILE, np_ * TILE), tdt),
            on=mk((1, 1, g.rows, g.width), ttnn.bfloat16),
            probe=probe,
        )  # fmt: skip

    # ------------------------------------------------------------------ the phase's parts
    def _cb(self, spec: CBSpec, crs):
        ttnn = self.ttnn
        fmt = ttnn.float32 if spec.fmt == "fp32" else ttnn.bfloat16
        return ttnn.CBDescriptor(total_size=spec.bytes, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=spec.index, data_format=fmt, page_size=spec.page_bytes)])  # fmt: skip

    def cbs(self) -> list:
        return [self._cb(s, self.item_crs) for s in self.plan.item] + [
            self._cb(s, self.scan_crs) for s in self.plan.scan
        ]

    def semaphores(self) -> list:
        ttnn, g = self.ttnn, self.geo
        sem = lambda i, crs, v=0: ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=v)  # noqa: E731
        out = [
            sem(SEM_QKV, self.item_crs, self.nprod * g.n_chunks if self.h3_preraised else 0),
            sem(SEM_KDA, self.item_crs),
        ]
        out += [sem(SEM_KDA, self.scan_crs), sem(SEM_RMS, self.scan_crs), sem(SEM_ONAGG, self.scan_crs)]
        if self.consumers:
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

            cons = merged_ranges([(int(c.x), int(c.y)) for c in self._consumer_logical()])
            out.append(sem(SEM_ON, cons))
        return out

    def _consumer_logical(self):
        return [c for c in self.consumers if hasattr(c, "x")]

    def consumer_noc(self) -> list:
        return [self.noc(c) if hasattr(c, "x") else tuple(c) for c in self.consumers]

    def roles(self, scr: dict, state_s, state_c, valid_end) -> dict:
        """{role: (kernel file, compile-time args, {(x, y): runtime args}, defines, config)}: item_reader /
        item_compute / item_writer on the item cores, scan_reader / scan_compute / scan_writer on the scan cores.
        state_s / state_c = FusedKDAState.S [1, H, 128, 128] fp32 TILE / .C [1, 3, 1, 3 W] bf16 TILE (one slot)."""
        if tuple(state_c.shape)[-2:] != (1, 3 * self.geo.width) or state_c.layout != self.ttnn.TILE_LAYOUT:
            raise ValueError(f"KDA phase: state C {tuple(state_c.shape)} {state_c.layout} (FusedKDAState.C at 1 slot)")
        ttnn, g = self.ttnn, self.geo
        kt, vt, h, nch, cph, sp = g.kt, g.vt, g.heads, g.n_chunks, g.cph, g.spread
        wt = g.width // TILE
        pw = lambda t: int(t.padded_shape[-1]) // TILE  # noqa: E731  tiles per row = the padded width (lr 264 -> 288)
        unt, lrnt, gnt = pw(scr["u"]), pw(scr["lr"]), pw(scr["f"])
        tb = g.term_bytes
        for k, dt in (
            ("terms", g.terms_dtype),
            ("gate_out", g.terms_dtype),
            ("u", "bf16"),
            ("lr", "bf16"),
            ("f", "fp32"),
        ):
            want = ttnn.float32 if dt == "fp32" else ttnn.bfloat16
            if scr[k].dtype != want:
                raise ValueError(
                    f"KDA phase: {k} is {scr[k].dtype}, the geometry wants {want} (terms_dtype {g.terms_dtype})"
                )
        btile = 2 * g.head_dim // TILE
        if 3 * wt * 64 > 2 * kt * vt * T32:  # the P9 carry row buffer lives in the drained SR ring
            raise ValueError("KDA phase: the carry row buffer does not fit the SR ring")
        addr = lambda t: int(t.buffer_address())  # noqa: E731
        cons = self.consumer_noc()
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        brisc = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
        ncrisc = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)

        def cc(specs, copy_only):
            c = ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=True,
            )
            modes = [ttnn.UnpackToDestMode.Default] * 64
            for sp_ in specs:
                if sp_.fmt == "fp32" and sp_.name in copy_only:  # copy / SFPU-only CBs; FPU operands stay SrcA/B
                    modes[sp_.index] = ttnn.UnpackToDestMode.UnpackToDestFp32
            c.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
            return c

        scan_xy = [self.noc(c) for c in self.scan_cores]
        item_xy = [self.noc(c) for c in self.item_cores]
        item_rt_r, item_rt_c, item_rt_w = {}, {}, {}
        for i, core in enumerate(self.item_cores):
            head, slot = divmod(i, cph)
            nxt = item_xy[head * cph + (slot + 1) % cph]  # the baton's next core
            xy = (int(core.x), int(core.y))
            item_rt_r[xy] = (
                [head, slot, addr(scr["u"]), addr(scr["lr"]), addr(scr["f"]), addr(state_c), addr(valid_end)]
                + [addr(t) for t in self.taps]
                + [addr(self.dt_bias), addr(self.rate), addr(scr["probe"])]
            )
            item_rt_c[xy] = [slot]
            item_rt_w[xy] = [head, slot, addr(scr["terms"]), addr(valid_end), nxt[0], nxt[1], addr(scr["probe"])] + [
                v for b in range(sp) for v in scan_xy[head * sp + b]
            ]
        scan_rt_r, scan_rt_c, scan_rt_w = {}, {}, {}
        lx, ly = scan_xy[0]
        for i, core in enumerate(self.scan_cores):
            head, vblk = divmod(i, sp)
            xy = (int(core.x), int(core.y))
            scan_rt_r[xy] = [
                head,
                vblk,
                addr(state_s),
                addr(scr["terms"]),
                addr(scr["gate_out"]),
                addr(self.w),
                addr(valid_end),
                addr(scr["probe"]),
                g.item_cores,
                sp,
            ]
            scan_rt_c[xy] = []
            peers = [v for b in range(sp) if b != vblk for v in scan_xy[head * sp + b]]
            scan_rt_w[xy] = [head, vblk, addr(scr["on"]), addr(state_s), addr(valid_end), addr(state_c), addr(scr["u"]),
                             int(i == 0), lx, ly, int(i == 0), addr(scr["probe"]), g.item_cores] + peers + [v for xy_ in cons for v in xy_]  # fmt: skip
        d = 1.0 / math.sqrt(g.head_dim)
        return dict(
            item_reader=("kda_item_r.cpp", [kt, kt, h, nch, cph, wt, unt, lrnt, btile, gnt, self.nprod], item_rt_r, [], brisc),
            item_compute=("kda_item_c.cpp", [kt, kt, nch, cph, f32_bits(d), f32_bits(1e-6), f32_bits(self.lower_bound)],
                          item_rt_c, [], cc(self.plan.item, ITEM_COPY_ONLY_CBS)),
            item_writer=("kda_item_w.cpp", [kt, kt, nch, cph, sp, tb], item_rt_w, [], ncrisc),
            scan_reader=("kda_scan_r.cpp", [kt, vt, kt, nch, cph, wt, tb], scan_rt_r, [], brisc),
            scan_compute=("kda_scan_c.cpp", [kt, vt, nch, f32_bits(self.eps), f32_bits(1.0 / g.head_dim), sp], scan_rt_c, [],
                          cc(self.plan.scan, SCAN_COPY_ONLY_CBS)),
            scan_writer=(
                "kda_scan_w.cpp",
                [kt, vt, kt, nch, sp, wt, g.scan_cores, len(cons), unt, 2 * in_pages(kt, vt) * tb], scan_rt_w, [], ncrisc),
        )  # fmt: skip

    def io(self, scr: dict, state_s, state_c, valid_end) -> list:
        return [scr[k] for k in ("u", "lr", "f", "gate_out", "terms", "on", "probe")] + [state_s, state_c, valid_end] + \
            list(self.taps) + [self.dt_bias, self.rate, self.w]  # fmt: skip

    def kernels(self, roles: dict) -> list:
        ttnn = self.ttnn
        out = []
        for name, (src, ct, rt, defs, cfg) in roles.items():
            rta = ttnn.RuntimeArgs()
            for (x, y), args in rt.items():
                rta[x][y] = [int(v) for v in args]
            crs = self.item_crs if name.startswith("item") else self.scan_crs
            kdefs = list(defs) + list(self.kdefines["item" if name.startswith("item") else "scan"])  # the CB maps
            out.append(ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, src),
                                             source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
                                             compile_time_args=[int(v) for v in ct], runtime_args=rta,
                                             defines=kdefs, config=cfg))  # fmt: skip
        return out

    def program(self, scr: dict, state_s, state_c, valid_end):
        """the standalone one-phase ProgramDescriptor (+ its io list)"""
        ttnn = self.ttnn
        roles = self.roles(scr, state_s, state_c, valid_end)
        return ttnn.ProgramDescriptor(kernels=self.kernels(roles), semaphores=self.semaphores(), cbs=self.cbs()), \
            self.io(scr, state_s, state_c, valid_end)  # fmt: skip

    def probe_rows(self, scr: dict, device_index: int = 0) -> dict:
        """decode the probe rows of one chip (kda_common.hpp): per item core (reader words 16.., writer 0..) and scan
        core: valid_end as read, counts, checksums; summary = the distinct values per field"""
        ttnn, g = self.ttnn, self.geo
        t = scr["probe"]
        parts = ttnn.get_device_tensors(t) if hasattr(t, "device") and self.device.get_num_devices() > 1 else [t]
        rows = ttnn.to_torch(parts[device_index]).to(__import__("torch").int64) & 0xFFFFFFFF
        ni = g.item_cores

        def col(r0, r1, w):
            return sorted({int(v) for v in rows[r0:r1, w].tolist()})

        return dict(
            item_w=dict(magic=col(0, ni, 0), ve=col(0, ni, 1), items=col(0, ni, 3), chk_terms=col(0, ni, 4)[:4]),
            item_r=dict(magic=col(0, ni, 16), ve=col(0, ni, 17), items=col(0, ni, 19), chk_u=col(0, ni, 20)[:4], chk_f=col(0, ni, 21)[:4]),
            scan_w=dict(magic=col(ni, ni + g.scan_cores, 0), ve=col(ni, ni + g.scan_cores, 1), nvalid=col(ni, ni + g.scan_cores, 2),
                        chunks=col(ni, ni + g.scan_cores, 3), chk_on=col(ni, ni + g.scan_cores, 4)[:4], chk_fs=col(ni, ni + g.scan_cores, 5)[:4],
                        carry_rows=col(ni, ni + g.scan_cores, 8)),
            scan_r=dict(magic=col(ni, ni + g.scan_cores, 16), ve=col(ni, ni + g.scan_cores, 17), nvalid=col(ni, ni + g.scan_cores, 18),
                        chunks=col(ni, ni + g.scan_cores, 19), chk_in=col(ni, ni + g.scan_cores, 20)[:4], chk_s=col(ni, ni + g.scan_cores, 21)[:4]),
        )  # fmt: skip

    def __call__(self, scr: dict, state_s, state_c, valid_end):
        """standalone launch (the mock unit / the device bench, captured in a trace by the caller): on = scr["on"]"""
        prog, io = self.program(scr, state_s, state_c, valid_end)
        self.ttnn.generic_op(io, prog)
        self.calls += 1
        return scr["on"]
