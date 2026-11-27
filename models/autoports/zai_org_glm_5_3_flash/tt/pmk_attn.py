# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill megakernel (PMK): the ATTENTION phase set of a sparse_dsa layer program (MLA absorbed + DSA), piece "attn"
of doc/ttnn/16-prefill-production/prefill_megakernel.md (§0 pieces; the contract: §4b hand-off map, §4c streaming
readers). One chip of the TP8 mesh = the head shard H_r = 8 (TP2 bench: 32); the bucket C = 2,048 rows (CT = 64 query
tiles); SP rows = the chip's 256 rows (TP2: 1,024) for the indexer and for the mHC post phase.

Phases of the sparse_dsa layer program between P2 (AG(x̂), coll) and P8 (RS(o), coll); the numbers continue §3's table:

  A1  q_a | kv_a (+ I1 w_x)     MM grid, PfStreamPhase (§4c)     x̂ SP rows [256, 4096] -> qk [256, 2048] bf16 (+ y_ix
                                                                [256, 288] fp32): REPLICATED weights, SP rows only
  A2  norms                     norm cores (lat_c.cpp lineage)   RMSNorm_512(kv_a) -> latent_sp [256, 512] bf16 (-> the
                                                                cache piece's page write AND the AG of A3); 1/rms(q_a)
  A3  AG(latent_sp, q_a_sp, ids)  coll (C1-style, 32-row blocks)  latent_all [2048, 512] lands in the MM grid's L1 store
                                                                (32 KB per core over 64 cores: the K/V of A6), q_a_all,
                                                                the sparse ids; landed counters (FabricStream 14 / 15)
  A4  q_b (folded norm)         MM grid, PfStreamPhase           q_a_all -> q [2048, H_r 256] bf16 (RS: qinv)
  A5  W_uk                      MM grid, PfStreamPhase (grouped) q -> q_lat [1, H_r, C, 512] bf16; SEM 5 (+1 per 32-row
                                                                block of ALL heads, row order)
  A6  ATTENTION (kernels/pmk/attn_*.cpp)                        work items (h, QG q tiles) on the 104-core MM grid: keys
                                                                = [context] ++ [chunk tiles 0..qt] (dense: paged cache
                                                                + the L1 latent store; sparse: the union of the rows'
                                                                selected pools from the ROW_MAJOR latent store + masks)
                                                                -> o_lat [1, H_r, C, 512]; SEM 7 (+1 per item)
  A7  W_uv                      MM grid, PfStreamPhase (grouped) o_lat head rows -> o [2048, H_r 256] bf16
  A8  o_proj                    MM grid, PfStreamPhase           o -> a [2048, 4096] bf16 partial -> H5 (SEM 8 per
                                                                32-row owner block) -> P8 RS(o) (coll)
  I2  pooled keys + ring rows   index cores (ix_c.cpp lineage)   y_ix SP rows -> the 64 pooled keys of the chip's rows
  I3  q_idx                     MM grid, PfStreamPhase           q_a SP rows -> q_idx [256, 32 x 128] bf16 (RS: qinv)
  I4  scores                    MM grid                          per (q tile, pool tile): 32 heads x 4 tile-matmuls +
                                                                head-weighted sum -> scores [256, P] fp32 (DRAM)
  I5  top-512 per row           MM-core NCRISCs                  per-row radix select -> ids [256, 512] u32 sorted
I4 / I5 are specified here (geometry, FLOPs, L1) and left to the next cycle (README §open).

Rev 3 (the silicon result of rev 2, README §7): the attention is a BYTES problem, not a FLOP problem -- every (head,
q tile) item re-read its K tiles from DRAM (64 FLOP/B against a HiFi4 ridge of ~340). Rev 3: Q once per item, QG
query tiles share every K tile (in-place running partials), the chunk's K/V from the L1-sharded latent store (NoC
L1 -> L1), batch-granular reads with one barrier. `math_estimate` carries the traffic model (DRAM / NoC / barriers)
calibrated on the measured 9,699 us.

Rev 4 (the silicon result of rev 3, README §8): rev 3 was SLOWER (18,735 us, the compute waiting 79 %) because every
core walked its items' chunk tiles in the same ascending order at the same pace: at any moment the 104 readers hit
the same NK owner cores of the L1-sharded store (tile k lives on core k), a 104-to-1 NoC fan-in that serialises
every batch on one owner's egress. `noc_sim` is the cycle-level host model of that reader (owner FIFOs, reader
ingress, the CB_AK double buffer, the compute consuming behind): the ascending order reproduces the measured span,
the rev 4 per-core ROTATED batch order (`batch_order`) spreads the readers over all owners and brings the reader
under the compute. Sparse: the row-major pool gather read 4 KB from a 1-KB-page interleaved store (rows p, p + 8,
p + 16, p + 24 instead of p .. p + 3): now one page read per row (`rm_rows_per_page`); the mask constant is -1e9
(exact under fp32 adds, bf16-safe) and `attention_kernel_model` is the pure-python trace of the kernel's batch
arithmetic (all-masked first batches, rotated orders) proving the stats stay finite.

Rev 7 (the silicon result of rev 6, README §9): rev 6 is numerically right (tp2-dense PCC 0.999996, tp2-sparse
0.999969) but the sparse unit took 35,051 us with the BRISC 97 % in mask building (one fp32 element store + bitmap
test per mask element, ~35,600 cycles per tile, 1,270 tiles per core) and the dense unit 9,058 us with the BRISC 76 %
in read barriers (2.4 B/cycle per reader against the L1-sharded store, the compute waiting 60 %). Rev 7:
  * sparse masks: the per-row membership is built on the NCRISC as POOL WORDS (one word per (pool, q tile), bit =
    row; `abm_words`), one item ahead in the double-buffered CB_ABM; the BRISC writes a mask tile as 32 bf16 rows
    from its K tile's 8 pool words (kernels/pmk/attn_common.hpp mask_row: branch-free, ~1,900 cycles per tile; the
    diagonal tile's causal bound + open-pool tail patch two words). The mask value is bf16(-1e9) = NEG_MASK_BF16;
    the output is BITWISE rev 6's (`attention_kernel_model(mask_neg=...)`, test_mask_value_independence);
  * dense fan-in: `dual_noc` (default on) splits every chunk batch's tile reads over both NoCs (BRISC even tiles on
    NoC0, NCRISC odd tiles on NoC1; CB_AGO pages + a flag word per CB_AK slot), so each owner serves through both of
    its NIUs and each reader lands through both of its own; `kslots` (CB_AK slots, default 2) deepens the batch
    pipeline when L1 allows; `noc_sim(noc_ports=2)` is the model's view.
  Silicon rev 7 (README §10): tp2-dense 4,773 us PASS (QG 2: 4,100 -- now the dense default), tp2-sparse 11,971 us:
  the BRISC 56-59 % writing masks (512 stores per tile ~7,300 cycles: an L1 store from a Tensix RISC is ~8-10 cycles)
  and the NCRISC 64-67 % building pool words (~36 cycles per id x 29.5k ids per item = the item's compute time).

Rev 8 (README §10): NO per-id and NO per-element RISC work on the sparse path.
  * the selection arrives as per-row BITMAPS (`select_bitmap_host`; sel [1, 1, C, W] uint32 ROW_MAJOR, bit q = pool
    q selected, context pools then the chunk's 512; W = `sel_row_words`): the NCRISC burst-reads the item's rows
    into CB_ABM and ORs them into the union (word ops only; one item ahead as in rev 7);
  * K tiles are FIXED 8-pool groups (group g <=> byte g of a row's bitmap; gathered iff the union touches it), so a
    row's membership byte for a tile is ONE byte load (`group_tiles`: the same tile count as the compact union for
    random or clustered selections; up to 8x more only for selections scattered one pool per group);
  * the BRISC writes a mask OPERAND of 32 words per (tile, q tile): column 0 of a bf16 tile = the row's membership
    byte as an integer (~800 cycles; the ring pages zeroed once per phase);
  * the TRISC expands it: one tile matmul B_col @ E8 (E8 row 0 = 2^-s on pool s's four columns, CB_AID page 2) gives
    byte x 2^-s per element exactly, and one SFPI pass (kernels/pmk/attn_mask_sfpu.hpp) adds the exact fp32 -1e9
    where floor(.) is even, with the scale multiply fused (one SFPU pass fewer per masked pair). The diagonal tile
    keeps the causal triangle add; CB_AID is bf16 (triangle, identity, E8). The output stays rev 6's (§9a).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import torch

try:
    import ttnn
except Exception:  # noqa: BLE001 -- the pure-python tests run without ttnn
    ttnn = None

KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
TILE = 32
DT = 16  # latent tiles (512 / 32)
ST = 2  # stats tiles (m, l)
KPOOL, TOPK, DENSE_MAX, POOLS_PER_TILE = 4, 512, 2051, 8
BLOCK = 64  # latent cache page (tokens), prefill_tp.BLOCK
NONE_ID = 0xFFFFFFFF
NEG_MASK = -1.0e9  # the mask / triangle constant (rev 4; rev 3's -1e30): s * scale + NEG_MASK stays exact in fp32 and
# representable in bf16, exp(NEG_MASK - m) == 0 on the SFPU as on the host; -1e9 << any finite score
NEG_MASK_BF16 = -998244352.0  # the bf16 triangle tile's value = bf16(-1e9) (0xCE6E; attn_common.hpp NEG16; rev 8's
# SFPU mask adds the exact fp32 -1e9). Any constant <= -2^29 (ulp 64 > 2 |s|) and <= -104 (exp underflows) gives the
# SAME output: masked keys contribute exp(NEG - m') == 0 exactly and an all-masked batch's garbage is rescaled by
# a = exp(NEG - m_real) == 0
# ---------------------------------------------------------------- CB map (§4b: Stage 14's indices on the MM cores;
# the attention takes 45..63 on the same cores: INVH 36, OUT32 38, STAGE 39, IN0 40, PART 41, OUT 42, rings 43..44 stay)
# Rev 7: the unused APB / APBO slots become the cross-RISC hand-off CBs: ABM (NCRISC -> BRISC: the item's pool words,
# sparse; 1 pad page in dense) and AGO (BRISC -> NCRISC: the dual-NoC batch pages, dense; 1 pad page in sparse)
ATT_CB0 = 45
ACB = dict(
    AQ=45, AK=46, AKRM=47, AKT=48, AMASK=49, AINFO=50, AS=51, AM=52, AP=53, AE=54, APA=55, ABM=56, APAO=57, AGO=58,
    ACOEF=59, ADIAG=60, AID=61, AOUT=62, ASCR=63,
)  # fmt: skip
PROBE_CB = 31  # kernels/mk/probe.hpp's scratch CB (probe builds only)
RINGS_MAX = ATT_CB0 - 43  # rings 43, 44: a sparse_dsa layer program may stream through at most 2 ring CBs
# ---------------------------------------------------------------- semaphores (Stage 16 geometry note, NOTE_16_GEOMETRY
# .txt: the attention phase owns 5 (QKV), 7 (ON), 8 (A) of the non-lane map; monotonic per layer, wait_min, reset once)
SEM_QLAT = 5  # "QKV": A5 -> A6, 32-row blocks of q_lat complete (all heads), row order
SEM_OL = 7  # "ON": A6 -> A7, items done on every W_uv consumer core (v1: W_uv waits >= items, ol_counts)
SEM_A = 8  # "A": A8 o_proj partial -> P8 lanes, §4b H5 (owner block)
# the AG-landed counters are the coll piece's (FabricStream 14 / 15): latent_all / q_a_all / ids blocks landed. A6 waits
# on the latent counter (LATB) before each chunk tile and on the ids counter (sparse) before an item; both ids are
# constructor arguments (defaults = FabricStream's 14 / 15, to be confirmed by coll / family)
SEM_LATB_DEFAULT, SEM_IDS_DEFAULT = 14, 15
ATT_REV = 8  # kernel revision (8: selection bitmaps, fixed 8-pool groups, 32-word mask operands expanded on the
# TRISC (matmul + SFPI), bf16 CB_AID; 7: pool words on the NCRISC + row-wise bf16 masks, dual-NoC dense reads, kslots;
# 6: tilize formats reconfigured + fixed NK x QG mask pages per masked batch; 5: the sparse context tilize writes K
# tile i to AKT tiles i DT (both indices); 4: rotated batch order, dm0 probe, row-page pool gather, -1e9 masks); 3 (the
# compile signature of a long-lived process); 3: QG items, Q once, in-place partials,
# batch reads, L1 latent store, probes; 2: semaphores 5 / 7 / 8, TP2 geometry


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _a64(n: int) -> int:
    return -(-max(n, 1) // 64) * 64


# ---------------------------------------------------------------- geometry
@dataclass(frozen=True)
class AttnGeo:
    """one chip's attention geometry at a bucket"""

    heads: int  # H_r: heads on this chip (TP8: 8, TP2: 32)
    rows: int  # C: the bucket (query rows of the chunk, every chip attends ALL rows for its heads)
    chunk_start: int  # position of row 0 (multiple of 128)
    cores: int = 104  # the MM grid (§2: 64 members + 8 roots + 32 MM-only)
    nk_: int | None = None  # K tiles per batch (PMK_ATT_NK); None = 4 dense (fewer softmax passes: +6 % FPU share),
    # 2 sparse (the gathered-tile CBs AKRM / AKT scale with nk)
    qg_: int | None = None  # q tiles per item (PMK_QG); None = auto_qg (4 / 2 / 1 by the estimated span x balance)
    max_seq: int = 32768  # the slot's capacity (page table width, bitmap sizes)
    sp_rows: int = 256  # the chip's SP rows (TP8: 256, TP2: 1,024; NOTE_16_GEOMETRY)
    lat_l1: bool = True  # the chunk latent store L1-sharded over the MM grid (else a DRAM tensor)
    obf16: bool | None = None  # bf16 O partials (None: fp32 when the L1 plan fits, else bf16 -- rev 4; rev 3 made the
    # sparse units bf16 by default)
    dual_noc: bool = True  # rev 7 (dense): the chunk-tile reads of a batch split over both NoCs (PMK_ATT_DUAL_NOC)
    kslots: int = 2  # rev 7: CB_AK slots (batches in flight + 1; PMK_ATT_KSLOTS); the L1 plan must fit
    bm_pages_: int | None = None  # rev 7 (sparse): CB_ABM pages (2: the next item's bitmaps fetched while this
    # item streams; 1: serial, the large slots); None = 2 when the L1 plan fits, else 1
    exp_approx: bool = False  # rev 9: the two softmax exps on the SFPU's APPROX path (PMK_ATT_EXP_APPROX) -- OFF by
    # the rev 9 ruling: PCC 0.98 with the per-batch rescale (the error compounds through a), 0.99977 with max_lazy
    exp_21f: bool = False  # rev 10: E only on the exp_21f fp32 SFPI path (~2^-8 relative, the bf16 resolution the
    # P V operand rounds E to; a keeps the accurate exp; PMK_ATT_EXP_21F)
    ctx_l1_: bool | None = None  # rev 10 (sparse): the context rows in an L1 store (32-KB tiles over the MM-only
    # cores, read whole over both NoCs, no tilize) instead of 1-KB DRAM row gathers; None = when the store's share
    # per owner core is <= CTX_L1_MAX_KB (PMK_CTX_L1)
    max_lazy: float | None = None  # rev 9: rescale the partial only when a row's max grows by more than this
    # threshold T (the batch keeps m as its stabiliser otherwise: exact for any T <= ~80, E <= e^T; PMK_ATT_MAX_LAZY =
    # the fp32 bits of T); None = the rev 3..8 rescale on every batch

    @property
    def nk(self) -> int:
        return self.nk_ if self.nk_ is not None else (2 if self.sparse else 4)

    @property
    def ctx_l1(self) -> bool:
        if not self.sparse:
            return False
        if self.ctx_l1_ is not None:
            return self.ctx_l1_
        return ctx_store_kb(self) <= CTX_L1_MAX_KB

    @property
    def bm_pages(self) -> int:
        if not self.sparse:
            return 1
        if self.bm_pages_ is not None:
            return self.bm_pages_
        import dataclasses

        return 2 if l1_plan(dataclasses.replace(self, bm_pages_=2))["fits"] else 1

    @property
    def qg(self) -> int:
        """q tiles per item: every K tile read serves qg query tiles (rev 8: 2 for both regimes -- the dense silicon
        ladder: QG 1 4,773 us, QG 2 4,100 us with dual_noc)"""
        return self.qg_ if self.qg_ is not None else 2

    @staticmethod
    def tp8(chunk_start: int = 0, **kw) -> "AttnGeo":
        return auto_qg(AttnGeo(heads=8, rows=2048, chunk_start=chunk_start, sp_rows=256, **kw))

    @staticmethod
    def tp2(chunk_start: int = 0, **kw) -> "AttnGeo":
        """the desktop-0 pair units: 1,024 SP rows, 32 heads per chip; the attention stores (q_lat 64 MB, o_lat) in DRAM"""
        return auto_qg(AttnGeo(heads=32, rows=2048, chunk_start=chunk_start, sp_rows=1024, **kw))

    @property
    def ct(self) -> int:
        return self.rows // TILE

    @property
    def ct0(self) -> int:
        return self.chunk_start // TILE

    @property
    def sparse(self) -> bool:
        return self.chunk_start + self.rows > DENSE_MAX

    @property
    def groups(self) -> int:
        return -(-self.ct // self.qg)

    @property
    def items(self) -> int:
        return self.heads * self.groups

    @property
    def o_bf16(self) -> bool:
        if self.obf16 is not None:
            return self.obf16
        import dataclasses

        return not l1_plan(dataclasses.replace(self, obf16=False, bm_pages_=self.bm_pages_ or 1))["fits"]


def item_qg(geo: AttnGeo, qt0: int) -> int:
    return min(geo.qg, geo.ct - qt0)


def item_compute(geo: AttnGeo, qt0: int) -> int:
    """(q tile, K tile) pairs of item qt0: context tiles x qg + the causal chunk pairs"""
    qg = item_qg(geo, qt0)
    return sum(geo.ct0 + qt0 + j + 1 for j in range(qg))


def item_loads(geo: AttnGeo, qt0: int) -> int:
    """K tiles READ by item qt0 (each once): context + chunk tiles 0 .. qt0 + qg - 1"""
    return geo.ct0 + qt0 + item_qg(geo, qt0)


def attn_items(geo: AttnGeo) -> list:
    """the balanced item lists: per core a list of (h, qt0). Items sorted by compute (descending), dealt in a zigzag
    over the cores (LPT-like), so every core's (q tile, K tile) total is within one item of the mean. Inside a core:
    ascending qt0 (the q_lat / latent block counters are reached in order)."""
    items = sorted(((item_compute(geo, qt0), h, qt0) for h in range(geo.heads) for qt0 in range(0, geo.ct, geo.qg)),
                   reverse=True)  # fmt: skip
    lists = [[] for _ in range(geo.cores)]
    for i, (_, h, qt0) in enumerate(items):
        p, k = divmod(i, geo.cores)
        core = k if p % 2 == 0 else geo.cores - 1 - k
        lists[core].append((h, qt0))
    for lst in lists:
        lst.sort(key=lambda hq: (hq[1], hq[0]))
    return lists


def ol_counts(geo: AttnGeo, lists: list) -> list:
    """per row block qt: the SEM_OL count that guarantees the block is done. Cores progress in their own list order
    and SEM_OL sums over cores, so no count short of `items` guarantees a given block (another core may have
    signalled later blocks first). v1 rule: W_uv starts after the whole attention phase: wait SEM_OL >= items."""
    return [geo.items] * geo.ct


def balance(geo: AttnGeo) -> dict:
    lists = attn_items(geo)
    comp = [sum(item_compute(geo, q) for _, q in lst) for lst in lists]
    loads = [sum(item_loads(geo, q) for _, q in lst) for lst in lists]
    return dict(items_per_core=(min(len(l) for l in lists), max(len(l) for l in lists)), pairs_min=min(comp),
                pairs_max=max(comp), pairs_mean=sum(comp) / len(comp), loads_max=max(loads))  # fmt: skip


def auto_qg(geo: AttnGeo) -> AttnGeo:
    """qg_ None -> the qg in (4, 2, 1) that fits the L1 budget and minimises the estimated span x the balance factor
    (slowest core's (q, K) pairs / the mean): the K reuse of a larger qg against its item granularity (at TP8 dense,
    8 x 16 items of qg 4 leave the largest item (250 pairs) above the per-core mean of 160)"""
    import dataclasses

    if geo.qg_ is not None:
        return geo
    best = None
    for qg in (2, 1):  # rev 8: QG 2 unless it does not fit (the silicon ladder, README §10); 4 never fit with fp32 O
        g = dataclasses.replace(geo, qg_=qg)
        if l1_plan(g)["fits"]:
            best = (0, g)
            break
    return best[1]


# ---------------------------------------------------------------- the per-core batch order (rev 4)
def batch_order(core: int, it: int, nb: int) -> list:
    """the order in which core `core` reads the nb chunk batches of its it-th item: rotated by (core + it) mod nb --
    the kernel's rule (attn_dm0.cpp `rot`), so at any moment the 104 readers of the L1-sharded store spread over
    its owners instead of all reading owner k0 .. k0 + NK - 1 together (rev 3's ascending order, README §8). The
    online softmax is order-free; the compute's diagonal / prefix rules use the batch's own k0."""
    r = (core + it) % nb
    return [(b + r) % nb for b in range(nb)]


def item_batches(geo: AttnGeo, core: int, it: int, qt0: int, rotate: bool = True) -> list:
    """the K batches of item (core, it, qt0) in read order: [(k0, nk, chunk)]; context batches first (dense: from
    the paged cache, DRAM), then the chunk batches (the L1 store) in `batch_order`"""
    nk = geo.nk
    out = [(j0, min(nk, geo.ct0 - j0), False) for j0 in range(0, geo.ct0, nk)]
    nchunk = qt0 + item_qg(geo, qt0)
    nb = -(-nchunk // nk)
    order = batch_order(core, it, nb) if rotate else list(range(nb))
    out += [(b * nk, min(nk, nchunk - b * nk), True) for b in order]
    return out


def batch_pairs(geo: AttnGeo, qt0: int, k0: int, nk: int, chunk: bool) -> int:
    """(q tile, K tile) pairs the compute performs on one batch (the chunk prefix rule per q tile)"""
    qg = item_qg(geo, qt0)
    if not chunk:
        return qg * nk
    return sum(sum(1 for i in range(nk) if k0 + i <= qt0 + j) for j in range(qg))


# NoC fan-in model (rev 4; the dm0 / dm1 "16 ms" of README §8). Owner egress = one L1's NoC read-response path under
# many requesters; fitted on the rev 3 silicon unit (u2: 16,014 us dm1 span, 79 % compute wait) -- see test_dm1_model.
NOC_OWNER_BPC = 11.0  # bytes per cycle an owner core returns under the 104-way fan-in (fitted on u2: the ascending
# order gives 15,620 us / 78.6 % wait against 16,014 / 79.0 % measured; 64 is the link peak)
NOC_READER_BPC = 64.0  # a reader's ingress (one NoC port)
NOC_LAT_CYC = 900  # request + response latency through the mesh (not fitted)
NOC_ISSUE_CYC = 50  # BRISC cycles per 2-KB page request (attn_dm0.cpp's read loop: ~35 instructions, disassembled)
DRAM_TILE_CYC = 1600  # a 32-KB context tile from DRAM at the calibrated 325 GB/s, per reader (dense context, Q)
MASK_CYC_REV6 = 35600  # BRISC cycles per rev 6 membership mask tile, SILICON (1,024 fp32 element stores with a
# bitmap test each: 34.5 ms x 97.1 % over 1,270 tiles per core on the rev 6 sparse unit; the rev 4 planning value
# was 2,500)
MASK_CYC_REV7 = 7300  # rev 7 SILICON: 32 rows x 16 word stores per tile at ~8-10 cycles per RISC L1 store (dm0 mask
# 56-59 % of the 11,971-us unit over 1,270 tiles); the planning value was 1,900
MASK_CYC = 800  # rev 8: 32 x (byte load + bf16 convert + 1 word store) per mask operand (attn_common.hpp mask_col_row)
BITMAP_CYC_PER_ID = 36  # rev 7 SILICON: one read-modify-write per id on the NCRISC (2 dependent L1 loads + 1 store;
# dm1 bitmaps 64-67 % over ~29.5k ids per item); rev 8 has no per-id work (fetch_item_bitmaps: word ORs)
UNION_CYC_PER_WORD = 10  # rev 8 NCRISC: one load + OR per (row, union word)
COMPUTE_CAL = 1.466  # the compute pipeline vs the planning cycles: u2's PACK buckets QK + softmax + PV = 3,345 us
# against math_estimate's 2,282 (TP2 dense): the SFPU passes cost ~2x the 256-cycle planning value


def noc_sim(geo: AttnGeo, rotate: bool = True, owner_bpc: float = NOC_OWNER_BPC, cal: float = COMPUTE_CAL,
            mhz: float = 1350.0, q_per_batch: bool = False, max_steps: int = 400000, replicas: int = 0,
            noc_ports: int = 1) -> dict:  # fmt: skip
    """cycle-level fluid model of the attention cores' readers against the L1-sharded chunk store: tile k of the
    store lives on owner core k; every reader (104 cores, their `attn_items` lists) reads its batches in
    `item_batches` order (rev 3: ascending, rev 4: rotated), NK tiles per batch, one barrier per batch, CB_AK
    double-buffered (a batch's reads start once the compute popped the batch two behind), the compute consuming
    each batch in the calibrated cycles. An owner's egress (owner_bpc) is shared FAIRLY among the readers with a
    request outstanding on it (the NoC's round-robin arbitration: the readers of one owner progress together, so
    rev 3's lockstep never heals -- a FIFO owner would let them drift apart); a reader's ingress is capped at
    NOC_READER_BPC; a batch pays NOC_LAT_CYC once. replicas = r > 0: tiles 0 .. r - 1 (the hottest, read by every
    item) also live on cores ct .. ct + r - 1 and odd readers take the replica (the next lever: halves the hottest
    owner's bytes; NOT in the rev 4 kernels, the store is the AG's landing store). noc_ports = 2 (rev 7 dual_noc):
    a batch's tiles are read through both NoCs -- the owner's egress and the reader's ingress double (each NIU has
    its own L1 port and links), the fan-in per NoC halves. Returns the span and the per-role shares (us)."""
    import heapq

    import numpy as np

    owner_bpc = owner_bpc * noc_ports
    reader_bpc = NOC_READER_BPC * noc_ports
    nk = geo.nk
    lists = attn_items(geo)
    mm = 64
    per_pair = 2 * DT * mm + (1 + 1 + 3) * 256
    per_upd = DT * 64 + (nk + 2 + 2 + 2 + nk + 3) * 256
    tile_bytes = float(DT * 2048)
    n = geo.cores
    plans = []
    for c, lst in enumerate(lists):
        flat = []
        for it, (_h, qt0) in enumerate(lst):
            first = True
            for k0, bnk, chunk in item_batches(geo, c, it, qt0, rotate):
                pairs = batch_pairs(geo, qt0, k0, bnk, chunk)
                upd = sum(1 for j in range(item_qg(geo, qt0)) if (not chunk) or k0 <= qt0 + j)
                cyc = (pairs * per_pair + upd * per_upd) * cal
                owners = [k0 + i for i in range(bnk)] if chunk else []
                if replicas and c % 2 == 1:
                    owners = [k + geo.ct if k < replicas else k for k in owners]
                ctx = 0 if chunk else bnk
                qcyc = DRAM_TILE_CYC * (item_qg(geo, qt0) if first or q_per_batch else 0)
                flat.append((cyc, owners, ctx, qcyc))
                first = False
            flat[-1] = (flat[-1][0] + item_qg(geo, qt0) * (DT * 64 + 6 * 256) * cal,) + flat[-1][1:]
        plans.append(flat)
    nb_max = max(len(pl) for pl in plans)
    comp_done = np.zeros((n, nb_max))
    cur = [0] * n  # the batch each reader is on
    rem = np.zeros((n, nk))  # outstanding bytes per reader slot
    own = np.full((n, nk), -1, dtype=np.int64)  # owner per slot (-1: none)
    active = np.zeros(n, dtype=bool)  # reader has outstanding reads
    wait_slot = np.zeros(n)
    wait_noc = np.zeros(n)
    batch_t0 = np.zeros(n)
    heap = [(0.0, c) for c in range(n)]  # (time, reader): reader may ISSUE its current batch
    heapq.heapify(heap)
    t = 0.0
    finished = 0
    steps = 0

    def issue(c, now):
        b = cur[c]
        cyc, owners, ctx, qcyc = plans[c][b]
        start = now + qcyc + ctx * DRAM_TILE_CYC
        if owners:
            own[c, :] = -1
            rem[c, :] = 0.0
            for i, k in enumerate(owners):
                own[c, i] = k
                rem[c, i] = tile_bytes
            active[c] = True
            batch_t0[c] = start
            return start
        return start

    def complete(c, now):  # the batch's reads landed (or none): push, compute, schedule the next issue
        b = cur[c]
        cyc = plans[c][b][0]
        landed = now + (NOC_LAT_CYC if plans[c][b][1] else 0)
        wait_noc[c] += landed - batch_t0[c]
        cs = max(landed, comp_done[c, b - 1] if b else 0.0)
        comp_done[c, b] = cs + cyc
        active[c] = False
        own[c, :] = -1
        cur[c] = b + 1
        if b + 1 < len(plans[c]):
            nxt = landed
            if b + 1 >= 2:
                nxt = max(nxt, comp_done[c, b - 1])
            wait_slot[c] += nxt - landed
            heapq.heappush(heap, (nxt, c))
            return False
        return True

    pending = []  # readers whose issue produced no NoC reads: complete at `start`
    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("noc_sim: too many steps")
        # rates of the outstanding slots: fair share of the owner, capped by the reader's ingress
        if active.any():
            flat_own = own[active]
            cnt = np.bincount(flat_own[flat_own >= 0], minlength=geo.ct + replicas)
            rate = np.zeros_like(rem)
            m = own >= 0
            rate[m] = owner_bpc / cnt[own[m]]
            tot = rate.sum(axis=1, keepdims=True)
            scale = np.where(tot > reader_bpc, reader_bpc / np.maximum(tot, 1e-9), 1.0)
            rate = rate * scale
            with np.errstate(divide="ignore", invalid="ignore"):
                dt_fluid = np.where(m, rem / np.maximum(rate, 1e-12), np.inf).min()
        else:
            dt_fluid = np.inf
            rate = None
        t_next = heap[0][0] if heap else np.inf
        if pending:
            t_next = min(t_next, min(pending)[0])
        if dt_fluid == np.inf and t_next == np.inf:
            break
        t_new = min(t + dt_fluid, t_next)
        if rate is not None and t_new > t:
            rem -= rate * (t_new - t)
        t = t_new
        # completions of the fluid reads
        if rate is not None:
            done_slots = (own >= 0) & (rem <= 1e-6)
            if done_slots.any():
                own[done_slots] = -1
                rem[done_slots] = 0.0
                for c in np.nonzero(active & ~(own >= 0).any(axis=1))[0]:
                    if complete(int(c), t):
                        finished += 1
        # timed issues
        while heap and heap[0][0] <= t + 1e-9:
            _, c = heapq.heappop(heap)
            start = issue(c, t)
            if not active[c]:  # no NoC reads in this batch (context-only): completes at `start`
                pending.append((start, c))
            elif start > t:  # the DRAM part precedes the NoC reads: hold the slots until then
                rem[c, :] = np.where(own[c, :] >= 0, rem[c, :] + (start - t) * 0.0, rem[c, :])
        if pending:
            keep = []
            for st, c in pending:
                if st <= t + 1e-9:
                    if complete(c, t):
                        finished += 1
                else:
                    keep.append((st, c))
            pending = keep
    spans = comp_done.max(axis=1)
    busy = [sum(x[0] for x in pl) for pl in plans]
    k = int(spans.argmax())
    span = float(spans[k])
    return dict(span_us=round(span / mhz, 1), compute_us=round(max(busy) / mhz, 1),
                compute_wait_pct=round(100 * (1 - busy[k] / span), 1), reader_slot_wait_us=round(wait_slot[k] / mhz, 1),
                reader_noc_wait_us=round(wait_noc[k] / mhz, 1), owner_bpc=owner_bpc, rotate=rotate,
                noc_ports=noc_ports, hottest_owner_mb=max(owner_load_mb(geo)), steps=steps)  # fmt: skip


def owner_load_mb(geo: AttnGeo) -> list:
    """bytes each owner core of the store serves over the phase (tile k: every item with qt0 + qg - 1 >= k)"""
    kb = DT * 2048
    return [round(sum(1 for h in range(geo.heads) for qt0 in range(0, geo.ct, geo.qg) if qt0 + item_qg(geo, qt0) - 1 >= k)
                  * kb / 1e6, 1) for k in range(geo.ct)]  # fmt: skip


# ---------------------------------------------------------------- L1
def scratch_layout(geo: AttnGeo) -> dict:
    """CB_ASCR (rev 7): the page-table row page at 0, the CB_AK slot flag words at `flag_off` (64 B: dual-NoC), the
    probe flush (64 B)"""
    pt = _a64(geo.max_seq // BLOCK * 4)
    return dict(pt_bytes=pt, flag_off=pt, total=pt + 64 + 64)


def scratch_bytes(geo: AttnGeo) -> int:
    return scratch_layout(geo)["total"]


def sel_row_words(ctx_pools: int) -> int:
    """words of one selection-bitmap row (attn_common.hpp sel_row_words): context pools + the chunk's 512, 64-B rounded"""
    return -(-(ctx_pools + 512) // 512) * 16


def abm_words(geo: AttnGeo) -> int:
    """the CB_ABM page of one item (attn_common.hpp abm_words): 32 qg selection rows + the union bitmap"""
    ctx_pools = geo.ct0 * POOLS_PER_TILE
    return 32 * geo.qg * sel_row_words(ctx_pools) + -(-ctx_pools // 32)


def cb_spec(geo: AttnGeo) -> list:
    """[(name, dtype name, page bytes, pages)] of the attention CBs on one core"""
    nk, qg = geo.nk, geo.qg
    ob = ("BF16", 2048) if geo.o_bf16 else ("FP32", 4096)
    sp = nk * DT if (geo.sparse and not geo.ctx_l1) else 1  # the row-major gather CBs (rev 10 ctx_l1: unused)
    abm = (_a64(abm_words(geo) * 4), geo.bm_pages) if geo.sparse else (64, 1)
    ago = (64, 4) if geo.dual_noc else (64, 1)
    return [("AQ", "BF16", 2048, qg * DT), ("AK", "BF16", 2048, geo.kslots * nk * DT), ("AKRM", "BF16", 2048, sp),
            ("AKT", "BF16", 2048, sp), ("AMASK", "BF16", 2048, 2 * nk * qg), ("AINFO", "U32", 64, 8),
            ("AS", "FP32", 4096, nk), ("AM", "FP32", 4096, 2), ("AP", "BF16", 2048, nk), ("AE", "FP32", 4096, nk),
            ("APA", "FP32", 4096, qg * ST), ("ABM", "U32", abm[0], abm[1]), ("APAO", ob[0], ob[1], qg * DT),
            ("AGO", "U32", ago[0], ago[1]), ("ACOEF", "FP32", 4096, 2), ("ADIAG", "FP32", 4096, 2), ("AID", "BF16", 2048, 3),
            ("AOUT", "BF16", 2048, DT), ("ASCR", "U32", scratch_bytes(geo), 1)]  # fmt: skip


def attention_cb_bytes(geo: AttnGeo) -> dict:
    """bytes per attention CB on one core (the ledger of the mock build is the authority; this is the plan)"""
    b = {name: page * n for name, _, page, n in cb_spec(geo)}
    b["total"] = sum(b.values())
    return b


LAT_STORE_KB = 32  # the chunk latent store's share per core (2 MB over 64 cores)
CTX_STORE_CORES = 40  # rev 10: the context L1 store's owners = the MM-only cores (grid cores 64 .. 103)
CTX_STORE_CORE0 = 64
CTX_L1_MAX_KB = 64  # the largest per-owner share of the context store (2 tiles: chunk_start <= 2,560)


def ctx_store_kb(geo: AttnGeo) -> int:
    """per owner core, the context store's share: ct0 tiles over CTX_STORE_CORES cores, 32 KB each"""
    return -(-geo.ct0 // CTX_STORE_CORES) * LAT_STORE_KB if geo.ct0 else 0


MM_RESIDENT_KB = 160 + 2 * 32 + 3 * 36 + 256  # §4 MM-only core in a weight phase
RING_RESIDENT_KB = 3 * 70 + 3 * 36  # §4 streamer rings (never rewound): bfp8 + bfp4 rings
CB_BUDGET_KB = 1026  # §4: Stage 14's sparse reader CB end minus the base


def l1_plan(geo: AttnGeo) -> dict:
    """the attention core's peak against §4's budget: the attention CBs + the resident rings + the latent store share
    (the MM phase CBs of A5 / A7 are NOT live during A6: IN0 / PART / OUT are rewound at the phase boundary)"""
    att = attention_cb_bytes(geo)["total"]
    ctx_kb = ctx_store_kb(geo) if geo.ctx_l1 else 0  # the MM-only owners carry it (not the chunk store's owners)
    peak = att / 1024 + RING_RESIDENT_KB + max(LAT_STORE_KB if geo.lat_l1 else 0, ctx_kb)
    return dict(attention_kb=round(att / 1024, 1), rings_kb=RING_RESIDENT_KB, store_kb=LAT_STORE_KB if geo.lat_l1 else 0,
                ctx_store_kb=ctx_kb, peak_kb=round(peak, 1), budget_kb=CB_BUDGET_KB, fits=peak <= CB_BUDGET_KB)  # fmt: skip


# ---------------------------------------------------------------- compute + traffic estimate
# Blackhole Tensix at 1.35 GHz: 1024 / 2048 FLOP per cycle at HiFi4 / HiFi2 (architecture.md §10) -> a 32^3 tile matmul
# (65,536 FLOP) = 64 / 32 cycles. SFPU tile ops (fp32, 32 x 32): ~256 cycles each (planning value; the harness piece's
# TRISC1 busy / the PMK_PROBE counters replace it).
# Traffic (rev 3, calibrated on rev 2's silicon unit: 3.15 GB of DRAM reads in 9,699 us = 325 GB/s effective for 104
# cores doing 2-KB page reads with a barrier per tile): DRAM_BW; a core's NoC ingress from other cores' L1 ~32 B/cycle
# (NOC_BW_CORE); a read barrier ~1.2 us from DRAM, ~0.6 us from L1 (latency, hidden only by the next batch's reads).
DRAM_BW_GBS, NOC_BW_CORE_GBS, DRAM_BARRIER_US, L1_BARRIER_US = 325.0, 43.0, 1.2, 0.6
UNION_TILES = 128  # sparse: context K tiles per item = the union of the item's rows' top-512 pools / 8 per tile


def _ctx_tiles(geo: AttnGeo, union_tiles: int) -> int:
    return min(geo.ct0, union_tiles) if geo.sparse else geo.ct0


def math_estimate(geo: AttnGeo, fid_qk: str = "HiFi4", mhz: float = 1350.0, union_tiles: int = UNION_TILES,
                  sim: bool = False, cal: float = COMPUTE_CAL) -> dict:  # fmt: skip
    """compute / DRAM / NoC / barrier terms (rev 3) with the rev 4 calibration of the compute pipeline (cal = the u2
    probe's QK + softmax + PV against the planning cycles); sim = True adds the fan-in model `noc_sim` (rotated order)
    as span_sim_us and takes the span from it"""
    mm = 64 if fid_qk == "HiFi4" else 32  # cycles per tile matmul of the two bf16 products
    nk, qg = geo.nk, geo.qg
    ctx = _ctx_tiles(geo, union_tiles)
    per_pair_fpu = 2 * DT * mm  # S = Q K^T (16 tile matmuls) + O += E V (16) per (q tile, K tile)
    per_upd_fpu = DT * 64  # the fp32 rescale diag(a) O: 16 tile matmuls at the kernel fidelity (HiFi4) per update
    sfpu_tile = 256
    per_pair_sfpu = (1 + 1 + 3) * sfpu_tile  # scale, (mask / triangle add), E = exp(S - m) -> 2 packs
    per_upd_sfpu = (nk + 2 + 2 + 2 + nk + 3) * sfpu_tile  # rowmax chain, reduce, exp(a), diag, rowsum chain, coefs
    pairs = batches = loads_ctx = loads_chunk = 0
    for qt0 in range(0, geo.ct, qg):
        g = item_qg(geo, qt0)
        pairs += g * ctx + sum(qt0 + j + 1 for j in range(g))
        # per q tile j: ceil(ctx / nk) context updates + ceil((qt0 + j + 1) / nk) chunk updates
        batches += sum(-(-ctx // nk) + -(-(qt0 + j + 1) // nk) for j in range(g))
        loads_ctx += ctx
        loads_chunk += qt0 + g
    pairs, batches, loads_ctx, loads_chunk = (v * geo.heads for v in (pairs, batches, loads_ctx, loads_chunk))
    fpu_useful = pairs * per_pair_fpu
    fpu_total = fpu_useful + batches * per_upd_fpu
    busy = fpu_total + pairs * per_pair_sfpu + batches * per_upd_sfpu + geo.items * qg * (DT * 64 + 6 * sfpu_tile)
    compute_us = busy / geo.cores / mhz * cal
    # bytes: K tiles 32 KB each; context from DRAM (sparse: 4-KB row reads, same bytes), chunk from the L1 store or DRAM
    kb = DT * 2048
    q_bytes = geo.items * qg * kb
    dram_bytes = loads_ctx * kb + q_bytes + (0 if geo.lat_l1 else loads_chunk * kb)
    noc_bytes_core = (loads_chunk * kb if geo.lat_l1 else 0) / geo.cores
    dram_us = dram_bytes / DRAM_BW_GBS / 1e3
    noc_us = noc_bytes_core / NOC_BW_CORE_GBS / 1e3
    nbatch_core = (loads_ctx + loads_chunk) / nk / geo.cores
    barrier_us = nbatch_core * (DRAM_BARRIER_US if (loads_ctx or not geo.lat_l1) else L1_BARRIER_US)
    # sparse: the BRISC writes one 32-word mask operand per (K tile, q tile) (MASK_CYC; rev 7: a 2-KB bf16 tile at
    # MASK_CYC_REV7, rev 6: a 4-KB fp32 tile at MASK_CYC_REV6) -- serial on the reader, overlapping the compute only
    # across batches; the item's 32 qg selection rows are fetched + OR-ed on the NCRISC one item ahead (bitmap_us:
    # its own time, hidden unless it exceeds the item's compute)
    mask_us = bitmap_us = 0.0
    if geo.sparse:
        masks = pairs  # one mask per (q tile, K tile) pair
        mask_us = masks * MASK_CYC / geo.cores / mhz
        bitmap_us = geo.items * qg * 32 * (ctx // 32 + 1) * UNION_CYC_PER_WORD / geo.cores / mhz
    span = max(compute_us, dram_us, noc_us, barrier_us, mask_us)
    span_sim = None
    if sim and geo.lat_l1 and geo.ct0 == 0:
        span_sim = noc_sim(geo, rotate=True, cal=cal)["span_us"]  # the sim's compute is the HiFi4 pipeline
        span = max(span, span_sim)
    flop = pairs * 2 * DT * 65536
    return dict(pairs=pairs, batches=batches, gflop=round(flop / 1e9, 2), fpu_useful_pct=round(100 * fpu_useful / busy, 1),
                span_sim_us=span_sim, cal=cal,
                fpu_pct_incl_rescale=round(100 * fpu_total / busy, 1), compute_us=round(compute_us, 1),
                dram_gb=round(dram_bytes / 1e9, 3), dram_us=round(dram_us, 1), noc_mb_core=round(noc_bytes_core / 1e6, 1),
                noc_us=round(noc_us, 1), barrier_us=round(barrier_us, 1), mask_us=round(mask_us, 1),
                bitmap_us=round(bitmap_us, 1), span_us=round(span, 1),
                bound=max((("compute", compute_us), ("dram", dram_us), ("noc", noc_us), ("barrier", barrier_us),
                           ("mask", mask_us), ("noc-sim", span_sim or 0.0)), key=lambda kv: kv[1])[0],
                tflops=round(flop / span / 1e6, 1), fid_qk=fid_qk, nk=nk, qg=qg, lat_l1=geo.lat_l1)  # fmt: skip


# ---------------------------------------------------------------- rev 9: the softmax levers (README §11a)
# The rev 8 sparse unit (12,139 us HiFi4, compute wait 8.6 %): TRISC busy ~11.1 ms for 1,270 (q tile, K tile) pairs
# per core = ~11.8k cycles per pair; the FPU floor is ~2.5k (Q K + P V 32 tile matmuls at 64 cycles + the rescale),
# the rest ~16 SFPU pass-equivalents of ~400 cycles (a tile op on 32 rows: load / op / store on 32-lane vectors).
PASS_CYC = 400  # one SFPU tile pass (rev 8 silicon: softmax 5.8 ms / 1,270 pairs = 6.2k cycles ~ 16 passes)
REV8_SPARSE_US = 12139.0


def softmax_levers(pairs_per_core: int = 1270, updates_per_core: int = 615, mhz: float = 1350.0) -> dict:
    """expected savings (us) of the rev 9 levers on the unit (per-pair / per-update pass counts, README §11a):
    exp_approx: the two exps 25 -> 15 SFP instructions per row (~3.5 -> ~2 passes each; E per pair, a per update);
    max_lazy: per update a = exp(m - m') (~4.5), diag(a) (~3), the 16-tile diag(a) O product (~2.5) and the l
    rescale (~1) vanish on the batches that do not rescale, against the decision's sub + column reduce + pack + one
    mailbox read (~4) -- net ~7 of the update's ~18.5 passes; paired_e: half the E acquires / m loads (~0.5 per
    pair); hifi2: the two bf16 products halve (silicon rev 8: -1,178 us). The first-batch-only max was rejected: a
    row whose first batch is all masked overflows at its first real batch (test_softmax_levers)"""
    cyc = lambda passes, n: passes * PASS_CYC * n / mhz  # noqa: E731
    out = dict(
        exp_approx_us=round(cyc(1.5, pairs_per_core) + cyc(1.5, updates_per_core), 0),
        max_lazy_us=round(cyc(7.0, updates_per_core), 0),
        paired_e_us=round(cyc(0.5, pairs_per_core), 0),
        hifi2_us=1178.0,
        two_batch_update_us=round(cyc(11.5, updates_per_core) / 2, 0),  # not built: the next exact lever (README §11c)
        fused_sub_exp_us=round(cyc(1.0, pairs_per_core), 0),  # not built: the column-broadcast replay in SFPI
        bf16_rowsum_us=round(cyc(0.5, pairs_per_core), 0),  # not built: drop the fp32 E pack
    )
    built = out["hifi2_us"] + out["exp_approx_us"] + out["max_lazy_us"] + out["paired_e_us"]
    out["expected_us"] = dict(
        exact_hifi2=round(REV8_SPARSE_US - out["hifi2_us"]),
        hifi2_exp_approx=round(REV8_SPARSE_US - out["hifi2_us"] - out["exp_approx_us"]),
        rev9_all=round(REV8_SPARSE_US - built),
        rev9_all_two_batch=round(REV8_SPARSE_US - built - out["two_batch_update_us"]),
    )
    # rev 10 (README §12): the rev 9 silicon row (max_lazy + HiFi2, exact) is 10,303 us with the TRISC 71 % busy
    # (~7.3 ms) and the compute waiting 29 % on the reader: the BRISC serialises ~1.5 ms of mask operands, the
    # single-NoC chunk reads (10.6 MB at the dense-measured ~3.5 B/cycle: ~2.2 ms) and the context GATHERS -- every
    # item's union touches all 32 context groups at ctx 1,024, so each core re-reads the same 1 MB per item from
    # DRAM as 1-KB pages (104 x 10 x 1 MB = 1 GB at 325 GB/s: ~3.5 ms) and the TRISC tilizes it (~0.4 ms).
    rev9_us = 10303.0
    out["rev10"] = dict(
        trisc_busy_us=round(rev9_us * 0.71),
        tilize_us=400.0,  # 32 ctx tiles x 16 tilize_block per item, 10 items (TRISC)
        exp_21f_us=round(cyc(1.0, pairs_per_core), 0),  # E only (a keeps the accurate exp)
        reader_rev9_us=dict(masks=1550, chunk_noc=2200, ctx_dram=3500, q=300),
        reader_rev10_us=dict(
            masks=1550, l1_tiles_dual_noc=4800, q=300
        ),  # 21 MB over both NoCs (dense rev 7: 4,773 at 21 MB)
    )
    out["rev10"]["expected_us"] = dict(
        ctx_l1_dual=round(max(out["rev10"]["trisc_busy_us"] - out["rev10"]["tilize_us"], 4800 + 1550 + 300) * 1.03),
        ctx_l1_dual_exp21f=round(max(out["rev10"]["trisc_busy_us"] - 400 - out["rev10"]["exp_21f_us"], 6650) * 1.03),
        ctx_l1_dual_two_batch=round(max(out["rev10"]["trisc_busy_us"] - 400 - out["two_batch_update_us"], 6650) * 1.03),
    )
    return out


# ---------------------------------------------------------------- the §4c projection / indexer phases (specs)
@dataclass
class PfStreamSpec:
    """the §4c PfStreamPhase fields this piece needs (the family owns the class; the integrator instantiates)"""

    name: str
    weight: str  # which DS weight (DSLinear / GroupedWeight role) streams
    k: int  # K (elements)
    n: int  # N per chip (elements)
    m_tiles: int  # rows / 32
    kb: int  # K block (tiles)
    fidelity: str
    wait: str  # the H counter it waits on (per 32-row block)
    signal: str  # the H counter it raises
    in0: str  # activation source
    out: str  # hand-off written
    grouped_heads: int = 0  # GroupedWeight phases: heads per chip (per-head sub-matmuls on the head's rows)
    rs: str = ""  # row-scale tile (the folded RMSNorm's 1/rms)
    out_fp32: bool = False
    note: str = ""

    @property
    def gflop(self) -> float:
        return 2 * self.m_tiles * TILE * self.k * self.n / 1e9

    @property
    def weight_mb(self) -> float:
        tb = 1088 if self.fidelity == "HiFi2" else 2048  # bfp8 DS weights at HiFi2, bf16 at HiFi4
        return self.k // TILE * self.n // TILE * tb / 1e6


def projection_phases(geo: AttnGeo, sp_rows: int | None = None) -> list:
    """the attention half's weight phases on the MM grid (§4c), in program order"""
    h = geo.heads
    sp_rows = sp_rows or geo.sp_rows
    m_all, m_sp = geo.ct, sp_rows // TILE
    return [
        PfStreamSpec("A1 q_a|kv_a", "ds_qa_kva", 4096, 2048, m_sp, 8, "HiFi2", "H1 x̂ (SEM_XH)", "SEM_QK (block)",
                     "x̂ SP rows (member OUT 11)", "qk [256, 2048] bf16 (L1)", rs="invh (deferred 1/rms(h))",
                     note="replicated weight, SP rows only"),
        PfStreamSpec("I1 w_x", "dsa_wx", 4096, 288, m_sp, 8, "HiFi2", "H1 x̂", "SEM_YIX", "x̂ SP rows",
                     "y_ix [256, 288] fp32 (DRAM)", rs="invh", out_fp32=True, note="sparse_dsa indexer projection"),
        PfStreamSpec("A4 q_b (fold)", "mla_qb_fold", 1536, h * 256, m_all, 4, "HiFi2", "SEM_QAB (A3 block)",
                     "SEM_Q (block)", "q_a_all (A3 store)", "q [2048, H_r 256] bf16", rs="qinv rows"),
        PfStreamSpec("I3 wq_b (fold)", "dsa_wqb_fold", 1536, 4096, m_sp, 8, "HiFi2", "SEM_QAB", "SEM_QI",
                     "q_a SP rows", "q_idx [256, 32 x 128] bf16", rs="qinv rows", note="sparse chunks only"),
        PfStreamSpec("A5 W_uk", "mla_w_uk", 256, 512, m_all, 4, "HiFi2", "SEM_Q", "SEM 5 (block, all heads)",
                     "q head columns", "q_lat [1, H_r, C, 512] bf16", grouped_heads=h),
        PfStreamSpec("A7 W_uv", "mla_w_uv", 512, 256, m_all, 8, "HiFi2", "SEM 7 >= items", "SEM_O (block)",
                     "o_lat [1, H_r, C, 512] (whole tiles)", "o [2048, H_r 256] bf16", grouped_heads=h),
        PfStreamSpec("A8 o_proj", "mla_o", h * 256, 4096, m_all, 16, "HiFi2", "SEM_O", "H5 SEM 8 (owner block)",
                     "o", "a [2048, 4096] bf16 partial -> P8 RS"),
    ]  # fmt: skip


def indexer_estimate(geo: AttnGeo, sp_rows: int | None = None, fid: str = "HiFi2") -> dict:
    """I4 scoring FLOPs / cycles of the chip's SP rows against the slot's pools (every chip scores its own rows only:
    the selection is per row, all-gathered in A3); I5 select RISC cycles (3 radix passes of 11 / 11 / 10 bits + the
    selection pass over P keys per row, ~6 cycles per key per pass)"""
    sp_rows = sp_rows or geo.sp_rows
    pools = (geo.chunk_start + geo.rows) // KPOOL
    ptiles = -(-pools // TILE)
    qtiles = sp_rows // TILE
    mm = 32 if fid == "HiFi2" else 64
    cycles = qtiles * ptiles * (32 * 4 + 32) * mm
    sel = sp_rows * pools * 4 * 6
    return dict(pools=pools, score_gflop=round(qtiles * ptiles * 32 * 4 * 65536 * 2 / 1e9, 2),
                score_us_on_grid=round(cycles / geo.cores / 1350, 1), select_us_per_core=round(sel / geo.cores / 1350, 1),
                scores_mb=round(sp_rows * pools * 4 / 1e6, 1))  # fmt: skip


# ---------------------------------------------------------------- host reference (the spec of A6; torch fp32)
def mask_bank_host() -> torch.Tensor:
    """[3 x 32, 32] (bf16 on the device, CB_AID): tile 0 = the causal triangle (0 on / below the diagonal, NEG_MASK
    above -> bf16(-1e9)), tile 1 = identity, tile 2 = E8 (rev 8: row 0 = 2^-(c // 4): the mask operand's expansion)"""
    i = torch.arange(TILE)
    tri = torch.where(i.reshape(-1, 1) >= i.reshape(1, -1), 0.0, NEG_MASK)
    e8 = torch.zeros(TILE, TILE)
    e8[0] = 2.0 ** -(torch.arange(TILE) // KPOOL).float()
    return torch.cat([tri, torch.eye(TILE), e8])


def select_bitmap_host(ids: torch.Tensor, ctx_pools: int) -> torch.Tensor:
    """rev 8: the per-row selection bitmaps the attention phase takes instead of the ids: [C, sel_row_words] int32,
    bit q of row r <=> pool q in ids[r] (context pools 0 .. ctx_pools - 1 then the chunk's 512; NONE_ID ignored)"""
    c = int(ids.shape[0])
    w = sel_row_words(ctx_pools)
    out = torch.zeros(c, w, dtype=torch.int64)
    for r in range(c):
        q = ids[r][ids[r] != NONE_ID].long()
        assert bool((q < ctx_pools + 512).all()), (r, int(q.max()))
        out[r].index_add_(0, q // 32, torch.ones_like(q) << (q % 32))  # distinct pools: a plain sum sets the bits
    return out


def group_tiles(ids: torch.Tensor, ctx_pools: int, qt0: int, qg: int) -> int:
    """rev 8: context K tiles of item (qt0, qg) = the fixed 8-pool groups any of its rows selected into"""
    rows = ids[qt0 * 32 : (qt0 + qg) * 32]
    q = rows[rows != NONE_ID]
    q = q[q < ctx_pools]
    return int(torch.unique(q // POOLS_PER_TILE).numel())


def select_rows_host(scores: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """ids [R, 512] int64 (NONE_ID padded, ascending) of the top-512 closed pools per row: scores [R, P] over pools
    (any order of pools), a pool is a candidate when its last token <= the row's position"""
    r, p = scores.shape
    pool_last = torch.arange(p) * KPOOL + KPOOL - 1
    out = torch.full((r, TOPK), NONE_ID, dtype=torch.int64)
    for i in range(r):
        vis = pool_last <= int(positions[i])
        n = int(vis.sum())
        if n == 0:
            continue
        k = min(TOPK, n)
        s = torch.where(vis, scores[i], torch.full_like(scores[i], -float("inf")))
        top = torch.topk(s, k).indices.sort().values
        out[i, :k] = top
    return out


def attention_host(q_lat: torch.Tensor, latent_all: torch.Tensor, chunk_start: int, ids=None, scale: float = 0.0625):
    """A6's reference. q_lat [H_r, C, 512], latent_all [chunk_start + C, 512] (positions 0 .. end: context ++ chunk),
    ids [C, 512] (sparse regime) -> o_lat [H_r, C, 512] fp32. Row i (position p = chunk_start + i) attends: dense -> every
    position <= p; sparse -> the tokens of its selected pools + the tail [p + 1 - (p + 1) % 4, p] (architecture.md §4).
    """
    hr, c, _ = q_lat.shape
    n = chunk_start + c
    q = q_lat.float()
    k = latent_all.float()[:n]
    s = torch.einsum("hcd,nd->hcn", q, k) * scale
    pos = chunk_start + torch.arange(c)
    allowed = torch.arange(n).reshape(1, -1) <= pos.reshape(-1, 1)  # causal [C, n]
    if ids is not None:
        pool_of_tok = torch.arange(n) // KPOOL
        selm = torch.zeros(c, n // KPOOL + 1, dtype=torch.bool)
        for i in range(c):
            sel = ids[i][ids[i] != NONE_ID]
            selm[i, sel] = True
        tok_sel = selm[:, pool_of_tok]  # [C, n]
        tail0 = pos + 1 - ((pos + 1) % KPOOL)
        tail = torch.arange(n).reshape(1, -1) >= tail0.reshape(-1, 1)
        allowed = allowed & (tok_sel | tail)
    s = torch.where(allowed.reshape(1, c, n), s, torch.full_like(s, NEG_MASK))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hcn,nd->hcd", p, k)


def attention_kernel_model(q_lat: torch.Tensor, latent_all: torch.Tensor, chunk_start: int, ids=None,
                           scale: float = 0.0625, nk: int = 2, qg: int = 1, core: int = 0, rotate: bool = True,
                           items=None, heads=None, mask_neg: float = NEG_MASK, max_lazy: float | None = None):  # fmt: skip
    """the kernel's arithmetic, batch by batch, in fp32 torch (attn_c.cpp update() / finalize(), attn_dm0.cpp's
    batches, masks and rotated order): per item (h, qt0 .. qt0 + qg - 1), per batch of nk key tiles: S = Q K^T scale
    (+ NEG_MASK where a key is masked for the row: the triangle on the diagonal dense tile, the per-row membership
    masks in the sparse regime), m' = max(m, rowmax S), a = exp(m - m'), E = exp(S - m'), l' = a l + rowsum E,
    O' = a O + E V; o = O / l. Context tiles: dense = all positions < chunk_start (tiles of 32); sparse = the union of
    the item's rows' selected pools, 8 pools per K tile, NONE_ID slots zero rows + masked, then the chunk tiles with
    the diagonal's causal bound and the open pool's tail. Returns (o_lat [H, C, 512], stats dict: min l, the number
    of (row, batch) updates whose keys were ALL masked, max |exp argument|). Rows whose early batches are entirely
    masked keep m = NEG_MASK (+ s) and l = the masked count until a real key arrives: a = exp(NEG_MASK - m') = 0
    then rescales them away exactly -- the finiteness argument of README §8. mask_neg: the mask tiles' constant
    (rev 7: NEG_MASK_BF16 on the sparse membership tiles; the dense diagonal triangle keeps NEG_MASK) -- the output
    is the same for any constant <= -2^29 (test_mask_value_independence). max_lazy = T (rev 9 PMK_ATT_MAX_LAZY): a
    batch raises the row's stabiliser (and rescales) only when its max exceeds m by more than T, as the kernel's
    per-tile decision (any row of the tile); otherwise E = exp(S - m) <= e^T accumulates without rescaling (stats
    max_exp_arg_pos: the largest S - m seen; rescales: the tile-level rescale count)."""
    hr, c, d = q_lat.shape
    n = chunk_start + c
    k_all = latent_all.float()[:n]
    q = q_lat.float()
    ct, ct0 = c // TILE, chunk_start // TILE
    chunk_pool0 = chunk_start // KPOOL
    o = torch.zeros(hr, c, d)
    stats = dict(min_l=float("inf"), all_masked_updates=0, max_exp_arg=0.0, batches=0, max_exp_arg_pos=0.0, rescales=0)
    heads = range(hr) if heads is None else heads
    if items is None:
        items = [(h, qt0) for h in heads for qt0 in range(0, ct, qg)]
    for it, (h, qt0) in enumerate(items):
        g = min(qg, ct - qt0)
        rows = torch.arange(qt0 * TILE, (qt0 + g) * TILE)  # the item's g x 32 query rows (chunk-relative)
        pos = chunk_start + rows
        m = torch.full((g * TILE,), NEG_MASK * 2)  # "no stats yet" (run == false): the first batch sets m, l, O
        have = torch.zeros(g * TILE, dtype=torch.bool)
        l = torch.zeros(g * TILE)
        O = torch.zeros(g * TILE, d)
        # key batches: [(keys [nkeys, 512], mask [g * 32, nkeys] (True = masked), q-tile applicability [g * 32])]
        batches = []
        if ids is None:  # dense: context tiles then chunk tiles, all keys allowed except above the diagonal
            keys = [(k_all[j * TILE : (j + 1) * TILE], j, False) for j in range(ct0)]
        else:
            # the union of the item's rows' selected context pools, ascending, 8 per tile
            sel = [set(int(x) for x in ids[r][ids[r] != NONE_ID].tolist()) for r in rows.tolist()]
            uni = sorted(pl for pl in set().union(*sel) if pl < chunk_pool0)
            keys = []
            for t0 in range(0, len(uni), POOLS_PER_TILE):
                pools = uni[t0 : t0 + POOLS_PER_TILE] + [NONE_ID] * (
                    POOLS_PER_TILE - len(uni[t0 : t0 + POOLS_PER_TILE])
                )
                ktile = torch.zeros(TILE, d)
                mask = torch.ones(g * TILE, TILE, dtype=torch.bool)
                for sidx, pl in enumerate(pools):
                    if pl == NONE_ID:
                        continue
                    ktile[sidx * KPOOL : (sidx + 1) * KPOOL] = k_all[pl * KPOOL : (pl + 1) * KPOOL]
                    for ri in range(g * TILE):
                        if pl in sel[ri]:
                            mask[ri, sidx * KPOOL : (sidx + 1) * KPOOL] = False
                keys.append((ktile, mask, None))
        ctx_batches = []
        for b0 in range(0, len(keys), nk):
            grp = keys[b0 : b0 + nk]
            kt = torch.cat([kk[0] for kk in grp])
            if ids is None:
                mk = torch.zeros(g * TILE, kt.shape[0], dtype=torch.bool)  # dense context: all allowed
            else:
                mk = torch.cat([kk[1] for kk in grp], dim=1)
            ctx_batches.append((kt, mk, torch.ones(g * TILE, dtype=torch.bool)))
        nchunk = qt0 + g
        nb = -(-nchunk // nk)
        order = batch_order(core, it, nb) if rotate else list(range(nb))
        chunk_batches = []
        for b in order:
            k0 = b * nk
            bnk = min(nk, nchunk - k0)
            kt = k_all[chunk_start + k0 * TILE : chunk_start + (k0 + bnk) * TILE]
            kpos = chunk_start + torch.arange(k0 * TILE, (k0 + bnk) * TILE)
            # applicability per row: its q tile qt = qt0 + j applies tile k iff k <= qt (prefix); the row's masked keys
            applies = torch.zeros(g * TILE, dtype=torch.bool)
            mk = torch.ones(g * TILE, kt.shape[0], dtype=torch.bool)
            for j in range(g):
                qt = qt0 + j
                rr = slice(j * TILE, (j + 1) * TILE)
                nk_j = sum(1 for i in range(bnk) if k0 + i <= qt)
                if nk_j == 0:
                    continue
                applies[rr] = True
                prow = pos[rr].reshape(-1, 1)
                allowed = kpos.reshape(1, -1) <= prow  # causal (the triangle on the diagonal tile; k < qt: all)
                allowed &= (torch.arange(kt.shape[0]) < nk_j * TILE).reshape(1, -1)  # tiles past the prefix: absent
                if ids is not None:
                    tail0 = prow + 1 - ((prow + 1) % KPOOL)
                    pool_of = (kpos // KPOOL).reshape(1, -1)
                    member = torch.zeros(TILE, kt.shape[0], dtype=torch.bool)
                    for ri in range(TILE):
                        member[ri] = torch.tensor([int(pp) in sel[j * TILE + ri] for pp in pool_of[0].tolist()])
                    allowed &= member | (kpos.reshape(1, -1) >= tail0)
                mk[rr] = ~allowed
            chunk_batches.append((kt, mk, applies))
        for kt, mk, applies in ctx_batches + chunk_batches:
            stats["batches"] += 1
            s_ = (q[h, rows] @ kt.T) * scale
            s_ = torch.where(mk, s_ + (mask_neg if ids is not None else NEG_MASK), s_)  # the mask tile ADDS its
            # constant (fp32), as the kernel: the sparse membership tiles mask_neg, the dense diagonal the triangle
            # tiles past a row's prefix are absent for that row: the kernel never multiplies them. Model: exclude
            absent = (~applies).reshape(-1, 1).expand_as(s_)
            rowmax = torch.where(absent, torch.full_like(s_, -float("inf")), s_).max(dim=1).values
            m_new = torch.where(have, torch.maximum(m, rowmax), rowmax)
            if max_lazy is not None:  # the kernel's decision is per q TILE (32 rows): rescale when any row grows > T
                for j in range(g):
                    rr = slice(j * TILE, (j + 1) * TILE)
                    if not bool(applies[rr].any()):
                        continue
                    grow = (m_new[rr] - m[rr])[have[rr]]
                    if grow.numel() and float(grow.max()) > max_lazy:
                        stats["rescales"] += 1
                    else:
                        m_new[rr] = torch.where(have[rr], m[rr], m_new[rr])
            a = torch.where(have, torch.exp(m - m_new), torch.zeros_like(m))
            stats["max_exp_arg"] = max(
                stats["max_exp_arg"], float((m - m_new)[have].abs().max()) if have.any() else 0.0
            )
            e = torch.exp(s_ - m_new.reshape(-1, 1))
            stats["max_exp_arg"] = max(stats["max_exp_arg"], float((s_ - m_new.reshape(-1, 1))[~absent].abs().max()))
            stats["max_exp_arg_pos"] = max(stats["max_exp_arg_pos"], float((s_ - m_new.reshape(-1, 1))[~absent].max()))
            e = torch.where(absent, torch.zeros_like(e), e)
            upd = applies
            stats["all_masked_updates"] += int((upd & mk.all(dim=1)).sum())
            l_new = a * l + e.sum(dim=1)
            O_new = a.reshape(-1, 1) * O + e.to(torch.bfloat16).float() @ kt  # E packed to bf16 for the PV matmul
            m = torch.where(upd, m_new, m)
            l = torch.where(upd, l_new, l)
            O = torch.where(upd.reshape(-1, 1), O_new, O)
            have |= upd
        assert bool(have.all()), "a q tile with no key"
        stats["min_l"] = min(stats["min_l"], float(l.min()))
        o[h, rows] = O / l.reshape(-1, 1)
    return o, stats


# ---------------------------------------------------------------- the device descriptors of A6
def _rt_args(rtd: dict):
    from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import _rt_args as rt

    return rt(rtd)


def rm_rows_per_page(lat_rm) -> int:
    """rows of the row-major latent store per page: [.., n, 512] -> 1 (the op path's lat_rm: 4 page reads per pool),
    [.., n / 4, 2048] -> 4 (one read per pool)"""
    w = int(lat_rm.shape[-1])
    assert w in (512, 2048), lat_rm.shape
    return w // 512


def latent_store(device, rows: int, cores, lat_host=None):
    """the chunk latent store [1, 1, rows, 512] bf16 TILE, HEIGHT-sharded over the first rows / 32 cores of `cores`
    (32 rows = 16 tiles = 32 KB per core): the AG's landing store in the layer program; here written from the host
    (the mock / the bench unit). The kernels read it through its TensorAccessor (sharded pages resolve to the owning
    core's L1)."""
    n = rows // TILE
    cl = list(ttnn.corerange_to_cores(cores, row_wise=True))[:n]
    assert len(cl) == n, (len(cl), n)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cl])
    spec = ttnn.ShardSpec(grid, (TILE, 512), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    t = lat_host if lat_host is not None else torch.zeros(1, 1, rows, 512)
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=mc)  # fmt: skip


def context_store(device, geo: AttnGeo, cores, ctx_host=None):
    """rev 10: the context rows [0, chunk_start) bf16 TILE, HEIGHT-sharded (32 x ceil(ct0 / 40) rows per core) over the
    MM-only cores 64 .. of `cores` (distinct owners from the chunk store's 0 .. 63): the sparse context batches read
    whole 32-KB tiles from it over both NoCs (attn_dm0.cpp read_ctx_l1_tile). In the layer program the cache piece
    copies it from the paged cache before the phase (no counter: landed before A6 starts)"""
    n = geo.ct0
    per = -(-n // CTX_STORE_CORES)
    ncores = -(-n // per)
    cl = list(ttnn.corerange_to_cores(cores, row_wise=True))[CTX_STORE_CORE0 : CTX_STORE_CORE0 + ncores]
    assert len(cl) == ncores, (len(cl), ncores)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cl])
    spec = ttnn.ShardSpec(grid, (TILE * per, 512), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    rows = ncores * per * TILE
    t = torch.zeros(1, 1, rows, 512)
    if ctx_host is not None:
        t[0, 0, : geo.chunk_start] = ctx_host.reshape(-1, 512)[: geo.chunk_start]
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=mc)  # fmt: skip


class PmkAttention:
    """A6 on a core set: `parts(...)` -> (kernels, cbs, semaphores) to splice into the layer program (the family's
    pmk_layer.py), or `standalone(...)` -> a generic_op of the phase alone (the mock job / the device bench unit)."""

    def __init__(self, device, geo: AttnGeo, cores, *, scale: float, fid_qk: str | None = None, probe=None,
                 sem_latb: int = SEM_LATB_DEFAULT, sem_ids: int = SEM_IDS_DEFAULT):  # fmt: skip
        self.device, self.geo, self.scale, self.probe = device, geo, scale, probe
        self.sem_latb, self.sem_ids = sem_latb, sem_ids
        self.cores = cores  # CoreRangeSet of the attention cores (the MM grid)
        self.core_list = [(int(c.x), int(c.y)) for c in ttnn.corerange_to_cores(cores, row_wise=True)]
        assert len(self.core_list) == geo.cores, (len(self.core_list), geo.cores)
        self.lists = attn_items(geo)
        self.fid_qk = fid_qk or os.environ.get("GLM53_PMK_ATT_FID_QK", "")
        mask = mask_bank_host()
        self.mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

    def parts(self, q_lat, cache, page_table, latent_all, sel, lat_rm, o_lat, consumers: list, ctx_store=None):
        """sel: the selection bitmaps [1, 1, C, sel_row_words] uint32 ROW_MAJOR (rev 8; select_bitmap_host; any
        uint32 row-major tensor in the dense regime). consumers: NoC (x, y) of the W_uv readers (SEM_OL +1 per item on
        each). probe (a [2 x cores, 16] uint32 DRAM tensor): dm0 marks [start, end, slot-wait, barrier, mask, Q,
        ncrisc-wait cycles] in row k, dm1 marks + the compute's cycle counters (QK, softmax, PV + rescale, waits) +
        its own (bitmap fetches, dual-NoC reads) in row cores + k."""
        geo = self.geo
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())  # noqa: E731
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        scr = scratch_layout(geo)
        pt_bytes = _a64(int(page_table.shape[-1]) * 4)
        assert pt_bytes == scr["pt_bytes"], (pt_bytes, scr, geo.max_seq)
        dm0_ct = [ACB["AQ"], ACB["AK"], ACB["AKRM"], ACB["AMASK"], ACB["AINFO"], ACB["ASCR"], ACB["ABM"], ACB["AID"],
                  geo.ct, pt_bytes, BLOCK, SEM_QLAT, self.sem_latb, rm_rows_per_page(lat_rm), ACB["AGO"],
                  scr["flag_off"]]  # fmt: skip
        if ctx_store is None:
            assert not geo.ctx_l1, "ctx_l1 needs the context L1 store (context_store)"
            ctx_store = latent_all  # a stand-in accessor (never read)
        dm0_ct += (acc(q_lat) + acc(cache) + acc(page_table) + acc(self.mask) + acc(latent_all) + acc(lat_rm)
                   + acc(ctx_store))  # fmt: skip
        dm1_ct = [ACB["AOUT"], geo.ct, SEM_OL, len(consumers), self.sem_latb, ACB["AK"], ACB["AGO"], ACB["ABM"],
                  ACB["ASCR"], scr["flag_off"]] + acc(o_lat) + acc(latent_all) + acc(sel) + acc(ctx_store)  # fmt: skip
        if geo.sparse:
            assert int(sel.shape[-1]) == sel_row_words(geo.ct0 * POOLS_PER_TILE), (sel.shape, geo.ct0)
        c_ct = [ACB[k] for k in ("AQ", "AK", "AKRM", "AMASK", "AINFO", "AS", "AM", "AP", "AE", "APA", "ABM", "APAO",
                                 "AGO", "ACOEF", "ADIAG", "AID", "AOUT")] + [f32_bits(self.scale), ACB["AKT"]]  # fmt: skip
        dm0_rt, dm1_rt, c_rt = {}, {}, {}
        flat = [v for xy in consumers for v in xy]
        # one runtime-arg width per kernel over the cores (the item lists differ by one item)
        width = 2 * max(len(l) for l in self.lists)
        pa = self.probe.buffer_address() if self.probe is not None else None
        for k, xy in enumerate(self.core_list):
            lst = self.lists[k]
            pairs = [v for hq in lst for v in hq]
            pairs += [0] * (width - len(pairs))
            dm0_rt[xy] = [q_lat.buffer_address(), cache.buffer_address(), page_table.buffer_address(),
                          self.mask.buffer_address(), latent_all.buffer_address(), lat_rm.buffer_address(),
                          ctx_store.buffer_address(), geo.chunk_start, len(lst), k] + pairs  # fmt: skip
            dm1_rt[xy] = [o_lat.buffer_address(), latent_all.buffer_address(), sel.buffer_address(),
                          ctx_store.buffer_address(), geo.chunk_start, len(lst), width // 2] + pairs + flat  # fmt: skip
            if pa is not None:
                dm0_rt[xy] += [pa, k]
                dm1_rt[xy] += [pa, geo.cores + k]
            c_rt[xy] = []
        # the compile signature carries the geometry: two geometries with the same kernels / CBs / cores would hash to
        # ONE cached program whose per-core runtime args have the first geometry's width (the mock's out-of-bounds
        # override, 2026-10-22); NK / QG always explicit
        defs = [("PMK_ATT_REV", str(ATT_REV)), ("PMK_ATT_NK", str(geo.nk)), ("PMK_QG", str(geo.qg)),
                ("PMK_HEADS", str(geo.heads)), ("PMK_MAX_ITEMS", str(width // 2)),
                ("PMK_CHUNK_START", str(geo.chunk_start)), ("PMK_ATT_KSLOTS", str(geo.kslots)),
                ("PMK_BM_PAGES", str(geo.bm_pages))]  # fmt: skip
        if geo.sparse:
            defs.append(("PMK_SPARSE", "1"))
            if geo.ctx_l1:
                defs.append(("PMK_CTX_L1", "1"))
        if geo.dual_noc:
            defs.append(("PMK_ATT_DUAL_NOC", "1"))
        if geo.lat_l1:
            defs.append(("PMK_LAT_L1", "1"))
        cdefs = list(defs) + [("PMK_ATT_OB", os.environ.get("GLM53_PMK_ATT_OB", "4"))]
        if geo.exp_approx:
            cdefs.append(("PMK_ATT_EXP_APPROX", "1"))
        if geo.exp_21f:
            cdefs.append(("PMK_ATT_EXP_21F", "1"))
        if geo.max_lazy is not None:
            cdefs.append(("PMK_ATT_MAX_LAZY", str(f32_bits(geo.max_lazy))))
        if self.fid_qk:
            cdefs.append(("PMK_ATT_FID_QK", str(int(getattr(ttnn.MathFidelity, self.fid_qk).value))))
        d0, d1 = list(defs), list(defs)
        if pa is not None:
            d0 += [("MK_PROBE_ARG", str(10 + width)), ("MK_PROBE_CB", str(PROBE_CB))]
            d1 += [("MK_PROBE_ARG", str(7 + width + len(flat))), ("MK_PROBE_CB", str(PROBE_CB)),
                   ("PMK_PROBE", str(PROBE_CB))]  # fmt: skip
            cdefs.append(("PMK_PROBE", str(PROBE_CB)))
        fid = getattr(ttnn.MathFidelity, os.environ.get("GLM53_PMK_ATT_FIDELITY", "HiFi4"))
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        # fp32 SFPU operands: exact unpack (rev 7 / 8: AMASK and AID are bf16 -- copy_tile expands them exactly)
        for key in ("AS", "AM", "AE", "APA", "ACOEF"):
            modes[ACB[key]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kd = lambda src, ct, rt, d, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=os.path.join(KDIR, "pmk", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores, compile_time_args=ct, runtime_args=_rt_args(rt), defines=d, config=cfg)  # fmt: skip
        kernels = [
            kd("attn_dm0.cpp", dm0_ct, dm0_rt, d0, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            kd("attn_c.cpp", c_ct, c_rt, cdefs, cc),
            kd("attn_dm1.cpp", dm1_ct, dm1_rt, d1, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
        ]
        cbs = self.cbs()
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.cores, initial_value=0)
                for i in (SEM_QLAT, self.sem_latb, SEM_OL, self.sem_ids)]  # fmt: skip
        return kernels, cbs, sems

    def cbs(self) -> list:
        dts = dict(BF16=ttnn.bfloat16, FP32=ttnn.float32, U32=ttnn.uint32)
        out = [ttnn.CBDescriptor(total_size=page * n, core_ranges=self.cores,
                                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=ACB[k], data_format=dts[dt],
                                                                             page_size=page)])
               for k, dt, page, n in cb_spec(self.geo)]  # fmt: skip
        if self.probe is not None:
            out.append(ttnn.CBDescriptor(total_size=128, core_ranges=self.cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=PROBE_CB, data_format=ttnn.uint32, page_size=128)]))  # fmt: skip
        return out

    def standalone(self, q_lat, cache, page_table, latent_all, sel, lat_rm, *, preset_counters: bool = True,
                   ctx_store=None):  # fmt: skip
        """the phase alone as one generic_op: every counter pre-raised (the producers are the layer's other phases;
        here the inputs are in place, so the kernels' wait_min succeed at once). The mock job / the bench unit."""
        geo = self.geo
        o_lat = ttnn.empty((1, geo.heads, geo.rows, 512), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device,
                           ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        kernels, cbs, sems = self.parts(
            q_lat, cache, page_table, latent_all, sel, lat_rm, o_lat, [], ctx_store=ctx_store
        )
        if preset_counters:  # the counters start at their final values: the inputs are in place
            sems = [ttnn.SemaphoreDescriptor(id=s.id, core_ranges=self.cores, initial_value=geo.ct) for s in sems]
        io = [q_lat, cache, page_table, self.mask, latent_all, sel, lat_rm, o_lat] + (
            [ctx_store] if ctx_store is not None else []
        )
        if self.probe is not None:
            io.append(self.probe)
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return o_lat


def probe_tensor(device, geo: AttnGeo):
    return ttnn.from_torch(torch.zeros(2 * geo.cores, 16, dtype=torch.int32), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


def decode_probe(p: torch.Tensor, geo: AttnGeo, mhz: float = 1350.0) -> dict:
    """the probe tensor [2 cores, 16] (host): dm0 rows k = [start, end, cyc_slot_wait, cyc_barrier, cyc_mask, cyc_q,
    cyc_ncrisc_wait]; dm1 rows cores + k = [start, end, cyc_qk, cyc_softmax, cyc_pv, cyc_wait, cyc_bitmaps,
    cyc_dual_noc] -> per-role spans (us, median / max over cores), the compute's cycle shares, the reader's (slot wait
    = compute-bound, barrier = NoC / DRAM-bound, ncrisc = the pool words / the dual-NoC flag) and the NCRISC's shares
    (bitmaps: the pool-word builds; dual: its half of the dense reads)"""
    import statistics

    p = p.reshape(-1, 16).long() & 0xFFFFFFFF
    n = geo.cores
    d0 = [((int(p[k, 1]) - int(p[k, 0])) & 0xFFFFFFFF) / mhz for k in range(n) if int(p[k, 1])]
    rc = [[int(p[k, 2 + i]) for i in range(5)] for k in range(n) if int(p[k, 1])]
    med = lambda v: round(statistics.median(v), 1) if v else None  # noqa: E731
    rcs = {name: med([100 * c[i] / max(((int(p[k, 1]) - int(p[k, 0])) & 0xFFFFFFFF), 1) for c, k in
                      zip(rc, [k for k in range(n) if int(p[k, 1])])])
           for i, name in enumerate(("dm0_slot_pct", "dm0_barrier_pct", "dm0_mask_pct", "dm0_q_pct",
                                     "dm0_ncrisc_pct"))}  # fmt: skip
    d1 = [((int(p[n + k, 1]) - int(p[n + k, 0])) & 0xFFFFFFFF) / mhz for k in range(n) if int(p[n + k, 1])]
    cyc = [[int(p[n + k, 2 + i]) for i in range(4)] for k in range(n) if int(p[n + k, 1])]
    tot = [sum(c) for c in cyc] or [1]
    shares = {name: med([100 * c[i] / max(t, 1) for c, t in zip(cyc, tot)])
              for i, name in enumerate(("qk_pct", "softmax_pct", "pv_pct", "wait_pct"))}  # fmt: skip
    nc = [[int(p[n + k, 6 + i]) for i in range(2)] for k in range(n) if int(p[n + k, 1])]
    ncs = {
        name: med(
            [
                100 * c[i] / max(((int(p[n + k, 1]) - int(p[n + k, 0])) & 0xFFFFFFFF), 1)
                for c, k in zip(nc, [k for k in range(n) if int(p[n + k, 1])])
            ]
        )
        for i, name in enumerate(("dm1_bitmaps_pct", "dm1_dual_pct"))
    }  # fmt: skip  (rev 8: bitmap FETCHES + unions)
    return dict(dm0_us=med(d0), dm0_max_us=max(d0) if d0 else None, dm1_us=med(d1), dm1_max_us=max(d1) if d1 else None,
                compute_cyc_us=med([t / mhz for t in tot]), **shares, **rcs, **ncs)  # fmt: skip


def attention_grid(device, n: int = 104):
    """the first n cores of the compute grid in row-major order (the mock / bench: the family's MM grid placement
    replaces this in the layer program)"""
    grid = device.compute_with_storage_grid_size()
    cores = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x))][:n]
    assert len(cores) == n, (len(cores), n)
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
