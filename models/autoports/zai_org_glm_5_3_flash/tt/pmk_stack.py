# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PMK STACK: the 45-layer prefill-megakernel driver (doc/ttnn/16-prefill-production/pmk/stack/README.md; contract
prefill_megakernel.md §0 / §1 / §1b / §3 / §4b / §7 B5).

What it is (host-side Python + program sequencing; no new kernels):
  * the LAYER SCHEDULE: layer L -> kind (31 sparse_kda, 11 sparse_dsa, 3 dense_kda from the HF config), the builder of
    its persistent program (the family engine `tt/pmk_layer.py` for the KDA kinds; the sparse_dsa plan of Stage 14's
    pmk-dsa branch through `KIND_BUILDERS` when it is on the tree, the family skeleton with the attention half
    PENDING otherwise) and the geometry (`PmkGeom.from_config(cfg, tp, c)`: TP8 = 256 SP rows / 8 heads / XSL 256,
    TP2 = 1,024 / 32 / 1,024);
  * the CHAIN: layers [0, n) run as ONE persistent program each on a PERSISTENT per-bucket residual `res[c]` (fp32
    [1, 1, C_local, 16384] DRAM, the layout of TPPrefill.streams_to_device); the program is in place (P1 reads the
    residual block into the members, P16 writes it back), so the only DRAM stream between two programs is `res` plus
    the KDA scratch (S / C) the next layer never reads. Layers [n, 45) keep the op graph (TPPrefill.layer), reading
    `res` as their input: with n = 0 nothing is installed and the op-graph path is bitwise the one without this module;
  * ONE CAPTURE PER BUCKET: `trace_capture(c)` = TPPrefill.capture((c,)) -> the key (c, 0) -> `replay(c)` =
    execute_trace. Every PMK program's first launch is the compile launch inside TPPrefill.compile() (every program
    compiled before the FIRST capture, contract §10.5); nothing runs eagerly afterwards;
  * the PREFILL -> DECODE HAND-OFF: the KDA state pages. P9 (cache piece, cache_state.cpp) writes every segment's final
    S + conv carry straight into the decode stack's state tensors kv.states[cidx][li] = (S [chunk, H_r, 128, 128]
    fp32, C [1, 3, chunk, W_r] bf16) at row rd = slot % chunk -- the SAME tensors and bytes StateIO.state_out
    (kernels/prefill/state_io.cpp) writes and the served StackBackend reads (_state_ops / _new_state);
    `state_handoff()` is the per-layer map and `state_layout()` the pure-python twin of the three layouts
    (tests/s16/test_pmk_stack.py proves them equal);
  * weights: the programs read the resident weights of the StackBackend's layers (pf.layers[li].kda / moe / xsl /
    shared / hc / norm and pl.mpf's fn / sb) by address -- no second upload; the only tensors this module creates are
    the per-bucket residual, the mhc piece's bf16 norm-weight tile per sublayer (SubWeights.w, 256 KB) and the mhc
    intermediates. All are plain from_torch / empty uploads at build time, so Stage 12's resident image (tp_image's
    TPWeightIO intercepts) replays them when PmkStack is built inside the backend build window
    (StackBackend.prepare_prefill, GLM53_PMK_LAYERS = n).

The piece bodies registered per program are what the integration branch carries: mhc (P1 / P10 / P16) through
tt/pmk_sparse_kda.py; the other pieces (kda P4..P6 / P9, moe P12..P14, coll P2 / P8 / P11 / P15, cache, attn) land
through `PIECE_WIRERS` as their parts() are adopted by Stage 16. `wired()` reports, per layer, which phases carry a
body and which are stubs, so a row's numerics are never mistaken for a full layer."""

from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace

import torch

from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_DSA, SPARSE_KDA

TILE = 32
HIDDEN, NSTREAM = 4096, 4
RES_COLS = NSTREAM * HIDDEN  # the flat residual [rows, 16384] fp32
HEADS, HEAD_DIM, CONV_K = 64, 128, 4
TILE_FP32, TILE_BF16 = 4096, 2048
KINDS = (SPARSE_KDA, SPARSE_DSA, DENSE_KDA)
# the served construction's layer-kind map (tt/stack_model.layer_kind): tt/config.py's names -> the evidence names
_KIND_OF = {"kda_dense": DENSE_KDA, "kda_moe": SPARSE_KDA, "mla_moe": SPARSE_DSA}
V1_BUCKET = 2048  # the only bucket the family engine plans today (64 blocks = 64 members; 512 -> 16 blocks: assert)


# ------------------------------------------------------------------------------------------------ layer schedule
def layer_schedule(cfg) -> list:
    """[(L, kind)] for every decoder layer of the config (45: 31 sparse_kda, 11 sparse_dsa, 3 dense_kda)"""
    n = int(getattr(cfg, "num_hidden_layers", 45))
    return [(L, _KIND_OF[cfg.layer_kind(L)]) for L in range(n)]


def kind_counts(cfg) -> dict:
    out = {k: 0 for k in KINDS}
    for _, k in layer_schedule(cfg):
        out[k] += 1
    return out


# ------------------------------------------------------------------------------------------------ kind builders
def _build_family(pf, li: int, c: int, sp: bool, probe: bool, kind: str):
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK

    geom = PMK.PmkGeom.from_config(pf.cfg, int(pf.ndev), c, kind=kind)
    return PMK.PmkSparseKdaLayer(pf, li, c, sp, probe=probe, geom=geom)


def build_sparse_kda(pf, li, c, sp, probe=False):
    return _build_family(pf, li, c, sp, probe, "sparse_kda")


def build_dense_kda(pf, li, c, sp, probe=False):
    return _build_family(pf, li, c, sp, probe, "dense_kda")


def build_sparse_dsa(pf, li, c, sp, probe=False):
    """Stage 14's sparse_dsa plan (branch aperezvicente/glm53-pmk-dsa: tt/pmk_dsa.py PmkSparseDsaPlan on the family
    engine) when it is on the tree; otherwise the family skeleton at the sparse_kda geometry with the attention half
    PENDING (its mHC / collective / FFN phases are the same set; the KDA phases P3..P9 stay stubs and the program
    says so in `pending`), so the 45-layer chain builds, captures and censuses today and 14's plan drops in."""
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_dsa as DSA  # noqa: PLC0415

        cls = getattr(DSA, "PmkSparseDsaLayer", None)
        if cls is not None:
            return cls(pf, li, c, sp, probe=probe)
        fn = getattr(DSA, "build_layer", None)
        if fn is not None:
            return fn(pf, li, c, sp, probe=probe)
    except ImportError:
        pass
    prog = _build_family(pf, li, c, sp, probe, "sparse_kda")
    prog.pending = (
        "sparse_dsa attention half (A1..A8 / I1..I5, D9): 14's PmkSparseDsaPlan, branch pmk-dsa, not on this tree"
    )
    prog.impl = "pmk:family-skeleton(sparse_dsa geometry pending)"
    return prog


KIND_BUILDERS = {SPARSE_KDA: build_sparse_kda, DENSE_KDA: build_dense_kda, SPARSE_DSA: build_sparse_dsa}


# ------------------------------------------------------------------------------------------------ piece wirers
def wire_mhc(stack: "PmkStack", prog, pl, res, *, weights: str = "resident") -> dict:
    """the mhc piece's P1 / P10 / P16 bodies on this layer's program (tt/pmk_sparse_kda.py's adapter): per sublayer
    the weights fn / sb / w and the intermediates hn, y, post, comb. weights = "resident": fn / sb are the layer's
    MhcPrePf tensors (pl.mpf, the same layout; one upload), w = the bf16 norm-weight tile from the host weight;
    "source": SubWeights.from_source (a second fn upload); "random": synthetic (host mocks without a weight cache).
    The sublayer outputs y (H6 / H10) are the RS lanes' landing; until coll's FabricStream is registered they are
    the DRAM tensors `ys` (zero) -- the program's residual is then mHC(x, y = 0), reported by wired()."""
    import ttnn  # noqa: PLC0415
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as MHC  # noqa: PLC0415
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_sparse_kda as INT  # noqa: PLC0415

    mesh, cfg, L = stack.pf.mesh, stack.pf.cfg, pl.L
    SP = prog.geom.sp_rows
    mp = MHC.Plan(sp_rows=SP, y_lane=False)
    norm = {"attn": "input_layernorm.weight", "ffn": "post_attention_layernorm.weight"}
    subs = []
    for w in ("attn", "ffn"):
        mpf = (getattr(pl, "mpf", None) or {}).get(w) if weights == "resident" else None
        if weights == "random":
            subs.append(MHC.SubWeights.random(mesh, seed=L * 2 + (w == "ffn")))
        elif mpf is not None and hasattr(mpf, "fn") and hasattr(mpf, "sb"):
            wt = torch.zeros(TILE, HIDDEN)
            wt[0] = stack.source.host(L, norm[w]).float().reshape(-1)
            w_t = ttnn.from_torch(wt.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            subs.append(MHC.SubWeights(fn=mpf.fn, sb=mpf.sb, w=w_t, host={}))
        else:
            subs.append(MHC.SubWeights.from_source(mesh, stack.source, cfg, L, w, norm[w]))
    mhc = MHC.PmkMhc(mesh, subs, mp, hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps,
                     iters=cfg.hc_sinkhorn_iters, placement=INT.family_mhc_placement(prog))  # fmt: skip
    empty = lambda shape, dt: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
    ydt = ttnn.float32 if mp.y_fp32 else ttnn.bfloat16
    inter = dict(hns=[empty((1, 1, SP, HIDDEN), ttnn.bfloat16) for _ in range(2)],
                 ys=[stack._zeros((1, 1, SP, HIDDEN), ydt) for _ in range(2)],
                 posts=[empty((1, 1, SP, NSTREAM), ttnn.float32) for _ in range(2)],
                 combs=[empty((1, 1, SP, NSTREAM * NSTREAM), ttnn.float32) for _ in range(2)])  # fmt: skip
    flags = MHC.F_XH_DRAM | MHC.F_Y_DRAM | MHC.F_PC_DRAM | MHC.F_INIT
    flags |= (MHC.F_X_LOAD | MHC.F_X_STORE) if mp.resident else 0
    bodies = INT.mhc_bodies(mhc, res, inter["hns"], inter["ys"], inter["posts"], inter["combs"], flags=flags, layer=0,
                            fn_gate=None)  # fmt: skip
    INT.register_mhc(prog, bodies)
    return dict(piece="mhc", phases=sorted(bodies), weights=weights, y="dram-zero (RS lanes not registered)",
                keep=(mhc, subs, inter))  # fmt: skip


# the pieces the stack wires onto every program, in registration order; a wirer returns its report dict. Pieces
# whose parts() are not adopted by the integration branch yet are absent here on purpose (their phases stay stubs and
# wired() lists them as pending) -- never a silent op-path fallback inside a PMK layer
PIECE_WIRERS = [wire_mhc]


# ------------------------------------------------------------------------------------------------ state hand-off
def state_layout(tp: int, chunk: int, slot: int) -> dict:
    """the slot's bytes in the decode stack's state tensors of ONE KDA layer at TP `tp`: the pure-python twin of the
    three writers / readers that must agree bitwise -- kernels/prefill/state_io.cpp STATE_OUT (StateIO, the op path),
    kernels/pmk/cache_state.cpp (P9, the PMK), tt/stack_model.StackBackend._state_ops (the decode side).
    S [chunk, H_r, 128, 128] fp32 TILE: user rd's tiles are contiguous pages [rd n_s, (rd + 1) n_s), n_s = 16 H_r;
    C [1, 3, chunk, W_r] bf16 TILE (chunk <= 32 rows = one tile row): user rd = row rd of every tile, 2 x 32-B face
    rows per tile at row_off_bf16(rd) and + 512."""
    h_r = HEADS // tp
    w_r = h_r * HEAD_DIM
    n_s = (HEAD_DIM // TILE) * (HEAD_DIM // TILE) * h_r  # 16 H_r fp32 pages
    n_c = (CONV_K - 1) * (3 * w_r // TILE)  # conv tiles: 3 history rows x 3 W_r / 32 column tiles
    cidx, rd = divmod(int(slot), int(chunk))
    s_pages = list(range(rd * n_s, (rd + 1) * n_s))
    row_off = (rd >> 4) * 1024 + (rd & 15) * 32  # cache_common.hpp row_off_bf16 == state_io.cpp tile_off(rd, 0) * 2
    c_rows = [(p, p * TILE_BF16 + row_off, p * TILE_BF16 + row_off + 512) for p in range(n_c)]
    return dict(tp=tp, h_r=h_r, w_r=w_r, n_s=n_s, n_c=n_c, chunk=chunk, cidx=cidx, rd=rd, s_pages=s_pages,
                s_bytes=[(p * TILE_FP32, TILE_FP32) for p in s_pages], c_rows=c_rows, row_bytes=32,
                s_total=n_s * TILE_FP32, c_total=n_c * 2 * 32)  # fmt: skip


def state_io_layout(n_s: int, n_c: int, chunk: int, slot: int) -> dict:
    """state_io.cpp STATE_OUT as written: S pages rd * n_s + p; conv tile p, dst row rd via tile_off(rd, c) for
    c in 0..31 (uint16 elements)"""
    cidx, rd = divmod(slot, chunk)
    tile_off = lambda r, c: (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)  # noqa: E731
    s = [(rd * n_s + p) * TILE_FP32 for p in range(n_s)]
    c = [(p, sorted({p * TILE_BF16 + 2 * tile_off(rd, col) for col in range(32)})) for p in range(n_c)]
    return dict(cidx=cidx, rd=rd, s=s, c=c)


def cache_state_layout(n_s: int, n_c: int, chunk: int, slot: int) -> dict:
    """cache_state.cpp as written: S pages rd * N_S + p (p over the cores' [first_s, +count_s) ranges = all of
    0..N_S); conv tile p: copy32 at row_off_bf16(rd) and + 512 (8 words = 32 B each)"""
    cidx, rd = divmod(slot, chunk)
    row_off = (rd >> 4) * 1024 + (rd & 15) * 32
    s = [(rd * n_s + p) * TILE_FP32 for p in range(n_s)]
    c = [(p, sorted({p * TILE_BF16 + row_off + w * 4 for w in range(8)} | {p * TILE_BF16 + row_off + 512 + w * 4 for w in range(8)}))
         for p in range(n_c)]  # fmt: skip
    return dict(cidx=cidx, rd=rd, s=s, c=c)


def decode_state_ops(n_s: int, n_c: int, chunk: int, slot: int) -> dict:
    """tt/stack_model.StackBackend._state_ops' reads of the same slot: op (1, src, rs * n_s, dst, rd * n_s, n_s, 4096,
    0) = S pages [rd n_s, +n_s); op (2, ..., _pages(C), 2048, rd) = row rd of every conv tile (SlotOps' row patch)"""
    cidx, rd = divmod(slot, chunk)
    return dict(
        cidx=cidx, rd=rd, s=[(rd * n_s + p) * TILE_FP32 for p in range(n_s)], c_tiles=list(range(n_c)), c_row=rd
    )


# ------------------------------------------------------------------------------------------------ time model
# TP8, 2,048 rows, per chip, per layer: the phase spans the stack plans with, in microseconds. Every number names its
# origin: MEASURED (a traced silicon row, geometry given), SCALED (a measured TP2 row scaled to TP8 by the work ratio
# -- stated), MODEL (a piece's traffic / FLOP model or the contract's committed span). The projection is honest about
# what is measured (little, at TP2) and what is not (everything at TP8).
TIME_MODEL_TP8 = {
    # phase: (us, origin, note)
    "P1+P10+P16 mHC": (
        200,
        "MODEL",
        "contract §3: 60 + 100 + 40 (residual L1-resident; mhc unit SP 256 bitwise hn / post)",
    ),
    "P3 qkv + low-rank stream": (
        445,
        "SCALED",
        "7a PfStream qkv 1,779 us MEASURED at TP2 (N = 12,288) / 4 (TP8 N = 3,072)",
    ),
    "P4-P6 KDA conv / prepare / scan": (
        552,
        "SCALED",
        "kda r6 2,209 us MEASURED TP2 (32 heads, 2,048 rows) / 4 (8 heads; head-parallel chains)",
    ),
    "P7 o_proj partial": (85, "MODEL", "contract §3: under P6 per chunk (17.2 GFLOP)"),
    "P2/P8/P11/P15 collective tails": (
        215,
        "MODEL",
        "contract §3: 40 (RS o) + 175 (RS FFN); AGs pipelined under P3 / P12",
    ),
    "P12 router + plan": (200, "MODEL", "contract §3 incl. P11"),
    "P13 shared expert": (45, "MODEL", "contract §3"),
    "P14 routed experts (XSL xffn)": (
        1940,
        "MODEL",
        "moe v1 DRAM floor at TP8 (509.6 MB bfp4 at ~263 GB/s effective); contract 1,062 at 480 GB/s",
    ),
    "P9 cache writes": (0, "MODEL", "under P8 (cache README: ~35 us on 8 tail cores, hidden)"),
}
TIME_MODEL_DSA_TP8 = {
    "P1+P10+P16 mHC": (200, "MODEL", "as sparse_kda"),
    "A1-A3 MLA projections + AG": (
        500,
        "MODEL",
        "op path 392 + 162 + 370 us at 31-40 % FPU (contract §1c); PfStream target ~half",
    ),
    "A4-A8 attention (absorbed, dense regime)": (
        570,
        "MODEL",
        "attn rev 3 traffic model at TP8 (silicon TP8-geometry: 2,589 us rev 2; rev 3 TP2 2,282 modelled vs 9,699 rev 2)",
    ),
    "P8/P11/P15 collective tails": (215, "MODEL", "as sparse_kda"),
    "P12 router + plan": (200, "MODEL", "as sparse_kda"),
    "P13 shared expert": (45, "MODEL", "as sparse_kda"),
    "P14 routed experts (EP, 36 per chip)": (1940, "MODEL", "same DRAM floor (509.6 MB per chip per layer)"),
    "D9 latent / indexer pages": (0, "MODEL", "under the attention phases (cache README: ~56 us on 8 cores, hidden)"),
}
TIME_MODEL_DENSE_TP8 = {
    "P1+P10+P16 mHC": (200, "MODEL", "as sparse_kda"),
    "P3 qkv + low-rank stream": (445, "SCALED", "as sparse_kda"),
    "P4-P6 KDA": (552, "SCALED", "as sparse_kda"),
    "P7 o_proj partial": (85, "MODEL", "as sparse_kda"),
    "P2/P8/P11/P15 collective tails": (215, "MODEL", "as sparse_kda"),
    "P13 dense SwiGLU slice (12,288 / 8)": (
        600,
        "MODEL",
        "77 GFLOP / 297 TFLOPS = 260 us FPU floor, 20 MB weights 42 us; planned at ~45 % FPU",
    ),
}
EMBED_HEAD_US = 1000  # contract §3: "about 1 (embed / head)" ms


def time_model(cfg=None, counts: dict | None = None) -> dict:
    """the TP8 45-layer time model: per-kind layer spans (sum of the tables above) x the kind counts + embed / head ->
    the TTFT projection for 2,048 tokens and what dominates"""
    counts = counts or (kind_counts(cfg) if cfg is not None else {SPARSE_KDA: 31, SPARSE_DSA: 11, DENSE_KDA: 3})
    per = {SPARSE_KDA: sum(v[0] for v in TIME_MODEL_TP8.values()),
           SPARSE_DSA: sum(v[0] for v in TIME_MODEL_DSA_TP8.values()),
           DENSE_KDA: sum(v[0] for v in TIME_MODEL_DENSE_TP8.values())}  # fmt: skip
    layers_us = {k: per[k] * counts[k] for k in KINDS}
    ttft_us = sum(layers_us.values()) + EMBED_HEAD_US
    moe_us = (TIME_MODEL_TP8["P14 routed experts (XSL xffn)"][0] * counts[SPARSE_KDA]
              + TIME_MODEL_DSA_TP8["P14 routed experts (EP, 36 per chip)"][0] * counts[SPARSE_DSA])  # fmt: skip
    kda_us = sum(TIME_MODEL_TP8[k][0] for k in ("P3 qkv + low-rank stream", "P4-P6 KDA conv / prepare / scan", "P7 o_proj partial")) \
        * (counts[SPARSE_KDA] + counts[DENSE_KDA])  # fmt: skip
    measured_tp8 = [
        k
        for t in (TIME_MODEL_TP8, TIME_MODEL_DSA_TP8, TIME_MODEL_DENSE_TP8)
        for k, v in t.items()
        if v[1] == "MEASURED"
    ]
    return dict(per_layer_us=per, counts=counts, layers_us=layers_us, embed_head_us=EMBED_HEAD_US, ttft_us=ttft_us,
                ttft_ms=round(ttft_us / 1e3, 1), moe_share=round(moe_us / ttft_us, 3), kda_share=round(kda_us / ttft_us, 3),
                dominant="P14 routed experts (DRAM-bound expert streaming: 42 MoE layers)",
                measured_at_tp8=measured_tp8, contract_ttft_ms=102, op_graph_ttft_ms=451.5, p4_bar_ms=300)  # fmt: skip


# ------------------------------------------------------------------------------------------------ the stack
class PmkStack:
    """Layers [0, n) of a TPPrefill as persistent programs chained on a per-bucket residual; the rest stay on the
    op graph. attach() installs the TPPrefill.pmk_layers hooks, detach() removes them (n = 0 == never attached)."""

    def __init__(self, pf, n: int, *, buckets=None, probe: bool = False, weights: str = "resident",
                 pieces=None, source=None, impl: str = "family", slots: dict | None = None, opts: dict | None = None):  # fmt: skip
        """impl "family": one family-engine program per layer with the wired piece bodies (the skeleton driver);
        "served": the per-layer program SEQUENCE of the delivered pieces (ServedLayer: A1..A5 + program B + op
        fillers), slots = slot_modes_from_env() by default, opts = the pieces' knobs (b_g, b_rect, b_fid, b_ids_u16,
        pfs_park, mhc_probe)"""
        assert impl in ("family", "served"), impl
        self.impl, self.slots = impl, dict(slots or slot_modes_from_env())
        self.opts = dict(opts or {})
        self.pf, self.n, self.probe, self.weights = pf, int(n), bool(probe), weights
        self.pieces = PIECE_WIRERS if pieces is None else list(pieces)
        self.source = source if source is not None else getattr(pf, "source", None)
        self.schedule = layer_schedule(pf.cfg)
        self.counts = kind_counts(pf.cfg)
        self.buckets = tuple(int(b) for b in (buckets or getattr(pf, "buckets", (V1_BUCKET,))))
        self.pmk_buckets = tuple(b for b in self.buckets if self.plans_bucket(b))
        assert 0 <= self.n <= len(pf.layers), (self.n, len(pf.layers))
        self.layers = [(li, pf.layers[li].L, pf.layers[li].kind) for li in range(self.n)]
        for li, L, kind in self.layers:
            assert dict(self.schedule)[L] == kind, (L, kind, dict(self.schedule)[L])
        self.shared: dict = {}  # (bucket, kind group) -> the ServedLayer owning the group's transient set
        self.res: dict = {}  # bucket -> the persistent residual
        self.progs: dict = {}  # (bucket, L) -> program
        self.reports: dict = {}  # (bucket, L) -> [piece reports]
        self.keep: list = []  # tensors / objects whose addresses the programs bake in (never freed before release)
        self.captured: dict = {}  # bucket -> trace key
        self.counters = dict(hook_calls=0, copies=0, captures=0, replays=0, eager=0)
        self._attached = False
        self.mock = os.environ.get("GLM53_MOCK") == "1"
        self.build()

    # ------------------------------------------------------------------ geometry
    def plans_bucket(self, c: int) -> bool:
        """the family engine plans ONE geometry per TP today: 64 row blocks (= 64 members) of a sequence-parallel
        bucket: c = 2,048. The 512 / 128 buckets (16 / 4 blocks) and TP8's replicated 128 fall to the op graph until
        the engine takes a blocks-per-member of 1/4 (README: open item)"""
        return int(c) == V1_BUCKET and self.pf.mode(int(c)) == "sp"

    def geometry(self, c: int) -> dict:
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK

        g = PMK.PmkGeom.from_config(self.pf.cfg, int(self.pf.ndev), int(c))
        return dict(tp=g.tp, rows=g.c_rows, sp_rows=g.sp_rows, sp_blocks=g.sp_blocks, heads_r=g.heads_r,
                    qkv_n=g.qkv_n, xsl_w=g.xsl_w, shared_r=g.shared_r, residual_mb=round(g.sp_rows * RES_COLS * 4 / 2**20, 1))  # fmt: skip

    # ------------------------------------------------------------------ build
    def _zeros(self, shape, dtype):
        import ttnn  # noqa: PLC0415

        t = ttnn.from_torch(torch.zeros(*shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.pf.mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=getattr(self.pf, "repl", None))  # fmt: skip
        self.keep.append(t)
        return t

    def _residual(self, c: int):
        """the per-bucket persistent residual in TPPrefill.streams_to_device's layout (sp: rows sharded)"""
        import ttnn  # noqa: PLC0415

        if c not in self.res:
            n = int(self.pf.ndev)
            sp = self.pf.mode(c) == "sp"
            mp = (ttnn.ShardTensorToMesh(self.pf.mesh, dim=2) if sp else self.pf.repl) if n > 1 else None
            self.res[c] = ttnn.from_torch(torch.zeros(1, 1, c, RES_COLS), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                          device=self.pf.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mp)  # fmt: skip
            # TPPrefill.layer / forward never deallocate it (they free their per-forward residuals)
            getattr(self.pf, "persistent", set()).add(id(self.res[c]))
        return self.res[c]

    def build(self):
        """one program per (bucket, layer): the kind's builder + the wired pieces' bodies, registered against the
        bucket's persistent residual (their args bake its address, so it must be the tensor the hook runs on)"""
        for c in self.pmk_buckets:
            res = self._residual(c)
            sp = self.pf.mode(c) == "sp"
            for li, L, kind in self.layers:
                if self.impl == "served":  # the pieces' program sequence (+ op fillers) on the persistent residual;
                    # ONE transient hand-off set per kind group (program A: both KDA kinds share the geometry; program B:
                    # the sparse_kda layers), owned by the group's first layer (Stage 12 item L3)
                    grp = "kda" if kind in PMK_KINDS else kind
                    self.progs[(c, L)] = ServedLayer(self, li, c, sp, self.slots, reuse=self.shared.get((c, grp)))
                    self.shared.setdefault((c, grp), self.progs[(c, L)])
                    self.reports[(c, L)] = [dict(piece="served-chain", modes=self.progs[(c, L)].modes,
                                                 pending=self.progs[(c, L)].pending)]  # fmt: skip
                    continue
                prog = KIND_BUILDERS[kind](self.pf, li, c, sp, probe=self.probe)
                reps = []
                for wire in self.pieces:
                    rep = wire(self, prog, self.pf.layers[li], res, weights=self.weights)
                    self.keep.append(rep.pop("keep", None))
                    reps.append(rep)
                self.progs[(c, L)] = prog
                self.reports[(c, L)] = reps
        return self

    # ------------------------------------------------------------------ the hook
    def _hook(self, L: int):
        def run(x, *, tpp, pl, ci, sp):
            import ttnn  # noqa: PLC0415

            c = int(ci.bucket)
            self.counters["hook_calls"] += 1
            prog = self.progs.get((c, L))
            if prog is None:  # a bucket the engine does not plan: the op-graph layer, bitwise the n = 0 path
                return self._op_layer(tpp, L, x, ci, sp)
            res = self.res[c]
            if x is not res:  # the first PMK layer of the chain: the embed's residual -> the persistent one
                ttnn.copy(x, res)
                self.counters["copies"] += 1
            prog(res, ci)
            return res  # in place: TPPrefill.layer keeps x alive only when out is x

        return run

    @staticmethod
    def _op_layer(tpp, L: int, x, ci, sp):
        li = L_index(tpp, L)
        saved = tpp.pmk_layers.pop(L)
        try:
            return tpp.layer(li, x, ci, sp, keep_input=True)
        finally:
            tpp.pmk_layers[L] = saved

    def attach(self) -> "PmkStack":
        # one-replay store A (GLM53_PF_KDA_SNAPSHOT): the KDA state at the checkpoint cut is written by the OP scan
        # (prefill_tp.kda_prefill_tp(snap=)); program A's in-program scan writes no cut scratch, so a PMK-owned KDA
        # attention would let export_kda_state(source="snapshot") read a stale state (reviews/
        # one_replay_store_A_scan_review.md finding 1) -> refused
        if getattr(self.pf, "kda_snapshot", False) and self.slots.get("attn") == "pmk" and any(
                k_ in PMK_KINDS for _, _, k_ in self.layers):  # fmt: skip
            raise ValueError(
                "GLM53_PF_KDA_SNAPSHOT=1 with the PMK chain's KDA attention: program A writes no cut state"
            )
        for _, L, _ in self.layers:
            self.pf.pmk_layers[L] = self._hook(L)
        self._attached = True
        return self

    def detach(self) -> "PmkStack":
        for _, L, _ in self.layers:
            self.pf.pmk_layers.pop(L, None)
        self._attached = False
        return self

    # ------------------------------------------------------------------ capture / replay (one per bucket)
    def trace_capture(self, c: int):
        """ONE captured trace of the whole bucket (embed -> the n PMK programs -> the op-graph layers -> head): the
        TPPrefill's key (c, 0). On the mock there is no trace: the key is recorded, the replay is the eager call."""
        c = int(c)
        assert self._attached, "attach() first: the capture must record the PMK hooks"
        assert c in self.pmk_buckets, (c, self.pmk_buckets)
        key = (c, 0)
        if key not in self.pf.traces and not self.mock:
            self.pf.capture(buckets=(c,))
            assert key in self.pf.traces, (key, list(self.pf.traces))
        self.captured[c] = key
        self.counters["captures"] += 1
        return key

    def replay(self, c: int, blocking: bool = False):
        c = int(c)
        assert c in self.captured, (c, list(self.captured))
        self.pf.replay(c, blocking=blocking)
        if self.mock:
            self.counters["eager"] += 1
        self.counters["replays"] += 1

    # ------------------------------------------------------------------ reports
    def wired(self) -> dict:
        """per (bucket, L): the phases with a registered body / the stubs / the pending note"""
        out = {}
        for (c, L), prog in self.progs.items():
            if isinstance(prog, ServedLayer):
                out[(c, L)] = dict(kind=prog.kind, impl=prog.impl, modes=prog.modes, pending=prog.pending, notes=prog.notes,
                                   programs=served_layer_plan(prog.kind, prog.tp, self.slots, xsl=getattr(prog.pl, "xsl", None) is not None)["programs"],
                                   bodies=[s for s, v in prog.modes.items() if v == "pmk"], stubs=[s for s, v in prog.modes.items() if v != "pmk"],
                                   pieces=["assembly-A"] * (prog.a is not None) + ["moe2-B"] * (prog.b is not None))  # fmt: skip
                continue
            eng = getattr(prog, "engine", None)
            phases = {p.pid: p.body.status for p in eng.phases} if eng is not None else {}
            out[(c, L)] = dict(kind=dict(self.schedule)[L], impl=getattr(prog, "impl", type(prog).__name__),
                               bodies=sorted(k for k, v in phases.items() if v == "body"),
                               stubs=sorted(k for k, v in phases.items() if v != "body"),
                               pending=getattr(prog, "pending", None), pieces=[r.get("piece") for r in self.reports[(c, L)]])  # fmt: skip
        return out

    def census(self) -> dict:
        """per kind: the planned census (every phase) and the LAUNCH census (registered phases only) -> CB ends"""
        out = {}
        for (c, L), prog in self.progs.items():
            kind = dict(self.schedule)[L]
            eng = getattr(prog, "engine", None)
            if eng is None or kind in out:
                continue
            if isinstance(
                prog, ServedLayer
            ):  # per scoped program (A1..A5) after a launch; program B's standalone ledger
                lc = getattr(prog.a.layer, "launch_census", {}) if prog.a is not None else {}
                ends = [v for d in lc.values() for v in (d.values() if isinstance(d, dict) else [d])]
                out[kind] = dict(layer=L, bucket=c, planned={k: v["cb_end"] for k, v in eng.census.items()},
                                 launch={tag: max(d.values()) if isinstance(d, dict) else d for tag, d in lc.items()},
                                 launch_by_role=lc, fits=all(v["fits"] for v in eng.census.values()),
                                 l1_max_cb_end=max(ends) if ends else max(v["cb_end"] for v in eng.census.values()),
                                 program_b_cb_total=(prog.b.l1_bytes(c) if prog.b is not None else None))  # fmt: skip
                continue
            launch = eng._census(eng.active_phases())
            out[kind] = dict(layer=L, bucket=c, planned={k: v["cb_end"] for k, v in eng.census.items()},
                             launch={k: v["cb_end"] for k, v in launch.items()},
                             fits=all(v["fits"] for v in eng.census.values()), l1_max_cb_end=max(v["cb_end"] for v in eng.census.values()))  # fmt: skip
        return out

    def streams(self, c: int) -> dict:
        """the DRAM streams between two consecutive programs of the chain, bytes per chip at bucket c"""
        g = self.geometry(c)
        tp = g["tp"]
        h_r = HEADS // tp
        s_bytes = 16 * h_r * TILE_FP32
        c_bytes = (CONV_K - 1) * (3 * h_r * HEAD_DIM // TILE) * TILE_BF16
        return dict(residual_bytes=g["sp_rows"] * RES_COLS * 4, residual_form="res[c] fp32 [1, 1, SP, 16384] DRAM (P16 writes, next P1 reads)",
                    kda_scratch_bytes=s_bytes + c_bytes, kda_scratch_form="pl.scratch S [1, H_r, 128, 128] fp32 + C [1, 3, 1, 3 W_r] bf16 (per layer, P9 -> slot pages)",
                    mhc_intermediates_bytes=2 * (g["sp_rows"] * HIDDEN * 2 + g["sp_rows"] * HIDDEN * 2 + g["sp_rows"] * NSTREAM * 4 + g["sp_rows"] * 16 * 4),
                    mhc_intermediates_form="hn bf16 x2, y x2 (lane landing), post / comb fp32 x2 -- DRAM at TP2, L1 at TP8 once coll registers",
                    chain_total_bytes=g["sp_rows"] * RES_COLS * 4, note="inside a program every hand-off is a CB (§4b); the residual is the only inter-program stream")  # fmt: skip

    def state_handoff(self) -> dict:
        """the prefill -> decode state hand-off per KDA layer: which decode tensors P9 writes (the same the op path's
        StateIO writes and the decode stack reads), and who writes the slot for the layers not on the PMK"""
        pf = self.pf
        b = getattr(pf, "b", None)
        kv = getattr(pf, "kv", None)
        chunk = int(getattr(b, "chunk", 4))
        out = dict(chunk=chunk, layers={})
        for li, pl in enumerate(pf.layers):
            if pl.kind == SPARSE_DSA:
                out["layers"][pl.L] = dict(kind=pl.kind, state="latent pages (D9 / cache_lat), no recurrent state")
                continue
            on_pmk = li < self.n
            entry = dict(kind=pl.kind, on_pmk=on_pmk, writer="P9 cache_state.cpp (in-program)" if on_pmk else "StateIO.state_out (op path, in-trace)",
                         scratch="pl.scratch (S, C)", layout=state_layout(int(pf.ndev), chunk, 0))  # fmt: skip
            if kv is not None and getattr(kv, "states", None):
                entry["slot_tensors"] = [f"kv.states[{ci}][{li}]" for ci in range(len(kv.states))]
            out["layers"][pl.L] = entry
        out[
            "state_in"
        ] = "StateIO.state_in (slot -> scratch, zeros when fresh) stays in the trace for every KDA layer: the PMK's P6 starts from pl.scratch"
        out["state_out_needed"] = any(
            not v.get("on_pmk", True) for v in out["layers"].values() if v.get("kind") != SPARSE_DSA
        )
        return out

    def signature(self) -> dict:
        """the built chain's signature: served_stack_signature's env hash + what was built (layers, modes, reuse
        owners, program B levers) -- deterministic for one env / construction (the build order is the layer order)"""
        import hashlib  # noqa: PLC0415
        import json  # noqa: PLC0415

        base = served_stack_signature(tp=int(self.pf.ndev), n_layers=len(self.pf.layers), cfg=self.pf.cfg)
        built = []
        for (c, L), prog in sorted(self.progs.items()):
            if isinstance(prog, ServedLayer):
                built.append(dict(c=c, L=L, kind=prog.kind, modes=prog.modes, reuse=prog.reuse.L if prog.reuse is not None else None,
                                  levers=str(prog.b_levers) if prog.b is not None else None, a=prog.a is not None,
                                  b_cfg={k: str(v) for k, v in sorted((getattr(prog, "b_cfg", None) or {}).items())},  # the resolved record
                                  b_form=type(prog.b).__name__ if prog.b is not None else None))  # fmt: skip
        blob = json.dumps(dict(env=base["signature"], built=built, buckets=list(self.pmk_buckets)), sort_keys=True)
        return dict(
            signature=hashlib.sha1(blob.encode()).hexdigest()[:16], env_signature=base["signature"], built=built
        )

    def summary(self) -> dict:
        return dict(n=self.n, impl=self.impl, slots=self.slots, layers=[(L, k) for _, L, k in self.layers], counts=self.counts, buckets=self.buckets,
                    pmk_buckets=self.pmk_buckets, programs=len(self.progs), captured=dict(self.captured),
                    counters=dict(self.counters), census=self.census(), wired={f"{c}:L{L}": v for (c, L), v in self.wired().items()})  # fmt: skip

    def release(self):
        self.detach()
        for prog in self.progs.values():
            rel = getattr(prog, "release", None)
            if callable(rel) and isinstance(prog, ServedLayer):
                rel()
        for t in self.res.values():
            getattr(self.pf, "persistent", set()).discard(id(t))
        self.progs.clear()
        self.keep.clear()
        self.res.clear()


def L_index(tpp, L: int) -> int:
    return next(i for i, pl in enumerate(tpp.layers) if pl.L == L)


# ------------------------------------------------------------------------------------------------ host shim
class PfShim:
    """a TPPrefill stand-in for host mocks WITHOUT weights: the mesh, the config, `ndev`, `repl`, `mode()`, the
    45-layer table as (L, kind) PrefillLayer-likes (no scratch / weights) and the hook dict -- enough for the family
    builders and the mhc wirer with random weights. Programs built on it launch on the mock mesh (generic_op)."""

    def __init__(self, mesh, cfg, layers=None, buckets=(V1_BUCKET,)):
        import ttnn  # noqa: PLC0415

        self.mesh, self.cfg = mesh, cfg
        self.ndev = int(mesh.get_num_devices())
        self.repl = ttnn.ReplicateTensorToMesh(mesh) if self.ndev > 1 else None
        self.buckets = tuple(buckets)
        sched = layer_schedule(cfg)
        self.layers = [
            SimpleNamespace(L=L, kind=k, scratch=None, mpf=None) for L, k in sched if layers is None or L in layers
        ]
        self.pmk_layers: dict = {}
        self.persistent: set = set()
        self.traces: dict = {}
        self.source = None
        self.counters = dict(eager=0)

    def mode(self, c: int) -> str:
        return "sp" if self.ndev > 1 and c % (TILE * self.ndev) == 0 else "rep"

    def capture(self, buckets=None):  # the mock has no traces
        return None

    def replay(self, c, blocking=False):
        self.counters["eager"] += 1


@dataclasses.dataclass
class StackJob:
    """the ONE device unit's arguments (README §5): n PMK layers vs n = 0 on the same TPPrefill, one bucket"""

    n: int = 1
    layer: int = 4
    rows: int = V1_BUCKET
    geometry: str = "pair"
    pcc_bar: float = 0.9999
    replays: int = 200
    tag: str = "w-d0-pmk-stack-n1-L4-pair"


# ==================================================================================================== LONG PROMPTS
# (coordinator follow-up 2026-10-22: 128K-1M prompts on TP8 = a loop of bucket replays; two levers + a cost model)
#
# Lever 1, LARGE BUCKETS. An 8,192-row bucket at TP8 = 1,024 sequence-parallel rows per chip = the family's TP2
# per-chip ROW geometry (NOTE_16_GEOMETRY: 1,024 SP rows, the stores in DRAM) with TP8's HEAD count (8). The expert
# weight stream (509.6 MB per chip per layer) is paid once per replay, so its bytes per token fall 4x (16,384: 8x).
# The engine today keeps the gathered x-hat store per member (64 blocks = 64 members) and sub-chunks larger buckets
# into weight re-stream passes (bucket_census() shows it: member 2.4 / 3.9 MB, passes 4 / 8); the large-bucket form
# is the TP2 one -- XS / chunk store / residual in DRAM, the member a 32-row streaming window -- whose member census
# is the engine's own TP2 number (bucket_census(2048, tp=2)).
#
# Lever 2, DEVICE-RESIDENT CHUNK LOOP (ChunkLoop): ONE trace = N consecutive chunk forwards (head-less, one key
# window t for all N: the DSA select scores every pool a chunk start <= t - c can see and masks with the DEVICE
# chunk start, prefill_tp.SparseCtx.cs_fixed), every per-chunk device input (tokens, valid_end, chunk / user /
# pool tables, chunk_start, ring rows, the StateIO ctl word) read from a device-resident ARG TABLE the host fills
# once per prompt -> the host issues ceil(L / (N c)) traces instead of L / c and never waits between chunks.
EXPERT_BYTES_TP8 = 509.6e6  # bfp4 XSL slices per chip per MoE layer (contract §3)
DENSE_W_BYTES = {
    SPARSE_KDA: 26.1e6,
    SPARSE_DSA: 29.0e6 + 2.4e6 + 3.3e6,
    DENSE_KDA: 20.4e6 + 20.0e6,
}  # non-expert weights
EXPERT_GBPS_FLOOR, EXPERT_GBPS_TARGET = (
    263.0,
    493.0,
)  # moe v1 effective (1,940 us) / the coordinator's 493 GB/s (1.034 ms)
PEAK_TFLOPS, FPU_EFF = 297.0, 0.4  # HiFi2 grid peak; the efficiency assumed for the attention / indexer terms
# per BUCKET ROW compute costs (us per row of the 2,048 bucket at TP8), from the §5 time model's row-proportional phases
PER_ROW_US = {
    SPARSE_KDA: dict(
        mhc=200 / 2048,
        qkv=445 / 2048,
        kda=552 / 2048,
        o_proj=85 / 2048,
        coll=215 / 2048,
        router=200 / 2048,
        shared=45 / 2048,
    ),
    SPARSE_DSA: dict(
        mhc=200 / 2048, mla=500 / 2048, attn_dense=570 / 2048, coll=215 / 2048, router=200 / 2048, shared=45 / 2048
    ),
    DENSE_KDA: dict(mhc=200 / 2048, qkv=445 / 2048, kda=552 / 2048, o_proj=85 / 2048, coll=215 / 2048, mlp=600 / 2048),
}
P14_FLOP_PER_ROW = 103.1e9 / 2048  # routed experts, LoFi (2x the HiFi2 rate)
INDEX_HEADS, INDEX_DIM, KPOOL, TOPK_TOKENS = 32, 128, 4, 2048
DENSE_EXACT = 2051
HOST_US_PER_TRACE = (
    13_000.0  # Stage 16 TP2: TTFT(2,048) 148 ms host-dominated vs 135 ms replay -> 13 ms per chunk round trip
)
EMBED_HEAD_PER_CHUNK_US = 1000.0


def bucket_geometry(c: int, tp: int = 8) -> dict:
    """the per-chip geometry of bucket c at TP tp and the DRAM forms of its stores (bytes per chip)"""
    sp = c // tp
    h_r = HEADS // tp
    return dict(
        bucket=c,
        tp=tp,
        sp_rows=sp,
        sp_blocks=sp // TILE,
        c_blocks=c // TILE,
        heads_r=h_r,
        qkv_n=3 * h_r * HEAD_DIM,
        residual_bytes=sp * RES_COLS * 4,
        xs_store_bytes=c * HIDDEN * 2,  # the gathered x-hat [c, 4096] bf16
        chunk_store_bytes=c * (3 * h_r * HEAD_DIM * 2 + h_r * HEAD_DIM * 4 + h_r * 4),  # q / k / v bf16, g fp32, beta
        x_ext_bytes=c * HIDDEN * 2,
        kda_chains=h_r * 4,
        kda_chunks=c // TILE,  # 32 chains x c / 32 serial steps
        latent_rows_bytes=c * 512 * 2,
        fabric_in_bytes=4 * c * HIDDEN * 2 * (tp - 1) // tp,
    )  # 2 AG + 2 RS


def bucket_census(c: int, tp: int = 8, place=None) -> dict:
    """the family engine's plan at (c, tp): census + policy (passes = weight re-streams of the member-resident form);
    and the DRAM-store form's member census = the engine's TP2 plan at the same per-chip rows (its stores are DRAM)"""
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK

    place = place or PMK.Placement(11, 10)
    g = PMK.PmkGeom(c_rows=c, tp=tp)
    p = PMK.PmkSparseKdaPlan(g, place).plan()
    out = dict(geometry=bucket_geometry(c, tp), engine_census={k: v["cb_end"] for k, v in p.census.items()},
               engine_policy={k: p.policy.get(k) for k in ("stationary", "passes", "re_stream_bytes", "chunk_store", "residual")},
               handoff_mib={h: round(p.handoffs[h].payload_bytes / 2**20, 1) for h in ("H2", "H3", "H5", "H8")})  # fmt: skip
    rows_per_chip = c // tp
    tp_same_rows = (
        2048 // rows_per_chip if rows_per_chip and 2048 % rows_per_chip == 0 and 2048 // rows_per_chip >= 2 else None
    )
    if tp_same_rows:  # the geometry whose per-chip rows match, with the stores in DRAM (NOTE_16_GEOMETRY)
        q = PMK.PmkSparseKdaPlan(PMK.PmkGeom(c_rows=2048, tp=tp_same_rows), place).plan()
        out["dram_form"] = dict(from_geometry=f"2048 @ tp{tp_same_rows} ({rows_per_chip} SP rows per chip, stores in DRAM)",
                                member_cb_end=q.census["member"]["cb_end"], root_cb_end=p.census["root"]["cb_end"],
                                mm_only_cb_end=p.census["mm_only"]["cb_end"], passes=1,
                                note="member = the TP2 census (rows), root / mm_only = this TP's rings (heads); XS / chunk / residual in DRAM")  # fmt: skip
    return out


def layer_cost_us(kind: str, rows: int, pos_end: int, *, expert_gbps: float = EXPERT_GBPS_TARGET, fpu_eff: float = FPU_EFF,
                  gather_union: float = 2.0) -> dict:  # fmt: skip
    """one layer, one replay of `rows` bucket rows whose last position is pos_end (TP8): weight term (paid once per
    replay) + compute term (per row at the phase targets) + the DSA attention term vs context position.
    sparse regime (pos_end > 2,051): the indexer scores every visible pool for every query (per chip FLOP = rows x
    (pos / 4) x 32 heads x 128 x 2 / 8 chips = rows x pos x 256: O(L^2) over a prompt), then top-512 pools -> 2,048
    tokens per query: attention FLOP per row = 8 heads x 2 x 2 x 2,048 x 512, and the selected latents' gather bytes
    = (2,048 x 1 KiB) x gather_union / 128 queries per q-block (the union of a block's selections, assumed 2x one
    query's)."""
    dense_w = DENSE_W_BYTES[kind] / (EXPERT_GBPS_TARGET * 1e3)  # us
    expert_w = EXPERT_BYTES_TP8 / (expert_gbps * 1e3) if kind != DENSE_KDA else 0.0
    per_row = sum(PER_ROW_US[kind].values())
    compute = rows * per_row
    p14_compute = rows * P14_FLOP_PER_ROW / (2 * PEAK_TFLOPS * 1e6 * 0.5) if kind != DENSE_KDA else 0.0  # LoFi, 50 %
    p14 = max(expert_w, p14_compute)
    attn = dict(indexer_us=0.0, sparse_attn_us=0.0, gather_us=0.0)
    if kind == SPARSE_DSA and pos_end > DENSE_EXACT:
        pos_mid = max(pos_end - rows // 2, DENSE_EXACT)  # the chunk's queries see pos_mid pools on average
        flop_idx = rows * pos_mid * 256.0
        attn["indexer_us"] = flop_idx / (PEAK_TFLOPS * 1e6 * fpu_eff)
        flop_attn = rows * 8 * 2 * 2 * TOPK_TOKENS * 512
        attn["sparse_attn_us"] = flop_attn / (PEAK_TFLOPS * 1e6 * fpu_eff)
        attn["gather_us"] = rows * (TOPK_TOKENS * 1024 * gather_union / 128) / (EXPERT_GBPS_TARGET * 1e3)
    total = dense_w + p14 + compute + sum(attn.values())
    return dict(kind=kind, rows=rows, pos_end=pos_end, dense_w_us=dense_w, expert_w_us=expert_w, p14_us=p14,
                compute_us=compute, **attn, total_us=total)  # fmt: skip


def chunk_cost_us(rows: int, pos_end: int, counts: dict | None = None, **kw) -> dict:
    counts = counts or {SPARSE_KDA: 31, SPARSE_DSA: 11, DENSE_KDA: 3}
    per = {k: layer_cost_us(k, rows, pos_end, **kw) for k in KINDS}
    tot = sum(per[k]["total_us"] * counts[k] for k in KINDS) + EMBED_HEAD_PER_CHUNK_US
    return dict(rows=rows, pos_end=pos_end, total_us=tot, per_kind=per,
                weight_us=sum((per[k]["dense_w_us"] + per[k]["expert_w_us"]) * counts[k] for k in KINDS),
                indexer_us=per[SPARSE_DSA]["indexer_us"] * counts[SPARSE_DSA],
                attn_us=(per[SPARSE_DSA]["sparse_attn_us"] + per[SPARSE_DSA]["gather_us"]) * counts[SPARSE_DSA],
                us_per_token=tot / rows)  # fmt: skip


def prefill_ttft(L: int, bucket: int, loop_n: int = 1, *, host_us_per_trace: float = HOST_US_PER_TRACE, **kw) -> dict:
    """TTFT of an L-token prompt as a loop of `bucket` replays, N chunks per trace: device time (the cost model
    per chunk at its end position) + the host round trip per TRACE"""
    chunks = -(-L // bucket)
    dev = 0.0
    parts = dict(weight_us=0.0, indexer_us=0.0, attn_us=0.0)
    for k in range(chunks):
        cc = chunk_cost_us(bucket, min(L, (k + 1) * bucket), **kw)
        dev += cc["total_us"]
        for p in parts:
            parts[p] += cc[p]
    traces = -(-chunks // loop_n)
    host = traces * host_us_per_trace
    return dict(L=L, bucket=bucket, loop_n=loop_n, chunks=chunks, traces=traces, device_us=dev, host_us=host,
                ttft_us=dev + host, ttft_s=round((dev + host) / 1e6, 3), **{k: round(v / 1e6, 3) for k, v in parts.items()},
                weight_share=round(parts["weight_us"] / dev, 3), indexer_share=round(parts["indexer_us"] / dev, 3))  # fmt: skip


def ttft_table(lengths=(32768, 131072, 1048576), buckets=(2048, 8192, 16384), loop_ns=(1, 4, 16), **kw) -> list:
    return [prefill_ttft(L, b, n, **kw) for L in lengths for b in buckets for n in loop_ns]


# ------------------------------------------------------------------------------------------------ the chunk loop
CHUNK_ARG_WORDS = 16  # per-chunk control row of the arg table (uint32): slot, fresh, chunk_start, n_valid, t, k, 0..


class ChunkLoop:
    """ONE trace = N consecutive chunks of one request (head-less: the head runs on the prompt's last chunk through
    the ordinary single-chunk key). Host side (no new kernels): N ChunkInputs + N StateIO ctl words = the
    device-resident ARG TABLE (rows k = 0..N-1; every per-chunk program of unrolled chunk k binds row k's tensors
    by address at capture), `fill()` writes every row once per prompt (one host -> device burst, no replay waits
    between chunks), `capture()` records the N forwards under one key ("loop", c, N, t), `replay()` runs them.
    PMK bodies: a body that reads per-chunk device words (the cache piece's SEGTAB address, the kda piece's
    valid_end / chunk_start) declares them as `chunk_args` (role, risc) -> [(rt index, base tensor name)]; `bind(k)`
    rewrites those runtime args to row k's address before the k-th unrolled launch and `audit()` refuses a body whose
    args reference a per-chunk tensor without declaring it (the family's audit_args runs on the rebound descriptor)."""

    def __init__(self, stack: "PmkStack", n: int, c: int | None = None):
        import ttnn  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import StateIO  # noqa: PLC0415

        self.stack, self.pf, self.n = stack, stack.pf, int(n)
        self.c = int(c or stack.pmk_buckets[0])
        assert self.n >= 1 and self.c in stack.pmk_buckets, (self.n, self.c, stack.pmk_buckets)
        pf = self.pf
        self.cis = [pf._new_chunk_inputs(self.c) for _ in range(self.n)]  # row k's per-chunk device inputs
        self.sios = [StateIO(pf) for _ in range(self.n)] if getattr(pf, "sio", None) is not None else None
        self.table = ttnn.from_torch(torch.zeros(self.n, CHUNK_ARG_WORDS, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=pf.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=pf.repl)  # fmt: skip
        self.key = None
        self.rows_filled = 0
        self.counters = dict(fills=0, captures=0, replays=0, forwards=0, binds=0)
        self.chunk_args: dict = {}  # (L, (role, risc)) -> [(rt index, name)] declared per-chunk runtime args

    # ------------------------------------------------------------------ the arg table
    def row_words(self, k: int, *, slot: int, chunk_start: int, n_valid: int, t: int) -> list:
        return [slot, int(chunk_start == 0), chunk_start, n_valid, t, k] + [0] * (CHUNK_ARG_WORDS - 6)

    def fill(self, tokens: torch.Tensor, *, slot: int, chunk_start: int, t: int) -> int:
        """rows 0..N-1 <- the N consecutive chunks of `tokens` (N x c tokens, the last may be short) from position
        chunk_start of `slot`; the host's only per-prompt writes (one burst). Returns the chunks written."""
        import ttnn  # noqa: PLC0415

        pf = self.pf
        tokens = tokens.reshape(-1)
        n_chunks = min(self.n, -(-int(tokens.numel()) // self.c))
        host = torch.zeros(self.n, CHUNK_ARG_WORDS, dtype=torch.int32)
        for k in range(n_chunks):
            tk = tokens[k * self.c : (k + 1) * self.c]
            cs = chunk_start + k * self.c
            if self.sios is not None:
                self.sios[k].write(slot, fresh=cs == 0)
            saved = pf.sio
            pf.sio = None  # write_inputs' own ctl write is per-k (above)
            try:
                pf.write_inputs(self.c, tk, slot=slot, chunk_start=cs, n_valid=int(tk.numel()), ci=self.cis[k])
            finally:
                pf.sio = saved
            host[k] = torch.tensor(
                self.row_words(k, slot=slot, chunk_start=cs, n_valid=int(tk.numel()), t=t), dtype=torch.int32
            )
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                        mesh_mapper=pf.repl), self.table)  # fmt: skip
        self.rows_filled = n_chunks
        self.counters["fills"] += 1
        return n_chunks

    # ------------------------------------------------------------------ PMK bodies: per-chunk args
    def declare(self, L: int, role: str, risc: str, args: list):
        """a registered body's per-chunk runtime args: [(rt index, name in {"segtab", "ctl", "valid_end",
        "chunk_start", "tokens"})] -- rewritten to row k's address by bind(k)"""
        self.chunk_args[(L, (role, risc))] = list(args)

    def _row_tensor(self, k: int, name: str):
        ci = self.cis[k]
        return dict(segtab=self.table, ctl=self.sios[k].ctl if self.sios else self.table, valid_end=ci.valid_end,
                    chunk_start=ci.chunk_start, tokens=ci.tokens)[name]  # fmt: skip

    def bind(self, k: int):
        """rewrite every declared per-chunk runtime arg of every PMK program to row k's address (before the k-th
        unrolled launch; the engine's descriptor() + audit_args then run on the rebound args)"""
        for (L, (role, risc)), args in self.chunk_args.items():
            prog = self.stack.progs[(self.c, L)]
            for p in prog.engine.phases:
                part = p.body.parts.get((role, risc)) if p.body.status == "body" else None
                if part is None:
                    continue
                for xy, rt in part["rt"].items():
                    for idx, name in args:
                        rt[idx] = int(self._row_tensor(k, name).buffer_address()) + (
                            CHUNK_ARG_WORDS * 4 * k if name == "segtab" else 0
                        )
        self.counters["binds"] += 1

    def audit(self) -> dict:
        """every registered body's runtime args vs the per-chunk tensors: an arg equal to a per-chunk base address
        (row 0's) that is not declared is a HIDDEN per-chunk dependency -> refused (it would replay chunk 0's
        inputs on every unrolled chunk)"""
        bases = {}
        for name in ("valid_end", "chunk_start", "tokens"):
            bases[int(self._row_tensor(0, name).buffer_address())] = name
        bases[int(self.table.buffer_address())] = "segtab"
        if self.sios:
            bases[int(self.sios[0].ctl.buffer_address())] = "ctl"
        hidden, declared = [], 0
        for (c, L), prog in self.stack.progs.items():
            if c != self.c:
                continue
            for p in prog.engine.phases:
                if p.body.status != "body":
                    continue
                for (role, risc), part in p.body.parts.items():
                    decl = {i for i, _ in self.chunk_args.get((L, (role, risc)), [])}
                    declared += len(decl)
                    for xy, rt in part["rt"].items():
                        for i, v in enumerate(rt):
                            if int(v) in bases and i not in decl:
                                hidden.append(
                                    dict(L=L, phase=p.pid, role=role, risc=risc, core=xy, index=i, tensor=bases[int(v)])
                                )
        if hidden:
            raise ValueError(f"ChunkLoop audit: undeclared per-chunk runtime args {hidden[:8]} ({len(hidden)} total)")
        return dict(declared=declared, hidden=0, bases=sorted(bases.values()))

    # ------------------------------------------------------------------ capture / replay
    def _forward_chunk(self, k: int, t: int):
        pf = self.pf
        saved = (pf.inputs.get(self.c), getattr(pf, "sio", None))
        pf.inputs[self.c] = self.cis[k]
        if self.sios is not None:
            pf.sio = self.sios[k]
        try:
            self.bind(k)
            pf.forward(self.c, head=False, t=t)
            self.counters["forwards"] += 1
        finally:
            pf.inputs[self.c] = saved[0]
            pf.sio = saved[1]

    def capture(self, t: int = 0, compile_first: bool = True):
        """compile launch (the N forwards once, eager: their programs' first launches, before the capture) then ONE
        capture of the N head-less forwards under ("loop", c, N, t). t = the key window of the LAST chunk
        (TPPrefill.plan(chunk_start + (N - 1) c, c)[1]); 0 = dense (no sparse_dsa, or the prompt < 2,052)."""
        import ttnn  # noqa: PLC0415

        pf = self.pf
        key = ("loop", self.c, self.n, int(t))
        self.audit()
        if compile_first:
            pf.quiesce()  # the eager compile pass may follow trace replays (TPPrefill.quiesce)
            for k in range(self.n):
                self._forward_chunk(k, t)
            ttnn.synchronize_device(pf.mesh)
        if self.stack.mock:
            self.key = key
            self.counters["captures"] += 1
            return key
        tid = ttnn.begin_trace_capture(pf.mesh, cq_id=0)
        try:
            for k in range(self.n):
                self._forward_chunk(k, t)
        except BaseException:
            ttnn.end_trace_capture(pf.mesh, tid, cq_id=0)
            ttnn.release_trace(pf.mesh, tid)
            raise
        ttnn.end_trace_capture(pf.mesh, tid, cq_id=0)
        pf.traces[key] = tid
        pf.counters["captures"] += 1
        self.key = key
        self.counters["captures"] += 1
        return key

    def replay(self, blocking: bool = False):
        import ttnn  # noqa: PLC0415

        assert self.key is not None, "capture() first"
        if self.stack.mock:
            for k in range(self.rows_filled or self.n):
                self._forward_chunk(k, self.key[3])
            self.counters["replays"] += 1
            return
        ttnn.execute_trace(self.pf.mesh, self.pf.traces[self.key], cq_id=0, blocking=blocking)
        self.pf.counters["replays"] += 1
        self.counters["replays"] += 1

    def prompt_plan(self, L: int, chunk_start: int = 0) -> dict:
        """how a prompt of L tokens runs: full N-chunk traces, then single chunks, the last with the head"""
        chunks = -(-L // self.c)
        loops = chunks // self.n
        rest = chunks - loops * self.n
        return dict(L=L, bucket=self.c, N=self.n, chunks=chunks, loop_traces=loops, single_chunks=rest,
                    traces=loops + rest, head_on="last single chunk" if rest else "one extra head-only replay of the last row",
                    host_round_trips_removed=chunks - (loops + rest))  # fmt: skip


# ==================================================================================================== SERVED STACK
# GLM53_PF_MODE=pmk (Stage 9's serving switch; doc/ttnn/16-prefill-production/pmk/stack/README.md §11). Every planned
# bucket's trace runs each KDA layer as the PER-LAYER PROGRAM SEQUENCE of the delivered pieces on the persistent
# residual:
#
#     A1 (mhc pre attn) -> A2 (AG + projections) + A3 (kda) -> [attention piece: sparse_dsa] -> A4 (o_proj + RS)
#     -> A5 (mhc post attn + pre ffn + AG) -> program B (routed experts + shared + combine) -> mhc post ffn
#
# grouped into five SLOTS. A slot runs on the PMK ("pmk") when its piece is on the tree for the layer's kind and the
# knob allows it, otherwise the OP GRAPH runs that slot INSIDE THE SAME TRACE, reading and writing the pieces' DRAM
# hand-off tensors (hn[0] / y / posts / combs / hn[1] / y_ffn: ProgramAHandoff). The chain is therefore correct at every
# adoption level, every launch is captured by the bucket's one trace (TPPrefill.capture -> the pmk_layers hook), the
# compile launch is TPPrefill.compile()'s eager forward, and nothing runs eagerly afterwards -- the op graph's own
# capture / replay contract. The moment a piece passes its TP8 unit, its slot flips to "pmk" and nothing else changes.
SLOTS = ("mhc_pre", "attn", "mhc_mid", "ffn", "mhc_post")
SLOT_PIECES = {
    "mhc_pre": "A1 = mhc part P1 (assembly program 0): x -> hn[0], posts[0], combs[0]",
    "attn": "A2 = C1 AG + PfStream qkv / lowrank / f_b / g_b (program 1), A3 = kda items + scan (program 2; S / C into "
    "pl.scratch), A4 = PfStream o_proj + RS (program 3; the pair's RSR lanes; TP8: the o_proj partial, the RS is "
    "an op reduce-scatter into y until coll's RS8 lands)",
    "mhc_mid": "A5 = pcoef(0) + mhc part P10 + P11 AG (program 4): y, posts[0] / combs[0] -> x1 (in place), hn[1], "
    "posts[1], combs[1]",
    "ffn": "program B (moe2 ProgramB: 24 adj streamers + receivers, P12 plan / P14 experts / P15 combine with the shared "
    "term) on x_tile = the op gather of hn[1]; router + shared expert + RS = op fillers (P12 / P13 / P15's "
    "in-program forms are moe2 / coll follow-ups)",
    "mhc_post": "P16 = mhc post (ffn): NOT a registered assembly program (pending) -> the op post reads A5's pages "
    "(posts[1] / combs[1]) and y_ffn; writes the residual in place",
}
PMK_KINDS = (SPARSE_KDA, DENSE_KDA)  # program A's kinds (sparse_dsa: pending, see ServedLayer.pending)


def slot_modes_from_env(env=None) -> dict:
    """GLM53_PMK_SLOTS: "auto" (default) = every slot on the PMK where its piece exists (pending slots fall to the op
    graph and are reported); "none" = every slot on the op graph (the chain object is still built: the unit's
    reference on the same TPPrefill); else a comma list of the slots to run on the PMK (the others op)."""
    env = os.environ if env is None else env
    v = (env.get("GLM53_PMK_SLOTS", "auto") or "auto").strip().lower()
    if v == "auto":
        return {s: "pmk" for s in SLOTS}
    if v == "none":
        return {s: "op" for s in SLOTS}
    want = {x.strip() for x in v.split(",") if x.strip()}
    bad = want - set(SLOTS)
    assert not bad, f"GLM53_PMK_SLOTS: unknown slots {sorted(bad)} (known: {SLOTS})"
    return {s: ("pmk" if s in want else "op") for s in SLOTS}


def dsa_mhc_from_env(env=None) -> bool:
    """GLM53_PMK_DSA_MHC (default 1): the sparse_dsa layers run A1 / A5 (the mhc-only program A) around their op
    attention; 0 = every sparse_dsa slot on the op graph (before 14b's stack45)"""
    env = os.environ if env is None else env
    return (env.get("GLM53_PMK_DSA_MHC", "1") or "1").strip() != "0"


def served_layer_plan(kind: str, tp: int, modes: dict, *, xsl: bool = True, dsa_mhc: bool | None = None) -> dict:
    """pure python: which slots of a layer of `kind` at TP `tp` run on the PMK under `modes`, and why the others do
    not (the same rules ServedLayer applies; tests/s16/test_pmk_stack.py)"""
    out = {s: modes.get(s, "op") for s in SLOTS}
    pending = {}
    dsa_mhc = dsa_mhc_from_env() if dsa_mhc is None else bool(dsa_mhc)
    if kind == SPARSE_DSA and dsa_mhc:  # 14b stack45: A1 + A5 (mhc-only program A) around the op MLA / DSA attention
        why = dict(attn="sparse_dsa attention: 16's PMK attention entry (rev 12.2) is not on this tree -> the op MLA / DSA "
                        "between A1 and A5 (reads hn[0], writes ys[0])",
                   ffn="program B needs the XSL-sliced experts; the sparse_dsa layers load EP experts (TPDsaLayer) -> op ffn "
                       "(router on the fp32 x-hat_f filler, as the sparse_kda chain)",
                   mhc_post="P16 (mhc post ffn) is not an assembly program yet -> op post on A5's pages")  # fmt: skip
        for s in ("attn", "ffn", "mhc_post"):
            if out[s] == "pmk":
                pending[s] = why[s]
                out[s] = "op"
        # GLM53_PMK_DSA_SLOTS (the DSA L3 race bisect): which of A1 (mhc_pre) / A5 (mhc_mid) run on the PMK on the
        # sparse_dsa layers only (default both); the others take the op graph inside the same trace
        dsa_slots = {
            v.strip() for v in str(os.environ.get("GLM53_PMK_DSA_SLOTS", "mhc_pre,mhc_mid")).split(",") if v.strip()
        }
        for s in ("mhc_pre", "mhc_mid"):
            if out[s] == "pmk" and s not in dsa_slots:
                pending[
                    s
                ] = f"GLM53_PMK_DSA_SLOTS={','.join(sorted(dsa_slots)) or 'none'}: {s} on the op graph (sparse_dsa)"
                out[s] = "op"
        progs = int(out["mhc_pre"] == "pmk") + int(out["mhc_mid"] == "pmk")
        return dict(modes=out, pending=pending, programs=progs, notes=dict(dsa="mhc-only program A (SCOPES_MHC)"))
    if kind not in PMK_KINDS:
        for s in SLOTS:
            if out[s] == "pmk":
                pending[s] = (
                    "sparse_dsa: program A's mhc / MLA projection programs and the attention piece's layer "
                    "seam (MlaPrefill q_lat / latent_all / sel / o_lat as persistent tensors) are not on the "
                    "tree -> op graph"
                )
                out[s] = "op"
        return dict(modes=out, pending=pending, programs=0)
    if out["mhc_post"] == "pmk":
        pending["mhc_post"] = "P16 (mhc post ffn) is not an assembly program yet -> op post on A5's pages"
        out["mhc_post"] = "op"
    if out["ffn"] == "pmk" and kind == DENSE_KDA:
        pending["ffn"] = "dense_kda: the dense SwiGLU slice (17's P13) has no program-B form -> op ffn"
        out["ffn"] = "op"
    if out["ffn"] == "pmk" and not xsl:
        pending["ffn"] = "program B needs the XSL-sliced experts (GLM53_TP_XSL=1 GLM53_TP_XSL_SLICED=1) -> op ffn"
        out["ffn"] = "op"
    progs = 0
    if out["mhc_pre"] == "pmk":
        progs += 1
    if out["attn"] == "pmk":
        progs += 3
    if out["mhc_mid"] == "pmk":
        progs += 1
    if out["ffn"] == "pmk":
        progs += 1
    notes = {}
    if out["attn"] == "pmk" and tp not in (2, 8):
        notes[
            "attn"
        ] = "A4 carries no RS at this TP (pair RSR / TP8 RS8 only): the o_proj partial is reduce-scattered by an op filler into y"
    return dict(modes=out, pending=pending, programs=progs, notes=notes)


def _dram_alloc_mb(mesh) -> float:
    """DRAM allocated per bank (MB; lock-step allocation: chip 0 = every chip)"""
    import ttnn  # noqa: PLC0415

    try:
        return int(ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank) / 2**20
    except Exception:  # noqa: BLE001
        return 0.0


def _assert_live(t, name: str):
    """t is an allocated DEVICE tensor on a mesh (host tensors have no device; a deallocated one is not allocated)"""
    ok_alloc = True
    try:
        ok_alloc = bool(t.is_allocated())
    except Exception:  # noqa: BLE001 -- a binding without is_allocated: the device check below
        pass
    dev = None
    try:
        dev = t.device()
    except Exception:  # noqa: BLE001
        dev = None
    assert (
        ok_alloc and dev is not None
    ), f"chain tensor {name} is not a live device tensor (allocated {ok_alloc}, device {dev})"


def _copy_into(src, dst):
    """src -> dst (persistent, address baked into a program), through a typecast when the dtypes differ"""
    import ttnn  # noqa: PLC0415

    if src.dtype == dst.dtype:
        ttnn.copy(src, dst)
        return
    t = ttnn.typecast(src, dst.dtype)
    ttnn.copy(t, dst)
    ttnn.deallocate(t)


def _l1_floor(mesh):
    """the lowest allocated L1 address per bank (chip 0's allocator = every chip's, lock-step; stack_model.l1_report's
    view), or None without an allocator view"""
    import ttnn  # noqa: PLC0415

    try:
        base = int(ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1))
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        addrs = [
            base + int(b["address"])
            for b in mv.block_table
            if str(b.get("allocated", "")).lower() in ("yes", "true", "1")
        ]
        return min(addrs) if addrs else None
    except Exception:  # noqa: BLE001
        return None


def _l1_floor_check(prog, mesh, what: str):
    """cf3h-k (14's review B1): a fused PMK program's CB / window ends (its launch census, every role) must sit BELOW the
    lowest live L1 buffer -- tt-metal validates only the launch-time static CBs, the windows' re-laid CBs are the PMK's
    own (laid up to the fixed 1,330,000 B ceiling). GLM53_PMK_L1_FLOOR_CHECK=0 turns it off (a diagnosis)"""
    if os.environ.get("GLM53_PMK_L1_FLOOR_CHECK", "1") == "0":
        return None
    floor = _l1_floor(mesh)
    cen = getattr(prog.layer, "launch_census", None) or {}
    ends = {
        f"{tag}/{role}": int(e_) for tag, roles in cen.items() if isinstance(roles, dict) for role, e_ in roles.items()
    }
    if floor is None or not ends:
        return dict(floor=floor, worst=None)
    worst = max(ends.items(), key=lambda kv: kv[1])
    if worst[1] > floor:
        over = {k: v for k, v in ends.items() if v > floor}
        try:  # name the live L1 blocks under the program's CB end (stack_model.l1_report: owners by a gc / object walk)
            from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM_  # noqa: PLC0415

            rows_ = SM_.l1_report(mesh, f"PMK L1 FLOOR {what}")
            print(f"PMK L1 FLOOR {what}: live blocks below the CB end {worst[1]:,}: "
                  f"{[(r_['address'], r_['size'], r_['tensors'][:2]) for r_ in rows_ if r_['address'] < worst[1]]}", flush=True)  # fmt: skip
        except Exception as e_:  # noqa: BLE001
            print(f"PMK L1 FLOOR {what}: l1_report failed {e_!r}", flush=True)
        raise ValueError(
            f"PMK L1 FLOOR ({what}): CB / window ends over the lowest live L1 buffer {floor:,}: {over} -- "
            f"the program's CBs would overwrite persistent L1 (14's review B1)"
        )
    return dict(floor=floor, worst=worst)


def _asm_descs(a, x, ci):
    """program A's per-scope descriptors for this forward (PmkSparseKdaLayer.__call__'s build, without its launches:
    the served chain launches the scoped programs one by one with the op fillers between them)"""
    import ttnn  # noqa: PLC0415

    lay, e = a.layer, a.engine
    pl = a.pl
    scratch = getattr(pl, "scratch", None)
    io = [x] + [t for t in (getattr(scratch, "S", None), getattr(scratch, "C", None)) if t is not None]
    seen = {id(t) for t in io}
    for p in e.phases:
        for t in p.body.io if p.body.status == "body" else ():
            if id(t) not in seen:
                io.append(t)
                seen.add(id(t))
    io.append(io.pop(0))  # generic_op's last io is the output: the residual (in place)
    dev = x.device() or a.pf.mesh
    n = int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1
    descs, census = [], {}
    try:
        for sc in e.programs:
            e.scope = set(sc)
            if n > 1:
                R, C_ = int(dev.shape[0]), int(dev.shape[1])
                desc = ttnn.MeshProgramDescriptor()
                for r in range(R):
                    for c_ in range(C_):
                        m = ttnn.MeshCoordinate(r, c_)
                        desc[ttnn.MeshCoordinateRange(m, m)] = e.descriptor(
                            dev, io, rank=r * C_ + c_, fs_sync_addr=lay.fs_sync_addr
                        )
            else:
                desc = e.descriptor(dev, io, rank=0, fs_sync_addr=lay.fs_sync_addr)
            launch = e._census(e.active_phases())
            tag = "+".join(sorted(sc, key=lambda q: int(q[1:])))
            census[tag] = {k: v["cb_end"] for k, v in launch.items()}
            over = {k: v["cb_end"] for k, v in launch.items() if not v["fits"]}
            if over and os.environ.get("GLM53_MOCK") != "1":
                raise ValueError(f"PMK launch census over the CB ceiling on a DEVICE: {over} (program {tag})")
            descs.append(desc)
    finally:
        e.scope = None
    lay.descs, lay.launch_census, lay.desc = descs, census, descs[0]
    return io, descs


def w7_from_env(env=None) -> bool:
    """GLM53_PMK_W7=1 (cf3h-k, operator directive 2026-10-26 09:52:27Z): every sparse_kda layer chunk of a PMK bucket runs
    Stage 16's fused W7 program -- A15 (P1 .. P10) + P11 (the x-hat_f AG) + P12 (the router tail) + P13 (the shared
    expert) + program B (P14, the routed experts) + P15 (the FFN partial's RS8) in ONE launch per layer -- instead of A15
    + the op-path MoE. dense_kda / sparse_dsa layers and the 512 bucket (op path) are unchanged. Knobs of record:
    doc/ttnn/16-prefill-production/pmk/w7/composition_handoff.md (the W7 gate: w7lv4 / w7lv5 at TP8)."""
    env = os.environ if env is None else env
    return (env.get("GLM53_PMK_W7", "0") or "0").strip() == "1"


def g_mode(env=None) -> str:
    """GLM53_PMK_G: "off" | "v2a" (the served RS op after G) | "v2b" (cf3h-k2, coordinator 2026-10-26 13:01:18Z: P15 =
    the FFN partial's RS8 inside G, the served RS op retired for those layers; w7k2 5,282 vs 7,855 us per DSA
    layer-chunk); "1" = v2a; unset = v2a when GLM53_PMK_W7=1"""
    env = os.environ if env is None else env
    v = (env.get("GLM53_PMK_G", None) or "").strip().lower()
    if v == "":
        v = "v2a" if (env.get("GLM53_PMK_W7", "0") or "0").strip() == "1" else "off"
    return {"0": "off", "1": "v2a", "off": "off", "v2a": "v2a", "v2b": "v2b"}.get(v, "off")


def g_from_env(env=None) -> bool:
    """GLM53_PMK_G=1 (cf3h-k; default = GLM53_PMK_W7): every sparse_dsa layer chunk of a PMK bucket runs its MoE half as
    Stage 16's G v2a program (P10 mhc post attn + FFN pre, P11 AG, P12 router, P13 shared expert, the PmkGang EP experts)
    built with reuse = the layer's mhc-only program A (posts[0] / combs[0] / ys[0] / hns shared: the G v2a NaN contract),
    then the served RS op and the op mhc post; A1 + the op MLA / DSA attention unchanged"""
    return g_mode(env) != "off"


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ScopedEnv:
    """the W7 / G construction knobs (14's review: PMK_* construction-scoped, never process env in the served process):
    GLM53_PMK_W7_ENV = "KEY=VALUE;KEY=VALUE" applied while a W7 / G program is constructed and its descriptors built,
    then restored (e.g. GLM53_PMK_B_SELFSTREAM / _BLOCKS / _FFN_FID / _VALID / _SNAKE / _PLAN_CORES, PMK_P12_RECTS,
    PMK_P15_STREAM, PMK_COMB_STRIDE, PMK_XEXT, GLM53_PMK_ITEM_CONV, GLM53_PMK_CB_END)"""

    # G v2a's own set (16's composition_handoff.md: PMK_P12_RECTS / PMK_P15_STREAM / PMK_COMB_STRIDE = 0 for G; the
    # W7 values on G put its mm_only CBs 4 KB over the served floor: w7g mock 10:47Z)
    # GLM53_PMK_A_LR_EXACT=0 in G's scope: G has no low-rank / f_b / g_b streams; under the served LR_EXACT=1 its P13 down
    # moved to its own bf16 PARK (16's 84c595eb7f5 / 5ec7b192ef7 for W7) and G's mm_only CBs ended 4 KB over the floor
    # (w7g mock 10:54Z, 978.5 KB); 16's G v2a gate (w7lv5) ran LR_EXACT 0
    G_DEFAULT = "PMK_XEXT=1;PMK_P12_RECTS=0;PMK_P15_STREAM=0;PMK_COMB_STRIDE=0;GLM53_PMK_A_LR_EXACT=0"

    def __init__(self, extra: dict | None = None, var: str = "GLM53_PMK_W7_ENV"):
        kv = {}
        src = os.environ.get(var, None)
        if src is None and var == "GLM53_PMK_G_ENV":  # G: the handoff's set + the W7 set's GLM53_PMK_CB_END / _ROW_MASK
            # (the in-program valid-row mask, coordinator 12:10:42Z: W7's B and G both read P12's routes)
            cb_ = [
                it
                for it in (os.environ.get("GLM53_PMK_W7_ENV", "") or "").split(";")
                if it.startswith(("GLM53_PMK_CB_END=", "GLM53_PMK_ROW_MASK="))
            ]
            src = ";".join([self.G_DEFAULT] + cb_)
        for item in (src or "").split(";"):
            if "=" in item:
                k_, v_ = item.split("=", 1)
                if k_.strip():
                    kv[k_.strip()] = v_.strip()
        kv.update(extra or {})
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k_, v_ in self.kv.items():
            self.old[k_] = os.environ.get(k_)
            os.environ[k_] = v_
        return self

    def __exit__(self, *exc):
        for k_, v_ in self.old.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
        return False


def w7_place(mesh):
    """program A's core placement exactly as PmkSparseKdaLayer builds it (pmk_layer: the grid + the device's
    bank-adjacent DS streamers to avoid) -- W7's program B takes its cores from it before program A exists
    (pmk_fuse3_unit reads it off a control program A; ServedLayer._build_w7 asserts the two agree)"""
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMKL  # noqa: PLC0415

    grid = mesh.compute_with_storage_grid_size()
    avoid = ()
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import reader_assignments  # noqa: PLC0415

        avoid = tuple((int(cc.x), int(cc.y)) for cc, _, _ in reader_assignments(mesh, 3, "MESH_ROW_PLUS_X"))
    except Exception:  # noqa: BLE001 -- no device query: the plan's own placement (as PmkSparseKdaLayer)
        pass
    return PMKL.Placement(gx=int(grid.x), gy=int(grid.y), avoid=avoid)


class ServedLayer:
    """one layer of the served stack at one bucket: the five slots on the pieces' programs or the op graph, chained
    on the persistent residual as explicit STEPS (`steps(ci)`: the unit captures each alone for its device span);
    `prog(res, ci)` runs them in order (the hook's contract). `reuse` (Stage 12 L3): the first layer of the kind
    owns the transient hand-off set; later layers share it (program A: pmk_assembly's `reuse`; program B: the
    scratch dict), so the per-layer DRAM is the weights only."""

    impl = "pmk:served-chain(A1..A5 + B + op fillers)"

    def __init__(self, stack: "PmkStack", li: int, c: int, sp: bool, modes: dict, reuse: "ServedLayer | None" = None, *,
                 res=None, ci=None, build_b: bool = True):  # fmt: skip
        """res / ci / build_b (GroupStack, design (b)): a chunk instance on ITS OWN persistent residual and ChunkInputs
        (program A bakes both addresses), without a program B of its own (the group runs the FFN slot)"""
        pf = stack.pf
        self.stack, self.pf, self.li, self.c, self.sp = stack, pf, int(li), int(c), bool(sp)
        self.pl = pl = pf.layers[li]
        self.L, self.kind = pl.L, pl.kind
        self.res = stack.res[c] if res is None else res
        self.ci_own = ci
        self.tp = int(pf.ndev)
        plan = served_layer_plan(self.kind, self.tp, modes, xsl=getattr(pl, "xsl", None) is not None)
        self.modes, self.pending, self.notes = plan["modes"], plan["pending"], plan.get("notes", {})
        # GLM53_PMK_W7 (cf3h-k): a sparse_kda layer whose attention half is on the PMK runs the whole layer chunk but the
        # FFN mhc post as ONE fused program (the "ffn" slot = W7's P11 .. P15); GroupStack chunk instances never do
        self.w7 = bool(w7_from_env() and self.kind == SPARSE_KDA and res is None and ci is None and build_b
                       and self.tp in (2, 8) and all(self.modes[s_] == "pmk" for s_ in ("mhc_pre", "attn", "mhc_mid")))  # fmt: skip
        if self.w7:
            self.modes = dict(self.modes, ffn="w7")
            self.pending.pop("ffn", None)
        self.w7_in = None
        self.a = self.b = None
        self.engine = None
        self.counters = dict(calls=0, launches=0, fillers=0, eager=0)
        self.router_in = str(stack.opts.get("router_in", "fp32"))
        assert self.router_in in ("fp32", "bf16"), self.router_in
        self.build_s = {}
        self.b_levers = None
        self.b_kw = {}
        self.reuse = reuse
        self.t: dict = {}  # the chain's current tensors (h0, post0, comb0, y0, h1, h1_in, post1, comb1, mb, out)
        self._own: set = set()  # the op-owned entries of t (freed at the end of the layer)
        self.spans_mode = False  # the unit's per-step captures: never free the previous chain tensors
        self._io = self._descs = None
        self.taps = {}  # opts["taps"]: persistent fp32 copies of the attention output (y0) and the FFN output (mb)
        taps = stack.opts.get("taps")  # True = every layer; a list = those layers only (each tap set costs DRAM)
        if taps and (not isinstance(taps, (list, tuple, set)) or self.L in {int(v) for v in taps}):
            import ttnn  # noqa: PLC0415

            sp_rows = int(self.res.shape[-2])
            mk = lambda shape: ttnn.from_torch(torch.zeros(*shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=pf.mesh,  # noqa: E731
                                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=getattr(pf, "repl", None))  # fmt: skip
            for nm in ("y0", "mb", "h1"):  # the attention output, the FFN output, the FFN input (fp32 x-hat_f)
                self.taps[nm] = mk((1, 1, sp_rows, HIDDEN))
            self.taps["ffn_pre"] = mk(
                (1, 1, c, HIDDEN)
            )  # the FFN output BEFORE the RS, per chip (program B's y / the op partial)
            if self.kind in (
                SPARSE_KDA,
                SPARSE_DSA,
            ):  # the routing per token (gathered rows: every chip holds all C): top-8 ids, selected scores
                topk = int(getattr(pl.moe, "top_k", 8))
                for nm in ("ids", "sel"):
                    self.taps[nm] = mk((1, 1, c, topk))
        self.swap_in: dict = (
            {}
        )  # the unit's tap-swap: {"h1_in": a persistent fp32 x-hat_f} replaces the ffn step's input
        # the unit's ablation inside program B's slot: ("slot", mask [1, 1, C, 8] fp32 with one weight column zeroed) ->
        # the combine without that expert slot; ("shared", zeros [1, 1, C, H] bf16) -> without the shared term
        self.b_ablate: tuple | None = None
        self.dram_mb = dict(_t0=_dram_alloc_mb(pf.mesh))  # the DRAM (MB per bank) program A / program B retain at build
        self.b_in: dict = (
            {}
        )  # persistent program-B inputs (x_ext / ids_rm / wt) when opts["b_persistent_inputs"] (default on)
        if self.w7:
            with _ScopedEnv():
                self._build_w7(pf.chunk_inputs(c) if ci is None else ci)
        elif any(self.modes[s] == "pmk" for s in ("mhc_pre", "attn", "mhc_mid")):
            import time  # noqa: PLC0415

            from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: PLC0415

            t0 = time.time()
            ci = pf.chunk_inputs(c) if ci is None else ci
            ra = reuse.a if (reuse is not None and reuse.a is not None) else None
            self.a = ASM.PmkProgramA(pf, li, c, sp, self.res, getattr(pl, "scratch", None), ci.valid_end, probe=stack.probe,
                                     mhc_probe=stack.opts.get("mhc_probe", False), pfs_park=int(stack.opts.get("pfs_park", 4)),
                                     reuse=ra, attn=self.kind in PMK_KINDS)  # fmt: skip
            self.engine = self.a.engine
            self.build_s["A"] = round(time.time() - t0, 1)
        self.dram_mb["A"] = _dram_alloc_mb(pf.mesh) - self.dram_mb.pop("_t0", 0.0)
        d0 = _dram_alloc_mb(pf.mesh)
        if self.modes["ffn"] == "pmk" and build_b:
            # program B's scratch (out_rm / y / y_s: ~200 MB per chip at TP8) is shared with the FIRST layer that HAS a
            # program B (key (c, "B")), not the kind group's owner: the group "kda" is owned by L0, a dense_kda layer
            # without B, so every sparse_kda layer used to allocate its own set (14b 45-layer mock r1: +29 MB/bank per
            # sparse layer -> DRAM OOM at L24)
            owner_b = stack.shared.get((c, "B"))
            src = owner_b if owner_b is not None else (reuse if (reuse is not None and reuse.b is not None) else None)
            self.b = self._build_b(src.b if src is not None else None)
            stack.shared.setdefault((c, "B"), self)
        self.dram_mb["B"] = round(_dram_alloc_mb(pf.mesh) - d0, 2)
        self.dram_mb["A"] = round(self.dram_mb["A"], 2)
        # GLM53_PMK_G (cf3h-k): the sparse_dsa MoE half on G v2a (reuse = this layer's mhc-only program A)
        self.g = self.gk = None
        self.g_p15 = False
        self._g_io = self._g_descs = None
        if (
            g_from_env()
            and self.kind == SPARSE_DSA
            and self.a is not None
            and res is None
            and self.ci_own is None
            and build_b
            and self.modes["mhc_pre"] == "pmk"
            and self.modes["mhc_mid"] == "pmk"
            and self.modes["attn"] == "op"
        ):
            d1 = _dram_alloc_mb(pf.mesh)
            with _ScopedEnv(var="GLM53_PMK_G_ENV"):
                self._build_g(pf.chunk_inputs(c))
            self.modes = dict(self.modes, ffn="g")
            self.pending.pop("ffn", None)
            self.dram_mb["G"] = round(_dram_alloc_mb(pf.mesh) - d1, 2)
        self.pz = self._poison_setup() if (self.w7 or self.g is not None) else []

    # ------------------------------------------------------------------ W7 (cf3h-k)
    def _build_w7(self, ci):
        """GLM53_PMK_W7: this sparse_kda layer's W7 program = PmkProgramA(a15_fuse, b_p11 / b_p12 / b_p13 / b_p15, b_window
        = its program B), built as Stage 16's pmk_fuse3_unit variant a15b15 (w7lv4 / w7lv5, TP8 bitwise det): B in the
        self-stream form on the root / MM-only / member cores (GLM53_PMK_B_SELFSTREAM=1), its plan EARLY on the idle
        MM-only cores (GLM53_PMK_W7_PLAN_EARLY, default 1) in GLM53_PMK_W7_PLAN_RUNS_NB expert batches (default 4), the
        router tail's uint16 ids / bf16 weights (ids_u16). The FIRST W7 layer of the bucket owns the shared set (program
        A's transient hand-offs + collectives + P15's RS8 via `reuse`, B's sync shard and scratch, the B window inputs);
        later W7 layers carry their weights only."""
        import time  # noqa: PLC0415

        import ttnn  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, EpMoe  # noqa: PLC0415

        t0 = time.time()
        pf, pl, mesh, c, stack = self.pf, self.pl, self.pf.mesh, self.c, self.stack
        # (14's review C2) W7's KDA scan writes the chunk-END state only (P6, as A15): KDA checkpoints come from the
        # exported state at chunk ends (GLM53_KDA_CKPT_IN_REPLAY=0, GLM53_PF_KDA_SNAPSHOT=0: the served cf3h-j3cngx env);
        # an in-replay snapshot at a cut inside the chunk has no W7 form
        for k_ in ("GLM53_PF_KDA_SNAPSHOT", "GLM53_KDA_CKPT_IN_REPLAY"):
            assert os.environ.get(k_, "0") in (
                "",
                "0",
            ), f"GLM53_PMK_W7: {k_}={os.environ.get(k_)!r} (no in-chunk KDA snapshot in W7)"
        owner = stack.shared.get((c, "W7"))
        ep = getattr(pl, "epmoe", None)
        if ep is None:
            ep = pl.epmoe = EpMoe(pf, pl.moe, xsl=pl.xsl)
        cfg = b_config(stack.opts, tp=self.tp)
        if not int(cfg.get("out_bfp8", 0)):
            cfg.update(yw_db=0, yw_coal=1, comb_split=0, yw_brisc=0)
        cfg["ids_u16"] = 1  # W7 step 3: B reads the router tail's [C, 8] uint16 ids / bf16 weights
        G = int(cfg["g"])
        place = w7_place(mesh)
        rc = {k_: [tuple(c_) for c_ in v_] for k_, v_ in place.role_cores().items()}
        members = list(rc["member"])
        a_streamers = [
            tuple(c_) for c_ in ASM.adj_streamers(mesh, members, place.streamers, rc["root"] + rc["mm_only"])
        ]
        strm = M2.adj_pmk_streamers(mesh, members=members, avoid=rc["lane"] + rc["tail"]) if self.tp == 8 else None
        b_kw = {}
        plan_early = int(os.environ.get("GLM53_PMK_W7_PLAN_EARLY", "1") or 0)
        plan_nb = int(
            os.environ.get("GLM53_PMK_W7_PLAN_RUNS_NB", "4") or 1
        )  # 4 (coordinator 10:30:20Z, B1: root / mm_only
        # under the served floor GLM53_PMK_CB_END=1289856; bitwise by construction)
        if int(cfg.get("selfstream", 0)):  # B7g: 14's self-stream form on the root / MM-only / member cores
            b_kw = dict(selfstream=True, cores=[c_ for r_ in ("root", "mm_only", "member") for c_ in rc[r_]])
            if int(cfg.get("plan_cores", 0) or 0):
                b_kw["plan_cores"] = int(cfg["plan_cores"])
            if cfg.get("ffn_fid"):
                b_kw["ffn_fid"] = str(cfg["ffn_fid"])
            if plan_early or int(cfg.get("plan_cores", 0) or 0):  # the plan cores first in B's core list
                idle = [c_ for c_ in rc["mm_only"] if c_ not in set(a_streamers)]
                n_pc = len(idle) if plan_early else int(cfg["plan_cores"])
                pool = idle + [c_ for c_ in rc["mm_only"] + rc["root"] if c_ not in idle]
                first = pool[:n_pc]
                b_kw["cores"] = first + [c_ for c_ in b_kw["cores"] if c_ not in first]
                b_kw["plan_cores"] = n_pc
                b_kw["plan_runs_nb"] = plan_nb
                if plan_early:  # ids 2 / 3 (pmk_fuse3_unit): never touched on MM-only by A / P12 / P13 / P15
                    b_kw["plan_early"] = dict(sig=2, sig_n=len(members), sem_a=3, sem_b=2)
            if not os.environ.get("GLM53_PMK_B_BLOCKS") and stack.opts.get("b_blocks") is None:
                cfg["blocks"] = (4, 8, 8)  # the 7g geometry
        lv = M2.Levers(x_l1=bool(cfg["x_l1"]), mcast=G, shared=True, ids_u16=True, out_bfp8=bool(cfg.get("out_bfp8", 0)),
                       yw_db=bool(cfg["yw_db"]), yw_coal=int(cfg["yw_coal"]), ring_slots=int(cfg["ring_slots"]),
                       gu_split=bool(cfg["gu_split"]), comb_split=bool(cfg["comb_split"]), yw_brisc=bool(cfg["yw_brisc"]),
                       **({k_: bool(int(cfg.get(k_, 0))) for k_ in ("valid", "snake", "xs_rows") if k_ in M2.Levers.__dataclass_fields__}))  # fmt: skip
        W = int(ep.down[0].shape[-2])
        lv = M2.fit_levers(M2.Geometry(c=c, w=W, p=110, h_split=int(cfg["h_split"])), lv)
        b = M2.ProgramB(mesh, g_size=G, levers=lv, streamers=strm, num_experts=ep.e_all, top_k=int(pl.moe.top_k),
                        hidden=int(pl.moe.hidden), limit=float(pl.moe.limit), gate_up=ep.gate_up, down=ep.down,
                        blocks=cfg["blocks"], fid=str(cfg["fid"]), probe=False, prefetch=bool(cfg["prefetch"]),
                        full_sync=bool(cfg["full_sync"]), rect_mode=str(cfg["rect"]), gmax=int(cfg["gmax"]),
                        mcast_linked=bool(cfg["mcast_linked"]), mcast_unicast=bool(cfg["mcast_unicast"]),
                        mcast_dual=bool(cfg["mcast_dual"]), lpt=int(cfg["lpt"]), gu_split=bool(cfg["gu_split"]),
                        h_split=int(cfg["h_split"]), inplace=bool(cfg["inplace"]), lpt_w=None,
                        comb_split=bool(cfg["comb_split"]), yw_brisc=bool(cfg["yw_brisc"]), compute_only=members, **b_kw)  # fmt: skip
        assert getattr(b, "inplace", False), "W7: program B reads pl.xsl in place (GLM53_PMK_B_INPLACE=1)"
        ob = owner.b if owner is not None else None
        if (
            ob is not None
            and getattr(b, "sync", None) is not None
            and getattr(ob, "sync", None) is not None
            and tuple(b.sync.shape) == tuple(ob.sync.shape)
        ):
            ttnn.deallocate(
                b.sync
            )  # the streamers' L1 sync shard: one per bucket (a per-layer shard lowers the L1 floor)
            b.sync = ob.sync
        if ob is not None:  # B's scratch (out_rm / y / y_s): one set per bucket
            key = ("scratch", c)
            assert key in ob.bufs, "W7: the owner's program B has no scratch"
            b.bufs[key] = ob.bufs[key]
        bs = b.scratch(c)
        K_, H_ = int(pl.moe.top_k), int(pl.moe.hidden)
        if owner is not None:
            BIN = owner.w7_in
        else:  # the B window's inputs, written in-program by P11 / P12 every launch (pmk_fuse3_unit BIN)
            dram = ttnn.DRAM_MEMORY_CONFIG
            BIN = dict(hb=ttnn.empty((1, 1, c, H_), BF16, ttnn.TILE_LAYOUT, mesh, dram),
                       x_ext=ttnn.empty((c + 32, H_), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
                       ids_rm=ttnn.empty((1, c * K_), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram),
                       wt=ttnn.empty((1, c * K_), BF16, ttnn.ROW_MAJOR_LAYOUT, mesh, dram))  # fmt: skip
        self.w7_in = BIN
        stack_ = self.stack
        self.a = ASM.PmkProgramA(pf, self.li, c, self.sp, self.res, getattr(pl, "scratch", None), ci.valid_end,
                                 probe=stack_.probe, mhc_probe=stack_.opts.get("mhc_probe", False),
                                 pfs_park=int(stack_.opts.get("pfs_park", 4)), reuse=self._w7_reuse(owner),
                                 moe_share=self._w7_moe_share(owner),
                                 attn=True, a15_fuse=True, b_p11=True, b_p12=True, b_p13=True, b_p15=True,
                                 b_window=dict(b=b, x_ext=BIN["x_ext"], ids_rm=BIN["ids_rm"], wt=BIN["wt"], x_tile=BIN["hb"],
                                               y_s=bs.get("y_s")))  # fmt: skip
        assert [tuple(s_) for s_ in self.a.scopes] == [tuple(ASM.A15B15_SCOPE)], (
            "W7: one A15B15 program",
            self.a.scopes,
        )
        assert list(map(tuple, self.a.members)) == members and list(map(tuple, self.a.streamers)) == a_streamers, \
            "W7: program A's placement differs from w7_place (B's cores were taken from it)"  # fmt: skip
        assert (self.a.rs8_15 if self.tp == 8 else self.a.rsr15) is not None, "W7: P15's RS (the FFN partial's landing)"
        self.engine = self.a.engine
        self.b, self.b_cfg, self.b_levers, self.b_kw = b, cfg, lv, b_kw
        stack.shared.setdefault((c, "W7"), self)
        self.build_s["W7"] = round(time.time() - t0, 1)
        print(f"PMK served L{self.L}: W7 (A15B15) built in {self.build_s['W7']} s -- B {cfg} levers {lv}; plan "
              f"{'early' if plan_early else 'after'} on {b_kw.get('plan_cores')} cores, nb {b_kw.get('plan_runs_nb')}; "
              f"owner L{owner.L if owner is not None else self.L}", flush=True)  # fmt: skip

    def _build_g(self, ci):
        """GLM53_PMK_G: G v2a as pmk_gv2_unit builds it (PmkGang over PBW.w7_gang_sets, EP experts in place, fid
        GLM53_PMK_G_FID default LoFi, ys / wt_bf16 / ids_u16) on this layer's residual, with reuse = the FIRST G layer of
        the bucket (its sync shard / scratch / tables; that one reuses its own mhc-only program A), so P10 reads the
        posts[0] / combs[0] / ys[0] that A1 and the attention wrote this replay"""
        import time  # noqa: PLC0415

        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_b_window as PBW  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_gang as PG  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMKL  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import EpMoe  # noqa: PLC0415

        t0 = time.time()
        pf, pl, mesh, c, stack = self.pf, self.pl, self.pf.mesh, self.c, self.stack
        owner = stack.shared.get((c, "G"))
        ep = getattr(pl, "epmoe", None)
        if ep is None:
            ep = pl.epmoe = EpMoe(pf, pl.moe, xsl=getattr(pl, "xsl", None))
        assert (
            getattr(pl, "xsl", None) is None
        ), f"L{self.L}: G v2a needs the EP experts (a sparse_dsa layer without XSL)"
        moe = pl.moe
        gk = PG.PmkGang(mesh, e_loc=ep.e_loc, top_k=int(moe.top_k), hidden=int(moe.hidden), limit=float(moe.limit),
                        gate_up=ep.gate_up, down=ep.down, gangs=PBW.w7_gang_sets(PMKL.Placement()),
                        lo=[ep.e_loc * r for r in range(self.tp)], fid=os.environ.get("GLM53_PMK_G_FID", "LoFi") or "LoFi",
                        ys=True, wt_bf16=True, ids_u16=True)  # fmt: skip
        if owner is not None:  # G's scratch (out_rm / y / y_s): one set per bucket, as program B's
            key = ("scratch", c)
            owner.gk.scratch(c)
            gk.bufs[key] = owner.gk.bufs[key]
        old_ = os.environ.get("GLM53_PMK_ARGS")
        os.environ["GLM53_PMK_ARGS"] = "table"  # (pmk_gv2_unit run_g: P10 .. P13 + G exceed 256 member BRISC args)
        try:
            self.g_p15 = g_mode() == "v2b"  # G v2b: the RS8 of G's partial inside the program (P15)
            self.g = ASM.PmkProgramA(pf, self.li, c, self.sp, self.res, None, ci.valid_end, attn=False,
                                     g_window=dict(g=gk, p15=self.g_p15), probe=False,
                                     pfs_park=int(os.environ.get("GLM53_PMK_G_PFS_PARK", "3") or 3),
                                     reuse=(owner.g if owner is not None else self.a))  # fmt: skip
        finally:
            if old_ is None:
                os.environ.pop("GLM53_PMK_ARGS", None)
            else:
                os.environ["GLM53_PMK_ARGS"] = old_
        assert self.g.posts[0] is self.a.posts[0] and self.g.combs[0] is self.a.combs[0] and self.g.ys[0] is self.a.ys[0], \
            "G v2a: P10 must read the layer's A1 / attention outputs (the shared transient set)"  # fmt: skip
        assert len(self.g.scopes) == 1, ("G v2a: one program", self.g.scopes)
        if self.g_p15:
            assert (
                self.g.rs8_15 if self.tp == 8 else self.g.rsr15
            ) is not None, "G v2b: P15's RS (the FFN rows' landing)"
        self.gk = gk
        stack.shared.setdefault((c, "G"), self)
        self.build_s["G"] = round(time.time() - t0, 1)
        print(f"PMK served L{self.L}: G v2a built in {self.build_s['G']} s -- {len(gk.gangs)} gangs, CB {gk.l1_bytes(c)} B; "
              f"owner L{owner.L if owner is not None else self.L}", flush=True)  # fmt: skip

    def _poison_setup(self):
        """GLM53_PMK_W7_POISON=1 (the 01:56:05Z rule of record, the cf3h-k stack unit): before EVERY W7 / G launch, inside
        the trace, the fused program's intermediate outputs are overwritten with NaN (device copies from NaN tensors
        allocated here, once per (shape, dtype, layout, memory config), before any capture) -- W7: B's y, P13's y_s, P15's
        y15, RS8 stage 1's r1 (P8 / P15); G: G's y / out_rm, P13's y_s. A phase that reads a row before its producer wrote
        it this replay reads NaN (the output goes non-finite) instead of the previous replay's identical value"""
        import torch as _t  # noqa: PLC0415

        import ttnn  # noqa: PLC0415

        if os.environ.get("GLM53_PMK_W7_POISON", "0") != "1":
            return []
        tg = []
        if self.w7:
            a = self.a
            tg += [("b_y", self.b.scratch(self.c)["y"]), ("y_s", a.sh_dn.y), ("y15", self.w7_y())]
            for k_, rs_ in (("r1_8", getattr(a, "rs8", None)), ("r1_15", getattr(a, "rs8_15", None))):
                if rs_ is not None:
                    tg.append((k_, rs_.rs1.y))
        if getattr(self, "g", None) is not None:
            sc_ = self.gk.scratch(self.c)
            tg += [("g_y", sc_["y"]), ("g_out_rm", sc_["out_rm"]), ("g_y_s", self.g.sh_dn.y)]
            if getattr(self, "g_p15", False):  # G v2b: P15's landing + RS8 stage 1's r1
                tg.append(("g_y15", self.g_y15()))
                if getattr(self.g, "rs8_15", None) is not None:
                    tg.append(("g_r1_15", self.g.rs8_15.rs1.y))
        pz = self.stack.__dict__.setdefault("_poison_src", {})
        out = []
        for nm_, t_ in tg:
            key = (tuple(t_.shape), str(t_.dtype), str(t_.layout), str(t_.memory_config()))
            if key not in pz:
                pz[key] = ttnn.from_torch(_t.full(tuple(t_.shape), float("nan")), dtype=t_.dtype, layout=t_.layout,
                                          device=self.pf.mesh, memory_config=t_.memory_config(),
                                          mesh_mapper=ttnn.ReplicateTensorToMesh(self.pf.mesh) if self.tp > 1 else None)  # fmt: skip
            out.append((nm_, pz[key], t_))
        return out

    def _poison(self):
        import ttnn  # noqa: PLC0415

        for _nm, src_, dst_ in self.pz:
            ttnn.copy(src_, dst_)

    def _w7_reuse(self, owner):
        """W7's program A reuse: the bucket's first W7 layer takes the KDA group owner's transient set (the dense-KDA A15:
        hns / posts / combs, C1X, the P8 RS8, the stream outputs, the KDA scratch -- the same geometry; its sync shard is
        shared only when it has W7's 8 slots), later W7 layers the first W7's (+ P15's RS8, B / MoE scratch). The w7m
        mock (11:29Z): the W7 group's own duplicate of that set was 21.1 MiB/bank of the W7 owner's 62.6, plus its C1X /
        RS8 global semaphores in L1. GLM53_PMK_W7_REUSE_KDA=0: the W7 group owns its set (the A/B)"""
        if owner is not None:
            return owner.a
        if os.environ.get("GLM53_PMK_W7_REUSE_KDA", "1") != "1":
            return None
        kda_ = self.stack.shared.get((self.c, "kda"))
        return kda_.a if (kda_ is not None and kda_ is not self and getattr(kda_, "a", None) is not None) else None

    def g_y15(self):
        """G v2b: P15's reduced FFN rows (fp32, this chip's owned rows) = what the served RS op returned for v2a"""
        return self.g.rs8_15.y if self.g.rs8_15 is not None else self.g.rsr15.y

    def _w7_moe_share(self, owner):
        """(cf3h-k2 v2, coordinator 16:02:20Z: the capture-phase DRAM low) the bucket's first W7 layer takes the G owner's
        MoE-half per-launch transients (P11x xext, P12 logits / ids / weights, P13 outputs: the same shapes, written
        before read in every launch, W7 and G never live together) instead of a second set (~6.6 MiB/bank).
        GLM53_PMK_W7_SHARE_G=0: the W7 group owns its set (the A/B)"""
        if owner is not None or os.environ.get("GLM53_PMK_W7_SHARE_G", "1") != "1":
            return None
        g_ = self.stack.shared.get((self.c, "G"))
        return getattr(g_, "g", None) if g_ is not None else None

    def w7_y(self):
        """W7: P15's reduced FFN rows (fp32, this chip's owned rows) = the op path's pf.reduce(ffn) for the mhc post"""
        return self.a.rs8_15.y if self.a.rs8_15 is not None else self.a.rsr15.y

    # ------------------------------------------------------------------ program B
    def _build_b(self, reuse_b=None, rows: int | None = None):
        """rows (GroupStack): program B at `rows` rows (n x c) for the group; the layer's own b_* fields are untouched"""
        import inspect  # noqa: PLC0415
        import time  # noqa: PLC0415

        own = rows is None
        c_b = self.c if own else int(rows)

        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import EpMoe  # noqa: PLC0415

        t0 = time.time()
        pf, pl, mesh = self.pf, self.pl, self.pf.mesh
        ep = getattr(pl, "epmoe", None)
        if ep is None:
            ep = pl.epmoe = EpMoe(pf, pl.moe, xsl=pl.xsl)
        W = int(ep.down[0].shape[-2])
        cfg = b_config(self.stack.opts, tp=self.tp)
        if own:
            self.b_cfg = cfg
        G = int(cfg["g"])
        ob = bool(cfg.get("out_bfp8", 0))  # bf16-row partials unless GLM53_PMK_B_OUT_BFP8=1 (the record's comment)
        if not ob:
            cfg.update(yw_db=0, yw_coal=1, comb_split=0, yw_brisc=0)
        lv = M2.Levers(x_l1=bool(cfg["x_l1"]), mcast=G, shared=True, ids_u16=bool(cfg["ids_u16"]), out_bfp8=ob,
                       yw_db=bool(cfg["yw_db"]), yw_coal=int(cfg["yw_coal"]), ring_slots=int(cfg["ring_slots"]),
                       gu_split=bool(cfg["gu_split"]), comb_split=bool(cfg["comb_split"]), yw_brisc=bool(cfg["yw_brisc"]),
                       **({k_: bool(cfg[k_]) for k_ in ("valid", "snake", "xs_rows") if k_ in M2.Levers.__dataclass_fields__}))  # fmt: skip
        lv = M2.fit_levers(M2.Geometry(c=c_b, w=W, p=110, h_split=int(cfg["h_split"])), lv)
        if own:
            self.b_levers = lv
        lpt_w = M2.lpt_weights_from(cfg["lpt_w"]) if hasattr(M2, "lpt_weights_from") else None
        kw = dict(g_size=G, levers=lv, num_experts=ep.e_all, top_k=int(pl.moe.top_k), hidden=int(pl.moe.hidden),
                  limit=float(pl.moe.limit), gate_up=ep.gate_up, down=ep.down, blocks=cfg["blocks"], fid=str(cfg["fid"]),
                  probe=False, prefetch=bool(cfg["prefetch"]), full_sync=bool(cfg["full_sync"]), rect_mode=str(cfg["rect"]),
                  gmax=int(cfg["gmax"]), mcast_linked=bool(cfg["mcast_linked"]), mcast_unicast=bool(cfg["mcast_unicast"]),
                  mcast_dual=bool(cfg["mcast_dual"]), lpt=int(cfg["lpt"]), gu_split=bool(cfg["gu_split"]), h_split=int(cfg["h_split"]),
                  inplace=bool(cfg["inplace"]), lpt_w=lpt_w, comb_split=bool(cfg["comb_split"]), yw_brisc=bool(cfg["yw_brisc"]),
                  **({"pack_rne": True} if int(cfg.get("pack_rne", 0)) else {}),
                  **({"ffn_fid": str(cfg["ffn_fid"])} if cfg.get("ffn_fid") else {}),
                  **({"plan_cores": int(cfg["plan_cores"])} if int(cfg.get("plan_cores", 0)) else {}),
                  **({"selfstream": True} if int(cfg.get("selfstream", 0)) else {}))  # fmt: skip
        sig = inspect.signature(M2.ProgramB.__init__).parameters
        var_kw = any(p_.kind == inspect.Parameter.VAR_KEYWORD for p_ in sig.values())
        base = set()  # the **kw chain: every base __init__ of ProgramB (PmkMoe2 -> pmk_moe.PmkMoe)
        for cls in M2.ProgramB.__mro__[1:]:
            if cls.__init__ is not object.__init__:
                base |= set(inspect.signature(cls.__init__).parameters)
        known = set(sig) | (base if var_kw else set())
        dropped = sorted(k for k in kw if k not in known)
        if "selfstream" in dropped:
            print(
                f"PMK served L{self.L}: GLM53_PMK_B_SELFSTREAM requested but this tree's ProgramB has no selfstream (moe2 cc9d5fffd4 "
                f"not on the tree): program B runs in the adj-streamer form",
                flush=True,
            )
        if "pack_rne" in dropped:
            print(
                f"PMK served L{self.L}: GLM53_PMK_B_PACK_RNE requested but this tree's ProgramB has no pack_rne (the moe2 SFPU "
                f"pre-round pass is not on the tree): program B runs with the truncating pack",
                flush=True,
            )
        b_kw = {k: v for k, v in kw.items() if k in known}
        if own:
            self.b_kw = b_kw
        b = M2.ProgramB(mesh, **b_kw)
        inplace = bool(getattr(b, "inplace", False))
        if inplace:
            pages, check = 0, dict(
                inplace=True, chips=int(mesh.get_num_devices()), note="pl.xsl read in place (device k = slice k)"
            )
        else:
            pages, check = load_streams_per_chip(b, ep.gate_up, ep.down)
        if own:
            self.b_pages, self.b_stream_check = pages, check
        if (
            reuse_b is not None
            and getattr(b, "sync", None) is not None
            and getattr(reuse_b, "sync", None) is not None
            and tuple(b.sync.shape) == tuple(reuse_b.sync.shape)
            and list(getattr(b, "streamers", [])) == list(getattr(reuse_b, "streamers", []))
        ):
            # the streamers' L1 sync shard shared too (read at parts() time): per layer it pushed the L1 floor down
            import ttnn  # noqa: PLC0415

            ttnn.deallocate(b.sync)
            b.sync = reuse_b.sync
        if reuse_b is not None:  # the scratch (out_rm / y / y_s) shared with the reused layer's program B
            key = ("scratch", c_b)
            assert key in reuse_b.bufs, "reuse: the reused program B has no scratch yet"
            b.bufs[key] = reuse_b.bufs[key]
        b.scratch(c_b)  # the persistent out_rm / y / y_s (one set per kind)
        print(f"PMK served L{self.L}: program B ({c_b} rows) of record {cfg} -> levers {lv}; kwargs {sorted(b_kw)} (dropped by the "
              f"tree's signature: {dropped}); blocks {b.blocks(c_b)}; streams {check}", flush=True)  # fmt: skip
        if own:
            self.b_dropped = dropped
            self.build_s["B"] = round(time.time() - t0, 1)
        else:
            self.build_s[f"B{c_b}"] = round(time.time() - t0, 1)
        return b

    def _ffn_b_steps(self) -> list:
        """the FFN slot on program B as SUB-STEPS (each a capturable span: the fillers around program B are what the
        slot costs over B standalone -- d0 card0 stk14u: 18,946 vs 13,671 us at L4): router (rms + the fp32 router
        matmul + sigmoid / top-8 + the selected scores, gathered), x gather (the bf16 x-hat_f all-gather = B's x_tile),
        B inputs (inputs_from_op_path: x_ext ROW_MAJOR pad / convert, ids transpose + convert, the weights' math),
        shared (swiglu_prefill -> y_s), B (one generic_op), rs (the op reduce-scatter to the owners' rows)"""
        import ttnn  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as PM  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32, swiglu_prefill  # noqa: PLC0415

        pf, pl, sp, c, T = self.pf, self.pl, self.sp, self.c, self.t

        def router():
            h1 = self._get("h1_in")
            if T.get("h1_in_normed"):
                pl._normed["ffn"] = h1
            x32 = pl.rms(h1, "ffn", FP32)
            idx, scores = pl.moe.router(x32, decode=False)
            self._set("x32", x32, x32 is not h1)
            # NO ttnn.slice in the chain (r14): the ids (fp32) and the selected scores gathered separately
            sel_l = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
            ids_l = ttnn.typecast(idx, FP32)
            for t_ in (scores, idx):
                ttnn.deallocate(t_)
            ids_f, sel = pf.gather(ids_l, sp), pf.gather(sel_l, sp)
            for t_ in (ids_l, sel_l):
                if t_ is not ids_f and t_ is not sel:
                    ttnn.deallocate(t_)
            if "ids" in self.taps:
                _copy_into(ids_f, self.taps["ids"])
                _copy_into(sel, self.taps["sel"])
            pin = getattr(pf, "_pad_now", None)
            rows = int(ids_f.shape[-2])
            if pin is not None and rows == pin["rows"]:  # GLM53_PF_MOE_PADMASK: rows >= n_valid -> the sentinel expert
                kept = ttnn.multiply(ids_f, pin["keep"])
                ttnn.deallocate(ids_f)
                ids_f = ttnn.add(kept, pin["pad_ids"])
                ttnn.deallocate(kept)
                sel_k = ttnn.multiply(sel, pin["keep"])
                ttnn.deallocate(sel)
                sel = sel_k
            idx16 = ttnn.typecast(ids_f, ttnn.uint16)
            ttnn.deallocate(ids_f)
            self._set("idx16", idx16, True)
            self._set("sel", sel, True)
            self.counters["fillers"] += 1

        def gather_x():
            x32 = self._get("x32")
            self._set("hb", pf.gather(ttnn.typecast(x32, BF16), sp), True)  # [1, 1, C, 4096] bf16 TILE = B's x_tile
            self.counters["fillers"] += 1

        def b_inputs():
            hb, idx16, sel = self._get("hb"), self._get("idx16"), self._get("sel")
            x_ext, ids_rm, wt = PM.inputs_from_op_path(hb, idx16, sel, float(pl.moe.scaling))
            if self.b_levers.ids_u16:
                ids_rm, wt = M2.tail_layout_inputs(idx16, sel, float(pl.moe.scaling))
            if (
                self.b_ablate is not None and self.b_ablate[0] == "slot"
            ):  # the unit's ablation: one expert slot's weight zeroed
                wt_m = ttnn.multiply(wt, self.b_ablate[1])
                ttnn.deallocate(wt)
                wt = wt_m
            if self.stack.opts.get(
                "b_persistent_inputs", False
            ):  # program B reads PERSISTENT inputs (the bench's form).
                # r18 on silicon: no gain (b_inputs 257 -> 351 us: the three copies cost more than the placement saved;
                # B's span unchanged) -> off by default; the +1.8 ms of B in the chain vs standalone is not the inputs'
                # placement (named, not closed: the chain's DRAM / NoC state from the preceding steps is the candidate)
                if not self.b_in:
                    for k_, t_ in (("x_ext", x_ext), ("ids_rm", ids_rm), ("wt", wt)):
                        self.b_in[k_] = ttnn.empty(
                            tuple(int(v) for v in t_.shape), t_.dtype, t_.layout, pf.mesh, ttnn.DRAM_MEMORY_CONFIG
                        )
                for k_, t_ in (("x_ext", x_ext), ("ids_rm", ids_rm), ("wt", wt)):
                    ttnn.copy(t_, self.b_in[k_])
                    ttnn.deallocate(t_)
                x_ext, ids_rm, wt = self.b_in["x_ext"], self.b_in["ids_rm"], self.b_in["wt"]
                self._set("x_ext", x_ext, False)
                self._set("ids_rm", ids_rm, False)
                self._set("wt", wt, False)
            else:
                self._set("x_ext", x_ext, True)
                self._set("ids_rm", ids_rm, True)
                self._set("wt", wt, True)
            self.counters["fillers"] += 1

        def shared():
            if self.b_ablate is not None and self.b_ablate[0] == "shared":  # the unit's ablation: no shared term
                _copy_into(self.b_ablate[1], self.b.scratch(c)["y_s"])
                return
            y_s = swiglu_prefill(pl.shared, self._get("hb"))  # the op P13: the (K + 1)-th combine term (bf16 TILE)
            dst = self.b.scratch(c)["y_s"]
            if (
                dst.layout != y_s.layout
            ):  # the self-stream form's bf16 combine reads ROW_MAJOR [C, H] rows (PERF.md sec 19)
                rows_ = ttnn.to_layout(
                    ttnn.reshape(y_s, tuple(int(v) for v in dst.shape)),
                    dst.layout,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(y_s)
                y_s = rows_
            _copy_into(y_s, dst)
            ttnn.deallocate(y_s)
            self.counters["fillers"] += 1

        def run_b():
            self.b(self._get("x_ext"), self._get("ids_rm"), self._get("wt"), c, x_tile=self._get("hb"))  # -> scratch y
            self.counters["launches"] += 1

        def rs():
            y = self.b.scratch(c)["y"]
            if self.taps:
                _copy_into(y, self.taps["ffn_pre"])  # program B's output per chip, before the RS
            self._set("mb", pf.reduce(y, sp), True)  # the op P15 (RS to the owners' rows)
            self.counters["fillers"] += 1

        return [
            ("ffn", "op:router", router),
            ("ffn", "op:gather_x", gather_x),
            ("ffn", "op:b_inputs", b_inputs),
            ("ffn", "op:shared", shared),
            ("ffn", "B", run_b),
            ("ffn", "op:rs", rs),
        ]

    # ------------------------------------------------------------------ the chain as steps
    def _launch(self, i: int):
        import ttnn  # noqa: PLC0415

        ttnn.generic_op(self._io, self._descs[i])
        self.counters["launches"] += 1

    def _launch_g(self):
        import ttnn  # noqa: PLC0415

        ttnn.generic_op(self._g_io, self._g_descs[0])
        self.counters["launches"] += 1

    def _get(self, name):
        """a chain tensor another step produced: asserted LIVE (allocated, on a mesh device) before any op reads it --
        d0 01:38Z: an all_gather on a chain tensor whose buffer had died (a per-slot capture's trace-owned output,
        released with its trace) segfaulted in MeshDevice::shape(); the mock's fake device tensors carry the same
        checks"""
        t = self.t.get(name)
        assert t is not None, f"chain tensor {name!r} missing (L{self.L}: the producing step did not run)"
        _assert_live(t, f"L{self.L} {name}")
        return t

    def snapshot(self) -> tuple:
        return dict(self.t), set(self._own)

    def restore(self, snap: tuple):
        """back to a snapshot WITHOUT freeing what came after (the per-slot capture's trace-owned outputs are dropped,
        never deallocated: their trace owns them)"""
        self.t, self._own = dict(snap[0]), set(snap[1])

    def _set(self, name, tensor, own: bool):
        """a chain tensor; an op-owned one from an earlier run of the same step is freed first (not in spans mode)"""
        import ttnn  # noqa: PLC0415

        old = self.t.get(name)
        if old is not None and name in self._own and old is not tensor and not self.spans_mode:
            ttnn.deallocate(old)
        self.t[name] = tensor
        if own:
            self._own.add(name)
        else:
            self._own.discard(name)

    def steps(self, ci) -> list:
        """[(slot, name, fn)] in chain order: fn() runs one piece program or one op filler on the chain's current
        tensors (self.t) and records its outputs there"""
        import ttnn  # noqa: PLC0415

        pf, pl, res, sp, m, a, T = self.pf, self.pl, self.res, self.sp, self.modes, self.a, self.t
        st = []
        # program A's launches, deduplicated ACROSS the slots (static, per steps() list): a fused scope (PMK_A15_FUSE: P1 ..
        # P10 in ONE program; A34 / A2F merges) owns several slots' phases, and each program must run ONCE per layer --
        # realgate5 (2026-10-24 22:22Z): slots 1 / 2 / 3 each launched the one A15 program, so it ran 3x in place on the
        # residual (L0 pcc 0.774)
        launched = []
        # ---- slot 1: mhc pre (attn) -> hn[0], posts[0], combs[0]
        if m["mhc_pre"] == "pmk":
            i1_ = a.index_of("P1")
            launched.append(i1_)
            pz1_ = self.w7  # W7: the poison copies go right before the one fused launch

            def s1():
                if pz1_:
                    self._poison()
                self._launch(i1_)
                self._set("h0", a.hns[0], False)
                self._set("post0", a.posts[0], False)
                self._set("comb0", a.combs[0], False)

            st.append(("mhc_pre", "A1", s1))
        else:

            def s1():
                h0, p0, c0 = pf._mhc_pre(pl, "attn", res)
                self._set("h0", h0, True)
                self._set("post0", p0, True)
                self._set("comb0", c0, True)
                # PrefillLayer.rms hands a pre-normalised h through only while pl._normed[sub] IS h, and resets it on
                # use: a ONE-SHOT handshake. The consuming step re-arms it itself (idempotent steps: the unit runs
                # every step twice, compile-style then inside its capture; d0 01:24Z: the second run fell into
                # ttnn.rms_norm, a program never run before -> "Cannot load new binaries during trace capture")
                T["h0_normed"] = pl._normed.get("attn") is h0
                self.counters["fillers"] += 1
                if a is not None and m["attn"] == "pmk":
                    _copy_into(h0, a.hns[0])  # A2's c1_stub_prod reads hn[0] by address
                if a is not None and m["mhc_mid"] == "pmk":
                    _copy_into(p0, a.posts[0])
                    _copy_into(c0, a.combs[0])

            st.append(("mhc_pre", "op:mhc_pre", s1))
        # ---- slot 2: attention -> y (fp32 [SP, 4096]; the mhc's ys[0] when A5 reads it)
        if m["attn"] == "pmk":
            for nm_, pid_ in (("A2", "P2"), ("A3", "P4"), ("A4", "P7")):  # by scope (A3 inside A2 under PMK_A3_MERGE)
                i_ = a.index_of(pid_)
                if i_ not in launched:
                    launched.append(i_)
                    st.append(("attn", nm_, (lambda i_=i_: self._launch(i_))))
            if (
                a.rsr is None and getattr(a, "rs8", None) is None
            ):  # no in-program RS (pair RSR / TP8 RS8) -> op RS of the o_proj partial into y

                def s2r():
                    y = pf.reduce(a.mm_o.y, sp)
                    _copy_into(y, a.ys[0])
                    ttnn.deallocate(y)
                    self.counters["fillers"] += 1

                st.append(("attn", "op:rs_o", s2r))
            st.append(("attn", "y", lambda: self._set("y0", a.ys[0], False)))
        else:

            def s2():
                h0 = self._get("h0")
                # _attn_sublayer FREES its input after the gather (the only consuming filler): program A's persistent
                # hn[0] and -- in spans mode, where every step runs twice (mock r10: "L3 h0 is not a live device
                # tensor" on the second run) -- the op-owned h0 go in as a copy; the plain chain hands its own h0 over
                if m["mhc_pre"] == "pmk" or self.spans_mode:
                    h0 = ttnn.clone(h0)
                    if m["mhc_pre"] == "pmk" or T.get("h0_normed"):
                        pl._normed["attn"] = h0
                else:
                    if T.get("h0_normed"):
                        pl._normed["attn"] = h0  # re-armed here: the step is idempotent
                    self._own.discard("h0")  # _attn_sublayer frees it
                y0 = pf._attn_sublayer(pl, h0, ci, self.li, sp)
                self.counters["fillers"] += 1
                if a is not None and m["mhc_mid"] == "pmk":
                    _copy_into(y0, a.ys[0])
                    ttnn.deallocate(y0)
                    self._set("y0", a.ys[0], False)
                else:
                    self._set("y0", y0, True)

            st.append(("attn", "op:attn", s2))
        # ---- slot 3: mhc post (attn) + pre (ffn) -> x1 (in place), hn[1], posts[1], combs[1]
        if self.g is not None:  # G v2a: P10 (mhc post attn + FFN pre) .. P14 (the EP experts) in ONE launch
            g_ = self.g

            def s3():
                self._poison()
                self._launch_g()
                self._set("h1", g_.hns[1], False)
                self._set("post1", g_.posts[1], False)
                self._set("comb1", g_.combs[1], False)

            st.append(("mhc_mid", "G", s3))
        elif m["mhc_mid"] == "pmk":
            i10_ = a.index_of("P10")
            run10_ = i10_ not in launched  # A5 inside an earlier slot's fused program (A15): its outputs are ready
            launched.append(i10_)

            def s3():
                if run10_:
                    self._launch(i10_)  # A5
                self._set("h1", a.hns[1], False)
                self._set("post1", a.posts[1], False)
                self._set("comb1", a.combs[1], False)

            st.append(("mhc_mid", "A5", s3))
        else:

            def s3():
                x1 = pf._post(self._get("y0"), res, self._get("post0"), self._get("comb0"))
                ttnn.copy(x1, res)
                ttnn.deallocate(x1)
                h1, p1, c1 = pf._mhc_pre(pl, "ffn", res)
                self._set("h1", h1, True)
                self._set("post1", p1, True)
                self._set("comb1", c1, True)
                T["h1_normed"] = pl._normed.get("ffn") is h1
                self.counters["fillers"] += 2

            st.append(("mhc_mid", "op:post+pre", s3))
        # ---- slot 4: ffn -> m (the reduced FFN output on the owner rows)
        # THE ROUTER'S INPUT (silicon 2026-10-23 00:04Z, pair unit L4 PCC 0.995 / rel 0.099): A5 stores x-hat_f as bf16
        # (hn[1]); the op path's fused7a routes on the fp32 x-hat (hn_dtype FP32 for MoE layers) and a bf16-rounded
        # router input flips one top-8 expert on ~2 % of the tokens (host: 43 of 2,048 on L4's rows). router_in
        # "fp32" (default): the FFN pre recomputed on the fp32 residual by the op's MhcPrePf (one filler; its post /
        # comb pages unused: A5's are the chain's) feeds the router AND the experts' gather, as the op path; "bf16":
        # hn[1] (the PMK-only path, PCC-explained)
        if m["ffn"] in ("w7", "g"):  # W7 / G: P11 .. (the router on its own x-hat_f gather) ran inside the fused launch

            def s4i():
                self._set("h1_in", self._get("h1"), False)

            st.append(("ffn", "w7:h1", s4i))
        elif (
            m["mhc_mid"] == "pmk" and self.kind in (SPARSE_KDA, SPARSE_DSA) and self.router_in == "fp32"
        ):  # the MoE kinds

            def s4i():
                h, p_, c_ = pf._mhc_pre(pl, "ffn", res)
                for t_ in (p_, c_):
                    ttnn.deallocate(t_)
                self._set("h1_in", h, True)
                T["h1_in_normed"] = pl._normed.get("ffn") is h
                self.counters["fillers"] += 1
                self._swap_and_tap_h1()

            st.append(("ffn", "op:router_in_fp32", s4i))
        else:

            def s4i():
                self._set("h1_in", self._get("h1"), False)
                T["h1_in_normed"] = True if m["mhc_mid"] == "pmk" else bool(T.get("h1_normed"))  # A5's hn[1] is x-hat_f
                self._swap_and_tap_h1()

            st.append(("ffn", "h1", s4i))

        def arm_ffn():  # the one-shot rms handshake, re-armed by the consuming step (idempotent)
            if T.get("h1_in_normed"):
                pl._normed["ffn"] = self._get("h1_in")

        def tap_routing_op():  # the op ffn's routing, tapped by the SAME router call on the same input (deterministic)
            if "ids" not in self.taps:
                return
            from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import FP32  # noqa: PLC0415

            arm_ffn()
            x32 = pl.rms(self._get("h1_in"), "ffn", FP32)
            idx, scores = pl.moe.router(x32, decode=False)
            sel_l = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
            ids_l = ttnn.typecast(idx, FP32)
            ids_g, sel_g = pf.gather(ids_l, sp), pf.gather(sel_l, sp)
            _copy_into(ids_g, self.taps["ids"])
            _copy_into(sel_g, self.taps["sel"])
            for t_ in (x32, idx, scores, sel_l, ids_l, ids_g, sel_g):
                if t_ is not self._get("h1_in"):
                    ttnn.deallocate(t_)

        if m["ffn"] == "pmk":
            st.extend(self._ffn_b_steps())  # router / gather_x / b_inputs / shared / B / rs as separate spans
        elif m["ffn"] == "w7":
            st.append(("ffn", "w7:y15", lambda: self._set("mb", self.w7_y(), False)))  # P15's reduced FFN rows (fp32)
        elif m["ffn"] == "g" and getattr(self, "g_p15", False):  # G v2b: P15's reduced rows (fp32, the owned rows)
            # (K2-M3, coordinator 13:25:28Z) P15's landing holds a sequence-parallel bucket's OWNED rows -- what
            # pf.reduce(y, sp) returns only when sp; a replicated bucket has no G v2b form
            assert sp, f"L{self.L}: G v2b (P15 in G) needs the sequence-parallel bucket (c {self.c})"
            st.append(("ffn", "g:y15", lambda: self._set("mb", self.g_y15(), False)))
        elif m["ffn"] == "g":

            def s4g():  # the served RS op on G's per-chip partial (GLM53_PF_REDUCE_FP32 as the op ffn)
                self._set("mb", pf.reduce(self.gk.scratch(self.c)["y"], sp), True)
                self.counters["fillers"] += 1

            st.append(("ffn", "g:rs", s4g))
        else:

            def s4():
                tap_routing_op()
                arm_ffn()
                f = pf.ffn(pl, self._get("h1_in"), sp)
                if self.taps:
                    _copy_into(f, self.taps["ffn_pre"])  # the op ffn's partial per chip, before the RS
                self._set("mb", pf.reduce(f, sp), True)
                ttnn.deallocate(f)
                self.counters["fillers"] += 1

            st.append(("ffn", "op:ffn", s4))
        if self.taps:  # the per-slot taps (inside the trace: one copy each)

            def tap_y0():
                _copy_into(self._get("y0"), self.taps["y0"])

            def tap_mb():
                _copy_into(self._get("mb"), self.taps["mb"])

            st.append(("attn", "tap:y0", tap_y0))
            st.append(("ffn", "tap:mb", tap_mb))

        # ---- slot 5: mhc post (ffn): the op post on A5's / the op's pages -> the residual, in place
        def s5():
            out = pf._post(self._get("mb"), res, self._get("post1"), self._get("comb1"))
            ttnn.copy(out, res)
            ttnn.deallocate(out)
            self.counters["fillers"] += 1

        st.append(("mhc_post", "op:post", s5))
        return st

    def _swap_and_tap_h1(self):
        """the unit's tap-swap (swap_in["h1_in"]: a persistent fp32 x-hat_f staged outside the trace replaces this
        layer's FFN input, so program B / the op ffn run on ANOTHER chain's exact input) and the h1 tap"""
        sw = self.swap_in.get("h1_in")
        if sw is not None:
            _assert_live(sw, f"L{self.L} swap h1_in")
            self._set("h1_in", sw, False)
            self.t["h1_in_normed"] = True
        if self.taps:
            _copy_into(self._get("h1_in"), self.taps["h1"])

    def _free_owned(self):
        import ttnn  # noqa: PLC0415

        for name in list(self._own):
            t_ = self.t.pop(name, None)
            if t_ is not None:
                ttnn.deallocate(t_)
        self._own.clear()
        self.t.clear()

    def prepare(self, x, ci):
        """program A's per-forward descriptors (addresses are persistent; built once per forward)"""
        assert (
            x is self.res
        ), "the served chain runs on the bucket's persistent residual (its address is baked into program A)"
        _assert_live(self.res, f"L{self.L} res")
        if self.a is not None:
            for nm, tl in (("hns", self.a.hns), ("ys", self.a.ys), ("posts", self.a.posts), ("combs", self.a.combs)):
                for i, t_ in enumerate(tl):
                    _assert_live(t_, f"L{self.L} A.{nm}[{i}]")
        if self.b is not None:
            for k_, t_ in self.b.scratch(self.c).items():
                _assert_live(t_, f"L{self.L} B.{k_}")
        if self.a is not None:
            with _ScopedEnv() if self.w7 else _NullCtx():
                self._io, self._descs = _asm_descs(self.a, self.res, ci)
            self.a.counters["calls"] += 1
            if self.w7:
                self.l1_floor = _l1_floor_check(self.a, self.pf.mesh, f"L{self.L} W7")
        if (
            self.g is not None
        ):  # G's composites read their args from its DRAM table (GLM53_PMK_ARGS=table at descriptor time)
            old_ = os.environ.get("GLM53_PMK_ARGS")
            os.environ["GLM53_PMK_ARGS"] = "table"
            try:
                with _ScopedEnv(var="GLM53_PMK_G_ENV"):
                    self._g_io, self._g_descs = _asm_descs(self.g, self.res, ci)
                self.l1_floor_g = _l1_floor_check(self.g, self.pf.mesh, f"L{self.L} G v2a")
            finally:
                if old_ is None:
                    os.environ.pop("GLM53_PMK_ARGS", None)
                else:
                    os.environ["GLM53_PMK_ARGS"] = old_
            self.g.counters["calls"] += 1

    def __call__(self, x, ci, keep: bool = False, **_):
        self.prepare(x, ci)
        self.counters["calls"] += 1
        for _slot, _name, fn in self.steps(ci):
            fn()
        if not keep:
            self._free_owned()
        return self.res

    # ------------------------------------------------------------------ reports
    def plan(self) -> dict:
        d = dict(impl=self.impl, L=self.L, kind=self.kind, bucket=self.c, tp=self.tp, modes=self.modes, pending=self.pending,
                 notes=self.notes, build_s=self.build_s, router_in=self.router_in, reuse=(self.reuse.L if self.reuse is not None else None),
                 steps=[(sl, nm) for sl, nm, _ in self.steps(None)] if self.a is not None or True else [])  # fmt: skip
        if self.a is not None:
            d["A"] = dict(scopes=[list(s) for s in self.a.scopes], launch_census=getattr(self.a.layer, "launch_census", {}),
                          streams=self.a.report.get("streams"), handoff=self.a.handoff.table())  # fmt: skip
        if self.b is not None:
            d["B"] = dict(levers=str(self.b_levers), kwargs=sorted(self.b_kw), rects=len(self.b.rects), rect_sizes=self.b.rect_sizes,
                          blocks=list(self.b.blocks(self.c)), cb_total=self.b.l1_bytes(self.c), stream_pages=self.b_pages,
                          streamers=len(self.b.streamers), streams_per_chip=self.b_stream_check)  # fmt: skip
        return d

    def release(self):
        if self.a is not None:
            self.a.release()
        if getattr(self, "g", None) is not None:
            self.g.release()


# program B's LEVERS OF RECORD = perf5's round-4 cell c18 (doc/ttnn/16-prefill-production/pmk/moe2/perf5_units/
# 18_perf5_lptw-lpt2-combsplit.job, pair L4 14,111 us -> 13.7 ms class; the driver's base levers ran 38.4 ms in the
# chain, d0 card0 r12). Every key is overridable by GLM53_PMK_B_<KEY> (upper case; ints / 0-1 flags / "a,b,c" tuples).
B_CONFIG_OF_RECORD = dict(g=2, x_l1=0, blocks="1,4,8", ids_u16=0, rect="cover", gmax=4, prefetch=1, full_sync=0, yw_db=1,
                          yw_coal=2, mcast_linked=1, mcast_unicast=0, mcast_dual=1, lpt=2, ring_slots=2, inplace=1,
                          comb_split=1, gu_split=0, yw_brisc=0, h_split=4, fid="HiFi2", lpt_w="auto",
                          pack_rne=0, selfstream=0, out_bfp8=0, valid=0, snake=0, xs_rows=0, ffn_fid="",
                          plan_cores=0)  # fmt: skip
# plan_cores (GLM53_PMK_B_PLAN_CORES=32; PMK_PLAN_P, 14 2026-10-26 00:29Z): the routing plan on the first plan_cores of B's
# cores only -- its tables are partition-invariant (bitwise), its cost ~138 + 4.6 us x P (d0 pair: 653 us on 110 cores,
# 285 on 32); 0 = all cores
# B2.3 (Stage 14, w8/progb_b23_design.md; the self-stream form only): valid (GLM53_PMK_B_VALID=1: XFFN_VALID_ROWS at x_l1 0),
# snake (GLM53_PMK_B_SNAKE=1: the op path's cost-class dealing in the G = 1 reader), ffn_fid (GLM53_PMK_B_FFN_FID=LoFi:
# PMK_FFN_FID, the FFN matmuls at LoFi under `fid` -- the combine stays at fid: its multiplies at LoFi lose ~2 % of y's
# norm), xs_rows (GLM53_PMK_B_XS_ROWS=1: x from an L1 row store; needs blocks 2,8,8 -- a loss at that MG on the pair);
# d0 pair in TP8's per-chip geometry (L4 TP8 slices, 2,048 rows, the shared term; every arm bitwise vs 7g, det 20/20;
# u-d0-b23-ab-*-r2, 2026-10-25 23:33Z): zipf 7g 4,619 -> FFN_FID LoFi 4,434 -> + VALID 3,960 -> + SNAKE 3,614 us;
# uniform 4,110 -> 3,968 -> 3,340 -> 3,358
# out_bfp8 0 (GLM53_PMK_B_OUT_BFP8; coordinator 2026-10-23 19:28:07Z, Stage 14's C-term A/B on the pair, HiFi2, L4): the
# expert partials as bf16 rows (the packer untilizes, the combine tilizes in hardware) instead of row-bfp8 (RISC assembly
# in the writer + RISC rebuild in the combine): 2,048 rows 13,599 -> 11,755 us, 4,096 rows 19,858 -> 17,231, PCC vs the
# op path 0.99997 -> 0.99999. yw_db / yw_coal / comb_split / yw_brisc are row-bfp8 levers: dropped when out_bfp8 is 0.
# selfstream (GLM53_PMK_B_SELFSTREAM=1; moe2 PERF.md sec 19, cc9d5fffd4): ProgramB(selfstream=True) returns a SelfStreamB
# = program B's TP8 form of record (every core streaming its own expert slices, bf16 partials, the bf16 combine; 7f 4,553
# us vs op 4,088 at HiFi2; 7g = + x_l1 0, blocks 4,8,8 projected 3.8-3.9 ms; 7i HiFi4). TP8 keys: GLM53_PMK_B_SELFSTREAM=1
# GLM53_PMK_B_X_L1=0 GLM53_PMK_B_BLOCKS=4,8,8 (+ FID=HiFi4 once 7i passes; HiFi4 is TP8-only: the pair's FFN slot is over the
# 20 ms ceiling at HiFi4). Its scratch["y_s"] is ROW_MAJOR [C, H] bf16 (the bf16 combine reads rows): the shared partial is
# laid out to the scratch's layout before the copy (shared() below). Not a TP2 form: the pair keeps selfstream 0.
# pack_rne (GLM53_PMK_B_PACK_RNE=1): program B's down output pre-rounded in DST to its face row's bfp8 grid (RNE) before
# the packer's truncating shared-exponent assembler (logs/bfp8_emul_L4.py rne_prepack_tile = the kernel's golden;
# truncation 0.9905 vs RNE 1.0000 of the fp32 norm on real L4 partials). The ProgramB kwarg lands with moe2's kernel;
# until the tree's signature has it the knob is reported as DROPPED (b_dropped), never silently ignored.
# c13's per-streamer us per chunk on the PAIR, normalised (work-s16-d0/q_r2bu/a_c13.out; the file exists on desktop-0
# only): the default lpt_w at TP2; TP8 has no row yet -> unweighted (lpt 2 by counts) unless GLM53_PMK_B_LPT_W names one
B_LPTW_PAIR_C13_FILE = "/home/alex/glm53/work-s16-d0/q_r2bu/a_c13.out"


def b_config(opts: dict | None = None, env=None, tp: int = 8) -> dict:
    """program B's construction knobs: the record above, then GLM53_PMK_B_<KEY> from env, then opts["b_<key>"] (a
    job's args); lpt_w "auto" = the c13 pair weights at TP2 (file or None), a path / list otherwise"""
    env = os.environ if env is None else env
    cfg = dict(B_CONFIG_OF_RECORD)
    for k in cfg:
        v = env.get(f"GLM53_PMK_B_{k.upper()}")
        if v is not None and v != "":
            cfg[k] = v
    for k, v in (opts or {}).items():
        if k.startswith("b_") and k[2:] in cfg and v is not None:
            cfg[k[2:]] = v
    out = {}
    for k, v in cfg.items():
        if k in ("rect", "fid", "lpt_w", "blocks", "ffn_fid"):
            out[k] = v
        else:
            out[k] = int(v) if not isinstance(v, bool) else int(v)
    if isinstance(out["blocks"], str):
        out["blocks"] = tuple(int(x) for x in out["blocks"].split(",")) if out["blocks"] else None
    lw = out["lpt_w"]
    if lw == "auto":
        lw = B_LPTW_PAIR_C13_FILE if (tp == 2 and os.path.exists(B_LPTW_PAIR_C13_FILE)) else None
    elif isinstance(lw, str) and lw and not os.path.exists(lw) and "," in lw:
        lw = [float(x) for x in lw.split(",")]
    elif isinstance(lw, str) and not lw:
        lw = None
    out["lpt_w"] = lw
    return out


def pack_streams_per_chip(gate_up_by_chip: list, down_by_chip: list):
    """pure torch (the test's twin of load_streams_per_chip): chip k's per-bank stream tiles from ITS slices ->
    (stacked tiles [N, rows, 32], m0, pages), asserting every chip packs the same page count / offsets"""
    import torch  # noqa: PLC0415

    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: PLC0415

    packs = [M2.pack_expert_streams(g, d) for g, d in zip(gate_up_by_chip, down_by_chip)]
    tiles0, m00, pages0 = packs[0]
    for tiles, m0, pages in packs[1:]:
        assert (
            tuple(tiles.shape) == tuple(tiles0.shape) and list(m0) == list(m00) and pages == pages0
        ), "chips pack differently"
    return torch.stack([t.reshape(-1, 32) for t, _, _ in packs]), list(m00), pages0


def load_streams_per_chip(b, gate_up_dev: list, down_dev: list):
    """program B's per-bank stream tensor built PER CHIP from that chip's own XSL slices (fused_decoder.load_xsl_experts:
    device k holds TP slice k) and placed on chip k (ShardTensorToMesh over the stacked chip dim, the mesh's device
    order = get_device_tensors' order). The replicated form (pmk_moe2.ProgramB.load_streams from chip 0's host copy,
    what the standalone bench feeds and compares chip 0 vs chip 0) gave every chip chip 0's experts: Stage 12's
    reading of the 00:04Z L4 row. Returns (pages, check) with check = {chips, bytes per chip, distinct across chips}."""
    import torch  # noqa: PLC0415

    import ttnn  # noqa: PLC0415

    mesh = b.mesh
    n = int(mesh.get_num_devices())
    host = (
        lambda t, k: ttnn.to_torch(ttnn.get_device_tensors(t)[k] if n > 1 else t)
        .to(torch.bfloat16)
        .reshape(-1, int(t.shape[-1]))
    )  # noqa: E731
    gus = [[host(t, k) for t in gate_up_dev] for k in range(n)]
    dns = [[host(t, k) for t in down_dev] for k in range(n)]
    stacked, m0, pages = pack_streams_per_chip(gus, dns)
    distinct = all(not torch.equal(stacked[0], stacked[k]) for k in range(1, n)) if n > 1 else True
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0) if n > 1 else None
    b.stream_t = ttnn.from_torch(stacked.reshape(n, 1, -1, 32), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)  # fmt: skip
    g = b.geometry(2048)
    m0v = torch.zeros(1, g.ep, dtype=torch.int32)
    m0v[0, : len(m0)] = torch.tensor(m0, dtype=torch.int32)
    b.m0_t = ttnn.from_torch(m0v, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if n > 1 else None)  # fmt: skip
    per_chip_shape = tuple(int(v) for v in (ttnn.get_device_tensors(b.stream_t)[0] if n > 1 else b.stream_t).shape)
    assert per_chip_shape == (1, 1, stacked.shape[1], 32), per_chip_shape
    return pages, dict(chips=n, rows_per_chip=int(stacked.shape[1]), distinct_across_chips=bool(distinct))


def served_stack_signature(env=None, *, tp: int = 8, n_layers: int = 45, cfg=None) -> dict:
    """the chain's build signature from the environment alone (Stage 12's prefill image window keys its recording on
    it, before anything is built): the knobs that change what the chain allocates / launches, hashed"""
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    env = os.environ if env is None else env
    keys = ("GLM53_PF_MODE", "GLM53_PMK_SLOTS", "GLM53_PMK_LAYERS", "GLM53_PMK_ROUTER_IN", "GLM53_PMK_B_G", "GLM53_PMK_B_RECT",
            "GLM53_PMK_B_FID", "GLM53_PMK_B_IDS_U16", "GLM53_PMK_A_PARK", "GLM53_PMK_PROBE", "GLM53_PREFILL_BUCKETS", "GLM53_PMK_DSA_MHC", "GLM53_PMK_DSA_SLOTS", "GLM53_PMK_DSA_P11",
            "GLM53_PF_MHC_PRE", "GLM53_PF_MHC_POST", "GLM53_TP_XSL", "GLM53_TP_XSL_SLICED")  # fmt: skip
    defaults = {k: v[0] for k, v in served_stack_env(tp).items()}
    defaults["GLM53_PF_MODE"] = "op"
    vals = {k: str(env.get(k, defaults.get(k, ""))) for k in keys}
    modes = slot_modes_from_env(env) if vals["GLM53_PF_MODE"] == "pmk" else {s: "op" for s in SLOTS}
    kinds = [k for _, k in layer_schedule(cfg)] if cfg is not None else None
    body = dict(v=1, tp=tp, n_layers=n_layers, env=vals, modes=modes,
                plans={k: served_layer_plan(k, tp, modes) for k in KINDS}, kinds=kinds)  # fmt: skip
    blob = json.dumps(body, sort_keys=True, default=str)
    return dict(signature=hashlib.sha1(blob.encode()).hexdigest()[:16], body=body)


def served_stack_env(tp: int = 8) -> dict:
    """the environment keys a served PMK restart adds to the op-graph line's env job (Stage 9: launch_resident.sh /
    the image mock), with their meaning; the value is the default"""
    return {
        "GLM53_PF_MODE": (
            "pmk",
            "the serving switch: 'op' (default) = the traced op graph per bucket; 'pmk' = this chain",
        ),
        "GLM53_PMK_SLOTS": ("auto", "slots on the PMK: auto | none | comma list of mhc_pre,attn,mhc_mid,ffn,mhc_post"),
        "GLM53_PMK_LAYERS": ("0", "0 = every layer of the construction; n = the first n layers only (bisection)"),
        "GLM53_PMK_ROUTER_IN": (
            "fp32",
            "MoE layers with A5 on the PMK: the router's input = the fp32 x-hat_f recomputed by the op's "
            "MhcPrePf (the op path's numerics; one filler) | bf16 = A5's hn[1] (2 % of the tokens re-route)",
        ),
        "GLM53_PMK_B_<KEY>": (
            "perf5 c18",
            "program B's levers of record (B_CONFIG_OF_RECORD: g 2, x_l1 0, blocks 1,4,8, prefetch 1, "
            "full_sync 0, yw_db 1, yw_coal 2, mcast_linked 1, mcast_dual 1, lpt 2, ring_slots 2, inplace 1, "
            "comb_split 1, gu_split 0, yw_brisc 0, h_split 4, fid HiFi2, rect cover, gmax 4, ids_u16 0, "
            "lpt_w auto = c13's pair weights at TP2 / unweighted at TP8); any key overridable",
        ),
        "GLM53_PMK_A_PARK": ("4", "program A: PfStream PARK depth (2..4)"),
        "GLM53_PMK_DSA_MHC": (
            "1",
            "sparse_dsa layers: A1 + A5 (the mhc-only program A) around the op MLA / DSA attention; 0 = all op",
        ),
        "GLM53_PMK_PROBE": ("0", "in-program probe rows on A's streams / kda (diagnostics under trace)"),
        "GLM53_PF_MHC_PRE": (
            "fused7a",
            "REQUIRED (unchanged): the layer's resident MhcPrePf = program A's mhc weights",
        ),
        "GLM53_TP_XSL": ("1", "REQUIRED (unchanged): program B's XSL expert slices"),
        "GLM53_TP_XSL_SLICED": ("1", "REQUIRED (unchanged)"),
        "GLM53_PF_MHC_POST": (
            "fused",
            "REQUIRED (unchanged): the fused-mHC layer (the chain's op fillers are its calls)",
        ),
        "GLM53_PREFILL_BUCKETS": (
            "256,512,2048",
            "unchanged: the 2,048 bucket is the only PMK bucket (v1); the others stay op",
        ),
        "GLM53_PF_NOHEAD": ("1", "unchanged: the head-less twin of every key is captured with the chain inside"),
        "GLM53_PF_PACKED": ("512:4,2048:4", "unchanged: packed keys keep the op graph (layer_packed has no hook: v2)"),
    }


def served_stack_dram_bytes(tp: int = 8, c: int = V1_BUCKET, counts: dict | None = None) -> dict:
    """the per-chip DRAM the served chain ADDS to the op-graph construction, per layer and for the 45-layer stack
    (host arithmetic; the mock measures the same per layer): program A's hand-off tensors + its second copy of the
    projection weights in the mk-stream-v1 layout (asm README conflict #10) + program B's per-bank stream tensor (a
    second copy of the routed experts: moe2 §12) + the persistent residual."""
    counts = counts or {SPARSE_KDA: 31, SPARSE_DSA: 11, DENSE_KDA: 3}
    sp = c // tp
    h_r = HEADS // tp
    w = h_r * HEAD_DIM
    hand = (
        2 * sp * HIDDEN * 2 + 2 * sp * HIDDEN * 4 + 2 * sp * NSTREAM * 4 + 2 * sp * 16 * 4
    )  # hn x2, y x2, post / comb x2
    stream_out = c * (3 * w * 2 + 512 * 2 + w * 4 + w * 4 + HIDDEN * 2)  # qkv, lr, f, gate_out, o_proj y (DRAM stores)
    proj_w = HIDDEN * (3 * w + 512) * 1 + 2 * 512 * w * 1 + w * HIDDEN * 1  # bfp8 ~1 B / element
    a_bytes = hand + stream_out + proj_w
    b_stream = 288 * (HIDDEN * 2 * (2048 // tp) + (2048 // tp) * HIDDEN) * 0.5  # bfp4 gate|up + down per slice
    b_scratch = c * HIDDEN * 2 + c * HIDDEN * 2 + c * 8 * 1088 // 2  # y, y_s, out_rm (row-bfp8 ~)
    per = {SPARSE_KDA: a_bytes + b_stream + b_scratch, DENSE_KDA: a_bytes, SPARSE_DSA: 0.0}
    tot = sum(per[k] * counts[k] for k in per) + sp * RES_COLS * 4
    return dict(tp=tp, bucket=c, per_layer_bytes={k: int(v) for k, v in per.items()}, residual_bytes=sp * RES_COLS * 4,
                program_a_bytes=int(a_bytes), program_b_stream_bytes=int(b_stream), program_b_scratch_bytes=int(b_scratch),
                stack_45_bytes=int(tot), stack_45_gib=round(tot / 2**30, 2),
                note="program B's stream tensor and program A's PfStream weights are SECOND copies of resident weights "
                     "(the served image holds the op path's layouts): the 45-layer stack needs the image to carry the "
                     "stream layouts instead (Stage 12), or the op copies dropped under GLM53_PF_MODE=pmk")  # fmt: skip


# ===================================================================================================== design (b)
class GroupStack:
    """DESIGN (b) (Stage 14; coordinator 2026-10-23 19:39:53Z / 19:43:49Z; doc/ttnn/14-production-tp8-megakernel/
    b_multichunk_design.md): a GROUP of n consecutive c-row chunks of ONE request through the served chain LAYER-MAJOR,
    so program B streams each layer's experts ONCE for n x c rows (its W term amortized n times).

    Per layer L:
      - chunk k = 0 .. n-1 runs slots 1..3 and the router input on its own persistent residual res[k]. Program A is
        built per chunk on that residual and that chunk's ChunkInputs; its transient set is shared through `reuse`.
        The KDA state S / C carries chunk to chunk through the layer's scratch, exactly as consecutive single-chunk
        forwards do. Each chunk runs under its own sparse / fresh context (TPPrefill.plan's window for that chunk).
      - the FFN slot runs ONCE on the per-chip concatenation of the n chunks' FFN inputs: router -> x gather -> B
        inputs -> shared -> program B at n x c rows -> RS. The gathered row order is chip-major (chip p: chunk 0..n-1
        rows) through B and back through the RS, so chip p's slice k of the reduced output is chunk k's.
      - chunk k's slot 5 runs on that slice, into res[k].
    A layer whose FFN slot is the op graph runs every slot per chunk.

    Around the layers: embed per chunk, the head on the last chunk, the slot's KDA state in / out once for the group
    (chunk 0's write_inputs carries the slot / fresh flags, chunks 1.. are segment inputs).

    One captured trace per window tuple (capture / replay). The gate (tests/s16/jobs/pmk_group_unit.py): the group
    vs n sequential single-chunk forwards of the same chain on the same tokens -- residual probes, logits, first
    token and KDA state BITWISE (program B is per token, the fillers are row-generic)."""

    def __init__(self, stack: "PmkStack", n: int, c: int = V1_BUCKET):
        assert stack.impl == "served", "GroupStack drives the served chain (ServedLayer)"
        assert not w7_from_env(), ("GLM53_PMK_W7 (cf3h-k): every sparse_kda layer chunk runs its own fused W7 program -- the W1 "
                                   "group (one MoE over n chunks) does not apply; serve with GLM53_PF_GROUP_MAX=0")  # fmt: skip
        c, n = int(c), int(n)
        assert n >= 2 and c in stack.pmk_buckets, (n, c, stack.pmk_buckets)
        self.stack, self.pf, self.n, self.c = stack, stack.pf, n, c
        pf = self.pf
        self.sp = pf.mode(c) == "sp"
        assert self.sp, "the group form is the sequence-parallel bucket"
        self.cb = n * c  # program B's rows
        # GLM53_PF_MOE_PADMASK (14b 2026-10-26, padm8b): the group is ONE request's n consecutive chunks, so only the LAST
        # chunk can be padded: its c-row routing mask (keep / pad_ids of TPPrefill.pad_inputs, allocated here before any
        # capture) is written per replay in write() and applied to that chunk's ids / weights in the op FFN arm, as the
        # single-chunk forward masks every chunk with its own n_valid (full chunks keep every row: bitwise the same)
        self.pad = pf.pad_inputs(("grp", n, c), c) if getattr(pf, "moe_padmask", False) else None
        self.cis = pf.seg_inputs(c, n)  # chunk k's persistent ChunkInputs (k = 0: the bucket's own)
        self.res = [stack.res[c]] + [self._residual() for _ in range(1, n)]
        self.inst: dict = {}  # (L, k) -> ServedLayer (k = 0: the stack's own program)
        self.ffn: dict = {}  # L -> program B at n x c rows (the layers whose FFN slot is program B)
        self.pc: dict = {}  # k -> persistent copies of A5's posts[1] / combs[1] (shared transients, read by slot 5)
        self.sparse: dict = {}  # (k, t) -> SparseCtx (begin() stores the chunk's masks on the object: one per chunk)
        self.traces: dict = {}  # windows -> trace id
        self.counters = dict(forwards=0, captures=0, replays=0, b_launches=0, fillers=0)
        # W1 B span (tests/s16/jobs/pmk_group_unit.py b_span_layer): layer keep_b's program-B inputs (x_ext, ids_rm, wt,
        # x_tile) are kept from the first FFN slot that runs (the eager compile forward) for a B-only trace afterwards
        self.keep_b, self.kept_b = None, None
        # W1's production-safe arm (coordinator 2026-10-24 21:29:57Z): opts["group_op_ffn"] 1 = the layers whose FFN slot
        # is the OP graph (kinds in opts["group_op_kinds"], default sparse_kda) run it layer-major too: the router / gathers /
        # shared expert per chunk at c rows (as the sequential form), the routed experts (EpMoe: xffn on XSL) ONCE on the
        # chunk-major concatenation, then the add + RS per chunk. EpMoe's own row groups (GLM53_PF_MOE_GROUP_ROWS: 0 or a
        # multiple of n c for one pass) and its mc2 plan window (GLM53_PF_MOE_PLAN_WIN) bound its peak DRAM / L1
        # (serving sets them by env: the served stack's opts come from StackBackend.prepare_prefill)
        self.op_ffn = bool(int(stack.opts.get("group_op_ffn", os.environ.get("GLM53_PF_GROUP_OP_FFN", "0")) or 0))
        self.op_kinds = {k_.strip() for k_ in str(stack.opts.get("group_op_kinds", os.environ.get("GLM53_PF_GROUP_OP_KINDS", SPARSE_KDA))).split(",")
                         if k_.strip()}  # fmt: skip
        # the FFN fillers: "chunk" (default) = router / gathers / shared / RS per chunk at c rows (the sequential forms'
        # programs: every M-dependent op sees the same M), program B ONCE on the chunk-major concatenation; "group" = the
        # fillers on the n-chunk concatenation too (one launch each; M3 on the pair: L4 1.65e-4 off the sequential chain)
        self.fillers = str(stack.opts.get("group_fillers", "chunk"))
        assert self.fillers in ("chunk", "group"), self.fillers
        b_owner = None
        for li, L, kind in stack.layers:
            p0 = stack.progs[(c, L)]
            self.inst[(L, 0)] = p0
            for k in range(1, n):
                self.inst[(L, k)] = ServedLayer(stack, li, c, self.sp, dict(p0.modes), reuse=p0, res=self.res[k],
                                                ci=self.cis[k], build_b=False)  # fmt: skip
            if p0.modes["ffn"] == "pmk":
                self.ffn[L] = p0._build_b(b_owner, rows=self.cb)  # the n x c scratch shared across the layers
                b_owner = b_owner or self.ffn[L]

    def _residual(self):
        import ttnn  # noqa: PLC0415

        mp = ttnn.ShardTensorToMesh(self.pf.mesh, dim=2) if int(self.pf.ndev) > 1 else None
        t = ttnn.from_torch(torch.zeros(1, 1, self.c, RES_COLS), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                            device=self.pf.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mp)  # fmt: skip
        getattr(self.pf, "persistent", set()).add(id(t))
        self.stack.keep.append(t)
        return t

    # ------------------------------------------------------------------ host inputs
    def write(
        self, tokens: torch.Tensor, *, slot: int, chunk_start: int, n_valid: list, state_src=None, state_snap=None
    ):
        """the group's host inputs (OUTSIDE any capture): chunk k = tokens[k c : k c + n_valid[k]] at chunk_start + k c;
        chunk 0 through the bucket's own inputs (the slot / fresh flags of the KDA state in, and the prefix-cache
        state_src / state_snap VIRTUAL slots of TPPrefill.prefill: the state I/O runs ONCE per group, in at the group's
        start and out at its end), chunks 1.. as segments"""
        c, n = self.c, self.n
        assert len(n_valid) == n and all(v == c for v in n_valid[:-1]) and 0 < n_valid[-1] <= c, n_valid
        tk = tokens.reshape(-1)
        for k in range(n):
            sio_kw = dict(state_src=state_src, state_snap=state_snap) if k == 0 else {}
            self.pf.write_inputs(c, tk[k * c : k * c + n_valid[k]], slot=slot, chunk_start=chunk_start + k * c,
                                 n_valid=n_valid[k], ci=None if k == 0 else self.cis[k], **sio_kw)  # fmt: skip
        if self.pad is not None:  # GLM53_PF_MOE_PADMASK: the last chunk's valid rows
            self.pf.write_pad(("grp", n, c), torch.arange(c) < n_valid[-1])

    def windows(self, chunk_start: int, n_valid: list) -> tuple:
        """the chunks' trace-key windows (TPPrefill.plan of each chunk): the group's capture key"""
        keys = [self.pf.plan(chunk_start + k * self.c, n_valid[k]) for k in range(self.n)]
        assert all(k_[0] == self.c for k_ in keys), (keys, self.c)
        return tuple(int(k_[1]) for k_ in keys)

    # ------------------------------------------------------------------ the forward
    def _ctx(self, k: int, t: int):
        from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PF  # noqa: PLC0415

        if t <= 0:
            return None
        if (k, t) not in self.sparse:
            self.sparse[(k, t)] = PF.SparseCtx(self.pf, self.c, t)
        return self.sparse[(k, t)]

    def _set_chunk(self, k: int, ts: tuple, ctxs: list):
        from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PF  # noqa: PLC0415

        self.pf._sparse_now = ctxs[k]
        self.pf._fresh_now = ts[k] == PF.FRESH

    def _post_comb_copy(self, inst, k: int):
        """A5's posts[1] / combs[1] are the SHARED transients (reuse): chunk k's copy, read by its slot 5 after the
        group's FFN (every later chunk's A5 overwrites the shared pair)"""
        import ttnn  # noqa: PLC0415

        a = inst.a
        if a is None or inst.modes["mhc_mid"] != "pmk":
            return  # the op mhc mid: post1 / comb1 are chunk-owned tensors already
        if k not in self.pc:
            self.pc[k] = tuple(ttnn.empty(tuple(int(v) for v in t_.shape), t_.dtype, t_.layout, self.pf.mesh,
                                          ttnn.DRAM_MEMORY_CONFIG) for t_ in (a.posts[1], a.combs[1]))  # fmt: skip
            self.stack.keep.extend(self.pc[k])
        for src, dst in zip((a.posts[1], a.combs[1]), self.pc[k]):
            ttnn.copy(src, dst)
        inst._set("post1", self.pc[k][0], False)
        inst._set("comb1", self.pc[k][1], False)

    def _ffn_chunk_fill(self, L: int, h1s: list, normed: bool) -> list:
        """the FFN slot with the fillers PER CHUNK (ServedLayer._ffn_b_steps' ops at c rows each) and program B ONCE on
        the chunk-major concatenation of their outputs (B is per token): router + ids / sel gathers, the x gather and
        the shared expert per chunk; B's x_tile / ids / weights / y_s concatenated chunk-major ([chunk 0 rows | chunk 1
        rows | ...], every chip all rows); B's y sliced per chunk and reduce-scattered per chunk -> n reduced outputs"""
        import ttnn  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as PM  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32, swiglu_prefill  # noqa: PLC0415

        p0 = self.inst[(L, 0)]
        pf, pl, sp, c, cb, b = self.pf, p0.pl, self.sp, self.c, self.cb, self.ffn[L]
        ids_g, sel_g, hb_g, ys_g, free = [], [], [], [], []
        for k, h1 in enumerate(h1s):
            if normed:
                pl._normed["ffn"] = h1
            x32 = pl.rms(h1, "ffn", FP32)
            idx, scores = pl.moe.router(x32, decode=False)
            sel_l = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
            ids_l = ttnn.typecast(idx, FP32)
            for t_ in (scores, idx):
                ttnn.deallocate(t_)
            ids_f, sel = pf.gather(ids_l, sp), pf.gather(sel_l, sp)
            for t_ in (ids_l, sel_l):
                if t_ is not ids_f and t_ is not sel:
                    ttnn.deallocate(t_)
            hb = pf.gather(ttnn.typecast(x32, BF16), sp)  # [1, 1, C, 4096]: chunk k's rows, every chip
            y_s = swiglu_prefill(pl.shared, hb)
            if x32 is not h1:
                ttnn.deallocate(x32)
            ids_g.append(ids_f)
            sel_g.append(sel)
            hb_g.append(hb)
            ys_g.append(y_s)
            self.counters["fillers"] += 4
        cat = lambda ts: ttnn.concat(ts, dim=2)  # noqa: E731  -- chunk-major rows
        ids_f, sel, hb, y_s = cat(ids_g), cat(sel_g), cat(hb_g), cat(ys_g)
        free += ids_g + sel_g + hb_g + ys_g
        idx16 = ttnn.typecast(ids_f, ttnn.uint16)
        ttnn.deallocate(ids_f)
        x_ext, ids_rm, wt = PM.inputs_from_op_path(hb, idx16, sel, float(pl.moe.scaling))
        if p0.b_levers is not None and p0.b_levers.ids_u16:
            ttnn.deallocate(ids_rm)
            ttnn.deallocate(wt)
            ids_rm, wt = M2.tail_layout_inputs(idx16, sel, float(pl.moe.scaling))
        dst = b.scratch(cb)["y_s"]
        if dst.layout != y_s.layout:
            rows_ = ttnn.to_layout(
                ttnn.reshape(y_s, tuple(int(v) for v in dst.shape)), dst.layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(y_s)
            y_s = rows_
        _copy_into(y_s, dst)
        b(x_ext, ids_rm, wt, cb, x_tile=hb)  # -> scratch y [1, 1, n C, H]: chunk k's rows at [k C, (k + 1) C)
        self.counters["b_launches"] += 1
        y = b.scratch(cb)["y"]
        out = []
        for k in range(self.n):
            y_k = ttnn.slice(y, (0, 0, k * c, 0), (1, 1, (k + 1) * c, int(y.shape[-1])))
            out.append(pf.reduce(y_k, sp))  # the sequential form's RS of B's output at c rows
            ttnn.deallocate(y_k)
            self.counters["fillers"] += 1
        keep = set()
        if self.keep_b == L and self.kept_b is None:
            self.kept_b = dict(x_ext=x_ext, ids_rm=ids_rm, wt=wt, x_tile=hb)
            keep = {id(v) for v in self.kept_b.values()}
        for t_ in {id(v): v for v in free + [idx16, sel, hb, y_s, x_ext, ids_rm, wt]}.values():
            if id(t_) not in keep:
                ttnn.deallocate(t_)
        return out

    def _ffn_op_chunk_fill(self, L: int, h1s: list, normed: bool) -> list:
        """the OP FFN slot layer-major (W1's production-safe arm): TPPrefill.ffn's MoE branch with the rms / router /
        gathers / shared expert per chunk at c rows and the routed experts (pl.epmoe) ONCE on the chunk-major concatenation
        of the n chunks' rows (per token: bitwise vs n calls), then routed + shared and the RS per chunk"""
        import ttnn  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import (  # noqa: PLC0415
            BF16,
            FP32,
            EpMoe,
            swiglu_prefill,
        )

        p0 = self.inst[(L, 0)]
        pf, pl, sp, c = self.pf, p0.pl, self.sp, self.c
        assert pf.moe_impl == "ep" and pf._pad_now is None, (
            pf.moe_impl,
            "the op group arm: EpMoe, the group's own pad mask",
        )
        hbs, ids_g, sel_g, shs = [], [], [], []
        for ci_, h1 in enumerate(h1s):
            if normed:
                pl._normed["ffn"] = h1
            x32 = pl.rms(h1, "ffn", FP32)
            idx, scores = pl.moe.router(x32, decode=False)
            hb = pf.gather(ttnn.typecast(x32, BF16), sp)
            if x32 is not h1:
                ttnn.deallocate(x32)
            k = int(idx.shape[-1])
            sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
            pack = ttnn.concat([ttnn.typecast(idx, FP32), sel], dim=-1)
            for t_ in (scores, idx, sel):
                ttnn.deallocate(t_)
            pack = pf.gather(pack, sp)
            rows = int(pack.shape[-2])
            ids_f = ttnn.slice(pack, (0, 0, 0, 0), (1, 1, rows, k))
            sel_c = ttnn.slice(pack, (0, 0, 0, k), (1, 1, rows, 2 * k))
            ttnn.deallocate(pack)
            if (
                self.pad is not None and ci_ == self.n - 1
            ):  # GLM53_PF_MOE_PADMASK: rows >= n_valid -> the sentinel expert
                assert rows == self.pad["rows"], (rows, self.pad["rows"])
                kept = ttnn.multiply(ids_f, self.pad["keep"])
                ttnn.deallocate(ids_f)
                ids_f = ttnn.add(kept, self.pad["pad_ids"])
                ttnn.deallocate(kept)
                sel_k = ttnn.multiply(sel_c, self.pad["keep"])
                ttnn.deallocate(sel_c)
                sel_c = sel_k
            sel_g.append(sel_c)
            ids_g.append(ttnn.typecast(ids_f, ttnn.uint16))
            ttnn.deallocate(ids_f)
            hbs.append(hb)
            shs.append(swiglu_prefill(pl.shared, hb))
            self.counters["fillers"] += 4
        if getattr(pl, "epmoe", None) is None:
            pl.epmoe = EpMoe(pf, pl.moe, xsl=getattr(pl, "xsl", None))
        cat = lambda ts: ttnn.concat(ts, dim=2)  # noqa: E731  -- chunk-major rows
        hb, idx, sel = cat(hbs), cat(ids_g), cat(sel_g)
        routed = pl.epmoe(hb, idx, sel)  # [1, 1, n C, 4096]: chunk k's rows at [k C, (k + 1) C)
        keep = set()
        if self.keep_b == L and self.kept_b is None:  # the W1 span probe: the routed experts alone on these inputs
            self.kept_b = dict(op=True, x=hb, idx=idx, sel=sel)
            keep = {id(hb), id(idx), id(sel)}
        self.counters["b_launches"] += 1
        out = []
        for k_ in range(self.n):
            r_k = ttnn.slice(routed, (0, 0, k_ * c, 0), (1, 1, (k_ + 1) * c, int(routed.shape[-1])))
            f_k = ttnn.add(r_k, shs[k_])
            out.append(pf.reduce(f_k, sp))  # the sequential form's RS at c rows
            for t_ in (r_k, f_k):
                ttnn.deallocate(t_)
            self.counters["fillers"] += 2
        for t_ in {id(v): v for v in hbs + ids_g + sel_g + shs + [hb, idx, sel, routed]}.values():
            if id(t_) not in keep:
                ttnn.deallocate(t_)
        return out

    def _ffn_group(self, L: int, h1s: list, normed: bool) -> list:
        """the FFN slot ONCE on the n chunks' rows (ServedLayer._ffn_b_steps on the concatenation) -> n reduced slices"""
        import ttnn  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as PM  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32, swiglu_prefill  # noqa: PLC0415

        p0 = self.inst[(L, 0)]
        pf, pl, sp, cb, b = self.pf, p0.pl, self.sp, self.cb, self.ffn[L]
        assert (
            self.pad is None
        ), "GLM53_PF_MOE_PADMASK: the group pad mask is wired for the op FFN arm only (not program B)"
        h1 = ttnn.concat(h1s, dim=2)  # [1, 1, n SP, 4096] per chip: chunk k's local rows at [k SP, (k + 1) SP)
        if normed:
            pl._normed["ffn"] = h1
        x32 = pl.rms(h1, "ffn", FP32)
        idx, scores = pl.moe.router(x32, decode=False)
        sel_l = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
        ids_l = ttnn.typecast(idx, FP32)
        for t_ in (scores, idx):
            ttnn.deallocate(t_)
        ids_f, sel = pf.gather(ids_l, sp), pf.gather(sel_l, sp)
        for t_ in (ids_l, sel_l):
            if t_ is not ids_f and t_ is not sel:
                ttnn.deallocate(t_)
        idx16 = ttnn.typecast(ids_f, ttnn.uint16)
        ttnn.deallocate(ids_f)
        hb = pf.gather(ttnn.typecast(x32, BF16), sp)  # [1, 1, n C, 4096] bf16 TILE = B's x_tile (chip-major rows)
        x_ext, ids_rm, wt = PM.inputs_from_op_path(hb, idx16, sel, float(pl.moe.scaling))
        if p0.b_levers is not None and p0.b_levers.ids_u16:
            ttnn.deallocate(ids_rm)
            ttnn.deallocate(wt)
            ids_rm, wt = M2.tail_layout_inputs(idx16, sel, float(pl.moe.scaling))
        y_s = swiglu_prefill(pl.shared, hb)  # the op P13 on n x c rows: the (K + 1)-th combine term
        dst = b.scratch(cb)["y_s"]
        if dst.layout != y_s.layout:
            rows_ = ttnn.to_layout(
                ttnn.reshape(y_s, tuple(int(v) for v in dst.shape)), dst.layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(y_s)
            y_s = rows_
        _copy_into(y_s, dst)
        ttnn.deallocate(y_s)
        b(
            x_ext, ids_rm, wt, cb, x_tile=hb
        )  # -> scratch y (program B: the layer's experts streamed ONCE for n x c rows)
        self.counters["b_launches"] += 1
        mb = pf.reduce(b.scratch(cb)["y"], sp)  # [1, 1, n SP, 4096] per chip (block k = chunk k's owner rows)
        keep = set()
        if self.keep_b == L and self.kept_b is None:
            self.kept_b = dict(x_ext=x_ext, ids_rm=ids_rm, wt=wt, x_tile=hb)
            keep = {id(v) for v in self.kept_b.values()}
        for t_ in {id(v): v for v in (h1, x32, idx16, sel, hb, x_ext, ids_rm, wt)}.values():
            if id(t_) not in keep:
                ttnn.deallocate(t_)
        spr = int(mb.shape[-2]) // self.n
        out = [ttnn.slice(mb, (0, 0, k * spr, 0), (1, 1, (k + 1) * spr, int(mb.shape[-1]))) for k in range(self.n)]
        ttnn.deallocate(mb)
        self.counters["fillers"] += 6
        return out

    def forward(self, ts: tuple, *, head: bool = True):
        """the whole group on device (the host wrote write()); ts = windows(): chunk k's trace-key window"""
        import ttnn  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import HIDDEN as PH  # noqa: PLC0415
        from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import NSTREAM as PN

        pf, n, sp = self.pf, self.n, self.sp
        assert len(ts) == n, (ts, n)
        ctxs = [self._ctx(k, int(t)) for k, t in enumerate(ts)]
        for k, cx in enumerate(ctxs):
            if cx is not None:
                cx.begin(self.cis[k])
        pf._pad_now = None
        if pf.sio is not None:
            pf.sio.state_in()  # the slot's KDA state (or zeros) -> every layer's scratch, once for the group
        for k in range(n):
            x = pf.embed(self.cis[k])
            ttnn.copy(x, self.res[k])
            if id(x) not in getattr(pf, "persistent", set()):
                ttnn.deallocate(x)
        for li, L, kind in self.stack.layers:
            insts = [self.inst[(L, k)] for k in range(n)]
            for k, inst in enumerate(insts):
                inst.prepare(self.res[k], self.cis[k])
            op_grp = L not in self.ffn and self.op_ffn and kind in self.op_kinds
            if L not in self.ffn and not op_grp:  # the op FFN (dense_kda / sparse_dsa): every slot per chunk
                for k, inst in enumerate(insts):
                    self._set_chunk(k, ts, ctxs)
                    for _s, _nm, fn in inst.steps(self.cis[k]):
                        fn()
                    inst._free_owned()
            else:
                tails, h1s, normed = [], [], True
                for k, inst in enumerate(insts):
                    self._set_chunk(k, ts, ctxs)
                    tail = []
                    for slot, name, fn in inst.steps(self.cis[k]):
                        if (
                            slot in ("mhc_pre", "attn", "mhc_mid")
                            or name in ("h1", "op:router_in_fp32")
                            or name == "tap:y0"
                        ):
                            fn()
                            if name in ("A5", "op:post+pre"):
                                self._post_comb_copy(inst, k)
                        elif slot == "mhc_post":
                            tail.append(fn)
                    h = inst._get("h1_in")
                    if "h1_in" not in inst._own:  # a shared transient (bf16 router input = A5's hn[1]): chunk k's copy
                        h = ttnn.clone(h)
                        inst._set("h1_in", h, True)
                    h1s.append(h)
                    normed = normed and bool(inst.t.get("h1_in_normed"))
                    tails.append(tail)
                pf._sparse_now, pf._fresh_now = None, False
                if op_grp:
                    mbs = self._ffn_op_chunk_fill(L, h1s, normed)
                else:
                    mbs = (
                        self._ffn_group(L, h1s, normed)
                        if self.fillers == "group"
                        else self._ffn_chunk_fill(L, h1s, normed)
                    )
                for k, inst in enumerate(insts):
                    inst._set("mb", mbs[k], True)
                    for fn in tails[k]:
                        fn()
                    inst._free_owned()
            pb = pf._probe.get((self.c, L))
            if (
                pb is not None
            ):  # GLM53_PF_PROBE_LAYERS: the LAST chunk's output residual (the sequential twin's last forward)
                ttnn.copy(self.res[-1], pb)
        for cx in ctxs:
            if cx is not None:
                cx.end()
        pf._sparse_now, pf._fresh_now = None, False
        ci, x = self.cis[-1], self.res[-1]
        row = pf.last_row(x, ci, sp) if head else None
        if pf.sio is not None:
            pf.sio.state_out()
        if head:  # TPPrefill.forward's head on the last chunk
            ttnn.copy(row, pf.row_buf())
            stk = ttnn.reshape(row, (1, 1, PN, PH))
            for res in pf.kv.res:
                if int(res.shape[-1]) != PH:
                    src = ttnn.repeat(row, (1, 1, int(res.shape[-2]), 1))
                elif int(res.shape[0]) > 1:
                    src = ttnn.repeat(stk, (int(res.shape[0]), 1, 1, 1))
                else:
                    src = None
                ttnn.copy(stk if src is None else src, res)
                if src is not None:
                    ttnn.deallocate(src)
            pf.b.head(pf.kv.res, pf.b.pos, pos_copies=pf.b.pos_c)
            ttnn.deallocate(row)
        self.counters["forwards"] += 1

    # ------------------------------------------------------------------ capture / replay
    @staticmethod
    def _key(ts: tuple, head: bool) -> tuple:
        return tuple(int(t) for t in ts), bool(head)

    def capture(self, ts: tuple, *, head: bool = True):
        """ONE trace of the group at windows ts (every program compiled before: forward() eagerly once first); the head
        is part of the key (a head-less group = a non-final run of a request, as TPPrefill's nohead traces)"""
        import ttnn  # noqa: PLC0415

        key = self._key(ts, head)
        if key in self.traces or self.stack.mock:
            return key
        mesh = self.pf.mesh
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            self.forward(key[0], head=key[1])
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        self.traces[key] = tid
        self.counters["captures"] += 1
        return key

    def ready(self, ts: tuple, *, head: bool = True) -> bool:
        """the (windows, head) trace exists (serving replays only prewarmed keys; a cold key falls back per chunk)"""
        return self.stack.mock or self._key(ts, head) in self.traces

    def prepare(self, keys: list):
        """prewarm (startup, before serving): every (windows, head) key's compile launch first (forward eagerly once:
        the programs of every key exist before the first capture -- a compile after a capture leaks a live buffer),
        then one capture per key"""
        import ttnn  # noqa: PLC0415

        keys = [self._key(ts, h) for ts, h in keys]
        for ts, h in keys:
            if (ts, h) not in self.traces:
                self.forward(ts, head=h)
        ttnn.synchronize_device(self.pf.mesh)
        for ts, h in keys:
            self.capture(ts, head=h)
        return keys

    def replay(self, ts: tuple, blocking: bool = True, *, head: bool = True):
        import ttnn  # noqa: PLC0415

        key = self._key(ts, head)
        if self.stack.mock:
            self.forward(key[0], head=key[1])
        else:
            ttnn.execute_trace(self.pf.mesh, self.traces[key], cq_id=0, blocking=blocking)
        self.counters["replays"] += 1

    def prefill(self, tokens: torch.Tensor, *, slot: int, chunk_start: int, n_valid: list, head: bool = True,
                state_src=None, state_snap=None):  # fmt: skip
        """TPPrefill.prefill for a GROUP of the request's chunks (W1, design/w1_grpmax_design.md §8): the DSA slot ops of
        a prefix attach / snapshot at the group's bounds, the group's inputs (state I/O once), ONE replay of the
        (windows, head) trace -- which must be prewarmed (ready); returns the windows key. The caller (StackBackend
        prefill_group) owns cur_pos / the sampler rows, as for a chunk."""
        pf = self.pf
        n_valid = [int(v) for v in n_valid]
        src = None if state_src is None else pf.max_batch + int(state_src)
        snap = None if state_snap is None else pf.max_batch + int(state_snap)
        for v in (src, snap):
            assert v is None or pf.max_batch <= v < pf.slot_rows, (v, pf.max_batch, pf.slot_rows)
        assert src is None or chunk_start > 0, "an attach continues a cached prefix (chunk_start > 0)"
        assert chunk_start % pf.start_align == 0, (chunk_start, pf.start_align)
        ts = self.windows(chunk_start, n_valid)
        assert self.ready(ts, head=head), f"group n {self.n} windows {ts} head {head}: not prewarmed"
        end = chunk_start + sum(n_valid)
        if pf.sio is None:
            pf.b.slot_ops.run(pf.state_ops(slot if src is None else src, into_scratch=True))
        copy = getattr(pf, "prefix_dsa_copy", True)
        dsa_in = [] if src is None or not copy else pf.b._dsa_ops(pf.kv, slot, src, upto=chunk_start)
        if dsa_in:
            pf.b.slot_ops.run(dsa_in)
            pf.counters["slot_ops"] += 1
        self.write(tokens, slot=slot, chunk_start=chunk_start, n_valid=n_valid, state_src=src, state_snap=snap)
        self.replay(ts, blocking=False, head=head)
        dsa_out = [] if snap is None or not copy else pf.b._dsa_ops(pf.kv, snap, slot, upto=end)
        if dsa_out:
            pf.b.slot_ops.run(dsa_out)
            pf.counters["slot_ops"] += 1
        if pf.sio is None:
            pf.b.slot_ops.run(pf.state_ops(slot, into_scratch=False)
                              + ([] if snap is None else pf.state_ops(snap, into_scratch=False)))  # fmt: skip
            pf.counters["slot_ops"] += 2
        self.counters["prefills"] = self.counters.get("prefills", 0) + 1
        return ts

    def release(self):
        import ttnn  # noqa: PLC0415

        for tid in self.traces.values():
            ttnn.release_trace(self.pf.mesh, tid)
        self.traces.clear()

    def plan(self) -> dict:
        opg = sorted(L for _, L, kd in self.stack.layers if L not in self.ffn and self.op_ffn and kd in self.op_kinds)
        return dict(n=self.n, c=self.c, b_rows=self.cb, ffn_layers=sorted(self.ffn), op_group_layers=opg, per_chunk_layers=sorted(
            L for _, L, _ in self.stack.layers if L not in self.ffn and L not in opg), b_blocks={L: list(b.blocks(self.cb)) for L, b in self.ffn.items()},
            b_cb_total={L: b.l1_bytes(self.cb) for L, b in self.ffn.items()}, counters=dict(self.counters))  # fmt: skip
