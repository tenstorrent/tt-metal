# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 prefill user capacity at 1M per 8x4 stage: a calculation, calibrated on GLM-5.3 bisections.

Per chip, a stage fits ``N`` users while

    W_stage + S_stage + N * U_stage <= A

* ``W_stage``: weight bytes per chip, summed per component. The shard shapes and dtypes are those of GLM-5.3's TTNN
  weight cache (``GLM-5.3-Cache-prefill/glm_5_3_bh_32dev/8x4`` and ``.../glm_5_3_bh_16dev/4x4_sp1``, read on host
  with ``ttnn.load_tensor``; bead tt_metal_tracker-q171.3.7 notes). Attention projections are TP-split over 4 and
  copied on every SP row; wkv_b1 (sparse layers at TP > 1, ``ttMLA.replicates_wkv_b1``) and the indexer's wq_b are
  replicated; routed experts are 256 / chips per chip (8 on 8x4, 16 on 4x4); the embedding (first rank, or a last
  rank running MTP) and the LM head (last rank running MTP) are TP-split bf16 tables.
* ``U_stage``: one user's caches per chip (``Glm53Adapter.allocate_kv_cache``): the scaled-FP8 KVPE cache, one
  704 B page per row (656 B packed row, 64 B aligned) for every KV-carrying layer, and the bfp8 index-K cache (128
  wide) for every full-indexer layer, both TP-deduped: max_seq / (sp * tp) rows per chip.
* ``S_stage``: stage-specific fixed buffers GLM-5.4 adds (G1/G2 hand-off staging, kept pass inputs); small.
* ``A``: what a chip has for weights and caches once every layer-independent buffer is placed (trace region,
  the chunk-proportional indexer score buffer, MoE dispatch, RoPE tables, activations) and fragmentation is paid.
  It is not computed from DRAM size: capacity limits are fragmentation-bound (knowledge note "Capacity sweeps
  allocate weight-shaped tensors instead of loading real weights"), so ``A`` is calibrated on a measured bisection,
  #62125's 4x4 stage 6-7 (rank 0 = layers 58-67 plus the embedding, 1,064,960 tokens, 2560-token chunks): traced
  pass 55 / fail 56, eager pass 56 / fail 57. A 4x4 chip at 2560-token chunks holds the same 640 token rows and 160
  query rows as an 8x4 chip at 5120, so most layer-independent buffers (indexer score buffer, trace region) are
  equal. Two are not, and ``mesh_budget_delta`` moves ``A`` by their difference: the MoE dispatch buffer and the
  routed-expert output buffer of its shape scale with the SP dispatch group (8x4 holds about twice the tokens of
  4x4), and the RoPE tables are SP-sharded (8x4 holds half). Net, an 8x4 chip has about 346 MB less than the 4x4
  calibration. ``capacity`` reports low / mid / high: mid and high are the adjusted bisection midpoint and upper
  bound; low is the 8x4 floor that SC4 CI proves (``CI_FLOOR``: GLM-5.3 runs 54 users, so its fullest stage fits).

GLM-5.3 check: the same formula on GLM-5.3's default SC4 splits (``compute_layer_split``: 19/19/20/20 trunk,
19/20/20/19 + MTP) against what SC4 CI runs (54 trunk, 52 MTP KV share; nightly 45115566753, both legs pass).
Those are configured counts (lower bounds), not bisected limits: no 8x4 GLM-5.3 bisection exists.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config as C
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import GLM54Config as C54
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry

MAX_SEQ_LEN = 1_064_960  # glm53.json PREFILL_MAX_SEQ_LEN
CAPACITY_FACTOR = 8  # prefill_runner.py PREFILL_CAPACITY_FACTOR default; no GLM manifest sets it
MB = 1e6

# Physical bytes per 32x32 tile.
BFP8_TILE = 1088
BFP4_TILE = 576
BF16_TILE = 2048
DRAM_ALIGNMENT = 64

_GEOMETRY = MlaKvCacheGeometry(latent_dim=C.KV_LORA_RANK, rope_dim=C.QK_ROPE_HEAD_DIM)
KVPE_PAGE_BYTES = -(-_GEOMETRY.packed_row_bytes // DRAM_ALIGNMENT) * DRAM_ALIGNMENT  # 656 -> 704


def _tiles(rows: int, cols: int) -> int:
    return -(-rows // 32) * -(-cols // 32)


@dataclass(frozen=True)
class Mesh:
    sp: int
    tp: int

    @property
    def chips(self) -> int:
        return self.sp * self.tp

    @property
    def experts_per_chip(self) -> int:
        return C.NUM_ROUTED_EXPERTS // self.chips


MESH_8X4 = Mesh(sp=8, tp=4)
MESH_4X4 = Mesh(sp=4, tp=4)


def component_bytes(mesh: Mesh) -> dict[str, int]:
    """Per-chip weight bytes per component; shapes as the TTNN cache shards them (bead q171.3.7 tensorbin logs)."""
    tp = mesh.tp
    h = C.EMB_SIZE // tp  # 1536: hidden rows of a TP shard
    heads = C.NUM_ATTENTION_HEADS // tp  # 16
    # Norm vectors as the cache stores them: attn_norm / ffn_norm [48, 32] bf16 each, q_a_layernorm [64, 32],
    # kv_a_layernorm [16, 32].
    norms = 2 * 48 * 32 * 2 + 64 * 32 * 2 + 16 * 32 * 2
    mla = BFP8_TILE * (
        _tiles(h, C.Q_LORA_RANK)  # q_a_proj
        + _tiles(C.Q_LORA_RANK, heads * (C.QK_NOPE_HEAD_DIM + C.QK_ROPE_HEAD_DIM))  # q_b_proj
        + _tiles(h, C.KV_LORA_RANK + C.QK_ROPE_HEAD_DIM)  # kv_a_proj_with_mqa
        + C.NUM_ATTENTION_HEADS * _tiles(C.QK_NOPE_HEAD_DIM, C.KV_LORA_RANK)  # wkv_b1, replicated (mla_repl)
        + heads * _tiles(C.KV_LORA_RANK, C.V_HEAD_DIM)  # wkv_b2
        + _tiles(heads * C.V_HEAD_DIM, C.EMB_SIZE)  # o_proj
    )
    indexer = (
        BFP8_TILE * _tiles(C.Q_LORA_RANK, C.INDEX_N_HEADS * C.INDEX_HEAD_DIM)  # wq_b, replicated
        + BFP8_TILE * _tiles(h, C.INDEX_HEAD_DIM)  # wk
        + BF16_TILE * _tiles(h, C.INDEX_N_HEADS)  # weights_proj
        + 2 * C.INDEX_HEAD_DIM * 2  # k_norm weight + bias
    )
    routed = mesh.experts_per_chip * 3 * BFP4_TILE * _tiles(C.EMB_SIZE, C.MOE_INTERMEDIATE_SIZE)
    shared = 3 * BFP8_TILE * _tiles(C.EMB_SIZE, C.MOE_INTERMEDIATE_SIZE // tp)
    router = BF16_TILE * _tiles(h, C.NUM_ROUTED_EXPERTS) + 2 * C.NUM_ROUTED_EXPERTS  # weight + correction bias
    dense_ffn = 3 * BFP8_TILE * _tiles(C.EMB_SIZE, C.INTERMEDIATE_SIZE // tp)
    vocab_table = C.VOCAB_SIZE * C.EMB_SIZE // tp * 2  # bf16, one of embedding / lm_head
    return {
        "mla": mla + norms,
        "indexer": indexer,
        "moe": routed + shared + router,
        "dense_ffn": dense_ffn,
        "embedding": vocab_table,
        "lm_head": vocab_table,
        "mtp_eh_proj": 2 * C.EMB_SIZE // tp * C.EMB_SIZE * 2,  # [12288 / tp, 6144] bf16
        # A trunk-only last rank builds its last layer kv-only (prefill_runner.py kv_only_last_layer): attn_norm and
        # the KV projection, no Q/O, wkv_b, indexer or FFN.
        "mla_kv_only": BFP8_TILE * _tiles(h, C.KV_LORA_RANK + C.QK_ROPE_HEAD_DIM) + 16 * 32 * 2 + 48 * 32 * 2,
        # GLM-5.4 damp gate (A1): RMSNorm gain + Linear(6144 -> width) + bias, bf16. Synthetic weights use the
        # scalar width; a per-channel gate [6144, 6144] would add at most 75.5 MB per chip if replicated.
        "damp_gate": (C.EMB_SIZE + C.EMB_SIZE * 1 + 1) * 2,
    }


def user_bytes_per_layer(mesh: Mesh, max_seq_len: int = MAX_SEQ_LEN) -> dict[str, int]:
    """One user's per-chip cache bytes per KV-carrying layer and per full-indexer layer (TP-deduped)."""
    rows = max_seq_len // mesh.chips
    assert rows * mesh.chips == max_seq_len and rows % 32 == 0
    return {
        "kvpe": rows * KVPE_PAGE_BYTES,
        "index_k": BFP8_TILE * _tiles(rows, C.INDEX_HEAD_DIM),
    }


def full_layers(first: int, last: int) -> int:
    """Full-indexer layers among 0-indexed first..last (inclusive), GLM-5.3 map."""
    types = C.indexer_types()
    return sum(1 for i in range(first, last + 1) if types[i] == "full")


@dataclass
class Stage:
    """One pipeline stage as one 8x4 (or 4x4) rank holds it."""

    name: str
    layers: tuple[int, int]  # 0-indexed inclusive weight range
    mesh: Mesh = MESH_8X4
    indexer: bool = True  # False on GLM-5.4 G2 (pass 2 runs no indexer, A5)
    mtp: bool = False  # last rank running one MTP level (A7)
    embedding: bool = False  # first rank, a stage-alone probe rank, or MTP
    damp_gate: bool = False
    # Per-user extras (bytes per chip) and fixed extras beyond the GLM-5.3 stage, with labels.
    user_extra: dict[str, int] = field(default_factory=dict)
    fixed_extra: dict[str, int] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return self.layers[1] - self.layers[0] + 1

    @property
    def kv_only_last_layer(self) -> bool:
        """prefill_runner.py: ``kv_only_last_layer = is_last_rank and not MTP_LEVELS`` (an offset rank too, when it
        ends at the model's last layer)."""
        return not self.mtp and self.layers[1] == C.NUM_LAYERS - 1

    @property
    def dense_layers(self) -> int:
        return max(0, min(self.layers[1] + 1, C.NUM_DENSE_LAYERS) - self.layers[0])

    @property
    def index_layers(self) -> int:
        if not self.indexer:
            return 0
        # The MTP layer owns an index-K slot (enable_mtp_indexer_slot).
        return full_layers(*self.layers) + (1 if self.mtp else 0)

    @property
    def kvpe_layers(self) -> int:
        return self.num_layers + (1 if self.mtp else 0)

    def weights(self) -> dict[str, float]:
        comp = component_bytes(self.mesh)
        kv_only = 1 if self.kv_only_last_layer else 0
        assert not kv_only or C.indexer_types()[self.layers[1]] == "shared", "a kv-only full layer owns no indexer"
        moe_layers = self.num_layers - kv_only - self.dense_layers + (1 if self.mtp else 0)
        w = {
            "mla": (self.num_layers - kv_only + (1 if self.mtp else 0)) * comp["mla"],
            "mla_kv_only": kv_only * comp["mla_kv_only"],
            "indexer": self.index_layers * comp["indexer"],
            "moe": moe_layers * comp["moe"],
            "dense_ffn": self.dense_layers * comp["dense_ffn"],
        }
        if self.embedding or self.mtp:
            w["embedding"] = comp["embedding"]
        if self.mtp:
            w["lm_head"] = comp["lm_head"]
            w["mtp_eh_proj"] = comp["mtp_eh_proj"]
        if self.damp_gate:
            w["damp_gate"] = comp["damp_gate"]
        return {k: v for k, v in w.items() if v}

    def per_user(self) -> dict[str, float]:
        u = user_bytes_per_layer(self.mesh)
        out = {"kvpe": self.kvpe_layers * u["kvpe"], "index_k": self.index_layers * u["index_k"]}
        out.update(self.user_extra)
        return {k: v for k, v in out.items() if v}

    def fixed(self) -> float:
        return sum(self.fixed_extra.values())


@dataclass(frozen=True)
class Bisection:
    """A measured capacity: pass at ``passes``, fail at ``passes + 1``."""

    stage: Stage
    passes: int
    source: str

    def budget_bounds(self) -> tuple[float, float]:
        """[low, high) bounds on A implied by pass at N / fail at N + 1."""
        w = sum(self.stage.weights().values()) + self.stage.fixed()
        u = sum(self.stage.per_user().values())
        return w + self.passes * u, w + (self.passes + 1) * u


# #62125 "User capacity at 1M on 4x4 stages": SC4 4x4 stages 6-7 under PREFILL_PP_LAYER_OFFSET; rank 0 = layers
# 58-67 and holds the embedding; the out-of-memory was on rank 0 at bring-up in both modes.
CAL_STAGE = Stage("4x4 stage 6 (offset rank 0)", (58, 67), mesh=MESH_4X4, embedding=True)
CALIBRATION = {
    "traced": Bisection(CAL_STAGE, 55, "#62125 4x4 stages 6-7, traced, job 169106: fits 55, fails 56"),
    "eager": Bisection(CAL_STAGE, 56, "#62125 4x4 stages 6-7, eager, job 168932: fits 56, fails 57 (12 MB short)"),
}


def mesh_layer_independent_bytes(mesh: Mesh, chunk: int, max_seq_len: int = MAX_SEQ_LEN) -> dict[str, float]:
    """Per-chip bytes of the layer-independent buffers whose size depends on the mesh, at one chunk size.

    * MoE dispatch: ``compute_constants`` (moe/init_helpers.py) sizes the buffer to dispatch_group_size (= the SP
      extent, ``MeshConfig``) x seq_len_per_chip (chunk / sp) x the capacity factor, plus one tile per local expert
      but one. GLM's fabric2d dispatch moves bf16 tokens (``TtDispatch2dModule``); the routed experts write a bfp8
      TILE buffer of the same shape (tt_routed_expert.py), live together with it (tt_moe.py).
    * RoPE: the whole-cache cos and sin tables, bf16, SP-sharded (mla/rope.py ``get_rope_tensors``).
    """
    raw = mesh.sp * (chunk // mesh.sp) * CAPACITY_FACTOR
    tokens = raw + 32 * (min(raw, mesh.experts_per_chip) - 1)
    return {
        "moe_dispatched_bf16": tokens * C.EMB_SIZE * 2,
        "moe_expert_outputs_bfp8": tokens * C.EMB_SIZE * BFP8_TILE / 1024,
        "rope_cos_sin": max_seq_len // mesh.sp * C.QK_ROPE_HEAD_DIM * 2 * 2,
    }


CAL_CHUNK = 2560  # glm53_4x4.json PREFILL_CHUNK_SIZE
TARGET_CHUNK = 5120


def mesh_budget_delta() -> float:
    """How much less an 8x4 chip at 5120 has for weights and caches than the 4x4 calibration chip at 2560."""
    target = sum(mesh_layer_independent_bytes(MESH_8X4, TARGET_CHUNK).values())
    cal = sum(mesh_layer_independent_bytes(MESH_4X4, CAL_CHUNK).values())
    return target - cal


def ci_floor() -> float:
    """Lower bound on the 8x4 budget A: GLM-5.3's fullest stage at the user counts SC4 CI runs and passes (54 trunk,
    52 MTP KV share; nightly 45115566753)."""
    configured = {"trunk": 54, "mtp_kv_share": 52}
    return max(
        sum(s.weights().values()) + s.fixed() + n * sum(s.per_user().values())
        for split, n in configured.items()
        for s in _glm53_stages()[split]
    )


def budget(mode: str) -> dict[str, float]:
    """8x4 budget A: low = the CI floor; mid / high = the 4x4 bisection's midpoint / exclusive upper bound, moved by
    ``mesh_budget_delta``. Eager has at least the traced budget, so the traced CI floor bounds both."""
    lo, hi = CALIBRATION[mode].budget_bounds()
    delta = mesh_budget_delta()
    return {"low": ci_floor(), "mid": (lo + hi) / 2 - delta, "high": hi - delta}


def capacity(stage: Stage, mode: str = "traced") -> dict:
    """Users that fit, at the low / mid / high end of the 8x4 budget, and the per-chip breakdown at mid."""
    a = budget(mode)
    lo, mid, hi = a["low"], a["mid"], a["high"]
    weights = stage.weights()
    per_user = stage.per_user()
    w = sum(weights.values()) + stage.fixed()
    u = sum(per_user.values())
    # The high bound is exclusive: a stage needing exactly `hi` does not fit.
    fits = {k: math.floor((a - w) / u) for k, a in (("low", lo), ("mid", mid), ("high", hi - 1))}
    n = fits["mid"]
    return {
        "users": fits,
        "per_chip_MB": {
            "weights": {k: round(v / MB, 1) for k, v in weights.items()},
            "weights_total": round(sum(weights.values()) / MB, 1),
            "stage_fixed_extra": {k: round(v / MB, 2) for k, v in stage.fixed_extra.items()},
            "per_user": {k: round(v / MB, 2) for k, v in per_user.items()},
            "per_user_total": round(u / MB, 2),
            "caches_at_mid_users": round(n * u / MB, 1),
            "budget_A": {k: round(v / MB, 1) for k, v in a.items()},
            "slack_at_mid_users": round((mid - w - n * u) / MB, 1),
        },
        "layers": {
            "weights": list(stage.layers),
            "kvpe_layers": stage.kvpe_layers,
            "index_k_layers": stage.index_layers,
            "dense_layers": stage.dense_layers,
            "kv_only_last_layer": stage.kv_only_last_layer,
            "mtp": stage.mtp,
            "embedding": stage.embedding or stage.mtp,
        },
    }


def _glm54_stages(stage_alone: bool = False, g3_mtp: bool = True) -> list[Stage]:
    """GLM-5.4 on 4 stages of 8x4 (epic DESIGN Q1). ``stage_alone``: one rank standing in for the stage via
    PREFILL_PP_LAYER_OFFSET, which is a first rank and so loads the embedding (#62125: "rank 0 also holds the
    embedding")."""
    loop = (C54.LOOP_FIRST_LAYER, C54.LOOP_LAST_LAYER)
    n_loop = loop[1] - loop[0] + 1
    rows = MESH_8X4  # per-chip shares of a 5120-token chunk: 640 token rows, 160 query rows after head-to-seq
    q_rows = 5120 // rows.chips
    x_bytes = (5120 // rows.sp) * (C.EMB_SIZE // rows.tp) * 2  # one [640, 1536] bf16 activation
    topk_sets = full_layers(*loop) * q_rows * C.INDEX_TOPK * 4  # 7 uint32 top-k sets per chunk per chip
    kvpe_rows = n_loop * q_rows * KVPE_PAGE_BYTES  # one chunk's pass-1 KVPE rows of the 28 loop layers per chip
    # Pass-2 window carry: 128 rows x 704 B per loop layer per user, bounded as if every chip kept all of them.
    window = n_loop * C54.LOOP_WINDOW_SIZE * KVPE_PAGE_BYTES
    return [
        Stage("G0", (0, loop[0] - 1), embedding=True),
        Stage(
            "G1",
            loop,
            embedding=stage_alone,
            damp_gate=True,
            fixed_extra={"x0_keep": x_bytes, "g1_to_g2_topk_staging": topk_sets, "g1_to_g2_kvpe_staging": kvpe_rows},
        ),
        Stage(
            "G2",
            loop,
            indexer=False,
            embedding=stage_alone,
            damp_gate=True,
            user_extra={"window_bound": window},
            fixed_extra={"x1_keep": x_bytes, "topk_receive": topk_sets, "kvpe_receive": kvpe_rows},
        ),
        Stage("G3", (loop[1] + 1, C.NUM_LAYERS - 1), mtp=g3_mtp, embedding=stage_alone),
    ]


def _glm53_stages() -> dict[str, list[Stage]]:
    """GLM-5.3's default SC4 splits (compute_layer_split): trunk 19/19/20/20; MTP (tail_levels=1) 19/20/20/19."""
    return {
        "trunk": [
            Stage("S0", (0, 18), embedding=True),
            Stage("S1", (19, 37)),
            Stage("S2", (38, 57)),
            Stage("S3", (58, 77)),
        ],
        "mtp_kv_share": [
            Stage("S0", (0, 18), embedding=True),
            Stage("S1", (19, 38)),
            Stage("S2", (39, 58)),
            Stage("S3", (59, 77), mtp=True),
        ],
    }


def _pipeline(stages: list[Stage], mode: str) -> dict:
    per = {s.name: capacity(s, mode) for s in stages}
    binding = min(per, key=lambda k: per[k]["users"]["mid"])
    users = {e: min(p["users"][e] for p in per.values()) for e in ("low", "mid", "high")}
    return {"stages": per, "pipeline_users": users, "binding_stage": binding}


ASSUMPTIONS = [
    "A (per-chip budget for weights + caches) is calibrated on #62125's 4x4 stage 6-7 bisection and moved to 8x4 at "
    "5120 by mesh_budget_delta (about -346 MB: MoE dispatched buffer bf16 and expert outputs bfp8 at 41,184 vs "
    "20,960 tokens per chip, RoPE tables half the size). Other layer-independent buffers (trace region, indexer "
    "score buffer: 160 query rows per chip on both) are taken as equal. The 4x4 failure was the 341 MB indexer "
    "score buffer; on 8x4 the largest transient is the ~506 MB dispatched buffer, so the fragmentation-bound limit "
    "may bind differently: the low end is the 8x4 floor SC4 CI proves (GLM-5.3 at 54 users), not the 4x4 bracket.",
    "GLM-5.4 G2 runs no indexer, so it allocates no indexer score buffer (about 341 MB per chip: 160 query rows x "
    "1,064,960 keys x bf16, the failing allocation of #62125's 4x4 stage 0); the calculation does not credit it, "
    "because the two-bank attention's own buffers are not implemented yet (q171.2.5). Crediting it adds about 0.5 "
    "user on G2.",
    "G2 pass-2 window carry is bounded as 128 rows x 704 B per loop layer per user on every chip (2.52 MB per user); "
    "the real storage (q171.2.6) is likely SP-local and smaller.",
    "Damp gates use the scalar width (synthetic weights); a per-channel gate adds at most 75.5 MB per chip on G1 "
    "and G2 (0.1 user).",
    "G3 holds one MTP level as GLM-5.3 (A7): embedding + LM head + one MoE layer + eh_proj, and one more KVPE and "
    "index-K slot per user. Without MTP G3 is reported separately; there layer 77 is built kv-only, as the runner "
    "does on a trunk-only last rank.",
    "Stage-alone probes (q171.3.5) put the stage on a first rank, which loads the embedding (475.8 MB per chip, "
    "about 0.7 user on G1/G2); predictions for that setup are reported under stage_alone_probe.",
    "Traced, forward_chunks = 1 (glm53.json), so no packed-forward scratch slot (kv_cache_slots).",
]


def report() -> dict:
    out = {
        "max_seq_len": MAX_SEQ_LEN,
        "mesh": "8x4 (SP8 x TP4) per stage",
        "inputs": {
            "kvpe_page_bytes": KVPE_PAGE_BYTES,
            "per_user_per_layer_8x4_MB": {k: v / MB for k, v in user_bytes_per_layer(MESH_8X4).items()},
            "per_user_per_layer_4x4_MB": {k: v / MB for k, v in user_bytes_per_layer(MESH_4X4).items()},
            "weights_per_layer_8x4_MB": {k: v / MB for k, v in component_bytes(MESH_8X4).items()},
            "weights_per_layer_4x4_MB": {k: v / MB for k, v in component_bytes(MESH_4X4).items()},
            "calibration": {
                mode: {"source": b.source, "A_bounds_4x4_MB": [round(x / MB, 1) for x in b.budget_bounds()]}
                for mode, b in CALIBRATION.items()
            },
            "mesh_layer_independent_MB": {
                "8x4@5120": {k: v / MB for k, v in mesh_layer_independent_bytes(MESH_8X4, TARGET_CHUNK).items()},
                "4x4@2560": {k: v / MB for k, v in mesh_layer_independent_bytes(MESH_4X4, CAL_CHUNK).items()},
                "budget_delta": mesh_budget_delta() / MB,
            },
            "ci_floor_A_MB": ci_floor() / MB,
            "budget_A_8x4_MB": {mode: {k: v / MB for k, v in budget(mode).items()} for mode in CALIBRATION},
        },
        "assumptions": ASSUMPTIONS,
    }
    for mode in ("traced", "eager"):
        out[mode] = {
            "glm54": _pipeline(_glm54_stages(), mode),
            "glm54_g3_without_mtp": capacity(_glm54_stages(g3_mtp=False)[3], mode),
            "glm54_stage_alone_probe": _pipeline(_glm54_stages(stage_alone=True), mode),
            "glm53_check": {k: _pipeline(v, mode) for k, v in _glm53_stages().items()},
        }
    out["glm53_configured"] = {
        "trunk": {"users": 54, "source": "glm53.json; SC4 CI nightly 45115566753 passes"},
        "mtp_kv_share": {"users": 52, "source": "glm53_mtp_kv_share.json; SC4 CI nightly 45115566753 passes"},
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()
    rep = report()
    text = json.dumps(rep, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    for mode in ("traced", "eager"):
        g = rep[mode]["glm54"]
        print(f"[{mode}] GLM-5.4 pipeline {g['pipeline_users']} binding {g['binding_stage']}")
        for name, s in g["stages"].items():
            m = s["per_chip_MB"]
            print(f"  {name}: users {s['users']}  weights {m['weights_total']} MB  per user {m['per_user_total']} MB")
        for k, p in rep[mode]["glm53_check"].items():
            print(f"  GLM-5.3 {k}: {p['pipeline_users']} binding {p['binding_stage']}")


if __name__ == "__main__":
    main()
