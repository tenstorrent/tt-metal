# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime options of the MiMo-V2 model stack (formerly ``MIMO_*`` environment variables read at call time).

Passed down explicitly: TtMiMoModel / MiMoRuntimeConfig -> TtDecoderLayer -> TtAttention / TtMoE / TtDenseMLP ->
MoeAgBlock. The defaults are the production configuration. ``MiMoRuntimeOptions.from_env()`` maps the old variables
onto the fields (tests, perf scripts and the engine adapter use it, so existing command lines keep working).
"""

import dataclasses
import os
from dataclasses import dataclass

import ttnn

_EXPERT_DTYPES = {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b}


@dataclass(frozen=True)
class MiMoRuntimeOptions:
    # Fabric links per CCL (ring SDPA, MoE dispatch / combine / reduce). 3 measured best on the BH QuietBox 2x2:
    # MoE dispatch 2.59 / 1.31 / 0.88 ms for 1 / 2 / 3 links at 640 tokens/chip; 4 does not fit the dispatch cores.
    num_links: int | None = None  # MIMO_NUM_LINKS (None: ccl.resolve_num_links, 3 on the QuietBox, 2 on a BH Galaxy)
    # TP all-reduce links; None: op default (on 2x2, 2 links made the reduce-scatter ~1.8x slower).
    ar_links: int | None = None  # MIMO_AR_LINKS
    # Routed-expert weight dtype (ttnn.bfloat4_b | ttnn.bfloat8_b). bf4 (DeepSeek's production default) halves the
    # expert weight bandwidth (experts 3.66 -> 2.73 ms at 640 tok/chip on 2x2); the source is MXFP4, so the loss is
    # small (6-layer stitched KV PCC >= 0.997 vs >= 0.9994 at bf8).
    expert_dtype: object = ttnn.bfloat4_b  # MIMO_EXPERT_DTYPE = bf4 | bf8
    # Routed-expert implementation: "op" the flat_routed_expert C++ op, "py" the Python generic_op builder it was
    # ported from, "unified" unified_routed_expert_moe.
    routed_expert: str = "op"  # MIMO_FLAT_EXPERT = 1 | py | 0
    # unified only: experts with <= T tokens go to moe_fused_swiglu, the rest to unified_routed_expert_moe (DeepSeek
    # / Kimi / GLM use 320). None: unified op only.
    re_hybrid_threshold: int | None = None  # MIMO_RE_HYBRID_THRESHOLD
    # The all-gather MoE block (tt/moe_ag.py); needs the flat_routed_expert op (routed_expert "op"). False: DeepSeek
    # dispatch / combine.
    moe_ag: bool = True  # MIMO_MOE_AG
    # Dispatch-buffer capacity factor; None: ffn.moe_capacity_factor's rule.
    moe_capacity: int | None = None  # MIMO_MOE_CAPACITY
    # A/B: build the flat expert's buffer locally with ttnn.embedding instead of the indexed x read.
    moe_ag_embedding: bool = False  # MIMO_MOE_AG_EMB
    # high_bw_all_gather links (None: every usable link on the axis; QuietBox 4 per axis, Galaxy 2).
    hbw_links: int | None = None  # MIMO_HBW_LINKS
    # reduce_scatter / all_gather links of the > 2-row send-back and the "rsag" TP all-reduce (None: op default).
    moe_ag_rs_links: int | None = None  # MIMO_MOE_AG_RS_LINKS
    # Sequence-parallel residual over TP (norms / adds on S/TP rows; blocks all-gather in, reduce-scatter out).
    sp_residual: bool = True  # MIMO_SP_RESIDUAL
    # Sequence-parallel residual: the MoE block takes this col's normed rows and gathers x / top-k over TP itself
    # (router + untilize on S/TP rows, no block-input all-gather).
    moe_ag_tp_in_gather: bool = True  # MIMO_MOE_AG_TP_IN_GATHER
    # The MoE block's all-gathers: "fabric" (ttnn.experimental.fabric_all_gather: ~10-20% faster on device, traced; eager
    # its per-call fence absorbs the chips' launch skew and it loses 2-3 ms per 6 layers on the LoudBox), or "high_bw".
    moe_ag_gather_op: str = "high_bw"  # MIMO_MOE_AG_GATHER_OP
    # Row reduce-scatters (attention / MLP / MoE TP out, the > 2-row MoE send-back): "ttnn" (ttnn.reduce_scatter) or
    # "fabric" (the fabric_reduce_scatter example: a line add-and-forward relay at the link rate).
    rs_op: str = "ttnn"  # MIMO_RS_OP
    # Per-layer static expert placement (JSON, tests/perf/expert_placement.py; all-gather block only). None: the EP table.
    expert_placement: str | None = None  # MIMO_EXPERT_PLACEMENT
    # Gather the top-k idx / w as tiles and untilize after the gather (high_bw_all_gather costs per page).
    moe_ag_tile_topk: bool = True  # MIMO_MOE_AG_TILE_TOPK
    # Gathered x pages per token row: 1 (one 8 KB page) or H / 1024 (2 KB pages over H / 1024 DRAM banks).
    moe_ag_x_pages_per_row: int = 1  # MIMO_MOE_AG_XPPR
    # TP all-reduce of the MoE output: "hbw" (high_bw_all_gather + one add / tilize pass) or "rsag" (reduce_scatter
    # + all_gather on tiles); None: "rsag" for > 2 mesh columns, else "hbw".
    moe_ag_tp: str | None = None  # MIMO_MOE_AG_TP
    # The flat expert writes y as row-major bf16 (pack-untilized on its down cores); False: bfp8 tiles + an untilize.
    moe_ag_y_row_major: bool = True  # MIMO_MOE_AG_YRM
    # 2 mesh rows: fused send-back (two LocalReduce phases around the exchange) instead of reduce + AddRows.
    moe_ag_fused_send_back: bool = True  # MIMO_MOE_AG_FUSED_SB
    # UntilizeActive tiles per block (y_row_major=False only).
    untilize_width: int = 32  # MIMO_UA_W
    # TTNN weight cache (ttnn.as_tensor tensorbins, keyed by mesh shape): root None -> <extracted checkpoint>/ttnn_cache.
    ttnn_cache: bool = True  # MIMO_TTNN_CACHE = 0 | off | none
    ttnn_cache_root: str | None = None  # MIMO_TTNN_CACHE = <dir>
    # Prefill runtime: synchronize the device before each layer ack (a KV migration burst never reads a layer the
    # device has not finished writing).
    ack_sync: bool = True  # MIMO_ACK_SYNC
    # GA ring SDPA K split; None: automatic (sdpa.default_k_split).
    sdpa_k_split: int | None = None
    # Ring SDPA two-level accumulation (SDPAProgramConfig.ring_two_level / ring_two_level_fold).
    sdpa_two_level: bool = False  # TT_METAL_SDPA_RING_TWO_LEVEL
    sdpa_two_level_fold: int = 0  # TT_METAL_SDPA_RING_TWO_LEVEL_FOLD

    def __post_init__(self):
        assert self.routed_expert in ("op", "py", "unified"), self.routed_expert
        if self.routed_expert == "py" and self.moe_ag:
            raise ValueError(
                'routed_expert="py" (the Python FlatExpert prototype) runs on the dispatch / combine path only: '
                "set moe_ag=False (the all-gather block needs the flat_routed_expert op's indexed mode)"
            )
        assert self.moe_ag_tp in (None, "hbw", "rsag"), self.moe_ag_tp
        assert self.moe_ag_gather_op in ("fabric", "high_bw"), self.moe_ag_gather_op

    @property
    def flat_expert(self) -> bool:
        """Routed experts on a flat streamed expert (the C++ op or its Python builder)."""
        return self.routed_expert in ("op", "py")

    @property
    def use_moe_ag(self) -> bool:
        return self.moe_ag and self.flat_expert

    def replace(self, **kw) -> "MiMoRuntimeOptions":
        return dataclasses.replace(self, **kw)

    @classmethod
    def from_env(cls, env=None, **overrides) -> "MiMoRuntimeOptions":
        """The options the ``MIMO_*`` environment variables select (unset: the default); ``overrides`` win."""
        env = os.environ if env is None else env
        get = lambda n: env.get(n) or None
        opt_int = lambda n: int(get(n)) if get(n) else None
        flag = lambda n, default: env.get(n, "1" if default else "0") == "1"
        kw = {}
        if get("MIMO_NUM_LINKS"):
            kw["num_links"] = int(get("MIMO_NUM_LINKS"))
        kw["ar_links"] = opt_int("MIMO_AR_LINKS")
        kw["expert_dtype"] = _EXPERT_DTYPES[env.get("MIMO_EXPERT_DTYPE", "bf4")]
        kw["routed_expert"] = {"1": "op", "py": "py"}.get(env.get("MIMO_FLAT_EXPERT", "1"), "unified")
        kw["re_hybrid_threshold"] = opt_int("MIMO_RE_HYBRID_THRESHOLD")
        kw["moe_ag"] = flag("MIMO_MOE_AG", kw["routed_expert"] != "py")  # the Python builder: dispatch path
        kw["moe_capacity"] = opt_int("MIMO_MOE_CAPACITY")
        kw["moe_ag_embedding"] = flag("MIMO_MOE_AG_EMB", False)
        kw["hbw_links"] = opt_int("MIMO_HBW_LINKS")
        kw["moe_ag_rs_links"] = opt_int("MIMO_MOE_AG_RS_LINKS")
        kw["moe_ag_x_pages_per_row"] = int(env.get("MIMO_MOE_AG_XPPR", "1"))
        kw["moe_ag_tp"] = env.get("MIMO_MOE_AG_TP")
        kw["moe_ag_y_row_major"] = flag("MIMO_MOE_AG_YRM", True)
        kw["moe_ag_fused_send_back"] = flag("MIMO_MOE_AG_FUSED_SB", True)
        kw["moe_ag_tile_topk"] = flag("MIMO_MOE_AG_TILE_TOPK", True)
        kw["expert_placement"] = env.get("MIMO_EXPERT_PLACEMENT") or None
        kw["sp_residual"] = flag("MIMO_SP_RESIDUAL", True)
        kw["moe_ag_tp_in_gather"] = flag("MIMO_MOE_AG_TP_IN_GATHER", True)
        kw["moe_ag_gather_op"] = env.get("MIMO_MOE_AG_GATHER_OP", "high_bw")
        kw["rs_op"] = env.get("MIMO_RS_OP", "ttnn")
        kw["untilize_width"] = int(env.get("MIMO_UA_W", "32"))
        root = env.get("MIMO_TTNN_CACHE")
        kw["ttnn_cache"] = root not in ("0", "off", "none", "")
        kw["ttnn_cache_root"] = root if kw["ttnn_cache"] else None
        kw["ack_sync"] = flag("MIMO_ACK_SYNC", True)
        kw["sdpa_k_split"] = opt_int("MIMO_SDPA_KSPLIT")
        kw["sdpa_two_level"] = flag("TT_METAL_SDPA_RING_TWO_LEVEL", False)
        kw["sdpa_two_level_fold"] = int(env.get("TT_METAL_SDPA_RING_TWO_LEVEL_FOLD") or 0)
        kw.update(overrides)
        return cls(**kw)
