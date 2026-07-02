# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GLM-4.7-Flash matmul targets and the brute-force sweep axes.

All shapes here are the FULL logical [K, N] (in_features, out_features).  The
harness applies the model's real mesh mapper (`shard`) so each of the 8 chips
ends up computing its true per-device shard -- you declare the logical shape +
shard scheme, ttnn does the split exactly as production.

Verified against config.json + the on-device weight-cache shapes:
  H=2048  q_lora=768  kv_lora=512  heads=20  qk_head_dim=256  v_head_dim=256
  moe_inter=1536  dense_inter=10240  experts=64  top-4  (layer0 dense, 1..46 MoE)
Mesh 2x4: TP=4 on cols (shard N=dim3 / K=dim2), DP=2 on rows, experts EP=8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- model dims (ground truth) ---
H = 2048
Q_LORA = 768
KV_LORA = 512
HEADS = 20
QK_HEAD_DIM = 256  # qk_nope(192) + qk_rope(64)
V_HEAD_DIM = 256
KVPE = KV_LORA + 64  # 576
MOE_INTER = 1536
DENSE_INTER = 10240
N_ROUTED_EXPERTS = 64
TP = 4  # cols


# shard schemes (string -> how the harness maps weight & activation to the mesh)
#   "replicate" : full weight on every chip; activation replicated
#   "col"       : shard N (weight dim 3) across TP cols; activation replicated
#   "row"       : shard K (weight dim 2) across TP cols; activation sharded on K
#   "head"      : batched, shard heads (dim 1) across TP cols   [phase 2]
#   "expert"    : shard experts (dim 0) across all 8 chips, sparse_matmul [phase 2]
SHARD_SCHEMES = ("replicate", "col", "row", "head", "expert")


@dataclass(frozen=True)
class MatmulTarget:
    name: str
    k_full: int
    n_full: int
    shard: str  # one of SHARD_SCHEMES
    collective: str = "none"  # "none" | "all_reduce" | "reduce_scatter" | "all_gather"
    batched: int | None = None  # heads for kv_b1/b2 (phase 2); None for 2D
    sparse: bool = False  # experts (phase 2)
    dp_split: bool = False  # attention matmuls are data-parallel across the mesh ROWS:
    #   per-device tokens = M // dp_rows.  MLP/MoE/router see all tokens.
    note: str = ""

    def __post_init__(self):
        assert self.shard in SHARD_SCHEMES, f"bad shard {self.shard!r}"


# ---- Phase 1: plain 2D matmuls (attention proj + shared/dense MLP) ----
# Row-parallel matmuls are followed by an all_reduce over the TP axis in the
# real model, so we tag them to also measure that collective.
GLM_TARGETS_2D: list[MatmulTarget] = [
    # attention (data-parallel across mesh rows -> dp_split=True)
    MatmulTarget("w_q_a", H, Q_LORA, "replicate", dp_split=True, note="q down-proj (attndp)"),
    MatmulTarget("w_kv_a", H, KVPE, "replicate", dp_split=True, note="kv down-proj (attndp)"),
    MatmulTarget("w_q_kv_a_fused", H, Q_LORA + KVPE, "replicate", dp_split=True, note="fused q+kv down (attndp)"),
    MatmulTarget("w_q_b_attndp", Q_LORA, HEADS * QK_HEAD_DIM, "replicate", dp_split=True, note="q up-proj replicated"),
    MatmulTarget(
        "w_q_b_headpar",
        Q_LORA,
        HEADS * QK_HEAD_DIM,
        "col",
        dp_split=True,
        note="q up-proj head/col-parallel -> [768,1280]/dev",
    ),
    MatmulTarget(
        "w_o", HEADS * V_HEAD_DIM, H, "row", collective="all_reduce", dp_split=True, note="attn out -> [1280,2048]/dev"
    ),
    # shared expert MLP (MoE layers)
    MatmulTarget("w_shared_gate", H, MOE_INTER, "col", note="shared gate -> [2048,384]/dev"),
    MatmulTarget("w_shared_up", H, MOE_INTER, "col", note="shared up -> [2048,384]/dev"),
    MatmulTarget("w_shared_gate_up", H, 2 * MOE_INTER, "col", note="fused shared gate+up -> [2048,768]/dev"),
    MatmulTarget("w_shared_down", MOE_INTER, H, "row", collective="all_reduce", note="shared down -> [384,2048]/dev"),
    # dense MLP (layer 0 only)
    MatmulTarget("w_dense_gate", H, DENSE_INTER, "col", note="dense gate -> [2048,2560]/dev"),
    MatmulTarget("w_dense_up", H, DENSE_INTER, "col", note="dense up -> [2048,2560]/dev"),
    MatmulTarget("w_dense_down", DENSE_INTER, H, "row", collective="all_reduce", note="dense down -> [2560,2048]/dev"),
    # router
    MatmulTarget("w_router", H, N_ROUTED_EXPERTS, "replicate", note="MoE router gate"),
]

# ---- Phase 2 (added after): batched head-parallel + sparse experts ----
GLM_TARGETS_PHASE2: list[MatmulTarget] = [
    MatmulTarget("w_kv_b1_headpar", 192, KV_LORA, "head", batched=HEADS, note="kv_b1 per-head -> [5,192,512]/dev"),
    MatmulTarget(
        "w_kv_b2_headpar", KV_LORA, V_HEAD_DIM, "head", batched=HEADS, note="kv_b2 per-head -> [5,512,256]/dev"
    ),
    MatmulTarget("expert_gate_up", H, 2 * MOE_INTER, "expert", sparse=True, note="routed experts fused gate+up (EP=8)"),
    MatmulTarget("expert_down", MOE_INTER, H, "expert", sparse=True, note="routed experts down (EP=8)"),
]


@dataclass
class PhaseSpec:
    """One (phase, batch, seq_len) point that fixes the matmul row count M.

    M (tokens the matmul processes) is derived exactly as the model does:
      decode : 1 token/sequence  -> M = batch            (tile-padded to 32)
      prefill: full sequence     -> M = batch * seq_len  (capped at prefill_chunk)
    Under attention data-parallelism across `dp_rows` mesh rows, attention
    matmuls (MatmulTarget.dp_split=True) see M // dp_rows per device.
    """

    phase: str  # "decode" | "prefill"
    batch: int
    seq_len: int = 1  # decode ignores this (1 tok/seq); prefill uses it

    def m_tokens(self, *, prefill_chunk: int, dp_rows: int, dp_split: bool) -> int:
        if self.phase == "decode":
            m = max(1, self.batch)
        else:  # prefill
            m = max(1, self.batch) * max(1, self.seq_len)
            if prefill_chunk and prefill_chunk > 0:
                m = min(m, prefill_chunk)
        if dp_split and dp_rows > 1:
            m = max(1, (m + dp_rows - 1) // dp_rows)  # tokens split across mesh rows
        return m


@dataclass
class SweepAxis:
    """Cartesian product of everything here = the brute-force grid.

    Pure Python: dtypes/fidelity are string tokens resolved to ttnn enums in the
    harness, so this module needs no ttnn import.
    """

    # phase/shape axis: each PhaseSpec fixes M (see PhaseSpec.m_tokens)
    phases: list[PhaseSpec] = field(
        default_factory=lambda: [
            PhaseSpec("decode", batch=1),
            PhaseSpec("decode", batch=32),
        ]
    )
    prefill_chunk: int = 128  # cap on prefill M (matches GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE)
    dp_rows: int = 1  # set 2 to split attention tokens across the 2x4 mesh rows (realistic attn_dp)

    in_dtype: list[str] = field(default_factory=lambda: ["bf16"])  # activation dtype
    wt_dtype: list[str] = field(default_factory=lambda: ["bf8", "bf4"])  # weight dtype
    fidelity: list[str] = field(default_factory=lambda: ["lofi", "hifi2"])
    fp32_acc: list[bool] = field(default_factory=lambda: [False])
    packer_l1_acc: list[bool] = field(default_factory=lambda: [True])
    out_mem: list[str] = field(default_factory=lambda: ["dram", "l1", "l1_width_sharded"])
    # program-config candidates; each dict is one prog cfg. "auto" lets the
    # harness derive a grid-capped 1D/2D config. Grids are clamped to the device
    # grid at runtime (your core-count guard), so nothing requests >8 wide on WH.
    prog: list = field(default_factory=lambda: ["auto"])
    iters: int = 5  # measured iterations per config (report min = denoised)
    warmup: int = 2  # warmup iterations (compile + cache) before measuring
