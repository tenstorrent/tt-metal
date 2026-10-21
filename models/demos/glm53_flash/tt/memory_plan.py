# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The serving envelope of one chip: users x context from the DRAM left after the weights.

Per chip (the hybrid 8x4 layout: experts over the columns, each row its own users):

    headroom(ISL) = dram - weights - reserve(ISL)
    users_per_chip x (ISL x cache_per_token + fixed_per_user) <= headroom(ISL)

``cache_per_token`` = the DSA layers' paged latent rows (``kv_lora_rank`` bf16) + the indexer's pooled keys
(``index_head_dim`` x pool dtype per ``index_kpool`` tokens); ``fixed_per_user`` = the KDA recurrent / conv states
(ISL-independent); ``reserve`` = the working set a prefill needs beside the caches (trace region, the regions path's
sub-chunk buffers, the index scorer's score block, the per-chunk whole-prefix latent copy of ``ISL x 576`` bf16).
``users_per_chip`` = users per row with the column-replicated cache (every column of a row holds the row's users),
users per row / columns with the user-partitioned cache (plus one scratch slot per chip while the sequence-parallel
prefill v1 still needs it). Keeping the stacked expert tensors next to the regions path's per-expert lists
(``GLM53_REGIONS_KEEP_STACKED=1``: decode on the stream op) costs a second copy of the expert weights.

Host-side estimates from the config (``MemoryPlan.estimate``); the live form (``MemoryPlan.from_device``) takes the
allocator's numbers after the weights are built. ``tests/test_memory_plan.py`` prints the 8x4 table.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 2**30
TILE = 32
BFP8_BYTES = 1.0625  # bfp8_b: 1 byte per element + one shared exponent per 16
BFP4_BYTES = 0.5625
BF16_BYTES = 2.0
P150_DRAM_BYTES = 8 * 4_205_232_512  # 8 banks x 3.916 GiB allocatable (the allocator's bank size on p150)


def cache_bytes_per_token(cfg, pool_bytes: float = BF16_BYTES) -> float:
    """DSA latent rows + pooled index keys of one user per token (every column of the user's row holds them)."""
    n_dsa = sum(1 for t in cfg.layer_types if t != "linear_attention") if cfg.layer_types else 11
    latent = cfg.kv_lora_rank * BF16_BYTES
    pools = cfg.index_head_dim * pool_bytes / cfg.index_kpool
    return n_dsa * (latent + pools)


def fixed_bytes_per_user(cfg, cols: int) -> float:
    """KDA recurrent [H/cols, K, V] fp32 + conv taps per KDA layer, the index rings and page tables (ISL-independent)."""
    n_kda = sum(1 for t in cfg.layer_types if t == "linear_attention") if cfg.layer_types else 34
    h_local = cfg.kda_num_heads // cols
    rec = h_local * cfg.kda_head_dim * cfg.kda_head_dim * 4
    conv = (cfg.kda_conv_kernel - 1) * (3 * h_local * cfg.kda_head_dim) * 2
    rings = 2 * TILE * cfg.index_head_dim * 2 * 11
    return n_kda * (rec + conv) + rings


def expert_bytes_per_chip(cfg, rows: int, cols: int, expert_bytes: float = BFP8_BYTES) -> float:
    """The routed experts of the hybrid layout: E/cols experts, each I/rows wide, per MoE layer."""
    n_moe = cfg.num_hidden_layers - 3  # first_k_dense_replace = 3
    e_loc, i_loc = cfg.n_routed_experts // cols, cfg.moe_intermediate_size // rows
    per_expert = 3 * cfg.hidden_size * i_loc
    return n_moe * e_loc * per_expert * expert_bytes


def weights_bytes_estimate(
    cfg, rows: int, cols: int, expert_bytes: float = BFP8_BYTES, keep_stacked: bool = False
) -> float:
    """Experts (x2 with keep_stacked) + the TP4-sharded dense weights (attention, shared expert, mHC, embedding, head)."""
    experts = expert_bytes_per_chip(cfg, rows, cols, expert_bytes) * (2 if keep_stacked else 1)
    other = 2.3 * GIB  # measured on the 8x4: 12.4 GB total with 10.1 GB of bfp8 experts
    return experts + other


def reserve_bytes(
    isl: int, trace_mb: int = 512, regions_t: int = 2048, qblock: int = 512, k: int = 8, e_local: int = 72
) -> float:
    """Working set beside the caches during a prefill chunk + the decode trace region."""
    trace = trace_mb * 2**20
    regions = 3 * (regions_t * k + e_local * TILE) * 4096 * 2  # region buffer, gather table, pick rows (bf16)
    score = qblock * (isl / 4) * 2  # the index scorer's [qblock, pools] bf16 block
    prefix_copy = isl * 576 * 2  # the per-chunk whole-prefix ROW_MAJOR latent copy (sparse_sdpa contract)
    misc = 256 * 2**20
    return trace + regions + score + prefix_copy + misc


@dataclass(frozen=True)
class MemoryPlan:
    dram_bytes: float
    weights_bytes: float
    cache_per_token: float
    fixed_per_user: float
    rows: int
    cols: int
    partition: bool
    scratch_slot: bool = True  # the SP prefill v1 scratch slot of the partitioned cache (sparse_1m.md section 7)
    trace_mb: int = 512

    @classmethod
    def estimate(
        cls,
        cfg,
        rows: int,
        cols: int,
        *,
        partition: bool,
        expert_bytes: float = BFP8_BYTES,
        keep_stacked: bool = False,
        scratch_slot: bool = True,
        trace_mb: int = 512,
        dram_bytes: float = P150_DRAM_BYTES,
    ) -> "MemoryPlan":
        return cls(
            dram_bytes,
            weights_bytes_estimate(cfg, rows, cols, expert_bytes, keep_stacked),
            cache_bytes_per_token(cfg),
            fixed_bytes_per_user(cfg, cols),
            rows,
            cols,
            partition,
            scratch_slot,
            trace_mb,
        )

    @classmethod
    def from_device(
        cls, device, cfg, rows: int, cols: int, *, partition: bool, scratch_slot: bool = True, trace_mb: int = 512
    ) -> "MemoryPlan":
        """After the weights are built and before the caches: the allocator's allocated bytes are the weights."""
        import ttnn

        mv = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
        return cls(
            mv.num_banks * mv.total_bytes_per_bank,
            mv.num_banks * mv.total_bytes_allocated_per_bank,
            cache_bytes_per_token(cfg),
            fixed_bytes_per_user(cfg, cols),
            rows,
            cols,
            partition,
            scratch_slot,
            trace_mb,
        )

    def slots_per_chip(self, users_per_row: int) -> int:
        if not self.partition:
            return users_per_row
        return -(-users_per_row // self.cols) + (1 if self.scratch_slot else 0)

    def headroom(self, isl: int) -> float:
        return self.dram_bytes - self.weights_bytes - reserve_bytes(isl, self.trace_mb)

    def bytes_for(self, users_per_row: int, isl: int) -> float:
        return self.slots_per_chip(users_per_row) * (isl * self.cache_per_token + self.fixed_per_user)

    def fits(self, users_per_row: int, isl: int) -> bool:
        return self.bytes_for(users_per_row, isl) <= self.headroom(isl)

    def max_isl(self, users_per_row: int, step: int = 16384) -> int:
        """The largest context (a multiple of ``step``) that fits ``users_per_row`` users per row."""
        slots = self.slots_per_chip(users_per_row)
        lo, hi = 0, 1 << 24
        while hi - lo > step:
            mid = (lo + hi) // 2
            if slots * (mid * self.cache_per_token + self.fixed_per_user) <= self.headroom(mid):
                lo = mid
            else:
                hi = mid
        return lo // step * step

    def max_users_per_row(self, isl: int) -> int:
        """The largest users-per-row count that fits ``isl`` for every user (users on the mesh = rows x that)."""
        u = 0
        while self.fits(u + 1, isl):
            u += 1
            if u >= 4096:
                break
        return u

    def table(self, isls=(16384, 65536, 131072, 262144, 524288, 1048576)) -> str:
        head = "| ISL | max users per row | users on the mesh | cache GiB per chip |\n|---|---|---|---|\n"
        rows = []
        for isl in isls:
            u = self.max_users_per_row(isl)
            rows.append(f"| {isl} | {u} | {u * self.rows} | {self.bytes_for(u, isl) / GIB:.1f} |")
        return head + "\n".join(rows)
