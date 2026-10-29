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


def layer_types_of(cfg, layers=None) -> list:
    """The ``layer_types`` of the layers a model instance builds (``GLM53_MODEL_LAYERS`` / ``n_layers`` subsets: the
    block tests run layers 0-3 on 2 rows), the whole config's when ``layers`` is None."""
    types = list(cfg.layer_types) if cfg.layer_types else []
    if layers is None or not types:
        return types
    return [types[int(i)] for i in layers]


def cache_bytes_per_token(cfg, pool_bytes: float = BF16_BYTES, layers=None) -> float:
    """DSA latent rows + pooled index keys of one user per token (every column of the user's row holds them)."""
    types = layer_types_of(cfg, layers)
    n_dsa = sum(1 for t in types if t != "linear_attention") if types else 11
    latent = cfg.kv_lora_rank * BF16_BYTES
    pools = cfg.index_head_dim * pool_bytes / cfg.index_kpool
    return n_dsa * (latent + pools)


def fixed_bytes_per_user(cfg, cols: int, layers=None) -> float:
    """KDA recurrent [H/cols, K, V] fp32 + conv taps per KDA layer, the index rings and page tables (ISL-independent)."""
    types = layer_types_of(cfg, layers)
    n_kda = sum(1 for t in types if t == "linear_attention") if types else 34
    h_local = cfg.kda_num_heads // cols
    rec = h_local * cfg.kda_head_dim * cfg.kda_head_dim * 4
    conv = (cfg.kda_conv_kernel - 1) * (3 * h_local * cfg.kda_head_dim) * 2
    rings = 2 * TILE * cfg.index_head_dim * 2 * 11
    return n_kda * (rec + conv) + rings


def expert_bytes_per_chip(cfg, rows: int, cols: int, expert_bytes: float = BFP8_BYTES, layers=None) -> float:
    """The routed experts of the hybrid layout: E/cols experts, each I/rows wide, per MoE layer (the layers at or
    past ``first_k_dense_replace`` = 3 among ``layers``; every one of the config's when None)."""
    n_moe = cfg.num_hidden_layers - 3 if layers is None else sum(1 for i in layers if int(i) >= 3)
    e_loc, i_loc = cfg.n_routed_experts // cols, cfg.moe_intermediate_size // rows
    per_expert = 3 * cfg.hidden_size * i_loc
    return n_moe * e_loc * per_expert * expert_bytes


def weights_bytes_estimate(
    cfg, rows: int, cols: int, expert_bytes: float = BFP8_BYTES, keep_stacked: bool = False, layers=None
) -> float:
    """Experts (x2 with keep_stacked) + the TP4-sharded dense weights (attention, shared expert, mHC, embedding, head).
    ``layers``: the subset an instance builds (block tests); the per-layer terms count its layers only and the
    static remainder is scaled by its share of the config's layers (floor 1 GiB: embedding + head shards, the traced
    prefill slot and its inputs stay whatever the depth)."""
    experts = expert_bytes_per_chip(cfg, rows, cols, expert_bytes, layers) * (2 if keep_stacked else 1)
    # the non-expert static footprint, calibrated on the 8x4 (2026-10-21): 28.85 GiB allocated after a 45-layer bfp8
    # build with ONE 1 M slot per chip (11.15 GiB of latent + pools) -> 17.7 GiB of weights + fixed buffers, i.e. ~8.2 GiB
    # beyond the bfp8 experts (dense TP4 weights, the stream / block expert layouts, embedding + head shards, the traced
    # prefill slot and its per-length inputs, page tables, CCL semaphores; attribution pending). With this term the
    # single-user 1 M run fits by ~60 MB, as observed (4.0 of 4.2 GB per bank in use at the first chunk).
    other = 8.2 * GIB
    if layers is not None:
        other = max(1.0 * GIB, other * len(list(layers)) / cfg.num_hidden_layers)
    return experts + other


def default_prefill_qblock(max_seq_len: int | None) -> int:
    """The sparse prefill's score sub-block rows when ``GLM53_PREFILL_QBLOCK`` is unset: 2048 up to 256 K, 512 at
    512 K, 256 at 1 M. The scorer keeps a [rows, pools] bf16 score block and its ranking copy (``_rank_pools``) live
    beside the chunk's whole-prefix latent copy, so at 1 M the three are 134 + 134 + 1200 MB with 256 rows. Served
    1 M runs on 2026-10-27: 2048 rows died at ~196 K (a 201 MB score block), 512 rows at chunk 62 of 64 (the 264 MB
    ranking copy, 0.5 GB free but fragmented); the harness 1 M rows ran 512 with the decode trace not yet captured."""
    if not max_seq_len or int(max_seq_len) <= 262144:
        return 2048
    return 512 if int(max_seq_len) <= 524288 else 256


def reserve_bytes(
    isl: int,
    trace_mb: int = 512,
    regions_t: int = 2048,
    qblock: int | None = None,
    k: int = 8,
    e_local: int = 72,
    chunk_tokens: int = 16384,
) -> float:
    """Working set beside the caches during a prefill chunk + the decode trace region."""
    trace = trace_mb * 2**20
    regions = 3 * (regions_t * k + e_local * TILE) * 4096 * 2  # region buffer, gather table, pick rows (bf16)
    qblock = default_prefill_qblock(isl) if qblock is None else qblock
    score = qblock * (isl / 4) * 2  # the index scorer's [qblock, pools] bf16 block
    prefix_copy = isl * 576 * 2  # the per-chunk whole-prefix ROW_MAJOR latent copy (sparse_sdpa contract)
    misc = 256 * 2**20
    # the floor of the prefill transient is a device chunk's dense working set: about four [chunk, 4384] fp32 buffers of
    # the mHC collapse / expand (1.15 GB at 16 K tokens), whatever the context. Two calibration points (2026-10-27): the
    # 1 M x 4-users-per-row served profile boots and serves with ~120 MB of modelled spare (its whole-prefix copy is the
    # larger term), and the 128 K x 32-users boot (0.55 GiB modelled spare before this term) failed a 287 MB
    # (= 16384 x 4384 x 4 B) allocation in the 32 K warm-up prompt with 613 MB free but fragmented -> 28 users per row
    # (7 slots per chip) at 128 K.
    chunk_ws = 4 * chunk_tokens * 4384 * 4
    # the regions buffers and the score block are released before the attention of a sub-block takes the whole-prefix
    # copy; the observed peak transient at 1 M (QBLOCK 512, regions 2048) was ~1.5 GB with ~1 GB left at the last chunk
    return trace + misc + max(prefix_copy, regions + score, chunk_ws)


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
    # waves (doc/mesh/row_ownership.md): every row runs W = wave_rows of its own chunk; the MoE gathers rows x W tokens
    # per chip, so the chunk working set is that of a rows x W global chunk: neutral at W = 2048 on the 8x4 (16 K)
    wave_rows: int = 2048

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
        layers=None,
    ) -> "MemoryPlan":
        """``layers``: the layer ids the instance builds (None = the whole config); the block tests' 4-layer models
        on 2 rows do not fit the full-depth estimate (the 2-row expert shards alone exceed a chip)."""
        return cls(
            dram_bytes,
            weights_bytes_estimate(cfg, rows, cols, expert_bytes, keep_stacked, layers),
            cache_bytes_per_token(cfg, layers=layers),
            fixed_bytes_per_user(cfg, cols, layers),
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
        chunk_tokens = max(16384, self.rows * int(self.wave_rows))  # the SP chunk, or the rows' wave chunks gathered
        return self.dram_bytes - self.weights_bytes - reserve_bytes(isl, self.trace_mb, chunk_tokens=chunk_tokens)

    def max_wave_rows(self, users_per_row: int, isl: int, step: int = 128) -> int:
        """The largest wave width W (a multiple of ``step``, >= 2048) at which ``users_per_row`` users per row fit
        ``isl``; 0 when not even W = 2048 fits (the planner refuses a wider wave at a full cell)."""
        from dataclasses import replace

        if not replace(self, wave_rows=2048).fits(users_per_row, isl):
            return 0
        lo, hi = 2048, 1 << 20
        while hi - lo > step:
            mid = ((lo + hi) // 2 // step) * step
            if replace(self, wave_rows=mid).fits(users_per_row, isl):
                lo = mid
            else:
                hi = mid
        return lo

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
