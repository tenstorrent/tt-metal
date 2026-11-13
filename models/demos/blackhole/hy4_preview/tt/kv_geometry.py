# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview KV pool geometry and per-chip byte budget (design task kv-design; CPU only, pure Python, no ttnn/torch).

Everything here is arithmetic on the checkpoint shapes (config.json / shape_table.txt) plus ASSUMPTIONS that are listed in ``ASSUMPTIONS`` with the way each one is MEASURED at first boot. Nothing in this file is
a measurement. ``python kv_geometry.py`` prints the tables used by DESIGN.md.

Layout decided in DESIGN.md (s2): one POOL PAGE = one 512-token chunk-block. Inside a page the 512 tokens are striped over the 8 mesh columns in 64-token slices (column c holds tokens [64c, 64c+64) of every
page); every column uses the same local page id (SPMD, the page table is replicated over the columns). Each mesh row owns an independent pool (users are pinned to a row).
"""
from dataclasses import dataclass

GiB = 1 << 30
MiB = 1 << 20

# ---- model constants (config.json, shape_table.txt) ----------------------------------------------------------------------------------------------------
LAYERS = 78
MTP_LAYERS = 1
KV_LAYERS = LAYERS + MTP_LAYERS  # every layer stores its own 576-wide latent (IndexCache shares INDICES only)
INDEX_FULL_LAYERS = 21  # indexer_types == "full": layers 0, 1, 5, 9, ..., 77
INDEX_LAYERS = INDEX_FULL_LAYERS + MTP_LAYERS  # the MTP layer has its own indexer (22 key stores)
LATENT = 512
ROPE = 64
K_DIM = LATENT + ROPE  # 576
INDEX_DIM = 128
INDEX_TOPK = 2048
MAX_CTX = 1 << 20

# ---- mesh / pool geometry --------------------------------------------------------------------------------------------------------------------------------
ROWS, COLS = 4, 8
CHUNK = 512  # prefill chunk = APC grid = pool page (tokens)
SLICE = CHUNK // COLS  # 64 tokens of one page on one column = 2 tiles of 32 rows
DRAM_ALIGN = 64  # Blackhole DRAM alignment (core_config.h LOG_BASE_2_OF_DRAM_ALIGNMENT = 6)


def align(n, a=DRAM_ALIGN):
    return -(-n // a) * a


# ---- per-token-per-layer row formats ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KvFormat:
    name: str
    logical_row: float  # bytes per token per layer as defined by the format
    stored_row: int  # bytes actually reserved per row in an interleaved DRAM buffer (aligned_page_size)
    kernel: str  # who reads it


def _fmt():
    scaled = LATENT + 4 * (LATENT // 128) + 2 * ROPE  # 512 fp8 + 4 fp32 scales + 64 bf16 = 656
    return {
        "bf16": KvFormat("bf16", 2 * K_DIM, align(2 * K_DIM), "stock sparse_sdpa BF16"),
        "scaled_fp8": KvFormat(
            "scaled_fp8",
            scaled,
            align(scaled),
            "stock sparse_sdpa SCALED_FP8 (row stride = buffer()->aligned_page_size())",
        ),
        "fp8": KvFormat("fp8", K_DIM, align(K_DIM), "stock sparse_sdpa FP8_E4M3 (no scales: accuracy risk)"),
        "bfp4": KvFormat("bfp4", K_DIM * 0.5625, int(K_DIM * 0.5625), "NO kernel (new C++)"),
    }


FORMATS = _fmt()
INDEX_KEY_ROW = int(INDEX_DIM * 1.0625)  # bfp8 tiles: 1088 B per 32x128-element... 136 B per key, 32-row tile aligned


def page_bytes_per_chip(fmt="scaled_fp8", packed=False, layers=KV_LAYERS, index_layers=INDEX_LAYERS):
    """bytes one pool page (512 tokens) occupies on ONE chip: its 64-token slice of all layers + the index keys of the full-indexer layers.
    ``packed``: rows at their logical size (a kernel that strides by 656 instead of the aligned page); default = stored (aligned) size.
    """
    f = FORMATS[fmt]
    row = f.logical_row if packed else f.stored_row
    return int(layers * SLICE * row + index_layers * SLICE * INDEX_KEY_ROW)


def kv_bytes_per_token_mesh(fmt="scaled_fp8", packed=False):
    """bytes per token summed over the 8 chips of a row (what a user pays per token in its row's pool)"""
    return page_bytes_per_chip(fmt, packed) * COLS / CHUNK


# ---- weights per chip --------------------------------------------------------------------------------------------------------------------------------------
BFP4, BFP8, BF16 = 0.5625, 1.0625, 2.0


def params():
    """parameter counts from the checkpoint shapes (shape_table.txt); MTP layer = one full layer"""
    H, I, E, Hd = 6144, 2048, 256, 6144
    exp_layer = E * 3 * H * I
    attn_layer = (
        2048 * H + 16384 * 2048 + 576 * H + 28672 * 512 + 16384 * H + H * 16384
    )  # q_a q_b kv_a kv_b linear_gate o_proj
    shared_layer = 3 * H * I
    idx_layer = 32 * H + 128 * H + 4096 * 2048
    return dict(
        routed=(77 + 1) * exp_layer,
        attn=(78 + 1) * attn_layer,
        shared=(77 + 1) * shared_layer,
        dense0=3 * H * 18432,
        indexer=INDEX_LAYERS * idx_layer,
        embed=120832 * H,
        lm_head=120832 * H,
        hc_misc=int(0.03e9),
    )


def weights_gib_per_chip(attn_bytes=BFP8, expert_bytes=BFP4, lm_head_bytes=4.0):
    p = params()
    tp8 = (p["attn"] + p["shared"] + p["dense0"] + p["indexer"]) * attn_bytes / COLS
    emb = (p["embed"] * BF16 + p["lm_head"] * lm_head_bytes) / COLS
    experts = p["routed"] * expert_bytes / (ROWS * COLS)
    return (experts + tp8 + emb + p["hc_misc"] * 4) / GiB


# ---- budget ---------------------------------------------------------------------------------------------------------------------------------------------
# reserve items per chip, GiB: ESTIMATES (see ASSUMPTIONS R); their sum is below the 5 GiB reserve of D2-1, the rest is slack for fragmentation / unknowns
RESERVE_ITEMS = {
    "trace region (device open parameter, 1.2e9 in DSV4 B=8/16)": 1.12,
    "prefill indexer scores+topk transients, Up=4, L=1M, /8 columns": 0.75,
    "MoE prefill dispatch/combine + iHC streams + q/gate transients": 0.70,
    "CCL / all_gather staging (decode q gather, (o,max,sum), xrow rows-mode G/OUT stages)": 0.60,
    "RoPE tables (compact, column-sharded) + sampler/lm_head temporaries": 0.30,
    "largest-free-block / fragmentation slack": 0.50,
}


@dataclass(frozen=True)
class Budget:
    usable_gib: float = 28.5  # allocator-visible DRAM per chip (DSV4 B=8/1M: 3650 MiB/bank x 8)
    reserve_gib: float = 5.0
    attn_bytes: float = BFP8
    expert_bytes: float = BFP4
    fmt: str = "scaled_fp8"
    packed: bool = False
    rows: int = ROWS

    @property
    def weights_gib(self):
        return weights_gib_per_chip(self.attn_bytes, self.expert_bytes)

    @property
    def free_gib(self):
        return self.usable_gib - self.weights_gib - self.reserve_gib

    @property
    def page_chip_bytes(self):
        return page_bytes_per_chip(self.fmt, self.packed)

    @property
    def pages_per_row(self):
        return int(self.free_gib * GiB // self.page_chip_bytes)

    @property
    def tokens_per_row(self):
        return self.pages_per_row * CHUNK

    @property
    def tokens_mesh(self):
        return self.tokens_per_row * self.rows

    def guaranteed_ctx(self, users_per_row, cache_frac=0.0):
        """largest per-user context (multiple of CHUNK) such that ``users_per_row`` users at that context fit in a row's pool, leaving ``cache_frac`` of the pool for cached (APC) blocks"""
        usable = int(self.pages_per_row * (1.0 - cache_frac))
        return min(MAX_CTX, usable // users_per_row * CHUNK)

    def one_user_max(self, users_per_row=1, cache_frac=0.0):
        return self.guaranteed_ctx(users_per_row, cache_frac)


ASSUMPTIONS = [
    (
        "A1",
        "28.5 GiB allocator-visible DRAM per chip (8 banks x 3650 MiB)",
        "DSV4 B=8/1M log: 3300 allocated + 350 free MiB/bank (APC_PAGED_DESIGN.md:172). Is the trace region inside or outside it?",
        "first boot of ANY Hy4 build: open device with trace_region_size=0 and =1.2e9, print per-bank allocatable (ttnn.get_memory_view / 'allocated X, free Y' of a deliberate OOM probe); difference = trace carve-out",
    ),
    (
        "A2",
        "weights 16.0 GiB/chip (experts bfp4 12.3, TP8 attn/shared/dense/indexer bfp8 3.0, embed bf16 + lm_head fp32 0.5, iHC 0.1)",
        "port-plan s2a with params recomputed from shape_table.txt (shared experts are 2.9B, not 1.96B: moved to the bfp8 group, +0.1 GiB)",
        "task M capacity probe: build N real layers, print bytes/bank per tensor group (experts, attn, shared, indexer, embed, head), multiply by layer count; compare 16.0",
    ),
    (
        "A3",
        "reserve 5.0 GiB/chip; itemised estimate 3.97 GiB + 1.0 slack",
        "RESERVE_ITEMS below; trace 1.12 and CCL are from DSV4, the prefill transient items are new for 2048-wide top-k and 8x larger index reads",
        "after the full warm-up + all trace captures at the target (B, Up, ctx): free per bank = A1 - weights - pool; reserve_measured = nothing may be allocated afterwards; run a 1M prefill and a 262144 B=16 decode and read the low-water mark of free bytes (TT_METAL_TRACE_ALLOC_TRACKING=1)",
    ),
    (
        "A4",
        "KV row stride 704 B for SCALED_FP8, not 656",
        "sparse_sdpa_program_factory.cpp:83 packed_page_bytes = kv.buffer()->aligned_page_size(); BH DRAM alignment 64 B (core_config.h:42) -> round_up(656, 64) = 704 (+7.3%). SOURCE-READ, not run",
        "allocate ttnn.empty([1,1,T,656], uint8, ROW_MAJOR, DRAM) for T = 65536 and compare the per-bank allocation delta with 656 x T / 8 (attn-op-probe host, 1 minute)",
    ),
    (
        "A5",
        "index key row = 136 B (bfp8, 128 values + shared exponents), 22 layers, 32-row tile aligned",
        "DSV4 indexer key store (bfp8 slab, 40 KiB/chip/block); Hy4 has no compressed keys, one key per token",
        "allocate the key store [22, pages*64, 128] bfp8 tile on one chip and read bytes/bank",
    ),
    (
        "A6",
        "every one of the 78 layers + MTP stores a 576 latent per token; no GQA 8-kv-head cache",
        "PORT_PLAN risk R1 (U): oracle will confirm absorbed-MLA cache is [T, 576]",
        "ref-oracle: shape of the cache tensor in the CPU reference forward; if it is 8 x 256 x 2 the budget is 7x worse (STOP and redesign)",
    ),
    (
        "A7",
        "pool is per mesh row; users are pinned to a row; KV is replicated over nothing (8 columns share it by sequence striping, 4 rows hold independent pools)",
        "D2-1 (sequence-sharded over columns); DSV4 row pinning of users",
        "page-table / pool allocation log at boot: pool tensor bytes/bank identical on all 32 chips; hit==cold test",
    ),
    (
        "A8",
        "no persistent per-user state besides KV pages and index keys (no halo/tail record store)",
        "DESIGN.md s3: the shared-layer index tensor is consumed inside the same forward step (per query token), never carried across chunks or APC hits",
        "oracle: run chunked prefill (512) vs single-shot on a 4-layer model with a shared layer; indices at the shared layer must be bit-equal to the previous full layer's of the SAME chunk; hit==cold gate at the first device test",
    ),
]


def fmt_table():
    rows = []
    for name, f in FORMATS.items():
        pp = page_bytes_per_chip(name)
        pk = page_bytes_per_chip(name, packed=True)
        rows.append(
            (
                name,
                f.logical_row,
                f.stored_row,
                pp,
                pk,
                kv_bytes_per_token_mesh(name) / 1024,
                kv_bytes_per_token_mesh(name, True) / 1024,
            )
        )
    return rows


def main():
    p = params()
    print("params (B):", {k: round(v / 1e9, 3) for k, v in p.items()}, "total", round(sum(p.values()) / 1e9, 2))
    b = Budget()
    print(
        f"weights {b.weights_gib:.2f} GiB/chip (attn bf16 {weights_gib_per_chip(attn_bytes=BF16):.2f}, attn bfp4 {weights_gib_per_chip(attn_bytes=BFP4):.2f}); free for KV {b.free_gib:.2f} GiB/chip"
    )
    print("reserve items:", round(sum(RESERVE_ITEMS.values()), 2), "GiB of", b.reserve_gib)
    print(
        f"{'fmt':11s} {'row':>6s} {'stored':>6s} {'page/chip B':>12s} {'packed':>10s} {'KiB/tok row':>11s} {'packed':>7s}"
    )
    for r in fmt_table():
        print(f"{r[0]:11s} {r[1]:6.0f} {r[2]:6d} {r[3]:12d} {r[4]:10d} {r[5]:11.1f} {r[6]:7.1f}")
    print()
    for fmt, packed in (("bf16", False), ("scaled_fp8", True), ("scaled_fp8", False), ("fp8", False), ("bfp4", False)):
        bb = Budget(fmt=fmt, packed=packed)
        print(
            f"{fmt:11s} packed={packed!s:5s} pages/row {bb.pages_per_row:5d} tokens/row {bb.tokens_per_row/1e6:5.3f}M mesh {bb.tokens_mesh/1e6:5.2f}M | "
            f"B=16 ctx {bb.guaranteed_ctx(4)//1024:4d}k (10% cache {bb.guaranteed_ctx(4,.1)//1024:4d}k) B=8 ctx {bb.guaranteed_ctx(2)//1024:4d}k (10% {bb.guaranteed_ctx(2,.1)//1024:4d}k) 1 user/row {bb.guaranteed_ctx(1)//1024}k"
        )


if __name__ == "__main__":
    main()
