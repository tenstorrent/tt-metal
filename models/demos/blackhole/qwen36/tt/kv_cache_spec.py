# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache specs of Qwen36ForCausalLM and the place of the GDN state inside a KV cache block.

vLLM's hybrid KV cache manager puts the attention layers into one KV cache group and the GDN layers into MambaSpec
groups of the same size, and overlays the groups on one block pool. The layers at position j of every group share one
pair of K and V buffers (a layer slot, the plugin's ``tensor_idx``), block id b names the same bytes in each of them,
and one group owns a block at a time. A GDN layer therefore keeps the state of a request in the K and V buffers of its
layer slot, at the block its group gave the request: the recurrent state as float32 pages, the convolution history as
bfloat16 pages, both read through ``ttnn.experimental.alias_view`` views of the buffers.

This module has no ttnn import, so host tests can check the specs and the page geometry with torch and vLLM alone.
"""

import ast
import math
import os
import re
from dataclasses import dataclass

import torch

FP32_PAGE_BYTES = 4096
BF16_PAGE_BYTES = 2048
# BFLOAT8_B stores a 32x32 tile as 1024 one-byte mantissas plus one shared exponent per 16 values.
BF8_TILE_BYTES = 1088
# The float32 views address the state in slices of half a value head ([Dk / 2, Dv] = 8 pages). One slice is one
# page row over 8 DRAM banks, so its pages sit at the same offset in every bank.
HALF_HEAD_PAGES = 8
# gdn_decode_step's packed history: 4 tiles per value head, slot 3 the newest token, slot 0 never read.
HIST_SLOTS = 4
# Blackhole P150 has 8 DRAM banks. The specs are built before the mesh opens; GdnBlockState checks the real count.
DRAM_BANKS = 8
# Attention block sizes the prefill supports: a power of two that divides the 2048-token prefill chunk.
BLOCK_SIZES = (64, 128, 256, 512, 1024, 2048)
# gdn_decode_step handles at most this many batch rows.
MAX_DECODE_ROWS = 32


class GdnLayoutError(ValueError):
    """The GDN state of one request does not fit one KV cache block of a layer slot."""


def kv_cache_bf8_enabled():
    """QWEN_SDPA_BF8=1 makes Qwen36Model allocate the paged K and V caches as bfloat8_b."""
    return os.environ.get("QWEN_SDPA_BF8", "0") == "1"


def _cfg(tc, name, default=None):
    return tc.get(name, default) if isinstance(tc, dict) else getattr(tc, name, default)


@dataclass(frozen=True)
class GdnDims:
    """Per-device GDN head geometry at one TP degree."""

    nk: int  # key heads
    nv: int  # value heads
    dk: int
    dv: int
    kconv: int  # convolution taps

    @classmethod
    def from_text_config(cls, text_config, tp):
        nk = int(_cfg(text_config, "linear_num_key_heads"))
        nv = int(_cfg(text_config, "linear_num_value_heads"))
        if nk % tp or nv % tp:
            raise ValueError(f"GDN heads ({nk} key, {nv} value) do not divide by TP={tp}")
        return cls(
            nk=nk // tp,
            nv=nv // tp,
            dk=int(_cfg(text_config, "linear_key_head_dim")),
            dv=int(_cfg(text_config, "linear_value_head_dim")),
            kconv=int(_cfg(text_config, "linear_conv_kernel_dim")),
        )

    @property
    def conv_dim(self):
        """Channels of one convolution input row, [q | k | v]."""
        return 2 * self.nk * self.dk + self.nv * self.dv

    @property
    def head_pages(self):
        """float32 pages of one value head's [Dk, Dv] recurrent state."""
        return self.dk * self.dv // 1024

    def decode_step_problems(self):
        """Why gdn_decode_step cannot run this geometry (empty when it can)."""
        problems = []
        if self.kconv != HIST_SLOTS:
            problems.append(f"conv kernel {self.kconv} != {HIST_SLOTS}")
        if self.nv % self.nk:
            problems.append(f"{self.nv} value heads per device are not a multiple of {self.nk} key heads")
        if 2 * self.nv > 32:
            problems.append(f"{self.nv} value heads per device > 16 (a and b share one 32-wide tile)")
        if self.dk % 32 or self.dv % 32 or self.dk != self.dv:
            problems.append(f"head dims {self.dk} x {self.dv} are not equal multiples of 32")
        if 2 * (2 * self.dk + self.dv) // 32 > 32:
            problems.append(f"a [q | k | v] head row of {2 * self.dk + self.dv} channels does not fit one history tile")
        return problems


def _first_row(block, bank_bytes, page_bytes):
    """First page row of a bank that lies wholly inside block ``block``, which owns bytes
    [block * bank_bytes, (block + 1) * bank_bytes) of every bank."""
    return -(-block * bank_bytes // page_bytes)


def _period(bank_bytes, page_bytes):
    """Blocks after which the page offsets inside a block repeat."""
    return page_bytes // math.gcd(bank_bytes, page_bytes)


@dataclass(frozen=True)
class GdnBlockLayout:
    """Where one request's GDN state lies in one K and V block of a layer slot.

    Interleaved page p of a buffer sits on bank p % banks at row p // banks. A block of ``tiles`` K tiles therefore owns
    ``bank_bytes`` bytes of every bank, and a float32 view page is wholly inside the block only from row
    ``_first_row(block)`` on; with bfloat8_b caches that row depends on the block. Value heads 0 .. heads_k - 1 take
    the first whole float32 rows of the K block, the other heads the first whole float32 rows of the V block. The
    packed history (4 bfloat16 pages per head) takes the last page rows of the V block, or its first rows when K holds
    every head. gdn_decode_step reads the V heads through its second state pool, so their index values are offset by
    the page count of the K view.
    """

    banks: int
    block_size: int
    tiles: int  # tiles of one K (or V) block
    kv_tile_bytes: int
    dims: GdnDims
    heads_k: int
    hist_first: int  # first history page of a block, counted from the block's first bfloat16 page

    @property
    def heads_v(self):
        return self.dims.nv - self.heads_k

    @property
    def bank_bytes(self):
        return self.tiles // self.banks * self.kv_tile_bytes

    @property
    def hist_tiles(self):
        """bfloat16 pages of one block in the bfloat16 view of V."""
        return self.tiles * self.kv_tile_bytes // BF16_PAGE_BYTES

    @property
    def fp32_slices_are_blocks(self):
        """True when every block is a whole number of float32 pages per bank (alias_view's default slice check)."""
        return self.bank_bytes % FP32_PAGE_BYTES == 0

    def fp32_pages(self, num_blocks):
        """Page count of a float32 view of a K or V buffer of ``num_blocks`` blocks: the whole page rows of every
        bank."""
        return self.banks * (num_blocks * self.bank_bytes // FP32_PAGE_BYTES)

    def fp32_view_shape(self, num_blocks):
        """Shape of the float32 views: one dim-0 slice per half value head."""
        return [self.fp32_pages(num_blocks) // HALF_HEAD_PAGES, 1, self.dims.dk // 2, self.dims.dv]

    def hist_view_shape(self, num_blocks):
        """Shape of the bfloat16 history view: one dim-0 slice per page."""
        return [num_blocks * self.hist_tiles, 1, 32, 32]

    def first_fp32_page(self, block):
        """First whole float32 page of ``block`` in the float32 view of K (and of V)."""
        return self.banks * _first_row(block, self.bank_bytes, FP32_PAGE_BYTES)

    def hist_page(self, block, head):
        return block * self.hist_tiles + self.hist_first + HIST_SLOTS * head

    def index_host(self, blocks, num_blocks):
        """gdn_decode_step ``state_page_idx`` for rows that hold ``blocks``: int32 [2 * Nv, len(blocks)]. Row h is the
        first float32 page of value head h, row Nv + h the first history page of head h."""
        b = torch.as_tensor(blocks, dtype=torch.int64).reshape(1, -1)
        first = self.banks * (-(-b * self.bank_bytes // FP32_PAGE_BYTES))
        h = torch.arange(self.dims.nv, dtype=torch.int64).reshape(-1, 1)
        hp = self.dims.head_pages
        rec = torch.where(
            h < self.heads_k,
            first + hp * h,
            self.fp32_pages(num_blocks) + first + hp * (h - self.heads_k),
        )
        hist = b * self.hist_tiles + self.hist_first + HIST_SLOTS * h
        return torch.cat([rec, hist]).to(torch.int32)

    def state_writes(self, block):
        """(pool, first half-head slice, first source half-head, end source half-head) of the recurrent state of
        ``block``: pool 0 is the float32 view of K, pool 1 the float32 view of V."""
        s0 = self.first_fp32_page(block) // HALF_HEAD_PAGES
        parts = [(0, s0, 0, 2 * self.heads_k)]
        if self.heads_v:
            parts.append((1, s0, 2 * self.heads_k, 2 * self.dims.nv))
        return parts

    def write_tables(self, block):
        """paged_fill_cache page tables that put one request's state into ``block``: per state_writes part the
        half-head slices of its view, then the history pages."""
        tables = [list(range(s0, s0 + hi - lo)) for _, s0, lo, hi in self.state_writes(block)]
        first = self.hist_page(block, 0)
        return tables + [list(range(first, first + HIST_SLOTS * self.dims.nv))]


def plan_gdn_block_layout(*, block_size, kv_heads, head_dim, kv_bf8, dims, banks=DRAM_BANKS):
    """The GdnBlockLayout of a K / V cache block of ``block_size`` tokens, or GdnLayoutError with the reason."""
    tile_bytes = BF8_TILE_BYTES if kv_bf8 else BF16_PAGE_BYTES
    if block_size % 32 or head_dim % 32:
        raise GdnLayoutError(f"block {block_size} x head_dim {head_dim} is not whole 32 x 32 tiles")
    tiles = kv_heads * block_size * head_dim // 1024
    if tiles % banks:
        raise GdnLayoutError(
            f"block {block_size}: {tiles} tiles per block do not spread evenly over {banks} DRAM banks"
        )
    if banks % HALF_HEAD_PAGES:
        raise GdnLayoutError(f"{banks} DRAM banks: the half-head float32 slices need a multiple of {HALF_HEAD_PAGES}")
    s = tiles // banks * tile_bytes
    if s % BF16_PAGE_BYTES:
        raise GdnLayoutError(f"block {block_size}: a block holds {s} B per bank, not whole bfloat16 history pages")
    hp = dims.head_pages
    period = _period(s, FP32_PAGE_BYTES)
    rows_k = min((b + 1) * s // FP32_PAGE_BYTES - _first_row(b, s, FP32_PAGE_BYTES) for b in range(period))
    heads_k = min(dims.nv, rows_k * banks // hp)
    heads_v = dims.nv - heads_k
    hist_rows = -(-HIST_SLOTS * dims.nv // banks)
    hist_tiles = tiles * tile_bytes // BF16_PAGE_BYTES
    if hist_rows * BF16_PAGE_BYTES > s:
        raise GdnLayoutError(f"block {block_size}: a V block cannot hold the {HIST_SLOTS * dims.nv}-page history")
    hist_first = 0
    if heads_v:
        rows_v = -(-heads_v * hp // banks)
        for b in range(period):
            end = (_first_row(b, s, FP32_PAGE_BYTES) + rows_v) * FP32_PAGE_BYTES
            if end > (b + 1) * s - hist_rows * BF16_PAGE_BYTES:
                raise GdnLayoutError(
                    f"block {block_size}: K holds {heads_k} of {dims.nv} GDN value heads and V cannot hold the "
                    f"other {heads_v} and the history"
                )
        hist_first = hist_tiles - hist_rows * banks
    return GdnBlockLayout(
        banks=banks,
        block_size=block_size,
        tiles=tiles,
        kv_tile_bytes=tile_bytes,
        dims=dims,
        heads_k=heads_k,
        hist_first=hist_first,
    )


def gdn_state_shapes(dims):
    """MambaSpec shapes and dtypes of one GDN layer on one device, as the block stores them: the packed history
    [Nv, 4, 32, 32] bfloat16 and the recurrent state [Nv, Dk, Dv] float32."""
    return (
        ((dims.nv, HIST_SLOTS, 32, 32), (dims.nv, dims.dk, dims.dv)),
        (torch.bfloat16, torch.float32),
    )


def pack_history_rows(dims):
    """int64 [Nv * 4 * 32]: for each row (head h, slot s, tile row r) of the packed history, the row of the convolution
    carry [K - 1, C] viewed as [(K - 1) * C / 32, 32] that it holds. Slot s >= 1 holds carry row s - 1 (slot 3 the
    newest token); tile rows 2c and 2c + 1 hold channel chunk c of the head's [q_hk | k_hk | v_h] row, hk = h // (Nv /
    Nk). Slot 0 and the rows past the head row are never read and hold row 0."""
    rf = dims.nv // dims.nk
    qn, vn = dims.dk // 32, dims.dv // 32
    kd32 = dims.nk * dims.dk // 32
    row_chunks = dims.conv_dim // 32
    out = torch.zeros(dims.nv, HIST_SLOTS, 32, dtype=torch.int64)
    for h in range(dims.nv):
        hk = h // rf
        chunks = (
            [hk * qn + c for c in range(qn)]
            + [kd32 + hk * qn + c for c in range(qn)]
            + [2 * kd32 + h * vn + c for c in range(vn)]
        )
        for s in range(1, HIST_SLOTS):
            for c, ch in enumerate(chunks):
                out[h, s, 2 * c] = out[h, s, 2 * c + 1] = (s - 1) * row_chunks + ch
    return out.reshape(-1)


def unpack_history_rows(dims):
    """int64 [(K - 1) * C / 32]: for each 32-channel row (carry row j, channel chunk ch) of the convolution carry
    [K - 1, C] viewed as [(K - 1) * C / 32, 32], the row of the packed history viewed as [Nv * 4 * 32, 32] that holds it
    (pack_history_rows read backwards). A q or k chunk of key head hk is read from value head hk * (Nv / Nk), the
    first head that holds it."""
    rf = dims.nv // dims.nk
    qn, vn = dims.dk // 32, dims.dv // 32
    kd32 = dims.nk * dims.dk // 32
    out = torch.zeros(dims.kconv - 1, dims.conv_dim // 32, dtype=torch.int64)
    for ch in range(dims.conv_dim // 32):
        if ch < kd32:
            h, c = (ch // qn) * rf, ch % qn
        elif ch < 2 * kd32:
            h, c = ((ch - kd32) // qn) * rf, qn + (ch - kd32) % qn
        else:
            h, c = (ch - 2 * kd32) // vn, 2 * qn + (ch - 2 * kd32) % vn
        for j in range(dims.kconv - 1):
            out[j, ch] = (h * HIST_SLOTS + j + 1) * 32 + 2 * c
    return out.reshape(-1)


def pack_conv_taps(taps, dims):
    """[Nv, 4, 32, 32] gdn_decode_step taps from 4 per-device tap rows [C] (tap 0 = oldest): tile (h, j) holds tap j
    of the head's [q_hk | k_hk | v_h] channels, chunk c in tile rows 2c and 2c + 1."""
    rows = torch.stack([t.reshape(-1) for t in taps])  # [4, C]
    flat = rows.reshape(-1, 32)  # [4 * C / 32, 32]
    idx = pack_history_rows(dims).reshape(dims.nv, HIST_SLOTS, 32)
    out = torch.zeros(dims.nv, HIST_SLOTS, 32, 32, dtype=rows.dtype)
    row_chunks = dims.conv_dim // 32
    used = 2 * (2 * dims.dk + dims.dv) // 32
    for j in range(HIST_SLOTS):
        # Slot 1's gather rows name history row 0; tap j takes the same channels from tap row j.
        out[:, j, :used] = flat[idx[:, 1, :used] + j * row_chunks]
    return out


def choose_block_size(*, kv_heads, head_dim, kv_bf8, dims, min_block_size=None, banks=DRAM_BANKS):
    """(block size, GdnBlockLayout): the smallest size in BLOCK_SIZES, at least ``min_block_size``, whose K and V block
    holds one GDN state. GdnLayoutError when none does."""
    sizes = [bs for bs in BLOCK_SIZES if min_block_size is None or bs >= min_block_size]
    if not sizes:
        raise GdnLayoutError(f"block size {min_block_size} is above the largest supported block, {BLOCK_SIZES[-1]}")
    reasons = []
    for bs in sizes:
        try:
            return bs, plan_gdn_block_layout(
                block_size=bs, kv_heads=kv_heads, head_dim=head_dim, kv_bf8=kv_bf8, dims=dims, banks=banks
            )
        except GdnLayoutError as e:
            reasons.append(str(e))
    raise GdnLayoutError(
        f"no KV cache block size in {sizes} holds one GDN state ({dims.nv} value heads per device, {'bfloat8_b' if kv_bf8 else 'bfloat16'} K and V): "
        + "; ".join(reasons)
    )


def qwen36_layer_names(text_config):
    """(attention layer names, GDN layer names), named as vLLM's Qwen3.5 model names them."""
    n = int(_cfg(text_config, "num_hidden_layers"))
    types = _cfg(text_config, "layer_types", None) or ["linear_attention"] * 3 + ["full_attention"]
    attn, gdn = [], []
    for i in range(n):
        if types[i % len(types)] == "full_attention":
            attn.append(f"model.layers.{i}.self_attn")
        else:
            gdn.append(f"model.layers.{i}.linear_attn")
    return attn, gdn


def layer_index(name):
    """Decoder layer index of a vLLM layer name (``model.layers.<i>.<module>``)."""
    m = re.search(r"layers\.(\d+)\.", name)
    if m is None:
        raise ValueError(f"layer name {name!r} has no 'layers.<index>.'")
    return int(m.group(1))


# The plugin's MESH_DEVICE presets (vllm_tt_plugin.utils.dp_discovery).
_MESH_PRESETS = {
    "N150": (1, 1),
    "P100": (1, 1),
    "P150": (1, 1),
    "P150x2": (1, 2),
    "N300": (1, 2),
    "P300": (1, 2),
    "N150x4": (1, 4),
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "P150x8": (1, 8),
    "P300x2": (1, 4),
}


def mesh_tp_degree(mesh_device=None, visible_devices=None):
    """Tensor-parallel degree of the mesh the plugin opens: the columns of MESH_DEVICE (a preset or a "(rows, cols)"
    tuple; rows hold replicas), or the TT_VISIBLE_DEVICES count when that differs from the MESH_DEVICE size, as the
    plugin's mesh resolution does. Defaults read the environment."""
    mesh_device = os.environ.get("MESH_DEVICE") if mesh_device is None else mesh_device
    visible_devices = os.environ.get("TT_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if not mesh_device:
        raise ValueError(
            "Qwen36ForCausalLM sizes its KV cache blocks per device before the mesh opens and reads the device count "
            "from MESH_DEVICE; set MESH_DEVICE (for example P150x8)"
        )
    grid = _MESH_PRESETS.get(mesh_device)
    if grid is None:
        try:
            parsed = ast.literal_eval(mesh_device)
            grid = tuple(int(d) for d in parsed)
        except (ValueError, SyntaxError, TypeError):
            grid = None
        if grid is None or len(grid) != 2:
            raise ValueError(f"MESH_DEVICE={mesh_device!r} is neither a mesh preset nor a (rows, cols) tuple")
    visible = [d for d in (visible_devices or "").split(",") if d.strip()]
    if visible and len(visible) != grid[0] * grid[1]:
        return len(visible)
    return grid[1]


def qwen36_kv_cache_spec(
    text_config, tp, *, kv_bf8, mamba_block_size, mamba_cache_mode="none", min_block_size=None, banks=DRAM_BANKS
):
    """dict[layer name, KVCacheSpec] for vLLM's hybrid KV cache manager, sizes per device at ``tp`` devices.

    One FullAttentionSpec per attention layer, with the smallest block size whose K and V block holds one GDN state
    (choose_block_size), and one MambaSpec per GDN layer, padded to the attention page so that vLLM keeps the
    attention block size. ``mamba_block_size``: tokens per state block (max_model_len in mamba_cache_mode "none", so a
    request keeps one state block per group). Raises ValueError for a TP degree or a geometry gdn_decode_step does not
    serve."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    tp = int(tp)
    dims = GdnDims.from_text_config(text_config, tp)
    problems = dims.decode_step_problems()
    if problems:
        raise ValueError(
            f"Qwen36ForCausalLM decodes the GDN state in KV cache blocks with gdn_decode_step, which cannot run this "
            f"checkpoint at TP={tp} (MESH_DEVICE): " + "; ".join(problems) + ". Use a mesh with more devices"
        )
    if tp < 2:
        raise ValueError(
            "Qwen36ForCausalLM keeps the GDN state in KV cache blocks only on the tensor-parallel path; the "
            "single-device GDN layer has no decode on KV cache blocks (not implemented). Use a mesh of 2 or more "
            "devices"
        )
    dim = int(_cfg(text_config, "hidden_size"))
    n_heads = int(_cfg(text_config, "num_attention_heads"))
    n_kv = int(_cfg(text_config, "num_key_value_heads"))
    head_dim = int(_cfg(text_config, "head_dim", None) or dim // n_heads)
    # TPAttention replicates the KV heads when TP exceeds their count.
    n_local_kv = max(1, n_kv // tp)
    block_size, layout = choose_block_size(
        kv_heads=n_local_kv, head_dim=head_dim, kv_bf8=kv_bf8, dims=dims, min_block_size=min_block_size, banks=banks
    )
    attn_kw = {"state_content_bytes": 2 * head_dim * BF8_TILE_BYTES // 1024} if kv_bf8 else {}
    attn = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=n_kv,
        num_head_slots=n_local_kv,
        head_size=head_dim,
        dtype=torch.bfloat16,
        **attn_kw,
    )
    assert attn.page_size_bytes == 2 * layout.tiles * layout.kv_tile_bytes
    shapes, dtypes = gdn_state_shapes(dims)
    gdn = MambaSpec(
        # A multiple of the attention block keeps vLLM's scheduler block size (the LCM of the group block sizes) at
        # this value.
        block_size=-(-int(mamba_block_size) // block_size) * block_size,
        shapes=shapes,
        dtypes=dtypes,
        page_size_padded=attn.page_size_bytes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=mamba_cache_mode,
    )
    attn_names, gdn_names = qwen36_layer_names(text_config)
    specs = {name: attn for name in attn_names}
    specs.update({name: gdn for name in gdn_names})
    return dict(sorted(specs.items(), key=lambda item: layer_index(item[0])))
