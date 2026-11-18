# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: S3-PB with option C's KV layout (update 4 item A; bench/packaging/update4/KV_ADMISSION_FINDINGS.md
§4, KV_OPTION_C_DESIGN_ADDENDUM.md, LEAD_DECISIONS_PKG P103). vLLM sees one full-attention KV group (the 10 full
layers); the 50 sliding layers keep adapter-owned rings outside vLLM's pool, so admission charges only the full
blocks the engine uses.

Gemma4ResidentMultiS32PBForCausalLM (unchanged) plus:
- get_kv_cache_spec: FullAttentionSpec for the full layers only, at the native page. The plugin's single-group path
  then passes `page_table` (the full group's table) and no `page_tables_per_layer`; prefill_forward / decode_forward
  hand the S-chain `[full table] * 60`, so S0's keys, S1's continuation rebind, _group_tables and _layout run as before
  over 3.2's six groups (layer_groups comes from the layer kinds alone).
- Pools: 10 full [N + B, 1, 128, 512] tensors (unaliased, full_page_scale 1, scratch N + slot) and 50 ring
  [RING_PAGES, 4, 128, 256] tensors. Ring page 0 is a null page; slot s owns ring pages 1 + 9 s + j (j < 9) and its
  write-only sink RING_PAGES - B + s. Page 0 cannot belong to a ring: the serving-table check accepts 0 only when the
  scratch pages are the last B pages (generator.py:250-257), and PB masks every sliding 0 to -1, which the fill writer
  skips (prefill_paged.py:611-619). N is sized so the KV bytes equal S3.2's pool to the byte (allocate_pools checks it).
- Prefill rows (one host row shared by every sliding layer): columns [max(lo, e - 8), e] of a chunk [start, end) go to
  the slot's ring, e = (end - 1) // 128 its live end and lo = start // 128; every other column is 0, i.e. skipped.
- Decode rows: every column up to the full row's last live block goes to the ring (c mod 9), past it to the sink. Live
  full blocks cover every scheduled row, so a block of M rows at p writes its pages into the ring (M7: row x overwrites
  only row x - 1152; the window of row p survives for M <= 129).
- What a window reads (the row-level claim the layout rests on): at decode position p, every row <= p of every window
  column was written by this request's own prefill or decode. At the first decode the window lies in the final chunk's
  ring columns (or column e + 1 at row 0); a column enters the window at its row 0, in the step that writes it. Rows
  above p of a window column may hold a previous occupant's or a warm pass's data (the rest of the current page, the
  fill's padding tiles); those are causally masked, as with a vLLM block handed over under S3.2.
"""

import os

import torch
from loguru import logger

from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import CacheState
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_vllm import Gemma4ResidentMultiS32PBForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import Gemma4ResidentForCausalLM

PAGE = Decoder.PAGE_SIZE
RING = Decoder.SLIDING_WINDOW_PAGES  # 9: an unaligned 1,024-row window straddles at most nine pages
FULL_LAYERS, SLIDING_LAYERS = 10, 50
TILE_BYTES = 1088  # one BF8 32 x 32 tile
SLIDING_PAGE_BYTES = 2 * 4 * (PAGE // 32) * (256 // 32) * TILE_BYTES  # K and V of [4, 128, 256] per chip: 278,528
FULL_PAGE_BYTES = 2 * 1 * (PAGE // 32) * (512 // 32) * TILE_BYTES  # K and V of [1, 128, 512] per chip: 139,264


def ring_pages(users):
    """Pages of one sliding layer's ring tensor: the null page, nine per slot, one sink per slot."""
    return 1 + (RING + 1) * users


def ring_page(slot, column):
    return 1 + RING * slot + column % RING


def sink_page(slot, users):
    return ring_pages(users) - users + slot


def s32_pool_bytes(max_model_len, users):
    """S3.2's KV bytes per chip: 10 pool tensors of N' = blocks + users sliding pages, blocks from M1's token budget
    (generator_resident_vllm.get_max_tokens_all_users) plus the plugin's output page per user."""
    tokens = Gemma4ResidentForCausalLM.get_max_tokens_all_users(max_num_seqs=users, max_model_len=max_model_len)
    blocks = -(-(tokens + PAGE * users) // PAGE)
    return FULL_LAYERS * (blocks + users) * SLIDING_PAGE_BYTES


def full_blocks(max_model_len, users):
    """vLLM's full-group block count N: what S3.2's bytes leave after the rings, less a scratch page per slot."""
    left = s32_pool_bytes(max_model_len, users) - SLIDING_LAYERS * ring_pages(users) * SLIDING_PAGE_BYTES
    pages, rem = divmod(left, FULL_LAYERS * FULL_PAGE_BYTES)
    if rem:
        raise ValueError(f"S3.2's pool does not split into whole full pages ({rem} B left)")
    return pages - users


class Gemma4ResidentMultiS32PBRingForCausalLM(Gemma4ResidentMultiS32PBForCausalLM):
    FIRST_RING_PAGE = 1  # ring pages follow the null page 0 (ring_page)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ring_chunks = None  # (start, end) per prefill row of the call in flight, for _tables
        self.kv_bytes = None

    # ------------------------------------------------------------------ vLLM's view

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Gemma4 QB2 does not implement prefix caching")
        if vllm_config.max_in_flight_tokens > 2 * cls.MAX_PREFILL_TOKENS:
            raise ValueError("Gemma4 QB2 supports at most 8192 batched tokens with async chunked prefill")
        config = vllm_config.model_config.hf_text_config
        block = vllm_config.cache_config.block_size
        if block != PAGE:
            raise ValueError("Gemma4 QB2 uses 128-token cache pages")
        # The full layers only; the sliding layers' rings are the adapter's (module docstring).
        return {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                block_size=block,
                num_kv_heads=config.num_global_key_value_heads,
                head_size=config.global_head_dim,
                dtype=torch.bfloat16,
            )
            for i, kind in enumerate(config.layer_types)
            if kind != "sliding_attention"
        }

    @classmethod
    def get_max_tokens_all_users(cls, *, max_num_seqs, max_model_len=262144, **kwargs):
        # The plugin adds one output page per user (worker.py:799), which makes the pool N blocks.
        if os.environ.get("GEMMA4_RESIDENT_KV_POOL", "") == "b32":
            raise ValueError("GEMMA4_RESIDENT_KV_POOL=b32 sizes S3.2's hybrid pool; the ring layout has its own")
        return PAGE * (full_blocks(max_model_len, max_num_seqs) - max_num_seqs)

    # ------------------------------------------------------------------ pools

    def allocate_pools(self, per_layer_specs, *, alias=True):
        """10 full pools and 50 rings as a CacheState (module docstring). The plugin's single-group specs give every
        layer the full group's shape (model_runner._build_per_layer_specs); the sliding entries only count layers."""
        layers = self.generator.model.layers
        users = self.max_batch_size
        if len(per_layer_specs) != len(layers):
            raise ValueError("The allocator must supply one spec per layer")
        shapes = {tuple(shape) for shape, _, _ in per_layer_specs}
        if len(shapes) != 1 or any(dtype != torch.bfloat16 or i != idx for i, (_, dtype, idx) in
                                   enumerate(per_layer_specs)):
            raise ValueError(f"Option C needs vLLM's single full-attention group (one shape, one tensor per layer): "
                             f"{sorted(shapes)}")
        shape = shapes.pop()
        full = [layer for layer in layers if not layer.sliding]
        sliding = [layer for layer in layers if layer.sliding]
        if (len(full), len(sliding)) != (FULL_LAYERS, SLIDING_LAYERS):
            raise ValueError(f"Expected {FULL_LAYERS} full and {SLIDING_LAYERS} sliding layers")
        if shape[1:] != (full[0].kv_heads, PAGE, full[0].head_dim) or any(
            (layer.kv_heads, layer.head_dim) != (4, 256) for layer in sliding
        ):
            raise ValueError(f"Unexpected allocator geometry for the full group: {shape}")
        expected = full_blocks(self.max_seq_len, users)
        if shape[0] != expected:
            raise ValueError(f"vLLM gave {shape[0]} full blocks; the ring layout sizes {expected} "
                             f"(get_max_tokens_all_users), so the KV bytes would not equal S3.2's")
        self.blocks = shape[0]
        rings = ring_pages(users)
        kv, scratch = [], {}
        for i, layer in enumerate(layers):
            pages = rings if layer.sliding else self.blocks + users
            kv.append(layer.allocate_cache(physical_pages=pages))
            scratch[i] = list(range(pages - users, pages))
        self.kv_bytes = (SLIDING_LAYERS * rings * SLIDING_PAGE_BYTES
                         + FULL_LAYERS * (self.blocks + users) * FULL_PAGE_BYTES)
        s32 = s32_pool_bytes(self.max_seq_len, users)
        if self.kv_bytes != s32:
            raise ValueError(f"Option C KV bytes {self.kv_bytes} per chip differ from S3.2's {s32}")
        tables = {i: torch.zeros(users, self.max_seq_len // PAGE, dtype=torch.int32) for i in range(len(kv))}
        state = CacheState(kv, tables, users, self.max_seq_len, scratch_pages=scratch, vllm_owned=True)
        state.full_page_scale = 1
        logger.info("Option C KV: {} full blocks (+{} scratch) x {} layers, rings {} pages x {} layers; {} B per chip "
                    "(S3.2: {} B)", self.blocks, users, FULL_LAYERS, rings, SLIDING_LAYERS, self.kv_bytes, s32)
        return state

    def _group_scratch(self):
        """scratch[g][slot]: the full group's N + slot, a sliding group's sink."""
        users = self.max_batch_size
        return [
            [(self.blocks + j) if g in self.full_groups else sink_page(j, users) for j in range(users)]
            for g in range(max(self.groups) + 1)
        ]

    # ------------------------------------------------------------------ tables

    def _single_group(self, page_table, page_tables_per_layer):
        """`[full table] * 60`: the plugin's single-group page_table, or the full layer's table of a per-layer list
        (M1's bucket warmup builds one per group)."""
        source = page_table if page_tables_per_layer is None else page_tables_per_layer[self.full_layer]
        return None if source is None else [source] * len(self.generator.model.layers)

    def _ring_rows(self, full, slots):
        """Decode rows: row r (engine slot slots[r], None if empty) maps each column up to its last live full block to
        the slot's ring and the rest to 0, which S0's sliding rule turns into the slot's sink."""
        ring = torch.zeros(full.shape, dtype=torch.int32)
        rows = [r for r, s in enumerate(slots[: full.shape[0]]) if s is not None]
        if not rows:
            return ring
        columns = torch.arange(full.shape[1])
        index = torch.tensor(rows)
        slot = torch.tensor([slots[r] for r in rows], dtype=torch.int32)[:, None]
        last = torch.where(full[index] > 0, columns, -1).amax(dim=1, keepdim=True)
        pages = self.FIRST_RING_PAGE + RING * slot + columns % RING
        ring[index] = torch.where(columns <= last, pages, 0).to(torch.int32)
        return ring

    def _tables(self, cache, page_tables_per_layer, *, slots=None, chunks=None):
        """Prefill tables: 3.2's translation (full blocks unscaled), then one ring row tensor for every sliding layer.
        Row k (engine slot slots[k]) of a chunk [start, end) names its ring pages at columns [max(start // 128,
        e - 8), e], e = (end - 1) // 128, and 0 (skipped) elsewhere. `chunks` defaults to the prefill call's own; a
        direct call without either derives e from the full row's last live block (counted: ring_rows_derived)."""
        result = super()._tables(cache, page_tables_per_layer, slots=slots)
        full = result[self.full_layer]
        ring = torch.zeros_like(full)
        if slots is not None:
            chunks = self._ring_chunks if chunks is None else chunks
            if chunks is not None and len(chunks) != len(slots):
                raise ValueError(f"{len(chunks)} ring chunks for {len(slots)} prefill rows")
            for k, slot in enumerate(slots):
                live = (full[slot] > 0).nonzero().flatten()
                if chunks is None:
                    if not len(live):
                        continue
                    lo, e = 0, int(live.max())
                    self.counters["ring_rows_derived"] += 1
                else:
                    start, end = (int(x) for x in chunks[k])
                    lo, e = start // PAGE, (end - 1) // PAGE
                    if not 0 <= lo <= e or not len(live) or e > int(live.max()):
                        raise ValueError(f"Prefill chunk [{start}, {end}) of slot {slot} ends past its full blocks")
                columns = torch.arange(max(lo, e - RING + 1), e + 1)
                ring[slot, columns] = (self.FIRST_RING_PAGE + RING * slot + columns % RING).to(ring.dtype)
        for i, layer in enumerate(self.generator.model.layers):
            if layer.sliding:
                result[i] = ring
        return result

    def _resident_tables(self, tables):
        """M1's engine tables (warmup_model_decode) with the decode ring rows: row r is slot r."""
        full = tables[self.full_layer]
        ring = self._ring_rows(full, list(range(full.shape[0])))
        tables = {i: (ring if layer.sliding else full) for i, layer in enumerate(self.generator.model.layers)}
        return super()._resident_tables(tables)

    def _write_slot_tables(self, sources, rows):
        """S0's slot table writes, with every sliding group's source replaced by the decode ring rows."""
        full = sources[self.groups[self.full_layer]]
        ring = self._ring_rows(full, rows)
        return super()._write_slot_tables([full if g in self.full_groups else ring for g in range(len(sources))], rows)

    # ------------------------------------------------------------------ entry points

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, *, page_tables_per_layer=None, start_pos=None,
                        **kwargs):
        lengths = [int(x) for x in prompt_lens]
        starts = [0] * len(lengths) if start_pos is None else [int(x) for x in start_pos]
        self._ring_chunks = list(zip(starts, lengths))
        try:
            return super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos=start_pos,
                                           page_tables_per_layer=self._single_group(page_table, page_tables_per_layer),
                                           **kwargs)
        finally:
            self._ring_chunks = None

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, *, page_tables_per_layer=None, **kwargs):
        return super().decode_forward(tokens, start_pos, page_table, kv_cache,
                                      page_tables_per_layer=self._single_group(page_table, page_tables_per_layer),
                                      **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        logger.info("Option C prefill warmup: {} ring rows derived from full tables (no chunk given)",
                    self.counters["ring_rows_derived"])
