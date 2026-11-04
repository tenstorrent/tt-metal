# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full GLM-5.3-Flash model on homelab-1's mesh (Stage 6): the device `ModelBackend` of `tt/generator.py`.

Model (HF `Glm5NextForConditionalGeneration`, text path): token ids -> embed_tokens (bf16) -> 4 identical mHC streams
[T, 4, 4096] -> 45 decoder layers -> mean of the 4 streams (`Glm5NextTextHyperHead`) -> RMSNorm (eps 1e-5) -> untied
lm_head [4096 -> 154,880] -> sampler.

Placement (the optimized multichip decoder of Stage 4, unchanged): `tt/multichip_decoder.MultichipDecoder` = byte-
weighted LAYER PIPELINE over eight 1x1 submeshes of the 2x4 Ethernet mesh, plan of record `pipeline8_ring_wrap`
(partition.json): chip 0 runs the head stage (embedding + L0 + L1.attn) and, after the closing hop 1 -> 0, the tail
stage (L41.ffn .. L44 + final norm / lm_head / sampler). Every decoder layer keeps the single-chip optimized decoder's
weights / dtypes / program configs / KV-cache layout (Stage 3 `OptimizedDecoder`); the inter-layer residual is the
[1, 1, rows, 16384] fp32 4-stream tensor handed over by MeshSocket send/recv inside each chip's trace (Stage 4). This
module adds only the model terminal around the stack, on chip 0:

* decode prologue (inside chip 0's head-stage decode trace): `ttnn.embedding(io.tokens)` -> rows 0..B-1 -> 4 copies
  -> fp32 -> the stack's persistent stage-0 input;
* decode epilogue (inside chip 0's tail-stage decode trace): stack output -> inactive rows (cur_pos == -1) zeroed ->
  mean of the 4 streams -> RMSNorm -> lm_head (bfp8, 4 DRAM width-sharded matmuls over 38,720-column vocab chunks,
  Stage 3 `DSLinear`) -> rows padded to 32 -> `io.logits` [1, 1, 32, 154880] bf16;
* split sampling (canonical, $tt-enable-tracing): a SEPARATE sampler trace on chip 0 (`models/common/sampling`
  `TTSampling`, chosen over `Sampling1D` in 06a contract.md section 7) over io.logits with `tt_out_tok = io.tokens`
  (the next decode step's token input: token feedback stays on chip 0), then a device copy into `io.sampled` (the
  buffer the generator reads);
* positions: every stage's decode trace ends with `ttnn.plus_one(cur_pos, skip_negative_entries=True)` on that
  stage's own persistent position copy (Stage 4 `plus_one=True`); the host writes positions only at request
  boundaries (fan-out to every stage's copy);
* page table: one [slots, blocks_per_seq] int32 table per MLA-holding stage (shared by that chip's MLA layers),
  rewritten in place only when the generator's page table changes;
* prefill: eager orchestration of Stage 4's traced per-(unit, bucket) prefill (`prefill_chunk_plan` "plan":
  power-of-two buckets + a < 64-token tail; Stage 4's padded "auto" policy corrupts the decode state, see
  `chunk_policy`); the chunk input is produced ON chip 0 by the embedding from a persistent per-bucket token buffer
  (padded rows zeroed by the row mask), the output rows the caller needs go through the same head: last row (one-hot
  row-select matmul into the slot's row -> head -> logits / device sampler) or all rows.

Everything is allocated before the first trace capture (allocate_kv_cache / allocate_decode_io), every program is
compiled in warmup phase 1 (prepare_prefill / prepare_decode), and captures only record.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.generator import (
    BLOCK_SIZE,
    MAX_DECODE_SLOTS,
    MODEL_PREFILL_ALIGN,
    PREFILL_CHUNK,
    SEED_SKIP,
    SEED_UNSEEDED,
    DecodeIO,
    ModelBackend,
    blocks_per_seq,
)
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import (
    HC_WIDTH,
    MODEL_ALIGN,
    MultichipDecoder,
    StagePlan,
    chip_coords,
    dram_allocated,
    model_chunk_plan,
    parse_units,
    plan_from_partition,
    prefill_chunk_plan,
    validate_plan,
)
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import VOCAB_SIZE, SamplingParams

DRAM = ttnn.DRAM_MEMORY_CONFIG
HIDDEN = 4096
HC_MULT = 4
TOK_ROWS = 32  # ttnn.sampling users per call / token buffer width (tt_transformers [1, 1, 1, 32] uint32 convention)
LOGITS_WIDTH = VOCAB_SIZE  # sampler-ready logits width (no padded columns)
LM_HEAD_SPLITS = 4  # vocab chunks of the DRAM-sharded lm_head (Terminal)
DEVICE_SEED_SKIP = 0xFFFFFFFF  # ttnn.manual_seed: keep the core's PRNG state
WS = os.environ.get("GLM53_WS", "/home/alex/mpi-shfs/tenstorrent/glm53-flash-bringup")
DEFAULT_PARTITION = os.path.join(WS, "doc/ttnn/04-multichip-decoder/partition.json")
DEFAULT_PARTITION_KEY = "pipeline8_ring_wrap"


def _dense_exact_max_kv() -> int:
    from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV

    return DSA_DENSE_EXACT_MAX_KV


# ================================================================================================ buffers / caches
class DevBuf:
    """One logical DecodeIO buffer: the device tensor(s) behind it (several for per-stage copies) + a name the backend
    dispatches host writes / reads on. `tensors[0]` is the canonical copy (chip 0)."""

    def __init__(self, name: str, tensors: list, rows: int):
        self.name = name
        self.tensors = tensors
        self.rows = rows

    @property
    def shape(self):
        return (self.rows,)

    def __repr__(self):
        return f"DevBuf({self.name}, copies={len(self.tensors)})"


@dataclass
class PipelineKVCache:
    """Opaque KV/state handle the generator passes back: the pipeline's micro-batch group `group` (KDA state + conv
    rows per KDA layer, paged MLA latent caches per MLA layer, all on the chips that run those layers)."""

    max_batch: int
    max_seq_len: int
    blocks_per_seq: int
    num_blocks: int
    group: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class _Trace:
    kind: str  # "decode" | "sampler"
    key: Any
    ids: list  # [(mesh, trace_id)]


# ================================================================================================ pipeline + terminal
class PipelineStack(MultichipDecoder):
    """Stage 4's MultichipDecoder with the model terminal hooked into the first and last stages' decode programs:
    `decode_prologue(g)` runs at the start of stage 0's decode sequence (embedding into the stack input), and
    `decode_epilogue(g)` at the end of the last stage's sequence (head into the sampler-ready logits). Both are part
    of the per-chip decode traces (warmed by the same eager step), so the whole token -> logits path is traced."""

    decode_prologue = None
    decode_epilogue = None
    # B > 1 (contract.md section 5, Stage 2 review C-9): rows with cur_pos == -1 are zeroed at every unit boundary and
    # at every MLA attention output, and enter every MoE FFN as a copy of row 0's FFN input (so they add no expert to
    # the fused_experts union and carry no NaN / Inf); the epilogue hands the sampler zero rows for them.
    mask_inactive = False

    def _stage_decode(self, k: int, g_idx: int):
        st = self.stages[k]
        g = st.groups[g_idx]
        if k == 0 and self.decode_prologue is not None:
            self.decode_prologue(g_idx)
        if not self.mask_inactive:
            super()._stage_decode(k, g_idx)
        else:
            if st.link_in is not None:
                st.link_in.recv(g.streams_in)
            m32 = _active_mask(g.cur_pos, self.max_batch)
            mb = ttnn.typecast(m32, ttnn.bfloat16)
            x = ttnn.where(m32, g.streams_in, 0.0)
            out = self._decode_units_masked(st, g, x, m32, mb)
            if st.link_out is not None:
                st.link_out.send(out)
            if g.out_buf is not None:
                ttnn.copy(out, g.out_buf)
            ttnn.deallocate(out)
            ttnn.deallocate(m32)
            ttnn.deallocate(mb)
            if st.plus_one:
                ttnn.plus_one(g.cur_pos, skip_negative_entries=True)
        if k == len(self.stages) - 1 and self.decode_epilogue is not None:
            self.decode_epilogue(g_idx)

    @staticmethod
    def _decode_units_masked(st, g, x, m32, mb):
        """PipelineStage.decode_units with inactive rows zeroed at every unit output and at every MLA attention
        output (the same layer programs: `layer.forward` with the decoder's own attention callables)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import decode_attention, forward_unit

        for u in st.plan.units:
            dec = st.decoders[u.layer]
            attn = None
            if u.part != "ffn":
                base = decode_attention(dec, g.caches[u.layer], g.page_table, g.cur_pos)
                attn = base if dec.is_kda else (lambda h, base=base: _where_rows(mb, base(h)))
            moe = u.part != "attn" and dec.spec.mlp == "moe"
            if moe:  # inactive rows route like row 0 (06a contract: their ids stay out of the fused_experts union)
                dec.layer._ffn = _row0_ffn(type(dec.layer)._ffn.__get__(dec.layer), m32, mb)
            try:
                if u.part == "full":
                    y = dec.layer.forward(x, attn, decode=True)
                else:
                    y = forward_unit(dec, x, attn, decode=True, part=u.part)
            finally:
                if moe:
                    del dec.layer._ffn
            ttnn.deallocate(x)
            x = ttnn.where(m32, y, 0.0)
            ttnn.deallocate(y)
        return x


def _row0_ffn(ffn, m32, mb):
    """The layer's FFN (router + experts) with the rows of inactive slots replaced by row 0's FFN input: a zero row would
    route on the router bias alone to one fixed expert set that joins the fused_experts union of the active rows
    (Stage 9 2026-10-19 03:16Z: a lone request on the 32-slot trace streamed 8..16 experts per MoE layer, 126-165 ms per
    token vs 75.5 on the 1-slot trace); row 0's copy adds no expert when row 0 is live (vLLM compacts live requests into
    the lowest slots: the plugin remaps slots at request boundaries, `copy_slot`); with row 0 inactive the rows route
    as before (zero input: a perf effect only). The inactive rows' FFN outputs are discarded (zeroed at the unit
    boundary); the active rows' input is unchanged (where-select)."""

    def run(h, decode):
        rows, width = h.shape[-2], h.shape[-1]
        r0 = ttnn.slice(h, (0, 0, 0, 0), (1, 1, 1, width))
        rep = ttnn.repeat(r0, (1, 1, rows, 1))
        ttnn.deallocate(r0)
        h2 = ttnn.where(m32 if h.dtype == ttnn.float32 else mb, h, rep)
        ttnn.deallocate(rep)
        y = ffn(h2, decode)
        ttnn.deallocate(h2)
        return y

    return run


def _where_rows(mask, t):
    y = ttnn.where(mask, t, 0.0)
    ttnn.deallocate(t)
    return y


def _active_mask(cur_pos, rows: int):
    """cur_pos [B] int32 -> [1, 1, B, 1] fp32 TILE: 1.0 where cur_pos >= 0."""
    cp = ttnn.to_layout(ttnn.reshape(cur_pos, (1, 1, rows, 1)), ttnn.TILE_LAYOUT)
    return ttnn.typecast(ttnn.ge(ttnn.typecast(cp, ttnn.float32), 0.0), ttnn.float32)


def _zero_inactive_rows(x, cur_pos, rows: int):
    m = _active_mask(cur_pos, rows)
    y = ttnn.where(m, x, 0.0)
    ttnn.copy(y, x)
    ttnn.deallocate(y)
    ttnn.deallocate(m)


class Terminal:
    """Chip-0 model terminal: embedding table (bf16 ROW_MAJOR), final RMSNorm weight, lm_head (bfp8, DRAM
    width-sharded `DSLinear` per vocab chunk)."""

    def __init__(self, mesh, cfg, *, cache_dir: str, manifest_path: str, ocfg=None):
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
            DSLinear,
            FdTileCacheSource,
            OptimizedDecoderConfig,
        )
        from models.autoports.zai_org_glm_5_3_flash.tt.precision import hifi2_fp32, hifi4_fp32

        self.mesh = mesh
        self.cfg = cfg
        self.eps = float(cfg.rms_norm_eps)
        src = FdTileCacheSource(cache_dir, manifest_path)
        ent = lambda name: src.tensors[name]
        t0 = time.perf_counter()
        self.embed_w = src._load(ent("embed_tokens.weight"), mesh)  # [V, 4096] bf16 ROW_MAJOR
        nw = src._load(ent("norm.weight"), mesh)
        self.norm_w = ttnn.reshape(nw, (1, 1, 1, HIDDEN))
        lm = src._load(ent("lm_head.weight"), mesh)  # [4096, V] bfp8 TILE
        self.ocfg = ocfg or OptimizedDecoderConfig()
        # lm_head as LM_HEAD_SPLITS DRAM-sharded decode matmuls over vocab chunks of 38,720 columns (1,210 tiles): a DS
        # matmul gives every DRAM reader N / (8 banks x readers) output tiles and allows <= 3 readers per bank, so the
        # whole 154,880-wide head (202 tiles per reader at 3 readers) overflows L1 (mock 2026-10-18 21:44Z: 2.98 MB of
        # CBs at 2 readers); per chunk: padded to 1,224 tiles = 24 readers x 51 tiles. Chunk outputs are sliced back to
        # 38,720 columns and concatenated: logits [.., 154,880] with no padded columns.
        self.ocfg.ds_readers_per_role = ";".join(filter(None, [self.ocfg.ds_readers_per_role, "lm_head:3"]))
        self.ckc_head = hifi2_fp32(mesh)
        self.ckc_norm = hifi4_fp32(mesh)
        w = VOCAB_SIZE // LM_HEAD_SPLITS
        self.lm_heads = []
        for i in range(LM_HEAD_SPLITS):
            part = ttnn.slice(lm, (0, i * w), (HIDDEN, (i + 1) * w), memory_config=DRAM)
            ds = DSLinear(mesh, part, role="lm_head", cfg=self.ocfg, ckc=self.ckc_head, dtype=ttnn.bfloat16)
            if ds.weight is not part:
                ttnn.deallocate(part)
            self.lm_heads.append(ds)
        ttnn.deallocate(lm)
        self.load_s = time.perf_counter() - t0

    # ---- pieces shared by decode and prefill
    def embed_streams(self, tokens_2d, rows: int):
        """tokens_2d [1, R] uint32 ROW_MAJOR (R >= rows) -> [1, 1, rows, 16384] fp32 (4 identical streams)."""
        e = ttnn.embedding(tokens_2d, self.embed_w, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)  # [1, R, 4096]
        r = e.shape[-2]
        e4 = ttnn.reshape(e, (1, 1, r, HIDDEN))
        if r != rows:
            s = ttnn.slice(e4, (0, 0, 0, 0), (1, 1, rows, HIDDEN), memory_config=DRAM)
            ttnn.deallocate(e4)
            e4 = s
        x = ttnn.concat([e4] * HC_MULT, dim=-1, memory_config=DRAM)
        ttnn.deallocate(e4)
        y = ttnn.typecast(x, ttnn.float32)
        ttnn.deallocate(x)
        return y

    def hidden(self, streams):
        """[1, 1, R, 16384] fp32 -> mean of the 4 streams -> RMSNorm -> [1, 1, R, 4096] bf16."""
        r = streams.shape[-2]
        parts = [ttnn.slice(streams, (0, 0, 0, i * HIDDEN), (1, 1, r, (i + 1) * HIDDEN)) for i in range(HC_MULT)]
        acc = ttnn.add(parts[0], parts[1])
        for p in parts[2:]:
            acc2 = ttnn.add(acc, p)
            ttnn.deallocate(acc)
            acc = acc2
        for p in parts:
            ttnn.deallocate(p)
        mean = ttnn.multiply(acc, 1.0 / HC_MULT)
        ttnn.deallocate(acc)
        h = ttnn.rms_norm(mean, weight=self.norm_w, epsilon=self.eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(mean)
        hb = ttnn.typecast(h, ttnn.bfloat16)
        ttnn.deallocate(h)
        return hb

    def logits_decode(self, streams32):
        """[1, 1, 32, 16384] fp32 -> [1, 1, 32, LOGITS_WIDTH] bf16 (DRAM-sharded decode matmuls)."""
        h = self.hidden(streams32)
        parts = [ds(h, out_memory_config=DRAM) for ds in self.lm_heads]
        ttnn.deallocate(h)
        y = ttnn.concat(parts, dim=-1, memory_config=DRAM)
        for p in parts:
            ttnn.deallocate(p)
        return y

    def logits_prefill(self, streams):
        """[1, 1, P, 16384] fp32 (any P) -> [1, 1, P, LOGITS_WIDTH] bf16 (2D multicast program on the DS weight)."""
        h = self.hidden(streams)
        parts = [ds.prefill(h, memory_config=DRAM, dtype=ttnn.bfloat16) for ds in self.lm_heads]
        ttnn.deallocate(h)
        y = ttnn.concat(parts, dim=-1, memory_config=DRAM)
        for p in parts:
            ttnn.deallocate(p)
        return y


# ================================================================================================ backend
class Glm53PipelineBackend(ModelBackend):
    """ModelBackend over the layer pipeline + chip-0 terminal (module docstring)."""

    vocab_size = VOCAB_SIZE
    logits_width = LOGITS_WIDTH
    block_size = BLOCK_SIZE
    prefill_chunk_tokens = PREFILL_CHUNK
    prefill_align = MODEL_PREFILL_ALIGN
    continuation_align = BLOCK_SIZE
    # capture_decode: every width replays the one full-width pipeline trace (KDA decodes all max_batch slots of its
    # batched state), so the generator keeps a single width and serving never compacts slots for nothing
    # (tt/per_width.py PerWidthBackend names its own trace widths)
    width_bucketed_decode = False
    # copy_slot re-seeds the destination's device sampler row (the per-core PRNG state cannot be copied)
    copy_slot_keeps_sampler_rng = False

    def __init__(self, stack: PipelineStack, terminal: Terminal, *, max_batch: int = MAX_DECODE_SLOTS, hf_config=None):
        self.m = stack
        self.term = terminal
        self.hf_config = hf_config
        self.max_batch = max_batch
        self.supported_context = _dense_exact_max_kv()  # min over layer kinds (sparse_dsa dense-exact regime)
        self.layer_kinds = tuple(stack.decoder(l).spec.kind for l in stack.layers)
        self.head_mesh = stack.stages[0].mesh
        if stack.stages[-1].mesh is not self.head_mesh:
            raise NotImplementedError(
                "the full model needs the embedding and the head on the same chip (wrap plan: token feedback on "
                "chip 0); this pipeline ends on another chip"
            )
        if terminal.mesh is not self.head_mesh:
            raise ValueError("the terminal must live on the pipeline's head chip")
        self.kv: Optional[PipelineKVCache] = None
        self.io: Optional[DecodeIO] = None
        self.sampler = None  # TTSampling (decode, all 32 rows)
        self.psampler = None  # TTSampling (prefill first token, scratch rows)
        self.traces: list[_Trace] = []
        self.captured_stack = False
        self.counters = {k: 0 for k in ("writes", "reads", "seed_skip_writes", "prefill_subchunks")}
        self._pending_seed_skip = False
        self._kpt = [None, None, None]  # host k / p / temp of the decode sampler (32 rows)
        self.ptok: dict = {}  # prefill token input per model bucket [1, P] uint32
        self.psel: dict = {}  # prefill last-row selector per bucket [1, 1, 32, P] bf16
        self.plogits = None  # prefill scratch logits [1, 1, 32, W] bf16
        self.ptoken = None  # prefill sampler output [1, 1, 1, 32] uint32
        self.dec_streams32 = None
        # prefill chunk plan inside one generator chunk (<= 2048 tokens): "plan" = power-of-two buckets + one < 64-row
        # tail (KDA layers run round_up(n, 32) rows of it). NOT Stage 4's "auto" / "padded" F1 policy: one chunk padded to
        # its bucket holds fully padded 32-row KDA recurrence chunks (forbidden by the Stage 1 contract) and the decode
        # state after such a prefill is corrupt (run 3 20261019T001656Z: AIME24 161 tokens padded to 256 -> prefill
        # logits right, decode garbage from the first step; exact-bucket prompts unaffected).
        self.chunk_policy = "plan"

    # ---------------------------------------------------------------------------------------------- state
    def allocate_kv_cache(self, *, max_batch: int, max_seq_len: int, num_blocks: Optional[int] = None):
        if not 1 <= max_batch <= self.max_batch:
            raise ValueError(f"max_batch {max_batch} outside [1, {self.max_batch}]")
        if max_seq_len > self.supported_context:
            raise ValueError(f"max_seq_len {max_seq_len} > supported context {self.supported_context}")
        if self.kv is not None:
            if self.traces:
                raise RuntimeError("one KV cache per pipeline: release the traces of the current one first")
            self.m.free()
        msl = {l: max_seq_len for l in self.m.layers}
        self.m.allocate(max_batch=max_batch, max_seq_len=msl, microbatches=1, plus_one=True)
        self.m.mask_inactive = max_batch > 1
        bps = self.m.blocks_per_seq() or blocks_per_seq(max_seq_len)
        default_blocks = max_batch * bps
        if num_blocks is not None and num_blocks < bps:
            raise ValueError(
                f"num_blocks {num_blocks} < blocks_per_seq {bps}: paged_update_cache needs a pool of at least one "
                f"sequence's page-table width (max_seq_len {max_seq_len})"
            )
        if num_blocks is not None and num_blocks != default_blocks:
            self._resize_latent_pools(num_blocks)
        self.kv = PipelineKVCache(max_batch, max_seq_len, bps, num_blocks or default_blocks)
        # prefill terminal buffers (persistent, before any capture)
        for p in self.m.buckets():
            if p % MODEL_ALIGN:
                continue
            self.ptok[p] = ttnn.from_torch(
                torch.zeros(1, p, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.head_mesh,
                memory_config=DRAM,
            )
            self.psel[p] = ttnn.from_torch(
                torch.zeros(1, 1, TOK_ROWS, p),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.head_mesh,
                memory_config=DRAM,
            )
        return self.kv

    def _resize_latent_pools(self, num_blocks: int):
        """Serving pool of `num_blocks` 64-token pages per MLA layer (vLLM block budget) instead of slots x bps."""
        for st in self.m.stages:
            for g in st.groups:
                for l in st.mla_layers:
                    c = g.caches[l]
                    ttnn.deallocate(c.latent)
                    c.latent = st.decoders[l].layer.attn.allocate_cache(num_blocks)

    def _mla_stages(self):
        return [st for st in self.m.stages if st.mla_layers]

    def latent_pages(self, kv_cache) -> dict:
        """{decoder layer: its paged latent [num_blocks, 1, 64, 512] bf16 on the chip that runs it} (KV connectors)."""
        return {l: st.groups[kv_cache.group].caches[l].latent for st in self._mla_stages() for l in st.mla_layers}

    def allocate_decode_io(self, kv_cache) -> DecodeIO:
        from models.common.sampling.tt_sampling import TTSampling

        B = kv_cache.max_batch
        g = kv_cache.group
        mesh = self.head_mesh
        tokens = ttnn.from_torch(
            torch.zeros(1, 1, 1, TOK_ROWS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=DRAM,
        )
        sampled = ttnn.from_torch(
            torch.zeros(1, 1, 1, TOK_ROWS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=DRAM,
        )
        logits = ttnn.from_torch(
            torch.zeros(1, 1, TOK_ROWS, LOGITS_WIDTH),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
        )
        self.plogits = ttnn.from_torch(
            torch.zeros(1, 1, TOK_ROWS, LOGITS_WIDTH),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
        )
        self.ptoken = ttnn.from_torch(
            torch.zeros(1, 1, 1, TOK_ROWS, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=DRAM,
        )
        args = SimpleNamespace(
            vocab_size=VOCAB_SIZE,
            padded_vocab_size=LOGITS_WIDTH,
            max_batch_size=TOK_ROWS,
            max_top_k=32,
            cluster_shape=(1, 1),
            sampling_dp=1,
        )
        self.sampler = TTSampling(mesh_device=mesh, tt_ccl=None, args=args)
        self.psampler = TTSampling(mesh_device=mesh, tt_ccl=None, args=args)
        self._kpt = [torch.ones(TOK_ROWS, dtype=torch.int64), torch.zeros(TOK_ROWS), torch.ones(TOK_ROWS)]
        stages = self.m.stages
        io = DecodeIO(
            tokens=DevBuf("tokens", [tokens], B),
            cur_pos=DevBuf("cur_pos", [st.groups[g].cur_pos for st in stages], B),
            page_table=DevBuf("page_table", [st.groups[g].page_table for st in self._mla_stages()], B),
            logits=DevBuf("logits", [logits], B),
            sampled=DevBuf("sampled", [sampled], B),
            k=DevBuf("k", [self.sampler.k_tensor], B),
            p=DevBuf("p", [self.sampler.p_tensor], B),
            temp=DevBuf("temp", [self.sampler.temp_tensor], B),
            seeds=DevBuf("seeds", [self.sampler.seeds_tt_tensor], B),
        )
        # decode head scratch input: the stack output padded to 32 rows (persistent; B < 32 rows are zero)
        self.dec_streams32 = ttnn.from_torch(
            torch.zeros(1, 1, TOK_ROWS, HC_WIDTH),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
        )
        rm = torch.zeros(1, 1, TOK_ROWS, 1)
        rm[0, 0, :B] = 1.0
        self.rowmask32 = ttnn.from_torch(
            rm, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        )
        self.io = io
        self.m.decode_prologue = lambda gi: self._prologue(io, gi)
        self.m.decode_epilogue = lambda gi: self._epilogue(io, gi)
        return io

    # ---- decode terminal (recorded inside the head / tail stage traces)
    def _prologue(self, io: DecodeIO, gi: int):
        B = io.tokens.rows
        tok = ttnn.reshape(io.tokens.tensors[0], (1, TOK_ROWS))
        x = self.term.embed_streams(tok, B)
        ttnn.copy(x, self.m.stages[0].groups[gi].streams_in)
        ttnn.deallocate(x)

    def _epilogue(self, io: DecodeIO, gi: int):
        """Stack output [1, 1, B, 16384] -> 32 head rows (rows >= B and inactive rows exactly 0: the sampler reads all
        32 rows and needs finite logits) -> io.logits. NOTE: ttnn.pad inside the tile's physical padding returns a
        VIEW of its input (same buffer; mock check 2026-10-18 22:20Z), so the padded view of the persistent stack
        output is never deallocated."""
        B = io.tokens.rows
        last = self.m.stages[-1]
        g = last.groups[gi]
        out = g.out_buf  # [1, 1, B, 16384] fp32, persistent
        xp = ttnn.pad(out, ((0, 0), (0, 0), (0, TOK_ROWS - B), (0, 0)), 0.0) if B < TOK_ROWS else out
        if B > 1:
            am = _active_mask(g.cur_pos, B)  # after this stage's plus_one: inactive rows stay -1
            amp = ttnn.pad(am, ((0, 0), (0, 0), (0, TOK_ROWS - B), (0, 0)), 0.0) if B < TOK_ROWS else am
            m = ttnn.multiply(amp, self.rowmask32)
            ttnn.deallocate(am)
        else:
            m = self.rowmask32
        xm = ttnn.where(m, xp, 0.0)
        if m is not self.rowmask32:
            ttnn.deallocate(m)
        ttnn.copy(xm, self.dec_streams32)
        ttnn.deallocate(xm)
        y = self.term.logits_decode(self.dec_streams32)
        ttnn.copy(y, io.logits.tensors[0])
        ttnn.deallocate(y)

    # ---------------------------------------------------------------------------------------------- host I/O
    def write(self, buf: DevBuf, value: torch.Tensor) -> None:
        self.counters["writes"] += 1
        v = torch.as_tensor(value)
        name = buf.name
        if name == "tokens":
            t = torch.zeros(1, 1, 1, TOK_ROWS, dtype=torch.int32)
            t.view(-1)[: v.numel()] = v.reshape(-1).to(torch.int32).clamp(min=0)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), buf.tensors[0]
            )
        elif name == "cur_pos":
            host = ttnn.from_torch(v.reshape(-1).to(torch.int32), dtype=ttnn.int32)
            for t in buf.tensors:
                ttnn.copy_host_to_device_tensor(host, t)
        elif name == "page_table":
            pt = v.to(torch.int32).reshape(buf.rows, -1)
            host = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for t in buf.tensors:
                ttnn.copy_host_to_device_tensor(host, t)
        elif name in ("k", "p", "temp"):
            i = ("k", "p", "temp").index(name)
            full = self._kpt[i].clone()
            vals = v.reshape(-1)
            if name == "temp":  # the generator writes T; ttnn.sampling multiplies by 1/T (format_sampling_params)
                vals = torch.where(vals > 0, 1.0 / vals.clamp(min=1e-6), torch.ones_like(vals))
            full[: vals.numel()] = vals.to(full.dtype)
            self._kpt[i] = full
            k, p, temp = self._kpt
            self.sampler.reset_params(k.clone(), p.tolist(), temp.clone())
        elif name == "seeds":
            self._write_seeds(v.reshape(-1))
        else:
            raise KeyError(name)

    def _seed_values(self, rows: torch.Tensor) -> list[int]:
        out = [DEVICE_SEED_SKIP] * TOK_ROWS
        for s, x in enumerate(rows.tolist()):
            if x == SEED_SKIP:
                continue
            if x == SEED_UNSEEDED:
                out[s] = int.from_bytes(os.urandom(4), "little") % 0xFFFFFFF0 + 1
            else:
                out[s] = (int(x) * 2654435761 + 0x9E3779B9) % 0xFFFFFFF0 + 1  # explicit seed -> device seed
        return out

    def _write_seeds(self, rows: torch.Tensor):
        """Request boundary: re-seed the listed rows on the NEXT sampler step (ttnn.manual_seed inside the sampler
        trace reads this buffer every step), then SKIP from the step after (one extra write per admission)."""
        vals = self._seed_values(rows)
        self._put_seeds(self.sampler, vals)
        self._pending_seed_skip = any(v != DEVICE_SEED_SKIP for v in vals)

    @staticmethod
    def _put_seeds(sampler, vals):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor(vals, dtype=torch.int64).to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            sampler.seeds_tt_tensor,
        )

    def _after_sampler_step(self):
        if self._pending_seed_skip:
            self._put_seeds(self.sampler, [DEVICE_SEED_SKIP] * TOK_ROWS)
            self._pending_seed_skip = False
            self.counters["seed_skip_writes"] += 1

    def read(self, buf: DevBuf) -> torch.Tensor:
        self.counters["reads"] += 1
        return self._host_rows(buf, ttnn.to_torch(buf.tensors[0]))

    def read_async(self, buf: DevBuf):
        """Non-blocking copy of the canonical (chip 0) tensor, ordered on chip 0's CQ 0 behind the sampler trace that
        wrote it; the event marks its completion (the vLLM async split waits on it before finish_read)."""
        self.counters["reads"] += 1
        t = buf.tensors[0]
        host = t.cpu(blocking=False)
        return host, [ttnn.record_event(self.head_mesh, 0)]

    def finish_read(self, buf: DevBuf, pending) -> torch.Tensor:
        return self._host_rows(buf, ttnn.to_torch(pending))

    def _host_rows(self, buf: DevBuf, t: torch.Tensor) -> torch.Tensor:
        if buf.name in ("tokens", "sampled"):
            return t.reshape(-1)[: buf.rows].to(torch.int64)
        if buf.name == "logits":
            return t.reshape(TOK_ROWS, -1)[: buf.rows].float()
        if buf.name in ("cur_pos",):
            return t.reshape(-1)[: buf.rows].to(torch.int64)
        return t

    def set_page_table_host(self, kv_cache, page_table: torch.Tensor) -> None:
        pt = page_table.to(torch.int32).reshape(kv_cache.max_batch, -1)
        for st in self._mla_stages():
            for l in st.mla_layers:
                st.groups[kv_cache.group].caches[l].page_table_host = pt

    def reset_slot(self, kv_cache, slot: int) -> None:
        self.m.reset_slot(slot, group=kv_cache.group)

    def copy_slot(self, kv_cache, src: int, dst: int) -> None:
        """vLLM slot remap (request boundary): KDA recurrent state S through the traced-prefill scratch state (the
        slot read programs are compiled for every slot in prepare_prefill) + fill_cache into `dst`; the conv rows C
        (users as rows of a bf16 TILE tensor) by a host round trip. MLA pages follow the page-table row."""
        for st in self.m.stages:
            g = st.groups[kv_cache.group]
            for l in st.state_layers:
                dec = st.decoders[l]
                if not dec.is_kda:
                    continue
                c = g.caches[l]
                dec._scratch_state_in(c, src)
                ttnn.fill_cache(c.kda.S, c.kda.prefill_S, dst)
                host = ttnn.to_torch(c.kda.C)
                host[:, :, dst] = host[:, :, src]
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(host, dtype=c.kda.C.dtype, layout=ttnn.TILE_LAYOUT), c.kda.C
                )
        # the decode sampler's per-slot RNG is device state (seeded at admission, then advanced in-trace): re-seed the
        # destination row like a fresh request (explicit seeds are reproducible per absolute position only on host)
        rows = torch.full((TOK_ROWS,), SEED_SKIP, dtype=torch.int64)
        rows[dst] = SEED_UNSEEDED
        self._write_seeds(rows)

    def reset_all(self, kv_cache) -> None:
        for s in range(kv_cache.max_batch):
            self.m.reset_slot(s, group=kv_cache.group)

    # ---------------------------------------------------------------------------------------------- prefill
    def _embed_chunk(self, ids: torch.Tensor, n: int, phys: int, dst, tail: bool):
        t = torch.zeros(1, phys, dtype=torch.int32)
        t[0, :n] = ids[:n].to(torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.ptok[phys]
        )
        x = self.term.embed_streams(self.ptok[phys], phys)
        if tail:  # rows >= n exactly 0 (the zero-padded chunk contract of the stack)
            st = self.m.stages[0]
            m = st.pmask[phys]
            v = torch.zeros(1, 1, phys, 1)
            v[0, 0, :n] = 1.0
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), m)
            y = ttnn.where(m, x, 0.0)
            ttnn.deallocate(x)
            x = y
        ttnn.copy(x, dst)
        ttnn.deallocate(x)

    def _run_prefill(self, ids: torch.Tensor, *, slot: int, start: int, group: int, eager: bool, on_output):
        """Stage 4 `_prefill_traced` with the chunk input embedded on chip 0. on_output(off, n, phys, y) consumes the
        last stage's output buffer of every model sub-chunk before the next sub-chunk overwrites it."""
        m = self.m
        first = m.stages[0]
        n_tok = int(ids.numel())
        plan = prefill_chunk_plan(n_tok, m.buckets(), self.chunk_policy)
        cov = self._covered_tokens(slot, group)
        if any(start + off + phys > cov for off, n, phys in plan):
            # a padded chunk would write MLA latent rows past the slot's page coverage (a caller row repeats its last
            # block there): keep the power-of-two plan, whose < 64-row tail stays in the request's own last page
            plan = model_chunk_plan(n_tok, m.buckets())
        for off, n, phys in plan:
            if phys not in first.pbufs:
                raise ValueError(f"prefill chunk of {phys} rows: no prepared bucket")
            self._embed_chunk(ids[off : off + n], n, phys, first.pbufs[phys][0], tail=n < phys)
            y = None
            for k, st in enumerate(m.stages):
                if k > 0:
                    src = m.stages[k - 1]
                    src_buf = src.pbufs[phys][len(src.plan.units) % 2]
                    if m.handoff == "host":
                        from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import host_handoff

                        host_handoff(src_buf, st.pbufs[phys][0])
                    else:
                        src.link_out.send(src_buf)
                        st.link_in.recv(st.pbufs[phys][0])
                y = m._stage_prefill_chunk(
                    k, group, phys=phys, n_valid=n, chunk_start=start + off, user_slot=slot, eager=eager
                )
            self.counters["prefill_subchunks"] += 1
            on_output(off, n, phys, y)

    def _covered_tokens(self, slot: int, group: int) -> int:
        """Positions the slot's page-table row addresses with distinct blocks (the generator pads a caller row by
        repeating its last entry): 64 x the index of the first repeat (inf without MLA layers)."""
        for st in self._mla_stages():
            row = st.groups[group].caches[st.mla_layers[0]].page_table_host[slot].tolist()
            n = next((i for i in range(1, len(row)) if row[i] == row[i - 1]), len(row))
            return n * BLOCK_SIZE
        return 1 << 62

    def _select_row(self, y, phys: int, row: int, slot: int):
        """One-hot row select: [1,1,32,P] x [1,1,P,16384] -> row `slot` = y[row], other rows 0."""
        sel = torch.zeros(1, 1, TOK_ROWS, phys)
        sel[0, 0, slot, row] = 1.0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.psel[phys]
        )
        return ttnn.matmul(
            self.psel[phys], y, dtype=ttnn.float32, compute_kernel_config=self.term.ckc_norm, memory_config=DRAM
        )

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None):
        ids = torch.as_tensor(tokens).reshape(-1)[:n_valid]
        out: dict = {}
        eager = not self.m.prefill_captured
        n_tot = n_valid

        def on_output(off, n, phys, y):
            if logits_rows == "all":
                lg = self.term.logits_prefill(y)
                out.setdefault("rows", []).append(ttnn.to_torch(lg).float().reshape(phys, -1)[:n])
                ttnn.deallocate(lg)
            elif logits_rows == "last" and off + n == n_tot:
                h = self._select_row(y, phys, n - 1, slot)
                lg = self.term.logits_decode(h)
                ttnn.deallocate(h)
                ttnn.copy(lg, self.plogits)
                ttnn.deallocate(lg)
                out["last"] = True

        self._run_prefill(ids, slot=slot, start=chunk_start, group=kv_cache.group, eager=eager, on_output=on_output)
        if logits_rows == "none":
            return None
        if logits_rows == "all":
            rows = torch.cat(out["rows"])
            res = torch.zeros(round_up(n_valid, MODEL_PREFILL_ALIGN), rows.shape[-1])
            res[:n_valid] = rows
            return res
        if sample is None:
            return ttnn.to_torch(self.plogits).float().reshape(TOK_ROWS, -1)[slot : slot + 1]
        tok = self._prefill_sample(slot, sample)
        self._write_token_row(io, slot, tok)
        return tok

    def _prefill_sample(self, slot: int, params: SamplingParams) -> int:
        k, p, temp = params.device_values()
        kk = torch.ones(TOK_ROWS, dtype=torch.int64)
        pp = [0.0] * TOK_ROWS
        tt = torch.ones(TOK_ROWS)
        kk[slot], pp[slot], tt[slot] = k, p, (1.0 / temp if temp > 0 else 1.0)
        self.psampler.reset_params(kk, pp, tt)
        seeds = [DEVICE_SEED_SKIP] * TOK_ROWS
        seeds[slot] = self._seed_values(torch.tensor([SEED_UNSEEDED if params.seed is None else params.seed]))[0]
        self._put_seeds(self.psampler, seeds)
        self.psampler(self.plogits, tt_out_tok=self.ptoken)
        return int(ttnn.to_torch(self.ptoken).reshape(-1)[slot])

    def _write_token_row(self, io: DecodeIO, slot: int, tok: int):
        """io.tokens[slot] = tok (request boundary: read the 32-row buffer, set the row, write it back)."""
        cur = ttnn.to_torch(io.tokens.tensors[0]).reshape(-1).to(torch.int64)
        cur[slot] = tok
        self.write(io.tokens, cur[: io.tokens.rows])

    def prepare_prefill(self, kv_cache) -> None:
        """Warmup phase 1 (prefill): one eager pass of every model bucket and both tail shapes through the whole
        pipeline (Stage 4 prepare_prefill, inputs embedded on chip 0), the last-row and all-rows heads of every
        bucket, the prefill sampler, and the slot-indexed KDA scratch-state reads of every slot."""
        m = self.m
        g = kv_cache.group
        lens = sorted({b for b in m.buckets() if b % MODEL_ALIGN == 0} | {20, 40})
        pad = torch.zeros(max(lens), dtype=torch.int64)

        def heads(off, n, phys, y):
            h = self._select_row(y, phys, n - 1, 0)
            lg = self.term.logits_decode(h)
            ttnn.deallocate(h)
            ttnn.copy(lg, self.plogits)
            ttnn.deallocate(lg)
            lg = self.term.logits_prefill(y)
            ttnn.deallocate(lg)

        for n in lens:
            self._run_prefill(pad[:n], slot=0, start=0, group=g, eager=True, on_output=heads)
        # padded chunks (chunk policy "auto" / "padded": one chunk padded to its bucket): the chip-0 embedding row mask
        # and every chip's unit-boundary row mask at EVERY bucket, compiled before any capture (Stage 4 1854f5f4d6a)
        first = m.stages[0]
        for p in sorted(b for b in m.buckets() if b % MODEL_ALIGN == 0):
            self._embed_chunk(pad[: p - 1], p - 1, p, first.pbufs[p][0], tail=True)
        done = set()
        for st in m.stages:
            if tuple(st.plan.coord) in done:
                continue
            done.add(tuple(st.plan.coord))
            for p in st.pmask:
                if p in st.pbufs:
                    m._mask_rows(st, st.pbufs[p][0], p, p - 1)
        self._prefill_sample(0, SamplingParams())
        self._prefill_sample(0, SamplingParams(temperature=1.0, top_p=0.95))
        for st in m.stages:
            gg = st.groups[g]
            for l in st.state_layers:
                dec, cache = st.decoders[l], gg.caches[l]
                if dec.is_kda:
                    for s in range(cache.max_batch):
                        dec._scratch_state_in(cache, s)
                    ttnn.fill_cache(cache.kda.S, cache.kda.prefill_S, 0)
        self.m.reset_slot(0, group=g)
        self.m.synchronize()

    # ---------------------------------------------------------------------------------------------- decode
    def prepare_decode(self, io: DecodeIO, kv_cache, widths: Sequence[int]) -> None:
        """Warmup phase 1 (decode): one eager decode step of the whole pipeline incl. the terminal prologue/epilogue
        (state + positions restored), the eager sampler in both modes, the sampled-token copy, the seed writes."""
        m = self.m
        gi = kv_cache.group
        snaps = [st.snapshot(st.groups[gi]) for st in m.stages]
        pos = m._snapshot_positions(gi)
        m.decode_eager(gi)
        m.synchronize()
        for st, s in zip(m.stages, snaps):
            st.restore(st.groups[gi], s)
        m._restore_positions(gi, pos)
        for mode in ("greedy", "sample"):
            self._sampler_body(io, mode)
        if len(widths) > 1 and getattr(m, "width", None) is None:
            # several decode widths -> serving compacts live requests into the lowest slots (copy_slot): compile the
            # move's programs now, before any capture, on two idle slots (the generator's reset_all follows phase 1)
            self.copy_slot(kv_cache, kv_cache.max_batch - 1, kv_cache.max_batch - 2)
        self._put_seeds(self.sampler, [DEVICE_SEED_SKIP] * TOK_ROWS)
        m.synchronize()

    def _sampler_body(self, io: DecodeIO, mode: str):
        self.sampler(io.logits.tensors[0], tt_out_tok=io.tokens.tensors[0])
        ttnn.copy(io.tokens.tensors[0], io.sampled.tensors[0])

    def decode(self, io: DecodeIO, kv_cache, width: int) -> None:
        self.m.decode_eager(kv_cache.group)

    def sample(self, io: DecodeIO, mode: str) -> None:
        self._sampler_body(io, mode)
        self._after_sampler_step()

    def capture_decode(self, io: DecodeIO, kv_cache, width: int):
        """Every width maps to the full-width pipeline trace (KDA decodes every slot of its batched state; rows
        >= width have cur_pos -1 and are left untouched, so the width-w contract holds). Captured once: one decode
        trace per chip stage, then (first time) the traced prefill bodies."""
        m = self.m
        gi = kv_cache.group
        if not self.captured_stack:
            for k, st in enumerate(m.stages):
                g = st.groups[gi]
                g.trace_id = ttnn.begin_trace_capture(st.mesh, cq_id=0)
                try:
                    m._stage_decode(k, gi)
                finally:
                    ttnn.end_trace_capture(st.mesh, g.trace_id, cq_id=0)
            m.captured = True
            if m.traced_prefill:
                m._capture_prefill()
            self.captured_stack = True
        tr = _Trace("decode", width, [(st.mesh, st.groups[gi].trace_id) for st in m.stages])
        self.traces.append(tr)
        return tr

    def capture_sampling(self, io: DecodeIO, mode: str):
        tid = ttnn.begin_trace_capture(self.head_mesh, cq_id=0)
        try:
            self._sampler_body(io, mode)
        finally:
            ttnn.end_trace_capture(self.head_mesh, tid, cq_id=0)
        tr = _Trace("sampler", mode, [(self.head_mesh, tid)])
        self.traces.append(tr)
        return tr

    def replay(self, trace: _Trace) -> None:
        if trace.kind == "decode":
            for mesh, tid in trace.ids:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            return
        for mesh, tid in trace.ids:
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        self._after_sampler_step()

    def release(self, trace: _Trace) -> None:
        if trace not in self.traces:
            return
        self.traces.remove(trace)
        if trace.kind == "sampler":
            for mesh, tid in trace.ids:
                ttnn.release_trace(mesh, tid)
            return
        if not any(t.kind == "decode" for t in self.traces):
            self.m.release()  # every chip's decode trace + the prefill traces
            self.captured_stack = False

    def close(self) -> None:
        """Model unload when the caller owns the mesh (vLLM worker shutdown, before the plugin closes the submeshes and
        the parent): what multichip_decoder.close_pipeline_mesh does before ttnn.close_mesh_device - release the
        traces, drain every chip, release the socket pairs (Stage 04a review C-3), quiesce the parent mesh."""
        for t in list(self.traces):
            self.release(t)
        self.m.release()
        self.m.synchronize()
        self.m.close_links()
        self.m.parent.quiesce_devices()

    # ---------------------------------------------------------------------------------------------- reports
    def dram_report(self) -> dict:
        return self.m.dram_report()

    def trace_report(self) -> dict:
        """Device-side trace state for serving evidence (Stage 9 mixed-shape check): per pipeline stage its decode
        trace id(s), captured prefill traces, the program-cache entries and allocated DRAM bytes of its chip (constant
        after warm-up = no compile and no growth while traces are live / across request lifecycles), the sampler trace
        ids, and whether ttnn's trace-allocation tracker is active in this process."""
        from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

        stages = []
        for st in self.m.stages:
            g = st.groups[0]
            n_prefill = sum(t.trace_id is not None for traces in g.prefill_traces.values() for t in traces.values())
            stages.append(
                {
                    "chip": st.plan.chip,
                    "decode_trace_id": g.trace_id,
                    # per-width decode traces (tt/per_width.py), width -> trace id
                    "width_trace_ids": {w: wb.trace_id for w, wb in getattr(g, "wbufs", {}).items()},
                    "prefill_traces": n_prefill,
                    "program_cache_entries": st.mesh.num_program_cache_entries(),
                    "dram_allocated": dram_allocated(st.mesh),
                }
            )
        return {
            "stages": stages,
            "sampler_trace_ids": [tid for t in self.traces if t.kind == "sampler" for _, tid in t.ids],
            "prefill_captured": self.m.prefill_captured,
            "counters": dict(self.counters),
            "trace_alloc_tracking": bool(TRACE_ALLOC_TRACKING),
        }


def round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# ================================================================================================ factory
def _slurp(path: str, bs: int = 1 << 20) -> int:
    n = 0
    try:
        with open(path, "rb", buffering=0) as f:
            while True:
                chunk = f.read(bs)
                if not chunk:
                    return n
                n += len(chunk)
    except FileNotFoundError:
        return 0


class _Prefetcher:
    """Page-cache prefetch of the NFS weight files of upcoming layers (doc/research/weight_loading_speed.md #1:
    ttnn.load_tensor page-faults each .tensorbin as a serial chain of small NFS reads, ~185 MB/s; a 24-thread pool of
    plain reads fills the page cache at GB/s while the previous layer uploads). Identical bytes and load calls."""

    def __init__(self, threads: int = 24):
        import concurrent.futures

        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
        from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource

        self.src = TileCacheSource(CACHE_DIR, MANIFEST)
        self.perm = os.environ.get("GLM53_FUSED_EXPERTS_CACHE", "")
        self.pool = concurrent.futures.ThreadPoolExecutor(threads)
        self.pending: dict = {}

    def files(self, key) -> list[str]:
        src = self.src
        if key == "terminal":
            return [
                os.path.join(src.cache_dir, src.tensors[n]["path"])
                for n in ("embed_tokens.weight", "norm.weight", "lm_head.weight")
            ]
        pre = f"layers.{key}."
        out = []
        for k, e in src.tensors.items():
            if not k.startswith(pre):
                continue
            if ".experts." in k and k.endswith("gate_up_proj") and self.perm:
                ex = int(k.split(".experts.")[1].split(".")[0])
                out.append(
                    os.path.join(self.perm, f"L{key:02d}", f"{ex:03d}.gate_up_proj.{e['sha256'][:16]}.tensorbin")
                )
            else:
                out.append(os.path.join(src.cache_dir, e["path"]))
        return out

    def start(self, key):
        if key not in self.pending:
            self.pending[key] = [self.pool.submit(_slurp, f) for f in self.files(key)]

    def wait(self, key) -> int:
        return sum(f.result() for f in self.pending.pop(key, []))


class _Loader:
    """Per-layer decoder loads in the pipeline's construction order, each preceded by a page-cache prefetch of the next
    `ahead` layers' files (and the terminal's at the start)."""

    def __init__(
        self, cfg, impl: str = "optimized", ocfg=None, order: Sequence[int] = (), ahead: int = 2, parallel: bool = False
    ):
        self.cfg, self.impl, self.ocfg = cfg, impl, ocfg
        self.load_s: dict = {}
        self.order = list(order)
        self.ahead = ahead
        self.calls = 0
        self.parallel = parallel  # chips load concurrently (MultichipDecoder parallel_load): queue every prefetch now
        self.pf = _Prefetcher() if order else None
        if self.pf is not None:
            self.pf.start("terminal")
            if parallel:
                for l in self.order:
                    self.pf.start(l)

    def __call__(self, layer, mesh, half=None):
        if self.pf is not None:
            if not self.parallel:
                for nxt in self.order[self.calls : self.calls + 1 + self.ahead]:
                    self.pf.start(nxt)
            self.pf.wait(layer)
        self.calls += 1
        t0 = time.perf_counter()
        if self.impl == "optimized":
            from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder

            dec = load_optimized_decoder(self.cfg, layer, mesh, self.ocfg)
        else:
            from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import load_fused_decoder

            dec = load_fused_decoder(self.cfg, layer, mesh)
        self.load_s[layer] = round(time.perf_counter() - t0, 2)
        print(f"[model] loaded L{layer} in {self.load_s[layer]} s", flush=True)
        return dec


# Decoder implementations build_model_backend(decoder=) serves (the vLLM adapter checks GLM53_DECODER against this
# before loading): "optimized" = the Stage 3 op-chain decoder; Stage 7a adds its per-layer persistent programs here.
SERVING_DECODERS = ("optimized",)


def load_order(stages) -> list[int]:
    """The order MultichipDecoder.__init__ requests layers: per stage, split (half) layers first."""
    out = []
    for p in stages:
        halves = {}
        for l in p.layers:
            parts = {u.part for u in p.units if u.layer == l}
            halves[l] = None if "full" in parts or parts == {"attn", "ffn"} else next(iter(parts))
        out += sorted(p.layers, key=lambda l: halves[l] is None)
    return out


def pipeline_fabric_config() -> dict:
    """`ttnn.set_fabric_config` keywords of the layer pipeline's socket hops (= multichip_decoder.open_pipeline_mesh:
    FABRIC_1D with the Stage 4 router payload), for a caller that opens the parent mesh itself (the vLLM plugin reads
    them from the serving adapter's model_capabilities["fabric_config"])."""
    from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import FABRIC_PAYLOAD_BYTES

    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = FABRIC_PAYLOAD_BYTES
    return {"config": ttnn.FabricConfig.FABRIC_1D, "router_config": router}


def plan_stages(mesh, *, partition: str | None = None, partition_key: str | None = None, units: str | None = None):
    """Pipeline plan: `units` ("chip:U,U;chip:U,.." e.g. "0:L0,L1,L2,L3" for the reduced probe) or a partition.json
    plan (default: Stage 4's pipeline8_ring_wrap)."""
    coords = chip_coords(mesh)
    if units:
        spec = [(int(seg.split(":")[0]), seg.split(":")[1].split(",")) for seg in units.split(";")]
        stages = [StagePlan(coord=coords[c], units=parse_units(us), chip=c) for c, us in spec]
        validate_plan(stages)
        return stages
    return plan_from_partition(partition or DEFAULT_PARTITION, coords, key=partition_key or DEFAULT_PARTITION_KEY)


def build_model_backend(model_dir, mesh_device, **kw) -> Glm53PipelineBackend:
    """Build the full model on `mesh_device` (the parent mesh opened with fabric: `multichip_decoder.
    open_pipeline_mesh`). kwargs: units / partition / partition_key (plan), decoder ("optimized"), handoff
    ("eth_direct"), max_batch (backend slot capacity, <= 32), submeshes (coord -> 1x1 submesh), ocfg, decode_widths
    (per-width decode traces for these widths, `tt/per_width.py`; default: env GLM53_DECODE_WIDTHS, e.g. "1,2,4"; unset =
    every width replays the full-width trace)."""
    from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config

    cfg = kw.pop("hf_config", None) or hf_config()
    widths = kw.pop("decode_widths", None)
    if widths is None and os.environ.get("GLM53_DECODE_WIDTHS"):
        widths = tuple(int(x) for x in os.environ["GLM53_DECODE_WIDTHS"].split(",") if x)
    stages = kw.pop("stages", None) or plan_stages(
        mesh_device,
        partition=kw.pop("partition", None),
        partition_key=kw.pop("partition_key", None),
        units=kw.pop("units", None),
    )
    # parallel_load (or GLM53_PARALLEL_LOAD=1): one loader thread per chip (Stage 4 MultichipDecoder parallel_load)
    parallel = bool(kw.pop("parallel_load", os.environ.get("GLM53_PARALLEL_LOAD", "0") == "1"))
    loader = kw.pop("loader", None) or _Loader(
        cfg,
        kw.pop("decoder", "optimized"),
        kw.pop("ocfg", None),
        order=load_order(stages) if kw.pop("prefetch", True) else (),
        parallel=parallel,
    )
    stack = PipelineStack(
        mesh_device,
        stages,
        loader=loader,
        handoff=kw.pop("handoff", "eth_direct"),
        submeshes=kw.pop("submeshes", None),
        parallel_load=parallel,
    )
    if getattr(loader, "pf", None) is not None:
        loader.pf.wait("terminal")
    term = Terminal(stack.stages[0].mesh, cfg, cache_dir=CACHE_DIR, manifest_path=MANIFEST)
    be = Glm53PipelineBackend(stack, term, max_batch=kw.pop("max_batch", MAX_DECODE_SLOTS), hf_config=cfg)
    be.loader = loader
    if widths:
        from models.autoports.zai_org_glm_5_3_flash.tt.per_width import PerWidthBackend, PerWidthStack

        be.m.__class__ = PerWidthStack  # the subclasses add methods only (state is created in allocate_decode_io)
        be.__class__ = PerWidthBackend
        be.pw_request = tuple(int(w) for w in widths)
    return be
