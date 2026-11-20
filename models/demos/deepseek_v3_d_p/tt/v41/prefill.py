# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill for the disaggregated split (tt-blaze DS41F-0037): the ENCODER (layers 0..19, Engram
before 1 and 14) in full and layer 20 KV-only, eager, chunked.

    pf = V41Prefill(mesh, cfg, checkpoint(), max_seq_len=..., chunk_tokens=...)
    pf.prefill(ids, engram_host)            # tt-blaze EngramHost: the n-gram hashes + table rows per position
    snap = pf.cache_snapshot()              # == tt-blaze golden ``cache_snapshot_of`` (prefill_trace.py) for layers 0..20

DS41F-0073 marked decoder chunks (``decoder_marked``, the runtime's V41_PREFILL_DECODER_MARKED=1 with V41_PREFILL_DECODER=1):
all 40 layers are built, but a chunk runs layers 20..39 only when asked (``_chunk(decoder=True)``, tt-d-gen marks the chunks
covering the migrated prefix's last ``pd_decoder_cone`` tokens); every other chunk runs the encoder and layer 20 KV-only into
layer 20's BLOCK state (``V41PrefillBlock.kv_only``: its entries, index keys, window carry and path-A slab rows), i.e. what
the default prefill computes. The first decoder chunk of a prompt starts layers 21..39 at its start position from ZERO windows
(the zero rows the ring's replay starts from today): by the replay argument below, every decoder layer's state is exact from
20 x 127 + 1 positions on, so the export at the prefix end is exact when the decoder started >= 2541 positions before it (or at
0). Zeroing (not keeping the slot's previous prompt's windows) changes no exact row; it makes every row independent of what the
slot held before, so runs compare byte for byte.

What the decode ring needs after the prompt (tt-blaze ``docs/deepseek_v4_1_flash/disagg_prefill_plan.md``): every layer's
window (the last 128 positions), the KV sources' compressed entries (2 / 8 / 14 at ratio 2; layer 20 at ratio 1) and the
index keys (2 / 8 / 14 / 20). The decoder layers 21..39 start empty: the ring replays the prompt's last R >= 2541 tokens,
after which its state equals a full prefill's (layer-20 entries and keys are exact because they come from the encoder
output; every decoder-side dependency is a window, a top-k or a candidate pick per query).

Streams are 4 bf16 ``[1, 1, S_l, D_l]`` (SP over mesh dim ``sp_axis``, TP over ``tp_axis``) as in V4-Flash; the embedding
lookup and the Engram hash / gather run on the host.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.v4.trace_island import TraceIsland, copy_into

from . import hostprof as HP
from .attention import V41CSA, v41_hf_config
from .block import HC, V41HyperConnection, V41PrefillBlock, identity_pre_row
from .engram import V41PrefillEngram


def _kv_name(kv_b) -> str:
    return "full" if kv_b is None else str(kv_b)


def _first_device(t) -> torch.Tensor:
    """A replicated device tensor -> the first chip's copy on the host (fp32)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


PROFILE = os.environ.get("V41_PREFILL_PROFILE", "0") == "1"
# WS4 (tt-blaze prefill TTFT plan, Phase 2): each run of consecutive CSA consumer blocks (encoder 3-7, 9-13, 15-19) replays as ONE
# trace on a full traced chunk -- A, A1c, the fixed-row glue, A2c, the carry and B of every block, the blocks chained in place
# (no copy-in). The sources, the SWA layers, Engram and chunk 0 / ragged / decoder chunks keep the per-layer islands. Default off.
# =2: the CSA sources too -- a segment runs from a source's tail (carry, B) through its consumers (and an Engram layer) to the next
# source's head (A + A1); between segments only the source's position-dependent glue (eager) and its A2 (one island per score
# bucket, reading the segment's outputs). The SWA layers 0 / 1 and the decoder layers keep the per-layer islands.
SUPER_LEVEL = int(os.environ.get("V41_PREFILL_SUPER_ISLANDS", "0") or 0)
SUPER = SUPER_LEVEL >= 1
PRE_CACHED = os.environ.get("V41_PREFILL_PRE_CACHED", "0") == "1"  # prefill TTFT plan 1.4 (WS1): identity pre row once
# DS41F-0068: the chunk embedding uploaded S-sharded / TP-replicated and split on device (0 = the D-over-TP mapper)
EMBED_PARTITION = os.environ.get("V41_PREFILL_EMBED_PARTITION", "1") == "1"
ENGRAM_EARLY = (
    os.environ.get("V41_PREFILL_ENGRAM_EARLY", "1") == "1"
)  # DS41F-0068: the Engram fetch before the embedding
# DS41F-0119 (prefill bench 10-29: chunks are host-issue bound): V41_PREFILL_DEVICE_EMBED=1 looks a full text chunk's rows up on
# device from the runner's H2D ids ([1, 1, S_l] uint32 per chip, SP-sharded, TP-replicated) in the table TP-sharded on D
# (~331 MB per chip), instead of the host gather (84 MB) + tilize + TP-replicated upload; ragged / image chunks keep the host
# path. Same bf16 rows by construction; V41_PREFILL_DEVICE_EMBED_CHECK=N compares both paths on the first N chunks.
DEVICE_EMBED = os.environ.get("V41_PREFILL_DEVICE_EMBED", "0") == "1"
DEVICE_EMBED_CHECK = int(os.environ.get("V41_PREFILL_DEVICE_EMBED_CHECK", "0"))
# WS6 (prefill TTFT): V41_PREFILL_EMBED_RAGGED=1 (needs DEVICE_EMBED) -- a RAGGED chunk (every prompt's last one) embeds on device
# too: its ids padded with one extra id that points at a zero row appended to the device table, uploaded as a [sp, 1, S_l] uint32
# tensor (16 KB) instead of the host path's gathered + zero-padded [chunk, dim] bf16 rows (~42 MB at chunk 4096). The streams are
# the host path's bit for bit (the device rows match the host gather, DEVICE_EMBED_CHECK; the pad rows are +0 like its zeros).
# Default off: the host path, which costs ~26 ms of host time at the chunk start while the device idles (WS6 baseline: embed
# 30.1 ms on ragged chunks vs 3.9 ms on full ones).
EMBED_RAGGED = os.environ.get("V41_PREFILL_EMBED_RAGGED", "0") == "1"


# tt-blaze DS41F-0163 (11-19, the Kernelize review's step 0): V41_PREFILL_DEVPROF=<jsonl> with V41_PREFILL_DEVPROF_STARTS=<chunk
# start positions, comma list>: on the EAGER path (V41_PREFILL_ISLANDS=0) and the device profiler (TT_METAL_DEVICE_PROFILER=1
# TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1), each listed chunk reads the device profiler at every
# phase mark (the PhaseProfile marks: embed / engram / block phases / kv_only) and writes, per phase, every program's device
# kernel and FW duration (slowest chip), core count and chip 0's FW start / end: the chunk's device work vs its wall time.
DEVPROF = os.environ.get("V41_PREFILL_DEVPROF", "")
DEVPROF_STARTS = {int(v) for v in os.environ.get("V41_PREFILL_DEVPROF_STARTS", "").split(",") if v.strip()}


class DevProf:
    KERNEL = "DEVICE KERNEL DURATION [ns]"
    FW = "DEVICE FW DURATION [ns]"

    def __init__(self, mesh, path: str, meta: dict):
        self.mesh, self.meta = mesh, meta
        self.f = open(path, "a")
        ttnn.synchronize_device(mesh)
        try:  # drain: the programs before this chunk are not its work
            ttnn.ReadDeviceProfiler(mesh)
            ttnn.get_latest_programs_perf_data()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[v41 devprof] drain failed: {e!r}")
        self.t0 = time.perf_counter()

    def __call__(self, name: str) -> None:
        t = time.perf_counter()
        try:
            ttnn.ReadDeviceProfiler(self.mesh)
            data = ttnn.get_latest_programs_perf_data()
        except Exception as e:  # noqa: BLE001 -- one lost phase, not the run
            self.f.write(json.dumps(dict(kind="devprof_fail", tag=name, error=repr(e)[:300], **self.meta)) + "\n")
            self.f.flush()
            return
        progs = {}
        chip0 = min(data) if data else None
        for chip, plist in data.items():
            for pr in plist:
                res = pr.program_analyses_results
                k, fw = res.get(self.KERNEL), res.get(self.FW)
                if k is None and fw is None:
                    continue
                e = progs.setdefault(
                    int(pr.program_execution_uid.runtime_id), [0, 0, None, None, 0, int(pr.core_count)]
                )
                if k is not None:
                    e[0] = max(e[0], int(k.duration))
                if fw is not None:
                    e[1] = max(e[1], int(fw.duration))
                    if chip == chip0:
                        e[2], e[3] = int(fw.start_timestamp), int(fw.end_timestamp)
                e[4] += 1
        row = dict(
            kind="devprof",
            tag=name,
            read_s=round(time.perf_counter() - t, 3),
            chips=len(data),
            progs=[[rid, *v] for rid, v in sorted(progs.items())],
            **self.meta,
        )
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def report(self, what: str):
        self.f.write(
            json.dumps(dict(kind="devprof_end", what=what, wall_s=round(time.perf_counter() - self.t0, 3), **self.meta))
            + "\n"
        )
        self.f.close()
        logger.info(f"[v41 devprof] {what}: profiled ({DEVPROF})")
        return None


class PhaseProfile:
    """V41_PREFILL_PROFILE=1: per phase name (summed over the layers), the HOST time to issue its ops and the DEVICE
    drain after it (a synchronize closes every phase). A phase whose drain is ~0 is host-issue bound: the device waited
    on Python. The syncs serialise host and device, so the total is an upper bound of the unprofiled chunk."""

    def __init__(self, mesh):
        self.mesh, self.acc = mesh, {}
        ttnn.synchronize_device(mesh)
        self.t = time.perf_counter()

    def __call__(self, name: str) -> None:
        t1 = time.perf_counter()
        ttnn.synchronize_device(self.mesh)
        t2 = time.perf_counter()
        a = self.acc.setdefault(name, [0.0, 0.0, 0])
        a[0], a[1], a[2] = a[0] + t1 - self.t, a[1] + t2 - t1, a[2] + 1
        self.t = t2

    def report(self, what: str) -> dict:
        tot_i = sum(v[0] for v in self.acc.values())
        tot_w = sum(v[1] for v in self.acc.values())
        rows = ", ".join(
            f"{k} {v[0] * 1e3:.0f}+{v[1] * 1e3:.0f} ms/{v[2]}"
            for k, v in sorted(self.acc.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
        )
        logger.info(
            f"[v41 prefill][profile] {what}: issue {tot_i:.2f} s + device drain {tot_w:.2f} s | phase issue+drain/calls: {rows}"
        )
        return dict(self.acc)


def _decoder_cache():
    """DS41F-0073: V41_PREFILL_DECODER_CACHE = a weight-cache dir for the decoder layers (unset: converted at every start)."""
    p = os.environ.get("V41_PREFILL_DECODER_CACHE", "")
    return Path(p) if p else None


def _decoder_expert_dtype():
    """DS41F-0073: V41_PREFILL_DECODER_EXPERT_DTYPE=bfp4|bfp8 for the decoder layers' routed experts (unset: the encoder's).
    BFP4 is what the decode ring runs those layers with (tt-blaze DS41F-0019) and halves their ~9 GB per chip."""
    v = os.environ.get("V41_PREFILL_DECODER_EXPERT_DTYPE", "")
    return {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[v] if v else None


class V41Prefill:
    CHUNK_ALIGN = (
        64  # tokens per chip: a multiple of the MoE routing's 64-core masked_bincount grid (and of TILE / ratio)
    )

    def __init__(
        self,
        mesh_device,
        cfg,
        ck,
        *,
        max_seq_len: int,
        chunk_tokens: int,
        n_layers: int | None = None,
        kv_only: bool = True,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        num_links: int = 2,
        weight_cache_path=None,
        load_routed_from_cache: bool = False,
        num_users: int = 1,
        decoder_marked: bool = False,
    ):
        self.mesh, self.cfg, self.ck = mesh_device, cfg, ck
        self.num_users, self.slot = int(num_users), 0
        self.last_profile = None  # V41_PREFILL_PROFILE: the last chunk's PhaseProfile
        self.islands = False  # enable_islands: blocks replay their mHC / MoE trace islands
        weight_cache_path = Path(weight_cache_path) if weight_cache_path else None  # the modules join it with `/`
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp, self.tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
        self.chunk = int(chunk_tokens)
        # 64 tokens per chip per step: the MoE's masked_bincount splits each chip's tokens over a 64-core grid
        assert self.chunk % (self.CHUNK_ALIGN * self.sp) == 0, (self.chunk, self.CHUNK_ALIGN * self.sp)
        self.max_seq_len = int(max_seq_len)
        self.n_layers = cfg.first_decoder_layer if n_layers is None else int(n_layers)
        self.kv_only_layer = (
            cfg.first_decoder_layer if kv_only else None
        )  # layer 20: the decoder's shared KV + index keys
        # DS41F-0073 marked decoder chunks: every layer built, layers 20..39 run only on the chunks _chunk is told to
        self.decoder_marked = bool(decoder_marked)
        assert not self.decoder_marked or (
            not kv_only and self.n_layers == cfg.n_layers
        ), "decoder_marked needs all layers built as blocks (n_layers = cfg.n_layers, kv_only=False)"
        # per slot: where the current prompt's decoder started (None = not since the last reset / KV-only chunk)
        self.dec_from = {s: None for s in range(self.num_users)}
        S_l = self.chunk // self.sp
        self.rotary_emb = DeepseekV4RotaryEmbedding(v41_hf_config(cfg, max_seq=self.max_seq_len))
        mk = dict(sp_axis=sp_axis, tp_axis=tp_axis, topology=topology)
        self.sources: dict = {}
        self.blocks, self.engrams = [], {}
        t0 = time.time()
        for L in range(self.n_layers):
            w = ck.layer(L)
            if cfg.role(L).engram:
                self.engrams[L] = V41PrefillEngram(mesh_device, cfg, L, w, sp_axis=sp_axis, tp_axis=tp_axis)
            # DS41F-0073: the decoder layers (20..39, V41_PREFILL_DECODER) take their own weight cache (default none: the shared
            # disk cannot hold another ~13.7 GB per BFP8 layer) and expert dtype (default the encoder's)
            dec = L >= cfg.first_decoder_layer
            blk = V41PrefillBlock(
                mesh_device,
                cfg,
                L,
                w,
                rotary_emb=self.rotary_emb,
                seq_len_per_chip=S_l,
                sources=self.sources,
                ck=ck,
                num_links=num_links,
                weight_cache_path=_decoder_cache() if dec else weight_cache_path,
                load_routed_from_cache=load_routed_from_cache and not dec,
                expert_dtype=_decoder_expert_dtype() if dec else None,
                **mk,
            )
            for slot in range(self.num_users):  # one attention state per user (the weights are shared)
                st = blk.alloc_state(self.max_seq_len, self.chunk, slot=slot)
                if self.decoder_marked and L > cfg.first_decoder_layer and getattr(self, "_zero_carry", None) is None:
                    # the zero window _decoder_begin copies into layers 21..39's carries: a fresh state's carry (zeros), cloned
                    # now -- before any trace-island capture
                    self._zero_carry = ttnn.clone(st.sliding_carry)
                if self.decoder_marked and L == cfg.first_decoder_layer and getattr(blk.attn, "sparse_path", False):
                    # layer 20's path-A slab now, not at its first write: a KV-only chunk writes it (kv_only(slab=True)),
                    # possibly first on a later request, after the trace-island capture (DS41F-0037 10-10 21:25 class)
                    blk.attn._slab_alloc(st)
            self.blocks.append(blk)
            logger.info(f"[v41 prefill] layer {L} ({cfg.role(L).mode}) built, {time.time() - t0:.0f} s")
        self.kv_attn = self.kv_hc = self.kv_norm = None
        if self.kv_only_layer is not None:
            L = self.kv_only_layer
            w = ck.layer(L)
            self.kv_hc = V41HyperConnection(
                mesh_device,
                hidden=cfg.dim,
                fn=w["hc_attn_fn"],
                base=w["hc_attn_base"],
                scale=w["hc_attn_scale"],
                rms_eps=cfg.norm_eps,
                hc_eps=cfg.hc_eps,
                sinkhorn_iters=cfg.hc_sinkhorn_iters,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
            )
            self.kv_norm = TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=cfg.dim,
                epsilon=cfg.norm_eps,
                torch_weight=w["attn_norm.weight"].float(),
                cluster_axis=tp_axis,
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"v41_layer_{L}.attn_norm",
            )
            self.kv_attn = V41CSA.from_weights(
                mesh_device, cfg, L, w, self.rotary_emb, weight_cache_path=weight_cache_path, **mk
            )
            self.kv_states = {
                s: self.kv_attn.alloc_state(self.max_seq_len, chunk_tokens=self.chunk) for s in range(self.num_users)
            }
        self._embed = None
        # DEVICE_EMBED: the table on device, allocated here -- before any trace-island capture (a buffer allocated after a
        # capture can sit in the islands' intermediates, DS41F-0037 10-10 21:25); _ids_dev = the current chunk's H2D ids
        self._ids_dev = None
        self._embed_dev = self._device_embedding() if DEVICE_EMBED else None
        self._embed_checks = DEVICE_EMBED_CHECK
        self.kv_actuals = {s: 0 for s in range(self.num_users)}
        # ViT V2 (tt-blaze vit_plan.md E1 / G1): the current prompt's image-span rows (virtual id vocab + k -> row k) and
        # each gate's (text, bias_vl) bias rows; ``_vl_live`` = some gate holds bias_vl rows from the previous chunk
        self._image_rows = None
        self._gate_bias_rows: dict = {}
        self._vl_live = False

    # the CURRENT slot's layer-20 state and token count (use_slot): the chunk / export / snapshot code reads these
    @property
    def kv_state(self):
        return self.kv_states[self.slot]

    @property
    def kv_actual(self) -> int:
        return self.kv_actuals[self.slot]

    @kv_actual.setter
    def kv_actual(self, v: int) -> None:
        self.kv_actuals[self.slot] = int(v)

    def use_slot(self, slot: int) -> None:
        assert 0 <= slot < self.num_users, (slot, self.num_users)
        self.slot = int(slot)
        for b in self.blocks:
            b.use_slot(self.slot)

    # ---- DS41F-0073 marked decoder chunks ------------------------------------------------------------------------------
    @property
    def decoder_from(self):
        """The CURRENT slot's prompt: where its decoder layers started (None = no decoder chunk since the last reset or the
        last KV-only chunk)."""
        return self.dec_from[self.slot]

    def window_blocks(self) -> list:
        """The blocks whose window carry a KV-only chunk keeps current (prefix reuse checkpoints them; the export's window rings
        from carries write them): every block, except with ``decoder_marked`` the decoder layers 21..39, whose windows
        only decoder chunks write."""
        if not self.decoder_marked:
            return list(self.blocks)
        return [b for b in self.blocks if b.layer <= self.cfg.first_decoder_layer]

    def _decoder_begin(self, start: int) -> None:
        """A decoder chunk at ``start``: the first of the prompt (or the first after a KV-only chunk) starts layers 21..39 here
        -- counters at ``start``, their window carries zero (the replay argument: exact >= 2541 positions on); a later one
        must continue them."""
        first = self.cfg.first_decoder_layer
        if self.dec_from[self.slot] is None:
            for b in self.blocks:
                if b.layer > first:
                    ratio = self.cfg.role(b.layer).compress_ratio or 1
                    b.state.kv_actual = int(start)
                    b.state.entry_count = int(start) // ratio
                    ttnn.copy(self._zero_carry, b.state.sliding_carry)  # INTO the buffer (the islands read its address)
            self.dec_from[self.slot] = int(start)
            return
        for b in self.blocks:
            if b.layer > first:
                assert b.state.kv_actual == int(start), (b.layer, b.state.kv_actual, start)

    def reset(self) -> None:
        """A new prompt in the CURRENT slot: its attention states' counters back to 0. Path B's masks are built from the counters
        (entries / window rows past them read -inf, chunk 0's carry columns are masked), so no device contents need to
        be cleared; V4.1's non-overlap compressor carries no state between chunks."""
        states = [b.state for b in self.blocks] + ([self.kv_state] if self.kv_attn is not None else [])
        for st in states:
            st.kv_actual, st.entry_count = 0, 0
        self.kv_actual = 0
        self.dec_from[self.slot] = None
        # DS41F-0068 path A3: the static scorer reads the slot's score mask (shared by the slot's sources), which each traced
        # chunk's glue opens column by column -> every entry column back to -inf (a device copy), once per mask
        seen = set()
        for b in self.blocks:
            m = getattr(b.state, "score_mask", None)
            if m is not None and b._attn_islands_enabled() and id(m) not in seen:
                seen.add(id(m))
                b.attn.reset_score_mask(b.state)

    # ---- host-side inputs ---------------------------------------------------------------------------------------
    def _embedding(self) -> torch.Tensor:
        if self._embed is None:
            self._embed = self.ck.raw("embed.weight")  # [vocab, D] bf16
        return self._embed

    def _device_embedding(self):
        """DEVICE_EMBED: the [vocab, D] bf16 table, replicated over SP and sharded on D over TP (TtParallelEmbedding's layout)."""
        assert (self.sp_axis, self.tp_axis) == (0, 1), (self.sp_axis, self.tp_axis)
        t0 = time.time()
        mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, 1))
        table = self._embedding().contiguous()
        if EMBED_RAGGED:  # the ragged chunks' pad id (= vocab) reads this zero row
            table = torch.cat([table, torch.zeros(1, table.shape[1], dtype=table.dtype)], dim=0)
        t = ttnn.from_torch(
            table,
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        logger.info(f"[v41 prefill] device embedding table {tuple(t.shape)} per chip in {time.time() - t0:.1f} s")
        return t

    def _ragged_ids_dev(self, ids: list):
        """EMBED_RAGGED: a ragged chunk's ids as the H2D ids tensor's spec ([sp, 1, S_l] uint32, SP-sharded, TP-replicated), the
        pad positions set to vocab (the table's appended zero row). Freed by the caller right after the embedding (an eager
        temporary: nothing outlives a replay)."""
        S_l, vocab = self.chunk // self.sp, int(self._embedding().shape[0])
        t = torch.full((self.chunk,), vocab, dtype=torch.int32)
        t[: len(ids)] = torch.tensor(ids, dtype=torch.int32)
        return ttnn.from_torch(
            t.view(self.sp, 1, S_l),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
        )

    def _streams_device(self, ids_dev):
        """DEVICE_EMBED: [1, 1, S_l, D_l] bf16 TILE per chip straight from the H2D ids (no host round trip)."""
        S_l, D_l = self.chunk // self.sp, self.cfg.dim // self.tp
        t = ttnn.embedding(ids_dev, self._embed_dev, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if tuple(t.shape) != (1, 1, S_l, D_l):
            t = ttnn.reshape(t, (1, 1, S_l, D_l))
        return t

    def _streams(self, ids: list[int]) -> list:
        ids_dev = self._ids_dev
        if (
            self._embed_dev is not None
            and ids_dev is not None
            and len(ids) == self.chunk
            and max(ids) < self._embedding().shape[0]
        ):
            t = self._streams_device(ids_dev)
            if (
                self._embed_checks > 0
            ):  # DEVICE_EMBED_CHECK: the host path's tensor for the same ids, compared on chip 0..n
                self._embed_checks -= 1
                self._ids_dev = None
                ref = self._streams(ids)[0]
                self._ids_dev = ids_dev
                a = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]
                b = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(ref)]
                same = all(
                    x.shape == y.shape and torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(a, b)
                )
                logger.warning(
                    f"[v41 prefill] DEVICE_EMBED_CHECK: device rows {'BIT-IDENTICAL' if same else 'DIFFER'} to the host path on "
                    f"{len(a)} chips (shape {tuple(a[0].shape)} vs {tuple(b[0].shape)})"
                )
            return [t] + [ttnn.clone(t) for _ in range(HC - 1)]
        if (
            EMBED_RAGGED
            and self._embed_dev is not None
            and 0 < len(ids) < self.chunk
            and max(ids) < self._embedding().shape[0]
        ):
            ids_t = self._ragged_ids_dev(ids)
            t = self._streams_device(ids_t)
            ttnn.deallocate(ids_t)
            return [t] + [ttnn.clone(t) for _ in range(HC - 1)]
        emb = self._embedding()
        t = torch.tensor(ids, dtype=torch.long)
        vl = t >= emb.shape[0]  # ViT V2 E1: image-span VIRTUAL ids (vocab + k) embed as the request's span row k
        rows = emb[t.clamp(max=emb.shape[0] - 1)].to(torch.bfloat16)
        if bool(vl.any()):
            assert self._image_rows is not None, "image-span (virtual) ids need prefill(..., image_rows=...)"
            k = t[vl] - emb.shape[0]
            assert int(k.max()) < self._image_rows.shape[0], (int(k.max()), tuple(self._image_rows.shape))
            rows[vl] = self._image_rows[k].to(torch.bfloat16)
        if len(ids) == self.chunk and rows.is_contiguous():
            h = rows.view(
                1, 1, self.chunk, self.cfg.dim
            )  # DS41F-0068: a full chunk needs no zero-filled copy (84 MB twice)
        else:
            h = torch.zeros(1, 1, self.chunk, self.cfg.dim, dtype=torch.bfloat16)
            h[0, 0, : len(ids)] = rows
        if EMBED_PARTITION:
            # DS41F-0068: S-sharded + TP-replicated upload (contiguous host slices: ~25 ms for the 8192-row chunk, vs ~104 ms for
            # the D-over-TP mapper's strided copies, MEASURED ut/a3/upload_bench.py), then each chip keeps its D / tp on device
            mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None))
            full = ttnn.from_torch(
                h, device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            t = ttnn.mesh_partition(full, dim=3, cluster_axis=self.tp_axis)
            ttnn.deallocate(full)
        else:
            mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 3))
            t = ttnn.from_torch(h, device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        return [t] + [ttnn.clone(t) for _ in range(HC - 1)]  # model.py: h.unsqueeze(2).repeat(1, 1, hc, 1)

    # ---- the prefill --------------------------------------------------------------------------------------------
    def set_image_rows(self, rows) -> None:
        """ViT V2: the CURRENT prompt's image-span rows ``[n_span, dim]`` (pod-NFS ``prof/vit_prompt.py``'s span rows:
        START, the aligner rows with a NEWLINE after each grid row, END); virtual id ``vocab + k`` embeds as row ``k`` and
        routes with ``bias_vl``. None = a text-only prompt."""
        self._image_rows = None if rows is None else torch.as_tensor(rows).to(torch.bfloat16).reshape(-1, self.cfg.dim)

    def _gate_rows(self, blk):
        """(text, bias_vl) bias rows of ``blk``'s gate, bf16 [n_experts]: the text row read back from the gate's own device
        bias (bit-identical to what it routes text with), bias_vl centred on its own midrange before the bf16 cast (DS41F-0040's
        rule: top-k is shift-invariant per token, so each vector may carry its own constant)."""
        L = blk.layer
        if L not in self._gate_bias_rows:
            g = blk.moe.gate
            text = ttnn.to_torch(ttnn.get_device_tensors(g.bias)[0])[0].to(torch.bfloat16)
            vl = self.ck.raw(f"layers.{L}.ffn.gate.bias_vl").float()
            vl = (vl - (vl.min() + vl.max()) / 2).to(torch.bfloat16)
            self._gate_bias_rows[L] = (text, vl)
        return self._gate_bias_rows[L]

    def _route_bias(self, part: list[int]) -> None:
        """ViT V2 G1: model.py routes image-span tokens with ``bias_vl`` (``Gate.forward``'s image_mask). Rewrite each gate's
        per-row bias [sp_dim, n_experts] (the islands captured its buffer; the CONTENT changes, as the ragged padding
        config does) with bias_vl rows at this chunk's span positions; the next text-only chunk restores the text rows.
        """
        if (
            os.environ.get("V41_PREFILL_VL_BIAS", "1") == "0"
        ):  # the causal control: span tokens route with the text bias
            return
        vocab = self._embedding().shape[0]
        vl = [i for i, t in enumerate(part) if int(t) >= vocab]
        if not vl and not self._vl_live:
            return
        mapper = ttnn.ShardTensor2dMesh(self.mesh, dims=(0, None), mesh_shape=tuple(self.mesh.shape))
        for blk in self.blocks:
            g = blk.moe.gate
            assert int(g.bias.shape[0]) * self.sp == self.chunk, (tuple(g.bias.shape), self.chunk, self.sp)
            text, vlb = self._gate_rows(blk)
            rows = text.unsqueeze(0).repeat(self.chunk, 1)
            if vl:
                rows[torch.tensor(vl, dtype=torch.long)] = vlb
            host = ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(host, g.bias)
        self._vl_live = bool(vl)

    def prefill(self, ids: list[int], engram_host=None, *, on_hidden=None, image_rows=None) -> None:
        """All of ``ids`` from position 0 (chunks of ``chunk_tokens``; every chunk but the last full). ``image_rows``: the
        prompt's image-span rows (``set_image_rows``) when ``ids`` carry virtual image ids."""
        assert self.kv_actual == 0, "one prompt per V41Prefill (states are not reset)"
        S = len(ids)
        assert S <= self.max_seq_len
        if image_rows is not None:
            self.set_image_rows(image_rows)
        for c0 in range(0, S, self.chunk):
            part = ids[c0 : c0 + self.chunk]
            t0 = time.time()
            self._chunk(part, c0, engram_host, on_hidden)
            ttnn.synchronize_device(self.mesh)
            logger.info(f"[v41 prefill] chunk [{c0}, {c0 + len(part)}) in {time.time() - t0:.1f} s")

    def enable_islands(self) -> None:
        """Capture every block's trace islands (block.enable_trace_islands) once, after eager warm-up chunks have built every
        program and every lazily cached constant at every live width (the runtime's warm-all). The CSA sources get their
        persistent selection-mask buffers first (they must exist before any capture). Templates: zero streams of the chunk
        shape, each block's capture feeding the next its persistent outputs (contents do not matter)."""
        assert not self.islands
        for blk in self.blocks:
            if hasattr(blk.attn, "precreate_shared_masks"):
                blk.attn.precreate_shared_masks(blk.state)
        streams = self._streams([0] * self.chunk)
        pre = identity_pre_row(self.mesh, self.chunk // self.sp, self.sp_axis)
        if PRE_CACHED:
            # V41_PREFILL_PRE_CACHED: the identity pre row every chunk rebuilt + uploaded (a constant), allocated ONCE and
            # BEFORE the captures -- a buffer allocated after a capture can sit in the islands' intermediates (DS41F-0037)
            self._pre_cached = identity_pre_row(self.mesh, self.chunk // self.sp, self.sp_axis)
        t0 = time.time()
        from .engram import ENGRAM_ISLAND

        for blk in self.blocks:
            if ENGRAM_ISLAND and blk.layer in self.engrams:
                streams = self.engrams[blk.layer].enable_island(
                    streams, self.chunk
                )  # DS41F-0163: its outputs feed the block
            streams, pre = blk.enable_trace_islands(streams, pre, self.chunk)
        self.islands = True
        logger.info(f"[v41 prefill] trace islands captured for {len(self.blocks)} blocks in {time.time() - t0:.1f} s")
        if SUPER_LEVEL >= 2:
            self._enable_super_plan()
        elif SUPER:
            self._enable_super_islands()

    # ---- WS4 level 2: the whole encoder run from the first source on, as segments split only at the sources' glue -------------
    def _enable_super_plan(self) -> None:
        from .attention import V41CSA, V41CSAConsumer

        assert all(int(getattr(b.state, "slot", 0)) == 0 for b in self.blocks), "super-islands assume one physical slot"
        t0 = time.time()
        C = self.chunk
        enc = [i for i, b in enumerate(self.blocks) if b.layer < self.cfg.first_decoder_layer]
        srcs = [i for i in enc if isinstance(self.blocks[i].attn, V41CSA) and self.blocks[i]._attn_islands_enabled()]
        self._plan = None
        if not srcs or srcs[0] == 0:
            logger.warning("[v41 super2] no traced CSA source after block 0: plan disabled")
            return
        s0, end = srcs[0], enc[-1]
        for i in range(s0, end + 1):
            b = self.blocks[i]
            cons = isinstance(b.attn, V41CSAConsumer) and b._attn_islands_enabled()
            if b.layer in self.engrams and getattr(self.engrams[b.layer], "_island", None) is None:
                logger.warning(
                    f"[v41 super2] Engram L{b.layer} has no island (V41_PREFILL_ENGRAM_ISLAND=1 needed): plan disabled"
                )
                return
            if not (i in srcs or cons) or (b.layer in self.engrams and i not in srcs):
                logger.warning(
                    f"[v41 super2] block {b.layer} is neither a traced source nor a traced consumer: plan disabled"
                )
                return
        # capture-time positions, as _capture_attn_islands: a valid full chunk just past chunk 0
        saved = {}
        for i in range(s0, end + 1):
            b = self.blocks[i]
            saved[i] = (b.state.kv_actual, b.state.entry_count, getattr(b.state, "mask_zeroed_upto", None))
            b.state.kv_actual = C
            if i in srcs:
                b.state.entry_count = C // b.attn.compressor.compress_rate
            b.attn.prepare_chunk(b.state, C)

        def head_ops(b, s, pre):
            attn_pre, attn_post, attn_comb, h, Xa = b._ops_a(s, pre)
            # WS7: the X5 stack would have to cross a segment boundary; super2 does not carry it
            assert Xa is None, "V41_MHC_REUSE_X5 is not supported with V41_PREFILL_SUPER_ISLANDS=2"
            outs = b.attn.forward_pre(h, b.state, C)
            return [attn_pre, attn_post, attn_comb, h] + list(outs)

        def tail_ops(b, src_in, attn_pre, attn_post, attn_comb, next_carry, y_a2):
            ttnn.copy(next_carry, b.state.sliding_carry)  # _a3_carry on a full chunk
            y_buf = b._islands[4]
            y = ttnn.typecast(y_a2, y_buf.dtype) if y_a2.dtype != y_buf.dtype else y_a2
            out = b._ops_b(src_in, y, attn_pre, attn_post, attn_comb, C)
            return list(out[:HC]), out[HC]

        n_out = None
        segs, prev_head = [], None
        bounds = srcs + [end + 1]
        start_inputs = list(
            self.blocks[s0 - 1]._islands[1].outputs
        )  # block s0-1's island-B outputs: HC streams + ffn_pre
        for k, si in enumerate(bounds):
            # segment k: [tail of source bounds[k-1]] + consumers (bounds[k-1], si) + [Engram of si] + [head of si]
            lo = bounds[k - 1] + 1 if k > 0 else s0
            cons = [i for i in range(lo, min(si, end + 1))]
            head = si if si <= end else None
            if k == 0:
                ins = start_inputs
            else:
                ph = prev_head  # outputs of the previous segment for its head source
                ins = list(ph["src_in"]) + [ph["attn_pre"], ph["attn_post"], ph["attn_comb"], ph["outs"][3]]
                ins.append(self.blocks[bounds[k - 1]].attn._a3_out[0][0])  # the source's A2 y (persistent)

            def body(*args, _k=k, _cons=cons, _head=head, _prev=bounds[k - 1] if k > 0 else None):
                if _k == 0:
                    s, pre = list(args[:HC]), args[HC]
                else:
                    src_in = list(args[:HC])
                    s, pre = tail_ops(self.blocks[_prev], src_in, *args[HC : HC + 5])
                for i in _cons:
                    out = self.blocks[i].super_consumer_ops(s, pre, C)
                    s, pre = list(out[:HC]), out[HC]
                if _head is None:
                    return tuple(s) + (pre,)
                hb = self.blocks[_head]
                if hb.layer in self.engrams:
                    e = self.engrams[hb.layer]
                    s = e.forward(s, e._isl_rows)
                return tuple(s) + tuple(head_ops(hb, s, pre))

            moes = []
            if k > 0:
                moes.append(self.blocks[bounds[k - 1]].moe)
            moes += [self.blocks[i].moe for i in cons]
            isl = TraceIsland(self.mesh, body, ins, moes=moes, name=f"v41.super2.seg{k}")
            outs = isl.capture()
            seg = dict(island=isl, cons=cons, head=head, prev=bounds[k - 1] if k > 0 else None)
            if head is not None:
                o = list(outs)
                prev_head = dict(
                    src_in=o[:HC],
                    attn_pre=o[HC],
                    attn_post=o[HC + 1],
                    attn_comb=o[HC + 2],
                    h=o[HC + 3],
                    outs=o[HC + 4 :],
                )
                seg["head_out"] = prev_head
                # the head source's A2, one island per score bucket (and the full width), reading this segment's outputs and
                # writing the source's persistent A2 outputs (the consumers' top-k buffer)
                hb = self.blocks[head]
                a2_out = hb.attn._a3_out[0]
                buckets = [None] + list(hb.attn.capacity_buckets(hb.state, C // self.sp))
                a2s = {}
                for kv_b in buckets:

                    def a2_fn(_b=hb, _ph=prev_head, _kv=kv_b, _dst=a2_out):
                        res = _b.attn.forward_attn(_ph["h"], _ph["outs"], _b.state, C, kv_bucket=_kv)
                        for dst, t in zip(_dst, res):
                            ttnn.copy(t, dst)
                            ttnn.deallocate(t)
                        return _dst

                    a = TraceIsland(self.mesh, a2_fn, [], name=f"v41.super2.A2.L{hb.layer}.b{_kv_name(kv_b)}")
                    a.capture()
                    a2s[kv_b] = a
                seg["a2s"] = a2s
            else:
                seg["final_out"] = list(outs)
            segs.append(seg)
        for i, (ka, ec, mz) in saved.items():
            b = self.blocks[i]
            b.state.kv_actual, b.state.entry_count = ka, ec
            if mz is not None:
                b.state.mask_zeroed_upto = mz
        self._plan = dict(s0=s0, end=end, srcs=srcs, segs=segs, start_inputs=start_inputs)
        logger.info(
            f"[v41 super2] blocks {self.blocks[s0].layer}..{self.blocks[end].layer} as {len(segs)} segment(s) "
            f"({[sg['island'].num_segments for sg in segs]} trace pieces), "
            f"sources {[self.blocks[i].layer for i in srcs]}, captured in {time.time() - t0:.1f} s"
        )

    def _super_plan_eligible(self, real_len: int, streams, pre, owned: bool) -> bool:
        pl = getattr(self, "_plan", None)
        if pl is None or owned or int(real_len) != self.chunk:
            return False
        if not all(a is b for a, b in zip(list(streams) + [pre], pl["start_inputs"])):
            return False
        for i in range(pl["s0"], pl["end"] + 1):
            b = self.blocks[i]
            if b._moe_pc_len != self.chunk or int(b.state.kv_actual) == 0 or not b.attn.trace_ready(b.state):
                return False
        return True

    def _run_super_plan(self, rows, mark) -> tuple:
        """Replay the level-2 plan on a full traced chunk -> (streams, pre) = the last encoder block's outputs."""
        pl = self._plan
        C = self.chunk
        for i in range(pl["s0"], pl["end"] + 1):  # every position scalar first (they depend on the chunk start only)
            b = self.blocks[i]
            b.attn.prepare_chunk(b.state, C)
        if HP.ON:
            HP.mark("super2.prepare")
        out = None
        for seg in pl["segs"]:
            hb = self.blocks[seg["head"]] if seg["head"] is not None else None
            if hb is not None and hb.layer in self.engrams:
                L = hb.layer
                if HP.ON:
                    HP.mark("other")
                r = rows[L].result() if hasattr(rows[L], "result") else rows[L]
                if HP.ON:
                    HP.mark("engram.wait")
                rd = self.engrams[L].upload_rows(r, C)
                copy_into(self.engrams[L]._isl_rows, rd)
                ttnn.deallocate(rd)
                if HP.ON:
                    HP.mark("engram.upload")
            out = seg["island"].replay()
            if HP.ON:
                HP.mark("super2.seg")
            for i in seg["cons"]:
                self.blocks[i].super_after(C)
            if hb is not None:
                ph = seg["head_out"]
                hb.attn.glue_chunk(hb.state, ph["outs"], C, None)
                if HP.ON:
                    HP.mark("super2.glue")
                a2 = seg["a2s"][None]
                for kv_b in sorted(k for k in seg["a2s"] if k is not None):
                    if hb.attn.bucket_need(hb.state, C) <= kv_b:
                        a2 = seg["a2s"][kv_b]
                        break
                a2.replay()
                if HP.ON:
                    HP.mark("super2.A2")
                # _source_epilogue_chunk without the carry copy (the next segment does it)
                st = hb.state
                st.entry_count += C // hb.attn.compressor.compress_rate
                st.kv_actual += C
                hb.attn.last_path = "A3"
                hb.attn.shared = dict(path="A3", idx=hb.attn._a3_out[0][1], state=st, entry_count=st.entry_count)
            mark(f"super2.seg{pl['segs'].index(seg)}")
        o = list(out)
        return o[:HC], o[HC]

    def _enable_super_islands(self) -> None:
        """WS4: one super-island per maximal run of consecutive CSA consumer blocks, captured AFTER every per-layer island (its
        inputs are the previous block's island-B persistent outputs, read in place; its intermediates may reuse the per-layer
        islands' freed ranges, which is safe because nothing eager lives across a replay). The capture runs the consumer's
        full-chunk traced path at a valid position just past chunk 0, as the per-layer capture does."""
        from .attention import V41CSAConsumer

        assert all(int(getattr(b.state, "slot", 0)) == 0 for b in self.blocks), "super-islands assume one physical slot"
        t0 = time.time()
        runs, cur = [], []
        for i, blk in enumerate(self.blocks):
            ok = isinstance(blk.attn, V41CSAConsumer) and blk._attn_islands_enabled() and blk.layer not in self.engrams
            if ok and i > 0:
                cur.append(i)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        self._super = {}
        for idx in runs:
            prev = self.blocks[idx[0] - 1]
            B_prev = prev._islands[1]
            ins = list(B_prev.outputs)  # the previous block's island-B persistent outputs: HC streams + ffn_pre
            group = [self.blocks[i] for i in idx]
            saved = [(b.state.kv_actual, b.state.entry_count) for b in group]
            for b in group:  # a valid position just past chunk 0 (as _capture_attn_islands): the rope scalar reads it
                b.state.kv_actual = self.chunk
                b.attn.prepare_chunk(b.state, self.chunk)

            def body(*args, _group=group):
                s, p = list(args[:HC]), args[HC]
                for b in _group:
                    out = b.super_consumer_ops(s, p, self.chunk)
                    s, p = list(out[:HC]), out[HC]
                return tuple(s) + (p,)

            isl = TraceIsland(
                self.mesh, body, ins, moes=[b.moe for b in group], name=f"v41.super.L{group[0].layer}-{group[-1].layer}"
            )
            isl.capture()
            for b, (ka, ec) in zip(group, saved):
                b.state.kv_actual, b.state.entry_count = ka, ec
            self._super[idx[0]] = (idx, isl)
            logger.info(f"[v41 super] blocks {[b.layer for b in group]}: one island, {isl.num_segments} segment(s)")
        logger.info(f"[v41 super] {len(self._super)} super-island(s) captured in {time.time() - t0:.1f} s")

    def _run_decoder(self, decoder, start: int) -> bool:
        """``_chunk``'s ``decoder`` -> run layers 20..39 on this chunk? None = every built layer (the default prefill: the
        encoder + layer 20 KV-only; V41_PREFILL_DECODER=1: all 40). False needs ``decoder_marked``."""
        if not self.decoder_marked:
            assert decoder is None or bool(decoder) == (self.kv_only_layer is None), (decoder, self.kv_only_layer)
            return False
        run = True if decoder is None else bool(decoder)
        if run:
            self._decoder_begin(start)
        else:
            self.dec_from[self.slot] = None  # a KV-only chunk: a later decoder chunk starts the decoder afresh
        return run

    def _chunk(self, part: list[int], start: int, engram_host, on_hidden, decoder=None):
        """One chunk at ``start``. ``decoder`` (``decoder_marked`` only; None = all built layers): True runs layers 20..39
        too, False the encoder + layer 20 KV-only."""
        kv_only20 = not self._run_decoder(decoder, start) and self.decoder_marked
        if self.islands:
            return self._chunk_islands(part, start, engram_host, kv_only20=kv_only20)
        real_len = len(part)
        if DEVPROF and int(start) in DEVPROF_STARTS and len(set(part[:256])) > 1:  # not a warm-up chunk (constant ids)
            prof = DevProf(
                self.mesh, DEVPROF, dict(start=int(start), real_len=real_len, decoder=not kv_only20, slot=self.slot)
            )
        else:
            prof = PhaseProfile(self.mesh) if PROFILE else None
        mark = prof if prof is not None else (lambda name: None)
        streams = self._streams(part)
        self._route_bias(part)
        pre = identity_pre_row(self.mesh, self.chunk // self.sp, self.sp_axis)
        mark("embed")
        rows = engram_host.rows_for(self.slot, part, start) if (self.engrams and engram_host is not None) else {}
        mark("engram.host")
        for blk in self.blocks:
            L = blk.layer
            if kv_only20 and L >= self.cfg.first_decoder_layer:
                blk.kv_only(streams, pre, real_len=real_len)  # layer 20 KV-only; 21..39 skipped
                mark("kv_only")
                break
            if L in self.engrams:
                assert L in rows, f"layer {L} needs Engram rows (pass an EngramHost)"
                rd = self.engrams[L].upload_rows(rows[L], self.chunk)
                s_in = streams
                streams = self.engrams[L](streams, rd)
                ttnn.deallocate(rd)
                mark("engram.dev")
                if (
                    on_hidden is not None
                ):  # the Engram output (block L's input) and its input (ViT V2: identity on the span)
                    on_hidden(f"{L}:engram", streams, s_in)
            streams, pre = blk(streams, pre, real_len=real_len, on_hidden=on_hidden, prof=prof)
            if on_hidden is not None:
                on_hidden(f"{L}:out", streams, pre)
        if self.kv_attn is not None:
            h = self.kv_norm(self.kv_hc.collapse(streams, pre))
            self.kv_attn.kv_only(h, seq_len_actual=real_len, state=self.kv_state)
            ttnn.deallocate(h)
            mark("kv_only")
        self.kv_actual += real_len
        self._run_after_engram_hook()
        if prof is not None:
            self.last_profile = prof.report(f"chunk [{start}, {start + real_len}) slot {self.slot}")

    def _chunk_islands(self, part: list[int], start: int, engram_host, kv_only20: bool = False):
        """``_chunk`` on the captured islands: the embedding streams, the pre row and the Engram outputs are eager (owned,
        freed by the block after its copy-in); each block's outputs are its island's persistent buffers. ``kv_only20``:
        layer 20 replays only its island A, then KV-only (``V41PrefillBlock.kv_only_islands``); 21..39 are skipped."""
        real_len = len(part)
        prof = PhaseProfile(self.mesh) if PROFILE else None
        mark = prof if prof is not None else (lambda name: None)
        if HP.ON:
            HP.start()
        rows = {}

        def _fetch():
            # layer 14's rows are only needed at block 14: gather them in the background while blocks 0..13 run
            fetch = getattr(engram_host, "rows_async", None)
            return fetch(self.slot, part, start) if fetch is not None else engram_host.rows_for(self.slot, part, start)

        if self.engrams and engram_host is not None and ENGRAM_EARLY:
            # DS41F-0068: the fetch starts BEFORE the embedding (its host gather + upload, ~140 ms at 8192) so layer 1's rows are
            # not waited for (V41_PREFILL_ENGRAM_EARLY=0: after it, as before)
            rows = _fetch()
            mark("engram.host")
        if HP.ON:
            HP.mark("engram.fetch_start")
        streams = self._streams(part)
        self._route_bias(part)
        cached = getattr(self, "_pre_cached", None) if PRE_CACHED else None
        pre = cached if cached is not None else identity_pre_row(self.mesh, self.chunk // self.sp, self.sp_axis)
        mark("embed")
        if HP.ON:
            HP.mark("embed")
        if self.engrams and engram_host is not None and not ENGRAM_EARLY:
            rows = _fetch()
            mark("engram.host")
        owned = True
        sup = getattr(self, "_super", None) or {}
        skip_until = -1
        pl = getattr(self, "_plan", None)
        for bi, blk in enumerate(self.blocks):
            L = blk.layer
            if bi < skip_until:
                continue
            if pl is not None and bi == pl["s0"] and self._super_plan_eligible(real_len, streams, pre, owned):
                streams, pre = self._run_super_plan(rows, mark)
                owned = False
                skip_until = pl["end"] + 1
                continue
            if bi in sup:
                idx, isl = sup[bi]
                group = [self.blocks[i] for i in idx]
                prev_out = list(self.blocks[bi - 1]._islands[1].outputs)
                if (
                    not owned
                    and all(a is b for a, b in zip(list(streams) + [pre], prev_out))
                    and all(b.super_eligible(real_len) for b in group)
                ):
                    for b in group:
                        b.super_before(real_len)
                    if HP.ON:
                        HP.mark("super.prepare")
                    outs = isl.replay()
                    if HP.ON:
                        HP.mark("super.replay")
                    for b in group:
                        b.super_after(real_len)
                    streams, pre = list(outs[:HC]), outs[HC]
                    owned = False
                    skip_until = idx[-1] + 1
                    mark(f"super.L{group[0].layer}-{group[-1].layer}")
                    continue
            if L in self.engrams:
                if HP.ON:
                    HP.mark("other")
                r = rows[L].result() if hasattr(rows[L], "result") else rows[L]
                mark("engram.wait")
                if HP.ON:
                    HP.mark("engram.wait")
                rd = self.engrams[L].upload_rows(r, self.chunk)
                if HP.ON:
                    HP.mark("engram.upload")
                isl = getattr(self.engrams[L], "_island", None) is not None
                # DS41F-0163 V41_PREFILL_ENGRAM_ISLAND: the island's outputs are persistent (not owned: never freed here)
                new = self.engrams[L].forward_island(streams, rd) if isl else self.engrams[L](streams, rd)
                ttnn.deallocate(rd)
                if owned:
                    for t in streams:
                        ttnn.deallocate(t)
                streams, owned = new, not isl
                mark("engram.dev")
                if HP.ON:
                    HP.mark("engram.forward")
                self._run_after_engram_hook()
            if kv_only20 and L >= self.cfg.first_decoder_layer:
                if HP.ON:
                    HP.mark("other")
                blk.kv_only_islands(
                    streams, pre, real_len=real_len, owned=owned, owned_pre=blk is self.blocks[0] and cached is None
                )
                mark("kv_only")
                if HP.ON:
                    HP.mark("kv_only20")
                break
            streams, pre = blk.forward_islands(
                streams, pre, real_len=real_len, owned=owned, owned_pre=blk is self.blocks[0] and cached is None
            )
            owned = False
            mark(f"block.{type(blk.attn).__name__}")
        if self.kv_attn is not None:
            h = self.kv_norm(self.kv_hc.collapse(streams, pre))
            self.kv_attn.kv_only(h, seq_len_actual=real_len, state=self.kv_state)
            ttnn.deallocate(h)
            mark("kv_only")
        self.kv_actual += real_len
        self._run_after_engram_hook()  # a chunk whose Engram layers ran inside a super segment (or none)
        if prof is not None:
            self.last_profile = prof.report(f"chunk [{start}, {start + real_len}) slot {self.slot} (islands)")
        if HP.ON:
            HP.mark("other")
            logger.info(
                f"[v41 hostprof] chunk [{start}, {start + real_len}) slot {self.slot} kv20={int(kv_only20)} {HP.report()}"
            )

    def _run_after_engram_hook(self) -> None:
        """V41_PREFILL_HOST_IDS (WS13): the runtime's once-per-chunk callback, run right after the first Engram layer is
        issued -- this chunk's own Engram hash is done (it advances the slot's n-gram history) and the device has the rest
        of the chunk queued, so host work here hides under device time."""
        hook = getattr(self, "_after_engram_hook", None)
        if hook is None:
            return
        self._after_engram_hook = None
        hook()
        if HP.ON:
            HP.mark("engram.prefetch")

    # ---- export ---------------------------------------------------------------------------------------------------
    def _window(self, state) -> torch.Tensor:
        """The carry (positions [S - 128, S) in order) -> model.py's ``window_kv_cache`` layout (slot = position % 128)."""
        carry = _first_device(state.sliding_carry)[0, 0]  # [128, 512]
        win = int(carry.shape[0])
        S = self.kv_actual
        out = torch.zeros_like(carry)
        for i in range(win):
            q = S - win + i
            if q >= 0:
                out[q % win] = carry[i]
        return out

    def cache_snapshot(self) -> dict:
        """tt-blaze ``prefill_trace.cache_snapshot_of`` for layers 0..n_layers-1 (+ the KV-only layer): window ring
        (``[1, 128, 512]``, slot p % 128), and for KV sources the compressed rows ``[1, S // ratio, 512]`` and the index keys
        ``[1, S // ratio, 128]``; ``kv_state`` / ``score_state`` are omitted (V4.1's pooling never straddles an even S).
        """
        S = self.kv_actual
        out = {}
        mods = [(b.layer, b.attn, b.state) for b in self.blocks]
        if self.kv_attn is not None:
            mods.append((self.kv_only_layer, self.kv_attn, self.kv_state))
        for L, attn, st in mods:
            d = dict(window_kv_cache=self._window(st)[None])
            r = self.cfg.role(L)
            if r.mode == "full":
                n = S // r.compress_ratio
                d["compress_kv_cache"] = _first_device(st.compressed_kv)[0, 0, :n][None]
                d["index_k_cache"] = _first_device(st.index_k)[0, 0, :n][None]
            out[L] = d
        return dict(end_pos=S, layers=out)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.cache_snapshot(), path)
