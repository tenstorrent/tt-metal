# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fast PUP prefill runtime: the session-5 grouped prefill (one row) behind the tt-d-gen runner contract.

Same contract and published state as ``runtime.TtQwen35PupPrefillRuntime`` (configs 00-09, one chunk per request,
``actual_start == 0``); three things differ, all inside the runner:

1. **System-prompt snapshot.** PUP's system turn is the same for every request of a task (553-640 tokens). The first
   request whose system turn has not been seen prefills ``[0, P)`` once (``P`` = the system turn floored to 128) and
   keeps a device snapshot: the 24 GDN states at ``P`` and the 8 attention layers' K / V rows ``[0, P)``. Every later
   request with the same ``[0, P)`` tokens restores it (only when the slot holds another prefix) and prefills just its
   tail ``[P, L)``. The engine still sees a request prefilled from 0 (``actual_start = 0``) and gets the full state
   ``[0, L)``; the K / V of ``[0, P)`` and the GDN carry at ``P`` are what a split prefill at ``P`` computes. Legal
   under the prefill contract: the runtime publishes the state after the last token it was given, and the engine's
   tokens fully determine the snapshot it resumes from (the tokens of ``[0, P)`` are compared, not hashed only).
2. **Traced buckets of every size.** The tail runs as the grouped prefill body (tt-blaze co_prefill / q36s:
   ``forward_prefill_group`` with the GDN initial state + conv history of the snapshot, attention ``chunk_start = P``,
   token-parallel residual stream from 1536 rows) captured per bucket up to 3584 rows; nothing runs eagerly at
   request time except the snapshot copies.
3. **Per-layer-group publish.** Each bucket is captured as ``segments`` traces of consecutive layers, each with the
   publish (K / V -> the engine cache, GDN -> the Snapshot slabs) of its own layers inside. All segments are enqueued
   back to back; the host waits on an event per segment and reports those layers to the completion sink, so the KV
   manager can migrate layer group k while group k + 1 computes.

One rank, one slot (``num_users == 1``: the K / V publish bakes slot 0 into the trace).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.runners import kv_table
from models.demos.blackhole.qwen36.tt.runners import media as pup_media
from models.demos.blackhole.qwen36.tt.runners.runtime import RuntimeConfig, TtQwen35PupPrefillRuntime


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


@dataclass
class _Prefix:
    """One system-prompt snapshot (device buffers preallocated before any capture)."""

    index: int
    rec: list  # per GDN layer [1, Nv, Dk, Dv] fp32
    hist: list  # per GDN layer [1, K-1, C] bf16 RM
    k: list  # per attention layer [1, 1, prefix_max, hd] (cache dtype)
    v: list
    tokens: torch.Tensor | None = None  # [P] int32; None = free
    P: int = 0
    last_used: float = 0.0
    hits: int = 0


@dataclass
class _Bucket:
    Tb: int
    cfg: object  # model._BatchedShortCfg (Bg = 1)
    cos_cat: object = None
    sin_cat: object = None
    vis_rm: object = None
    vmask: object = None
    export_rows: int = 0
    traces: list = field(default_factory=list)  # [(trace id, layer start, layer end)]
    outs: list = field(default_factory=list)  # segment outputs kept alive (the next segment's input)
    host_keys: dict = field(default_factory=dict)  # what each staged buffer holds now
    host_vis: object = None  # [tp, Tb, dim / tp] host staging of the image rows


class TtQwen35PupFastPrefillRuntime(TtQwen35PupPrefillRuntime):
    def __init__(self, *, mesh_device, model, config: RuntimeConfig, media_reader=None, prefix_end_token=None):
        super().__init__(mesh_device=mesh_device, model=model, config=config, media_reader=media_reader)
        pc = config.pup
        assert config.num_users == 1, "the fast runtime bakes slot 0 into the publish (num_users = 1)"
        self.buckets_rows = tuple(sorted(pc.fast_buckets))
        self.segments = int(pc.publish_segments)
        self.prefix_max = int(pc.prefix_max)
        self.prefix_align = int(pc.prefix_align)
        self.prefix_end_token = prefix_end_token
        self.n_prefix_slots = int(pc.prefix_slots)
        self.buf_blocks = -(-(self.blocks_per_slot + self.buckets_rows[-1] // self.block) // 32) * 32
        self.trash = config.num_users * self.blocks_per_slot
        self.args = model.args
        # the arbitrary-length tail: the conv-select shortcut assumes the last 3 rows sit in the bucket's last W rows
        self.args.prefill_gdn_convsel_tail = 0
        self.tokpar_min_rows = int(pc.tokpar_min_rows)
        n = len(model.layers)
        per = -(-n // self.segments)
        self.seg_bounds = [(s, min(n, s + per)) for s in range(0, n, per)]
        self.dn_list = [l.attention for l in model.layers if not l.is_full_attention]
        self._buckets = {}
        self._prefixes = []
        self._slot_prefix = None  # the _Prefix whose K / V rows the slot's blocks hold (None: unknown)
        self._init_prefix = None  # the _Prefix loaded into the bound GDN init (0 = zeros)
        self._kv = None
        self.stats = dict(snapshots=0, restores=0, hits=0, misses=0)

    # ------------------------------------------------------------------------------------------------ compile --
    def compile(self, kv_caches) -> None:
        m, dev = self.model, self.mesh_device
        rep = ttnn.ReplicateTensorToMesh(dev)
        self._kv = kv_caches
        self._kv_ref = kv_caches
        t0 = time.perf_counter()
        st = self._slot_scalar(0)
        m.prefill_trash_block = self.trash

        def up(t, dtype, layout, mapper=rep, mc=None):
            kw = {} if mc is None else {"memory_config": mc}
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, mesh_mapper=mapper, **kw)

        dn0 = self.dn_list[0]
        C, K = dn0.qkv_dim_tp, dn0.K
        # ---- every persistent buffer, before any capture ----
        # the bound GDN init the traces read (one row: recurrent state + the conv op's [1, K-1, C] RM history)
        self.init = {
            id(dn): {
                "rec": up(torch.zeros(1, dn.Nv, dn.Dk, dn.Dv), ttnn.float32, ttnn.TILE_LAYOUT),
                "hist": up(torch.zeros(1, K - 1, C, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            }
            for dn in self.dn_list
        }
        self.zero_init = {
            id(dn): {
                "rec": up(torch.zeros(1, dn.Nv, dn.Dk, dn.Dv), ttnn.float32, ttnn.TILE_LAYOUT),
                "hist": up(torch.zeros(1, K - 1, C, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            }
            for dn in self.dn_list
        }
        att0 = m.layers[self.attn_layer_ids[0]].attention
        kdt, hd = att0.paged_k.dtype, int(att0.paged_k.shape[-1])
        pb = self.prefix_max // self.block
        for i in range(self.n_prefix_slots):
            self._prefixes.append(
                _Prefix(
                    index=i,
                    rec=[up(torch.zeros(1, dn.Nv, dn.Dk, dn.Dv), ttnn.float32, ttnn.TILE_LAYOUT) for dn in self.dn_list],
                    hist=[
                        up(torch.zeros(1, K - 1, C, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                        for _ in self.dn_list
                    ],
                    k=[
                        up(torch.zeros(1, 1, self.prefix_max, hd), kdt, ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG)
                        for _ in self.attn_layer_ids
                    ],
                    v=[
                        up(torch.zeros(1, 1, self.prefix_max, hd), kdt, ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG)
                        for _ in self.attn_layer_ids
                    ],
                )
            )
        self.prefix_pt = up(torch.arange(pb, dtype=torch.int32).reshape(1, pb), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        shard = ttnn.ShardTensor2dMesh(dev, dims=(None, -1), mesh_shape=self.args.cluster_shape)
        rd = None
        for Tb in self.buckets_rows:
            cfg = m._alloc_batched_short_cfg(Tb, 1, self.buf_blocks)
            b = _Bucket(Tb=Tb, cfg=cfg)
            rd = int(cfg.cos.shape[-1])
            b.cos_cat = up(torch.ones(1, 1, Tb, rd), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.sin_cat = up(torch.zeros(1, 1, Tb, rd), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.vis_rm = up(
                torch.zeros(1, 1, Tb, self.args.dim, dtype=torch.bfloat16),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                mapper=shard,
                mc=ttnn.DRAM_MEMORY_CONFIG,
            )
            b.vmask = up(
                torch.zeros(1, 1, Tb, 1, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG
            )
            b.export_rows = min(self.config.max_seq_len, _next_pow2(self.prefix_max + Tb))
            self._buckets[Tb] = b
        logger.info(
            f"[qwen35_9b_pup fast] buffers: {len(self._buckets)} buckets {self.buckets_rows}, {self.n_prefix_slots} "
            f"prefix slots x {self.prefix_max} rows, segments {self.seg_bounds}"
        )
        # ---- warm: every program the request path runs, eagerly, twice ----
        self._build_rope(torch.zeros(1, 8, dtype=torch.int32), None)
        for Tb, b in self._buckets.items():
            self._stage(b, torch.zeros(1, Tb, dtype=torch.int32), P=0, n=Tb, vis=None)
            for _ in range(2):
                x = None
                for s, (l0, l1) in enumerate(self.seg_bounds):
                    xn = self._segment(b, x, s, l0, l1, st)
                    if x is not None:
                        ttnn.deallocate(x)
                    x = xn
                ttnn.synchronize_device(dev)
        snap = self._prefixes[0]
        self._take_snapshot(snap, self._buckets[self.buckets_rows[0]])
        self._restore(snap)
        self._load_init(None)
        ttnn.synchronize_device(dev)
        self._warm_s = time.perf_counter() - t0
        self._captured = False
        logger.info(f"[qwen35_9b_pup fast] warm {self._warm_s:.1f} s; traces are captured at the first request")

    def _capture(self):
        """Capture every bucket's segment traces. Deferred to the first request: the runner allocates its socket /
        migration buffers after compile(), and a buffer allocated after a capture can sit in memory the trace's
        intermediates reuse (seen: the next request's H2D metadata read back as garbage after one replay)."""
        dev, st = self.mesh_device, self._slot_scalar(0)
        t1 = time.perf_counter()
        for Tb, b in self._buckets.items():
            x = None
            for s, (l0, l1) in enumerate(self.seg_bounds):
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                x = self._segment(b, x, s, l0, l1, st)
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                b.traces.append((tid, l0, l1))
                b.outs.append(x)
            ttnn.synchronize_device(dev)
        for snap in self._prefixes:
            snap.tokens = None
        self._slot_prefix = None
        self._init_prefix = None
        self._captured = True
        self.model._log_trace_region("fast prefill captured")
        logger.info(
            f"[qwen35_9b_pup fast] captured {sum(len(b.traces) for b in self._buckets.values())} traces in "
            f"{time.perf_counter() - t1:.1f} s (warm {self._warm_s:.1f} s), export rows per bucket "
            f"{[b.export_rows for b in self._buckets.values()]}"
        )

    # ------------------------------------------------------------------------------------------- the body --
    def _segment(self, b, x, s, l0, l1, slot_tensor):
        """Layers [l0, l1) of the one-row grouped prefill on bucket b + the publish of those layers. Segment 0 embeds
        the tokens and splices the image rows; the token-parallel stream stays token-parallel across segments."""
        m, cfg, Tb = self.model, b.cfg, b.Tb
        tokpar = bool(getattr(self.args, "prefill_tokpar", False)) and Tb >= self.tokpar_min_rows
        if s == 0:
            x = m.embd(cfg.tok)
            x = ttnn.reshape(x, (1, 1, Tb, x.shape[-1]))
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            vis = ttnn.to_layout(b.vis_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x2 = ttnn.where(b.vmask, vis, x)
            ttnn.deallocate(vis)
            ttnn.deallocate(x)
            x = x2
            if tokpar:
                from models.demos.blackhole.qwen36.tt.layer import tokpar_a2a

                xt = tokpar_a2a(x, to_tokens=True)
                ttnn.deallocate(x)
                x = xt
        pg = {
            "Bg": 1,
            "Tb": Tb,
            "cos": [cfg.cos],
            "sin": [cfg.sin],
            "csi": cfg.csi,
            "full_pts": cfg.full_pts,
            "chunk_pts": cfg.chunk_pts,
            "gdn_masks": cfg.gdn_masks,
            "gdn_scratch": m._batched_gdn_scratch[1],
            "gdn_init": self.init,
            "cos_cat": b.cos_cat,
            "sin_cat": b.sin_cat,
        }
        for li in range(l0, l1):
            layer = m.layers[li]
            if tokpar:
                xn = layer.forward_tokpar(x, pg, chunk_size=self.args.gdn_chunk_size)
            else:
                xn = layer.forward(
                    x, cos=cfg.cos, sin=cfg.sin, mode="prefill", chunk_size=self.args.gdn_chunk_size, prefill_group=pg
                )
            if s > 0 and li == l0:
                pass  # the previous segment's output is a kept trace output: never freed
            else:
                ttnn.deallocate(x)
            x = xn
        self._publish_layers(b, l0, l1, slot_tensor)
        if l1 == len(m.layers):
            ttnn.deallocate(x)  # no head: the last hidden state is not needed
            return None
        # the hand-off to the next segment lives in DRAM: a kept L1 buffer would sit under later programs' static CBs
        if x.memory_config().buffer_type == ttnn.BufferType.DRAM:
            return x
        xd = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        return xd

    def _publish_layers(self, b, l0, l1, slot_tensor):
        """K / V rows [0, export_rows) of the slot's blocks -> the engine cache; GDN state of the group scratch -> the
        Snapshot slabs (slot read on device). Rows past the request's length are stale and never migrated."""
        m, kv = self.model, self._kv
        n = b.export_rows
        nb = n // self.block
        scr = m._batched_gdn_scratch[1]
        for li in range(l0, l1):
            layer = m.layers[li]
            if layer.is_full_attention:
                pos = self.attn_layer_ids.index(li)
                att = layer.attention
                for src, dst in ((att.paged_k, kv.k), (att.paged_v, kv.v)):
                    hd = int(src.shape[-1])
                    blk = ttnn.slice(src, [0, 0, 0, 0], [nb, 1, self.block, hd])
                    x = ttnn.reshape(blk, [1, 1, n, hd])
                    if dst is kv.k:
                        z = kv_table.rotate_half_to_interleaved(x)
                        ttnn.deallocate(x)
                        x = z
                    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                        dst, x, slot_idx=0, layer_idx=pos, num_layers=kv.num_attn_layers, kv_actual_global=0, cluster_axis=0
                    )
                    ttnn.deallocate(x)
            else:
                dn = layer.attention
                gs = scr[id(dn)]
                conv = ttnn.concat([gs.conv[k] for k in range(1, dn.K)], dim=1)  # [1, K-1, C]
                kv.gdn.export_layer(gs.rec, conv, slot_tensor, li)
                ttnn.deallocate(conv)

    # -------------------------------------------------------------------------------------- snapshot / restore --
    def _take_snapshot(self, snap, b):
        """GDN state (group scratch: the state after the bucket's last real row) + K / V rows [0, prefix_max) of the
        slot's blocks -> snap's buffers."""
        m = self.model
        scr = m._batched_gdn_scratch[1]
        for i, dn in enumerate(self.dn_list):
            gs = scr[id(dn)]
            ttnn.copy(gs.rec, snap.rec[i])
            h = ttnn.concat([gs.conv[k] for k in range(1, dn.K)], dim=1)
            hr = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy(hr, snap.hist[i])
            ttnn.deallocate(h)
            ttnn.deallocate(hr)
        pb = self.prefix_max // self.block
        for pos, li in enumerate(self.attn_layer_ids):
            att = m.layers[li].attention
            for src, dst in ((att.paged_k, snap.k[pos]), (att.paged_v, snap.v[pos])):
                hd = int(src.shape[-1])
                blk = ttnn.slice(src, [0, 0, 0, 0], [pb, 1, self.block, hd])
                x = ttnn.reshape(blk, [1, 1, self.prefix_max, hd])
                ttnn.copy(x, dst)
                ttnn.deallocate(x)

    def _restore(self, snap):
        """The snapshot's K / V rows [0, prefix_max) back into the slot's blocks (rows past P are overwritten by the
        tail or never read)."""
        m = self.model
        for pos, li in enumerate(self.attn_layer_ids):
            att = m.layers[li].attention
            ttnn.experimental.paged_fill_cache(att.paged_k, snap.k[pos], self.prefix_pt, batch_idx=0)
            ttnn.experimental.paged_fill_cache(att.paged_v, snap.v[pos], self.prefix_pt, batch_idx=0)
        self._slot_prefix = snap

    def _load_init(self, snap):
        """The traces' bound GDN init <- the snapshot's state (or zeros for a request prefilled from 0)."""
        for i, dn in enumerate(self.dn_list):
            dst = self.init[id(dn)]
            if snap is None:
                src_r, src_h = self.zero_init[id(dn)]["rec"], self.zero_init[id(dn)]["hist"]
            else:
                src_r, src_h = snap.rec[i], snap.hist[i]
            ttnn.copy(src_r, dst["rec"])
            ttnn.copy(src_h, dst["hist"])
        self._init_prefix = snap if snap is not None else 0

    # -------------------------------------------------------------------------------------------------- staging --
    def _build_rope(self, ids, grid):
        self.model._req_image_grid_thw = grid
        self.model._req_video_grid_thw = None
        self.model.rope.build_request_rope(ids, image_grid_thw=grid, video_grid_thw=None)

    def _bucket_for(self, n):
        for Tb in self.buckets_rows:
            if Tb >= n:
                return self._buckets[Tb]
        return None

    def _stage(self, b, tail_ids, *, P, n, vis):
        """Host -> device copies of one request's tail [P, P + n) on bucket b. vis: None or (positions in the tail,
        rows [k, dim] bf16)."""
        cfg, Tb, rep = b.cfg, b.Tb, ttnn.ReplicateTensorToMesh(self.mesh_device)
        keep = []

        def put(t, dtype, layout, dst, mapper=rep):
            h = ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper)
            keep.append(h)
            ttnn.copy_host_to_device_tensor(h, dst)

        tok = torch.zeros(1, Tb, dtype=torch.int32)
        tok[0, :n] = tail_ids[0, :n].to(torch.int32)
        put(tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, cfg.tok)
        L = P + n
        nreal = -(-L // self.block)
        cb0 = P // self.block
        row = torch.full((1, max(self.buf_blocks, cb0 + cfg.blocks_per_bucket)), self.trash, dtype=torch.int32)
        row[0, : self.blocks_per_slot] = torch.arange(self.blocks_per_slot, dtype=torch.int32)
        row[0, nreal:] = self.trash
        key_pt = (P, nreal)
        if b.host_keys.get("pt") != key_pt:
            put(row[:, : self.buf_blocks].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, cfg.full_pts[0])
            put(row[:, cb0 : cb0 + cfg.blocks_per_bucket].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, cfg.chunk_pts[0])
            put(torch.tensor([P], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, cfg.csi)
            b.host_keys["pt"] = key_pt
        cos, sin = self.model._rope_tp_cos_sin_torch(P, Tb)
        put(cos, ttnn.bfloat16, ttnn.TILE_LAYOUT, b.cos_cat)
        put(sin, ttnn.bfloat16, ttnn.TILE_LAYOUT, b.sin_cat)
        if b.host_keys.get("n") != n:
            bg, conv_sel, _ = self.model._batched_host_masks_torch(Tb, [n], cfg.n_hist)
            put(bg, ttnn.float32, ttnn.TILE_LAYOUT, cfg.m_bg)
            put(conv_sel, ttnn.bfloat16, ttnn.TILE_LAYOUT, cfg.conv_sel)
            b.host_keys["n"] = n
        if vis is not None:
            # image rows straight into a per-bucket host staging buffer laid out per chip ([tp, Tb, dim / tp]: chip c
            # holds hidden columns [c * dim / tp, ...), the vis_rm sharding); rows outside the images keep stale
            # values (the mask selects only image positions), so nothing is zeroed
            tp = int(self.mesh_device.get_num_devices())
            if b.host_vis is None:
                b.host_vis = torch.zeros(tp, Tb, self.args.dim // tp, dtype=torch.bfloat16)
            mask = torch.zeros(1, 1, Tb, 1, dtype=torch.bfloat16)
            for s0, r in vis:
                k = int(r.shape[0])
                src = torch.from_numpy(np.ascontiguousarray(r).view(np.int16)).view(torch.bfloat16)
                b.host_vis[:, s0 : s0 + k, :] = src.view(k, tp, -1).transpose(0, 1)
                mask[0, 0, s0 : s0 + k, 0] = 1.0
            h = ttnn.from_host_shards(
                [
                    ttnn.from_torch(b.host_vis[c].view(1, 1, Tb, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                    for c in range(tp)
                ],
                self.mesh_device.shape,
            )
            keep.append(h)
            ttnn.copy_host_to_device_tensor(h, b.vis_rm)
            put(mask, ttnn.bfloat16, ttnn.TILE_LAYOUT, b.vmask)
            b.host_keys["vis"] = True
        elif b.host_keys.get("vis", True):
            put(torch.zeros(1, 1, Tb, 1, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, b.vmask)
            b.host_keys["vis"] = False
        return keep

    def _run(self, b, slot_id, actual_start, actual_end, report):
        """Replay the bucket's segments back to back; report each layer group when its event completes."""
        evs = []
        for tid, l0, l1 in b.traces:
            ttnn.execute_trace(self.mesh_device, tid, cq_id=0, blocking=False)
            evs.append((ttnn.record_event(self.mesh_device, 0), l0, l1))
        for ev, l0, l1 in evs:
            ttnn.event_synchronize(ev)
            if report and self._sink is not None:
                self._sink.layers_completed(l0, l1, slot_id, actual_start, actual_end)

    # -------------------------------------------------------------------------------------------- the request --
    def _prefix_len(self, ids):
        """P for this prompt: its system turn (through the first prefix-end token; the whole chunk when the engine's cut
        falls inside the turn) floored to prefix_align, at most the chunk minus 3 tokens; 0 when that leaves no aligned
        prefix within prefix_max."""
        if self.prefix_end_token is None:
            return 0
        L = int(ids.shape[1])
        hit = (ids[0] == int(self.prefix_end_token)).nonzero()
        end = int(hit[0, 0]) + 1 if hit.numel() else L
        P = min(end, L - 3) // self.prefix_align * self.prefix_align
        if P < self.prefix_align or P > self.prefix_max:
            return 0
        if self.image_token is not None and bool((ids[0, :P] == self.image_token).any()):
            return 0
        return P

    def _find_prefix(self, ids, P):
        for snap in self._prefixes:
            if snap.tokens is not None and snap.P == P and torch.equal(snap.tokens, ids[0, :P]):
                return snap
        return None

    def prefill_chunk(
        self,
        input_tensor,
        kv_cache,
        *,
        slot_id,
        actual_start,
        actual_end,
        request_id=0,
        media_refs=None,
        d2h_service=None,
        metadata_msg=None,
    ):
        del d2h_service, metadata_msg
        if actual_start != 0:
            raise NotImplementedError(
                f"qwen35_9b_pup: chunk [{actual_start}, {actual_end}) of slot {slot_id} resumes a request; one chunk "
                "per request (no engine resume of a recurrent model without a snapshot)"
            )
        assert slot_id == 0, "fast runtime: one slot"
        L = int(actual_end)
        ids = self._ids_for(input_tensor, L).to(torch.int32)  # host copy first: nothing below reads the input tensor
        if not self._captured:
            self._capture()
        t0 = time.perf_counter()
        refs = pup_media.check_refs(
            pup_media.refs_in_chunk(media_refs, 0) if media_refs and hasattr(media_refs[0], "chunk_index") else [],
            actual_start,
            actual_end,
            ids[0].tolist(),
            self.image_token,
        )
        grid, rows = None, None
        if refs:
            if self.media_reader is None:
                raise pup_media.MediaSpanError("media refs but no embedding store reader is configured")
            values = pup_media.fetch_values(self.media_reader, refs)
            grid = torch.tensor([list(h.grid_thw) for h, _ in values], dtype=torch.long)
            rows = [(int(r.start_pos), np.asarray(v)) for r, (_, v) in zip(refs, values)]  # (span start, rows) per image
        t_media = time.perf_counter()
        P = self._prefix_len(ids)
        snap = self._find_prefix(ids, P) if P else None
        t_snap = 0.0
        if P and snap is None:
            # first request with this system turn: prefill [0, P) from zero on its bucket, keep the snapshot
            ts = time.perf_counter()
            snap = min(self._prefixes, key=lambda s: (s.tokens is not None, s.last_used))
            snap.tokens, snap.P = None, P
            bp = self._bucket_for(P)
            self._build_rope(ids[:, :P], None)
            if self._init_prefix != 0:
                self._load_init(None)
            keep = self._stage(bp, ids[:, :P], P=0, n=P, vis=None)
            self._run(bp, slot_id, 0, P, report=False)
            keep.clear()
            self._take_snapshot(snap, bp)
            ttnn.synchronize_device(self.mesh_device)
            snap.tokens = ids[0, :P].clone()
            self._slot_prefix = snap
            self.stats["snapshots"] += 1
            t_snap = time.perf_counter() - ts
            logger.info(f"[qwen35_9b_pup fast] system-turn snapshot P={P} into prefix slot {snap.index} ({t_snap * 1e3:.1f} ms)")
        if snap is None:
            P = 0
            self.stats["misses"] += 1
        else:
            snap.last_used = time.time()
            snap.hits += 1
            self.stats["hits"] += 1
        n = L - P
        b = self._bucket_for(n)
        if b is None or n < 3:
            raise NotImplementedError(f"qwen35_9b_pup fast: tail of {n} tokens (P={P}) has no bucket {self.buckets_rows}")
        # restore: the slot's K / V prefix rows and the traces' GDN init
        tr = time.perf_counter()
        restored = 0
        if snap is not None and self._slot_prefix is not snap:
            self._restore(snap)
            restored += 1
        want = snap if snap is not None else 0
        if self._init_prefix is not want:
            self._load_init(snap)
            restored += 1
        if restored:
            self.stats["restores"] += 1
        t_restore = time.perf_counter() - tr
        # rope over the whole prompt (M-RoPE from the stored grids), tail staging
        self._build_rope(ids, grid)
        vis = None
        if rows is not None:
            n_img = int((ids[0, P:] == self.image_token).sum())
            assert n_img == sum(int(r.shape[0]) for _, r in rows), (n_img, [tuple(r.shape) for _, r in rows])
            vis = [(s0 - P, r) for s0, r in rows]
        keep = self._stage(b, ids[:, P:], P=P, n=n, vis=vis)
        keep.append(self._publish_slot_record(slot_id, grid, len(refs)))
        t_stage = time.perf_counter()
        self._run(b, slot_id, actual_start, actual_end, report=True)
        keep.clear()
        if snap is None:
            self._slot_prefix = None  # a from-0 request overwrote the slot's prefix rows
        t_end = time.perf_counter()
        self.last_timing = dict(
            media_ms=(t_media - t0) * 1e3,
            snapshot_ms=t_snap * 1e3,
            restore_ms=t_restore * 1e3,
            stage_ms=(t_stage - tr - t_restore) * 1e3,
            prefill_ms=(t_end - t_stage) * 1e3,
            export_ms=0.0,
            total_ms=(t_end - t0) * 1e3,
            images=len(refs),
            tokens=L,
            prefix=P,
            bucket=b.Tb,
            rope_delta=self._last_record[0],
            presence=self._last_record[1],
        )
        logger.info(
            "[qwen35_9b_pup] PUPPRE "
            + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in self.last_timing.items())
        )
        return None
