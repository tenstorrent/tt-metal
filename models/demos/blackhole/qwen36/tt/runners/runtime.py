# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The PUP prefill runtime the common prefill runner drives (``ADDING_A_PREFILL_MODEL.md`` section 2).

One (1, 4) column, one rank (first and last). Per chunk:
  1. the chunk's image rows from the embedding store (``media.py``; one batched read, grids from the headers);
  2. the s5-profile prefill of the chunk on the model's own B=1 working state (``Qwen36Model.prefill_traced_chunked``:
     traced per bucket, M-RoPE from the ids + the stored grids, GDN from zero at a request's head chunk);
  3. state publish in the d-gen form: the 8 attention layers' K / V rows ``[actual_start, actual_end)`` into the
     engine-owned M3-layout cache (``update_padded_kv_cache``), every GDN layer's recurrent + conv state into the
     engine-owned Snapshot slabs (``indexed_entry_copy``, slot read on device);
  4. one layer-completion record for the chunk (all 32 layers) once 3 is enqueued.
No LM head output is used: decode computes the last prompt token (the engine prefills L - 1 tokens of a recurrent
model, so the published GDN state is the one after position L - 2).

Limits of this version (the runner refuses what it cannot serve with a clear error):
  - a request is one chunk (``chunk_size`` 4096 covers every PUP request); a chunk with ``actual_start > 0`` (a
    multi-chunk prompt, or an engine prefix-reuse resume) is refused: resuming needs the GDN carry imported from the
    slab into the model's working state, and the engine must not resume a recurrent model at a block boundary it has no
    snapshot for;
  - packed forwards (F > 1) run chunk by chunk through ``prefill_chunk``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.runners import kv_table
from models.demos.blackhole.qwen36.tt.runners import media as pup_media
from models.demos.blackhole.qwen36.tt.runners.config import PupPrefillConfig


@dataclass
class RuntimeConfig:
    """The fields the runner reads off ``runtime.config``."""

    chunk_size: int
    max_seq_len: int
    num_users: int
    first_layer_idx: int = 0
    num_layers: int = 32
    is_first_rank: bool = True
    is_last_rank: bool = True
    use_trace: bool = True
    pup: PupPrefillConfig = field(default_factory=PupPrefillConfig)


class TtQwen35PupPrefillRuntime:
    def __init__(self, *, mesh_device, model, config: RuntimeConfig, media_reader=None):
        self.mesh_device = mesh_device
        self.model = model
        self.config = config
        self.media_reader = media_reader
        args = model.args
        self.block = config.pup.block_size
        self.blocks_per_slot = config.max_seq_len // self.block
        # slot s owns blocks [s * bps, (s + 1) * bps) of the model's paged cache (identity page table)
        self.page_table = torch.arange(config.num_users * self.blocks_per_slot, dtype=torch.int32).reshape(
            config.num_users, self.blocks_per_slot
        )
        self.attn_layer_ids = tuple(i for i, layer in enumerate(model.layers) if layer.is_full_attention)
        self.gdn_layer_ids = tuple(i for i, layer in enumerate(model.layers) if not layer.is_full_attention)
        self.image_token = int(getattr(args.hf_config, "image_token_id", 0) or 0) or None
        self._sink = None
        self._host_ids = None  # (device input, host ids) of the last make_chunk_input
        self._slot_scalars = {}
        self.last_timing = {}
        self.last_logits = None
        self.profile = False

    # --- runner contract ---------------------------------------------------------------------------------------
    def compile(self, kv_caches) -> None:
        """Allocate and warm the model's traced prefill paths, then capture them. Everything that must persist across
        replays (every slot's index scalar, the publish ops' programs) is created BEFORE the first capture: a device
        buffer allocated after a capture can land on a captured trace's freed intermediates and be overwritten by
        every replay (seen here as a slot's published GDN state staying zero)."""
        m = self.model
        full_pt = self.page_table.reshape(1, -1)
        for slot in range(self.config.num_users):
            self._slot_scalar(slot)
        # the block past every slot's range: padded bucket K / V writes land there (the adapter allocates it)
        m.prefill_trash_block = self.config.num_users * self.blocks_per_slot
        seg = self.config.pup.segment_tokens
        m.prepare_prefill_trace_chunked(self.mesh_device, full_pt[:, : self.blocks_per_slot], chunk_size=seg)
        if self.config.use_trace:
            m.prepare_prefill_traces_short(self.mesh_device, full_pt[:, : self.blocks_per_slot])
        # Masked buckets above the short traces (a > 2048-token request runs as one bucket of the next 128 multiple)
        # compile here, before any capture: a program compiled after a trace is parked clobbers it (#48536). Seen
        # with 10 references: the 3200-row bucket compiled at request time gave garbage once a short trace had
        # replayed in between. PUP_PREFILL_WARM_BUCKETS lists them (comma-separated rows).
        long_buckets = [int(b) for b in os.environ.get("PUP_PREFILL_WARM_BUCKETS", "").split(",") if b.strip()]
        if long_buckets:
            t0 = time.perf_counter()
            m.warmup_prefill_masked_buckets(full_pt[:, : self.blocks_per_slot], buckets=long_buckets)
            logger.info(f"[qwen35_9b_pup] warmed masked buckets {long_buckets} in {time.perf_counter() - t0:.1f} s")
        self._export(kv_caches, slot=0, actual_start=0, actual_end=128)
        ttnn.synchronize_device(self.mesh_device)
        if self.config.use_trace:
            m.record_prefill_trace_chunked(self.mesh_device)
            m.capture_prefill_traces_short(self.mesh_device, full_pt[:, : self.blocks_per_slot])
        ttnn.synchronize_device(self.mesh_device)
        logger.info(
            f"[qwen35_9b_pup] compiled: slots {self.config.num_users} x {self.config.max_seq_len} tokens, "
            f"{len(self.attn_layer_ids)} attention + {len(self.gdn_layer_ids)} GDN layers, segment {seg}, "
            f"trace={self.config.use_trace}"
        )

    def make_chunk_input(self, token_ids):
        ids = torch.as_tensor(token_ids, dtype=torch.int32).reshape(1, -1)
        dev = ttnn.from_torch(
            ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._host_ids = (dev, ids)
        return dev

    def set_layer_completion_sink(self, sink) -> None:
        self._sink = sink

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
        del d2h_service, metadata_msg  # layer acks go through the completion sink; no D2H activation (one rank)
        if actual_start != 0:
            raise NotImplementedError(
                f"qwen35_9b_pup: chunk [{actual_start}, {actual_end}) of slot {slot_id} resumes a request; this runtime "
                "prefills a request in one chunk (chunk_size >= the prompt, no prefix-reuse resume of a recurrent model)"
            )
        n = int(actual_end) - int(actual_start)
        t0 = time.perf_counter()
        ids = self._ids_for(input_tensor, n)
        refs = pup_media.check_refs(
            pup_media.refs_in_chunk(media_refs, 0) if media_refs and hasattr(media_refs[0], "chunk_index") else [],
            actual_start,
            actual_end,
            ids[0].tolist(),
            self.image_token,
        )
        vt = None
        t_media = 0.0
        if refs:
            if self.media_reader is None:
                raise pup_media.MediaSpanError("media refs but no embedding store reader is configured")
            tm = time.perf_counter()
            values = pup_media.fetch_values(self.media_reader, refs)
            vt = self.model.media_values_to_device(values)
            if self.profile:
                ttnn.synchronize_device(self.mesh_device)
            t_media = time.perf_counter() - tm
        else:
            self.model._req_image_grid_thw = None
            self.model._req_video_grid_thw = None
        tp = time.perf_counter()
        # host logits of the chunk's last token: unused by the engine (decode computes the last prompt token); kept
        # for the harness's first-token check
        self.last_logits = self.model.prefill_traced_chunked(
            ids, self.page_table[slot_id : slot_id + 1], actual_len=n, vision_tokens=vt, return_host_logits=True
        )
        if vt is not None:
            ttnn.deallocate(vt)
        if self.profile:
            ttnn.synchronize_device(self.mesh_device)
        t_prefill = time.perf_counter() - tp
        te = time.perf_counter()
        self._export(kv_cache, slot=slot_id, actual_start=actual_start, actual_end=actual_end)
        if self.profile:
            ttnn.synchronize_device(self.mesh_device)
        t_export = time.perf_counter() - te
        if self._sink is not None:
            self._sink.layers_completed(0, len(self.model.layers), slot_id, actual_start, actual_end)
        self.last_timing = dict(
            media_ms=t_media * 1e3,
            prefill_ms=t_prefill * 1e3,
            export_ms=t_export * 1e3,
            total_ms=(time.perf_counter() - t0) * 1e3,
            images=len(refs),
            tokens=n,
        )
        logger.info("[qwen35_9b_pup] PUPPRE " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in self.last_timing.items()))
        return None  # one rank: the published caches are the output

    # --- KV chunk table ----------------------------------------------------------------------------------------
    def kv_migration_stages(self, kv_cache, first_layer_idx=None, num_my_layers=None):
        return kv_table.migration_stages(kv_cache)

    def build_kv_chunk_table(self, kv_cache, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        if stage_layouts is None:
            stage_layouts = kv_table.local_layouts(self.mesh_device, kv_cache, self.mesh_device.dram_grid_size().x)
        table = kv_table.build_table(
            stage_layouts=stage_layouts,
            num_users=kv_cache.num_users,
            max_seq_len=kv_cache.max_seq_len,
            head_dim=kv_cache.head_dim,
            num_kv_heads=int(self.mesh_device.shape[1]),
            attn_layer_ids=kv_cache.attn_layer_ids,
            gdn_layer_ids=kv_cache.gdn_layer_ids,
            num_model_layers=len(self.model.layers),
            geometry=kv_cache.gdn.geometry,
        )
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
        logger.info(f"[qwen35_9b_pup] KV chunk table -> {path} (configs 10, entries {table.total_entries()})")
        return path

    # --- internals ---------------------------------------------------------------------------------------------
    def _ids_for(self, input_tensor, n):
        if self._host_ids is not None and self._host_ids[0] is input_tensor:
            return self._host_ids[1][:, :n]
        # request mode: the tokens arrived over the H2D socket; one small read (v1; M3 embeds them on device)
        host = ttnn.to_torch(ttnn.get_device_tensors(input_tensor)[0]).reshape(1, -1)
        return host[:, :n].to(torch.int32)

    def _slot_scalar(self, slot):
        t = self._slot_scalars.get(slot)
        if t is None:
            t = ttnn.from_torch(
                torch.tensor([slot], dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            self._slot_scalars[slot] = t
        return t

    def _export(self, kv, *, slot, actual_start, actual_end):
        """Publish the chunk's state into the engine-owned caches (device ops only)."""
        a0, a1 = int(actual_start), int(actual_end)
        # update_padded_kv_cache writes a window whose length divides the cache length: the next power of two >= the
        # chunk (rows past actual_end are the slot's stale rows, at positions the migration never reads)
        n = max(32, 1 << (a1 - a0 - 1).bit_length())
        assert a0 % n == 0 and a0 + n <= self.config.max_seq_len, (a0, a1, n)
        b0 = slot * self.blocks_per_slot + a0 // self.block
        b1 = b0 + -(-n // self.block)
        rows = (b1 - b0) * self.block
        for pos, li in enumerate(self.attn_layer_ids):
            att = self.model.layers[li].attention
            for src, dst in ((att.paged_k, kv.k), (att.paged_v, kv.v)):
                hd = int(src.shape[-1])
                blk = ttnn.slice(src, [b0, 0, 0, 0], [b1, 1, self.block, hd])
                x = ttnn.reshape(blk, [1, 1, rows, hd])
                if rows != n:  # n = 32: half a block
                    y = ttnn.slice(x, [0, 0, 0, 0], [1, 1, n, hd])
                    x = y
                if dst is kv.k:  # the decode ring's interleaved-pair RoPE order, permuted in bf16
                    z = kv_table.rotate_half_to_interleaved(x)
                    if x is not blk:
                        ttnn.deallocate(x)
                    x = z
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    dst,
                    x,
                    slot_idx=slot,
                    layer_idx=pos,
                    num_layers=kv.num_attn_layers,
                    kv_actual_global=a0,
                    cluster_axis=0,
                )
                ttnn.deallocate(x)
                if blk is not x:
                    ttnn.deallocate(blk)
        st = self._slot_scalar(slot)
        for li in self.gdn_layer_ids:
            dn = self.model.layers[li].attention
            conv = ttnn.concat([dn.conv_states[m] for m in range(1, dn.K)], dim=1)  # [1, K-1, C]
            kv.gdn.export_layer(dn.rec_state, conv, st, li)
            ttnn.deallocate(conv)
