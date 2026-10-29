# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill runtime for the generic prefill runner (``models/demos/common/prefill``): one rank (one
galaxy, sp x tp), by default the encoder 0..19 + layer 20 KV-only (``tt/v41/prefill.py``; V41_PREFILL_DECODER=1: all 40
layers, DS41F-0073), the decode contract's export caches and KV chunk table (``tt/v41/kv_export.py``). tt-blaze
DS41F-0037 M2.

Per chunk: token ids to the host (the embedding and the Engram hash / gather are host work), ``V41Prefill`` runs the
chunk (eagerly, or on per-block trace islands with V41_PREFILL_ISLANDS=1, captured in ``on_services_ready``), its new
rows go into the export caches on device, then one ack per CONTRACT layer (``cfg.n_layers`` = 40, which both sides'
PREFILL_NUM_LAYERS must match; not the 21 layers computed by default) -- the export is complete at every chunk
boundary. The first token is NOT produced here: the ring replays the prompt's tail itself (R >= 2541; under 128 tokens
with V41_PREFILL_DECODER=1), which yields it (``disagg_prefill_plan.md``). A request's first chunk (actual_start 0)
resets that slot's states and zeroes its export rows. ``num_users`` slots: one attention-state set per slot
(``V41Prefill.use_slot``), the weights shared.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn

from .kv_export import allocate_v41_kv_export, build_v41_kv_chunk_table, export_chunk
from .prefill import V41Prefill

# DS41F-0073: V41_PREFILL_DECODER=1 builds all 40 layers (layer 20 a full ratio-1 source, the Reindex 24 / 28 / 32 / 36, the
# decoder Reuse layers) and exports every decoder layer's window ring, so the ring replays only the prompt's last < 128
# tokens instead of R >= 2541 (V4-Flash's prefill also ran every layer). Default 0: the encoder + layer 20 KV-only.
DECODER = os.environ.get("V41_PREFILL_DECODER", "0") == "1"


class V41PrefillRuntime:
    def __init__(
        self,
        mesh_device,
        cfg,
        ck,
        *,
        chunk_size: int,
        max_seq_len: int,
        num_users: int = 1,
        sp_axis: int = 0,
        tp_axis: int = 1,
        weight_cache_path=None,
        model_dir: str | None = None,
    ):
        self.mesh_device, self.cfg = mesh_device, cfg
        # what the generic runner reads (prefill_runner.py: is_first_rank / is_last_rank / use_trace): one untraced rank
        self.config = SimpleNamespace(
            is_first_rank=True,
            is_last_rank=True,
            use_trace=False,
            num_users=num_users,
            chunk_size=int(chunk_size),
            max_seq_len=int(max_seq_len),
            mesh_shape=tuple(mesh_device.shape),
            # main's runner sizes the D2D planes from these (prefill_runner._serve_request): this one rank holds the
            # whole contract, layers [0, n_layers)
            first_layer_idx=0,
            num_layers=int(cfg.n_layers),
        )
        self.chunk_size, self.max_seq_len, self.num_users = int(chunk_size), int(max_seq_len), int(num_users)
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.mesh_shape = tuple(mesh_device.shape)
        self.pf = V41Prefill(
            mesh_device,
            cfg,
            ck,
            max_seq_len=max_seq_len,
            chunk_tokens=chunk_size,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            weight_cache_path=weight_cache_path,
            n_layers=cfg.n_layers if DECODER else None,
            kv_only=not DECODER,
            num_users=num_users,
        )
        from blaze.models.deepseek_v4_1_flash_4_GLX.engram_host import EngramHost  # tt-blaze's ttnn-free host half

        kw = {} if model_dir is None else {"model_dir": model_dir}
        self.engram = EngramHost(
            max_seq_len=max_seq_len + 64,
            n_slots=num_users,
            table_dir=os.environ.get("DSV41_ENGRAM_TABLE_DIR") or None,
            **kw,
        )
        # ViT in the runner (vit_plan V2 E1 step 1): V41_PREFILL_VIT=1 builds the ttnn ViT + aligner on a (1, 1) submesh of this
        # mesh NOW, before any trace-island capture (a buffer allocated after a capture can sit in the islands' intermediates,
        # DS41F-0037 10-10 21:25), so a request's sidecar may name its IMAGE instead of precomputed rows (_request_image_rows).
        # a previous run's per-request sidecars must not leak into this one (the warm-up chunk at 0 would load them; seen
        # 2026-10-12 21:08): the producer writes this run's before each request, so clear the dir now
        _sd = os.environ.get("PREFILL_REQUEST_SIDECAR_DIR")
        if _sd and os.path.isdir(_sd):
            for _f in os.listdir(_sd):
                if _f.startswith("slot_") and _f.endswith(".json"):
                    os.remove(os.path.join(_sd, _f))
        self.vit = None
        if os.environ.get("V41_PREFILL_VIT", "0") == "1":
            self._init_vit(ck, model_dir)
        # ViT V2 (tt-blaze vit_plan.md): V41_PREFILL_IMAGE_ROWS=<prof/vit_prompt.py rows .pt> = the image-span rows of the
        # prompt this runner serves (its ids carry the span on virtual ids vocab + k). One image prompt per runner launch,
        # like the ring's build-time table (V3); per-request rows are the next step.
        _img = os.environ.get("V41_PREFILL_IMAGE_ROWS")
        if _img:
            _rows = torch.load(_img, map_location="cpu")
            self.pf.set_image_rows(_rows["rows"] if isinstance(_rows, dict) else _rows)
            logger.info(f"[v41 runtime] image-span rows from {_img}: {tuple(self.pf._image_rows.shape)}")
        # the CONTRACT's layer count (40), not the 21 layers this rank computes: the driver migrates layers [0, num_layers) and
        # drains num_layers acks per chunk (PREFILL_NUM_LAYERS on both sides), and the export holds rows for EVERY layer (the
        # decoder 21..39's entries / keys alias layer 20's) -- with 21, layers 21..39 were never migrated (DS41F-0037 attempt 4)
        self.num_layers = int(cfg.n_layers)
        self._ack = None
        self._request_id = 0

    # ---- KV export ----------------------------------------------------------------------------------------------------
    def allocate_kv_cache(self):
        return allocate_v41_kv_export(
            self.mesh_device,
            self.cfg,
            max_seq_len=self.max_seq_len,
            num_users=self.num_users,
            mesh_shape=self.mesh_shape,
            sp_axis=self.sp_axis,
            decoder=DECODER,
        )

    def kv_migration_base_address(self, kv_caches) -> int:
        return int(kv_caches.swa.buffer_address())

    def kv_migration_stages(self, kv_caches, first_layer_idx: int, num_my_layers: int):
        """Main's migration contract (common/prefill/runners/migration.py): ONE stage -- the contract export's base address
        (the same address kv_migration_base_address gave the old single-stage protocol) spanning this rank's layers."""
        from models.demos.common.prefill.runners.migration import KvCacheStage

        return [KvCacheStage(self.kv_migration_base_address(kv_caches), int(first_layer_idx), int(num_my_layers))]

    def build_kv_chunk_table(
        self,
        kv_caches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers=None,
        stage_layouts=None,
        stage_layout=None,
    ) -> str:
        # single rank: the table comes from the local export tensors; the gathered stage layouts carry nothing more
        return build_v41_kv_chunk_table(kv_caches, self.cfg, self.mesh_device, path)

    # ---- the engine contract ----------------------------------------------------------------------------------------
    def compile(self, kv_caches) -> None:
        """One warm chunk of zeros (compiles every program of a full chunk), then a clean slate."""
        t0 = time.time()
        self.prefill_chunk(
            [0] * self.chunk_size, kv_caches, slot_id=0, actual_start=0, actual_end=self.chunk_size, warmup=True
        )
        self.pf.reset()
        kv_caches.zero_all()
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"[v41 runtime] compiled (one warm chunk of {self.chunk_size}) in {time.time() - t0:.1f} s")

    def make_chunk_input(self, token_ids: list):
        return list(token_ids)

    def _token_ids_from_device(self, input_tensor) -> list:
        """Request mode: the chunk's ids arrive over H2D as [sp, 1, S / sp] uint32 per mesh (SP-sharded, TP-replicated)."""
        full = ttnn.to_torch(
            input_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, mesh_shape=self.mesh_shape, dims=(0, 1)),
        )
        return full[:, 0, :].reshape(-1).to(torch.int64).tolist()

    def prefill_chunk(self, input_tensor, kv_caches, *, metadata_msg=None, **kwargs):
        """Main's runner hands each chunk's metadata tensor to the runtime; this single untraced rank never forwards it, so it
        frees it after the chunk (as the reference runtimes do; the pre-merge runner leaked it as record_dev)."""
        try:
            return self._prefill_chunk(input_tensor, kv_caches, **kwargs)
        finally:
            if metadata_msg is not None and self.config.is_last_rank:
                ttnn.deallocate(metadata_msg)

    def _prefill_chunk(
        self,
        input_tensor,
        kv_caches,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        record_dev=None,
        warmup: bool = False,
    ):
        assert 0 <= slot_id < self.num_users, (slot_id, self.num_users)
        self.pf.use_slot(int(slot_id))
        ids = input_tensor if isinstance(input_tensor, list) else self._token_ids_from_device(input_tensor)
        real = int(actual_end) - int(actual_start)
        assert 0 < real <= self.chunk_size and len(ids) >= real
        if not warmup and os.environ.get("V41_PREFILL_ISLANDS", "0") == "1" and not self.pf.islands:
            # capture at the FIRST REQUEST, not at the warm-up: the runner builds its H2D input service AFTER the warm-up,
            # and a buffer allocated after a capture can sit in the islands' intermediates -- the socket's next chunk then
            # arrived as garbage ids (index 930504704 of 129280, 19:49). This chunk's ids were read above. ~6 s, once.
            self.pf.enable_islands()
        if int(actual_start) == 0:
            self.pf.reset()
            kv_caches.zero_slot(int(slot_id))
            self._request_image_rows(int(slot_id))
        assert self.pf.kv_actual == int(actual_start), (self.pf.kv_actual, actual_start)
        self._request_id = int(request_id)
        t0 = time.time()
        # DS41F-0119 V41_PREFILL_DEVICE_EMBED: the prefill may embed a full chunk straight from the H2D ids tensor
        self.pf._ids_dev = None if isinstance(input_tensor, list) else input_tensor
        try:
            self.pf._chunk([int(v) for v in ids[:real]], int(actual_start), self.engram, None)
        finally:
            self.pf._ids_dev = None
        t1 = time.time()
        export_chunk(kv_caches, self.pf, slot_id, int(actual_start), int(actual_end))
        t2 = time.time()
        ttnn.synchronize_device(self.mesh_device)
        t3 = time.time()
        if self._ack is not None and not warmup:
            for L in range(self.num_layers):
                self._ack(L)
        t4 = time.time()
        # DS41F-0119: the first request at a NEW prompt length is 0.3-4.6 s slower (bench 10-29) -- log the program-cache growth
        # per chunk, so a length-keyed program build shows up as "+N programs" on the request path
        pc = self._program_cache_entries()
        if pc is not None and pc != getattr(self, "_pc_last", None):
            if getattr(self, "_pc_last", None) is not None and not warmup:
                logger.info(
                    f"[v41 runtime] program cache {pc} entries (+{pc - self._pc_last}) after chunk [{actual_start}, {actual_end})"
                )
            self._pc_last = pc
        islands = os.environ.get("V41_PREFILL_ISLANDS", "0") == "1"
        if warmup and (islands or os.environ.get("V41_PREFILL_WARM_ALL", "0") == "1"):
            # islands need every width's programs and cached constants built BEFORE the capture: warm-all first
            self._warm_all(ids[:real], kv_caches, int(slot_id), int(actual_end))

        logger.info(
            f"[v41 runtime] chunk [{actual_start}, {actual_end}) slot {slot_id} in {t4 - t0:.2f} s "
            f"(compute issue {t1 - t0:.2f} + export issue {t2 - t1:.2f} + sync {t3 - t2:.2f} + acks {t4 - t3:.2f})"
            + (" (warm-up)" if warmup else "")
        )
        return None

    def _program_cache_entries(self):
        try:
            return int(self.mesh_device.num_program_cache_entries())
        except Exception:  # not every build exposes it on a mesh
            return None

    def _init_vit(self, ck, model_dir) -> None:
        import json as _json
        import sys as _sys
        import types as _types

        from .config import DEFAULT_MODEL_DIR
        from .vision import V41Vision, load_vision_state

        mdir = model_dir or DEFAULT_MODEL_DIR
        _sys.path.insert(0, os.path.join(mdir, "inference"))
        import image_processor as _ip  # the checkpoint's own preprocessing / span layout

        self._ip = _ip
        self._vit_args = _types.SimpleNamespace(**_json.load(open(os.path.join(mdir, "inference", "config.json"))))
        t0 = time.time()
        state = load_vision_state(ck)
        self._vit_delims = {
            k: state[k].to(torch.bfloat16).reshape(-1) for k in ("image_start", "image_end", "image_newline")
        }
        # REPLICATED on the runner's own mesh (each chip computes the same rows; the host reads chip 0's): a (1, 1) submesh
        # of this FABRIC_2D mesh hung the runner's next prefill chunk (DS41F-0049, 2026-10-12 21:08)
        self.vit = V41Vision(self.mesh_device, state)
        logger.info(
            f"[v41 runtime] ViT + aligner replicated on the {tuple(self.mesh_device.shape)} mesh, built in {time.time() - t0:.1f} s"
        )

    def _vit_span_rows(self, image_path: str):
        """The image's span rows in span order (START, the aligner rows with a NEWLINE after each grid row, END) --
        the layout of prof/vit_prompt.py (pod NFS only, see scripts/v41_vit_rows.py), from the device ViT."""
        t0 = time.time()
        patches, n_h, n_w, n_llm_h, n_llm_w = self._ip.load_image({"url": image_path}, self._vit_args)
        rows = self.vit.aligner(self.vit.vit(patches, n_h, n_w), n_h, n_w)  # [n_llm_h * n_llm_w, dim] bf16
        d, out, r = self._vit_delims, [], 0
        for t in self._ip.image_token_types(n_llm_h, n_llm_w).tolist():
            if t == self._ip.IMAGE_START:
                out.append(d["image_start"])
            elif t == self._ip.IMAGE_END:
                out.append(d["image_end"])
            elif t == self._ip.IMAGE_NEW_LINE:
                out.append(d["image_newline"])
            else:
                out.append(rows[r])
                r += 1
        span = torch.stack(out)
        logger.info(
            f"[v41 runtime] ViT {image_path}: grid {n_h} x {n_w} -> {tuple(span.shape)} span rows in {time.time() - t0:.2f} s"
        )
        return span

    def _request_image_rows(self, slot_id: int) -> None:
        """ViT V2, per request: ``PREFILL_REQUEST_SIDECAR_DIR/slot_<s>.json`` (written by the producer before the slot's chunks)
        names the request's image-span rows (``metadata["image_rows"]``); absent = a text prompt. Without the sidecar dir the
        launch-wide ``V41_PREFILL_IMAGE_ROWS`` stays in force."""
        d = os.environ.get("PREFILL_REQUEST_SIDECAR_DIR")
        if not d:
            return
        import json

        f = os.path.join(d, f"slot_{slot_id}.json")
        side = json.load(open(f)) if os.path.exists(f) else {}
        meta = side.get("metadata", {})
        path = meta.get("image_rows")
        if meta.get("image") and self.vit is not None:
            # the request names its IMAGE: the ViT runs here; its rows also go next to the trace's full ids, where the decode
            # ring's replay looks for them (<full ids prefix>.rows.pt) when the span reaches past the prefill's end
            span = self._vit_span_rows(meta["image"])
            self.pf.set_image_rows(span)
            self._image_rows_path = None
            if side.get("trace"):
                out = os.path.join(side["trace"], "full_prompt.rows.pt")
                tmp = out + ".tmp"
                torch.save({"rows": span}, tmp)
                os.replace(tmp, out)
            return
        if path != getattr(self, "_image_rows_path", None):
            rows = torch.load(path, map_location="cpu") if path else None
            self.pf.set_image_rows(rows["rows"] if isinstance(rows, dict) else rows)
            self._image_rows_path = path
            logger.info(f"[v41 runtime] slot {slot_id}: image-span rows {path or 'none (text prompt)'}")

    def on_services_ready(self, kv_caches) -> None:
        """Called by the runner once every persistent service buffer (H2D / D2H / the KV table) exists, right before its
        request loop: capture the trace islands HERE, so the first request no longer pays the ~6 s capture (DS41F-0037
        10-11). The first-request capture in prefill_chunk stays as the fallback for callers without this hook (tests).
        """
        if os.environ.get("V41_PREFILL_ISLANDS", "0") == "1" and not self.pf.islands:
            self.pf.enable_islands()
            # The first replay after the capture is 1.3-3 s slower than the rest (MEASURED: a request's first 8192 chunk
            # 3.07-4.71 s vs 1.4-2.1 s, 64K runs 10-11): absorb it with one dummy chunk on slot 0. Safe: the first real
            # request starts at 0, which resets the slot's states and zeroes its export rows, and the Engram n-gram cache is
            # position-indexed (a chunk at 0 overwrites every position it hashes over).
            t0 = time.time()
            self.pf.use_slot(0)
            self.pf.reset()
            self.pf._chunk([0] * self.chunk_size, 0, self.engram, None)
            if any(b._attn_islands_enabled() for b in self.pf.blocks) and 2 * self.chunk_size <= self.max_seq_len:
                # DS41F-0068: chunk 0 runs its attention eagerly; a second dummy chunk absorbs the attention islands' (A3)
                # first replays too
                self.pf._chunk([0] * self.chunk_size, self.chunk_size, self.engram, None)
            ttnn.synchronize_device(self.mesh_device)
            self.pf.reset()
            logger.info(f"[v41 runtime] islands warm replay at setup: {time.time() - t0:.2f} s")

    def _warm_all(self, ids, kv_caches, slot: int, start: int) -> None:
        """V41_PREFILL_WARM_ALL=1 (DS41F-0037 C-P1): after the warm-up chunk, run the same ids as dummy chunks at every later
        chunk position up to the capacity, export included, so every live-width program (the CSA live extent steps every
        4096 entries) is built before the first request instead of inside it (5-13 s per new width, MEASURED 10-10). The
        slot's states and export rows are reset by the next request's first chunk."""
        t0 = time.time()
        n = 0
        for s0 in range(start, self.max_seq_len, self.chunk_size):
            real = min(self.chunk_size, self.max_seq_len - s0)
            self.pf._chunk([int(v) for v in ids[:real]], s0, self.engram, None)
            export_chunk(kv_caches, self.pf, slot, s0, s0 + real)
            n += 1
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"[v41 runtime] warm-all: {n} more chunk(s) to {self.max_seq_len} in {time.time() - t0:.1f} s")

    def release(self) -> None:
        """Free the trace islands BEFORE the mesh closes (a trace buffer destroyed during close segfaults, DS4F-0258)."""
        for blk in self.pf.blocks:
            blk.release_islands()
        self.pf.islands = False

    def set_layer_ack_channel(self, channel) -> None:
        self._ack = lambda layer_idx: channel.inject(1)

    def set_layer_completion_sink(self, sink) -> None:
        self._ack = lambda layer_idx: sink(layer_idx, self._request_id)

    def flush_acks(self) -> None:
        return None

    def tail_hidden_row(self, tail_out, row: int):
        raise NotImplementedError("V4.1's first token comes from the ring's replay of the prompt tail (DS41F-0037)")
