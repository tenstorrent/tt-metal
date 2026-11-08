# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 Flash runtime for the common prefill runner (tt-blaze G53F-0256 G4b).

One 8x4 mesh, one rank, one prefill slot: ``GLM53MeshModel`` (45 layers, no LM head -- the decode ring replays the
prompt's tail and samples the first token itself) prefills each engine chunk at its absolute start (``start == 0`` begins
a new prompt and resets the KDA state), then ``GlmRingExport`` writes the chunk's KV in the ring's byte format and every
layer (0..44 and the MTP row 45) is acknowledged with its GLOBAL index. The MTP layer's chunks stay zero for now: the
ring drafts from a cold MTP cache after a landing (lossless; acceptance recovers as it decodes).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import KvCacheStage
from models.demos.glm53_flash.tt.runners import prefix_reuse
from models.demos.glm53_flash.tt.runners.kv_export import (
    DSA_LAYERS,
    LATENT_DIM,
    MTP_LAYER,
    POOL_CHUNK_TOKENS,
    POOL_RATIO,
    POOL_WIDTH,
    TABLE_LAYERS,
    TPP,
    GlmRingExport,
    log_export,
    read_tile_rows,
)


def pad_bucket(n: int) -> int | None:
    """The padded length a chunk of ``n`` tokens runs at (``GLM53F_RUNNER_BUCKETS``, read per call). Default (unset):
    None, the model's own 256-token steps. ``coarse``: 256-token steps to 4096, 512 to 8192, 1024 above, i.e. 32
    lengths up to a 16384-token chunk instead of 64. Every padded length keeps DRAM resident, and a 16384-chunk runner
    warmed with all 64 OOMed at the 63rd (tt-blaze G53F-0256, 14:52). ``mesh_model.prefill(pad_to=)`` is the tail-
    bucket path of G53F-0221. ``STEP:UPTO,...`` (e.g. ``512:4096,1024:8192,2048:16384``): STEP-token steps up to
    each UPTO, the last step past the last UPTO -- a 1M cache leaves room for only ~24 lengths (G53F-0256, 19:53)."""
    spec = os.environ.get("GLM53F_RUNNER_BUCKETS", "")
    if not spec:
        return None
    if spec == "coarse":
        spec = "256:4096,512:8192,1024:16384"
    tiers = [tuple(int(x) for x in t.split(":")) for t in spec.split(",")]
    step = next((st for st, upto in tiers if n <= upto), tiers[-1][0])
    return -(-n // step) * step


class GLM53FlashPrefillRuntime:
    def __init__(self, *, mesh_device, hf_config, params):
        from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

        _DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
        self.mesh_device = mesh_device
        self.config = SimpleNamespace(
            chunk_size=params.chunk_size,
            max_seq_len=params.max_seq_len,
            first_layer_idx=0,
            num_layers=TABLE_LAYERS,
            is_first_rank=True,
            is_last_rank=True,
            use_trace=False,
        )
        assert params.num_users == 1, f"the GLM prefill serves one slot (PREFILL_NUM_USERS=1), got {params.num_users}"
        assert params.first_layer_idx == 0 and params.num_layers == TABLE_LAYERS, (
            params.first_layer_idx,
            params.num_layers,
        )
        unit = 32 * tuple(mesh_device.shape)[0]
        assert params.chunk_size % max(unit, POOL_CHUNK_TOKENS) == 0, f"chunk {params.chunk_size} % {unit}"
        t0 = time.time()
        self.model = GLM53MeshModel(
            mesh_device,
            hf_config,
            list(range(45)),
            max_batch=1,
            max_seq_len=params.max_seq_len,
            weight_cache_path=params.weight_cache_path,
            expert_dtype=_DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")],
            expert_down_dtype=_DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")],
            build_head=False,
        )
        self.decs = {d.layer_idx: d for d in self.model.layers}
        self._sink = None
        # tt-blaze G53F-0256 export pipeline (GLM53F_EXPORT_ASYNC=1): the device runs chunk i+1 while one worker thread
        # raw-reads chunk i's DSA pages (UMD, no command queue), packs and writes them, then acks the chunk's layers in
        # order (the ack ring is single-producer, so EVERY ack goes through this worker once it is on).
        # GLM53F_EXPORT_ASYNC: "1" = the worker raw-reads too (REFUTED 18:1x: UMD reads 34x slower than the command
        # queue, and they slowed the device's own prefill 3.5x); "pack" = the main thread reads through the command queue,
        # the worker packs + writes + acks
        self._async = os.environ.get("GLM53F_EXPORT_ASYNC", "") in ("1", "pack")
        self._async_pack = os.environ.get("GLM53F_EXPORT_ASYNC", "") == "pack"
        self._xq: queue.Queue = queue.Queue()
        # tt-blaze G53F-0311 prefill-side prefix reuse (GLM53F_PREFIX_REUSE=1; off = this runner exactly as before): built
        # in compile() (the checkpoint pool is allocated before the warm-up, so the warm-up runs beside it)
        self._prefix: prefix_reuse.PrefixRetention | None = None
        self._prefix_on = False
        self._ckpt: list[dict] = []
        self._kv = None
        if self._async:
            threading.Thread(target=self._export_worker, name="glm53f-export", daemon=True).start()
        logger.info(f"[glm53f-runtime] model built in {time.time() - t0:.0f} s (max_seq_len {params.max_seq_len})")

    # ------------------------------------------------------------------ the runner contract
    def set_layer_completion_sink(self, sink) -> None:
        self._sink = sink

    def make_chunk_input(self, token_ids):
        return torch.tensor([int(t) for t in token_ids], dtype=torch.int64)

    def _ids_host(self, inp) -> torch.Tensor:
        if isinstance(inp, torch.Tensor):
            return inp.flatten().to(torch.int64)
        devs = ttnn.get_device_tensors(inp)
        rows, cols = tuple(self.mesh_device.shape)
        # the H2D mapper shards the chunk over the SP rows (mesh axis 0) and replicates it over the TP columns
        return torch.cat([ttnn.to_torch(devs[r * cols]).flatten() for r in range(rows)]).to(torch.int64)

    def compile(self, kv_caches) -> None:
        """Warm the programs at the lengths the engine sends, export included (``GLM53F_RUNNER_WARMUP``, comma list):
        ``N:S`` one call of N tokens at start S (S > 0 must continue the previous item), ``cN`` a whole N-token prompt in
        engine-sized chunks (its last chunk is the ragged tail; long prompts also create the DSA pool-bucket widths of
        every start). Every 256-token padded length is its own program set (a first request at a new one pays ~6-50 s,
        tt-blaze G53F-0256 G4b), so the default covers the cuts S' = 4 * floor((S - 1) / 4) of ISL 1k / 2k / 4k / 16k.
        """
        chunk = self.config.chunk_size
        if prefix_reuse.enabled():
            self._build_prefix(kv_caches)
        spec = os.environ.get("GLM53F_RUNNER_WARMUP", f"1020:0,2044:0,4092:0,c{2 * chunk - 4}")
        for item in [v.strip() for v in spec.split(",") if v.strip()]:
            if item.startswith("a"):  # tt-blaze G53F-0256: a whole prompt through prefill_chunk (the async export path)
                self._warm_async(kv_caches, int(item[1:]), chunk)
                continue
            if item.startswith("c"):
                n = int(item[1:])
                parts = [(s, min(n, s + chunk)) for s in range(0, n, chunk)]
            else:
                n_, _, s_ = item.partition(":")
                parts = [(int(s_ or 0), int(s_ or 0) + int(n_))]
            if parts[-1][1] > self.config.max_seq_len:
                logger.info(f"[glm53f-runtime] warm-up {item} skipped (past max_seq_len {self.config.max_seq_len})")
                continue
            tw = time.perf_counter()
            tp = te = 0.0
            for s, e in parts:
                n = e - s
                ids = torch.tensor(([785, 6722, 315, 9621] * (n // 4 + 1))[:n], dtype=torch.int64)
                t0 = time.perf_counter()
                self._prefill(ids, s)
                ttnn.synchronize_device(self.mesh_device)
                t1 = time.perf_counter()
                self._export(kv_caches, s, e)
                tp, te = tp + (t1 - t0), te + (time.perf_counter() - t1)
            logger.info(
                f"[glm53f-runtime] warm-up {item} ({len(parts)} chunk(s), {parts[-1][1]} tokens): prefill {tp:.2f} s, "
                f"export {te:.2f} s, wall {time.perf_counter() - tw:.2f} s{self._dram_note()}"
            )
        if self._prefix is not None:
            # the warm-up wrote the caches with its own ids outside the retention's view: start the serving empty
            self._prefix.reset("warm-up done")
            self._prefix_on = True
        self._arm_dram_guard()

    def _dram_note(self) -> str:
        """DRAM held after a warm-up item (tt-blaze G53F-0256: each padded chunk length keeps DRAM resident and the
        warm-up OOMs past ~24 lengths at a 1M cache), with the program-cache size, to say what grows per length."""
        try:
            v = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
            n = getattr(self.mesh_device, "num_program_cache_entries", lambda: -1)()
            return (
                f", DRAM {v.total_bytes_allocated_per_bank / 2**20:.1f} MiB/bank held, largest free "
                f"{v.largest_contiguous_bytes_free_per_bank / 2**20:.1f} MiB, programs {n}"
            )
        except Exception as e:  # noqa: BLE001 -- a report must not fail the warm-up
            return f", DRAM view unavailable ({type(e).__name__})"

    _CONST_CACHE_ATTRS = ("_const_cache", "_slab_consts", "_cu_windows", "_pad_cache")
    _const_dicts: list | None = None
    _last_growth: tuple | None = None

    def _growth_note(self) -> str:
        """A served chunk's DRAM held, program-cache entries and cached device constants, appended to its log line only
        when one of them moved (tt-blaze G53F-0313: a 1M runner OOM'd in the routed MoE after ~4,700 mixed-length
        chunks, DRAM held 3786 -> 3991 MiB/bank since warm-up; this says which cache grows)."""
        try:
            if self._const_dicts is None:
                found, seen, todo = [], set(), [self.model]
                while todo:
                    o = todo.pop()
                    if id(o) in seen:
                        continue
                    seen.add(id(o))
                    if isinstance(o, dict):
                        todo.extend(v for v in o.values() if not isinstance(v, ttnn.Tensor))
                    elif isinstance(o, (list, tuple)):
                        todo.extend(v for v in o if not isinstance(v, ttnn.Tensor))
                    elif type(o).__module__.startswith("models.") and hasattr(o, "__dict__"):
                        for k, v in vars(o).items():
                            if k in self._CONST_CACHE_ATTRS and isinstance(v, dict):
                                found.append(v)
                            elif not isinstance(v, (ttnn.Tensor, torch.Tensor, str, int, float, bool)):
                                todo.append(v)
                self._const_dicts = found
            v = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
            g = (
                round(v.total_bytes_allocated_per_bank / 2**20, 1),
                round(v.largest_contiguous_bytes_free_per_bank / 2**20, 1),
                getattr(self.mesh_device, "num_program_cache_entries", lambda: -1)(),
                sum(len(d) for d in self._const_dicts),
            )
        except Exception as e:  # noqa: BLE001 -- a report must not fail a request
            return f", growth note unavailable ({type(e).__name__})"
        if g == self._last_growth:
            return ""
        self._last_growth = g
        return f", DRAM {g[0]} MiB/bank held, largest free {g[1]} MiB, programs {g[2]}, device consts {g[3]}"

    # tt-blaze G53F-0313: every program the cache keeps holds its kernel binaries in DRAM (tt_metal program.cpp
    # allocate_kernel_bin_buf_on_device), and the DSA layers compile new programs (and cache new device constants) per
    # (chunk start, padded length) geometry -- ~1,200 at a 1M cache, far more than the ~290 MiB/bank warm-up leaves
    # free. Served long enough on mixed lengths, a chunk's transient buffers stop fitting (a routed-MoE OOM after
    # ~4,700 chunks). The guard: before a chunk, when the largest free DRAM block is more than this many MiB/bank below
    # its post-warm-up size, drop the program cache and the cached device constants (both rebuild on demand: the JIT
    # disk cache keeps the kernels). 0 disables. Prefill traces are off in serving (GLM53_MODEL_TRACE_MB=0), and the
    # guard refuses to arm when a trace region is configured, since a captured trace replays binaries by address.
    _dram_floor_mib: float | None = None
    _guard_clears = 0

    def _arm_dram_guard(self) -> None:
        budget = float(os.environ.get("GLM53F_RUNNER_DRAM_GUARD_MIB", "64"))
        if budget <= 0 or int(os.environ.get("GLM53_MODEL_TRACE_MB", "0") or 0) > 0:
            logger.info(
                f"[glm53f-runtime] DRAM guard off (budget {budget:g} MiB, trace MB {os.environ.get('GLM53_MODEL_TRACE_MB')})"
            )
            return
        try:
            v = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[glm53f-runtime] DRAM guard off: no DRAM view ({type(e).__name__})")
            return
        free = v.largest_contiguous_bytes_free_per_bank / 2**20
        self._dram_floor_mib = free - budget
        logger.info(
            f"[glm53f-runtime] DRAM guard armed: largest free {free:.1f} MiB/bank after warm-up, clear the program cache "
            f"+ device consts below {self._dram_floor_mib:.1f} MiB (GLM53F_RUNNER_DRAM_GUARD_MIB={budget:g})"
        )
        if os.environ.get("GLM53F_RUNNER_DRAM_GUARD_SELFTEST", "0") == "1":
            # one forced clear now: the first requests then run on rebuilt programs + constants (a correctness gate
            # after this proves a guard clear is safe) and the log says what a clear frees and costs
            floor, self._dram_floor_mib = self._dram_floor_mib, float("inf")
            self._dram_guard(-1)
            self._dram_floor_mib = floor

    def _dram_guard(self, start: int) -> None:
        if self._dram_floor_mib is None:
            return
        v = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
        free = v.largest_contiguous_bytes_free_per_bank / 2**20
        if free >= self._dram_floor_mib:
            return
        t0 = time.perf_counter()
        if self._async:
            self._xq.join()  # the export thread enqueues device work too: nothing in flight while the caches go
        ttnn.synchronize_device(self.mesh_device)  # no program still reading the binaries about to be freed
        self._growth_note()  # builds the const-dict list on first use
        n_prog = self.mesh_device.num_program_cache_entries()
        n_const = sum(len(d) for d in self._const_dicts or ())
        for d in self._const_dicts or ():
            d.clear()  # references only: a tensor another module still holds stays allocated
        self.mesh_device.clear_program_cache()
        self._last_growth = None
        self._guard_clears += 1
        w = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
        logger.warning(
            f"[glm53f-runtime] DRAM guard #{self._guard_clears} "
            f"{'self-test (forced)' if start < 0 else f'before the chunk at {start}'}: largest free {free:.1f} MiB/bank "
            f"(floor {self._dram_floor_mib:.1f}): dropped {n_prog} programs + {n_const} device consts in "
            f"{time.perf_counter() - t0:.2f} s -> largest free {w.largest_contiguous_bytes_free_per_bank / 2**20:.1f} MiB, "
            f"held {w.total_bytes_allocated_per_bank / 2**20:.1f} MiB/bank"
        )

    def _warm_async(self, kv_caches, n: int, chunk: int) -> None:
        """Warm-up item ``aN``: an N-token prompt through ``prefill_chunk`` exactly as the engine drives it (export pipeline
        included when GLM53F_EXPORT_ASYNC=1), with a counting sink; the wall clock runs to the last chunk's acks."""
        if n > self.config.max_seq_len:
            logger.info(f"[glm53f-runtime] warm-up a{n} skipped (past max_seq_len {self.config.max_seq_len})")
            return
        acks, saved = [0], self._sink
        self._sink = lambda layer, request_id: acks.__setitem__(0, acks[0] + 1)
        try:
            tw = time.perf_counter()
            for k, s in enumerate(range(0, n, chunk)):
                e = min(n, s + chunk)
                ids = torch.tensor(([785, 6722, 315, 9621] * ((e - s) // 4 + 1))[: e - s], dtype=torch.int64)
                self.prefill_chunk(ids, kv_caches, slot_id=0, actual_start=s, actual_end=e, request_id=k)
            self._xq.join()
            nchunks = -(-n // chunk)
            logger.info(
                f"[glm53f-runtime] warm-up a{n} ({nchunks} chunk(s), async export {'ON' if self._async else 'OFF'}): "
                f"wall to the last ack {time.perf_counter() - tw:.2f} s, acks {acks[0]} (expect {nchunks * TABLE_LAYERS})"
            )
        finally:
            self._sink = saved

    def prefill_chunk(self, input_tensor, kv_caches, *, slot_id, actual_start, actual_end, request_id=0, **_):
        # tt-d-gen device.prefill.mark_last_chunk: bit 29 of the slot word on every chunk, bit 30 on a burst's last one
        slot_id, last = prefix_reuse.split_slot_word(slot_id)
        pr = self._prefix if self._prefix_on else None
        if not isinstance(input_tensor, torch.Tensor) and actual_end <= actual_start:
            ttnn.deallocate(input_tensor)
        if actual_end <= actual_start:
            if pr is not None:
                pr.note_empty()
            # tt-d-gen prefill_wait_for_arm_us: a request the decode side declined before its first chunk sends one
            # EMPTY chunk only to retire its burst -- nothing to compute or export, every layer acknowledged
            logger.info(f"[glm53f-runtime] chunk {request_id} slot {slot_id} empty (declined before compute)")
            if self._async:
                self._xq.put(("ack", None, 0, 0, request_id, None))
            elif self._sink is not None:
                for layer in range(TABLE_LAYERS):
                    self._sink(layer, request_id)
            return None
        ids = self._ids_host(input_tensor)[: actual_end - actual_start]
        if not isinstance(input_tensor, torch.Tensor):
            ttnn.deallocate(input_tensor)
        self._dram_guard(actual_start)
        t0 = time.perf_counter()
        if pr is not None:
            if actual_start == 0:
                rss = f"; runner RSS {self._rss_gib():.1f} GiB" if pr.store is not None else ""
                logger.info(f"[glm53f-prefix] new prompt at chunk {request_id}: {pr.summary()}{rss}")
            if pr.begin_chunk(ids, actual_start, actual_end, last):
                self._skip_chunk(slot_id, actual_start, actual_end, request_id, t0)
                self._after_last(pr, last)
                return None
            if pr.last_resume is not None:  # once per resumed prompt: what the resume cost the device
                ttnn.synchronize_device(self.mesh_device)
                logger.info(
                    f"[glm53f-prefix] chunk {request_id}: resumed at {actual_start} ({pr.last_resume['how']}) in "
                    f"{time.perf_counter() - t0:.3f} s after skipping {actual_start // self.config.chunk_size} chunk(s)"
                )
            t0 = time.perf_counter()
            try:
                self._prefill(ids, actual_start)
            except Exception:
                pr.invalidate("a chunk failed")
                raise
            pr.end_compute(ids, actual_start, actual_end)
        else:
            self._prefill(ids, actual_start)
        t1 = time.perf_counter()
        if self._async:
            # KDA states: device copies now (the next chunk mutates them in place), then one sync; the DSA rows of
            # [actual_start, actual_end) are final, and the next chunk writes only rows past them
            rows = self._dsa_rows_cq(actual_start, actual_end) if self._async_pack else None
            if pr is not None and rows is not None and pr.wants_rows(actual_start, actual_end):
                pr.store_rows(ids, actual_start, actual_end, *self._store_payload(rows))
            for layer in kv_caches.kda_layers:
                kv_caches.export_kda(layer, self.decs[layer].kda_batched)
            ttnn.synchronize_device(self.mesh_device)
            self._xq.put(("chunk", kv_caches, actual_start, actual_end, request_id, rows))
            logger.info(
                f"[glm53f-runtime] chunk {request_id} slot {slot_id} [{actual_start}, {actual_end}): prefill {t1 - t0:.3f} s, "
                f"kda export + sync {time.perf_counter() - t1:.3f} s (DSA export async){self._growth_note()}"
            )
            self._after_last(pr, last)
            return None
        capture = {} if pr is not None and pr.wants_rows(actual_start, actual_end) else None
        self._export(kv_caches, actual_start, actual_end, capture)
        if capture:
            pr.store_rows(ids, actual_start, actual_end, *self._store_payload(capture))
        logger.info(
            f"[glm53f-runtime] chunk {request_id} slot {slot_id} [{actual_start}, {actual_end}): prefill {t1 - t0:.3f} s, "
            f"export {time.perf_counter() - t1:.3f} s{self._growth_note()}"
        )
        if self._sink is not None:
            for layer in range(TABLE_LAYERS):
                self._sink(layer, request_id)
        self._after_last(pr, last)
        return None

    # ------------------------------------------------------------------ prefix reuse (tt-blaze G53F-0311)
    def _skip_chunk(self, slot_id, start, end, request_id, t0) -> None:
        """A chunk the device already holds (``PrefixRetention.begin_chunk``): acknowledge every layer, nothing computed."""
        if self._async:
            self._xq.put(("ack", None, 0, 0, request_id, None))
        elif self._sink is not None:
            for layer in range(TABLE_LAYERS):
                self._sink(layer, request_id)
        logger.info(
            f"[glm53f-runtime] chunk {request_id} slot {slot_id} [{start}, {end}): SKIPPED (retained prefix) in "
            f"{time.perf_counter() - t0:.3f} s"
        )
        return None

    def _build_prefix(self, kv_caches) -> None:
        """The retention + its device-DRAM checkpoint pool: ``GLM53F_PREFIX_CKPTS`` slots (default 8), each a copy of
        every KDA layer's carries (recurrent fp32 [1, H, 128, 128] + conv history, the model's own layouts)."""
        n = int(os.environ.get("GLM53F_PREFIX_CKPTS", "8"))
        self._kv = kv_caches
        kda = [L for L in sorted(self.decs) if self.decs[L].kda_batched is not None]
        before = self._dram_bytes()
        self._ckpt = []
        for _ in range(n):
            self._ckpt.append(
                {
                    L: SimpleNamespace(
                        recurrent=ttnn.clone(self.decs[L].kda_batched.recurrent),
                        convolution=ttnn.clone(self.decs[L].kda_batched.convolution),
                    )
                    for L in kda
                }
            )
        after = self._dram_bytes()
        per = (after - before) / max(n, 1) / 2**20 if before is not None and after is not None else float("nan")
        host_gb = float(os.environ.get("GLM53F_PREFIX_HOST_GB", "0"))
        store = None
        if host_gb > 0:
            # phase 2 (several retained prompts): the restore writes the caches through the model's own fills and
            # packs the slab pages with the export's functions; both assume the bf16 caches of the serving config
            for L in DSA_LAYERS:
                c = self.decs[L].cache
                assert c.dtype == ttnn.bfloat16 and c.index.pool_dtype == ttnn.bfloat16, (
                    L,
                    c.dtype,
                    c.index.pool_dtype,
                )
                assert not c.partition, "phase 2 needs GLM53_PARTITION_USERS=0 (one user, identity page table)"
            assert not (self._async and not self._async_pack), "phase 2 needs the sync or 'pack' export (rows read)"
            store = prefix_reuse.ChunkStore(int(host_gb * 2**30), log=logger.info)
        self._prefix = prefix_reuse.PrefixRetention(
            chunk=self.config.chunk_size,
            n_ckpts=n,
            save=self._ckpt_save,
            restore=self._ckpt_restore,
            export_carries=self._export_carries,
            set_next_start=self._set_next_start,
            compute_silent=lambda ids, start: self._prefill(ids, start),
            log=logger.info,
            store=store,
            restore_rows=self._restore_rows,
            spill=self._spill_carries,
            load_carries=self._load_carries,
            spill_tips=int(os.environ.get("GLM53F_PREFIX_SPILL_TIPS", "2")),
        )
        if store is not None:
            logger.info(
                f"[glm53f-prefix] phase 2 ON: host chunk store {host_gb:g} GiB (~"
                f"{int(host_gb * 2**30 // (len(DSA_LAYERS) * self.config.chunk_size * (LATENT_DIM + POOL_WIDTH // POOL_RATIO) * 2))}"
                f" chunks of {self.config.chunk_size} tokens), spill {self._prefix.spill_tips} tip(s) per prompt left"
            )
        logger.info(
            f"[glm53f-prefix] prefix reuse ON: {n} checkpoint slot(s) x {len(kda)} KDA layers, {per:.2f} MiB per bank "
            f"each{self._dram_note()}"
        )

    def _dram_bytes(self):
        try:
            return ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        except Exception:  # noqa: BLE001 -- a report must not fail the build
            return None

    def _ckpt_save(self, slot: int) -> None:
        for L, st in self._ckpt[slot].items():
            ttnn.copy(self.decs[L].kda_batched.recurrent, st.recurrent)
            ttnn.copy(self.decs[L].kda_batched.convolution, st.convolution)

    def _ckpt_restore(self, slot: int) -> None:
        for L, st in self._ckpt[slot].items():
            ttnn.copy(st.recurrent, self.decs[L].kda_batched.recurrent)
            ttnn.copy(st.convolution, self.decs[L].kda_batched.convolution)

    def _export_carries(self, slot) -> None:
        """The KDA carries of checkpoint ``slot`` (None = the live ones) into the export slabs: what the KV Manager ships
        after a burst whose last chunk was skipped."""
        for L in self._kv.kda_layers:
            self._kv.export_kda(L, self.decs[L].kda_batched if slot is None else self._ckpt[slot][L])
        ttnn.synchronize_device(self.mesh_device)

    def _set_next_start(self, pos: int) -> None:
        for dec in self.model.layers:
            dec.next_start[0] = int(pos)

    # ---- phase 2: the host chunk store (several retained prompts)
    def _after_last(self, pr, last) -> None:
        """After a burst's last chunk is acknowledged: read the pinned checkpoints of the prompt the device left."""
        if pr is not None and last and pr.pending_spills:
            t0 = time.perf_counter()
            n = len(pr.pending_spills)
            pr.flush_spills()
            logger.info(
                f"[glm53f-prefix] spilled {n} checkpoint(s) to the host store in {time.perf_counter() - t0:.3f} s"
            )

    @staticmethod
    def _rss_gib() -> float:
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        except OSError:
            return float("nan")

    @staticmethod
    def _store_payload(rows: dict) -> tuple:
        """(rows, bytes) of one chunk for the store: {DSA layer: (latent [n, 512] bf16, pools [n / 4, 128] bf16)}."""
        size = lambda t: 0 if t is None else int(t.numel()) * t.element_size()  # noqa: E731
        nbytes = sum(size(a) + size(b) for a, b in rows.values())
        return {L: (a.contiguous(), None if b is None else b.contiguous()) for L, (a, b) in rows.items()}, nbytes

    def _restore_rows(self, rows: dict, start: int) -> None:
        """A stored chunk back into the device: every DSA layer's latent rows and pooled keys into every chip's caches
        (the model's own ``fill`` / ``write_pools``, replicated uploads), then its export-slab pages packed again from
        the same rows (``export_dsa``, the export's own functions)."""
        t0 = time.perf_counter()
        for L, (latent, pools) in rows.items():
            c = self.decs[L].cache
            n = int(latent.shape[0])
            dev = self._upload_replicated(latent.reshape(1, 1, n, LATENT_DIM))
            c.fill(dev, 0, start=start)
            ttnn.deallocate(dev)
            if pools is not None:
                devp = self._upload_replicated(pools.reshape(1, 1, int(pools.shape[0]), POOL_WIDTH))
                c.index.write_pools(devp, 0, start_pool=start // POOL_RATIO)
                ttnn.deallocate(devp)
            c.release_prefix()  # GLM53_PREFIX_INCREMENTAL's arena (off in serving) must not keep older rows
        t1 = time.perf_counter()
        for L, (latent, pools) in rows.items():
            self._kv.export_dsa(L, latent, start // TPP, pools, start // POOL_CHUNK_TOKENS)
        logger.info(
            f"[glm53f-prefix] restored [{start}, {start + int(next(iter(rows.values()))[0].shape[0])}) from the host "
            f"store: caches {t1 - t0:.3f} s (dispatch), slab pages {time.perf_counter() - t1:.3f} s"
        )

    def _upload_replicated(self, t: torch.Tensor):
        """bf16 TILE on every chip (the model's own replicated-upload form, ``mesh.replicate()``)."""
        return ttnn.from_torch(
            t.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.model.mesh.replicate(),
        )

    def _spill_carries(self, slot: int) -> tuple:
        """Checkpoint ``slot`` -> host: per KDA layer the recurrent + conv carries of each TP column (row 0 chips; the
        SP rows hold replicas)."""
        cols = tuple(self.mesh_device.shape)[1]
        out, nbytes = {}, 0
        for L, st in self._ckpt[slot].items():
            pair = []
            for t in (st.recurrent, st.convolution):
                devs = ttnn.get_device_tensors(t)
                pair.append([ttnn.to_torch(devs[c]) for c in range(cols)])
                nbytes += sum(int(x.numel()) * x.element_size() for x in pair[-1])
            out[L] = tuple(pair)
        return out, nbytes

    def _load_carries(self, carries: dict) -> None:
        """Host carries -> the live KDA carries (each TP column's tensors, replicated over the SP rows)."""
        mesh = self.model.mesh
        for L, (rec_cols, conv_cols) in carries.items():
            dst = self.decs[L].kda_batched
            for parts, t in ((rec_cols, dst.recurrent), (conv_cols, dst.convolution)):
                shard = len(parts[0].shape) - 1 if len(parts[0].shape) == 3 else 1  # conv: the width; rec: the heads
                tmp = mesh.upload(torch.cat(parts, dim=shard), t.dtype, cols=shard, layout=t.layout)
                ttnn.copy(tmp, t)
                ttnn.deallocate(tmp)

    # ------------------------------------------------------------------ KV table
    def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
        return [
            KvCacheStage(base_addr=int(t.buffer_address()), first_layer=0, count=TABLE_LAYERS)
            for t in (kv_caches.latent, kv_caches.pool, kv_caches.kda_rec, kv_caches.kda_conv)
        ]

    def build_kv_chunk_table(self, kv_caches, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        return kv_caches.build_table(path, stage_layouts)

    # ------------------------------------------------------------------ internals
    def _prefill(self, ids: torch.Tensor, start: int) -> None:
        streams, logits, _ = self.model.prefill(ids, user=0, start=start, pad_to=pad_bucket(int(ids.numel())))
        ttnn.deallocate(streams)
        if logits is not None:
            ttnn.deallocate(logits)

    _read_dev: int | None = None  # GLM53F_EXPORT_READ_DEV: the replica the export reads (None = the first)

    def _rows(self, t, b0: int, b1: int) -> torch.Tensor:
        """Blocks [b0, b1) of a mesh-replicated paged cache ``[blocks, 1, 32, W]`` (first replica, or the replica
        ``GLM53F_EXPORT_READ_DEV`` names: ``auto`` times one read from every chip once and keeps the fastest -- 4 of a
        galaxy's 32 chips have a x8 PCIe link, the rest x1, tt-blaze G53F-0256)."""
        rest = [int(t.shape[i]) for i in range(1, len(t.shape))]
        part = ttnn.slice(t, [b0, 0, 0, 0], [b1, *rest])
        mode = os.environ.get("GLM53F_EXPORT_READ_DEV", "")
        if mode and self._read_dev is None:
            devs = self.model.mesh.device_tensors(part)
            if mode == "auto":
                times = []
                for k, d in enumerate(devs):
                    t0 = time.perf_counter()
                    ttnn.to_torch(d)
                    times.append((time.perf_counter() - t0, k))
                self._read_dev = min(times)[1]
                logger.info(
                    "[glm53f-export] read replica: chip index "
                    f"{self._read_dev} ({min(times)[0] * 1e3:.1f} ms; median {sorted(times)[len(times) // 2][0] * 1e3:.1f} ms, "
                    f"slowest {max(times)[0] * 1e3:.1f} ms over {len(times)} chips for {b1 - b0} blocks)"
                )
            else:
                self._read_dev = int(mode)
        if self._read_dev is not None:
            out = ttnn.to_torch(self.model.mesh.device_tensors(part)[self._read_dev])
        else:
            out = self.model.mesh.to_host(part)
        if part.buffer_address() != t.buffer_address():
            ttnn.deallocate(part)
        return out.reshape(-1, out.shape[-1])

    _prof_pk = _prof_wr = 0.0  # GLM53F_EXPORT_PROFILE: the export stats' pack / write totals at the last profile line
    _chk_cq = _chk_raw = 0.0  # GLM53F_EXPORT_CHECK: summed read times, command queue vs raw

    def _dsa_rows_raw(self, ex: GlmRingExport, layer: int, start: int, end: int):
        """Latent rows [start, end) and pool rows of DSA layer ``layer``, raw-read from the first replica (bf16 TILE
        caches, slot 0 = the first blocks: the one-slot runner's identity page table)."""
        c = self.decs[layer].cache
        dev = ex._dev_ids[0]
        latent = read_tile_rows(ex._banks, dev, c.cache.buffer_address(), ex._nb, LATENT_DIM, start, end)
        pools = None
        idx = getattr(c, "index", None)
        if idx is not None:
            q0, q1 = start // POOL_RATIO, -(-end // POOL_RATIO)
            pools = read_tile_rows(ex._banks, dev, idx.pool_cache.buffer_address(), ex._nb, POOL_WIDTH, q0, q1)
        return latent, pools

    def _dsa_rows_cq(self, start: int, end: int) -> dict:
        """{layer: (latent, pools)} of every DSA layer for rows [start, end), read through the command queue."""
        out = {}
        for layer in DSA_LAYERS:
            c = self.decs[layer].cache
            latent = self._rows(c.cache, start // TPP, -(-end // TPP))[: end - start].to(torch.bfloat16)
            pools = None
            idx = getattr(c, "index", None)
            if idx is not None:
                q0, q1 = start // POOL_RATIO, -(-end // POOL_RATIO)
                r0, r1 = q0 // 32, min(-(-q1 // 32), int(idx.pool_blocks))
                pools = self._rows(idx.pool_cache, r0, r1)[: q1 - q0]
            out[layer] = (latent, pools)
        return out

    def _export_worker(self) -> None:
        """GLM53F_EXPORT_ASYNC: chunk jobs in arrival order -> DSA export (raw read, pack, write) -> the chunk's acks."""
        while True:
            kind, ex, start, end, request_id, rows = self._xq.get()
            try:
                if kind == "chunk":
                    t0 = time.perf_counter()
                    for layer in DSA_LAYERS:
                        latent, pools = rows[layer] if rows is not None else self._dsa_rows_raw(ex, layer, start, end)
                        ex.export_dsa(layer, latent, start // TPP, pools, start // POOL_CHUNK_TOKENS)
                    log_export(ex.stats, start, end, t0)
                if self._sink is not None:
                    for layer in range(TABLE_LAYERS):
                        self._sink(layer, request_id)
            except Exception as e:  # noqa: BLE001 -- a lost ack strands the request: say so loudly
                logger.exception(f"[glm53f-export] async export of [{start}, {end}) failed: {e}")
            finally:
                self._xq.task_done()

    def _export(self, ex: GlmRingExport, start: int, end: int, capture: dict | None = None) -> None:
        t0 = time.perf_counter()
        prof = os.environ.get("GLM53F_EXPORT_PROFILE") == "1"  # tt-blaze G53F-0256: where a chunk's export time goes
        t_read = t_dsa = 0.0
        nbytes = 0
        for layer in DSA_LAYERS:
            c = self.decs[layer].cache
            b0, b1 = start // TPP, -(-end // TPP)
            tr = time.perf_counter()
            if os.environ.get("GLM53F_EXPORT_RAWREAD") == "1":
                latent, pools = self._dsa_rows_raw(ex, layer, start, end)
            else:
                latent = self._rows(c.cache, b0, b1)[: end - start].to(torch.bfloat16)
                pools = None
                idx = getattr(c, "index", None)
                if idx is not None:
                    q0, q1 = start // POOL_RATIO, -(-end // POOL_RATIO)
                    r0, r1 = q0 // 32, min(-(-q1 // 32), int(idx.pool_blocks))
                    pools = self._rows(idx.pool_cache, r0, r1)[: q1 - q0]
                if os.environ.get("GLM53F_EXPORT_CHECK") == "1":
                    t_cq = time.perf_counter() - tr
                    t_raw0 = time.perf_counter()
                    rl, rp = self._dsa_rows_raw(ex, layer, start, end)
                    t_raw = time.perf_counter() - t_raw0
                    self._chk_cq, self._chk_raw = self._chk_cq + t_cq, self._chk_raw + t_raw
                    ok = torch.equal(rl.view(torch.int16), latent.view(torch.int16)) and (
                        pools is None or torch.equal(rp.view(torch.int16), pools.to(torch.bfloat16).view(torch.int16))
                    )
                    logger.info(
                        f"[glm53f-export] raw-read check layer {layer} [{start}, {end}): {'EQUAL' if ok else 'DIFFERENT'} "
                        f"(read: command queue {t_cq * 1e3:.1f} ms, raw {t_raw * 1e3:.1f} ms)"
                    )
            tw = time.perf_counter()
            t_read += tw - tr
            if capture is not None:  # G53F-0311 phase 2: the rows this export read, kept for the host store
                capture[layer] = (latent.clone(), None if pools is None else pools.clone())
            nbytes += ex.export_dsa(layer, latent, start // TPP, pools, start // POOL_CHUNK_TOKENS)
            t_dsa += time.perf_counter() - tw
        tk = time.perf_counter()
        for layer in ex.kda_layers:
            ex.export_kda(layer, self.decs[layer].kda_batched)
        ttnn.synchronize_device(self.mesh_device)
        if prof:
            pk, wr = ex.stats.get("pack_s", 0.0), ex.stats.get("write_s", 0.0)
            logger.info(
                f"[glm53f-export] profile [{start}, {end}): dsa read {t_read:.3f} s, dsa pack+write {t_dsa:.3f} s "
                f"(pack {pk - self._prof_pk:.3f} write {wr - self._prof_wr:.3f}), kda + sync {time.perf_counter() - tk:.3f} s"
            )
            self._prof_pk, self._prof_wr = pk, wr
        log_export(ex.stats, start, end, t0)
        ex.stats["bytes"] = ex.stats.get("bytes", 0) + nbytes


__all__ = ["GLM53FlashPrefillRuntime", "MTP_LAYER"]
