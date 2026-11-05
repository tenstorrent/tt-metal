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
        if not isinstance(input_tensor, torch.Tensor) and actual_end <= actual_start:
            ttnn.deallocate(input_tensor)
        if actual_end <= actual_start:
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
        t0 = time.perf_counter()
        self._prefill(ids, actual_start)
        t1 = time.perf_counter()
        if self._async:
            # KDA states: device copies now (the next chunk mutates them in place), then one sync; the DSA rows of
            # [actual_start, actual_end) are final, and the next chunk writes only rows past them
            rows = self._dsa_rows_cq(actual_start, actual_end) if self._async_pack else None
            for layer in kv_caches.kda_layers:
                kv_caches.export_kda(layer, self.decs[layer].kda_batched)
            ttnn.synchronize_device(self.mesh_device)
            self._xq.put(("chunk", kv_caches, actual_start, actual_end, request_id, rows))
            logger.info(
                f"[glm53f-runtime] chunk {request_id} slot {slot_id} [{actual_start}, {actual_end}): prefill {t1 - t0:.3f} s, "
                f"kda export + sync {time.perf_counter() - t1:.3f} s (DSA export async)"
            )
            return None
        self._export(kv_caches, actual_start, actual_end)
        logger.info(
            f"[glm53f-runtime] chunk {request_id} slot {slot_id} [{actual_start}, {actual_end}): prefill {t1 - t0:.3f} s, "
            f"export {time.perf_counter() - t1:.3f} s"
        )
        if self._sink is not None:
            for layer in range(TABLE_LAYERS):
                self._sink(layer, request_id)
        return None

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

    def _export(self, ex: GlmRingExport, start: int, end: int) -> None:
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
