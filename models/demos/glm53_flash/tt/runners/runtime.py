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
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import KvCacheStage
from models.demos.glm53_flash.tt.runners.kv_export import (
    DSA_LAYERS,
    MTP_LAYER,
    POOL_CHUNK_TOKENS,
    POOL_RATIO,
    TABLE_LAYERS,
    TPP,
    GlmRingExport,
    log_export,
)


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
                f"export {te:.2f} s, wall {time.perf_counter() - tw:.2f} s"
            )

    def prefill_chunk(self, input_tensor, kv_caches, *, slot_id, actual_start, actual_end, request_id=0, **_):
        ids = self._ids_host(input_tensor)[: actual_end - actual_start]
        if not isinstance(input_tensor, torch.Tensor):
            ttnn.deallocate(input_tensor)
        t0 = time.perf_counter()
        self._prefill(ids, actual_start)
        t1 = time.perf_counter()
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
        streams, logits, _ = self.model.prefill(ids, user=0, start=start)
        ttnn.deallocate(streams)
        if logits is not None:
            ttnn.deallocate(logits)

    def _rows(self, t, b0: int, b1: int) -> torch.Tensor:
        """Blocks [b0, b1) of a mesh-replicated paged cache ``[blocks, 1, 32, W]`` (first replica)."""
        rest = [int(t.shape[i]) for i in range(1, len(t.shape))]
        part = ttnn.slice(t, [b0, 0, 0, 0], [b1, *rest])
        out = self.model.mesh.to_host(part)
        if part.buffer_address() != t.buffer_address():
            ttnn.deallocate(part)
        return out.reshape(-1, out.shape[-1])

    def _export(self, ex: GlmRingExport, start: int, end: int) -> None:
        t0 = time.perf_counter()
        nbytes = 0
        for layer in DSA_LAYERS:
            c = self.decs[layer].cache
            b0, b1 = start // TPP, -(-end // TPP)
            latent = self._rows(c.cache, b0, b1)[: end - start].to(torch.bfloat16)
            pools = None
            idx = getattr(c, "index", None)
            if idx is not None:
                q0, q1 = start // POOL_RATIO, -(-end // POOL_RATIO)
                r0, r1 = q0 // 32, min(-(-q1 // 32), int(idx.pool_blocks))
                pools = self._rows(idx.pool_cache, r0, r1)[: q1 - q0]
            nbytes += ex.export_dsa(layer, latent, start // TPP, pools, start // POOL_CHUNK_TOKENS)
        for layer in ex.kda_layers:
            ex.export_kda(layer, self.decs[layer].kda_batched)
        ttnn.synchronize_device(self.mesh_device)
        log_export(ex.stats, start, end, t0)
        ex.stats["bytes"] = ex.stats.get("bytes", 0) + nbytes


__all__ = ["GLM53FlashPrefillRuntime", "MTP_LAYER"]
