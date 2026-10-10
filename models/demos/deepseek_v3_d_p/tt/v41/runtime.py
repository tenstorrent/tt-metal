# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill runtime for the generic prefill runner (``models/demos/common/prefill``): one rank (one
galaxy, sp x tp), the encoder 0..19 + layer 20 KV-only (``tt/v41/prefill.py``), the decode contract's export caches and
KV chunk table (``tt/v41/kv_export.py``). tt-blaze DS41F-0037 M2.

Per chunk: token ids to the host (the embedding and the Engram hash / gather are host work), ``V41Prefill`` runs the
chunk eagerly, its new rows go into the export caches on device, then one ack per layer (21) -- the export is complete at
every chunk boundary. The first token is NOT produced here: the ring replays the prompt's tail (R >= 2541) itself, which
yields it (``disagg_prefill_plan.md``). A request's first chunk (actual_start 0) resets that slot's states and zeroes its export rows.
``num_users`` slots: one attention-state set per slot (``V41Prefill.use_slot``), the weights shared.
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
            kv_only=True,
            num_users=num_users,
        )
        from blaze.models.deepseek_v4_1_flash.engram_host import EngramHost  # tt-blaze's ttnn-free host half

        kw = {} if model_dir is None else {"model_dir": model_dir}
        self.engram = EngramHost(
            max_seq_len=max_seq_len + 64,
            n_slots=num_users,
            table_dir=os.environ.get("DSV41_ENGRAM_TABLE_DIR") or None,
            **kw,
        )
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
        )

    def kv_migration_base_address(self, kv_caches) -> int:
        return int(kv_caches.swa.buffer_address())

    def build_kv_chunk_table(
        self, kv_caches, path: str, *, first_layer_idx: int = 0, num_my_layers=None, stage_layout=None
    ) -> str:
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

    def prefill_chunk(
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
        if int(actual_start) == 0:
            self.pf.reset()
            kv_caches.zero_slot(int(slot_id))
        assert self.pf.kv_actual == int(actual_start), (self.pf.kv_actual, actual_start)
        self._request_id = int(request_id)
        t0 = time.time()
        self.pf._chunk([int(v) for v in ids[:real]], int(actual_start), self.engram, None)
        t1 = time.time()
        export_chunk(kv_caches, self.pf, slot_id, int(actual_start), int(actual_end))
        t2 = time.time()
        ttnn.synchronize_device(self.mesh_device)
        t3 = time.time()
        if self._ack is not None and not warmup:
            for L in range(self.num_layers):
                self._ack(L)
        t4 = time.time()
        logger.info(
            f"[v41 runtime] chunk [{actual_start}, {actual_end}) slot {slot_id} in {t4 - t0:.2f} s "
            f"(compute issue {t1 - t0:.2f} + export issue {t2 - t1:.2f} + sync {t3 - t2:.2f} + acks {t4 - t3:.2f})"
            + (" (warm-up)" if warmup else "")
        )
        return None

    def set_layer_ack_channel(self, channel) -> None:
        self._ack = lambda layer_idx: channel.inject(1)

    def set_layer_completion_sink(self, sink) -> None:
        self._ack = lambda layer_idx: sink(layer_idx, self._request_id)

    def flush_acks(self) -> None:
        return None

    def tail_hidden_row(self, tail_out, row: int):
        raise NotImplementedError("V4.1's first token comes from the ring's replay of the prompt tail (DS41F-0037)")
