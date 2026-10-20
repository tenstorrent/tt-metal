# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident multi-layer GLM-5.3-Flash decode model on a mesh: embedding -> N decoder layers -> head, one
traced decode step for ``rows * users_per_row`` users (design section 5: TP over the columns, one user group per
row, hybrid experts). Every layer is a stage-1 ``FunctionalDecoder`` built on the mesh; the embedding table and
the head are replicated on every chip (the head's vocab split over the columns is a later optimization).

Memory per chip at 8x4: experts 42 layers x 7.7 GB / 32 ~ 10 GB, non-expert weights / 4 ~ 2.2 GB, embedding
1.27 GB, head 0.63 GB (bfp8), caches and states per layer.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.embedding import TtEmbedding
from models.demos.glm53_flash.tt.functional_decoder import DecodePositions, FunctionalDecoder
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.model_head import TtModelHead


class GLM53MeshModel:
    def __init__(
        self,
        mesh_device,
        cfg: GLM53FlashConfig,
        layers: list[int],
        *,
        max_batch: int,
        max_seq_len: int,
        snapshot: Path | None = None,
        weight_cache_path: Path | None = None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        build_head: bool = True,
    ):
        self.device = mesh_device
        self.cfg = cfg
        self.mesh = GLMMesh.from_device(mesh_device)
        self.layer_ids = list(layers)
        self.max_batch = max_batch  # per row
        self.total_users = max_batch * self.mesh.rows
        self.max_seq_len = max_seq_len
        snapshot = snapshot or default_snapshot_dir()
        headers = shard_headers(snapshot)
        t0 = time.time()
        self.embedding = TtEmbedding.from_checkpoint(mesh_device, cfg, snapshot, headers)
        self.head = TtModelHead.from_checkpoint(mesh_device, cfg, snapshot, headers) if build_head else None
        logger.info(f"mesh model: embedding + head built in {time.time() - t0:.1f} s")
        self.layers: list[FunctionalDecoder] = []
        self.build_times: dict[int, float] = {}
        for i in self.layer_ids:
            t0 = time.time()
            raw = load_layer_raw(snapshot, i, headers)
            t1 = time.time()
            dec = FunctionalDecoder.from_state_dict(
                raw,
                hf_config=cfg,
                layer_idx=i,
                mesh_device=mesh_device,
                max_batch=max_batch,
                max_seq_len=max_seq_len,
                weight_cache_path=weight_cache_path / f"layer{i}" if weight_cache_path else None,
                weight_dtype=weight_dtype,
                expert_dtype=expert_dtype,
                expert_down_dtype=expert_down_dtype,
            )
            del raw
            self.layers.append(dec)
            self.build_times[i] = time.time() - t0
            logger.info(f"mesh model: layer {i} built in {self.build_times[i]:.1f} s (host load {t1 - t0:.1f} s)")
        self.positions: DecodePositions | None = None
        self.ids_dev: ttnn.Tensor | None = None

    # ------------------------------------------------------------------ inputs
    def ids_to_device(self, ids: torch.Tensor) -> ttnn.Tensor:
        """``[rows*B]`` token ids (row r's users first) -> the persistent ``[1,1,1,rows*B]`` uint32 input,
        row-sharded on a multi-row mesh (each row embeds its own B tokens)."""
        host = self.embedding.tokens_to_host(ids, pad_to=self.total_users)
        if self.mesh.rows > 1:
            return self.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def load_ids(self, ids: torch.Tensor):
        host = self.embedding.tokens_to_host(ids, pad_to=self.total_users)
        mapper = self.mesh.shard_rows(3) if self.mesh.rows > 1 else None
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper), self.ids_dev
        )

    def set_positions(self, host_positions: list[int]):
        if self.positions is None:
            self.positions = self.layers[0].make_decode_positions(host_positions)
        else:
            self.layers[0].update_decode_positions(self.positions, host_positions)

    # ------------------------------------------------------------------ step
    def decode_step(self, ids_dev: ttnn.Tensor, positions: DecodePositions):
        """ids [1,1,1,B] (per row) -> (streams after the last layer [1,1,B,16384], logits, greedy ids)."""
        streams = self.embedding(ids_dev)
        for dec in self.layers:
            out = dec.decode_forward(streams, positions=positions)
            ttnn.deallocate(streams)
            streams = out
        if self.head is None:
            return streams, None, None
        logits, ids = self.head.forward_greedy(streams)
        return streams, logits, ids

    # ------------------------------------------------------------------ prefill (TTFT)
    def prefill(self, prompt_ids: torch.Tensor, user: int = 0):
        """One prompt through every layer (sequence-parallel over the rows on a multi-row mesh) and the
        head on its last token. Returns ``(streams_local, logits_last [1,1,1,V] per chip, owner_row)``:
        the row holding the last token is ``owner_row`` (every row computes the head on its own local
        row ``(S-1) % T_local``; only the owner's result is the prompt's)."""
        S = int(prompt_ids.numel())
        unit = 32 * self.mesh.rows
        T_pad = -(-S // unit) * unit
        host = self.embedding.tokens_to_host(prompt_ids, pad_to=T_pad)  # [1,1,1,T_pad]
        if self.mesh.rows > 1:
            ids_dev = self.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            ids_dev = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        streams = self.embedding(ids_dev)  # [1,1,T_pad/rows,16384] per row (its slab)
        ttnn.deallocate(ids_dev)
        for dec in self.layers:
            out = dec.prefill_forward(streams, user=user, seq_len=S)
            ttnn.deallocate(streams)
            streams = out
        T_loc = T_pad // self.mesh.rows
        owner_row, local_last = divmod(S - 1, T_loc)
        logits = self.head.last_token_logits(streams, local_last + 1) if self.head is not None else None
        return streams, logits, owner_row

    def first_token(self, logits_last: ttnn.Tensor, owner_row: int) -> int:
        ids = self.head.greedy(logits_last)  # [1,1,1,1] per chip
        shards = [ttnn.to_torch(t).view(torch.int32).reshape(-1) for t in self.mesh.device_tensors(ids)]
        return int(shards[owner_row * self.mesh.cols][0])

    def ids_to_host(self, ids: ttnn.Tensor) -> torch.Tensor:
        """greedy ids [1,1,B,1] uint32 per chip -> [rows*B] int64."""
        t = self.mesh.to_host(ids, rows=2 if self.mesh.rows > 1 else None)
        return t.view(torch.int32).reshape(-1).long()

    # ------------------------------------------------------------------ trace
    def capture(self, ids: torch.Tensor, host_positions: list[int], cq_id: int = 0):
        """Warm-up (compile) + capture one decode step on persistent inputs; returns the trace id and the
        captured output tensors. The warm-up and the capture both consume the token (state advances twice)."""
        self.ids_dev = self.ids_to_device(ids)
        self.set_positions(host_positions)
        self.decode_step(self.ids_dev, self.positions)
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
        outs = self.decode_step(self.ids_dev, self.positions)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=cq_id)
        return trace_id, outs

    def replay(self, trace_id, cq_id: int = 0, blocking: bool = True):
        ttnn.execute_trace(self.device, trace_id, cq_id=cq_id, blocking=blocking)

    def reset_kda_states(self):
        for dec in self.layers:
            if dec.kda_batched is None:
                continue
            for t in (dec.kda_batched.recurrent, dec.kda_batched.convolution):
                z = ttnn.zeros(
                    tuple(t.shape),
                    dtype=t.dtype,
                    layout=t.layout,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(z, t)
                ttnn.deallocate(z)
