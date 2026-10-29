# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident multi-layer GLM-5.3-Flash decode model on a mesh: embedding -> N decoder layers -> head, one
traced decode step for ``rows * users_per_row`` users (design section 5: TP over the columns, one user group per
row, hybrid experts). Every layer is a stage-1 ``FunctionalDecoder`` built on the mesh; the embedding table and
the head: embedding replicated on every chip, the head vocab split over the TP columns (all-gathered logits).

Memory per chip at 8x4: experts 42 layers x 7.7 GB / 32 ~ 10 GB, non-expert weights / 4 ~ 2.2 GB, embedding
1.27 GB, head 0.63 GB (bfp8), caches and states per layer.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.decoder_layer import WaveInfo
from models.demos.glm53_flash.tt.dsa import dense_exact_max_pos
from models.demos.glm53_flash.tt.embedding import TtEmbedding
from models.demos.glm53_flash.tt.functional_decoder import TILE, DecodePositions, FunctionalDecoder, PrefillTraceCall
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.model_head import TtModelHead
from models.demos.glm53_flash.tt.moe import pad_mask_enabled
from models.demos.glm53_flash.tt.packed_inputs import UINT32, PackedDecodeLayout
from models.demos.glm53_flash.tt.trace_prefill import (
    DEFAULT_LENGTHS,
    DENSE_ROWS,
    PrefillSlot,
    PrefillTraceInputs,
    PrefillTraceResult,
    WaveSlot,
    WaveTraceInputs,
    pick_prefill_length,
    supported_prefill_lengths,
)

# the eager prefill's chunk-length buckets live in the ttnn-free tt/vllm_config.py (the served warm-up ladder walks
# them); re-exported here for tt/dsa.py and the tests
from models.demos.glm53_flash.tt.vllm_config import (  # noqa: E402  (re-exports)
    DEFAULT_PREFILL_BUCKETS,
    PREFILL_BUCKETS_ENV,
    eager_pad_length,
    prefill_buckets,
)

__all__ = ["DEFAULT_PREFILL_BUCKETS", "PREFILL_BUCKETS_ENV", "eager_pad_length", "prefill_buckets"]

PACKED_INPUTS_ENV = "GLM53_DECODE_PACKED_INPUTS"


def packed_inputs_enabled() -> bool:
    """``GLM53_DECODE_PACKED_INPUTS=1``: one packed host->device upload per decode step (``tt/packed_inputs.py``)
    instead of the per-field refreshes. Default off until validated on the device
    (``tests/test_decode_packed_inputs_mesh.py``)."""
    return os.environ.get(PACKED_INPUTS_ENV, "1") == "1"  # default on since 2026-10-27 (smoke r12)


class PackedDecodeInputs:
    """The device side of ``GLM53_DECODE_PACKED_INPUTS`` (layout: ``tt/packed_inputs.py``): one persistent int32
    ``[1, 1, H, W]`` ROW_MAJOR segment per chip, refreshed with ONE ``copy_host_to_device_tensor`` per step
    (``upload``), and the trace-safe unpack that copies every field into the EXISTING persistent decode inputs at
    the start of the decode step (``unpack``): one ``ttnn.bitcast`` of the segment to uint32 (the ids / tail_pool /
    valid_len fields), then per field a ROW_MAJOR ``ttnn.slice`` (row fields ``[1,1,1,n]``, column fields
    ``[1,1,n,1]``), a free ``ttnn.reshape`` view to the field's shape (the last dim is unchanged) and a ``ttnn.copy``
    into the field tensor; the consumers of ``DecodePositions`` / ``SparseDecodeMeta`` / ``ids_dev`` never change.
    Allocate it before any trace capture (``GLM53MeshModel.warmup_decode``)."""

    def __init__(self, mesh: GLMMesh, layout: PackedDecodeLayout, host: torch.Tensor):
        self.mesh = mesh
        self.layout = layout
        if tuple(host.shape) != layout.host_shape:
            raise ValueError(f"packed host tensor {tuple(host.shape)} is not {layout.host_shape}")
        self.host = host
        self.dev = ttnn.from_torch(
            host.contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mapper(),
        )

    def mapper(self):
        """Dim 2 (the stacked segments) over the chips: one segment per chip in row-major chip order with the user
        partition (``ShardTensorToMesh``), one per row replicated over the columns otherwise, none on one chip
        (``load_ids``' placement)."""
        if self.layout.segment_cols > 1:
            return self.mesh.shard_2d(2, 2)
        if self.mesh.rows > 1:
            return self.mesh.shard_rows(2)
        return None

    def upload(self, host: torch.Tensor):
        """The step's packed host values -> the persistent device segments (one ``copy_host_to_device_tensor``)."""
        if tuple(host.shape) != self.layout.host_shape:
            raise ValueError(f"packed host tensor {tuple(host.shape)} is not {self.layout.host_shape}")
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                host.contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper()
            ),
            self.dev,
        )
        self.host = host

    def targets(self, ids_dev: ttnn.Tensor, dp: DecodePositions) -> dict[str, ttnn.Tensor]:
        """The persistent field tensors the unpack writes, keyed like ``layout.fields``."""
        lay = self.layout
        t = {"ids": ids_dev, "positions": dp.positions, "pool_ids": dp.pool_ids, "ring_slots": dp.ring_slots}
        if lay.dense != (dp.dense_positions is not None):
            raise ValueError(
                f"packed layout dense={lay.dense} but the positions carry "
                f"{'a' if dp.dense_positions is not None else 'no'} dense_positions tensor"
            )
        if lay.dense:
            t["dense_positions"] = dp.dense_positions
        sp = dp.sparse
        if sp is not None:
            t["cur_pos"] = sp.cur_pos
            t["tail_pool"] = sp.tail_pool
            if sp.valid_len_rows is not None:
                t["valid_len_rows"] = sp.valid_len_rows
            if lay.per_user_valid_len:
                if len(sp.valid_len) != lay.users_per_chip:
                    raise ValueError(f"{len(sp.valid_len)} valid_len tensors for {lay.users_per_chip} users per chip")
                for u, vt in enumerate(sp.valid_len):
                    t[f"valid_len[{u}]"] = vt
        return t

    def unpack(self, ids_dev: ttnn.Tensor, dp: DecodePositions):
        """Device ops only (captured with the decode step): every field of the segment into its persistent tensor."""
        fields = self.layout.fields
        dram = ttnn.DRAM_MEMORY_CONFIG  # every transient stays in DRAM (an L1 buffer alive across ops can clash
        # with the static circular buffers of a later program: the mesh.py ``const`` note)
        u32 = ttnn.bitcast(self.dev, ttnn.uint32, memory_config=dram)
        for name, dst in self.targets(ids_dev, dp).items():
            f = fields[name]
            want = ttnn.uint32 if f.dtype == UINT32 else ttnn.int32
            if dst.dtype != want or dst.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise ValueError(
                    f"packed field {name}: the persistent tensor is {dst.dtype} {dst.layout}, expected {want} ROW_MAJOR"
                )
            if math.prod(list(dst.shape)) != f.n:
                raise ValueError(f"packed field {name}: {f.n} values for a persistent tensor of shape {dst.shape}")
            s = ttnn.slice(u32 if f.dtype == UINT32 else self.dev, list(f.start), list(f.end), memory_config=dram)
            shape = list(dst.shape)
            v = s if list(s.shape) == shape else ttnn.reshape(s, shape)
            ttnn.copy(v, dst)
            ttnn.deallocate(s)
        ttnn.deallocate(u32)


class GLM53MeshModel:
    # the attached MTP drafter (``attach_mtp``; ``tt/mtp.py`` ``TtMTPBlock``) and its own level-1 positions
    # (``set_mtp_positions``): the speculative-decode draft step (``decode_step_with_draft``, ``tt/spec_decode.py``)
    mtp = None
    mtp_positions: DecodePositions | None = None

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
        if os.environ.get("GLM53_PREFILL_EXPERTS", "auto") == "auto" and max_seq_len <= 4096:
            os.environ[
                "GLM53_PREFILL_EXPERTS"
            ] = "blocks"  # short prompts: the traced prefill is as fast on the blocks path
            logger.info(f"mesh model: GLM53_PREFILL_EXPERTS=auto -> blocks prefill path for max_seq_len {max_seq_len}")
        if os.environ.get("GLM53_PREFILL_EXPERTS", "auto") == "auto":
            # the regions prefill path, with the stacked expert copy kept for the decode block op when the memory
            # planner says both fit beside the caches of (max_batch users per row x max_seq_len); resolved for the
            # whole process before the layers are built (tt/moe.py reads the knobs at construction)
            from models.demos.glm53_flash.tt.memory_plan import BF16_BYTES, BFP4_BYTES, BFP8_BYTES, MemoryPlan

            eb = {ttnn.bfloat8_b: BFP8_BYTES, ttnn.bfloat4_b: BFP4_BYTES, ttnn.bfloat16: BF16_BYTES}.get(
                expert_dtype, BFP8_BYTES
            )
            partition = os.environ.get("GLM53_PARTITION_USERS", "0") == "1"
            scratch = os.environ.get("GLM53_PREFILL_PARTITION", "scratch") != "heads2users"
            plan = MemoryPlan.estimate(
                cfg,
                self.mesh.rows,
                self.mesh.cols,
                partition=partition,
                expert_bytes=eb,
                keep_stacked=True,
                scratch_slot=scratch,
                trace_mb=int(os.environ.get("GLM53_MODEL_TRACE_MB", "512")),
            )
            from models.demos.glm53_flash.custom_ops.expert_stream.regions import (
                REGIONS_DECODE_ENV,
                regions_decode_mode,
                resolve_keep_stacked,
            )

            fits = plan.fits(max_batch, max_seq_len)
            mode = regions_decode_mode()
            # GLM53_REGIONS_DECODE=stream: the regions lists serve decode too (one copy), whatever the planner says
            keep = resolve_keep_stacked(fits, mode)
            os.environ["GLM53_PREFILL_EXPERTS"] = "regions"
            os.environ["GLM53_REGIONS_KEEP_STACKED"] = "1" if keep else "0"
            logger.info(
                f"mesh model: GLM53_PREFILL_EXPERTS=auto -> regions prefill, stacked decode copy "
                f"{'kept' if keep else 'dropped'} for {max_batch} users/row x {max_seq_len} (planner: "
                f"{'fits' if fits else 'does not fit'} beside the caches; {REGIONS_DECODE_ENV}={mode}"
                f"{': the stream ops read the regions lists, one copy' if mode == 'stream' else ''})"
            )
        snapshot = snapshot or default_snapshot_dir()
        headers = shard_headers(snapshot)
        t0 = time.time()
        self.embedding = TtEmbedding.from_checkpoint(mesh_device, cfg, snapshot, headers)
        self.head = (
            TtModelHead.from_checkpoint(mesh_device, cfg, snapshot, headers, mesh=self.mesh) if build_head else None
        )
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
        if os.environ.get("GLM53_CACHES_LAST", "0") == "1":
            t0 = time.time()
            for dec in self.layers:
                dec.allocate_cache()
            logger.info(
                f"mesh model: GLM53_CACHES_LAST=1: the DSA caches allocated after every weight in {time.time() - t0:.1f} s"
            )
        self.positions: DecodePositions | None = None
        self.ids_dev: ttnn.Tensor | None = None
        # GLM53_DECODE_PACKED_INPUTS: the one-upload decode inputs (allocated by ``warmup_decode``, before any capture)
        # and the last token ids they carry (``[rows*B]`` int32, padded); None = the direct per-field refreshes
        self.packed: PackedDecodeInputs | None = None
        self._host_ids: torch.Tensor | None = None
        # the traced prefill's device-side user slot (tt/trace_prefill.py): allocated here, before any trace
        # capture, because a persistent tensor allocated after a capture may alias that trace's transient buffers
        self.prefill_slot: PrefillSlot | None = PrefillSlot.allocate(self.mesh, self.layers) if self.layers else None
        self.wave_slot: WaveSlot | None = (
            WaveSlot.allocate(self.mesh, self.layers) if self.layers and self.mesh.rows > 1 else None
        )
        self.wave_traces: dict[int, WaveTraceInputs] = {}  # per-row length -> the captured wave
        self._wave_ready: dict[int, WaveTraceInputs] = {}  # prepared (allocated + compiled), not yet captured
        self.prefill_traces: dict[int, PrefillTraceInputs] = {}
        # the slot-level prefix cache's KDA snapshot store (tt/kda_snapshot.py): allocated by the served adapter
        # through ``allocate_kda_snapshots`` before any capture; None = no snapshots
        self.kda_snapshots = None

    # ------------------------------------------------------------------ inputs
    def ids_to_device(self, ids: torch.Tensor) -> ttnn.Tensor:
        """``[rows*B]`` token ids (row r's users first) -> the persistent ``[1,1,1,rows*B]`` uint32 input,
        row-sharded on a multi-row mesh (each row embeds its own B tokens)."""
        host = self.embedding.tokens_to_host(ids, pad_to=self.total_users)
        if self.mesh.rows > 1:
            return self.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def load_ids(self, ids: torch.Tensor):
        """The step's token ids -> the persistent ``ids_dev`` (one row-sharded ``copy_host_to_device_tensor``). With
        the packed inputs (``self.packed``) the ids go into the packed segment instead (re-uploaded with the current
        positions; ``load_decode_inputs`` is the one-upload form of ``load_ids`` + ``set_positions``)."""
        host = self.embedding.tokens_to_host(ids, pad_to=self.total_users)
        if self.packed is not None:
            self._host_ids = host.reshape(-1)
            self.packed.upload(self.packed.layout.pack(self._host_ids, self.positions.host))
            return
        self._load_ids_direct(host)

    def _load_ids_direct(self, host: torch.Tensor):
        mapper = self.mesh.shard_rows(3) if self.mesh.rows > 1 else None
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper), self.ids_dev
        )

    def set_positions(self, host_positions: list[int]):
        """The step's positions -> the persistent ``DecodePositions`` (``update_decode_positions``: the per-field
        refreshes, or, with the packed inputs, the host bookkeeping + one packed upload with the current ids)."""
        if self.positions is None:
            self.positions = self.layers[0].make_decode_positions(host_positions)
        elif self.packed is not None:
            self.layers[0].update_decode_positions(self.positions, host_positions, upload=False)
            self.packed.upload(self.packed.layout.pack(self._host_ids, host_positions))
        else:
            self.layers[0].update_decode_positions(self.positions, host_positions)

    def load_decode_inputs(self, ids: torch.Tensor, host_positions: list[int]):
        """``load_ids`` + ``set_positions`` of one decode step. With ``GLM53_DECODE_PACKED_INPUTS=1`` (after
        ``warmup_decode``) this is ONE ``copy_host_to_device_tensor`` of the packed segments (``tt/packed_inputs.py``)
        plus the host bookkeeping of ``update_decode_positions``; otherwise the direct per-field refreshes."""
        if self.packed is None:
            self.load_ids(ids)
            self.set_positions(host_positions)
            return
        self._host_ids = self.embedding.tokens_to_host(ids, pad_to=self.total_users).reshape(-1)
        self.layers[0].update_decode_positions(self.positions, host_positions, upload=False)
        self.packed.upload(self.packed.layout.pack(self._host_ids, host_positions))

    def packed_layout(self) -> PackedDecodeLayout:
        """The packed layout of this model (``self.positions`` must exist: it decides ``dense``). The per-user
        ``valid_len`` list is unpacked only when something reads it (the unbatched top-k, ``GLM53_TOPK_BATCHED=0``)."""
        if self.positions is None:
            raise RuntimeError("set_positions first")
        dec = self.layers[0]
        return PackedDecodeLayout(
            rows=self.mesh.rows,
            cols=self.mesh.cols,
            users_per_row=self.max_batch,
            index_kpool=self.cfg.index_kpool,
            index_select_pools=self.cfg.index_select_pools,
            dense_max_pos=dense_exact_max_pos(self.cfg),
            partition=dec.partition,
            dense=self.positions.dense_positions is not None,
            per_user_valid_len=os.environ.get("GLM53_TOPK_BATCHED", "1") != "1",
        )

    def _stage_decode_inputs(self, ids: torch.Tensor, host_positions: list[int]):
        """The warm-ups' input staging: the direct uploads (allocating the persistent inputs on the first call) and,
        with the knob, the packed segments holding the same values (allocated on the first call: before any capture)."""
        host = self.embedding.tokens_to_host(ids, pad_to=self.total_users)
        if self.ids_dev is None:
            self.ids_dev = self.ids_to_device(ids)
        else:
            self._load_ids_direct(host)
        if self.positions is None:
            self.positions = self.layers[0].make_decode_positions(host_positions)
        else:
            self.layers[0].update_decode_positions(self.positions, host_positions)
        self._host_ids = host.reshape(-1)
        if self.packed is None and packed_inputs_enabled():
            layout = self.packed_layout()
            self.packed = PackedDecodeInputs(self.mesh, layout, layout.pack(self._host_ids, host_positions))
            logger.info(
                f"mesh model: {PACKED_INPUTS_ENV}=1: packed decode inputs {layout.segment_shape} int32 per chip "
                f"({layout.segments} segments, {len(layout.fields)} fields, users/chip {layout.users_per_chip})"
            )
        elif self.packed is not None:
            self.packed.upload(self.packed.layout.pack(self._host_ids, host_positions))

    def _unpack_inputs(self, ids_dev: ttnn.Tensor, positions: DecodePositions):
        """The packed inputs' device-side unpack when the step runs on the model's persistent inputs."""
        if self.packed is not None and ids_dev is self.ids_dev and positions is self.positions:
            self.packed.unpack(ids_dev, positions)

    # ------------------------------------------------------------------ step
    def decode_step(self, ids_dev: ttnn.Tensor, positions: DecodePositions, keep_layer_outputs: bool = False):
        """ids [1,1,1,B] (per row) -> (streams after the last layer [1,1,B,16384], logits, greedy ids).
        ``keep_layer_outputs`` (eager diagnostics only) keeps every layer's output in ``self.layer_outputs``.
        With the packed inputs the step starts with their device-side unpack into ``ids_dev`` / ``positions``
        (when they are the model's persistent inputs: device ops, captured with the step)."""
        self._unpack_inputs(ids_dev, positions)
        streams = self.embedding(ids_dev)
        self.layer_outputs = []
        last = len(self.layers) - 1
        for k, dec in enumerate(self.layers):
            # the last layer feeds the head: its FFN wrap emits the mean over the streams directly
            collapse = k == last and self.head is not None and not keep_layer_outputs and dec.supports_collapse
            out = dec.decode_forward(streams, positions=positions, collapse=collapse)
            if keep_layer_outputs:
                self.layer_outputs.append(out)
            else:
                ttnn.deallocate(streams)
            streams = out
        self.mesh.release_transients()  # constants built under live traces live for this step only (GLMMesh.lazy)
        if self.head is None:
            return streams, None, None
        logits, ids = self.head.forward_greedy(streams)
        return streams, logits, ids

    # ------------------------------------------------------------------ prefill (TTFT)
    def prefill(self, prompt_ids: torch.Tensor, user: int = 0, start: int = 0):
        """One prompt (or one chunk of it: ``start`` > 0 continues the user's prompt, the
        ``FunctionalDecoder`` chunked-prefill contract) through every layer (sequence-parallel over the
        rows on a multi-row mesh) and the head on its last token. Returns ``(streams_local, logits_last
        [1,1,1,V] per chip, owner_row)``: the row holding the last token is ``owner_row`` (every row
        computes the head on its own local row ``(S-1) % T_local``; only the owner's result is the
        prompt's)."""
        S = int(prompt_ids.numel())
        unit = 32 * self.mesh.rows
        T_pad = eager_pad_length(S, unit)
        host = self.embedding.tokens_to_host(prompt_ids, pad_to=T_pad)  # [1,1,1,T_pad]
        if self.mesh.rows > 1:
            ids_dev = self.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            ids_dev = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        streams = self.embedding(ids_dev)  # [1,1,T_pad/rows,16384] per row (its slab)
        ttnn.deallocate(ids_dev)
        return self.prefill_streams(streams, S, user=user, start=start)

    def prefill_streams(self, streams: ttnn.Tensor, seq_len: int, user: int = 0, start: int = 0):
        """``prefill`` from ready-made input streams (this row's slab ``[1,1,T_pad/rows,16384]`` fp32 of
        a ``T_pad = ceil(seq_len / (32 * rows)) * 32 * rows`` chunk; consumed): the multimodal path
        (``tt/multimodal.py``). Same return as ``prefill``."""
        S = seq_len
        T_pad = streams.shape[-2] * self.mesh.rows
        for dec in self.layers:
            out = dec.prefill_forward(streams, user=user, seq_len=S, start=start)
            ttnn.deallocate(streams)
            streams = out
        T_loc = T_pad // self.mesh.rows
        owner_row, local_last = divmod(S - 1, T_loc)
        logits = self.head.last_token_logits(streams, local_last + 1) if self.head is not None else None
        self.mesh.release_transients()  # constants built under live traces live for this chunk only (GLMMesh.lazy)
        return streams, logits, owner_row

    # ------------------------------------------------------------------- waves
    wave_per_row_users = False  # phase 1 (doc/mesh/row_ownership.md section 3): one slot index per wave

    def prefill_wave(self, call) -> dict[int, tuple[int, torch.Tensor]]:
        """One wave of per-row prompt chunks (``doc/mesh/row_ownership.md`` mechanism (a); ``call``: a
        ``tt/vllm_config.py`` ``WaveCall``): row r embeds and runs its member's chunk ``call.ids[r]`` (the tokens of
        ``[start, end)`` of that prompt; at most one member per row, idle rows run pad tokens into nothing), every
        layer row-locally (KDA: the row's own chronology and carry; MoE: the row's own pad mask; DSA: step 3, not yet),
        then the head on each member row's last token. Returns ``{row: (first greedy token, last-token logits [vocab]
        fp32)}`` for the member rows. Phase 1: ``call.user`` is the slot index on every row."""
        rows = self.mesh.rows
        if int(call.rows) != rows:
            raise ValueError(f"a wave of {call.rows} rows on a {rows}-row mesh")
        if call.user is None:
            raise NotImplementedError("phase 2 (a per-row slot index) is not implemented: a wave needs call.user")
        members = {int(m.row): m for m in call.members}
        T = eager_pad_length(int(call.t_pad), TILE)  # the per-row bucket: one ttKDA layer / head program per bucket
        starts = tuple(int(members[r].start) if r in members else 0 for r in range(rows))
        valids = tuple(int(members[r].n_tokens) if r in members else 0 for r in range(rows))
        host = torch.full((rows, T), int(self.embedding.pad_token_id), dtype=torch.int32)
        for r, m in members.items():
            ids = torch.as_tensor(call.ids[r]).reshape(-1)
            if int(ids.numel()) != valids[r]:
                raise ValueError(f"row {r}: {int(ids.numel())} ids for a chunk of {valids[r]} tokens")
            host[r, : valids[r]] = ids.to(torch.int32)
        host = self.embedding.tokens_to_host(host.reshape(-1))  # [1,1,1,rows*T]: row r's chunk is slab r
        if rows > 1:
            ids_dev = self.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            ids_dev = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        streams = self.embedding(ids_dev)  # [1,1,T,16384] per row: its own chunk
        ttnn.deallocate(ids_dev)
        kda = next((dec for dec in self.layers if dec.is_kda), None)  # the chronology scalars live on a KDA layer
        attn = kda._require_weights().attn if kda is not None else None
        ends = tuple(s0 + n for s0, n in zip(starts, valids))
        info = WaveInfo(
            T=T,
            starts=starts,
            valids=valids,
            start_t=attn.scalar_rows(starts, "wave_start") if attn is not None else None,
            end_t=attn.scalar_rows(ends, "wave_end") if attn is not None else None,
        )
        for dec in self.layers:
            out = dec.prefill_wave_forward(streams, user=int(call.user), wave=info)
            ttnn.deallocate(streams)
            streams = out
        res: dict[int, tuple[int, torch.Tensor]] = {}
        if self.head is not None:
            logits = self.head.last_token_logits_rows(streams, valids)  # [1,1,1,V] per chip: row r's on row r
            host_logits = self.mesh.to_host(logits, rows=0).float().reshape(rows, -1)
            ttnn.deallocate(logits)
            for r in members:
                res[r] = (int(torch.argmax(host_logits[r])), host_logits[r])
        ttnn.deallocate(streams)
        return res

    # ----------------------------------------------------------------- traced waves
    def wave_trace_lengths(self, lengths=DEFAULT_LENGTHS) -> list[int]:
        """The per-row lengths a wave can be captured at: multiples of 32 within the dense range."""
        return [int(T) for T in lengths if T % TILE == 0 and T <= DENSE_ROWS]

    def has_wave_trace(self, T: int) -> bool:
        return pick_prefill_length(T, self.wave_traces) is not None

    def _wave_host(self, call, T: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        rows = self.mesh.rows
        members = {int(m.row): m for m in call.members}
        valids = tuple(int(members[r].n_tokens) if r in members else 0 for r in range(rows))
        host = torch.full((rows, T), int(self.embedding.pad_token_id), dtype=torch.int32)
        for r, m in members.items():
            ids = torch.as_tensor(call.ids[r]).reshape(-1)
            if int(ids.numel()) != valids[r] or valids[r] > T:
                raise ValueError(f"row {r}: {int(ids.numel())} ids for a chunk of {valids[r]} tokens in a {T}-row wave")
            host[r, : valids[r]] = ids.to(torch.int32)
        return self.embedding.tokens_to_host(host.reshape(-1)), valids

    def _wave_graph(self, inp: WaveTraceInputs):
        """The traced wave graph on the persistent inputs ``inp`` + ``self.wave_slot`` (no host value of the wave
        inside): embedding -> every layer (the device-slot wave form) -> the head on each row's selected token."""
        slot = self.wave_slot
        info = WaveInfo(T=inp.T, starts=(0,) * self.mesh.rows, valids=inp.valids, start_t=inp.starts, end_t=inp.ends)
        streams = self.embedding(inp.ids)  # [1,1,T,16384] per row: its own chunk
        for dec in self.layers:
            out = dec.prefill_wave_traced(
                streams,
                wave=info,
                slot=slot,
                ring_sel=inp.ring_sel,
                row_mask=inp.row_mask if pad_mask_enabled() else None,
            )
            ttnn.deallocate(streams)
            streams = out
        if self.head is None:
            return streams
        logits = self.head.last_token_logits_rows_from(streams, inp.tile_start, inp.tile_end, inp.onehot)
        ttnn.copy(logits, inp.out_logits)
        ttnn.deallocate(logits)
        return streams

    def prepare_waves(self, lengths=None, warmup_user: int = 0) -> list[int]:
        """Allocate the persistent inputs of every requested per-row wave length (default ``wave_trace_lengths``)
        and compile each graph eagerly (every row a member, dummy tokens into slot ``warmup_user``) without
        capturing: the served warm-up prepares the waves before ANY capture (an allocation after a capture may alias
        that trace's transient buffers) and ``capture_waves`` captures them after the prefill traces. Returns the
        lengths prepared now."""
        if self.wave_slot is None:
            raise RuntimeError("waves need a multi-row mesh with layers")
        todo = [
            T
            for T in (self.wave_trace_lengths() if lengths is None else lengths)
            if T not in self.wave_traces and T not in self._wave_ready
        ]
        if not todo:
            return []
        vocab = self.cfg.vocab_size if self.head is not None else None
        new = {T: WaveTraceInputs.allocate(self.mesh, T, self.cfg.hidden_size, vocab) for T in todo}
        rows = self.mesh.rows
        self.wave_slot.set(warmup_user, ())
        for T, inp in new.items():
            t0 = time.perf_counter()
            dummy = (torch.arange(rows * T, dtype=torch.long) % 1000 + 1000).to(torch.int32)
            inp.set_wave(self.mesh, dummy.reshape(1, 1, 1, -1), (T,) * rows)
            ttnn.deallocate(self._wave_graph(inp))
            ttnn.synchronize_device(self.device)
            self._wave_ready[T] = inp
            logger.info(f"traced wave: T={T} per row compiled in {time.perf_counter() - t0:.1f} s")
        return todo

    def capture_waves(self, lengths=None, cq_id: int = 0, warmup_user: int = 0) -> list[int]:
        """Capture one wave trace per per-row length (default ``wave_trace_lengths``), preparing (``prepare_waves``:
        inputs + eager compile) any length not prepared yet; like ``capture_prefill`` (before ``capture`` or after
        ``warmup_decode``). Returns the lengths captured now."""
        todo = [T for T in (self.wave_trace_lengths() if lengths is None else lengths) if T not in self.wave_traces]
        if not todo:
            return []
        self.prepare_waves(todo, warmup_user=warmup_user)
        self.wave_slot.set(warmup_user, ())
        for T in todo:
            inp = self._wave_ready.pop(T)
            t0 = time.perf_counter()
            inp.trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
            inp.streams = self._wave_graph(inp)
            ttnn.end_trace_capture(self.device, inp.trace_id, cq_id=cq_id)
            self.mesh.allocations_frozen[0] = True
            ttnn.synchronize_device(self.device)
            self.wave_traces[T] = inp
            logger.info(f"traced wave: T={T} per row captured in {time.perf_counter() - t0:.1f} s")
        return todo

    def prefill_traced_wave(self, call, cq_id: int = 0) -> dict[int, tuple[int, torch.Tensor]]:
        """``prefill_wave`` through the captured wave of the smallest per-row length that holds every member (all
        members at start 0): refreshes the wave slot (slot index + idle rows) and the per-length inputs, replays,
        reads each member row's last-token logits. Returns ``{row: (first greedy token, logits [vocab] fp32)}``."""
        rows = self.mesh.rows
        if int(call.rows) != rows or call.user is None:
            raise ValueError("a traced wave needs call.rows == mesh rows and a slot index (phase 1)")
        if any(int(m.start) for m in call.members):
            raise ValueError("a traced wave starts every member at 0")
        n_max = max(int(m.n_tokens) for m in call.members)
        T = pick_prefill_length(n_max, self.wave_traces)
        if T is None:
            raise ValueError(f"no wave trace for {n_max} tokens per row (captured: {sorted(self.wave_traces)})")
        inp = self.wave_traces[T]
        host, valids = self._wave_host(call, T)
        idle = tuple(r for r, v in enumerate(valids) if v == 0)
        self.wave_slot.set(int(call.user), idle)
        inp.set_wave(self.mesh, host, valids)
        ttnn.execute_trace(self.device, inp.trace_id, cq_id=cq_id, blocking=True)
        for dec in self.layers:
            for r, v in enumerate(valids):
                if v:
                    dec.next_start_rows[r][int(call.user)] = v
        res: dict[int, tuple[int, torch.Tensor]] = {}
        if inp.out_logits is not None:
            host_logits = self.mesh.to_host(inp.out_logits, rows=0).float().reshape(rows, -1)
            for m in call.members:
                r = int(m.row)
                res[r] = (int(torch.argmax(host_logits[r])), host_logits[r])
        return res

    def release_wave_traces(self):
        for inp in self.wave_traces.values():
            ttnn.release_trace(self.device, inp.trace_id)
        self.wave_traces = {}
        self._wave_ready = {}

    def rewind_user(self, user: int, position: int) -> None:
        """Set every layer's chunked-prefill cursor of slot ``user`` to ``position`` (a 32-multiple)."""
        if position < 0 or position % 32:
            raise ValueError(f"rewind position {position} must be a non-negative multiple of 32")
        for dec in self.layers:
            dec.set_next_start(user, int(position))

    def prefill_chunked(self, prompt_ids: torch.Tensor, user: int = 0, chunk: int = 32768, on_chunk=None):
        """A long prompt through ``prefill`` in consecutive chunks (every chunk but the last is a multiple
        of ``32 * rows`` tokens, so the row slabs stay tile-aligned). ``on_chunk(start, end, seconds)`` is
        called after every chunk (synchronised). Returns the LAST chunk's ``(streams, logits_last,
        owner_row)``: its ``owner_row`` / local row index the prompt's last token."""
        S = int(prompt_ids.numel())
        unit = 32 * self.mesh.rows
        chunk = max(unit, (chunk // unit) * unit)
        out = None
        for s0 in range(0, S, chunk):
            e = min(S, s0 + chunk)
            if out is not None:
                ttnn.deallocate(out[0])
                if out[1] is not None:
                    ttnn.deallocate(out[1])
            t0 = time.perf_counter()
            out = self.prefill(prompt_ids[s0:e], user=user, start=s0)
            if on_chunk is not None:
                ttnn.synchronize_device(self.device)
                on_chunk(s0, e, time.perf_counter() - t0)
        return out

    def first_token(self, logits_or_ids, owner_row: int = 0, row: int = 0) -> int:
        """The prompt's first greedy token from the owner row's chip: ``logits_or_ids`` is the eager
        ``prefill``'s last-token logits (argmax taken here), a greedy-id tensor (``row`` of it), or a
        ``PrefillTraceResult`` (its own tile row)."""
        if isinstance(logits_or_ids, PrefillTraceResult):
            res = logits_or_ids
            logits_or_ids, owner_row, row = res.ids, res.owner_row, res.row
        ids = logits_or_ids if logits_or_ids.dtype == ttnn.uint32 else self.head.greedy(logits_or_ids)
        shard = self.mesh.device_tensors(ids)[owner_row * self.mesh.cols]
        return int(ttnn.to_torch(shard).view(torch.int32).reshape(-1)[row])

    # ------------------------------------------------------------- traced prefill
    def trace_prefill_lengths(self, lengths=DEFAULT_LENGTHS) -> list[int]:
        """The traced chunk lengths this model supports (``trace_prefill.supported_prefill_lengths``)."""
        max_rows = -(-self.max_seq_len // 32) * 32
        return supported_prefill_lengths(self.mesh.rows, max_rows, lengths)

    def has_prefill_trace(self, S: int) -> bool:
        return pick_prefill_length(S, self.prefill_traces) is not None

    def _prefill_graph(self, inp: PrefillTraceInputs):
        """The traced prefill graph on the persistent inputs ``inp`` + ``self.prefill_slot`` (no host value
        of the prompt inside): embedding -> every layer (device-slot form) -> head on the selected row."""
        # the pad-row mask (rows >= S out of the MoE router) is a per-length input refreshed per prompt; the knob is
        # read at capture (an eager compile pass precedes it), so the trace replays the mode it was captured in
        call = PrefillTraceCall(
            slot=self.prefill_slot,
            end=inp.end,
            ring_sel=inp.ring_sel,
            row_mask=inp.row_mask if pad_mask_enabled() else None,
        )
        streams = self.embedding(inp.ids)  # [1,1,T_loc,16384] per row
        for dec in self.layers:
            out = dec.prefill_forward(streams, user=self.prefill_slot.user, seq_len=inp.T, start=0, trace=call)
            ttnn.deallocate(streams)
            streams = out
        if self.head is None:
            return streams
        logits = self.head.tile_logits(streams, inp.tile_start, inp.tile_end)  # [1,1,32,V]: the last token's tile
        ids = self.head.greedy(logits)
        # the tile outputs go to buffers that predate every capture (they survive other traces' replays and the
        # capture-created tensors die inside the trace); the streams stay the capture's transient output
        ttnn.copy(logits, inp.out_logits)
        ttnn.copy(ids, inp.out_ids)
        ttnn.deallocate(logits)
        ttnn.deallocate(ids)
        return streams

    def capture_prefill(self, lengths=None, cq_id: int = 0, warmup_user: int = 0) -> list[int]:
        """Allocate the persistent inputs of every requested length (default ``trace_prefill_lengths``),
        compile each graph eagerly (a dummy prompt of the padded length into slot ``warmup_user``), then
        capture one trace per length. Every allocation precedes the first capture; the warm-ups and the
        captures leave dummy data in slot ``warmup_user`` (the next prefill of that slot overwrites it).
        Call it before ``capture`` (the decode trace) or after ``warmup_decode``, so the decode step's
        persistent inputs exist before any capture too. Returns the lengths captured now."""
        if self.prefill_slot is None:
            raise RuntimeError("no layers")
        todo = [
            T for T in (self.trace_prefill_lengths() if lengths is None else lengths) if T not in self.prefill_traces
        ]
        if not todo:
            return []
        vocab = self.cfg.vocab_size if self.head is not None else None
        new = {T: PrefillTraceInputs.allocate(self.mesh, T, self.cfg.hidden_size, vocab) for T in todo}
        self.prefill_slot.set_user(warmup_user)
        for T, inp in new.items():  # program-cache warm-up with the exact shapes and argument values
            t0 = time.perf_counter()
            dummy = torch.arange(T, dtype=torch.long) % 1000 + 1000
            inp.set_prompt(self.mesh, self.embedding.tokens_to_host(dummy, pad_to=T), T)
            ttnn.deallocate(self._prefill_graph(inp))
            ttnn.synchronize_device(self.device)
            logger.info(f"traced prefill: T={T} compiled in {time.perf_counter() - t0:.1f} s")
        for T, inp in new.items():
            t0 = time.perf_counter()
            inp.trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
            inp.streams = self._prefill_graph(inp)
            ttnn.end_trace_capture(self.device, inp.trace_id, cq_id=cq_id)
            self.mesh.allocations_frozen[0] = True  # no lazily allocated constants from here on (tt/kda.py)
            ttnn.synchronize_device(self.device)
            self.prefill_traces[T] = inp
            logger.info(f"traced prefill: T={T} captured in {time.perf_counter() - t0:.1f} s")
        return todo

    def prefill_traced(self, prompt_ids: torch.Tensor, user: int = 0, cq_id: int = 0) -> PrefillTraceResult:
        """One prompt through the captured trace of the smallest supported length >= its token count
        (``ValueError`` when none fits: use ``prefill``). Refreshes the slot / per-length inputs, replays,
        and returns a ``PrefillTraceResult``: the last layer's ``streams`` (``[1,1,T_loc,16384]`` per row,
        transient: read before another replay), the ``logits`` ``[1,1,32,V]`` / greedy ``ids`` ``[1,1,32,1]``
        of the last token's tile (persistent buffers) and ``owner_row`` / ``row`` locating the prompt's
        token (``first_token(result)``)."""
        S = int(prompt_ids.numel())
        T = pick_prefill_length(S, self.prefill_traces)
        if T is None:
            raise ValueError(f"no prefill trace for {S} tokens (captured: {sorted(self.prefill_traces)})")
        inp = self.prefill_traces[T]
        self.prefill_slot.set_user(user)
        inp.set_prompt(self.mesh, self.embedding.tokens_to_host(prompt_ids, pad_to=T), S)
        ttnn.execute_trace(self.device, inp.trace_id, cq_id=cq_id, blocking=True)
        for dec in self.layers:
            dec.set_next_start(user, S)
        return PrefillTraceResult(inp.streams, inp.out_logits, inp.out_ids, inp.owner_row(S), inp.last_row(S), S, T)

    def release_prefill_traces(self):
        for inp in self.prefill_traces.values():
            ttnn.release_trace(self.device, inp.trace_id)
        self.prefill_traces = {}

    def ids_to_host(self, ids: ttnn.Tensor) -> torch.Tensor:
        """greedy ids [1,1,B,1] uint32 per chip -> [rows*B] int64: one chip read per mesh row (the first column;
        the columns hold the same ids after the head's all-gather), not one per chip."""
        t = self.mesh.to_host(ids, rows=2 if self.mesh.rows > 1 else None)
        return t.view(torch.int32).reshape(-1).long()

    # ------------------------------------------------------------------ trace
    def warmup_decode(self, ids: torch.Tensor, host_positions: list[int]):
        """Allocate the persistent decode inputs (once; with ``GLM53_DECODE_PACKED_INPUTS=1`` the packed segments
        too) and run one eager decode step (compiles the programs, the packed unpack included, creates every lazily
        built constant) so that ``capture_prefill`` and ``capture`` can follow in either order with nothing
        allocated after a capture. Consumes the token."""
        self._stage_decode_inputs(ids, host_positions)
        for t in self.decode_step(self.ids_dev, self.positions):
            if t is not None:
                ttnn.deallocate(t)
        ttnn.synchronize_device(self.device)

    def capture(self, ids: torch.Tensor, host_positions: list[int], cq_id: int = 0):
        """Warm-up (compile) + capture one decode step on persistent inputs; returns the trace id and the
        captured output tensors. The warm-up and the capture both consume the token (state advances twice)."""
        self.warmup_decode(ids, host_positions)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
        outs = self.decode_step(self.ids_dev, self.positions)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=cq_id)
        self.mesh.allocations_frozen[0] = True  # no lazily allocated constants from here on (tt/kda.py)
        return trace_id, outs

    def replay(self, trace_id, cq_id: int = 0, blocking: bool = True):
        """Replay a captured trace. ``blocking=True`` (the default, what the served decode uses) returns after the
        device finished it (``FDMeshCommandQueue::enqueue_trace`` -> ``finish``), so ops enqueued afterwards never
        overlap the replay; the served adapter still synchronizes explicitly before a KDA snapshot save that
        follows a replay (``GLM53FlashForCausalLM._snapshot``) so that order does not rest on this default."""
        ttnn.execute_trace(self.device, trace_id, cq_id=cq_id, blocking=blocking)

    def dram_allocated(self) -> int:
        """The DRAM allocator's allocated bytes per bank (a host-side query, no device op): the footprint the served
        adapter fingerprints after the decode capture and re-checks after every prefix-cache call site
        (``GLM53_VLLM_APC_ALLOC_CHECK``); a change means a buffer allocated under live traces persists."""
        mv = ttnn.device.get_memory_view(self.device, ttnn.BufferType.DRAM)
        return int(mv.total_bytes_allocated_per_bank)

    # ------------------------------------------------------------- prefix cache (KDA snapshots)
    def allocate_kda_snapshots(self, entries: int, store: str = "device"):
        """The KDA snapshot store of the served prefix cache (``doc/serving/prefix_caching.md``): ``entries`` per
        user slot in a DRAM ring (``store="device"``: allocated here, before any trace capture) or as host copies
        (``"host"``: no device memory). Returns the store (``save(user, entry)`` / ``load(user, entry)``)."""
        from models.demos.glm53_flash.tt.kda_snapshot import DeviceKdaSnapshotRing, HostKdaSnapshotStore

        if self.kda_snapshots is not None:
            raise RuntimeError("the KDA snapshot store is already allocated")
        if store == "device":
            if self.mesh.allocations_frozen[0]:
                raise RuntimeError("the KDA snapshot ring must be allocated before the first trace capture")
            self.kda_snapshots = DeviceKdaSnapshotRing(self, entries)
        elif store == "host":
            self.kda_snapshots = HostKdaSnapshotStore(self, entries)
        else:
            raise ValueError(f"snapshot store {store!r}: expected 'device' or 'host'")
        logger.info(
            f"mesh model: KDA snapshot store {store}, {entries} entries per user slot ({self.max_batch} slots per row)"
        )
        return self.kda_snapshots

    def snapshot_kda(self, user: int, entry: int) -> None:
        """Copy slot ``user``'s live KDA carries into snapshot ``entry`` (between replays, never traced)."""
        if self.kda_snapshots is None:
            raise RuntimeError("allocate_kda_snapshots first")
        self.kda_snapshots.save(user, entry)

    def restore_kda(self, user: int, entry: int, position: int) -> None:
        """Snapshot ``entry`` -> slot ``user``'s live carries, and re-arm the chunked prefill at ``position`` (the
        next chunk of that slot starts there: ``FunctionalDecoder.next_start``)."""
        if self.kda_snapshots is None:
            raise RuntimeError("allocate_kda_snapshots first")
        self.kda_snapshots.load(user, entry)
        self.rewind_user(user, position)

    def rewind_user(self, user: int, position: int) -> None:
        """Set every layer's chunked-prefill cursor of slot ``user`` to ``position`` (a 32-multiple)."""
        if position < 0 or position % 32:
            raise ValueError(f"rewind position {position} must be a non-negative multiple of 32")
        for dec in self.layers:
            dec.next_start[user] = int(position)

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

    # ------------------------------------------------------------- MTP draft step (speculative decoding, step 3)
    def head_forward_with_hidden(self, streams: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """``TtModelHead.forward_greedy`` that also hands back the normed hidden: packed streams (or the last
        layer's collapsed mean) ``[1,1,T,*]`` -> ``(hn [1,1,T,4096] bf16, logits [1,1,T,V] bf16, ids [1,1,T,1]
        uint32)``. ``hn`` is the LM-head input = the MTP's ``H`` row (``tt/mtp.py``); the caller deallocates all
        three. Trace-capturable (device ops only)."""
        h = self.head.collapse(streams)
        hn = self.head.norm(h)
        if h is not streams:
            ttnn.deallocate(h)
        logits = self.head.logits(hn)
        ids = self.head.greedy(logits)
        return hn, logits, ids

    def attach_mtp(self, mtp):
        """Attach the MTP drafter (a ``TtMTPBlock`` built with this model's embedding / head, ``max_batch`` user
        slots per row and a cache of at least ``max_seq_len`` positions). Allocate it, like the layers, before
        any trace capture."""
        if mtp.max_batch != self.max_batch or mtp.mesh.rows != self.mesh.rows:
            raise ValueError(f"MTP block ({mtp.max_batch} users/row) does not match the model ({self.max_batch})")
        if mtp.max_seq_len < self.max_seq_len:
            raise ValueError(f"MTP cache holds {mtp.max_seq_len} positions, the model decodes up to {self.max_seq_len}")
        self.mtp = mtp
        self.mtp_positions = None

    def set_mtp_positions(self, host_positions: list[int]):
        """The MTP's level-1 positions of the step: the same values as ``set_positions`` (the level-1 row of main
        position ``p`` sits at MTP position ``p``), held in the MTP's own persistent ``DecodePositions``."""
        if self.mtp is None:
            raise RuntimeError("attach_mtp first")
        if self.mtp_positions is None:
            self.mtp_positions = self.mtp.make_decode_positions(host_positions)
        else:
            self.mtp.update_decode_positions(self.mtp_positions, host_positions)

    def set_spec_positions(self, host_positions: list[int]):
        """``set_positions`` + ``set_mtp_positions`` with the same values (one call per step)."""
        self.set_positions(host_positions)
        self.set_mtp_positions(host_positions)

    def decode_step_with_draft(self, ids_dev: ttnn.Tensor, positions: DecodePositions, mtp_positions: DecodePositions):
        """``decode_step`` followed by MTP level 1: the normal step on ``ids_dev`` ``[1,1,1,B]`` (per row) at
        ``positions``, then the drafter on the head's normed hidden and the step's greedy id (re-embedded on
        device from the argmax output) at ``mtp_positions`` (= the same positions). Returns ``(logits [1,1,B,V],
        ids [1,1,B,1], draft ids [1,1,B,1], hn [1,1,B,4096] bf16)``: the step's greedy token per user slot and
        the MTP's prediction of the token after it. Every op is a device op on persistent inputs (trace-safe;
        ``capture_with_draft``). ``hn`` is returned for diagnostics (the eager MTP re-run of a test)."""
        if self.mtp is None:
            raise RuntimeError("attach_mtp first")
        self._unpack_inputs(ids_dev, positions)  # the main step's inputs; the MTP positions stay direct uploads
        streams = self.embedding(ids_dev)
        last = len(self.layers) - 1
        for k, dec in enumerate(self.layers):
            collapse = k == last and dec.supports_collapse
            out = dec.decode_forward(streams, positions=positions, collapse=collapse)
            ttnn.deallocate(streams)
            streams = out
        hn, logits, ids = self.head_forward_with_hidden(streams)
        ttnn.deallocate(streams)
        # the id row is a view of ``ids`` (never deallocated); ``project`` reads it and ``hn`` without consuming
        out, normed, mlogits, draft = self.mtp.decode_forward(self.mtp.ids_row(ids), hn, mtp_positions)
        for t in (out, normed, mlogits):
            ttnn.deallocate(t)
        return logits, ids, draft, hn

    def warmup_decode_with_draft(self, ids: torch.Tensor, host_positions: list[int]):
        """``warmup_decode`` for the draft step: allocates the persistent inputs (ids, both position sets; the
        packed segments with the knob) and runs one eager step (compiles the programs, builds the lazy constants).
        Consumes the token."""
        self._stage_decode_inputs(ids, host_positions)
        self.set_mtp_positions(host_positions)
        for t in self.decode_step_with_draft(self.ids_dev, self.positions, self.mtp_positions):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.device)

    def capture_with_draft(self, ids: torch.Tensor, host_positions: list[int], cq_id: int = 0):
        """Warm-up + capture of ``decode_step_with_draft`` on the persistent inputs (``capture`` with the drafter):
        returns ``(trace id, (logits, ids, draft ids, hn))``, transient outputs read after each ``replay``. Both
        passes consume the token (the KDA states advance twice; the MTP cache row at the position is rewritten)."""
        self.warmup_decode_with_draft(ids, host_positions)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
        outs = self.decode_step_with_draft(self.ids_dev, self.positions, self.mtp_positions)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=cq_id)
        self.mesh.allocations_frozen[0] = True  # no lazily allocated constants from here on (tt/kda.py)
        return trace_id, outs
