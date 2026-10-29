# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional decoder contract for GLM-5.3-Flash (tt-model-bringup stage 1, single device).

One ``FunctionalDecoder`` is one decoder layer (``layer_idx``) of any of the three layer kinds
(KDA + dense MLP: layers 0-2; DSA + MoE: 3+4k; KDA + MoE: the rest) with its device-owned
autoregressive state:

* KDA layers: a persistent ``[max_batch, ...]`` ``KdaState`` (recurrent ``[B,64,128,128]`` fp32 +
  conv history ``[B,3,3*8192]`` bf16); prefill writes slot ``user``, decode updates all slots in place;
* DSA layers: a paged latent cache ``[max_blocks, 1, 32, 512]`` bf16 shared by ``max_batch`` users
  (``[max_batch, blocks_per_seq]`` page table, any unique block mapping) and the pooled-key index
  (``TtIndexCache``), both maintained on every prefill/decode call.

Tensor contract (the HF residual is the mHC stream tensor ``[B, S, 4, 4096]``):
* ``prefill_forward(streams, user, seq_len, start=0)``: one user, one chunk; ``streams`` is the
  packed fp32 device tensor ``[1, 1, T_pad, 4*4096]`` (stream i at columns [i*4096, (i+1)*4096),
  T_pad % 32 == 0, rows past ``seq_len`` are padding and come back unspecified). Row i is the
  token at absolute position ``start + i``. Returns the packed output streams.
  Chunked prefill: a prompt may be prefilled in consecutive chunks. ``start == 0`` begins a new
  prompt (the user's KDA state is reset / the DSA cache is refilled from 0); every later chunk
  must satisfy ``start % 32 == 0``, ``start == next_start[user]`` (the previous chunk's
  ``start + seq_len``, so chunks are contiguous and a non-32-aligned ``seq_len`` is only possible
  in the last chunk of a prompt) and ``start + seq_len <= max_seq_len``. The chunk continues from
  the device-owned state: KDA reads the user's recurrent + conv carries out of the batched state
  and writes the new ones back; DSA writes the chunk's latent into the user's pages from block
  ``start // 32``, maintains the pooled-key index from position ``start`` (the ring holds the
  previous chunk's tail rows) and attends every chunk row over the full prefix [0, start + T_pad):
  rows at absolute positions < 2048 with the dense causal SDPA (K/V from the cached latent of
  [0, start) plus the chunk's), rows from 2048 on with the indexer + sparse_sdpa, so the result
  equals the single-shot prefill of the whole prompt (same kernels, same row classes).
* ``decode_forward(streams, positions)``: ``streams`` ``[1, 1, B, 4*4096]`` fp32, one token for
  each of the B == max_batch user slots (user b = row b); ``positions`` is a ``DecodePositions``
  built by ``make_decode_positions(host_positions)`` and refreshed in place by
  ``update_decode_positions`` (persistent ``[B]`` int32 device tensors: the position, position // 4
  and position % 4, plus the host values). Returns ``[1, 1, B, 4*4096]``.
  DSA layers append the latent, maintain the pooled-key index and run the dense flash-MLA decode
  for EVERY user slot (HF-exact through position 2050 = index_topk + index_kpool - 2, trace-safe);
  the rows of the users whose host position is > 2050 are then overwritten with the per-user
  indexer + sparse_sdpa result (host chunk geometry, not traced). Mixed batches therefore stay
  exact for the short users; the routing is per user, never a batch-wide switch.
* ``capture_decode_trace`` / ``replay_decode``: the dense decode graph is capturable; the captured
  graph reads the persistent position tensors and updates the state and the pooled-key index in
  place. Capturing a DSA layer whose ``max_seq_len`` exceeds 2051 is refused unless the caller opts
  in (``allow_sparse_range``), because a replay at a position > 2050 would attend densely, which is
  not HF semantics there (the sparse rows are never part of the captured graph).
* ``release_weights`` / ``attach_layer``: weight streaming. ``release_weights`` deallocates every
  device tensor the layer module holds (weights, folded constants, cached geometry scalars) and
  detaches it, while the decoder keeps the autoregressive state (KDA slots / DSA latent cache +
  pooled-key index) and ``next_start``; ``attach_layer`` re-arms it with a freshly built
  ``TtDecoderLayer`` of the same layer. ``tt/model.py`` runs all 45 layers through one chip this
  way (build -> run -> release per layer, the states persist across the passes).
Use ``host_streams_to_device`` / ``device_streams_to_host`` at the test boundary only.

Multi-chip (Phase 2, ``tt/mesh.py``): on a ``(rows, cols)`` mesh the columns are the tensor-parallel
group of a row and each row decodes its own ``max_batch`` user slots (data parallel), so a decode
step carries ``rows * max_batch`` users: ``streams`` ``[1, 1, rows*B, 4*4096]`` on the host is
row-sharded (row r holds users ``[r*B, (r+1)*B)``), ``host_positions`` lists all ``rows * B``
positions in that order, and every per-row chip returns the same (replicated) output. Prefill and
the sparse DSA decode path are single-row only for now (ttKDA is sequence-parallel over the rows,
the sparse chunk geometry is per user); a multi-row decoder refuses them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.decoder_layer import LayerState, TtDecoderLayer, WaveInfo
from models.demos.glm53_flash.tt.dsa import (
    SparseDecodeMeta,
    TtLatentCache,
    decode_position_tensors,
    sparse_decode_enabled,
)
from models.demos.glm53_flash.tt.mesh import GLMMesh

MEM_LOG_LAYERS_ENV = "GLM53_MEM_LOG_LAYERS"  # "1": log the DRAM allocator after every layer's prefill chunk

TILE = 32

# objects a weight walk must not enter: devices / meshes, and the per-mesh shared KDA history
# exchange (its buffer and semaphores outlive every layer)
_WALK_SKIP_TYPES = ("_HistoryExchange", "GlobalSemaphore", "MeshDevice", "Device", "GLMMesh")


def device_tensors_of(obj, *, _seen: set | None = None, _out: list | None = None, _depth: int = 0) -> list[ttnn.Tensor]:
    """Every ``ttnn.Tensor`` reachable from ``obj`` through attributes of the model's own objects
    (``models.*`` classes: modules, dataclasses) and plain containers, deduplicated. Used by
    ``FunctionalDecoder.release_weights``; the shared / device-level objects are not entered."""
    seen = set() if _seen is None else _seen
    out = [] if _out is None else _out
    if id(obj) in seen or _depth > 16:
        return out
    seen.add(id(obj))
    if isinstance(obj, ttnn.Tensor):
        out.append(obj)
    elif type(obj).__name__ in _WALK_SKIP_TYPES:
        pass
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            device_tensors_of(v, _seen=seen, _out=out, _depth=_depth + 1)
    elif isinstance(obj, dict):
        for v in obj.values():
            device_tensors_of(v, _seen=seen, _out=out, _depth=_depth + 1)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("models."):
        for v in vars(obj).values():
            device_tensors_of(v, _seen=seen, _out=out, _depth=_depth + 1)
    return out


def host_streams_to_device(
    streams: torch.Tensor, device, pad_to: int | None = None, mesh: GLMMesh | None = None
) -> ttnn.Tensor:
    """HF ``[1, S, n, C]`` (any dtype) -> packed ``[1, 1, T_pad, n*C]`` fp32 on device. With a
    multi-row ``mesh`` the rows (= ``rows * B`` decode users) are sharded over the mesh rows."""
    _, S, n, C = streams.shape
    unit = TILE * (mesh.rows if mesh is not None else 1)  # a multi-row prefill chunk splits into row slabs
    T_pad = pad_to if pad_to is not None else -(-S // unit) * unit
    packed = torch.zeros(1, 1, T_pad, n * C, dtype=torch.float32)
    packed[0, 0, :S] = streams.reshape(S, n * C).float()
    if mesh is not None and mesh.rows > 1:
        return mesh.upload(packed, ttnn.float32, rows=2)
    return ttnn.from_torch(packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def device_streams_to_host(
    packed: ttnn.Tensor, n: int, seq_len: int | None = None, mesh: GLMMesh | None = None
) -> torch.Tensor:
    """Packed ``[1, 1, T, n*C]`` -> HF ``[1, S, n, C]`` fp32 (S = seq_len rows). With a multi-row
    ``mesh`` the rows' shards are concatenated (first column of each row)."""
    if mesh is not None and not mesh.is_single:
        t = mesh.to_host(packed, rows=2 if mesh.rows > 1 else None).float()[0, 0]
    else:
        t = ttnn.to_torch(packed).float()[0, 0]
    T, nC = t.shape
    S = T if seq_len is None else seq_len
    return t[:S].reshape(1, S, n, nC // n)


@dataclass
class DecodePositions:
    """Persistent decode-step position inputs (one entry per user slot)."""

    positions: ttnn.Tensor  # [B] int32: position of the token being decoded (= context length)
    pool_ids: ttnn.Tensor  # [B] int32: position // index_kpool
    ring_slots: ttnn.Tensor  # [B] int32: position % index_kpool
    host: list[int]  # the same values on the host (chunk geometry for the sparse path)
    # [B] int32: min(position, 2050), the bound of the dense flash-MLA decode (DSA layers). A user beyond
    # 2050 gets the sparse row instead, so its dense row is discarded: bounding the op there keeps its cost
    # at <= 2051 keys instead of the whole prefix (1 GiB per layer at 1 M positions); identity otherwise.
    dense_positions: ttnn.Tensor | None = None
    # the traced multi-user sparse decode's per-user metadata (``SparseDecodeMeta``): with it a DSA layer
    # attends every user, dense or sparse, inside one trace-safe graph (default); ``GLM53_SPARSE_DECODE=eager``
    # keeps the dense flash-MLA + per-user eager sparse rows instead (A/B reference)
    sparse: SparseDecodeMeta | None = None


@dataclass
class PrefillTraceCall:
    """The device-side inputs of one traced prefill chunk (``tt/trace_prefill.py``): the user slot
    (``PrefillSlot``), the valid end ``[1,1,1,1]`` uint32 (ttKDA ``actual_end`` = the prompt length) and
    the ring-row selector ``[1,1,32,T]`` bf16 of the pooled-key index."""

    slot: object
    end: ttnn.Tensor
    ring_sel: ttnn.Tensor
    # [1,1,T,32] bf16 row-sharded: column 0 = 1.0 on rows < S (the MoE router's pad-row mask, ``tt/moe.py``
    # ``pad_row_mask_host``); None under GLM53_ROUTER_PAD_MASK=0
    row_mask: ttnn.Tensor | None = None


class FunctionalDecoder(LightweightModule):
    def __init__(
        self,
        layer: TtDecoderLayer,
        *,
        max_batch: int,
        max_seq_len: int,
        block_size: int = TILE,
        partition_users: bool | None = None,
        prefill_partition: str | None = None,
    ):
        """``partition_users``: the user-partitioned DSA cache on a multi-column mesh (column c of a row owns users
        ``[c*B/cols, (c+1)*B/cols)``; ``max_batch`` % cols == 0; decode through the traced sparse path only).
        ``None`` = the ``GLM53_PARTITION_USERS`` knob (default off). ``prefill_partition``: how that cache serves a
        prefill, ``scratch`` | ``heads2users`` (``None`` = the ``GLM53_PREFILL_PARTITION`` knob, default
        ``scratch``; ``TtLatentCache``)."""
        if partition_users is None:
            partition_users = os.environ.get("GLM53_PARTITION_USERS", "0") == "1"
        self.layer: TtDecoderLayer | None = layer
        self.cfg = layer.cfg
        self.device = layer.device
        self.mesh: GLMMesh = layer.mesh
        self.layer_idx = layer.layer_idx
        self.is_kda = layer.is_kda
        self.is_moe = layer.is_moe
        self.max_batch = max_batch  # user slots per mesh row
        self.total_users = max_batch * self.mesh.rows
        self.max_seq_len = max_seq_len
        self.cache = None
        self.partition = bool(partition_users) and self.mesh.cols > 1
        # GLM53_CACHES_LAST=1 (placement experiment, 2026-10-27): the DSA cache is NOT allocated here but by
        # ``allocate_cache`` once every layer's weights are resident (mesh_model), so the bank-sharded expert weights of
        # all 45 layers land contiguously below the caches instead of interleaving with eleven ~1 GiB caches at 1 M
        self._cache_args = dict(
            max_batch=max_batch,
            max_seq_len=max_seq_len,
            block_size=block_size,
            partition_users=partition_users,
            prefill_partition=prefill_partition,
        )
        if not layer.is_kda and os.environ.get("GLM53_CACHES_LAST", "0") != "1":
            self.allocate_cache()
        # persistent batched KDA state (max_batch slots): prefill writes slot `user`, decode reads
        # and updates all slots in place, so captured decode traces always see the live state.
        self.kda_batched: KdaState | None = None
        if layer.is_kda:
            self.kda_batched = layer.attn.allocate_state(max_batch)
        # chunked prefill bookkeeping: the position the next chunk of each user must start at. Under the sequence-
        # parallel prefill slot ``user`` of every row holds the prompt (one cursor); a wave (``prefill_wave_forward``)
        # advances row r's slot ``user`` alone, so the per-row cursors are the truth after one.
        self.next_start: list[int] = [0] * max_batch
        self.next_start_rows: list[list[int]] = [[0] * max_batch for _ in range(self.mesh.rows)]

    def allocate_cache(self) -> None:
        """Allocate this DSA layer's latent cache (``TtLatentCache``); a no-op for KDA layers or when it exists."""
        if self.is_kda or self.cache is not None:
            return
        a = self._cache_args
        self.cache = TtLatentCache(
            self.device,
            self.cfg,
            max_batch=a["max_batch"],
            max_seq_len=a["max_seq_len"],
            block_size=a["block_size"],
            mesh=self.mesh,
            partition_users=a["partition_users"],
            prefill_partition=a["prefill_partition"],
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        max_batch: int = 1,
        max_seq_len: int = 4096,
        weight_cache_path: Path | None = None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        **kwargs,
    ) -> "FunctionalDecoder":
        """``state_dict``: this layer's raw checkpoint tensors (prefix stripped, fp8 dequantized).
        ``hf_config``: a ``GLM53FlashConfig``, or a path to the checkpoint ``config.json``.
        ``expert_dtype`` / ``expert_down_dtype``: routed-expert gate/up and down weight formats
        (``bfloat8_b`` default; ``bfloat4_b`` is the design's accuracy-gated candidate for gate/up)."""
        cfg = hf_config if isinstance(hf_config, GLM53FlashConfig) else GLM53FlashConfig.from_hf_config_json(hf_config)
        layer = TtDecoderLayer(
            mesh_device,
            cfg,
            layer_idx,
            state_dict,
            weight_cache_path=weight_cache_path,
            weight_dtype=weight_dtype,
            expert_dtype=expert_dtype,
            expert_down_dtype=expert_down_dtype,
            mesh=GLMMesh.from_device(mesh_device),
        )
        return cls(layer, max_batch=max_batch, max_seq_len=max_seq_len, **kwargs)

    # ------------------------------------------------------- weight streaming
    @property
    def weights_attached(self) -> bool:
        return self.layer is not None

    def release_weights(self) -> int:
        """Deallocate every device tensor of the attached layer module and detach it (the KDA slots /
        DSA cache / index and ``next_start`` stay). Returns the number of buffers released. The
        decoder refuses forwards until ``attach_layer``."""
        if self.layer is None:
            return 0
        n = 0
        for t in device_tensors_of(self.layer):
            if t.is_allocated():
                ttnn.deallocate(t)
                n += 1
        self.layer = None
        return n

    def attach_layer(self, layer: TtDecoderLayer):
        """Re-arm the decoder with freshly built weights of the same layer (after ``release_weights``)."""
        if self.layer is not None:
            raise ValueError("release_weights first")
        if layer.layer_idx != self.layer_idx or layer.is_kda != self.is_kda or layer.is_moe != self.is_moe:
            raise ValueError(f"layer {layer.layer_idx} does not match this decoder (layer {self.layer_idx})")
        if layer.device is not self.device:
            raise ValueError("the new layer must live on the decoder's device")
        self.layer = layer

    def _require_weights(self) -> TtDecoderLayer:
        if self.layer is None:
            raise RuntimeError(f"layer {self.layer_idx}: weights released; attach_layer before a forward")
        return self.layer

    # ------------------------------------------------------------------ state
    def _state_for_prefill(self, user: int, start: int = 0) -> tuple[LayerState, list[ttnn.Tensor]]:
        """The layer state a prefill chunk of ``user`` starts from, plus the temporaries to release
        after the chunk. KDA: a zero state for a new prompt (start == 0), otherwise the user's
        current carries read out of the persistent batched state (slot ``user``)."""
        if not self.is_kda:
            return LayerState(cache=self.cache), []
        if start == 0:
            return LayerState(kda=self._require_weights().attn.allocate_state(1)), []
        src = self.kda_batched
        if self.max_batch == 1:  # ttKDA only reads its input state; the new carries are fresh tensors
            return LayerState(kda=src), []
        rec = self._user_recurrent(src.recurrent, user)
        conv = self._require_weights().attn.user_conv_state(src, user)  # ttKDA's [1, taps, W] row-major
        return LayerState(kda=KdaState(recurrent=rec, convolution=conv)), [rec, conv]

    @staticmethod
    def _user_recurrent(batched: ttnn.Tensor, user: int) -> ttnn.Tensor:
        """Slot ``user`` of the batched recurrent state ``[B, H, K, V]`` as a fresh ``[1, H, K, V]`` tensor through the
        tensor-argument ``ttnn.slice`` (the program keys on the shapes, the slot index is a runtime tensor): one
        program whatever ``user`` is. The host-int slice was a program per user index, which the first continued
        chunk of a slot the warm-up had not used compiled under the sealed cache (2026-10-28 served APC, 8x4)."""
        B, H, K, V = (int(v) for v in batched.shape)
        dev = batched.device()
        rm = lambda v: ttnn.from_torch(
            torch.tensor(v, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )
        start, end = rm([user, 0, 0, 0]), rm([user + 1, H, K, V])
        out = ttnn.slice(input_tensor=batched, starts=start, ends=end, slice_dim=0, num_devices=B)
        ttnn.deallocate(start)
        ttnn.deallocate(end)
        return out

    def _write_user_kda_state_slot(self, slot, new: KdaState):
        """``_write_user_kda_state`` with the slot chosen on device (the traced prefill): the recurrent
        state is a whole-tile copy through ``paged_fill_cache`` on the ``[B*H*K/32, 1, 32, V]`` view of
        the batched state (``slot.rec_page_table`` = identity, ``slot.user_idx`` = the batch index), the
        tap-major conv history is blended with the one-hot user masks (``dst * keep + new * mask``:
        every product is x * 1.0 or x * 0.0, exact in bf16). B == 1 keeps the plain copies."""
        dst = self.kda_batched
        attn = self._require_weights().attn
        if self.max_batch == 1:
            ttnn.copy(new.recurrent, dst.recurrent)
            ttnn.copy(new.convolution, dst.convolution)
        else:
            if not attn.conv_tiled:
                raise NotImplementedError("the traced prefill needs the block KDA decode (tiled conv history)")
            B, H, K, V = (int(v) for v in dst.recurrent.shape)
            dst_view = ttnn.experimental.view(dst.recurrent, ttnn.Shape([B * H * K // TILE, 1, TILE, V]))
            new_view = ttnn.experimental.view(new.recurrent, ttnn.Shape([1, 1, H * K, V]))
            ttnn.experimental.paged_fill_cache(dst_view, new_view, slot.rec_page_table, batch_idx_tensor=slot.user_idx)
            ttnn.multiply(dst.convolution, slot.conv_keep, output_tensor=dst.convolution)  # zero the slot's rows
            new_rows = ttnn.multiply(new.convolution, slot.conv_mask)  # [taps,1,W] x [taps,B,1] -> [taps,B,W]
            ttnn.add(dst.convolution, new_rows, output_tensor=dst.convolution)
            ttnn.deallocate(new_rows)
        ttnn.deallocate(new.recurrent)
        ttnn.deallocate(new.convolution)

    def _write_user_kda_state(self, user: int, new: KdaState):
        """Copy a freshly prefilled single-user state into slot ``user`` of the persistent batched state."""
        dst = self.kda_batched
        attn = self._require_weights().attn
        if self.max_batch == 1:
            ttnn.copy(new.recurrent, dst.recurrent)
            ttnn.copy(new.convolution, dst.convolution)
        else:
            if attn.conv_tiled:  # the tap-major tile history: a paged row write per tap
                attn.write_user_conv_state(dst, user, new.convolution)
                pairs = ((new.recurrent, dst.recurrent),)
            else:
                pairs = ((new.recurrent, dst.recurrent), (new.convolution, dst.convolution))
            for src_t, dst_t in pairs:
                rows = []
                for u in range(self.max_batch):
                    if u == user:
                        rows.append(src_t)
                    else:
                        end = list(dst_t.shape)
                        start = [0] * len(end)
                        start[0], end[0] = u, u + 1
                        rows.append(ttnn.slice(dst_t, start, end))
                stacked = ttnn.concat(rows, dim=0)
                ttnn.copy(stacked, dst_t)
                ttnn.deallocate(stacked)
        ttnn.deallocate(new.recurrent)
        ttnn.deallocate(new.convolution)

    def _state_for_decode(self, batch: int) -> LayerState:
        if batch != self.max_batch:
            raise ValueError(f"decode batch {batch} must equal max_batch {self.max_batch} (fixed user slots)")
        if not self.is_kda:
            return LayerState(cache=self.cache)
        return LayerState(kda=self.kda_batched)

    # -------------------------------------------------------------- positions
    def make_decode_positions(self, host_positions: list[int]) -> DecodePositions:
        """Upload the three persistent [B] int32 tensors for ``host_positions`` (one per user slot;
        ``rows * max_batch`` entries on a multi-row mesh, row-sharded)."""
        if len(host_positions) != self.total_users:
            raise ValueError(f"{len(host_positions)} positions for {self.total_users} user slots")
        pos, pool_ids, ring_slots = decode_position_tensors(
            self.device, host_positions, self.cfg.index_kpool, mesh=self.mesh, partition=self.partition
        )
        dense = None
        if not self.layer.is_kda and not self.partition:
            dense = decode_position_tensors(
                self.device, self._dense_bound(host_positions), self.cfg.index_kpool, mesh=self.mesh
            )[0]
        # the sparse-decode metadata is built for every layer kind: one DecodePositions serves all layers of
        # a model (mesh_model builds it from layer 0, a KDA layer) and the DSA layers need it
        sparse = SparseDecodeMeta.from_positions(
            self.device, host_positions, self.cfg, mesh=self.mesh, partition=self.partition
        )
        return DecodePositions(pos, pool_ids, ring_slots, list(host_positions), dense, sparse)

    def _dense_bound(self, host_positions: list[int]) -> list[int]:
        return [min(p, self.layer.attn.dense_exact_max_pos) for p in host_positions]

    def update_decode_positions(self, dp: DecodePositions, host_positions: list[int], *, upload: bool = True):
        """Refresh the persistent position tensors in place (trace-replay safe): ``3 + users_per_chip`` (+1 with
        ``dense_positions``, +1 per sparse tensor) ``copy_host_to_device_tensor`` calls. ``upload=False`` keeps the
        host bookkeeping only (``dp.host``, the sparse regime refresh / guard): the packed decode inputs
        (``GLM53_DECODE_PACKED_INPUTS``, ``GLM53MeshModel.load_decode_inputs``) carry the device values."""
        if len(host_positions) != self.total_users:
            raise ValueError(f"{len(host_positions)} positions for {self.total_users} user slots")
        P = self.cfg.index_kpool
        from models.demos.glm53_flash.tt.dsa import _meta_mapper

        mapper = _meta_mapper(self.mesh, self.partition)
        updates = [
            (dp.positions, host_positions),
            (dp.pool_ids, [p // P for p in host_positions]),
            (dp.ring_slots, [p % P for p in host_positions]),
        ]
        if dp.dense_positions is not None:
            updates.append((dp.dense_positions, self._dense_bound(host_positions)))
        if dp.sparse is not None:
            dp.sparse.update(host_positions, self.cfg, mesh=self.mesh, partition=self.partition, upload=upload)
        if not upload:
            dp.host = list(host_positions)
            return
        for t, vals in updates:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor(vals, dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mapper,
                ),
                t,
            )
        dp.host = list(host_positions)

    # ---------------------------------------------------------------- forwards
    def prefill_forward(
        self,
        streams: ttnn.Tensor,
        *,
        user: int = 0,
        seq_len: int | None = None,
        start: int = 0,
        trace: PrefillTraceCall | None = None,
    ) -> ttnn.Tensor:
        """One chunk of one user's prompt: row i of ``streams`` is the token at absolute position
        ``start + i``. ``start == 0`` starts a new prompt; a later chunk must start where the previous
        one ended (``next_start[user]``), on a 32-row boundary, and end within ``max_seq_len`` (see
        the module docstring for the full chunked-prefill contract).
        ``trace`` (``PrefillTraceCall``): the trace-replayable form of a new prompt (start == 0): the
        user slot, the prompt length and the ring rows come from persistent device tensors, so the
        graph is the same for every prompt of the padded length ``T`` (``seq_len`` is then ``T``)."""
        T = streams.shape[-2] * self.mesh.rows  # multi-row mesh: streams hold this row's slab of the chunk
        seq_len = T if seq_len is None else seq_len
        if trace is not None:
            if start or seq_len != T:
                raise ValueError("a traced prefill chunk starts a new prompt (start == 0) at its padded length")
            if user >= self.max_batch:
                raise ValueError(f"user {user} >= max_batch {self.max_batch}")
            return self._prefill_forward_traced(streams, trace, user, T)
        if T % TILE or seq_len > T or seq_len <= 0:
            raise ValueError(f"T_pad={T} must be a multiple of {TILE} and 0 < seq_len={seq_len} <= T_pad")
        if user >= self.max_batch:
            raise ValueError(f"user {user} >= max_batch {self.max_batch}")
        if start < 0 or start % TILE:
            raise ValueError(f"chunk start {start} must be a non-negative multiple of {TILE}")
        if start + seq_len > self.max_seq_len:
            raise ValueError(f"start {start} + seq_len {seq_len} > max_seq_len {self.max_seq_len}")
        if start and start != self.next_start[user]:
            raise ValueError(
                f"user {user}: chunk start {start} is not contiguous with the previous chunk "
                f"(expected {self.next_start[user]}; a chunk with a non-{TILE}-aligned seq_len must be the last)"
            )
        # multi-row mesh (sequence-parallel prefill v1): the rows hold consecutive slabs of the chunk; the
        # KDA carries come back replicated over the rows and the DSA cache / index are written on every
        # row, so slot ``user`` of EVERY row ends up holding this prompt (a per-row ownership mask is the
        # next step; until then a prefilled user is decodable from any row's slot ``user``).
        layer = self._require_weights()
        state, temporaries = self._state_for_prefill(user, start)
        out, new_state = layer.prefill_forward(streams, state, user=user, seq_len=seq_len, start=start)
        for t in temporaries:
            ttnn.deallocate(t)
        if self.is_kda:
            self._write_user_kda_state(user, new_state.kda)
        self.set_next_start(user, start + seq_len)
        if os.environ.get(MEM_LOG_LAYERS_ENV, "0") == "1":  # which layer grows the resident DRAM per chunk (2026-10-27)
            mv = ttnn.device.get_memory_view(self.mesh.device, ttnn.BufferType.DRAM)
            logger.info(
                f"GLM53 mem/layer {'KDA' if self.is_kda else 'DSA'} chunk [{start}, {start + seq_len}) user {user}: "
                f"allocated {mv.total_bytes_allocated_per_bank / 2**20:.0f} MiB/bank, free "
                f"{getattr(mv, 'total_bytes_free_per_bank', 0) / 2**20:.0f}, largest "
                f"{getattr(mv, 'largest_contiguous_bytes_free_per_bank', 0) / 2**20:.0f}"
            )
        return out

    def _prefill_forward_traced(self, streams: ttnn.Tensor, trace: PrefillTraceCall, user: int, T: int) -> ttnn.Tensor:
        """The device-slot form of a new-prompt chunk (no host value of the call enters the graph):
        KDA starts from the slot's read-only zero state, runs to ``trace.end`` (ttKDA clips its
        chronology there) and writes the carries into the slot chosen on device; DSA fills the latent
        pages / index of that slot and selects the ring rows with ``trace.ring_sel``. ``user`` is the
        host-side bookkeeping only (the slot the caller refreshed ``trace.slot`` for)."""
        layer = self._require_weights()
        slot = trace.slot
        if self.is_kda:
            out, new_state = layer.prefill_forward(
                streams,
                LayerState(kda=slot.kda_zero),
                user=user,
                seq_len=T,
                start=0,
                actual_start=layer.attn.scalar(0),
                actual_end=trace.end,
                row_mask=trace.row_mask,
            )
            self._write_user_kda_state_slot(slot, new_state.kda)
        else:
            out, _ = layer.prefill_forward(
                streams,
                LayerState(cache=self.cache),
                user=user,
                seq_len=T,
                start=0,
                slot=slot,
                ring_sel=trace.ring_sel,
                row_mask=trace.row_mask,
            )
        self.set_next_start(user, T)  # the caller (GLM53MeshModel.prefill_traced) sets the prompt's real length
        return out

    def set_next_start(self, user: int, position: int) -> None:
        """Slot ``user`` holds (the same prompt) up to ``position`` on EVERY row: the sequence-parallel prefill's and
        the traced prefill's bookkeeping (and ``GLM53MeshModel.rewind_user``)."""
        self.next_start[user] = int(position)
        for r in range(self.mesh.rows):
            self.next_start_rows[r][user] = int(position)

    # ----------------------------------------------------------------- waves
    def _row_flags(self, values: tuple[float, ...], dtype, rank: int = 4) -> ttnn.Tensor:
        """A ``rank``-D all-ones-shaped tensor of ``dtype`` on every chip of row r holding ``values[r]`` (a per-row
        0.0 / 1.0 select factor broadcast over a tensor of that rank: every product is x * 1.0 or x * 0.0, exact)."""
        t = torch.tensor(values, dtype=torch.float32).reshape(self.mesh.rows, *([1] * (rank - 1)))
        return self.mesh.upload(t, dtype, rows=0)

    def _wave_state(self, user: int, wave: WaveInfo):
        """The KDA state a wave's chunk of slot ``user`` starts from on every row: the row's own carry (slot ``user``
        of ITS batched state) when the row continues a prompt (``start > 0``), zeros when it starts one (and on idle
        rows, whose scan is discarded). Returns ``(state, temporaries, (current recurrent, current conv tiles))``;
        the current tensors are what an idle / new-prompt row writes back (``_write_wave_kda_state``)."""
        attn = self._require_weights().attn
        if not attn.conv_tiled:
            raise NotImplementedError("waves need the block KDA decode (tiled conv history)")
        src = self.kda_batched
        if self.max_batch == 1:
            cur_rec, cur_conv = src.recurrent, src.convolution  # [1,H,K,V] fp32, [taps,1,W] bf16 tiles
            fresh = []
        else:
            cur_rec = self._user_recurrent(src.recurrent, user)
            cur_conv = attn.user_conv_tiles(src, user)
            fresh = [cur_rec, cur_conv]
        cont = tuple(1.0 if (v > 0 and s > 0) else 0.0 for s, v in zip(wave.starts, wave.valids))
        c32, c16 = self._row_flags(cont, ttnn.float32), self._row_flags(cont, ttnn.bfloat16, rank=3)
        in_rec = ttnn.multiply(cur_rec, c32)  # a continuing row keeps its carry, the others start from zeros
        in_conv = ttnn.multiply(cur_conv, c16)
        state = LayerState(kda=KdaState(recurrent=in_rec, convolution=in_conv))
        return state, [in_rec, in_conv, c32, c16], (cur_rec, cur_conv, fresh)

    def _write_wave_kda_state(self, user: int, new: KdaState, current, wave: WaveInfo) -> None:
        """Write every member row's fresh carry into slot ``user`` of that row's batched state and an idle row's
        current carry back unchanged: ``sel = new * m + current * (1 - m)`` with the per-row member flag ``m``
        (exact 0 / 1 products), then the batched-slot blend of ``_write_user_kda_state``."""
        cur_rec, cur_conv, fresh = current
        member = tuple(1.0 if v > 0 else 0.0 for v in wave.valids)
        keep = tuple(1.0 - m for m in member)
        m32, m16 = self._row_flags(member, ttnn.float32), self._row_flags(member, ttnn.bfloat16, rank=3)
        k32, k16 = self._row_flags(keep, ttnn.float32), self._row_flags(keep, ttnn.bfloat16, rank=3)
        sel_rec = ttnn.multiply(new.recurrent, m32)
        old_rec = ttnn.multiply(cur_rec, k32)
        ttnn.add(sel_rec, old_rec, output_tensor=sel_rec)
        sel_conv = ttnn.multiply(new.convolution, m16)
        old_conv = ttnn.multiply(cur_conv, k16)
        ttnn.add(sel_conv, old_conv, output_tensor=sel_conv)
        for t in (old_rec, old_conv, m32, m16, k32, k16, new.recurrent, new.convolution, *fresh):
            ttnn.deallocate(t)
        self._write_user_kda_state(user, KdaState(recurrent=sel_rec, convolution=sel_conv))

    def prefill_wave_forward(self, streams: ttnn.Tensor, *, user: int, wave: WaveInfo) -> ttnn.Tensor:
        """One wave (``doc/mesh/row_ownership.md`` mechanism (a), phase 1: the same slot ``user`` on every row): row
        r of the mesh runs its own chunk ``[starts[r], starts[r] + valids[r])`` of the ``T = streams.shape[-2]``
        padded rows it holds (an idle row runs pad tokens; nothing of it is kept). KDA: every row scans its own
        chronology from its own carry and writes its own carry back into slot ``user`` of ITS batched state (an idle
        row's slot stays as it was). DSA: not yet (step 3). Row r's cursor ``next_start_rows[r][user]`` advances."""
        T = int(streams.shape[-2])
        if T % TILE or wave.T != T or wave.rows != self.mesh.rows:
            raise ValueError(
                f"wave T {wave.T} x {wave.rows} rows does not match streams {tuple(streams.shape)} on {self.mesh.rows} rows"
            )
        if user >= self.max_batch:
            raise ValueError(f"user {user} >= max_batch {self.max_batch}")
        if not wave.member_rows:
            raise ValueError("a wave needs at least one member row")
        for r, (s0, n) in enumerate(zip(wave.starts, wave.valids)):
            if n == 0:
                continue
            if s0 < 0 or s0 % TILE or n > T:
                raise ValueError(
                    f"row {r}: chunk start {s0} must be a non-negative multiple of {TILE} and valid rows {n} <= T {T}"
                )
            if s0 + n > self.max_seq_len:
                raise ValueError(f"row {r}: start {s0} + {n} > max_seq_len {self.max_seq_len}")
            if s0 and s0 != self.next_start_rows[r][user]:
                raise ValueError(
                    f"row {r} user {user}: chunk start {s0} is not contiguous with the previous chunk "
                    f"(expected {self.next_start_rows[r][user]})"
                )
        layer = self._require_weights()
        current = None
        if self.is_kda:
            state, temporaries, current = self._wave_state(user, wave)
        else:
            state, temporaries = LayerState(cache=self.cache), []
        out, new_state = layer.prefill_forward(streams, state, user=user, seq_len=T, start=0, wave=wave)
        for t in temporaries:
            ttnn.deallocate(t)
        if self.is_kda:
            self._write_wave_kda_state(user, new_state.kda, current, wave)
        for r, (s0, n) in enumerate(zip(wave.starts, wave.valids)):
            if n:
                self.next_start_rows[r][user] = s0 + n
        return out

    def decode_forward(
        self, streams: ttnn.Tensor, *, positions: DecodePositions, collapse: bool = False
    ) -> ttnn.Tensor:
        """One token per user slot. DSA layers pick the dense (traceable) or sparse path from the
        host positions carried by ``positions``. ``collapse`` (the last layer): return the LM head's
        mean over the 4 new streams ``[1,1,B,4096]`` fp32 instead of the streams (``TtMHCFast`` only)."""
        B = streams.shape[-2]
        layer = self._require_weights()
        if any(p >= self.max_seq_len for p in positions.host):
            raise ValueError(f"positions {positions.host} exceed max_seq_len {self.max_seq_len}")
        traced_sparse = positions.sparse is not None and sparse_decode_enabled()
        if (
            self.mesh.rows > 1
            and not self.is_kda
            and not traced_sparse
            and any(p > layer.attn.dense_exact_max_pos for p in positions.host)
        ):
            raise NotImplementedError("eager sparse DSA decode (positions > 2050) on a multi-row mesh is not wired")
        out, new_state = layer.decode_forward(
            streams, self._state_for_decode(B), positions=positions, collapse=collapse
        )
        if self.is_kda:
            assert new_state.kda is self.kda_batched, "KDA decode must update the persistent state in place"
        self.next_start = [p + 1 for p in positions.host]  # a prefill chunk may continue after decode
        return out

    # ------------------------------------------------------------------- trace
    @property
    def supports_collapse(self) -> bool:
        return bool(getattr(self.layer.ffn_hc, "supports_collapse", False)) if self.layer is not None else False

    def capture_decode_trace(
        self, streams: ttnn.Tensor, positions: DecodePositions, *, cq_id: int = 0, allow_sparse_range: bool = False
    ):
        """Warm up (compile) one decode step on the persistent inputs, then capture the next one.
        Returns ``(trace_id, output tensor, warm-up output)``. Replay with ``replay_decode``; the
        state tensors are updated in place by the captured graph. The warm-up and the capture both
        consume the token at ``positions`` (the state advances twice), so capture on a throw-away
        token or re-prefill afterwards. The captured graph is the dense path (flash-MLA for every
        slot + index maintenance), HF-exact through position 2050: a DSA layer whose ``max_seq_len``
        exceeds 2051 is refused unless ``allow_sparse_range`` (a replay at a position > 2050 would
        attend densely instead of taking the eager per-user sparse row, which is not HF semantics)."""
        layer = self._require_weights()
        traced_sparse = positions.sparse is not None and sparse_decode_enabled()
        if (
            not self.is_kda
            and not traced_sparse
            and self.max_seq_len > layer.attn.dense_exact_max_pos + 1
            and not allow_sparse_range
        ):
            raise ValueError(
                f"traced decode is HF-exact only through position {layer.attn.dense_exact_max_pos}; this decoder "
                f"allows max_seq_len={self.max_seq_len}. Use eager decode_forward beyond, or pass allow_sparse_range=True"
            )
        warm = self.decode_forward(streams, positions=positions)
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
        out = self.decode_forward(streams, positions=positions)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=cq_id)
        return trace_id, out, warm

    def replay_decode(self, trace_id, *, cq_id: int = 0, blocking: bool = True):
        ttnn.execute_trace(self.device, trace_id, cq_id=cq_id, blocking=blocking)
