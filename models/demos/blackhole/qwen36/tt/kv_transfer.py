# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6/3.8 KV-transfer model hook for prefill/decode disaggregation (PHASE2_DESIGN.md section 5).

``Qwen36KVTransfer(model)`` moves ONE request's device state between two nodes through transport ``Sink`` /
``Source`` objects (design 4.1; the connector/transport live in the vLLM TT plugin and are never imported here):

* producer (prefill node): ``describe_request_state`` -> ``export_request_state`` after the T-1 prefill returned;
* consumer (decode node): ``import_kv_blocks`` (request-private paged K/V blocks, any step end) ->
  ``validate_gdn_parts`` (host checks BEFORE ``finished_recving``) -> ``install_gdn_state`` (the GDN row of the slot
  whose FIRST decode is in the current step's batch: rec row + 4 conv taps, plus the packed conv history on a fused-conv
  decode; invariant I11).

Per-request state at TP=1 (design 5.2): 32 paged K/V tensors ``[num_blocks+1, 4, 64, 256]`` TILE bfp8_b (block-major
chunks of 32 blocks on the wire), 48 x ``rec_state[slot]`` ``[1, 48, 128, 128]`` fp32 TILE and 48 x 4 conv tap rows
``conv_states[m][0, slot, :]`` bf16 (``[4, 10240]`` host rows on the wire). Every device op runs eagerly on cq 0 from
the engine thread, IN PLACE into the trace-baked buffers (never a rebind, I5), with a fixed program set compiled by
``warmup_kv_transfer`` before any decode trace is captured (I6). ``TT_PD_STRICT_SHAPES=1`` asserts that no hook call
grows the device program cache (a request-time compile can clobber a parked trace, risk R5).

Spike-decided choices (profiles/pd/p2_spike_kv_xfer.md): the dumpfile chunk build uses the tensor-args form of
``ttnn.slice`` (start/end in device tensors refreshed by ``copy_host_to_device_tensor``; ONE program for every block
index, unlike the Python-int form whose start is a program-cache key) + one 32-way ``ttnn.concat``; the consumer's
block-major -> head-major relayout is ``ttnn.permute`` on bfp8 (bit-exact); the rec row is written with
``ttnn.fill_cache`` (``batch_idx`` is a runtime arg, byte-exact vs ``_write_index``; ``TT_PD_REC_WRITE=write_index``
selects the slice/concat/copy fallback); the 4 tap rows use ``_write_index`` (the exact host-path primitive).

Env knobs: ``TT_PD_STRICT_SHAPES`` (1: raise on program-cache growth inside a hook), ``TT_PD_REC_WRITE``
(``fill_cache`` | ``write_index``), ``TT_PD_IMPORT_VIA_WRITE_SLOT`` (1: install through ``TPGatedDeltaNet.write_slot``,
bisect only), ``TT_PD_CHECKSUM`` (1: check ``Source.crc32c()`` in ``validate_gdn_parts``), ``TT_PD_TIMING`` (1: log a
per-call timing line).
"""

import os
import time
from dataclasses import dataclass, field

import torch
from loguru import logger

import ttnn

LAYOUT_VERSION = 1
_TILE_ELEMS = 1024
_TILE_NBYTES = {"bfloat8_b": 1088, "bfloat4_b": 576, "bfloat16": 2048, "float32": 4096}
_ELEM_NBYTES = {"bfloat16": 2, "float32": 4, "int32": 4, "uint32": 4}
_TTNN_DTYPE = {"bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16, "float32": ttnn.float32}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def dtype_name(dt) -> str:
    """ttnn.DataType -> the wire name ("bfloat8_b" | "bfloat16" | "float32")."""
    return str(dt).split(".")[-1].lower()


def tile_nbytes(shape, dtype: str) -> int:
    numel = 1
    for d in shape:
        numel *= int(d)
    assert numel % _TILE_ELEMS == 0, f"{tuple(shape)} is not tile-aligned"
    return (numel // _TILE_ELEMS) * _TILE_NBYTES[dtype]


def row_major_nbytes(shape, dtype: str) -> int:
    numel = 1
    for d in shape:
        numel *= int(d)
    return numel * _ELEM_NBYTES[dtype]


# Field-compatible with the plugin's ``kv_transfer.metadata.PartSpec`` / ``Manifest`` (design 3.4): the transport
# coerces by attribute, so this module needs no plugin import.
@dataclass
class PartSpec:
    name: str  # "kv.L{i}.k" | "kv.L{i}.v" | "gdn.L{j}.rec" | "gdn.L{j}.taps"
    kind: str  # "kv_blocks" | "gdn_rec" | "gdn_taps"
    shape: tuple  # on-wire CHUNK shape
    dtype: str  # "bfloat8_b" | "bfloat16" | "float32"
    layout: str  # "TILE" | "ROW_MAJOR"
    nchunks: int
    chunk_nbytes: int
    chunk_offsets: list = field(default_factory=list)
    nbytes: int = 0

    def __post_init__(self):
        self.shape = tuple(int(s) for s in self.shape)
        if not self.chunk_offsets:
            self.chunk_offsets = [c * self.chunk_nbytes for c in range(self.nchunks)]
        if not self.nbytes:
            self.nbytes = self.nchunks * self.chunk_nbytes


@dataclass
class Manifest:
    layout_version: int
    model_sig: str
    num_tokens: int
    nblk: int
    block_size: int
    chunk_tokens: int
    kv_dtype: str
    rec_dtype: str
    parts: list
    prompt_hash: str

    @property
    def total_nbytes(self) -> int:
        return sum(p.nbytes for p in self.parts)

    def part(self, name: str) -> PartSpec:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)


def _spec_key(spec) -> tuple:
    return (tuple(int(s) for s in spec.shape), str(spec.dtype), str(spec.layout))


class Qwen36KVTransfer:
    """Model-side hook (design 5.1). ``model`` is the ``Qwen36Model`` (TP code path, paged KV allocated)."""

    BLOCKS_PER_CHUNK = 32

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh_device
        if self.mesh.get_num_devices() != 1:
            # The layout_version-1 wire format carries the TP=1 per-device shapes ([4, 10240] taps, [48,128,128] rec).
            raise RuntimeError("Qwen36KVTransfer: layout_version 1 is defined for a (1,1) mesh (TP=1 mode) only")
        self.strict = os.environ.get("TT_PD_STRICT_SHAPES", "0") == "1"
        self.rec_write = os.environ.get("TT_PD_REC_WRITE", "fill_cache")
        assert self.rec_write in ("fill_cache", "write_index"), f"TT_PD_REC_WRITE={self.rec_write!r}"
        self.via_write_slot = os.environ.get("TT_PD_IMPORT_VIA_WRITE_SLOT", "0") == "1"
        self.checksum = os.environ.get("TT_PD_CHECKSUM", "0") == "1"
        self.timing = os.environ.get("TT_PD_TIMING", "0") == "1"
        self._kv_staging = []  # 2 x [1, heads, chunk_tokens, head_dim] cache dtype (consumer)
        self._rec_staging = None  # [1, Nv, Dk, Dv] rec dtype (consumer, raw mode)
        self._slice_start = None  # int32 [4] device tensors for the tensor-args slice (producer, dumpfile)
        self._slice_end = None
        self._pt_host = {}
        self._warmed = set()  # {"producer", "consumer"}
        self._mode = None
        self._chunk_tokens = None
        self._n_program_cache_after_warmup = None
        self.request_time_compiles = 0  # entries the program cache grew by inside hook calls after warmup
        self._shape_keys = set()  # (op, key) recorded during warmup; first unknown request-time key is logged
        self._recording = False

    # ----------------------------------------------------------------------------------------------------------- #
    # geometry
    # ----------------------------------------------------------------------------------------------------------- #
    @property
    def attn_layers(self):
        return [self.model.layers[i].attention for i in self.model._attention_layer_indices]

    @property
    def gdn_layers(self):
        return [layer.attention for layer in self.model.layers if not layer.is_full_attention]

    def kv_transfer_gdn_layers(self):
        return self.gdn_layers

    def _kv_tensors(self):
        """[(name, device tensor)] in wire order: kv.L{i}.k, kv.L{i}.v for the model's attention layers."""
        out = []
        for i, a in enumerate(self.attn_layers):
            assert a.paged_k is not None and a.paged_v is not None, "paged KV not bound (allocate_kv_caches first)"
            out.append((f"kv.L{i}.k", a.paged_k))
            out.append((f"kv.L{i}.v", a.paged_v))
        return out

    def _geometry(self):
        k0 = self.attn_layers[0].paged_k
        assert k0 is not None, "paged KV not bound (allocate_kv_caches first)"
        num_blocks, heads, block_size, head_dim = (int(d) for d in k0.shape)
        dn = self.gdn_layers[0]
        assert dn.rec_state is not None and dn.conv_states is not None, "GDN state not allocated"
        return dict(
            num_blocks=num_blocks,
            heads=heads,
            block_size=block_size,
            head_dim=head_dim,
            kv_dtype=dtype_name(k0.dtype),
            rec_shape=tuple(int(d) for d in dn.rec_state.shape)[1:],  # (Nv, Dk, Dv)
            rec_dtype=dtype_name(dn.rec_state.dtype),
            K=int(dn.K),
            conv_dim=int(dn.conv_states[0].shape[-1]),
            B=int(dn.B),
        )

    @property
    def pad_block(self) -> int:
        pad = self.model._pad_kv_block
        assert pad is not None and pad != 0, f"pad KV block {pad} unusable (block 0 aliases a real request)"
        return int(pad)

    def _bpc(self, chunk_tokens=None) -> int:
        g = self._geometry()
        ct = chunk_tokens or self._chunk_tokens or self.BLOCKS_PER_CHUNK * g["block_size"]
        bpc = ct // g["block_size"]
        assert bpc * g["block_size"] == ct and bpc >= 1, f"chunk_tokens {ct} not a block multiple"
        return bpc

    # ----------------------------------------------------------------------------------------------------------- #
    # STRICT_SHAPES guard (design 5.5, C4/G4 decision)
    # ----------------------------------------------------------------------------------------------------------- #
    def _npc(self) -> int:
        return int(self.mesh.num_program_cache_entries())

    class _Guard:
        def __init__(self, hook, name):
            self.hook, self.name = hook, name

        def __enter__(self):
            self.n0 = self.hook._npc()
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, et, ev, tb):
            if et is not None:
                return False
            grew = self.hook._npc() - self.n0
            if self.hook.timing:
                logger.info(f"[PD_TIMING] {self.name}: {1e3 * (time.perf_counter() - self.t0):.1f} ms (+{grew} programs)")
            if grew > 0 and self.hook._n_program_cache_after_warmup is not None:
                self.hook.request_time_compiles += grew
                msg = f"PD: request-time program compile (+{grew} entries) in {self.name}"
                if self.hook.strict:
                    raise RuntimeError(msg)
                logger.warning(msg)
            return False

    def _guard(self, name):
        return self._Guard(self, name)

    def _note_shape(self, op, key):
        """Secondary shape-key recorder (5.5): log -- not raise -- the first request-time key outside the warmed set."""
        k = (op, key)
        if self._recording:
            self._shape_keys.add(k)
        elif self._n_program_cache_after_warmup is not None and k not in self._shape_keys:
            self._shape_keys.add(k)
            logger.warning(f"PD: op shape outside the warmed set: {op} {key}")

    # ----------------------------------------------------------------------------------------------------------- #
    # describe
    # ----------------------------------------------------------------------------------------------------------- #
    def describe_request_state(self, num_tokens: int, block_ids, *, model_sig: str = "", prompt_hash: str = "") -> Manifest:
        """The Manifest for ``num_tokens`` (= T-1) tokens in ``block_ids`` (design 3.4): 2 x n_attn ``kv_blocks`` parts of
        ``cdiv(nblk, 32)`` chunks, one ``gdn_rec`` and one ``gdn_taps`` part per GDN layer."""
        g = self._geometry()
        assert num_tokens >= 1, f"num_tokens {num_tokens}"
        nblk = cdiv(num_tokens, g["block_size"])
        assert len(block_ids) >= nblk, f"{len(block_ids)} blocks for {num_tokens} tokens (need {nblk})"
        bpc = self._bpc()
        nchunks = cdiv(nblk, bpc)
        kv_shape = (bpc, g["heads"], g["block_size"], g["head_dim"])
        kv_chunk = tile_nbytes(kv_shape, g["kv_dtype"])
        rec_shape = (1,) + g["rec_shape"]
        rec_nbytes = tile_nbytes(rec_shape, g["rec_dtype"])
        taps_shape = (g["K"], g["conv_dim"])
        taps_nbytes = row_major_nbytes(taps_shape, "bfloat16")
        parts = []
        for name, _ in self._kv_tensors():
            parts.append(PartSpec(name, "kv_blocks", kv_shape, g["kv_dtype"], "TILE", nchunks, kv_chunk))
        for j in range(len(self.gdn_layers)):
            parts.append(PartSpec(f"gdn.L{j}.rec", "gdn_rec", rec_shape, g["rec_dtype"], "TILE", 1, rec_nbytes))
        for j in range(len(self.gdn_layers)):
            parts.append(PartSpec(f"gdn.L{j}.taps", "gdn_taps", taps_shape, "bfloat16", "ROW_MAJOR", 1, taps_nbytes))
        return Manifest(
            layout_version=LAYOUT_VERSION,
            model_sig=model_sig,
            num_tokens=int(num_tokens),
            nblk=nblk,
            block_size=g["block_size"],
            chunk_tokens=bpc * g["block_size"],
            kv_dtype=g["kv_dtype"],
            rec_dtype=g["rec_dtype"],
            parts=parts,
            prompt_hash=prompt_hash,
        )

    # ----------------------------------------------------------------------------------------------------------- #
    # warmup (design 5.5)
    # ----------------------------------------------------------------------------------------------------------- #
    def _upload_starts(self, b: int):
        g = self._geometry()
        sh = ttnn.from_torch(torch.tensor([b, 0, 0, 0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        eh = ttnn.from_torch(
            torch.tensor([b + 1, g["heads"], g["block_size"], g["head_dim"]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(sh, self._slice_start)
        ttnn.copy_host_to_device_tensor(eh, self._slice_end)
        return sh, eh  # keep alive until the slice has been issued (in-order cq: until the next upload is enough)

    def _unit_slice(self, t, num_blocks):
        """Block ``b`` (set by ``_upload_starts``) of the paged tensor as a fresh ``[1, heads, block, head_dim]`` device
        tensor; start/end are runtime tensor args -> one program per cache shape (spike 6m)."""
        self._note_shape("slice_block", (tuple(int(d) for d in t.shape), dtype_name(t.dtype)))
        return ttnn.slice(t, self._slice_start, self._slice_end, slice_dim=0, num_devices=num_blocks)

    def _rep(self):
        return ttnn.ReplicateTensorToMesh(self.mesh)

    def warmup_kv_transfer(self, *, role: str, mode: str = "dumpfile", chunk_tokens: int = 2048, slots=None) -> None:
        """Compile every export/import program shape and allocate the staging tensors (outside the trace region).

        Called from the runner's warmup Phase 1, before any decode trace is captured; the slot rows hold no live
        sequence at that point (the per-slot warm loop writes zeros into them).
        role: "kv_producer" | "kv_consumer" | "kv_both"; mode: "dumpfile" | "raw".
        """
        assert mode in ("dumpfile", "raw"), mode
        roles = {"kv_producer": ("producer",), "kv_consumer": ("consumer",), "kv_both": ("producer", "consumer")}[role]
        g = self._geometry()
        self._mode = mode
        self._chunk_tokens = int(chunk_tokens)
        bpc = self._bpc(chunk_tokens)
        dns = self.gdn_layers
        kv = self._kv_tensors()
        num_blocks = g["num_blocks"]
        pad = self.pad_block
        if slots is None:
            slots = range(g["B"])
        slots = [int(s) for s in slots]
        n0 = self._npc()
        t0 = time.perf_counter()
        self._recording = True
        rep = self._rep()
        try:
            if "producer" in roles:
                if mode == "dumpfile":
                    if self._slice_start is None:
                        z = torch.zeros(4, dtype=torch.int32)
                        self._slice_start = ttnn.to_device(
                            ttnn.from_torch(z, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.mesh
                        )
                        self._slice_end = ttnn.to_device(
                            ttnn.from_torch(z, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self.mesh
                        )
                    # the two dumpfile chunk-build programs: unit tensor-args slice + bpc-way concat (both cache shapes
                    # are the same across the 32 tensors, so one tensor warms all)
                    keep = []
                    parts = []
                    for b in (0, pad):
                        keep.append(self._upload_starts(b))
                        parts.append(self._unit_slice(kv[0][1], num_blocks))
                    parts += [parts[-1]] * (bpc - len(parts))
                    blk = ttnn.concat(parts, dim=0)
                    self._note_shape("concat_chunk", (bpc, tuple(int(d) for d in blk.shape), dtype_name(blk.dtype)))
                    ttnn.synchronize_device(self.mesh)
                    h = ttnn.from_device(blk)  # the Sink's D2H (no program)
                    del h
                    ttnn.deallocate(blk)
                    for p in parts[: min(2, len(parts))]:
                        ttnn.deallocate(p)
                # GDN D2H reads (no programs): rec whole buffer + 4 tap tensors
                dn0 = dns[0]
                _ = ttnn.from_device(dn0.rec_state)
                for m in range(dn0.K):
                    _ = ttnn.to_torch(dn0.conv_states[m])
            if "consumer" in roles:
                cache_dtype = kv[0][1].dtype
                st_shape = (1, g["heads"], bpc * g["block_size"], g["head_dim"])
                if not self._kv_staging:
                    for _ in range(2):
                        self._kv_staging.append(
                            ttnn.from_torch(
                                torch.zeros(*st_shape, dtype=torch.bfloat16),
                                dtype=cache_dtype,
                                layout=ttnn.TILE_LAYOUT,
                                device=self.mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                mesh_mapper=rep,
                            )
                        )
                if self._rec_staging is None:
                    self._rec_staging = ttnn.from_torch(
                        torch.zeros(1, *g["rec_shape"], dtype=torch.float32),
                        dtype=dns[0].rec_state.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=rep,
                    )
                # paged_fill_cache from the staging tensor with an all-pad page table (writes zeros into the pad block)
                pt = self._pt_tensor([pad] * bpc)
                for name, t in kv:
                    self._note_shape("paged_fill_cache", (tuple(int(d) for d in t.shape), st_shape, dtype_name(t.dtype)))
                    ttnn.experimental.paged_fill_cache(t, self._kv_staging[0], pt, batch_idx=0)
                ttnn.deallocate(pt)
                if mode == "dumpfile":
                    # block-major chunk -> head-major staging: permute + reshape + copy (once; shape-keyed)
                    blk = ttnn.from_torch(
                        torch.zeros(bpc, g["heads"], g["block_size"], g["head_dim"], dtype=torch.bfloat16),
                        dtype=cache_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=rep,
                    )
                    self._relayout_into_staging(blk, self._kv_staging[0])
                    self._relayout_into_staging(blk, self._kv_staging[1])
                    ttnn.deallocate(blk)
                # per-slot GDN row writes on the first GDN layer (programs are shape-keyed, identical across layers):
                # rec via fill_cache (runtime batch_idx) or _write_index, 4 taps via _write_index, packed row on fused-conv
                dn0 = dns[0]
                zero_row = torch.zeros(1, 1, g["conv_dim"], dtype=torch.bfloat16)
                zero_rec_dev = None
                for slot in slots:
                    assert 0 <= slot < dn0.B, f"slot {slot} outside [0, {dn0.B})"
                    if self.rec_write == "fill_cache":
                        self._note_shape("fill_cache_rec", (tuple(int(d) for d in dn0.rec_state.shape), slot))
                        ttnn.fill_cache(dn0.rec_state, self._rec_staging, slot)
                    else:
                        self._note_shape("write_index_rec", (tuple(int(d) for d in dn0.rec_state.shape), slot))
                        dn0._write_index(dn0.rec_state, ttnn.clone(self._rec_staging), slot, dim=0)
                    for m in range(dn0.K):
                        c = self._tap_row_tensor(zero_row, dn0)
                        self._note_shape("write_index_tap", (tuple(int(d) for d in dn0.conv_states[m].shape), slot))
                        dn0._write_index(dn0.conv_states[m], c, slot, dim=1)
                    if getattr(dn0, "_decode_fused_conv", False) and dn0.conv_hist_packed is not None:
                        packed = self._packed_slot_tensor(dn0, torch.zeros(dn0.K, g["conv_dim"], dtype=torch.bfloat16), slot)
                        self._note_shape("write_index_packed", (tuple(int(d) for d in dn0.conv_hist_packed.shape), slot))
                        dn0._write_index(dn0.conv_hist_packed, packed, slot, dim=0)
                del zero_rec_dev
            ttnn.synchronize_device(self.mesh)
        finally:
            self._recording = False
        self._warmed |= set(roles)
        self._n_program_cache_after_warmup = self._npc()
        logger.info(
            f"[PD] warmup_kv_transfer role={role} mode={mode} chunk_tokens={chunk_tokens} slots={slots}: "
            f"+{self._n_program_cache_after_warmup - n0} programs ({self._n_program_cache_after_warmup} total), "
            f"{len(self._shape_keys)} shape keys, {1e3 * (time.perf_counter() - t0):.0f} ms; "
            f"strict_shapes={self.strict} rec_write={self.rec_write} via_write_slot={self.via_write_slot}"
        )

    # ----------------------------------------------------------------------------------------------------------- #
    # small device helpers
    # ----------------------------------------------------------------------------------------------------------- #
    def _pt_tensor(self, rows):
        return ttnn.from_torch(
            torch.tensor([list(rows)], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=self._rep(),
        )

    def _tap_row_tensor(self, row, dn):
        """[1, 1, D] bf16 host row -> device TILE tensor in conv_states' dtype (the write_slot tap path's upload)."""
        return ttnn.from_torch(
            row.reshape(1, 1, -1),
            dtype=dn.conv_states[0].dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=self._rep(),
        )

    def _packed_slot_tensor(self, dn, rows, slot):
        """rows [K, D] bf16 (tap m = row m; tap 3 newest) -> this slot's packed conv history [1, Nv, 4, 32, 32] bf16
        device tensor with parity ``slot & 1`` -- the same bytes ``TPGatedDeltaNet._packed_slot_tensor`` produces
        (one vectorized gather via ``_pack_rows_vec``; the per-head Python loop is gone since 5234d22)."""
        packed = dn._pack_rows_vec(rows.to(torch.bfloat16), int(slot) & 1)  # [K, Nv, 32, 32]
        packed = packed.permute(1, 0, 2, 3).contiguous().unsqueeze(0)  # [1, Nv, K, 32, 32]
        return ttnn.from_torch(
            packed,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )

    def _relayout_into_staging(self, blk, st):
        """dumpfile consumer: block-major [bpc, heads, block, hd] chunk -> head-major staging [1, heads, bpc*block, hd]
        (bfp8 permute is bit-exact, spike 6a)."""
        hm = ttnn.permute(blk, (1, 0, 2, 3))
        hm2 = ttnn.reshape(hm, tuple(int(d) for d in st.shape))  # tile-aligned view of hm's buffer
        ttnn.copy(hm2, st)
        ttnn.deallocate(hm2)
        del hm

    # ----------------------------------------------------------------------------------------------------------- #
    # export (producer; design 5.3)
    # ----------------------------------------------------------------------------------------------------------- #
    def export_request_state(self, block_ids, num_tokens: int, slot: int, sinks) -> None:
        """Write the request's K/V blocks + GDN row into ``sinks`` (one Sink per part name). Two producer shapes:
        ``dn.B == 1`` (R2 prefill node / the bound B=1 prefill scratch: the layer buffers ARE the request's state) and
        ``dn.B > 1`` (kv_both node: row ``slot``, read as a byte region in raw mode or a host slice in dumpfile mode).
        Device synchronized on return; every device temporary freed after the final sync (rule N2)."""
        with self._guard("export_request_state"):
            g = self._geometry()
            nblk = cdiv(num_tokens, g["block_size"])
            ids = [int(b) for b in block_ids[:nblk]]
            assert len(ids) == nblk, f"{len(block_ids)} blocks for {num_tokens} tokens (need {nblk})"
            bpc = self._bpc()
            pad = self.pad_block
            num_blocks = g["num_blocks"]
            kv = self._kv_tensors()
            tmp = []
            t0 = time.perf_counter()
            for c in range(cdiv(nblk, bpc)):
                chunk_ids = ids[bpc * c : bpc * c + bpc]
                regions = getattr(sinks[kv[0][0]], "supports_regions", False)
                if regions:
                    blk_nbytes = tile_nbytes((1, g["heads"], g["block_size"], g["head_dim"]), g["kv_dtype"])
                    for name, t in kv:
                        sink = sinks[name]
                        for k0, b0, n in _contiguous_runs(chunk_ids):
                            sink.write_region_from_device(
                                t, b0 * blk_nbytes, n * blk_nbytes, chunk=c, dst_offset_bytes=k0 * blk_nbytes, blocking=False
                            )
                    continue
                # dumpfile: one runtime-start unit slice per (block, tensor) + one bpc-way concat per tensor; the block
                # index is set once and reused by all 32 tensors (in-order cq). Blocking sink writes so the chunk's
                # temporaries can be freed right away (a 32k prompt would otherwise pin 2.3 GB of them).
                rows = chunk_ids + [pad] * (bpc - len(chunk_ids))
                parts = {name: [] for name, _ in kv}
                keep = []
                for b in rows:
                    keep.append(self._upload_starts(b))
                    for name, t in kv:
                        parts[name].append(self._unit_slice(t, num_blocks))
                for name, t in kv:
                    blk = ttnn.concat(parts[name], dim=0)
                    self._note_shape("concat_chunk", (bpc, tuple(int(d) for d in blk.shape), dtype_name(blk.dtype)))
                    sinks[name].write_from_device(blk, chunk=c, blocking=True)
                    ttnn.deallocate(blk)
                    for p in parts[name]:
                        ttnn.deallocate(p)
                del keep
            t1 = time.perf_counter()
            for j, dn in enumerate(self.gdn_layers):
                rs = sinks[f"gdn.L{j}.rec"]
                if dn.B == 1:
                    rs.write_from_device(dn.rec_state, chunk=0, blocking=False)
                elif getattr(rs, "supports_regions", False):
                    row_nbytes = tile_nbytes((1,) + g["rec_shape"], g["rec_dtype"])
                    rs.write_region_from_device(dn.rec_state, slot * row_nbytes, row_nbytes, chunk=0, dst_offset_bytes=0, blocking=False)
                else:
                    h = ttnn.from_device(dn.rec_state)  # whole [B, Nv, Dk, Dv] (24 MiB at B=8), blocking
                    row = ttnn.from_torch(ttnn.to_torch(h)[slot : slot + 1], dtype=dn.rec_state.dtype, layout=ttnn.TILE_LAYOUT)
                    rs.write_host(row, chunk=0)
                rows = torch.stack([ttnn.to_torch(dn.conv_states[m])[0, slot] for m in range(dn.K)])  # [K, D] bf16 host
                sinks[f"gdn.L{j}.taps"].write_rows(rows.to(torch.bfloat16).contiguous())
            ttnn.synchronize_device(self.mesh)
            for t in tmp:
                ttnn.deallocate(t)
            if self.timing:
                logger.info(
                    f"[PD_TIMING] export T={num_tokens} nblk={nblk} slot={slot} B={dn.B}: kv {1e3 * (t1 - t0):.1f} ms, "
                    f"gdn {1e3 * (time.perf_counter() - t1):.1f} ms"
                )

    # ----------------------------------------------------------------------------------------------------------- #
    # import (consumer; design 5.4)
    # ----------------------------------------------------------------------------------------------------------- #
    def import_kv_blocks(self, sources, block_ids, num_tokens: int, *, chunk_range=None) -> int:
        """K/V chunks only (request-private blocks; any step's END hook). Returns the number of chunks done. The caller
        synchronizes the device and reports KV_DONE (design 3.3)."""
        with self._guard("import_kv_blocks"):
            g = self._geometry()
            nblk = cdiv(num_tokens, g["block_size"])
            ids = [int(b) for b in block_ids[:nblk]]
            assert len(ids) == nblk, f"{len(block_ids)} blocks for {num_tokens} tokens (need {nblk})"
            bpc = self._bpc()
            nchunks = cdiv(nblk, bpc)
            chunks = range(nchunks) if chunk_range is None else range(nchunks)[chunk_range]
            assert self._kv_staging, "warmup_kv_transfer(role=consumer) must run first"
            kv = self._kv_tensors()
            done = 0
            t0 = time.perf_counter()
            for c in chunks:
                cids = ids[bpc * c : bpc * c + bpc]
                pt_rows = cids + [self.pad_block] * (bpc - len(cids))  # never 0; the pad block owns the tail garbage
                pt = self._pt_tensor(pt_rows)
                for name, t in kv:
                    src = sources[name].chunk(c)
                    st = self._kv_staging[c % 2]
                    if getattr(src, "is_head_major", False) and getattr(src, "is_device_readable", False):
                        src.read_into_device(st)  # raw mode: host relayout done off-thread, raw H2D
                    else:
                        blk = src.read_device(self.mesh)  # dumpfile: fresh [bpc, heads, block, hd] block-major tensor
                        self._relayout_into_staging(blk, st)
                        ttnn.deallocate(blk)
                    self._note_shape("paged_fill_cache", (tuple(int(d) for d in t.shape), tuple(int(d) for d in st.shape), dtype_name(t.dtype)))
                    ttnn.experimental.paged_fill_cache(t, st, pt, batch_idx=0)
                ttnn.deallocate(pt)
                done += 1
            if self.timing:
                logger.info(f"[PD_TIMING] import_kv_blocks T={num_tokens} chunks={list(chunks)}: {1e3 * (time.perf_counter() - t0):.1f} ms")
            return done

    def validate_gdn_parts(self, sources) -> None:
        """Host-only checks of every GDN part (sizes, dtypes, layouts, optional crc). Raises -> the worker fails the load
        BEFORE ``finished_recving``; nothing device-side has been touched for the GDN row yet."""
        g = self._geometry()
        rec_spec = ((1,) + g["rec_shape"], g["rec_dtype"], "TILE")
        rec_nbytes = tile_nbytes(rec_spec[0], g["rec_dtype"])
        taps_spec = ((g["K"], g["conv_dim"]), "bfloat16", "ROW_MAJOR")
        taps_nbytes = row_major_nbytes(taps_spec[0], "bfloat16")
        for j in range(len(self.gdn_layers)):
            for name, spec, nbytes in ((f"gdn.L{j}.rec", rec_spec, rec_nbytes), (f"gdn.L{j}.taps", taps_spec, taps_nbytes)):
                s = sources[name]  # KeyError -> missing part
                got = _spec_key(s.spec)
                if got != spec:
                    raise RuntimeError(f"PD: {name} spec {got} != expected {spec}")
                present = int(getattr(s, "nbytes_present", nbytes))
                if present != nbytes:
                    raise RuntimeError(f"PD: {name} has {present} bytes, expected {nbytes}")
                if self.checksum:
                    crc = s.crc32c()
                    if crc != int(getattr(s, "spec_crc", crc)):
                        raise RuntimeError(f"PD: {name} crc32c {crc:#x} != header {int(s.spec_crc):#x}")
        for name, _ in self._kv_tensors():
            if name not in sources:
                raise RuntimeError(f"PD: missing K/V part {name}")

    def install_gdn_state(self, sources, slot: int) -> None:
        """Write the rec row + 4 conv tap rows (+ the packed conv history on a fused-conv decode) into ``slot`` of every
        GDN layer, in place. JOIN-step step-BEGIN hook only (I11): the composite TP=1 decode advances every idle row's
        taps and, inside the pow2 bucket, rewrites its rec row, so an installed row must decode in the very next
        forward. Device synchronized on return (I4); every host buffer is referenced until then."""
        with self._guard("install_gdn_state"):
            dns = self.gdn_layers
            fresh = []
            keep = []
            slot = int(slot)
            t0 = time.perf_counter()
            for j, dn in enumerate(dns):
                assert 0 <= slot < dn.B, f"slot {slot} outside [0, {dn.B})"
                src = sources[f"gdn.L{j}.rec"].chunk(0)
                if getattr(src, "is_device_readable", False):
                    r = self._rec_staging
                    assert r is not None, "warmup_kv_transfer(role=consumer) must run first"
                    src.read_into_device(r)
                else:
                    r = src.read_device(self.mesh)  # dumpfile: load_tensor -> fresh DRAM tensor (outside the trace region)
                    fresh.append(r)
                if r.dtype != dn.rec_state.dtype:
                    rc = ttnn.typecast(r, dn.rec_state.dtype)
                    fresh.append(rc)
                    r = rc
                rows = sources[f"gdn.L{j}.taps"].read_rows().to(torch.bfloat16)  # [K, D] host
                keep.append(rows)
                assert tuple(rows.shape) == (dn.K, int(dn.conv_states[0].shape[-1])), f"taps {tuple(rows.shape)}"
                if self.via_write_slot:
                    # bisect path: TPGatedDeltaNet.write_slot verbatim (consumes rec + convs; its tail re-syncs the packed
                    # history: per-slot when valid, else a FULL rebuild -- never use this on a live fused-conv node)
                    convs = [self._tap_row_tensor(rows[m], dn) for m in range(dn.K)]
                    dn.write_slot(slot, ttnn.clone(r), convs)
                    continue
                if self.rec_write == "fill_cache":
                    self._note_shape("fill_cache_rec", (tuple(int(d) for d in dn.rec_state.shape), slot))
                    ttnn.fill_cache(dn.rec_state, r, slot)  # does not consume r
                else:
                    self._note_shape("write_index_rec", (tuple(int(d) for d in dn.rec_state.shape), slot))
                    dn._write_index(dn.rec_state, ttnn.clone(r), slot, dim=0)
                for m in range(dn.K):
                    c = self._tap_row_tensor(rows[m], dn)
                    self._note_shape("write_index_tap", (tuple(int(d) for d in dn.conv_states[m].shape), slot))
                    dn._write_index(dn.conv_states[m], c, slot, dim=1)  # exact host-path primitive (write_slot's tap path)
                if getattr(dn, "_decode_fused_conv", False):
                    # fused-conv decode (QWEN36_GDN_DECODE_FUSED=2, AM3 M2): the op reads conv_hist_packed[slot] (parity
                    # slot & 1) inside the decode trace, so the row we just installed must also land there. Never a full
                    # rebuild here (it would revert every live user's history from their stale conv_states, I4).
                    if not dn._hist_packed_valid or dn.conv_hist_packed is None:
                        raise RuntimeError(f"PD: GDN layer {j}: conv_hist_packed invalid while slots are live")
                    packed = self._packed_slot_tensor(dn, rows, slot)
                    self._note_shape("write_index_packed", (tuple(int(d) for d in dn.conv_hist_packed.shape), slot))
                    dn._write_index(dn.conv_hist_packed, packed, slot, dim=0)
            ttnn.synchronize_device(self.mesh)  # every H2D done before any host buffer above goes away
            for t in fresh:
                ttnn.deallocate(t)
            del keep
            if self.timing:
                logger.info(f"[PD_TIMING] install_gdn_state slot={slot}: {1e3 * (time.perf_counter() - t0):.1f} ms")

    def import_request_state(self, sources, block_ids, num_tokens: int, slot: int, *, chunk_range=None) -> None:
        """import_kv_blocks (all chunks) + validate_gdn_parts + install_gdn_state; tests / TT_PD_VERIFY_IMPORT only."""
        self.import_kv_blocks(sources, block_ids, num_tokens, chunk_range=chunk_range)
        ttnn.synchronize_device(self.mesh)
        self.validate_gdn_parts(sources)
        g = self._geometry()
        nchunks = cdiv(cdiv(num_tokens, g["block_size"]), self._bpc())
        if chunk_range is None or chunk_range.stop is None or chunk_range.stop >= nchunks:
            self.install_gdn_state(sources, slot)


def _contiguous_runs(ids):
    """[(index in chunk, first block, run length)] over consecutive physical block ids."""
    out = []
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append((i, ids[i], j - i + 1))
        i = j + 1
    return out
