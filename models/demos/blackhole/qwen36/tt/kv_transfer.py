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

Handoff fast paths (p1d1_opt lane B, 2026-09-22):

* producer K/V: the prefill traces MIRROR each chunk's head-major K/V (``paged_fill_cache``'s input = the bytes the
  cache holds) into persistent staging tensors bound into the attention layers by ``warmup_kv_transfer`` (one
  ``fill_cache`` per K/V part per chunk, inside the trace); the worker opens the export at step BEGIN
  (``begin_export``) and the model's chunk observer (``on_prefill_chunk``) copies the staging into the transport's
  pool buffers at every chunk boundary, so ``export_request_state`` only writes chunks the mirror missed (the
  paged-cache gather = 32 unit slices + concat + relayout per part, ~20-30 ms per chunk) plus the GDN row;
* producer taps: ONE device concat + untilize + D2H read of all 48 x 4 tap rows (2 ms) instead of 192 blocking tile
  reads (30 ms);
* consumer GDN install: the rec row is ``fill_cache``'d straight from the received pool buffer (no staging copy) and
  the 192 tap rows go up as ONE compact ROW_MAJOR tensor, are tilized on device and written with ``update_cache``
  over a width-split zero-copy view of ``conv_states[m]`` (one op per row; ``TT_PD_TAPS_WRITE=write_index`` keeps
  the slice/concat/copy path, 4 ops + a 640 KiB tile upload per row).

Env knobs: ``TT_PD_STRICT_SHAPES`` (1: raise on program-cache growth inside a hook), ``TT_PD_REC_WRITE``
(``fill_cache`` | ``write_index``), ``TT_PD_IMPORT_VIA_WRITE_SLOT`` (1: install through ``TPGatedDeltaNet.write_slot``,
bisect only), ``TT_PD_CHECKSUM`` (1: check ``Source.crc32c()`` in ``validate_gdn_parts``), ``TT_PD_TIMING`` (1: log a
per-call timing line), ``TT_PD_TAPS_WRITE`` (``update_cache`` | ``write_index``), ``TT_PD_TAPS_SPLIT`` (width slabs of
the tap-row view, default 40 -> 256 columns each; ``update_cache`` stages ``32 x slab`` tiles per core),
``TT_PD_HIST_WRITE`` (``batched`` | ``legacy``: the fused-conv packed conv-history row of an install -- one host pack
of all layers + per layer one H2D into persistent staging and one ``fill_cache``, or the per-layer pack/upload +
``_write_index``), ``TT_PD_ROWS_PREFETCH`` (1: ``validate_gdn_parts`` loads the taps rows while the K/V fills run and
``install_gdn_state`` reuses them; 0: read at install),
``TT_PD_EXPORT_MIRROR`` (0: never bind the staging; the export gathers from the paged cache as before),
``TT_PD_CHUNK_SYNC`` (1: before a chunk boundary at which the pump could send -- a published export is waiting for
its claim -- the observer synchronizes the device, so the sends follow a FINISHED chunk; a request with no pending
handoff keeps QWEN36_PREFILL_OVERLAP's host/device pipelining. 0: no per-chunk sync and therefore NO chunk-boundary
pump: the traced loop runs the host up to 8 replays (~14 s) ahead of the device, so a send marker written at a
host-time boundary would make the consumer post recvs that park its cq behind the producer's queued replays; the
mirror copies stay, the claim-gated sends wait for the step-begin hold / step end as before).
"""

import collections
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


@dataclass
class _ActiveExport:
    """The export the worker opened at step begin for the request whose prefill runs in this step."""

    block_ids: list
    num_tokens: int
    nblk: int
    nchunks: int
    sinks: object  # {part name: Sink}
    written: set = field(default_factory=set)  # chunk indices the mirror copied into the sinks (all K/V parts)
    t0: float = 0.0


class Qwen36KVTransfer:
    """Model-side hook (design 5.1). ``model`` is the ``Qwen36Model`` (TP code path, paged KV allocated)."""

    BLOCKS_PER_CHUNK = 32
    _ROWS_CACHE_MAX = 16

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh_device
        if self.mesh.get_num_devices() != 1:
            # The layout_version-1 wire format carries the TP=1 per-device shapes ([4, 10240] taps, [48,128,128] rec).
            raise RuntimeError("Qwen36KVTransfer: layout_version 1 is defined for a (1,1) mesh (TP=1 mode) only")
        if getattr(model, "mtp", None) is not None:
            # install_gdn_state writes the slot's recurrent/conv state directly; it bypasses the speculative decoders'
            # spec-mirror protocol (the durable-state ring a verify replays against), so a PD node must run PLAIN
            # decode. The MTP head is only built when a speculative mode asks for it (QWEN36_MTP=1 / QWEN36_DRAFTER=mtp).
            raise RuntimeError(
                "Qwen36KVTransfer: the PD nodes run plain decode; this model built the MTP drafter head "
                "(speculative substrate). Unset QWEN36_MTP / QWEN36_DRAFTER=mtp on the PD launch."
            )
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
        self._zero_blk = (
            None  # [1, heads, block, head_dim] zeros (cache dtype): the dumpfile chunk's filler rows (producer)
        )
        self._pt_host = {}
        self._warmed = set()  # {"producer", "consumer"}
        self._mode = None
        self._chunk_tokens = None
        self._n_program_cache_after_warmup = None
        self.request_time_compiles = 0  # entries the program cache grew by inside hook calls after warmup
        self._shape_keys = set()  # (op, key) recorded during warmup; first unknown request-time key is logged
        self._recording = False
        # producer: the prefill traces mirror each chunk's head-major K/V into these persistent staging tensors (one
        # per K/V part, bound into the attention layers); on_prefill_chunk copies them into the open export's pool
        # buffers at the chunk boundary, so export_request_state re-gathers nothing from the paged cache.
        self._export_staging = {}  # part name -> [1, heads, chunk_tokens, head_dim] cache-dtype device tensor
        self._active_export = None  # _ActiveExport between begin_export and export_request_state / end_export
        self._chunk_pump = None  # worker callback run at every prefill chunk boundary (claim-gated sends' clock)
        self._pump_wanted = None  # host-only callable: could the pump send something now? (published-unsent exports)
        self._chunk_sync = False  # the observer synchronizes the device before a boundary the pump may use
        self._observer_installed = False
        self.export_mirror = os.environ.get("TT_PD_EXPORT_MIRROR", "1") != "0"
        # consumer: all tap rows of one install as ONE compact ROW_MAJOR upload + device tilize, each row written with
        # update_cache over a width-split zero-copy view of conv_states[m]; write_index = the slice/concat/copy path.
        self.taps_write = os.environ.get("TT_PD_TAPS_WRITE", "update_cache")
        assert self.taps_write in ("update_cache", "write_index"), f"TT_PD_TAPS_WRITE={self.taps_write!r}"
        self.taps_split = int(os.environ.get("TT_PD_TAPS_SPLIT", "40"))
        # consumer, fused-conv decode: how the packed conv history row of an install is written. batched = ONE
        # host pack of all layers into a persistent per-parity host buffer, then per layer an H2D into persistent
        # staging + ONE fill_cache; legacy = per layer host pack + fresh upload + _write_index (slice/concat/copy).
        self.hist_write = os.environ.get("TT_PD_HIST_WRITE", "batched")
        assert self.hist_write in ("batched", "legacy"), f"TT_PD_HIST_WRITE={self.hist_write!r}"
        self._hist_stage = None  # L x [1, Nv*K, 32, 32] bf16 TILE device staging (one per GDN layer)
        self._hist_host = None  # {parity: torch bf16 [L, Nv, K, 32, 32]} (rows other than 2c+parity stay zero)
        self.last_install_timing = None
        # consumer: validate_gdn_parts loads the taps rows (host) while the K/V fills run; install_gdn_state reuses them
        self.rows_prefetch = os.environ.get("TT_PD_ROWS_PREFETCH", "1") != "0"
        # id(sources) -> (sources, [K, D] bf16 rows per GDN layer). A job that fails after validate never installs, so its
        # entry (~3.9 MB of rows + a reference to its sources dict; transports release explicitly, no device memory)
        # stays until _ROWS_CACHE_MAX newer validates evict it: <= ~63 MB host under an abort-heavy load. If more than
        # _ROWS_CACHE_MAX validated jobs wait for their join at once, the oldest falls back to reading at install.
        self._rows_cache = collections.OrderedDict()
        self._taps_stage_rm = None  # [1, 1, R32, D] bf16 ROW_MAJOR device staging (R32 = tap rows rounded up to 32)
        self.mirrored_chunks = 0  # chunks copied by the mirror (all requests)
        self.gathered_chunks = 0  # chunks the export had to gather from the paged cache

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
                logger.info(
                    f"[PD_TIMING] {self.name}: {1e3 * (time.perf_counter() - self.t0):.1f} ms (+{grew} programs)"
                )
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
    def describe_request_state(
        self, num_tokens: int, block_ids, *, model_sig: str = "", prompt_hash: str = ""
    ) -> Manifest:
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
        sh = ttnn.from_torch(
            torch.tensor([b, 0, 0, 0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
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
                    if self._zero_blk is None:
                        # Filler for the chunk rows past the request's blocks. NOT a slice of the pad block: the masked-
                        # bucket prefill writes its pad rows there, and their bytes are not run-to-run deterministic
                        # (the rows are never read by a real request), so an export's filler would depend on the
                        # prefill that last ran and two exports of one request could differ byte-for-byte.
                        self._zero_blk = ttnn.from_torch(
                            torch.zeros(1, g["heads"], g["block_size"], g["head_dim"], dtype=torch.bfloat16),
                            dtype=kv[0][1].dtype,
                            layout=ttnn.TILE_LAYOUT,
                            device=self.mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=rep,
                        )
                    # the two dumpfile chunk-build programs: unit tensor-args slice + bpc-way concat (both cache shapes
                    # are the same across the 32 tensors, so one tensor warms all)
                    keep = []
                    parts = []
                    for b in (0, pad):
                        keep.append(self._upload_starts(b))
                        parts.append(self._unit_slice(kv[0][1], num_blocks))
                    parts += [self._zero_blk] * (bpc - len(parts))
                    blk = ttnn.concat(parts, dim=0)
                    self._note_shape("concat_chunk", (bpc, tuple(int(d) for d in blk.shape), dtype_name(blk.dtype)))
                    ttnn.synchronize_device(self.mesh)
                    h = ttnn.from_device(blk)  # the Sink's D2H (no program)
                    del h
                    ttnn.deallocate(blk)
                    for p in parts[: min(2, len(parts))]:
                        ttnn.deallocate(p)
                # GDN D2H reads: rec whole buffer (no program) + the batched taps read (concat + untilize programs)
                dn0 = dns[0]
                _ = ttnn.from_device(dn0.rec_state)
                _ = self._read_taps_batched(dns, 0)
                # the export mirror: persistent head-major staging per K/V part, bound into the attention layers so
                # every prefill trace captured from now on fill_cache's its chunk's K/V there (attention/tp.py)
                if self.export_mirror:
                    self._bind_export_staging(g, kv, bpc)
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
                    self._note_shape(
                        "paged_fill_cache", (tuple(int(d) for d in t.shape), st_shape, dtype_name(t.dtype))
                    )
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
                # rec via fill_cache (runtime batch_idx) or _write_index, taps via update_cache over the width-split
                # view (all layers, one compact upload) or 4 x _write_index, packed row on fused-conv
                dn0 = dns[0]
                zero_row = torch.zeros(1, 1, g["conv_dim"], dtype=torch.bfloat16)
                zero_rec_dev = None
                if self.taps_write == "update_cache":
                    self._alloc_taps_stage(g, len(dns), dn0.K)
                hist_batched = (
                    self.hist_write == "batched"
                    and not self.via_write_slot
                    and self.mesh.get_num_devices() == 1  # replicated staging == the legacy dim-0 shard only at 1x1
                    and all(getattr(dn, "_decode_fused_conv", False) and dn.conv_hist_packed is not None for dn in dns)
                )
                if hist_batched:
                    self._alloc_hist_stage(dns)
                for slot in slots:
                    assert 0 <= slot < dn0.B, f"slot {slot} outside [0, {dn0.B})"
                    if self.rec_write == "fill_cache":
                        self._note_shape("fill_cache_rec", (tuple(int(d) for d in dn0.rec_state.shape), slot))
                        ttnn.fill_cache(dn0.rec_state, self._rec_staging, slot)
                    else:
                        self._note_shape("write_index_rec", (tuple(int(d) for d in dn0.rec_state.shape), slot))
                        dn0._write_index(dn0.rec_state, ttnn.clone(self._rec_staging), slot, dim=0)
                    if self.taps_write == "update_cache":
                        # the slot rows hold no live sequence at warmup: zeros into every layer's row `slot`
                        keep = self._write_taps_update_cache(
                            dns, torch.zeros(len(dns), dn0.K, g["conv_dim"], dtype=torch.bfloat16), slot
                        )
                        ttnn.synchronize_device(self.mesh)
                        del keep
                    else:
                        for m in range(dn0.K):
                            c = self._tap_row_tensor(zero_row, dn0)
                            self._note_shape("write_index_tap", (tuple(int(d) for d in dn0.conv_states[m].shape), slot))
                            dn0._write_index(dn0.conv_states[m], c, slot, dim=1)
                    if hist_batched:
                        # zeros into every layer's packed row `slot` (= the zero tap rows written above)
                        keep = self._write_hist_batched(
                            dns, torch.zeros(len(dns), dn0.K, g["conv_dim"], dtype=torch.bfloat16), slot
                        )
                        ttnn.synchronize_device(self.mesh)
                        del keep
                    elif getattr(dn0, "_decode_fused_conv", False) and dn0.conv_hist_packed is not None:
                        packed = self._packed_slot_tensor(
                            dn0, torch.zeros(dn0.K, g["conv_dim"], dtype=torch.bfloat16), slot
                        )
                        self._note_shape(
                            "write_index_packed", (tuple(int(d) for d in dn0.conv_hist_packed.shape), slot)
                        )
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
            f"strict_shapes={self.strict} rec_write={self.rec_write} via_write_slot={self.via_write_slot} "
            f"hist_write={self.hist_write} (staged={self._hist_stage is not None}) rows_prefetch={self.rows_prefetch}"
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
    # export mirror (producer): staging bound into the prefill traces + the chunk observer
    # ----------------------------------------------------------------------------------------------------------- #
    def _bind_export_staging(self, g, kv, bpc) -> None:
        """Allocate one head-major staging tensor per K/V part and bind them into the attention layers (the prefill
        traces captured afterwards fill_cache each chunk's K/V there); warm the fill programs; install the observer.
        Skipped (with a warning; the paged-cache gather stays) when the model's chunking cannot match the wire
        chunks: batched prefill (max_batch_size > 1), a chunk trace of another size, or a bucket wider than a chunk."""
        if self._export_staging:
            return
        args = getattr(self.model, "args", None)
        mbs = int(getattr(args, "max_batch_size", 1) or 1)
        ct = bpc * g["block_size"]
        mcs = int(getattr(self.model, "_chunked_chunk_size", None) or 2048)
        buckets = tuple(int(b) for b in getattr(self.model, "_PREFILL_MASK_BUCKETS", ()))
        why = None
        if mbs != 1:
            why = f"batched prefill model (max_batch_size {mbs})"
        elif mcs != ct:
            why = f"model chunk trace {mcs} tokens != wire chunk {ct}"
        elif buckets and max(buckets) > ct:
            why = f"masked bucket {max(buckets)} wider than the wire chunk {ct}"
        elif not hasattr(self.model, "set_prefill_chunk_observer"):
            why = "model has no set_prefill_chunk_observer"
        if why is not None:
            logger.warning(f"PD: export mirror disabled ({why}); export_request_state gathers from the paged cache")
            return
        cache_dtype = kv[0][1].dtype
        st_shape = (1, g["heads"], ct, g["head_dim"])
        rep = self._rep()
        for name, _ in kv:
            self._export_staging[name] = ttnn.from_torch(
                torch.zeros(*st_shape, dtype=torch.bfloat16),
                dtype=cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
        for i, a in enumerate(self.attn_layers):
            a._pd_export_k = self._export_staging[f"kv.L{i}.k"]
            a._pd_export_v = self._export_staging[f"kv.L{i}.v"]
        # every fill width the traces produce: the full chunk and each masked bucket (rows [0, S) of the staging)
        # Warm fill_cache(staging <- [1, heads, S, hd]) for EVERY fill width the attention layer can produce: the full
        # chunk and each bucket (the traced replays) AND every 64-token multiple below the chunk -- the eager masked
        # forward of an un-traced bucket (QWEN36_PREFILL_BUCKET_TRACE naming a subset) fills page_len = blocks x 64
        # (model._warmup_paged_fill_widths sweeps the same widths for paged_fill_cache). A width missed here would
        # compile inside the model forward after the traces are parked (the compile-clobbers-trace hang class; outside
        # any _guard). fill_cache hashes on both tensor specs incl. the memory config, so the input is built in the
        # layer's prefill memory config (_pf_mc: DRAM at TP=1), exactly as forward_prefill_paged's k_fill.
        pf_mc = getattr(self.attn_layers[0], "_pf_mc", ttnn.DRAM_MEMORY_CONFIG)
        bs = g["block_size"]
        widths = sorted(set(buckets) | {ct} | {w * bs for w in range(1, ct // bs + 1)})
        st0 = self._export_staging[kv[0][0]]
        t_w = time.perf_counter()
        for S in widths:
            x = ttnn.from_torch(
                torch.zeros(1, g["heads"], S, g["head_dim"], dtype=torch.bfloat16),
                dtype=cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=pf_mc,
                mesh_mapper=rep,
            )
            self._note_shape("fill_cache_export", (st_shape, S, dtype_name(cache_dtype)))
            ttnn.fill_cache(st0, x, 0)
            ttnn.deallocate(x)
        ttnn.synchronize_device(self.mesh)
        self._install_observer()
        logger.info(
            f"[PD] export mirror bound: {len(self._export_staging)} staging tensors {st_shape} "
            f"({len(self._export_staging) * tile_nbytes(st_shape, dtype_name(cache_dtype)) / 2**20:.0f} MiB), "
            f"{len(widths)} fill widths ({widths[0]}..{widths[-1]}, every {bs}) warmed in "
            f"{1e3 * (time.perf_counter() - t_w):.0f} ms"
        )

    def _install_observer(self) -> None:
        if self._observer_installed:
            return
        self._chunk_sync = os.environ.get("TT_PD_CHUNK_SYNC", "1") != "0"
        self.model.set_prefill_chunk_observer(
            self._on_prefill_chunk, sync_each_chunk=self._chunk_sync, needs_sync=self._needs_chunk_sync
        )
        self._observer_installed = True
        logger.info(
            f"[PD] prefill chunk observer installed (sync_each_chunk={self._chunk_sync}, "
            f"chunk pump {'on' if self._chunk_sync else 'OFF (needs the sync)'})"
        )

    def _needs_chunk_sync(self) -> bool:
        """Model callback before a chunk boundary: synchronize the device first? Only when the boundary pump could
        send something (a published export is waiting for its claim, ``_pump_wanted``); the mirror copies need no
        sync (in-order cq), so a request without a pending handoff keeps the traced loop's pipelining."""
        if self._chunk_pump is None or not self._chunk_sync:
            return False
        wants = self._pump_wanted
        return True if wants is None else bool(wants())

    def begin_export(self, block_ids, num_tokens: int, sinks) -> bool:
        """Producer step BEGIN (before the prefill of this request runs): remember the open export so the chunk
        observer can copy every mirrored chunk into ``sinks`` as the prefill goes. Returns True when the mirror path
        is active (staging bound), False when ``export_request_state`` will gather everything at step end."""
        g = self._geometry()
        nblk = cdiv(int(num_tokens), g["block_size"])
        ids = [int(b) for b in block_ids[:nblk]]
        assert len(ids) == nblk, f"{len(block_ids)} blocks for {num_tokens} tokens (need {nblk})"
        prev = self._active_export
        if prev is not None and prev.sinks is not sinks:
            # one export window at a time (the prefill rank runs one request per step): the stale window is dropped;
            # its export then gathers and, if any chunk was already mirrored into its sinks, FAILS on the sink's
            # write-twice check -> the consumer recomputes (never a wrong K/V)
            logger.warning(
                f"PD: begin_export while an export window is open ({prev.num_tokens} tokens, "
                f"{len(prev.written)}/{prev.nchunks} chunks mirrored): replacing it"
            )
        self._active_export = _ActiveExport(
            ids, int(num_tokens), nblk, cdiv(nblk, self._bpc()), sinks, set(), time.perf_counter()
        )
        return bool(self._export_staging)

    def end_export(self, sinks=None) -> None:
        """Close the export window; with ``sinks`` only when it is the open window's (a stale close never drops a
        newer window)."""
        if sinks is None or (self._active_export is not None and self._active_export.sinks is sinks):
            self._active_export = None

    def set_chunk_pump(self, fn, wants=None) -> None:
        """``fn()`` runs on the engine thread at every non-final prefill chunk boundary (the worker's transport pump:
        the claim-gated sends of an earlier export go out mid-prefill instead of at the step's end) -- only while the
        per-chunk device sync is on (``TT_PD_CHUNK_SYNC``, see the module docstring). ``wants()`` (host-only, optional)
        says whether the pump could send anything now; it gates the per-chunk sync (``_needs_chunk_sync``)."""
        self._chunk_pump = fn
        self._pump_wanted = wants

    def _on_prefill_chunk(self, *, chunk_start: int, n_tokens: int, final: bool) -> None:
        """Model chunk observer (engine thread; device ops). Copies the mirrored chunk into the open export's pool
        buffers (32 head-major device copies, enqueued before the next chunk's replay: the in-order cq runs them
        between the two chunks), then pumps the transport."""
        exp = self._active_export
        if exp is not None and self._export_staging:
            ct = self._chunk_tokens
            c = chunk_start // ct
            if chunk_start % ct == 0 and c < exp.nchunks and c not in exp.written:
                # The whole staging is copied. Rows past the request's last real block are stale filler (the previous
                # chunk's or request's K/V) that the consumer's chunk page table maps to its pad block, never to a
                # real block; unlike the gather path's zero filler the wire bytes are therefore not identical across
                # two exports of one prompt (only a byte-level comparison of the wire would notice).
                t0 = time.perf_counter()
                try:
                    with self._guard("mirror_chunk"):
                        for name, _ in self._kv_tensors():
                            exp.sinks[name].write_from_device(self._export_staging[name], chunk=c, blocking=False)
                    exp.written.add(c)
                    self.mirrored_chunks += 1
                    if self.timing:
                        logger.info(
                            f"[PD_TIMING] mirror chunk {c}/{exp.nchunks} ({n_tokens} tokens): "
                            f"{1e3 * (time.perf_counter() - t0):.2f} ms"
                        )
                except Exception:
                    # a part written before the failure is noted in its sink: the step-end gather of this chunk then
                    # fails the export (write twice) and the consumer recomputes -- never a wrong K/V
                    logger.exception(f"PD: mirror copy of chunk {c} failed; export_request_state gathers it")
        pump = self._chunk_pump
        if pump is not None and not final and self._chunk_sync:
            # only with the per-chunk sync: the marker the pump writes must follow sends that run right after THIS
            # finished chunk, not behind the replays the host queued ahead (TT_PD_CHUNK_SYNC, module docstring)
            try:
                pump()
            except Exception:
                logger.exception("PD: chunk-boundary transport pump raised")

    # ----------------------------------------------------------------------------------------------------------- #
    # GDN taps: batched device read (producer) and update_cache row writes (consumer)
    # ----------------------------------------------------------------------------------------------------------- #
    def _read_taps_batched(self, dns, slot: int):
        """All conv taps of ``slot`` as ONE host tensor [n_layers, K, D] bf16: stack the n_layers x K ``[1, B, D]``
        TILE tap tensors along dim 0 (tile-row stacking, one concat), untilize + unpad on device, one D2H read
        (2 ms for 192 rows vs 192 blocking 640 KiB tile reads + host untilizes = 30 ms)."""
        parts = [dn.conv_states[m] for dn in dns for m in range(dn.K)]
        R = len(parts)
        B = int(parts[0].shape[1])
        D = int(parts[0].shape[-1])
        assert 0 <= int(slot) < B, f"slot {slot} outside [0, {B})"
        self._note_shape("taps_concat", (R, B, D))
        cat = ttnn.concat(parts, dim=0)  # [R, B, D] TILE (B rows per part live in one tile row)
        self._note_shape("taps_untilize", (R, B, D))
        rm = ttnn.untilize_with_unpadding(cat, (R - 1, B - 1, D - 1))
        rows = ttnn.to_torch(rm)  # [R, B, D] bf16 host
        ttnn.deallocate(rm)
        ttnn.deallocate(cat)
        K = int(dns[0].K)
        return rows.reshape(R, B, D)[:, int(slot), :].reshape(len(dns), K, D).to(torch.bfloat16).contiguous()

    def _alloc_taps_stage(self, g, n_layers: int, K: int) -> None:
        if self._taps_stage_rm is not None:
            return
        D = g["conv_dim"]
        NS = self.taps_split
        assert (
            NS >= 1 and D % NS == 0 and (D // NS) % 32 == 0
        ), f"TT_PD_TAPS_SPLIT={NS} must split D={D} into tile-aligned slabs"
        # update_cache stages input_rows x slab tiles per core: measured on chip 0 (laneB/mb_view*.log) slab 256
        # (NS=40) 0.5 MiB OK, 512 (NS=20) OK, 1280 (NS=8) 3.4 MiB and the unsplit 10240-wide row 26 MiB overflow L1
        assert (
            D // NS <= 512
        ), f"TT_PD_TAPS_SPLIT={NS}: slab {D // NS} > 512 columns overflows the core's L1 in update_cache"
        R32 = cdiv(n_layers * K, 32) * 32
        self._taps_stage_rm = ttnn.from_torch(
            torch.zeros(1, 1, R32, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._rep(),
        )

    def _alloc_hist_stage(self, dns) -> None:
        # TP=1 only: the staging and the host pack are replicated (``_rep``) where the legacy path shards the packed
        # row over dim 0 (``ShardTensorToMesh``); the two write the same bytes only on a 1x1 mesh. A TP>1 consumer
        # takes the legacy path (the warmup does not allocate this staging, so install_gdn_state falls back).
        n_dev = self.mesh.get_num_devices()
        assert n_dev == 1, f"batched hist write is TP=1 only ({n_dev} devices)"
        if self._hist_stage is not None:
            return
        dn0 = dns[0]
        L, Nv, K = len(dns), int(dn0.Nv), int(dn0.K)
        for dn in dns:
            assert tuple(int(d) for d in dn.conv_hist_packed.shape)[1:] == (Nv, K, 32, 32), tuple(
                dn.conv_hist_packed.shape
            )
        self._hist_host = {p: torch.zeros(L, Nv, K, 32, 32, dtype=torch.bfloat16) for p in (0, 1)}
        z = torch.zeros(1, Nv * K, 32, 32, dtype=torch.bfloat16)
        self._hist_stage = [
            ttnn.from_torch(
                z,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._rep(),
            )
            for _ in range(L)
        ]

    @staticmethod
    def _pack_hist_into(out, dn, rows, parity: int) -> None:
        """rows [L, K, D] bf16 -> out [L, Nv, K, 32, 32]: row 2c + parity of tile (l, h, m) = channel chunk c of head h's
        [q(hk) | k(hk) | v(h)] row of tap m -- the bytes ``dn._pack_rows_vec`` + permute(1, 0, 2, 3) produce per layer
        (``_packed_slot_tensor``), for every layer in one vectorized copy. Only the parity rows are written: the other
        rows of ``out`` (the persistent per-parity host buffer) are zero from allocation and never touched."""
        Nv, Nk, Dk, Dv = int(dn.Nv), int(dn.Nk), int(dn.Dk), int(dn.Dv)
        kd = Nk * Dk
        L, K = int(rows.shape[0]), int(rows.shape[1])
        hk = torch.arange(Nv) // (Nv // Nk)
        rows = rows.to(torch.bfloat16)
        q = rows[..., :kd].reshape(L, K, Nk, Dk)[:, :, hk, :]
        k = rows[..., kd : 2 * kd].reshape(L, K, Nk, Dk)[:, :, hk, :]
        v = rows[..., 2 * kd : 2 * kd + Nv * Dv].reshape(L, K, Nv, Dv)
        chunks = torch.cat([q, k, v], dim=-1).reshape(L, K, Nv, -1, 32)  # [L, K, Nv, n, 32]
        n = int(chunks.shape[-2])
        assert 2 * n <= 32, f"{n} channel chunks per head do not fit a parity-interleaved 32-row tile"
        out[:, :, :, parity : 2 * n + parity : 2, :] = chunks.permute(0, 2, 1, 3, 4)

    def _write_hist_batched(self, dns, rows, slot: int, tm=None):
        """rows [n_layers, K, D] bf16 host -> packed conv history row ``slot`` (parity ``slot & 1``) of every layer, in
        place: one vectorized host pack of all layers into the persistent parity buffer, then per layer a host tilize
        (``from_torch`` TILE of a [1, Nv*K, 32, 32] view: 2 KiB pages, unlike a ROW_MAJOR [.., 32] upload whose 64 B
        pages make the H2D 5x slower), one H2D copy into the layer's persistent TILE staging and ONE ``fill_cache``
        over the zero-copy [B, Nv*K, 32, 32] view of ``conv_hist_packed`` (``batch_idx`` = slot is a runtime arg).
        Returns the host tensors, to keep alive until the device is synchronized."""
        dn0 = dns[0]
        L, Nv, K = len(dns), int(dn0.Nv), int(dn0.K)
        assert self._hist_stage is not None, "warmup_kv_transfer(role=consumer) must run first"
        assert self.mesh.get_num_devices() == 1, "batched hist write is TP=1 only (see _alloc_hist_stage)"
        assert tuple(int(d) for d in rows.shape[:2]) == (L, K), tuple(rows.shape)
        slot = int(slot)
        t0 = time.perf_counter()
        host = self._hist_host[slot & 1]
        self._pack_hist_into(host, dn0, rows, slot & 1)
        t_pack = time.perf_counter()
        hs = []
        for j, dn in enumerate(dns):
            buf = dn.conv_hist_packed
            B = int(buf.shape[0])
            assert 0 <= slot < B, f"slot {slot} outside [0, {B})"
            h = ttnn.from_torch(host[j].view(1, Nv * K, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            hs.append(h)
            ttnn.copy_host_to_device_tensor(h, self._hist_stage[j])
            cache = ttnn.experimental.view(buf, (B, Nv * K, 32, 32))
            self._note_shape("fill_cache_hist", ((B, Nv * K, 32, 32), slot))
            ttnn.fill_cache(cache, self._hist_stage[j], slot)
        if tm is not None:
            tm["pack"] += t_pack - t0
            tm["hist"] += time.perf_counter() - t_pack
        return hs

    def _write_taps_update_cache(self, dns, rows, slot: int):
        """rows [n_layers, K, D] bf16 host -> row ``slot`` of every ``conv_states[m]`` of every layer, in place:
        ONE compact ROW_MAJOR upload into the persistent staging, one device tilize, one tile-aligned slice per 32
        rows, and one ``update_cache`` per tap row over zero-copy width-split views (``[1, NS, rows, D/NS]``: the
        same linear tile order as ``[1, rows, D]``, so the op stages ``32 x D/NS`` tiles per core instead of the
        whole 10240-wide row, which does not fit L1). Exact: the kernel untilizes the tile row, overwrites the bf16
        row bytes and retilizes. Returns the host tensor, to keep alive until the device is synchronized."""
        n, K, D = (int(d) for d in rows.shape)
        R = n * K
        assert self._taps_stage_rm is not None, "warmup_kv_transfer(role=consumer) must run first"
        R32 = int(self._taps_stage_rm.shape[2])
        assert R <= R32 and int(self._taps_stage_rm.shape[-1]) == D, (R, R32, D)
        t0 = time.perf_counter()
        host = torch.zeros(1, 1, R32, D, dtype=torch.bfloat16)
        host[0, 0, :R] = rows.reshape(R, D).to(torch.bfloat16)
        h = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        t_host = time.perf_counter()
        ttnn.copy_host_to_device_tensor(h, self._taps_stage_rm)
        t_h2d = time.perf_counter()
        self._note_shape("taps_tilize", (R32, D))
        tiled = ttnn.tilize(self._taps_stage_rm)
        NS = self.taps_split
        WS = D // NS
        tile_rows = []
        views = []
        for t in range(R32 // 32):
            self._note_shape("taps_tilerow_slice", (R32, D, t))
            x = ttnn.slice(tiled, (0, 0, 32 * t, 0), (1, 1, 32 * t + 32, D))
            tile_rows.append(x)
            views.append(ttnn.experimental.view(x, (1, NS, 32, WS)))
        slot = int(slot)
        for j, dn in enumerate(dns):
            for m in range(dn.K):
                r = j * dn.K + m
                conv = dn.conv_states[m]
                # the width-split view keeps the linear tile order only while every slot row lives in ONE tile row
                assert (
                    int(conv.shape[1]) <= 32
                ), f"conv_states rows {int(conv.shape[1])} > 32: the width-split view would interleave tile rows"
                cache = ttnn.experimental.view(conv, (1, NS, int(conv.shape[1]), WS))
                self._note_shape("update_cache_tap", (tuple(int(d) for d in cache.shape), slot))
                ttnn.update_cache(cache, views[r // 32], slot, batch_offset=r % 32)
        for x in tile_rows:
            ttnn.deallocate(x)
        ttnn.deallocate(tiled)
        if self.timing:
            logger.info(
                f"[PD_TIMING] taps update_cache x{R}: host prep {1e3 * (t_host - t0):.1f} + H2D {1e3 * (t_h2d - t_host):.1f} "
                f"+ device ops enqueue {1e3 * (time.perf_counter() - t_h2d):.1f} ms"
            )
        return h

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
            num_blocks = g["num_blocks"]
            kv = self._kv_tensors()
            tmp = []
            t0 = time.perf_counter()
            # chunks the prefill's mirror already copied into these very sinks (begin_export -> on_prefill_chunk)
            exp = self._active_export
            self._active_export = None
            mirrored = set()
            if exp is not None and exp.sinks is sinks and exp.num_tokens == int(num_tokens) and exp.block_ids == ids:
                mirrored = set(exp.written)
            elif exp is not None:
                logger.warning(
                    f"PD: open export ({exp.num_tokens} tokens, {len(exp.written)} mirrored chunks) does not match this "
                    f"export_request_state ({num_tokens} tokens); gathering every chunk from the paged cache"
                )
            nchunks = cdiv(nblk, bpc)
            self.gathered_chunks += nchunks - len(mirrored)
            for c in range(nchunks):
                if c in mirrored:
                    continue
                chunk_ids = ids[bpc * c : bpc * c + bpc]
                regions = getattr(sinks[kv[0][0]], "supports_regions", False)
                if regions:
                    blk_nbytes = tile_nbytes((1, g["heads"], g["block_size"], g["head_dim"]), g["kv_dtype"])
                    for name, t in kv:
                        sink = sinks[name]
                        for k0, b0, n in _contiguous_runs(chunk_ids):
                            sink.write_region_from_device(
                                t,
                                b0 * blk_nbytes,
                                n * blk_nbytes,
                                chunk=c,
                                dst_offset_bytes=k0 * blk_nbytes,
                                blocking=False,
                            )
                    continue
                # dumpfile: one runtime-start unit slice per (block, tensor) + one bpc-way concat per tensor; the block
                # index is set once and reused by all 32 tensors (in-order cq). Blocking sink writes so the chunk's
                # temporaries can be freed right away (a 32k prompt would otherwise pin 2.3 GB of them).
                # Filler rows past the request's blocks are the persistent zero block (warmup_kv_transfer), never a
                # slice of the live pad block (see there): the exported bytes depend on the request alone.
                assert self._zero_blk is not None, "warmup_kv_transfer(role=producer, mode=dumpfile) must run first"
                parts = {name: [] for name, _ in kv}
                keep = []
                for b in chunk_ids:
                    keep.append(self._upload_starts(b))
                    for name, t in kv:
                        parts[name].append(self._unit_slice(t, num_blocks))
                for name, t in kv:
                    blk = ttnn.concat(parts[name] + [self._zero_blk] * (bpc - len(chunk_ids)), dim=0)
                    self._note_shape("concat_chunk", (bpc, tuple(int(d) for d in blk.shape), dtype_name(blk.dtype)))
                    sinks[name].write_from_device(blk, chunk=c, blocking=True)
                    ttnn.deallocate(blk)
                    for p in parts[name]:
                        ttnn.deallocate(p)
                del keep
            t1 = time.perf_counter()
            dns = self.gdn_layers
            taps = self._read_taps_batched(dns, slot)  # [n_layers, K, D] bf16, one device read for every layer
            t_taps = time.perf_counter()
            for j, dn in enumerate(dns):
                rs = sinks[f"gdn.L{j}.rec"]
                if dn.B == 1:
                    rs.write_from_device(dn.rec_state, chunk=0, blocking=False)
                elif getattr(rs, "supports_regions", False):
                    row_nbytes = tile_nbytes((1,) + g["rec_shape"], g["rec_dtype"])
                    rs.write_region_from_device(
                        dn.rec_state, slot * row_nbytes, row_nbytes, chunk=0, dst_offset_bytes=0, blocking=False
                    )
                else:
                    h = ttnn.from_device(dn.rec_state)  # whole [B, Nv, Dk, Dv] (24 MiB at B=8), blocking
                    row = ttnn.from_torch(
                        ttnn.to_torch(h)[slot : slot + 1], dtype=dn.rec_state.dtype, layout=ttnn.TILE_LAYOUT
                    )
                    rs.write_host(row, chunk=0)
            t_rec = time.perf_counter()
            for j in range(len(dns)):
                # .clone(): a fresh 80 KiB storage. taps[j] is a view of the [n_layers, K, D] batch and torch.save (the
                # dumpfile taps sink) serializes a view's WHOLE storage: 48 x 3.75 MiB files = 50 ms (py-spy, 10:32).
                sinks[f"gdn.L{j}.taps"].write_rows(taps[j].clone())
            t_rows = time.perf_counter()
            ttnn.synchronize_device(self.mesh)
            t_sync = time.perf_counter()
            for t in tmp:
                ttnn.deallocate(t)
            if self.timing:
                logger.info(
                    # `kv X ms, gdn Y ms` first: profiles/pd/p3f_handoff_table.py parses exactly that prefix
                    f"[PD_TIMING] export T={num_tokens} nblk={nblk} slot={slot} B={dn.B}: kv {1e3 * (t1 - t0):.1f} ms, "
                    f"gdn {1e3 * (t_sync - t1):.1f} ms (mirrored {len(mirrored)}/{nchunks} chunks; gdn = taps read "
                    f"{1e3 * (t_taps - t1):.1f} + rec copies {1e3 * (t_rec - t_taps):.1f} + rows files "
                    f"{1e3 * (t_rows - t_rec):.1f} + sync {1e3 * (t_sync - t_rows):.1f})"
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
                    self._note_shape(
                        "paged_fill_cache",
                        (tuple(int(d) for d in t.shape), tuple(int(d) for d in st.shape), dtype_name(t.dtype)),
                    )
                    ttnn.experimental.paged_fill_cache(t, st, pt, batch_idx=0)
                ttnn.deallocate(pt)
                done += 1
            if self.timing:
                logger.info(
                    f"[PD_TIMING] import_kv_blocks T={num_tokens} chunks={list(chunks)}: {1e3 * (time.perf_counter() - t0):.1f} ms"
                )
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
            for name, spec, nbytes in (
                (f"gdn.L{j}.rec", rec_spec, rec_nbytes),
                (f"gdn.L{j}.taps", taps_spec, taps_nbytes),
            ):
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
        if self.rows_prefetch:
            # Host-only: load the taps rows now (the worker calls this right after enqueueing the K/V fills and before
            # its device sync, so the ~48 row-file reads overlap the fills instead of sitting inside install_gdn_state
            # at the join step). Cached per sources dict; install_gdn_state pops the entry.
            want = (g["K"], g["conv_dim"])
            rows = []
            for j in range(len(self.gdn_layers)):
                r = sources[f"gdn.L{j}.taps"].read_rows().to(torch.bfloat16)
                if tuple(r.shape) != want:
                    raise RuntimeError(f"PD: gdn.L{j}.taps rows {tuple(r.shape)} != expected {want}")
                rows.append(r)
            self._rows_cache[id(sources)] = (sources, rows)
            while len(self._rows_cache) > self._ROWS_CACHE_MAX:  # a job that failed after validate never installs
                self._rows_cache.popitem(last=False)

    def install_gdn_state(self, sources, slot: int) -> None:
        """Write the rec row + 4 conv tap rows (+ the packed conv history on a fused-conv decode) into ``slot`` of every
        GDN layer, in place. JOIN-step step-BEGIN hook only (I11): the composite TP=1 decode advances every idle row's
        taps and, inside the pow2 bucket, rewrites its rec row, so an installed row must decode in the very next
        forward. Device synchronized on return (I4); every host buffer is referenced until then."""
        with self._guard("install_gdn_state"):
            dns = self.gdn_layers
            fresh = []
            keep = []
            rows_all = []
            slot = int(slot)
            t0 = time.perf_counter()
            batched_taps = self.taps_write == "update_cache" and not self.via_write_slot
            # packed conv history of a fused-conv decode: batched = one vectorized host pack of all layers, then per layer
            # a host tilize + one TILE H2D into persistent staging + one fill_cache, after the loop (_write_hist_batched);
            # legacy = per layer host pack/tilize + fresh upload + _write_index (slice/slice/concat/copy)
            batched_hist = (
                self.hist_write == "batched"
                and not self.via_write_slot
                and self._hist_stage is not None  # allocated by the warmup only on a single-device mesh
                and all(getattr(dn, "_decode_fused_conv", False) for dn in dns)
            )
            tm = {"rec": 0.0, "rows": 0.0, "pack": 0.0, "hist": 0.0, "taps": 0.0}
            cached = self._rows_cache.pop(id(sources), None)
            pre_rows = cached[1] if cached is not None and cached[0] is sources else None
            for j, dn in enumerate(dns):
                assert 0 <= slot < dn.B, f"slot {slot} outside [0, {dn.B})"
                ta = time.perf_counter()
                src = sources[f"gdn.L{j}.rec"].chunk(0)
                dev_tensor = getattr(src, "device_tensor", None)
                if callable(dev_tensor):
                    r = dev_tensor()  # fabric: the received pool buffer itself (valid until finish_import); no copy
                elif getattr(src, "is_device_readable", False):
                    r = self._rec_staging
                    assert r is not None, "warmup_kv_transfer(role=consumer) must run first"
                    src.read_into_device(r)
                else:
                    r = src.read_device(
                        self.mesh
                    )  # dumpfile: load_tensor -> fresh DRAM tensor (outside the trace region)
                    fresh.append(r)
                if r.dtype != dn.rec_state.dtype:
                    rc = ttnn.typecast(r, dn.rec_state.dtype)
                    fresh.append(rc)
                    r = rc
                tb = time.perf_counter()
                if pre_rows is not None:
                    rows = pre_rows[j]  # loaded by validate_gdn_parts
                else:
                    rows = sources[f"gdn.L{j}.taps"].read_rows().to(torch.bfloat16)  # [K, D] host
                keep.append(rows)
                tc = time.perf_counter()
                tm["rows"] += tc - tb
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
                td = time.perf_counter()
                tm["rec"] += (tb - ta) + (td - tc)
                if batched_taps or batched_hist:
                    rows_all.append(rows)  # written below, all layers in one pass
                if not batched_taps:
                    for m in range(dn.K):
                        c = self._tap_row_tensor(rows[m], dn)
                        self._note_shape("write_index_tap", (tuple(int(d) for d in dn.conv_states[m].shape), slot))
                        dn._write_index(
                            dn.conv_states[m], c, slot, dim=1
                        )  # exact host-path primitive (write_slot's tap path)
                if getattr(dn, "_decode_fused_conv", False):
                    # fused-conv decode (QWEN36_GDN_DECODE_FUSED=2, AM3 M2): the op reads conv_hist_packed[slot] (parity
                    # slot & 1) inside the decode trace, so the row we just installed must also land there. Never a full
                    # rebuild here (it would revert every live user's history from their stale conv_states, I4).
                    if not dn._hist_packed_valid or dn.conv_hist_packed is None:
                        raise RuntimeError(f"PD: GDN layer {j}: conv_hist_packed invalid while slots are live")
                    if not batched_hist:
                        te = time.perf_counter()
                        packed = self._packed_slot_tensor(dn, rows, slot)
                        tf = time.perf_counter()
                        self._note_shape("write_index_packed", (tuple(int(d) for d in dn.conv_hist_packed.shape), slot))
                        dn._write_index(dn.conv_hist_packed, packed, slot, dim=0)
                        tm["pack"] += tf - te
                        tm["hist"] += time.perf_counter() - tf
            t_layers = time.perf_counter()
            stacked = torch.stack(rows_all) if rows_all else None
            if batched_taps and rows_all:
                keep.append(self._write_taps_update_cache(dns, stacked, slot))
            t_taps = time.perf_counter()
            tm["taps"] = t_taps - t_layers
            if batched_hist:
                keep.append(self._write_hist_batched(dns, stacked, slot, tm))
            t_hist = time.perf_counter()
            ttnn.synchronize_device(self.mesh)  # every H2D done before any host buffer above goes away
            t_sync = time.perf_counter()
            for t in fresh:
                ttnn.deallocate(t)
            del keep
            self.last_install_timing = dict(
                total=time.perf_counter() - t0,
                layers=t_layers - t0,
                sync=t_sync - t_hist,
                hist_mode="batched" if batched_hist else "legacy",
                rows_prefetched=pre_rows is not None,
                **tm,
            )
            if self.timing:
                logger.info(
                    f"[PD_TIMING] install_gdn_state slot={slot}: {1e3 * (time.perf_counter() - t0):.1f} ms = layer loop "
                    f"{1e3 * (t_layers - t0):.1f} + taps {1e3 * (t_taps - t_layers):.1f} + hist "
                    f"{1e3 * (t_hist - t_taps):.1f} + sync {1e3 * (t_sync - t_hist):.1f} (hist={'batched' if batched_hist else 'legacy'}: "
                    f"rec {1e3 * tm['rec']:.1f}, rows read {1e3 * tm['rows']:.1f}, hist pack {1e3 * tm['pack']:.1f}, hist "
                    f"write {1e3 * tm['hist']:.1f} ms; rows {'prefetched' if pre_rows is not None else 'read here'})"
                )

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
