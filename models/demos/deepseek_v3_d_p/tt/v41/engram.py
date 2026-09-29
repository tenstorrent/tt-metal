# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 Engram (graph nodes N2-N5; bead F5, dev-spec D-D host-first placement).

Reference (``inference/model.py`` ``Engram`` / ``ParallelEngramEmbedding``, ``inference/engram.py``
``NgramHashState``), applied to the 4 residual streams **before** the block of each Engram layer (1, 14):

* N2 hash (host, token-only): tokens -> compressed ids (tokenizer-normalized map; image tokens -> DEAD) ->
  per position the 2..4-grams ending there (look-back stops at the sequence start and at DEAD, filled with
  the pad id) -> per (layer, n-gram, head) a prime-bucket hash id: ``[L, n_layers, 24]``. N-grams span chunk
  boundaries, so the last 3 compressed ids of the previous chunk are carried (graph.md §4 rule 5); padded
  tail tokens are never hashed and never enter the history (rule 8).
* N3 lookup: FP8 e4m3 rows with E8M0 per-32 scales, dequantized on device exactly as the reference does -> bf16
  ``[L, 24 * 256]``. Two table placements, chosen at construction, same rows (the sign of zero aside):
  host table (``V41EngramTable``): the host gathers the stored rows and uploads them packed (1 byte per value,
  1/chips of the lookups per chip), the device decodes and all-gathers over TP; device table (``TtV41EngramTable``,
  row-sharded over all chips): only row ids go up, every chip gathers its shard's rows (zero row elsewhere), the
  byte sum over the mesh is reduce-scattered, decoded and all-gathered over TP. Both prepare steps depend on
  tokens only (prefetchable).
* N4 wkv (device): FP8 linear 6144 -> (hc_mult + 1) * hidden with the reference's FP8 activation QDQ;
  its output columns are permuted at load so each TP chip gets its hidden slice of the 4 keys and the value.
* N5 gate/add (device): per stream copy ``gate = sigmoid(signed_sqrt(rstd(h) rstd(key) <h * q_w * k_w, key>
  / sqrt(dim)))`` (fp32; the three per-copy sums are TP partials, all-gathered and summed in a fixed order),
  zero where the token mask is False (image tokens); ``h += gate * value``.

Streams are ``[1, 1, S/sp, hc_mult * hidden/tp]`` fp32 (per chip its hidden slice of every stream, copy-major),
as in ``tt/v41/mhc.py``; the output stays fp32 (the reference rounds it to bf16, dev-spec D-E).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.engram import (
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp8_qdq

DEAD = -1  # compressed id of a token that takes no part in any n-gram (reference NgramHashState.DEAD)
SCALE_BLOCK = 32  # E8M0 scale per 32 table values (reference fp8_block_size)
GATE_CLAMP = 1e-6  # reference Engram.clamp_value
STAT_WIDTH = 32  # per-chip partial sums padded to one tile row

ENGRAM_COMPUTE_CONFIG = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# Packed row: uint16 containers; container k < head_dim/2 holds value bytes k (low) and k + head_dim/2 (high),
# container head_dim/2 + g the scale bytes g and g + groups/2; padded to whole tiles (and 64 B DRAM pages).
PACKED_WIDTH = 160
_SUBNORMAL_OFFSET = 2.0**-6  # e4m3 with the exponent field forced to 1 minus this = the subnormal value


def pack_rows(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Table rows as stored (float8_e4m3fn [..., head_dim], E8M0 [..., head_dim / 32]) -> uint16 [..., PACKED_WIDTH]
    (a byte rearrangement: the device decodes it bit-exactly, see ``_decode_packed``)."""
    half = weight.shape[-1] // 2
    data, scales = weight.view(torch.uint8), scale.view(torch.uint8)
    groups = scales.shape[-1] // 2
    assert half + groups <= PACKED_WIDTH and half % SCALE_BLOCK == 0, (weight.shape, scale.shape)
    # little-endian containers: byte 2k is the low byte, 2k + 1 the high byte
    packed = torch.zeros(*weight.shape[:-1], PACKED_WIDTH, 2, dtype=torch.uint8)
    packed[..., :half, 0], packed[..., :half, 1] = data[..., :half], data[..., half:]
    packed[..., half : half + groups, 0], packed[..., half : half + groups, 1] = (
        scales[..., :groups],
        scales[..., groups:],
    )
    return packed.flatten(-2).view(torch.uint16)


def _decode_packed(low: ttnn.Tensor, high: ttnn.Tensor, head_dim: int) -> ttnn.Tensor:
    """The low and high bytes of ``pack_rows`` containers, int32 [1, 1, N, PACKED_WIDTH] TILE each -> bf16
    [1, 1, N, head_dim]: the reference's
    ``fp32(e4m3) * fp32(2^(e8m0 - 127))`` per group of 32, cast to bf16 (exact: every step is exact in fp32 and the
    product has <= 4 significant bits). Equal as values; the sign of zero is not kept (the device writes e4m3 -0,
    code 0x80, as +0; no effect downstream)."""
    rows, half = low.shape[2], head_dim // 2
    groups = half // SCALE_BLOCK
    expand = torch.zeros(32, half)  # scale group g -> its 32 values (one-hot: exact for powers of two)
    for g in range(groups):
        expand[g, g * SCALE_BLOCK : (g + 1) * SCALE_BLOCK] = 1.0
    expand = ttnn.from_torch(
        expand[None, None],
        device=low.device(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(low.device()),
    )
    halves = []
    for byte in (low, high):
        value = ttnn.slice(byte, [0, 0, 0, 0], [1, 1, rows, half])
        scale_bits = ttnn.bitwise_left_shift(ttnn.slice(byte, [0, 0, 0, half], [1, 1, rows, half + 32]), 23)
        scale = ttnn.matmul(ttnn.bitcast(scale_bits, ttnn.float32), expand, compute_kernel_config=ENGRAM_COMPUTE_CONFIG)
        exponent = ttnn.bitwise_and(ttnn.bitwise_right_shift(value, 3), 15)
        bits = ttnn.add(
            ttnn.bitwise_left_shift(ttnn.add(ttnn.maximum(exponent, 1), 120), 23),
            ttnn.bitwise_left_shift(ttnn.bitwise_and(value, 7), 20),
        )
        subnormal = ttnn.multiply(ttnn.typecast(ttnn.eq(exponent, 0), ttnn.float32), _SUBNORMAL_OFFSET)
        magnitude = ttnn.subtract(ttnn.bitcast(bits, ttnn.float32), subnormal)
        negative = ttnn.bitwise_right_shift(value, 7)
        signed = ttnn.where(negative, ttnn.neg(magnitude), magnitude)
        halves.append(ttnn.multiply(signed, scale))
    return ttnn.typecast(ttnn.concat(halves, dim=-1), ttnn.bfloat16)


def engram_layout(config) -> EngramLayout:
    """The released bucket layout (primes per layer / n-gram / head) of every Engram layer of ``config``."""
    return EngramLayout.from_args(
        SimpleNamespace(
            engram_layer_ids=tuple(config.ENGRAM_LAYER_IDS),
            engram_max_ngram_size=config.ENGRAM_MAX_NGRAM_SIZE,
            engram_vocab_size=config.ENGRAM_VOCAB_SIZE,
            engram_n_heads=config.ENGRAM_N_HEADS,
            engram_head_dim=config.ENGRAM_HEAD_DIM,
            engram_num_embeddings=tuple(config.ENGRAM_NUM_EMBEDDINGS),
        )
    )


class V41EngramHash:
    """Host n-gram hashing of all Engram layers (N2). Stateless: the caller owns the per-request history."""

    def __init__(self, config, token_map: torch.Tensor):
        """``token_map``: [vocab] int64 compressed id of every token (``from_tokenizer`` builds it)."""
        self.config = config
        self.layout = engram_layout(config)
        vocab = int(token_map.max()) + 1
        # every multiplier derives from the compressed vocab size: a mismatch silently rehashes the table
        assert vocab == config.ENGRAM_COMPRESSED_VOCAB_SIZE, (vocab, config.ENGRAM_COMPRESSED_VOCAB_SIZE)
        self.token_map = token_map.to(torch.int64)
        self.pad_id = int(self.token_map[config.ENGRAM_PAD_ID])
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in self.layout.primes]
        self.primes = torch.tensor(self.layout.primes)  # [n_layers, max_ngram - 1, n_heads]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]))  # [n_layers, 24]
        self.multipliers = compute_hash_multipliers(self.layout.layer_ids, self.layout.max_ngram_size, vocab)

    @classmethod
    def from_tokenizer(cls, config, tokenizer) -> "V41EngramHash":
        token_map, _ = build_compressed_token_map(tokenizer)
        return cls(config, torch.tensor(token_map, dtype=torch.int64))

    @property
    def n_hash_cols(self) -> int:
        return (self.layout.max_ngram_size - 1) * self.layout.n_heads

    def layer_index(self, layer: int) -> int:
        """Position of V4.1 layer ``layer`` in the hash output's layer axis."""
        return self.layout.layer_ids.index(layer)

    def new_history(self) -> torch.Tensor:
        """The history of a fresh request (no tokens before position 0)."""
        return torch.empty(0, dtype=torch.int64)

    def __call__(
        self, tokens: torch.Tensor, history: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hash the valid tokens [L] of a chunk that follows ``history``.

        ``token_mask`` [L] bool: False for tokens that take no part in an n-gram (image spans); None = all text.
        Returns (hash ids [L, n_layers, n_hash_cols] int64 global table rows, the history after the chunk)."""
        n = self.layout.max_ngram_size
        compressed = self.token_map[tokens.to(torch.int64)]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, DEAD)
        seq = torch.cat([history, compressed])
        length, back = compressed.numel(), history.numel()
        # the reference blocks look-back at global position < shift; the history holds min(start, n - 1) ids,
        # so that is an index before the history's start here
        index = torch.arange(length) + back
        grams, blocked = [], torch.zeros(length, dtype=torch.bool)
        for shift in range(n):
            source = seq[(index - shift).clamp_min(0)]
            blocked = blocked | (index < shift) | (source == DEAD)
            grams.append(torch.where(blocked, self.pad_id, source))
        grams = torch.stack(grams, dim=-1)  # [L, n]
        # XOR of the multiplied ids, one look-back at a time: the running value after step i hashes the (i+1)-gram
        products = grams.unsqueeze(1) * self.multipliers  # [L, n_layers, n]
        rolling, hashes = products[..., 0], []
        for i in range(1, n):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets, seq[-(n - 1) :].clone()


class V41EngramTable:
    """One Engram layer's table on the host (N3), from the checkpoint's float8_e4m3fn rows ``[R, head_dim]`` and
    float8_e8m0fnu scales ``[R, head_dim / 32]``, held packed once (``pack_rows``: the same bytes, rearranged for
    the device, 320 B per row), so a chunk's upload is one row gather. ``row_ids`` (sorted int64): the global rows
    these ``R`` rows are, for a row subset (e.g. the rows a prompt needs); None = the full table."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, row_ids: torch.Tensor | None = None):
        assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu, (weight.dtype, scale.dtype)
        assert scale.shape == (weight.shape[0], weight.shape[1] // SCALE_BLOCK), (weight.shape, scale.shape)
        assert row_ids is None or row_ids.shape == (weight.shape[0],)
        self.head_dim, self.row_ids = weight.shape[1], row_ids
        self.rows = pack_rows(weight, scale)

    def _local(self, ids: torch.Tensor) -> torch.Tensor:
        if self.row_ids is None:
            return ids
        local = torch.searchsorted(self.row_ids, ids.contiguous()).clamp_max(len(self.row_ids) - 1)
        if not torch.equal(self.row_ids[local], ids):
            raise KeyError("Engram rows requested that this table subset does not hold")
        return local

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [...] int64 global rows -> bf16 [..., head_dim]: ``fp32(row) * fp32(scale)`` per 32, cast to bf16
        (the reference ``ParallelEngramEmbedding``)."""
        half, groups = self.head_dim // 2, self.head_dim // SCALE_BLOCK // 2
        byte = self.rows[self._local(ids)].view(torch.uint8).unflatten(-1, (PACKED_WIDTH, 2))
        weight = torch.cat([byte[..., :half, 0], byte[..., :half, 1]], dim=-1).view(torch.float8_e4m3fn)
        scale = torch.cat([byte[..., half : half + groups, 0], byte[..., half : half + groups, 1]], dim=-1)
        values = weight.float().unflatten(-1, (-1, SCALE_BLOCK)) * scale.view(torch.float8_e8m0fnu).float().unsqueeze(
            -1
        )
        return values.flatten(-2).to(torch.bfloat16)

    def packed(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        """Gather the packed rows of ids [N] int64 global rows into ``out`` uint16 [N, PACKED_WIDTH]."""
        torch.index_select(self.rows, 0, self._local(ids), out=out)


class TtV41EngramTable:
    """One Engram layer's table resident in device DRAM: rows as stored (``pack_rows``), row-sharded over all chips
    in (sp, tp) order, each shard followed by an all-zero row that out-of-shard lookups read."""

    def __init__(self, mesh_device, table: V41EngramTable):
        self.table = table
        self.chips = mesh_device.get_num_devices()
        rows = table.rows.shape[0]
        self.shard_rows = -(-rows // self.chips)
        sp, tp = mesh_device.shape
        packed = torch.zeros(self.chips * self.shard_rows, PACKED_WIDTH, dtype=torch.uint16)
        packed[:rows] = table.rows
        packed = packed.view(sp, tp, self.shard_rows, PACKED_WIDTH)
        packed = torch.cat([packed, torch.zeros(sp, tp, 1, PACKED_WIDTH, dtype=torch.uint16)], dim=2)
        # ttnn.embedding gathers bf16 rows: the containers travel as bf16 bit patterns and are never computed on
        self.rows = ttnn.from_torch(
            packed.view(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
        )

    def shard_ids(self, ids: torch.Tensor, count: int) -> torch.Tensor:
        """ids [N] int64 global rows, N <= count -> int32 [chips, count]: each chip's local row of every id, or its
        zero row where another chip holds the id (and for the padding beyond N)."""
        local = self.table._local(ids)
        out = torch.full((self.chips, count), self.shard_rows, dtype=torch.int32)
        out[local // self.shard_rows, torch.arange(ids.numel())] = (local % self.shard_rows).to(torch.int32)
        return out


@dataclass
class EngramChunkInputs:
    """Per-chunk device inputs of one Engram layer. ``lookup``: with a host table, this chip's share of the chunk's
    packed rows (uint16 [1, 1, chunk * n_hash_cols / chips, PACKED_WIDTH], token-major, (sp, tp) order); with a
    device table, this chip's local row ids of all the chunk's lookups (uint32 [1, 1, 1, chunk * n_hash_cols]).
    ``mask`` [1, 1, chunk/sp, 1] fp32 (1 = text, 0 = image token; None = all text)."""

    lookup: ttnn.Tensor
    mask: ttnn.Tensor | None


class TtV41Engram(LightweightModule):
    """One Engram layer. ``prepare`` (host, per chunk, prefetchable) uploads the chunk's packed rows (host table) or
    only its row ids (device-resident table); ``forward`` dequantizes the rows on device, runs wkv and the gated
    stream update. The two table placements are interchangeable at construction and give identical rows."""

    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        table: "V41EngramTable | TtV41EngramTable",
        topology=ttnn.Topology.Linear,
        weights_dtype=ttnn.bfloat8_b,
    ):
        """``weights``: ``wkv`` [(hc_mult + 1) * hidden, n_hash_cols * head_dim] (FP8 dequantized, checkpoint
        orientation [out, in]), ``q_weight`` and ``k_weight`` [hc_mult, hidden]; ``table``: this layer's rows, on the
        host (``V41EngramTable``) or resident on the device (``TtV41EngramTable``)."""
        assert layer in config.ENGRAM_LAYER_IDS, f"layer {layer} has no Engram"
        self.mesh_device, self.config, self.layer, self.table = mesh_device, config, layer, table
        self.sp, self.tp = mesh_device.shape
        self.hc, self.dim, self.eps = config.HC_MULT, config.EMB_SIZE, config.RMS_NORM_EPS
        assert self.dim % (32 * self.tp) == 0, f"hidden {self.dim} does not split over TP={self.tp} in tiles"
        self.local = self.dim // self.tp
        self.head_dim = config.ENGRAM_HEAD_DIM
        self.n_hash_cols = (config.ENGRAM_MAX_NGRAM_SIZE - 1) * config.ENGRAM_N_HEADS
        self.in_features = self.n_hash_cols * self.head_dim
        self.chips = self.sp * self.tp
        self.ccl = V41Collectives(mesh_device, topology)
        shape = tuple(mesh_device.shape)
        tp_cols = ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(None, 3))

        wkv = weights["wkv"].detach().to(torch.bfloat16)
        assert wkv.shape == ((self.hc + 1) * self.dim, self.in_features), wkv.shape
        # [key_0 .. key_{hc-1}, value] x hidden -> per TP chip its hidden slice of each, in that order
        per_chip = (
            wkv.view(self.hc + 1, self.tp, self.local, self.in_features).transpose(0, 1).reshape(-1, self.in_features)
        )
        self.wkv = ttnn.from_torch(
            per_chip.transpose(0, 1).contiguous()[None, None],
            device=mesh_device,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=tp_cols,
        )
        # only ever used as the fp32 product (reference); packed like the streams
        qk = weights["q_weight"].detach().float() * weights["k_weight"].detach().float()
        qk = qk.view(self.hc, self.tp, self.local).transpose(0, 1).reshape(1, 1, 1, -1)
        self.qk_weight = ttnn.from_torch(
            qk.contiguous(), device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=tp_cols
        )

    # --- host side -------------------------------------------------------------------------------------
    def _per_chip(self, host: torch.Tensor, dtype) -> ttnn.Tensor:
        """[chips, ...] -> chip (sp, tp) gets entry sp * tp_size + tp, row-major."""
        return ttnn.from_torch(
            host.reshape(self.sp, self.tp, *host.shape[1:]),
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(0, 1)),
        )

    def prepare(self, hash_ids: torch.Tensor, chunk: int, token_mask: torch.Tensor | None = None) -> EngramChunkInputs:
        """Upload one chunk's Engram inputs; depends on tokens only, so it can run ahead of the device.
        ``hash_ids`` [L, n_hash_cols]: this layer's ids of the chunk's valid tokens (rows beyond L are zero);
        ``token_mask`` [L] bool: False for image tokens (their gate is zero); None = all text."""
        lookups = chunk * self.n_hash_cols
        assert (
            lookups % (32 * self.chips) == 0
        ), f"chunk {chunk}: its lookups must split over {self.chips} chips in tiles"
        if isinstance(self.table, TtV41EngramTable):
            lookup = self._per_chip(self.table.shard_ids(hash_ids.flatten(), lookups)[:, None], ttnn.uint32)
        else:
            packed = torch.zeros(lookups, PACKED_WIDTH, dtype=torch.uint16)
            self.table.packed(hash_ids.flatten(), packed[: hash_ids.numel()])
            lookup = self._per_chip(packed.view(self.chips, lookups // self.chips, PACKED_WIDTH), ttnn.uint16)
        mask = None
        if token_mask is not None:
            full = torch.ones(chunk, 1)
            full[: token_mask.numel(), 0] = token_mask.float()
            mask = ttnn.from_torch(
                full[None, None],
                device=self.mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, None)),
            )
        return EngramChunkInputs(lookup, mask)

    # --- device side -----------------------------------------------------------------------------------
    def _copy(self, t: ttnn.Tensor, c: int, width: int) -> ttnn.Tensor:
        s = t.shape[2]
        return ttnn.slice(t, [0, 0, 0, c * width], [1, 1, s, (c + 1) * width])

    def _sum(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.sum(t, dim=-1, keepdim=True, compute_kernel_config=ENGRAM_COMPUTE_CONFIG)

    def _reduce_scatter(self, t: ttnn.Tensor, axis: int, topology) -> ttnn.Tensor:
        """Sum over mesh axis ``axis`` and keep this chip's slice of dim 2."""
        ccl = self.ccl.tt_ccl
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            persistent_output_buffers=None,
            dim=2,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=axis),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=axis),
            num_links=self.ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            cluster_axis=axis,
        )

    @staticmethod
    def _bytes(packed: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """uint16 containers (ROW_MAJOR) -> their low and high bytes, int32 TILE."""
        containers = ttnn.typecast(ttnn.to_layout(packed, ttnn.TILE_LAYOUT), ttnn.int32)
        return ttnn.bitwise_and(containers, 0xFF), ttnn.bitwise_right_shift(containers, 8)

    def rows(self, inputs: EngramChunkInputs) -> ttnn.Tensor:
        """The chunk's looked-up rows ``embed(hash_ids).flatten(-2)``: [1, 1, chunk/sp, n_hash_cols * head_dim] bf16,
        SP-sharded, replicated over TP (bit-identical to ``V41EngramTable.lookup``)."""
        if isinstance(self.table, TtV41EngramTable):
            ids = ttnn.reshape(inputs.lookup, (1, inputs.lookup.shape[-1]))
            gathered = ttnn.embedding(ids, self.table.rows, layout=ttnn.ROW_MAJOR_LAYOUT)
            count = ids.shape[-1]
            packed = ttnn.reshape(ttnn.bitcast(gathered, ttnn.uint16), (1, 1, count, PACKED_WIDTH))
            low, high = self._bytes(packed)
            # every lookup is held by one chip, the others read their zero row: the sum over the mesh is exact
            # for bytes (0..255, exact in bf16; 16-bit containers are not preserved by the reduction), scattered
            # so each chip decodes 1/chips of the lookups
            summed = ttnn.concat([ttnn.typecast(low, ttnn.bfloat16), ttnn.typecast(high, ttnn.bfloat16)], dim=-1)
            summed = self._reduce_scatter(summed, 0, ttnn.Topology.Linear) if self.sp > 1 else summed
            summed = self._reduce_scatter(summed, 1, self.ccl.topology) if self.tp > 1 else summed
            summed = ttnn.typecast(summed, ttnn.int32)
            rows = summed.shape[2]
            low = ttnn.slice(summed, [0, 0, 0, 0], [1, 1, rows, PACKED_WIDTH])
            high = ttnn.slice(summed, [0, 0, 0, PACKED_WIDTH], [1, 1, rows, 2 * PACKED_WIDTH])
        else:
            low, high = self._bytes(inputs.lookup)
        values = self.ccl.tp_all_gather(_decode_packed(low, high, self.head_dim), dim=2)  # [1, 1, lookups/sp, hd]
        return ttnn.reshape(values, (1, 1, values.shape[2] // self.n_hash_cols, self.in_features))

    def forward(self, x: ttnn.Tensor, inputs: EngramChunkInputs) -> ttnn.Tensor:
        """x [1, 1, chunk/sp, hc_mult * hidden/tp] fp32 streams -> the same after the Engram update."""
        hc, local = self.hc, self.local
        # [1, 1, S, (hc + 1) * local] bf16 like the reference GEMM; HiFi4 + fp32 accumulation (HiFi2: update PCC 0.99985)
        kv = ttnn.linear(fp8_qdq(self.rows(inputs)), self.wkv, compute_kernel_config=ENGRAM_COMPUTE_CONFIG)
        seq = kv.shape[2]
        key = ttnn.typecast(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, seq, hc * local]), ttnn.float32)
        value = ttnn.typecast(ttnn.slice(kv, [0, 0, 0, hc * local], [1, 1, seq, (hc + 1) * local]), ttnn.float32)

        # per-chip partial sums over its hidden slice: [sum h^2 (hc) | sum key^2 (hc) | sum h*qk*key (hc) | 0 ...]
        weighted = ttnn.multiply(ttnn.multiply(x, self.qk_weight), key)
        terms = [ttnn.multiply(x, x), ttnn.multiply(key, key), weighted]
        stats = [self._sum(self._copy(t, c, local)) for t in terms for c in range(hc)]
        stats = ttnn.concat(stats, dim=-1)
        stats = ttnn.pad(stats, [(0, 0), (0, 0), (0, 0), (0, STAT_WIDTH - 3 * hc)], 0.0)
        # all-gather the partials over TP and add them in chip order: the same fp32 sum on every chip
        gathered = self.ccl.tp_all_gather(stats, dim=3)
        total = self._copy(gathered, 0, STAT_WIDTH)
        for t in range(1, self.tp):
            total = ttnn.add(total, self._copy(gathered, t, STAT_WIDTH))
        h_sq, k_sq, dot = (self._copy(total, q, hc) for q in range(3))

        inv_dim = 1.0 / self.dim
        rstd = ttnn.multiply(
            ttnn.rsqrt(ttnn.add(ttnn.multiply(h_sq, inv_dim), self.eps)),
            ttnn.rsqrt(ttnn.add(ttnn.multiply(k_sq, inv_dim), self.eps)),
        )
        dot = ttnn.multiply(ttnn.multiply(dot, rstd), self.dim**-0.5)
        root = ttnn.sqrt(ttnn.maximum(ttnn.abs(dot), GATE_CLAMP))
        gate = ttnn.sigmoid(ttnn.where(ttnn.ltz(dot), ttnn.neg(root), root))  # [1, 1, S, hc]
        if inputs.mask is not None:
            gate = ttnn.multiply(gate, inputs.mask)
        update = ttnn.concat([ttnn.multiply(self._copy(gate, c, 1), value) for c in range(hc)], dim=-1)
        return ttnn.add(x, update)
