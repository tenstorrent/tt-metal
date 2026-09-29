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
* N3 lookup (host): FP8 e4m3 rows with E8M0 per-32 scales, dequantized as the reference does -> bf16
  ``[L, 24, 256]``, flattened to ``[L, 6144]`` and uploaded per chunk (prefetchable: depends on tokens only).
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
    """One Engram layer's table held as the checkpoint stores it (N3): float8_e4m3fn rows ``[R, head_dim]`` and
    float8_e8m0fnu scales ``[R, head_dim / 32]``. ``row_ids`` (sorted int64): the global rows these ``R`` rows are,
    for a row subset (e.g. the rows a prompt needs); None = the full table (rows are global ids)."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, row_ids: torch.Tensor | None = None):
        assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu, (weight.dtype, scale.dtype)
        assert scale.shape == (weight.shape[0], weight.shape[1] // SCALE_BLOCK), (weight.shape, scale.shape)
        assert row_ids is None or row_ids.shape == (weight.shape[0],)
        self.weight, self.scale, self.row_ids = weight, scale, row_ids

    def _local(self, ids: torch.Tensor) -> torch.Tensor:
        if self.row_ids is None:
            return ids
        local = torch.searchsorted(self.row_ids, ids.contiguous()).clamp_max(len(self.row_ids) - 1)
        if not torch.equal(self.row_ids[local], ids):
            raise KeyError("Engram rows requested that this table subset does not hold")
        return local

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [...] int64 global rows -> bf16 [..., head_dim]: ``fp32(row) * fp32(scale)`` per 32, cast to bf16."""
        local = self._local(ids)
        values = self.weight[local].float().unflatten(-1, (-1, SCALE_BLOCK)) * self.scale[local].float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16)


@dataclass
class EngramChunkInputs:
    """Per-chunk device inputs of one Engram layer: ``rows`` [1, 1, chunk/sp, n_hash_cols * head_dim] bf16
    (SP-sharded, replicated over TP; rows beyond the valid length are zero) and ``mask`` [1, 1, chunk/sp, 1]
    fp32 (1 = text, 0 = image token; None = all text)."""

    rows: ttnn.Tensor
    mask: ttnn.Tensor | None


class TtV41Engram(LightweightModule):
    """One Engram layer: host row lookup + upload (``prepare``), device wkv and gated stream update (``forward``)."""

    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        table: V41EngramTable,
        topology=ttnn.Topology.Linear,
        weights_dtype=ttnn.bfloat8_b,
    ):
        """``weights``: ``wkv`` [(hc_mult + 1) * hidden, n_hash_cols * head_dim] (FP8 dequantized, checkpoint
        orientation [out, in]), ``q_weight`` and ``k_weight`` [hc_mult, hidden]; ``table``: this layer's rows."""
        assert layer in config.ENGRAM_LAYER_IDS, f"layer {layer} has no Engram"
        self.mesh_device, self.config, self.layer, self.table = mesh_device, config, layer, table
        self.sp, self.tp = mesh_device.shape
        self.hc, self.dim, self.eps = config.HC_MULT, config.EMB_SIZE, config.RMS_NORM_EPS
        assert self.dim % (32 * self.tp) == 0, f"hidden {self.dim} does not split over TP={self.tp} in tiles"
        self.local = self.dim // self.tp
        self.in_features = (config.ENGRAM_MAX_NGRAM_SIZE - 1) * config.ENGRAM_N_HEADS * config.ENGRAM_HEAD_DIM
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
    def _sp_rows(self, host: torch.Tensor, dtype) -> ttnn.Tensor:
        """[chunk, W] token rows -> each SP rank its contiguous chunk/sp rows, replicated over TP."""
        return ttnn.from_torch(
            host[None, None],
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, None)),
        )

    def host_rows(self, hash_ids: torch.Tensor, chunk: int) -> torch.Tensor:
        """This layer's hash ids [L, n_hash_cols] of the valid tokens -> bf16 [chunk, n_hash_cols * head_dim]
        (the reference's ``embed(hash_ids).flatten(-2)``), zero rows beyond L."""
        rows = torch.zeros(chunk, self.in_features, dtype=torch.bfloat16)
        rows[: hash_ids.shape[0]] = self.table.lookup(hash_ids).flatten(-2)
        return rows

    def prepare(self, hash_ids: torch.Tensor, chunk: int, token_mask: torch.Tensor | None = None) -> EngramChunkInputs:
        """Host lookup + upload of one chunk (depends on tokens only, so it can run ahead of the device).
        ``token_mask`` [L] bool: False for image tokens (their gate is zero); None = all text."""
        mask = None
        if token_mask is not None:
            full = torch.ones(chunk, 1)
            full[: token_mask.numel(), 0] = token_mask.float()
            mask = self._sp_rows(full, ttnn.float32)
        return EngramChunkInputs(self._sp_rows(self.host_rows(hash_ids, chunk), ttnn.bfloat16), mask)

    # --- device side -----------------------------------------------------------------------------------
    def _copy(self, t: ttnn.Tensor, c: int, width: int) -> ttnn.Tensor:
        s = t.shape[2]
        return ttnn.slice(t, [0, 0, 0, c * width], [1, 1, s, (c + 1) * width])

    def _sum(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.sum(t, dim=-1, keepdim=True, compute_kernel_config=ENGRAM_COMPUTE_CONFIG)

    def forward(self, x: ttnn.Tensor, inputs: EngramChunkInputs) -> ttnn.Tensor:
        """x [1, 1, chunk/sp, hc_mult * hidden/tp] fp32 streams -> the same after the Engram update."""
        hc, local = self.hc, self.local
        # [1, 1, S, (hc + 1) * local] bf16 like the reference GEMM; HiFi4 + fp32 accumulation (HiFi2: update PCC 0.99985)
        kv = ttnn.linear(fp8_qdq(inputs.rows), self.wkv, compute_kernel_config=ENGRAM_COMPUTE_CONFIG)
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
