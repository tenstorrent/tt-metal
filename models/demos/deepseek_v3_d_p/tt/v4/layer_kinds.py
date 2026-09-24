# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash layer schedule and per-kind KV geometry -- pure Python, no device.

The checkpoint describes each layer's attention by a compression ratio (``compress_ratios``: 0 = sliding-window,
4 = compressed sparse attention with the lightning indexer, 128 = heavily compressed attention) and each layer's
MLP by ``mlp_layer_types`` (the first 3 are hash-routed MoE, the rest top-k MoE). Everything downstream -- which
``Tt*`` attention module a block builds, how many rows each KV group needs, which chunk-table config a layer's
cache lives in -- is derived from ``DeepseekV4Config.layer_types`` through this module and nowhere else.

KV GROUPS are the prefill <-> decode contract and live in ``kv_contract`` (plain data shared with tt-blaze). One
config per DECODE TENSOR KIND, because decode keeps window + compressed entries in one unified tensor per layer
and uses a different dtype per layer kind (contract finding 2026-09-24, DS4F-0238):
  0 ``swa_window``   layers 0,1: the 128-row window ring            (_CACHE_DTYPE tile, 512 wide)
  1 ``hca_unified``  HCA layers: window ring + entries (1 per 128)   (_CACHE_DTYPE tile, 512 wide)
  2 ``csa_unified``  CSA layers: window ring + entries (1 per 4)     (bf16 ROW_MAJOR, 512 wide)
  3 ``csa_index_k``  CSA layers: lightning-indexer keys, 1 per entry (bfp8 tile, 128 wide)
  4 ``csa_pending``  CSA layers: compressor partial-window state     (bf16 ROW_MAJOR, 1024 wide; contract gap)
  5 ``hca_pending``  HCA layers: compressor partial-window state     (bf16 ROW_MAJOR, 1024 wide; contract gap)

Every group is REPLICATED over the sequence-parallel axis and addressed LINEARLY by unified row (not the MLA
family's block-cyclic per-shard layout): the V4 compressors all-gather their entries anyway, and HCA emits 40
entries per 5120-token chunk, which a 32-rows-per-shard block-cyclic table cannot describe (DS4F-0239).
"""

from __future__ import annotations

from dataclasses import dataclass

from models.demos.deepseek_v3_d_p.tt.v4.kv_contract import (
    CONTRACT,
    KV_GROUPS,
    MIGRATED_GROUPS,
    PENDING_GROUPS,
    TILE,
    KvGroupSpec,
    compressed_entries,
)

SLIDING = "sliding_attention"
CSA = "compressed_sparse_attention"
HCA = "heavily_compressed_attention"
LAYER_KINDS = (SLIDING, CSA, HCA)

__all__ = [
    "SLIDING",
    "CSA",
    "HCA",
    "LAYER_KINDS",
    "KV_GROUPS",
    "MIGRATED_GROUPS",
    "PENDING_GROUPS",
    "CONTRACT",
    "TILE",
    "flash_compress_ratios",
    "layer_kinds",
    "kind_counts",
    "layers_of_kind",
    "compressed_entries",
    "V4FlashKvGeometry",
]


def flash_compress_ratios(num_layers: int) -> list[int]:
    """DeepSeek-V4-Flash's per-layer compression ratios, as the checkpoint's ``config.json`` states them:
    layers 0 and 1 are sliding-window (0), then even layers are CSA (4) and odd layers are HCA (128).
    43 layers -> 2 SWA, 21 CSA, 20 HCA."""
    out = []
    for i in range(int(num_layers)):
        out.append(0 if i < 2 else (4 if i % 2 == 0 else 128))
    return out


def layer_kinds(cfg) -> list[str]:
    """``cfg.layer_types`` validated against the three V4 kinds (a DeepseekV4Config or any object carrying it)."""
    kinds = list(cfg.layer_types)
    bad = sorted(set(kinds) - set(LAYER_KINDS))
    if bad:
        raise ValueError(f"unknown V4 layer kinds {bad}; expected a subset of {LAYER_KINDS}")
    return kinds


def kind_counts(cfg) -> dict[str, int]:
    kinds = layer_kinds(cfg)
    return {k: kinds.count(k) for k in LAYER_KINDS}


def layers_of_kind(cfg, kind: str, *, first_layer_idx: int = 0, num_layers: int | None = None) -> list[int]:
    """GLOBAL layer indices of ``kind`` inside this rank's slice ``[first_layer_idx, first_layer_idx + num_layers)``."""
    kinds = layer_kinds(cfg)
    n = len(kinds) if num_layers is None else int(num_layers)
    lo = int(first_layer_idx)
    return [i for i in range(lo, lo + n) if kinds[i] == kind]


@dataclass(frozen=True)
class V4FlashKvGeometry:
    """Rows per KV group for this rank's layer slice, given the (global) context length.

    Every group is a per-layer, per-user tensor ``[users * layers_in_group, 1, rows, width]`` holding the SAME
    rows on every chip (replicated over SP). ``rows`` is the contract extent for ``max_seq_len`` (window ring
    128 + tile-aligned entries), so the allocation and the migratable extent coincide. ``sp_factor`` is carried
    only to size ``init_kvpe_cache``'s ``seq_len`` argument (it divides by the SP extent internally)."""

    max_seq_len: int
    sp_factor: int
    sliding_window: int
    head_dim: int
    index_head_dim: int
    csa_rate: int
    hca_rate: int
    swa_layers: tuple  # GLOBAL layer indices of each kind inside this rank's slice
    hca_layers: tuple
    csa_layers: tuple

    @classmethod
    def from_config(cls, cfg, *, max_seq_len: int, sp_factor: int, first_layer_idx: int = 0, num_layers=None):
        kinds = layer_kinds(cfg)
        n = len(kinds) if num_layers is None else int(num_layers)
        lo = int(first_layer_idx)
        if lo < 0 or lo + n > len(kinds):
            raise ValueError(f"layer slice [{lo}, {lo + n}) exceeds the {len(kinds)}-layer model")
        geom = cls(
            max_seq_len=int(max_seq_len),
            sp_factor=int(sp_factor),
            sliding_window=int(cfg.sliding_window),
            head_dim=int(cfg.head_dim),
            index_head_dim=int(cfg.index_head_dim),
            csa_rate=int(cfg.compress_rates[CSA]),
            hca_rate=int(cfg.compress_rates[HCA]),
            swa_layers=tuple(i for i in range(lo, lo + n) if kinds[i] == SLIDING),
            hca_layers=tuple(i for i in range(lo, lo + n) if kinds[i] == HCA),
            csa_layers=tuple(i for i in range(lo, lo + n) if kinds[i] == CSA),
        )
        geom.validate()
        return geom

    def validate(self) -> None:
        """The contract's constants are the model's: this catches a config that drifted from the decode side."""
        from models.demos.deepseek_v3_d_p.tt.v4 import kv_contract as kc

        if self.sliding_window != kc.WINDOW or self.head_dim != kc.HEAD_DIM or self.index_head_dim != kc.INDEX_HEAD_DIM:
            raise ValueError(
                f"config (window {self.sliding_window}, head_dim {self.head_dim}, index_head_dim {self.index_head_dim}) "
                f"disagrees with the KV contract ({kc.WINDOW}, {kc.HEAD_DIM}, {kc.INDEX_HEAD_DIM})"
            )
        if self.csa_rate != kc.CSA_RATE or self.hca_rate != kc.HCA_RATE:
            raise ValueError(f"compress rates ({self.csa_rate}, {self.hca_rate}) disagree with the contract")
        if self.max_seq_len <= 0 or self.max_seq_len % self.hca_rate:
            raise ValueError(
                f"max_seq_len {self.max_seq_len} must be a positive multiple of the HCA rate {self.hca_rate}"
            )
        if self.sp_factor <= 0:
            raise ValueError(f"sp_factor {self.sp_factor} must be positive")

    # ---- entries / rows -------------------------------------------------------------------------------------
    @property
    def hca_entries(self) -> int:
        return compressed_entries(self.max_seq_len, self.hca_rate)

    @property
    def csa_entries(self) -> int:
        return compressed_entries(self.max_seq_len, self.csa_rate)

    def rows(self, group: str) -> int:
        """Rows allocated (= migratable extent) for ``group`` at this context length."""
        return self._spec(group).extent(self.max_seq_len)

    @staticmethod
    def _spec(group: str) -> KvGroupSpec:
        for g in CONTRACT:
            if g.name == group:
                return g
        raise KeyError(group)

    def layers(self, group: str) -> tuple:
        kind = self._spec(group).kind
        return {SLIDING: self.swa_layers, HCA: self.hca_layers, CSA: self.csa_layers}[kind]

    def group_shapes(self, num_users: int = 1, *, include_pending: bool = True) -> dict[str, tuple[int, int, int, int]]:
        """``{group: (users*layers, 1, rows, width)}`` in contract order; a group with no layers on this rank is absent."""
        u = int(num_users)
        out = {}
        for g in CONTRACT:
            if g.pending and not include_pending:
                continue
            n_layers = len(self.layers(g.name))
            if n_layers:
                out[g.name] = (u * n_layers, 1, g.extent(self.max_seq_len), g.width)
        return out

    def group_bytes(self, num_users: int = 1) -> dict[str, int]:
        """DRAM per group PER CHIP (replicated), from the contract's chunk bytes."""
        out = {}
        for name, (b, _, rows, _) in self.group_shapes(num_users).items():
            g = self._spec(name)
            out[name] = b * (rows // TILE) * g.chunk_size_bytes
        return out
