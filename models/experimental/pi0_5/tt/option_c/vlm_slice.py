# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Submesh-aware VLM transformer slice for Option C (no TP, L1-resident).

Option C target: one VLM transformer layer per chip on the 18-chip prefill
submesh. This first cut uses REPLICATED weights across the submesh — same
construction-path-validation strategy Option B's first cut used. Layer-paired
sharding (the real Option C placement) comes next; the slice API doesn't
change, only the upload helpers do.

Difference vs `option_b.vlm_slice`:
  - Default `memory_config=L1_MEMORY_CONFIG` on every upload (vs DRAM in B).
  - No TP path — Option C's whole premise is "no collectives within a stage",
    so the TP=8 variant doesn't exist here.

Memory budget per chip (real config, replicated, single VLM layer):
  ~110 MB weights bf8 + 2 MB activation + 250 KB KV + 5 MB scratch ≈ 118 MB
  → fits inside the 180 MB L1 cap with ~60 MB headroom (per plan §3.1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import (
    GemmaBlockTTNN,
    precompute_freqs_cis_meta_format,
)

from .transport import send_activation_via_host


# ---------------------------------------------------------------------------- #
# Upload helpers — Option C defaults every placement to L1.                     #
# ---------------------------------------------------------------------------- #


def _upload_l1_replicated(
    t: torch.Tensor,
    submesh,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
) -> "ttnn.Tensor":
    """Upload a torch tensor replicated across every chip in `submesh`.

    Default is DRAM despite the helper name (kept for call-site stability while
    we move toward layer-paired L1 placement). The "L1-everywhere" claim in
    the original docstring assumed that the absence of all_reduces made L1
    safe — but ops like rms_norm reserve their own low-L1 static CB region
    that clashes with any L1-resident buffer that lands at a low address.
    Replicated bring-up uploads also blow past the per-chip L1 budget because
    each chip holds a full copy.

    Once layer-paired distribution is in (1 VLM layer / chip across 18 chips),
    callers can pass `memory_config=ttnn.L1_MEMORY_CONFIG` per tensor to
    move only the per-layer working set into L1 while keeping shared
    activations / KV cache in DRAM.
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    mapper = ttnn.replicate_tensor_to_mesh_mapper(submesh)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=submesh,
        mesh_mapper=mapper,
        memory_config=memory_config,
    )


def _upload_single_chip_l1(
    t: torch.Tensor,
    micro_submesh,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
) -> "ttnn.Tensor":
    """Upload a torch tensor to a 1-chip micro-submesh.

    Used by the layer-paired path: each VLM layer's weights live on exactly
    one chip, so the per-chip L1 budget (~118 MB for one VLM layer) sits
    well inside the 180 MB cap and we can default `memory_config` to L1.

    `micro_submesh` must be a 1-chip MeshDevice (typically built by
    `mesh_setup.create_per_chip_submeshes`).
    """
    if micro_submesh.get_num_devices() != 1:
        raise ValueError("_upload_single_chip_l1 requires a 1-chip submesh; got " f"{micro_submesh.get_num_devices()}")
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG
    mapper = ttnn.replicate_tensor_to_mesh_mapper(micro_submesh)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=micro_submesh,
        mesh_mapper=mapper,
        memory_config=memory_config,
    )


def _load_vlm_block_weights_l1(
    full_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    submesh,
) -> Dict[str, "ttnn.Tensor"]:
    """Upload one VLM layer's weights onto `submesh` (replicated, L1-resident).

    Identical key mapping to `option_b.vlm_slice._load_vlm_block_weights` but
    every upload routes through `_upload_l1_replicated`.
    """
    prefix = f"model.layers.{layer_idx}."
    block_weights: Dict[str, "ttnn.Tensor"] = {}

    q_key = f"{prefix}self_attn.q_proj.weight"
    k_key = f"{prefix}self_attn.k_proj.weight"
    v_key = f"{prefix}self_attn.v_proj.weight"

    if q_key in full_weights and k_key in full_weights and v_key in full_weights:
        wq = _upload_l1_replicated(full_weights[q_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        wk = _upload_l1_replicated(full_weights[k_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        wv = _upload_l1_replicated(full_weights[v_key].T.contiguous(), submesh, ttnn.bfloat8_b)
        block_weights["self_attn.wqkv"] = ttnn.concat([wq, wk, wv], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(wq)
        ttnn.deallocate(wk)
        ttnn.deallocate(wv)

    for key, value in full_weights.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix) :]
        if new_key in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            continue

        is_norm = "layernorm" in new_key or "norm" in new_key
        if "weight" in new_key and not is_norm:
            value = value.T
        if is_norm:
            value = value + 1.0  # Gemma-style +1 offset, pre-baked

        if len(value.shape) == 1:
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, submesh, dtype=ttnn.bfloat16)
        else:
            weight_dtype = ttnn.bfloat16 if is_norm else ttnn.bfloat8_b
            block_weights[new_key] = _upload_l1_replicated(value.contiguous(), submesh, weight_dtype)
    return block_weights


class Pi0_5OptionCVLMSlice:
    """VLM transformer layers on the Option C prefill submesh (no TP, L1-resident).

    Args:
        config:               full PaliGemma config.
        weights:              full weights dict (we slice by layer index).
        submesh:              the prefill MeshDevice (18 chips for Option C).
        layer_range:          half-open (lo, hi). For the scaffolding pass
                              this is typically small (1-2 layers) so the
                              replicated weights fit the per-chip L1 budget;
                              real Option C will use layer-paired sharding
                              and pass the full (0, 18) range.
        holds_vlm_final_norm: if True, also upload model.norm.weight and
                              apply it at the tail of forward(). Set on the
                              slice that owns the last VLM layer.
        holds_embed_tokens:   currently ignored — Option C plans to host the
                              embed_tokens table on the vision-embed chip,
                              not the prefill submesh. Accepted as a flag so
                              the scaffolding-mode shrunk test can still pass
                              a single layout to both slices.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        submesh,
        layer_range: Tuple[int, int],
        holds_embed_tokens: bool = False,
        holds_vlm_final_norm: bool = False,
    ) -> None:
        if not (0 <= layer_range[0] < layer_range[1] <= config.vlm_config.depth):
            raise ValueError(
                f"layer_range {layer_range} out of bounds for " f"vlm_config.depth={config.vlm_config.depth}"
            )

        self.config = config
        self.submesh = submesh
        self.layer_lo, self.layer_hi = layer_range
        self.num_layers = self.layer_hi - self.layer_lo
        self.holds_embed_tokens = holds_embed_tokens
        self.holds_vlm_final_norm = holds_vlm_final_norm

        lang = weights["vlm_language"]

        # Final RMSNorm (only when this slice owns the tail; prefill chip 17
        # in default Option C).
        self.vlm_norm: Optional["ttnn.Tensor"] = None
        if holds_vlm_final_norm:
            self.vlm_norm = tensor_1d_to_2d_ttnn(lang["model.norm.weight"] + 1.0, submesh, dtype=ttnn.bfloat16)

        # Optional embed_tokens table — accepted for shrunk-config tests but
        # the real Option C placement puts this on the vision-embed chip
        # (see deployment plan §3.1, option (a)).
        self.vlm_embed_tokens: Optional["ttnn.Tensor"] = None
        if holds_embed_tokens:
            embed = lang.get("model.embed_tokens.weight") or lang.get("lm_head.weight")
            if embed is None:
                raise KeyError(
                    "holds_embed_tokens=True but neither model.embed_tokens.weight "
                    "nor lm_head.weight is in weights['vlm_language']"
                )
            self.vlm_embed_tokens = _upload_l1_replicated(
                embed.contiguous(),
                submesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        # RoPE tables — precomputed once on this submesh.
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(
            config.vlm_config.head_dim,
            config.max_seq_len,
            submesh,
        )

        # Per-layer blocks.
        self.vlm_blocks: List = []
        for i in range(self.layer_lo, self.layer_hi):
            block_weights = _load_vlm_block_weights_l1(lang, i, submesh)
            self.vlm_blocks.append(
                GemmaBlockTTNN(
                    config.vlm_config,
                    block_weights,
                    i,
                    submesh,
                    self.cos_meta,
                    self.sin_meta,
                )
            )

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def embed_language_tokens(self, token_ids: "ttnn.Tensor") -> "ttnn.Tensor":
        if self.vlm_embed_tokens is None:
            raise RuntimeError("embed_language_tokens called on a slice without holds_embed_tokens=True")
        return ttnn.embedding(token_ids, self.vlm_embed_tokens)

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_values: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        use_cache: bool = False,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
    ) -> Tuple["ttnn.Tensor", Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]]]:
        """Forward through layers [layer_lo, layer_hi).

        past_key_values is indexed by GLOBAL layer index over the full VLM
        depth — slicing happens here so callers can pass the full list.
        Returned new_cache is also keyed by global index; only this slice's
        entries are populated when use_cache=True.
        """
        new_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = (
            [None] * self.config.vlm_config.depth if use_cache else None
        )

        for local_i, block in enumerate(self.vlm_blocks):
            global_i = self.layer_lo + local_i
            past_kv = past_key_values[global_i] if past_key_values is not None else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                cos_override,
                sin_override,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache and new_kv is not None:
                new_cache[global_i] = new_kv

        if self.holds_vlm_final_norm and self.vlm_norm is not None:
            # The default-config (non-sharded) rms_norm reserves a static
            # circular-buffer region at low L1 addresses and writes its OUTPUT
            # to L1. With Option C's L1-resident weights + KV cache + residual
            # stream, both the upstream activation AND the rms_norm output land
            # at L1 addresses that overlap the static CB region — failing
            # `validate_circular_buffer_region` ("Statically allocated CBs in
            # program N clash with L1 buffers on core range ...").
            # Bouncing input/output through DRAM avoids the clash entirely; the
            # next-stage host-bounce transport reads from DRAM anyway, so this
            # adds no extra round-trip on the production path.
            h_dram = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hidden_states)
            hidden_states = ttnn.rms_norm(
                h_dram,
                weight=self.vlm_norm,
                epsilon=self.config.vlm_config.rms_norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(h_dram)

        return hidden_states, new_cache

    # ------------------------------------------------------------------ #
    # KV migration emitter                                                #
    # ------------------------------------------------------------------ #

    def get_kv_cache_for_slice(
        self,
        new_cache: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]],
    ) -> List[Tuple[int, Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """Return (global_layer_idx, (K, V)) tuples for this slice's layers.

        kv_migration.KVMigration consumes this to ship per-layer KV to the
        denoise stage after prefill completes.
        """
        out = []
        for local_i in range(self.num_layers):
            global_i = self.layer_lo + local_i
            kv = new_cache[global_i] if new_cache is not None else None
            if kv is not None:
                out.append((global_i, kv))
        return out


# ---------------------------------------------------------------------------- #
# Layer-paired slice — 1 VLM layer per chip, L1-resident, host-bounce between   #
# adjacent chips.                                                               #
# ---------------------------------------------------------------------------- #


def _load_vlm_block_weights_single_chip_l1(
    full_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    micro_submesh,
    attn_dram: bool = False,
) -> Dict[str, "ttnn.Tensor"]:
    """Upload one VLM layer's weights onto a single-chip micro-submesh, L1-resident.

    Identical key mapping to `_load_vlm_block_weights_l1` but every upload
    is single-chip + L1. With only one layer's weights per chip we sit well
    inside the 180 MB L1 budget (~110 MB weights bf8 + headroom).

    `attn_dram`: when True, keep `o_proj` in DRAM alongside the already-DRAM
    fused `wqkv`. Frees the ~4.5 MB / chip that o_proj occupies on L1 — buys
    fragmentation headroom for the Tilize transient on the prefill forward
    path at S=1024 + no-TP + bf8.
    """
    prefix = f"model.layers.{layer_idx}."
    block_weights: Dict[str, "ttnn.Tensor"] = {}

    q_key = f"{prefix}self_attn.q_proj.weight"
    k_key = f"{prefix}self_attn.k_proj.weight"
    v_key = f"{prefix}self_attn.v_proj.weight"

    if q_key in full_weights and k_key in full_weights and v_key in full_weights:
        wq = _upload_single_chip_l1(full_weights[q_key].T.contiguous(), micro_submesh, ttnn.bfloat8_b)
        wk = _upload_single_chip_l1(full_weights[k_key].T.contiguous(), micro_submesh, ttnn.bfloat8_b)
        wv = _upload_single_chip_l1(full_weights[v_key].T.contiguous(), micro_submesh, ttnn.bfloat8_b)
        # Final fused QKV in DRAM — keeps the static-CB clash that pushed every
        # replicated weight to DRAM from biting the per-chip rms_norm at low L1.
        block_weights["self_attn.wqkv"] = ttnn.concat([wq, wk, wv], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(wq)
        ttnn.deallocate(wk)
        ttnn.deallocate(wv)

    for key, value in full_weights.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix) :]
        if new_key in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            continue

        is_norm = "layernorm" in new_key or "norm" in new_key
        is_o_proj = new_key == "self_attn.o_proj.weight"
        if "weight" in new_key and not is_norm:
            value = value.T
        if is_norm:
            value = value + 1.0  # Gemma-style +1 offset, pre-baked

        if len(value.shape) == 1:
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, micro_submesh, dtype=ttnn.bfloat16)
        else:
            weight_dtype = ttnn.bfloat16 if is_norm else ttnn.bfloat8_b
            mem_cfg = ttnn.DRAM_MEMORY_CONFIG if (attn_dram and is_o_proj) else None
            block_weights[new_key] = _upload_single_chip_l1(
                value.contiguous(), micro_submesh, weight_dtype, memory_config=mem_cfg
            )
    return block_weights


class Pi0_5OptionCVLMSlicePaired:
    """Layer-paired VLM slice — one VLM layer per single-chip micro-submesh.

    This is Option C's target placement (deployment plan §3.1): each prefill
    chip owns exactly one VLM transformer layer, weights L1-resident, no
    cross-chip collectives. Activation transport between consecutive layers
    is host-bounce (same fallback Option B's stage-to-stage transport uses)
    until on-fabric D2D copy lands.

    External contract is identical to `Pi0_5OptionCVLMSlice`:
        forward(hidden_states, ...) -> (hidden_states_on_last_chip, new_cache)
        get_kv_cache_for_slice(new_cache) -> List[(global_layer_idx, (K, V))]

    Args:
        config:               full PaliGemma config.
        weights:              full categorized weights dict.
        micro_submeshes:      list of 1-chip MeshDevices, one per layer in the
                              slice's `layer_range`. Build via
                              `mesh_setup.create_per_chip_submeshes`.
        layer_range:          half-open (lo, hi). len(micro_submeshes) must
                              equal hi - lo.
        holds_vlm_final_norm: if True, place model.norm.weight on the
                              LAST micro-submesh (chip 17 in the default
                              Option C layout) and apply it after the
                              final block runs there.
        holds_embed_tokens:   accepted for API parity with the replicated
                              slice; the embed table is host-side or on
                              the vision-embed chip in Option C and never
                              lives on the prefill submesh.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        micro_submeshes: List,
        layer_range: Tuple[int, int],
        holds_embed_tokens: bool = False,
        holds_vlm_final_norm: bool = False,
        attn_dram: bool = False,
    ) -> None:
        if not (0 <= layer_range[0] < layer_range[1] <= config.vlm_config.depth):
            raise ValueError(
                f"layer_range {layer_range} out of bounds for " f"vlm_config.depth={config.vlm_config.depth}"
            )
        if len(micro_submeshes) != layer_range[1] - layer_range[0]:
            raise ValueError(
                f"micro_submeshes count ({len(micro_submeshes)}) must equal "
                f"layer_range span ({layer_range[1] - layer_range[0]})"
            )
        for i, sm in enumerate(micro_submeshes):
            if sm.get_num_devices() != 1:
                raise ValueError(f"micro_submeshes[{i}] is not a 1-chip submesh " f"({sm.get_num_devices()} devices)")
        if holds_embed_tokens:
            # Accept the flag but don't materialize the table — Option C's
            # embed_tokens lives off the prefill submesh.
            pass

        self.config = config
        self.micro_submeshes = micro_submeshes
        self.layer_lo, self.layer_hi = layer_range
        self.num_layers = self.layer_hi - self.layer_lo
        self.holds_embed_tokens = holds_embed_tokens
        self.holds_vlm_final_norm = holds_vlm_final_norm

        lang = weights["vlm_language"]

        # RoPE tables — one set per chip (each layer needs its own copy
        # since chips can't share L1 across submeshes).
        self.cos_metas: List = []
        self.sin_metas: List = []
        for sm in micro_submeshes:
            cos, sin = precompute_freqs_cis_meta_format(
                config.vlm_config.head_dim,
                config.max_seq_len,
                sm,
            )
            self.cos_metas.append(cos)
            self.sin_metas.append(sin)

        # Per-layer blocks — each block owns its chip.
        self.vlm_blocks: List = []
        for local_i in range(self.num_layers):
            global_i = self.layer_lo + local_i
            sm = micro_submeshes[local_i]
            block_weights = _load_vlm_block_weights_single_chip_l1(lang, global_i, sm, attn_dram=attn_dram)
            self.vlm_blocks.append(
                GemmaBlockTTNN(
                    config.vlm_config,
                    block_weights,
                    global_i,
                    sm,
                    self.cos_metas[local_i],
                    self.sin_metas[local_i],
                )
            )

        # Final RMSNorm on the LAST chip in the chain when this slice owns it.
        self.vlm_norm: Optional["ttnn.Tensor"] = None
        if holds_vlm_final_norm:
            self.vlm_norm = tensor_1d_to_2d_ttnn(
                lang["model.norm.weight"] + 1.0,
                micro_submeshes[-1],
                dtype=ttnn.bfloat16,
            )

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_values: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        use_cache: bool = False,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
    ) -> Tuple["ttnn.Tensor", Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]]]:
        """Run layers [layer_lo, layer_hi) sequentially across chips.

        On entry, `hidden_states` must live on `micro_submeshes[0]` and any
        `attention_mask` argument must live on the same chip. Between
        consecutive layers we host-bounce the activation; the mask is
        re-uploaded onto each chip exactly once (cached by id internally —
        callers can pass the same mask object every call).

        Returns: (hidden_on_last_chip, new_cache_keyed_by_global_idx)
        """
        if past_key_values is not None and any(
            past_key_values[self.layer_lo + i] is not None for i in range(self.num_layers)
        ):
            raise NotImplementedError(
                "Pi0_5OptionCVLMSlicePaired.forward does not yet accept past_key_values; "
                "this stage is prefill-only in the current pipeline."
            )

        new_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = (
            [None] * self.config.vlm_config.depth if use_cache else None
        )

        # Cache per-chip masks so a multi-step prefill (or repeated calls)
        # doesn't pay the host bounce N times. Keyed by id(attention_mask)
        # so callers can pass the same object every iteration.
        mask_cache_key = id(attention_mask) if attention_mask is not None else None
        if not hasattr(self, "_per_chip_mask_cache"):
            self._per_chip_mask_cache: Dict = {}
        masks_per_chip = self._per_chip_mask_cache.get(mask_cache_key)
        if attention_mask is not None and masks_per_chip is None:
            masks_per_chip = [attention_mask]
            for i in range(1, self.num_layers):
                # SDPA requires masks to live in DRAM. `send_activation_via_host`
                # lands its output in L1 by default — flip to DRAM here.
                broadcast = send_activation_via_host(masks_per_chip[0], self.micro_submeshes[i])
                dram_mask = ttnn.to_memory_config(broadcast, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(broadcast)
                masks_per_chip.append(dram_mask)
            self._per_chip_mask_cache[mask_cache_key] = masks_per_chip

        h = hidden_states
        for local_i, block in enumerate(self.vlm_blocks):
            global_i = self.layer_lo + local_i
            mask_i = masks_per_chip[local_i] if masks_per_chip is not None else None
            h_new, new_kv = block.forward(
                h,
                cos_override,
                sin_override,
                mask_i,
                position_ids,
                None,  # past_kv handled per-chip; prefill-only for now
                use_cache,
            )
            if use_cache and new_kv is not None:
                new_cache[global_i] = new_kv
            # Host-bounce activation to the next chip before deallocating
            # the local copy.
            if local_i + 1 < self.num_layers:
                h = send_activation_via_host(h_new, self.micro_submeshes[local_i + 1])
                ttnn.deallocate(h_new)
            else:
                h = h_new

        # Final RMSNorm on the last chip (only when this slice owns it).
        if self.holds_vlm_final_norm and self.vlm_norm is not None:
            # Same low-L1 / static-CB clash as the replicated slice — bounce
            # input/output through DRAM on the last chip.
            h_dram = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(h)
            h = ttnn.rms_norm(
                h_dram,
                weight=self.vlm_norm,
                epsilon=self.config.vlm_config.rms_norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(h_dram)

        return h, new_cache

    # ------------------------------------------------------------------ #
    # KV migration emitter                                                #
    # ------------------------------------------------------------------ #

    def get_kv_cache_for_slice(
        self,
        new_cache: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]],
    ) -> List[Tuple[int, Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """Return (global_layer_idx, (K, V)) tuples for this slice's layers.

        Each (K, V) lives on the layer's owning micro-submesh; the migrator
        ships them to the matching denoise chip via host-bounce.
        """
        out = []
        for local_i in range(self.num_layers):
            global_i = self.layer_lo + local_i
            kv = new_cache[global_i] if new_cache is not None else None
            if kv is not None:
                out.append((global_i, kv))
        return out


# ---------------------------------------------------------------------------- #
# TP=2 slice — 9 (2,1) col-pair sub-meshes × 2 VLM layers per sub-mesh.        #
# Mirrors Pi0_5OptionCVLMSlicePaired's host-bounce-between-chips pattern, only #
# the unit is a (2,1) sub-mesh running one TP=2 block instead of a single chip #
# running a full-replication block.                                            #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Parent-mesh slice (D2D Option A) — 18 layers via per-chip-sharded weights + #
# P2P transitions. Replaces the host-bouncing layer-paired path.               #
# ---------------------------------------------------------------------------- #


def _load_vlm_weights_stacked_sharded(
    full_weights: Dict[str, torch.Tensor],
    layer_range: Tuple[int, int],
    parent_mesh,
    parent_shape: Tuple[int, int],
    prefill_offset: Tuple[int, int],
    prefill_shape: Tuple[int, int],
    dtype=ttnn.bfloat8_b,
) -> Dict[str, "ttnn.Tensor"]:
    """Upload all VLM layer weights as parent-mesh sharded tensors.

    For each weight category (q_proj, k_proj, v_proj, o_proj, gate_proj,
    up_proj, down_proj, layernorm weights), build a stacked tensor over the
    parent mesh (8×4 = 32 slots) where:
      - Slot at parent linear-idx for prefill chip i holds layer i's weight.
      - Other slots (non-prefill chips) hold zero tensors of matching shape.
    Then upload with `ShardTensorToMesh(parent_mesh, dim=0)`.

    This is the foundation for the parent-mesh D2D model: chip i has layer
    i's weights, ttnn.linear on the parent mesh runs the matmul on every
    chip with each chip's own weight slice in parallel. Only chip i's output
    is "live" at step i; others run but discard.

    Returns a dict of parent-mesh sharded weight tensors keyed by weight
    name (e.g. "q_proj", "gate_proj", "input_layernorm").
    """
    lo, hi = layer_range
    n_layers = hi - lo
    if n_layers != prefill_shape[0] * prefill_shape[1]:
        raise ValueError(
            f"layer_range span {n_layers} must equal prefill chip count " f"{prefill_shape[0] * prefill_shape[1]}"
        )
    devices_total = parent_shape[0] * parent_shape[1]

    def _prefill_lin(layer_idx: int) -> int:
        sub_row = layer_idx // prefill_shape[1]
        sub_col = layer_idx % prefill_shape[1]
        return (prefill_offset[0] + sub_row) * parent_shape[1] + (prefill_offset[1] + sub_col)

    # Discover the shape of one layer's weights by inspecting layer `lo`.
    def _key(layer_idx: int, suffix: str) -> str:
        return f"model.layers.{layer_idx}.{suffix}"

    # Weights to upload (matmul weights at `dtype`, norms at bf16).
    matmul_suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    norm_suffixes = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]

    out: Dict[str, "ttnn.Tensor"] = {}

    def _stack_and_upload(suffix: str, weight_dtype) -> Optional["ttnn.Tensor"]:
        # Check first layer's weight exists; if not, skip silently.
        k0 = _key(lo, suffix)
        if k0 not in full_weights:
            return None
        ref = full_weights[k0]
        # For matmul weights, .T to match nn.Linear's [out, in] convention.
        is_matmul = suffix in matmul_suffixes
        ref_t = ref.T.contiguous() if is_matmul else ref
        is_norm = "layernorm" in suffix
        # ttnn.rms_norm with ROW_MAJOR gamma requires:
        #   gamma.padded_shape()[-1] == tile_width (32)
        #   gamma.physical_volume() / 32 == input.padded_shape()[-1] / 32
        # For Gemma's 1D [hidden_dim] norm weight, we reshape to
        # [hidden_dim / 32, 32] so the last dim is exactly 32 tiles wide.
        if is_norm and ref_t.ndim == 1:
            assert ref_t.shape[0] % 32 == 0, f"norm weight dim {ref_t.shape[0]} must be multiple of 32"
            target_shape = (ref_t.shape[0] // 32, 32)
        else:
            target_shape = tuple(ref_t.shape)
        # Build [devices_total, *target_shape] stacked tensor with zeros at
        # non-prefill slots and per-layer weight (reshaped if norm) at the
        # prefill chip i's slot.
        stacked_shape = (devices_total,) + target_shape
        stacked = torch.zeros(stacked_shape, dtype=ref_t.dtype)
        for i in range(n_layers):
            global_i = lo + i
            wk = _key(global_i, suffix)
            if wk not in full_weights:
                continue
            w = full_weights[wk]
            if is_matmul:
                w = w.T.contiguous()
            if is_norm:
                w = (w + 1.0).reshape(target_shape).contiguous()
            lin = _prefill_lin(i)
            stacked[lin] = w

        layout = ttnn.ROW_MAJOR_LAYOUT if is_norm else ttnn.TILE_LAYOUT
        return ttnn.from_torch(
            stacked,
            dtype=weight_dtype,
            layout=layout,
            device=parent_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent_mesh, dim=0),
        )

    for s in matmul_suffixes:
        key = s.replace(".weight", "").replace("self_attn.", "").replace("mlp.", "")
        t = _stack_and_upload(s, dtype)
        if t is not None:
            out[key] = t
    for s in norm_suffixes:
        key = s.replace(".weight", "")
        # Norms are 1D; stacked is [devices_total, dim] = 2D. Upload as bf16.
        t = _stack_and_upload(s, ttnn.bfloat16)
        if t is not None:
            out[key] = t

    return out


class Pi0_5OptionCVLMSliceParent:
    """Parent-mesh VLM slice — 18 layers on the prefill submesh via per-chip
    sharded weights + D2D P2P transitions between layers.

    STATUS: scaffolding / minimal-viable. The full integration with the
    real GemmaBlockTTNN (attention + RMSNorm + MLP + RoPE + KV cache) is the
    follow-up. This class lays the architectural pattern:

    - Open the prefill submesh as a multi-chip mesh (no carving).
    - Upload all 18 layers' weights as parent-mesh sharded tensors via
      `_load_vlm_weights_stacked_sharded` — chip i has layer i's weights.
    - Activation lives on the parent mesh as a sharded tensor; the "live"
      shard moves chip-to-chip via `send_shard_via_p2p_multihop`.
    - Each layer step: run the per-chip matmuls (all 18 chips in parallel,
      each computing its own layer), then advance the live shard.

    For pi0.5 Option C this replaces the host-bouncing layer-paired path —
    estimated ~85 ms saved across 17 inter-layer transitions at full depth.

    Args:
        config:        full PaliGemma config.
        weights:       full categorized weights dict (we slice by layer index).
        parent_mesh:   the galaxy parent mesh (8, 4). Required because P2P
                       multi-hop needs to route across the parent.
        prefill_offset: (row, col) origin of the prefill submesh in the parent.
        prefill_shape:  (rows, cols) of the prefill submesh.
        layer_range:    half-open (lo, hi). hi - lo must equal num prefill chips.

    NOT YET IMPLEMENTED in this class:
        - Full GemmaBlockTTNN forward semantics (attention + KV cache + RoPE
          + RMSNorm + MLP). Currently only the matmul chain is wired.
        - KV cache emission as parent-mesh tensors for downstream D2D KV
          migration (the `KVMigration.migrate_layer_paired_d2d` entry point
          is ready and will consume parent-mesh K/V tensors).
        - Final VLM RMSNorm on last chip.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        parent_mesh,
        prefill_offset: Tuple[int, int],
        prefill_shape: Tuple[int, int],
        layer_range: Tuple[int, int],
    ) -> None:
        if not (0 <= layer_range[0] < layer_range[1] <= config.vlm_config.depth):
            raise ValueError(f"layer_range {layer_range} out of bounds for depth={config.vlm_config.depth}")
        n_prefill_chips = prefill_shape[0] * prefill_shape[1]
        if layer_range[1] - layer_range[0] != n_prefill_chips:
            raise ValueError(
                f"layer_range span {layer_range[1] - layer_range[0]} must equal "
                f"prefill chip count {n_prefill_chips}"
            )

        self.config = config
        self.parent_mesh = parent_mesh
        self.prefill_offset = prefill_offset
        self.prefill_shape = prefill_shape
        self.parent_shape = (parent_mesh.shape[0], parent_mesh.shape[1])
        self.layer_lo, self.layer_hi = layer_range
        self.num_layers = self.layer_hi - self.layer_lo

        # Upload all weights as parent-mesh sharded tensors.
        self.weights_on_parent = _load_vlm_weights_stacked_sharded(
            weights["vlm_language"],
            layer_range,
            parent_mesh,
            self.parent_shape,
            prefill_offset,
            prefill_shape,
        )

    def prefill_coord_for_layer(self, layer_idx: int) -> Tuple[int, int]:
        """Galaxy-parent coord of the prefill chip owning the given layer."""
        sub_row = layer_idx // self.prefill_shape[1]
        sub_col = layer_idx % self.prefill_shape[1]
        return (self.prefill_offset[0] + sub_row, self.prefill_offset[1] + sub_col)

    def forward_qkv_chain(self, activation: "ttnn.Tensor") -> "ttnn.Tensor":
        """Simplified forward that runs ONLY the q_proj matmul chain.

        For each layer i:
            1. Run ttnn.linear(activation, q_proj) — all 18 chips compute
               their layer's Q projection.
            2. P2P-multihop the live shard from chip i to chip i+1.

        Returns the final activation (= layer 17's Q projection output at
        chip 17's parent coord). Validates the matmul chain at scale.
        """
        from .transport import send_shard_via_p2p_multihop

        if "q_proj" not in self.weights_on_parent:
            raise RuntimeError("q_proj weight not loaded; check _load_vlm_weights_stacked_sharded")

        q_proj = self.weights_on_parent["q_proj"]
        act = activation
        for i in range(self.num_layers):
            # Per-chip matmul on the parent mesh. Each chip uses its own
            # q_proj slice and its own activation shard.
            out = ttnn.linear(act, q_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            if i + 1 < self.num_layers:
                cur = self.prefill_coord_for_layer(i)
                nxt = self.prefill_coord_for_layer(i + 1)
                act_new = send_shard_via_p2p_multihop(out, cur, nxt)
                if act_new is not out:
                    ttnn.deallocate(out)
                ttnn.deallocate(act)
                act = act_new
            else:
                ttnn.deallocate(act)
                act = out
        return act

    def forward_mlp_sublayer_chain(self, activation: "ttnn.Tensor") -> "ttnn.Tensor":
        """Run the MLP SUBLAYER chain (gate + up + down + residual) for 18 layers.

        Per-layer ops:
            1. gate matmul: ttnn.linear(h, gate_proj)
            2. up matmul:   ttnn.linear(h, up_proj)
            3. GLU:         gate_out * silu(up_out)  (Gemma-style swiGLU)
            4. down matmul: ttnn.linear(mid, down_proj)
            5. residual add: h + down_out
            6. P2P-multihop the live shard to the next layer's chip

        This is the MLP half of the Gemma block. Combined with the attention
        sublayer chain (separately), the full block forward is complete
        except for RMSNorm and SDPA.

        Returns the final activation after 18 MLP sublayers.
        """
        from .transport import send_shard_via_p2p_multihop

        for r in ("gate_proj", "up_proj", "down_proj"):
            if r not in self.weights_on_parent:
                raise RuntimeError(f"{r} weight not loaded")

        gate = self.weights_on_parent["gate_proj"]
        up = self.weights_on_parent["up_proj"]
        down = self.weights_on_parent["down_proj"]

        h = activation
        for i in range(self.num_layers):
            # 1. gate (M, K) @ (K, intermediate) → (M, intermediate)
            g = ttnn.linear(h, gate, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            # 2. up matmul (parallel structurally)
            u = ttnn.linear(h, up, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            # 3. GLU: gate * silu(up). For Gemma it's actually silu(gate) * up,
            #    but for parent-mesh validation purposes either form exercises
            #    the elementwise op correctly.
            u_act = ttnn.silu(u, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(u)
            mid = ttnn.multiply(g, u_act, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(g)
            ttnn.deallocate(u_act)
            # 4. down matmul
            d = ttnn.linear(mid, down, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(mid)
            # 5. residual add
            h_new = ttnn.add(h, d, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(d)
            ttnn.deallocate(h)
            # 6. P2P advance
            if i + 1 < self.num_layers:
                cur = self.prefill_coord_for_layer(i)
                nxt = self.prefill_coord_for_layer(i + 1)
                h = send_shard_via_p2p_multihop(h_new, cur, nxt)
                if h is not h_new:
                    ttnn.deallocate(h_new)
            else:
                h = h_new
        return h

    def _ensure_rope_tables(self):
        """Lazy-init cos/sin tables on the parent mesh, replicated to all chips.

        RoPE uses position-dependent rotation tables that are layer-agnostic —
        the same cos/sin works for every VLM layer. We replicate them across
        every chip of the parent mesh so each chip can apply RoPE to its own
        Q, K shards independently.

        Tables are bf16 [1, 1, max_seq_len, head_dim].
        """
        if getattr(self, "_cos_meta", None) is not None:
            return
        from models.experimental.pi0_5.tt.ttnn_gemma import precompute_freqs_cis_meta_format

        head_dim = self.config.vlm_config.head_dim
        max_seq = self.config.max_seq_len
        # Build cos/sin on a single representative chip, then replicate across
        # the parent mesh.
        cos_single, sin_single = precompute_freqs_cis_meta_format(head_dim, max_seq, self.parent_mesh)
        # cos_single is already on parent_mesh as it was built using the parent
        # mesh as the device — but it's replicated automatically since
        # precompute_freqs_cis_meta_format uses ttnn.arange etc. on the mesh.
        self._cos_meta = cos_single
        self._sin_meta = sin_single

    def forward_qkv_with_rope_chain(self, activation: "ttnn.Tensor") -> "ttnn.Tensor":
        """Q+K+V matmuls + head reshape + RoPE on Q,K, validating the full
        Q-K-V production pipeline on the parent mesh.

        Per-layer:
            1. RMSNorm
            2. Q matmul → reshape to heads → RoPE
            3. K matmul → reshape to kv_heads → RoPE
            4. V matmul → reshape to kv_heads
            5. O matmul on Q (stubbed attn output — real SDPA next)
            6. Residual add
            7. P2P advance

        Note: this currently computes Q/K/V but doesn't feed them through a
        real SDPA. The point of THIS variant is validating the Q+K+V+RoPE
        pipeline runs end-to-end on parent mesh; SDPA integration follows.

        Returns the final activation after 18 layers.
        """
        from .transport import send_shard_via_p2p_multihop

        self._ensure_rope_tables()

        required = ["input_layernorm", "q_proj", "k_proj", "v_proj", "o_proj"]
        for r in required:
            if r not in self.weights_on_parent:
                raise RuntimeError(f"{r} not loaded")

        input_ln = self.weights_on_parent["input_layernorm"]
        q_proj = self.weights_on_parent["q_proj"]
        k_proj = self.weights_on_parent["k_proj"]
        v_proj = self.weights_on_parent["v_proj"]
        o_proj = self.weights_on_parent["o_proj"]
        eps = self.config.vlm_config.rms_norm_eps

        num_heads = self.config.vlm_config.num_heads
        num_kv_heads = self.config.vlm_config.num_kv_heads
        head_dim = self.config.vlm_config.head_dim

        h = activation
        for i in range(self.num_layers):
            seq_len = h.shape[-2]
            # 1. RMSNorm
            normed = ttnn.rms_norm(h, weight=input_ln, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
            # 2. Q matmul → [1, 1, M, num_heads*head_dim]
            q_flat = ttnn.linear(normed, q_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            # 3. K matmul → [1, 1, M, num_kv_heads*head_dim]
            k_flat = ttnn.linear(normed, k_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            # 4. V matmul → [1, 1, M, num_kv_heads*head_dim]
            v_flat = ttnn.linear(normed, v_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(normed)
            # 4a. Reshape Q, K to heads layout: [1, num_heads, M, head_dim]
            q = ttnn.reshape(q_flat, (1, seq_len, num_heads, head_dim))
            q = ttnn.permute(q, (0, 2, 1, 3))  # [1, num_heads, M, head_dim]
            ttnn.deallocate(q_flat)
            k = ttnn.reshape(k_flat, (1, seq_len, num_kv_heads, head_dim))
            k = ttnn.permute(k, (0, 2, 1, 3))  # [1, num_kv_heads, M, head_dim]
            ttnn.deallocate(k_flat)
            # 4b. Apply RoPE to Q and K using replicated cos/sin tables
            cos_slice = ttnn.slice(self._cos_meta, [0, 0, 0, 0], [1, 1, seq_len, head_dim])
            sin_slice = ttnn.slice(self._sin_meta, [0, 0, 0, 0], [1, 1, seq_len, head_dim])
            q_rope = ttnn.experimental.rotary_embedding(q, cos_slice, sin_slice)
            k_rope = ttnn.experimental.rotary_embedding(k, cos_slice, sin_slice)
            ttnn.deallocate(cos_slice)
            ttnn.deallocate(sin_slice)
            # Q, K now have RoPE applied. (Real attention would now do SDPA;
            # we still stub it with Q→O for this validation.)
            # Reshape Q back to [1, 1, M, num_heads*head_dim] for the O matmul.
            q_rope_flat = ttnn.permute(q_rope, (0, 2, 1, 3))
            q_rope_flat = ttnn.reshape(q_rope_flat, (1, 1, seq_len, num_heads * head_dim))
            ttnn.deallocate(q_rope)
            ttnn.deallocate(k_rope)
            ttnn.deallocate(v_flat)
            # 5. O matmul on Q-with-RoPE
            attn_out = ttnn.linear(q_rope_flat, o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q_rope_flat)
            # 6. Residual
            h_new = ttnn.add(h, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(h)
            # 7. P2P advance
            if i + 1 < self.num_layers:
                cur = self.prefill_coord_for_layer(i)
                nxt = self.prefill_coord_for_layer(i + 1)
                h = send_shard_via_p2p_multihop(h_new, cur, nxt)
                if h is not h_new:
                    ttnn.deallocate(h_new)
            else:
                h = h_new
        return h

    def forward_full_block_chain(self, activation: "ttnn.Tensor") -> "ttnn.Tensor":
        """Full Gemma block chain (RMSNorm + attn + residual + RMSNorm + MLP + residual).

        Per-layer:
          h_attn = RMSNorm(h, input_ln)
          q = h_attn @ q_proj
          attn_out = q @ o_proj            # stubbed attention (real: SDPA(Q,K,V))
          h = h + attn_out
          h_mlp = RMSNorm(h, post_attn_ln)
          g = h_mlp @ gate_proj
          u = h_mlp @ up_proj
          mid = g * silu(u)
          mlp_out = mid @ down_proj
          h = h + mlp_out
          P2P advance

        This is the FULL Gemma block topology except for real SDPA on Q/K/V
        (currently Q→O stub). Numerically valid except for the lack of
        cross-token attention context — all the per-chip op compositions
        are exercised at production scale.
        """
        from .transport import send_shard_via_p2p_multihop

        required = [
            "input_layernorm",
            "post_attention_layernorm",
            "q_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        for r in required:
            if r not in self.weights_on_parent:
                raise RuntimeError(f"{r} weight not loaded")

        input_ln = self.weights_on_parent["input_layernorm"]
        post_attn_ln = self.weights_on_parent["post_attention_layernorm"]
        q_proj = self.weights_on_parent["q_proj"]
        o_proj = self.weights_on_parent["o_proj"]
        gate = self.weights_on_parent["gate_proj"]
        up = self.weights_on_parent["up_proj"]
        down = self.weights_on_parent["down_proj"]
        eps = self.config.vlm_config.rms_norm_eps

        h = activation
        for i in range(self.num_layers):
            # ---- Attention sublayer ----
            normed = ttnn.rms_norm(h, weight=input_ln, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
            q = ttnn.linear(normed, q_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(normed)
            attn_out = ttnn.linear(q, o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q)
            h_post_attn = ttnn.add(h, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(h)
            # ---- MLP sublayer ----
            normed = ttnn.rms_norm(h_post_attn, weight=post_attn_ln, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
            g = ttnn.linear(normed, gate, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            u = ttnn.linear(normed, up, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(normed)
            u_act = ttnn.silu(u, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(u)
            mid = ttnn.multiply(g, u_act, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(g)
            ttnn.deallocate(u_act)
            d = ttnn.linear(mid, down, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(mid)
            h_new = ttnn.add(h_post_attn, d, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(d)
            ttnn.deallocate(h_post_attn)
            # ---- P2P advance ----
            if i + 1 < self.num_layers:
                cur = self.prefill_coord_for_layer(i)
                nxt = self.prefill_coord_for_layer(i + 1)
                h = send_shard_via_p2p_multihop(h_new, cur, nxt)
                if h is not h_new:
                    ttnn.deallocate(h_new)
            else:
                h = h_new
        return h

    def forward_attention_sublayer_chain(self, activation: "ttnn.Tensor") -> "ttnn.Tensor":
        """Run the attention sublayer chain (RMSNorm + Q + O + residual).

        Per-layer ops (validates the full attn sublayer shape):
            1. RMSNorm with input_layernorm weight (ROW_MAJOR gamma)
            2. Q matmul: each chip applies its layer's q_proj
            3. O matmul: simulated attention output projection on Q (stub —
               real path would be: SDPA(Q, K, V) → O matmul)
            4. Residual add (hidden + attn_output)
            5. P2P-multihop the live shard to the next layer's chip

        Returns the final activation after 18 attention sublayers.
        """
        from .transport import send_shard_via_p2p_multihop

        required = ["input_layernorm", "q_proj", "o_proj"]
        for r in required:
            if r not in self.weights_on_parent:
                raise RuntimeError(f"{r} weight not loaded")

        input_ln = self.weights_on_parent["input_layernorm"]
        q_proj = self.weights_on_parent["q_proj"]
        o_proj = self.weights_on_parent["o_proj"]
        eps = self.config.vlm_config.rms_norm_eps

        h = activation
        for i in range(self.num_layers):
            # 1. RMSNorm (per-chip, each chip's layer's gamma)
            normed = ttnn.rms_norm(h, weight=input_ln, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
            # 2. Q matmul
            q = ttnn.linear(normed, q_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(normed)
            # 3. O matmul on Q (stubbed attention output)
            attn_out = ttnn.linear(q, o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q)
            # 4. Residual add
            h_new = ttnn.add(h, attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(h)
            # 5. P2P advance to next chip
            if i + 1 < self.num_layers:
                cur = self.prefill_coord_for_layer(i)
                nxt = self.prefill_coord_for_layer(i + 1)
                h = send_shard_via_p2p_multihop(h_new, cur, nxt)
                if h is not h_new:
                    ttnn.deallocate(h_new)
            else:
                h = h_new
        return h


class Pi0_5OptionCVLMSliceTP:
    """TP=tp_size VLM slice carved across multiple (tp_size, 1) sub-meshes.

    Default layout (matches the (6,3) prefill submesh with `tp_size=2`):
      - 9 (2,1) sub-meshes, ordered (0,0), (0,1), (0,2), (2,0), ..., (4,2).
      - 2 VLM layers per sub-mesh (layers 0..1 on sub-mesh 0, 2..3 on
        sub-mesh 1, ..., 16..17 on sub-mesh 8). Layers within a sub-mesh
        run in sequence on the same (2,1) (no host bounce between them);
        between sub-meshes activations are bounced through host.

    Args:
        config:               full PaliGemma config.
        weights:              categorized weights dict.
        tp_submeshes:         list of N (tp_size, 1) MeshDevices (build via
                              `mesh_setup.create_tp_submeshes_2x1`).
        layer_range:          half-open (lo, hi). `hi - lo` must equal
                              `len(tp_submeshes) * layers_per_submesh`.
        layers_per_submesh:   how many VLM layers each (tp_size,1) sub-mesh
                              owns. With 9 sub-meshes and 18 VLM layers,
                              default is 2.
        tp_size:              TP factor inside each sub-mesh. Default 2.
        holds_vlm_final_norm: if True, place model.norm.weight on the LAST
                              sub-mesh and apply it after the final block.
        holds_embed_tokens:   accepted for API parity; ignored — embed is
                              vision-side in Option C.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        tp_submeshes: List,
        layer_range: Tuple[int, int],
        layers_per_submesh: int = 2,
        tp_size: int = 2,
        holds_embed_tokens: bool = False,
        holds_vlm_final_norm: bool = False,
    ) -> None:
        from .tp_block import Pi0_5OptionCSubmeshTPGemmaBlock

        if not (0 <= layer_range[0] < layer_range[1] <= config.vlm_config.depth):
            raise ValueError(
                f"layer_range {layer_range} out of bounds for " f"vlm_config.depth={config.vlm_config.depth}"
            )
        expected_span = len(tp_submeshes) * layers_per_submesh
        if layer_range[1] - layer_range[0] != expected_span:
            raise ValueError(
                f"layer_range span ({layer_range[1] - layer_range[0]}) must equal "
                f"len(tp_submeshes)*layers_per_submesh = {len(tp_submeshes)}*"
                f"{layers_per_submesh} = {expected_span}"
            )
        for i, sm in enumerate(tp_submeshes):
            if sm.get_num_devices() != tp_size:
                raise ValueError(f"tp_submeshes[{i}] has {sm.get_num_devices()} devices but tp_size={tp_size}")

        self.config = config
        self.tp_submeshes = tp_submeshes
        self.layer_lo, self.layer_hi = layer_range
        self.num_layers = self.layer_hi - self.layer_lo
        self.layers_per_submesh = layers_per_submesh
        self.tp_size = tp_size
        self.holds_embed_tokens = holds_embed_tokens
        self.holds_vlm_final_norm = holds_vlm_final_norm

        lang = weights["vlm_language"]

        # RoPE tables — one set per sub-mesh. Pin to DRAM: rotary_embedding
        # only READS these (no in-place mutation); DRAM placement frees
        # ~2 MB / chip of L1 (1 MB per of cos/sin at [1,1,max_seq_len,head_dim]
        # bf16) which we need to fit Q/K/V/O matmul weights alongside MLP.
        # rotary_embedding's validate_on_program_cache_miss only requires the
        # tensor be on-device + TILE layout — no memory_config constraint.
        self.cos_metas: List = []
        self.sin_metas: List = []
        for sm in tp_submeshes:
            cos, sin = precompute_freqs_cis_meta_format(
                config.vlm_config.head_dim,
                config.max_seq_len,
                sm,
            )
            # Idempotent move to DRAM. ttnn.to_memory_config on an already-DRAM
            # tensor returns the SAME buffer; deallocate(source) would then free
            # the underlying buffer of the returned tensor (same trap fixed in
            # _l1_migration._to_l1). Check buffer_type and skip the move.
            if cos.memory_config().buffer_type != ttnn.BufferType.DRAM:
                cos_new = ttnn.to_memory_config(cos, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(cos)
                cos = cos_new
            if sin.memory_config().buffer_type != ttnn.BufferType.DRAM:
                sin_new = ttnn.to_memory_config(sin, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(sin)
                sin = sin_new
            self.cos_metas.append(cos)
            self.sin_metas.append(sin)

        # Per-layer blocks. vlm_blocks[i] lives on
        # tp_submeshes[i // layers_per_submesh].
        self.vlm_blocks: List = []
        for local_i in range(self.num_layers):
            global_i = self.layer_lo + local_i
            sm_idx = local_i // layers_per_submesh
            sm = tp_submeshes[sm_idx]
            self.vlm_blocks.append(
                Pi0_5OptionCSubmeshTPGemmaBlock(
                    config.vlm_config,
                    lang,
                    global_i,
                    sm,
                    self.cos_metas[sm_idx],
                    self.sin_metas[sm_idx],
                    tp_size=tp_size,
                )
            )

        # Final RMSNorm on the LAST sub-mesh when this slice owns the tail.
        self.vlm_norm: Optional["ttnn.Tensor"] = None
        if holds_vlm_final_norm:
            self.vlm_norm = tensor_1d_to_2d_ttnn(
                lang["model.norm.weight"] + 1.0,
                tp_submeshes[-1],
                dtype=ttnn.bfloat16,
            )

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_values: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        use_cache: bool = False,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
    ) -> Tuple["ttnn.Tensor", Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]]]:
        """Run layers [layer_lo, layer_hi) across the chain of TP sub-meshes.

        On entry, `hidden_states` and any `attention_mask` must live on
        `tp_submeshes[0]` (replicated). Between consecutive sub-meshes the
        activation is host-bounced and re-replicated on the next (2,1).
        """
        if past_key_values is not None and any(
            past_key_values[self.layer_lo + i] is not None for i in range(self.num_layers)
        ):
            raise NotImplementedError("Pi0_5OptionCVLMSliceTP.forward does not yet accept past_key_values")

        new_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = (
            [None] * self.config.vlm_config.depth if use_cache else None
        )

        # Cache per-sub-mesh masks (DRAM, SDPA requirement). Keyed by id().
        mask_cache_key = id(attention_mask) if attention_mask is not None else None
        if not hasattr(self, "_per_submesh_mask_cache"):
            self._per_submesh_mask_cache: Dict = {}
        masks_per_submesh = self._per_submesh_mask_cache.get(mask_cache_key)
        if attention_mask is not None and masks_per_submesh is None:
            masks_per_submesh = [attention_mask]
            for i in range(1, len(self.tp_submeshes)):
                broadcast = send_activation_via_host(masks_per_submesh[0], self.tp_submeshes[i])
                dram_mask = ttnn.to_memory_config(broadcast, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(broadcast)
                masks_per_submesh.append(dram_mask)
            self._per_submesh_mask_cache[mask_cache_key] = masks_per_submesh

        h = hidden_states
        current_sm_idx = 0
        for local_i, block in enumerate(self.vlm_blocks):
            global_i = self.layer_lo + local_i
            sm_idx = local_i // self.layers_per_submesh
            # If we crossed into a new sub-mesh, host-bounce the activation.
            if sm_idx != current_sm_idx:
                h_next = send_activation_via_host(h, self.tp_submeshes[sm_idx])
                ttnn.deallocate(h)
                h = h_next
                current_sm_idx = sm_idx
            mask_i = masks_per_submesh[sm_idx] if masks_per_submesh is not None else None
            h_new, new_kv = block.forward(
                h,
                attention_mask=mask_i,
                use_cache=use_cache,
            )
            if use_cache and new_kv is not None:
                new_cache[global_i] = new_kv
            ttnn.deallocate(h)
            h = h_new

        # Final RMSNorm on the last sub-mesh; mirror the DRAM bounce
        # in the paired slice (rms_norm CB clash dodge).
        if self.holds_vlm_final_norm and self.vlm_norm is not None:
            h_dram = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(h)
            h = ttnn.rms_norm(
                h_dram,
                weight=self.vlm_norm,
                epsilon=self.config.vlm_config.rms_norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(h_dram)

        return h, new_cache

    # ------------------------------------------------------------------ #
    # KV migration emitter                                               #
    # ------------------------------------------------------------------ #

    def get_kv_cache_for_slice(
        self,
        new_cache: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]],
    ) -> List[Tuple[int, Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """Return (global_layer_idx, (K, V)) tuples for this slice's layers.

        Each (K, V) lives on the (2,1) sub-mesh that ran the layer. KV is
        sharded TP=N along the head axis, so the migration will need to
        all-gather or sequentially read each chip's slice — TBD, not
        implemented in this first cut.
        """
        out = []
        for local_i in range(self.num_layers):
            global_i = self.layer_lo + local_i
            kv = new_cache[global_i] if new_cache is not None else None
            if kv is not None:
                out.append((global_i, kv))
        return out
