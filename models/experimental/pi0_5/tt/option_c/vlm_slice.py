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
) -> Dict[str, "ttnn.Tensor"]:
    """Upload one VLM layer's weights onto a single-chip micro-submesh, L1-resident.

    Identical key mapping to `_load_vlm_block_weights_l1` but every upload
    is single-chip + L1. With only one layer's weights per chip we sit well
    inside the 180 MB L1 budget (~110 MB weights bf8 + headroom).
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
        if "weight" in new_key and not is_norm:
            value = value.T
        if is_norm:
            value = value + 1.0  # Gemma-style +1 offset, pre-baked

        if len(value.shape) == 1:
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, micro_submesh, dtype=ttnn.bfloat16)
        else:
            weight_dtype = ttnn.bfloat16 if is_norm else ttnn.bfloat8_b
            block_weights[new_key] = _upload_single_chip_l1(value.contiguous(), micro_submesh, weight_dtype)
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
            block_weights = _load_vlm_block_weights_single_chip_l1(lang, global_i, sm)
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
