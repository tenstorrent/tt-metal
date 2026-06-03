# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Action-expert slice for Option C (no TP, L1-resident).

Mirrors `option_c.vlm_slice` but for the Gemma-300M action expert with adaRMS
modulation. Per the deployment plan §3.1, each denoise chip holds 3 expert
layers; total weights per chip are ~18 MB, well inside the L1 budget.

This first cut uses REPLICATED weights across the 6-chip denoise submesh.
Layer-paired sharding (3 layers per chip, not replicated) is the follow-up
that brings real-config memory in line.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import (
    AdaRMSGemmaBlockTTNN,
    ada_rms_norm_no_gate_ttnn,
    ada_rms_norm_no_gate_precomputed_ttnn,
    precompute_freqs_cis_meta_format,
)

from .transport import send_activation_via_host
from .vlm_slice import _upload_l1_replicated, _upload_single_chip_l1


def _load_expert_block_weights_l1(
    full_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    submesh,
) -> Dict[str, "ttnn.Tensor"]:
    """Upload one expert layer's weights onto `submesh` (replicated, L1-resident).

    Identical key mapping to `option_b.expert_slice._load_expert_block_weights`
    but routes every upload through `_upload_l1_replicated`.
    """
    prefix = f"model.layers.{layer_idx}."
    block_weights: Dict[str, "ttnn.Tensor"] = {}

    # Fused QKV (col-concat of q/k/v after .T)
    q_key, k_key, v_key = (
        f"{prefix}self_attn.q_proj.weight",
        f"{prefix}self_attn.k_proj.weight",
        f"{prefix}self_attn.v_proj.weight",
    )
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

        # Skip individual Q/K/V (already fused) and adaRMS Denses (handled below).
        if new_key in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "input_layernorm.dense.weight",
            "input_layernorm.dense.bias",
            "post_attention_layernorm.dense.weight",
            "post_attention_layernorm.dense.bias",
        ):
            continue
        if new_key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            continue

        is_norm = "norm" in new_key
        if "weight" in new_key and not is_norm:
            value = value.T

        if len(value.shape) == 1:
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, submesh, dtype=ttnn.bfloat16)
        else:
            block_weights[new_key] = _upload_l1_replicated(
                value.contiguous(),
                submesh,
                ttnn.bfloat16 if is_norm else ttnn.bfloat8_b,
            )

    # Fused adaRMS modulation weight = concat([pre_attn.dense, pre_ffw.dense], dim=0).
    w_keys = [
        f"{prefix}input_layernorm.dense.weight",
        f"{prefix}post_attention_layernorm.dense.weight",
    ]
    for wk in w_keys:
        if wk not in full_weights:
            raise KeyError(f"expert layer {layer_idx} missing adaRMS weight '{wk}'")
    fused_w = torch.cat([full_weights[wk] for wk in w_keys], dim=0).contiguous()
    block_weights["adarms_mod.weight"] = _upload_l1_replicated(
        fused_w.T.contiguous(),
        submesh,
        ttnn.bfloat16,
    )

    b_keys = [
        f"{prefix}input_layernorm.dense.bias",
        f"{prefix}post_attention_layernorm.dense.bias",
    ]
    biases = [full_weights[bk] for bk in b_keys if bk in full_weights]
    if biases:
        assert len(biases) == 2, "expected both adaRMS biases or neither"
        fused_b = torch.cat(biases, dim=0).contiguous()
        block_weights["adarms_mod.bias"] = tensor_1d_to_2d_ttnn(fused_b, submesh, dtype=ttnn.bfloat16)

    return block_weights


class Pi0_5OptionCExpertSlice:
    """Action-expert layers + final adaRMS-norm Dense on the denoise submesh.

    Args:
        config:              full PaliGemma config.
        weights:             full weights dict (we slice by layer index).
        submesh:             the denoise MeshDevice (6 chips for Option C).
        expert_layer_range:  half-open (lo, hi). For scaffolding pass this is
                             narrow so the replicated weights fit per-chip;
                             real Option C uses layer-paired sharding to
                             distribute 3 layers per chip.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        submesh,
        expert_layer_range: Tuple[int, int] = (0, 18),
    ) -> None:
        if not (0 <= expert_layer_range[0] < expert_layer_range[1] <= config.expert_config.depth):
            raise ValueError(
                f"expert_layer_range {expert_layer_range} out of bounds for "
                f"expert_config.depth={config.expert_config.depth}"
            )

        self.config = config
        self.submesh = submesh
        self.layer_lo, self.layer_hi = expert_layer_range
        self.num_layers = self.layer_hi - self.layer_lo

        ae = weights["action_expert"]

        # Expert RoPE — same precompute as VLM but with expert head_dim.
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(
            config.expert_config.head_dim,
            config.max_seq_len,
            submesh,
        )

        self.expert_blocks: List = []
        for i in range(self.layer_lo, self.layer_hi):
            block_weights = _load_expert_block_weights_l1(ae, i, submesh)
            self.expert_blocks.append(
                AdaRMSGemmaBlockTTNN(
                    config.expert_config,
                    block_weights,
                    i,
                    submesh,
                    self.cos_meta,
                    self.sin_meta,
                )
            )

        # Final adaRMS norm Dense.
        if "model.norm.dense.weight" not in ae:
            raise KeyError("expert checkpoint missing 'model.norm.dense.weight'")
        self.final_norm_mod_weight = _upload_l1_replicated(
            ae["model.norm.dense.weight"].T.contiguous(),
            submesh,
            ttnn.bfloat16,
        )
        self.final_norm_mod_bias = None
        if "model.norm.dense.bias" in ae:
            self.final_norm_mod_bias = tensor_1d_to_2d_ttnn(ae["model.norm.dense.bias"], submesh, dtype=ttnn.bfloat16)

        device_grid = submesh.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        prefix_kv_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None,
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        precomputed_block_mods: Optional[List[Tuple["ttnn.Tensor", ...]]] = None,
        precomputed_final_mod: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
        keep_padded: bool = False,
    ) -> "ttnn.Tensor":
        """Run all expert layers in the slice. Returns post-final-norm hidden.

        prefix_kv_cache is indexed by GLOBAL expert layer index. The denoise
        stage passes the layer-paired migrated VLM KV here.
        """
        for local_i, block in enumerate(self.expert_blocks):
            global_i = self.layer_lo + local_i
            past_kv = prefix_kv_cache[global_i] if prefix_kv_cache is not None else None
            block_mod = precomputed_block_mods[local_i] if precomputed_block_mods is not None else None
            hidden_states, _new_kv = block.forward(
                hidden_states,
                cos_override,
                sin_override,
                adarms_cond,
                attention_mask,
                position_ids,
                past_kv,
                use_cache=False,
                precomputed_mod=block_mod,
                keep_padded=keep_padded,
            )

        if precomputed_final_mod is not None:
            sf1, tf = precomputed_final_mod
            hidden_states = ada_rms_norm_no_gate_precomputed_ttnn(
                hidden_states, sf1, tf, self.config.expert_config.rms_norm_eps
            )
        else:
            hidden_states = ada_rms_norm_no_gate_ttnn(
                hidden_states,
                adarms_cond,
                self.final_norm_mod_weight,
                self.final_norm_mod_bias,
                self.config.expert_config.rms_norm_eps,
                self.core_grid,
            )
        return hidden_states


# ---------------------------------------------------------------------------- #
# Layer-paired expert slice — 3 layers per chip × 6 chips, L1-resident.         #
# ---------------------------------------------------------------------------- #


def _load_expert_block_weights_single_chip_l1(
    full_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    micro_submesh,
) -> Dict[str, "ttnn.Tensor"]:
    """Single-chip + L1 mirror of `_load_expert_block_weights_l1`."""
    prefix = f"model.layers.{layer_idx}."
    block_weights: Dict[str, "ttnn.Tensor"] = {}

    q_key, k_key, v_key = (
        f"{prefix}self_attn.q_proj.weight",
        f"{prefix}self_attn.k_proj.weight",
        f"{prefix}self_attn.v_proj.weight",
    )
    if q_key in full_weights and k_key in full_weights and v_key in full_weights:
        wq = _upload_single_chip_l1(full_weights[q_key].T.contiguous(), micro_submesh, ttnn.bfloat8_b)
        wk = _upload_single_chip_l1(full_weights[k_key].T.contiguous(), micro_submesh, ttnn.bfloat8_b)
        wv = _upload_single_chip_l1(full_weights[v_key].T.contiguous(), micro_submesh, ttnn.bfloat8_b)
        block_weights["self_attn.wqkv"] = ttnn.concat([wq, wk, wv], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(wq)
        ttnn.deallocate(wk)
        ttnn.deallocate(wv)

    for key, value in full_weights.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix) :]
        if new_key in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "input_layernorm.dense.weight",
            "input_layernorm.dense.bias",
            "post_attention_layernorm.dense.weight",
            "post_attention_layernorm.dense.bias",
        ):
            continue
        if new_key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            continue

        is_norm = "norm" in new_key
        if "weight" in new_key and not is_norm:
            value = value.T

        if len(value.shape) == 1:
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, micro_submesh, dtype=ttnn.bfloat16)
        else:
            block_weights[new_key] = _upload_single_chip_l1(
                value.contiguous(),
                micro_submesh,
                ttnn.bfloat16 if is_norm else ttnn.bfloat8_b,
            )

    w_keys = [
        f"{prefix}input_layernorm.dense.weight",
        f"{prefix}post_attention_layernorm.dense.weight",
    ]
    for wk in w_keys:
        if wk not in full_weights:
            raise KeyError(f"expert layer {layer_idx} missing adaRMS weight '{wk}'")
    fused_w = torch.cat([full_weights[wk] for wk in w_keys], dim=0).contiguous()
    block_weights["adarms_mod.weight"] = _upload_single_chip_l1(fused_w.T.contiguous(), micro_submesh, ttnn.bfloat16)

    b_keys = [
        f"{prefix}input_layernorm.dense.bias",
        f"{prefix}post_attention_layernorm.dense.bias",
    ]
    biases = [full_weights[bk] for bk in b_keys if bk in full_weights]
    if biases:
        assert len(biases) == 2, "expected both adaRMS biases or neither"
        fused_b = torch.cat(biases, dim=0).contiguous()
        block_weights["adarms_mod.bias"] = tensor_1d_to_2d_ttnn(fused_b, micro_submesh, dtype=ttnn.bfloat16)

    return block_weights


class Pi0_5OptionCExpertSlicePaired:
    """Action-expert slice with layer-paired L1 placement on the denoise submesh.

    Target placement (deployment plan §3.1):
        chip 0 holds expert layers 0–2
        chip 1 holds expert layers 3–5
        ...
        chip 5 holds expert layers 15–17

    Each chip is a 1-chip MeshDevice (carved from the 6-chip denoise submesh
    via `mesh_setup.create_per_chip_submeshes`). Weights for a chip's 3 layers
    live in L1 on that chip; activation host-bounces between consecutive chips.

    Final adaRMS norm Dense lives on the LAST chip in the chain.

    External contract matches `Pi0_5OptionCExpertSlice.forward(...)`:
        forward(h_on_first_chip, adarms_cond_on_first_chip, prefix_kv_cache,
                attention_mask_on_first_chip) -> h_on_last_chip
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        micro_submeshes: List,
        expert_layer_range: Tuple[int, int] = (0, 18),
        layers_per_chip: int = 3,
    ) -> None:
        if not (0 <= expert_layer_range[0] < expert_layer_range[1] <= config.expert_config.depth):
            raise ValueError(
                f"expert_layer_range {expert_layer_range} out of bounds for "
                f"expert_config.depth={config.expert_config.depth}"
            )
        if layers_per_chip <= 0:
            raise ValueError(f"layers_per_chip must be > 0; got {layers_per_chip}")

        lo, hi = expert_layer_range
        num_layers = hi - lo
        num_chips_needed = (num_layers + layers_per_chip - 1) // layers_per_chip
        if num_chips_needed != len(micro_submeshes):
            raise ValueError(
                f"layer range {expert_layer_range} with layers_per_chip={layers_per_chip} "
                f"needs {num_chips_needed} chips, got {len(micro_submeshes)}"
            )
        for i, sm in enumerate(micro_submeshes):
            if sm.get_num_devices() != 1:
                raise ValueError(f"micro_submeshes[{i}] must be a 1-chip submesh " f"({sm.get_num_devices()} devices)")

        self.config = config
        self.micro_submeshes = micro_submeshes
        self.layer_lo, self.layer_hi = expert_layer_range
        self.num_layers = num_layers
        self.layers_per_chip = layers_per_chip

        ae = weights["action_expert"]

        # RoPE tables + per-layer blocks, grouped by owning chip.
        self.cos_metas: List = []
        self.sin_metas: List = []
        self.expert_blocks: List = []  # flat list, ordered by global layer idx
        self.chip_for_layer: List[int] = []  # local layer i → micro_submesh index

        for chip_idx, sm in enumerate(micro_submeshes):
            cos, sin = precompute_freqs_cis_meta_format(
                config.expert_config.head_dim,
                config.max_seq_len,
                sm,
            )
            self.cos_metas.append(cos)
            self.sin_metas.append(sin)

            chip_layer_lo = self.layer_lo + chip_idx * layers_per_chip
            chip_layer_hi = min(self.layer_hi, chip_layer_lo + layers_per_chip)
            for global_i in range(chip_layer_lo, chip_layer_hi):
                block_weights = _load_expert_block_weights_single_chip_l1(ae, global_i, sm)
                self.expert_blocks.append(
                    AdaRMSGemmaBlockTTNN(
                        config.expert_config,
                        block_weights,
                        global_i,
                        sm,
                        cos,
                        sin,
                    )
                )
                self.chip_for_layer.append(chip_idx)

        # Final adaRMS norm Dense on the last chip.
        last_sm = micro_submeshes[-1]
        if "model.norm.dense.weight" not in ae:
            raise KeyError("expert checkpoint missing 'model.norm.dense.weight'")
        self.final_norm_mod_weight = _upload_single_chip_l1(
            ae["model.norm.dense.weight"].T.contiguous(),
            last_sm,
            ttnn.bfloat16,
        )
        self.final_norm_mod_bias = None
        if "model.norm.dense.bias" in ae:
            self.final_norm_mod_bias = tensor_1d_to_2d_ttnn(ae["model.norm.dense.bias"], last_sm, dtype=ttnn.bfloat16)
        last_grid = last_sm.compute_with_storage_grid_size()
        self.last_core_grid = ttnn.CoreGrid(y=last_grid.y, x=last_grid.x)

        # Per-chip caches for shared inputs (mask, adarms_cond), keyed by id(obj).
        self._per_chip_mask_cache: Dict = {}
        self._per_chip_adarms_cache: Dict = {}

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def _broadcast_to_chips(
        self,
        t: "ttnn.Tensor",
        cache: Dict,
        force_dram: bool = False,
    ) -> List["ttnn.Tensor"]:
        """Materialize one copy of `t` on each micro-submesh, caching by id.

        `force_dram=True` flips each broadcast copy to DRAM after the host
        bounce. Use this for the attention mask — SDPA TT_FATALs on L1 masks.
        """
        if t is None:
            return [None] * len(self.micro_submeshes)
        key = id(t)
        hit = cache.get(key)
        if hit is not None:
            return hit
        per_chip = [t]
        for sm in self.micro_submeshes[1:]:
            broadcast = send_activation_via_host(t, sm)
            if force_dram:
                dram = ttnn.to_memory_config(broadcast, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(broadcast)
                per_chip.append(dram)
            else:
                per_chip.append(broadcast)
        cache[key] = per_chip
        return per_chip

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        prefix_kv_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None,
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        precomputed_block_mods: Optional[List[Tuple["ttnn.Tensor", ...]]] = None,
        precomputed_final_mod: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
        keep_padded: bool = False,
    ) -> "ttnn.Tensor":
        """Run all expert layers across chips. Returns post-final-norm hidden
        on the LAST chip in the chain.

        `hidden_states`, `adarms_cond`, and `attention_mask` must enter on
        the FIRST chip. `prefix_kv_cache` is indexed by GLOBAL expert layer
        idx; entries must live on the chip that owns that layer.
        """
        if precomputed_block_mods is not None or precomputed_final_mod is not None:
            raise NotImplementedError("precomputed mods not yet supported on the layer-paired expert slice")

        # Replicate the per-step shared inputs onto each chip once per call,
        # then run each layer on its chip. Mask broadcast must be DRAM-resident
        # for SDPA; adarms_cond is fine in L1.
        masks_per_chip = self._broadcast_to_chips(attention_mask, self._per_chip_mask_cache, force_dram=True)
        adarms_per_chip = self._broadcast_to_chips(adarms_cond, self._per_chip_adarms_cache)

        h = hidden_states
        current_chip = 0
        for local_i, block in enumerate(self.expert_blocks):
            owner_chip = self.chip_for_layer[local_i]
            if owner_chip != current_chip:
                # Transport activation to the next chip before running the layer.
                h_next = send_activation_via_host(h, self.micro_submeshes[owner_chip])
                ttnn.deallocate(h)
                h = h_next
                current_chip = owner_chip

            global_i = self.layer_lo + local_i
            past_kv = prefix_kv_cache[global_i] if prefix_kv_cache is not None else None
            h, _new_kv = block.forward(
                h,
                cos_override,
                sin_override,
                adarms_per_chip[current_chip],
                masks_per_chip[current_chip],
                position_ids,
                past_kv,
                use_cache=False,
                precomputed_mod=None,
                keep_padded=keep_padded,
            )

        # Final adaRMS norm on the last chip.
        last_chip = len(self.micro_submeshes) - 1
        if current_chip != last_chip:
            h_next = send_activation_via_host(h, self.micro_submeshes[last_chip])
            ttnn.deallocate(h)
            h = h_next
        h = ada_rms_norm_no_gate_ttnn(
            h,
            adarms_per_chip[last_chip],
            self.final_norm_mod_weight,
            self.final_norm_mod_bias,
            self.config.expert_config.rms_norm_eps,
            self.last_core_grid,
        )
        return h
