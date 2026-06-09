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


def _upload_l1_sharded(
    t: torch.Tensor,
    submesh,
    dim: int,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
) -> "ttnn.Tensor":
    """Shard `t` along axis `dim` across every chip in `submesh`, L1-resident.

    Mirrors `option_b.tp_block._shard_along` but defaults to L1 instead of DRAM
    (Option C's whole premise). Used to split the adaRMS modulation Dense's
    6144 output axis across the 6 denoise chips so each chip only holds 1/6
    of the modulation weight — cuts the dominant per-chip load by 6x.

    The matmul output is sharded; an `all_gather` along `dim` materializes the
    full modulation tensor on every chip before applying it per-token.
    """
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG
    mapper = ttnn.shard_tensor_to_mesh_mapper(submesh, dim=dim)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=submesh,
        mesh_mapper=mapper,
        memory_config=memory_config,
    )


def _load_expert_block_weights_l1(
    full_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    submesh,
    mod_sharded: bool = False,
) -> Dict[str, "ttnn.Tensor"]:
    """Upload one expert layer's weights onto `submesh` (replicated, L1-resident).

    Identical key mapping to `option_b.expert_slice._load_expert_block_weights`
    but routes every upload through `_upload_l1_replicated`.

    All weights and biases (norms, mod weight/bias, etc.) are uploaded at
    bf8_b to halve the per-chip load. Matmul Q/K/V/O + gate/up/down were
    already bf8_b. The only thing left at native precision is the host-side
    embed_table, which doesn't live on chip at all.

    When `mod_sharded=True`, the adaRMS mod weight is sharded along its 6144
    output axis across `submesh` (each chip holds 1/6 of the mod output dim).
    The block forward needs an all_gather to materialize the full mod output
    before splitting into the 6 (scale/shift/gate per attn/ffw) tensors —
    see `Pi0_5OptionCExpertSlice.forward` below.
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
            # Keep LN 1D bias at bf16. bf8_b compounds a measurable PCC drop
            # at depth=18 (validated by bisect: 0.40 with bf8 vs target ≥0.85).
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, submesh, dtype=ttnn.bfloat16)
        else:
            # Keep LN 2D weights at bf16 (same rationale as the 1D case).
            block_weights[new_key] = _upload_l1_replicated(
                value.contiguous(),
                submesh,
                ttnn.bfloat16 if is_norm else ttnn.bfloat8_b,
            )

    # Fused adaRMS modulation weight = concat([pre_attn.dense, pre_ffw.dense], dim=0).
    # Pre-transpose shape: [1024, 6144]. The 6144 axis is the output dim that's
    # split into 6 W=1024 chunks (sa, ta, ga, sf, tf, gf) inside the block.
    w_keys = [
        f"{prefix}input_layernorm.dense.weight",
        f"{prefix}post_attention_layernorm.dense.weight",
    ]
    for wk in w_keys:
        if wk not in full_weights:
            raise KeyError(f"expert layer {layer_idx} missing adaRMS weight '{wk}'")
    fused_w = torch.cat([full_weights[wk] for wk in w_keys], dim=0).contiguous()
    if mod_sharded:
        # Shard the 6144 output axis across `submesh.get_num_devices()` chips.
        # Each chip holds [1024, 6144/N]; an all_gather at forward time
        # reconstitutes the full mod output before the per-token modulation.
        # Keep bf16 (not bf8_b) for numerics — the mod weight scales the
        # residual stream per token and bf8_b compounds a PCC drop across
        # depth.
        block_weights["adarms_mod.weight"] = _upload_l1_sharded(
            fused_w.T.contiguous(),
            submesh,
            dim=-1,
            dtype=ttnn.bfloat16,
        )
    else:
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
        if mod_sharded:
            # Bias must match the sharded mod output: shape [1, 6144] sharded
            # on its last axis. Use a separate path because `tensor_1d_to_2d_ttnn`
            # is replicate-only. Keep bf16 (matches mod_weight dtype).
            fused_b_2d = fused_b.reshape(1, -1).contiguous()
            block_weights["adarms_mod.bias"] = _upload_l1_sharded(
                fused_b_2d,
                submesh,
                dim=-1,
                dtype=ttnn.bfloat16,
            )
        else:
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
        mod_sharded: bool = False,
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
        # When True, the per-block adaRMS mod weight + bias are sharded across
        # the denoise submesh along the 6144 output axis. The forward path
        # gathers the sharded mod output via ttnn.all_gather before splitting
        # into the 6 per-token modulation tensors (sa, ta, ga, sf, tf, gf).
        # Requires the submesh's parent to have been opened with
        # `open_galaxy_mesh(enable_fabric=True)` so the fabric is available
        # for all_gather. See pipeline.py's `denoise_mod_sharded` flag.
        self.mod_sharded = mod_sharded
        # Number of chips the mod weight is sharded across — used to compute
        # the gather output shape and the per-chip mod chunk size.
        self.mod_shard_world = submesh.get_num_devices() if mod_sharded else 1

        ae = weights["action_expert"]

        # Expert RoPE — same precompute as VLM but with expert head_dim.
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(
            config.expert_config.head_dim,
            config.max_seq_len,
            submesh,
        )

        self.expert_blocks: List = []
        for i in range(self.layer_lo, self.layer_hi):
            block_weights = _load_expert_block_weights_l1(ae, i, submesh, mod_sharded=mod_sharded)
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
        # Keep final_norm_mod_weight + bias at bf16. bf8_b on these compounds
        # PCC drift with the per-block mod weight changes (see paired path).
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

    def _precompute_mod_sharded(self, block, adarms_cond: "ttnn.Tensor") -> Tuple["ttnn.Tensor", ...]:
        """Compute the 6 per-block modulation tensors using a sharded mod weight.

        Steps:
          1. matmul(adarms_cond, block.mod_weight) — output is sharded along
             the last (6144) axis: each chip holds shape [B, 1, 6144/N].
          2. all_gather along the last axis — every chip ends up with the
             full [B, 1, 6144] mod tensor.
          3. split into 6 chunks of width W=1024.
          4. Add 1.0 to the two scale tensors (sa, sf) so they match the
             `precomputed_mod`/`pre_added=True` contract the block uses.

        The block contract: when `precomputed_mod is not None`, the block sets
        `mod_owned=False ⇒ pre_added=True` and calls `_modulated_rms_norm` with
        scale already-baked (i.e. `sa = 1 + raw_scale_a`). We mirror that here
        so the precomputed-path numerics match the in-block path bit-for-bit.

        Requires fabric: `set_fabric_config(FABRIC_1D)` must have been called
        before the parent mesh opened (see open_galaxy_mesh(enable_fabric=True)).
        """
        mod_partial = ttnn.linear(
            adarms_cond,
            block.mod_weight,
            bias=block.mod_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
            compute_kernel_config=block.mod_compute_kernel_config,
        )
        # Materialize the full mod output on every chip by gathering the
        # sharded last axis. Fabric must be enabled at parent-open time.
        mod_full = ttnn.all_gather(
            mod_partial,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(mod_partial)
        # Split into 6 W=1024 chunks. The block expects (B, 1, W) for each.
        B = mod_full.shape[0]
        total = mod_full.shape[-1]
        W = total // 6
        mod3 = ttnn.reshape(mod_full, (B, 1, total))
        # NOTE: ttnn.deallocate(mod_full) is unsafe here — `mod3` is a reshape
        # alias backed by the same buffer. Deallocating it would invalidate
        # the chunk slices below. See `_split_modulation_6` in ttnn_gemma.py
        # for the same pattern.
        sa_raw = mod3[:, :, 0 * W : 1 * W]
        ta = mod3[:, :, 1 * W : 2 * W]
        ga = mod3[:, :, 2 * W : 3 * W]
        sf_raw = mod3[:, :, 3 * W : 4 * W]
        tf = mod3[:, :, 4 * W : 5 * W]
        gf = mod3[:, :, 5 * W : 6 * W]
        # Bake the +1 into the two scale tensors to match the precomputed_mod
        # contract (block sees `pre_added=True`).
        # TODO(full-l1): if validation agent sees PCC drift, the most likely
        # culprit is here — verify the +1 baking matches the in-block path's
        # `scale_plus_one = ttnn.add(scale, 1.0)`.
        sa1 = ttnn.add(sa_raw, 1.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        sf1 = ttnn.add(sf_raw, 1.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        return (sa1, ta, ga, sf1, tf, gf)

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

        When `mod_sharded=True`, the per-block mod weight is sharded across
        the submesh, so this method precomputes the modulation outputs (with
        an internal all_gather) for each layer and passes them via
        `precomputed_mod` to the block. The block then sees the already-gathered
        modulation tensors and runs its non-fast-path matmul-free modulation.
        Mod weights replicated (mod_sharded=False) keeps today's behavior:
        the block internally does its own ttnn.linear with the replicated
        weight, no all_gather needed.
        """
        for local_i, block in enumerate(self.expert_blocks):
            global_i = self.layer_lo + local_i
            past_kv = prefix_kv_cache[global_i] if prefix_kv_cache is not None else None
            block_mod = precomputed_block_mods[local_i] if precomputed_block_mods is not None else None
            # Sharded mod path: we precompute + all_gather inside the slice
            # and pass the 6 chunk tensors via `precomputed_mod`. Only
            # applies when the caller didn't already provide one.
            if block_mod is None and self.mod_sharded:
                block_mod = self._precompute_mod_sharded(block, adarms_cond)
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
    """Single-chip + L1 mirror of `_load_expert_block_weights_l1`.

    All weights and biases (norms, mod weight/bias) are bf8_b. Sharding does
    not apply to 1-chip submeshes — the paired path already has each layer
    on exactly one chip, and the mod weight at bf8_b is ~6 MB / layer.
    """
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
            # Keep LN 1D bias at bf16 — bf8_b on tensors that scale the residual
            # stream accumulates a measurable PCC drop at depth=18 (0.40 vs 0.99
            # at depth=2). LN bias is tiny (kBs) so the L1 cost is negligible.
            block_weights[new_key] = tensor_1d_to_2d_ttnn(value, micro_submesh, dtype=ttnn.bfloat16)
        else:
            # Keep LN 2D weights at bf16 (same rationale as the 1D case).
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
    # Keep adarms_mod.weight at bf16 — its output is the per-token scale/shift
    # for every layer's residual stream, so bf8_b quantization compounds across
    # 18 layers × 10 denoise steps into a ~0.4 PCC drift. ~6 MB / layer / chip
    # extra (= 18 MB / chip across 3 layers per paired chip) — still well
    # under the L1 cap with vision/prefill/denoise at 83 / 108 / ~100 MB.
    block_weights["adarms_mod.weight"] = _upload_single_chip_l1(fused_w.T.contiguous(), micro_submesh, ttnn.bfloat16)

    b_keys = [
        f"{prefix}input_layernorm.dense.bias",
        f"{prefix}post_attention_layernorm.dense.bias",
    ]
    biases = [full_weights[bk] for bk in b_keys if bk in full_weights]
    if biases:
        assert len(biases) == 2, "expected both adaRMS biases or neither"
        fused_b = torch.cat(biases, dim=0).contiguous()
        # Keep adarms_mod.bias at bf16 (same rationale as the weight).
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
        # Keep final_norm_mod_weight + bias at bf16 — same rationale as the
        # per-block mod weight; the final norm is applied once but its scale
        # multiplies the accumulated 18-layer residual.
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


# ---------------------------------------------------------------------------- #
# Parent-mesh expert slice (D2D) — 3 layers per chip × 6 chips, weights         #
# sharded on parent mesh, activation flows chip→chip via fabric P2P every       #
# 3 layers. Same recipe as `vlm_slice.Pi0_5OptionCVLMSliceParent`, but for the  #
# denoise submesh column (rows 2..7, col 3 — all P2P hops are same-column,     #
# single-hop fabric, no multihop needed).                                      #
# ---------------------------------------------------------------------------- #


def _load_expert_weights_stacked_sharded(
    full_weights: Dict[str, torch.Tensor],
    layer_range: Tuple[int, int],
    parent_mesh,
    parent_shape: Tuple[int, int],
    denoise_offset: Tuple[int, int],
    denoise_shape: Tuple[int, int],
    layers_per_chip: int,
    dtype=ttnn.bfloat8_b,
) -> Dict[str, List["ttnn.Tensor"]]:
    """Upload all expert layer weights as parent-mesh sharded tensors.

    The denoise submesh holds `layers_per_chip` layers per chip (3 by default).
    For each weight category and each per-chip position `j ∈ [0, layers_per_chip)`,
    we build one parent-mesh stacked tensor where chip d's slot holds the
    weights for layer `layer_lo + d*layers_per_chip + j`. Other slots are zero.

    At forward time, when running layer i with local position j on chip d:
      - we use the position-j tensor for that weight category
      - chip d's matmul produces the right output (its slot has the right weights)
      - other chips' matmuls produce garbage that we discard (only chip d's
        shard is live).

    Returns a dict keyed by weight name (e.g. "q_proj", "gate_proj", "wqkv",
    "input_layernorm", "post_attention_layernorm", "adarms_mod_weight",
    "adarms_mod_bias"), each mapping to a list of `layers_per_chip` parent-
    mesh tensors indexed by per-chip position.
    """
    lo, hi = layer_range
    n_layers = hi - lo
    expected = denoise_shape[0] * denoise_shape[1] * layers_per_chip
    if n_layers != expected:
        raise ValueError(
            f"layer_range span {n_layers} must equal denoise chip count "
            f"{denoise_shape[0] * denoise_shape[1]} × layers_per_chip={layers_per_chip} "
            f"= {expected}"
        )
    devices_total = parent_shape[0] * parent_shape[1]

    def _denoise_lin(chip_idx: int) -> int:
        sub_row = chip_idx // denoise_shape[1]
        sub_col = chip_idx % denoise_shape[1]
        return (denoise_offset[0] + sub_row) * parent_shape[1] + (denoise_offset[1] + sub_col)

    def _key(layer_idx: int, suffix: str) -> str:
        return f"model.layers.{layer_idx}.{suffix}"

    matmul_suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    # AdaRMS norm Dense weights (per-block modulation) and biases.
    mod_weight_keys = [
        "input_layernorm.dense.weight",
        "post_attention_layernorm.dense.weight",
    ]
    mod_bias_keys = [
        "input_layernorm.dense.bias",
        "post_attention_layernorm.dense.bias",
    ]

    out: Dict[str, List["ttnn.Tensor"]] = {}
    num_chips = denoise_shape[0] * denoise_shape[1]

    def _alloc_for_position(
        suffix: str,
        position: int,
        weight_dtype,
        is_matmul: bool,
    ) -> Optional["ttnn.Tensor"]:
        """Build the position-`j` parent-mesh sharded tensor for `suffix`.
        Chip d holds layer (lo + d*layers_per_chip + j)'s weight."""
        # Probe layer at position j of chip 0 for the reference shape.
        ref_layer = lo + position
        ref_key = _key(ref_layer, suffix)
        if ref_key not in full_weights:
            return None
        ref = full_weights[ref_key]
        ref_t = ref.T.contiguous() if is_matmul else ref
        target_shape = tuple(ref_t.shape)
        stacked = torch.zeros((devices_total,) + target_shape, dtype=ref_t.dtype)
        for d in range(num_chips):
            global_i = lo + d * layers_per_chip + position
            if global_i >= hi:
                continue
            wk = _key(global_i, suffix)
            if wk not in full_weights:
                continue
            w = full_weights[wk]
            if is_matmul:
                w = w.T.contiguous()
            lin = _denoise_lin(d)
            stacked[lin] = w
        return ttnn.from_torch(
            stacked,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=parent_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent_mesh, dim=0),
        )

    def _alloc_fused_wqkv_for_position(position: int) -> Optional["ttnn.Tensor"]:
        """Build a fused wqkv for position j (matches the paired slice's
        ttnn.concat([wq, wk, wv], dim=-1) pattern, but done on host so we
        upload one tensor instead of three + a concat op).

        Layer i's wqkv lives at the layer's owner chip's slot.
        """
        # Probe shape at lo + position.
        ref_layer = lo + position
        keys = (
            _key(ref_layer, "self_attn.q_proj.weight"),
            _key(ref_layer, "self_attn.k_proj.weight"),
            _key(ref_layer, "self_attn.v_proj.weight"),
        )
        if not all(k in full_weights for k in keys):
            return None
        wq_t = full_weights[keys[0]].T.contiguous()
        wk_t = full_weights[keys[1]].T.contiguous()
        wv_t = full_weights[keys[2]].T.contiguous()
        target_shape = (wq_t.shape[0], wq_t.shape[1] + wk_t.shape[1] + wv_t.shape[1])
        stacked = torch.zeros((devices_total,) + target_shape, dtype=wq_t.dtype)
        for d in range(num_chips):
            global_i = lo + d * layers_per_chip + position
            if global_i >= hi:
                continue
            ks = (
                _key(global_i, "self_attn.q_proj.weight"),
                _key(global_i, "self_attn.k_proj.weight"),
                _key(global_i, "self_attn.v_proj.weight"),
            )
            if not all(k in full_weights for k in ks):
                continue
            wq = full_weights[ks[0]].T.contiguous()
            wk = full_weights[ks[1]].T.contiguous()
            wv = full_weights[ks[2]].T.contiguous()
            fused = torch.cat([wq, wk, wv], dim=-1)
            lin = _denoise_lin(d)
            stacked[lin] = fused
        return ttnn.from_torch(
            stacked,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=parent_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent_mesh, dim=0),
        )

    def _alloc_fused_adarms_mod_for_position(position: int, is_weight: bool) -> Optional["ttnn.Tensor"]:
        """Build the fused adaRMS mod weight/bias for position j (matches the
        paired slice's torch.cat([input_ln.dense, post_attn_ln.dense]) pattern).
        Shape: [W, 6W] (weight) or [6W] (bias)."""
        keys = mod_weight_keys if is_weight else mod_bias_keys
        ref_layer = lo + position
        ref_keys = [_key(ref_layer, k) for k in keys]
        if not all(k in full_weights for k in ref_keys):
            return None
        ref_parts = [full_weights[k] for k in ref_keys]
        ref_cat = torch.cat(ref_parts, dim=0)  # cat on out-dim (the "6" axis)
        if is_weight:
            ref_cat = ref_cat.T.contiguous()  # [W, 6W]
        target_shape = tuple(ref_cat.shape)
        stacked = torch.zeros((devices_total,) + target_shape, dtype=ref_cat.dtype)
        for d in range(num_chips):
            global_i = lo + d * layers_per_chip + position
            if global_i >= hi:
                continue
            layer_keys = [_key(global_i, k) for k in keys]
            if not all(k in full_weights for k in layer_keys):
                continue
            parts = [full_weights[k] for k in layer_keys]
            cat = torch.cat(parts, dim=0)
            if is_weight:
                cat = cat.T.contiguous()
            lin = _denoise_lin(d)
            stacked[lin] = cat
        # adarms mod is kept at bf16 (same rationale as the paired slice: the
        # scale/translate multiplies the residual; bf8_b compounds drift).
        if is_weight:
            return ttnn.from_torch(
                stacked,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=parent_mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(parent_mesh, dim=0),
            )
        # Bias is 1D [6W]; tile-pad to [tile, 6W] for the kernel.
        # We use ROW_MAJOR + a [1, 6W] shape, then upload as tiled via reshape.
        # Simpler: keep as 1D and let the caller reshape/broadcast at use site.
        return ttnn.from_torch(
            stacked.unsqueeze(-2),  # → [devices, 1, 6W]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=parent_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(parent_mesh, dim=0),
        )

    # Build per-position tensors for each matmul suffix.
    for suffix in matmul_suffixes:
        key = suffix.replace(".weight", "").replace("self_attn.", "").replace("mlp.", "")
        per_pos: List["ttnn.Tensor"] = []
        for j in range(layers_per_chip):
            t = _alloc_for_position(suffix, j, dtype, is_matmul=True)
            if t is None:
                break
            per_pos.append(t)
        if per_pos:
            out[key] = per_pos

    # Fused wqkv per position (q_proj + k_proj + v_proj along output axis).
    wqkv_per_pos: List["ttnn.Tensor"] = []
    for j in range(layers_per_chip):
        t = _alloc_fused_wqkv_for_position(j)
        if t is None:
            break
        wqkv_per_pos.append(t)
    if wqkv_per_pos:
        out["wqkv"] = wqkv_per_pos

    # AdaRMS modulation Dense (fused [input_ln.dense + post_attn_ln.dense]).
    mod_w_per_pos: List["ttnn.Tensor"] = []
    mod_b_per_pos: List["ttnn.Tensor"] = []
    for j in range(layers_per_chip):
        tw = _alloc_fused_adarms_mod_for_position(j, is_weight=True)
        if tw is None:
            break
        mod_w_per_pos.append(tw)
        tb = _alloc_fused_adarms_mod_for_position(j, is_weight=False)
        if tb is not None:
            mod_b_per_pos.append(tb)
    if mod_w_per_pos:
        out["adarms_mod_weight"] = mod_w_per_pos
    if mod_b_per_pos:
        out["adarms_mod_bias"] = mod_b_per_pos

    return out


class Pi0_5OptionCExpertSliceParent:
    """6-chip parent-mesh expert slice — 3 layers per chip × 6 chips on the
    denoise submesh (single column at parent coords (2..7, 3)), weights
    sharded across the parent mesh, activation flows chip→chip via fabric
    P2P every `layers_per_chip` layers.

    STATUS: scaffolding / minimal-viable. The full block forward (adaRMS
    modulation + cross-attention over migrated VLM prefix KV + RoPE + MLP)
    is the follow-up. This class lays the architectural pattern:

    - Open the denoise submesh as part of the galaxy parent (no carving).
    - Upload all 18 layers' weights as parent-mesh sharded tensors via
      `_load_expert_weights_stacked_sharded` — chip d, position j, holds
      layer (d*layers_per_chip + j)'s weights.
    - Activation lives on the parent mesh as a sharded tensor; the "live"
      shard moves chip-to-chip via `send_shard_via_p2p` (same column →
      single-hop fabric, no multihop) every `layers_per_chip` layers.

    For pi0.5 Option C this replaces the host-bouncing layer-paired denoise
    expert chain — estimated ~30-60 ms saved across 5 inter-chip transitions
    plus the per-Euler-step wrap-back (last chip → chip 0).

    Args:
        config:         full PaliGemma config.
        weights:        full categorized weights dict (we slice action_expert).
        parent_mesh:    the galaxy parent mesh (8, 4). Required because
                        weights are parent-mesh sharded.
        denoise_offset: (row, col) origin of the denoise submesh in the parent.
        denoise_shape:  (rows, cols) of the denoise submesh.
        expert_layer_range: half-open (lo, hi). hi - lo must equal num denoise
                            chips × layers_per_chip.
        layers_per_chip: layers held per denoise chip (default 3 → 18 total).

    NOT YET IMPLEMENTED in this class:
        - Full AdaRMSGemmaBlockTTNN forward semantics. Currently only the
          per-chip matmul chain + P2P advance is wired (smoke validation).
        - Cross-attention over migrated prefix KV (which lives on the same
          chip as its consuming expert layer post-migrate_layer_paired_d2d).
        - Final adaRMS norm on the last denoise chip.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        parent_mesh,
        denoise_offset: Tuple[int, int],
        denoise_shape: Tuple[int, int],
        expert_layer_range: Tuple[int, int] = (0, 18),
        layers_per_chip: int = 3,
    ) -> None:
        if not (0 <= expert_layer_range[0] < expert_layer_range[1] <= config.expert_config.depth):
            raise ValueError(
                f"expert_layer_range {expert_layer_range} out of bounds for "
                f"expert_config.depth={config.expert_config.depth}"
            )
        n_denoise_chips = denoise_shape[0] * denoise_shape[1]
        expected_layers = n_denoise_chips * layers_per_chip
        if expert_layer_range[1] - expert_layer_range[0] != expected_layers:
            raise ValueError(
                f"expert_layer_range span {expert_layer_range[1] - expert_layer_range[0]} "
                f"must equal denoise chip count {n_denoise_chips} × layers_per_chip="
                f"{layers_per_chip} = {expected_layers}"
            )

        self.config = config
        self.parent_mesh = parent_mesh
        self.denoise_offset = denoise_offset
        self.denoise_shape = denoise_shape
        self.parent_shape = (parent_mesh.shape[0], parent_mesh.shape[1])
        self.layer_lo, self.layer_hi = expert_layer_range
        self.num_layers = self.layer_hi - self.layer_lo
        self.layers_per_chip = layers_per_chip
        self.num_chips = n_denoise_chips

        # Upload all weights as parent-mesh sharded tensors (per-position).
        if "action_expert" not in weights:
            raise KeyError("weights dict must contain 'action_expert'")
        self.weights_on_parent = _load_expert_weights_stacked_sharded(
            weights["action_expert"],
            expert_layer_range,
            parent_mesh,
            self.parent_shape,
            denoise_offset,
            denoise_shape,
            layers_per_chip,
        )

    def expert_chip_for_layer(self, local_layer_idx: int) -> int:
        return local_layer_idx // self.layers_per_chip

    def position_in_chip(self, local_layer_idx: int) -> int:
        return local_layer_idx % self.layers_per_chip

    def denoise_coord_for_layer(self, local_layer_idx: int) -> Tuple[int, int]:
        """Galaxy-parent coord of the denoise chip owning the given local layer."""
        chip_idx = self.expert_chip_for_layer(local_layer_idx)
        sub_row = chip_idx // self.denoise_shape[1]
        sub_col = chip_idx % self.denoise_shape[1]
        return (self.denoise_offset[0] + sub_row, self.denoise_offset[1] + sub_col)

    def forward_qo_chain(self, activation: "ttnn.Tensor") -> "ttnn.Tensor":
        """Simplified forward that runs the q_proj → o_proj matmul pair per layer.

        Each layer does:
            1. h = linear(h, q_proj[position_in_chip(i)])   → [..., num_heads*head_dim]
            2. h = linear(h, o_proj[position_in_chip(i)])   → [..., width]
        Then at each chip boundary (i % layers_per_chip == layers_per_chip-1
        and not the last layer), P2P advances the live shard to the next
        denoise chip (same column → single-hop fabric).

        Q→O is the outer shape of the attention sublayer (skipping SDPA).
        Unlike the VLM's square q_proj `[W=2048, W=2048]`, the expert's
        q_proj is rectangular `[W=1024, num_heads*head_dim=2048]` so Q output
        cannot feed the next layer's Q input directly. Chaining through O
        brings the activation back to `[..., W=1024]` which feeds the next
        layer's Q. Validates the matmul chain + P2P advance at scale.

        Returns the final activation at the last expert chip's parent coord.
        """
        from .transport import send_shard_via_p2p

        for r in ("q_proj", "o_proj"):
            if r not in self.weights_on_parent:
                raise RuntimeError(f"{r} weights not loaded; check _load_expert_weights_stacked_sharded")
        q_proj_per_pos = self.weights_on_parent["q_proj"]
        o_proj_per_pos = self.weights_on_parent["o_proj"]

        h = activation
        for local_i in range(self.num_layers):
            j = self.position_in_chip(local_i)
            q_out = ttnn.linear(
                h,
                q_proj_per_pos[j],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(h)
            h = ttnn.linear(
                q_out,
                o_proj_per_pos[j],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q_out)

            at_chip_boundary = (j == self.layers_per_chip - 1) and (local_i + 1 < self.num_layers)
            if at_chip_boundary:
                cur = self.denoise_coord_for_layer(local_i)
                nxt = self.denoise_coord_for_layer(local_i + 1)
                h_advanced = send_shard_via_p2p(h, cur, nxt)
                if h_advanced is not h:
                    ttnn.deallocate(h)
                h = h_advanced
        return h

    def _ensure_rope_tables(self) -> None:
        """Lazy-init cos/sin tables on the parent mesh (replicated to all chips).

        RoPE rotations are position-dependent but layer-agnostic. We build
        them once on the parent mesh; every chip applies the same rotation
        to its own Q,K shards independently.
        """
        if getattr(self, "_cos_meta", None) is not None:
            return
        from models.experimental.pi0_5.tt.ttnn_gemma import precompute_freqs_cis_meta_format

        head_dim = self.config.expert_config.head_dim
        max_seq = self.config.max_seq_len
        cos, sin = precompute_freqs_cis_meta_format(head_dim, max_seq, self.parent_mesh)
        self._cos_meta = cos
        self._sin_meta = sin

    def forward_attn_sublayer_chain(
        self,
        activation: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
    ) -> "ttnn.Tensor":
        """Full attention SUBLAYER chain across all expert layers — adaRMS
        modulation + modulated RMSNorm + fused wqkv + heads split + RoPE +
        SDPA (no past_kv, self-attention only) + O projection + gated residual.

        Per layer:
            1. mod = linear(adarms_cond, mod_weight, mod_bias)  → [B, 6W]
            2. (sa1, ta, ga, _, _, _) = split_modulation_6(mod)
            3. normed = modulated_rms_norm(h, sa1, ta)
            4. xqkv = linear(normed, wqkv)
            5. (q, k, v) = nlp_create_qkv_heads(xqkv, num_heads, num_kv_heads)
            6. q_rope = rotary(q, cos, sin) ; k_rope = rotary(k, cos, sin)
            7. attn_out = SDPA(q_rope, k_rope, v, attn_mask=mask)
            8. attn_concat = nlp_concat_heads(attn_out)
            9. o_out = linear(attn_concat, o_proj)
           10. gated = o_out * ga
           11. h = h + gated
           12. at chip boundary: P2P advance (same column → single hop)

        No KV cache support yet — `past_kv = None` (self-attention over the
        64-token suffix). KV cache integration + cross-attention with the
        migrated VLM prefix lands in the next sub-commit.

        Args:
            activation: parent-mesh tensor [1, 1, M_padded, W=1024] with live
                data at the first denoise chip's parent coord.
            adarms_cond: parent-mesh tensor [B, 1, W] (replicated across all
                chips). The conditioning vector from the suffix slice's
                time/state embedding.
            attention_mask: optional parent-mesh mask for SDPA. None →
                is_causal=True self-attention.

        Returns: final activation at the last denoise chip's parent coord.
        """
        from .transport import send_shard_via_p2p
        from models.experimental.pi0_5.tt.ttnn_gemma import (
            _modulated_rms_norm,
            _split_modulation_6,
        )

        for r in ("wqkv", "o_proj", "adarms_mod_weight"):
            if r not in self.weights_on_parent:
                raise RuntimeError(f"{r} weights not loaded; check _load_expert_weights_stacked_sharded")
        self._ensure_rope_tables()

        wqkv_per_pos = self.weights_on_parent["wqkv"]
        o_proj_per_pos = self.weights_on_parent["o_proj"]
        mod_w_per_pos = self.weights_on_parent["adarms_mod_weight"]
        mod_b_per_pos = self.weights_on_parent.get("adarms_mod_bias", [None] * self.layers_per_chip)
        eps = self.config.expert_config.rms_norm_eps
        num_heads = self.config.expert_config.num_heads
        num_kv_heads = self.config.expert_config.num_kv_heads
        head_dim = self.config.expert_config.head_dim

        h = activation
        for local_i in range(self.num_layers):
            j = self.position_in_chip(local_i)
            seq_len = h.shape[-2]

            # 1. Modulation Dense. adarms_cond [B, 1, W] @ mod_w [1, W, 6W] (+ bias)
            #    → mod [B, 1, 6W]. Each chip computes its own mod with its
            #    position-j weight slot.
            mod_b = mod_b_per_pos[j] if j < len(mod_b_per_pos) else None
            mod = ttnn.linear(
                adarms_cond,
                mod_w_per_pos[j],
                bias=mod_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            sa1, ta, ga, _sf1, _tf, _gf = _split_modulation_6(mod)
            ttnn.deallocate(mod)
            ttnn.deallocate(_sf1)
            ttnn.deallocate(_tf)
            ttnn.deallocate(_gf)

            # 2. Modulated RMSNorm: normed = (rms_norm(h) * (1+sa1)) + ta.
            normed = _modulated_rms_norm(h, sa1, ta, eps, pre_added=False)
            ttnn.deallocate(sa1)
            ttnn.deallocate(ta)

            # 3. Fused wqkv matmul. wqkv was uploaded as the concat of
            #    [wq, wk, wv] along the output axis, so xqkv has shape
            #    [B, 1, M, num_heads*head_dim + 2*num_kv_heads*head_dim].
            xqkv = ttnn.linear(
                normed,
                wqkv_per_pos[j],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(normed)

            # 4. Split into per-head Q, K, V. Same op the paired path uses.
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                xqkv,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            # NOTE: do NOT deallocate xqkv — nlp_create_qkv_heads may view-alias.

            # 5. RoPE on Q, K. Slice the precomputed cos/sin tables to current seq_len.
            cos_slice = ttnn.slice(self._cos_meta, [0, 0, 0, 0], [1, 1, seq_len, head_dim])
            sin_slice = ttnn.slice(self._sin_meta, [0, 0, 0, 0], [1, 1, seq_len, head_dim])
            q_rope = ttnn.experimental.rotary_embedding(q, cos_slice, sin_slice)
            k_rope = ttnn.experimental.rotary_embedding(k, cos_slice, sin_slice)
            ttnn.deallocate(cos_slice)
            ttnn.deallocate(sin_slice)

            # 6. SDPA. Pass is_causal=False explicitly when an attn_mask is
            #    given — the kernel defaults is_causal=True and TT_FATALs if
            #    both are supplied.
            if attention_mask is not None:
                attn_out = ttnn.transformer.scaled_dot_product_attention(
                    q_rope,
                    k_rope,
                    v,
                    attn_mask=attention_mask,
                    is_causal=False,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                attn_out = ttnn.transformer.scaled_dot_product_attention(
                    q_rope, k_rope, v, is_causal=True, memory_config=ttnn.L1_MEMORY_CONFIG
                )
            ttnn.deallocate(q_rope)
            ttnn.deallocate(k_rope)
            ttnn.deallocate(v)

            # 7. Concat heads back to flat [B, 1, M, num_heads*head_dim].
            attn_flat = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)

            # 8. O projection brings the activation back to width = 1024.
            o_out = ttnn.linear(
                attn_flat,
                o_proj_per_pos[j],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_flat)

            # 9. Gated residual: h = h + (o_out * ga).
            gated = ttnn.multiply(o_out, ga, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(o_out)
            ttnn.deallocate(ga)
            h_new = ttnn.add(h, gated, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gated)
            ttnn.deallocate(h)
            h = h_new

            # 10. Inter-chip P2P advance (same column → single hop).
            at_chip_boundary = (j == self.layers_per_chip - 1) and (local_i + 1 < self.num_layers)
            if at_chip_boundary:
                cur = self.denoise_coord_for_layer(local_i)
                nxt = self.denoise_coord_for_layer(local_i + 1)
                h_advanced = send_shard_via_p2p(h, cur, nxt)
                if h_advanced is not h:
                    ttnn.deallocate(h)
                h = h_advanced
        return h

    def forward_real_block_chain(
        self,
        activation: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        prefix_kv_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None,
    ) -> "ttnn.Tensor":
        """Complete Gemma expert block forward on parent mesh — REAL attention +
        MLP sublayers + gated residuals + adaRMS modulation, chained across
        18 layers with 5 same-column P2P advances.

        Per layer:
          Attention sublayer (same as forward_attn_sublayer_chain):
            1. mod = linear(adarms_cond, mod_w[j], mod_b[j])
            2. (sa1, ta, ga, sf1, tf, gf) = split_modulation_6(mod)
            3. normed = modulated_rms_norm(h, sa1, ta)
            4. xqkv = linear(normed, wqkv[j])
            5. (q, k, v) = nlp_create_qkv_heads(xqkv, num_heads, num_kv_heads)
            6. RoPE on q, k
            7. SDPA (optionally cross-attention with prefix_kv_cache[layer])
            8. attn_flat = nlp_concat_heads
            9. o_out = linear(attn_flat, o_proj[j])
           10. h = h + (o_out * ga)
          MLP sublayer:
           11. normed_mlp = modulated_rms_norm(h, sf1, tf)
           12. g_mlp = linear(normed_mlp, gate_proj[j])
           13. u_mlp = linear(normed_mlp, up_proj[j])
           14. mid = g_mlp * silu(u_mlp)
           15. d = linear(mid, down_proj[j])
           16. h = h + (d * gf)
          Transport:
           17. at chip boundary: P2P advance (single-hop, same column)

        When `prefix_kv_cache` is provided, the entry at layer i (global
        index) must be a parent-mesh (K, V) tuple whose live shard resides
        at the same denoise chip that owns expert layer i — i.e. the
        post-`migrate_layer_paired_d2d` layout. This enables cross-attention
        with the migrated VLM prefix.

        Args:
            activation: [1, 1, M_padded, W] parent-mesh tensor, live at the
                first denoise chip's coord.
            adarms_cond: [B, 1, W] parent-mesh tensor (replicated).
            attention_mask: optional joint mask (parent-mesh tensor).
            prefix_kv_cache: optional depth-indexed list of (K, V); each
                entry None or a parent-mesh tensor with its live shard at
                the denoise chip owning the matching expert layer.

        Returns: final activation at the last denoise chip's parent coord.
        """
        from .transport import send_shard_via_p2p
        from models.experimental.pi0_5.tt.ttnn_gemma import (
            _modulated_rms_norm,
            _split_modulation_6,
        )

        required = (
            "wqkv",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "adarms_mod_weight",
        )
        for r in required:
            if r not in self.weights_on_parent:
                raise RuntimeError(f"{r} weights not loaded; check _load_expert_weights_stacked_sharded")
        self._ensure_rope_tables()

        wqkv_per_pos = self.weights_on_parent["wqkv"]
        o_proj_per_pos = self.weights_on_parent["o_proj"]
        gate_per_pos = self.weights_on_parent["gate_proj"]
        up_per_pos = self.weights_on_parent["up_proj"]
        down_per_pos = self.weights_on_parent["down_proj"]
        mod_w_per_pos = self.weights_on_parent["adarms_mod_weight"]
        mod_b_per_pos = self.weights_on_parent.get("adarms_mod_bias", [None] * self.layers_per_chip)
        eps = self.config.expert_config.rms_norm_eps
        num_heads = self.config.expert_config.num_heads
        num_kv_heads = self.config.expert_config.num_kv_heads
        head_dim = self.config.expert_config.head_dim

        h = activation
        for local_i in range(self.num_layers):
            j = self.position_in_chip(local_i)
            global_i = self.layer_lo + local_i
            seq_len = h.shape[-2]

            # ===== Modulation =====
            mod_b = mod_b_per_pos[j] if j < len(mod_b_per_pos) else None
            mod = ttnn.linear(adarms_cond, mod_w_per_pos[j], bias=mod_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            sa1, ta, ga, sf1, tf, gf = _split_modulation_6(mod)
            ttnn.deallocate(mod)

            # ===== Attention sublayer =====
            normed = _modulated_rms_norm(h, sa1, ta, eps, pre_added=False)
            ttnn.deallocate(sa1)
            ttnn.deallocate(ta)

            xqkv = ttnn.linear(normed, wqkv_per_pos[j], dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(normed)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                xqkv,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            cos_slice = ttnn.slice(self._cos_meta, [0, 0, 0, 0], [1, 1, seq_len, head_dim])
            sin_slice = ttnn.slice(self._sin_meta, [0, 0, 0, 0], [1, 1, seq_len, head_dim])
            q_rope = ttnn.experimental.rotary_embedding(q, cos_slice, sin_slice)
            k_rope = ttnn.experimental.rotary_embedding(k, cos_slice, sin_slice)
            ttnn.deallocate(cos_slice)
            ttnn.deallocate(sin_slice)

            # Cross-attention: concat the migrated VLM prefix KV when provided.
            past_kv = prefix_kv_cache[global_i] if prefix_kv_cache is not None else None
            if past_kv is not None:
                past_k, past_v = past_kv
                k_rope = ttnn.concat([past_k, k_rope], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
                v = ttnn.concat([past_v, v], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

            if attention_mask is not None:
                attn_out = ttnn.transformer.scaled_dot_product_attention(
                    q_rope,
                    k_rope,
                    v,
                    attn_mask=attention_mask,
                    is_causal=False,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                attn_out = ttnn.transformer.scaled_dot_product_attention(
                    q_rope, k_rope, v, is_causal=True, memory_config=ttnn.L1_MEMORY_CONFIG
                )
            ttnn.deallocate(q_rope)
            ttnn.deallocate(k_rope)
            ttnn.deallocate(v)

            attn_flat = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)

            o_out = ttnn.linear(attn_flat, o_proj_per_pos[j], dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_flat)

            gated_attn = ttnn.multiply(o_out, ga, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(o_out)
            ttnn.deallocate(ga)
            h_post_attn = ttnn.add(h, gated_attn, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gated_attn)
            ttnn.deallocate(h)

            # ===== MLP sublayer =====
            normed_mlp = _modulated_rms_norm(h_post_attn, sf1, tf, eps, pre_added=False)
            ttnn.deallocate(sf1)
            ttnn.deallocate(tf)

            g_mlp = ttnn.linear(normed_mlp, gate_per_pos[j], dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            u_mlp = ttnn.linear(normed_mlp, up_per_pos[j], dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(normed_mlp)
            u_act = ttnn.silu(u_mlp, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(u_mlp)
            mid = ttnn.multiply(g_mlp, u_act, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(g_mlp)
            ttnn.deallocate(u_act)
            d = ttnn.linear(mid, down_per_pos[j], dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(mid)

            gated_mlp = ttnn.multiply(d, gf, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(d)
            ttnn.deallocate(gf)
            h_new = ttnn.add(h_post_attn, gated_mlp, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gated_mlp)
            ttnn.deallocate(h_post_attn)
            h = h_new

            # ===== P2P advance =====
            at_chip_boundary = (j == self.layers_per_chip - 1) and (local_i + 1 < self.num_layers)
            if at_chip_boundary:
                cur = self.denoise_coord_for_layer(local_i)
                nxt = self.denoise_coord_for_layer(local_i + 1)
                h_advanced = send_shard_via_p2p(h, cur, nxt)
                if h_advanced is not h:
                    ttnn.deallocate(h)
                h = h_advanced
        return h
