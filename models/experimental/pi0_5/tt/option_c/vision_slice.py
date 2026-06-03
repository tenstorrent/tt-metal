# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 (vision) slice for Option C.

Target placement (PI0_5_GALAXY_DEPLOYMENT_PLAN.md §3.1, file map in
option_c/README.md):

    vision submesh (4 chips, shape (2, 2)) =
        3 SigLIP chips × 9 SigLIP layers each
      + 1 vision-embed chip (patch_conv + pos_embed + final LN + mm_projector)

For the scaffolding pass (this file), SigLIP + mm_projector run on the
HOST (torch) — same pragmatic choice Option B made (see
`option_b/vision_slice.py`) — and we upload the projected vision features
+ language-token embeddings onto the 4-chip vision submesh. The on-device
3-chip split is a follow-up; the slice's external contract (`embed_images`,
`embed_language_tokens`, `build_prefix_hidden`) is stable and the
orchestrator (`stage_vision.py`) talks to it through that contract only.

Memory footprint (per chip, scaffolding mode, replicated):
  - Vision feature cache from host:        [B, 256, 1152] bf8 ≈ 0.6 MB
  - Projected vision tokens:                [B, 256, 2048] bf8 ≈ 1 MB
  - (Optional) host-lookup lang embedding:  [B, S_lang, 2048] bf8 ≈ 0.4 MB
  Total ≈ 2 MB / chip — trivially fits.

When the device-side SigLIP lands, per-chip weight footprint per
PI0_5_GALAXY_DEPLOYMENT_PLAN.md §3.1: 9 layers × 15.6 MB (bf8 attn) ≈
140 MB / chip — sits inside the 180 MB L1 cap with ~40 MB headroom.

Language embed table (527 MB) lives on HOST. We do not put it on a chip;
the recommendation in §3.1 option (a) is host-side text embedding lookup
followed by activation upload.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.reference.torch_siglip_hf import HFSigLIPVisionTower
from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector as _TorchMMProjector
from models.experimental.pi0_5.tt.ttnn_siglip import (
    MultiModalProjectorTTNN,
    SigLIPVisionTowerTTNN,
)

from .transport import send_activation_via_host
from .vlm_slice import _upload_l1_replicated


# ----------------------------------------------------------------------------#
# L1 weight migration helpers                                                 #
# ----------------------------------------------------------------------------#


def _to_l1(t: Optional["ttnn.Tensor"]) -> Optional["ttnn.Tensor"]:
    """Move a tensor to L1 and deallocate the DRAM source.

    `ttnn.to_memory_config(t, L1)` allocates a NEW L1 tensor and copies
    from `t` — leaving `t` (in DRAM) alive until Python GC drops the
    reference. With ~135 MB of weights to migrate per vision chip, that
    transient peak (DRAM + L1 simultaneously) trips the L1 allocator on
    the first big buffer. Explicit deallocate keeps the per-step peak
    bounded by one buffer.
    """
    if t is None:
        return None
    new_t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t)
    return new_t


def _migrate_tower_weights_to_l1(tower: SigLIPVisionTowerTTNN) -> None:
    """Move every weight tensor in a SigLIPVisionTowerTTNN to L1, in place.

    The upstream SigLIPVisionTowerTTNN uploads weights with ttnn's default
    DRAM memory config (see ttnn_siglip.py). Threading a memory_config
    parameter through 6 nested constructors would be a wide cross-cutting
    refactor that affects single-device callers too; a post-construction
    walk is contained, reversible, and easy to audit.

    Attribute names are verified against ttnn_siglip.py — break if a name
    changes there; the change should be caught by the next probe run.
    """
    # PatchEmbedding (only on the first chip's tower).
    pe = getattr(tower, "patch_embed", None)
    if pe is not None:
        if hasattr(pe, "_linear_weight") and pe._linear_weight is not None:
            pe._linear_weight = _to_l1(pe._linear_weight)
        if hasattr(pe, "_linear_bias") and pe._linear_bias is not None:
            pe._linear_bias = _to_l1(pe._linear_bias)

    # Positional embedding (only on the chip that holds it).
    if getattr(tower, "pos_emb_weights", None) is not None:
        tower.pos_emb_weights = _to_l1(tower.pos_emb_weights)

    # Post-LN (only on the last slice's tower).
    if getattr(tower, "post_ln_weight", None) is not None:
        tower.post_ln_weight = _to_l1(tower.post_ln_weight)
    if getattr(tower, "post_ln_bias", None) is not None:
        tower.post_ln_bias = _to_l1(tower.post_ln_bias)

    # Every transformer block on this chip.
    for block in tower.blocks:
        if getattr(block, "ln1_weight", None) is not None:
            block.ln1_weight = _to_l1(block.ln1_weight)
        if getattr(block, "ln1_bias", None) is not None:
            block.ln1_bias = _to_l1(block.ln1_bias)
        if getattr(block, "ln2_weight", None) is not None:
            block.ln2_weight = _to_l1(block.ln2_weight)
        if getattr(block, "ln2_bias", None) is not None:
            block.ln2_bias = _to_l1(block.ln2_bias)
        # Attention QKV (concatenated) + output projection.
        attn = block.attention
        attn.wqkv = _to_l1(attn.wqkv)
        if attn.bqkv is not None:
            attn.bqkv = _to_l1(attn.bqkv)
        attn.wo = _to_l1(attn.wo)
        # MLP fc1 / fc2.
        mlp = block.mlp
        mlp.fc1_weight = _to_l1(mlp.fc1_weight)
        if getattr(mlp, "fc1_bias", None) is not None:
            mlp.fc1_bias = _to_l1(mlp.fc1_bias)
        mlp.fc2_weight = _to_l1(mlp.fc2_weight)
        if getattr(mlp, "fc2_bias", None) is not None:
            mlp.fc2_bias = _to_l1(mlp.fc2_bias)


def _migrate_projector_weights_to_l1(projector: MultiModalProjectorTTNN) -> None:
    """Move MultiModalProjectorTTNN's `weight` and `bias` to L1 in place."""
    projector.weight = _to_l1(projector.weight)
    if projector.bias is not None:
        projector.bias = _to_l1(projector.bias)


class Pi0_5OptionCVisionSlice:
    """Stage 0 — produces prefix hidden_states for stage 1 from images + lang
    token IDs.

    Args:
        config:        full PaliGemma config (uses .siglip_config + .vlm_config).
        weights:       full categorized weights dict; needs vlm_vision +
                       vlm_projector + vlm_language.
        submesh:       the 4-chip vision MeshDevice.
        embed_on_host: if True (default for the scaffolding pass), text
                       embeddings are looked up on HOST via
                       weights['vlm_language']['model.embed_tokens.weight'].
                       This keeps the 527 MB embed table off the vision
                       submesh until vocab sharding lands. Set False once
                       you've sharded the table across vision chips.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        submesh,
        embed_on_host: bool = True,
    ) -> None:
        self.config = config
        self.submesh = submesh
        self.embed_on_host = embed_on_host

        # SigLIP-27 + multimodal projector run on host (torch) for the
        # scaffolding pass. The 3-chip device split (9 SigLIP layers per
        # chip, last chip holds mm_projector) is the follow-up — when it
        # lands, swap these two refs for an on-device equivalent and the
        # rest of the file (orchestrator, pipeline) is unchanged.
        self._host_vision_tower = HFSigLIPVisionTower(config.siglip_config, weights["vlm_vision"])
        self._host_mm_projector = _TorchMMProjector(weights["vlm_projector"])

        lang = weights["vlm_language"]
        embed_torch = lang.get("model.embed_tokens.weight") or lang.get("lm_head.weight")
        if embed_torch is None:
            raise KeyError(
                "vlm_language must contain 'model.embed_tokens.weight' or 'lm_head.weight' "
                "for Pi0_5OptionCVisionSlice to perform language embedding lookup"
            )

        if embed_on_host:
            self._host_embed_table: Optional[torch.Tensor] = embed_torch
            self._device_embed_table: Optional["ttnn.Tensor"] = None
        else:
            # 527 MB at bf16; over the 180 MB L1 cap without vocab sharding.
            # Forced to DRAM to keep the construction succeeding while we
            # validate the rest of the pipeline.
            self._host_embed_table = None
            self._device_embed_table = _upload_l1_replicated(
                embed_torch.contiguous(),
                submesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    # ------------------------------------------------------------------ #
    # Image path                                                         #
    # ------------------------------------------------------------------ #

    def embed_images(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """`pixel_values`: torch float tensor [B, 3, H, W].

        Returns a replicated tensor [B, num_patches, vlm_W] on the vision
        submesh, dtype bf16, TILE_LAYOUT, L1-resident.
        """
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)
        with torch.no_grad():
            features = self._host_vision_tower.forward(pixel_values)  # [B, 256, 1152]
            projected = self._host_mm_projector.forward(features)  # [B, 256, vlm_W]
        return _upload_l1_replicated(
            projected.to(torch.float32).contiguous(),
            self.submesh,
            dtype=ttnn.bfloat16,
        )

    # ------------------------------------------------------------------ #
    # Language path                                                      #
    # ------------------------------------------------------------------ #

    def embed_language_tokens(self, token_ids: torch.Tensor) -> "ttnn.Tensor":
        """`token_ids`: torch int tensor [B, S_lang].

        Returns a replicated tensor [B, S_lang, vlm_W] on the vision submesh.
        """
        if self.embed_on_host:
            embedded = torch.nn.functional.embedding(
                token_ids.to(torch.long), self._host_embed_table
            )  # [B, S_lang, vlm_W]
            return _upload_l1_replicated(
                embedded.to(torch.float32).contiguous(),
                self.submesh,
                dtype=ttnn.bfloat16,
            )
        # On-device embedding lookup path.
        if not isinstance(token_ids, ttnn.Tensor):
            token_ids = ttnn.from_torch(
                token_ids.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.submesh,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.submesh),
            )
        return ttnn.embedding(token_ids, self._device_embed_table)

    # ------------------------------------------------------------------ #
    # Combined prefix builder                                            #
    # ------------------------------------------------------------------ #

    def build_prefix_hidden(
        self,
        pixel_values: torch.Tensor,
        language_token_ids: torch.Tensor,
    ) -> "ttnn.Tensor":
        """Concat image and language embeddings along the seq dim.

        Returns: [B, num_patches + S_lang, vlm_W] on the vision submesh.

        Note: the openpi prefix path also appends a robot-state slot. Like
        Option B's vision slice, we leave that to the orchestrator so the
        bring-up dry run can be exercised with image+text only.
        """
        image_hidden = self.embed_images(pixel_values)
        lang_hidden = self.embed_language_tokens(language_token_ids)
        return ttnn.concat([image_hidden, lang_hidden], dim=1)


# ---------------------------------------------------------------------------- #
# On-device 3-chip SigLIP split + 1 projector chip                              #
# ---------------------------------------------------------------------------- #


class Pi0_5OptionCVisionSliceSplit:
    """On-device SigLIP-27 split across 3 chips + mm_projector on chip 4.

    Target placement (deployment plan §3.1):
        chip 0: patch_embed + pos_embed + SigLIP layers 0–8  (no post_ln)
        chip 1: SigLIP layers 9–17                           (no post_ln)
        chip 2: SigLIP layers 18–26 + post_ln
        chip 3: mm_projector (vision features → vlm_W)

    Activation host-bounces between consecutive chips. Language token
    embedding stays on HOST (the 527 MB embed_tokens table doesn't fit on
    a single chip; vocab sharding is a later follow-up). The host-resolved
    language embedding is uploaded to chip 3 to be concatenated with the
    projected vision features there.

    Construction is dry-run-safe even on shrunk configs because the SigLIP
    config / weights are independent of `vlm_depth` / `expert_depth`.

    Args:
        config:           full PaliGemma config.
        weights:          full categorized weights dict.
        micro_submeshes:  list of 4 single-chip MeshDevices (carved from the
                          4-chip vision submesh). micro_submeshes[0..2] are
                          the SigLIP chips, micro_submeshes[3] is the
                          projector chip.
        layers_per_chip:  number of SigLIP layers on each of the 3 SigLIP
                          chips. Defaults to 9 (per §3.1).
        siglip_depth:     total SigLIP depth (default 27, = 3 × 9).
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        micro_submeshes: List,
        layers_per_chip: int = 9,
        siglip_depth: int = 27,
        weights_in_l1: bool = False,
    ) -> None:
        if len(micro_submeshes) != 4:
            raise ValueError(
                f"Pi0_5OptionCVisionSliceSplit needs 4 micro-submeshes "
                f"(3 SigLIP + 1 projector); got {len(micro_submeshes)}"
            )
        for i, sm in enumerate(micro_submeshes):
            if sm.get_num_devices() != 1:
                raise ValueError(f"micro_submeshes[{i}] must be a 1-chip submesh " f"({sm.get_num_devices()} devices)")
        num_siglip_chips = 3
        if layers_per_chip * num_siglip_chips != siglip_depth:
            raise ValueError(
                f"layers_per_chip ({layers_per_chip}) × num_siglip_chips "
                f"({num_siglip_chips}) must equal siglip_depth ({siglip_depth})"
            )

        self.config = config
        self.micro_submeshes = micro_submeshes
        self.layers_per_chip = layers_per_chip
        self.siglip_depth = siglip_depth

        vlm_vision = weights["vlm_vision"]
        vlm_projector = weights["vlm_projector"]

        # Three SigLIP slices, one per chip.
        self.siglip_chunks: List[SigLIPVisionTowerTTNN] = []
        for chunk_idx in range(num_siglip_chips):
            lo = chunk_idx * layers_per_chip
            hi = lo + layers_per_chip
            is_first = chunk_idx == 0
            is_last = chunk_idx == num_siglip_chips - 1
            tower = SigLIPVisionTowerTTNN(
                config=config.siglip_config,
                weights=vlm_vision,
                device=micro_submeshes[chunk_idx],
                layer_range=(lo, hi),
                holds_patch_embed=is_first,
                holds_pos_embed=is_first,
                holds_post_ln=is_last,
            )
            if weights_in_l1:
                _migrate_tower_weights_to_l1(tower)
            self.siglip_chunks.append(tower)

        # mm_projector lives on chip 3.
        self.mm_projector = MultiModalProjectorTTNN(vlm_projector, micro_submeshes[3])
        if weights_in_l1:
            _migrate_projector_weights_to_l1(self.mm_projector)

        # Host-resident language embed table — same fallback as the host
        # vision slice. Vocab sharding is the next-step follow-up.
        lang = weights["vlm_language"]
        embed_torch = lang.get("model.embed_tokens.weight") or lang.get("lm_head.weight")
        if embed_torch is None:
            raise KeyError(
                "vlm_language must contain 'model.embed_tokens.weight' or "
                "'lm_head.weight' for Pi0_5OptionCVisionSliceSplit"
            )
        self._host_embed_table = embed_torch

    # ------------------------------------------------------------------ #
    # Image path                                                         #
    # ------------------------------------------------------------------ #

    def embed_images(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """Run SigLIP-27 across 3 chips, then mm_projector on chip 3.

        Returns a tensor [B, num_patches, vlm_W] on micro_submeshes[3].
        """
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)

        # Upload pixel_values to chip 0. SigLIPVisionTowerTTNN.forward expects
        # a ttnn.Tensor on the device that owns patch_embed (chip 0 here) —
        # the first op it runs is `ttnn.permute(x, (0, 2, 3, 1))` (NCHW → NHWC)
        # which can't accept a torch tensor. Match the single-device convention
        # from test_perf_ttnn_full_e2e_trace.py:_build_inputs (bf16, TILE,
        # DRAM_MEMORY_CONFIG).
        chip0 = self.micro_submeshes[0]
        pixel_values_ttnn = ttnn.from_torch(
            pixel_values.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=chip0,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(chip0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Chip 0: patch_embed + pos_embed + SigLIP layers 0–8.
        h = self.siglip_chunks[0].forward(pixel_values_ttnn)
        ttnn.deallocate(pixel_values_ttnn)

        # Chip 1: SigLIP layers 9–17 (no patch / pos_emb add — already done).
        h_next = send_activation_via_host(h, self.micro_submeshes[1])
        ttnn.deallocate(h)
        h = self.siglip_chunks[1].forward_from_hidden(h_next)

        # Chip 2: SigLIP layers 18–26 + final post_ln.
        h_next = send_activation_via_host(h, self.micro_submeshes[2])
        ttnn.deallocate(h)
        h = self.siglip_chunks[2].forward_from_hidden(h_next)

        # Chip 3: mm_projector.
        h_proj_in = send_activation_via_host(h, self.micro_submeshes[3])
        ttnn.deallocate(h)
        return self.mm_projector.forward(h_proj_in)

    # ------------------------------------------------------------------ #
    # Language path                                                      #
    # ------------------------------------------------------------------ #

    def embed_language_tokens(self, token_ids: torch.Tensor) -> "ttnn.Tensor":
        """Host-resolved language embedding uploaded to the projector chip."""
        embedded = torch.nn.functional.embedding(token_ids.to(torch.long), self._host_embed_table)
        proj_chip = self.micro_submeshes[3]
        return ttnn.from_torch(
            embedded.to(torch.float32).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=proj_chip,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(proj_chip),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------ #
    # Combined prefix builder                                            #
    # ------------------------------------------------------------------ #

    def build_prefix_hidden(
        self,
        pixel_values: torch.Tensor,
        language_token_ids: torch.Tensor,
    ) -> "ttnn.Tensor":
        """[B, num_patches + S_lang, vlm_W] on micro_submeshes[3]."""
        image_hidden = self.embed_images(pixel_values)
        lang_hidden = self.embed_language_tokens(language_token_ids)
        return ttnn.concat([image_hidden, lang_hidden], dim=1)

    @property
    def last_chip_submesh(self):
        """Where build_prefix_hidden's output lives — the projector chip."""
        return self.micro_submeshes[3]
