# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 (vision) slice for Option C.

Target placement (post full-L1 redesign, PI0_5_GALAXY_DEPLOYMENT_PLAN.md
§3.1, file map in option_c/README.md):

    vision submesh (8 chips, shape (2, 4)) =
        SigLIP-27 split 4+4+4+4+3+3+3+2 across the 8 chips
      + post_ln + mm_projector co-located on the last chip

Two slice variants exist:
  * `Pi0_5OptionCVisionSlice` runs SigLIP + mm_projector on HOST (torch)
    and uploads projected features onto the vision submesh — kept for the
    scaffolding / dry-run regression path.
  * `Pi0_5OptionCVisionSliceSplit` runs SigLIP on-device across the 8
    micro-submeshes; the on-device path is the one used when
    `device_siglip=True` and is the target for the full-L1 deployment.

Memory footprint (per chip, scaffolding mode, replicated):
  - Vision feature cache from host:        [B, 256, 1152] bf8 ≈ 0.6 MB
  - Projected vision tokens:                [B, 256, 2048] bf8 ≈ 1 MB
  - (Optional) host-lookup lang embedding:  [B, S_lang, 2048] bf8 ≈ 0.4 MB
  Total ≈ 2 MB / chip — trivially fits.

Device-SigLIP, 8-chip layout (default split 4+4+4+4+3+3+3+2 = 27 layers):
  Busiest chip carries 4 SigLIP layers (~4×18 MB bf8_b ≈ 72 MB) plus
  patch_embed + pos_embed on chip 0 (~3 MB) → ~75 MB / chip peak.
  Well inside the 175.4 MB L1 cap; full-L1 residence (LN, post_ln,
  patch_embed, pos_embed, attn, MLP) leaves ~100 MB / chip headroom
  for activations, BS-shard scratch, and matmul kernel static CB
  regions.

Language embed table (527 MB) lives on HOST. We do not put it on a chip;
the recommendation in §3.1 option (a) is host-side text embedding lookup
followed by activation upload.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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

    Idempotent: when the source is already L1-resident, `to_memory_config`
    is a no-op that returns the SAME buffer reference; `deallocate(t)`
    would then free the underlying buffer that the returned tensor also
    references — dangling reference, next op throws "Tensor is not
    allocated". Return `t` unchanged in that case.
    """
    if t is None:
        return None
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        return t
    new_t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t)
    return new_t


def _migrate_tower_weights_to_l1(
    tower: SigLIPVisionTowerTTNN,
    migrate_patch_embed: bool = True,
    migrate_pos_embed: bool = True,
    migrate_post_ln: bool = True,
    migrate_layer_norms: bool = True,
    migrate_attention: bool = True,
    migrate_mlp: bool = True,
) -> None:
    """Move every weight tensor in a SigLIPVisionTowerTTNN to L1, in place.

    The upstream SigLIPVisionTowerTTNN uploads weights with ttnn's default
    DRAM memory config (see ttnn_siglip.py). Threading a memory_config
    parameter through 6 nested constructors would be a wide cross-cutting
    refactor that affects single-device callers too; a post-construction
    walk is contained, reversible, and easy to audit.

    Full-L1 default (8-chip (2,4) vision submesh):
        With the wider vision submesh and the 4/4/4/4/3/3/3/2 SigLIP split
        (at most 4 SigLIP layers per chip + the small projector on the
        last chip), per-chip weight load drops well below the L1 cap
        even when ALL weight categories migrate. Defaults therefore turn
        ON every category — patch_embed / pos_embed / post_ln / LNs /
        attention / MLP — so vision is fully L1-resident. The bf16 bias
        and LN tensors are also converted to bf8_b on upload in
        ttnn_siglip.py to keep per-chip footprint inside budget.

    Attribute names are verified against ttnn_siglip.py — break if a name
    changes there; the change should be caught by the next probe run.
    """
    # PatchEmbedding (only on the first chip's tower).
    pe = getattr(tower, "patch_embed", None)
    if pe is not None and migrate_patch_embed:
        if hasattr(pe, "_linear_weight") and pe._linear_weight is not None:
            pe._linear_weight = _to_l1(pe._linear_weight)
        if hasattr(pe, "_linear_bias") and pe._linear_bias is not None:
            pe._linear_bias = _to_l1(pe._linear_bias)

    # Positional embedding (only on the chip that holds it).
    if migrate_pos_embed and getattr(tower, "pos_emb_weights", None) is not None:
        tower.pos_emb_weights = _to_l1(tower.pos_emb_weights)

    # Post-LN (only on the last slice's tower).
    if migrate_post_ln:
        if getattr(tower, "post_ln_weight", None) is not None:
            tower.post_ln_weight = _to_l1(tower.post_ln_weight)
        if getattr(tower, "post_ln_bias", None) is not None:
            tower.post_ln_bias = _to_l1(tower.post_ln_bias)

    # Every transformer block on this chip.
    for block in tower.blocks:
        # bf16 LayerNorm weights/biases — skip by default (see docstring).
        if migrate_layer_norms:
            if getattr(block, "ln1_weight", None) is not None:
                block.ln1_weight = _to_l1(block.ln1_weight)
            if getattr(block, "ln1_bias", None) is not None:
                block.ln1_bias = _to_l1(block.ln1_bias)
            if getattr(block, "ln2_weight", None) is not None:
                block.ln2_weight = _to_l1(block.ln2_weight)
            if getattr(block, "ln2_bias", None) is not None:
                block.ln2_bias = _to_l1(block.ln2_bias)
        # Attention QKV (concatenated) + output projection — bf8_b workhorses.
        if migrate_attention:
            attn = block.attention
            attn.wqkv = _to_l1(attn.wqkv)
            if attn.bqkv is not None:
                attn.bqkv = _to_l1(attn.bqkv)
            attn.wo = _to_l1(attn.wo)
            if getattr(attn, "bo", None) is not None:
                attn.bo = _to_l1(attn.bo)
        # MLP fc1 / fc2 — bf8_b workhorses.
        if migrate_mlp:
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
        submesh:       the 8-chip (2,4) vision MeshDevice.
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
        # scaffolding pass. The 8-chip device split (Pi0_5OptionCVisionSliceSplit
        # — see the class below; default 4+4+4+4+3+3+3+2 layer layout with
        # mm_projector co-located on the last chip) is the device path —
        # selected via `device_siglip=True` on Pi0_5PipelineC.
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

        pixel_values: [N_cams, 3, H, W] — cameras stack along openpi's image-slot
        axis. Per-camera SigLIP outputs are flattened into the seq dim so the
        prefix is [1, N_cams*256 + S_lang, vlm_W].

        Note: the openpi prefix path also appends a robot-state slot. Like
        Option B's vision slice, we leave that to the orchestrator so the
        bring-up dry run can be exercised with image+text only.
        """
        image_hidden = self.embed_images(pixel_values)  # [N_cams, 256, W]
        n_cams, n_patches, width = image_hidden.shape
        if n_cams != 1:
            image_hidden = ttnn.reshape(image_hidden, (1, n_cams * n_patches, width))
        lang_hidden = self.embed_language_tokens(language_token_ids)
        return ttnn.concat([image_hidden, lang_hidden], dim=1)


# ---------------------------------------------------------------------------- #
# On-device 8-chip SigLIP split (projector co-located on last chip)             #
# ---------------------------------------------------------------------------- #


class Pi0_5OptionCVisionSliceSplit:
    """On-device SigLIP-27 split across all 8 vision chips + co-located projector.

    Placement (one micro-submesh = one chip on the 8-chip (2,4) vision submesh):
        chip 0: patch_embed + pos_embed + SigLIP layers  0– 3   (4 layers)
        chip 1: SigLIP layers  4– 7                             (4 layers)
        chip 2: SigLIP layers  8–11                             (4 layers)
        chip 3: SigLIP layers 12–15                             (4 layers)
        chip 4: SigLIP layers 16–18                             (3 layers)
        chip 5: SigLIP layers 19–21                             (3 layers)
        chip 6: SigLIP layers 22–24                             (3 layers)
        chip 7: SigLIP layers 25–26 + post_ln + mm_projector    (2 layers)

    Why this split (4+4+4+4+3+3+3+2 = 27 total):
        - Busiest chip carries 4 SigLIP layers (~4×18 MB ≈ 72 MB at bf8_b)
          plus the patch_embed / pos_embed (~3 MB) on chip 0 — well below
          the 175 MB L1 cap. Plenty of room for L1-resident LN / norm
          weights, biases, and per-step activations + the matmul kernel's
          static CB region.
        - The projector chip (last chip) carries only 2 SigLIP layers
          plus post_ln + mm_projector (~36 + 5 + 4 ≈ 45 MB) — the
          smallest chip, intentionally, since it also has to hold the
          language-token embedding output on the same chip.

    Wall-clock impact vs the previous 7/7/7/6 split:
        - Total compute unchanged (27 layer-times).
        - Host bounces grow from 3 to 7 (chip 0→1→2→…→7), but each hop
          carries the same [B, 256, 1152] BS-sharded hidden — small enough
          that the additional bounces are dominated by the per-chip
          matmul work that now runs without a DRAM-read penalty.
        - Last chunk → mm_projector handoff stays in L1 (same chip).

    Language token embedding stays on HOST (the 527 MB embed_tokens table
    doesn't fit on a single chip; vocab sharding is a later follow-up).
    The host-resolved language embedding is uploaded to chip 7 to be
    concatenated with the projected vision features there.

    Construction is dry-run-safe even on shrunk configs because the SigLIP
    config / weights are independent of `vlm_depth` / `expert_depth`.

    Args:
        config:           full PaliGemma config.
        weights:          full categorized weights dict.
        micro_submeshes:  list of 8 single-chip MeshDevices (carved from
                          the 8-chip (2,4) vision submesh). Each holds one
                          SigLIP chunk; the last chip ALSO holds the
                          mm_projector.
        layers_per_chip:  per-chip layer counts. Default
                          `(4, 4, 4, 4, 3, 3, 3, 2)` sums to 27. Length
                          must equal len(micro_submeshes).
        siglip_depth:     total SigLIP depth (default 27 = sum of
                          `layers_per_chip`).
        weights_in_l1:    if True, post-construction migrate every
                          vision tower weight (patch_embed, pos_embed,
                          LN, post_ln, attn, MLP) to L1 — full-L1
                          residence per the project's NO-DRAM mandate.
                          Off by default to keep regression mode safe.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        micro_submeshes: List,
        layers_per_chip: Optional[List[int]] = None,
        siglip_depth: int = 27,
        weights_in_l1: bool = False,
        keep_pos_embed_dram: bool = False,
    ) -> None:
        if len(micro_submeshes) not in (4, 8):
            raise ValueError(f"Pi0_5OptionCVisionSliceSplit needs 4 or 8 micro-submeshes; got {len(micro_submeshes)}")
        for i, sm in enumerate(micro_submeshes):
            if sm.get_num_devices() != 1:
                raise ValueError(f"micro_submeshes[{i}] must be a 1-chip submesh " f"({sm.get_num_devices()} devices)")

        # Default split:
        #   8 chips: 4+4+4+4+3+3+3+2 (production layout — busiest chip 4 layers)
        #   4 chips: 7+7+7+6           (4-chip "tile" — busiest chip 7 layers,
        #                               mm_projector co-locates with the 6-layer
        #                               chip so per-chip L1 stays comfortable)
        # The mm_projector co-locates with the last SigLIP chunk so the
        # SigLIP → projector handoff is in-L1.
        if layers_per_chip is None:
            if len(micro_submeshes) == 8:
                layers_per_chip = [4, 4, 4, 4, 3, 3, 3, 2]
            else:
                layers_per_chip = [7, 7, 7, 6]
        if len(layers_per_chip) != len(micro_submeshes):
            raise ValueError(
                f"layers_per_chip length ({len(layers_per_chip)}) must equal "
                f"number of micro_submeshes ({len(micro_submeshes)})"
            )
        if any(n < 0 for n in layers_per_chip):
            raise ValueError(f"layers_per_chip entries must be ≥ 0; got {layers_per_chip}")
        if sum(layers_per_chip) != siglip_depth:
            raise ValueError(
                f"sum(layers_per_chip) ({sum(layers_per_chip)}) must equal " f"siglip_depth ({siglip_depth})"
            )

        self.config = config
        self.micro_submeshes = micro_submeshes
        self.layers_per_chip = layers_per_chip
        self.siglip_depth = siglip_depth
        self.num_siglip_chips = len(layers_per_chip)
        # mm_projector co-locates with the last SigLIP chunk (chip 3 in
        # the default layout). This frees the chip that used to be
        # projector-only for additional SigLIP layers, and the in-L1
        # handoff from SigLIP chunk → projector skips one host bounce.
        self.projector_chip_idx = self.num_siglip_chips - 1

        vlm_vision = weights["vlm_vision"]
        vlm_projector = weights["vlm_projector"]

        # Build N SigLIP chunks per `layers_per_chip`; each chunk holds
        # the range of layers assigned to its chip.
        self.siglip_chunks: List[SigLIPVisionTowerTTNN] = []
        lo = 0
        for chunk_idx, n_layers in enumerate(layers_per_chip):
            hi = lo + n_layers
            is_first = chunk_idx == 0
            is_last = chunk_idx == self.num_siglip_chips - 1
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
                # Keeping pos_embed in DRAM avoids the ttnn.embedding op's
                # static CB region clashing with adjacent L1 weight buffers
                # on tight per-chip layouts (e.g. 4-chip vision with 7
                # SigLIP layers + patch_embed on chip 0).
                _migrate_tower_weights_to_l1(tower, migrate_pos_embed=not keep_pos_embed_dram)
            self.siglip_chunks.append(tower)
            lo = hi

        # mm_projector lives on the SAME chip as the last SigLIP chunk
        # (chip 3 by default). Sharing the submesh means the SigLIP →
        # projector handoff stays in L1 — no host bounce.
        self.mm_projector = MultiModalProjectorTTNN(vlm_projector, micro_submeshes[self.projector_chip_idx])
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
        """Run SigLIP-27 across all SigLIP chunks, then mm_projector.

        Layout (default `layers_per_chip=[4, 4, 4, 4, 3, 3, 3, 2]` on 8 chips):
            chip 0: patch_embed + pos_embed + SigLIP 0..3
            chip 1: SigLIP 4..7
            chip 2: SigLIP 8..11
            chip 3: SigLIP 12..15
            chip 4: SigLIP 16..18
            chip 5: SigLIP 19..21
            chip 6: SigLIP 22..24
            chip 7: SigLIP 25..26 + post_ln + mm_projector
        Host-bounces between consecutive SigLIP chips; the last chunk →
        mm_projector handoff is in-L1 (same chip), so we skip the bounce
        after the final SigLIP chunk.

        Returns: [B, num_patches, vlm_W] on `micro_submeshes[projector_chip_idx]`.
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

        # Chip 0: patch_embed + pos_embed + first chunk's SigLIP layers.
        h = self.siglip_chunks[0].forward(pixel_values_ttnn)
        ttnn.deallocate(pixel_values_ttnn)

        # Chip 1..N-1: forward from hidden, host-bouncing between chips.
        for chunk_idx in range(1, self.num_siglip_chips):
            h_next = send_activation_via_host(h, self.micro_submeshes[chunk_idx])
            ttnn.deallocate(h)
            h = self.siglip_chunks[chunk_idx].forward_from_hidden(h_next)

        # mm_projector lives on the same chip as the last SigLIP chunk
        # (chip 3 by default) — feed it directly without a host bounce.
        return self.mm_projector.forward(h)

    # ------------------------------------------------------------------ #
    # Language path                                                      #
    # ------------------------------------------------------------------ #

    def embed_language_tokens(self, token_ids: torch.Tensor) -> "ttnn.Tensor":
        """Host-resolved language embedding uploaded to the projector chip."""
        embedded = torch.nn.functional.embedding(token_ids.to(torch.long), self._host_embed_table)
        proj_chip = self.micro_submeshes[self.projector_chip_idx]
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
        """[1, N_cams*num_patches + S_lang, vlm_W] on micro_submeshes[projector_chip_idx]."""
        image_hidden = self.embed_images(pixel_values)  # [N_cams, 256, W]
        n_cams, n_patches, width = image_hidden.shape
        if n_cams != 1:
            image_hidden = ttnn.reshape(image_hidden, (1, n_cams * n_patches, width))
        lang_hidden = self.embed_language_tokens(language_token_ids)
        return ttnn.concat([image_hidden, lang_hidden], dim=1)

    @property
    def last_chip_submesh(self):
        """Where build_prefix_hidden's output lives — the projector chip."""
        return self.micro_submeshes[self.projector_chip_idx]


# ---------------------------------------------------------------------------- #
# Parent-mesh vision slice (D2D SigLIP).                                        #
#                                                                              #
# Same logical mapping as Pi0_5OptionCVisionSliceSplit (4+4+4+4+3+3+3+2 split   #
# of SigLIP-27 across 8 vision chips + co-located mm_projector), but weights   #
# and activations live on the VISION SUBMESH as sharded tensors, and chunk-    #
# to-chunk transitions use ttnn.point_to_point on the vision submesh's fabric  #
# (validated by tests/test_vision_submesh_p2p_smoke.py). 6 same-row hops + 1  #
# diagonal (chip 3 → 4, (0,3) → (1,0)) handled by send_shard_via_p2p_multihop. #
#                                                                              #
# Weights are uploaded chunk-LOCAL via chunk-position-stacked sharded tensors  #
# (one tensor per per-chunk-layer-position per weight category). Per chip      #
# L1 cost matches the existing single-chip-submesh budget (~140 MB busy chip). #
# ---------------------------------------------------------------------------- #


def _siglip_layer_key(layer_idx: int, suffix: str) -> str:
    """Build a SigLIP weight key for an absolute layer index."""
    return f"vision_model.encoder.layers.{layer_idx}.{suffix}"


def _siglip_layer_key_alt(layer_idx: int, suffix: str) -> str:
    """Legacy SigLIP weight key (no vision_model. prefix)."""
    return f"encoder.layers.{layer_idx}.{suffix}"


def _fetch_layer_weight(
    full_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    suffix: str,
) -> Optional[torch.Tensor]:
    """Return one SigLIP layer's weight tensor, trying both prefix conventions."""
    k = _siglip_layer_key(layer_idx, suffix)
    if k in full_weights:
        return full_weights[k]
    k_alt = _siglip_layer_key_alt(layer_idx, suffix)
    if k_alt in full_weights:
        return full_weights[k_alt]
    return None


def _load_siglip_weights_chunk_stacked_sharded(
    full_weights: Dict[str, torch.Tensor],
    layers_per_chip: List[int],
    vision_submesh,
    num_heads: int,
    head_dim: int,
    dtype=ttnn.bfloat8_b,
) -> Dict[str, List[Optional["ttnn.Tensor"]]]:
    """Upload all SigLIP layer weights as vision-submesh sharded tensors.

    For each per-chunk layer position `j ∈ [0, max(layers_per_chip))` and each
    weight category, build one vision-submesh stacked tensor where chunk i's
    slot holds the weights for layer (sum(layers_per_chip[:i]) + j). When a
    chunk has fewer than `j+1` layers, its slot is zeros and the forward will
    skip the matmul at that position (no live data, no garbage propagates).

    Per-chip L1 cost = (max layers per chunk) × (per-layer weights) ≈ 140 MB
    on the busiest chip — matches the existing single-chip-submesh budget.

    Returns a dict keyed by weight name (e.g. "q_proj_w", "fc1_w",
    "layer_norm1_w") → list of `max_layers_per_chunk` parent-mesh tensors,
    indexed by per-chunk position. Entry j is None if no chunk has a layer
    at position j (impossible with current splits but defensive).
    """
    num_chunks = len(layers_per_chip)
    max_layers_per_chunk = max(layers_per_chip) if layers_per_chip else 0
    num_devices = vision_submesh.get_num_devices()
    if num_devices != num_chunks:
        raise ValueError(
            f"vision_submesh device count {num_devices} must equal " f"num_chunks {num_chunks} (one chunk per chip)"
        )

    # Build the absolute layer range for each chunk.
    chunk_layer_ranges: List[Tuple[int, int]] = []
    lo = 0
    for n in layers_per_chip:
        chunk_layer_ranges.append((lo, lo + n))
        lo += n

    # SigLIP attention weights need head-dim padding (72 → 96 for SigLIP-base)
    # because TT-Metal tile alignment requires head_dim % 32 == 0. Q/K/V get
    # padded then concat'd into a fused wqkv matrix (matches the existing
    # SigLIPAttentionTTNN __init__); out_proj gets padded on its input axis.
    padded_head_dim = ((head_dim + 31) // 32) * 32
    pad_amount = padded_head_dim - head_dim

    def _pad_head_dim_weight(w: torch.Tensor, heads_out: bool) -> torch.Tensor:
        """Pad a Q/K/V or O weight matrix's head_dim from head_dim to padded_head_dim.

        For Q/K/V (heads_out=True): weight is [num_heads*head_dim, hidden]. Pad
        head_dim axis to padded_head_dim → [num_heads*padded_head_dim, hidden].
        For O (heads_out=False): weight is [hidden, num_heads*head_dim]. Pad the
        head_dim axis on the input side → [hidden, num_heads*padded_head_dim].
        """
        if pad_amount == 0:
            return w
        if heads_out:
            # w: [num_heads * head_dim, hidden] → reshape to [num_heads, head_dim, hidden]
            w_r = w.reshape(num_heads, head_dim, w.shape[-1])
            # pad last-axis-but-one (head_dim): F.pad pads in reverse, so (left, right) of each dim
            # We want to pad dim=1 (head_dim). torch.nn.functional.pad takes pairs starting from last.
            w_p = torch.nn.functional.pad(w_r, (0, 0, 0, pad_amount))  # pad head_dim
            return w_p.reshape(num_heads * padded_head_dim, w.shape[-1])
        else:
            # w: [hidden, num_heads * head_dim] → reshape to [hidden, num_heads, head_dim]
            w_r = w.reshape(w.shape[0], num_heads, head_dim)
            w_p = torch.nn.functional.pad(w_r, (0, pad_amount))  # pad head_dim
            return w_p.reshape(w.shape[0], num_heads * padded_head_dim)

    def _pad_head_dim_bias(b: torch.Tensor) -> torch.Tensor:
        """Pad a Q/K/V bias from [num_heads*head_dim] to [num_heads*padded_head_dim]."""
        if pad_amount == 0:
            return b
        b_r = b.reshape(num_heads, head_dim)
        b_p = torch.nn.functional.pad(b_r, (0, pad_amount))
        return b_p.reshape(num_heads * padded_head_dim)

    # MLP and LN weight suffixes — uploaded with no special handling.
    matmul_suffixes_mlp = [
        "mlp.fc1.weight",
        "mlp.fc2.weight",
    ]
    bias_suffixes_mlp_and_ln = [
        "mlp.fc1.bias",
        "mlp.fc2.bias",
        "layer_norm1.bias",
        "layer_norm2.bias",
    ]
    ln_weight_suffixes = [
        "layer_norm1.weight",
        "layer_norm2.weight",
    ]
    # Out-proj weight + bias get special handling for input-side head-dim padding.
    out_proj_w_suffix = "self_attn.out_proj.weight"
    out_proj_b_suffix = "self_attn.out_proj.bias"

    def _build_position_tensor(
        suffix: str,
        position: int,
        weight_dtype,
        is_matmul: bool,
        is_norm: bool = False,
    ) -> Optional["ttnn.Tensor"]:
        """Build the chunk-position-`j` vision-submesh sharded tensor for `suffix`."""
        # Discover a reference shape from the first chunk that has a layer at this position.
        ref: Optional[torch.Tensor] = None
        for chunk_idx in range(num_chunks):
            cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
            if cl_lo + position < cl_hi:
                global_layer = cl_lo + position
                ref = _fetch_layer_weight(full_weights, global_layer, suffix)
                if ref is not None:
                    break
        if ref is None:
            return None
        ref_t = ref.T.contiguous() if is_matmul else ref
        target_shape = tuple(ref_t.shape)
        stacked = torch.zeros((num_chunks,) + target_shape, dtype=ref_t.dtype)
        for chunk_idx in range(num_chunks):
            cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
            if cl_lo + position >= cl_hi:
                continue  # this chunk has no layer at this position; leave zeros
            global_layer = cl_lo + position
            w = _fetch_layer_weight(full_weights, global_layer, suffix)
            if w is None:
                continue
            if is_matmul:
                w = w.T.contiguous()
            stacked[chunk_idx] = w
        return ttnn.from_torch(
            stacked,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=vision_submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(vision_submesh, dim=0),
        )

    out: Dict[str, List[Optional["ttnn.Tensor"]]] = {}

    # MLP weights (no padding).
    for s in matmul_suffixes_mlp:
        key = s.replace(".weight", "").replace("mlp.", "")
        per_pos: List[Optional["ttnn.Tensor"]] = []
        for j in range(max_layers_per_chunk):
            t = _build_position_tensor(s, j, dtype, is_matmul=True)
            per_pos.append(t)
        out[key] = per_pos

    # Fused wqkv per position: concat([wq, wk, wv], dim=-1) after .T and per-head padding.
    # Each layer's contribution at chunk i's slot: [hidden, 3*num_heads*padded_head_dim].
    def _build_wqkv_position(position: int) -> Optional["ttnn.Tensor"]:
        # Probe reference shape from first chunk that has a layer at this position.
        ref: Optional[torch.Tensor] = None
        for chunk_idx in range(num_chunks):
            cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
            if cl_lo + position < cl_hi:
                wq_ref = _fetch_layer_weight(full_weights, cl_lo + position, "self_attn.q_proj.weight")
                if wq_ref is not None:
                    ref = wq_ref
                    break
        if ref is None:
            return None
        hidden = ref.shape[-1]
        wqkv_n = 3 * num_heads * padded_head_dim
        # Stacked across chunks: [num_chunks, hidden, wqkv_n] (uploaded post-.T form).
        stacked = torch.zeros((num_chunks, hidden, wqkv_n), dtype=ref.dtype)
        for chunk_idx in range(num_chunks):
            cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
            if cl_lo + position >= cl_hi:
                continue
            global_layer = cl_lo + position
            wq = _fetch_layer_weight(full_weights, global_layer, "self_attn.q_proj.weight")
            wk = _fetch_layer_weight(full_weights, global_layer, "self_attn.k_proj.weight")
            wv = _fetch_layer_weight(full_weights, global_layer, "self_attn.v_proj.weight")
            if wq is None or wk is None or wv is None:
                continue
            wq_p = _pad_head_dim_weight(wq, heads_out=True).T.contiguous()  # [hidden, num_heads*padded]
            wk_p = _pad_head_dim_weight(wk, heads_out=True).T.contiguous()
            wv_p = _pad_head_dim_weight(wv, heads_out=True).T.contiguous()
            fused = torch.cat([wq_p, wk_p, wv_p], dim=-1)
            stacked[chunk_idx] = fused
        return ttnn.from_torch(
            stacked,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=vision_submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(vision_submesh, dim=0),
        )

    def _build_bqkv_position(position: int) -> Optional["ttnn.Tensor"]:
        # Reference probe.
        ref: Optional[torch.Tensor] = None
        for chunk_idx in range(num_chunks):
            cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
            if cl_lo + position < cl_hi:
                bq_ref = _fetch_layer_weight(full_weights, cl_lo + position, "self_attn.q_proj.bias")
                if bq_ref is not None:
                    ref = bq_ref
                    break
        if ref is None:
            return None
        bqkv_n = 3 * num_heads * padded_head_dim
        # Tile-pad 1D bias to [1, bqkv_n] for the kernel.
        stacked = torch.zeros((num_chunks, 1, bqkv_n), dtype=ref.dtype)
        for chunk_idx in range(num_chunks):
            cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
            if cl_lo + position >= cl_hi:
                continue
            global_layer = cl_lo + position
            bq = _fetch_layer_weight(full_weights, global_layer, "self_attn.q_proj.bias")
            bk = _fetch_layer_weight(full_weights, global_layer, "self_attn.k_proj.bias")
            bv = _fetch_layer_weight(full_weights, global_layer, "self_attn.v_proj.bias")
            if bq is None or bk is None or bv is None:
                continue
            bq_p = _pad_head_dim_bias(bq)
            bk_p = _pad_head_dim_bias(bk)
            bv_p = _pad_head_dim_bias(bv)
            fused = torch.cat([bq_p, bk_p, bv_p], dim=0).reshape(1, -1)
            stacked[chunk_idx] = fused
        return ttnn.from_torch(
            stacked,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=vision_submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(vision_submesh, dim=0),
        )

    def _build_out_proj_w_position(position: int) -> Optional["ttnn.Tensor"]:
        ref: Optional[torch.Tensor] = None
        for chunk_idx in range(num_chunks):
            cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
            if cl_lo + position < cl_hi:
                ref = _fetch_layer_weight(full_weights, cl_lo + position, out_proj_w_suffix)
                if ref is not None:
                    break
        if ref is None:
            return None
        hidden = ref.shape[0]
        out_n = num_heads * padded_head_dim
        # Uploaded as .T form: shape [hidden, out_n] per slot.
        # Wait — out_proj is [hidden, num_heads*head_dim]. After padding input axis:
        # [hidden, num_heads*padded_head_dim]. After .T: [num_heads*padded, hidden].
        # ttnn.linear(attn_concat [..., num_heads*padded], W [num_heads*padded, hidden]) → [..., hidden]
        # So the stacked uploaded shape is [num_chunks, num_heads*padded, hidden].
        stacked = torch.zeros((num_chunks, out_n, hidden), dtype=ref.dtype)
        for chunk_idx in range(num_chunks):
            cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
            if cl_lo + position >= cl_hi:
                continue
            global_layer = cl_lo + position
            wo = _fetch_layer_weight(full_weights, global_layer, out_proj_w_suffix)
            if wo is None:
                continue
            wo_p = _pad_head_dim_weight(wo, heads_out=False).T.contiguous()  # [num_heads*padded, hidden]
            stacked[chunk_idx] = wo_p
        return ttnn.from_torch(
            stacked,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=vision_submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(vision_submesh, dim=0),
        )

    wqkv_per_pos: List[Optional["ttnn.Tensor"]] = []
    bqkv_per_pos: List[Optional["ttnn.Tensor"]] = []
    out_proj_per_pos: List[Optional["ttnn.Tensor"]] = []
    for j in range(max_layers_per_chunk):
        wqkv_per_pos.append(_build_wqkv_position(j))
        bqkv_per_pos.append(_build_bqkv_position(j))
        out_proj_per_pos.append(_build_out_proj_w_position(j))
    out["wqkv"] = wqkv_per_pos
    out["bqkv"] = bqkv_per_pos
    out["out_proj"] = out_proj_per_pos

    # LN weights at bf16 (matches the existing slice's precision policy).
    for s in ln_weight_suffixes:
        key = s.replace(".weight", "").replace(".", "_")
        per_pos = []
        for j in range(max_layers_per_chunk):
            t = _build_position_tensor(s, j, ttnn.bfloat16, is_matmul=False, is_norm=True)
            per_pos.append(t)
        out[key + "_w"] = per_pos

    # Biases (1D). Keep at bf8_b for matmul biases, bf16 for LN biases.
    # Attention Q/K/V biases handled via the fused bqkv above. Out-proj bias
    # is regular (input axis isn't padded).
    for s in bias_suffixes_mlp_and_ln + [out_proj_b_suffix]:
        key = s.replace(".bias", "_b").replace("self_attn.", "").replace("mlp.", "").replace(".", "_")
        per_pos = []
        for j in range(max_layers_per_chunk):
            # Probe for reference shape.
            ref: Optional[torch.Tensor] = None
            for chunk_idx in range(num_chunks):
                cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
                if cl_lo + j < cl_hi:
                    ref = _fetch_layer_weight(full_weights, cl_lo + j, s)
                    if ref is not None:
                        break
            if ref is None:
                per_pos.append(None)
                continue
            target_shape = (1, ref.shape[0])  # tile-pad to [1, dim]
            stacked = torch.zeros((num_chunks,) + target_shape, dtype=ref.dtype)
            for chunk_idx in range(num_chunks):
                cl_lo, cl_hi = chunk_layer_ranges[chunk_idx]
                if cl_lo + j >= cl_hi:
                    continue
                w = _fetch_layer_weight(full_weights, cl_lo + j, s)
                if w is None:
                    continue
                stacked[chunk_idx] = w.reshape(1, -1)
            b_dtype = ttnn.bfloat16 if "layer_norm" in s else ttnn.bfloat8_b
            t = ttnn.from_torch(
                stacked,
                dtype=b_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=vision_submesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(vision_submesh, dim=0),
            )
            per_pos.append(t)
        out[key] = per_pos

    return out


class Pi0_5OptionCVisionSliceSplitParent:
    """8-chip vision-submesh SigLIP slice — D2D variant of Pi0_5OptionCVisionSliceSplit.

    STATUS: scaffolding. Weight upload + per-chunk coord helpers; the
    inline forward (patch_embed + per-chunk encoder layers + P2P advance +
    post_ln + mm_projector) is the next sub-commit.

    Same logical placement as the host-bounce variant:
        chip (0,0): patch_embed + pos_embed + SigLIP layers 0-3
        chip (0,1): SigLIP layers 4-7
        chip (0,2): SigLIP layers 8-11
        chip (0,3): SigLIP layers 12-15
        chip (1,0): SigLIP layers 16-18                  ← diagonal P2P entry
        chip (1,1): SigLIP layers 19-21
        chip (1,2): SigLIP layers 22-24
        chip (1,3): SigLIP layers 25-26 + post_ln + mm_projector

    But all weights live as chunk-stacked sharded tensors on the vision
    submesh (8 chips), with each chip's slot holding ONLY its chunk's
    weights — per-chip L1 cost matches the existing budget (~140 MB busy
    chip). Activations flow chip-to-chip via send_shard_via_p2p (single
    hop for same-row transitions, send_shard_via_p2p_multihop for the
    one diagonal transition (0,3) → (1,0)).

    Args:
        config:           full PaliGemma config (uses .siglip_config).
        weights:          full categorized weights dict.
        vision_submesh:   the 2x4 vision submesh (carved from the parent).
        layers_per_chip:  default (4, 4, 4, 4, 3, 3, 3, 2).
        siglip_depth:     default 27.

    NOT YET IMPLEMENTED:
        - patch_embed forward on chip (0,0)
        - per-chunk encoder forward on parent-mesh tensors
        - P2P transitions
        - post_ln + mm_projector on chip (1,3)
        - mm_projector wiring
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        vision_submesh,
        layers_per_chip: Optional[List[int]] = None,
        siglip_depth: int = 27,
    ) -> None:
        if layers_per_chip is None:
            layers_per_chip = [4, 4, 4, 4, 3, 3, 3, 2]
        if sum(layers_per_chip) != siglip_depth:
            raise ValueError(f"sum(layers_per_chip) ({sum(layers_per_chip)}) must equal siglip_depth ({siglip_depth})")
        n_chunks = len(layers_per_chip)
        if vision_submesh.get_num_devices() != n_chunks:
            raise ValueError(
                f"vision_submesh device count {vision_submesh.get_num_devices()} must equal "
                f"num_chunks {n_chunks} (one chunk per chip)"
            )

        self.config = config
        self.vision_submesh = vision_submesh
        self.layers_per_chip = layers_per_chip
        self.siglip_depth = siglip_depth
        self.num_chunks = n_chunks
        self.max_layers_per_chunk = max(layers_per_chip)
        self.projector_chunk_idx = n_chunks - 1
        # Vision submesh shape (rows, cols); reads from the submesh handle.
        # ttnn submeshes expose .shape as (rows, cols).
        self.submesh_shape = (vision_submesh.shape[0], vision_submesh.shape[1])

        # SigLIP attention head config — needed for QKV padding + the forward.
        siglip_cfg = config.siglip_config
        self.num_heads = siglip_cfg.num_attention_heads
        self.head_dim = siglip_cfg.head_dim
        self.padded_head_dim = ((self.head_dim + 31) // 32) * 32  # 72 → 96
        self.hidden_size = siglip_cfg.hidden_size
        self.layer_norm_eps = siglip_cfg.layer_norm_eps

        # Upload all SigLIP layer weights as chunk-stacked sharded tensors.
        self.weights_on_vision = _load_siglip_weights_chunk_stacked_sharded(
            weights["vlm_vision"],
            layers_per_chip,
            vision_submesh,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

    def chunk_coord(self, chunk_idx: int) -> Tuple[int, int]:
        """Vision-submesh coord (row, col) of the given chunk's chip."""
        cols = self.submesh_shape[1]
        return (chunk_idx // cols, chunk_idx % cols)

    def chunk_for_layer(self, global_layer_idx: int) -> int:
        """Map a global SigLIP layer index → owning chunk index."""
        lo = 0
        for chunk_idx, n in enumerate(self.layers_per_chip):
            if global_layer_idx < lo + n:
                return chunk_idx
            lo += n
        raise ValueError(f"global_layer_idx {global_layer_idx} out of range for siglip_depth {self.siglip_depth}")

    def position_in_chunk(self, global_layer_idx: int) -> int:
        """Position of `global_layer_idx` within its owning chunk (0-indexed)."""
        lo = 0
        for n in self.layers_per_chip:
            if global_layer_idx < lo + n:
                return global_layer_idx - lo
            lo += n
        raise ValueError(f"global_layer_idx {global_layer_idx} out of range for siglip_depth {self.siglip_depth}")

    def forward_chunk(self, hidden_states: "ttnn.Tensor", chunk_idx: int) -> "ttnn.Tensor":
        """Run all encoder layers of chunk `chunk_idx` on the vision submesh.

        Per layer (positions 0..layers_per_chip[chunk_idx]-1):
            1. LayerNorm1 (weight + bias)
            2. Fused wqkv linear → nlp_create_qkv_heads (MHA, num_kv_heads=num_heads)
            3. SDPA (is_causal=False, scale=1/sqrt(head_dim))
            4. nlp_concat_heads → out_proj linear
            5. Residual add
            6. LayerNorm2
            7. fc1 (linear + GELU) → fc2 (linear)
            8. Residual add

        `hidden_states` must be a vision-submesh tensor [B, 1, M, hidden]
        (4D for nlp_create_qkv_heads). Live data at `chunk_idx`'s chip slot.
        Returns the same-shape tensor with the chunk's encoder run applied
        — live data still at `chunk_idx`'s chip slot.

        Doesn't apply post_ln or mm_projector; those are chunk-final-only
        (the last chunk's caller handles them).

        Doesn't include P2P advance between chunks — that's done by the
        outer chain (the next sub-commit).
        """
        if chunk_idx < 0 or chunk_idx >= self.num_chunks:
            raise ValueError(f"chunk_idx {chunk_idx} out of range [0, {self.num_chunks})")
        n_layers = self.layers_per_chip[chunk_idx]
        if n_layers == 0:
            return hidden_states  # nothing to do

        # Get the per-position weight slots used in this forward.
        wqkv = self.weights_on_vision["wqkv"]
        bqkv = self.weights_on_vision["bqkv"]
        out_proj = self.weights_on_vision["out_proj"]
        out_proj_b = self.weights_on_vision["out_proj_b"]
        fc1 = self.weights_on_vision["fc1"]
        fc2 = self.weights_on_vision["fc2"]
        fc1_b = self.weights_on_vision["fc1_b"]
        fc2_b = self.weights_on_vision["fc2_b"]
        ln1_w = self.weights_on_vision["layer_norm1_w"]
        ln1_b = self.weights_on_vision["layer_norm1_b"]
        ln2_w = self.weights_on_vision["layer_norm2_w"]
        ln2_b = self.weights_on_vision["layer_norm2_b"]

        scale = 1.0 / (self.head_dim**0.5)
        h = hidden_states

        for j in range(n_layers):
            # ---- LayerNorm 1 ----
            normed = ttnn.layer_norm(
                h,
                weight=ln1_w[j],
                bias=ln1_b[j],
                epsilon=self.layer_norm_eps,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # ---- Fused QKV ----
            xqkv = ttnn.linear(
                normed,
                wqkv[j],
                bias=bqkv[j],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(normed)

            # ---- Split heads (MHA: num_kv_heads = num_heads) ----
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                xqkv,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # ---- SDPA (no causal mask, no past_kv — vision is bidirectional) ----
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=scale,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            # ---- Concat heads ----
            attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn)

            # ---- Out projection ----
            o = ttnn.linear(
                attn_flat,
                out_proj[j],
                bias=out_proj_b[j],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_flat)

            # ---- Residual add (attention) ----
            h_post_attn = ttnn.add(h, o, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(o)
            ttnn.deallocate(h)

            # ---- LayerNorm 2 ----
            normed = ttnn.layer_norm(
                h_post_attn,
                weight=ln2_w[j],
                bias=ln2_b[j],
                epsilon=self.layer_norm_eps,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # ---- MLP: fc1 + GELU + fc2 ----
            x = ttnn.linear(
                normed,
                fc1[j],
                bias=fc1_b[j],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                activation="gelu",
            )
            ttnn.deallocate(normed)
            mlp_out = ttnn.linear(
                x,
                fc2[j],
                bias=fc2_b[j],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x)

            # ---- Residual add (MLP) ----
            h_new = ttnn.add(h_post_attn, mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(mlp_out)
            ttnn.deallocate(h_post_attn)
            h = h_new

        return h
