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

        Returns: [B, num_patches + S_lang, vlm_W] on the vision submesh.

        Note: the openpi prefix path also appends a robot-state slot. Like
        Option B's vision slice, we leave that to the orchestrator so the
        bring-up dry run can be exercised with image+text only.
        """
        image_hidden = self.embed_images(pixel_values)
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
    ) -> None:
        if len(micro_submeshes) != 8:
            raise ValueError(f"Pi0_5OptionCVisionSliceSplit needs 8 micro-submeshes; got {len(micro_submeshes)}")
        for i, sm in enumerate(micro_submeshes):
            if sm.get_num_devices() != 1:
                raise ValueError(f"micro_submeshes[{i}] must be a 1-chip submesh " f"({sm.get_num_devices()} devices)")

        # Default split: 4+4+4+4+3+3+3+2 across all 8 chips of the (2,4)
        # vision submesh. Busiest chip carries 4 SigLIP layers — keeps
        # per-chip L1 well below the cap so ALL weight categories
        # (LN/patch_embed/pos_embed/post_ln/attn/MLP) can migrate to L1.
        # The mm_projector co-locates with the last SigLIP chunk (chip 7
        # by default) so the SigLIP → projector handoff is in-L1.
        if layers_per_chip is None:
            layers_per_chip = [4, 4, 4, 4, 3, 3, 3, 2]
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
                _migrate_tower_weights_to_l1(tower)
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
        """[B, num_patches + S_lang, vlm_W] on micro_submeshes[projector_chip_idx]."""
        image_hidden = self.embed_images(pixel_values)
        lang_hidden = self.embed_language_tokens(language_token_ids)
        return ttnn.concat([image_hidden, lang_hidden], dim=1)

    @property
    def last_chip_submesh(self):
        """Where build_prefix_hidden's output lives — the projector chip."""
        return self.micro_submeshes[self.projector_chip_idx]
