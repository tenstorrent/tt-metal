# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Image-conditioned Qwen3-VL-8B text encoder (the TI2I / editing path of Qwen-Image-2.1) in TTNN.

Adds three things to the text-only stack in `text_encoder.py`:

1. The vision tower (`model.visual.*`), reused from `models/tt_dit/encoders/qwen3vl/vision_qwen3vl.py`
   on a 1x1 mesh (no CCL, no parallel config). It turns `pixel_values` into one 4096-d token per 2x2
   patch group, plus one "deepstack" feature per entry of `deepstack_visual_indexes` ([8, 16, 24]).
2. Injection: the merged tokens REPLACE the token embeddings at the `<|image_pad|>` rows (host-side,
   in the embedding gather), and the three deepstack features are ADDED to those same rows after
   decoder layers 0, 1 and 2 (device-side, as a dense additive tensor that is zero on text rows).
3. mRoPE: 3-D (t, h, w) position ids and the `mrope_interleaved` frequency-to-axis assignment
   (mrope_section [24, 20, 20]). The position grid and the interleaved selection are reused from
   `models/tt_dit/encoders/qwen3vl/model_qwen3vl.py`, which is validated bitwise against
   `Qwen3VLModel.get_rope_index`. The resulting per-token angles are turned into the PERMUTED
   (adjacent-pair) cos/sin layout that `ttnn.experimental.rotary_embedding_llama` needs, matching the
   q/k row permutation the text-only path already applies.

Text-only prompts keep going through `Qwen3VLTextEncoder`; pass an existing instance as `te=` to share
its 6.9 GiB of device weights instead of loading the 8B model twice. The tower adds 1.19 GiB.

The vision tower's bf16 rounding can amplify through the decoder. Component tests compare both the
bf16 pipeline and an independent fp32 encoder reference; full-pipeline tests compare the final image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import ttnn
from models.tt_dit.encoders.qwen3vl.model_qwen3vl import _apply_interleaved_mrope, mrope_position_ids
from models.tt_dit.encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel, vision_cu_seqlens
from models.tt_dit.layers.linear import Linear as TTDitLinear

from ..common import rope as rope_mod
from ..common.config import IMAGE_PAD_TOKEN_ID, TE, TextEncoderConfig
from ..common.weights import LazyCheckpoint
from .text_encoder import TILE, Qwen3VLTextEncoder, TEPrecision

VIDEO_PAD_TOKEN_ID = 151656
VISUAL_PFX = "model.visual."
MROPE_SECTION: Tuple[int, int, int] = (24, 20, 20)


@dataclass(frozen=True)
class VisionConfig:
    """`vision_config` of Qwen/Qwen-Image-2.1's text_encoder (Qwen3-VL-8B)."""

    hidden_size: int = 1152
    num_heads: int = 16
    depth: int = 27
    intermediate_size: int = 4304
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 4096
    hidden_act: str = "gelu_pytorch_tanh"
    norm_eps: float = 1e-6  # hard-coded in transformers' Qwen3VLVisionBlock, not in the config
    deepstack_visual_indexes: Tuple[int, ...] = (8, 16, 24)


VISION = VisionConfig()


# ------------------------------------------------------------------------------------ host helpers
def visual_state_dict(ckpt: LazyCheckpoint) -> Dict[str, torch.Tensor]:
    """The `model.visual.*` sub-state-dict, keys relative to the tower (what tt_dit's loader wants)."""
    return {k[len(VISUAL_PFX) :]: ckpt.get(k) for k in ckpt.keys() if k.startswith(VISUAL_PFX)}


def mm_token_type_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """[1, L] token ids -> [1, L] modality ids (0 text, 1 image, 2 video), as the processor emits."""
    ids = input_ids.reshape(1, -1)
    out = torch.zeros_like(ids)
    out[ids == IMAGE_PAD_TOKEN_ID] = 1
    out[ids == VIDEO_PAD_TOKEN_ID] = 2
    return out


def vl_angles(
    position_ids: torch.Tensor,
    head_dim: int = TE.head_dim,
    theta: float = TE.rope_theta,
    section: Sequence[int] = MROPE_SECTION,
) -> torch.Tensor:
    """(3, L) or (3, 1, L) mRoPE positions -> [L, 64] rotation angles (fp64).

    Frequency slot i of the 64 gets the position of axis (i mod 3) -> (t, h, w) for i < 3*section[k],
    with the tail beyond 3*min(section) staying on t; see `_apply_interleaved_mrope`. For a text-only
    prompt all three axes agree and this collapses to `rope.te_cos_sin`'s 1-D table.
    """
    pos = position_ids.reshape(3, -1)
    inv = rope_mod._freqs(head_dim, theta)  # [64] fp64, same table as the text-only path
    freqs = pos[:, None, :, None].to(torch.float64) * inv[None, None, None, :]  # (3, 1, L, 64)
    return _apply_interleaved_mrope(freqs, section)[0]


def vl_cos_sin(position_ids: torch.Tensor, dtype=torch.bfloat16, **kw):
    """cos/sin [L, 128] in the permuted adjacent-pair layout used by the RoPE kernel."""
    return rope_mod.angles_to_cos_sin(vl_angles(position_ids, **kw), dtype)


def _set_linear_fidelity(mod, ck) -> None:
    """Override the matmul compute kernel config of every tt_dit `Linear` under `mod`."""
    for _, child in mod.named_children():
        if isinstance(child, TTDitLinear):
            child.compute_config = ck
        _set_linear_fidelity(child, ck)


# ------------------------------------------------------------------------------------------- model
class Qwen3VLEncoderVL:
    """Qwen3-VL-8B with condition images: vision tower + mRoPE + deepstack on top of the text stack.

    `encode(input_ids, pixel_values, image_grid_thw)` reproduces
    `Qwen3VLForConditionalGeneration(...).hidden_states[-1]` with the pipeline's identity hook on the
    final norm, i.e. the last decoder layer's output BEFORE `model.language_model.norm`.
    """

    def __init__(
        self,
        dev,
        ckpt: LazyCheckpoint,
        te: Optional[Qwen3VLTextEncoder] = None,
        prec: Optional[TEPrecision] = None,
        cfg: TextEncoderConfig = TE,
        layers: Optional[int] = None,
        vcfg: VisionConfig = VISION,
        load_vision: bool = True,
        vision_mm_fidelity=ttnn.MathFidelity.HiFi4,
        vision_fp32_residual: bool = True,
    ):
        self.dev = dev
        self.cfg = cfg
        self.vcfg = vcfg
        self.ckpt = ckpt
        self.vision_mm_fidelity = vision_mm_fidelity
        self.vision_fp32_residual = vision_fp32_residual
        self.te = te if te is not None else Qwen3VLTextEncoder(dev, ckpt, prec, cfg, layers)
        self.tower: Optional[Qwen3VlVisionModel] = None
        if load_vision:
            self.load_vision()

    def load_vision(self) -> None:
        v = self.vcfg
        self.tower = Qwen3VlVisionModel(
            hidden_size=v.hidden_size,
            num_heads=v.num_heads,
            depth=v.depth,
            intermediate_size=v.intermediate_size,
            in_channels=v.in_channels,
            patch_size=v.patch_size,
            temporal_patch_size=v.temporal_patch_size,
            spatial_merge_size=v.spatial_merge_size,
            num_position_embeddings=v.num_position_embeddings,
            out_hidden_size=v.out_hidden_size,
            hidden_act=v.hidden_act,
            norm_eps=v.norm_eps,
            deepstack_visual_indexes=v.deepstack_visual_indexes,
            mesh_device=self.dev,
            parallel_config=None,
            ccl_manager=None,
        )
        self.tower.load_torch_state_dict(visual_state_dict(self.ckpt))
        if self.vision_mm_fidelity is not None:
            # tt_dit's Linear defaults to HiFi2 for bf16 weights. The tower's last block amplifies the
            # residual stream ~75x (RMS 2.6 -> 199, absmax 15k: the usual ViT "massive activation"
            # channels), so every bit of error accumulated over the 27 blocks lands magnified in the
            # merger's LayerNorm. HiFi4 on the projections moves the merged tokens from PCC 0.9888 to
            # 0.9929 against the golden; the tower runs once per image, so the cost is irrelevant.
            ck = ttnn.init_device_compute_kernel_config(
                self.dev.arch(),
                math_fidelity=self.vision_mm_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            _set_linear_fidelity(self.tower, ck)

    # -------------------------------------------------------------------------------- vision tower
    def _run_blocks(self, h, rope, cu_seqlens, deepstack_out: List):
        """The tower's block loop with an fp32 residual stream, plus the deepstack taps.

        `Qwen3VlVisionModel.forward` keeps the residual in bf16, which is what the reference does too,
        but bf16 cannot hold this stream: the last block's output reaches 15k with an RMS of 199, so a
        rounding step there is worth ~64 absolute and the merger's LayerNorm divides by exactly those
        channels. Accumulating the two residual adds per block in fp32 while every matmul stays bf16
        costs one extra 19 MB tensor and takes the merged tokens from PCC 0.9929 to 0.9942 against
        the golden (0.9888 with neither this nor the HiFi4 override above).
        """
        t = self.tower
        fp32 = self.vision_fp32_residual
        cast = lambda x, dt: x if x.get_dtype() == dt else ttnn.typecast(x, dt)
        if fp32:
            h = cast(h, ttnn.float32)
        for i, blk in enumerate(t.blocks):
            n1 = cast(blk.norm1.forward(h), ttnn.bfloat16)
            a = blk.attn.forward(n1, pos_embeds=rope, cu_seqlens=cu_seqlens)
            ttnn.deallocate(n1)
            h2 = ttnn.add(h, a, dtype=ttnn.float32 if fp32 else None)
            ttnn.deallocate(a)
            ttnn.deallocate(h)
            n2 = cast(blk.norm2.forward(h2), ttnn.bfloat16)
            mo = blk.mlp.forward(n2)
            ttnn.deallocate(n2)
            h = ttnn.add(h2, mo, dtype=ttnn.float32 if fp32 else None)
            ttnn.deallocate(mo)
            ttnn.deallocate(h2)
            if i in t.deepstack_visual_indexes:
                merger = t.deepstack_merger_list[t.deepstack_visual_indexes.index(i)]
                deepstack_out.append(merger.forward(cast(h, ttnn.bfloat16)))
        return h

    def encode_image(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        """pixel_values [n_patches, 1536], grid [n_img, 3] -> (merged [n_merged, 4096],
        [deepstack_k [n_merged, 4096]] * 3), all host bf16."""
        assert self.tower is not None, "vision tower not loaded"
        grid = image_grid_thw.reshape(-1, 3).to(torch.long)
        total = int(grid.prod(-1).sum())
        assert pixel_values.shape[0] == total, f"{pixel_values.shape[0]} patch rows for a grid of {total}"

        cos, sin = self.tower.prepare_rope(grid)
        pos = self.tower.prepare_pos_embeds(grid)
        up = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rope = (up(cos), up(sin))
        px, pe = up(pixel_values), up(pos)
        h = ttnn.add(self.tower.patch_embed.forward(px), pe)
        ttnn.deallocate(px)
        ttnn.deallocate(pe)
        deepstack: List = []
        h = self._run_blocks(h, rope, vision_cu_seqlens(grid), deepstack)
        hb = h if h.get_dtype() == ttnn.bfloat16 else ttnn.typecast(h, ttnn.bfloat16)
        tokens = self.tower.merger.forward(hb)
        n = total // self.vcfg.spatial_merge_size**2
        merged = ttnn.to_torch(tokens).reshape(n, self.vcfg.out_hidden_size)
        feats = [ttnn.to_torch(f).reshape(n, self.vcfg.out_hidden_size) for f in deepstack]
        for tsr in (tokens, hb, h, *rope, *deepstack):
            if tsr.is_allocated():  # hb aliases h when the residual is bf16
                ttnn.deallocate(tsr)
        return merged.to(torch.bfloat16), [f.to(torch.bfloat16) for f in feats]

    # ------------------------------------------------------------------------------------ host prep
    def embed_with_vision(self, input_ids: torch.Tensor, merged: torch.Tensor) -> torch.Tensor:
        """[L, 4096] token embeddings with the `<|image_pad|>` rows replaced by the merged tokens."""
        ids = input_ids.reshape(-1)
        emb = self.te.embed(ids)
        mask = (ids == IMAGE_PAD_TOKEN_ID) | (ids == VIDEO_PAD_TOKEN_ID)
        n = int(mask.sum())
        assert n == merged.shape[0], f"{n} vision placeholders but {merged.shape[0]} merged tokens"
        emb[mask] = merged.to(emb.dtype)
        return emb

    def _upload_rope(self, position_ids: torch.Tensor, Sp: int):
        cos, sin = vl_cos_sin(position_ids, head_dim=self.cfg.head_dim, theta=self.cfg.rope_theta)
        L = cos.shape[0]
        f = lambda t: ttnn.from_torch(
            torch.cat([t, torch.zeros(Sp - L, t.shape[1], dtype=t.dtype)]).reshape(1, 1, Sp, -1)
            if Sp > L
            else t.reshape(1, 1, Sp, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # The tail rows are padding that causal attention never lets a real row see; cos=0 there is
        # harmless and cheaper than extending the position grid.
        return f(cos), f(sin)

    def _upload_deepstack(self, feats: List[torch.Tensor], vision_rows: torch.Tensor, Sp: int):
        """Dense [1, 1, Sp, 4096] additive tensors, zero everywhere except the vision rows."""
        out = {}
        for li, feat in enumerate(feats):
            dense = torch.zeros(1, 1, Sp, self.cfg.hidden, dtype=torch.bfloat16)
            dense[0, 0, vision_rows] = feat.to(torch.bfloat16)
            out[li] = ttnn.from_torch(
                dense,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return out

    # ----------------------------------------------------------------------------------------- api
    def encode(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
        taps: Optional[List[int]] = None,
        return_vision: bool = False,
        vision=None,
    ):
        """input_ids [1, L], pixel_values [n_patches, 1536], image_grid_thw [n_img, 3]
        -> hidden [L, 4096] bf16 (last decoder layer, before the final norm).

        `taps` returns a {layer_index: [L, 4096]} dict of layer outputs taken BEFORE that layer's
        deepstack add, which is what transformers' `output_hidden_states` records (its hooks sit on
        the decoder layer, and the add happens outside it).

        `vision` supplies `(merged_tokens, deepstack_features)` from an earlier `encode_image`, which
        skips the tower. Useful for re-encoding several prompts over the same condition image, and for
        measuring the decoder against a reference without the tower's error in the way.
        """
        ids = input_ids.reshape(1, -1)
        merged, feats = self.encode_image(pixel_values, image_grid_thw) if vision is None else vision
        emb = self.embed_with_vision(ids, merged)
        L = emb.shape[0]
        Sp = (L + TILE - 1) // TILE * TILE

        tt = mm_token_type_ids(ids) if type_ids is None else type_ids.reshape(1, -1)
        position_ids = mrope_position_ids(
            tt, image_grid_thw=image_grid_thw.reshape(-1, 3), spatial_merge_size=self.vcfg.spatial_merge_size
        )
        cos_sin = self._upload_rope(position_ids, Sp)
        rows = (tt[0] != 0).nonzero().reshape(-1)
        post_add = self._upload_deepstack(feats, rows, Sp)

        h, L, out_taps = self.te.forward_device(emb, taps=taps, cos_sin=cos_sin, post_add=post_add)
        out = ttnn.to_torch(h)[0, 0, :L]
        ttnn.deallocate(h)
        for t in cos_sin:
            ttnn.deallocate(t)
        for t in post_add.values():
            ttnn.deallocate(t)
        extra = {}
        if taps is not None:
            extra["taps"] = out_taps
        if return_vision:
            extra["merged"] = merged
            extra["deepstack"] = feats
            extra["position_ids"] = position_ids
        return (out, extra) if extra else out
