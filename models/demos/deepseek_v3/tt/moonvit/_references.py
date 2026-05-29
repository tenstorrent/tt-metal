# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
HF reference module extraction.

Pulls torch submodules out of the loaded K2.6 vision model (the vision-only
stand-in from `_k26_loader`) so per-submodule PCC tests have a known-good
torch comparison target.

Attribute paths are resolved at first use against the actual model graph
(K2.6's classes live in the downloaded checkpoint, not the installed
transformers package). If a path is wrong, the helpers below raise a clear
error pointing at the expected location.

The attribute paths below match modeling_kimi_k25.py at the time of writing
(May 2026); update if the released model is re-laid-out.
"""
from __future__ import annotations

import copy
from typing import Any

import torch

# --- 2D <-> 3D adapters -------------------------------------------------------
#
# K2.6's MoonViT is the 3D (video) variant. For image inputs (T=1) it is
# numerically identical to the 2D MoonViT this module targets, but three
# call-signature differences leak through to the reference modules:
#
#   1. Grids are (N,3) [t,h,w], not (N,2) [h,w].
#   2. `Rope2DPosEmbRepeated.get_freqs_cis(grid_thws, device)` takes a
#      device arg a plain 2D rope wouldn't.
#   3. `attention_qkvpacked` / encoder-layer `forward` take an extra
#      positional `max_seqlen` argument.
#
# The PCC tests and TT `from_torch` builders speak the 2D interface (2D
# grids, no max_seqlen). Rather than fork every test, we bridge here — the
# reference layer is exactly the right place, since its whole job is to
# present a known-good comparison target.


def _promote_grid(grid):
    """Promote a 2D (N,2) [h,w] grid to K2.6's 3D (N,3) [t=1,h,w].

    An image is the T=1 case of a video, so prepending a unit temporal
    dimension is exact. (N,3) inputs pass through unchanged.
    """
    g = grid if torch.is_tensor(grid) else torch.tensor(grid, dtype=torch.long)
    if g.dim() == 2 and g.shape[-1] == 2:
        t = torch.ones((g.shape[0], 1), dtype=g.dtype, device=g.device)
        g = torch.cat([t, g], dim=1)
    return g


class _DelegatingModule(torch.nn.Module):
    """Base for adapters that hold one inner module and delegate attrs to it.

    `.float()` / `.to()` cast the inner module (it's a registered child) and
    return the adapter. Attribute access that misses on the adapter falls
    through to the inner module, so `from_torch` builders that read
    `.weight`, `.proj`, `.wqkv`, etc. work transparently.
    """

    _INNER = "inner"

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            inner = modules.get(self._INNER)
            if inner is not None and name != self._INNER:
                return getattr(inner, name)
            raise


class _GridForwardRef(_DelegatingModule):
    """Wrap a K2.6 module whose `forward(x, grid_thws)` wants (N,3) grids.

    Presents a 2D `forward(x, grid_hws)` signature accepting (N,2).
    Covers MoonVision3dPatchEmbed, Learnable2DInterpPosEmbDivided_fixed,
    and the full MoonViT3dPretrainedModel tower.
    """

    def forward(self, x, grid, *args, **kwargs):
        return self.inner(x, _promote_grid(grid), *args, **kwargs)


class _RopeRef:
    """Present K2.6 `Rope2DPosEmbRepeated` via a plain 2D-rope interface.

    Not an nn.Module — the rope holds only buffers and the tests/`Rope2DSetup`
    read plain scalars off it. Exposes `.dim/.max_height/.max_width/
    .theta_base` (for `Rope2DSetup.from_torch`) and a `get_freqs_cis(grid_hws)`
    that accepts (N,2) grids and supplies the device arg.
    """

    def __init__(self, ref):
        self._ref = ref
        self.dim = int(ref.dim)
        self.max_height = int(ref.max_height)
        self.max_width = int(ref.max_width)
        self.theta_base = float(ref.theta_base)

    def get_freqs_cis(self, grid_hws):
        g = _promote_grid(grid_hws)
        return self._ref.get_freqs_cis(g, device=torch.device("cpu"))


class _EncoderLayerRef(_DelegatingModule):
    """Present a K2.6 `MoonViTEncoderLayer` via the 2D interface.

    K2.6's `attention_qkvpacked(x, cu_seqlens, max_seqlen, rope_freqs_cis)`
    and `forward(x, cu_seqlens, max_seqlen, rope_freqs_cis)` both take a
    `max_seqlen` the 2D callers don't pass — we derive it from cu_seqlens.
    Also forces the layer's attention to the requested implementation
    (default `sdpa`, registered by `_k26_loader._register_sdpa_attention`,
    which masks multi-image correctly unlike K2.6's bundled `eager`).
    """

    _INNER = "layer"

    def __init__(self, layer):
        super(_DelegatingModule, self).__init__()
        self.layer = layer
        # Default to the correctly-masked sdpa; tests may override.
        self.attn_implementation = "sdpa"

    @staticmethod
    def _max_seqlen(cu_seqlens):
        cu = cu_seqlens if torch.is_tensor(cu_seqlens) else torch.tensor(cu_seqlens)
        return int((cu[1:] - cu[:-1]).max().item())

    def attention_qkvpacked(self, x, cu_seqlens, rope_freqs_cis=None):
        self.layer.attn_implementation = self.attn_implementation
        return self.layer.attention_qkvpacked(
            x, cu_seqlens, self._max_seqlen(cu_seqlens), rope_freqs_cis=rope_freqs_cis
        )

    def forward(self, x, cu_seqlens, rope_freqs_cis=None):
        self.layer.attn_implementation = self.attn_implementation
        return self.layer(x, cu_seqlens, self._max_seqlen(cu_seqlens), rope_freqs_cis=rope_freqs_cis)


class _ProjectorRef(_DelegatingModule):
    """Present K2.6 `PatchMergerMLP` via a `.pre_norm/.linear_1/.linear_2` interface.

    K2.6 uses a Sequential `proj` (proj.0 Linear / proj.1 GELU / proj.2 Linear)
    and `pre_norm`, and its forward returns a per-image list. The PCC test and
    `MoonViTProjector.from_torch` expect `.pre_norm`, `.linear_1`, `.linear_2`
    and a forward(list)->concatenated tensor. This bridges both.
    """

    def __init__(self, pm):
        super().__init__(pm)
        seq = pm.proj
        # Expose the Sequential entries under linear_1/linear_2 names.
        self.pre_norm = pm.pre_norm
        self.linear_1 = seq[0]
        self.act = seq[1]
        self.linear_2 = seq[2]

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        outs = []
        for item in x:
            h = self.pre_norm(item).reshape(item.shape[0], -1)
            h = self.linear_1(h)
            h = self.act(h)
            h = self.linear_2(h)
            outs.append(h)
        return torch.cat(outs, dim=0)


# --- Attribute-path helpers ---------------------------------------------------


def _resolve(root, dotted: str):
    """Walk a dotted attribute path. Raise a useful error if missing."""
    cur = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                f"HF model has no attribute `{dotted}` "
                f"(failed at `{part}` on {type(cur).__name__}). "
                "If MoonViT's HF layout has changed, update "
                "models/demos/deepseek_v3/tt/moonvit/_references.py."
            )
        cur = getattr(cur, part)
    return cur


def _vision_tower(args):
    """The K2.6 MoonViT vision tower, wrapped in `_GridForwardRef`.

    The wrapper lets callers pass (N,2) [h,w] grids (promoted to the (N,3)
    [t=1,h,w] form K2.6 expects); attribute resolution through it is
    transparent.
    """
    return _GridForwardRef(_resolve(args.hf_model, "vision_tower"))


def _projector(args):
    """The K2.6 multimodal projector (PatchMergerMLP)."""
    return _resolve(args.hf_model, "mm_projector")


def _encoder_layer(args, layer_num: int):
    """One MoonViTEncoderLayer from the encoder's block list.

    Attribute name for the block list isn't fixed across HF revisions —
    try a few candidates.
    """
    enc = _resolve(_vision_tower(args), "encoder")
    for path in ("blocks", "layers", "encoder_layers"):
        if hasattr(enc, path):
            blocks = getattr(enc, path)
            if layer_num >= len(blocks):
                raise IndexError(
                    f"Requested encoder layer {layer_num} but encoder.{path} " f"has only {len(blocks)} entries."
                )
            return blocks[layer_num]
    raise AttributeError(
        f"Could not find a block list on encoder ({type(enc).__name__}). " "Tried: blocks, layers, encoder_layers."
    )


def _clone(module):
    """Return an eval-mode deep copy on CPU for use as a PCC reference."""
    out = copy.deepcopy(module).cpu().eval()
    for p in out.parameters():
        p.requires_grad_(False)
    return out


# --- Public reference factories ----------------------------------------------


def reference_layernorm(args, layer_num: int = 0, which: str = "norm0") -> Any:
    """Return one LayerNorm submodule from an encoder layer.

    Args:
        layer_num: index into the encoder block list.
        which: "norm0" (pre-attn) or "norm1" (pre-MLP) — matches the HF
            attribute names in MoonViTEncoderLayer.
    """
    layer = _encoder_layer(args, layer_num)
    if not hasattr(layer, which):
        raise AttributeError(
            f"Encoder layer {layer_num} has no `{which}` " f"(found: {[n for n, _ in layer.named_children()]})."
        )
    return _clone(getattr(layer, which))


def reference_mlp(args, layer_num: int = 0) -> Any:
    """Return one MLP2 submodule."""
    layer = _encoder_layer(args, layer_num)
    if hasattr(layer, "mlp"):
        return _clone(layer.mlp)
    raise AttributeError(
        f"Encoder layer {layer_num} has no `mlp` " f"(found: {[n for n, _ in layer.named_children()]})."
    )


def reference_attention(args, layer_num: int = 0) -> Any:
    """Return one encoder layer wrapped in `_EncoderLayerRef`.

    MoonViT's attention isn't a clean nn.Module — it's a method on the
    layer that takes (x, cu_seqlens, rope_freqs_cis). The test driver
    invokes that method directly. `_EncoderLayerRef` injects the K2.6
    `max_seqlen` arg and forces correct sdpa masking; cloning the layer
    gives access to wqkv / wo / norms for the TT `from_torch` builders.
    """
    return _EncoderLayerRef(_clone(_encoder_layer(args, layer_num)))


def reference_block(args, layer_num: int = 0) -> Any:
    """Return one MoonViTEncoderLayer (full block including both norms)."""
    return _EncoderLayerRef(_clone(_encoder_layer(args, layer_num)))


def reference_patch_embed(args) -> Any:
    """Return MoonVision3dPatchEmbed (Conv2d + learned posemb), grid-adapted."""
    return _GridForwardRef(_clone(_resolve(_vision_tower(args), "patch_embed")))


def reference_pos_emb(args) -> Any:
    """Return the Learnable2DInterpPosEmbDivided_fixed instance, grid-adapted.

    Lives inside MoonVision3dPatchEmbed.pos_emb.
    """
    return _GridForwardRef(_clone(_resolve(_vision_tower(args), "patch_embed.pos_emb")))


def reference_patch_merger(args) -> Any:
    """K2.6's `tpool_patch_merger` free function, adapted to (N,2) grids.

    Wrapped so it accepts 2D (N,2) grids (T=1 prepended). For T=1
    the temporal pool is identity, so it reduces to spatial 2x2 merging.
    """
    # Ensure the modeling module is imported/registered.
    args.hf_model

    import sys

    for name, mod in sys.modules.items():
        if name.endswith("modeling_kimi_k25") and hasattr(mod, "tpool_patch_merger"):
            tpool = mod.tpool_patch_merger

            def patch_merger(x, grid_hws, merge_kernel_size=(2, 2)):
                return tpool(x, _promote_grid(grid_hws), merge_kernel_size=merge_kernel_size)

            return patch_merger
    raise RuntimeError(
        "Could not locate tpool_patch_merger in loaded K2.6 modules. "
        "Searched sys.modules for a *modeling_kimi_k25 module exposing tpool_patch_merger."
    )


def reference_projector(args) -> Any:
    """The multimodal projector (K2.6 PatchMergerMLP), 2D-interface adapted."""
    return _ProjectorRef(_clone(_projector(args)))


def reference_vision_tower(args) -> Any:
    """Full MoonViT vision tower — patch_embed + encoder, no projector."""
    return _clone(_vision_tower(args))


def reference_rope_2d(args) -> Any:
    """Rope2DPosEmbRepeated module (vision_tower.encoder.rope_2d), grid-adapted."""
    return _RopeRef(_clone(_resolve(_vision_tower(args), "encoder.rope_2d")))


def reference_final_layernorm(args) -> Any:
    """The final LayerNorm at the end of the MoonViT encoder.

    Applied after the last encoder block, before the patch merger:
        hidden_states = self.final_layernorm(hidden_states)
    """
    return _clone(_resolve(_vision_tower(args), "encoder.final_layernorm"))
