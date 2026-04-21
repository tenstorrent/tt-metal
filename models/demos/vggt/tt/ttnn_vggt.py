"""ttnn port of VGGT, built up operator-by-operator.

Strategy: keep the torch reference structure intact and monkey-patch
specific sub-modules to route their compute through ttnn. Each port lifts
one op class off CPU; weights are uploaded once at install time and then
re-used for every subsequent forward call.

Stage 1 (this file): MLP blocks — all 72 Mlp instances in the reference
(24 DINOv2 patch_embed + 24 frame + 24 global aggregator blocks) route
their fc1 + gelu + fc2 through ttnn.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


_CACHED_MODEL = None
_INSTALL_DONE: dict = {}


def _get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from reference.torch_vggt import load_vggt
        _CACHED_MODEL = load_vggt(eval_mode=True)
    return _CACHED_MODEL


# ---------- ttnn MLP port ----------

def _install_ttnn_mlp(model, device):
    """Attach ttnn-resident weights to every Mlp and swap in a ttnn forward."""
    import ttnn
    from vggt.layers.mlp import Mlp  # type: ignore

    # Preload weights for each Mlp instance.
    for m in model.modules():
        if isinstance(m, Mlp) and not getattr(m, "_tt_ready", False):
            w1 = m.fc1.weight.detach().t().contiguous().to(torch.bfloat16)
            b1 = m.fc1.bias.detach().reshape(1, 1, -1).to(torch.bfloat16)
            w2 = m.fc2.weight.detach().t().contiguous().to(torch.bfloat16)
            b2 = m.fc2.bias.detach().reshape(1, 1, -1).to(torch.bfloat16)
            m._tt_w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            m._tt_b1 = ttnn.from_torch(b1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            m._tt_w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            m._tt_b2 = ttnn.from_torch(b2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            m._tt_device = device
            m._tt_ready = True

    # Patch Mlp.forward once.
    if getattr(Mlp, "_tt_patched", False):
        return

    def ttnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_tt_ready", False):
            # Unported instance (e.g., not pre-loaded) — fall back to torch.
            return self._orig_forward(x)
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        tt_in = ttnn.from_torch(
            x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self._tt_device
        )
        tt_mid = ttnn.linear(tt_in, self._tt_w1, bias=self._tt_b1)
        tt_mid = ttnn.gelu(tt_mid)
        tt_out = ttnn.linear(tt_mid, self._tt_w2, bias=self._tt_b2)
        out = ttnn.to_torch(tt_out).to(x.dtype)
        return out

    Mlp._orig_forward = Mlp.forward
    Mlp.forward = ttnn_forward
    Mlp._tt_patched = True


def _install_ttnn_attention_qkv(model, device):
    """Route only `self.qkv(x)` through ttnn. proj stays on CPU for precision."""
    import ttnn
    import torch.nn.functional as F
    from vggt.layers.attention import Attention  # type: ignore

    for m in model.modules():
        if isinstance(m, Attention) and not getattr(m, "_tt_attn_ready", False):
            w = m.qkv.weight.detach().t().contiguous().to(torch.bfloat16)
            b = m.qkv.bias.detach().reshape(1, 1, -1).to(torch.bfloat16) if m.qkv.bias is not None else None
            m._tt_qkv_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            m._tt_qkv_b = (
                ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                if b is not None else None
            )
            m._tt_device = device
            m._tt_attn_ready = True

    if getattr(Attention, "_tt_patched", False):
        return

    def ttnn_attn_forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        if not getattr(self, "_tt_attn_ready", False):
            return self._orig_forward(x, pos=pos)
        B, N, C = x.shape
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        tt_in = ttnn.from_torch(
            x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self._tt_device
        )
        tt_qkv = ttnn.linear(tt_in, self._tt_qkv_w, bias=self._tt_qkv_b)
        qkv = ttnn.to_torch(tt_qkv).to(x.dtype)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

    Attention._orig_forward = Attention.forward
    Attention.forward = ttnn_attn_forward
    Attention._tt_patched = True


def _ensure_installed(device):
    if _INSTALL_DONE.get(id(device)):
        return
    model = _get_model()
    _install_ttnn_mlp(model, device)
    _install_ttnn_attention_qkv(model, device)
    _INSTALL_DONE[id(device)] = True


def vggt_forward(images: torch.Tensor, device: Any = None,
                 query_points: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Run VGGT end-to-end. MLP blocks run on ttnn; everything else is torch."""
    if device is None:
        raise RuntimeError("ttnn device handle required")
    _ensure_installed(device)
    model = _get_model()
    with torch.no_grad():
        return model(images, query_points=query_points)
