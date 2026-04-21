"""ttnn port of VGGT on a single Tenstorrent p150a Blackhole chip.

Strategy: keep the torch reference structure intact and monkey-patch
specific sub-modules to route their compute through ttnn. Each port lifts
one op class off CPU; weights are uploaded once at install time and then
re-used for every subsequent forward call.

Ports applied:
  - Every Mlp (72 instances: 24 DINOv2 patch_embed + 24 frame + 24 global)
    runs fc1 + gelu + fc2 on device in bf16 LoFi.
  - Every Attention (72 instances) runs qkv + scores + softmax + context
    + proj on device. qkv is bf16 LoFi; attention scores / softmax are
    held in fp32 with HiFi4 so the 1374-long softmax row doesn't collapse
    world_points_conf; proj is bf16 HiFi4 + fp32 dest for the same reason.
  - Block.norm1 is fused into the attention's device call (upload x,
    LN + qkv on device, download qkv). Block.norm1 becomes Identity on host.

Ports NOT applied (tested and discarded — see results.tsv):
  - Block.norm1/norm2 as standalone device LN: LN compute < host round-trip.
  - Full sdpa on device (LoFi or HiFi4 fused kernel): PCC collapse.
  - All nn.Linear generically: head Linears are precision-sensitive and
    overhead-bound; ported only the ones inside Attention/Mlp.
  - bf16 autocast over aggregator: head confidence channels need fp32.
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


_HIFI_KCONFIG = None


def _hifi_kconfig(device):
    """HiFi4 compute kernel with fp32 dest accumulation for precision-
    sensitive matmuls (e.g., attn.proj feeding the depth/world_points_conf
    heads). Cached per process."""
    global _HIFI_KCONFIG
    if _HIFI_KCONFIG is None:
        import ttnn
        _HIFI_KCONFIG = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return _HIFI_KCONFIG


def _install_ttnn_attention(model, device):
    """Route qkv (LoFi) and proj (HiFi4) through ttnn.

    proj uses HiFi4 + fp32 dest so the downstream world_points_conf head
    stays above the 0.99 PCC floor (bf16 LoFi was 0.988 — FAIL).

    Also fuse Block.norm1 into the attention's device path: upload x_bf16,
    LN on device, qkv matmul on device, download qkv. Saves one CPU LN
    call per block and makes the norm1 Identity on the host side.
    """
    import ttnn
    import torch.nn.functional as F
    import torch.nn as nn
    from vggt.layers.attention import Attention  # type: ignore
    from vggt.layers.block import Block  # type: ignore

    attn_owner_block: dict = {}
    for blk in model.modules():
        if isinstance(blk, Block):
            attn_owner_block[id(blk.attn)] = blk

    for m in model.modules():
        if isinstance(m, Attention) and not getattr(m, "_tt_attn_ready", False):
            qw = m.qkv.weight.detach().t().contiguous().to(torch.bfloat16)
            qb = m.qkv.bias.detach().reshape(1, 1, -1).to(torch.bfloat16) if m.qkv.bias is not None else None
            pw = m.proj.weight.detach().t().contiguous().to(torch.bfloat16)
            pb = m.proj.bias.detach().reshape(1, 1, -1).to(torch.bfloat16) if m.proj.bias is not None else None
            m._tt_qkv_w = ttnn.from_torch(qw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            m._tt_qkv_b = (
                ttnn.from_torch(qb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                if qb is not None else None
            )
            m._tt_proj_w = ttnn.from_torch(pw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            m._tt_proj_b = (
                ttnn.from_torch(pb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                if pb is not None else None
            )
            m._tt_device = device
            # Fuse Block.norm1 if owned by a Block.
            blk = attn_owner_block.get(id(m))
            if blk is not None and isinstance(blk.norm1, nn.LayerNorm):
                g = blk.norm1.weight.detach().reshape(1, 1, -1).to(torch.bfloat16)
                b = blk.norm1.bias.detach().reshape(1, 1, -1).to(torch.bfloat16)
                m._tt_ln_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                m._tt_ln_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                m._tt_ln_eps = float(blk.norm1.eps)
                blk.norm1 = nn.Identity()
                m._tt_fused_ln = True
            else:
                m._tt_fused_ln = False
            m._tt_attn_ready = True

    if getattr(Attention, "_tt_patched", False):
        return

    import math

    def ttnn_attn_forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        if not getattr(self, "_tt_attn_ready", False):
            return self._orig_forward(x, pos=pos)
        B, N, C = x.shape
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        tt_in = ttnn.from_torch(
            x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self._tt_device
        )
        if self._tt_fused_ln:
            tt_in = ttnn.layer_norm(tt_in, weight=self._tt_ln_g, bias=self._tt_ln_b, epsilon=self._tt_ln_eps)
        tt_qkv = ttnn.linear(tt_in, self._tt_qkv_w, bias=self._tt_qkv_b)
        qkv = ttnn.to_torch(tt_qkv)  # bf16
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        # Attention on device: bf16 uploads (small) but hold the softmax +
        # scores in fp32 via HiFi4 kernels and dtype=float32. bf16 softmax
        # accumulates enough error over a 1374-long row to collapse
        # world_points_conf; fp32 intermediate restores PCC.
        dh = self.head_dim
        dev = self._tt_device
        kcfg = _hifi_kconfig(dev)
        tt_q = ttnn.from_torch(q.contiguous(),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tt_kt = ttnn.from_torch(k.transpose(-2, -1).contiguous(),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tt_v = ttnn.from_torch(v.contiguous(),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tt_scores = ttnn.matmul(tt_q, tt_kt, compute_kernel_config=kcfg, dtype=ttnn.float32)
        tt_scores = ttnn.multiply(tt_scores, 1.0 / math.sqrt(dh))
        tt_attn = ttnn.softmax(tt_scores, dim=-1, compute_kernel_config=kcfg)
        tt_ctx = ttnn.matmul(tt_attn, tt_v, compute_kernel_config=kcfg, dtype=ttnn.float32)
        x = ttnn.to_torch(tt_ctx).to(v.dtype)
        x = x.transpose(1, 2).reshape(B, N, C)
        # proj with HiFi4 + fp32 dest to preserve the confidence-head PCC.
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        tt_in = ttnn.from_torch(
            x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        tt_out = ttnn.linear(
            tt_in, self._tt_proj_w, bias=self._tt_proj_b,
            compute_kernel_config=kcfg,
        )
        return ttnn.to_torch(tt_out).to(x.dtype)

    Attention._orig_forward = Attention.forward
    Attention.forward = ttnn_attn_forward
    Attention._tt_patched = True


def _ensure_installed(device):
    if _INSTALL_DONE.get(id(device)):
        return
    model = _get_model()
    _install_ttnn_mlp(model, device)
    _install_ttnn_attention(model, device)
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
