"""Minimal pure-torch DUSt3R reference for PCC validation.

Implements the subset of naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt needed
to reproduce forward passes layer-by-layer.

Architecture (from config.json):
  enc: ViT-L, 24 blocks, dim=1024, heads=16, RoPE100
  dec: 12 blocks, dim=768, heads=12, self-attn + cross-attn, RoPE100
  dual decoder branches (dec_blocks / dec_blocks2) for two views
  head: DPT with act_postprocess + refinement
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


CKPT_DIR = Path(
    "/home/ttuser/.cache/huggingface/hub/"
    "models--naver--DUSt3R_ViTLarge_BaseDecoder_512_dpt/snapshots"
)


def load_checkpoint():
    snap = next(CKPT_DIR.iterdir())
    return load_file(str(snap / "model.safetensors"))


# ---------- RoPE (100 base, 2D) ----------

class RoPE2D:
    """2D rotary positional embedding (base=100) as in DUSt3R."""

    def __init__(self, base: float = 100.0):
        self.base = base
        self.cache: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_cos_sin(self, D: int, seq_len: int, device, dtype):
        key = (D, seq_len, device, dtype)
        if key not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cache[key] = (emb.cos().to(dtype), emb.sin().to(dtype))
        return self.cache[key]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        # tokens: (B, H, N, Dhead)  pos1d: (B, N)
        cos = cos[pos1d]  # (B, N, D)
        sin = sin[pos1d]
        cos = cos[:, None, :, :]  # (B, 1, N, D)
        sin = sin[:, None, :, :]
        return tokens * cos + self.rotate_half(tokens) * sin

    def __call__(self, tokens, positions):
        # tokens: (B, heads, N, D)
        # positions: (B, N, 2) -> y,x
        assert tokens.shape[-1] % 2 == 0
        D = tokens.shape[-1] // 2
        y = tokens[..., :D]
        x = tokens[..., D:]
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        y = self.apply_rope1d(y, positions[..., 0], cos, sin)
        x = self.apply_rope1d(x, positions[..., 1], cos, sin)
        return torch.cat((y, x), dim=-1)


ROPE = RoPE2D(base=100.0)


def make_positions(B: int, H: int, W: int, device) -> torch.Tensor:
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack((grid_y, grid_x), dim=-1).reshape(H * W, 2)
    return pos.unsqueeze(0).expand(B, -1, -1).contiguous()


# ---------- Patch embed ----------

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 1024, patch: int = 16, in_ch: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.patch = patch

    def forward(self, img: torch.Tensor):
        x = self.proj(img)  # (B, E, H/p, W/p)
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, E)
        pos = make_positions(B, H, W, img.device)
        return x, pos, (H, W)


# ---------- Encoder block (self-attn + MLP, with RoPE) ----------

class EncoderAttn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, pos):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, N, dh)
        q = ROPE(q, pos)
        k = ROPE(k, pos)
        attn = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 1024, heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EncoderAttn(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, depth: int = 24, dim: int = 1024, heads: int = 16):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, img):
        x, pos, (H, W) = self.patch_embed(img)
        for blk in self.blocks:
            x = blk(x, pos)
        x = self.norm(x)
        return x, pos, (H, W)


# ---------- Decoder block (self + cross attention) ----------

class DecoderSelfAttn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, pos):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = ROPE(q, pos)
        k = ROPE(k, pos)
        attn = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class DecoderCrossAttn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.projq = nn.Linear(dim, dim, bias=True)
        self.projk = nn.Linear(dim, dim, bias=True)
        self.projv = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, y, pos_q, pos_kv):
        B, N, D = x.shape
        _, M, _ = y.shape
        q = self.projq(x).reshape(B, N, self.h, self.dh).transpose(1, 2)
        k = self.projk(y).reshape(B, M, self.h, self.dh).transpose(1, 2)
        v = self.projv(y).reshape(B, M, self.h, self.dh).transpose(1, 2)
        q = ROPE(q, pos_q)
        k = ROPE(k, pos_kv)
        attn = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DecoderSelfAttn(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = DecoderCrossAttn(dim, heads)
        self.norm_y = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, y, pos_x, pos_y):
        x = x + self.attn(self.norm1(x), pos_x)
        y_normed = self.norm_y(y)
        x = x + self.cross_attn(self.norm2(x), y_normed, pos_x, pos_y)
        x = x + self.mlp(self.norm3(x))
        return x


# ---------- DPT head ----------

class ResConvUnit(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.resConfUnit1 = ResConvUnit(ch)
        self.resConfUnit2 = ResConvUnit(ch)
        self.out_conv = nn.Conv2d(ch, ch, 1, bias=True)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        x = self.out_conv(x)
        return x


class DPTHead(nn.Module):
    """Full DPT head matching downstream_head{1,2} checkpoint layout."""

    def __init__(self, enc_dim: int = 1024, dec_dim: int = 768, patch: int = 16):
        super().__init__()
        self.patch = patch
        # act_postprocess: 4 stages. Each stage: 1x1 projection, then resample.
        self.ap0_proj = nn.Conv2d(enc_dim, 96, 1, bias=True)
        self.ap0_up = nn.ConvTranspose2d(96, 96, 4, stride=4, bias=True)
        self.ap1_proj = nn.Conv2d(dec_dim, 192, 1, bias=True)
        self.ap1_up = nn.ConvTranspose2d(192, 192, 2, stride=2, bias=True)
        self.ap2_proj = nn.Conv2d(dec_dim, 384, 1, bias=True)
        self.ap3_proj = nn.Conv2d(dec_dim, 768, 1, bias=True)
        self.ap3_down = nn.Conv2d(768, 768, 3, stride=2, padding=1, bias=True)

        # scratch: 3x3 no-bias convs from each scale to 256 channels.
        self.layer1_rn = nn.Conv2d(96, 256, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(192, 256, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(384, 256, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(768, 256, 3, padding=1, bias=False)

        self.refinenet4 = FeatureFusionBlock(256)
        self.refinenet3 = FeatureFusionBlock(256)
        self.refinenet2 = FeatureFusionBlock(256)
        self.refinenet1 = FeatureFusionBlock(256)

        self.head0 = nn.Conv2d(256, 128, 3, padding=1, bias=True)
        self.head2 = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        self.head4 = nn.Conv2d(128, 4, 1, bias=True)

    def _tokens_to_2d(self, feat, hw):
        H, W = hw
        B, N, D = feat.shape
        return feat.transpose(1, 2).reshape(B, D, H, W).contiguous()

    def forward(self, feats_list, hw):
        # feats_list: 4 token tensors (B, N, D). [0] is enc (1024), [1..3] are decoder at 3 depths (768).
        H, W = hw
        f0 = self._tokens_to_2d(feats_list[0], hw)
        f1 = self._tokens_to_2d(feats_list[1], hw)
        f2 = self._tokens_to_2d(feats_list[2], hw)
        f3 = self._tokens_to_2d(feats_list[3], hw)

        l1 = self.ap0_up(self.ap0_proj(f0))      # (B, 96,  4H, 4W)
        l2 = self.ap1_up(self.ap1_proj(f1))      # (B, 192, 2H, 2W)
        l3 = self.ap2_proj(f2)                    # (B, 384, H,  W)
        l4 = self.ap3_down(self.ap3_proj(f3))    # (B, 768, H/2,W/2)

        l1 = self.layer1_rn(l1)
        l2 = self.layer2_rn(l2)
        l3 = self.layer3_rn(l3)
        l4 = self.layer4_rn(l4)

        p4 = self.refinenet4(l4)
        p3 = self.refinenet3(p4, l3)
        p2 = self.refinenet2(p3, l2)
        p1 = self.refinenet1(p2, l1)

        x = self.head0(p1)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        x = self.head2(x)
        x = F.relu(x)
        x = self.head4(x)
        return x  # (B, 4, H_img, W_img)


def load_dpt_head(state: dict, branch: int = 1) -> DPTHead:
    h = DPTHead()
    prefix = f"downstream_head{branch}.dpt."
    sub = _prefix(state, prefix)
    # Remap flat state into our attribute names.
    mapping = {
        "act_postprocess.0.0.weight": "ap0_proj.weight",
        "act_postprocess.0.0.bias": "ap0_proj.bias",
        "act_postprocess.0.1.weight": "ap0_up.weight",
        "act_postprocess.0.1.bias": "ap0_up.bias",
        "act_postprocess.1.0.weight": "ap1_proj.weight",
        "act_postprocess.1.0.bias": "ap1_proj.bias",
        "act_postprocess.1.1.weight": "ap1_up.weight",
        "act_postprocess.1.1.bias": "ap1_up.bias",
        "act_postprocess.2.0.weight": "ap2_proj.weight",
        "act_postprocess.2.0.bias": "ap2_proj.bias",
        "act_postprocess.3.0.weight": "ap3_proj.weight",
        "act_postprocess.3.0.bias": "ap3_proj.bias",
        "act_postprocess.3.1.weight": "ap3_down.weight",
        "act_postprocess.3.1.bias": "ap3_down.bias",
        "scratch.layer1_rn.weight": "layer1_rn.weight",
        "scratch.layer2_rn.weight": "layer2_rn.weight",
        "scratch.layer3_rn.weight": "layer3_rn.weight",
        "scratch.layer4_rn.weight": "layer4_rn.weight",
        "head.0.weight": "head0.weight",
        "head.0.bias": "head0.bias",
        "head.2.weight": "head2.weight",
        "head.2.bias": "head2.bias",
        "head.4.weight": "head4.weight",
        "head.4.bias": "head4.bias",
    }
    # refinenets
    for r in (1, 2, 3, 4):
        for cu in (1, 2):
            mapping[f"scratch.refinenet{r}.resConfUnit{cu}.conv1.weight"] = f"refinenet{r}.resConfUnit{cu}.conv1.weight"
            mapping[f"scratch.refinenet{r}.resConfUnit{cu}.conv1.bias"] = f"refinenet{r}.resConfUnit{cu}.conv1.bias"
            mapping[f"scratch.refinenet{r}.resConfUnit{cu}.conv2.weight"] = f"refinenet{r}.resConfUnit{cu}.conv2.weight"
            mapping[f"scratch.refinenet{r}.resConfUnit{cu}.conv2.bias"] = f"refinenet{r}.resConfUnit{cu}.conv2.bias"
        mapping[f"scratch.refinenet{r}.out_conv.weight"] = f"refinenet{r}.out_conv.weight"
        mapping[f"scratch.refinenet{r}.out_conv.bias"] = f"refinenet{r}.out_conv.bias"

    remapped = {}
    for src, dst in mapping.items():
        if src in sub:
            remapped[dst] = sub[src]
    missing, unexpected = h.load_state_dict(remapped, strict=False)
    assert not missing, f"DPT head missing keys: {missing[:5]}..."
    return h.eval()


# ---------- Weight loading helpers ----------

def _prefix(state: dict, prefix: str) -> dict:
    out = {}
    plen = len(prefix)
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
    return out


def load_encoder_block(state: dict, idx: int) -> EncoderBlock:
    blk = EncoderBlock()
    sub = _prefix(state, f"enc_blocks.{idx}.")
    missing, unexpected = blk.load_state_dict(sub, strict=False)
    assert not missing, f"missing keys in enc_blocks.{idx}: {missing}"
    return blk.eval()


def load_patch_embed(state: dict) -> PatchEmbed:
    pe = PatchEmbed()
    pe.load_state_dict(_prefix(state, "patch_embed."), strict=True)
    return pe.eval()


def load_encoder(state: dict) -> Encoder:
    enc = Encoder()
    enc.patch_embed.load_state_dict(_prefix(state, "patch_embed."), strict=True)
    for i, blk in enumerate(enc.blocks):
        blk.load_state_dict(_prefix(state, f"enc_blocks.{i}."), strict=True)
    enc.norm.load_state_dict(_prefix(state, "enc_norm."), strict=True)
    return enc.eval()


def load_decoder_block(state: dict, idx: int, branch: int = 1) -> DecoderBlock:
    blk = DecoderBlock()
    prefix = "dec_blocks." if branch == 1 else "dec_blocks2."
    blk.load_state_dict(_prefix(state, f"{prefix}{idx}."), strict=True)
    return blk.eval()


class Decoder(nn.Module):
    """Dual-branch decoder: 12 blocks × 2 branches + final dec_norm."""

    def __init__(self, enc_dim: int = 1024, dec_dim: int = 768, depth: int = 12, heads: int = 12):
        super().__init__()
        self.embed = nn.Linear(enc_dim, dec_dim, bias=True)
        self.blocks1 = nn.ModuleList([DecoderBlock(dec_dim, heads) for _ in range(depth)])
        self.blocks2 = nn.ModuleList([DecoderBlock(dec_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dec_dim, eps=1e-6)

    def forward(self, feat1, feat2, pos):
        f1 = self.embed(feat1)
        f2 = self.embed(feat2)
        for b1, b2 in zip(self.blocks1, self.blocks2):
            new_f1 = b1(f1, f2, pos, pos)
            new_f2 = b2(f2, f1, pos, pos)
            f1, f2 = new_f1, new_f2
        return self.norm(f1), self.norm(f2)


def load_decoder(state: dict) -> Decoder:
    dec = Decoder()
    dec.embed.load_state_dict(_prefix(state, "decoder_embed."), strict=True)
    for i, blk in enumerate(dec.blocks1):
        blk.load_state_dict(_prefix(state, f"dec_blocks.{i}."), strict=True)
    for i, blk in enumerate(dec.blocks2):
        blk.load_state_dict(_prefix(state, f"dec_blocks2.{i}."), strict=True)
    dec.norm.load_state_dict(_prefix(state, "dec_norm."), strict=True)
    return dec.eval()
