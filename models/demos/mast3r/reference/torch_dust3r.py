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

class DPTHead(nn.Module):
    """Simplified DPT head matching checkpoint structure.

    act_postprocess: 4 stages that reshape encoder tokens back to 2D and
      project to different channels (96, 192, 384, 768).
    Refinement + head convs produce the final 4-channel (pts3d + conf) map.
    """

    def __init__(self, enc_dim: int = 1024, dec_dim: int = 768):
        super().__init__()
        # Stage 0 uses encoder features (1024), stages 1-3 use decoder (768).
        self.reassemble = nn.ModuleList([
            nn.Sequential(nn.Conv2d(enc_dim, 96, 1), nn.ConvTranspose2d(96, 96, 4, 4)),
            nn.Sequential(nn.Conv2d(dec_dim, 192, 1), nn.ConvTranspose2d(192, 192, 2, 2)),
            nn.Sequential(nn.Conv2d(dec_dim, 384, 1)),
            nn.Sequential(nn.Conv2d(dec_dim, 768, 1), nn.Conv2d(768, 768, 3, padding=1)),
        ])
        # Refinement / fusion convs are the remaining keys in downstream_head1/2.
        # We keep a placeholder so tests can instantiate before full wiring.
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, 1),
        )


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
