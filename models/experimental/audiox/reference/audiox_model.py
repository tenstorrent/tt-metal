import torch
import torch.nn as nn
import math


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        return self.linear_2(self.act(self.linear_1(t)))


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(*t.shape[:-1], self.heads, -1).transpose(-3, -2), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-3, -2).reshape(*x.shape[:-1], -1)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context):
        q = self.to_q(x).view(*x.shape[:-1], self.heads, -1).transpose(-3, -2)
        k = self.to_k(context).view(*context.shape[:-1], self.heads, -1).transpose(-3, -2)
        v = self.to_v(context).view(*context.shape[:-1], self.heads, -1).transpose(-3, -2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-3, -2).reshape(*x.shape[:-1], -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, mult=4):
        super().__init__()
        hidden_dim = hidden_dim or dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, context_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head) if context_dim else None
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, context=None, timestep_emb=None):
        x = x + self.attn(self.norm1(x))
        if self.cross_attn and context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class MultimodalAdaptiveFusion(nn.Module):
    def __init__(self, dim, num_modalities=4):
        super().__init__()
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_modalities)
        ])
        self.fusion_gate = nn.Linear(dim * num_modalities, num_modalities)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, *modalities):
        projected = [proj(m) for proj, m in zip(self.modality_projections, modalities)]
        stacked = torch.stack(projected, dim=-1)
        gate_input = torch.cat(projected, dim=-1)
        gates = torch.softmax(self.fusion_gate(gate_input), dim=-1)
        fused = (stacked * gates.unsqueeze(-2)).sum(dim=-1)
        return self.output_proj(fused)


class AudioXDiffusionTransformer(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12, dim_head=64, context_dim=768):
        super().__init__()
        self.input_proj = nn.Linear(80, dim)
        self.time_embed = TimestepEmbedding(dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 4),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, context_dim)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, 80)

    def forward(self, x, timestep, context=None):
        x = self.input_proj(x)
        t = self.time_embed(timestep)
        t = self.time_mlp(t)
        for block in self.blocks:
            x = block(x, context)
        x = self.norm_out(x)
        return self.output_proj(x)


class AudioXModel(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12, context_dim=768, num_modalities=4):
        super().__init__()
        self.text_encoder = nn.Linear(512, context_dim)
        self.audio_encoder = nn.Linear(80, context_dim)
        self.video_encoder = nn.Linear(1024, context_dim)
        self.image_encoder = nn.Linear(1024, context_dim)
        self.fusion = MultimodalAdaptiveFusion(context_dim, num_modalities)
        self.diffusion = AudioXDiffusionTransformer(dim, depth, heads, context_dim=context_dim)
        self.vocoder = nn.Conv1d(80, 1, kernel_size=7, padding=3)

    def forward(self, audio_latents, timestep, text_embeds=None, video_embeds=None,
                image_embeds=None, audio_embeds=None):
        context = None
        modalities = []
        if text_embeds is not None:
            modalities.append(self.text_encoder(text_embeds))
        if video_embeds is not None:
            modalities.append(self.video_encoder(video_embeds))
        if image_embeds is not None:
            modalities.append(self.image_encoder(image_embeds))
        if audio_embeds is not None:
            modalities.append(self.audio_encoder(audio_embeds))

        if len(modalities) > 1:
            context = self.fusion(*modalities)
        elif len(modalities) == 1:
            context = modalities[0]

        latent = self.diffusion(audio_latents, timestep, context)
        return self.vocoder(latent.transpose(1, 2)).transpose(1, 2)

    def generate(self, text=None, video=None, image=None, audio=None,
                 num_steps=50, guidance_scale=3.0, length=160):
        batch_size = 1
        device = next(self.parameters()).device

        text_embeds = torch.randn(batch_size, 77, 512, device=device) if text else None
        video_embeds = torch.randn(batch_size, 8, 1024, device=device) if video else None
        image_embeds = torch.randn(batch_size, 1, 1024, device=device) if image else None
        audio_embeds = torch.randn(batch_size, length, 80, device=device) if audio else None

        x = torch.randn(batch_size, length, 80, device=device)
        for t in torch.linspace(1, 0, num_steps, device=device):
            t_emb = t.expand(batch_size, 1)
            noise_pred = self.forward(x, t_emb, text_embeds, video_embeds, image_embeds, audio_embeds)
            if guidance_scale > 1.0:
                uncond_pred = self.forward(x, t_emb)
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)
            x = x - 0.02 * noise_pred

        return self.vocoder(x.transpose(1, 2)).squeeze(1)
