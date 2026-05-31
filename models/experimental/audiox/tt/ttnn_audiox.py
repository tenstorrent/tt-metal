import math
from typing import Optional

import torch
import ttnn


def ttnn_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor)


def torch_to_ttnn(tensor: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, device=device)


def create_linear_weight(weight: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(weight, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def create_conv_weight(weight: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(weight, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TtnnTimestepEmbedding:
    def __init__(self, ref_module, device):
        self.device = device
        self.linear_1_weight = create_linear_weight(ref_module.linear_1.weight.data, device)
        self.linear_1_bias = create_linear_weight(ref_module.linear_1.bias.data, device)
        self.linear_2_weight = create_linear_weight(ref_module.linear_2.weight.data, device)
        self.linear_2_bias = create_linear_weight(ref_module.linear_2.bias.data, device)

    def __call__(self, t: ttnn.Tensor) -> ttnn.Tensor:
        h = ttnn.linear(t, self.linear_1_weight, bias=self.linear_1_bias)
        h = ttnn.silu(h)
        h = ttnn.linear(h, self.linear_2_weight, bias=self.linear_2_bias)
        return h


class TtnnAttention:
    def __init__(self, ref_module, device, heads=8):
        self.device = device
        self.heads = heads
        self.scale = 1.0 / math.sqrt(ref_module.scale ** -2)
        self.to_qkv_weight = create_linear_weight(ref_module.to_qkv.weight.data, device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return x  # placeholder: full attention requires ttnn.scaled_dot_product_attention


class TtnnFeedForward:
    def __init__(self, ref_module, device):
        self.device = device
        self.net_0_weight = create_linear_weight(ref_module.net[0].weight.data, device)
        self.net_0_bias = create_linear_weight(ref_module.net[0].bias.data, device)
        self.net_2_weight = create_linear_weight(ref_module.net[2].weight.data, device)
        self.net_2_bias = create_linear_weight(ref_module.net[2].bias.data, device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = ttnn.linear(x, self.net_0_weight, bias=self.net_0_bias)
        h = ttnn.gelu(h)
        h = ttnn.linear(h, self.net_2_weight, bias=self.net_2_bias)
        return h


class TtnnTransformerBlock:
    def __init__(self, ref_block, device, heads=8, context_dim=None):
        self.device = device
        self.norm1_weight = create_linear_weight(ref_block.norm1.weight.data, device)
        self.norm1_bias = create_linear_weight(ref_block.norm1.bias.data, device)
        self.attn = TtnnAttention(ref_block.attn, device, heads)
        self.ff = TtnnFeedForward(ref_block.ff, device)

    def __call__(self, x: ttnn.Tensor, context: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        residual = x
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)
        x = self.attn(x)
        x = ttnn.add(x, residual)
        residual = x
        x = self.ff(x)
        x = ttnn.add(x, residual)
        return x


class TtnnMultimodalAdaptiveFusion:
    def __init__(self, ref_module, device, num_modalities=4):
        self.device = device
        self.projections = [
            create_linear_weight(ref_module.modality_projections[i].weight.data, device)
            for i in range(num_modalities)
        ]
        self.fusion_gate_weight = create_linear_weight(ref_module.fusion_gate.weight.data, device)
        self.fusion_gate_bias = create_linear_weight(ref_module.fusion_gate.bias.data, device)
        self.output_weight = create_linear_weight(ref_module.output_proj.weight.data, device)
        self.output_bias = create_linear_weight(ref_module.output_proj.bias.data, device)

    def __call__(self, *modalities: ttnn.Tensor) -> ttnn.Tensor:
        fused = modalities[0]
        for m in modalities[1:]:
            fused = ttnn.add(fused, m)
        fused = fused / len(modalities)
        out = ttnn.linear(fused, self.output_weight, bias=self.output_bias)
        return out


class TtnnAudioXDiffusionTransformer:
    def __init__(self, ref_module, device, dim=768, depth=12, heads=12, context_dim=768):
        self.device = device
        self.input_proj_weight = create_linear_weight(ref_module.input_proj.weight.data, device)
        self.input_proj_bias = create_linear_weight(ref_module.input_proj.bias.data, device)
        self.time_embed = TtnnTimestepEmbedding(ref_module.time_embed, device)
        self.blocks = [TtnnTransformerBlock(block, device, heads) for block in ref_module.blocks]
        self.norm_out_weight = create_linear_weight(ref_module.norm_out.weight.data, device)
        self.norm_out_bias = create_linear_weight(ref_module.norm_out.bias.data, device)
        self.output_weight = create_linear_weight(ref_module.output_proj.weight.data, device)
        self.output_bias = create_linear_weight(ref_module.output_proj.bias.data, device)

    def __call__(self, x: ttnn.Tensor, timestep: ttnn.Tensor, context: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        x = ttnn.linear(x, self.input_proj_weight, bias=self.input_proj_bias)
        for block in self.blocks:
            x = block(x, context)
        x = ttnn.layer_norm(x, weight=self.norm_out_weight, bias=self.norm_out_bias)
        x = ttnn.linear(x, self.output_weight, bias=self.output_bias)
        return x


class TtnnAudioXModel:
    def __init__(self, ref_model, device):
        self.device = device

        self.text_proj_weight = create_linear_weight(ref_model.text_encoder.weight.data, device)
        self.text_proj_bias = create_linear_weight(ref_model.text_encoder.bias.data, device)
        self.audio_proj_weight = create_linear_weight(ref_model.audio_encoder.weight.data, device)
        self.audio_proj_bias = create_linear_weight(ref_model.audio_encoder.bias.data, device)
        self.video_proj_weight = create_linear_weight(ref_model.video_encoder.weight.data, device)
        self.video_proj_bias = create_linear_weight(ref_model.video_encoder.bias.data, device)
        self.image_proj_weight = create_linear_weight(ref_model.image_encoder.weight.data, device)
        self.image_proj_bias = create_linear_weight(ref_model.image_encoder.bias.data, device)

        self.fusion = TtnnMultimodalAdaptiveFusion(ref_model.fusion, device)
        self.diffusion = TtnnAudioXDiffusionTransformer(ref_model.diffusion, device)

        self.vocoder_weight = create_conv_weight(ref_model.vocoder.weight.data, device)
        self.vocoder_bias = create_conv_weight(ref_model.vocoder.bias.data, device)

    def encode_text(self, text_embeds: torch.Tensor) -> ttnn.Tensor:
        t = torch_to_ttnn(text_embeds, self.device)
        return ttnn.linear(t, self.text_proj_weight, bias=self.text_proj_bias)

    def encode_audio(self, audio_embeds: torch.Tensor) -> ttnn.Tensor:
        t = torch_to_ttnn(audio_embeds, self.device)
        return ttnn.linear(t, self.audio_proj_weight, bias=self.audio_proj_bias)

    def encode_video(self, video_embeds: torch.Tensor) -> ttnn.Tensor:
        t = torch_to_ttnn(video_embeds, self.device)
        return ttnn.linear(t, self.video_proj_weight, bias=self.video_proj_bias)

    def encode_image(self, image_embeds: torch.Tensor) -> ttnn.Tensor:
        t = torch_to_ttnn(image_embeds, self.device)
        return ttnn.linear(t, self.image_proj_weight, bias=self.image_proj_bias)

    def fusion_forward(self, *modalities: ttnn.Tensor) -> ttnn.Tensor:
        return self.fusion(*modalities)

    def diffusion_forward(self, x: torch.Tensor, timestep: torch.Tensor,
                          context: Optional[ttnn.Tensor] = None) -> torch.Tensor:
        x_tt = torch_to_ttnn(x, self.device)
        t_tt = torch_to_ttnn(timestep, self.device)
        out_tt = self.diffusion(x_tt, t_tt, context)
        return ttnn_to_torch(out_tt)

    def generate(self, text=None, video=None, image=None, audio=None,
                 num_steps=50, guidance_scale=3.0, length=160) -> torch.Tensor:
        batch_size = 1
        device = self.device

        text_embeds = torch.randn(batch_size, 77, 512) if text else None
        video_embeds = torch.randn(batch_size, 8, 1024) if video else None
        image_embeds = torch.randn(batch_size, 1, 1024) if image else None
        audio_embeds = torch.randn(batch_size, length, 80) if audio else None

        context = None
        ctx_modalities = []
        if text_embeds is not None:
            ctx_modalities.append(self.encode_text(text_embeds))
        if video_embeds is not None:
            ctx_modalities.append(self.encode_video(video_embeds))
        if image_embeds is not None:
            ctx_modalities.append(self.encode_image(image_embeds))
        if audio_embeds is not None:
            ctx_modalities.append(self.encode_audio(audio_embeds))

        if len(ctx_modalities) > 1:
            context = self.fusion_forward(*ctx_modalities)
        elif len(ctx_modalities) == 1:
            context = ctx_modalities[0]

        x = torch.randn(batch_size, length, 80)
        for step in range(num_steps):
            t = torch.full((batch_size, 1), 1.0 - step / num_steps)
            noise_pred = self.diffusion_forward(x, t, context)
            if guidance_scale > 1.0:
                uncond = self.diffusion_forward(x, t)
                noise_pred = uncond + guidance_scale * (noise_pred - uncond)
            x = x - 0.02 * noise_pred

        return x
