from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from diffusers.models.attention import FeedForward, JointTransformerBlock
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed, Timesteps
from diffusers.models.normalization import AdaLayerNormContinuous
from loguru import logger

from ...utils.check import assert_quality
from . import configuration_motifimage, modeling_dit

if TYPE_CHECKING:
    from collections.abc import Mapping


class MotifJointTransformerBlock(JointTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hidden_states_residual = hidden_states
        encoder_hidden_states_residual = encoder_hidden_states

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        # different in Motif:
        # hidden_states = hidden_states + ff_output
        hidden_states = hidden_states_residual + ff_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        # different in Motif:
        # encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        encoder_hidden_states = encoder_hidden_states_residual + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


class MotifDiT(torch.nn.Module):
    """Motif Image Diffusion Transformer.

    An implementation of the Motif Image 6B transformer based on diffusers.SD3Transformer2DModel. It
    is missing the rescaling of each query head (mmdit_blocks.*.attn.q_scale in the state), but the
    effect of doing so is quite small.
    """

    ENCODED_TEXT_DIM = 4096
    SD3_LATENT_CHANNEL = 16

    def __init__(
        self,
        *,
        patch_size: int,
        num_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        pooled_projection_dim: int,
        pos_embed_max_size: int,
        modulation_dim: int,
        time_embed_dim: int,
        register_token_num: int,
    ) -> None:
        super().__init__()

        self.out_channels = self.SD3_LATENT_CHANNEL
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size

        sample_size = 64

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=self.SD3_LATENT_CHANNEL,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = torch.nn.Linear(self.ENCODED_TEXT_DIM, self.inner_dim)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                MotifJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=False,
                    qk_norm="rms_norm",
                    use_dual_attention=False,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, modulation_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = torch.nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        # new in Motif:
        self.register_tokens = torch.nn.Parameter(torch.randn(1, register_token_num, self.inner_dim))
        # self.t_token_proj = torch.nn.Linear(modulation_dim, self.inner_dim) if use_time_token_in_attn else None

        # changed in Motif:
        self.time_text_embed.timestep_embedder.linear_1 = torch.nn.Linear(time_embed_dim, 4 * time_embed_dim)
        self.time_text_embed.timestep_embedder.linear_2 = torch.nn.Linear(4 * time_embed_dim, modulation_dim)
        self.time_text_embed.text_embedder.linear_1 = torch.nn.Linear(pooled_projection_dim, 4 * pooled_projection_dim)
        self.time_text_embed.text_embedder.linear_2 = torch.nn.Linear(4 * pooled_projection_dim, modulation_dim)
        self.time_text_embed.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        for block in self.transformer_blocks:
            block.norm1.linear = torch.nn.Linear(modulation_dim, 6 * self.inner_dim)
            block.norm1_context.linear = torch.nn.Linear(modulation_dim, 6 * self.inner_dim)
            block.ff = FeedForward(dim=self.inner_dim, dim_out=self.inner_dim, activation_fn="linear-silu")
            block.ff_context = FeedForward(dim=self.inner_dim, dim_out=self.inner_dim, activation_fn="linear-silu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # new in Motif
        hidden_states = torch.cat((self.register_tokens.expand(hidden_states.shape[0], -1, -1), hidden_states), dim=1)
        # if self.t_token_proj is not None:
        #     t_token = self.t_token_proj(temb).unsqueeze(1)
        #     encoder_hidden_states = torch.cat([encoder_hidden_states, t_token], dim=1)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        # new in Motif
        hidden_states = hidden_states[:, self.register_tokens.shape[1] :]

        # same as in SD3 but without norm and silu
        emb = self.norm_out.linear(temb)
        scale, shift = torch.chunk(emb, 2, dim=1)
        hidden_states = hidden_states * (1 + scale)[:, None, :] + shift[:, None, :]

        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

    @staticmethod
    def convert_state_dict(state_dict: Mapping[str, torch.Tensor], /, *, num_layers: int) -> Mapping[str, torch.Tensor]:
        renames = {
            "pos_embed": "pos_embed.pos_embed",
            "text_cond.projection.weight": "context_embedder.weight",
            "text_cond.projection.bias": "context_embedder.bias",
            "final_modulation.weight": "norm_out.linear.weight",
            "final_modulation.bias": "norm_out.linear.bias",
            "final_linear_SD3.weight": "proj_out.weight",
            "final_linear_SD3.bias": "proj_out.bias",
            "patching.projection_SD3.weight": "pos_embed.proj.weight",
            "patching.projection_SD3.bias": "pos_embed.proj.bias",
            "time_emb.time_emb.linear_1.weight": "time_text_embed.timestep_embedder.linear_1.weight",
            "time_emb.time_emb.linear_1.bias": "time_text_embed.timestep_embedder.linear_1.bias",
            "time_emb.time_emb.linear_2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
            "time_emb.time_emb.linear_2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
            "time_emb.pooled_text_emb.linear_1.weight": "time_text_embed.text_embedder.linear_1.weight",
            "time_emb.pooled_text_emb.linear_1.bias": "time_text_embed.text_embedder.linear_1.bias",
            "time_emb.pooled_text_emb.linear_2.weight": "time_text_embed.text_embedder.linear_2.weight",
            "time_emb.pooled_text_emb.linear_2.bias": "time_text_embed.text_embedder.linear_2.bias",
        }

        block_renames = {
            "attn.o_proj.weight": "attn.to_out.0.weight",
            "attn.add_o_proj.weight": "attn.to_add_out.weight",
            "attn.q_norm_x.weight": "attn.norm_q.weight",
            "attn.k_norm_x.weight": "attn.norm_k.weight",
            "attn.q_norm_c.weight": "attn.norm_added_q.weight",
            "attn.k_norm_c.weight": "attn.norm_added_k.weight",
            "affine_params_c.projection.weight": "norm1_context.linear.weight",
            "affine_params_c.projection.bias": "norm1_context.linear.bias",
            "affine_params_x.projection.weight": "norm1.linear.weight",
            "affine_params_x.projection.bias": "norm1.linear.bias",
            "mlp_3_c.gate_proj.weight": "ff_context.net.0.proj.weight",
            "mlp_3_c.gate_proj.bias": "ff_context.net.0.proj.bias",
            "mlp_3_c.down_proj.weight": "ff_context.net.2.weight",
            "mlp_3_c.down_proj.bias": "ff_context.net.2.bias",
            "mlp_3_x.gate_proj.weight": "ff.net.0.proj.weight",
            "mlp_3_x.gate_proj.bias": "ff.net.0.proj.bias",
            "mlp_3_x.down_proj.weight": "ff.net.2.weight",
            "mlp_3_x.down_proj.bias": "ff.net.2.bias",
        }

        def swap_scale_and_shift(prefix: str) -> None:
            ts = out[f"{prefix}.weight"].chunk(6, dim=0)
            out[f"{prefix}.weight"] = torch.concat([ts[1], ts[0], ts[2], ts[4], ts[3], ts[5]], dim=0)

            ts = out[f"{prefix}.bias"].chunk(6, dim=0)
            out[f"{prefix}.bias"] = torch.concat([ts[1], ts[0], ts[2] + 1, ts[4], ts[3] - 1, ts[5]], dim=0)

        out = dict(state_dict)

        for src, dst in renames.items():
            out[dst] = out.pop(src)

        for i in range(num_layers):
            src_prefix = f"mmdit_blocks.{i}"
            dst_prefix = f"transformer_blocks.{i}"

            for src, dst in block_renames.items():
                out[f"{dst_prefix}.{dst}"] = out.pop(f"{src_prefix}.{src}")

            x_weight = out.pop(f"{src_prefix}.linear_1_x.weight")
            x_bias = out.pop(f"{src_prefix}.linear_1_x.bias")

            q_weight = out.pop(f"{src_prefix}.attn.q_proj.weight")
            out[f"{dst_prefix}.attn.to_q.weight"] = q_weight @ x_weight
            out[f"{dst_prefix}.attn.to_q.bias"] = q_weight @ x_bias
            k_weight = out.pop(f"{src_prefix}.attn.k_proj.weight")
            out[f"{dst_prefix}.attn.to_k.weight"] = k_weight @ x_weight
            out[f"{dst_prefix}.attn.to_k.bias"] = k_weight @ x_bias
            v_weight = out.pop(f"{src_prefix}.attn.v_proj.weight")
            out[f"{dst_prefix}.attn.to_v.weight"] = v_weight @ x_weight
            out[f"{dst_prefix}.attn.to_v.bias"] = v_weight @ x_bias

            c_weight = out.pop(f"{src_prefix}.linear_1_c.weight")
            c_bias = out.pop(f"{src_prefix}.linear_1_c.bias")

            add_q_weight = out.pop(f"{src_prefix}.attn.add_q_proj.weight")
            out[f"{dst_prefix}.attn.add_q_proj.weight"] = add_q_weight @ c_weight
            out[f"{dst_prefix}.attn.add_q_proj.bias"] = add_q_weight @ c_bias
            add_k_weight = out.pop(f"{src_prefix}.attn.add_k_proj.weight")
            out[f"{dst_prefix}.attn.add_k_proj.weight"] = add_k_weight @ c_weight
            out[f"{dst_prefix}.attn.add_k_proj.bias"] = add_k_weight @ c_bias
            add_v_weight = out.pop(f"{src_prefix}.attn.add_v_proj.weight")
            out[f"{dst_prefix}.attn.add_v_proj.weight"] = add_v_weight @ c_weight
            out[f"{dst_prefix}.attn.add_v_proj.bias"] = add_v_weight @ c_bias

            out[f"{dst_prefix}.attn.to_out.0.bias"] = torch.zeros_like(out[f"{dst_prefix}.attn.to_out.0.weight"][0])
            out[f"{dst_prefix}.attn.to_add_out.bias"] = torch.zeros_like(out[f"{dst_prefix}.attn.to_add_out.weight"][0])

            swap_scale_and_shift(f"{dst_prefix}.norm1.linear")
            swap_scale_and_shift(f"{dst_prefix}.norm1_context.linear")

        out["pos_embed.pos_embed"] = out["pos_embed.pos_embed"].unsqueeze(0)

        # unused:
        del out["t_token_proj.weight"]
        del out["t_token_proj.bias"]

        # not implemented:
        for i in range(num_layers):
            del out[f"mmdit_blocks.{i}.attn.q_scale"]

        return out


def substate(state: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    prefix = f"{key}."
    prefix_len = len(prefix)

    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}


def combine_prompt_embeddings(t5_xxl, clip_a, clip_b):
    clip_emb = torch.cat([clip_a, clip_b], dim=-1)
    clip_emb = torch.nn.functional.pad(clip_emb, (0, t5_xxl.shape[-1] - clip_emb.shape[-1]))
    return torch.cat([clip_emb, t5_xxl], dim=-2)


def main() -> None:
    model_path = "motif_image_preview.bin"

    modulation_dim = 4096  # new parameter (is hidden_dim in SD3 large in some places)
    time_embed_dim = 4096  # new parameter (related constant in SD3 large is 256)
    register_token_num = 4  # new parameter
    num_layers = 30  # SD3 large: 38
    num_attention_heads = 30  # SD3 large: 38
    hidden_dim = 1920  # SD3 large: 2432
    pos_emb_size = 64  # SD3 large: 192
    patch_size = 2  # same in SD3 large
    pooled_text_dim = 2048  # same in SD3 large

    height = 1024
    width = 1024

    torch.manual_seed(0)
    latents = torch.randn([2, 16, height // 8, width // 8])
    timestep = torch.full([2], fill_value=500, dtype=torch.int64)
    text_embs = [
        torch.randn([2, 256, 4096]),
        torch.randn([2, 77, 768]),
        torch.randn([2, 77, 1280]),
    ]
    pooled_text_embs = torch.randn([2, 2048])

    logger.info("loading state dict from disk...")
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), mmap=True)

    logger.info("creating model...")
    sd3_model = MotifDiT(
        patch_size=patch_size,
        num_layers=num_layers,
        attention_head_dim=hidden_dim // num_attention_heads,
        num_attention_heads=num_attention_heads,
        pooled_projection_dim=pooled_text_dim,
        pos_embed_max_size=pos_emb_size,
        modulation_dim=modulation_dim,
        time_embed_dim=time_embed_dim,
        register_token_num=register_token_num,
    )

    logger.info("converting state dict...")
    hf_dit_state_dict = MotifDiT.convert_state_dict(substate(state_dict, "dit"), num_layers=num_layers)

    logger.info("loading state dict into model...")
    sd3_model.load_state_dict(hf_dit_state_dict)

    logger.info("running model...")
    out1 = sd3_model.forward(
        hidden_states=latents,
        encoder_hidden_states=combine_prompt_embeddings(*text_embs),
        pooled_projections=pooled_text_embs,
        timestep=timestep,
    )

    del sd3_model

    logger.info("creating model...")
    ref = modeling_dit.MotifDiT(
        configuration_motifimage.MotifImageConfig(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            in_channel=4,
            out_channel=4,
            time_embed_dim=time_embed_dim,
            attn_embed_dim=4096,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=30,
            use_scaled_dot_product_attention=True,
            dropout=0.0,
            mlp_hidden_dim=7680,
            use_modulation=True,
            modulation_type="film",
            register_token_num=register_token_num,
            additional_register_token_num=0,
            skip_register_token_num=0,
            height=height,
            width=width,
            attn_mode="flash",
            use_final_layer_norm=False,
            pos_emb_size=pos_emb_size,
            conv_header=False,
            use_time_token_in_attn=True,
            modulation_dim=modulation_dim,
            pooled_text_dim=pooled_text_dim,
        )
    )

    logger.info("loading state dict into model...")
    ref.load_state_dict(substate(state_dict, "dit"), strict=False)

    logger.info("running model...")
    out2 = ref.forward(latents, timestep, text_embs, pooled_text_embs)

    assert_quality(out1, out2)


if __name__ == "__main__":
    main()
