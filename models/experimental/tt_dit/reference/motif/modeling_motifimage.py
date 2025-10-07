# SPDX-FileCopyrightText: Copyright (c) 2025 Motif Technologies

# SPDX-License-Identifier: MIT License

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import tqdm
from diffusers.models import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from loguru import logger
from PIL import Image, ImageFilter
from transformers import CLIPTextModel, CLIPTokenizerFast, T5EncoderModel, T5Tokenizer

from .modeling_dit import MotifDiT

TOKEN_MAX_LENGTH: int = 256
VAE_DOWNSCALE_FACTOR: int = 8


class MotifImage(nn.Module):
    """
    MotifImage wraps a Diffusion Transformer (DiT) with multi-encoder text conditioning
    and a VAE for image decoding, providing sampling utilities for rectified flow.

    Args:
        config (MMDiTConfig): Configuration for model construction and encoder/decoder backends.

    Attributes:
        config: The configuration object passed at initialization.
        dit: The underlying DiT backbone that predicts velocity given latents, timestep, and text conditioning.
        vae: VAE used to decode latent tensors into images.
        t5, clip_l, clip_g: Text encoders for conditioning (T5, CLIP-L, CLIP-G).
        t5_tokenizer, clip_l_tokenizer, clip_g_tokenizer: Corresponding tokenizers.
        tokenizers (List): Convenience list of tokenizers in encoder order.
        text_encoders (List): Convenience list of text encoders in encoder order.
        cond_drop_prob (float): Drop probability for CFG-related behavior (kept for compatibility).
        use_weighting (bool): Placeholder flag for optional loss weighting (not used at inference time).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dit = MotifDiT(config)
        self.cond_drop_prob = 0.1
        self.use_weighting = False
        self._get_encoders()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This model is not used for training.")

    def _get_encoders(
        self,
        vae_path: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        t5_path: str = "google/flan-t5-xxl",
        clip_l_path: str = "openai/clip-vit-large-patch14",
        clip_g_path: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ) -> None:
        """Initialize the VAE and text encoders.

        Args:
            vae_path: Hugging Face model repo or local path for the SD3/SDXL VAE (subfolder may be used).
            t5_path: HF repo or local path for the T5 encoder.
            clip_l_path: HF repo or local path for CLIP-L text encoder.
            clip_g_path: HF repo or local path for CLIP-G text encoder.

        Raises:
            ValueError: If `self.config.vae_type` is not one of {"SD3", "SDXL"}.
        """
        if self.config.vae_type == "SD3":
            self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
        elif self.config.vae_type == "SDXL":
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        else:
            raise ValueError(f"VAE type must be `SD3` or `SDXL`  but self.config.vae_type is {self.config.vae_type}")

        # Text encoders
        # 1. T5-XXL from Google
        self.t5 = T5EncoderModel.from_pretrained(t5_path)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_path)

        # 2. CLIP-L from OpenAI
        self.clip_l = CLIPTextModel.from_pretrained(clip_l_path)
        self.clip_l_tokenizer = CLIPTokenizerFast.from_pretrained(clip_l_path)

        # 3. CLIP-G from LAION
        self.clip_g = CLIPTextModel.from_pretrained(clip_g_path)
        self.clip_g_tokenizer = CLIPTokenizerFast.from_pretrained(clip_g_path)

        self.tokenizers = [
            self.t5_tokenizer,
            self.clip_l_tokenizer,
            self.clip_g_tokenizer,
        ]
        self.text_encoders = [self.t5, self.clip_l, self.clip_g]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Return a state dict excluding large, external backbones.

        The returned state dictionary omits parameters from `t5`, `clip_l`, `clip_g`, and `vae` so that
        the checkpoint focuses on the DiT backbone weights and optional PEFT/LoRA weights.

        Args:
            destination: See `nn.Module.state_dict`.
            prefix: See `nn.Module.state_dict`.
            keep_vars: See `nn.Module.state_dict`.

        Returns:
            OrderedDict: Filtered state dictionary.
        """
        state_dict = super(MotifImage, self).state_dict(destination, prefix, keep_vars)
        exclude_keys = ["t5.", "clip_l.", "clip_g.", "vae."]
        for key in list(state_dict.keys()):
            if any(key.startswith(exclude_key) for exclude_key in exclude_keys):
                state_dict.pop(key)
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        """Load weights and (optionally) merge LoRA parameters.

        If the provided `state_dict` contains LoRA parameters, the model will enable LoRA (when not
        already enabled), load the weights non-strictly, then merge and unload LoRA modules so the
        resulting model behaves as a vanilla backbone at inference time.

        Args:
            state_dict (dict): Parameters to load.
            strict (bool): Whether to strictly enforce key matching.

        Returns:
            Tuple[List[str], List[str]]: Missing and unexpected keys, as returned by `nn.Module.load_state_dict`.
        """
        # Check if state_dict contains LoRA parameters
        has_lora = any("lora_" in key for key in state_dict.keys())

        if has_lora:
            # If model doesn't have LoRA enabled but state_dict has LoRA params, enable it
            if not hasattr(self.dit, "peft_config"):
                logger.info("Enabling LoRA for parameter merging...")
                # Use default values if not already configured
                lora_rank = getattr(self.config, "lora_rank", 64)
                lora_alpha = getattr(self.config, "lora_alpha", 8)
                self.enable_lora(lora_rank, lora_alpha)

        if has_lora:
            try:
                missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
                # Merge LoRA weights with base model
                logger.info("Merging LoRA parameters with base model...")
                for name, module in self.dit.named_modules():
                    if hasattr(module, "merge_and_unload"):
                        module.merge_and_unload()

                logger.info("Successfully merged LoRA parameters")

            except Exception as e:
                logger.error(f"Error merging LoRA parameters: {str(e)}")
                raise
        else:
            missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)

        # Log summary of missing/unexpected parameters
        missing_top_levels = set()
        for key in missing_keys:
            top_level_name = key.split(".")[0]
            missing_top_levels.add(top_level_name)
        if missing_top_levels:
            logger.debug("Missing keys during loading at top level:")
            for name in missing_top_levels:
                logger.debug(name)

        if unexpected_keys:
            logger.debug("Unexpected keys found:")
            for key in unexpected_keys:
                logger.debug(key)

        return missing_keys, unexpected_keys

    def tokenization(
        self, raw_texts: List[str], repeat_if_short: bool = False, get_rare_negative_token: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Batch-tokenize texts for all supported tokenizers.

        Optionally repeat short texts to better utilize context length per tokenizer. This method
        returns parallel lists of input ids and attention masks aligned with `self.tokenizers`.

        Args:
            raw_texts: Batch of input strings.
            repeat_if_short: If True, repeat each short text to approximate the max length for the given tokenizer.
            get_rare_negative_token: If True, replace inputs with rare negative tokens per tokenizer (used for CFG).

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Two lists (tokens, masks). Each list has one tensor per tokenizer with
            shape [batch, max_length].
        """
        final_batch_tokens = []
        final_batch_masks = []

        # Process the batch with each tokenizer
        for tokenizer in self.tokenizers:
            if get_rare_negative_token:
                if tokenizer == self.t5_tokenizer:
                    raw_texts = ["<extra_id_42>" for _ in range(len(raw_texts))]
                elif tokenizer == self.clip_l_tokenizer:
                    raw_texts = ["[◊◊◊]" for _ in range(len(raw_texts))]
                elif tokenizer == self.clip_g_tokenizer:
                    raw_texts = ["[◊◊◊]" for _ in range(len(raw_texts))]

            effective_max_length = min(TOKEN_MAX_LENGTH, tokenizer.model_max_length)

            # 1. Pre-process the batch: Create a new list of potentially repeated strings.
            processed_texts_for_tokenizer = []
            for text_item in raw_texts:
                # Start with the original text for this item
                processed_text = text_item

                if repeat_if_short:
                    # Apply repetition logic individually based on text_item's length
                    num_initial_tokens = len(text_item.split())
                    available_length = effective_max_length - 2  # Heuristic

                    if num_initial_tokens > 0 and num_initial_tokens < available_length:
                        num_additional_repeats = available_length // (num_initial_tokens + 1)
                        if num_additional_repeats > 0:
                            total_repeats = 1 + num_additional_repeats
                            processed_text = " ".join([text_item] * total_repeats)

                # Add the processed text (original or repeated) to the list for this tokenizer
                processed_texts_for_tokenizer.append(processed_text)

            # 2. Tokenize the entire batch of processed texts at once.
            #    Pass the list `processed_texts_for_tokenizer` directly to the tokenizer.
            #    The tokenizer's __call__ method should handle the batch efficiently.
            batch_tok_output = tokenizer(  # Call the tokenizer ONCE with the full list
                processed_texts_for_tokenizer,
                padding="max_length",
                max_length=effective_max_length,
                return_tensors="pt",
                truncation=True,
            )

            # 3. Store the resulting batch tensors directly.
            #    The tokenizer should return tensors with shape [batch_size, max_length].
            final_batch_tokens.append(batch_tok_output.input_ids)
            final_batch_masks.append(batch_tok_output.attention_mask)

        return final_batch_tokens, final_batch_masks

    @torch.no_grad()
    def text_encoding(
        self, tokens: List[torch.Tensor], masks, zero_masking=True
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Encode tokenized texts with T5, CLIP-L, and CLIP-G.

        Applies optional masking of padding tokens and optional noise on padding positions. Returns
        sequence-level embeddings per encoder and pooled embeddings from CLIP encoders.

        Args:
            tokens: List of token id tensors for each tokenizer.
            masks: List of attention mask tensors aligned with `tokens`.
            zero_masking: If True, zero-out embeddings at padding tokens.

        Returns:
            Tuple[List[Tensor], Tensor]:
            - List of sequence embeddings [B, L, C] for T5, CLIP-L, CLIP-G (in that order).
            - Concatenated pooled CLIP embeddings [B, 2048].
        """
        t5_tokens, clip_l_tokens, clip_g_tokens = tokens
        t5_masks, clip_l_masks, clip_g_masks = masks
        t5_emb = self.t5(t5_tokens, attention_mask=t5_masks)[0]

        if zero_masking:
            t5_emb = t5_emb * (t5_tokens != self.t5_tokenizer.pad_token_id).unsqueeze(-1)

        clip_l_emb = self.clip_l(input_ids=clip_l_tokens, output_hidden_states=True)
        clip_g_emb = self.clip_g(input_ids=clip_g_tokens, output_hidden_states=True)
        clip_l_emb_pooled = clip_l_emb.pooler_output  # B x 768
        clip_g_emb_pooled = clip_g_emb.pooler_output  # B x 1280

        clip_l_emb = clip_l_emb.last_hidden_state  # B x L x 768,
        clip_g_emb = clip_g_emb.last_hidden_state  # B x L x 1280,

        def masking_wo_first_eos(token, eos):
            idx = (token != eos).sum(dim=1)
            mask = token != eos
            arange = torch.arange(mask.size(0))
            mask[arange, idx] = True
            mask = mask.unsqueeze(-1)  # B x L x 1
            return mask

        if zero_masking:
            clip_l_emb = clip_l_emb * masking_wo_first_eos(
                clip_l_tokens, self.clip_l_tokenizer.eos_token_id
            )  # B x L x 768,
            clip_g_emb = clip_g_emb * masking_wo_first_eos(
                clip_g_tokens, self.clip_g_tokenizer.eos_token_id
            )  # B x L x 768,

        encodings = [t5_emb, clip_l_emb, clip_g_emb]
        pooled_encodings = torch.cat([clip_l_emb_pooled, clip_g_emb_pooled], dim=-1)  # cat by channel, B x 2048

        return encodings, pooled_encodings

    @torch.no_grad()
    def prompt_embedding(self, prompts: str, device, zero_masking=True, get_rare_negative_token=False):
        """Convenience wrapper that tokenizes and encodes prompts.

        Args:
            prompts: List of text prompts.
            device: Target device for tensors.
            zero_masking: Whether to zero-out padding tokens.
            get_rare_negative_token: If True, use rare tokens (for unconditional CFG branch).

        Returns:
            Tuple[List[Tensor], Tensor]: Sequence embeddings list and pooled CLIP embedding tensor.
        """
        tokens, masks = self.tokenization(prompts, get_rare_negative_token=get_rare_negative_token)
        tokens = [token.to(device) for token in tokens]
        masks = [mask.to(device) for mask in masks]
        text_embeddings, pooled_text_embeddings = self.text_encoding(tokens, masks, zero_masking=zero_masking)
        return text_embeddings, pooled_text_embeddings

    @torch.no_grad()
    def sample(
        self,
        raw_text: List[str],
        steps: int = 50,
        guidance_scale: float = 7.5,
        resolution: List[int] = (256, 256),
        pre_latent=None,
        pre_timestep=None,
        zero_masking=False,
        zero_embedding_for_cfg=False,
        negative_prompt: Optional[List[str]] = None,
        device: str = "cpu",
        rescale_cfg=-1.0,
        clip_t=[0.0, 1.0],
        use_linear_quadratic_schedule=False,
        linear_quadratic_emulating_steps=250,
        get_intermediate_steps: bool = False,
        get_rare_negative_token=False,
        negative_strategy_switch_t: Optional[float] = 0.15,
    ) -> Union[List[Image.Image], Tuple[List[Image.Image], List[List[Image.Image]]]]:
        """Generate images with the rectified flow sampler.

        This method integrates the learned velocity field using a simple Euler update on a time schedule.
        Optionally returns intermediate predictions derived from the observed average velocity.

        Args:
            raw_text: Batch of text prompts.
            steps: Number of ODE integration steps.
            guidance_scale: Classifier-free guidance scale (>1.0 enables CFG).
            resolution: Output image resolution as (H, W).
            pre_latent: Optional initial latent to start from (e.g., editing or partial denoising).
            pre_timestep: Normalized start time in [0, 1]. If provided, the schedule starts at this value.
            zero_masking: Mask out padding tokens of text encoders.
            zero_embedding_for_cfg: Use zero embeddings for the unconditional branch when `negative_prompt` is None.
            negative_prompt: Optional negative prompts for CFG.
            device: Torch device for sampling.
            rescale_cfg: Strength for CFG rescaling (<=0 disables).
            clip_t: Time window [t_start, t_end] where CFG is active.
            use_linear_quadratic_schedule: If True, use a linear-quadratic schedule (MovieGen-inspired).
            linear_quadratic_emulating_steps: N parameter used by the linear-quadratic schedule approximation.
            get_intermediate_steps: If True, also return intermediate images reconstructed from averaged velocity.
            get_rare_negative_token: If True, use rare tokens when building empty negative prompts.
            negative_strategy_switch_t: If set to a value in [0, 1], dynamically switch the negative CFG strategy per
                step: for timesteps t >= this threshold (higher noise), use empty-prompt negatives; for timesteps
                t < this threshold (lower noise), use zero-tensor negatives. Only applies when CFG is enabled,
                `negative_prompt` is None, and `zero_embedding_for_cfg` is True. Otherwise ignored.

        Returns:
            If `get_intermediate_steps` is False: List of final PIL images.
            If `get_intermediate_steps` is True: Tuple of (final images, list of per-step image batches).
        """
        prompts = raw_text

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        b = len(prompts)
        h, w = resolution

        # --- [Initial Latent Noise (e = x_1)] ---
        latent_channels = 16
        if pre_latent is None:
            initial_noise = randn_tensor(  # Store initial noise separately
                (
                    b,
                    latent_channels,
                    h // VAE_DOWNSCALE_FACTOR,
                    w // VAE_DOWNSCALE_FACTOR,
                ),
                device=device,
                dtype=torch.float32,  # Use float32 for calculations
            )
        else:
            initial_noise = pre_latent.to(device=device, dtype=torch.float32)
            if pre_timestep is not None and pre_timestep < 1.0:  # Check if it's truly intermediate
                logger.warning(
                    "Using pre_latent as initial_noise for average calculation, but pre_timestep suggests it's not pure noise. Results might be unexpected."
                )

        latents = initial_noise.clone()  # Working latents for the ODE solver

        # --- [Text Embeddings & CFG Setup] ---
        text_embeddings, pooled_text_embeddings = self.prompt_embedding(
            prompts, latents.device, zero_masking=zero_masking
        )
        text_embeddings = [emb.to(device=latents.device) for emb in text_embeddings]
        pooled_text_embeddings = pooled_text_embeddings.to(device=latents.device)

        # Keep conditional embeddings separate for potential per-step CFG handling
        cond_text_embeddings = text_embeddings
        cond_pooled_text_embeddings = pooled_text_embeddings

        do_classifier_free_guidance = guidance_scale > 1.0

        # Dynamic negative switch is only considered when zero_embedding_for_cfg is True and no explicit negatives
        use_dynamic_negative_switch = (
            do_classifier_free_guidance
            and negative_prompt is None
            and zero_embedding_for_cfg
            and negative_strategy_switch_t is not None
        )

        if do_classifier_free_guidance:
            if negative_prompt is not None:
                # Explicit negatives provided: use them uniformly for all steps
                negative_text_embeddings, negative_pooled_text_embeddings = self.prompt_embedding(
                    negative_prompt, latents.device, zero_masking=zero_masking
                )
                negative_text_embeddings = [emb.to(device=latents.device) for emb in negative_text_embeddings]
                negative_pooled_text_embeddings = negative_pooled_text_embeddings.to(device=latents.device)
                text_embeddings = [
                    torch.cat([cond, neg], dim=0) for cond, neg in zip(cond_text_embeddings, negative_text_embeddings)
                ]
                pooled_text_embeddings = torch.cat(
                    [cond_pooled_text_embeddings, negative_pooled_text_embeddings], dim=0
                )
            elif use_dynamic_negative_switch:
                # Precompute both negative variants; concatenate per step in the sampling loop
                empty_text_embeddings, empty_pooled_text_embeddings = self.prompt_embedding(
                    ["" for _ in range(len(prompts))],
                    latents.device,
                    get_rare_negative_token=get_rare_negative_token,
                )
                empty_text_embeddings = [emb.to(device=latents.device) for emb in empty_text_embeddings]
                empty_pooled_text_embeddings = empty_pooled_text_embeddings.to(device=latents.device)

                zero_text_embeddings = [
                    torch.zeros_like(text_embedding, device=text_embedding.device)
                    for text_embedding in cond_text_embeddings
                ]
                zero_pooled_text_embeddings = torch.zeros_like(
                    cond_pooled_text_embeddings, device=cond_pooled_text_embeddings.device
                )
            else:
                # Single-strategy negatives for all steps
                if zero_embedding_for_cfg:
                    negative_text_embeddings = [
                        torch.zeros_like(text_embedding, device=text_embedding.device)
                        for text_embedding in cond_text_embeddings
                    ]
                    negative_pooled_text_embeddings = torch.zeros_like(
                        cond_pooled_text_embeddings, device=cond_pooled_text_embeddings.device
                    )
                else:
                    negative_text_embeddings, negative_pooled_text_embeddings = self.prompt_embedding(
                        ["" for _ in range(len(prompts))],
                        latents.device,
                        get_rare_negative_token=get_rare_negative_token,
                    )
                    negative_text_embeddings = [emb.to(device=latents.device) for emb in negative_text_embeddings]
                    negative_pooled_text_embeddings = negative_pooled_text_embeddings.to(device=latents.device)

                text_embeddings = [
                    torch.cat([cond, neg], dim=0) for cond, neg in zip(cond_text_embeddings, negative_text_embeddings)
                ]
                pooled_text_embeddings = torch.cat(
                    [cond_pooled_text_embeddings, negative_pooled_text_embeddings], dim=0
                )
        else:
            # CFG disabled: use zeros
            text_embeddings = [
                torch.zeros_like(text_embedding, device=text_embedding.device)
                for text_embedding in cond_text_embeddings
            ]
            pooled_text_embeddings = torch.zeros_like(
                cond_pooled_text_embeddings, device=cond_pooled_text_embeddings.device
            )

        # --- [Timestep Schedule (Sigmas)] ---
        # linear t schedule
        sigmas = torch.linspace(1, 0, steps + 1) if pre_timestep is None else torch.linspace(pre_timestep, 0, steps + 1)

        if use_linear_quadratic_schedule:
            # liner-quadratic t schedule
            assert steps % 2 == 0
            N = linear_quadratic_emulating_steps
            sigmas = torch.concat(
                [
                    torch.linspace(1, 0, N + 1)[: steps // 2],
                    torch.linspace(0, 1, steps // 2 + 1) ** 2 * (steps // 2 * 1 / N - 1) - (steps // 2 * 1 / N - 1),
                ]
            )

        # --- [Initialization for Intermediate Step Calculation] ---
        # intermediate_latents will store the latent states for intermediate steps
        intermediate_latents = [] if get_intermediate_steps else None
        predicted_velocities = []  # Store dx from each step
        sigma_history = []
        # --- [Sampling Loop] ---
        for infer_step, t in tqdm.tqdm(enumerate(sigmas[:-1]), total=len(sigmas[:-1]), desc="Sampling"):
            # Prepare input for DiT model
            if do_classifier_free_guidance:
                input_latents = torch.cat([latents] * 2, dim=0)
            else:
                input_latents = latents

            # Prepare timestep input
            timestep = (t * 1000).round().long().to(latents.device)
            timestep = timestep.expand(input_latents.shape[0])

            # Choose per-step text embeddings if dynamic switching is enabled
            if use_dynamic_negative_switch:
                t_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
                if t_scalar >= float(negative_strategy_switch_t):
                    neg_step_text_embeddings = empty_text_embeddings
                    neg_step_pooled_text_embeddings = empty_pooled_text_embeddings
                else:
                    neg_step_text_embeddings = zero_text_embeddings
                    neg_step_pooled_text_embeddings = zero_pooled_text_embeddings

                text_embeddings_step = [
                    torch.cat([cond, neg], dim=0) for cond, neg in zip(cond_text_embeddings, neg_step_text_embeddings)
                ]
                pooled_text_embeddings_step = torch.cat(
                    [cond_pooled_text_embeddings, neg_step_pooled_text_embeddings], dim=0
                )
            else:
                text_embeddings_step = text_embeddings
                pooled_text_embeddings_step = pooled_text_embeddings

            # Predict velocity dx = v(x_t, t) ≈ e - x_0
            dx = self.dit(
                input_latents,
                timestep,
                text_embeddings_step,
                pooled_text_embeddings_step,
            )
            dt = sigmas[infer_step + 1] - sigmas[infer_step]  # dt is negative
            sigma_history.append(dt)

            # Apply Classifier-Free Guidance
            if do_classifier_free_guidance:
                cond_dx, uncond_dx = dx.chunk(2)
                current_guidance_scale = guidance_scale if clip_t[0] <= t and t <= clip_t[1] else 1.0
                dx = uncond_dx + current_guidance_scale * (cond_dx - uncond_dx)

                if rescale_cfg > 0.0:
                    std_pos = torch.std(cond_dx, dim=[1, 2, 3], keepdim=True, unbiased=False) + 1e-5
                    std_cfg = torch.std(dx, dim=[1, 2, 3], keepdim=True, unbiased=False) + 1e-5
                    factor = std_pos / std_cfg
                    factor = rescale_cfg * factor + (1.0 - rescale_cfg)
                    dx = dx * factor

            # --- Store the predicted velocity for averaging ---
            predicted_velocities.append(dx.clone())

            # --- Update Latents using standard Euler step ---
            latents = latents + dt * dx

            # --- Calculate and Store Intermediate Latent State (if requested) ---
            if get_intermediate_steps:
                dxs = torch.stack(predicted_velocities)

                sigma_sum = sum(sigma_history)
                normalized_sigma_history = [s / (sigma_sum) for s in sigma_history]
                dts = torch.tensor(normalized_sigma_history, device=dxs.device, dtype=dxs.dtype).view(-1, 1, 1, 1, 1)

                avg_dx = torch.sum(dxs * dts, dim=0)
                observed_state = initial_noise - avg_dx  # Calculate the desired intermediate state
                intermediate_latents.append(observed_state.clone())  # Store its latent representation

        # --- [Decode Final Latents to PIL Images] ---
        self.vae = self.vae.to(device=latents.device, dtype=torch.float32)  # Ensure VAE is ready
        final_latents_scaled = latents.to(torch.float32) / self.vae.config.scaling_factor
        final_image_tensors = self.vae.decode(final_latents_scaled, return_dict=False)[0] + self.vae.config.shift_factor
        final_image_tensors = ((final_image_tensors + 1.0) / 2.0).clamp(0.0, 1.0)

        final_pil_images = []
        for i, image_tensor in enumerate(final_image_tensors):
            img = T.ToPILImage()(image_tensor.cpu())
            final_pil_images.append(img)

        # --- [Decode Intermediate Latents to PIL Images (if requested)] ---
        if get_intermediate_steps:
            intermediate_pil_images = []
            # Ensure VAE is still ready (it should be from final decoding)
            for step_latents in tqdm.tqdm(intermediate_latents, desc="Decoding intermediates"):
                step_latents_scaled = step_latents.to(dtype=torch.float32) / self.vae.config.scaling_factor
                step_image_tensors = (
                    self.vae.decode(step_latents_scaled, return_dict=False)[0] + self.vae.config.shift_factor
                )
                step_image_tensors = ((step_image_tensors + 1.0) / 2.0).clamp(0.0, 1.0)

                current_step_pil = []
                for i, image_tensor in enumerate(step_image_tensors):
                    img = T.ToPILImage()(image_tensor.cpu())
                    current_step_pil.append(img)
                intermediate_pil_images.append(current_step_pil)  # Append list of images for this step

            return (
                final_pil_images,
                intermediate_pil_images,
            )  # Return both final and intermediate images
        else:
            return final_pil_images  # Return only final images
