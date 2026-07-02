# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end DiffusionGemma pipeline.

Mirrors the autoregressive-over-canvases + diffusion-within-canvas loop from
``transformers.models.diffusion_gemma.generation_diffusion_gemma.DiffusionGemmaGenerationMixin.generate``,
but routes the encoder + decoder forwards through our TT model. The sampler /
logits processor / stopping criteria classes are imported directly from HF so we
don't duplicate the (pure-torch, host-side) math.

Scope notes:
  * Single batch (``batch_size == 1``) — the simpler inference case. Multi-batch
    requires per-row finished tracking which is straightforward to add but not
    needed for the first parity pass.
  * Per-canvas the encoder re-encodes the full prefix (no incremental cache
    extension between canvases). This is suboptimal for throughput but
    semantically equivalent and avoids the cache-rolling complexity. Optimization
    follow-up: extend the KV cache rather than recomputing it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

import ttnn

from ...models.transformers.diffusion_gemma._state_utils import per_layer_moe_substates
from ...models.transformers.diffusion_gemma.encoder_model import DiffusionGemmaEncoderModel
from ...models.transformers.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion, DiffusionGemmaModel
from ...models.transformers.diffusion_gemma.text_decoder import DiffusionGemmaDecoderModel
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils.tensor import bf16_tensor, local_device_to_torch


@dataclasses.dataclass
class DiffusionGemmaPipelineConfig:
    mesh_device: ttnn.MeshDevice
    tp_axis: int = 0
    num_links: int = 1
    topology: ttnn.Topology = ttnn.Topology.Linear
    # Sampler tuning. Defaults match HF generation_config.json defaults.
    max_denoising_steps: int = 48
    entropy_bound: float = 0.1
    temperature_start: float = 0.8
    temperature_end: float = 0.4
    expert_dtype: ttnn.DataType = ttnn.bfloat16
    router_dtype: ttnn.DataType = ttnn.bfloat16


class DiffusionGemmaPipeline:
    """Driver: HF processor → encoder → decoder loop → text."""

    def __init__(
        self,
        config: DiffusionGemmaPipelineConfig,
        tt_model: DiffusionGemmaForBlockDiffusion,
        hf_model,  # HF DiffusionGemmaForBlockDiffusion (used for non-TT scaffolding only)
        hf_processor,
        hf_generation_config,
    ) -> None:
        self.config = config
        self.tt_model = tt_model
        self.hf_model = hf_model
        self.hf_processor = hf_processor
        self.hf_generation_config = hf_generation_config

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        config: DiffusionGemmaPipelineConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "DiffusionGemmaPipeline":
        """Build the pipeline from an HF checkpoint.

        The HF model is kept in memory for: (a) the multi-modal preprocessor +
        generation_config, (b) the text encoder's vision tower's std_bias/scale
        buffers when needed. The model weights are also used to populate our
        TT model via ``load_state_dict``.
        """
        from transformers import AutoProcessor
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            DiffusionGemmaForBlockDiffusion as HFForBlockDiffusion,
        )

        hf_model = HFForBlockDiffusion.from_pretrained(model_id, dtype=torch_dtype).eval()
        hf_processor = AutoProcessor.from_pretrained(model_id)
        hf_gen_config = hf_model.generation_config

        tt_model = cls._build_tt_model_from_hf(hf_model, config)
        return cls(config, tt_model, hf_model, hf_processor, hf_gen_config)

    @staticmethod
    def _build_tt_model_from_hf(hf_model, config: DiffusionGemmaPipelineConfig) -> DiffusionGemmaForBlockDiffusion:
        """Construct our TT model with hyperparameters from the HF config and load weights."""
        hf_cfg = hf_model.config
        text_cfg = hf_cfg.text_config
        vision_cfg = hf_cfg.vision_config

        mesh_device = config.mesh_device
        tp_factor = tuple(mesh_device.shape)[config.tp_axis]
        parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=config.tp_axis, factor=tp_factor),
            sequence_parallel=ParallelFactor(
                mesh_axis=1 - config.tp_axis, factor=tuple(mesh_device.shape)[1 - config.tp_axis]
            ),
            cfg_parallel=None,
        )
        ccl_manager = CCLManager(mesh_device=mesh_device, num_links=config.num_links, topology=config.topology)

        # Extract per-layer MoE substates from the encoder + decoder state dicts.
        hf_state = hf_model.state_dict()

        def _per_layer_moe(prefix: str) -> list[dict]:
            return per_layer_moe_substates(hf_state, num_layers=text_cfg.num_hidden_layers, prefix=prefix)

        text_kwargs = dict(
            vocab_size=text_cfg.vocab_size,
            hidden_size=text_cfg.hidden_size,
            intermediate_size=text_cfg.intermediate_size,
            num_hidden_layers=text_cfg.num_hidden_layers,
            layer_types=list(text_cfg.layer_types),
            num_attention_heads=text_cfg.num_attention_heads,
            num_key_value_heads=text_cfg.num_key_value_heads,
            num_global_key_value_heads=text_cfg.num_global_key_value_heads,
            head_dim=text_cfg.head_dim,
            global_head_dim=text_cfg.global_head_dim,
            sliding_window=text_cfg.sliding_window,
            num_experts=text_cfg.num_experts,
            top_k_experts=text_cfg.top_k_experts,
            moe_intermediate_size=text_cfg.moe_intermediate_size,
            rms_norm_eps=text_cfg.rms_norm_eps,
            max_position_embeddings=text_cfg.max_position_embeddings,
            sliding_rope_theta=text_cfg.rope_parameters["sliding_attention"]["rope_theta"],
            full_rope_theta=text_cfg.rope_parameters["full_attention"]["rope_theta"],
            full_partial_rotary_factor=text_cfg.rope_parameters["full_attention"]["partial_rotary_factor"],
            pad_token_id=text_cfg.pad_token_id,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            num_links=config.num_links,
            topology=config.topology,
            expert_dtype=config.expert_dtype,
            router_dtype=config.router_dtype,
        )

        encoder_text_kwargs = {**text_kwargs, "moe_state_dicts": _per_layer_moe("model.encoder.language_model.")}
        decoder_text_kwargs = {**text_kwargs, "moe_state_dicts": _per_layer_moe("model.decoder.")}

        # Vision tower kwargs (only when vision_config is present and has standardize).
        vision_kwargs = None
        if vision_cfg is not None:
            # Pad head_dim 72 → 96 (the standard configuration). Caller can override later.
            vision_kwargs = dict(
                hidden_size=vision_cfg.hidden_size,
                intermediate_size=vision_cfg.intermediate_size,
                num_hidden_layers=vision_cfg.num_hidden_layers,
                num_attention_heads=vision_cfg.num_attention_heads,
                head_dim=vision_cfg.head_dim,
                head_dim_padded=96 if vision_cfg.head_dim == 72 else vision_cfg.head_dim,
                patch_size=vision_cfg.patch_size,
                position_embedding_size=vision_cfg.position_embedding_size,
                pooling_kernel_size=vision_cfg.pooling_kernel_size,
                default_output_length=vision_cfg.default_output_length,
                rms_norm_eps=vision_cfg.rms_norm_eps,
                rope_theta=vision_cfg.rope_parameters["rope_theta"],
                standardize=vision_cfg.standardize,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )

        # Build encoder + decoder.
        encoder = DiffusionGemmaEncoderModel(
            text_kwargs=encoder_text_kwargs,
            vision_kwargs=vision_kwargs,
            multimodal_hidden_size=(vision_cfg.hidden_size if vision_cfg else text_cfg.hidden_size),
            text_hidden_size=text_cfg.hidden_size,
            rms_norm_eps=text_cfg.rms_norm_eps,
            image_token_id=hf_cfg.image_token_id,
            pad_token_id=text_cfg.pad_token_id,
            mesh_device=mesh_device,
        )
        decoder = DiffusionGemmaDecoderModel(**decoder_text_kwargs)

        model = DiffusionGemmaModel(encoder=encoder, decoder=decoder)
        tt_for_diffusion = DiffusionGemmaForBlockDiffusion(
            model=model,
            text_hidden_size=text_cfg.hidden_size,
            vocab_size=text_cfg.vocab_size,
            final_logit_softcapping=text_cfg.final_logit_softcapping,
            mesh_device=mesh_device,
        )
        tt_for_diffusion.load_state_dict(hf_state)
        return tt_for_diffusion

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        text_prompt: str,
        images: list | None = None,
        max_new_tokens: int = 256,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Generate text from a prompt (optionally with images).

        Returns a dict with ``sequences`` (LongTensor [B, total_len]),
        ``generated_text`` (decoded string), and ``num_canvases``.
        """
        from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
            EntropyBoundSampler,
            EntropyBoundSamplerConfig,
            LinearTemperatureScheduleLogitsProcessor,
            StableAndConfidentStoppingCriteria,
        )

        torch.manual_seed(seed)
        device = torch.device("cpu")

        # 1. Tokenize + image-process.
        messages = self._build_chat_messages(text_prompt, images)
        proc_inputs = self.hf_processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        input_ids = proc_inputs["input_ids"]
        pixel_values = proc_inputs.get("pixel_values")
        pixel_position_ids = proc_inputs.get("pixel_position_ids")

        text_cfg = self.hf_model.config.text_config
        canvas_length = self.hf_model.config.canvas_length
        vocab_size = text_cfg.vocab_size

        # 2. Sampler / processor / stopping setup (HF classes, host-side).
        sampler = EntropyBoundSampler(
            EntropyBoundSamplerConfig(entropy_bound=self.config.entropy_bound),
            canvas_length=canvas_length,
            vocab_size=vocab_size,
            max_denoising_steps=self.config.max_denoising_steps,
        )
        logits_processor = LinearTemperatureScheduleLogitsProcessor(
            temperature_start=self.config.temperature_start,
            temperature_end=self.config.temperature_end,
            num_steps=self.config.max_denoising_steps,
        )
        stopping = StableAndConfidentStoppingCriteria()

        # 3. Outer (canvas) loop.
        cur_input_ids = input_ids
        max_canvases = max(1, (max_new_tokens + canvas_length - 1) // canvas_length)
        for canvas_idx in range(max_canvases):
            # 3a. Encode the prompt + previously generated canvases. Build masks via HF mask utilities.
            encoder_kv, encoder_attention_masks, encoder_position_ids = self._run_encoder(
                cur_input_ids, pixel_values, pixel_position_ids
            )

            # 3b. Inner (diffusion) loop on the new canvas.
            current_canvas = sampler.initialize_canvas(batch_size=cur_input_ids.shape[0], device=device)
            self_cond_logits: torch.Tensor | None = None
            finished_denoising = torch.zeros(cur_input_ids.shape[0], dtype=torch.bool, device=device)
            decoder_position_ids = torch.arange(
                cur_input_ids.shape[1], cur_input_ids.shape[1] + canvas_length, dtype=torch.long
            ).unsqueeze(0)
            argmax_canvas = current_canvas.clone()

            for step in range(self.config.max_denoising_steps):
                logits = self._run_decoder(
                    current_canvas,
                    encoder_kv,
                    decoder_position_ids,
                    encoder_attention_masks,
                    encoder_seq_len=cur_input_ids.shape[1],
                    self_cond_logits=self_cond_logits,
                )
                argmax_canvas = logits.argmax(dim=-1)
                processed_logits = logits_processor(logits, step)
                # Sample (categorical) for the denoiser canvas.
                probs = torch.softmax(processed_logits.float(), dim=-1)
                denoiser_canvas = torch.distributions.Categorical(probs=probs).sample()
                # Accept and renoise per the sampler.
                accepted = sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits, step)
                current_canvas = sampler.renoise_canvas(accepted, step)
                # Adaptive stopping.
                finished_denoising = stopping(argmax_canvas, processed_logits)
                self_cond_logits = processed_logits
                if torch.all(finished_denoising):
                    break

            # 3c. Append the argmax canvas (final accepted tokens) to the sequence.
            cur_input_ids = torch.cat([cur_input_ids, argmax_canvas], dim=-1)

            # 3d. Stop if EOS appears in the canvas.
            if self._sequence_finished(argmax_canvas, text_cfg):
                break

        new_tokens = cur_input_ids[:, input_ids.shape[1] :]
        generated_text = self.hf_processor.batch_decode(new_tokens, skip_special_tokens=False)
        return {
            "sequences": cur_input_ids,
            "generated_text": generated_text[0] if cur_input_ids.shape[0] == 1 else generated_text,
            "num_canvases": canvas_idx + 1,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_chat_messages(self, text_prompt: str, images: list | None) -> list[dict]:
        content: list[dict] = []
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text_prompt})
        return [{"role": "user", "content": content}]

    def _run_encoder(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None,
        pixel_position_ids: torch.Tensor | None,
    ):
        """Run the TT encoder. Returns (encoder_kv, attention_masks, position_ids)."""
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        text_cfg = self.hf_model.config.text_config
        mesh_device = self.config.mesh_device

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Build per-layer-type masks via HF's mask utilities (no hand-coded mask math).
        with torch.no_grad():
            inputs_embeds_for_mask = self.hf_model.model.encoder.language_model.embed_tokens(input_ids)
        mask_kwargs = {
            "config": text_cfg,
            "inputs_embeds": inputs_embeds_for_mask,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        hf_masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        # HF's create_causal_mask returns a BOOLEAN mask under the SDPA path
        # (True=attend, False=masked). ttnn SDPA expects an ADDITIVE mask
        # (0.0=attend, -inf=masked). Convert before upload.
        def _to_additive(m: torch.Tensor) -> torch.Tensor:
            if m.dtype == torch.bool:
                m = torch.where(m, torch.tensor(0.0), torch.tensor(float("-inf")))
            return m.to(torch.bfloat16)

        tt_masks = {
            lt: (
                ttnn.from_torch(_to_additive(m), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                if m is not None
                else None
            )
            for lt, m in hf_masks.items()
        }

        tt_input_ids = ttnn.from_torch(input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        tt_pixels = None
        padding_positions = None
        if pixel_values is not None:
            # Pixel preprocessing: HF expects raw [0, 1] pixels but our patch embedder
            # expects pre-scaled (2*p - 1) input. Apply on host.
            scaled_px = 2.0 * (pixel_values.float() - 0.5)
            tt_pixels = bf16_tensor(scaled_px.unsqueeze(0), device=mesh_device)
            if pixel_position_ids is not None:
                padding_positions = (pixel_position_ids == -1).all(dim=-1)

        # Run our TT encoder.
        _hidden, per_layer_kv = self.tt_model.model.encoder(
            tt_input_ids,
            position_ids,
            tt_masks,
            pixel_values=tt_pixels,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            input_ids_host=input_ids,
        )
        return per_layer_kv, tt_masks, position_ids

    def _run_decoder(
        self,
        canvas_ids: torch.LongTensor,
        encoder_kv: list,
        decoder_position_ids: torch.Tensor,
        encoder_attention_masks: dict,
        encoder_seq_len: int,
        self_cond_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one decoder forward + lm_head softcap. Returns torch logits on host."""
        mesh_device = self.config.mesh_device
        text_cfg = self.hf_model.config.text_config
        canvas_length = canvas_ids.shape[1]

        # Decoder mask: bidirectional across [encoder cache | canvas]. Additive zeros.
        decoder_mask_dict = {
            "full_attention": torch.zeros(
                canvas_ids.shape[0], 1, canvas_length, encoder_seq_len + canvas_length, dtype=torch.bfloat16
            ),
            "sliding_attention": torch.zeros(
                canvas_ids.shape[0], 1, canvas_length, encoder_seq_len + canvas_length, dtype=torch.bfloat16
            ),
        }
        tt_decoder_masks = {
            lt: ttnn.from_torch(m, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            for lt, m in decoder_mask_dict.items()
        }

        tt_canvas = ttnn.from_torch(canvas_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Self-conditioning signal: prior step's logits → soft embeddings. The decoder
        # module accepts ``self_conditioning_signal`` (already projected to hidden). We
        # compute the soft embedding here.
        tt_self_cond_signal = None
        if self_cond_logits is not None:
            soft_probs = self_cond_logits.softmax(dim=-1, dtype=torch.float32).to(torch.bfloat16)
            embed_w = self.hf_model.model.decoder.embed_tokens.weight.detach().to(torch.bfloat16)
            soft_emb = (soft_probs @ embed_w).to(torch.bfloat16) * (text_cfg.hidden_size**0.5)
            tt_self_cond_signal = bf16_tensor(soft_emb.unsqueeze(0), device=mesh_device)

        # Single clean decoder step: decoder + lm_head + softcap on-device.
        out = self.tt_model.decoder_step(
            tt_canvas,
            decoder_position_ids,
            encoder_kv_cache=encoder_kv,
            decoder_attention_masks=tt_decoder_masks,
            self_conditioning_signal=tt_self_cond_signal,
        )
        # Bring to host as fp32 for the sampler math. local_device_to_torch returns the
        # single-device replica with its on-device shape preserved — typically [B, canvas, vocab].
        # The HF sampler expects exactly that shape; do NOT squeeze (otherwise we lose the
        # batch dim and Categorical treats `canvas` as batch).
        out_host = local_device_to_torch(out).to(torch.float32)
        # Defensive: handle the edge case where the upload added a leading mesh dim.
        if out_host.ndim == 4 and out_host.shape[0] == 1:
            out_host = out_host.squeeze(0)
        return out_host

    def _sequence_finished(self, canvas: torch.LongTensor, text_cfg) -> bool:
        eos = text_cfg.eos_token_id
        if isinstance(eos, int):
            eos = [eos]
        for eid in eos:
            if (canvas == eid).any():
                return True
        return False
