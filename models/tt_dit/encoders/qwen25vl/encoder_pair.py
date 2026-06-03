# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

import ttnn

from ...layers.linear import ColParallelLinear
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from .model_qwen25vl import Qwen25VlTextEncoder, Qwen25VlVisionEncoder
from .multimodal_preprocess import Qwen25VlMultimodalPreprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

_EDIT_PROMPT_DROP = 64
_EDIT_PROMPT_MAX_SEQ = 512


class Qwen25VlTokenizerEncoderPair:
    def __init__(
        self,
        checkpoint: str,
        *,
        tokenizer_subfolder: str | None = None,
        encoder_subfolder: str | None = None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_torch: bool,
        is_fsdp: bool = False,
        build_vision_encoder: bool = False,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._checkpoint = checkpoint
        self._encoder_subfolder = encoder_subfolder
        self._use_torch = use_torch
        self._encoder_loaded = True
        self._is_fsdp = is_fsdp
        self._build_vision_encoder = build_vision_encoder

        if tokenizer_subfolder is not None:
            self._tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint, subfolder=tokenizer_subfolder)
        else:
            self._tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint)
        self._vision_encoder: Qwen25VlVisionEncoder | None = None
        self._image_token_id: int | None = None
        self._vision_spatial_merge_size: int | None = None
        self._vision_head_dim: int | None = None
        self._vision_rope_theta: float | None = None
        self._vision_rope_cache: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._lm_rope_cache: OrderedDict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._rope_cache_max = 32
        self._encoder = self._load_encoder(checkpoint, encoder_subfolder, use_torch=use_torch)
        self._multimodal_preprocessor: Qwen25VlMultimodalPreprocessor | None = None
        if self._build_vision_encoder and not self._use_torch:
            self._multimodal_preprocessor = Qwen25VlMultimodalPreprocessor.from_hub(self._tokenizer)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def _finalize_edit_prompt_device(
        self,
        tt_last: ttnn.Tensor,
        batch: int,
        v_valid: int,
        hidden_sz: int,
    ) -> tuple[ttnn.Tensor, torch.Tensor]:
        if v_valid <= _EDIT_PROMPT_DROP:
            msg = "valid token count must exceed prompt drop"
            raise ValueError(msg)
        core_len = v_valid - _EDIT_PROMPT_DROP
        if core_len > _EDIT_PROMPT_MAX_SEQ:
            msg = f"prompt length {core_len} exceeds {_EDIT_PROMPT_MAX_SEQ}"
            raise ValueError(msg)
        pad_amt = _EDIT_PROMPT_MAX_SEQ - core_len
        tt_cut = ttnn.slice(tt_last, [0, _EDIT_PROMPT_DROP, 0], [batch, v_valid, hidden_sz])
        if pad_amt > 0:
            tt_rm = ttnn.to_layout(tt_cut, ttnn.ROW_MAJOR_LAYOUT)
            tt_pd = ttnn.pad(tt_rm, [(0, 0), (0, pad_amt), (0, 0)], 0.0)
            tt_out = ttnn.to_layout(tt_pd, ttnn.TILE_LAYOUT)
        else:
            tt_out = tt_cut
        mask_cpu = torch.zeros(batch, _EDIT_PROMPT_MAX_SEQ, dtype=torch.long)
        mask_cpu[:, :core_len] = 1
        tt_m = tensor.from_torch(mask_cpu.to(torch.bfloat16), device=self._device)
        tt_m3 = ttnn.reshape(tt_m, (batch, _EDIT_PROMPT_MAX_SEQ, 1))
        tt_out = ttnn.multiply(tt_out, tt_m3)
        return tt_out, mask_cpu

    def _load_encoder(
        self, checkpoint: str, subfolder: str | None, *, use_torch: bool
    ) -> Qwen2_5_VLForConditionalGeneration | Qwen25VlTextEncoder:
        if subfolder is not None:
            torch_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, subfolder=subfolder)
        else:
            torch_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint)

        if use_torch:
            return torch_model

        model = Qwen25VlTextEncoder(
            vocab_size=torch_model.config.vocab_size,
            hidden_size=torch_model.config.hidden_size,
            intermediate_size=torch_model.config.intermediate_size,
            hidden_act=torch_model.config.hidden_act,
            num_hidden_layers=torch_model.config.num_hidden_layers,
            num_attention_heads=torch_model.config.num_attention_heads,
            num_key_value_heads=torch_model.config.num_key_value_heads,
            rms_norm_eps=torch_model.config.rms_norm_eps,
            rope_theta=torch_model.config.rope_theta,
            mrope_section=torch_model.config.rope_scaling["mrope_section"],
            device=self._device,
            ccl_manager=self._ccl_manager,
            parallel_config=self._parallel_config,
            is_fsdp=self._is_fsdp,
        )

        torch_text_model = torch_model.model.language_model
        torch_state_dict = torch_text_model.state_dict()

        if self._build_vision_encoder:
            self._build_device_vision_encoder(torch_model)
            self._image_token_id = int(torch_model.config.image_token_id)

        del torch_model
        del torch_text_model

        self._encoder_state_dict = torch_state_dict

        cache.load_model(
            tt_model=model,
            get_torch_state_dict=lambda: torch_state_dict,
            model_name=checkpoint,
            subfolder=subfolder if subfolder is not None else "",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            is_fsdp=self._is_fsdp,
        )

        return model

    def _build_device_vision_encoder(self, torch_model: Qwen2_5_VLForConditionalGeneration) -> None:
        visual = torch_model.model.visual
        vcfg = visual.config

        vision_encoder = Qwen25VlVisionEncoder(
            hidden_size=vcfg.hidden_size,
            intermediate_size=vcfg.intermediate_size,
            num_heads=vcfg.num_heads,
            depth=vcfg.depth,
            patch_size=vcfg.patch_size,
            temporal_patch_size=vcfg.temporal_patch_size,
            in_channels=vcfg.in_channels,
            out_hidden_size=vcfg.out_hidden_size,
            spatial_merge_size=vcfg.spatial_merge_size,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            device=self._device,
            parallel_config=self._parallel_config,
            ccl_manager=self._ccl_manager,
            is_fsdp=self._is_fsdp,
        )

        vision_state = visual.state_dict()

        cache.load_model(
            tt_model=vision_encoder,
            get_torch_state_dict=lambda: vision_state,
            model_name=self._checkpoint,
            subfolder="vision_encoder",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            is_fsdp=self._is_fsdp,
        )

        self._vision_encoder = vision_encoder
        self._vision_spatial_merge_size = vcfg.spatial_merge_size
        self._vision_head_dim = vcfg.hidden_size // vcfg.num_heads
        self._vision_rope_theta = 10000.0

    def encoder_loaded(self) -> bool:
        return self._use_torch or self._encoder.is_loaded()

    def reload_encoder_weights(self) -> None:
        if self._use_torch or self._encoder_loaded:
            return

        logger.info("reloading encoder weights to device...")
        self._vision_rope_cache.clear()
        self._lm_rope_cache.clear()
        cache.load_model(
            tt_model=self._encoder,
            get_torch_state_dict=lambda: self._encoder_state_dict,
            model_name=self._checkpoint,
            subfolder=self._encoder_subfolder if self._encoder_subfolder is not None else "",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._device.shape),
            is_fsdp=self._is_fsdp,
        )

        self._encoder_loaded = True
        ttnn.synchronize_device(self._device)

    def deallocate_encoder_weights(self) -> None:
        if self._use_torch or not self._encoder_loaded:
            return

        self._encoder.deallocate_weights()
        self._encoder_loaded = False
        self._vision_rope_cache.clear()
        self._lm_rope_cache.clear()
        if hasattr(self, "_tp_image_col_proj"):
            delattr(self, "_tp_image_col_proj")
        if hasattr(self, "_tp_image_col_proj_key"):
            delattr(self, "_tp_image_col_proj_key")
        ttnn.synchronize_device(self._device)

    @staticmethod
    def _ordered_cache_put(cache: OrderedDict, key, value: tuple[torch.Tensor, torch.Tensor], max_items: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)

    def _get_tp_image_col_proj(self, hidden_sz: int, tp_axis: int) -> ColParallelLinear:
        qkv = self._encoder.layers[0].self_attn.qkv_proj
        fsdp_a = qkv.fsdp_mesh_axis
        cached = getattr(self, "_tp_image_col_proj", None)
        key = (hidden_sz, tp_axis, fsdp_a)
        if isinstance(cached, ColParallelLinear) and getattr(self, "_tp_image_col_proj_key", None) == key:
            return cached
        proj = ColParallelLinear(
            hidden_sz,
            hidden_sz,
            bias=False,
            mesh_device=self._device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_a,
            ccl_manager=self._ccl_manager,
        )
        proj.weight.load_torch_tensor(torch.eye(hidden_sz, dtype=torch.bfloat16))
        self._tp_image_col_proj_key = key
        self._tp_image_col_proj = proj
        return proj

    def encode(
        self, prompts: Sequence[str], *, num_images_per_prompt: int, sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_qwen_prompt_embeds(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer,
            text_encoder=self._encoder,
            sequence_length=sequence_length,
            mesh_device=self._device,
        )

    def encode_with_images(
        self,
        formatted_prompts: Sequence[str],
        images: Sequence,
        *,
        num_images_per_prompt: int,
        omit_final_host_gather: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[ttnn.Tensor, torch.Tensor]:
        """Multimodal encode using the built-in host preprocessor (replaces ``Qwen2VLProcessor``).

        Produces the same ``input_ids``, ``attention_mask``, ``pixel_values``, and
        ``image_grid_thw`` contract as HuggingFace ``Qwen2VLProcessor`` for the image
        + text path, then runs the TT vision tower and text encoder on device.
        """
        if self._vision_encoder is None:
            raise RuntimeError("encode_with_images requires a built vision encoder.")
        if self._image_token_id is None:
            raise RuntimeError("encode_with_images requires image_token_id.")
        if not isinstance(self._encoder, Qwen25VlTextEncoder):
            raise RuntimeError("encode_with_images requires the device-side text encoder (use_torch=False)")
        if self._multimodal_preprocessor is None:
            raise RuntimeError("encode_with_images requires build_vision_encoder=True (multimodal preprocessor).")

        model_inputs = self._multimodal_preprocessor(
            text=list(formatted_prompts),
            images=list(images),
            padding=True,
            return_tensors="pt",
        )

        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        pixel_values = model_inputs.pixel_values
        grid_thw = model_inputs.image_grid_thw

        batch = input_ids.shape[0]
        num_images = grid_thw.shape[0]

        from .model_qwen25vl import build_vision_rope_tensors

        grid_list = [tuple(int(v) for v in row) for row in grid_thw]
        padded_head_dim = ((self._vision_head_dim + 31) // 32) * 32
        vrope_key = (
            tuple(grid_list),
            self._vision_head_dim,
            padded_head_dim,
            self._vision_spatial_merge_size,
            self._vision_rope_theta,
        )
        if vrope_key in self._vision_rope_cache:
            cos_t, sin_t = self._vision_rope_cache[vrope_key]
            self._vision_rope_cache.move_to_end(vrope_key)
        else:
            cos_t, sin_t = build_vision_rope_tensors(
                grid_list,
                head_dim=self._vision_head_dim,
                spatial_merge_size=self._vision_spatial_merge_size,
                theta=self._vision_rope_theta,
                pad_to=padded_head_dim,
            )
            self._ordered_cache_put(self._vision_rope_cache, vrope_key, (cos_t, sin_t), self._rope_cache_max)

        in_features = (
            self._vision_encoder.patch_embed.in_channels
            * self._vision_encoder.patch_embed.temporal_patch_size
            * self._vision_encoder.patch_embed.patch_size
            * self._vision_encoder.patch_embed.patch_size
        )

        pixel_flat = pixel_values.to(torch.bfloat16).reshape(1, -1, in_features).contiguous()
        tt_pixel = tensor.from_torch(pixel_flat, device=self._device)
        tt_cos = tensor.from_torch(cos_t.unsqueeze(0).to(torch.bfloat16), device=self._device)
        tt_sin = tensor.from_torch(sin_t.unsqueeze(0).to(torch.bfloat16), device=self._device)

        tt_image_hidden = self._vision_encoder.forward(tt_pixel, pos_embeds=(tt_cos, tt_sin))
        n_vis = int(tt_image_hidden.shape[1])
        hidden_sz = int(tt_image_hidden.shape[2])
        image_mask = input_ids == self._image_token_id
        n_image_slots = int(image_mask.sum().item())
        if n_image_slots != n_vis:
            raise RuntimeError(
                f"image token slot count {n_image_slots} does not match vision features "
                f"{n_vis} (num_images={num_images}, grid_thw={grid_thw.tolist()})"
            )

        seq_len = int(input_ids.shape[1])
        is_image_i = (input_ids == self._image_token_id).long()
        row_ids = ((torch.cumsum(is_image_i, dim=-1) - 1) * is_image_i).clamp(min=0).to(torch.int32)

        lm_tp = self._encoder._tp_axis
        tt_ids = tensor.from_torch(input_ids.to(torch.int32), device=self._device, dtype=ttnn.uint32)
        img_id_tt = tensor.from_torch(
            torch.full_like(input_ids, int(self._image_token_id), dtype=torch.int32),
            device=self._device,
            dtype=ttnn.uint32,
        )
        is_img = ttnn.eq(tt_ids, img_id_tt)
        is_img_bsh = ttnn.reshape(is_img, (batch, seq_len, 1))
        tt_row_ids = tensor.from_torch(row_ids, device=self._device, dtype=ttnn.uint32)
        tt_token = self._encoder.embed_tokens.forward(tt_ids)
        if lm_tp is None:
            img_table = ttnn.reshape(tt_image_hidden, (n_vis, hidden_sz))
            img_rows = ttnn.embedding(tt_row_ids, img_table, layout=ttnn.TILE_LAYOUT)
            tt_inputs_embeds = ttnn.where(is_img_bsh, img_rows, tt_token)
        else:
            proj = self._get_tp_image_col_proj(hidden_sz, lm_tp)
            img_table_full = ttnn.reshape(tt_image_hidden, (n_vis, hidden_sz))
            img_table_shard = proj.forward(img_table_full)
            img_rows = ttnn.embedding(tt_row_ids, img_table_shard, layout=ttnn.TILE_LAYOUT)
            tt_inputs_embeds = ttnn.where(is_img_bsh, img_rows, tt_token)

        lm_key = (batch, seq_len)
        if attention_mask is not None and bool(attention_mask.bool().all().item()):
            if lm_key in self._lm_rope_cache:
                cos_lm, sin_lm = self._lm_rope_cache[lm_key]
                self._lm_rope_cache.move_to_end(lm_key)
            else:
                cos_lm, sin_lm = self._encoder.create_rope_tensors(batch, seq_len, attention_mask)
                self._ordered_cache_put(self._lm_rope_cache, lm_key, (cos_lm, sin_lm), self._rope_cache_max)
        else:
            cos_lm, sin_lm = self._encoder.create_rope_tensors(batch, seq_len, attention_mask)

        tt_attention_mask = tensor.from_torch(attention_mask, device=self._device)
        tt_cos_lm = tensor.from_torch(cos_lm, device=self._device)
        tt_sin_lm = tensor.from_torch(sin_lm, device=self._device)

        tt_hidden = self._encoder.forward_embeds(
            tt_inputs_embeds,
            attention_mask=tt_attention_mask,
            pos_embeds=(tt_cos_lm, tt_sin_lm),
        )
        tt_last = tt_hidden[-1]
        tt_mask_bf = tensor.from_torch(attention_mask.to(torch.bfloat16), device=self._device)
        tt_mask_3 = ttnn.reshape(tt_mask_bf, (batch, seq_len, 1))
        tt_last = ttnn.multiply(tt_last, tt_mask_3)
        if omit_final_host_gather:
            vm = attention_mask.sum(dim=1).long() if attention_mask is not None else None
            uniform = vm is not None and vm.numel() > 0 and bool((vm == vm[0]).all().item())
            if uniform:
                v_valid = int(vm[0].item())
                try:
                    tt_padded, mask_cpu = self._finalize_edit_prompt_device(tt_last, batch, v_valid, hidden_sz)
                    return tt_padded, mask_cpu
                except (ValueError, RuntimeError):
                    pass
            if num_images_per_prompt != 1:
                tt_last = ttnn.repeat(tt_last, (num_images_per_prompt, 1, 1))
            attention_mask_out = attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
            return tt_last, attention_mask_out

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_last)[0])

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        attention_mask_out = attention_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, attention_mask_out


# adapted from https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L188
def _get_qwen_prompt_embeds(
    prompts: Sequence[str],
    text_encoder: Qwen25VlTextEncoder | Qwen2_5_VLForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    mesh_device: ttnn.MeshDevice | None,
    sequence_length: int,
    num_images_per_prompt: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
    )

    tokens = tokenizer_out.input_ids
    attention_mask = tokenizer_out.attention_mask

    untruncated_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
    ).input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("input text was truncated")

    if isinstance(text_encoder, Qwen25VlTextEncoder):
        assert mesh_device is not None

        cos, sin = text_encoder.create_rope_tensors(tokens.shape[0], tokens.shape[1], attention_mask)

        tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
        tt_attention_mask = tensor.from_torch(attention_mask, device=mesh_device)
        tt_cos = tensor.from_torch(cos, device=mesh_device)
        tt_sin = tensor.from_torch(sin, device=mesh_device)

        tt_hidden_states = text_encoder.forward(
            tt_tokens, attention_mask=tt_attention_mask, pos_embeds=(tt_cos, tt_sin)
        )
        tt_prompt_embeds = tt_hidden_states[-1]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)

        with torch.no_grad():
            output = text_encoder.forward(
                tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        prompt_embeds = output.hidden_states[-1].to("cpu")

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, attention_mask
