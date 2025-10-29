# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py

import html
from typing import Any, Callable, Dict, List, Optional, Union

import ftfy
import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel as TorchWanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

import ttnn
from loguru import logger
from ...parallel.manager import CCLManager
from ...models.transformers.wan2_2.transformer_wan import WanTransformer3DModel
from ...models.vae.vae_wan2_1 import WanDecoder
from ...utils.cache import get_and_create_cache_path, cache_dict_exists, save_cache_dict, load_cache_dict
from ...utils.conv3d import conv_pad_in_channels, conv_pad_height
from ...utils.tensor import bf16_tensor_2dshard


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, WanPipeline
        >>> from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        >>> # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=720,
        ...     width=1280,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer_2 ([`WanTransformer3DModel`], *optional*):
            Conditional Transformer to denoise the input latents during the low-noise stage. If provided, enables
            two-stage denoising where `transformer` handles high-noise stages and `transformer_2` handles low-noise
            stages. If not provided, only `transformer` is used.
        boundary_ratio (`float`, *optional*, defaults to `None`):
            Ratio of total timesteps to use as the boundary for switching between transformers in two-stage denoising.
            The actual boundary timestep is calculated as `boundary_ratio * num_train_timesteps`. When provided,
            `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
    """

    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2"]

    def __init__(
        self,
        mesh_device,
        parallel_config,
        vae_parallel_config,
        num_links,
        use_cache,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
        dynamic_load=False,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="tokenizer", trust_remote_code=True
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="text_encoder", trust_remote_code=True
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae", trust_remote_code=True
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="scheduler", trust_remote_code=True
        )
        self.torch_transformer = TorchWanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer", trust_remote_code=True
        )
        self.torch_transformer_2 = TorchWanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer_2", trust_remote_code=True
        )

        self.dit_ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=num_links,
            topology=topology,
        )
        self.vae_ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=num_links,
            topology=ttnn.Topology.Linear,  # NOTE: VAE always uses Linear topology. TODO: enable ring if given.
        )
        self.parallel_config = parallel_config
        self.vae_parallel_config = vae_parallel_config
        self.use_cache = use_cache
        self.mesh_device = mesh_device
        self.dynamic_load = dynamic_load
        if not self.dynamic_load:
            self._load_transformer1()
            self._load_transformer2()

        self.tt_vae = WanDecoder(
            base_dim=self.vae.config.base_dim,
            z_dim=self.vae.config.z_dim,
            dim_mult=self.vae.config.dim_mult,
            num_res_blocks=self.vae.config.num_res_blocks,
            attn_scales=self.vae.config.attn_scales,
            temperal_downsample=self.vae.config.temperal_downsample,
            out_channels=self.vae.config.out_channels,
            is_residual=self.vae.config.is_residual,
            mesh_device=self.mesh_device,
            ccl_manager=self.vae_ccl_manager,
            parallel_config=self.vae_parallel_config,
        )

        self.tt_vae.load_state_dict(self.vae.state_dict())

        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _load_transformer1(self):
        self.transformer = WanTransformer3DModel(
            patch_size=self.torch_transformer.config.patch_size,
            num_heads=self.torch_transformer.config.num_attention_heads,
            dim=self.torch_transformer.config.num_attention_heads * self.torch_transformer.config.attention_head_dim,
            in_channels=self.torch_transformer.config.in_channels,
            out_channels=self.torch_transformer.config.out_channels,
            text_dim=self.torch_transformer.config.text_dim,
            freq_dim=self.torch_transformer.config.freq_dim,
            ffn_dim=self.torch_transformer.config.ffn_dim,
            cross_attn_norm=self.torch_transformer.config.cross_attn_norm,
            eps=self.torch_transformer.config.eps,
            rope_max_seq_len=self.torch_transformer.config.rope_max_seq_len,
            mesh_device=self.mesh_device,
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=True,
        )

        if self.use_cache:
            cache_path = get_and_create_cache_path(
                model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer",
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                dtype="bf16",
            )
            # create cache if it doesn't exist
            if not cache_dict_exists(cache_path):
                logger.info(
                    f"Cache does not exist. Creating cache: {cache_path} and loading transformer weights from PyTorch state dict"
                )
                self.transformer.load_state_dict(self.torch_transformer.state_dict())
                save_cache_dict(self.transformer.to_cached_state_dict(cache_path), cache_path)
            else:
                logger.info(f"Loading transformer weights from cache: {cache_path}")
                self.transformer.from_cached_state_dict(load_cache_dict(cache_path))
        else:
            logger.info("Loading transformer weights from PyTorch state dict")
            self.transformer.load_state_dict(self.torch_transformer.state_dict())

    def _load_transformer2(self):
        self.transformer_2 = WanTransformer3DModel(
            patch_size=self.torch_transformer_2.config.patch_size,
            num_heads=self.torch_transformer_2.config.num_attention_heads,
            dim=self.torch_transformer_2.config.num_attention_heads
            * self.torch_transformer_2.config.attention_head_dim,
            in_channels=self.torch_transformer_2.config.in_channels,
            out_channels=self.torch_transformer_2.config.out_channels,
            text_dim=self.torch_transformer_2.config.text_dim,
            freq_dim=self.torch_transformer_2.config.freq_dim,
            ffn_dim=self.torch_transformer_2.config.ffn_dim,
            cross_attn_norm=self.torch_transformer_2.config.cross_attn_norm,
            eps=self.torch_transformer_2.config.eps,
            rope_max_seq_len=self.torch_transformer_2.config.rope_max_seq_len,
            mesh_device=self.mesh_device,
            ccl_manager=self.dit_ccl_manager,
            parallel_config=self.parallel_config,
            is_fsdp=True,
        )

        if self.use_cache:
            cache_path = get_and_create_cache_path(
                model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                subfolder="transformer_2",
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                dtype="bf16",
            )
            # create cache if it doesn't exist
            if not cache_dict_exists(cache_path):
                logger.info(
                    f"Cache does not exist. Creating cache: {cache_path} and loading transformer weights from PyTorch state dict"
                )
                self.transformer_2.load_state_dict(self.torch_transformer_2.state_dict())
                save_cache_dict(self.transformer_2.to_cached_state_dict(cache_path), cache_path)
            else:
                logger.info(f"Loading transformer weights from cache: {cache_path}")
                self.transformer_2.from_cached_state_dict(load_cache_dict(cache_path))
        else:
            logger.info("Loading transformer weights from PyTorch state dict")
            self.transformer_2.load_state_dict(self.torch_transformer_2.state_dict())

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # device = device or self._execution_device
        device = "cpu"
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

        # NOTE: while the reference impl does not pad to max_sequence_length, for some reason this seems to be necessary for correctness in this pipeline.
        # TODO: investigate
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        # device = device or self._execution_device
        device = "cpu"

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # device = self._execution_device
        device = "cpu"

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        # prompt_embeds = prompt_embeds.to(transformer_dtype)
        # if negative_prompt_embeds is not None:
        #     negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        # num_channels_latents = (
        #     self.transformer.config.in_channels
        #     if self.transformer is not None
        #     else self.transformer_2.config.in_channels
        # )
        num_channels_latents = self.torch_transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    if self.dynamic_load:
                        if hasattr(self, "transformer_2"):
                            del self.transformer_2
                        if not hasattr(self, "transformer"):
                            self._load_transformer1()
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    if self.dynamic_load:
                        if hasattr(self, "transformer"):
                            # Offload transformer1 to make space for transformer2
                            del self.transformer
                        if not hasattr(self, "transformer_2"):
                            self._load_transformer2()
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                # latent_model_input = latents.to(transformer_dtype)
                latent_model_input = latents.clone()
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                logger.info(f"step {i} of {num_inference_steps}")
                logger.info(f"latent_model_input.shape: {latent_model_input.shape}")
                logger.info(f"timestep.shape: {timestep.shape}")
                logger.info(f"prompt_embeds.shape: {prompt_embeds.shape}")
                logger.info(f"negative_prompt_embeds.shape: {negative_prompt_embeds.shape}")
                logger.info(f"attention_kwargs: {attention_kwargs}")

                # with current_model.cache_context("cond"):
                noise_pred = current_model(
                    spatial=latent_model_input,
                    timestep=timestep,
                    prompt=prompt_embeds,
                    # attention_kwargs=attention_kwargs,
                    # return_dict=False,
                )

                if self.do_classifier_free_guidance:
                    # with current_model.cache_context("uncond"):
                    noise_uncond = current_model(
                        spatial=latent_model_input,
                        timestep=timestep,
                        prompt=negative_prompt_embeds,
                        # attention_kwargs=attention_kwargs,
                        # return_dict=False,
                    )
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            # VAE on device
            tt_latents_BTHWC = latents.permute(0, 2, 3, 4, 1)
            tt_latents_BTHWC = conv_pad_in_channels(tt_latents_BTHWC)
            tt_latents_BTHWC, logical_h = conv_pad_height(
                tt_latents_BTHWC, self.vae_parallel_config.height_parallel.factor
            )
            tt_latents_BTHWC = bf16_tensor_2dshard(
                tt_latents_BTHWC,
                self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping={
                    self.vae_parallel_config.height_parallel.mesh_axis: 2,
                    self.vae_parallel_config.width_parallel.mesh_axis: 3,
                },
            )
            tt_video_BCTHW, new_logical_h = self.tt_vae(tt_latents_BTHWC, logical_h)

            concat_dims = [None, None]
            concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
            concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
            video_torch = ttnn.to_torch(
                tt_video_BCTHW,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=concat_dims
                ),
            )
            video_torch = video_torch[:, :, :, :new_logical_h, :]

            video = self.video_processor.postprocess_video(video_torch, output_type=output_type)
        else:
            video = latents

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
