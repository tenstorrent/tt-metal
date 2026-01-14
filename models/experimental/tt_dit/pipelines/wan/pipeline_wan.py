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
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

import ttnn
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler
from ...parallel.manager import CCLManager
from ...parallel.config import DiTParallelConfig, VaeHWParallelConfig, ParallelFactor
from ...models.transformers.wan2_2.transformer_wan import WanTransformer3DModel
from ...models.vae.vae_wan2_1 import WanDecoder
from ...utils import cache
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
        checkpoint_name: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        boundary_ratio: Optional[float] = 0.875,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
        dynamic_load=False,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        is_fsdp: bool = True,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer", trust_remote_code=True)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            checkpoint_name, subfolder="text_encoder", trust_remote_code=True
        )
        self.vae = AutoencoderKLWan.from_pretrained(checkpoint_name, subfolder="vae", trust_remote_code=True)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            checkpoint_name, subfolder="scheduler", trust_remote_code=True
        )
        self.torch_transformer = TorchWanTransformer3DModel.from_pretrained(
            checkpoint_name, subfolder="transformer", trust_remote_code=True
        )
        self.torch_transformer_2 = TorchWanTransformer3DModel.from_pretrained(
            checkpoint_name, subfolder="transformer_2", trust_remote_code=True
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

        self.is_fsdp = is_fsdp
        self.parallel_config = parallel_config
        self.vae_parallel_config = vae_parallel_config
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

    @staticmethod
    def create_pipeline(
        mesh_device,
        checkpoint_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        sp_axis=None,
        tp_axis=None,
        num_links=None,
        dynamic_load=None,
        topology=None,
        is_fsdp=None,
    ):
        device_configs = {}
        if ttnn.device.is_blackhole():
            device_configs[(1, 4)] = {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
            device_configs[(1, 8)] = {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 2,
                "dynamic_load": False,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": False,
            }
        else:
            device_configs[(2, 4)] = {
                "sp_axis": 0,
                "tp_axis": 1,
                "num_links": 1,
                "dynamic_load": True,
                "topology": ttnn.Topology.Linear,
                "is_fsdp": True,
            }
            device_configs[(4, 8)] = {
                "sp_axis": 1,
                "tp_axis": 0,
                "num_links": 4,
                "dynamic_load": False,
                "topology": ttnn.Topology.Ring,
                "is_fsdp": True,
            }

        config = device_configs[tuple(mesh_device.shape)]

        sp_axis = sp_axis or config["sp_axis"]
        tp_axis = tp_axis or config["tp_axis"]

        parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
            sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
            cfg_parallel=None,
        )
        vae_parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(
                factor=tuple(mesh_device.shape)[tp_axis],
                mesh_axis=tp_axis,
            ),
            width_parallel=ParallelFactor(
                factor=tuple(mesh_device.shape)[sp_axis],
                mesh_axis=sp_axis,
            ),
        )

        return WanPipeline(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            num_links=num_links or config["num_links"],
            boundary_ratio=0.875,
            dynamic_load=dynamic_load if dynamic_load is not None else config["dynamic_load"],
            topology=topology or config["topology"],
            is_fsdp=is_fsdp if is_fsdp is not None else config["is_fsdp"],
            checkpoint_name=checkpoint_name,
        )

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
            is_fsdp=self.is_fsdp,
        )

        if not cache.initialize_from_cache(
            self.transformer,
            self.torch_transformer.state_dict(),
            "Wan2.2-T2V-A14B-Diffusers",
            "transformer",
            self.parallel_config,
            tuple(self.mesh_device.shape),
        ):
            logger.info("Loading transformer weights from PyTorch state dict")
            self.transformer.load_torch_state_dict(self.torch_transformer.state_dict())

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
            is_fsdp=self.is_fsdp,
        )

        if not cache.initialize_from_cache(
            self.transformer_2,
            self.torch_transformer_2.state_dict(),
            "Wan2.2-T2V-A14B-Diffusers",
            "transformer_2",
            self.parallel_config,
            tuple(self.mesh_device.shape),
        ):
            logger.info("Loading transformer weights from PyTorch state dict")
            self.transformer_2.load_torch_state_dict(self.torch_transformer_2.state_dict())

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

        latents = torch.randn(shape, dtype=torch.float32, device=torch.device(device))
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
        negative_prompt: Union[
            str, List[str]
        ] = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        guidance_scale_2: Optional[float] = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
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
        traced: bool = False,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
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
            seed (`int`, *optional*):
                A random generator seed to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `seed`.
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
        if profiler:
            profiler.start("encoder", profiler_iteration)
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
        if profiler:
            profiler.end("encoder", profiler_iteration)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.torch_transformer.config.in_channels
        if seed is not None:
            torch.manual_seed(seed)
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
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

        if profiler:
            profiler.start("denoising", profiler_iteration)

        permuted_latent = None
        rope_args = None
        prompt_embeds_map = {"transformer": None, "transformer_2": None}
        negative_prompt_embeds_map = {"transformer": None, "transformer_2": None}
        current_model_name = None

        latent_frames, latent_height, latent_width = latents.shape[2], latents.shape[3], latents.shape[4]

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
                    current_model_name = "transformer"
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
                    current_model_name = "transformer_2"
                    current_guidance_scale = guidance_scale_2

                if permuted_latent is None:
                    # First iteration, preprocess spatial input and prepare rope features
                    permuted_latent, patchified_seqlen = current_model.preprocess_spatial_input_host(latents)

                    rope_cos_1HND, rope_sin_1HND, trans_mat = current_model.prepare_rope_features(latents)
                    rope_args = {
                        "rope_cos_1HND": rope_cos_1HND,
                        "rope_sin_1HND": rope_sin_1HND,
                        "trans_mat": trans_mat,
                    }

                # Cache text conditioning
                if prompt_embeds_map[current_model_name] is None:
                    prompt_embeds_map[current_model_name] = current_model.prepare_text_conditioning(prompt_embeds)
                if self.do_classifier_free_guidance and negative_prompt_embeds_map[current_model_name] is None:
                    negative_prompt_embeds_map[current_model_name] = current_model.prepare_text_conditioning(
                        negative_prompt_embeds
                    )

                # latent_model_input = latents.to(transformer_dtype)
                # latent_model_input = latents.clone()
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                permuted_noise_pred = current_model.inner_step(
                    spatial_1BNI_torch=permuted_latent,
                    prompt_1BLP=prompt_embeds_map[current_model_name],
                    N=patchified_seqlen,
                    timestep_torch=timestep,
                    **rope_args,
                )

                if self.do_classifier_free_guidance:
                    permuted_noise_uncond = current_model.inner_step(
                        spatial_1BNI_torch=permuted_latent,
                        prompt_1BLP=negative_prompt_embeds_map[current_model_name],
                        N=patchified_seqlen,
                        timestep_torch=timestep,
                        **rope_args,
                    )
                    permuted_noise_pred = permuted_noise_uncond + current_guidance_scale * (
                        permuted_noise_pred - permuted_noise_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                permuted_latent = self.scheduler.step(permuted_noise_pred, t, permuted_latent, return_dict=False)[0]

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
        if profiler:
            profiler.end("denoising", profiler_iteration)

        self._current_timestep = None

        # Postprocess spatial output
        latents = current_model.postprocess_spatial_output_host(
            permuted_latent, F=latent_frames, H=latent_height, W=latent_width, N=patchified_seqlen
        )

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
            if profiler:
                profiler.start("vae", profiler_iteration)
            tt_video_BCTHW, new_logical_h = self.tt_vae(tt_latents_BTHWC, logical_h)
            if profiler:
                profiler.end("vae", profiler_iteration)

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

    def run_single_prompt(self, *args, **kwargs):
        return self.__call__(*args, **kwargs).frames

    def synchronize_devices(self):
        ttnn.synchronize_device(self.mesh_device)
