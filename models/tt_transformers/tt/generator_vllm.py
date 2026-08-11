# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import List, Mapping, Union

import torch
from loguru import logger
from PIL.Image import Image
from tqdm import tqdm
from vllm.model_executor.models.gemma3_mm import (
    Gemma3DummyInputsBuilder,
    Gemma3MultiModalProcessor,
    Gemma3ProcessingInfo,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.mistral3 import (
    Mistral3DummyInputsBuilder,
    Mistral3MultiModalProcessor,
    Mistral3ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseDummyInputsBuilder

try:
    # vLLM >= 0.24.0 exposes MultiModalDataDict from vllm.inputs; older
    # versions export it from vllm.multimodal.inputs.
    from vllm.inputs import MultiModalDataDict
except ImportError:
    from vllm.multimodal.inputs import MultiModalDataDict

import ttnn
from models.common.llama_models import create_vision_mask
from models.common.utility_functions import is_wormhole_b0, nearest_32
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, TensorGroup


def allocate_vllm_kv_cache_per_layer(per_layer_specs, dp_model: List[Transformer], tt_cache_path):
    """Allocate KV cache tensors with optional cross-layer DRAM sharing.

    Args:
        per_layer_specs: list of ``(kv_cache_shape, dtype, tensor_idx)``
            triples, one per layer in model layer-index order. Layers with
            the same ``tensor_idx`` share one underlying TT tensor — this
            is upstream's HMA tensor-sharing layout (e.g. for Gemma3 5:1,
            one full-attention layer and several sliding-window layers
            collapse to a single DRAM buffer; per-group block tables keep
            their slot accesses disjoint at runtime). Layers with unique
            ``tensor_idx`` get their own buffer.
        dp_model: list of replicated TT model handles, one per data-parallel
            submesh.
        tt_cache_path: path used for on-disk weight cache file naming.

    Returns:
        ``list[submesh][layer_idx][k_or_v]`` of TT tensors. Multiple
        ``layer_idx`` entries may refer to the same underlying tensor
        objects when they share a ``tensor_idx``.
    """
    submesh_devices = [model.mesh_device for model in dp_model]
    kv_cache = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        # tensor_idx -> [k, v] ttnn handles; reused across all layers that
        # share a buffer.
        unique_buffers: dict[int, list] = {}
        kv_tt = []
        for layer_num, (kv_cache_shape, dtype, tensor_idx) in enumerate(
            tqdm(per_layer_specs, desc=f"Allocating TT kv caches for each layer (submesh {mesh_idx+1})")
        ):
            existing = unique_buffers.get(tensor_idx)
            if existing is not None:
                kv_tt.append(existing)
                continue
            cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
            # Get the dtype for the kv cache based on the configured optimizations in the model
            if dp_model[mesh_idx].args.optimizations is not None:
                kv_cache_dtype = dp_model[mesh_idx].args.optimizations.get_tensor_dtype(
                    decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
                )
            else:
                logger.info("No dtype specified for the model KV cache - defaulting to ttnn.bfloat8_b.")
                kv_cache_dtype = None
            # Set default to bfloat8_b when no optimizations are configured
            kv_cache_dtype = ttnn.bfloat8_b if kv_cache_dtype is None else kv_cache_dtype
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    # TODO: this could be ShardTensorToMesh, removing the need for vLLM to know about TP for num_kv_heads.
                    # Could affect other calculations which use TTCacheEngine.num_kv_heads, though.
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_cache_dtype,
                    # Separate cache files for K and V to avoid collision.
                    # ``tensor_idx`` distinguishes shared buffers that have the
                    # same shape but back different layer subsets.
                    cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}_t{tensor_idx}",
                )
                for kv in ["k", "v"]
            ]

            unique_buffers[tensor_idx] = kv_tt_i
            kv_tt.append(kv_tt_i)
        kv_cache.append(kv_tt)
    return kv_cache


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, dp_model: List[Transformer], tt_cache_path):
    """Uniform-shape KV cache allocator for non-hybrid models.

    Hybrid attention models should use :func:`allocate_vllm_kv_cache_per_layer`,
    which takes a per-layer ``(shape, dtype, tensor_idx)`` list so layers
    can share DRAM buffers per upstream's HMA tensor-sharing model.
    """
    return allocate_vllm_kv_cache_per_layer(
        [(kv_cache_shape, dtype, i) for i in range(num_layers)],
        dp_model=dp_model,
        tt_cache_path=tt_cache_path,
    )


class HybridAttentionForCausalLM(Generator):
    """vLLM wrapper base for hybrid attention models.

    Models with mixed sliding-window + full-attention layers (Gemma3,
    Gemma4, GPT-OSS, ...) inherit from this class instead of plain
    :class:`Generator` so they can opt in to upstream's hybrid kv cache
    manager. The shared ``get_kv_cache_spec`` classmethod here builds
    the per-layer KV cache spec from ``hf_config.text_config.layer_types``
    — the standard HF convention used by all of these models — emitting
    ``SlidingWindowSpec`` for sliding layers and ``FullAttentionSpec``
    for full-attention layers.

    Subclasses are responsible for the model-specific pieces:

    * ``initialize_vllm_model``: load the underlying TT model.
    * ``prefill_forward`` / ``decode_forward``: consume the
      ``page_tables_per_layer`` list (one tensor per decoder layer, layer-
      aligned with the model's ``self.layers``) and pass each entry to its
      corresponding attention layer. The plugin pre-expands
      ``block_tables_per_group`` into this per-layer view at submission
      time so bridges don't have to re-derive vLLM's group construction
      order — see ``TTModelRunner._block_tables_per_layer``.
    * ``allocate_kv_cache_per_layer``: typically just delegates to
      :func:`allocate_vllm_kv_cache_per_layer` with the model handles.

    Until a subclass overrides them, ``prefill_forward`` and
    ``decode_forward`` raise :class:`NotImplementedError` to make the
    contract explicit. Legacy (non-hybrid) models never see the
    ``page_tables_per_layer`` kwarg — vLLM's plugin opts in via the
    presence of ``get_kv_cache_spec`` on the model class.
    """

    # Keep this in sync with get_kv_cache_spec below and with the TT vLLM
    # worker's token-budget calculation. While SlidingWindowSpec is disabled,
    # vLLM produces one full-attention KV group and the legacy single page_table
    # path is sufficient. When SlidingWindowSpec is restored, flip this back so
    # warmup also exercises the per-layer persistent page-table path.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Build per-layer KVCacheSpec from HF config ``layer_types``.

        Returns a dict keyed by ``model.layers.<idx>.self_attn`` (the
        upstream attention-layer naming convention vLLM's KVCacheGroup
        machinery understands) so :func:`_parse_layer_index` on the
        runner side can map each spec back to its model layer index.
        """
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

        # SlidingWindowSpec import intentionally dropped; restore alongside the
        # branch below when re-enabling kv cache groups.
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise ValueError(
                f"{cls.__name__}.get_kv_cache_spec requires "
                "hf_config.text_config.layer_types (one of 'full_attention' / "
                "'sliding_attention' per layer); none found on this model"
            )
        num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        block_size = cache_config.block_size

        common = dict(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )

        # SlidingWindowSpec is temporarily disabled: TT-side decode passes the
        # absolute position to paged_update_cache / paged_sdpa_decode, but vLLM
        # zero-pads the sliding group's page_table past sliding_window/block_size
        # entries, so positions beyond the sliding window collapse onto physical
        # block 0 and silently corrupt the cache. Emit FullAttentionSpec for every
        # layer so vLLM allocates a max_model_len cache per layer; the SDPA op's
        # own sliding_window_size kwarg still trims attention correctly on the
        # read side.
        spec_per_layer = {}
        for i, lt in enumerate(layer_types):
            name = f"model.layers.{i}.self_attn"
            if lt not in ("sliding_attention", "full_attention"):
                raise ValueError(
                    f"Unsupported layer_type {lt!r} at layer {i} on "
                    f"{cls.__name__}; expected 'full_attention' or "
                    "'sliding_attention'"
                )
            spec_per_layer[name] = FullAttentionSpec(**common)
        return spec_per_layer

    def prefill_forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must override prefill_forward to consume "
            "`page_tables_per_layer` and pass each entry to the matching "
            "attention layer."
        )

    def decode_forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must override decode_forward to consume "
            "`page_tables_per_layer` and pass each entry to the matching "
            "attention layer."
        )

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return allocate_vllm_kv_cache_per_layer(per_layer_specs, dp_model=self.model, tt_cache_path=self.cache_path)

    def _ensure_page_tables_per_layer(self, page_tables_per_layer, page_table):
        """When invoked outside the vLLM hybrid plugin (e.g. by warmup
        which only knows about the legacy single ``page_table``), optionally
        broadcast the single page table to a per-layer list.

        Broadcasting is only correct while hybrid KV cache groups are enabled:
        trace capture then needs to exercise the per-layer code path inside
        ``Transformer.forward`` so replay reads the persistent per-layer device
        tensors updated before each call. While hybrid groups are temporarily
        disabled, all layers use one full-attention KV group, so we intentionally
        keep warmup/runtime on the legacy single-page-table path.
        """
        if page_tables_per_layer is not None or page_table is None or not self._HYBRID_KV_CACHE_GROUPS_ENABLED:
            return page_tables_per_layer
        # Broadcast the same torch tensor across every layer in every
        # submesh — content is identical, persistent allocation gives each
        # layer its own device tensor at a stable address.
        num_layers = len(self.model[0].layers)
        return [page_table] * num_layers

    def _chunk_page_tables_per_dp(self, page_tables_per_layer):
        """Split a global per-layer list along DP into one per-layer list
        per submesh.

        The plugin pads each per-layer table to the global
        ``(max_num_seqs * data_parallel, max_num_blocks_per_req)`` shape
        (see ``TTModelRunner._block_tables_per_layer``); warmup likewise
        builds a global-batch tensor. ``Generator.decode_forward`` already
        does ``torch.chunk(page_table, self.data_parallel, 0)`` for the
        legacy single-page-table path before the per-submesh
        ``prepare_inputs_decode``; the hybrid bridge has to do the
        equivalent so each submesh's ``_page_tables_to_ttnn`` receives a
        per-DP slice whose batch dim matches the submesh's K/V tensors.
        Without this, ``paged_update_cache`` asserts a batch-size mismatch
        on multi-DP runs.
        """
        if page_tables_per_layer is None:
            return None
        dp = self.data_parallel
        if dp <= 1:
            return [page_tables_per_layer]
        per_submesh = [list() for _ in range(dp)]
        for pt in page_tables_per_layer:
            if pt is None or isinstance(pt, ttnn.Tensor):
                # Already-resolved or absent entries pass through unchanged
                # to every submesh — chunking only applies to torch tensors
                # carrying global batch.
                for s in per_submesh:
                    s.append(pt)
                continue
            chunks = torch.chunk(pt, dp, dim=0)
            for s, c in zip(per_submesh, chunks):
                s.append(c)
        return per_submesh

    def _route_per_layer_page_tables(self, per_submesh_page_tables):
        """Stash each submesh's per-layer page-table list on its model
        handle for the duration of a forward call.

        ``Generator``'s prefill/decode paths invoke
        ``model[i].ttnn_prefill_forward`` / ``ttnn_decode_forward`` from
        many sites (warmup, trace capture, traced replay, etc.) without
        forwarding an arbitrary kwarg. Threading the per-layer list through
        every site would be a wide change for a feature only this hybrid
        bridge consumes, so we use a localised attribute injection: each
        model reads ``getattr(self, "_active_page_tables_per_layer", None)``
        when its own kwarg is None. ``per_submesh_page_tables[i]`` is the
        per-layer slice that submesh ``i`` should see; ``None`` clears the
        stash entirely (legacy fallback).
        """

        class _Stash:
            def __init__(self, models, per_submesh):
                self._models = models
                self._per_submesh = per_submesh

            def __enter__(self):
                if self._per_submesh is None:
                    return
                for m, value in zip(self._models, self._per_submesh):
                    m._active_page_tables_per_layer = value

            def __exit__(self, *_):
                if self._per_submesh is None:
                    return
                for m in self._models:
                    if hasattr(m, "_active_page_tables_per_layer"):
                        del m._active_page_tables_per_layer

        return _Stash(self.model, per_submesh_page_tables)


def initialize_vllm_text_transformer(
    hf_config,
    tt_data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
    optimizations=DecodersPrecision.performance,
):
    submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
    # Load model args, weights
    model_args = []
    for submesh in submesh_devices:
        model_args_i = ModelArgs(
            submesh,
            instruct=(
                "Instruct" in hf_config._name_or_path or "DeepSeek-R1-Distill-Llama-70B" in hf_config._name_or_path
            ),
            max_batch_size=max_batch_size // tt_data_parallel,
            optimizations=lambda model_args: optimizations(model_args.n_layers, model_args.model_name),
            max_seq_len=max_seq_len,
        )

        assert model_args_i.model_name.replace("-", "") in hf_config._name_or_path.replace(
            "-", ""
        ), f"The model specified in vLLM ({hf_config._name_or_path}) does not match the model name ({model_args_i.model_name}) with model weights ({model_args_i.CKPT_DIR})."
        if n_layers is not None:
            model_args_i.n_layers = n_layers

        model_args.append(model_args_i)

    state_dict = model_args[0].load_state_dict()

    tt_model = []
    for i, submesh in enumerate(submesh_devices):
        tt_model_i = Transformer(
            args=model_args[i],
            mesh_device=submesh,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args[i].weight_cache_path(dtype),
            use_paged_kv_cache=True,
        )
        tt_model.append(tt_model_i)

    return tt_model, model_args


class DummyInputsBuilder(BaseDummyInputsBuilder):
    """
    We don't need to implement a dummy input builder since we don't do profiling in vLLM.
    Create callable class just for processor registration.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        raise NotImplementedError

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        raise NotImplementedError


class CustomNamespace(SimpleNamespace):
    def __contains__(self, key):
        return key in self.__dict__


@MULTIMODAL_REGISTRY.register_processor(
    Mistral3MultiModalProcessor,
    info=Mistral3ProcessingInfo,
    dummy_inputs=Mistral3DummyInputsBuilder,
)
class Mistral3ForConditionalGeneration(Generator, SupportsMultiModal):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.MISTRAL_IMAGE_TOKEN_ID = 151655
        self.max_gen_len = self.model_args[0].max_seq_len - 1

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=131072,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        from models.tt_transformers.demo.simple_vision_demo import create_multimodal_model

        max_seq_len = 1024 * 128

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None

        for submesh in submesh_devices:
            model_args_i, model_i, state_dict = create_multimodal_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                use_paged_kv_cache=True,
                checkpoint=state_dict,
            )
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        self.tokenizer = self.model_args[0].tokenizer
        pad_token_id = self.tokenizer.pad_token_id

        tokens = kwargs["tokens"]
        prompt_lens = kwargs["prompt_lens"]
        inputs = CustomNamespace()
        inputs.input_ids = tokens
        data = kwargs.get("images", None)
        for i in range(tokens.shape[0]):
            tokens[i][prompt_lens[i] :] = pad_token_id
        pixel_values, image_sizes = None, None

        if data and hasattr(data[0], "pixel_values"):
            pixel_values = [im.pixel_values for im in data if hasattr(im, "pixel_values")]
            image_sizes = [im.image_sizes for im in data if hasattr(im, "image_sizes")]

        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)

        return super().prefill_forward_text(
            tokens=inputs.input_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            pixel_values=pixel_values if pixel_values else None,
            image_sizes=image_sizes if image_sizes else None,
        )

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


# Mllama is currently not supported in vLLM V1.
# TODO: Remove or re-enable when Mllama is supported in vLLM V1.
# @MULTIMODAL_REGISTRY.register_processor(
#     MllamaMultiModalProcessor, info=TT_MllamaProcessingInfo, dummy_inputs=DummyInputsBuilder
# )
class MllamaForConditionalGeneration(Generator, SupportsMultiModal):
    # Class-level capabilities
    # Note: Mllama doesn't support prefix caching (it's V0 only)
    # decode_forward calls decode_forward_llama_vision and discards anything
    # but logits, so sampling_params never reach a sampler — explicitly
    # declare on-device sampling unsupported.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": False,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        **kwargs,
    ) -> int:
        """Returns config-specific all-user KV-cache token capacity."""
        devices_per_dp_cache = num_devices // tt_data_parallel
        is_wormhole = is_wormhole_b0()

        # Llama90B on WH T3K
        if "Llama-3.2-90B" in model_name and devices_per_dp_cache == 8 and is_wormhole:
            return 65_536
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            **kwargs,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.MLLAMA_IMAGE_TOKEN_ID = 128256
        self.max_gen_len = self.model_args[0].max_seq_len - 1  # TODO: double check what this should be

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations: str = None
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        from models.tt_transformers.demo.simple_vision_demo import create_multimodal_model

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None

        for submesh in submesh_devices:
            model_args_i, model_i, state_dict = create_multimodal_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                use_paged_kv_cache=True,
                checkpoint=state_dict,
            )
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    @property
    def max_cross_attn_tokens(self):
        return self.model_args[0].vision_max_num_chunks * nearest_32(self.model_args[0].vision_chunk_ntok)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        images: Union[List[Image], List[List[Image]]],
        page_table: torch.Tensor,
        kv_cache,
        prompt_lens,
        cross_page_table: torch.Tensor,
    ):
        """
        Replaces prefill_forward from Generator with a version that supports mask creation.
        """
        batch = tokens.shape[0]

        vision_images = []
        vision_masks = []
        total_lens = []
        for user_id in range(batch):
            image = images[user_id]
            if isinstance(image, list):
                assert len(image) == 1, "Only one image is supported for each user in the batch"
                image = image[0]
            vision_images.append([image] if image else None)
            prompt_tokens = [int(tokens[user_id, i]) for i in range(prompt_lens[user_id])]
            vision_masks.append(create_vision_mask(prompt_tokens, self.MLLAMA_IMAGE_TOKEN_ID) if image else None)
            total_lens.append(prompt_lens[user_id] + self.max_gen_len)

        return super().prefill_forward(
            vision_images,
            vision_masks,
            tokens,
            None,
            total_lens,
            prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
            cross_page_table=cross_page_table,
        )

    def decode_forward(self, *args, **kwargs):
        logits = super().decode_forward_llama_vision(*args, **kwargs)
        if isinstance(logits, tuple):
            return logits[0]
        else:
            return logits

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class LlamaForCausalLM(Generator):
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        **kwargs,
    ) -> int:
        """Returns config-specific all-user KV-cache token capacity."""
        devices_per_dp_cache = num_devices // tt_data_parallel
        is_wormhole = is_wormhole_b0()

        # Llama8B on N150
        if "Llama-3.1-8B" in model_name and devices_per_dp_cache == 1 and is_wormhole:
            return 32_768
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            **kwargs,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        hf_model_name = hf_config._name_or_path
        if (
            ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
            and mesh_device.get_num_devices() == 1
            and is_wormhole_b0()
        ):
            MAX_PROMPT_LEN = 32768
            if max_seq_len > MAX_PROMPT_LEN:
                raise ValueError(
                    f"TT-LLama8B and TT-Llama11B do not support max_model_len greater than {MAX_PROMPT_LEN} on N150 "
                    f"(received {max_seq_len}). Set --max_model_len to {MAX_PROMPT_LEN} or lower in vLLM."
                )

        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class QwenForCausalLM(Generator):
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        **kwargs,
    ) -> int:
        """Returns config-specific all-user KV-cache token capacity."""
        devices_per_dp_cache = num_devices // tt_data_parallel
        is_wormhole = is_wormhole_b0()

        # Qwen3-8B on N150 (same constraint as Llama8B-N150)
        if "Qwen3-8B" in model_name and devices_per_dp_cache == 1 and is_wormhole:
            return 32_768
        # DeepSeek-R1-Distill-Qwen-14B / Qwen2.5-14B on N300
        if (
            ("DeepSeek-R1-Distill-Qwen-14B" in model_name or "Qwen2.5-14B" in model_name)
            and devices_per_dp_cache == 2
            and is_wormhole
        ):
            return 65_536
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            **kwargs,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class MistralForCausalLM(Generator):
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        **kwargs,
    ) -> int:
        """Returns config-specific all-user KV-cache token capacity."""
        devices_per_dp_cache = num_devices // tt_data_parallel
        is_wormhole = is_wormhole_b0()

        # Mistral-7B on N150
        if "Mistral-7B" in model_name and devices_per_dp_cache == 1 and is_wormhole:
            return 65_536
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            **kwargs,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=DecodersPrecision.from_string(optimizations)
            if optimizations is not None
            else DecodersPrecision.performance,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


@MULTIMODAL_REGISTRY.register_processor(
    Gemma3MultiModalProcessor,
    info=Gemma3ProcessingInfo,
    dummy_inputs=Gemma3DummyInputsBuilder,
)
class Gemma3ForConditionalGeneration(HybridAttentionForCausalLM, SupportsMultiModal):
    """Gemma3 multimodal — hybrid attention (sliding-window + full).

    Gemma3's text decoder alternates ``sliding_attention`` and
    ``full_attention`` per ``hf_config.text_config.layer_types`` (a 5:1
    ratio in the 27B variant), so the bridge inherits from
    :class:`HybridAttentionForCausalLM` to opt into vLLM's hybrid kv cache
    manager. Sliding-window layers index a smaller paged pool than
    full-attention layers — the per-layer KV cache shape difference is
    where the asymmetric-hybrid memory savings live, and was the original
    motivation for kv-cache-groups (it's what unblocks the 62-layer × 107
    MB-per-layer DRAM OOM on T3K seen in run 25437459815).

    Mirrors the ``GptOssForCausalLM`` plumbing: ``prefill_forward`` /
    ``decode_forward`` stash ``page_tables_per_layer`` on each
    ``self.model[i]`` for the duration of a single
    ``super().{prefill_forward_text,decode_forward}`` call. The underlying
    ``Transformer.ttnn_*_forward`` (which ``TtGemmaModel`` inherits)
    picks up the stash via ``_active_page_tables_per_layer`` and routes
    each layer's attention to its own page table.
    """

    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        **kwargs,
    ) -> int:
        """Returns config-specific all-user KV-cache token capacity."""
        devices_per_dp_cache = num_devices // tt_data_parallel
        is_wormhole = is_wormhole_b0()

        # gemma-3-4b on wormhole configurations with up to 2 devices per DP shard
        if "gemma-3-4b" in model_name.lower() and devices_per_dp_cache in (1, 2) and is_wormhole:
            return 65_536
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            **kwargs,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=131072,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        from models.demos.multimodal.gemma3.demo.vision_demo import create_multimodal_model

        optimizations = (
            DecodersPrecision.from_string(optimizations) if optimizations is not None else DecodersPrecision.performance
        )

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None

        for submesh in submesh_devices:
            model_args_i, model_i, state_dict = create_multimodal_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                use_paged_kv_cache=True,
                checkpoint=state_dict,
                optimizations=lambda model_args: optimizations(model_args.n_layers, model_args.model_name),
            )
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        # While hybrid KV cache groups are disabled (one full-attention group
        # for every layer), the per-layer page-table routing inside this
        # bridge is buggy for users_row_sharded models: it shards page tables
        # naively by mesh row, which doesn't match the gpt-oss
        # slot // max_local_batch_size → row mapping and produces null
        # content on the rows whose page-table chunks point at the wrong
        # KV blocks. Until a row-aware per-layer routing lands, skip the
        # hybrid path entirely and let the legacy single page_table flow
        # through Generator.prefill_forward_text reach the model untouched.
        if not self._HYBRID_KV_CACHE_GROUPS_ENABLED:
            return super().prefill_forward_text(*args, **kwargs)
        page_tables_per_layer = self._ensure_page_tables_per_layer(page_tables_per_layer, kwargs.get("page_table"))
        per_submesh = self._chunk_page_tables_per_dp(page_tables_per_layer)
        # Push the per-layer block IDs into the persistent device buffers
        # *before* entering ``Generator.prefill_forward_text`` — that path
        # may execute a captured trace, which reads block IDs from the
        # persistent addresses and forbids in-trace writes. Allocation
        # itself happens lazily in ``Transformer._page_tables_to_ttnn``
        # the first time the inner forward runs (warmup compile).
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                m.update_persistent_per_layer_page_tables(pt_for_submesh)
        with self._route_per_layer_page_tables(per_submesh):
            return super().prefill_forward_text(**kwargs)

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        # See prefill_forward note above. Skip the hybrid path while
        # _HYBRID_KV_CACHE_GROUPS_ENABLED is False.
        if not self._HYBRID_KV_CACHE_GROUPS_ENABLED:
            return super(HybridAttentionForCausalLM, self).decode_forward(*args, **kwargs)
        page_tables_per_layer = self._ensure_page_tables_per_layer(page_tables_per_layer, kwargs.get("page_table"))
        per_submesh = self._chunk_page_tables_per_dp(page_tables_per_layer)
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                m.update_persistent_per_layer_page_tables(pt_for_submesh)
        with self._route_per_layer_page_tables(per_submesh):
            # Skip ``HybridAttentionForCausalLM.decode_forward``, which is a
            # NotImplementedError placeholder; route to ``Generator``'s
            # actual decode implementation.
            return super(HybridAttentionForCausalLM, self).decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


class GptOssForCausalLM(HybridAttentionForCausalLM):
    """GPT-OSS model for vLLM integration.

    GPT-OSS is a hybrid attention model — its layers alternate between
    full attention and sliding-window attention per ``hf_config.layer_types``.
    Inheriting from :class:`HybridAttentionForCausalLM` opts into vLLM's
    hybrid kv cache manager so sliding-window layers can index a smaller
    paged pool than full-attention layers, recovering the asymmetric-hybrid
    memory waste described in vLLM's hybrid kv cache manager design.

    The bridge accepts ``page_tables_per_layer`` from the plugin (one tensor
    per decoder layer, layer-aligned with the underlying TT model's
    ``self.layers``) and stashes it on each TT model handle as
    ``_active_page_tables_per_layer`` so the model's ``ttnn_prefill_forward``
    / ``ttnn_decode_forward`` pick it up without us having to thread the
    kwarg through every call site in :class:`Generator`. The attribute is
    cleared on the way out so a subsequent legacy single-page-table call
    isn't accidentally affected.
    """

    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,  # Sliding window => no prefix caching
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        # While hybrid KV cache groups are disabled (one full-attention group
        # for every layer), the per-layer page-table routing inside this
        # bridge is buggy for users_row_sharded models: it shards page tables
        # naively by mesh row, which doesn't match the gpt-oss
        # slot // max_local_batch_size → row mapping and produces null
        # content on the rows whose page-table chunks point at the wrong
        # KV blocks. Until a row-aware per-layer routing lands, skip the
        # hybrid path entirely and let the legacy single page_table flow
        # through Generator.prefill_forward_text reach the model untouched.
        if not self._HYBRID_KV_CACHE_GROUPS_ENABLED:
            return super().prefill_forward_text(*args, **kwargs)
        page_tables_per_layer = self._ensure_page_tables_per_layer(page_tables_per_layer, kwargs.get("page_table"))
        per_submesh = self._chunk_page_tables_per_dp(page_tables_per_layer)
        # See ``Gemma3ForConditionalGeneration.prefill_forward`` for why
        # the persistent-buffer update has to happen *before* the inner
        # decode/prefill path that may run captured traces.
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                m.update_persistent_per_layer_page_tables(pt_for_submesh)
        with self._route_per_layer_page_tables(per_submesh):
            return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        # See prefill_forward note above. Skip the hybrid path while
        # _HYBRID_KV_CACHE_GROUPS_ENABLED is False.
        if not self._HYBRID_KV_CACHE_GROUPS_ENABLED:
            return super(HybridAttentionForCausalLM, self).decode_forward(*args, **kwargs)
        page_tables_per_layer = self._ensure_page_tables_per_layer(page_tables_per_layer, kwargs.get("page_table"))
        per_submesh = self._chunk_page_tables_per_dp(page_tables_per_layer)
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                m.update_persistent_per_layer_page_tables(pt_for_submesh)
        with self._route_per_layer_page_tables(per_submesh):
            # Skip ``HybridAttentionForCausalLM.decode_forward``, which is a
            # NotImplementedError placeholder; route to ``Generator``'s
            # actual decode implementation.
            return super(HybridAttentionForCausalLM, self).decode_forward(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        from models.demos.gpt_oss.tt.common import create_tt_model

        model_args = []
        model = []
        state_dict = None
        # GPT-OSS throughput profile uses user-row sharding on
        # multi-row meshes with large max batch sizes (e.g., 128 on 4x8).
        # This must be selected at model init time to ensure correct sharding
        # and input preparation.
        users_row_sharded = bool(mesh_device.shape[0] > 1 and max_batch_size > 32)
        if users_row_sharded:
            # For users_row_sharded, we internally manage DP=4 in attention so we don't need to create submeshes
            tt_data_parallel = 1
        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
        for submesh in submesh_devices:
            # Use the existing create_tt_model function
            model_args_i, model_i, _, state_dict = create_tt_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                paged_attention_config=None,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
                num_layers=n_layers,
                mesh_config=None,
                create_kv_cache=False,
                users_row_sharded=users_row_sharded,
                use_throughput_experts=submesh.shape[0] > 1 and (max_batch_size > 1),
            )

            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat8_b)

    # prefill_forward / decode_forward are defined above with the
    # per-layer page-table stash; allocate_kv_cache_per_layer is inherited
    # from HybridAttentionForCausalLM.
    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)
