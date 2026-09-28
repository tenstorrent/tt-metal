# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from vllm.platforms.interface import Platform, PlatformEnum
from vllm_tt_plugin.config import (
    get_tt_config,
    get_tt_data_parallel_size,
    store_tt_lane_count,
    uses_tt_lane_coordinator,
    validate_tt_lane_config,
)
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.inputs import ProcessorInputs, PromptType
    from vllm.pooling_params import PoolingParams
    from vllm.renderers.inputs import DictPrompt, TokPrompt
    from vllm.sampling_params import SamplingParams
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = object

logger = init_tt_logger(__name__)

TT_SCHEDULER_CLS = "vllm_tt_plugin.scheduler.TTScheduler"
TT_LANE_SCHEDULER_CLS = "vllm_tt_plugin.lane_scheduler.TTLaneCoordinator"

# TT model versions backed by the single-execute Galaxy generator
# (models.demos.llama3_70b_galaxy.tt.generator:Generator). For these, gathered
# multi-process DP is deprecated in favor of single-process TT lanes. Maps the
# selecting env var to the version value that routes through that generator.
_GALAXY_GENERATOR_VERSIONS = {
    "TT_LLAMA_TEXT_VER": "llama3_70b_galaxy",
    "TT_QWEN3_TEXT_VER": "qwen3_32b_galaxy",
}

# TT model types that have been validated with real chunked prefill.
_CHUNKED_PREFILL_MODEL_TYPES = {"gemma4", "gemma4_unified"}


def _apply_chunked_prefill_policy(vllm_config: "VllmConfig") -> None:
    """Restricts token-chunked prefill to model types Metal has validated."""
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config
    model_type = getattr(model_config.hf_config, "model_type", None)

    if model_type in _CHUNKED_PREFILL_MODEL_TYPES:
        # A chunk boundary inside a multimodal item would split its embeddings
        # from their positions. Only meaningful while prefill can be split, and
        # vLLM rejects the flag outright when one item exceeds the token budget,
        # so it stays off for every model type below.
        scheduler_config.disable_chunked_mm_input = True
        return

    if scheduler_config.enable_chunked_prefill:
        logger.info(
            "Chunked prefill is not validated for `model_type=%s`; disabling it.",
            model_type,
        )
        scheduler_config.enable_chunked_prefill = False

        max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        max_model_len = model_config.max_model_len
        if max_num_batched_tokens < max_model_len:
            logger.warning(
                "`max_num_batched_tokens=%d < max_model_len=%d` with "
                "chunked prefill "
                "disabled, bumping `max_num_batched_tokens` to match.",
                max_num_batched_tokens,
                max_model_len,
            )

            scheduler_config.max_num_batched_tokens = max_model_len

    # The scheduler applies this threshold before checking
    # ``enable_chunked_prefill``. Leaving it nonzero can still split a
    # prefill despite the disabled flag.
    scheduler_config.long_prefill_token_threshold = 0


def _renormalize_mamba_cache_config(vllm_config: "VllmConfig") -> None:
    """Re-apply core's no-prefix-caching Mamba sizing after we turn it off.

    ``MambaModelConfig.verify_and_update_config`` runs as the first statement of
    ``VllmConfig.__post_init__``, far ahead of this platform hook, and sizes
    ``mamba_block_size`` and ``mamba_cache_mode`` against prefix caching being
    enabled. Disabling the flag afterwards leaves a pair that
    ``VllmConfig.validate_mamba_block_size`` rejects outright, so mirror the
    branch core takes when prefix caching is off: that validator reads
    ``mamba_block_size == max_model_len`` as "unset".

    ``block_size`` is deliberately left alone. TT reads it when it builds the KV
    spec, and it already holds the value core's disabled branch computes.
    """
    cache_config = vllm_config.cache_config
    if getattr(cache_config, "mamba_block_size", None) is None:
        return

    cache_config.mamba_cache_mode = "none"
    cache_config.mamba_block_size = vllm_config.model_config.max_model_len
    logger.info(
        "Prefix caching is disabled, so mamba_block_size is reset to "
        "max_model_len=%d and mamba_cache_mode to 'none'.",
        cache_config.mamba_block_size,
    )


def _galaxy_generator_version() -> str | None:
    """Return the active Galaxy-generator model version, or None.

    Both ``llama3_70b_galaxy`` (Llama3 70B) and ``qwen3_32b_galaxy`` (Qwen3-32B)
    are served by ``models.demos.llama3_70b_galaxy.tt.generator:Generator``.
    """
    for env_var, version in _GALAXY_GENERATOR_VERSIONS.items():
        if os.getenv(env_var) == version:
            return version
    return None


def _collapse_parallel_config_to_single_process(parallel_config) -> None:
    """Reset DP-derived ParallelConfig fields to single-process values.

    ``ParallelConfig.__post_init__`` has already derived multi-process DP state
    (rank, local size, master port, LB mode) from ``data_parallel_size`` by the
    time the platform hook runs. When we fold gathered DP into single-process TT
    lanes we must undo that so vLLM does not stand up multi-process DP
    coordination. ``world_size`` stays 1 because the TT backend requires
    ``tensor_parallel_size == pipeline_parallel_size == 1`` and DP does not
    multiply it (no external launcher), so ``world_size_across_dp`` collapses to
    1 automatically once ``data_parallel_size`` is reset.

    ``data_parallel_rank_local`` must be ``0`` here. The TT plugin gates device
    bring-up on ``data_parallel_rank_local == 0``: that rank opens the mesh,
    loads the model, and allocates the KV cache (see
    ``worker.init_device``/``load_model`` and
    ``TTModelRunner.initialize_kv_cache``). A single-process lane run owns the
    one device mesh, so it is that device rank and must be ``0`` for those gates
    to fire and bring the device up. ``0`` is also the value a genuine
    single-process run resolves to (``ParallelConfig.__post_init__`` defaults
    ``data_parallel_rank_local`` from ``VLLM_DP_RANK_LOCAL`` / ``VLLM_DP_RANK``,
    both ``0``).
    """
    parallel_config.data_parallel_size = 1
    parallel_config.data_parallel_size_local = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_rank_local = 0
    parallel_config.data_parallel_index = 0
    parallel_config.data_parallel_external_lb = False
    parallel_config.data_parallel_hybrid_lb = False

    # A single-process lane run owns one in-process worker, so pin the uniproc
    # executor. Newer vLLM derives it from
    # ``world_size_across_dp`` and latches "mp" from the user's
    # --data_parallel_size before this hook runs; pinning "uni" keeps lane-DP
    # single-process there too, so the worker's runtime
    # ``num_gpu_blocks_override`` still reaches the engine's KV-cache sizing.
    parallel_config.distributed_executor_backend = "uni"


def _convert_galaxy_gather_dp_to_lanes(vllm_config: "VllmConfig") -> None:
    """Transparently convert gathered multi-process DP into in-process TT lanes.

    Galaxy-generator models (``llama3_70b_galaxy``, ``qwen3_32b_galaxy``) are
    served by a single Galaxy device mesh. Gathered multi-process DP is
    deprecated in favor of single-process TT lanes. Rather than asking users to
    migrate flags, we run ``--data_parallel_size N`` as ``N`` in-process lanes:
    record the resolved lane count and reset ``data_parallel_size`` to 1.

    To preserve the historical capacity contract -- where each of the ``N``
    gathered DP ranks handled ``max_num_seqs`` requests -- the global
    ``max_num_seqs`` is scaled by the lane count. Lane mode then partitions that
    global capacity evenly across lanes, so the per-lane capacity stays at the
    value the user requested (e.g. ``--data_parallel_size 4 --max_num_seqs 8``
    becomes 4 lanes, each with max 8 seqs, for a global max of 32).

    No-op unless a Galaxy-generator model is active with
    ``data_parallel_size > 1``. Idempotent: after conversion
    ``data_parallel_size == 1``, so re-entry short-circuits.
    """
    parallel_config = vllm_config.parallel_config
    data_parallel_size = parallel_config.data_parallel_size
    if data_parallel_size <= 1:
        return
    galaxy_version = _galaxy_generator_version()
    if galaxy_version is None:
        return

    lanes = data_parallel_size
    scheduler_config = vllm_config.scheduler_config
    per_lane_max_num_seqs = int(scheduler_config.max_num_seqs)
    global_max_num_seqs = per_lane_max_num_seqs * lanes

    store_tt_lane_count(vllm_config, lanes)
    scheduler_config.max_num_seqs = global_max_num_seqs
    _collapse_parallel_config_to_single_process(parallel_config)

    logger.info(
        "Galaxy model %s requested DP "
        "(--data_parallel_size=%d); running single-process TT lane-DP instead "
        "(%d lanes, per-lane max_num_seqs=%d, global max_num_seqs=%d).",
        galaxy_version,
        lanes,
        lanes,
        per_lane_max_num_seqs,
        global_max_num_seqs,
    )


def _register_model_if_missing(ModelRegistry, model_arch: str, model_path: str) -> None:
    """Register `model_arch` only if not already registered.

    This keeps TT model registration idempotent across multiple call sites
    (e.g. APIServer pre-register, TT worker import, and platform config hook).
    """
    if model_arch not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(model_arch, model_path)


def _should_pre_register_tt_test_models_from_cli() -> bool:
    """Return True iff CLI TT config enables TT test models.

    `TTPlatform.pre_register_and_update()` runs before `VllmConfig` is
    constructed, but ModelConfig may inspect architectures early.
    """
    argv = list(sys.argv[1:])

    def _parse_namespaced_config(raw: str) -> dict | None:
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    cfg = None
    for i, arg in enumerate(argv):
        if "=" in arg:
            flag, value = arg.split("=", 1)
            if flag.replace("_", "-") == "--additional-config":
                cfg = _parse_namespaced_config(value) or cfg
        elif arg.replace("_", "-") == "--additional-config" and i + 1 < len(argv):
            cfg = _parse_namespaced_config(argv[i + 1]) or cfg

    tt_config = cfg.get("tt", {}) if cfg else {}
    return bool(
        isinstance(tt_config, dict) and tt_config.get("register_test_models") is True
    )


def _install_tt_harmony_truncation_patch() -> None:
    """Use right truncation for TT GPT-OSS tokenizers.

    GPT-OSS harmony prompts have important template/control tokens at the
    beginning. Left truncation can remove those tokens when prompt truncation is
    requested, so TT keeps the prefix and truncates from the right for these
    models.

    TODO: remove this once fixed in vLLM core.
    """
    import vllm.tokenizers.registry as tokenizer_registry

    if hasattr(tokenizer_registry, "_tt_original_tokenizer_args_from_config"):
        return

    original = tokenizer_registry.tokenizer_args_from_config
    tokenizer_registry._tt_original_tokenizer_args_from_config = original

    def tokenizer_args_from_config_tt(config, **kwargs):
        tokenizer_mode, tokenizer_name, args, tokenizer_kwargs = original(
            config, **kwargs
        )
        if (
            "truncation_side" not in tokenizer_kwargs
            and config.runner_type in ("generate", "draft")
            and "gpt-oss" in str(tokenizer_name or "").lower()
        ):
            tokenizer_kwargs["truncation_side"] = "right"
        return tokenizer_mode, tokenizer_name, args, tokenizer_kwargs

    tokenizer_registry.tokenizer_args_from_config = tokenizer_args_from_config_tt

    renderer_registry = sys.modules.get("vllm.renderers.registry")
    if renderer_registry is not None:
        renderer_registry.tokenizer_args_from_config = tokenizer_args_from_config_tt


def _iter_extra_model_bundles():
    """Yield ``(folder, arch, main_class)`` for each bundle under ``EXTRA_MODELS_DIR``.

    ``EXTRA_MODELS_DIR`` is a directory of self-contained per-model bundle
    folders. Each folder holds a ``vllm_metadata.json`` (``arch`` = HF
    architecture name, ``main_class`` = ``"module:Class"`` implementing the vLLM
    generator adapter) plus the adapter class and its dependencies. Any
    distribution tool (e.g. tt-kernel) can drop a bundle folder here and have it
    registered with no source edit to this plugin. Malformed / incomplete folders
    are skipped with a warning.
    """
    base = os.getenv("EXTRA_MODELS_DIR")
    if not base:
        return
    if not os.path.isdir(base):
        logger.warning("EXTRA_MODELS_DIR=%s is not a directory; ignoring.", base)
        return
    for name in sorted(os.listdir(base)):
        folder = os.path.join(base, name)
        if not os.path.isdir(folder):
            continue
        meta_path = os.path.join(folder, "vllm_metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Skipping %s: cannot read vllm_metadata.json (%s)", folder, exc
            )
            continue
        arch = data.get("arch")
        main_class = data.get("main_class")
        if not arch or not main_class:
            logger.warning(
                "Skipping %s: vllm_metadata.json needs 'arch' and 'main_class'.",
                folder,
            )
            continue
        yield folder, arch, main_class


def _register_models_from_extra_dir(ModelRegistry) -> int:
    """Register every model found under ``EXTRA_MODELS_DIR``; return the count.

    Registration is lazy (a ``"module:Class"`` string resolved by vLLM later), so
    a bundle folder that carries its own adapter module must stay importable when
    that resolution happens. We ``append`` the folder to ``sys.path`` (never
    ``insert(0)``, so an installed package of the same name always wins and
    nothing is shadowed); built-in adapters given as a full dotted path resolve
    normally and need no path entry. The arch is registered under the plugin's
    ``TT``-prefixed convention (mirroring ``check_and_update_config``).
    """
    count = 0
    for folder, arch, main_class in _iter_extra_model_bundles():
        if folder not in sys.path:
            sys.path.append(folder)
        tt_arch = arch if arch.startswith("TT") else "TT" + arch
        _register_model_if_missing(ModelRegistry, tt_arch, main_class)
        logger.info(
            "Registered TT model %s -> %s (from EXTRA_MODELS_DIR/%s)",
            tt_arch,
            main_class,
            os.path.basename(folder),
        )
        count += 1
    return count


def _builtin_models_enabled() -> bool:
    """Whether to register the built-in (hard-coded) TT model map.

    Defaults to enabled for backward compatibility. Set
    ``TT_VLLM_BUILTIN_MODELS=0`` to rely solely on ``EXTRA_MODELS_DIR`` (the
    intended end-state once all models ship as bundles). Any of 0/false/no/off
    disables it.
    """
    val = os.getenv("TT_VLLM_BUILTIN_MODELS")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off")


def register_tt_models(register_test_models=False) -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    # Dynamic hook: register any bundles dropped under EXTRA_MODELS_DIR. Runs
    # first so a distributed bundle can supply a model without touching this file.
    _register_models_from_extra_dir(ModelRegistry)

    # Built-in map. Kept for compatibility; disable with TT_VLLM_BUILTIN_MODELS=0.
    if not _builtin_models_enabled():
        if register_test_models:
            register_tt_test_models()
        return

    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    elif llama_text_version == "llama3_70b_galaxy":
        path_llama_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM"
        )
    elif llama_text_version == "llama2_70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Llama version: {llama_text_version}, "
            "pick one of [tt_transformers, llama3_70b_galaxy, llama2_70b]"
        )

    # Llama3.1/3.2 - Text
    _register_model_if_missing(ModelRegistry, "TTLlamaForCausalLM", path_llama_text)

    # Llama3.2 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTMllamaForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration",
    )

    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    _register_model_if_missing(ModelRegistry, "TTQwen2ForCausalLM", path_qwen_text)

    # Qwen3 - Text
    qwen3_text_version = os.getenv("TT_QWEN3_TEXT_VER", "tt_transformers")
    if qwen3_text_version == "tt_transformers":
        path_qwen3_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    elif qwen3_text_version == "qwen3_32b_galaxy":
        path_qwen3_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:QwenForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Qwen3 version: {qwen3_text_version}, "
            "pick one of [tt_transformers, qwen3_32b_galaxy]"
        )

    _register_model_if_missing(ModelRegistry, "TTQwen3ForCausalLM", path_qwen3_text)

    # Qwen3.5 - Text
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen3_5ForConditionalGeneration",
        "models.demos.blackhole.qwen36.tt.qwen36_vllm:Qwen36ForCausalLM",
    )

    # Qwen2.5 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen2_5_VLForConditionalGeneration",
        "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration",
    )

    # Qwen3 - Vision
    _register_model_if_missing(
        ModelRegistry,
        "TTQwen3VLForConditionalGeneration",
        "models.demos.qwen3_vl.tt.generator_vllm:Qwen3VLForConditionalGeneration",
    )

    # Mistral - Text only
    _register_model_if_missing(
        ModelRegistry,
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM",
    )

    # Mistral 3 - Multimodal (Vision + Text)
    _register_model_if_missing(
        ModelRegistry,
        "TTMistral3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Mistral3ForConditionalGeneration",
    )

    # Gemma3
    _register_model_if_missing(
        ModelRegistry,
        "TTGemma3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration",
    )

    # Gemma4 — text-only TT bridge.
    #
    # Gemma4 isn't in vLLM's upstream registry, so without an entry here
    # the upstream architecture resolver falls back to
    # ``TransformersMultiModalForCausalLM`` (because ``hf_config !=
    # hf_text_config`` for Gemma4's nested config — see
    # ``ModelConfig._get_transformers_backend_cls``) and crashes on the
    # ``_processor_factory`` assertion in the multimodal registry. The
    # plugin's later ``TT``-prefix logic runs after that resolution, so
    # it can't help.
    #
    # We register the plain HF arch names directly so upstream resolution
    # finds our class. Since ``Gemma4ForCausalLM`` (the TT class) does not
    # use ``SupportsMultiModal``, vLLM's ``_model_info.supports_multimodal``
    # is False, ``multimodal_config`` is not populated, and the request
    # path stays text-only — which matches what the TT model implements.
    # The ``TT``-prefixed aliases satisfy the plugin's later validation
    # in ``check_and_update_config`` so no override is needed.
    #
    # The 12B checkpoint is the "unified" multimodal variant: its config
    # declares ``architectures: ['Gemma4UnifiedForConditionalGeneration']``
    # with ``model_type: gemma4_unified`` and nested text/vision/audio
    # configs. Without the unified arch registered, the same nested-config
    # fallback resolves it to ``TransformersMultiModalForCausalLM``. We map
    # the unified arch (and its ``TT`` alias) to the same text-only TT class
    # so text-only inference runs on the unified checkpoint.
    _gemma4_target = "models.demos.gemma4.tt.generator_vllm:Gemma4ForCausalLM"
    for arch in (
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
        "TTGemma4ForCausalLM",
        "TTGemma4ForConditionalGeneration",
        "TTGemma4UnifiedForConditionalGeneration",
    ):
        _register_model_if_missing(ModelRegistry, arch, _gemma4_target)

    _register_model_if_missing(
        ModelRegistry,
        "TTAutoportGemma4ForCausalLM",
        "models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm:AutoportGemma4ForCausalLM",
    )

    # DeepseekV3
    _register_model_if_missing(
        ModelRegistry,
        "TTDeepseekV3ForCausalLM",
        "models.demos.deepseek_v3.tt.generator_vllm:DeepseekV3ForCausalLM",
    )

    # GPT-OSS
    _register_model_if_missing(
        ModelRegistry,
        "TTGptOssForCausalLM",
        "models.tt_transformers.tt.generator_vllm:GptOssForCausalLM",
    )

    # Optionally register test models if explicitly enabled
    if register_test_models:
        register_tt_test_models()


def register_tt_test_models():
    """Register non-production TT models which are only used for testing."""
    from vllm.model_executor.models.registry import ModelRegistry

    # Fake model for testing multi-process inference on T3000
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyT3000MultiProcessModel",
        "models.vllm_test_utils.t3000_multiproc_test.test_model:DummyT3000MultiProcessModel",
    )

    # Fake model which does nothing, for measuring vLLM host overheads
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyNoOpModel",
        "models.vllm_test_utils.no_op_test.test_model:DummyNoOpModel",
    )

    # Fake model for testing multi-host inference on dual Galaxy
    _register_model_if_missing(
        ModelRegistry,
        "TTDummyDualGlxModel",
        "models.vllm_test_utils.dual_glx_ccl_test.test_model:DummyDualGlxModel",
    )


class TTPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "tt"
    device_type: str = "tt"
    sample_on_device_mode: ClassVar[Literal["all", "decode_only"] | None] = None
    # Disable torch.compile on TT platform - the triton version in tt-metal
    # is incompatible with torch's inductor backend.
    simple_compile_backend: str = "eager"

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        # Hybrid models (Gemma3/4, GPT-OSS) opt in to upstream's HMA via
        # ``HybridAttentionForCausalLM.get_kv_cache_spec`` so layers from
        # different attention groups can share DRAM tensors. Without this
        # override the base ``Platform`` returns ``False`` and HMA collapses
        # every ``SlidingWindowSpec`` back to ``FullAttentionSpec`` in
        # ``unify_hybrid_kv_cache_specs`` — defeating the entire point.
        return True

    @classmethod
    def pre_register_and_update(
        cls, parser: FlexibleArgumentParser | None = None
    ) -> None:
        # Called during CLI/parser setup (APIServer). ModelConfig may
        # validate/inspect architectures before VllmConfig is constructed in
        # this process, so we must ensure TT test models are registered early
        # when explicitly requested via CLI override.
        super().pre_register_and_update(parser)
        _install_tt_harmony_truncation_patch()
        if _should_pre_register_tt_test_models_from_cli():
            register_tt_test_models()

    @classmethod
    def import_kernels(cls) -> None:
        # Do not import vllm._C or vllm._moe_C
        pass

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        _install_tt_harmony_truncation_patch()
        _apply_chunked_prefill_policy(vllm_config)

        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TT backend"
        )
        assert (
            vllm_config.parallel_config.tensor_parallel_size == 1
            and vllm_config.parallel_config.pipeline_parallel_size == 1
        ), "TT backend does not support distributed execution"
        assert not vllm_config.lora_config, "LoRA is not supported for TT backend"

        # Device computes top-32 logprobs but the OpenAI API limits to 20
        MAX_TOP_K = 20

        model_config = vllm_config.model_config
        if model_config.max_logprobs > MAX_TOP_K:
            logger.warning(
                "max_logprobs=%d exceeds TT device limit of %d, clamping to %d",
                model_config.max_logprobs,
                MAX_TOP_K,
                MAX_TOP_K,
            )
            model_config.max_logprobs = MAX_TOP_K

        # Import and register models from tt-metal.
        #
        # NOTE: We also register TT models early in `vllm_tt_plugin.worker`
        # (at module import time). That registration is required to handle
        # engine/worker subprocess startup ordering where model architectures
        # may be inspected (e.g. multimodal processor cache init) before this
        # `check_and_update_config()` hook is reached in that process.
        tt_config = get_tt_config(vllm_config)
        register_test_models = False
        if tt_config and "register_test_models" in tt_config:
            register_test_models = tt_config["register_test_models"]
            assert register_test_models in [True, False], (
                f"Invalid option register_test_models: {register_test_models}"
            )
        register_tt_models(register_test_models)

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_tt_plugin.worker.TTWorker"
        parallel_config.engine_core_cls = "vllm.v1.engine.core.EngineCore"
        parallel_config.engine_core_proc_cls = "vllm.v1.engine.core.EngineCoreProc"
        parallel_config.dp_engine_core_proc_cls = (
            "vllm_tt_plugin.engine.TTDPEngineCoreProc"
        )
        parallel_config.engine_core_launcher_cls = (
            "vllm_tt_plugin.launcher.TTCoreEngineLauncher"
        )

        # For TT models, prepend "TT" to the architecture name,
        # e.g. "TTLlamaForCausalLM"
        arch_names = vllm_config.model_config.hf_config.architectures
        for i in range(len(arch_names)):
            if not arch_names[i].startswith("TT"):
                arch_names[i] = "TT" + arch_names[i]

        # Verify that the TT architecture is registered in the model registry
        from vllm.model_executor.models.registry import ModelRegistry

        supported_archs = ModelRegistry.get_supported_archs()
        if not any(arch_name in supported_archs for arch_name in arch_names):
            tt_archs = sorted(
                [arch for arch in supported_archs if arch.startswith("TT")]
            )
            raise ValueError(
                f"No TT model architecture is registered for "
                f"model: '{vllm_config.model_config.model}'. "
                f"Available TT architectures: {tt_archs}"
            )

        # Setting attributes on the class level is kind of hacky, but
        # it's the only way to make validate_request depend on vllm_config
        # This is needed to catch incompatible requests early enough
        # to return an error instead of crashing.
        # TODO move this to tt_model_runner when request validation
        # stops depending on vllm_config

        if tt_config is not None and "sample_on_device_mode" in tt_config:
            sample_on_device_mode = tt_config["sample_on_device_mode"]
            assert sample_on_device_mode in [
                "all",
                "decode_only",
            ], f"Invalid sample_on_device_mode: {sample_on_device_mode}"
        else:
            sample_on_device_mode = None
        cls.sample_on_device_mode = sample_on_device_mode  # type: ignore[attr-defined]

        # Compat sampling uses the full vLLM sampling pipeline,
        # with logit processors and sampler, instead of our custom sampling.
        # It is enabled only if any of the requests in the batch requires it,
        # or if always_compat_sampling is enabled.

        always_compat_sampling = False
        if tt_config is not None and "always_compat_sampling" in tt_config:
            always_compat_sampling = tt_config["always_compat_sampling"]
            assert always_compat_sampling in [True, False], (
                "always_compat_sampling must be a boolean"
            )
            if always_compat_sampling:
                raise ValueError(
                    "always_compat_sampling is not yet supported for V1 TT backend."
                )
        cls.always_compat_sampling = always_compat_sampling  # type: ignore[attr-defined]

        # must perform local import to get around circular import
        from vllm.model_executor.model_loader.utils import get_model_architecture

        model_class, _ = get_model_architecture(vllm_config.model_config)

        # Get model capabilities from the class
        model_capabilities: dict | None = getattr(
            model_class, "model_capabilities", None
        )

        # A model either supports the full on-device sampling pipeline or it
        # doesn't — there is no greedy-only mode. Models opt in by setting
        # `supports_sample_on_device` in their `model_capabilities` dict.
        supports_sample_on_device = (
            model_capabilities.get("supports_sample_on_device", False)
            if model_capabilities
            else False
        )
        if sample_on_device_mode is not None and not supports_sample_on_device:
            raise ValueError(
                f"sample_on_device_mode={sample_on_device_mode!r} was requested, "
                f"but model {model_class.__name__} "
                f"({model_class.__module__}) does not support on-device sampling. "
                "Unset sample_on_device_mode or use a model that supports it."
            )

        # Model-gated async scheduling. Async overlap requires generators that
        # support split decode submission via `decode_forward(...,
        # read_from_device=False)` followed by `read_decode_output(...,
        # async_read=True)`.
        supports_async_decode = (
            model_capabilities.get("supports_async_decode", False)
            if model_capabilities
            else False
        )
        if vllm_config.scheduler_config.async_scheduling and not supports_async_decode:
            logger.warning(
                "Async scheduling was requested, but TT model %s (%s) does not "
                "declare support (`model_capabilities['supports_async_decode']`). "
                "Disabling async scheduling.",
                model_class.__name__,
                model_class.__module__,
            )
            vllm_config.scheduler_config.async_scheduling = False

        # Galaxy-generator models (Llama3 70B, Qwen3-32B) are served by a single
        # device mesh. Convert
        # it transparently into single-process TT lanes so users keep passing
        # --data_parallel_size with no other flag changes. Must run before the
        # validation/routing below so the lane path is selected.
        _convert_galaxy_gather_dp_to_lanes(vllm_config)

        if uses_tt_lane_coordinator(vllm_config):
            # Fail fast on misconfiguration: lane mode requires max_num_seqs to
            # split evenly across the internal TT lanes.
            validate_tt_lane_config(vllm_config)
            vllm_config.scheduler_config.scheduler_cls = TT_LANE_SCHEDULER_CLS
            logger.info(
                "Using TTLaneCoordinator with %d in-process TT lanes",
                get_tt_data_parallel_size(vllm_config),
            )
        else:
            vllm_config.scheduler_config.scheduler_cls = TT_SCHEDULER_CLS

        if vllm_config.cache_config.enable_prefix_caching:
            # Check prefix caching support from capabilities (default to False)
            supports_prefix_caching = (
                model_capabilities.get("supports_prefix_caching", False)
                if model_capabilities
                else False
            )

            if not supports_prefix_caching:
                vllm_config.cache_config.enable_prefix_caching = False
                logger.warning(
                    "Prefix caching is not supported in TT backend for %s, "
                    "disabling it",
                    model_class.__module__,
                )
                _renormalize_mamba_cache_config(vllm_config)
            else:
                # Check if the model architecture uses sliding window
                uses_sliding_window = (
                    vllm_config.model_config.get_sliding_window() is not None
                )
                if uses_sliding_window:
                    vllm_config.cache_config.enable_prefix_caching = False
                    logger.warning(
                        "Prefix caching is not supported in TT backend for "
                        "models with sliding window, disabling it"
                    )
                    _renormalize_mamba_cache_config(vllm_config)

        logger.info(
            "Automatic prefix caching is %s",
            "enabled" if vllm_config.cache_config.enable_prefix_caching else "disabled",
        )
        # Check that all invariants are satisfied after all rewriting
        vllm_config.scheduler_config.verify_max_model_len(
            vllm_config.model_config.max_model_len
        )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # The regular v0 vLLM sampling code tries
        # to use pinned memory in case we're using GPUs.
        return False

    @classmethod
    def uses_host_device_handling(cls) -> bool:
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: "PromptType | DictPrompt | TokPrompt",
        params: "SamplingParams | PoolingParams",
        processed_inputs: "ProcessorInputs | None" = None,
    ) -> None:
        """Validate current two-argument and legacy three-argument requests."""
        from vllm.sampling_params import SamplingParams

        dev = cls.device_name

        if isinstance(params, SamplingParams) and params.prompt_logprobs is not None:
            raise ValueError(f"Not yet supporting prompt_logprobs on {dev}")

    @staticmethod
    def compat_sampling_required(sampling_params, num_devices) -> bool:
        # Device logprobs only supported on multi-device setups and only
        # the sampled token's logprob is returned (not top-k alternatives).
        # Single device: any logprobs require host sampling.
        # Multi-device: logprobs > 1 requires host sampling because device
        # can only return the sampled token's logprob.
        # https://github.com/tenstorrent/tt-metal/issues/34077
        if (
            sampling_params.logprobs is not None
            and sampling_params.logprobs > 0
            and (num_devices == 1 or sampling_params.logprobs > 1)
        ):
            return True

        # all of the following sampling params require compat sampling
        return (
            sampling_params.min_p != 0.0
            or (
                sampling_params.bad_words is not None
                and len(sampling_params.bad_words) > 0
            )
            or sampling_params.prompt_logprobs is not None
            or sampling_params.structured_outputs is not None
            or sampling_params.logit_bias is not None
            or sampling_params.allowed_token_ids is not None
            or sampling_params.min_tokens != 0
        )
