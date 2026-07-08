# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for deepseek_v3_d_p transformer tests.

Provides:
- HF model creation and weight extraction helpers
- Tokenization helpers (JSON prompts, InfiniteBench)
- Host reference caching for PCC validation
- InfiniteBench dataset download and caching
"""

import gc
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch
from loguru import logger
from transformers import DynamicCache

from models.common.utility_functions import hf_cache_layer_kv

# transformers 5.x moved no_init_weights to transformers.initialization; fall back
# to the old location for transformers < 5.x.
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

import ttnn


@dataclass
class LayerByLayerResult:
    """Result from load_and_compute_layer_by_layer function."""

    state_dict: dict | None  # TT format state dict, or None if extract_tt_weights=False
    ref_snapshots: list[torch.Tensor] | None  # Reference outputs, or None if compute_reference=False
    ref_kvpe_list: list[torch.Tensor] | None  # Reference KV cache, or None if compute_reference=False


def _log_memory(label: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / 1024**3
    logger.info(f"[MEMORY] {label}: {rss_gb:.2f} GB")


# --- Constants ---

PROMPT_1K_PATH = Path("models/demos/deepseek_v3/demo/test_prompts_1024.json")
ABC_1K_PATH = Path("models/demos/deepseek_v3_d_p/demo/test_prompt_ABC_1k.json")
ABC_SHORT_PATH = Path("models/demos/deepseek_v3_d_p/demo/test_prompt_ABC_short.json")
P64TOK_PATH = Path("models/demos/deepseek_v3_d_p/demo/test_prompt_64tok.json")
P960TOK_PATH = Path("models/demos/deepseek_v3_d_p/demo/test_prompt_960tok.json")
PIE960_PATH = Path("models/demos/deepseek_v3_d_p/demo/test_pie_960tok.json")
PROMPT_5K_PATH = Path("models/demos/deepseek_v3_d_p/demo/test_prompt_5k.json")
PROMPT_25K_PATH = Path("models/demos/deepseek_v3_d_p/demo/test_prompt_25k.json")

TRACE_DIR_BASE = Path(os.getenv("DEEPSEEK_V3_TRACE_DIR", "/mnt/MLPerf/deepseek-prefill-cache")).resolve()
ILLIAD_1024_TRACE = TRACE_DIR_BASE / "illiad_prefill_fa2"
ILLIAD_25024_TRACE = TRACE_DIR_BASE / "illiad_prefill_fa2_25024"
ABC_1K_PAD_RIGHT_1024 = TRACE_DIR_BASE / "ABC_1k_prefill_padd_right_1024"
ABC_1K_PAD_LEFT_1024 = TRACE_DIR_BASE / "ABC_1k_prefill_padd_left_1024"
LONGBOOK_QA_ENG_25600 = TRACE_DIR_BASE / "longbook_qa_eng_prefill_25600_nopad"
LONGBOOK_QA_ENG_5120 = TRACE_DIR_BASE / "longbook_qa_eng_prefill_5120_nopad"
LONGBOOK_QA_ENG_56320 = TRACE_DIR_BASE / "longbook_qa_eng_prefill_56320_nopad"

# Identity-based trace lookup: (input_source, isl_total, padding_side) -> Path, where
# isl_total is the trace's NATIVE (generated) sequence length.
# Traces are only used when use_pretrained=True and n_routed_experts=256, since they
# were generated from the full pretrained model.
# A test may request an isl that is not a native trace length: find_trace_dir() falls
# back to the smallest native trace with the same (input_source, padding_side) whose
# length is >= the requested isl, and the caller slices it (see slice_debug_trace).
TRACE_LOOKUP: dict[tuple[str, int, str], Path] = {
    ("json_prompts", 1024, "right"): ILLIAD_1024_TRACE,
    ("json_prompts", 25600, "right"): ILLIAD_25024_TRACE,
    ("abc_1k", 1024, "right"): ABC_1K_PAD_RIGHT_1024,
    ("abc_1k", 1024, "left"): ABC_1K_PAD_LEFT_1024,
    ("longbook_qa_eng", 5120, "right"): LONGBOOK_QA_ENG_5120,
    ("longbook_qa_eng", 25600, "right"): LONGBOOK_QA_ENG_25600,
    ("longbook_qa_eng", 56320, "right"): LONGBOOK_QA_ENG_56320,
}


def _trace_dir_ready(path: Path) -> bool:
    """A trace dir is usable only if it exists and carries a metadata.json."""
    return path.exists() and (path / "metadata.json").exists()


def find_trace_dir(
    input_source: str,
    isl_total: int,
    padding_side: str,
    use_pretrained: bool,
    n_routed_experts: int,
) -> tuple[Path, int] | None:
    """Return ``(trace_dir, trace_isl)`` for a test configuration, or ``None``.

    ``trace_isl`` is the trace's NATIVE sequence length. When it is larger than the
    requested ``isl_total`` the caller must slice the trace down to ``isl_total``
    (see :func:`slice_debug_trace`) — valid for causal, nopad prefill traces.

    A trace is eligible only when:
    - the model uses pretrained weights with 256 experts (traces were generated from
      the full pretrained DeepSeek-R1 model)
    - the directory exists and contains a metadata.json

    Resolution order:
    1. Exact ``(input_source, isl_total, padding_side)`` match (no slicing).
    2. Otherwise the smallest ready trace with the same ``(input_source, padding_side)``
       whose native isl is ``>= isl_total`` (caller slices the first ``isl_total`` tokens).
    """
    if not use_pretrained or n_routed_experts != 256:
        return None

    # 1. Exact native-length match — preferred, no slicing needed.
    exact = TRACE_LOOKUP.get((input_source, isl_total, padding_side))
    if exact is not None and _trace_dir_ready(exact):
        return exact, isl_total

    # 2. Fall back to the smallest ready trace that is at least as long as requested,
    #    with matching input_source + padding_side.
    candidates = sorted(
        (trace_isl, path)
        for (src, trace_isl, pad), path in TRACE_LOOKUP.items()
        if src == input_source and pad == padding_side and trace_isl >= isl_total and _trace_dir_ready(path)
    )
    if candidates:
        trace_isl, path = candidates[0]
        return path, trace_isl
    return None


def check_first_token_match_host_ref(
    ref_snapshots: list | None,
    number_of_non_padded_tokens: int,
    padding_side: str,
    first_token_id: int,
    tokenizer,
) -> bool | None:
    """Check TT's first token vs HF reference argmax at the expected first token position.

    Returns:
        True if match, False if mismatch, None if no reference available.
    """
    if not ref_snapshots:
        return None
    hf_logits_full = ref_snapshots[-1]  # [1, seq_len, vocab]
    last_real_idx = number_of_non_padded_tokens - 1 if padding_side == "right" else hf_logits_full.shape[-2] - 1
    hf_token_id = int(hf_logits_full[0, last_real_idx, :].argmax().item())
    hf_token_text = tokenizer.decode([hf_token_id]) if tokenizer else "N/A"
    match = hf_token_id == first_token_id
    logger.info(
        f"HF reference token at position {last_real_idx}: "
        f"ID={hf_token_id} [{repr(hf_token_text)}] | TT==HF match: {match}"
    )
    return match


def check_first_token_match(trace, trace_dir: Path, first_token_id: int, first_token_prob: float) -> bool | None:
    """Check whether the produced first token matches the trace reference.

    Looks up the expected token ID from trace metadata or output_metadata.json.

    Returns:
        True if match, False if mismatch, None if no reference available.
    """
    ref_token_id = trace.metadata.get("next_token_id")
    ref_token_text = trace.metadata.get("next_token_text")

    if ref_token_id is None or ref_token_text is None:
        output_meta_path = (trace_dir / "output_metadata.json").resolve()
        if output_meta_path.exists():
            with open(output_meta_path) as f:  # noqa: S108
                output_meta = json.load(f)
            ref_token_id = ref_token_id or output_meta.get("next_token_id")
            ref_token_text = ref_token_text or output_meta.get("next_token_text")

    if ref_token_text is None:
        ref_token_text = "N/A"

    token_match = first_token_id == ref_token_id if ref_token_id is not None else None
    logger.info(
        f"Trace first token: TT={first_token_id} (prob={first_token_prob:.4f}), "
        f"Trace={ref_token_id} [{repr(ref_token_text)}], "
        f"Match={'YES' if token_match else 'NO' if token_match is not None else 'N/A'}"
    )
    return token_match


# Subset name -> JSONL filename on HuggingFace
INFINITEBENCH_SUBSETS = {
    "passkey": "passkey.jsonl",
    "kv_retrieval": "kv_retrieval.jsonl",
    "longdialogue_qa_eng": "longdialogue_qa_eng.jsonl",
    "longbook_qa_eng": "longbook_qa_eng.jsonl",
}


def _default_infinitebench_cache_dir() -> str:
    # Prefer a test-specific override, then HF_HOME/infinitebench, then a temp dir.
    explicit = os.environ.get("TT_DS_PREFILL_INFINITEBENCH_CACHE")
    if explicit:
        return explicit
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "infinitebench")
    return "/tmp/deepseek_v3_transformer_inputs"


INFINITEBENCH_CACHE_DIR = Path(_default_infinitebench_cache_dir())


# --- HF model helpers ---


def create_hf_model(variant, config, num_layers, n_routed_experts=None):
    """Create the variant's reference model with num_layers and random weights."""
    test_config = deepcopy(config)
    test_config.num_hidden_layers = num_layers
    test_config._attn_implementation = "eager"
    if n_routed_experts is not None:
        test_config.n_routed_experts = n_routed_experts

    model = variant.reference_model_cls(test_config)
    return model.eval().to(torch.bfloat16)


def extract_layer_state_dict(variant, full_sd, layer_idx, hf_layer):
    """Extract one layer's weights from HF state_dict into TtPrefillBlock format."""
    prefix = f"layers.{layer_idx}."
    is_moe = isinstance(hf_layer.mlp, variant.reference_moe_cls)

    layer_sd = {
        "attn_norm_weight": full_sd[f"{prefix}input_layernorm.weight"],
        "mla_weights": {
            "q_a_proj.weight": full_sd[f"{prefix}self_attn.q_a_proj.weight"],
            "q_a_layernorm.weight": full_sd[f"{prefix}self_attn.q_a_layernorm.weight"],
            "q_b_proj.weight": full_sd[f"{prefix}self_attn.q_b_proj.weight"],
            "kv_a_proj_with_mqa.weight": full_sd[f"{prefix}self_attn.kv_a_proj_with_mqa.weight"],
            "kv_a_layernorm.weight": full_sd[f"{prefix}self_attn.kv_a_layernorm.weight"],
            "kv_b_proj.weight": full_sd[f"{prefix}self_attn.kv_b_proj.weight"],
            "o_proj.weight": full_sd[f"{prefix}self_attn.o_proj.weight"],
        },
        "ffn_norm_weight": full_sd[f"{prefix}post_attention_layernorm.weight"],
    }

    if is_moe:
        layer_sd["gate_weights"] = {
            "weight": full_sd[f"{prefix}mlp.gate.weight"],
            "e_score_correction_bias": full_sd[f"{prefix}mlp.gate.e_score_correction_bias"],
        }
        layer_sd["routed_expert_weights"] = [
            {
                "gate_proj": full_sd[f"{prefix}mlp.experts.{j}.gate_proj.weight"],
                "up_proj": full_sd[f"{prefix}mlp.experts.{j}.up_proj.weight"],
                "down_proj": full_sd[f"{prefix}mlp.experts.{j}.down_proj.weight"],
            }
            for j in range(len(hf_layer.mlp.experts))
        ]
        layer_sd["shared_expert_weights"] = {
            "gate_proj": full_sd[f"{prefix}mlp.shared_experts.gate_proj.weight"],
            "up_proj": full_sd[f"{prefix}mlp.shared_experts.up_proj.weight"],
            "down_proj": full_sd[f"{prefix}mlp.shared_experts.down_proj.weight"],
        }
    else:
        layer_sd["ffn_weights"] = {
            "gate_proj": full_sd[f"{prefix}mlp.gate_proj.weight"],
            "up_proj": full_sd[f"{prefix}mlp.up_proj.weight"],
            "down_proj": full_sd[f"{prefix}mlp.down_proj.weight"],
        }

    return layer_sd


def extract_tt_state_dict(variant, hf_model):
    """Extract state_dict in TtPrefillTransformer format from HF model."""
    sd = hf_model.state_dict()
    num_layers = len(hf_model.layers)

    result = {
        "embed_weight": sd["embed_tokens.weight"].float(),
        "norm_weight": sd["norm.weight"],
        "layers": [],
    }

    for i in range(num_layers):
        layer_sd = extract_layer_state_dict(variant, sd, i, hf_model.layers[i])
        result["layers"].append(layer_sd)

    return result


def tt_state_dict_to_hf_state_dict(tt_sd):
    """Reverse key mapping from TT format to HF DeepseekV3Model state_dict keys.

    Tensor references are shared (no copy).
    """
    hf_sd = {}
    hf_sd["embed_tokens.weight"] = tt_sd["embed_weight"]
    hf_sd["norm.weight"] = tt_sd["norm_weight"]

    for i, layer in enumerate(tt_sd["layers"]):
        prefix = f"layers.{i}."
        hf_sd[f"{prefix}input_layernorm.weight"] = layer["attn_norm_weight"]

        for key, value in layer["mla_weights"].items():
            hf_sd[f"{prefix}self_attn.{key}"] = value

        hf_sd[f"{prefix}post_attention_layernorm.weight"] = layer["ffn_norm_weight"]

        if "ffn_weights" in layer:
            # Dense FFN
            for key in ("gate_proj", "up_proj", "down_proj"):
                hf_sd[f"{prefix}mlp.{key}.weight"] = layer["ffn_weights"][key]
        else:
            # MoE
            hf_sd[f"{prefix}mlp.gate.weight"] = layer["gate_weights"]["weight"]
            hf_sd[f"{prefix}mlp.gate.e_score_correction_bias"] = layer["gate_weights"]["e_score_correction_bias"]

            for j, expert in enumerate(layer["routed_expert_weights"]):
                for key in ("gate_proj", "up_proj", "down_proj"):
                    hf_sd[f"{prefix}mlp.experts.{j}.{key}.weight"] = expert[key]

            for key in ("gate_proj", "up_proj", "down_proj"):
                hf_sd[f"{prefix}mlp.shared_experts.{key}.weight"] = layer["shared_expert_weights"][key]

    return hf_sd


def create_hf_model_with_weights(variant, config, num_layers, hf_sd):
    """Create the variant's reference model with pretrained weights (no random init)."""
    test_config = deepcopy(config)
    test_config.num_hidden_layers = num_layers
    test_config._attn_implementation = "eager"

    logger.info(f"Creating {variant.reference_model_cls.__name__} with {num_layers} layers...")
    with no_init_weights():
        model = variant.reference_model_cls(test_config)
    logger.info("Model structure created successfully")

    # Load state dict layer-by-layer to avoid OOM (loading all layers at once doubles memory usage)
    # strict=False: rotary embedding buffers are computed from config, not from weights
    logger.info(f"Loading state dict with {len(hf_sd)} keys (layer-by-layer to minimize memory)...")

    all_missing = set()
    all_unexpected = set()

    try:
        # Load embeddings first
        embedding_keys = [k for k in hf_sd.keys() if k.startswith("embed_tokens")]
        if embedding_keys:
            logger.info(f"Loading embeddings ({len(embedding_keys)} keys)...")
            embedding_sd = {k: hf_sd[k] for k in embedding_keys}
            missing, unexpected = model.load_state_dict(embedding_sd, strict=False)
            all_missing.update(missing)
            all_unexpected.update(unexpected)
            # Free memory immediately
            for k in embedding_keys:
                del hf_sd[k]
            del embedding_sd
            gc.collect()
            logger.info("Embeddings loaded, memory freed")

        # Load each layer sequentially
        for i in range(num_layers):
            layer_keys = [k for k in hf_sd.keys() if k.startswith(f"layers.{i}.")]
            if layer_keys:
                logger.info(f"Loading layer {i} ({len(layer_keys)} keys)...")
                layer_sd = {k: hf_sd[k] for k in layer_keys}
                missing, unexpected = model.load_state_dict(layer_sd, strict=False)
                all_missing.update(missing)
                all_unexpected.update(unexpected)
                # Free memory immediately
                for k in layer_keys:
                    del hf_sd[k]
                del layer_sd
                gc.collect()
                logger.info(f"Layer {i} loaded, memory freed")

        # Load norm last
        norm_keys = [k for k in hf_sd.keys() if k.startswith("norm")]
        if norm_keys:
            logger.info(f"Loading norm ({len(norm_keys)} keys)...")
            norm_sd = {k: hf_sd[k] for k in norm_keys}
            missing, unexpected = model.load_state_dict(norm_sd, strict=False)
            all_missing.update(missing)
            all_unexpected.update(unexpected)
            # Free memory immediately
            for k in norm_keys:
                del hf_sd[k]
            del norm_sd
            gc.collect()
            logger.info("Norm loaded, memory freed")

        logger.info("State dict loaded successfully (layer-by-layer)")

    except Exception as e:
        logger.error(f"FAILED to load state dict: {type(e).__name__}: {e}")
        import traceback

        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

    if all_missing:
        real_missing = [k for k in all_missing if "rotary_emb" not in k]
        if real_missing:
            logger.warning(f"Missing weight keys (not rotary_emb): {real_missing}")
        logger.info(f"Skipped {len(all_missing)} model keys not in weights (rotary_emb buffers)")
    if all_unexpected:
        logger.warning(f"Unexpected keys in state dict: {all_unexpected}")

    logger.info("Converting model to eval mode and bfloat16...")
    result = model.eval().to(torch.bfloat16)
    logger.info("Model conversion complete")
    return result


def get_4d_causal_mask(attention_mask, causal_only=False):
    "Get 4D causal attention mask for prefill. If causal_only=True, returns a purely causal mask without any padding mask (equivalent to is_causal=True in ttnn)."

    if causal_only:
        # torch.where(torch.tril(torch.ones(5,5)) == 1, 0, -1e38)
        # tensor([[ 0.0000e+00, -1.0000e+38, -1.0000e+38, -1.0000e+38, -1.0000e+38],
        #         [ 0.0000e+00,  0.0000e+00, -1.0000e+38, -1.0000e+38, -1.0000e+38],
        #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+38, -1.0000e+38],
        #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+38],
        #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]])
        return (
            torch.where(torch.tril(torch.ones(attention_mask.shape[-1], attention_mask.shape[-1])) == 1, 0, -1e38)
            .unsqueeze(0)
            .unsqueeze(0)
        )
    else:
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask,
            (1, attention_mask.shape[-1]),
            inputs_embeds=torch.zeros_like(attention_mask, dtype=torch.bfloat16),
            past_key_values_length=0,
        )
        return attention_mask_4d


def load_and_compute_layer_by_layer(
    variant,
    model_path: Path,
    config,
    num_layers: int,
    token_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    compute_reference: bool = True,
    build_ttnn_cache: bool = True,
    weight_cache_path: Path | None = None,
    mesh_device: ttnn.MeshDevice | None = None,
    seq_len: int = 1024,
    num_links: int = 2,
    topology: ttnn.Topology = ttnn.Topology.Linear,
    sp_axis: int = 0,
    tp_axis: int = 1,
    gate_fallback_mode=None,
    routed_expert_activations_dtype=ttnn.bfloat8_b,
    routed_expert_weights_dtype=ttnn.bfloat4_b,
    shared_expert_activations_dtype=ttnn.bfloat16,
    shared_expert_weights_dtype=ttnn.bfloat8_b,
    causal_only=True,
) -> LayerByLayerResult:
    """
    Process layers one-at-a-time: load → compute reference → build cache → clear → next.

    Peak memory: ~21GB regardless of num_layers (no accumulation).

    Args:
        model_path: Path to HF model directory with safetensors
        config: HF model config
        num_layers: Number of layers to process
        token_ids: Input tokens [1, seq_len] (required if compute_reference=True)
        compute_reference: If True, compute forward passes and return reference outputs
        build_ttnn_cache: If True, build .ttnn cache files using device=None
        weight_cache_path: Cache directory (required if build_ttnn_cache=True)
        mesh_device: Mesh device reference (required if build_ttnn_cache=True)
        attention_mask: Attention mask [shape TBD] (required if compute_reference=True)

    Returns:
        LayerByLayerResult(state_dict=None, ref_snapshots, ref_kvpe_list)
        Note: state_dict is always None (cache built to disk instead)
    """
    from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
    from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
    from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
    from models.demos.deepseek_v3_d_p.utils.test_utils import dequantize_state_dict, detect_language_model_prefix

    if gate_fallback_mode is None:
        gate_fallback_mode = GateComputeMode.HOST_ALL

    # Validation
    if not (compute_reference or build_ttnn_cache):
        raise ValueError("At least one of compute_reference or build_ttnn_cache must be True")
    if compute_reference and token_ids is None:
        raise ValueError("token_ids required when compute_reference=True")
    if build_ttnn_cache and (weight_cache_path is None or mesh_device is None):
        raise ValueError("weight_cache_path and mesh_device required when build_ttnn_cache=True")

    logger.info(
        f"Processing {num_layers} layers from {model_path} "
        f"(compute_reference={compute_reference}, build_ttnn_cache={build_ttnn_cache})"
    )

    _log_memory("Start of load_and_compute_layer_by_layer")

    # Create LazyStateDict
    lazy_sd = LazyStateDict(Path(model_path))

    prefix = detect_language_model_prefix(lazy_sd)

    # Initialize outputs
    ref_snapshots = [] if compute_reference else None
    ref_kvpe_list = None
    ref_cache = None

    # Create hf_model only if computing reference
    hf_model = None
    h_ref = None
    if compute_reference:
        test_config = deepcopy(config)
        test_config.num_hidden_layers = num_layers
        test_config._attn_implementation = "eager"

        logger.info(f"Creating empty {variant.reference_model_cls.__name__} for reference computation...")
        with no_init_weights():
            hf_model = variant.reference_model_cls(test_config).eval()
        _log_memory("After creating HF model structure")

        # Setup forward pass inputs
        seq_len = token_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        ref_cache = DynamicCache()

    # --- Process Embeddings ---
    logger.info("Processing embeddings...")
    embed_sd = sub_state_dict(lazy_sd, f"{prefix}model.embed_tokens.")
    embed_dequant = dequantize_state_dict(embed_sd, config)

    if compute_reference:
        embed_with_prefix = {f"embed_tokens.{k}": v for k, v in embed_dequant.items()}
        hf_model.load_state_dict(embed_with_prefix, strict=False)
        with torch.no_grad():
            h_ref = hf_model.embed_tokens(token_ids).to(torch.bfloat16)
        ref_snapshots.append(h_ref)
        # Clear embedding weights from hf_model
        hf_model.embed_tokens.weight.data = torch.empty(0)
        del embed_with_prefix

    attention_mask = get_4d_causal_mask(attention_mask, causal_only=causal_only)

    if build_ttnn_cache:
        # Build embedding cache (device=None, no accumulation!)
        TtParallelEmbedding.build_ttnn_cache(
            torch_weight=embed_dequant["weight"].float(),
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=weight_cache_path,
            tp_axis=tp_axis,
            dtype=ttnn.bfloat16,
        )

    for k in embed_sd.keys():
        lazy_sd.evict(k)
    del embed_sd, embed_dequant
    gc.collect()
    _log_memory("After embeddings processed and cleared")
    logger.debug("Embeddings processed, cache cleared")

    # --- Process Layers ---
    first_k_dense = config.first_k_dense_replace
    n_routed = config.n_routed_experts

    for i in range(num_layers):
        logger.info(f"Processing layer {i}/{num_layers}...")

        layer_sd = sub_state_dict(lazy_sd, f"{prefix}model.layers.{i}.")
        layer_dequant = dequantize_state_dict(layer_sd, config)

        if compute_reference:
            layer_with_prefix = {f"layers.{i}.{k}": v for k, v in layer_dequant.items()}
            hf_model.load_state_dict(layer_with_prefix, strict=False)

            with torch.no_grad():
                layer_out = hf_model.layers[i](
                    h_ref,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=ref_cache,
                    use_cache=True,
                )
                h_ref = layer_out[0]
            ref_snapshots.append(h_ref)

            # Clear layer weights from hf_model
            for param in hf_model.layers[i].parameters():
                param.data = torch.empty(0)
            del layer_with_prefix

        if build_ttnn_cache:
            # Build TTNN cache for this layer (device=None, no accumulation!)
            is_dense = i < first_k_dense

            # Build layer_dict in same format as before
            layer_dict = {
                "attn_norm_weight": layer_dequant["input_layernorm.weight"],
                "mla_weights": {
                    "q_a_proj.weight": layer_dequant["self_attn.q_a_proj.weight"],
                    "q_a_layernorm.weight": layer_dequant["self_attn.q_a_layernorm.weight"],
                    "q_b_proj.weight": layer_dequant["self_attn.q_b_proj.weight"],
                    "kv_a_proj_with_mqa.weight": layer_dequant["self_attn.kv_a_proj_with_mqa.weight"],
                    "kv_a_layernorm.weight": layer_dequant["self_attn.kv_a_layernorm.weight"],
                    "kv_b_proj.weight": layer_dequant["self_attn.kv_b_proj.weight"],
                    "o_proj.weight": layer_dequant["self_attn.o_proj.weight"],
                },
                "ffn_norm_weight": layer_dequant["post_attention_layernorm.weight"],
            }

            # DSA-sparse variants (GLM-5.1, DeepSeek-V3.2) carry lightning-indexer weights; include them
            # so ttMLA.build_ttnn_cache writes a complete sparse cache — it resolves has_indexer from the
            # config and errors if the indexer host weights are missing. Auto-engages only when present
            # (dense DeepSeek-R1 / Kimi checkpoints have no self_attn.indexer.*). The checkpoint's
            # k_norm.bias maps to the indexer's k_norm_bias.weight slot (see TtIndexer.WEIGHT_NAMES).
            if "self_attn.indexer.wq_b.weight" in layer_dequant:
                layer_dict["mla_weights"].update(
                    {
                        "indexer.wq_b.weight": layer_dequant["self_attn.indexer.wq_b.weight"],
                        "indexer.wk.weight": layer_dequant["self_attn.indexer.wk.weight"],
                        "indexer.k_norm.weight": layer_dequant["self_attn.indexer.k_norm.weight"],
                        "indexer.k_norm_bias.weight": layer_dequant["self_attn.indexer.k_norm.bias"],
                        "indexer.weights_proj.weight": layer_dequant["self_attn.indexer.weights_proj.weight"],
                    }
                )

            if is_dense:
                layer_dict["ffn_weights"] = {
                    "gate_proj": layer_dequant["mlp.gate_proj.weight"],
                    "up_proj": layer_dequant["mlp.up_proj.weight"],
                    "down_proj": layer_dequant["mlp.down_proj.weight"],
                }
            else:
                layer_dict["gate_weights"] = {
                    "weight": layer_dequant["mlp.gate.weight"],
                    "e_score_correction_bias": layer_dequant["mlp.gate.e_score_correction_bias"],
                }
                layer_dict["routed_expert_weights"] = [
                    {
                        "gate_proj": layer_dequant[f"mlp.experts.{j}.gate_proj.weight"],
                        "up_proj": layer_dequant[f"mlp.experts.{j}.up_proj.weight"],
                        "down_proj": layer_dequant[f"mlp.experts.{j}.down_proj.weight"],
                    }
                    for j in range(n_routed)
                ]
                layer_dict["shared_expert_weights"] = {
                    "gate_proj": layer_dequant["mlp.shared_experts.gate_proj.weight"],
                    "up_proj": layer_dequant["mlp.shared_experts.up_proj.weight"],
                    "down_proj": layer_dequant["mlp.shared_experts.down_proj.weight"],
                }

            # Build TTNN cache (device=None) - NOT accumulated in memory!
            TtPrefillBlock.build_ttnn_cache(
                state_dict=layer_dict,
                layer_idx=i,
                cache_path=weight_cache_path,
                mesh_device=mesh_device,
                config=config,
                model_cfg=variant.model_config,
                seq_len=seq_len,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                gate_fallback_mode=gate_fallback_mode,
                routed_expert_activations_dtype=routed_expert_activations_dtype,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_activations_dtype=shared_expert_activations_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
            )

            # Free layer_dict immediately!
            del layer_dict

        for k in layer_sd.keys():
            lazy_sd.evict(k)
        del layer_sd, layer_dequant
        gc.collect()
        _log_memory(f"After layer {i} cleared")
        logger.debug(f"Layer {i} processed, cache cleared")

    # Extract KVPE if computed reference
    if compute_reference:
        ref_kvpe_list = [hf_cache_layer_kv(ref_cache, i)[0] for i in range(num_layers)]

    # --- Process Norm ---
    logger.info("Processing norm...")
    norm_sd = sub_state_dict(lazy_sd, f"{prefix}model.norm.")
    norm_dequant = dequantize_state_dict(norm_sd, config)

    if compute_reference:
        norm_with_prefix = {f"norm.{k}": v for k, v in norm_dequant.items()}
        hf_model.load_state_dict(norm_with_prefix, strict=False)
        logger.debug(f"[norm] h_ref {h_ref.dtype=}, norm_weight dtype={norm_dequant['weight'].dtype}")
        with torch.no_grad():
            h_ref = hf_model.norm(h_ref)
        ref_snapshots.append(h_ref)
        del norm_with_prefix

    if build_ttnn_cache:
        # Build norm cache
        TtDistributedRmsNorm.build_ttnn_cache(
            torch_weight=norm_dequant["weight"],
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=weight_cache_path,
            cache_name_prefix="norm",
        )

    for k in norm_sd.keys():
        lazy_sd.evict(k)
    del norm_sd, norm_dequant
    gc.collect()

    # --- Process LM Head ---
    logger.info("Processing lm_head...")
    lm_head_sd = sub_state_dict(lazy_sd, f"{prefix}lm_head.")
    lm_head_dequant = dequantize_state_dict(lm_head_sd, config)

    if compute_reference:
        # Apply lm_head projection: logits = h_ref @ lm_head_weight.T
        logger.debug(f"[lm_head] h_ref {h_ref.dtype=}, lm_head_weight.dtype={lm_head_dequant['weight'].dtype}")
        lm_head_weight = lm_head_dequant["weight"].to(torch.bfloat16)
        with torch.no_grad():
            h_ref_lm = torch.nn.functional.linear(h_ref.to(torch.bfloat16), lm_head_weight)
        ref_snapshots.append(h_ref_lm)
        del lm_head_weight

    if build_ttnn_cache:
        TtLMHead.build_ttnn_cache(
            torch_weight=lm_head_dequant["weight"],
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            mesh_device=mesh_device,
            cache_path=weight_cache_path,
            is_column_parallel=True,
        )

    for k in lm_head_sd.keys():
        lazy_sd.evict(k)
    del lm_head_sd, lm_head_dequant
    gc.collect()
    _log_memory("After lm_head processed and cleared")

    # Cleanup
    lazy_sd.close()

    _log_memory("End of load_and_compute_layer_by_layer")
    logger.info(f"Processing complete ({num_layers} layers, peak memory minimized)")

    return LayerByLayerResult(
        state_dict=None,  # Never built! Cache saved to disk instead
        ref_snapshots=ref_snapshots,
        ref_kvpe_list=ref_kvpe_list,
    )


@dataclass(frozen=True)
class ReferenceCacheKey:
    """All parameters that affect reference output identity.

    Changing any field produces a different cache filename, so stale results
    are never reused silently.
    """

    weight_type: str  # "pretrained" or "random"
    input_source: str  # "random", "json_prompts", "abc_1k", or InfiniteBench subset
    isl_total: int
    num_layers: int
    n_routed_experts: int
    padding_side: str  # "right" or "left"

    def __str__(self) -> str:
        return (
            f"{self.weight_type}_{self.input_source}"
            f"_isl{self.isl_total}_layers{self.num_layers}"
            f"_experts{self.n_routed_experts}_pad{self.padding_side}"
        )


def _ref_cache_dir(variant) -> Path:
    env = variant.ref_cache_env or "TT_DS_PREFILL_HOST_REF_CACHE"
    return Path(os.environ.get(env, f"/tmp/{variant.name}_transformer_ref_cache"))


def check_reference_cache_exists(variant, cache_key: ReferenceCacheKey) -> bool:
    cache_path = _ref_cache_dir(variant) / f"{cache_key}.pt"
    exists = cache_path.exists()
    if exists:
        logger.info(f"Reference cache found: {cache_path}")
    else:
        logger.debug(f"Reference cache not found: {cache_path}")
    return exists


def save_reference_cache(variant, cache_key: ReferenceCacheKey, ref_snapshots, ref_kvpe_list):
    cache_dir = _ref_cache_dir(variant)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.pt"
    torch.save({"ref_snapshots": ref_snapshots, "ref_kvpe_list": ref_kvpe_list}, cache_path)
    logger.info(f"Saved reference to {cache_path} ({len(ref_snapshots)} snapshots, {len(ref_kvpe_list)} KVPE)")


def load_reference_cache(variant, cache_key: ReferenceCacheKey) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    cache_path = _ref_cache_dir(variant) / f"{cache_key}.pt"

    if not cache_path.exists():
        raise FileNotFoundError(f"Reference cache not found: {cache_path}")

    cached = torch.load(cache_path, weights_only=True)

    if "ref_snapshots" not in cached or "ref_kvpe_list" not in cached:
        raise ValueError(f"Invalid cache format in {cache_path}")

    logger.info(f"Loaded reference from {cache_path}")
    return cached["ref_snapshots"], cached["ref_kvpe_list"]


def slice_non_padded(tensor: torch.Tensor, num_real_tokens: int, padding_side: str, seq_dim: int = -2) -> torch.Tensor:
    """Slice a tensor to keep only the non-padded tokens along the sequence dimension.

    Args:
        tensor: Input tensor with a sequence dimension
        num_real_tokens: Number of real (non-padded) tokens
        padding_side: "right" (padding at end) or "left" (padding at start)
        seq_dim: Which dimension is the sequence dimension (default: -2)

    Returns:
        Tensor with only the non-padded tokens along seq_dim
    """
    if padding_side == "right":
        return tensor.narrow(seq_dim, 0, num_real_tokens)
    else:
        start = tensor.shape[seq_dim] - num_real_tokens
        return tensor.narrow(seq_dim, start, num_real_tokens)


# --- Tokenization helpers ---
def tokenize_prompt_to_isl(
    tokenizer, max_isl: int, prompt_text: str = "Capital of France is", debug: bool = False
) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
    """Tokenize a prompt, padding/truncating to exactly max_isl tokens.

    Returns:
        (input_ids, attention_mask, tokens): input_ids and attention_mask are [1, max_isl] tensors;
        tokens is a list of token strings (only when debug=True, else None).
    """
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",  # return PyTorch tensors
        padding="max_length",  # pad batch to max_length
        truncation=True,  # truncate if longer than max_length
        max_length=max_isl,  # hard cap at <max_isl> tokens
        return_attention_mask=True,
    )

    input_ids: torch.Tensor = inputs.input_ids  # shape [B, 1024] (padded or capped)
    attention_mask: torch.Tensor = inputs.attention_mask
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist()) if debug else None

    return input_ids, attention_mask, tokens


def tokenize_prompt_to_chat_template(
    tokenizer,
    max_isl: int,
    user_prompt: str = "What is the capital of France",
    system_prompt: str = "You are a friendly assistant",
    debug: bool = False,
) -> tuple[torch.Tensor, list[str] | None]:
    """Apply chat template and tokenize, padding/truncating to max_isl tokens.

    Returns:
        (input_ids, tokens): input_ids is a [1, max_isl] tensor;
        tokens is a list of token strings (only when debug=True, else None).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # apply chat template AND tokenize in one shot
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding="max_length",  # pad batch to max_len
        truncation=True,  # truncate if longer
        max_length=max_isl,  # hard cap at <max_isl> tokens
        return_attention_mask=True,  # <-- ignored for some reason
    )
    logger.debug(f"Input IDs shape: {inputs.shape}")
    tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist()) if debug else None

    return inputs, tokens


# --- InfiniteBench download ---


def download_infinitebench_subset(subset: str) -> Path:
    """
    Download one InfiniteBench subset (longest entry) and cache as JSON.

    Args:
        subset: One of the keys in INFINITEBENCH_SUBSETS.

    Returns:
        Path to the cached JSON file containing the prompt.
    """
    if subset not in INFINITEBENCH_SUBSETS:
        raise ValueError(f"Unknown InfiniteBench subset '{subset}'. Choose from: {list(INFINITEBENCH_SUBSETS)}")

    filename = f"infinitebench_{subset}.json"
    cached_path = INFINITEBENCH_CACHE_DIR / filename

    if cached_path.exists():
        logger.info(f"InfiniteBench '{subset}' already cached at {cached_path}")
        return cached_path

    logger.info(f"Downloading InfiniteBench subset '{subset}' from HuggingFace...")

    from datasets import load_dataset

    ds = load_dataset(
        "xinrongzhang2022/InfiniteBench",
        data_files=INFINITEBENCH_SUBSETS[subset],
        split="train",
    )

    # Find the longest entry by character count (avoids tokenizer dependency)
    best = max(ds, key=lambda e: len(e.get("context") or ""))
    ctx = best["context"] or ""

    logger.info(f"Selected entry id={best.get('id', 'N/A')}, {len(ctx):,} chars")

    output = {
        "subset": subset,
        "source": "xinrongzhang2022/InfiniteBench",
        "license": "Apache-2.0",
        "id": best.get("id"),
        "prompt": ctx,
    }

    INFINITEBENCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cached_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {cached_path.name} ({cached_path.stat().st_size:,} bytes)")
    return cached_path


# --- Debug trace helpers ---
@dataclass
class DebugTraceData:
    """Data loaded from a bit_sculpt debug trace directory."""

    token_ids: torch.Tensor  # [1, seq_len] int64
    ref_snapshots: dict[str, torch.Tensor]  # label -> [1, seq, hidden_dim] bfloat16
    ref_kvpe_list: list[torch.Tensor]  # per-layer [1, 1, seq, kv_lora_rank + qk_rope_head_dim]
    logits: torch.Tensor | None  # [seq, vocab_size] float32
    metadata: dict  # raw metadata.json contents


def _read_trace_rows(tensor_dir: Path, key: str, end: int | None):
    """Read `key` from a chunked_group_a_v1 tensor dir (rows_<s>_<e>.safetensors shards), concatenating
    up to `end` rows (or all). Used to slice a long trace (e.g. the 55k GLM golden) down to a test's isl."""
    import glob as _glob

    from safetensors import safe_open

    parts, got = [], 0
    for shard in sorted(_glob.glob(str(tensor_dir / "rows_*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            t = f.get_tensor(key)
        parts.append(t)
        got += t.shape[0]
        if end is not None and got >= end:
            break
    out = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
    return out[:end] if end is not None else out


def load_debug_trace(trace_dir: Path, num_layers: int | None = None, isl: int | None = None) -> DebugTraceData:
    """
    Load reference tensors from a bit_sculpt debug trace directory.

    The trace contains real intermediate outputs from a pretrained model run,
    stored as safetensors files alongside a metadata.json.

    Args:
        trace_dir: Path to trace directory (must contain metadata.json + .safetensors)
        num_layers: Number of layers to load (default: all layers from metadata)

    Returns:
        DebugTraceData with token_ids, per-layer reference snapshots, KVPE cache, and logits
    """
    from safetensors import safe_open

    trace_dir = Path(trace_dir).resolve()
    if not trace_dir.exists():
        raise FileNotFoundError(f"Debug trace directory not found: {trace_dir}")

    with open(trace_dir / "metadata.json") as f:
        metadata = json.load(f)

    if num_layers is None:
        num_layers = metadata["n_layers"]

    token_ids = torch.tensor([metadata["token_ids"]], dtype=torch.int64)
    if isl is not None:
        token_ids = token_ids[:, :isl]  # chop a long trace (e.g. the 55k GLM golden) to this test's isl
    # chunked_group_a_v1 layout (row-sharded decoder_io/ + kv_cache/layer_i/) — used by the GLM 55k golden;
    # read + slice to isl instead of requiring a dedicated isl-sized standard-layout trace.
    chunked_dir = trace_dir / "decoder_io"
    is_chunked_layout = chunked_dir.is_dir()
    logger.info(f"Loaded {token_ids.shape[1]} tokens from {trace_dir.name}")

    ref_snapshots = {}
    hs_dir = trace_dir / "hidden_states"
    hs_flat = trace_dir / "hidden_states.safetensors"
    per_layer_format = hs_dir.is_dir()

    if is_chunked_layout:
        for i in range(num_layers):
            key = f"decoder_output_layer_{i}"
            t = _read_trace_rows(chunked_dir / key, key, isl)
            ref_snapshots[f"layer_{i}"] = t.unsqueeze(0)
        logger.info(f"Loaded {len(ref_snapshots)} layer snapshots from decoder_io/ (chunked_group_a_v1, isl={isl})")
    elif per_layer_format:
        for i in range(num_layers):
            layer_path = hs_dir / f"layer_{i}.safetensors"
            with safe_open(layer_path, framework="pt") as f:
                key = f"decoder_output_layer_{i}"
                t = f.get_tensor(key)
                ref_snapshots[f"layer_{i}"] = (t[:isl] if isl is not None else t).unsqueeze(0)
        logger.info(f"Loaded {len(ref_snapshots)} layer snapshots from hidden_states/ (per-layer files)")
    else:
        with safe_open(hs_flat, framework="pt") as f:
            for i in range(num_layers):
                key = f"decoder_output_layer_{i}"
                t = f.get_tensor(key)
                ref_snapshots[f"layer_{i}"] = (t[:isl] if isl is not None else t).unsqueeze(0)
        logger.info(f"Loaded {len(ref_snapshots)} layer snapshots from hidden_states.safetensors")

    ref_kvpe_list = []
    kv_dir = trace_dir / "kv_cache"
    kv_flat = trace_dir / "kv_cache.safetensors"

    if is_chunked_layout:
        for i in range(num_layers):
            key = f"kv_post_transform_layer_{i}"
            kv = _read_trace_rows(kv_dir / f"layer_{i}", key, isl)
            ref_kvpe_list.append(kv.unsqueeze(0).unsqueeze(0))
        logger.info(f"Loaded {len(ref_kvpe_list)} KVPE layers from kv_cache/ (chunked_group_a_v1, isl={isl})")
    elif per_layer_format and kv_dir.is_dir():
        # Detect key prefix from the first layer file
        with safe_open(kv_dir / "layer_0.safetensors", framework="pt") as f:
            available_keys = set(f.keys())
        use_post_transform = "kv_post_transform_layer_0" in available_keys
        key_prefix = "kv_post_transform_layer_" if use_post_transform else "compressed_kv_layer_"
        if not use_post_transform:
            logger.warning(
                "kv_post_transform not found in trace — falling back to compressed_kv (pre-RMSNorm, pre-RoPE). "
                "KVPE PCC will be unreliable. Re-generate the trace to fix."
            )
        for i in range(num_layers):
            layer_path = kv_dir / f"layer_{i}.safetensors"
            with safe_open(layer_path, framework="pt") as f:
                kv = f.get_tensor(f"{key_prefix}{i}")
                ref_kvpe_list.append(kv.unsqueeze(0).unsqueeze(0))
        kv_format = "post-transform" if use_post_transform else "pre-transform (legacy)"
        logger.info(f"Loaded {len(ref_kvpe_list)} KVPE layers from kv_cache/ (per-layer, {kv_format})")
    else:
        with safe_open(kv_flat, framework="pt") as f:
            available_keys = set(f.keys())
            use_post_transform = "kv_post_transform_layer_0" in available_keys
            key_prefix = "kv_post_transform_layer_" if use_post_transform else "compressed_kv_layer_"
            if not use_post_transform:
                logger.warning(
                    "kv_post_transform not found in trace — falling back to compressed_kv (pre-RMSNorm, pre-RoPE). "
                    "KVPE PCC will be unreliable. Re-generate the trace to fix."
                )
            for i in range(num_layers):
                kv = f.get_tensor(f"{key_prefix}{i}")
                ref_kvpe_list.append(kv.unsqueeze(0).unsqueeze(0))
        kv_format = "post-transform" if use_post_transform else "pre-transform (legacy)"
        logger.info(f"Loaded {len(ref_kvpe_list)} KVPE layers from kv_cache.safetensors ({kv_format})")

    logits = None
    logits_path = trace_dir / "logits.safetensors"
    if logits_path.exists():
        with safe_open(logits_path, framework="pt") as f:
            logits = f.get_tensor("logits")
        logger.info(f"Loaded logits: shape={list(logits.shape)}, dtype={logits.dtype}")

    return DebugTraceData(
        token_ids=token_ids,
        ref_snapshots=ref_snapshots,
        ref_kvpe_list=ref_kvpe_list,
        logits=logits,
        metadata=metadata,
    )


def slice_debug_trace(trace: DebugTraceData, isl_total: int) -> DebugTraceData:
    """Slice a debug trace down to its first ``isl_total`` sequence positions.

    This is exact for causal (autoregressive) prefill traces generated WITHOUT padding
    (the ``*_nopad`` traces): a transformer's per-layer decoder output and KV-cache entry
    at position ``i`` depend only on positions ``0..i`` (causal attention + absolute-position
    RoPE), so they are identical whether the full sequence or only its first ``isl_total``
    tokens are prefilled.

    The stored ``logits`` / ``next_token_id`` are the FULL sequence's final-position
    products and are meaningless for the shorter prefill, so ``logits`` is dropped
    (set to ``None``); callers must skip the logits / first-token checks for a sliced
    trace (``metadata`` is left untouched, so ``next_token_id`` must not be trusted).

    Args:
        trace: Trace to slice (typically longer than the requested isl).
        isl_total: Target sequence length; must be <= the trace's native length.

    Returns:
        A new :class:`DebugTraceData` truncated along the sequence dimension.
    """
    trace_len = trace.token_ids.shape[1]
    if isl_total > trace_len:
        raise ValueError(f"Cannot slice trace of length {trace_len} up to isl_total={isl_total}")
    return DebugTraceData(
        token_ids=trace.token_ids[:, :isl_total],
        ref_snapshots={label: snap[:, :isl_total, :] for label, snap in trace.ref_snapshots.items()},
        ref_kvpe_list=[kv[:, :, :isl_total, :] for kv in trace.ref_kvpe_list],
        logits=None,  # full-sequence final-position logits are invalid after slicing
        metadata=trace.metadata,
    )


# Golden bit_sculpt prefill trace (DeepSeek-R1-0528, 256 experts, hidden_dim 7168).
# Layer 3 is the first MoE layer (metadata moe_layer_offset == 3).
GOLDEN_LONGBOOK_TRACE = Path("/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad")


def load_trace_gate_input(
    trace_dir: Path,
    layer_idx: int,
    max_seq_len: int,
    dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor | None:
    """Load the MoE/gate block input from a bit_sculpt golden trace.

    The trace stores ``post_attn_norm_layer_{i}`` per layer — the post-attention
    RMSNorm output, which is exactly the tensor fed into the gate + experts at
    layer ``i`` (unlike ``decoder_output_layer_{i}``, the unnormalized residual
    stream). Returns ``[max_seq_len, dim]`` (tiled if the trace is shorter than
    ``max_seq_len``), or ``None`` if the trace/key is unavailable or ``dim``
    exceeds the trace hidden dim.

    A sliced read is used so only the requested ``[max_seq_len, dim]`` block is
    materialized rather than the full (seq, hidden_dim) tensor.
    """
    from safetensors import safe_open

    layer_path = Path(trace_dir) / "hidden_states" / f"layer_{layer_idx}.safetensors"
    if not layer_path.exists():
        logger.warning(f"Trace file not found: {layer_path}. Falling back to synthetic input.")
        return None

    key = f"post_attn_norm_layer_{layer_idx}"
    try:
        with safe_open(layer_path, framework="pt") as f:
            if key not in f.keys():
                logger.warning(f"{key} not in {layer_path}. Falling back to synthetic input.")
                return None
            sl = f.get_slice(key)
            seq_total, hidden_dim = sl.get_shape()
            if dim > hidden_dim:
                logger.warning(
                    f"Requested dim {dim} > trace hidden_dim {hidden_dim} ({key}). Falling back to synthetic input."
                )
                return None
            n = min(max_seq_len, seq_total)
            hidden = sl[:n, :dim].to(dtype)
    except Exception as e:  # safetensors / IO errors — fall back to synthetic
        logger.warning(f"Could not load {key} from {layer_path}: {e}. Falling back to synthetic input.")
        return None

    if hidden.shape[0] < max_seq_len:
        repeats = (max_seq_len + hidden.shape[0] - 1) // hidden.shape[0]
        hidden = hidden.repeat(repeats, 1)[:max_seq_len]

    logger.info(
        f"Loaded gate input from {Path(trace_dir).name} {key} "
        f"(trace {seq_total}x{hidden_dim}, sliced to {tuple(hidden.shape)})"
    )
    return hidden
