# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test factory and helpers for Gemma4 unit tests.

Uses HF_MODEL env var to determine which model variant to test against.
All HF reference configs and layers are created from the real checkpoint.
"""

import json
import os
from functools import lru_cache

import pytest
import torch
from loguru import logger

import ttnn

from ..config import MeshConfig, ModeConfig
from ..tt.model_config import Gemma4ModelArgs

_DEFAULT_MODEL_PATH = "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
_PCC_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "pcc_thresholds.json")

# Canonical prefill length buckets — three lengths chosen to bracket Gemma4's
# sliding-window attention: 128 (< sliding_window), 1024 (== sliding_window),
# 4096 (> sliding_window). Tests parametrize over these so each
# sliding-window regime gets a prefill kernel run. Long lengths are gated by
# the --max-prefill CLI option (see conftest); tests above the cap are
# auto-skipped to keep the routine loop fast.
PREFILL_BUCKETS = [128, 1024, 4096]


def _get_model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", _DEFAULT_MODEL_PATH)


# HF reference loading -- one path shared by every test that scores TT against
# the real checkpoint (test_teacher_forcing_e2e), so dtype/device_map never drift
# between the demo-agreement and accuracy numbers.
_HF_DEVICE_MAP_ENV = "GEMMA4_HF_DEVICE_MAP"


def from_pretrained_gemma4_causal_lm(model_path, **kwargs):
    """``from_pretrained`` that also loads the 12B unified multimodal release.

    Local 12B trees often keep an adapted text ``config.json`` (so TT can read
    ``text_config``) next to unified safetensors, with the original HF config
    saved as ``config.json.unified``. ``AutoModelForCausalLM`` then instantiates
    ``Gemma4ForConditionalGeneration`` and dies on the vision-tower shape
    mismatch. Prefer the sidecar + ``Gemma4UnifiedForConditionalGeneration``
    when present; otherwise use AutoModel (31B / CI NAS 12B).
    """
    from pathlib import Path

    from transformers import AutoConfig, AutoModelForCausalLM

    sidecar = Path(model_path) / "config.json.unified"
    if sidecar.is_file():
        config = AutoConfig.from_pretrained(sidecar, trust_remote_code=True)
        try:
            from transformers import Gemma4UnifiedForConditionalGeneration
        except ImportError:
            from transformers.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration

        logger.info("Loading HF reference via {} (unified sidecar)", sidecar.name)
        return Gemma4UnifiedForConditionalGeneration.from_pretrained(model_path, config=config, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def load_hf_reference_model(model_path=None):
    """Load the full HF checkpoint used as the reference model.

    bf16 weights + ``trust_remote_code``, sharded per ``GEMMA4_HF_DEVICE_MAP``
    (default ``auto``); set that to ``cpu`` to force host. Caller owns the
    returned model and should ``del`` it when done -- these checkpoints are large.
    """
    model_path = model_path or _get_model_path()
    device_map = os.environ.get(_HF_DEVICE_MAP_ENV, "auto")
    load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if device_map.lower() not in ("cpu", "none", ""):
        load_kwargs["device_map"] = device_map
        logger.info("Loading HF reference model with device_map={}", device_map)
    else:
        logger.warning("Loading HF reference model on CPU (set {}=auto for GPU)", _HF_DEVICE_MAP_ENV)

    model = from_pretrained_gemma4_causal_lm(model_path, **load_kwargs)
    model.eval()
    return model


def hf_reference_model_device(model) -> torch.device:
    """Device of the first parameter -- where inputs for ``model`` must live."""
    return next(model.parameters()).device


def build_hf_prefill_mask(seq_len, sliding_window=None):
    """Build the HF-format prefill attention mask [1, 1, seq_len, seq_len].

    Always causal. When sliding_window is set, also masks any (i, j) with
    j < i - sliding_window + 1, matching what HF's Gemma4 mask construction
    does internally for sliding_attention layers. Without this, an HF
    reference run with a pure causal mask diverges from the TT prefill
    once seq_len > sliding_window — TT applies the sliding window via the
    SDPA op's sliding_window_size, the reference doesn't, and PCC tanks.
    """
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    if sliding_window is not None and sliding_window > 0 and seq_len > sliding_window:
        idx = torch.arange(seq_len)
        outside_window = idx.unsqueeze(0) < (idx.unsqueeze(1) - sliding_window + 1)
        mask = mask.masked_fill(outside_window.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask


def find_layer_idx(hf_text_config, layer_type):
    """Return the first layer index whose type matches.

    layer_type: "sliding_attention" or "full_attention". Raises if no match —
    callers should pytest.skip when a model lacks a layer of the requested
    type (e.g. early-layer-only configs that have no global layer in the
    first N layers).
    """
    for i, lt in enumerate(hf_text_config.layer_types):
        if lt == layer_type:
            return i
    raise ValueError(f"No layer of type {layer_type} in layer_types={hf_text_config.layer_types}")


def num_layers_for_full_attention_group(hf_text_config):
    """Smallest prefix of layers that includes one full-attention block.

    Gemma4 stacks N sliding layers followed by one full layer; this helper
    returns N+1 so that a model truncated to that many layers exercises both
    the sliding and the full path. The group size depends on the variant:
    E2B has 4 sliding then full (returns 5); the larger variants have 5
    sliding then full (returns 6).
    """
    return find_layer_idx(hf_text_config, "full_attention") + 1


@lru_cache(maxsize=1)
def _load_pcc_thresholds():
    """Read the PCC threshold table from disk, once per process."""
    with open(_PCC_THRESHOLDS_PATH) as f:
        return json.load(f)


def _model_key():
    """Bucket key matching the top level of pcc_thresholds.json — the
    basename of the resolved model path (e.g. "gemma-4-E2B-it")."""
    return os.path.basename(_get_model_path().rstrip("/"))


def _lookup_model_entry(table, model_key):
    """Resolve a model entry from pcc_thresholds.json, case-insensitively.

    HF_MODEL may use ``google/gemma-4-31b-it`` while the table keys use
    ``gemma-4-31B-it``; basename casing must not force the 0.99 default.
    """
    if model_key in table:
        return table[model_key]
    key_lower = model_key.lower()
    for entry_key, entry in table.items():
        if entry_key.lower() == key_lower:
            return entry
    return {}


@lru_cache(maxsize=1)
def _model_key_candidates():
    """Return possible threshold-table keys for the active HF_MODEL.

    Local runs sometimes point HF_MODEL at a HuggingFace cache/snapshot path
    whose basename is a hash, not "gemma-4-31B-it". Keep the fast basename path,
    then infer well-known Gemma4 variants from the loaded config as a fallback.
    """
    candidates = [_model_key()]
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
        tc = getattr(config, "text_config", config)
        hidden = getattr(tc, "hidden_size", None)
        is_moe = bool(getattr(tc, "enable_moe_block", False))
        if hidden == 5376 and not is_moe:
            candidates.append("gemma-4-31B-it")
        elif hidden == 3840 and not is_moe:
            candidates.append("gemma-4-12B-it")
        elif is_moe:
            candidates.append("gemma-4-26B-A4B-it")
    except Exception:
        # Config inference is best-effort; fall back to the HF_MODEL basename.
        return tuple(dict.fromkeys(candidates))
    return tuple(dict.fromkeys(candidates))


def _mesh_key_from_node_name(node_name):
    """Extract the mesh-shape suffix (e.g. "1x1") from a pytest node name.

    parametrize_mesh_with_fabric appends an id like "1x1" / "1x2" / "1x8";
    pytest joins it with "-" as the trailing param. Returns None if no
    recognisable mesh-shape suffix is found.
    """
    if "[" not in node_name:
        return None
    inside = node_name[node_name.index("[") + 1 : node_name.rindex("]")]
    last = inside.rsplit("-", 1)[-1]
    if "x" in last and last.replace("x", "").isdigit():
        return last
    return None


def get_pcc_threshold(request, default=0.99):
    """Look up the PCC threshold for the current pytest node.

    Returns the per-test threshold from pcc_thresholds.json under
    [<model>][<mesh-shape>][<node-name>]. Falls back to ``default`` (0.99)
    when the entry is missing — that's the "haven't measured yet" path that
    new tests / unmeasured (model, system) combinations land on.

    Tests should call this with the pytest ``request`` fixture instead of
    hardcoding 0.95 / 0.90 inline so the table stays the single source of
    truth across runs.
    """
    table = _load_pcc_thresholds()
    node_name = request.node.name
    mesh_key = _mesh_key_from_node_name(node_name)
    # Try each candidate model key (HF_MODEL basename first, then the canonical
    # names inferred from the loaded config) so a HF_MODEL that drops the "-it"
    # suffix (e.g. google/gemma-4-26B-A4B) or points at a hashed cache snapshot
    # still resolves to the right entry instead of falling back to 0.99.
    # _lookup_model_entry matches case-insensitively.
    for model_key in _model_key_candidates():
        model_entry = _lookup_model_entry(table, model_key)
        mesh_entry = model_entry.get(mesh_key, {}) if mesh_key else {}
        if node_name in mesh_entry:
            return mesh_entry[node_name]
    return default


def is_moe_model():
    """Check if the current model has MoE enabled."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    except Exception as e:
        # IMPORTANT: this helper is evaluated at import time (see skip_if_not_moe),
        # so any failure here breaks *collection* for the whole Gemma4 unit-test suite.
        #
        # In environments where the HF checkpoint's `model_type="gemma4"` is not
        # recognized by the installed Transformers version (or where the model's
        # remote code isn't available offline), default to "not MoE" so only the
        # MoE-specific tests are skipped rather than crashing collection.
        import warnings

        warnings.warn(f"Unable to load HF config for is_moe_model(): {e}. Treating model as non-MoE.")
        return False

    tc = getattr(config, "text_config", config)
    return getattr(tc, "enable_moe_block", False)


skip_if_not_moe = pytest.mark.skipif(not is_moe_model(), reason="Model does not use MoE")

_GEMMA4_CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
_CONFIG_ONLY_SKIP_REASON = (
    "Real HF checkpoint required (weights + tokenizer); "
    "CI unit job uses config-only HF_MODEL under models/demos/gemma4/configs/"
)


def uses_ci_config_only_checkpoint():
    """True when HF_MODEL points at a checked-in config stub without weight files."""
    model_path = _get_model_path()
    if not os.path.isdir(model_path):
        return False
    resolved = os.path.abspath(model_path)
    if not resolved.startswith(_GEMMA4_CONFIGS_DIR + os.sep):
        return False
    if os.path.isfile(os.path.join(resolved, "model.safetensors")):
        return False
    if os.path.isfile(os.path.join(resolved, "pytorch_model.bin")):
        return False
    if any(name.startswith("model") and name.endswith(".safetensors") for name in os.listdir(resolved)):
        return False
    return True


def skip_if_config_only_checkpoint():
    """Skip tests that load HF weights or tokenizers when only config.json is available."""
    if uses_ci_config_only_checkpoint():
        pytest.skip(_CONFIG_ONLY_SKIP_REASON)


# ── Real checkpoint weights ────────────────────────────────────────────────
#
# ``create_hf_reference_layer`` and friends build HF modules from the *config*
# and leave the constructor's init in place — N(0, 0.02) for the router/expert
# tensors, all-ones for every RMSNorm scale, HF's default for the projections.
# That has none of the dynamic range or the outlier channels a trained
# checkpoint carries, which is exactly what bf16/bfp8 storage and multi-step
# accumulation get wrong. The helpers below pull the *real* tensors so a test
# can gate the device op at the magnitudes the model actually runs at.
#
# Only the requested tensors are read: the safetensors headers give a
# key -> shard map, then one ``get_tensor`` per key. Materializing a 31B
# state dict to fetch one projection would dominate every test that used it.


@lru_cache(maxsize=1)
def real_checkpoint_dir():
    """Local directory holding the checkpoint's safetensors shards.

    Cached: for a hub-id ``HF_MODEL`` this goes through ``snapshot_download``,
    which round-trips to the Hub to revalidate the revision even when every
    shard is already local. Once per process, not once per parametrized case.
    """
    model_path = _get_model_path()
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, allow_patterns=["*.safetensors", "*.safetensors.index.json"])


@lru_cache(maxsize=1)
def real_weight_index():
    """``{tensor key: shard path}`` from the safetensors headers — no tensor data read."""
    from pathlib import Path

    from safetensors import safe_open

    index = {}
    for shard in sorted(Path(real_checkpoint_dir()).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                index[key] = str(shard)
    return index


@lru_cache(maxsize=1)
def real_text_prefix():
    """Checkpoint prefix for the text stack, trailing dot included.

    The plain text releases name these ``model.layers.0.…`` and the unified
    multimodal ones ``model.language_model.layers.0.…``. Derived from wherever
    ``embed_tokens.weight`` lives rather than guessed, so a third naming
    convention resolves too.
    """
    for key in sorted(real_weight_index()):
        if key.endswith("embed_tokens.weight") and "per_layer" not in key:
            return key[: -len("embed_tokens.weight")]
    return "model."


def skip_unless_real_weights():
    """Skip unless HF_MODEL resolves to a checkpoint with readable safetensors."""
    skip_if_config_only_checkpoint()
    if not real_weight_index():
        pytest.skip(f"No safetensors shards under {real_checkpoint_dir()}; this test needs real weights")


def load_real_substate(relative_prefix):
    """Real weights under ``<text prefix><relative_prefix>.``, keys made relative.

    ``load_real_substate("layers.0.self_attn")`` returns ``{"q_proj.weight": …,
    "k_norm.weight": …}`` — the shape an HF submodule's ``load_state_dict``
    wants. fp32 tensors are cast to bf16 to match what
    ``Gemma4ModelArgs.load_state_dict`` hands the real model.
    """
    from collections import defaultdict

    from safetensors import safe_open

    assert relative_prefix, "relative_prefix must be non-empty — an empty one would load the whole text stack"
    skip_unless_real_weights()
    index = real_weight_index()
    prefix = f"{real_text_prefix()}{relative_prefix}."
    by_shard = defaultdict(list)
    for key in index:
        if key.startswith(prefix):
            by_shard[index[key]].append(key)
    if not by_shard:
        pytest.skip(f"Checkpoint {_get_model_path()} has no weights under '{prefix}'")

    out = {}
    for shard, keys in by_shard.items():
        with safe_open(shard, framework="pt") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                out[key[len(prefix) :]] = tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
    return out


def load_real_model_substate(num_layers):
    """Real weights for a *truncated* text stack, keyed for an HF reference model.

    Returns ``embed_tokens.weight``, ``layers.{0..num_layers-1}.…`` and
    ``norm.weight`` — deliberately not the whole checkpoint, so a 1-layer test
    reads one layer's tensors rather than all 48.
    """
    state = {}
    for name in ("embed_tokens", "norm"):
        for key, value in load_real_substate(name).items():
            state[f"{name}.{key}"] = value
    for layer_idx in range(num_layers):
        for key, value in load_real_substate(f"layers.{layer_idx}").items():
            state[f"layers.{layer_idx}.{key}"] = value
    return state


def load_real_weights_into(module, relative_prefix):
    """Load the checkpoint's weights for ``relative_prefix`` into an HF module.

    Returns the loaded state dict (keys relative to ``relative_prefix``) so a
    caller can log or reuse the tensors it just installed.

    ``strict=False``: the TT port drops tensors the reference module still
    declares (``v_norm``) and vice versa. Asserts that *something* landed, so a
    prefix typo fails loudly instead of quietly leaving the constructor init —
    which would make the test look like it passed on real weights.
    """
    state = load_real_substate(relative_prefix)
    _, unexpected = module.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    assert (
        loaded > 0
    ), f"No weight under '{relative_prefix}' matched {type(module).__name__}; unexpected={unexpected[:5]}"
    return state


class TestFactory:
    """Common test setup for Gemma4 unit tests."""

    BATCH_SEQ_CONFIGS = [
        (1, 1),  # Decode: single token
        (1, 128),  # Prefill: short sequence
    ]

    @staticmethod
    def create_hf_config():
        """Create Gemma4ModelArgs from the real model checkpoint (HF_MODEL env var)."""
        from transformers import AutoConfig

        model_path = _get_model_path()
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return Gemma4ModelArgs.from_hf_config(hf_config)

    @staticmethod
    def create_mesh_config(mesh_shape=(1, 1)):
        """Create a single-device MeshConfig for testing."""
        return MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1]))

    @staticmethod
    def create_random_state_dict(hf_config, prefix=""):
        """Generate random state dict for a given config and module prefix."""
        return {}

    @staticmethod
    def create_hf_text_config(num_experts=None, top_k=None):
        """Create HF Gemma4TextConfig from real model checkpoint.

        Optionally override num_experts/top_k for faster testing.
        """
        from transformers import AutoConfig

        model_path = _get_model_path()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tc = config.text_config
        if num_experts is not None:
            tc.num_experts = num_experts
        if top_k is not None:
            tc.top_k_experts = top_k
        tc._attn_implementation = "eager"
        return tc

    @staticmethod
    def create_hf_reference_layer(hf_text_config, layer_idx=0):
        """Create HF Gemma4TextDecoderLayer with random weights."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer as HFLayer

        hf_layer = HFLayer(hf_text_config, layer_idx=layer_idx)
        with torch.no_grad():
            for name, param in hf_layer.named_parameters():
                if any(k in name for k in ["router", "experts"]):
                    if "scale" in name:
                        param.data.fill_(1.0)
                    else:
                        param.data.normal_(0, 0.02)
            hf_layer.layer_scalar.fill_(1.0)
        hf_layer.eval()
        return hf_layer

    @staticmethod
    def create_hf_rope(hf_text_config, seq_len, layer_idx):
        """Create HF RoPE position embeddings (cos, sin) for torch reference."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        x_dummy = torch.randn(1, seq_len, hf_text_config.hidden_size)
        pos_ids = torch.arange(seq_len).unsqueeze(0)
        layer_type = hf_text_config.layer_types[layer_idx]
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        return cos, sin

    @staticmethod
    def create_tt_rope_cache(device, hf_text_config, max_seq_len, layer_idx):
        """Create HF-format cos/sin cache on TT device using HF Gemma4TextRotaryEmbedding.

        Returns (cos_cache, sin_cache) each [1, 1, max_seq_len, head_dim] on device.
        Matches exactly what HF produces (including identity padding for partial RoPE).
        """
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        x_dummy = torch.randn(1, max_seq_len, hf_text_config.hidden_size)
        pos_ids = torch.arange(max_seq_len).unsqueeze(0)
        layer_type = hf_text_config.layer_types[layer_idx]
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)
        # cos, sin: [1, max_seq_len, head_dim] -> [1, 1, max_seq_len, head_dim]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        is_mesh = hasattr(device, "shape")
        cos_tt = ttnn.from_torch(
            cos,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
        )
        sin_tt = ttnn.from_torch(
            sin,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
        )
        return cos_tt, sin_tt

    @staticmethod
    def create_tt_rope_cache_2d(device, hf_text_config, max_seq_len, layer_idx):
        """Create the 2D cos/sin cache used by the decode embedding-lookup RoPE path.

        Returns (cos_cache, sin_cache) each [max_seq_len, head_dim] on device,
        matching the layout of ``Gemma4Model.rope_caches_2d``. Decode attention
        detects the 2D shape and gathers per-user cos/sin via ``ttnn.embedding``
        (one row per user position), which is the path true batched decode takes.
        """
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        rope = Gemma4TextRotaryEmbedding(hf_text_config)
        x_dummy = torch.randn(1, max_seq_len, hf_text_config.hidden_size)
        pos_ids = torch.arange(max_seq_len).unsqueeze(0)
        layer_type = hf_text_config.layer_types[layer_idx]
        cos, sin = rope(x_dummy, pos_ids, layer_type=layer_type)  # [1, max_seq_len, head_dim]

        is_mesh = hasattr(device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(device) if is_mesh else None
        cos_tt = ttnn.from_torch(
            cos.squeeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        sin_tt = ttnn.from_torch(
            sin.squeeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        return cos_tt, sin_tt


def compare_tensors(tt_tensor, torch_tensor, mesh_device=None, pcc_threshold=0.99):
    """Compare TT and torch tensors using PCC. Logs the PCC value."""
    from loguru import logger

    from models.common.utility_functions import comp_pcc

    if isinstance(tt_tensor, torch.Tensor):
        tt_torch = tt_tensor
    else:
        tt_torch = ttnn.to_torch(tt_tensor)

    passing, pcc_value = comp_pcc(torch_tensor, tt_torch, pcc_threshold)
    status = "PASS" if passing else "FAIL"
    logger.info(f"PCC check: {pcc_value} (threshold={pcc_threshold}) [{status}]")
    return passing, pcc_value


def parametrize_batch_seq(configs=None, ids=None):
    """Parametrize test with batch/seq combinations.

    Default covers decode (seq=1) plus every PREFILL_BUCKETS length. Prefill
    lengths above the --max-prefill CLI option are auto-skipped at runtime by
    the _enforce_max_prefill fixture in conftest.py.
    """
    configs = configs or [(1, 1)] + [(1, L) for L in PREFILL_BUCKETS]
    ids = ids or ["decode" if seq_len == 1 else f"prefill_{seq_len}" for _, seq_len in configs]
    return pytest.mark.parametrize("batch_size, seq_len", configs, ids=ids)


# UMD device-enumeration failures that mean the *runner* is unhealthy (a dead
# ETH core, failed topology discovery, etc.) rather than a bug in our code. When
# `ttnn.get_num_devices()` raises one of these at import/collection time, there
# is no device to test on, so the honest outcome is a skip — not a fatal pytest
# collection error (exit 4) that masks which test was even meant to run.
_DEVICE_DISCOVERY_FAILURE_MARKERS = (
    "eth core heartbeat check failed",
    "topology discovery",
    "cluster initialization",
)


def _is_device_discovery_failure(exc):
    msg = str(exc).lower()
    return any(marker in msg for marker in _DEVICE_DISCOVERY_FAILURE_MARKERS)


def parametrize_mesh_with_fabric(mesh_shapes=None, device_params_extra=None):
    """Universal mesh parametrization with FABRIC_1D.

    Generates paired mesh_device + device_params parametrization for tests at
    any TP factor. Only includes mesh shapes that fit on the current system.

    ``device_params_extra`` (dict) is merged into every param's device_params —
    e.g. ``{"trace_region_size": 256_000_000}`` for tests that capture a trace.

    Fabric is enabled (FABRIC_1D) for multi-device shapes, and disabled for
    (1, 1). Launching fabric on a 1x1 mesh on a multi-device system fails the
    is_device_active() check because fabric expects every device in the system
    to be opened, but only device 0 is open in a 1x1 mesh.

    Default shapes: (1,1) single card, (1,2) N300, (1,8) T3K.

    When ``CI=true`` is set in the environment, only the largest mesh shape
    that fits on the current system is parametrized. This lets the same test
    entry in the pipeline yamls run on any SKU (N150, N300, T3K) without
    needing per-SKU ``-k "1xN"`` filters or duplicate yaml entries — each SKU
    automatically picks the largest mesh its device count supports.

    Usage:
        @parametrize_mesh_with_fabric()           # default: all shapes that fit
        @parametrize_mesh_with_fabric([(1,8)])     # explicit shapes

        pytest -k "1x1"   # single card (TP=1)         (manual / non-CI)
        pytest -k "1x2"   # N300 (TP=2)                (manual / non-CI)
        pytest -k "1x8"   # T3K (TP=8)                 (manual / non-CI)
    """
    try:
        num_devices = ttnn.get_num_devices()
    except RuntimeError as e:
        # Only swallow genuine hardware-enumeration failures (bad runner); let any
        # other RuntimeError propagate so real software regressions stay visible.
        if not _is_device_discovery_failure(e):
            raise
        params = [
            pytest.param(
                (1, 1),
                {"fabric_config": None, **dict(device_params_extra or {})},
                id="device-unavailable",
                marks=pytest.mark.skip(reason=f"Device discovery failed (unhealthy runner): {e}"),
            )
        ]

        def decorator(func):
            return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

        return decorator

    if mesh_shapes is None:
        all_shapes = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 32)]
        mesh_shapes = [s for s in all_shapes if s[0] * s[1] <= num_devices]
    else:
        # User-provided shapes: still filter to those that fit, so an explicit
        # mesh_shapes=[(1,8)] decorator gracefully skips on smaller systems.
        mesh_shapes = [s for s in mesh_shapes if s[0] * s[1] <= num_devices]

    # CI mode: pick only the largest fitting shape so that one yaml entry can
    # target multiple SKUs and let each runner select the appropriate mesh.
    if os.getenv("CI") == "true" and len(mesh_shapes) > 1:
        mesh_shapes = [max(mesh_shapes, key=lambda s: s[0] * s[1])]

    extra = dict(device_params_extra or {})
    # Match demo Fabric packet defaults so Tracy/PCC multi-device paths see the
    # same AG packing as text_demo_v2 (WH 31B → 6144). Explicit extra wins.
    # Skip on 1x1 (no fabric).
    router = None
    if "fabric_router_config" not in extra:
        from models.demos.gemma4.tt.ccl import fabric_router_config_from_env

        router = fabric_router_config_from_env()

    def _device_params_for(shape):
        params = {"fabric_config": None if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D, **extra}
        if shape != (1, 1) and router is not None and "fabric_router_config" not in params:
            params["fabric_router_config"] = router
        return params

    if not mesh_shapes:
        params = [
            pytest.param(
                (1, 1),
                _device_params_for((1, 1)),
                id="1x1",
                marks=pytest.mark.skip(reason="Not enough devices"),
            )
        ]
    else:
        params = [
            pytest.param(
                s,
                _device_params_for(s),
                id=f"{s[0]}x{s[1]}",
            )
            for s in mesh_shapes
        ]

    def decorator(func):
        return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

    return decorator
