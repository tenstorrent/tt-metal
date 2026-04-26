"""HuggingFace state-dict loader for Gemma4.

Phase 2 source-of-truth for static weights. Reads safetensors via
`safetensors.safe_open` (mmap-backed; no full-tensor reads until each
slot is accessed). Returns a torch state_dict keyed by HF parameter
names.

Lifted constants (rotary inv_freq, embed_scale) are NOT in the
safetensors; they're produced by `extract_lifted_constants()` which
instantiates transformers' Gemma4TextRotaryEmbedding directly on CPU
(small) to read its computed buffers.

Naming note: gemma-4-31B-it is a multimodal model. The language-model
weights live under `model.language_model.X` in the HF state_dict, but
the codegen's `arg_mapping.json` refers to them as `model.X` (relative
to the language model). `slot_to_hf_key()` adds the
`model.language_model.` prefix where needed.
"""
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Dict

import torch
from safetensors import safe_open
from transformers import AutoConfig

_DEFAULT_HF_CACHE = pathlib.Path("~/.cache/huggingface/hub").expanduser()
_HF_MODEL_NAME = "google/gemma-4-31B-it"


@dataclass
class HfWeights:
    """Bundle of everything needed to construct gemma4 classes from HF.

    `state_dict`: torch.Tensor map keyed by HF param name.
    `lifted`: dict of computed buffers not in safetensors.
    `config`: transformers.Gemma4Config (raw HF config).
    """

    state_dict: Dict[str, torch.Tensor]
    lifted: Dict[str, torch.Tensor]
    config: object


def load_hf_weights(model_name: str = _HF_MODEL_NAME, cache_dir: pathlib.Path = _DEFAULT_HF_CACHE) -> HfWeights:
    """Load gemma-4-31B-it weights from a local HF cache.

    Returns HfWeights with state_dict (from safetensors), lifted
    constants (from a meta-instantiated transformers model), and the
    HF config. Tensors are CPU-resident; the caller moves them onto
    the ttnn mesh.
    """
    snapshot_dir = _resolve_snapshot_dir(model_name, cache_dir)
    state_dict: Dict[str, torch.Tensor] = {}
    for shard in sorted(snapshot_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    lifted = extract_lifted_constants(config)
    return HfWeights(state_dict=state_dict, lifted=lifted, config=config)


def _resolve_snapshot_dir(model_name: str, cache_dir: pathlib.Path) -> pathlib.Path:
    """Resolve the symlinked snapshot directory under HF's cache layout."""
    safe_name = "models--" + model_name.replace("/", "--")
    snapshots = cache_dir / safe_name / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(
            f"HF snapshot not found at {snapshots}. " f"Run `huggingface-cli download {model_name}` first."
        )
    candidates = sorted(snapshots.iterdir())
    if not candidates:
        raise FileNotFoundError(f"No snapshots under {snapshots}")
    return candidates[0]


def extract_lifted_constants(config) -> Dict[str, torch.Tensor]:
    """Meta-instantiate Gemma4ForCausalLM to read computed buffers.

    Buffers in question:
      - model.language_model.rotary_emb.sliding_attention_inv_freq
      - model.language_model.rotary_emb.full_attention_inv_freq
      - model.language_model.embed_tokens.embed_scale

    Uses torch's "meta" device so weights are not allocated; only the
    buffer registrations run (which compute the inv_freq tables and
    embed_scale = sqrt(hidden_size)).

    Returns keys that mirror codegen-arg-mapping naming (`model.X` —
    without the `language_model.` prefix), matching how
    `arg_mapping.json` refers to these constants.
    """
    # Direct sub-module instantiation on CPU (the rotary buffers and
    # embed_scale are O(KB), no full-model allocation needed).
    # gemma-4-31B-it is a multimodal model with nested text_config.
    import math

    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    text_config = getattr(config, "text_config", config)
    rotary = Gemma4TextRotaryEmbedding(text_config, device="cpu")
    embed_scale_value = math.sqrt(text_config.hidden_size)
    out = {}
    out["model.rotary_emb.sliding_attention_inv_freq"] = rotary.sliding_attention_inv_freq.detach().clone()
    out["model.rotary_emb.full_attention_inv_freq"] = rotary.full_attention_inv_freq.detach().clone()
    out["model.embed_tokens.embed_scale"] = torch.tensor(embed_scale_value, dtype=torch.bfloat16)
    return out


# ============================================================================
# Slot-name → HF-state_dict-key translation.
# ============================================================================

# Cached arg_mapping.json contents per file
_ARG_MAPPING_CACHE: Dict[str, Dict[int, dict]] = {}


def _load_arg_mapping(arg_mapping_path: pathlib.Path) -> Dict[int, dict]:
    key = str(arg_mapping_path)
    if key not in _ARG_MAPPING_CACHE:
        with open(arg_mapping_path) as f:
            raw = json.load(f)
        _ARG_MAPPING_CACHE[key] = {int(k): v for k, v in raw.items()}
    return _ARG_MAPPING_CACHE[key]


# Encoded leaf-path → HF dotted leaf path
_LEAF_DECODE = {
    # L__ entries (no __parameters__ suffix)
    "input_layernorm_weight": "input_layernorm.weight",
    "post_attention_layernorm_weight": "post_attention_layernorm.weight",
    "pre_feedforward_layernorm_weight": "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm_weight": "post_feedforward_layernorm.weight",
    "self_attn_q_norm_weight": "self_attn.q_norm.weight",
    "self_attn_k_norm_weight": "self_attn.k_norm.weight",
    "layer_scalar": "layer_scalar",
    # self___ entries (with __parameters__weight suffix)
    "self_attn_q_proj__parameters__weight": "self_attn.q_proj.weight",
    "self_attn_k_proj__parameters__weight": "self_attn.k_proj.weight",
    "self_attn_v_proj__parameters__weight": "self_attn.v_proj.weight",
    "self_attn_o_proj__parameters__weight": "self_attn.o_proj.weight",
    "mlp_gate_proj__parameters__weight": "mlp.gate_proj.weight",
    "mlp_up_proj__parameters__weight": "mlp.up_proj.weight",
    "mlp_down_proj__parameters__weight": "mlp.down_proj.weight",
}

_LAYERED_PREFIX_RE = re.compile(r"^(?:L__self___|self___)model_layers__modules__(\d+)___(.+)$")

# Top-level (non-per-layer) entries
_TOP_LEVEL_DECODE = {
    "model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
    "model.norm.weight": "model.language_model.norm.weight",
    "self___lm_head__parameters__weight": "model.language_model.embed_tokens.weight",  # tied
    # Lifted constants (NOT in state_dict; signaled to caller)
    "model.embed_tokens.embed_scale": "<LIFTED>model.embed_tokens.embed_scale",
    "model.rotary_emb.sliding_attention_inv_freq": "<LIFTED>model.rotary_emb.sliding_attention_inv_freq",
    "model.rotary_emb.full_attention_inv_freq": "<LIFTED>model.rotary_emb.full_attention_inv_freq",
    # Tracer-lifted scalar bf16 constants (softcap / rms_eps / similar);
    # actual derivation handled via `lifted_tensor_value` lookup at use-site.
    "L['self'].model.lifted_tensor_0": "<LIFTED>lifted_tensor_0",
    "L['self'].model.lifted_tensor_1": "<LIFTED>lifted_tensor_1",
}


def slot_to_hf_key(slot_name: str) -> str:
    """Translate a codegen-slot name (from `arg_mapping.json`) to an
    HF state_dict key. Returns a key prefixed with '<LIFTED>' if the
    target is a lifted constant rather than a state_dict entry. Raises
    ValueError if the slot is a runtime arg or otherwise unrecognizable.
    """
    if slot_name.startswith("args_"):
        raise ValueError(f"slot {slot_name!r} is a runtime arg, not a weight")
    if slot_name in _TOP_LEVEL_DECODE:
        return _TOP_LEVEL_DECODE[slot_name]
    m = _LAYERED_PREFIX_RE.match(slot_name)
    if m:
        layer_idx, leaf = m.group(1), m.group(2)
        if leaf not in _LEAF_DECODE:
            raise ValueError(f"unknown leaf path {leaf!r} (layer {layer_idx})")
        decoded_leaf = _LEAF_DECODE[leaf]
        return f"model.language_model.layers.{layer_idx}.{decoded_leaf}"
    raise ValueError(f"unrecognized slot name: {slot_name!r}")


# ============================================================================
# Per-role mesh-mapper lookup.
# ============================================================================

# Role → (sharding_kind, dim). 'replicate' has dim=None.
# Each gemma4-class from_state_dict picks a role from this table and asks
# `mesh_mapper_for_role(role, mesh_device)` to construct the right ttnn
# mesh mapper for `ttnn.as_tensor(...)`.
#
# The dim choices are educated guesses based on each weight's role in
# the matmul / norm op; the allclose verification harness in Task 5
# catches any wrong dim immediately by comparing against the consteval
# cache.
_ROLE_TO_SHARDING = {
    "input_layernorm": ("shard", 0),
    "post_attention_layernorm": ("shard", 0),
    "pre_feedforward_layernorm": ("shard", 0),
    "post_feedforward_layernorm": ("shard", 0),
    "norm": ("shard", 0),
    "q_proj": ("shard", 0),
    "k_proj": ("shard", 0),
    "v_proj": ("shard", 0),
    "o_proj": ("shard", 1),
    "q_norm": ("replicate", None),
    "k_norm": ("replicate", None),
    "gate_proj": ("shard", 0),
    "up_proj": ("shard", 0),
    "down_proj": ("shard", 1),
    "embed_tokens": ("shard", 1),
    "lm_head": ("shard", 0),  # sharded along vocab dim (matmul w transpose_b)
    "embed_scale": ("replicate", None),
    "rotary_inv_freq": ("replicate", None),
    "layer_scalar": ("replicate", None),
    "lifted_scalar": ("replicate", None),
}


def mesh_mapper_for_role(role: str, mesh_device):
    """Return the right ttnn mesh mapper for a given weight role."""
    import ttnn  # imported here to avoid module-load failure on import-only flows

    if role not in _ROLE_TO_SHARDING:
        raise ValueError(f"unknown role for mesh sharding: {role!r}")
    kind, dim = _ROLE_TO_SHARDING[role]
    if kind == "replicate":
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.ShardTensorToMesh(mesh_device, dim=dim)


def role_for_hf_key(hf_key: str) -> str:
    """Identify the role tag for an HF state_dict key."""
    if hf_key == "model.language_model.embed_tokens.weight":
        # Could be either embed_tokens or lm_head (tied); caller picks the
        # appropriate role.
        return "embed_tokens"
    if hf_key == "model.language_model.norm.weight":
        return "norm"
    if hf_key == "model.embed_tokens.embed_scale":
        return "embed_scale"
    if hf_key.startswith("model.rotary_emb.") and hf_key.endswith("_inv_freq"):
        return "rotary_inv_freq"
    if hf_key.startswith("model.language_model.layers."):
        # model.language_model.layers.0.self_attn.q_proj.weight  →  'q_proj'
        # model.language_model.layers.0.input_layernorm.weight   →  'input_layernorm'
        # model.language_model.layers.0.layer_scalar             →  'layer_scalar'
        parts = hf_key.split(".")
        # parts: ['model', 'language_model', 'layers', '0', ...rest...]
        rest = parts[4:]
        if rest[-1] == "weight":
            rest = rest[:-1]
        if len(rest) == 1:
            return rest[0]  # input_layernorm / layer_scalar
        if len(rest) == 2:
            return rest[1]  # self_attn.q_proj → q_proj
    raise ValueError(f"unknown HF key for role lookup: {hf_key!r}")


def _verify_slot_translation(arg_mapping_path: pathlib.Path, hf_state_dict: Dict[str, torch.Tensor]) -> list:
    """Sanity check: every non-runtime slot's HF key resolves in the
    state_dict (or is a lifted-constant marker).
    """
    mapping = _load_arg_mapping(arg_mapping_path)
    missing = []
    for slot, info in mapping.items():
        name = info["name"]
        if name.startswith("args_"):
            continue
        try:
            key = slot_to_hf_key(name)
        except ValueError as e:
            missing.append((slot, name, str(e)))
            continue
        if key.startswith("<LIFTED>"):
            continue  # lifted-constant marker; handled separately
        if key not in hf_state_dict:
            missing.append((slot, name, f"key {key!r} not in HF state_dict"))
    return missing


# ============================================================================
# HF helper utilities for prelude construction.
#
# `apply_hf_scalar_overrides` builds the no-input scalar tensors (zeros,
# fills, aranges, one-hot mask helpers) that the prelude classes consume
# during construction. The result is a transient dict; once the preludes
# (and other from_state_dict consumers) have extracted what they need,
# the dict is discarded.
# ============================================================================

_HF_BUNDLE_CACHE: Dict[str, "HfWeights"] = {}


def get_hf_weights_cached() -> "HfWeights":
    """Module-level cache for the HF state_dict + lifted constants.
    First call takes ~10-30s; subsequent calls are O(1).
    """
    if "default" not in _HF_BUNDLE_CACHE:
        _HF_BUNDLE_CACHE["default"] = load_hf_weights()
    return _HF_BUNDLE_CACHE["default"]


# Top-level cached_main key map. Per-side because the codegen-key
# numbering differs between prefill and decode.
_TOP_LEVEL_KEYS = {
    "prefill": {
        "embed_weight": "main_const_eval_436",
        "embed_scale": "main_const_eval_255",
        "final_norm": "main_const_eval_50",
        "lm_head_weight": "main_const_eval_171",
    },
    "decode": {
        "embed_weight": "main_const_eval_435",
        "embed_scale": "main_const_eval_255",
        "final_norm": "main_const_eval_254",
        "lm_head_weight": "main_const_eval_50",
    },
}

# RMSNorm role → _LAYER_TABLE weight-key field
_RMSNORM_LT_KEYS = {
    "input_layernorm": "input_layernorm",
    "post_attention_layernorm": "post_attn_ln",
    "pre_feedforward_layernorm": "pre_ff_ln",
    "post_feedforward_layernorm": "post_ff_ln",
}


def apply_hf_scalar_overrides(cached_main: dict, hf, mesh_device, *, is_decode: bool) -> None:
    """Override no-input scalar consteval keys with HF-config-derived
    ttnn.Tensors.

    Each `main_const_eval_X` no-input function in the legacy consteval.py
    builds a constant tensor (zeros / full / Tensor-literal) with no
    inputs — meaning its output is purely a function of the codegen
    artifact, not the model state. We recreate each one exactly using
    ttnn.as_tensor + ReplicateTensorToMesh.

    Recipes are encoded in the per-side tables below (verified by
    reading gemma4_{prefill,decode}/consteval.py).
    """
    import torch as _torch

    import ttnn

    rms_eps = hf.config.text_config.rms_norm_eps
    softcap = hf.config.text_config.final_logit_softcapping or 30.0

    _DT = {
        "BFLOAT16": ttnn.DataType.BFLOAT16,
        "FLOAT32": ttnn.DataType.FLOAT32,
        "INT32": ttnn.DataType.INT32,
        "UINT32": ttnn.DataType.UINT32,
    }
    _TORCH_DT = {
        "BFLOAT16": _torch.bfloat16,
        "FLOAT32": _torch.float32,
        "INT32": _torch.int32,
        "UINT32": _torch.int32,  # torch lacks uint32; ttnn casts on as_tensor
    }

    _LAYOUT = {
        "TILE": ttnn.Layout.TILE,
        "ROW_MAJOR": ttnn.Layout.ROW_MAJOR,
    }

    def _build(
        shape,
        dtype,
        *,
        fill=None,
        arange_start=None,
        arange_mod=None,
        arange_clamp=None,
        one_hot_last=False,
        layout="TILE",
    ):
        torch_dt = _TORCH_DT[dtype]
        if fill is not None:
            t = _torch.full(list(shape), float(fill), dtype=torch_dt)
        elif arange_start is not None:
            n = 1
            for s in shape:
                n *= s
            base = _torch.arange(arange_start, arange_start + n, dtype=_torch.int64)
            if arange_mod is not None:
                base = base % arange_mod
            if arange_clamp is not None:
                base = base.clamp(min=arange_clamp[0], max=arange_clamp[1])
            t = base.to(torch_dt).reshape(*shape)
        elif one_hot_last:
            # Tensor of zeros, with last element = 1.0
            t = _torch.zeros(list(shape), dtype=torch_dt)
            t.flatten()[-1] = 1.0
        else:
            t = _torch.zeros(list(shape), dtype=torch_dt)
        return ttnn.as_tensor(
            t,
            dtype=_DT[dtype],
            layout=_LAYOUT[layout],
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Per-side recipes for SAFE no-input scalars (zeros, simple-fill).
    # Skipped for now: ce_0 (multi-output with floor+clamp+gt computation),
    # ce_123/266/242 prefill (Tensor literals dependent on seq_len/range),
    # ce_242/316/334/509 decode (Tensor literal arange-style arrays).
    # Those will be migrated alongside the RoPESetup work in a future
    # pass.
    # NOTE: We skip ce_535 (decode) and ce_536 (prefill): those are
    # Tensor literals with shape [1,1,256,1] bf16 where most values are
    # 0.0 but the last position is 1.0 (a one-hot mask). That's not a
    # plain zeros and would need a special recipe — defer to RoPESetup
    # work where these mask-helper tensors are reanalyzed.
    # bf16 attention-mask sentinel: torch.finfo(bf16).min ≈ -3.3895e+38.
    # Standard pattern for "masked attention slot" — using bf16-min instead
    # of -inf avoids NaN propagation through softmax.
    bf16_min = float(_torch.finfo(_torch.bfloat16).min)
    if is_decode:
        recipes = [
            # zeros / fills
            ("main_const_eval_123", [(1, 1, 1, 1), "BFLOAT16", {"fill": float("-inf")}]),
            ("main_const_eval_171", [(1, 1, 1), "BFLOAT16", {"fill": softcap}]),
            ("main_const_eval_240", [(1, 1, 1), "BFLOAT16", {"fill": rms_eps}]),
            ("main_const_eval_266", [(1,), "INT32", {"fill": 256}]),
            ("main_const_eval_337", [(1, 1, 1, 1), "FLOAT32", {"fill": 0.0}]),
            ("main_const_eval_400", [(1, 1), "INT32", {"fill": 0}]),
            ("main_const_eval_486", [(1, 1, 1, 1), "BFLOAT16", {"fill": 0.0}]),
            ("main_const_eval_489", [(1, 1, 1, 1), "FLOAT32", {"fill": float("-inf")}]),
            # lifted_tensor_0/1 — attention mask sentinels (consumed via
            # `where(cond, ce_lifted0, ce_lifted1)` inside the prelude).
            ("main_const_eval_621", [(1, 1, 1, 1), "BFLOAT16", {"fill": 0.0}]),  # lifted_tensor_0
            ("main_const_eval_543", [(1, 1, 1, 1), "BFLOAT16", {"fill": bf16_min}]),  # lifted_tensor_1
            # arange-style position arrays
            ("main_const_eval_242", [(256,), "INT32", {"arange_start": 0}]),
            ("main_const_eval_316", [(1, 1, 1, 256), "INT32", {"arange_start": 0}]),
            (
                "main_const_eval_334",
                [(1, 256), "UINT32", {"arange_start": 1, "arange_mod": 256, "layout": "ROW_MAJOR"}],
            ),
            ("main_const_eval_509", [(1, 256), "UINT32", {"arange_start": 0, "layout": "ROW_MAJOR"}]),
            # one-hot mask helper
            ("main_const_eval_535", [(1, 1, 256, 1), "BFLOAT16", {"one_hot_last": True}]),
        ]
    else:
        recipes = [
            # zeros / fills
            ("main_const_eval_186", [(1, 1, 1, 1), "FLOAT32", {"fill": 0.0}]),
            ("main_const_eval_217", [(1, 1, 1, 1), "BFLOAT16", {"fill": float("-inf")}]),
            ("main_const_eval_240", [(1, 19, 1), "BFLOAT16", {"fill": rms_eps}]),
            ("main_const_eval_242", [(1,), "INT32", {"fill": 256}]),
            ("main_const_eval_314", [(1, 1, 1), "BFLOAT16", {"fill": softcap}]),
            ("main_const_eval_335", [(1, 1, 1, 1), "BFLOAT16", {"fill": 0.0}]),
            ("main_const_eval_338", [(1, 1, 1, 1), "FLOAT32", {"fill": float("-inf")}]),
            ("main_const_eval_544", [(1, 1), "INT32", {"fill": 0}]),
            # lifted_tensor_0/1 — attention mask sentinels (consumed via
            # `where(cond, ce_lifted0, ce_lifted1)` inside the prelude).
            ("main_const_eval_490", [(1, 1, 1, 1), "BFLOAT16", {"fill": 0.0}]),  # lifted_tensor_0
            ("main_const_eval_622", [(1, 1, 1, 1), "BFLOAT16", {"fill": bf16_min}]),  # lifted_tensor_1
            # arange-style position arrays
            ("main_const_eval_123", [(1, 256), "UINT32", {"arange_start": 0, "layout": "ROW_MAJOR"}]),
            (
                "main_const_eval_266",
                [(1, 256), "UINT32", {"arange_start": 19, "arange_mod": 256, "layout": "ROW_MAJOR"}],
            ),
            ("main_const_eval_401", [(256,), "INT32", {"arange_start": 0}]),
            ("main_const_eval_510", [(1, 1, 1, 256), "INT32", {"arange_start": 0}]),
            ("main_const_eval_627", [(19,), "INT32", {"arange_start": 0}]),
            # one-hot mask helper
            ("main_const_eval_536", [(1, 1, 256, 1), "BFLOAT16", {"one_hot_last": True}]),
        ]

    for ce_key, recipe in recipes:
        shape, dtype, kwargs = recipe
        cached_main[ce_key] = [_build(shape, dtype, **kwargs)]

    # ce_0: multi-output (3 returns). Inputs to the prelude as
    # var_184 / var_189 / one_hot-style position helper.
    #   Output 1: zeros [1] INT32 TILE
    #   Output 2: full[1] INT32 TILE — fill=19 (prefill), fill=1 (decode)
    #   Output 3: clamp(arange(start, start+256), lo, hi).reshape(256,1)
    #             UINT32 ROW_MAJOR
    #             prefill: arange(-237, 19), clamp(0, 18)
    #             decode:  arange(-255, 1),  clamp(0, 0)  → all zeros
    if is_decode:
        ce0_arange_start = -255
        ce0_clamp = (0, 0)
        ce0_full_fill = 1
    else:
        ce0_arange_start = -237
        ce0_clamp = (0, 18)
        ce0_full_fill = 19
    cached_main["main_const_eval_0"] = [
        _build((1,), "INT32", fill=0),
        _build((1,), "INT32", fill=ce0_full_fill),
        _build(
            (256, 1),
            "UINT32",
            arange_start=ce0_arange_start,
            arange_clamp=ce0_clamp,
            layout="ROW_MAJOR",
        ),
    ]
