# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The single normalisation point for Llama-3.1-8B model dimensions.

HF anchor: ``transformers.models.llama.configuration_llama.LlamaConfig``.

P5.1 scope (this file today): ``LlamaHFConfig`` + ``llama_hf_config()`` only.
P6.2 extends it with ``ModelArgs`` (real-checkpoint state-dict loading + weight-cache pathing) —
``DEC-014``, contract in ``bringup_log/03_OUTLINE.md`` §3.3.

Why this file exists at all — ``DEC-009`` / ``DEC-010``:

* Every ``tt/`` module takes ``hf_config`` as a **normalised object** and reads plain attributes
  (``hf_config.hidden_size``). No module ever calls ``getattr(hf_config, ..., default)``.
* ``rope_theta`` and the llama3 scaling parameters are read in **exactly one place**: here, through
  ``models/tt_transformers/tt/common.py:165`` ``get_rope_theta`` and ``:183`` ``get_rope_scaling``,
  both of which take a **dict**.

That second rule is not stylistic. Measured on this box (``transformers`` 5.12.1):

| expression                              | result                                          |
|-----------------------------------------|-------------------------------------------------|
| ``LlamaConfig(...).rope_theta``         | raises ``AttributeError`` (the attr is gone)    |
| ``getattr(cfg, "rope_theta", 10000.0)`` | **10000.0** — a silently wrong theta            |
| ``cfg.to_dict()``                       | has neither ``rope_theta`` nor ``rope_scaling`` |
| ``get_rope_theta(cfg.to_dict())``       | **500000.0** ✓                                  |

so the ``getattr`` pattern used at ``models/demos/gpt_oss_d_p/tt/model_config.py:76`` would produce
a RoPE that is wrong at every position with no exception anywhere — the highest-severity
silent-wrongness trap in this bring-up (``BRINGUP_RECIPE.md`` Appendix F.2).

Mechanical rule for the rest of P5/P6: *if a module needs a model dimension, it is a field on
``LlamaHFConfig``; if it is not there, add it there — do not reach past the object.*
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from models.tt_transformers.tt.common import get_rope_scaling, get_rope_theta

# `models/tt_transformers/tt/common.py:405` compute_llama3_parameters hard-codes these two as local
# constants (`:407`, `:408`) instead of reading them from the config, so a checkpoint that changed
# them would be silently ignored. Asserted at construction, and re-asserted in tt/rope.py from the
# fields below. DEC-007 / R-006.
_HARDCODED_LOW_FREQ_FACTOR = 1.0
_HARDCODED_HIGH_FREQ_FACTOR = 4.0

# `apply_scaling` (`common.py:437`) only implements "default" / "linear" / "llama3" and falls through
# silently for anything else, so the type is asserted rather than branched on.
_SUPPORTED_ROPE_TYPE = "llama3"


@dataclass(frozen=True)
class LlamaHFConfig:
    """Every model dimension a ``tt/`` module is allowed to need, all resolved and non-``None``.

    Frozen so a module cannot mutate what another module already read.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    hidden_act: str
    attention_bias: bool
    mlp_bias: bool
    # --- RoPE, resolved here and nowhere else (DEC-010) ---
    rope_theta: float
    rope_type: str
    rope_scaling_factor: float
    rope_orig_context_len: int
    rope_low_freq_factor: float
    rope_high_freq_factor: float

    @property
    def gqa_group_size(self) -> int:
        """Q heads per KV head. 32 / 8 = 4 for Llama-3.1-8B."""
        return self.num_attention_heads // self.num_key_value_heads


def llama_hf_config(source) -> LlamaHFConfig:
    """Normalise a config into ``LlamaHFConfig``.

    Args:
        source: a plain ``dict`` (the bundled ``config.json``, or ``llama_config_dims()``), or
            anything exposing ``to_dict()`` (a ``transformers`` ``PretrainedConfig``). Converted to
            a dict **first**, so the RoPE helpers — which take dicts — see one layout.

    Returns:
        A frozen ``LlamaHFConfig`` with every field non-``None``.

    Raises:
        TypeError: ``source`` is neither a dict nor ``to_dict()``-able.
        AssertionError: a required dimension is missing/``None``, the RoPE type is not ``llama3``,
            or the llama3 limb factors are not the ones the repo helper hard-codes.
    """
    if isinstance(source, dict):
        cfg = dict(source)
    elif hasattr(source, "to_dict"):
        cfg = dict(source.to_dict())
    else:
        raise TypeError(f"llama_hf_config expects a dict or an object with to_dict(); got {type(source).__name__}")

    # head_dim is absent from the Llama-3.1-8B config.json; HF derives it the same way
    # (python_env/lib/python3.12/site-packages/transformers/models/llama/configuration_llama.py:87-88).
    head_dim = cfg.get("head_dim")
    if head_dim is None:
        assert cfg["hidden_size"] % cfg["num_attention_heads"] == 0, (
            f"hidden_size {cfg['hidden_size']} not divisible by num_attention_heads "
            f"{cfg['num_attention_heads']}; head_dim cannot be derived"
        )
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    # --- the ONE place theta and scaling are read (DEC-010) ---
    theta = get_rope_theta(cfg)
    assert theta is not None, (
        "rope_theta resolved to None; refusing to build a RoPE. Never fall back to a default here — "
        "getattr(cfg, 'rope_theta', 10000.0) returns 10000.0 on transformers 5.x and would give a "
        "silently wrong RoPE at every position (BRINGUP_RECIPE.md Appendix F.2)."
    )
    scaling = get_rope_scaling(cfg)
    assert scaling is not None, "rope_scaling / rope_parameters resolved to None; Llama-3.1 requires llama3 scaling"
    assert (
        scaling.get("rope_type") == _SUPPORTED_ROPE_TYPE
    ), f"expected rope_type {_SUPPORTED_ROPE_TYPE!r}, got {scaling.get('rope_type')!r}"

    low = float(scaling["low_freq_factor"])
    high = float(scaling["high_freq_factor"])
    assert low == _HARDCODED_LOW_FREQ_FACTOR, (
        f"low_freq_factor is {low}, but models/tt_transformers/tt/common.py:407 hard-codes "
        f"{_HARDCODED_LOW_FREQ_FACTOR} and would silently ignore it"
    )
    assert high == _HARDCODED_HIGH_FREQ_FACTOR, (
        f"high_freq_factor is {high}, but models/tt_transformers/tt/common.py:408 hard-codes "
        f"{_HARDCODED_HIGH_FREQ_FACTOR} and would silently ignore it"
    )

    assert (
        cfg["num_attention_heads"] % cfg["num_key_value_heads"] == 0
    ), f"num_attention_heads {cfg['num_attention_heads']} % num_key_value_heads {cfg['num_key_value_heads']} != 0"

    out = LlamaHFConfig(
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        head_dim=int(head_dim),
        vocab_size=int(cfg["vocab_size"]),
        max_position_embeddings=int(cfg["max_position_embeddings"]),
        rms_norm_eps=float(cfg["rms_norm_eps"]),
        tie_word_embeddings=bool(cfg["tie_word_embeddings"]),
        hidden_act=str(cfg["hidden_act"]),
        attention_bias=bool(cfg["attention_bias"]),
        mlp_bias=bool(cfg["mlp_bias"]),
        rope_theta=float(theta),
        rope_type=str(scaling["rope_type"]),
        rope_scaling_factor=float(scaling["factor"]),
        rope_orig_context_len=int(scaling["original_max_position_embeddings"]),
        rope_low_freq_factor=low,
        rope_high_freq_factor=high,
    )

    # Nothing may be None: a None dimension propagates into a shape and fails somewhere unrelated.
    missing = [f.name for f in fields(out) if getattr(out, f.name) is None]
    assert not missing, f"LlamaHFConfig fields resolved to None: {missing}"
    return out


class RuntimeLlamaHFConfig(LlamaHFConfig):
    """A :class:`LlamaHFConfig` that the prefill engine may stamp ``max_seq_len`` onto.

    **This exists because the engine mutates the config it is handed.**
    ``models/demos/common/prefill/runners/prefill_runner.py:475`` does::

        hf_config = ADAPTER.load_hf_config()
        hf_config.max_seq_len = MAX_SEQ_LEN

    and ``ADDING_A_PREFILL_MODEL.md`` §1 states it as part of the contract ("The engine sets
    ``.max_seq_len`` on the returned config"). :class:`LlamaHFConfig` is ``frozen=True``
    (``DEC-009``), whose generated ``__setattr__`` raises ``FrozenInstanceError`` for **any**
    attribute name — declared field or not — as long as ``type(self) is LlamaHFConfig``. So
    returning a plain ``LlamaHFConfig`` from ``load_hf_config`` makes the runner die on its next
    line, before a single device op. ``DEC-100``.

    A subclass is the exact fix rather than a workaround: CPython's frozen ``__setattr__`` is
    ``if type(self) is cls or name in fields: raise``, so on a subclass instance every **declared
    dimension stays frozen** (``hidden_size``, ``rope_theta``, … all still raise) and only a new,
    undeclared attribute like ``max_seq_len`` gets through. The invariant ``DEC-009`` bought — no
    module can mutate a dimension another module already read — is preserved in full.

    Do not add fields here. ``max_seq_len`` is deliberately *not* declared: it is the engine's
    per-run knob, it is not a model dimension, and declaring it would re-freeze it.
    """


def runtime_llama_hf_config(source) -> RuntimeLlamaHFConfig:
    """:func:`llama_hf_config`, but returning the engine-mutable subclass.

    Every field is resolved and validated by :func:`llama_hf_config` first, so there is exactly one
    normalisation path and this adds no second place for the ``rope_theta`` trap (``R-014``) to
    reappear.
    """
    base = llama_hf_config(source)
    return RuntimeLlamaHFConfig(**{f.name: getattr(base, f.name) for f in fields(base)})


# ======================================================================================
# P6.2 — real-checkpoint state-dict loading + weight-cache pathing (`ModelArgs`)
# ======================================================================================
#
# Contract: `bringup_log/03_OUTLINE.md` §3.3. Template: `models/demos/minimax_m3/tt/model_config.py:22`
# (class), `:125` (`load_state_dict`), `:211` (`weight_cache_path`), `:246`
# (`get_state_dict_prefix`).
#
# Three things this class deliberately does NOT do, each a `DEC`:
#
# 1. **It does not convert keys to Meta naming, and it does not permute Q/K.** `DEC-039`. The
#    outline's `load_state_dict(..., convert_to_meta_format=True)` and the recipe's
#    `map_hf_to_meta_keys` / `convert_hf_qkv_to_meta_format` suggestion are both wrong *for this
#    package*, in two independent ways, and either one alone is a silent-wrongness bug.
# 2. **It does not subclass `models/tt_transformers/tt/model_config.py:539` `ModelArgs`** — that one
#    raises without `HF_MODEL` (`:705`) and pulls in the whole tt_transformers config machinery
#    (`R-005`).
# 3. **It owns no device tensors.** It is a host-side path/dict helper; every device tensor is built
#    by the module that uses it.


# Imported HERE rather than in the module header on purpose: `03_OUTLINE.md` §3.3,
# `05_DECISIONS.md:1083` and `scripts/verify_citations.py` all cite `LlamaHFConfig` and
# `llama_hf_config` by line number in the P5.1 section above. Adding three lines to the header
# would shift every one of them, which is exactly the class of silent doc rot Appendix F.7 exists
# to prevent. The P6.2 section is append-only for the same reason.
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

_META_KEY_NAMES = ("tok_embeddings", "attention_norm", "ffn_norm", "wq", "wk", "wv", "wo", "w1", "w2", "w3")

# Per-layer HF weight names, in the order 03_OUTLINE.md §4.1 tabulates them (9 per layer).
_PER_LAYER_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

# The three non-layer HF weight names. `lm_head.weight` is present because Llama-3.1-8B is UNTIED
# (`tie_word_embeddings: false`), unlike Llama-3.2-1B/3B — 03_OUTLINE.md §3.15.
_GLOBAL_KEYS = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")

# `get_state_dict_prefix` module aliases -> the HF key segment they own. The two norms are listed
# by their HF names so a caller cannot mistype one into a silent `{}` sub-dict.
_MODULE_KEY = {
    "layer": "",
    "self_attn": "self_attn",
    "mlp": "mlp",
    "input_layernorm": "input_layernorm",
    "post_attention_layernorm": "post_attention_layernorm",
    "embedding": "model.embed_tokens",
    "norm": "model.norm",
    "lm_head": "lm_head",
}
_GLOBAL_MODULES = ("embedding", "norm", "lm_head")

_DTYPE_TAG = {"BFLOAT16": "bf16", "BFLOAT8_B": "bfp8", "BFLOAT4_B": "bfp4", "FLOAT32": "fp32"}


def state_dict_uses_meta_keys(state_dict) -> bool:
    """True if any key looks Meta-named (``wq`` / ``w1`` / ``attention_norm`` / ...).

    Used as a tripwire, not a branch: this package consumes **HF** keys everywhere (``DEC-039``),
    and a Meta-keyed dict handed to a module produces an empty ``substate()`` rather than an error.
    """
    return any(
        any(f".{m}." in k or k.startswith(f"{m}.") or k.endswith(f".{m}.weight") for m in _META_KEY_NAMES)
        for k in state_dict
    )


class ModelArgs:
    """Host-side checkpoint loading, key auditing and weight-cache pathing for Llama-3.1-8B.

    Owns no device tensors and no model dimensions of its own — dimensions live on
    :class:`LlamaHFConfig` (``DEC-009``).
    """

    def __init__(self, mesh_device, *, weights_path=None, hf_config=None):
        """
        Args:
            mesh_device: the open mesh. Only ``.shape`` is read, and only by
                :meth:`weight_cache_path` — the mesh shape is **part of the cache path** because
                ``ttnn.as_tensor`` caches the already-sharded per-device tensor, so a cache written
                at one mesh shape is garbage at another (``R-017``).
            weights_path: the checkpoint directory. ``None`` reads ``HF_MODEL``.
            hf_config: a :class:`LlamaHFConfig`. ``None`` builds one from
                ``{weights_path}/config.json``.
        """
        self.mesh_device = mesh_device
        self.mesh_shape = tuple(mesh_device.shape)

        path = weights_path if weights_path is not None else os.getenv("HF_MODEL")
        assert path, (
            "ModelArgs needs a checkpoint directory: pass weights_path=..., or export HF_MODEL. "
            "(There is no default path and no silent dummy-weight fallback — an empty state_dict "
            "must be requested explicitly by the caller, 03_OUTLINE.md §1 convention 5.)"
        )
        self.weights_path = Path(path)
        assert self.weights_path.is_dir(), f"weights_path {self.weights_path} is not a directory"

        if hf_config is None:
            config_json = self.weights_path / "config.json"
            assert config_json.is_file(), f"{config_json} not found; pass hf_config= explicitly"
            with open(config_json) as f:
                hf_config = llama_hf_config(json.load(f))
        self.hf_config = hf_config

    # ---------------------------------------------------------------------------------
    # state dict
    # ---------------------------------------------------------------------------------
    @staticmethod
    def load_state_dict(weights_path) -> dict:
        """Load the safetensors checkpoint. Keys and layout stay **exactly HF** (``DEC-039``).

        Delegates to ``models/tt_transformers/tt/load_checkpoints.py:18`` ``load_hf_state_dict``,
        which reads ``model.safetensors.index.json`` when present (4 shards here) and
        ``model.safetensors`` otherwise.

        Returns:
            ``{hf_key: torch.Tensor}`` — **291** tensors for Llama-3.1-8B (``9*32 + 3``,
            ``03_OUTLINE.md`` §4.1), every projection in HF ``[out, in]`` layout, no bias anywhere,
            and Q/K **not** permuted.

        Why no ``convert_to_meta_format`` flag (``DEC-039``), restated where a future editor will
        see it before adding one back:

        * ``convert_hf_qkv_to_meta_format`` would ``reverse_permute`` Q and K, and
          ``tt/attention/weights.py:71`` ``load_attention_weights`` **already does that**
          (``DEC-033``). Doing both is the identity's inverse applied twice — measured at PCC
          0.9475 when the swizzle is simply omitted, and a double permute is no better.
        * ``map_hf_to_meta_keys`` would rename ``mlp`` -> ``feed_forward``, ``gate_proj`` -> ``w1``,
          ``self_attn`` -> ``attention``, ``q_proj`` -> ``wq``, ... Every module in this package is
          handed a **stripped HF sub-dict** (``substate(sd, "mlp")``), so renamed keys make every
          ``substate()`` return ``{}``; with a populated ``tensor_cache_path`` that is not even an
          error — the modules load whatever is in the cache. That is exactly the
          "a renamed key means a layer quietly runs on the wrong weights" failure ``G-WEIGHTS``
          exists to catch, so it is caught by :meth:`audit_state_dict_keys` instead of being
          created here.
        """
        from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict

        state_dict = load_hf_state_dict(str(weights_path))
        assert not state_dict_uses_meta_keys(state_dict), (
            "load_state_dict got Meta-named keys from the checkpoint; every tt/ module in this "
            "package strips HF names (DEC-039)"
        )
        return state_dict

    def expected_state_dict_keys(self, num_layers=None) -> set:
        """The exact HF key set this package consumes: ``9 * num_layers + 3``.

        Derived from ``hf_config``, never from the checkpoint, so a checkpoint that is *missing* a
        key fails instead of shrinking the expectation to match.
        """
        n = self.hf_config.num_hidden_layers if num_layers is None else num_layers
        keys = set(_GLOBAL_KEYS)
        for i in range(n):
            keys.update(f"model.layers.{i}.{k}" for k in _PER_LAYER_KEYS)
        assert len(keys) == 9 * n + 3, f"expected 9*{n}+3 = {9 * n + 3} keys, built {len(keys)}"
        return keys

    def audit_state_dict_keys(self, state_dict, num_layers=None):
        """``(missing, unused)`` — the two sets ``G-WEIGHTS`` prints and asserts empty.

        * ``missing`` = keys this package will look for and the checkpoint does not have. Each one
          is a module that silently falls back to the weight cache (or fails at its first matmul).
        * ``unused`` = keys the checkpoint carries and nothing here reads. Each one is either a
          feature this package does not implement or, far worse, the *renamed* form of a key that
          is simultaneously in ``missing``.
        """
        expected = self.expected_state_dict_keys(num_layers)
        actual = set(state_dict)
        return expected - actual, actual - expected

    # ---------------------------------------------------------------------------------
    # weight cache
    # ---------------------------------------------------------------------------------
    def weight_cache_path(self, dtype) -> Path:
        """``<root>/llama31_8b_d_p_<arch>_<N>dev/<rows>x<cols>/tensor_cache_<dtype>``.

        Layout mirrors ``models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:75`` so P6-P8 and
        P10 share one cache tree, with two additions:

        * the **mesh shape** segment (``4x8``) — mandatory, not cosmetic. ``ttnn.as_tensor`` caches
          the *already-sharded* per-device tensor, so a ``(1,1)`` cache replayed on ``(4,8)`` hands
          every chip the full unsharded weight. Nothing downstream notices; it presents as "one
          layer runs on garbage", first visible at ``G-MESH-KV`` two phases later (``R-017``).
        * the **dtype** segment — bf8_b and bf16 tensors are not interchangeable and ttnn's own
          filename suffix (``_dtype_<DT>_layout_<L>.tensorbin``) makes them coexist rather than
          conflict, which is worse: a bf16 run would read nothing and rebuild, silently doubling
          load time, while a *dtype-tagged* directory keeps the two trees separate and inspectable.

        Root, in order: ``$LLAMA31_8B_TTNN_CACHE``, ``$TT_CACHE_PATH``, then
        ``{weights_path}/ttnn_cache``.
        """
        from models.common.utility_functions import is_blackhole

        root = os.getenv("LLAMA31_8B_TTNN_CACHE") or os.getenv("TT_CACHE_PATH")
        root = Path(root) if root else self.weights_path / "ttnn_cache"

        arch = "bh" if is_blackhole() else "wh"
        rows, cols = self.mesh_shape
        tag = _DTYPE_TAG.get(str(dtype).rsplit(".", 1)[-1].upper())
        assert tag, f"no cache tag for dtype {dtype}; add it to _DTYPE_TAG rather than sharing a directory"

        path = root / f"llama31_8b_d_p_{arch}_{rows * cols}dev" / f"{rows}x{cols}" / f"tensor_cache_{tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---------------------------------------------------------------------------------
    # key prefixes
    # ---------------------------------------------------------------------------------
    @staticmethod
    def get_state_dict_prefix(module_name, layer_idx=None) -> str:
        """The HF key prefix for one module, e.g. ``("self_attn", 3) -> "model.layers.3.self_attn."``.

        ``module_name`` is one of ``layer``, ``self_attn``, ``mlp``, ``input_layernorm``,
        ``post_attention_layernorm``, ``embedding``, ``norm``, ``lm_head`` — asserted, not
        best-effort, because a typo would otherwise produce a valid-looking prefix that matches
        nothing and a module built on an empty sub-dict.
        """
        assert module_name in _MODULE_KEY, f"unknown module {module_name!r}; expected one of {sorted(_MODULE_KEY)}"
        if module_name in _GLOBAL_MODULES:
            assert layer_idx is None, f"{module_name} is not a per-layer module; got layer_idx={layer_idx}"
            return f"{_MODULE_KEY[module_name]}."
        assert layer_idx is not None, f"{module_name} is per-layer; pass layer_idx"
        seg = _MODULE_KEY[module_name]
        return f"model.layers.{layer_idx}." + (f"{seg}." if seg else "")
