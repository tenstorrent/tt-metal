# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The model's precision policy, as one value that construction consumes.

Before this module the policy was module-level constants in
``optimized_decoder.py`` (``EXPERT_WEIGHT_DTYPE``, ``EXPERT_MATH_FIDELITY``,
``ATTENTION_WEIGHT_DTYPE``, the two ``EXPERT_IN0_BLOCK_W_*``) plus a handful of
literal ``dtype=ttnn.bfloat16`` arguments scattered through ``model.py`` and
``multichip_decoder.py``. That is enough to *ship* a policy but not enough to
*sweep* one: varying any of it meant editing source between runs, and a JSON
file written next to unchanged source would be a claim rather than a
configuration.

``PrecisionConfig`` is the value those constants became. It is frozen, it
round-trips through JSON losslessly (:meth:`to_json` / :meth:`from_json`), and
every field below is read at model-construction or forward time by code that
would behave differently if the field changed. The module constants still exist
-- probes and stage-02/04 tests import them by name -- but they are now
*derived* from ``DEFAULT_PRECISION`` rather than being the source of truth, so
there is exactly one place a shipped value is written down.

**The default is the shipped policy, and stage 07 moved it.** When this module
was introduced its default reproduced stages 02-06 exactly. It no longer does:
the stage-07 sweep selected new values for the two expert block widths
(``experts_gate_up_in0_block_w`` 16 -> 64, ``experts_down_in0_block_w`` 12 ->
24) and ``DEFAULT_PRECISION`` carries them, because the goal requires the
selection to be what the default construction path consumes. **Every other
field still reads back exactly what stages 02-06 measured**, and the two that
moved are scheduling choices rather than numerical ones -- the graph is the same
graph and the tokens are the same tokens; only the expert matmuls' inner
blocking differs. See the block-width fields below and
``doc/datatype_sweep/README.md``.

Three fields are ``None`` by default and that is deliberate rather than an
omission:

``attention_fidelity``
    The attention projections pass no ``compute_kernel_config`` today, so they
    take the op default. ``None`` reproduces that exactly; any other value
    builds a config and passes it. Encoding "op default" as an explicit
    ``MathFidelity`` would be a guess about what the op picks.
``ccl_dtype``
    The collectives run at whatever dtype the activation arrives in. ``None``
    means "inherit", which is today's behaviour and costs no ops; a named dtype
    casts into and out of the collective.
``experts_gate_up_fidelity`` has no ``None`` counterpart -- the expert matmuls
    have always passed an explicit config -- so it is a plain value.

The block widths live here too. They are not dtypes, but they were tuned
*against* the dtype, so a sweep that varies expert dtype and cannot vary the
block width alongside it would be measuring a mis-tuned point.

That mattered more than expected. Stage 07's sweep found the two block widths
to be **the only fields worth moving in the entire twenty-field config**: taking
each to its full-K ceiling (gate_up 16 -> 64, down 12 -> 24) bought +2.83%
traced decode at bit-identical accuracy, while every dtype lever the sweep tried
either regressed, landed inside the run-to-run band, or hit a TTNN blocker. The
old values were inherited from single-chip stage-02 tuning and expert
parallelism had since cut per-die N four-fold, changing which blocking the
matmul wants. See ``doc/datatype_sweep/README.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import ttnn

__all__ = [
    "PrecisionConfig",
    "DEFAULT_PRECISION",
    "dtype_from_name",
    "dtype_to_name",
    "fidelity_from_name",
    "fidelity_to_name",
]


# --- name <-> object tables ---------------------------------------------------
#
# Spelled out rather than derived from ``str(dtype)`` so the JSON is a stable
# contract: a rename inside ttnn's binding would silently invalidate every
# archived config if the names were scraped.

_DTYPES: dict[str, "ttnn.DataType"] = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "float32": ttnn.float32,
    "uint8": ttnn.uint8,
    "uint16": ttnn.uint16,
    "int32": ttnn.int32,
    "uint32": ttnn.uint32,
}
_DTYPE_NAMES: dict["ttnn.DataType", str] = {v: k for k, v in _DTYPES.items()}

_FIDELITIES: dict[str, "ttnn.MathFidelity"] = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
_FIDELITY_NAMES: dict["ttnn.MathFidelity", str] = {v: k for k, v in _FIDELITIES.items()}


def dtype_from_name(name):
    """``"bfloat4_b"`` -> ``ttnn.bfloat4_b``. ``None`` and ttnn dtypes pass through."""
    if name is None or isinstance(name, ttnn.DataType):
        return name
    try:
        return _DTYPES[str(name)]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; known: {sorted(_DTYPES)}") from None


def dtype_to_name(dtype):
    if dtype is None:
        return None
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError:
        raise ValueError(f"dtype {dtype!r} has no serialised name; add it to precision._DTYPES") from None


def fidelity_from_name(name):
    """``"LoFi"`` -> ``ttnn.MathFidelity.LoFi``. ``None`` and fidelities pass through."""
    if name is None or isinstance(name, ttnn.MathFidelity):
        return name
    try:
        return _FIDELITIES[str(name)]
    except KeyError:
        raise ValueError(f"unknown math fidelity {name!r}; known: {sorted(_FIDELITIES)}") from None


def fidelity_to_name(fidelity):
    if fidelity is None:
        return None
    try:
        return _FIDELITY_NAMES[fidelity]
    except KeyError:
        raise ValueError(f"fidelity {fidelity!r} has no serialised name") from None


# Which coercion each field takes on the way in from JSON. Every field of
# ``PrecisionConfig`` must appear here or in ``_INT_FIELDS``; ``__post_init__``
# asserts that, so a field added without a serialisation rule fails at import
# rather than producing a JSON file that silently drops it.
_DTYPE_FIELDS = frozenset(
    {
        "experts_gate_up_dtype",
        "experts_down_dtype",
        "attention_qkv_dtype",
        "attention_wo_dtype",
        "lm_head_dtype",
        "router_dtype",
        "embedding_dtype",
        "norm_weight_dtype",
        "activation_dtype",
        "ccl_dtype",
        "kv_cache_dtype",
        "logits_dtype",
        "sampling_dtype",
    }
)
_FIDELITY_FIELDS = frozenset(
    {
        "experts_fidelity",
        "attention_fidelity",
        "router_window_fidelity",
        "lm_head_fidelity",
        "norm_fidelity",
    }
)
_INT_FIELDS = frozenset({"experts_gate_up_in0_block_w", "experts_down_in0_block_w"})


@dataclass(frozen=True)
class PrecisionConfig:
    """One model's dtype / fidelity policy.

    Constructed with no arguments this is ``DEFAULT_PRECISION``: the shipped
    stage-06 policy. Vary a field with :meth:`with_overrides` (or plain
    ``dataclasses.replace``) to get a different model out of the same source.
    """

    # -- weight dtypes, per group ---------------------------------------------
    # The two expert weights are separate fields even though they ship at the
    # same dtype: they have different K, different tuned block widths, and
    # different sensitivity (down feeds the residual directly), so a sweep that
    # could only move them together would be unable to price them apart.
    experts_gate_up_dtype: "ttnn.DataType" = ttnn.bfloat4_b
    experts_down_dtype: "ttnn.DataType" = ttnn.bfloat4_b
    # Likewise qkv and wo: both bfloat8_b today, but wo is the one whose output
    # goes straight into the attention all-reduce.
    attention_qkv_dtype: "ttnn.DataType" = ttnn.bfloat8_b
    attention_wo_dtype: "ttnn.DataType" = ttnn.bfloat8_b
    lm_head_dtype: "ttnn.DataType" = ttnn.bfloat8_b
    router_dtype: "ttnn.DataType" = ttnn.bfloat16
    embedding_dtype: "ttnn.DataType" = ttnn.bfloat16
    # RMSNorm weights and the per-head q_norm/k_norm vectors. 4 KB each; here
    # for completeness of the picture rather than because it is a lever.
    norm_weight_dtype: "ttnn.DataType" = ttnn.bfloat16

    # -- per-group compute fidelity -------------------------------------------
    experts_fidelity: "ttnn.MathFidelity" = ttnn.MathFidelity.LoFi
    # ``None`` == the op default, which is what the projections take today.
    attention_fidelity: "ttnn.MathFidelity | None" = None
    # HiFi4 so the one-hot expert-window matmul selects rather than approximates
    # -- see ``multichip_decoder._exact_matmul_config``. Lowering this is a
    # correctness change, not a speed/accuracy trade; it is configurable so a
    # sweep can *demonstrate* that rather than assert it.
    router_window_fidelity: "ttnn.MathFidelity" = ttnn.MathFidelity.HiFi4
    lm_head_fidelity: "ttnn.MathFidelity" = ttnn.MathFidelity.HiFi2
    norm_fidelity: "ttnn.MathFidelity" = ttnn.MathFidelity.HiFi4

    # -- expert matmul inner block widths -------------------------------------
    # Tuned against ``experts_fidelity``; see the module docstring.
    #
    # **Stage 07 moved these, and they are the only fields the sweep moved.**
    # They were 16 and 12, inherited from the single-chip stage-02 tuning. The
    # 48-layer sweep measured both brackets end to end and found each monotonic
    # upward to its full-K ceiling, at *identical* accuracy:
    #
    #   gate_up (K = hidden_size 2048 = 64 tiles):  8 -> 41.33, 16 -> 42.34,
    #                                              32 -> 42.94, 64 -> 43.23 t/s/u
    #   down    (K = moe_intermediate_size 768 = 24 tiles):
    #                                               6 -> 41.67, 12 -> 42.34,
    #                                              24 -> 42.99 t/s/u
    #
    # and the combination at both ceilings measured 43.54 t/s/u -- +2.83% over
    # the shipped default, top-1 0.990 / top-5 1.000 / top-100 1.000, i.e. no
    # accuracy cost at all, because a block width is a *scheduling* choice and
    # not a numerical one. The stage-02 comment that 16 wins at LoFi predates
    # expert parallelism, which cut per-die N four-fold and changed which
    # blocking the matmul wants.
    #
    # Both values are exact divisors of K in tiles, so
    # ``_tuned_sparse_matmul_config`` does not clamp them; ``fallback_audit``
    # reports the resolved widths and stage 07 asserts on those, not on these.
    # See ``doc/datatype_sweep/README.md``.
    experts_gate_up_in0_block_w: int = 64
    experts_down_in0_block_w: int = 24

    # -- activations ----------------------------------------------------------
    # The dtype of every hidden state, including the inter-layer residual. The
    # residual *layout* (replicated ``[1, 1, rows, 2048]``, DRAM interleaved) is
    # a contract and is not configurable here.
    activation_dtype: "ttnn.DataType" = ttnn.bfloat16
    # ``None`` == run the collective at the activation dtype, no cast. A named
    # dtype casts in and out around the reduce-scatter/all-gather pair.
    ccl_dtype: "ttnn.DataType | None" = None

    # -- kv cache -------------------------------------------------------------
    kv_cache_dtype: "ttnn.DataType" = ttnn.bfloat16

    # -- terminal path --------------------------------------------------------
    logits_dtype: "ttnn.DataType" = ttnn.bfloat16
    # What the sampler is handed. Equal to ``logits_dtype`` by default, so the
    # shipped path casts nothing.
    sampling_dtype: "ttnn.DataType" = ttnn.bfloat16

    def __post_init__(self) -> None:
        names = {f.name for f in fields(self)}
        unclassified = names - _DTYPE_FIELDS - _FIDELITY_FIELDS - _INT_FIELDS
        assert not unclassified, f"PrecisionConfig fields with no serialisation rule: {sorted(unclassified)}"
        for name in _DTYPE_FIELDS:
            object.__setattr__(self, name, dtype_from_name(getattr(self, name)))
        for name in _FIDELITY_FIELDS:
            object.__setattr__(self, name, fidelity_from_name(getattr(self, name)))
        for name in _INT_FIELDS:
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
            object.__setattr__(self, name, value)
        # ``None`` is legal only where the docstring says it is.
        for name in _DTYPE_FIELDS | _FIDELITY_FIELDS:
            if getattr(self, name) is None and name not in ("ccl_dtype", "attention_fidelity"):
                raise ValueError(f"{name} may not be None")

    # -- convenience ----------------------------------------------------------

    def with_overrides(self, **overrides) -> "PrecisionConfig":
        """A copy with ``overrides`` applied; values may be names or objects."""
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown precision fields: {sorted(unknown)}")
        return replace(self, **overrides)

    @property
    def effective_ccl_dtype(self):
        """The dtype the collectives actually run at, resolving ``None``."""
        return self.activation_dtype if self.ccl_dtype is None else self.ccl_dtype

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-ready ``{field: name}``. Every field of the dataclass appears."""
        out: dict = {}
        for name, value in asdict(self).items():
            if name in _DTYPE_FIELDS:
                out[name] = dtype_to_name(value)
            elif name in _FIDELITY_FIELDS:
                out[name] = fidelity_to_name(value)
            else:
                out[name] = int(value)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "PrecisionConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown precision fields in config: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "PrecisionConfig":
        return cls.from_dict(json.loads(text))

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def read_json(cls, path: str | Path) -> "PrecisionConfig":
        return cls.from_json(Path(path).read_text())


#: The shipped policy, as stage 07 selected it.
#:
#: Every stage-02..06 number was measured at this config **except for the two
#: expert block widths**, which stage 07 moved from 16/12 to 64/24; those stages
#: ran at 16/12. Nothing else here has changed since stage 02.
DEFAULT_PRECISION = PrecisionConfig()
