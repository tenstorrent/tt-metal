# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The selected precision policy, as a file the construction path reads.

``$datatype-sweep`` requires that the winning weight / activation / CCL /
KV-cache / compute-fidelity policy is not merely *recorded* but *consumed*: the
model, ``build_generator`` and any later serving adapter must all construct the
same numbers the sweep measured, by default, without the caller repeating them.

So this module makes ``doc/datatype_sweep/selected_precision_config.json`` a
**required build input**.  :func:`selected_build_kwargs` turns it into the exact
keyword arguments :func:`~tt.generator.build_generator` passes down, and
``build_generator`` calls it when the caller did not override a knob.  A missing
or malformed file is an error rather than a silent fall back to a module
constant -- a fallback is precisely how a selected policy stops being the one
that runs.

Three kinds of field live in the artifact, and they are consumed three different
ways.  All three are checked; none is decoration:

* **plumbed** -- weight dtypes per group, layer exceptions, per-role decode and
  prefill math fidelity, activation dtype, KV-cache dtype, CCL payload dtypes,
  LM-head dtype/fidelity/accumulation/output dtype.  These become constructor
  arguments, and :meth:`~tt.model.MuseGlimmerModel.precision_report` reads the
  realised values back off the built device tensors.
* **structural** -- the embedding table dtype, the norm weight dtype and the
  residual-stream dtype.  This model has no separate knob for these (the
  embedding table must be BF16 ROW_MAJOR for ``ttnn.embedding``; the residual
  stream *is* the activation dtype), so the loader **validates** them and raises
  on a mismatch.  A wrong value is rejected, not ignored, which is the property
  the sweep needs.
* **provenance** -- the measured numbers and the run that produced them.  Not
  consumed by the build; :func:`check_propagation` ignores it.

Round-tripping the other way is :func:`config_from_policy`, which the sweep
driver uses to write a candidate out before measuring it.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import replace
from typing import Any

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    DEFAULT_PRECISION,
    PROJECTION_ROLES,
    PrecisionPolicy,
)

#: ``models/autoports/meta_models_muse_glimmer_30b/``.
ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The one artifact every construction path reads.
SELECTED_PRECISION_CONFIG_PATH = ROOT / "doc/datatype_sweep/selected_precision_config.json"

#: The current schema version.  Bumped when a field changes meaning, so an old
#: artifact fails loudly instead of being half-understood.
SCHEMA_VERSION = 1

DTYPES: dict[str, ttnn.DataType] = {
    "bfloat4_b": ttnn.bfloat4_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat16": ttnn.bfloat16,
    "float32": ttnn.float32,
    "uint32": ttnn.uint32,
}
DTYPE_NAMES = {v: k for k, v in DTYPES.items()}

FIDELITIES: dict[str, ttnn.MathFidelity] = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
FIDELITY_NAMES = {v: k for k, v in FIDELITIES.items()}

#: ``weight group -> the projection roles it covers``.  The groups are the
#: semantic ones ``$datatype-sweep`` names; the roles are what the layer keys on.
WEIGHT_GROUPS: dict[str, tuple[str, ...]] = {
    "attn_projections": ("wqkv", "attn_gate", "o_proj"),
    "mlp_gate_up": ("mlp_gate", "mlp_up"),
    "mlp_down": ("mlp_down",),
}

#: ``PrecisionPolicy`` field for each weight group.
_GROUP_FIELD = {
    "attn_projections": "attn_weight_dtype",
    "mlp_gate_up": "mlp_gate_up_weight_dtype",
    "mlp_down": "mlp_down_weight_dtype",
}


#: Non-dtype decoder settings a precision policy may carry with it.
#:
#: A dtype is sometimes only *legal* alongside a layout or collective change --
#: a BFP4 prefill CCL payload, for instance, lands in
#: ``layernorm_pre_all_gather``, which takes BF16/BFP8/FP32 only, so the payload
#: is legal only with the fractured prefill norm off.  Carrying that companion
#: setting in the artifact is what makes such a candidate a *configuration*
#: rather than a crash.  The list is closed on purpose: this is not a general
#: escape hatch into ``decoder_kwargs``.
ALLOWED_DECODER_OVERRIDES = frozenset(
    {
        "prefill_fractured_norm",
        "prefill_ccl_mode",
        "decode_ccl_mode",
        "prefill_ccl_impl",
        "decode_ccl_impl",
    }
)


class PrecisionConfigError(ValueError):
    """A precision artifact that cannot be turned into a build."""


# ------------------------------------------------------------------- decoding


def _dtype(name: Any, *, field: str) -> ttnn.DataType | None:
    if name is None:
        return None
    if name not in DTYPES:
        raise PrecisionConfigError(f"{field}: unknown dtype {name!r}; choose from {sorted(DTYPES)}")
    return DTYPES[name]


def _fidelity(name: Any, *, field: str) -> ttnn.MathFidelity:
    if name not in FIDELITIES:
        raise PrecisionConfigError(f"{field}: unknown math fidelity {name!r}; choose from {sorted(FIDELITIES)}")
    return FIDELITIES[name]


def load_precision_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Read and shape-check the artifact.  Raises rather than defaulting."""
    path = pathlib.Path(path) if path is not None else SELECTED_PRECISION_CONFIG_PATH
    if not path.exists():
        raise PrecisionConfigError(
            f"the selected precision config is missing: {path}\n"
            "It is a required build input, not an optional override: the datatype sweep "
            "selects the weight/activation/CCL/KV-cache/fidelity policy and every later "
            "construction path (full model, benchmarks, vLLM) reads it from this file so "
            "they cannot diverge from the configuration that was measured."
        )
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - operator error
        raise PrecisionConfigError(f"{path}: not valid JSON: {exc}") from exc
    version = config.get("schema_version")
    if version != SCHEMA_VERSION:
        raise PrecisionConfigError(f"{path}: schema_version {version!r}, expected {SCHEMA_VERSION}")
    for key in ("config_id", "weights", "compute_fidelity", "activations", "ccl", "kv_cache", "logits"):
        if key not in config:
            raise PrecisionConfigError(f"{path}: missing required key {key!r}")
    overrides = config.get("decoder_overrides") or {}
    if not isinstance(overrides, dict):
        raise PrecisionConfigError(f"{path}: decoder_overrides must be an object")
    for key in overrides:
        if key not in ALLOWED_DECODER_OVERRIDES:
            raise PrecisionConfigError(
                f"{path}: decoder_overrides.{key} is not an allowed companion setting; "
                f"choose from {sorted(ALLOWED_DECODER_OVERRIDES)}"
            )
    return config


def precision_policy_from_config(config: dict[str, Any]) -> PrecisionPolicy:
    """The decoder-layer part of the artifact, as a :class:`PrecisionPolicy`."""
    weights = config["weights"]
    fidelity = config["compute_fidelity"]
    changes: dict[str, Any] = {"name": str(config["config_id"])}
    for group, field in _GROUP_FIELD.items():
        if group not in weights:
            raise PrecisionConfigError(f"weights: missing group {group!r}")
        changes[field] = _dtype(weights[group]["dtype"], field=f"weights.{group}.dtype")

    activations = config["activations"]
    activation_dtype = _dtype(activations["activation_dtype"], field="activations.activation_dtype")
    residual_dtype = _dtype(activations["residual_dtype"], field="activations.residual_dtype")
    if residual_dtype is not activation_dtype:
        raise PrecisionConfigError(
            "activations.residual_dtype must equal activations.activation_dtype: the residual stream in "
            "this model *is* the activation tensor (every layer boundary is the same width-sharded L1 "
            "fixed point), so there is no knob that could make them differ. Requesting different values "
            "would silently run the activation dtype for both."
        )
    changes["activation_dtype"] = activation_dtype
    changes["kv_cache_dtype"] = _dtype(config["kv_cache"]["dtype"], field="kv_cache.dtype")

    for phase, field, by_field in (
        ("decode", "decode_math_fidelity", "decode_math_fidelity_by_role"),
        ("prefill", "prefill_math_fidelity", "prefill_math_fidelity_by_role"),
    ):
        spec = fidelity[phase]
        changes[field] = _fidelity(spec["default"], field=f"compute_fidelity.{phase}.default")
        by_role = spec.get("by_role") or {}
        for role in by_role:
            if role not in PROJECTION_ROLES:
                raise PrecisionConfigError(
                    f"compute_fidelity.{phase}.by_role: unknown role {role!r}; " f"choose from {list(PROJECTION_ROLES)}"
                )
        changes[by_field] = tuple(
            (role, _fidelity(by_role[role], field=f"compute_fidelity.{phase}.by_role.{role}"))
            for role in PROJECTION_ROLES
            if role in by_role
        )

    exceptions = []
    for i, entry in enumerate(config.get("layer_exceptions") or []):
        indices = tuple(int(x) for x in entry["layers"])
        if not indices:
            raise PrecisionConfigError(f"layer_exceptions[{i}]: empty layer list")
        fields: dict[str, Any] = {}
        for group, dtype_name in (entry.get("weights") or {}).items():
            if group not in _GROUP_FIELD:
                raise PrecisionConfigError(
                    f"layer_exceptions[{i}].weights: unknown group {group!r}; choose from {sorted(_GROUP_FIELD)}"
                )
            fields[_GROUP_FIELD[group]] = _dtype(dtype_name, field=f"layer_exceptions[{i}].weights.{group}")
        if "kv_cache_dtype" in entry:
            fields["kv_cache_dtype"] = _dtype(entry["kv_cache_dtype"], field=f"layer_exceptions[{i}].kv_cache_dtype")
        for phase, by_field in (
            ("decode", "decode_math_fidelity_by_role"),
            ("prefill", "prefill_math_fidelity_by_role"),
        ):
            by_role = ((entry.get("compute_fidelity") or {}).get(phase) or {}).get("by_role") or {}
            if by_role:
                fields[by_field] = tuple(
                    (role, _fidelity(by_role[role], field=f"layer_exceptions[{i}].compute_fidelity.{phase}.{role}"))
                    for role in PROJECTION_ROLES
                    if role in by_role
                )
        if not fields:
            raise PrecisionConfigError(f"layer_exceptions[{i}]: no field to change")
        exceptions.append((indices, tuple(sorted(fields.items(), key=lambda kv: kv[0]))))
    changes["layer_exceptions"] = tuple(exceptions)

    return replace(DEFAULT_PRECISION, **changes)


def _validate_structural(config: dict[str, Any]) -> None:
    """Fields the model has no knob for.  Rejected on mismatch, never ignored."""
    from models.autoports.meta_models_muse_glimmer_30b.tt.model import EMBED_DTYPE

    weights = config["weights"]
    embedding = _dtype(weights["embedding"]["dtype"], field="weights.embedding.dtype")
    if embedding is not EMBED_DTYPE:
        raise PrecisionConfigError(
            f"weights.embedding.dtype={weights['embedding']['dtype']!r} but the embedding table is built at "
            f"{DTYPE_NAMES[EMBED_DTYPE]}: ttnn.embedding needs a ROW_MAJOR table and this port builds it BF16. "
            "There is no knob for this, so the artifact must state the value the build actually has."
        )
    norms = _dtype(weights["norms"]["dtype"], field="weights.norms.dtype")
    if norms is not ttnn.bfloat16:
        raise PrecisionConfigError(
            f"weights.norms.dtype={weights['norms']['dtype']!r} but every RMSNorm weight is built BF16 "
            "(the `1 + w` fold is done in float32 on the host and packed to BF16). No knob exists."
        )
    logits = config["logits"]
    expected = weights["lm_head"].get("output_dtype", logits.get("lm_head_output_dtype"))
    if logits.get("sampling_input_dtype") != expected:
        raise PrecisionConfigError(
            "logits.sampling_input_dtype must equal the LM head's output dtype: the sampler consumes the "
            "LM head's output tensor directly, with no conversion between them."
        )


# ------------------------------------------------------------- build kwargs


LM_HEAD_GEOMETRY_KEYS = ("matmul", "cores", "in0_block_w")


def lm_head_geometry_for_topology(head: dict[str, Any], num_devices: int | None = None) -> dict[str, Any]:
    """Resolve an LM-head geometry without changing its precision policy.

    The P150x4-selected DRAM-sharded program projects one quarter of the
    vocabulary per device. Wider P150/P150x2 shards need a different legal
    program, recorded alongside the selected head rather than hidden in a
    source default. A caller that does not provide a topology retains the
    historical flat geometry for compatibility with sweep artifacts.
    """
    geometry = {key: head[key] for key in LM_HEAD_GEOMETRY_KEYS}
    overrides = head.get("topology_overrides") or {}
    if not isinstance(overrides, dict):
        raise PrecisionConfigError("weights.lm_head.topology_overrides must be an object")
    for topology, override in overrides.items():
        if not isinstance(override, dict):
            raise PrecisionConfigError(f"weights.lm_head.topology_overrides.{topology} must be an object")
        unknown = sorted(set(override) - set(LM_HEAD_GEOMETRY_KEYS))
        if unknown:
            raise PrecisionConfigError(
                f"weights.lm_head.topology_overrides.{topology}: unknown fields {unknown}; "
                f"choose from {list(LM_HEAD_GEOMETRY_KEYS)}"
            )
    if num_devices is not None:
        geometry.update(overrides.get(str(int(num_devices)), {}))
    if geometry["matmul"] not in ("dram_sharded", "mcast1d"):
        raise PrecisionConfigError(
            f"weights.lm_head.matmul: expected 'dram_sharded' or 'mcast1d', got {geometry['matmul']!r}"
        )
    for key in ("cores", "in0_block_w"):
        geometry[key] = int(geometry[key])
        if geometry[key] <= 0:
            raise PrecisionConfigError(f"weights.lm_head.{key} must be positive, got {geometry[key]}")
    return geometry


def build_kwargs_from_config(config: dict[str, Any], *, num_devices: int | None = None) -> dict[str, Any]:
    """The artifact as ``build_generator`` keyword arguments.

    Returns ``lm_head_*`` at the top level and everything the decoder layers need
    under ``decoder_kwargs``, which is exactly the shape
    ``MuseGlimmerModel.from_pretrained`` consumes.
    """
    _validate_structural(config)
    policy = precision_policy_from_config(config)
    head = config["weights"]["lm_head"]
    head_geometry = lm_head_geometry_for_topology(head, num_devices)
    head_fidelity = config["compute_fidelity"]["lm_head"]
    ccl = config["ccl"]
    return {
        "lm_head_dtype": _dtype(head["dtype"], field="weights.lm_head.dtype"),
        "lm_head_fidelity": _fidelity(head_fidelity["fidelity"], field="compute_fidelity.lm_head.fidelity"),
        "lm_head_fp32_acc": bool(head_fidelity["fp32_dest_acc_en"]),
        "lm_head_output_dtype": _dtype(head["output_dtype"], field="weights.lm_head.output_dtype"),
        # The LM-head matmul geometry travels with its dtype: the static
        # circular-buffer budget is dtype-scaled, so the shipped BFP4 geometry
        # (52 cores, in0_block_w=2) overflows L1 at BFP8 and a candidate that
        # changes the head's dtype has to bring a legal geometry with it.
        "lm_head_matmul": str(head_geometry["matmul"]),
        "lm_head_cores": int(head_geometry["cores"]),
        "lm_head_in0_block_w": int(head_geometry["in0_block_w"]),
        "decoder_kwargs": {
            "precision": policy,
            "prefill_ccl_dtype": _dtype(ccl["prefill_dtype"], field="ccl.prefill_dtype"),
            "decode_ccl_dtype": _dtype(ccl["decode_dtype"], field="ccl.decode_dtype"),
            **(config.get("decoder_overrides") or {}),
        },
    }


def selected_build_kwargs(
    path: str | pathlib.Path | None = None, *, num_devices: int | None = None
) -> tuple[str, dict[str, Any]]:
    """``(config_id, build kwargs)`` for the selected artifact."""
    config = load_precision_config(path)
    return str(config["config_id"]), build_kwargs_from_config(config, num_devices=num_devices)


# ----------------------------------------------------------------- writing


def config_from_policy(
    *,
    config_id: str,
    description: str = "",
    policy: PrecisionPolicy,
    prefill_ccl_dtype: ttnn.DataType | None,
    decode_ccl_dtype: ttnn.DataType | None,
    lm_head_dtype: ttnn.DataType,
    lm_head_fidelity: ttnn.MathFidelity,
    lm_head_fp32_acc: bool,
    lm_head_output_dtype: ttnn.DataType,
    lm_head_matmul: str,
    lm_head_cores: int,
    lm_head_in0_block_w: int,
    decoder_overrides: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A serialisable artifact for one candidate.  The inverse of the loader."""

    def by_role(pairs: tuple[tuple[str, ttnn.MathFidelity], ...]) -> dict[str, str]:
        return {role: FIDELITY_NAMES[fid] for role, fid in pairs}

    config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config_id": config_id,
        "description": description,
        "weights": {
            "attn_projections": {
                "dtype": DTYPE_NAMES[policy.attn_weight_dtype],
                "roles": list(WEIGHT_GROUPS["attn_projections"]),
            },
            "mlp_gate_up": {
                "dtype": DTYPE_NAMES[policy.mlp_gate_up_weight_dtype],
                "roles": list(WEIGHT_GROUPS["mlp_gate_up"]),
            },
            "mlp_down": {
                "dtype": DTYPE_NAMES[policy.mlp_down_weight_dtype],
                "roles": list(WEIGHT_GROUPS["mlp_down"]),
            },
            "lm_head": {
                "dtype": DTYPE_NAMES[lm_head_dtype],
                "output_dtype": DTYPE_NAMES[lm_head_output_dtype],
                "matmul": lm_head_matmul,
                "cores": int(lm_head_cores),
                "in0_block_w": int(lm_head_in0_block_w),
            },
            "embedding": {"dtype": "bfloat16", "note": "structural: ttnn.embedding needs a ROW_MAJOR BF16 table"},
            "norms": {"dtype": "bfloat16", "note": "structural: every RMSNorm weight is packed BF16"},
        },
        "layer_exceptions": [
            {
                "layers": list(indices),
                **_exception_payload(fields),
            }
            for indices, fields in policy.layer_exceptions
        ],
        "compute_fidelity": {
            "decode": {
                "default": FIDELITY_NAMES[policy.decode_math_fidelity],
                "by_role": by_role(policy.decode_math_fidelity_by_role),
            },
            "prefill": {
                "default": FIDELITY_NAMES[policy.prefill_math_fidelity],
                "by_role": by_role(policy.prefill_math_fidelity_by_role),
            },
            "lm_head": {
                "fidelity": FIDELITY_NAMES[lm_head_fidelity],
                "fp32_dest_acc_en": bool(lm_head_fp32_acc),
            },
        },
        "activations": {
            "activation_dtype": DTYPE_NAMES[policy.activation_dtype],
            "residual_dtype": DTYPE_NAMES[policy.activation_dtype],
        },
        "ccl": {
            "prefill_dtype": None if prefill_ccl_dtype is None else DTYPE_NAMES[prefill_ccl_dtype],
            "decode_dtype": None if decode_ccl_dtype is None else DTYPE_NAMES[decode_ccl_dtype],
            "decode_dtype_note": "null means the collective carries the activation dtype",
        },
        "kv_cache": {"dtype": DTYPE_NAMES[policy.kv_cache_dtype]},
        "decoder_overrides": dict(decoder_overrides or {}),
        "logits": {
            "lm_head_output_dtype": DTYPE_NAMES[lm_head_output_dtype],
            "sampling_input_dtype": DTYPE_NAMES[lm_head_output_dtype],
            "sampling_implementation": "models.common.sampling.SamplingGenerator",
        },
    }
    if extra:
        config.update(extra)
    return config


def _exception_payload(fields: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    field_to_group = {v: k for k, v in _GROUP_FIELD.items()}
    payload: dict[str, Any] = {}
    for field, value in fields:
        if field in field_to_group:
            payload.setdefault("weights", {})[field_to_group[field]] = DTYPE_NAMES[value]
        elif field == "kv_cache_dtype":
            payload["kv_cache_dtype"] = DTYPE_NAMES[value]
        elif field in ("decode_math_fidelity_by_role", "prefill_math_fidelity_by_role"):
            phase = field.split("_", 1)[0]
            payload.setdefault("compute_fidelity", {})[phase] = {
                "by_role": {role: FIDELITY_NAMES[fid] for role, fid in value}
            }
        else:  # pragma: no cover - guarded by the loader
            raise PrecisionConfigError(f"cannot serialise layer exception field {field!r}")
    return payload


# -------------------------------------------------------------- propagation


def check_propagation(config: dict[str, Any], realised: dict[str, Any]) -> list[str]:
    """Compare a requested artifact against a built model's ``precision_report``.

    Returns the list of mismatches, empty when every requested field is the one
    the build actually has.  This is the check that distinguishes "the JSON says
    BFP4" from "the matmul read BFP4".
    """
    problems: list[str] = []
    policy = precision_policy_from_config(config)

    def name_of(value: Any) -> str:
        """``DataType.BFLOAT8_B`` / ``MathFidelity.LoFi`` -> the artifact spelling.

        ``str()`` on a ttnn enum is upper-cased for dtypes and mixed-case for
        fidelities, while the artifact uses the lower-case ttnn attribute name
        (``bfloat8_b``) and the ``MathFidelity`` member name (``LoFi``).  Compare
        case-insensitively rather than teaching the artifact two spellings.
        """
        return str(value).split(".")[-1].lower()

    groups = realised["layer_groups"]
    seen_layers = {idx for group in groups for idx in group["layers"]}
    for group in groups:
        got = group["precision"]
        for layer_idx in group["layers"]:
            want = policy.for_layer(layer_idx)
            for role in PROJECTION_ROLES:
                role_got = got["roles"][role]
                checks = (
                    ("weight_dtype", DTYPE_NAMES[want.weight_dtype(role)]),
                    ("decode_fidelity", FIDELITY_NAMES[want.decode_fidelity(role)]),
                    ("prefill_fidelity", FIDELITY_NAMES[want.prefill_fidelity(role)]),
                )
                for key, expected in checks:
                    actual = name_of(role_got[key])
                    if actual != expected.lower():
                        problems.append(f"layer {layer_idx} {role}.{key}: requested {expected}, built {actual}")
            for key, expected in (
                ("activation_dtype", DTYPE_NAMES[want.activation_dtype]),
                ("kv_cache_dtype", DTYPE_NAMES[want.kv_cache_dtype]),
            ):
                actual = name_of(got[key])
                if actual != expected.lower():
                    problems.append(f"layer {layer_idx} {key}: requested {expected}, built {actual}")
            ccl = got.get("ccl", {})
            for key, requested in (
                ("prefill_payload_dtype", config["ccl"]["prefill_dtype"]),
                ("decode_payload_dtype", config["ccl"]["decode_dtype"]),
            ):
                expected = requested if requested is not None else DTYPE_NAMES[want.activation_dtype]
                actual = name_of(ccl.get(key, ""))
                if actual != expected.lower():
                    problems.append(f"layer {layer_idx} ccl.{key}: requested {expected}, built {actual}")
            break  # one layer per distinct group is enough; the group is identical by construction

    head = realised["lm_head"]
    head_want = config["weights"]["lm_head"]
    head_geometry_want = lm_head_geometry_for_topology(head_want, realised.get("num_devices"))
    fidelity_want = config["compute_fidelity"]["lm_head"]
    for key, expected in (
        ("weight_dtype", head_want["dtype"]),
        ("output_dtype", head_want["output_dtype"]),
        ("fidelity", fidelity_want["fidelity"]),
    ):
        actual = name_of(head[key])
        if actual != expected.lower():
            problems.append(f"lm_head.{key}: requested {expected}, built {actual}")
    for key in ("matmul", "cores", "in0_block_w"):
        if str(head[key]) != str(head_geometry_want[key]):
            problems.append(f"lm_head.{key}: requested {head_geometry_want[key]}, built {head[key]}")
    if bool(head["fp32_dest_acc_en"]) != bool(fidelity_want["fp32_dest_acc_en"]):
        problems.append(
            f"lm_head.fp32_dest_acc_en: requested {fidelity_want['fp32_dest_acc_en']}, built {head['fp32_dest_acc_en']}"
        )
    logits_want = config["logits"]
    logits_got = realised.get("logits", {})
    for key in ("logits_dtype", "sampling_input_dtype"):
        expected = logits_want.get("lm_head_output_dtype" if key == "logits_dtype" else key)
        actual = name_of(logits_got.get(key, ""))
        if actual != expected.lower():
            problems.append(f"logits.{key}: requested {expected}, built {actual}")
    # The requested path is the public re-export
    # (``models.common.sampling.SamplingGenerator``) and the built one is the
    # defining module (``models.common.sampling.generator.SamplingGenerator``).
    # Resolve the request to a class object and compare identities, so a genuine
    # substitution is caught and a re-export is not.
    if not _same_class(logits_want["sampling_implementation"], logits_got.get("sampling_implementation", "")):
        problems.append(
            f"logits.sampling_implementation: requested {logits_want['sampling_implementation']}, "
            f"built {logits_got.get('sampling_implementation')}"
        )
    for key, expected in (config.get("decoder_overrides") or {}).items():
        actual = (realised.get("decoder_overrides") or {}).get(key, "<not reported>")
        if actual != expected:
            problems.append(f"decoder_overrides.{key}: requested {expected}, built {actual}")
    embedding_want = config["weights"]["embedding"]["dtype"]
    embedding_got = name_of(realised["embedding"]["weight_dtype"])
    if embedding_got != embedding_want.lower():
        problems.append(f"embedding.weight_dtype: requested {embedding_want}, built {embedding_got}")
    norms_want = config["weights"]["norms"]["dtype"]
    for key, value in realised["terminal_norms"].items():
        actual = name_of(value)
        if actual != norms_want.lower():
            problems.append(f"terminal_norms.{key}: requested {norms_want}, built {actual}")
    if realised["num_layers"] != len(seen_layers):
        problems.append(f"precision_report covers {len(seen_layers)} layers of {realised['num_layers']}")
    return problems


def _same_class(requested: str, built: str) -> bool:
    """Do two dotted paths name the same class?  Falls back to string equality."""
    if requested == built:
        return True

    def resolve(path: str):
        module, _, name = path.rpartition(".")
        if not module:
            return None
        try:
            import importlib

            return getattr(importlib.import_module(module), name, None)
        except Exception:
            return None

    left, right = resolve(requested), resolve(built)
    return left is not None and left is right


__all__ = [
    "DTYPES",
    "DTYPE_NAMES",
    "FIDELITIES",
    "FIDELITY_NAMES",
    "PrecisionConfigError",
    "SCHEMA_VERSION",
    "SELECTED_PRECISION_CONFIG_PATH",
    "WEIGHT_GROUPS",
    "build_kwargs_from_config",
    "check_propagation",
    "config_from_policy",
    "load_precision_config",
    "precision_policy_from_config",
    "selected_build_kwargs",
]
