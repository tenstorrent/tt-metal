"""Op-level classifier for the bring-up loop.

The existing pipeline classifies whole HF components as REUSE / ADAPT / NEW
against a sibling tt-demo. That works at the COMPONENT level (e.g. "this
encoder block has a sibling we can adapt"). For everything classified as
NEW, the current loop dumps the entire component on an LLM.

But most of what's INSIDE a NEW component is mechanical — `nn.Linear`,
`nn.LayerNorm`, `nn.Conv2d`, activations, embeddings — and has a 1:1
mapping to a ttnn primitive. There is no reason to pay an LLM call to
re-derive that `nn.Linear` becomes `ttnn.linear`.

This module walks `named_modules()` of a resolved HF submodule and
classifies each leaf op as:

    op-REUSE  : a known atomic torch op with a deterministic ttnn
                template (`Linear`, `LayerNorm`, `Conv2d`, `Embedding`,
                `GELU`/`SiLU`/`ReLU`/`Sigmoid`/`Tanh`).
    op-ADAPT  : a known torch op with non-default args we can still
                template, but the user/agent should double-check (e.g.
                `Conv2d` with non-square stride and a custom output
                grouping).
    op-NEW    : everything else — custom HF blocks, two-way attention,
                positional encoding with sin/cos on float coords, etc.
                These are the ONLY ops the LLM still needs to write.

Output is consumed by `op_emitter.py` (which emits the deterministic ttnn
code) and by the CLI / handoff prompt (which only asks the LLM to fill
the op-NEW gaps instead of the whole component).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


_KIND_BY_CLASS_NAME: Dict[str, str] = {
    "Linear": "linear",
    "LayerNorm": "layer_norm",
    "RMSNorm": "rms_norm",
    "Conv2d": "conv2d",
    "Conv1d": "conv1d",
    "ConvTranspose2d": "conv_transpose2d",
    "Embedding": "embedding",
    "GELU": "activation",
    "SiLU": "activation",
    "ReLU": "activation",
    "Sigmoid": "activation",
    "Tanh": "activation",
    "GELUActivation": "activation",
    "FastGELUActivation": "activation",
    "QuickGELUActivation": "activation",
    "NewGELUActivation": "activation",
    "AccurateGELUActivation": "activation",
    "PytorchGELUTanh": "activation",
    "SiLUActivation": "activation",
    "MishActivation": "activation",
    "ReLUSquaredActivation": "activation",
    "LinearActivation": "identity",
    "Softmax": "softmax",
    "Dropout": "dropout",
    "Identity": "identity",
    "Flatten": "flatten",
}


_HF_ACTIVATION_TO_TTNN: Dict[str, str] = {
    "GELUActivation": "gelu",
    "FastGELUActivation": "gelu",
    "QuickGELUActivation": "gelu",
    "NewGELUActivation": "gelu",
    "AccurateGELUActivation": "gelu",
    "PytorchGELUTanh": "gelu",
    "SiLUActivation": "silu",
    "MishActivation": "mish",
    "ReLUSquaredActivation": "relu",
}


_OP_REUSE_KINDS = {
    "linear",
    "layer_norm",
    "embedding",
    "activation",
    "identity",
    "dropout",
}


_OP_ADAPT_KINDS = {
    "conv2d",
    "conv1d",
    "conv_transpose2d",
    "rms_norm",
    "softmax",
}


from .module_tree import _CONTAINER_CLASS_NAMES as _ALL_CONTAINER_CLASSES

_CONTAINER_CLASSES = {name for name in _ALL_CONTAINER_CLASSES if name in ("Sequential", "ModuleList", "ModuleDict")}


@dataclass
class OpDescriptor:
    """One leaf op inside a NEW component, plus enough metadata for the
    emitter to write its ttnn code without further introspection."""

    name: str
    """Dotted path within the component, e.g. `mask_embed.conv1`."""

    torch_class: str
    """Concrete torch class name, e.g. `Linear`, `Conv2d`."""

    kind: str
    """Coarse category used to pick the emitter: `linear`, `layer_norm`, ..."""

    status: str
    """`op-REUSE`, `op-ADAPT`, or `op-NEW`."""

    params: Dict[str, Any] = field(default_factory=dict)
    """Constructor args needed by the emitter (in_features, out_features,
    has_bias, normalized_shape, eps, kernel_size, stride, padding, ...)."""

    state_dict_keys: List[str] = field(default_factory=list)
    """Parameter/buffer keys for this op within the component's state_dict
    (with the leading op-name prefix included)."""

    ttnn_target: Optional[str] = None
    """The ttnn primitive this maps to (e.g. `ttnn.linear`). None for op-NEW."""

    notes: str = ""
    """Free-form classifier notes (e.g. "grouped conv — emitter does not
    yet support groups>1")."""

    def is_reusable(self) -> bool:
        return self.status in ("op-REUSE", "op-ADAPT")


def _torch_module_class_name(module: Any) -> str:
    try:
        return type(module).__name__
    except Exception:
        return ""


def _is_leaf(module: Any) -> bool:
    """A leaf module has no torch children. We treat Sequential / ModuleList
    as containers — even though they're nn.Module subclasses — so the
    classifier recurses into them via `named_modules()`."""
    cls_name = _torch_module_class_name(module)
    if cls_name in _CONTAINER_CLASSES:
        return False
    try:
        children = list(module.children())
    except Exception:
        return True
    return len(children) == 0


def _extract_linear_params(module: Any) -> Dict[str, Any]:
    return {
        "in_features": int(getattr(module, "in_features", 0) or 0),
        "out_features": int(getattr(module, "out_features", 0) or 0),
        "has_bias": getattr(module, "bias", None) is not None,
    }


def _extract_layer_norm_params(module: Any) -> Dict[str, Any]:
    normalized_shape = getattr(module, "normalized_shape", None)
    if normalized_shape is not None:
        try:
            normalized_shape = list(normalized_shape)
        except Exception:
            normalized_shape = [int(normalized_shape)]
    eps = float(getattr(module, "eps", 1e-5) or 1e-5)
    has_weight = getattr(module, "elementwise_affine", True) and getattr(module, "weight", None) is not None
    has_bias = getattr(module, "elementwise_affine", True) and getattr(module, "bias", None) is not None
    return {
        "normalized_shape": normalized_shape,
        "eps": eps,
        "has_weight": bool(has_weight),
        "has_bias": bool(has_bias),
    }


def _extract_conv_params(module: Any, *, dims: int) -> Dict[str, Any]:
    def _to_tuple(v: Any) -> Tuple[int, ...]:
        if isinstance(v, (tuple, list)):
            return tuple(int(x) for x in v)
        try:
            return (int(v),) * dims
        except Exception:
            return tuple([0] * dims)

    return {
        "in_channels": int(getattr(module, "in_channels", 0) or 0),
        "out_channels": int(getattr(module, "out_channels", 0) or 0),
        "kernel_size": _to_tuple(getattr(module, "kernel_size", 0)),
        "stride": _to_tuple(getattr(module, "stride", 1)),
        "padding": _to_tuple(getattr(module, "padding", 0)),
        "dilation": _to_tuple(getattr(module, "dilation", 1)),
        "groups": int(getattr(module, "groups", 1) or 1),
        "has_bias": getattr(module, "bias", None) is not None,
    }


def _extract_embedding_params(module: Any) -> Dict[str, Any]:
    return {
        "num_embeddings": int(getattr(module, "num_embeddings", 0) or 0),
        "embedding_dim": int(getattr(module, "embedding_dim", 0) or 0),
        "padding_idx": getattr(module, "padding_idx", None),
    }


def _extract_activation_params(module: Any) -> Dict[str, Any]:
    cls = _torch_module_class_name(module)
    if cls in _HF_ACTIVATION_TO_TTNN:
        variant = _HF_ACTIVATION_TO_TTNN[cls]
    else:
        variant = cls.lower()
    params: Dict[str, Any] = {"variant": variant, "source_class": cls}
    if cls == "GELU":
        params["approximate"] = getattr(module, "approximate", "none")
    return params


def _classify_one(module: Any, dotted_name: str) -> OpDescriptor:
    cls = _torch_module_class_name(module)
    kind = _KIND_BY_CLASS_NAME.get(cls, "unknown")
    params: Dict[str, Any] = {}
    ttnn_target: Optional[str] = None
    notes = ""

    if kind == "linear":
        params = _extract_linear_params(module)
        ttnn_target = "ttnn.linear"
    elif kind == "layer_norm":
        params = _extract_layer_norm_params(module)
        ttnn_target = "ttnn.layer_norm"
    elif kind == "rms_norm":
        params = _extract_layer_norm_params(module)
        ttnn_target = "ttnn.rms_norm"
    elif kind == "conv2d":
        params = _extract_conv_params(module, dims=2)
        ttnn_target = "ttnn.conv2d"
        if params.get("groups", 1) != 1:
            notes = "grouped conv (groups>1) — emitter falls back to a per-group lowering"
    elif kind == "conv1d":
        params = _extract_conv_params(module, dims=1)
        ttnn_target = "ttnn.conv1d"
    elif kind == "conv_transpose2d":
        params = _extract_conv_params(module, dims=2)
        ttnn_target = "ttnn.conv_transpose2d"
    elif kind == "embedding":
        params = _extract_embedding_params(module)
        ttnn_target = "ttnn.embedding"
    elif kind == "activation":
        params = _extract_activation_params(module)
        variant = params.get("variant", "")
        if variant in ("gelu", "silu", "relu", "sigmoid", "tanh"):
            ttnn_target = f"ttnn.{variant}"
        else:
            ttnn_target = None
    elif kind == "softmax":
        params = {"dim": int(getattr(module, "dim", -1) or -1)}
        ttnn_target = "ttnn.softmax"
    elif kind == "dropout":
        params = {"p": float(getattr(module, "p", 0.0) or 0.0)}
        ttnn_target = None
    elif kind == "identity":
        ttnn_target = None
    else:
        kind = "unknown"

    if kind in _OP_REUSE_KINDS:
        status = "op-REUSE"
    elif kind in _OP_ADAPT_KINDS:
        status = "op-ADAPT"
    else:
        status = "op-NEW"

    sd_keys: List[str] = []
    try:
        for pname, _p in module.named_parameters(recurse=False):
            sd_keys.append(f"{dotted_name}.{pname}" if dotted_name else pname)
        for bname, _b in module.named_buffers(recurse=False):
            sd_keys.append(f"{dotted_name}.{bname}" if dotted_name else bname)
    except Exception:
        pass

    return OpDescriptor(
        name=dotted_name or "<root>",
        torch_class=cls,
        kind=kind,
        status=status,
        params=params,
        state_dict_keys=sd_keys,
        ttnn_target=ttnn_target,
        notes=notes,
    )


def classify_ops_in_component(submodule: Any) -> List[OpDescriptor]:
    """Walk `named_modules()` of a resolved HF submodule and return one
    OpDescriptor per leaf. Containers are skipped (their leaves are
    enumerated by `named_modules` anyway). The root submodule itself is
    skipped unless it has no children.

    Args:
        submodule: a `torch.nn.Module` instance — usually the result of
            `_resolve_torch_submodule_for_component(...)`.

    Returns:
        Ordered list of OpDescriptors. Order matches `named_modules()`.
    """
    if submodule is None:
        return []

    out: List[OpDescriptor] = []
    try:
        items = list(submodule.named_modules())
    except Exception:
        return []

    for dotted, mod in items:
        if dotted == "":
            if _is_leaf(submodule):
                out.append(_classify_one(submodule, ""))
            continue
        if not _is_leaf(mod):
            continue
        out.append(_classify_one(mod, dotted))
    return out


def summarize_ops(ops: List[OpDescriptor]) -> Dict[str, Any]:
    """Aggregate counts useful for the bring-up summary printout."""
    counts = {"op-REUSE": 0, "op-ADAPT": 0, "op-NEW": 0}
    by_kind: Dict[str, int] = {}
    new_ops_by_class: Dict[str, int] = {}
    for op in ops:
        counts[op.status] = counts.get(op.status, 0) + 1
        by_kind[op.kind] = by_kind.get(op.kind, 0) + 1
        if op.status == "op-NEW":
            new_ops_by_class[op.torch_class] = new_ops_by_class.get(op.torch_class, 0) + 1
    total = sum(counts.values())
    reusable = counts.get("op-REUSE", 0) + counts.get("op-ADAPT", 0)
    return {
        "total": total,
        "counts": counts,
        "reusable": reusable,
        "reusable_fraction": (reusable / total) if total else 0.0,
        "by_kind": by_kind,
        "new_ops_by_class": new_ops_by_class,
    }


def format_op_plan(component_name: str, ops: List[OpDescriptor]) -> str:
    """Pretty-print the op-level plan for a component."""
    if not ops:
        return f"  {component_name}: (no leaf ops detected)\n"
    summary = summarize_ops(ops)
    lines: List[str] = []
    counts = summary["counts"]
    reusable_frac = summary["reusable_fraction"]
    lines.append(
        f"  {component_name}: total ops = {summary['total']}, "
        f"op-REUSE = {counts.get('op-REUSE', 0)}, "
        f"op-ADAPT = {counts.get('op-ADAPT', 0)}, "
        f"op-NEW   = {counts.get('op-NEW', 0)}  "
        f"({reusable_frac * 100:.0f}% deterministic)"
    )
    width = max(len(op.name) for op in ops)
    for op in ops:
        ttnn = op.ttnn_target or "—"
        tag = op.status
        line = f"    [{tag:<9}]  {op.name:<{width}}  {op.torch_class:<18}  -> {ttnn}"
        if op.notes:
            line += f"    ({op.notes})"
        lines.append(line)
    return "\n".join(lines) + "\n"
