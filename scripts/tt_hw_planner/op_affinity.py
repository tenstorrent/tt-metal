"""TTNN op-affinity catalog.

Maps each ttnn op name to a device-affinity score so the cold-evidence
classifier can directly answer "does putting this component on device
add value?" rather than approximating it via ops-per-byte heuristics.

Scoring is at the TTNN OP LEVEL, not the model level — no model
knowledge embedded. Any HF model whose stub uses these ops gets the
same scoring. A new model never has to be added to a list.

Three buckets:

  +1  DEVICE_FAVORABLE   compute-bound ops that fundamentally beat
                         CPU due to TT's parallelism + on-chip math.
                         Examples: matmul, conv2d, attention.

  -1  DEVICE_UNFAVORABLE  bandwidth/memory-layout ops where host↔device
                         transfer cost dominates any speedup.
                         Examples: permute, reshape, tilize, gather.

   0  NEUTRAL            small enough that the verdict depends on
                         shape/dtype/context; treated as a wash.

Unknown ops default to 0 — neutral by safe-default.

The component score = Σ count_i × affinity_i over the component's opplan
manifest. Positive score → device adds value. Negative → CPU wins.
"""

from __future__ import annotations

from typing import Iterable, Union


# Compute-bound ops where TT device parallelism + on-chip math
# fundamentally beats CPU. The dominant inference workloads (transformers,
# convnets, attention) live here.
DEVICE_FAVORABLE = frozenset(
    {
        "ttnn.matmul",
        "ttnn.linear",
        "ttnn.conv1d",
        "ttnn.conv2d",
        "ttnn.conv_transpose1d",
        "ttnn.conv_transpose2d",
        "ttnn.scaled_dot_product_attention",
        "ttnn.transformer.scaled_dot_product_attention",
        "ttnn.transformer.attention_softmax",
        "ttnn.embedding",
        "ttnn.max_pool2d",
        "ttnn.avg_pool2d",
        "ttnn.layer_norm",
        "ttnn.rms_norm",
        "ttnn.group_norm",
        "ttnn.batch_norm",
        # Large elementwise / reductions: device wins when tensors are big.
        # We treat them as favorable; if a model uses them on tiny tensors
        # the per-component compute_density signal will catch it.
        "ttnn.softmax",
        "ttnn.argmax",
        "ttnn.sum",
        "ttnn.mean",
    }
)


# Memory-layout / bandwidth-bound ops where TT device transfer cost
# dominates. No real compute → no real device speedup.
DEVICE_UNFAVORABLE = frozenset(
    {
        "ttnn.permute",
        "ttnn.transpose",
        "ttnn.reshape",
        "ttnn.view",
        "ttnn.expand",
        "ttnn.repeat",
        "ttnn.repeat_interleave",
        "ttnn.unsqueeze",
        "ttnn.squeeze",
        "ttnn.flatten",
        "ttnn.cat",
        "ttnn.concat",
        "ttnn.split",
        "ttnn.chunk",
        "ttnn.stack",
        "ttnn.gather",
        "ttnn.scatter",
        "ttnn.index_select",
        "ttnn.where",
        # Layout / device-movement primitives — pure transfer cost.
        "ttnn.tilize",
        "ttnn.untilize",
        "ttnn.from_torch",
        "ttnn.to_torch",
        "ttnn.from_device",
        "ttnn.to_device",
    }
)


# Small ops where the verdict depends on context (shape, dtype, fusion).
# Treated as neutral; let other signals dominate.
NEUTRAL = frozenset(
    {
        "ttnn.gelu",
        "ttnn.relu",
        "ttnn.silu",
        "ttnn.swish",
        "ttnn.tanh",
        "ttnn.sigmoid",
        "ttnn.sin",
        "ttnn.cos",
        "ttnn.exp",
        "ttnn.log",
        "ttnn.add",
        "ttnn.subtract",
        "ttnn.multiply",
        "ttnn.divide",
        "ttnn.pow",
        "ttnn.sqrt",
        "ttnn.abs",
        "ttnn.clamp",
        "ttnn.minimum",
        "ttnn.maximum",
    }
)


def op_affinity_score(op_name: str) -> int:
    """Return ``+1`` if op is device-favorable, ``-1`` if unfavorable,
    ``0`` if neutral or unknown.

    Accepts either a bare name (``matmul``) or a fully-qualified name
    (``ttnn.matmul``). Non-ttnn names (e.g. raw torch ops) score 0 —
    we don't make claims about their device affinity, since they'd
    need to be ported through a ttnn op first anyway."""
    name = (op_name or "").strip()
    if not name:
        return 0
    if not name.startswith("ttnn."):
        # Allow bare names too; normalize.
        if "." not in name:
            name = f"ttnn.{name}"
        else:
            return 0
    if name in DEVICE_FAVORABLE:
        return 1
    if name in DEVICE_UNFAVORABLE:
        return -1
    if name in NEUTRAL:
        return 0
    return 0  # unknown op — safe-default neutral


def component_affinity_score(opplan_data: Union[dict, list, None]) -> int:
    """Score a component from its opplan manifest content.

    Accepts either:
      - dict ``{op_name: count}``  (opplan ``counts`` schema)
      - list ``[op_name, ...]``    (opplan ``palette`` schema)
      - None                       (no opplan → score 0)

    Returns: integer score. Positive = component has more device-favorable
    ops than unfavorable → graduation worthwhile. Negative = unfavorable
    dominates → CPU wins. Zero = neutral / no opplan data.
    """
    if opplan_data is None:
        return 0
    if isinstance(opplan_data, dict):
        return sum(int(cnt) * op_affinity_score(op) for op, cnt in opplan_data.items() if isinstance(cnt, (int, float)))
    if isinstance(opplan_data, list):
        return sum(op_affinity_score(op) for op in opplan_data)
    return 0


def affinity_label(score: int) -> str:
    """Short human-readable label for an affinity score, for display
    in CLI output / reports."""
    if score > 0:
        return "device-favorable"
    if score < 0:
        return "device-unfavorable"
    return "neutral"
