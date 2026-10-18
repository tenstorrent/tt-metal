# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: gate trailing inactive B8 rows before both packed MoE consumers."""

import copy
import hashlib
import inspect
import textwrap
import types
from pathlib import Path

PRODUCTION_SHA256 = "b376bf4ecd58e0ccbbf977731b2c48717fb561ce9af05601825fc387f40a025c"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one production fragment: {old}")
    return source.replace(old, new)


def build_source(source, masked):
    source = textwrap.dedent(source)
    for fragment in (
        "[source, ids, scores, self.rank, counts, mapping, capacity]",
        "[down, ids, scores, self.rank, mapping, compact]",
        "scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)",
    ):
        if source.count(fragment) != 1:
            raise ValueError(f"Changed shared routing-score graph: {fragment}")
    source = replace_once(
        source,
        "    n = x.shape[2]\n",
        "    n = x.shape[2]\n    if n != 8:\n        raise ValueError('Probe supports exactly B8 decode rows')\n",
    )
    anchor = "    ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)"
    mask = "    selected = ttnn.where(self._probe_active, selected, 0.0)\n" if masked else ""
    source = replace_once(
        source,
        anchor,
        "    ttnn.copy(selected, self._probe_buffers['unmasked_scores'])\n"
        + mask
        + "    ttnn.copy(selected, self._probe_buffers['scores'])\n"
        + anchor
        + "\n    ttnn.copy(ids, self._probe_buffers['ids'])",
    )
    source = replace_once(
        source,
        "    down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(",
        "    ttnn.copy(counts, self._probe_buffers['counts'])\n"
        "    down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(",
    )
    return source


def make(owner, active, buffers, *, masked):
    from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as production

    if hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest() != PRODUCTION_SHA256:
        raise ValueError("Production source drift: review candidate against the new source")
    source = build_source(inspect.getsource(production.ParallelMoE.group_packed), masked)
    namespace = dict(vars(production))
    exec(compile(source, __file__, "exec"), namespace)
    result = copy.copy(owner)
    result._probe_active = active
    result._probe_buffers = buffers
    result.group_packed = types.MethodType(namespace["group_packed"], result)
    return result, source


def route_counts(ids, scores, rank, active=8):
    """Independent host reference to the existing dispatch predicate."""
    if len(ids) != 8 or len(scores) != 8 or not 0 <= active <= 8 or rank not in range(4):
        raise ValueError("Expected B8 and a valid TP4 rank")
    counts = [0] * 128
    for row in range(active):
        if len(ids[row]) != 32 or len(scores[row]) != 32:
            raise ValueError("Expected 32 selected columns")
        for expert, score in zip(ids[row][:10], scores[row][:10]):
            if not 0 <= expert < 512:
                raise ValueError("Router selected an invalid expert")
            if expert // 128 == rank and score != 0:
                counts[expert % 128] += 1
    return counts
