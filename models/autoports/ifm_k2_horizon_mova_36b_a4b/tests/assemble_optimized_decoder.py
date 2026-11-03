# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Assemble a standalone candidate from the measured experimental method family.

This development utility writes a draft only. Final runtime imports no test code.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_decoder"


def methods(path):
    source = path.read_text()
    lines = source.splitlines()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    return {
        n.name: "\n".join(lines[min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1 : n.end_lineno])
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }


def build():
    base = methods(ROOT / "tests/candidate_optimized_base.py")
    dram = methods(ROOT / "tests/candidate_optimized_dram.py")
    layout = methods(ROOT / "tests/candidate_optimized_layout.py")
    indexed = methods(ROOT / "tests/candidate_optimized_indexed.py")
    experts = methods(ROOT / "tests/candidate_optimized_experts.py")
    cache = methods(ROOT / "tests/candidate_optimized_cache.py")
    result = dict(base)
    for name, renamed in [
        ("_norm", "_norm_base"),
        ("_mlp", "_mlp_base"),
        ("_finish", "_finish_base"),
        ("_experts", "_experts_expanded"),
    ]:
        result[renamed] = base[name].replace("def " + name + "(", "def " + renamed + "(", 1)
    for name in ["_role", "_linear", "_mlp"]:
        result[name] = dram[name].replace("super()._mlp(", "self._mlp_base(")
    for name in ["_residual_memory", "_norm", "_finish"]:
        result[name] = (
            layout[name].replace("super()._norm(", "self._norm_base(").replace("super()._finish(", "self._finish_base(")
        )
    result["_experts_indexed"] = (
        indexed["_experts"]
        .replace("def _experts(", "def _experts_indexed(", 1)
        .replace("super()._experts(", "self._experts_expanded(")
    )
    for name in ["_experts", "_indexed_mm", "_sparse_mm"]:
        result[name] = experts[name].replace("super()._experts(", "self._experts_indexed(")
    for name in ["_prefill_chunk", "decode_forward"]:
        result[name] = cache[name]
    helper = dram["from_state_dict"].splitlines()
    helper = [
        line
        for line in helper
        if "@classmethod" not in line
        and "def from_state_dict" not in line
        and "self = super().from_state_dict" not in line
        and "sd=args[0]" not in line
    ]
    result["_prepare_decode_weights"] = "    def _prepare_decode_weights(self, sd):\n" + "\n".join(helper)
    result["from_state_dict"] = (
        result["from_state_dict"]
        .replace(
            "self.sparse = layer_idx not in c.mlp_only_layers",
            """self.sparse = layer_idx not in c.mlp_only_layers
        self.options = {**(MOE_POLICY if self.sparse else DENSE_POLICY), **(options or {})}
        self.kv_cache_dtype = getattr(ttnn, self.options["cache_dtype"])""",
        )
        .replace("        return self", "        self._prepare_decode_weights(sd)\n        return self", 1)
    )
    dense = json.loads((DOC / "candidates/dense_selected_options.json").read_text())
    sparse = json.loads((DOC / "candidates/sparse_selected_options.json").read_text())
    header = '''# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized single-device K2 Horizon decoder with paged, non-aligned prefill.

All forward arguments are device tensors. Constructor-only weight materialization
prepares both phases. The caller owns cache/page-table and RoPE input refresh.
B1..32 and context524288 retain the fused-stage contract. BFP8 is the preferred
cache dtype; BF16 caller caches remain supported. Runtime has no host fallback.
"""
import math
import ttnn
from models.common.lightweightmodule import LightweightModule

'''
    content = (
        header
        + "DENSE_POLICY = "
        + repr(dense)
        + "\nMOE_POLICY = "
        + repr(sparse)
        + "\n\nclass OptimizedDecoder(LightweightModule):\n"
        + "\n\n".join(result.values())
        + "\n"
    )
    ast.parse(content)
    (DOC / "candidates/integrated_draft.py.txt").write_text(content)


if __name__ == "__main__":
    build()
