# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated output-activation folds for the selected precise GDN frontend.

Import is host-only. Call install(name) before constructing the decoder, then
snapshot_provenance(directory). Neither candidate alters the precise generic
recurrence kernels or key normalization. Correctness and timing remain gates.
"""

import ast
import hashlib
import inspect
import json
import textwrap
from pathlib import Path

CANDIDATE_NAMES = ("gdn_query_scale", "gdn_decay_exp", "gdn_adjacent_both")
_ORIGINAL = None
_SELECTED = None


def _function_source(source):
    source = textwrap.dedent(source)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    # Staticmethod decorators belong to the class; generated helper functions
    # live in a private namespace and must remain ordinary callable functions.
    return "\n".join(source.splitlines()[node.lineno - 1 :]) + "\n"


def _replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one GDN fold target: {old!r}")
    return source.replace(old, new, 1)


def generated_sources(name, *, chunk, recurrence, helper, l2norm):
    """Pure text transformation for review and host-only syntax validation."""
    if name not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown GDN adjacent fold: {name}")
    query = name in ("gdn_query_scale", "gdn_adjacent_both")
    decay = name in ("gdn_decay_exp", "gdn_adjacent_both")
    chunk, recurrence, helper, l2norm = map(_function_source, (chunk, recurrence, helper, l2norm))
    helper = _replace(helper, "def _precise_recurrence(", "def _adjacent_precise_recurrence(")
    recurrence = _replace(
        recurrence,
        "return _precise_recurrence(self, q, k, v, g, beta)",
        "return _adjacent_precise_recurrence(self, q, k, v, g, beta)",
    )
    sources = {}
    if query:
        l2norm = _replace(l2norm, "def _l2norm(x):", "def _adjacent_query_l2norm(x):")
        l2norm = _replace(
            l2norm,
            "return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(squared_norm, 1e-6)))",
            "return ttnn.multiply(\n"
            "        x, ttnn.rsqrt(ttnn.add(squared_norm, 1e-6)),\n"
            "        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 128**-0.5)],\n"
            "    )",
        )
        recurrence = _replace(
            recurrence,
            "q = self._l2norm(self._heads(q, b, seq, 16))",
            "q = _adjacent_query_l2norm(self._heads(q, b, seq, 16))",
        )
        helper = _replace(
            helper,
            "    q = ttnn.multiply(q, query_scale)\n",
            "    # Query normalization already applied the scale as its output activation.\n",
        )
        sources["query_l2norm"] = l2norm
    if decay:
        chunk = _replace(
            chunk,
            "g = ttnn.multiply(self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)))",
            "g = ttnn.multiply(\n"
            "        self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)),\n"
            "        activations=[ttnn.UnaryOpType.EXP],\n"
            "    )  # g carries exp(log_decay) in this isolated candidate.",
        )
        helper = _replace(
            helper,
            "    decay = ttnn.exp(g)\n",
            "    decay = g  # Already exponentiated by the multiply output activation.\n",
        )
        sources["chunk"] = chunk
    sources["helper"] = helper
    sources["recurrence"] = recurrence
    for key, source in sources.items():
        compile(source, f"{__file__}:{name}:{key}", "exec")
    return sources


def install(name):
    """Patch methods only; preserve production class identity and other knobs."""
    global _ORIGINAL, _SELECTED
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    if _ORIGINAL is None:
        _ORIGINAL = {
            "class": fused.FusedGatedDeltaNet,
            "chunk": fused.FusedGatedDeltaNet._chunk,
            "recurrence": fused.FusedGatedDeltaNet._recurrence,
            "helper": fused._precise_recurrence,
            "l2norm": fused.FusedGatedDeltaNet._l2norm,
        }
    sources = generated_sources(
        name, **{key: inspect.getsource(_ORIGINAL[key]) for key in ("chunk", "recurrence", "helper", "l2norm")}
    )
    namespace = dict(_ORIGINAL["helper"].__globals__)
    for key, source in sources.items():
        exec(compile(source, f"{__file__}:{name}:{key}", "exec"), namespace)
    cls = _ORIGINAL["class"]
    cls._chunk = namespace["_chunk"] if "chunk" in sources else _ORIGINAL["chunk"]
    cls._recurrence = namespace["_recurrence"]
    # The production module's helper is intentionally untouched: only this
    # candidate method resolves the generated helper in its private namespace.
    _SELECTED = {"candidate": name, "generated": sources, "runtime_path": Path(fused.__file__).resolve()}
    return cls


def snapshot_provenance(directory):
    if _SELECTED is None:
        raise RuntimeError("Install a GDN adjacent candidate before snapshotting")
    output = Path(directory) / "gdn_adjacent_sources"
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest = {"candidate": _SELECTED["candidate"], "sources": {}, "generated": {}}
    for path in (Path(__file__).resolve(), _SELECTED["runtime_path"], root / "tt/gdn.py"):
        data = path.read_bytes()
        filename = path.name + ".txt"
        (output / filename).write_bytes(data)
        manifest["sources"][str(path.relative_to(root))] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "snapshot": filename,
        }
    for key, source in _SELECTED["generated"].items():
        filename = "generated_" + key + ".py.txt"
        (output / filename).write_text(source)
        manifest["generated"][key] = {"sha256": hashlib.sha256(source.encode()).hexdigest(), "snapshot": filename}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
