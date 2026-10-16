# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Captured-region walk of the position-generic decode body (no device, no ttnn import).

The body is captured once per GDN conv-ring phase and each graph is replayed
for every position of its residue class, so nothing inside the captured region
may shape a program from a host integer: ``ttnn.slice`` hashes its bounds into
program identity, ``ttnn.pad`` its amounts, ``ttnn.reshape``/``repeat`` their
shapes, ``ttnn.concat`` its piece count.  This test resolves the static call
graph from the three captured roots (model body, greedy candidates, device
greedy resolve) through every module of the ``ttnn`` package and requires the
shape argument of every such call to be built only from literals, module
constants, construction-time ``self`` attributes, tensor shapes, or locals
assigned from those.  A preallocated ``output_tensor`` is a buffer binding, not
a shape argument: it does not enter program identity, and the ring's
phase-selected slot is the one such binding, carried by the residue-class
captures.  The test also pins which functions the region reaches and which
per-position/host paths it does not.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "ttnn"

# (module, class, method) of the ops the runner captures, in capture order.
ROOTS = (
    ("model", "Qwen38TTNNTextModel", "forward_decode_generic"),
    ("embedding", "Qwen38TTNNLMHead", "greedy_candidates"),
    ("embedding", "Qwen38TTNNLMHead", "resolve_greedy_on_device"),
    # the composed GDN step: forward_decode reaches it through the resolved ``self._step`` (ttnn/fused/gdn_step)
    ("gdn", "Qwen38TTNNGDN", "_split_projection"),
    ("gdn", "Qwen38TTNNGDN", "_causal_conv_decode"),
    ("gdn", "Qwen38TTNNGDN", "_make_recurrent_inputs"),
    ("gdn", "Qwen38TTNNGDN", "_recurrent_decode"),
    ("gdn", "Qwen38TTNNGDN", "_gate"),
)
# Receivers whose method name exists on several classes: the class the body
# actually dispatches to (None: a branch the generic body never takes).  A
# callable object maps to its ``__call__``.
RECEIVERS = {
    "self.attention.forward_decode": ("Qwen38TTNNGDN", "forward_decode"),  # the QSA branch is forward_decode_generic
    "self.attention.forward_decode_generic": ("Qwen38TTNNQSA", "forward_decode_generic"),
    "self.attention._validate_state": ("Qwen38TTNNGDN", "_validate_state"),  # QSA validates its generic state
    "self.ple.forward_decode": None,  # eager host-token PLE path; the generic body always passes prepared_ple
    "self._routed_partial_blocks": None,  # the prefill slab's routed stream: a rows form the decode body never takes
    "self._weighted_reduce_slab_blocks": None,  # the one-call slab's weighted reduce: slab rows only
    "self.attention_gr": ("Qwen38TTNNGatedResidual", None),
    "self.mlp_gr": ("Qwen38TTNNGatedResidual", None),
    "self.moe": ("Qwen38TTNNMoE", None),
    "self.ple": ("Qwen38TTNNPLE", None),
    "self.final_mixer": ("Qwen38TTNNFinalMixer", "__call__"),
    "self.model_io.lm_head": ("Qwen38TTNNLMHead", "__call__"),
    "self.model_io.embedding": ("Qwen38TTNNTokenEmbedding", None),
    "self.rope_table": ("Qwen38TTNNRoPETable", None),
    "state.position": ("Qwen38TTNNDevicePosition", None),
    "layer": ("Qwen38TTNNDecoderLayer", None),
    "self.mesh_contract": ("Qwen38MeshContract", None),
}
SHAPE_OPS = {
    "slice",
    "pad",
    "concat",
    "reshape",
    "repeat",
    "repeat_interleave",
    "split",
    "view",
    "permute",
    "transpose",
    "squeeze",
    "unsqueeze",
}
STATIC_CALLS = {"ttnn.Shape", "tuple", "list", "int", "min", "max"}  # static when every argument is
# Read a tensor's shape (fixed per capture); _validate_logits returns int(logits.global_shape[2]).
SHAPE_READERS = {"_shape", "len", "self._validate_logits"}
SHAPE_ATTRIBUTES = {"shape", "padded_shape", "logical_shape", "global_shape"}
BUFFER_BINDINGS = {"output_tensor"}  # a preallocated output binds a buffer, not a shape
# Teardown and Python container methods: many classes define them, none shapes a program.
IGNORED_METHODS = {"deallocate", "release", "release_tensors", "append", "extend", "add", "get", "items", "pop"}
HOST_ONLY = ("to_torch(", "from_torch(", ".item()", "synchronize(", "copy_host_to_device_tensor(", "as_tensor(")
HOST_POSITION = (
    "next_position",
    "raw_tail_count",
    "state.compressed_blocks",
    "position %",
    "position //",
    "position +",
)


def _modules() -> dict[str, ast.Module]:
    return {path.stem: ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for path in PACKAGE.glob("*.py")}


class _Index:
    """Every class method and module function of the package, by name."""

    def __init__(self, modules: dict[str, ast.Module]) -> None:
        self.modules = modules
        self.functions: dict[tuple[str, str], ast.FunctionDef] = {}  # (module, name) -> module-level function
        self.methods: dict[tuple[str, str], tuple[str, ast.FunctionDef]] = {}  # (class, name) -> (module, node)
        self.classes_with: dict[str, set[str]] = {}
        for module, tree in modules.items():
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    self.functions[(module, node.name)] = node
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            self.methods[(node.name, item.name)] = (module, item)
                            self.classes_with.setdefault(item.name, set()).add(node.name)

    def resolve(self, call: ast.Call, *, module: str, cls: str | None) -> list[tuple[str, str | None, str]]:
        """The package functions the call may dispatch to (empty for ttnn/torch/builtins).

        A method name defined on several classes resolves through RECEIVERS when
        the receiver is listed; otherwise every candidate is walked, so an
        over-approximation can only make this test stricter, never weaker.
        """

        func = call.func
        if isinstance(func, ast.Name):
            if (module, func.id) in self.functions:
                return [(module, None, func.id)]
            return [(m, None, name) for (m, name) in self.functions if name == func.id]
        if not isinstance(func, ast.Attribute):
            return []
        full = ast.unparse(func)
        receiver = ast.unparse(func.value)
        if full in RECEIVERS:
            if RECEIVERS[full] is None:
                return []
            target_cls, method = RECEIVERS[full]
            return [self._method(target_cls, method or func.attr)]
        if receiver in RECEIVERS:
            return [self._method(RECEIVERS[receiver][0], func.attr)]
        if receiver.endswith("_module") and (receiver[: -len("_module")], func.attr) in self.functions:
            return [(receiver[: -len("_module")], None, func.attr)]  # `from ttnn import qsa as qsa_module`
        if receiver.split(".")[0] in {"ttnn", "torch", "os", "time", "json", "math"} or func.attr in IGNORED_METHODS:
            return []
        if receiver == "self" and cls is not None and (cls, func.attr) in self.methods:
            return [self._method(cls, func.attr)]
        return [self._method(owner, func.attr) for owner in sorted(self.classes_with.get(func.attr, ()))]

    def _method(self, cls: str, name: str) -> tuple[str, str, str]:
        assert (cls, name) in self.methods, f"{cls}.{name} is not defined in the package"
        return (self.methods[(cls, name)][0], cls, name)

    def node(self, key: tuple[str, str | None, str]) -> ast.FunctionDef:
        module, cls, name = key
        return self.functions[(module, name)] if cls is None else self.methods[(cls, name)][1]


def _reach(index: _Index) -> dict[tuple[str, str | None, str], ast.FunctionDef]:
    pending = [(module, cls, name) for module, cls, name in ROOTS]
    reached: dict[tuple[str, str | None, str], ast.FunctionDef] = {}
    while pending:
        key = pending.pop()
        if key in reached:
            continue
        node = index.node(key)
        reached[key] = node
        for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
            pending.extend(t for t in index.resolve(call, module=key[0], cls=key[1]) if t not in reached)
    return reached


def _static_locals(function: ast.FunctionDef) -> set[str]:
    """Locals whose every assignment is a static expression (fixpoint)."""

    assignments: dict[str, list[ast.expr]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments.setdefault(node.targets[0].id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for name in [n for n in ast.walk(node.target) if isinstance(n, ast.Name)]:
                assignments.setdefault(name.id, []).append(ast.Name(id="__loop__", ctx=ast.Load()))
    static: set[str] = set()
    while True:
        grown = {name for name, values in assignments.items() if all(_is_static(v, static) for v in values)}
        if grown <= static:
            return static
        static |= grown


def _is_static(node: ast.expr, static_locals: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id.isupper() or node.id in static_locals
    if isinstance(node, ast.Attribute):
        chain = ast.unparse(node)
        root = chain.split(".")[0]
        return root in {"ttnn", "torch", "self"} or root.isupper() or node.attr in SHAPE_ATTRIBUTES
    if isinstance(node, ast.Subscript):
        return _is_static(node.value, static_locals) and _is_static(node.slice, static_locals)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_static(item, static_locals) for item in node.elts)
    if isinstance(node, ast.BinOp):
        return _is_static(node.left, static_locals) and _is_static(node.right, static_locals)
    if isinstance(node, ast.UnaryOp):
        return _is_static(node.operand, static_locals)
    if isinstance(node, ast.Starred):
        return _is_static(node.value, static_locals)
    if isinstance(node, ast.Call):
        if ast.unparse(node.func) in SHAPE_READERS:
            return True
        return ast.unparse(node.func) in STATIC_CALLS and all(
            _is_static(arg, static_locals) for arg in [*node.args, *(kw.value for kw in node.keywords)]
        )
    return False


def _shape_op_calls(function: ast.FunctionDef) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in SHAPE_OPS
        and ast.unparse(node.func).startswith("ttnn.")
    ]


def _label(key: tuple[str, str | None, str]) -> str:
    module, cls, name = key
    return f"{module}.{cls}.{name}" if cls else f"{module}.{name}"


def test_captured_region_reaches_the_generic_body_and_not_the_per_position_paths() -> None:
    reached = {_label(key) for key in _reach(_Index(_modules()))}
    for required in (
        "model.Qwen38TTNNTextModel.forward_decode_generic",
        "model.Qwen38TTNNTextModel.forward_decode_generic_head",
        "model.Qwen38TTNNTextModel.forward_decode_generic_tail",
        "model.Qwen38TTNNTextModel._validate_generic_head",
        "model.Qwen38TTNNTextModel._embed_residual_from_device_token",
        "embedding.Qwen38TTNNTokenEmbedding.embed_device_token",
        "contracts.Qwen38TTNNDevicePosition.index_row",
        "contracts.Qwen38TTNNDevicePosition.block_start_index_row",
        "contracts.Qwen38TTNNDevicePosition.advance",
        "model.Qwen38TTNNRoPETable.rows",
        "model.Qwen38TTNNRoPETable._lookup",
        "qsa.derive_qsa_position_inputs",
        "layer.Qwen38TTNNDecoderLayer.forward_decode_generic",
        "layer.Qwen38TTNNDecoderLayer._apply_ple",
        "layer.Qwen38TTNNDecoderLayer._route_through_gr_and_moe",
        "ple.Qwen38TTNNPLE.forward_prepared",
        "gr.Qwen38TTNNGatedResidual.read",
        "gr.Qwen38TTNNGatedResidual.write",
        "gdn.Qwen38TTNNGDN.forward_decode",
        "qsa.Qwen38TTNNQSA.forward_decode_generic",
        "qsa.Qwen38TTNNQSA._write_compressed_index_generic",
        "qsa.Qwen38TTNNQSA._score_blocks_generic",
        "qsa.Qwen38TTNNQSA._materialize_row_generic",
        "qsa.Qwen38TTNNQSA._write_packed_kv_generic",
        "qsa.Qwen38TTNNQSA._sparse_value_attention",
        "moe.Qwen38TTNNMoE.forward",
        "final_mixer.Qwen38TTNNFinalMixer.__call__",
        "embedding.Qwen38TTNNLMHead.__call__",
        "embedding.Qwen38TTNNLMHead.greedy_candidates",
        "embedding.Qwen38TTNNLMHead.resolve_greedy_on_device",
    ):
        assert required in reached, required
    # The one PLE layer runs its prepared row: the eager host-token branch of _apply_ple is dead here.
    apply_ple = ast.unparse(_Index(_modules()).node(("layer", "Qwen38TTNNDecoderLayer", "forward_decode_generic")))
    assert (
        "self._apply_ple(residual_sharded, state, token_id=None, prepared_ple=prepared_ple, release_input=release_input)"
        in apply_ple
    )
    for forbidden in (
        "ple.Qwen38TTNNPLE.forward_decode",
        "ple.Qwen38TTNNPLE._upload_embedding",
        "qsa.Qwen38TTNNQSA.forward_decode",
        "qsa.Qwen38TTNNQSA._append_packed_kv",
        "qsa.Qwen38TTNNQSA._append_raw_index_key",
        "qsa.Qwen38TTNNQSA._score_complete_blocks",
        "qsa.Qwen38TTNNQSA._materialize_selection",
        "qsa.Qwen38TTNNQSA._select",
        "qsa.Qwen38TTNNQSA._take_next_kv_staging",
        "model.Qwen38TTNNRoPE.for_position",
        "model.Qwen38TTNNTextModel.forward_decode",
        "model.Qwen38TTNNTextModel._embed_residual",
        "model.Qwen38TTNNTextModel.prepare_decode_inputs",
        "layer.Qwen38TTNNDecoderLayer.forward_decode",
        "contracts.Qwen38TTNNDevicePosition.read",
        "contracts.Qwen38TTNNDevicePosition.reset",
        "embedding.Qwen38TTNNLMHead.resolve_greedy",
        "embedding.Qwen38TTNNTokenEmbedding.upload_tokens",
        "embedding.Qwen38TTNNTokenEmbedding.upload_token_row",
    ):
        assert forbidden not in reached, forbidden


def test_captured_region_has_no_host_int_dependent_shape_op_and_no_host_io() -> None:
    index = _Index(_modules())
    offenders: list[str] = []
    checked = 0
    for key, function in _reach(index).items():
        static_locals = _static_locals(function)
        source = ast.unparse(function)
        for call in _shape_op_calls(function):
            checked += 1
            # Argument 0 is the tensor (or the tensor list of concat); everything
            # else but a buffer binding shapes the program.
            for argument in [*call.args[1:], *(kw.value for kw in call.keywords if kw.arg not in BUFFER_BINDINGS)]:
                if not _is_static(argument, static_locals):
                    offenders.append(f"{_label(key)}:{call.lineno} {ast.unparse(call)}")
        for text in (*HOST_ONLY, *HOST_POSITION):
            if text in source:
                offenders.append(f"{_label(key)} contains {text!r}")
    assert checked >= 20, checked  # gdn projection slices, RoPE/embedding reshapes, greedy pad/reshape, QSA row ops
    assert not offenders, "\n".join(offenders)


def test_the_only_buffer_bound_shape_op_is_the_gdn_ring_slot_write() -> None:
    """The conv ring's slot binding is the body's one host-integer dependence; the residue-class traces carry it.

    ``ttnn.slice(..., output_tensor=newest)`` writes the newest projection into
    the slot ``conv_phase`` selects: the same program at every phase, a different
    buffer per phase.  That is why the single-trace runner captures one graph per
    phase; no other captured shape op binds a buffer.
    """

    index = _Index(_modules())
    bound = [
        f"{_label(key)} {ast.unparse(call.func)} {keyword.arg}={ast.unparse(keyword.value)}"
        for key, function in _reach(index).items()
        for call in _shape_op_calls(function)
        for keyword in call.keywords
        if keyword.arg in BUFFER_BINDINGS
    ]
    assert bound == ["gdn.Qwen38TTNNGDN._split_projection ttnn.slice output_tensor=newest"]
    forward = ast.unparse(index.node(("gdn", "Qwen38TTNNGDN", "forward_decode")))
    assert "window = state.conv_window()" in forward
    assert "projected = self._project(full_hidden)" in forward
    assert "step = self._gdn_step()" in forward and "gated = step(self, projected, window, state)" in forward
    assert "state.advance_conv_window()" in forward
