# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only conservative Python call-graph audit of the TP4 decoder runtime.

Parses source; never imports model modules, torch, TTNN, or device code. Follows
both forward roots, inherited dispatch/super calls and imported model-local
helpers across all policy branches. Static success is NOT runtime-sentinel
success. Pair the output with the existing device runner's conversion guards.
"""

import argparse
import ast
import builtins
import hashlib
import json
import sys
import tempfile
from pathlib import Path

PACKAGE = "models.autoports.qwen_qwen3_6_27b.tt"
ROOT_CLASS = PACKAGE + ".multichip_decoder.MultichipDecoder"
CORE_MODULES = ("optimized_decoder", "multichip_decoder", "flat_prefill", "recurrent_variants")
HOST_TTNN = {
    "to_torch",
    "from_torch",
    "as_tensor",
    "to_device",
    "from_device",
    "to_host",
    "copy_host_to_device_tensor",
    "copy_device_to_host_tensor",
    "full",
    "dump_tensor",
    "load_tensor",
}
SETUP_METHODS = {
    "from_state_dict",
    "_weight",
    "_upload",
    "_upload_shards",
    "_prepare_matmuls",
    "create_recurrent_state",
    "create_paged_state",
    "replace_paged_table_from_host",
    "upload_residual",
    "upload_decode_residual",
}
HOST_FRAMEWORKS = {"torch", "numpy", "scipy", "tensorflow", "jax"}
PACKED_MODULE = PACKAGE + ".packed_correction_candidate"
PACKED_CLASS = PACKED_MODULE + ".PackedCorrection"
PACKED_SETUP = {PACKED_CLASS + ".__init__", PACKED_MODULE + ".install"}
NORM_MODULE = PACKAGE + ".fused_decode_norm"
NORM_CLASS = NORM_MODULE + ".FusedDecodeNormOwner"
NORM_SETUP = {NORM_CLASS + ".__init__", NORM_CLASS + "._views", NORM_MODULE + ".install"}
RUNTIME_SETUP = PACKED_SETUP | NORM_SETUP
OWNER_ARGUMENTS = {
    **{
        PACKED_CLASS + "." + name: "owner"
        for name in ("project", "project_prefill", "_project_compact", "_phase_config", "_finish_projection")
    },
    NORM_CLASS + ".normalize": "model",
}
OWNER_METHODS = set(OWNER_ARGUMENTS)
REQUIRED_HELPERS = OWNER_METHODS | {
    PACKED_CLASS + "._combine",
    PACKED_MODULE + ".prefill_plan",
    ROOT_CLASS + "._prefill_swiglu_with_options",
    PACKAGE + ".flat_prefill._finish_flat",
    PACKAGE + ".flat_prefill._can_flatten",
    PACKAGE + ".prefill_qkv_candidate.prefill_qkv",
    PACKAGE + ".prefill_qkv_candidate.geometry",
    PACKAGE + ".prefill_qkv_candidate.partition",
    PACKAGE + ".attention_gate_candidate.attention_gate",
    PACKAGE + ".attention_gate_candidate.band_sources",
    PACKAGE + ".recurrent_next_candidate.local_conv",
    PACKAGE + ".recurrent_next_candidate.local_conv_sources",
    PACKAGE + ".multichip_decoder.MultichipDecoder._prefill_cache_fill",
    PACKAGE + ".multichip_decoder.MultichipDecoder._attention_page_table",
    PACKAGE + ".multichip_decoder.MultichipDecoder._attention_state_view",
    PACKAGE + ".page_table_ownership.page_table_for_attention",
    PACKAGE + ".page_table_ownership.page_table_certified",
    PACKAGE + ".batched_cache_fill_candidate.batched_fill_pair",
    PACKAGE + ".packed_swiglu_candidate.packed_swiglu",
    PACKAGE + ".packed_swiglu_candidate.output_memory_config",
    PACKAGE + ".local_affine_gate_candidate.local_affine_gate",
    PACKAGE + ".local_affine_gate_candidate.local_affine_sources",
    PACKAGE + ".parallel_qk_norm_candidate.compact_qk_norm",
    PACKAGE + ".flat_gdn_prefill_candidate.flat_packed_gdn",
    PACKAGE + ".tiled_prefill_conv_candidate.tiled_prefill_conv",
    PACKAGE + ".tiled_prefill_conv_candidate.native_compute",
}


def direct_walk(node):
    """Walk a scope, leaving nested bodies to their separately indexed nodes."""
    for child in ast.iter_child_nodes(node):
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Defaults/decorators execute in the surrounding scope.
            for value in child.decorator_list:
                yield value
                yield from direct_walk(value)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for value in (*child.args.defaults, *(v for v in child.args.kw_defaults if v is not None)):
                    yield value
                    yield from direct_walk(value)
            continue
        yield from direct_walk(child)


class Audit:
    def __init__(self, sources, root_class=ROOT_CLASS):
        self.sources, self.root_class = sources, root_class
        self.trees = {module: ast.parse(source, filename=module) for module, source in sources.items()}
        self.functions, self.classes, self.bindings, self.parents = {}, {}, {}, {}
        for module, tree in self.trees.items():
            self.bindings[module] = self.imports(module, list(direct_walk(tree)))
            self.index(module, tree, module)
        self.edges, self.import_records, self.external, self.unresolved, self.violations = [], [], [], [], []
        self.instance_types, self.audited_bindings, self.literal_calls = {}, [], []
        self.literal_values = self.module_literals()
        self.bind_packed_owner()
        self.bind_norm_owner()
        self.verify_owner_parameters()

    def bind_packed_owner(self):
        """Explicit type binding, checked against the actual setup assignments.

        Setup is inspected for identity only, not added to runtime roots. Its
        host weight reads remain excluded unless a forward actually calls it.
        """
        record = {
            "owner_class": self.root_class,
            "attribute": "_packed_correction_candidate",
            "instance_class": PACKED_CLASS,
            "basis": "explicit binding verified against setup AST",
            "setup_function": PACKED_MODULE + ".install",
            "evidence": [],
            "pass": False,
        }
        setup = self.functions.get(record["setup_function"])
        factory = self.functions.get(self.root_class + ".from_state_dict")
        if setup and factory and PACKED_CLASS in self.classes:
            module, fn, _ = setup
            nodes = list(direct_walk(fn))
            bindings = {**self.bindings[module], **self.imports(module, nodes)}
            instances = set()
            for node in nodes:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    symbol = self.symbol(node.value.func, module, bindings)
                    if symbol == PACKED_CLASS or module + "." + str(symbol) == PACKED_CLASS:
                        instances.update(t.id for t in node.targets if isinstance(t, ast.Name))
                        record["evidence"].append(
                            self.detail(record["setup_function"], node, kind="constructor_binding")
                        )
            assigned = False
            for node in nodes:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in instances:
                    for target in node.targets:
                        if self.symbol(target, module, bindings) == "layer._packed_correction_candidate":
                            assigned = True
                            record["evidence"].append(
                                self.detail(record["setup_function"], node, kind="owner_attribute_assignment")
                            )
            module, fn, _ = factory
            nodes = list(direct_walk(fn))
            bindings = {**self.bindings[module], **self.imports(module, nodes)}
            called = False
            for node in nodes:
                if (
                    isinstance(node, ast.Call)
                    and self.symbol(node.func, module, bindings) == record["setup_function"]
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "self"
                ):
                    called = True
                    record["evidence"].append(
                        self.detail(self.root_class + ".from_state_dict", node, kind="setup_receives_owner")
                    )
            record["pass"] = assigned and called
        if record["pass"]:
            self.instance_types[(self.root_class, record["attribute"])] = PACKED_CLASS
        else:
            self.violations.append({"category": "runtime_instance_binding_unverified", "binding": record})
        self.audited_bindings.append(record)

    def verify_owner_parameters(self):
        for scope, argument in sorted(OWNER_ARGUMENTS.items()):
            fn = self.functions.get(scope)
            valid = bool(fn and [arg.arg for arg in fn[1].args.args][:2] == ["self", argument])
            self.audited_bindings.append(
                {
                    "function": scope,
                    "argument": argument,
                    "instance_class": self.root_class,
                    "basis": "explicit owner parameter; verified signature",
                    "pass": valid,
                }
            )
            if not valid:
                self.violations.append({"category": "runtime_owner_parameter_unverified", "function": scope})

    def bind_norm_owner(self):
        """Prove the actual conditional installer, constructor owner and call site.

        A name/attribute match alone is insufficient: the constructor must receive
        the same model and real layer index that the decoder factory supplies.
        """
        setup_name = NORM_MODULE + ".install"
        record = {
            "owner_class": self.root_class,
            "attribute": "_fused_decode_norm",
            "instance_class": NORM_CLASS,
            "basis": "conditional setup assignment and explicit owner identity verified against AST",
            "setup_function": setup_name,
            "evidence": [],
            "pass": False,
        }
        setup = self.functions.get(setup_name)
        factory = self.functions.get(self.root_class + ".from_state_dict")
        constructor = self.functions.get(NORM_CLASS + ".__init__")
        normalize = self.functions.get(NORM_CLASS + ".normalize")
        if setup and factory and constructor and normalize:
            module, fn, _ = setup
            nodes = list(direct_walk(fn))
            bindings = {**self.bindings[module], **self.imports(module, nodes)}
            expected = ast.parse(
                'model._fused_decode_norm = FusedDecodeNormOwner(model, layer_idx=layer_idx) if family != "off" else None'
            ).body[0]
            assignments = [
                n
                for n in nodes
                if isinstance(n, ast.Assign)
                and any(self.symbol(t, module, bindings) == "model._fused_decode_norm" for t in n.targets)
            ]
            writes = [
                n
                for n in nodes
                if isinstance(n, ast.Attribute)
                and isinstance(n.ctx, ast.Store)
                and self.symbol(n, module, bindings) == "model._fused_decode_norm"
            ]
            assigned = len(assignments) == len(writes) == 1 and ast.dump(
                assignments[0], include_attributes=False
            ) == ast.dump(expected, include_attributes=False)
            constructor_symbol = self.symbol(assignments[0].value.body.func, module, bindings) if assigned else None
            constructor_symbol = (
                module + "." + constructor_symbol
                if constructor_symbol and "." not in constructor_symbol
                else constructor_symbol
            )
            assigned = assigned and constructor_symbol == NORM_CLASS
            family_assign = ast.parse('family = model.policy.get("distributed_decode_norm", "off")').body[0]
            family_nodes = [
                n
                for n in nodes
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "family" for t in n.targets)
            ]
            assigned = (
                assigned
                and len(family_nodes) == 1
                and ast.dump(family_nodes[0], include_attributes=False)
                == ast.dump(family_assign, include_attributes=False)
            )
            signature = [a.arg for a in fn.args.args] == ["model"] and [a.arg for a in fn.args.kwonlyargs] == [
                "layer_idx"
            ]
            # Explicit source identity guard prevents a clone from silently using
            # another decoder's normalization owner.
            identity = ast.parse("self.owner_id, self.mesh, self.layer_idx = id(model), model.mesh, layer_idx").body[0]
            identity_ok = any(
                ast.dump(n, include_attributes=False) == ast.dump(identity, include_attributes=False)
                for n in direct_walk(constructor[1])
                if isinstance(n, ast.Assign)
            )
            identity_ok = (
                identity_ok
                and sum(
                    isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and ast.unparse(n) == "self.owner_id"
                    for n in direct_walk(constructor[1])
                )
                == 1
            )
            identity_guard = ast.parse("id(model) == self.owner_id", mode="eval").body
            guard_ok = any(
                isinstance(n, ast.Assert)
                and ast.dump(n.test, include_attributes=False) == ast.dump(identity_guard, include_attributes=False)
                for n in normalize[1].body
            )
            ctor_ok = [a.arg for a in constructor[1].args.args][:2] == ["self", "model"]
            module, fn, _ = factory
            nodes = list(direct_walk(fn))
            bindings = {**self.bindings[module], **self.imports(module, nodes)}
            calls = [
                n for n in nodes if isinstance(n, ast.Call) and self.symbol(n.func, module, bindings) == setup_name
            ]
            called = (
                len(calls) == 1
                and len(calls[0].args) == 1
                and isinstance(calls[0].args[0], ast.Name)
                and calls[0].args[0].id == "self"
            )
            called = (
                called
                and len(calls[0].keywords) == 1
                and calls[0].keywords[0].arg == "layer_idx"
                and ast.unparse(calls[0].keywords[0].value) == "self.layer_idx"
            )
            if assigned:
                record["evidence"].append(
                    self.detail(setup_name, assignments[0], kind="conditional_constructor_owner_binding")
                )
            if called:
                record["evidence"].append(
                    self.detail(
                        self.root_class + ".from_state_dict", calls[0], kind="setup_receives_owner_and_layer_index"
                    )
                )
            record["checks"] = {
                "assignment": assigned,
                "installer_signature": signature,
                "constructor_signature": ctor_ok,
                "constructor_owner_identity": identity_ok,
                "normalize_owner_identity_guard": guard_ok,
                "factory_owner_and_layer_index": called,
            }
            record["pass"] = all(record["checks"].values())
        if record["pass"]:
            self.instance_types[(self.root_class, record["attribute"])] = NORM_CLASS
        else:
            self.violations.append({"category": "runtime_instance_binding_unverified", "binding": record})
        self.audited_bindings.append(record)

    def module_literals(self):
        """Only singly assigned AST literals qualify; naming conventions do not.

        Imported/reassigned objects, constructors, computed expressions and
        declared global mutation are not proven literal source strings.
        """
        result = {}
        for module, tree in self.trees.items():
            nodes = list(direct_walk(tree))
            writes = {}
            for node in nodes:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    writes[node.id] = writes.get(node.id, 0) + 1
            global_mutations = {name for n in ast.walk(tree) if isinstance(n, ast.Global) for name in n.names}
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                if type(value) is not str:
                    continue
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and writes.get(target.id) == 1
                        and target.id not in global_mutations
                    ):
                        result[module + "." + target.id] = value
        return result

    def literal_receiver(self, expr, scope, bindings, *, seen=frozenset()):
        """Prove local aliases and string-preserving reassignments, fail closed.

        Flow-insensitive aliases must not hide a later object assignment. A
        self-replacement is accepted only with a separate proven string seed.
        """
        module, fn, _ = self.functions[scope]
        if isinstance(expr, ast.Constant):
            return type(expr.value) is str
        if isinstance(expr, ast.Attribute):
            return self.symbol(expr, module, bindings) in self.literal_values
        if not isinstance(expr, ast.Name) or expr.id in seen:
            return False
        if expr.id in {a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}:
            return False
        nodes = list(direct_walk(fn))
        assignments = []
        writes = 0
        for node in nodes:
            if isinstance(node, ast.Name) and node.id == expr.id and isinstance(node.ctx, ast.Store):
                writes += 1
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == expr.id for t in targets):
                    assignments.append(node.value)
        if not assignments:
            return self.symbol(expr, module, bindings) in self.literal_values and writes == 0
        if writes != len(assignments):
            return False
        seed = False
        for value in assignments:
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "replace"
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == expr.id
            ):
                continue  # Builtin str.replace preserves type once a seed is proved.
            if not self.literal_receiver(value, scope, bindings, seen=seen | {expr.id}):
                return False
            seed = True
        return seed

    def literal_method(self, scope, node, canonical, bindings):
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in ("count", "replace"):
            return False
        receiver = canonical.rsplit(".", 1)[0]
        if receiver not in self.literal_values or not self.literal_receiver(node.func.value, scope, bindings):
            return False
        self.literal_calls.append(
            self.detail(
                scope,
                node,
                call=canonical,
                basis="AST-proven singly assigned str literal and checked local alias assignments",
                literal_source=receiver,
                literal_sha256=hashlib.sha256(self.literal_values[receiver].encode()).hexdigest(),
            )
        )
        return True

    def imports(self, module, nodes):
        bindings = {}
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bindings[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.split(".")[: -node.level]
                    prefix = ".".join(base + ([node.module] if node.module else []))
                else:
                    prefix = node.module or ""
                for alias in node.names:
                    bindings[alias.asname or alias.name] = prefix + "." + alias.name
        return bindings

    def index(self, module, owner, prefix, owner_class=None):
        for node in owner.body:
            if isinstance(node, ast.ClassDef):
                name = prefix + "." + node.name
                self.classes[name] = (module, node)
                self.index(module, node, name, name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + "." + node.name
                self.functions[name] = (module, node, owner_class)
                self.parents[name] = prefix
                self.index(module, node, name, owner_class)

    def symbol(self, expr, module, bindings):
        if isinstance(expr, ast.Name):
            return bindings.get(expr.id, expr.id)
        if isinstance(expr, ast.Attribute):
            base = self.symbol(expr.value, module, bindings)
            return base + "." + expr.attr if base else None
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "getattr":
            if len(expr.args) >= 2 and isinstance(expr.args[1], ast.Constant) and isinstance(expr.args[1].value, str):
                base = self.symbol(expr.args[0], module, bindings)
                return base + "." + expr.args[1].value if base else None
        return None

    def method(self, cls, name, *, super_only=False, seen=None):
        seen = set() if seen is None else seen
        if cls in seen or cls not in self.classes:
            return None
        seen.add(cls)
        target = cls + "." + name
        if not super_only and target in self.functions:
            return target
        module, node = self.classes[cls]
        for base in node.bases:
            parent = self.symbol(base, module, self.bindings[module])
            if parent not in self.classes and module + "." + str(parent) in self.classes:
                parent = module + "." + parent
            result = self.method(parent, name, seen=seen)
            if result:
                return result
        return None

    def resolve(self, symbol, scope, owner_class):
        if symbol is None:
            return None
        if symbol.startswith(("self.", "layer.", "decoder.", "owner.", "model.")):
            # Explicit layer arguments in model-local helpers denote the TP4
            # decoder, just like self in an inherited method.
            cls = self.root_class
            if (
                symbol.startswith("self.")
                and owner_class
                and owner_class not in (self.root_class, PACKAGE + ".optimized_decoder.OptimizedDecoder")
            ):
                cls = owner_class
            if symbol.startswith(("owner.", "model.")) and OWNER_ARGUMENTS.get(scope) != symbol.split(".")[0]:
                return None
            parts = symbol.split(".")
            for attribute in parts[1:-1]:
                cls = self.instance_types.get((cls, attribute))
                if cls is None:
                    return None
            return self.method(cls, parts[-1])
        if symbol in self.functions:
            return symbol
        if symbol in self.classes:
            constructor = self.method(symbol, "__init__")
            if constructor:
                return constructor
        module = self.functions[scope][0]
        parent = scope
        while parent.startswith(module):
            candidate = parent + "." + symbol
            if candidate in self.functions:
                return candidate
            if candidate in self.classes:
                return self.method(candidate, "__init__")
            parent = parent.rsplit(".", 1)[0]
        return None

    def symbol_options(self, expr, module, bindings, aliases):
        if isinstance(expr, ast.Name) and expr.id in aliases:
            return aliases[expr.id]
        if isinstance(expr, ast.IfExp):
            return self.symbol_options(expr.body, module, bindings, aliases) | self.symbol_options(
                expr.orelse, module, bindings, aliases
            )
        if isinstance(expr, ast.Attribute):
            return {s + "." + expr.attr for s in self.symbol_options(expr.value, module, bindings, aliases)}
        symbol = self.symbol(expr, module, bindings)
        return {symbol} if symbol else set()

    def classify(self, scope, node, canonical, spelling, bindings):
        root = canonical.split(".")[0]
        category = None
        if root in HOST_FRAMEWORKS:
            category = "host_framework_call_in_forward"
        elif root == "ttnn" and canonical.rsplit(".", 1)[-1] in HOST_TTNN:
            category = "host_tensor_conversion_readback_or_materialization"
        elif "." in canonical and canonical.rsplit(".", 1)[-1] in ("cpu", "numpy", "item", "tolist"):
            category = "possible_tensor_readback_method"
        elif canonical.endswith(".to") and any(
            isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value.startswith("cpu")
            for v in (*node.args, *(k.value for k in node.keywords))
        ):
            category = "tensor_to_cpu"
        elif canonical in ("eval", "exec", "__import__", "importlib.import_module"):
            category = "dynamic_code_or_module_execution_requires_review"
        if category:
            self.violations.append(self.detail(scope, node, category=category, call=canonical))
        elif root == "ttnn":
            self.external.append(self.detail(scope, node, call=canonical, boundary="TTNN API/native implementation"))
        elif root in ("math", "struct", "re") or canonical in vars(builtins):
            pass  # Scalar metadata, source construction, language builtins.
        elif canonical in ("self.policy.get", "layer.policy.get", "decoder.policy.get", "owner.policy.get") or (
            canonical == "model.policy.get" and OWNER_ARGUMENTS.get(scope) == "model"
        ):
            pass  # Constructor-owned Python policy dictionary; no tensor payload.
        elif canonical.startswith(("self.mesh.", "layer.mesh.", "decoder.mesh.", "owner.mesh.")):
            self.external.append(self.detail(scope, node, call=canonical, boundary="mesh metadata/API"))
        elif self.literal_method(scope, node, canonical, bindings):
            pass  # Receiver/type provenance is recorded; not a generic model-call exemption.
        elif canonical.startswith(PACKAGE + "."):
            # Data-only state constructors are the two known local classes.
            if canonical not in (
                PACKAGE + ".optimized_decoder.RecurrentState",
                PACKAGE + ".optimized_decoder.PagedState",
            ):
                finding = self.detail(scope, node, call=canonical, reason="unresolved_model_local_callable")
                self.unresolved.append(finding)
                self.violations.append({**finding, "category": "model_local_call_graph_gap"})
        elif any(a in spelling for a in ("._packed_correction_candidate.", "._fused_decode_norm.")) or (
            spelling.startswith(("owner._", "model._")) and scope in OWNER_METHODS
        ):
            finding = self.detail(scope, node, call=spelling, reason="unresolved_audited_owner_helper")
            self.unresolved.append(finding)
            self.violations.append({**finding, "category": "model_local_call_graph_gap"})
        elif spelling.startswith(("self.", "layer.", "decoder.", "owner.", "model.")):
            self.unresolved.append(
                self.detail(scope, node, call=spelling, reason="instance_member_or_external_CCL_boundary")
            )
        elif isinstance(node.func, ast.Name) and canonical not in ("RecurrentState", "PagedState"):
            self.unresolved.append(self.detail(scope, node, call=canonical, reason="unresolved_callable_or_callback"))
        elif isinstance(node.func, (ast.Call, ast.Subscript)):
            self.unresolved.append(
                self.detail(scope, node, call=spelling, reason="dynamically_returned_or_selected_callable")
            )
        elif canonical != spelling and "." in canonical:
            self.unresolved.append(self.detail(scope, node, call=canonical, reason="external_import_or_alias_boundary"))

    def detail(self, scope, node, **extra):
        return {"function": scope, "line": node.lineno, **extra}

    def run(self, extra_roots=()):
        roots = [self.method(self.root_class, name) for name in ("prefill_forward", "decode_forward")]
        if None in roots:
            raise ValueError("Missing TP4 prefill/decode roots")
        for name in extra_roots:
            if name not in self.functions:
                raise ValueError(f"Unknown explicit audit root {name}")
        pending, seen = roots + list(extra_roots), set()
        while pending:
            scope = pending.pop()
            if scope in seen:
                continue
            seen.add(scope)
            module, fn, owner_class = self.functions[scope]
            nodes = list(direct_walk(fn))
            lexical_scopes = [scope]
            ancestor = self.parents[scope]
            while ancestor in self.functions:
                lexical_scopes.append(ancestor)
                ancestor = self.parents[ancestor]
            lexical_nodes = [n for name in reversed(lexical_scopes) for n in direct_walk(self.functions[name][1])]
            bindings = {**self.bindings[module], **self.imports(module, lexical_nodes)}
            # Include simple/conditional function aliases without picking one
            # branch or erasing an earlier possible target on reassignment.
            aliases = {}
            for node in lexical_nodes:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    values = self.symbol_options(node.value, module, bindings, aliases) if node.value else set()
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if values:
                        for target in targets:
                            if isinstance(target, ast.Name):
                                aliases.setdefault(target.id, set()).update(values)
            if fn.name in SETUP_METHODS or scope in RUNTIME_SETUP:
                self.violations.append(self.detail(scope, fn, category="setup_reachable_from_forward", call=fn.name))
            for node in nodes:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.import_records.append(
                        self.detail(
                            scope,
                            node,
                            statement=ast.unparse(node),
                            classification="import_binding_not_tensor_fallback",
                        )
                    )
                if not isinstance(node, ast.Call):
                    continue
                spelling = ast.unparse(node.func)
                symbols = self.symbol_options(node.func, module, bindings, aliases)
                targets = set()
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == "super"
                ):
                    resolved = self.method(owner_class, node.func.attr, super_only=True)
                    if resolved:
                        targets.add(resolved)
                external = set()
                for symbol in symbols:
                    target = self.resolve(symbol, scope, owner_class)
                    if target:
                        targets.add(target)
                    else:
                        external.add(symbol)
                for target in sorted(targets):
                    extra = {}
                    if target in OWNER_METHODS:
                        argument = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else None
                        valid = (
                            argument == "self"
                            and owner_class in (self.root_class, PACKAGE + ".optimized_decoder.OptimizedDecoder")
                            or argument is not None
                            and argument == OWNER_ARGUMENTS.get(scope)
                            or argument in ("layer", "decoder")
                            and owner_class not in (PACKED_CLASS, NORM_CLASS)
                        )
                        extra = {"explicit_owner_argument": argument, "owner_binding_verified": bool(valid)}
                        if not valid:
                            self.violations.append(
                                self.detail(
                                    scope,
                                    node,
                                    category="runtime_owner_argument_unverified",
                                    call=spelling,
                                    target=target,
                                )
                            )
                    self.edges.append(self.detail(scope, node, call=spelling, target=target, **extra))
                    pending.append(target)
                if not targets and not external:
                    external.add(spelling)
                for symbol in sorted(external):
                    self.classify(scope, node, symbol, spelling, bindings)
        missing_helpers = sorted(REQUIRED_HELPERS - seen)
        for helper in missing_helpers:
            self.violations.append({"category": "required_runtime_helper_not_reached", "function": helper})
        reached_modules = {self.functions[name][0] for name in seen}
        literal_modules = {r["literal_source"].rsplit(".", 1)[0] for r in self.literal_calls}
        inspected = (
            reached_modules
            | literal_modules
            | {PACKAGE + "." + name for name in CORE_MODULES if PACKAGE + "." + name in self.sources}
        )
        hashes = {
            module.rsplit(".", 1)[-1]: hashlib.sha256(self.sources[module].encode()).hexdigest()
            for module in sorted(inspected)
        }
        source_transforms = [
            name
            for name in sorted(seen)
            if name.rsplit(".", 1)[0].endswith(("recurrent_variants", "recurrent_next_candidate"))
        ]
        return {
            "status": "pass" if not self.violations else "fail",
            "pass": not self.violations,
            "source_hashes": hashes,
            "roots": roots,
            "extra_roots": list(extra_roots),
            "audited_instance_and_owner_bindings": self.audited_bindings,
            "required_runtime_helpers": {
                "pass": not missing_helpers,
                "covered": sorted(REQUIRED_HELPERS & seen),
                "missing": missing_helpers,
            },
            "reachable_functions": sorted(seen),
            "call_graph": sorted(self.edges, key=lambda r: (r["function"], r["line"], r["target"])),
            "violations": self.violations,
            "runtime_imports": self.import_records,
            "source_string_transform_helpers": source_transforms,
            "unresolved_calls_for_review": self.unresolved,
            "ttnn_boundary_calls": self.external,
            "indexed_but_unreached_modules": sorted(set(self.sources) - reached_modules),
            "setup_entrypoints_outside_runtime": sorted(
                name
                for name in self.functions
                if (name.rsplit(".", 1)[-1] in SETUP_METHODS or name in RUNTIME_SETUP) and name not in seen
            ),
            "proven_literal_method_calls": self.literal_calls,
            "norm_setup_exclusion": {
                "functions": sorted(NORM_SETUP - seen),
                "reason": "Installer/constructor/view setup inspected for owner identity only; any runtime reachability is a violation.",
                "runtime_exclusion_verified": not bool(NORM_SETUP & seen),
            },
            "packed_setup_exclusion": {
                "functions": sorted(PACKED_SETUP - seen),
                "reason": "Constructor/installer repack weights and perform host readbacks at setup; inspected for binding identity only. Reaching either from a forward is a violation.",
                "runtime_exclusion_verified": not bool(PACKED_SETUP & seen),
            },
            "scope": "Conservative source-level call reachability from TP4 prefill/decode, across all policy branches; no tensor dataflow/type proof or device execution.",
            "limitations": [
                "Dynamic instance hooks, MethodType installations, callbacks, subscript-selected functions and nonliteral getattr cannot be exhaustively resolved. Experimental installers are not default roots; pass --extra-root for the device helper and review the binding separately.",
                "All policy branches are traversed; branch conditions are not proven. Unreachable candidate-only guards can therefore appear in unresolved_calls_for_review.",
                "Model-local imported functions, nested helpers and the explicitly verified packed-correction and fused-normalization instance bindings are followed; external CCLManager, LightweightModule, TTNN Python/native implementations and device C++ source-string contents require separate inspection/runtime evidence.",
                "Property getter/setter dispatch, module import side effects, decorators, generated classes and monkey patches are not completely modeled. In particular duplicate property setter/getter names share an AST index key; inspect model-local property bodies separately when they are on a runtime path. A local import alone is not a tensor fallback.",
                "String replacements, kernel-source specialization and scalar launch geometry are host control work, not host evaluation of model tensors. Their synchronization/numerical correctness is outside this audit.",
                "Simple aliases are flow-insensitive. Unknown object methods are not proved device-only; unresolved call metadata and the runtime guards must be reviewed before stage acceptance.",
            ],
            "runtime_sentinel": {
                "required": True,
                "status": "not_provided",
                "evidence": [],
                "contract": "Existing run_multichip_decoder runtime_conversion_sentinels must pass for both final-default layer kinds. The current guard patches six public transfer APIs during initial eager prefill, initial eager decode and decode capture. It is authoritative only for those executed paths/API surfaces; timed eager prefill, every eager call, arbitrary Torch operators and every readback API are not guarded. Static pass does not replace runtime guards, watcher, numerical checks or allocation tracking.",
            },
        }


def attach_runtime_results(report, paths):
    kinds = set()
    for path in paths:
        d = json.loads(Path(path).read_text())
        valid = (
            d.get("pass") is True
            and d.get("source_sha256") == report["source_hashes"]["multichip_decoder"]
            and d.get("candidate_policy") == {}
            and d.get("provenance", {}).get("mesh") == [1, 4]
            and d.get("trace", {}).get("runtime_conversion_sentinels") == "pass"
        )
        report["runtime_sentinel"]["evidence"].append({"path": str(path), "pass": valid, "layer": d.get("layer")})
        if not valid:
            report["violations"].append({"category": "runtime_sentinel_evidence_invalid", "path": str(path)})
        else:
            kinds.add(d["layer"])
    if paths:
        report["runtime_sentinel"]["status"] = (
            "pass"
            if kinds >= {0, 3} and all(v["pass"] for v in report["runtime_sentinel"]["evidence"])
            else "incomplete_or_failed"
        )
    report["pass"] = not report["violations"]
    report["status"] = "pass" if report["pass"] else "fail"


def negative_controls(sources):
    """Inject forbidden operations into temporary source copies, never imports."""
    cases = []
    controls = [
        (scope, "import ttnn\nttnn.to_torch(None)", "host_tensor_conversion_readback_or_materialization", scope)
        for scope in sorted(REQUIRED_HELPERS)
    ]
    controls.append(
        (
            PACKED_CLASS + ".project",
            "PackedCorrection(owner)",
            "setup_reachable_from_forward",
            PACKED_CLASS + ".__init__",
        )
    )
    controls.extend(
        [
            (
                NORM_CLASS + ".normalize",
                "model._upload(None)",
                "setup_reachable_from_forward",
                PACKAGE + ".multichip_decoder.MultichipDecoder._upload",
            ),
            (
                NORM_CLASS + ".normalize",
                "self._views([], 5120)",
                "setup_reachable_from_forward",
                NORM_CLASS + "._views",
            ),
            (
                NORM_CLASS + ".normalize",
                "install(model, layer_idx=self.layer_idx)",
                "setup_reachable_from_forward",
                NORM_MODULE + ".install",
            ),
            (
                NORM_CLASS + ".normalize",
                "FusedDecodeNormOwner(model, layer_idx=self.layer_idx)",
                "setup_reachable_from_forward",
                NORM_CLASS + ".__init__",
            ),
            (
                ROOT_CLASS + "._norm",
                "self._fused_decode_norm.normalize(None, x, key)",
                "runtime_owner_argument_unverified",
                ROOT_CLASS + "._norm",
            ),
            (
                PACKAGE + ".attention_gate_candidate.band_sources",
                "writer = object()",
                "model_local_call_graph_gap",
                PACKAGE + ".attention_gate_candidate.band_sources",
            ),
            (
                PACKAGE + ".attention_gate_candidate.band_sources",
                "_WRITER = object()",
                "model_local_call_graph_gap",
                PACKAGE + ".attention_gate_candidate.band_sources",
            ),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="runtime-audit-controls-") as directory:
        for index, (scope, injected, expected, violation_scope) in enumerate(controls):
            module = max((name for name in sources if scope.startswith(name + ".")), key=len)
            tree = ast.parse(sources[module])
            target = tree
            for part in scope[len(module) + 1 :].split("."):
                target = next(
                    node
                    for node in target.body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == part
                )
            offset = int(
                bool(
                    target.body
                    and isinstance(target.body[0], ast.Expr)
                    and isinstance(target.body[0].value, ast.Constant)
                    and isinstance(target.body[0].value.value, str)
                )
            )
            target.body[offset:offset] = ast.parse(injected).body
            ast.fix_missing_locations(tree)
            path = Path(directory) / f"control_{index}.py"
            path.write_text(ast.unparse(tree) + "\n")
            modified = {**sources, module: path.read_text()}
            result = Audit(modified).run()
            matches = [
                finding
                for finding in result["violations"]
                if finding.get("category") == expected and finding.get("function") == violation_scope
            ]
            cases.append(
                {
                    "injected_into": scope,
                    "injected_call": injected.splitlines()[-1],
                    "temporary_source_sha256": hashlib.sha256(modified[module].encode()).hexdigest(),
                    "expected_category": expected,
                    "matched_violations": matches,
                    "pass": not result["pass"] and bool(matches),
                }
            )
        # Binding drift must fail instead of silently dropping the helper graph.
        module = PACKED_MODULE
        before = "layer._packed_correction_candidate = candidate"
        assert sources[module].count(before) == 1
        path = Path(directory) / "broken_binding.py"
        path.write_text(sources[module].replace(before, "layer._packed_correction_candidate = None", 1))
        result = Audit({**sources, module: path.read_text()}).run()
        expected = "runtime_instance_binding_unverified"
        cases.append(
            {
                "injected_into": PACKED_MODULE + ".install",
                "injected_call": "replace helper binding with None",
                "expected_category": expected,
                "pass": not result["pass"] and any(v["category"] == expected for v in result["violations"]),
            }
        )
        # New conditional norm binding: constructor, owner identity and setup
        # connection must each be proved, not simply named in an allowlist.
        mutations = [
            (
                NORM_MODULE,
                'model._fused_decode_norm = FusedDecodeNormOwner(model, layer_idx=layer_idx) if family != "off" else None',
                "model._fused_decode_norm = None",
                "norm_binding_removed",
                "runtime_instance_binding_unverified",
            ),
            (
                NORM_MODULE,
                "FusedDecodeNormOwner(model, layer_idx=layer_idx)",
                "FusedDecodeNormOwner(other_model, layer_idx=layer_idx)",
                "norm_constructor_wrong_owner",
                "runtime_instance_binding_unverified",
            ),
            (
                NORM_MODULE,
                "def normalize(self, model, x, key):",
                "def normalize(self, different_model, x, key):",
                "norm_parameter_renamed",
                "runtime_owner_parameter_unverified",
            ),
            (
                NORM_MODULE,
                'assert id(model) == self.owner_id, "Shallow decoder clones require a new norm owner"',
                "assert True",
                "norm_identity_guard_removed",
                "runtime_instance_binding_unverified",
            ),
            (
                PACKAGE + ".multichip_decoder",
                "install_decode_norm(self, layer_idx=self.layer_idx)",
                "install_decode_norm(self, layer_idx=0)",
                "norm_factory_wrong_layer_index",
                "runtime_instance_binding_unverified",
            ),
        ]
        for module, before, after, label, expected in mutations:
            assert sources[module].count(before) == 1, (module, before)
            changed = sources[module].replace(before, after, 1)
            result = Audit({**sources, module: changed}).run()
            matches = [v for v in result["violations"] if v["category"] == expected]
            cases.append(
                {
                    "injected_into": module,
                    "injected_call": label,
                    "temporary_source_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                    "expected_category": expected,
                    "matched_violations": matches,
                    "pass": not result["pass"] and bool(matches),
                }
            )
        # A module symbol that merely looks like a kernel literal is not exempt.
        literal_module = PACKAGE + ".packed_swiglu_candidate"
        for label, duplicate in (("literal_replaced_with_object", False), ("literal_reassigned_to_object", True)):
            tree = ast.parse(sources[literal_module])
            assignment = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_WRITER" for t in n.targets)
            )
            replacement = ast.parse("_WRITER = object()").body[0]
            if duplicate:
                tree.body.append(replacement)
            else:
                assignment.value = replacement.value
            ast.fix_missing_locations(tree)
            changed = ast.unparse(tree) + "\n"
            result = Audit({**sources, literal_module: changed}).run()
            matches = [
                v
                for v in result["violations"]
                if v["category"] == "model_local_call_graph_gap"
                and v.get("function") == PACKAGE + ".attention_gate_candidate.band_sources"
            ]
            cases.append(
                {
                    "injected_into": literal_module,
                    "injected_call": label,
                    "temporary_source_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                    "expected_category": "model_local_call_graph_gap",
                    "matched_violations": matches,
                    "pass": not result["pass"] and bool(matches),
                }
            )
    return {
        "pass": all(case["pass"] for case in cases),
        "count": len(cases),
        "cases": cases,
        "scope": "Temporary Python source copies only; no model imports, TTNN imports or device execution.",
    }


def compact_report(report):
    """Retain gaps/bindings/control evidence without every ordinary TTNN call."""
    result = {
        key: value
        for key, value in report.items()
        if key
        not in (
            "reachable_functions",
            "call_graph",
            "ttnn_boundary_calls",
            "runtime_imports",
            "indexed_but_unreached_modules",
            "setup_entrypoints_outside_runtime",
            "source_string_transform_helpers",
        )
    }
    result["counts"] = {
        key: len(report.get(key, []))
        for key in (
            "reachable_functions",
            "call_graph",
            "ttnn_boundary_calls",
            "runtime_imports",
            "unresolved_calls_for_review",
        )
    }
    result["candidate_call_graph"] = [
        edge
        for edge in report.get("call_graph", [])
        if PACKED_MODULE in edge["function"] or PACKED_MODULE in edge["target"] or edge["target"] in REQUIRED_HELPERS
    ]
    result["compact_report"] = True
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "doc/optimized_multichip_decoder/runtime_audit.json")
    parser.add_argument(
        "--extra-root", action="append", default=[], help="Fully qualified model-local helper to inspect additionally"
    )
    parser.add_argument(
        "--runtime-result",
        type=Path,
        action="append",
        default=[],
        help="Existing final-default decoder result; never executes it",
    )
    parser.add_argument(
        "--stdout", action="store_true", help="Print complete report without writing the default artifact"
    )
    parser.add_argument(
        "--negative-controls", action="store_true", help="Audit forbidden-call mutations in temporary source copies"
    )
    parser.add_argument(
        "--compact", action="store_true", help="Keep binding/control/gap evidence but omit ordinary API-call listings"
    )
    args = parser.parse_args()
    try:
        sources = {PACKAGE + "." + p.stem: p.read_bytes().decode() for p in sorted((root / "tt").glob("*.py"))}
        report = Audit(sources).run(args.extra_root)
        report["auditor_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.negative_controls:
            report["negative_controls"] = negative_controls(sources)
            if not report["negative_controls"]["pass"]:
                report["violations"].append({"category": "negative_control_failed"})
        attach_runtime_results(report, args.runtime_result)
        if args.compact:
            report = compact_report(report)
    except (OSError, ValueError, SyntaxError, KeyError) as exc:
        report = {"pass": False, "status": "fail", "violations": [{"category": "audit_error", "error": str(exc)}]}
    if args.stdout:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "pass": report["pass"],
                    "output": str(args.output),
                    "reachable_functions": report.get("counts", {}).get(
                        "reachable_functions", len(report.get("reachable_functions", []))
                    ),
                    "violations": len(report["violations"]),
                    "runtime_sentinel": report.get("runtime_sentinel", {}).get("status", "not_provided"),
                }
            )
        )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
