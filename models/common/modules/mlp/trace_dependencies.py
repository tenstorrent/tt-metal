#!/usr/bin/env python3
"""
Static analysis tool to trace parameter dependencies in Python code.

Given a target file (e.g., mlp.py), this tool:
1. Finds all config/attribute accesses
2. Traces them to their definitions
3. Follows function calls to find nested dependencies
4. Builds a dependency graph showing what parameters affect runtime behavior
"""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ConfigAccess:
    """Represents an access to a config value"""

    key: str  # e.g., "DECODE_MLP_W1_W3_PRG_CONFIG"
    file: str
    line: int
    context: str  # the line of code
    in_condition: bool = False  # whether it's used in an if/ternary condition


@dataclass
class AttrAccess:
    """Represents an attribute access like self.dim or self.args.is_galaxy"""

    chain: str  # e.g., "self.args.is_galaxy"
    file: str
    line: int
    context: str
    in_condition: bool = False


@dataclass
class ConfigDefinition:
    """Represents where a config value is defined"""

    key: str
    file: str
    line_start: int
    line_end: int
    source: str  # the actual definition code
    dependencies: Set[str] = field(default_factory=set)  # what it references


class DependencyTracer(ast.NodeVisitor):
    """AST visitor that extracts config accesses and attribute chains"""

    def __init__(self, source_lines: List[str], filename: str):
        self.source_lines = source_lines
        self.filename = filename
        self.config_accesses: List[ConfigAccess] = []
        self.attr_accesses: List[AttrAccess] = []
        self.in_condition_depth = 0
        self.current_function: Optional[str] = None

    def visit_FunctionDef(self, node):
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_If(self, node):
        # Mark that we're inside a condition
        self.in_condition_depth += 1
        self.visit(node.test)
        self.in_condition_depth -= 1
        # Visit body and orelse normally
        for child in node.body + node.orelse:
            self.visit(child)

    def visit_IfExp(self, node):
        # Ternary expression: x if cond else y
        self.in_condition_depth += 1
        self.visit(node.test)
        self.in_condition_depth -= 1
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Subscript(self, node):
        """Catch things like self.model_config["KEY"]"""
        # Check if this is self.model_config[...]
        if isinstance(node.value, ast.Attribute):
            attr_chain = self._get_attr_chain(node.value)
            if attr_chain and attr_chain.endswith("model_config"):
                # Get the key
                if isinstance(node.slice, ast.Constant):
                    key = node.slice.value
                    line = node.lineno
                    context = self.source_lines[line - 1].strip() if line <= len(self.source_lines) else ""
                    self.config_accesses.append(
                        ConfigAccess(
                            key=key,
                            file=self.filename,
                            line=line,
                            context=context,
                            in_condition=self.in_condition_depth > 0,
                        )
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Catch attribute chains like self.dim, self.args.is_galaxy"""
        chain = self._get_attr_chain(node)
        if chain and chain.startswith("self."):
            line = node.lineno
            context = self.source_lines[line - 1].strip() if line <= len(self.source_lines) else ""
            self.attr_accesses.append(
                AttrAccess(
                    chain=chain,
                    file=self.filename,
                    line=line,
                    context=context,
                    in_condition=self.in_condition_depth > 0,
                )
            )
        self.generic_visit(node)

    def _get_attr_chain(self, node) -> Optional[str]:
        """Recursively build attribute chain: a.b.c -> "a.b.c" """
        if isinstance(node, ast.Attribute):
            base = self._get_attr_chain(node.value)
            if base:
                return f"{base}.{node.attr}"
            return None
        elif isinstance(node, ast.Name):
            return node.id
        return None


class ConfigDefinitionFinder(ast.NodeVisitor):
    """Find where model_config keys are defined"""

    def __init__(self, source: str, source_lines: List[str], filename: str):
        self.source = source
        self.source_lines = source_lines
        self.filename = filename
        self.definitions: Dict[str, ConfigDefinition] = {}

    def visit_Assign(self, node):
        self._check_config_assignment(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.target:
            self._check_config_assignment([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def _check_config_assignment(self, targets, value, lineno):
        """Check if this is a model_config["KEY"] = ... assignment"""
        for target in targets:
            if isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Attribute):
                    chain = self._get_attr_chain(target.value)
                    if chain and "model_config" in chain:
                        if isinstance(target.slice, ast.Constant):
                            key = target.slice.value
                            # Get the full definition including multi-line
                            end_line = self._find_statement_end(lineno)
                            source_chunk = "\n".join(self.source_lines[lineno - 1 : end_line])

                            # Extract dependencies from the value
                            deps = self._extract_dependencies(value)

                            self.definitions[key] = ConfigDefinition(
                                key=key,
                                file=self.filename,
                                line_start=lineno,
                                line_end=end_line,
                                source=source_chunk,
                                dependencies=deps,
                            )

    def _find_statement_end(self, start_line: int) -> int:
        """Find where a statement ends (handles multi-line)"""
        # Simple heuristic: count parens/brackets
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        for i in range(start_line - 1, len(self.source_lines)):
            line = self.source_lines[i]
            # Skip strings (rough approximation)
            in_string = False
            for j, char in enumerate(line):
                if char in ('"', "'") and (j == 0 or line[j - 1] != "\\"):
                    in_string = not in_string
                if not in_string:
                    if char == "(":
                        paren_depth += 1
                    elif char == ")":
                        paren_depth -= 1
                    elif char == "[":
                        bracket_depth += 1
                    elif char == "]":
                        bracket_depth -= 1
                    elif char == "{":
                        brace_depth += 1
                    elif char == "}":
                        brace_depth -= 1

            if paren_depth <= 0 and bracket_depth <= 0 and brace_depth <= 0:
                return i + 1

        return len(self.source_lines)

    def _extract_dependencies(self, node) -> Set[str]:
        """Extract all self.* references from an AST node"""
        deps = set()

        class DepExtractor(ast.NodeVisitor):
            def visit_Attribute(inner_self, n):
                chain = self._get_attr_chain(n)
                if chain and chain.startswith("self."):
                    deps.add(chain)
                inner_self.generic_visit(n)

            def visit_Name(inner_self, n):
                # Capture local variables that might be parameters
                if n.id in ("is_wormhole_b0", "is_blackhole", "nearest_32", "math"):
                    deps.add(f"builtin:{n.id}")
                inner_self.generic_visit(n)

        DepExtractor().visit(node)
        return deps

    def _get_attr_chain(self, node) -> Optional[str]:
        if isinstance(node, ast.Attribute):
            base = self._get_attr_chain(node.value)
            if base:
                return f"{base}.{node.attr}"
            return None
        elif isinstance(node, ast.Name):
            return node.id
        return None


def analyze_file(filepath: Path) -> DependencyTracer:
    """Analyze a Python file for config/attribute accesses"""
    source = filepath.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    tracer = DependencyTracer(source_lines, str(filepath))
    tracer.visit(tree)
    return tracer


def find_config_definitions(filepath: Path) -> Dict[str, ConfigDefinition]:
    """Find all model_config definitions in a file"""
    source = filepath.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    finder = ConfigDefinitionFinder(source, source_lines, str(filepath))
    finder.visit(tree)
    return finder.definitions


def trace_mlp_dependencies(mlp_path: Path, model_config_path: Path):
    """Main analysis: trace all dependencies from mlp.py"""

    print("=" * 80)
    print(f"Analyzing: {mlp_path.name}")
    print("=" * 80)

    # Step 1: Find all accesses in mlp.py
    mlp_tracer = analyze_file(mlp_path)

    # Step 2: Find all config definitions in model_config.py
    config_defs = find_config_definitions(model_config_path)

    # Step 3: Report config accesses
    print("\n## Config Accesses in MLP forward()")
    print("-" * 60)

    config_keys_used = set()
    for access in mlp_tracer.config_accesses:
        config_keys_used.add(access.key)
        cond_marker = " [IN CONDITION]" if access.in_condition else ""
        print(f'  L{access.line}: model_config["{access.key}"]{cond_marker}')

    # Step 4: Report attribute accesses that are in conditions
    print("\n## Attributes Used in Conditions")
    print("-" * 60)

    condition_attrs = set()
    for access in mlp_tracer.attr_accesses:
        if access.in_condition:
            condition_attrs.add(access.chain)

    for attr in sorted(condition_attrs):
        print(f"  {attr}")

    # Step 5: For each config key, show its dependencies
    print("\n## Config Key Dependencies")
    print("-" * 60)

    all_root_deps = set()

    for key in sorted(config_keys_used):
        if key in config_defs:
            defn = config_defs[key]
            print(f"\n### {key}")
            print(f"    Defined at: model_config.py L{defn.line_start}-{defn.line_end}")
            if defn.dependencies:
                print(f"    Direct dependencies:")
                for dep in sorted(defn.dependencies):
                    print(f"      - {dep}")
                    all_root_deps.add(dep)
        else:
            print(f"\n### {key}")
            print(f"    WARNING: Definition not found in model_config.py")

    # Step 6: Summary
    print("\n" + "=" * 80)
    print("## SUMMARY: Root Parameters Affecting MLP Behavior")
    print("=" * 80)

    # Categorize
    self_attrs = sorted([d for d in all_root_deps if d.startswith("self.")])
    builtins = sorted([d for d in all_root_deps if d.startswith("builtin:")])

    print("\n### From ModelArgs (self.*):")
    for attr in self_attrs:
        print(f"  - {attr}")

    print("\n### Architecture/Builtin Checks:")
    for b in builtins:
        print(f"  - {b.replace('builtin:', '')}")

    print("\n### Direct Condition Variables in forward():")
    for attr in sorted(condition_attrs):
        print(f"  - {attr}")

    return {
        "config_keys": config_keys_used,
        "condition_attrs": condition_attrs,
        "root_deps": all_root_deps,
        "config_defs": config_defs,
    }


def extract_function_params_and_locals(filepath: Path, func_name: str) -> Dict[str, any]:
    """Extract parameters and local variable assignments from a function"""
    source = filepath.read_text()
    tree = ast.parse(source)

    result = {"params": [], "locals": {}, "runtime_inputs": set()}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Get parameters
            for arg in node.args.args:
                result["params"].append(arg.arg)
                if arg.arg not in ("self",):
                    result["runtime_inputs"].add(arg.arg)

            # Get local assignments
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            # Try to get what it's assigned from
                            result["locals"][target.id] = ast.unparse(stmt.value)

    return result


def trace_decoders_optimizations(model_config_path: Path):
    """Trace what DecodersPrecision / DECODERS_OPTIMIZATIONS depends on"""
    source = model_config_path.read_text()
    source_lines = source.splitlines()

    info = {
        "class_methods": [],
        "tensor_groups": [],
        "op_groups": [],
        "precision_settings": [],
        "math_fidelity_settings": [],
    }

    # Find the enums and class
    in_class = None
    for i, line in enumerate(source_lines, 1):
        if "class TensorGroup" in line:
            in_class = "TensorGroup"
        elif "class OpGroup" in line:
            in_class = "OpGroup"
        elif "class PrecisionSetting" in line:
            in_class = "PrecisionSetting"
        elif "class MathFidelitySetting" in line:
            in_class = "MathFidelitySetting"
        elif "class DecodersPrecision" in line:
            in_class = "DecodersPrecision"
        elif line.strip().startswith("class "):
            in_class = None

        if in_class == "TensorGroup" and "=" in line and not line.strip().startswith("class"):
            match = re.match(r"\s+(\w+)\s*=", line)
            if match:
                info["tensor_groups"].append(match.group(1))
        elif in_class == "OpGroup" and "=" in line and not line.strip().startswith("class"):
            match = re.match(r"\s+(\w+)\s*=", line)
            if match:
                info["op_groups"].append(match.group(1))
        elif in_class == "PrecisionSetting" and "=" in line:
            match = re.match(r"\s+(\w+)\s*=", line)
            if match:
                info["precision_settings"].append(match.group(1))
        elif in_class == "MathFidelitySetting" and "=" in line:
            match = re.match(r"\s+(\w+)\s*=", line)
            if match:
                info["math_fidelity_settings"].append(match.group(1))

    return info


def build_full_dependency_graph(mlp_path: Path, model_config_path: Path):
    """Build a complete dependency graph with transitive closure"""

    result = trace_mlp_dependencies(mlp_path, model_config_path)

    # Extract runtime parameters from forward()
    print("\n" + "=" * 80)
    print("## RUNTIME PARAMETERS (from forward() signature and locals)")
    print("=" * 80)

    func_info = extract_function_params_and_locals(mlp_path, "forward")
    print(f"\n### Function parameters:")
    for p in func_info["params"]:
        if p != "self":
            print(f"  - {p}")

    print(f"\n### Key local variables:")
    key_locals = ["seq_len", "TG", "layer_num", "mode", "pc_1", "pc_2", "pc_3", "activation_dtype", "memory_config"]
    for local, expr in func_info["locals"].items():
        if local in key_locals:
            print(f"  - {local} = {expr[:70]}{'...' if len(expr) > 70 else ''}")

    # Trace DECODERS_OPTIMIZATIONS
    print("\n" + "=" * 80)
    print("## DECODERS_OPTIMIZATIONS Structure")
    print("=" * 80)

    opt_info = trace_decoders_optimizations(model_config_path)

    print("\n### TensorGroup enum (weight/activation dtypes):")
    for tg in opt_info["tensor_groups"]:
        print(f"  - {tg}")

    print("\n### OpGroup enum (compute kernel fidelity):")
    for og in opt_info["op_groups"]:
        print(f"  - {og}")

    print("\n### PrecisionSetting values:")
    for ps in opt_info["precision_settings"]:
        print(f"  - {ps}")

    print("\n### MathFidelitySetting values:")
    for mf in opt_info["math_fidelity_settings"]:
        print(f"  - {mf}")

    # Deep trace
    print("\n" + "=" * 80)
    print("## DEEP TRACE: Following self.* references in definitions")
    print("=" * 80)

    # Parse model_config to find where self.* attrs are set
    source = model_config_path.read_text()
    source_lines = source.splitlines()

    # Find assignments to self.* in __init__
    init_assignments = {}

    # Simple regex-based extraction for self.X = ...
    for i, line in enumerate(source_lines, 1):
        match = re.match(r"\s+self\.(\w+)\s*=\s*(.+)", line)
        if match:
            attr_name = match.group(1)
            value_expr = match.group(2).strip()
            if attr_name not in init_assignments:
                init_assignments[attr_name] = []
            init_assignments[attr_name].append((i, value_expr))

    # For each root dependency, show where it's set
    print("\n### Where root parameters are defined:")
    for dep in sorted(result["root_deps"]):
        if dep.startswith("self."):
            attr = dep.split(".")[1]  # Get first attr after self
            if attr in init_assignments:
                print(f"\n  {dep}:")
                for line_no, expr in init_assignments[attr][:3]:  # Show first 3
                    print(f"    L{line_no}: self.{attr} = {expr[:60]}{'...' if len(expr) > 60 else ''}")

    # Scan mlp.py for ALL self.args.* and self.* accesses
    print("\n" + "=" * 80)
    print("## ALL self.args.* ACCESSES in mlp.py")
    print("=" * 80)

    mlp_source = mlp_path.read_text()

    # Find all self.args.* patterns
    args_pattern = re.compile(r"self\.args\.(\w+(?:\(\))?)")
    args_matches = set(args_pattern.findall(mlp_source))

    print("\n### self.args.* attributes used:")
    for match in sorted(args_matches):
        print(f"  - self.args.{match}")

    # Find all self.tt_ccl.* patterns
    ccl_pattern = re.compile(r"self\.tt_ccl\.(\w+)")
    ccl_matches = set(ccl_pattern.findall(mlp_source))

    print("\n### self.tt_ccl.* methods used:")
    for match in sorted(ccl_matches):
        print(f"  - self.tt_ccl.{match}()")

    # Find self.mesh_device usages
    mesh_pattern = re.compile(r"self\.mesh_device")
    mesh_count = len(mesh_pattern.findall(mlp_source))
    print(f"\n### self.mesh_device used: {mesh_count} times")

    # Final summary: categorize into data params vs method params
    print("\n" + "=" * 80)
    print("## FINAL CATEGORIZED PARAMETER LIST")
    print("=" * 80)

    data_params = set()
    method_refs = set()

    for dep in result["root_deps"]:
        if dep.startswith("self."):
            attr = dep.split(".")[-1]
            # Check if it's a method by looking at usage
            if attr in (
                "matmul_config",
                "matmul_1d_config_from_tensor_shapes",
                "dram_matmul_config",
                "find_prefill_grid",
                "find_grid",
            ):
                method_refs.add(attr)
            else:
                data_params.add(dep)

    print("\n### Data Parameters (scalar/config values):")
    for p in sorted(data_params):
        print(f"  - {p}")

    print("\n### Method References (compute helpers):")
    for m in sorted(method_refs):
        print(f"  - self.{m}()")

    print("\n### Runtime Inputs (per-call):")
    print("  - mode: 'decode' | 'prefill'")
    print("  - x.shape[-2] → seq_len")
    print("  - layer_num (instance variable, used for optimization lookup)")

    # Add self.args params
    print("\n### From self.args (ModelArgs):")
    for match in sorted(args_matches):
        is_method = "()" in match or match in ("ccl_topology",)
        suffix = "()" if is_method and "()" not in match else ""
        print(f"  - self.args.{match}{suffix}")

    print("\n### CCL Infrastructure (self.tt_ccl):")
    for match in sorted(ccl_matches):
        print(f"  - self.tt_ccl.{match}()")

    return result


class FunctionCallFinder(ast.NodeVisitor):
    """Find all function calls in a file/function"""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.calls: List[Tuple[str, int, str]] = []  # (func_name, line, context)

    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # e.g., tt_all_reduce, self.foo, ttnn.linear
            chain = self._get_attr_chain(node.func)
            if chain:
                func_name = chain

        if func_name:
            line = node.lineno
            context = self.source_lines[line - 1].strip() if line <= len(self.source_lines) else ""
            self.calls.append((func_name, line, context))

        self.generic_visit(node)

    def _get_attr_chain(self, node) -> Optional[str]:
        if isinstance(node, ast.Attribute):
            base = self._get_attr_chain(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return None


class FunctionAnalyzer(ast.NodeVisitor):
    """Analyze a function for conditions and dependencies"""

    def __init__(self, source_lines: List[str], func_name: str):
        self.source_lines = source_lines
        self.func_name = func_name
        self.conditions: List[Tuple[int, str]] = []  # (line, condition_expr)
        self.param_names: List[str] = []
        self.local_vars: Dict[str, str] = {}
        self.terminal_ops: List[Tuple[str, int]] = []  # ttnn ops called

    def visit_FunctionDef(self, node):
        if node.name == self.func_name:
            # Get parameters
            for arg in node.args.args:
                self.param_names.append(arg.arg)
            # Visit body
            for stmt in node.body:
                self.visit(stmt)

    def visit_If(self, node):
        line = node.lineno
        cond = ast.unparse(node.test)
        self.conditions.append((line, cond))
        self.generic_visit(node)

    def visit_IfExp(self, node):
        line = node.lineno
        cond = ast.unparse(node.test)
        self.conditions.append((line, cond))
        self.generic_visit(node)

    def visit_Call(self, node):
        # Track ttnn.* calls as terminal ops
        if isinstance(node.func, ast.Attribute):
            chain = []
            n = node.func
            while isinstance(n, ast.Attribute):
                chain.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name) and n.id == "ttnn":
                op_name = "ttnn." + ".".join(reversed(chain))
                self.terminal_ops.append((op_name, node.lineno))
        self.generic_visit(node)


def analyze_function_in_file(filepath: Path, func_name: str) -> Optional[FunctionAnalyzer]:
    """Analyze a specific function in a file"""
    source = filepath.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    analyzer = FunctionAnalyzer(source_lines, func_name)
    analyzer.visit(tree)
    return analyzer


def find_function_calls_in_forward(filepath: Path) -> List[Tuple[str, int, str]]:
    """Find all function calls in forward() method"""
    source = filepath.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    # Find the forward method
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            finder = FunctionCallFinder(source_lines)
            finder.visit(node)
            return finder.calls
    return []


def trace_called_functions(mlp_path: Path, ccl_path: Path):
    """Trace into functions called from mlp.py forward()"""

    print("\n" + "=" * 80)
    print("## FUNCTION CALL TRACE (following into called functions)")
    print("=" * 80)

    # Find all calls in forward()
    calls = find_function_calls_in_forward(mlp_path)

    # Filter to interesting calls (not ttnn.*, not self.*)
    interesting_calls = set()
    for func_name, line, context in calls:
        if func_name.startswith("tt_") or func_name in ("tt_all_reduce", "tt_all_gather"):
            interesting_calls.add(func_name)

    print(f"\n### Functions called from MLP.forward() that need tracing:")
    for func in sorted(interesting_calls):
        print(f"  - {func}")

    # Analyze tt_all_reduce
    print("\n### tt_all_reduce analysis:")
    analyzer = analyze_function_in_file(ccl_path, "tt_all_reduce")
    if analyzer:
        print(f"\n  Parameters: {', '.join(analyzer.param_names)}")

        print(f"\n  Conditions (branching points):")
        for line, cond in analyzer.conditions:
            print(f"    L{line}: if {cond}")

        print(f"\n  Terminal ttnn ops called:")
        seen_ops = set()
        for op, line in analyzer.terminal_ops:
            if op not in seen_ops:
                print(f"    - {op}")
                seen_ops.add(op)

    # Analyze tt_all_gather
    print("\n### tt_all_gather analysis:")
    analyzer_ag = analyze_function_in_file(ccl_path, "tt_all_gather")
    if analyzer_ag:
        print(f"\n  Parameters: {', '.join(analyzer_ag.param_names)}")

        print(f"\n  Conditions (branching points):")
        for line, cond in analyzer_ag.conditions:
            print(f"    L{line}: if {cond}")

        print(f"\n  Terminal ttnn ops called:")
        seen_ops = set()
        for op, line in analyzer_ag.terminal_ops:
            if op not in seen_ops:
                print(f"    - {op}")
                seen_ops.add(op)

    # Summary: what variables control CCL branching
    print("\n" + "=" * 80)
    print("## CCL BRANCHING VARIABLES")
    print("=" * 80)

    all_conditions = []
    if analyzer:
        all_conditions.extend(analyzer.conditions)
    if analyzer_ag:
        all_conditions.extend(analyzer_ag.conditions)

    cond_vars = extract_condition_variables(all_conditions)

    print("\n### Variables that control CCL path selection:")
    for var in sorted(cond_vars):
        print(f"  - {var}")

    # Map back to MLP call sites
    print("\n### How MLP.forward() passes these to tt_all_reduce:")
    mlp_source = mlp_path.read_text()

    # Find tt_all_reduce calls
    ar_calls = re.findall(r"tt_all_reduce\([^)]+\)", mlp_source, re.DOTALL)
    for i, call in enumerate(ar_calls[:3], 1):  # Show first 3
        # Clean up whitespace
        call_clean = " ".join(call.split())
        print(f"\n  Call {i}: {call_clean[:100]}{'...' if len(call_clean) > 100 else ''}")

    # Final comprehensive summary
    print("\n" + "=" * 80)
    print("## COMPLETE MINIMAL PARAMETER SET")
    print("=" * 80)

    print(
        """
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 0: HARDWARE (opaque - cannot trace further)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ • mesh_device.shape           → [rows, cols] e.g., [1,1], [1,2], [4,8]     │
│ • mesh_device.get_num_devices() → 1, 2, 8, 32                               │
│ • mesh_device.dram_grid_size() → architecture-specific                      │
│ • is_wormhole_b0() / is_blackhole() → architecture detection               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: MODEL CONFIG (from checkpoint params.json / config.json)           │
├─────────────────────────────────────────────────────────────────────────────┤
│ • dim                          → model hidden dimension                     │
│ • hidden_dim                   → MLP intermediate dimension                 │
│ • n_layers                     → number of decoder layers                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: DERIVED FROM HARDWARE + MODEL                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ • is_galaxy                    → num_devices == 32                          │
│ • cluster_shape                → mesh_device.shape                          │
│ • prefill_len_cutoff           → 512 (BH) or 1024 (WH)                      │
│ • ccl_topology                 → Ring (T3K) or Linear                       │
│ • num_reduce_scatter_links     → 1                                          │
│ • num_all_gather_links         → 2 (TG) or 1                                │
│ • ccl_dtype                    → ttnn.bfloat8_b                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 3: RUNTIME CONFIG (set at model init)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ • max_batch_size               → tile_padded_batch_rows                     │
│ • optimizations                → "accuracy" | "performance" | custom        │
│   └─ TensorGroup.FF1_FF3       → BFP4 | BFP8 | BF16                        │
│   └─ TensorGroup.FF2           → BFP4 | BFP8 | BF16                        │
│   └─ TensorGroup.ACTIVATION    → BFP8 | BF16 | None                        │
│   └─ OpGroup.LI_FF1_FF3        → LOFI | HIFI2 | HIFI2_FP16 | HIFI4         │
│   └─ OpGroup.LI_FF2            → LOFI | HIFI2 | HIFI2_FP16 | HIFI4         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 4: PER-CALL RUNTIME INPUTS                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ • mode                         → "decode" | "prefill"                       │
│ • seq_len                      → from x.shape[-2]                           │
│ • layer_num                    → 0..n_layers-1 (per-layer optimization)     │
│ • input_tensor.is_sharded()    → runtime tensor state                       │
│ • input_tensor.dtype           → runtime tensor dtype                       │
│ • input_tensor.shape           → runtime tensor shape                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 5: TERMINAL TTNN OPS (opaque - implementation in C++/kernels)         │
├─────────────────────────────────────────────────────────────────────────────┤
│ MLP Core:                                                                   │
│ • ttnn.linear                  → matmul with optional fused activation      │
│ • ttnn.mul                     → element-wise multiply with activation      │
│                                                                             │
│ Memory Management:                                                          │
│ • ttnn.to_memory_config        → move between memory layouts                │
│ • ttnn.reshape                 → tensor reshaping                           │
│ • ttnn.deallocate              → free tensor memory                         │
│                                                                             │
│ Collective Communication (CCL):                                             │
│ • ttnn.experimental.reduce_scatter_minimal_async                            │
│ • ttnn.experimental.all_gather_async                                        │
│ • ttnn.experimental.fast_reduce_nc                                          │
│ • ttnn.sharded_to_interleaved                                               │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    )

    return interesting_calls


def extract_condition_variables(conditions: List[Tuple[int, str]]) -> Set[str]:
    """Extract variable names from condition expressions"""
    vars_used = set()

    for line, cond in conditions:
        # Simple regex to find identifiers
        # This is approximate but catches most cases
        identifiers = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\b", cond)
        for ident in identifiers:
            # Filter out Python keywords and common functions
            if ident not in (
                "if",
                "else",
                "and",
                "or",
                "not",
                "in",
                "is",
                "None",
                "True",
                "False",
                "list",
                "len",
                "range",
                "min",
                "max",
                "int",
                "str",
                "bool",
            ):
                vars_used.add(ident)

    return vars_used


def trace_matmul_config_helpers(model_config_path: Path) -> Dict[str, Any]:
    """Trace into matmul_config, dram_matmul_config, etc. helper methods"""

    source = model_config_path.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    helpers = {}

    # Methods to analyze
    target_methods = [
        "matmul_config",
        "dram_matmul_config",
        "matmul_1d_config",
        "matmul_1d_config_from_tensor_shapes",
        "create_dram_sharded_mem_config",
        "dram_shard_core_grid_for_k",
        "dram_shard_core_grid_for_k_and_n",
        "find_grid",
        "find_prefill_grid",
        "find_largest_divisor",
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_methods:
            # Extract parameters
            params = []
            for arg in node.args.args:
                param_info = {"name": arg.arg}
                if arg.annotation:
                    param_info["type"] = ast.unparse(arg.annotation)
                params.append(param_info)

            # Extract defaults
            defaults = node.args.defaults
            if defaults:
                # defaults align to end of params
                offset = len(params) - len(defaults)
                for i, default in enumerate(defaults):
                    params[offset + i]["default"] = ast.unparse(default)

            # Find what self.* attributes are used
            self_attrs = set()

            class AttrFinder(ast.NodeVisitor):
                def visit_Attribute(inner_self, n):
                    if isinstance(n.value, ast.Name) and n.value.id == "self":
                        self_attrs.add(f"self.{n.attr}")
                    elif isinstance(n.value, ast.Attribute):
                        # Handle self.foo.bar
                        chain = []
                        curr = n
                        while isinstance(curr, ast.Attribute):
                            chain.append(curr.attr)
                            curr = curr.value
                        if isinstance(curr, ast.Name) and curr.id == "self":
                            chain.append("self")
                            self_attrs.add(".".join(reversed(chain)))
                    inner_self.generic_visit(n)

            AttrFinder().visit(node)

            # Find return type / what it returns
            returns = []
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and stmt.value:
                    ret_expr = ast.unparse(stmt.value)
                    if len(ret_expr) < 100:
                        returns.append(ret_expr)
                    else:
                        returns.append(ret_expr[:100] + "...")

            helpers[node.name] = {
                "params": params,
                "uses_self_attrs": sorted(self_attrs),
                "returns": returns[:3],  # First 3 return statements
                "line_start": node.lineno,
                "line_end": node.end_lineno,
            }

    return helpers


def build_json_output(mlp_path: Path, model_config_path: Path, ccl_path: Path) -> Dict[str, Any]:
    """Build a complete JSON-serializable dependency graph"""

    # Analyze MLP
    mlp_tracer = analyze_file(mlp_path)
    config_defs = find_config_definitions(model_config_path)

    # Config accesses
    config_accesses = []
    for access in mlp_tracer.config_accesses:
        config_accesses.append({"key": access.key, "line": access.line, "in_condition": access.in_condition})

    # Config definitions with dependencies
    config_definitions = {}
    for key, defn in config_defs.items():
        config_definitions[key] = {
            "line_start": defn.line_start,
            "line_end": defn.line_end,
            "dependencies": sorted(defn.dependencies),
        }

    # Analyze CCL functions
    ccl_functions = {}
    for func_name in ["tt_all_reduce", "tt_all_gather", "tt_distributed_rmsnorm", "tt_sharded_distributed_rmsnorm"]:
        analyzer = analyze_function_in_file(ccl_path, func_name)
        if analyzer and analyzer.param_names:
            ccl_functions[func_name] = {
                "parameters": analyzer.param_names,
                "conditions": [{"line": line, "expr": cond} for line, cond in analyzer.conditions],
                "terminal_ops": list(set(op for op, _ in analyzer.terminal_ops)),
            }

    # Trace matmul helpers
    matmul_helpers = trace_matmul_config_helpers(model_config_path)

    # DECODERS_OPTIMIZATIONS structure
    opt_info = trace_decoders_optimizations(model_config_path)

    # Extract self.args.* from mlp.py
    mlp_source = mlp_path.read_text()
    args_pattern = re.compile(r"self\.args\.(\w+)")
    args_used = sorted(set(args_pattern.findall(mlp_source)))

    # Build hierarchical parameter structure
    parameters = {
        "level_0_hardware": {
            "description": "Opaque hardware parameters",
            "params": [
                {"name": "mesh_device.shape", "type": "List[int]", "example": "[1,1], [1,2], [4,8]"},
                {"name": "mesh_device.get_num_devices()", "type": "int", "values": [1, 2, 8, 32]},
                {"name": "mesh_device.dram_grid_size()", "type": "CoreCoord", "description": "Architecture-specific"},
                {"name": "is_wormhole_b0()", "type": "bool", "description": "Architecture detection"},
                {"name": "is_blackhole()", "type": "bool", "description": "Architecture detection"},
            ],
        },
        "level_1_model_config": {
            "description": "From checkpoint params.json / config.json",
            "params": [
                {"name": "dim", "type": "int", "description": "Model hidden dimension"},
                {"name": "hidden_dim", "type": "int", "description": "MLP intermediate dimension"},
                {"name": "n_layers", "type": "int", "description": "Number of decoder layers"},
            ],
        },
        "level_2_derived": {
            "description": "Derived from hardware + model",
            "params": [
                {"name": "is_galaxy", "derivation": "num_devices == 32"},
                {"name": "cluster_shape", "derivation": "mesh_device.shape"},
                {"name": "prefill_len_cutoff", "derivation": "512 (BH) or 1024 (WH)"},
                {"name": "ccl_topology", "derivation": "Ring (T3K) or Linear"},
                {"name": "num_reduce_scatter_links", "default": 1},
                {"name": "num_all_gather_links", "derivation": "2 (TG) or 1"},
                {"name": "ccl_dtype", "default": "ttnn.bfloat8_b"},
            ],
        },
        "level_3_runtime_config": {
            "description": "Set at model init",
            "params": [
                {"name": "max_batch_size", "derives": "tile_padded_batch_rows"},
                {
                    "name": "optimizations",
                    "type": "str | DecodersPrecision",
                    "values": ["accuracy", "performance", "custom"],
                },
            ],
            "optimization_settings": {
                "tensor_groups": opt_info["tensor_groups"],
                "op_groups": opt_info["op_groups"],
                "precision_values": opt_info["precision_settings"],
                "fidelity_values": opt_info["math_fidelity_settings"],
            },
        },
        "level_4_per_call": {
            "description": "Per-call runtime inputs",
            "params": [
                {"name": "mode", "type": "str", "values": ["decode", "prefill"]},
                {"name": "seq_len", "derivation": "x.shape[-2]"},
                {"name": "layer_num", "type": "int", "range": "0..n_layers-1"},
                {"name": "input_tensor.is_sharded()", "type": "bool"},
                {"name": "input_tensor.dtype", "type": "ttnn.DataType"},
                {"name": "input_tensor.shape", "type": "Tuple[int, ...]"},
            ],
        },
        "level_5_terminal_ops": {
            "description": "Terminal ttnn ops (opaque C++/kernel implementation)",
            "mlp_core": ["ttnn.linear", "ttnn.mul"],
            "memory_management": ["ttnn.to_memory_config", "ttnn.reshape", "ttnn.deallocate"],
            "ccl": [
                "ttnn.experimental.reduce_scatter_minimal_async",
                "ttnn.experimental.all_gather_async",
                "ttnn.experimental.fast_reduce_nc",
                "ttnn.sharded_to_interleaved",
            ],
        },
    }

    return {
        "source_file": str(mlp_path),
        "config_file": str(model_config_path),
        "ccl_file": str(ccl_path),
        "config_accesses": config_accesses,
        "config_definitions": config_definitions,
        "ccl_functions": ccl_functions,
        "matmul_helpers": matmul_helpers,
        "args_used": args_used,
        "parameters": parameters,
    }


def print_matmul_helpers(model_config_path: Path):
    """Print analysis of matmul helper methods"""

    print("\n" + "=" * 80)
    print("## MATMUL CONFIG HELPER METHODS")
    print("=" * 80)

    helpers = trace_matmul_config_helpers(model_config_path)

    for name, info in sorted(helpers.items()):
        print(f"\n### {name}()")
        print(f"    Lines: {info['line_start']}-{info['line_end']}")

        print(f"    Parameters:")
        for p in info["params"]:
            type_str = f": {p.get('type', '?')}" if "type" in p else ""
            default_str = f" = {p['default']}" if "default" in p else ""
            print(f"      - {p['name']}{type_str}{default_str}")

        if info["uses_self_attrs"]:
            print(f"    Uses self.* attrs:")
            for attr in info["uses_self_attrs"]:
                print(f"      - {attr}")

        if info["returns"]:
            print(f"    Returns:")
            for ret in info["returns"][:2]:
                print(f"      - {ret}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trace parameter dependencies in Python code")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--json-file", type=str, help="Write JSON to file")
    parser.add_argument("--matmul-helpers", action="store_true", help="Include detailed matmul helper analysis")
    parser.add_argument("mlp_path", nargs="?", help="Path to MLP file")
    parser.add_argument("config_path", nargs="?", help="Path to model_config file")

    args = parser.parse_args()

    # Default paths
    base = Path(__file__).parent.parent.parent.parent  # models/
    mlp_path = Path(args.mlp_path) if args.mlp_path else base / "tt_transformers" / "tt" / "mlp.py"
    config_path = Path(args.config_path) if args.config_path else base / "tt_transformers" / "tt" / "model_config.py"
    ccl_path = base / "tt_transformers" / "tt" / "ccl.py"

    if args.json or args.json_file:
        # JSON output mode
        json_data = build_json_output(mlp_path, config_path, ccl_path)

        if args.json_file:
            with open(args.json_file, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"JSON output written to: {args.json_file}")
        else:
            print(json.dumps(json_data, indent=2))
    else:
        # Text output mode
        result = build_full_dependency_graph(mlp_path, config_path)

        # Trace into called functions
        trace_called_functions(mlp_path, ccl_path)

        # Optionally include matmul helpers
        if args.matmul_helpers:
            print_matmul_helpers(config_path)
