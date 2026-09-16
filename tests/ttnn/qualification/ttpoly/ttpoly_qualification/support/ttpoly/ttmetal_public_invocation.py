# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source-bound public API calls for exhaustive collection, not evaluators.

The same descriptor is used for stock and candidate. Backward collection fixes
the incoming gradient to one; it does not claim exhaustive two-tensor coverage.
Explicit DRAM output selection exercises the presently packaged device route.
"""
from dataclasses import dataclass
from pathlib import Path
import re
from types import SimpleNamespace

from ttpoly.ttmetal_dispatch import _balanced_end, _code, _signature_parameters


@dataclass(frozen=True)
class PublicInvocation:
    public_operation: str
    tensor_roles: tuple[str, ...]
    public_args: tuple = ()
    public_kwargs: tuple = ()
    single_result_container: bool = False
    explicit_dram_output: bool = False

    def invoke(self, ttnn, activation, gradient=None):
        """Call public TTNN only; the caller supplies the unit-gradient tensor."""
        if ("incoming_gradient" in self.tensor_roles) != (gradient is not None):
            raise ValueError("gradient presence must match the typed public tensor roles")
        tensors = {"activation_input": activation, "incoming_gradient": gradient}
        if not self.tensor_roles or any(role not in tensors for role in self.tensor_roles):
            raise ValueError("unknown public tensor role")
        keywords = {}
        for name, value in self.public_kwargs:
            if name in keywords:
                raise ValueError("duplicate public keyword")
            if isinstance(value, dict):
                if set(value) != {"enum"} or not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", value["enum"]):
                    raise ValueError("invalid public enum selector")
                enum, member = value["enum"].split(".")
                value = getattr(getattr(ttnn, enum), member)
            keywords[name] = value
        if self.explicit_dram_output:
            if "memory_config" in keywords:
                raise ValueError("output memory is specified twice")
            keywords["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        result = getattr(ttnn, self.public_operation)(
            *(tensors[role] for role in self.tensor_roles), *self.public_args, **keywords
        )
        if self.single_result_container:
            if not isinstance(result, (tuple, list)) or len(result) != 1 or result[0] is None:
                raise ValueError("public backward result must contain exactly one nonempty tensor")
            result = result[0]
        if not isinstance(result, ttnn.Tensor):
            raise ValueError("public operation returned a non-tensor")
        return result


def unit_gradient(ttnn, torch, activation):
    """Allocate transport input only; never compose a derivative in the harness."""
    return ttnn.from_torch(
        torch.ones(tuple(activation.shape), dtype=torch.bfloat16),
        device=activation.device(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _python_parameters(source, operation):
    """Resolve the actual registration helper's literal nb::arg names.

    Dynamic helper names are bound from its template invocation. Unknown
    overload/reordering shapes fail closed, not guessed from C++ arg spelling.
    """
    # Documentation raw strings may contain apostrophes and unmatched braces.
    masked = re.sub(r'R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"', lambda m: re.sub(r"[^\n]", " ", m[0]), source)
    code = _code(masked)
    calls = list(re.finditer(r'\b(bind_\w+)<"' + re.escape(operation) + r'"[^>]*>\s*\(', source))
    if len(calls) != 1:
        raise ValueError("public Python registration must be unique")
    call = calls[0]
    opening = source.index("(", call.start())
    arguments = _signature_parameters(source[opening + 1 : _balanced_end(code, opening, "(", ")") - 1])
    definition = re.search(r"\bvoid\s+" + call[1] + r"\s*\(", source)
    if not definition:
        raise ValueError("public Python helper definition is absent")
    opening = source.index("(", definition.start())
    end = _balanced_end(code, opening, "(", ")")
    formals = _signature_parameters(source[opening + 1 : end - 1])
    variables = {}
    for formal, value in zip(formals, arguments):
        name = re.search(r"([A-Za-z_]\w*)\s*(?:=.*)?$", formal)
        if name and re.fullmatch(r'"[A-Za-z_]\w*"', value):
            variables[name[1]] = value[1:-1]
    start = source.index("{", end)
    body = source[start : _balanced_end(code, start, "{", "}")]
    # Require the registered callable to be forwarded directly.
    if not re.search(
        r"bind_function<\w+>\s*\(\s*mod,\s*doc.c_str\(\),\s*(?:ttnn::overload_t\s*\{\s*)?(?:func|Func)\s*,", body
    ):
        raise ValueError("public Python callable is not a direct binding")
    names = []
    for match in re.finditer(r'nb::arg\(\s*(?:"(\w+)"|(\w+)\.c_str\(\))\s*\)', body):
        name = match[1] or variables.get(match[2])
        if name is None or name in names:
            raise ValueError("ambiguous public Python argument binding")
        names.append(name)
    return names


def public_invocation(root: Path, operation: str, program) -> PublicInvocation:
    """Derive invocation from the same verified bindings used by packaging."""
    from ttpoly.ttmetal_unary_factory import public_binding

    io = program.execution_abi.io_contract
    resources = program.execution_abi.resources
    if not io.fuse_grad:
        try:
            binding = public_binding(root, operation, program)
        except ValueError:
            pass
        else:
            return PublicInvocation(
                binding.public_operation, ("activation_input",), binding.public_args, binding.public_kwargs
            )
    declared = resources.public_scalar_bindings
    fixed = dict(resources.fixed_scalar_bindings if declared is None else declared)
    if io.fuse_grad:
        from ttpoly.ttmetal_backward_binding import inspect_backward_binding
        from ttpoly.ttmetal_backward_factory import _public_scalar_guards

        binding = inspect_backward_binding(root, operation, io)
        _public_scalar_guards(root, SimpleNamespace(binding=binding, resources=resources))
        symbol = binding.public_symbol
        parameters = binding.parameters
        roles = tuple(p.role for p in parameters if p.role in io.tensor_input_roles)
        registration = root / "ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward_nanobind.cpp"
        scalar_values = list(fixed.values())
        scalar_parameters = [p for p in parameters if p.role == "scalar_or_mode"]
        if set(fixed) == {p.name for p in scalar_parameters}:
            scalar_values = [fixed[p.name] for p in scalar_parameters]
        else:
            # The factory above has already verified this exact enum codec.
            enum_type = scalar_parameters[0].cpp_type.split("::")[-1]
            member = "ACCURATE" if fixed["approximate"] == "none" else "TANH"
            source = (root / "ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp").read_text()
            enum = re.search(r"nb::enum_<" + enum_type + r'>\(mod,\s*"(\w+)"\)(.*?);', source, re.S)
            value = (
                re.search(r'\.value\(\s*"(\w+)",\s*' + enum_type + r"::" + member + r"\s*,", enum[2]) if enum else None
            )
            if not value:
                raise ValueError("selected enum has no actual Python member")
            scalar_values = [{"enum": enum[1] + "." + value[1]}]
    else:
        from ttpoly.ttmetal_composite_factory import _binding, _integer_binding, PUBLIC_SOURCE

        binding = _binding(root, program.public_operation_name)
        symbol, parameters = binding.public_symbol, binding.parameters
        roles = ("activation_input",)
        body = (root / PUBLIC_SOURCE).read_text()[binding.body_start : binding.body_end]
        _integer_binding(SimpleNamespace(binding=binding, resources=resources), body)
        scalar_values = []
        for parameter in parameters:
            if parameter.role != "scalar_or_mode":
                continue
            keys = (
                [parameter.name]
                if parameter.name in fixed
                else [
                    key
                    for key in fixed
                    if re.search(
                        r"\bfloat\s+"
                        + re.escape(key)
                        + r"\s*=\s*static_cast<float>\(\s*"
                        + re.escape(parameter.name)
                        + r"\s*\)\s*;",
                        _code(body),
                    )
                ]
            )
            scalar_values.append(fixed[keys[0]])
        registration = root / "ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp"
    names = _python_parameters(registration.read_text(), symbol)
    if len(names) != len(parameters) or "memory_config" not in names:
        raise ValueError("Python and C++ public parameter counts differ")
    scalar_names = [name for name, parameter in zip(names, parameters) if parameter.role == "scalar_or_mode"]
    if len(scalar_names) != len(scalar_values):
        raise ValueError("public scalar arguments are incomplete")
    return PublicInvocation(
        symbol,
        roles,
        public_kwargs=tuple(zip(scalar_names, scalar_values)),
        single_result_container=io.fuse_grad,
        explicit_dram_output=True,
    )
