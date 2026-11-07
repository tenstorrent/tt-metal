# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re
import ttnn

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from sphinx.ext.autodoc import FunctionDocumenter, DataDocumenter

import ttnn.decorators


def _public_type_names(signature: str) -> str:
    """Use exported TTNN type names instead of nanobind's implementation paths."""

    def public_name(match):
        name = match.group(1)
        return f"ttnn.{name}" if hasattr(ttnn, name) else match.group(0)

    return re.sub(r"\bttnn\._ttnn\.(?:\w+\.)*(\w+)", public_name, signature)


def _binding_signature(operation) -> str | None:
    """Read the callable's signature independently of its prose docstring.

    C++ operations wrap callable nanobind instances, whose signatures live on
    the type's __call__. Other operations wrap nanobind or Python functions.
    Nanobind stores each overload and its default values separately.
    """
    function = operation.function
    signatures = getattr(function, "__nb_signature__", None)
    is_method = False
    if not signatures:
        signatures = getattr(getattr(type(function), "__call__", None), "__nb_signature__", None)
        is_method = bool(signatures)

    if not signatures:
        try:
            return _public_type_names(str(inspect.signature(function)))
        except (TypeError, ValueError):
            return None

    result = []
    for signature, _, defaults in signatures:
        signature = signature[signature.index("(") :]
        if is_method:
            signature = re.sub(r"^\(self(?:,\s*|(?=\)))", "(", signature)
            signature = re.sub(r"^\(/(?:,\s*|(?=\)))", "(", signature)
        signature = _public_type_names(signature)

        def substitute_default(match):
            value = defaults[int(match.group(2))]
            if match.group(1):  # nb::arg::sig overrides the displayed expression.
                return value
            if isinstance(value, Enum):
                return _public_type_names(f"{type(value).__module__}.{type(value).__qualname__}.{value.name}")
            return repr(value)

        # Substitute in a single pass: a string default may itself contain \\0,
        # and index 1 must not consume the prefix of index 10.
        signature = re.sub(r"\\(=?)(\d+)", substitute_default, signature)
        result.append(signature)
    return "\n".join(result)


@dataclass
class Param:
    name: str
    type: str = ""
    optional: bool = False
    default: str | None = None

    param_pattern: ClassVar[re.Pattern] = re.compile(
        r"""
        ^[ ]{4}                  # Match exactly 4 spaces at the start
        (\w+)                    # Capture the parameter name
        \s*                      # Optional whitespace
        (?:\(                    # Start of optional type group (non-capturing)
            (                    # Capture group for the type
                [^)]+            # Capture everything inside parentheses
            )
        \))?                     # End of optional type group
        :                        # Colon after the parameter name (and optional type)
        \s*(.*)$                 # Capture the rest of the line as description
        """,
        re.VERBOSE,
    )

    def to_string(self) -> str:
        """Return a string representation of the parameter."""
        ret = f"{self.name}{f': {self.type}' if self.type else ''}"
        # Only add "| None" if optional but has no default value
        # If there's a default value, it already indicates the parameter is optional
        if self.type and self.optional and not self.default:
            ret += " | None"

        return f"{ret} = {self.default}" if self.default else ret

    @staticmethod
    def from_param_line(line) -> "Param | None":
        match = Param.param_pattern.match(line)
        if not match:
            return None
        name = match.group(1)
        param_type = match.group(2) or ""  # Use an empty string if type is not specified
        description = match.group(3)

        # Check if the parameter is optional and remove ", optional" from the end of the type if present
        param_type, optional = re.subn(r",\s*optional$", "", param_type.strip())

        # Parse the default value from the description
        default = None
        if "Defaults to" in description:
            # Match the default value - handles numbers (including floats), strings in backticks/quotes, None, etc.
            # First try to match values enclosed in backticks or quotes
            default_match = re.search(r"Defaults to [`']([^`']+)[`']", description)
            if not default_match:
                # If no backticks/quotes, match until comma, period followed by space, or closing paren
                default_match = re.search(r"Defaults to ([^,\)]+?)(?:\.\s|,|\)|\.$|$)", description)
            if default_match:
                default = default_match.group(1).strip()

        return Param(name=name, type=param_type, optional=bool(optional), default=default)


@dataclass
class Return:
    type: str
    optional: bool = False

    def to_string(self) -> str:
        """Return a string representation of the return type."""
        return f"{self.type} | None" if self.optional else self.type

    @staticmethod
    def from_return_line(line: str) -> "Return | None":
        type = line.split(":", 1)[0].strip()
        if not type:
            return None

        # Check if the return type is optional and remove ", optional" from the end of the type if present
        type, optional = re.subn(r",\s*optional$", "", type.strip())

        return Return(type=type, optional=bool(optional))


@dataclass
class DocstringParser:
    docstring: str
    sections: dict[str, list[str]] = field(default_factory=dict, init=False)
    section_pattern: ClassVar[re.Pattern] = re.compile(r"^(\w[\w ]*):\s*$")

    args: list[Param] = field(default_factory=list, init=False)
    kwargs: list[Param] = field(default_factory=list, init=False)
    returns: Return | None = None

    def split_sections(self):
        current_section = None
        current_content = []

        for line in self.docstring.split("\n"):
            match = self.section_pattern.match(line)
            if match:
                if current_section:
                    self.sections[current_section] = current_content
                current_section = match.group(1).strip().lower()
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            self.sections[current_section] = current_content

    def parse_args_section(self, content: list[str]) -> list[Param]:
        args = []
        for line in content:
            param = Param.from_param_line(line)
            if param:
                args.append(param)
        return args

    def parse_returns_section(self, content) -> Return | None:
        return_type = None
        for line in content:
            return_type = Return.from_return_line(line)
            if return_type:
                break
        return return_type

    def parse(self):
        self.split_sections()

        self.args = self.parse_args_section(self.sections.get("args", []))
        self.kwargs = self.parse_args_section(self.sections.get("keyword args", []))
        self.returns = self.parse_returns_section(self.sections.get("returns", ""))

    def construct_signature(self):
        signature = f"({', '.join(map(Param.to_string, self.args))}"
        if self.kwargs:
            signature += f", *, {', '.join(map(Param.to_string, self.kwargs))}"
        signature += f") -> {self.returns.to_string() if self.returns else 'None'}"

        return signature


class FastOperationDocumenter(FunctionDocumenter):
    objtype = "fastoperation"
    directivetype = "function"
    priority = FunctionDocumenter.priority + 10

    def format_signature(self, **kwargs: Any) -> str:
        signature = _binding_signature(self.object)
        if signature is not None:
            return signature

        docstrings = self.get_doc()
        if len(docstrings) == 0 or "Overloaded function" in self.object.__doc__:
            return "(*args, **kwargs)"

        if len(docstrings) > 1:
            raise ValueError("Multiple docstrings found for FastOperation")

        lines = inspect.cleandoc(self.object.__doc__)
        parser = DocstringParser(lines)
        parser.parse()

        return parser.construct_signature()

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, ttnn.decorators.FastOperation)


class OperationDocumenter(FunctionDocumenter):
    objtype = "operation"
    directivetype = "function"
    priority = FunctionDocumenter.priority + 10

    def format_signature(self, **kwargs: Any) -> str:
        signature = _binding_signature(self.object)
        if signature is not None:
            return signature

        docstrings = self.get_doc()
        if len(docstrings) == 0 or "Overloaded function" in self.object.__doc__:
            return "(*args, **kwargs)"

        if len(docstrings) > 1:
            raise ValueError("Multiple docstrings found for Operation")

        lines = inspect.cleandoc(self.object.__doc__)
        parser = DocstringParser(lines)
        parser.parse()

        return parser.construct_signature()

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, ttnn.decorators.Operation)


class OperationDataDocumenter(DataDocumenter):
    """Custom documenter for Operation and FastOperation objects when used with autodata."""

    objtype = "operationdata"
    directivetype = "function"  # Use function directive to avoid "name =" format
    priority = DataDocumenter.priority + 10

    def should_suppress_value_header(self) -> bool:
        # Operations use py:function, which does not accept DataDocumenter's
        # :value: option containing the operation object's repr.
        return True

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, (ttnn.decorators.Operation, ttnn.decorators.FastOperation))

    def format_signature(self, **kwargs: Any) -> str:
        """Format signature for Operation objects."""
        signature = _binding_signature(self.object)
        if signature is not None:
            return signature

        # Try to parse the docstring to extract signature
        try:
            if self.object.__doc__:
                lines = inspect.cleandoc(self.object.__doc__)
                parser = DocstringParser(lines)
                parser.parse()
                return parser.construct_signature()
        except Exception:
            pass

        # Fallback to generic signature
        return "(*args, **kwargs)"


def setup(app):
    app.add_autodocumenter(FastOperationDocumenter)
    app.add_autodocumenter(OperationDocumenter)
    app.add_autodocumenter(OperationDataDocumenter)
