# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re
import ttnn

from dataclasses import dataclass, field
from typing import Any, ClassVar

from sphinx.ext.autodoc import FunctionDocumenter, DataDocumenter

import ttnn.decorators


@dataclass
class Param:
    name: str
    type: str = ""
    optional: bool = False
    default: str | None = None

    param_pattern: ClassVar[re.Pattern] = re.compile(
        r"""
        ^\s*                     # Match any amount of leading whitespace (including zero)
        (?:\*\s+)?               # Optional bullet point (asterisk) followed by whitespace
        (?:\*\*)?                # Optional bold opening (**) for rst formatting
        (\w+)                    # Capture the parameter name
        (?:\*\*)?                # Optional bold closing (**) for rst formatting
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
        # Remove bullet point and bold formatting: "* **type**: description" -> "type: description"
        line = re.sub(r"^\s*\*\s+\*\*(.+?)\*\*\s*:", r"\1:", line)
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
    section_pattern: ClassVar[re.Pattern] = re.compile(r"^\s*(\w[\w ]*):\s*$")  # Allow leading whitespace

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
            # Skip empty lines
            if not line.strip():
                continue
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
        docstrings = self.get_doc()
        if len(docstrings) == 0:
            return "(*args, **kwargs)"

        # get_doc() returns list of lists, join the first one
        doc_to_parse = "\n".join(docstrings[0])

        if not doc_to_parse:
            return "(*args, **kwargs)"

        try:
            lines = inspect.cleandoc(doc_to_parse)
            parser = DocstringParser(lines)
            parser.parse()

            # If we have args or returns, construct the signature
            if parser.args or parser.kwargs or parser.returns:
                return parser.construct_signature()
        except Exception as e:
            # If parsing fails, fall back to generic signature
            pass

        return "(*args, **kwargs)"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, ttnn.decorators.FastOperation)


class OperationDocumenter(FunctionDocumenter):
    objtype = "operation"
    directivetype = "function"
    priority = FunctionDocumenter.priority + 10

    def format_signature(self, **kwargs: Any) -> str:
        docstrings = self.get_doc()
        if len(docstrings) == 0:
            return "(*args, **kwargs)"

        # get_doc() returns list of lists, join the first one
        doc_to_parse = "\n".join(docstrings[0])

        if not doc_to_parse:
            return "(*args, **kwargs)"

        try:
            lines = inspect.cleandoc(doc_to_parse)
            parser = DocstringParser(lines)
            parser.parse()

            # If we have args or returns, construct the signature
            if parser.args or parser.kwargs or parser.returns:
                return parser.construct_signature()
        except Exception as e:
            # If parsing fails, fall back to generic signature
            pass

        return "(*args, **kwargs)"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, (ttnn.decorators.Operation, ttnn.decorators.FastOperation))


class OperationDataDocumenter(DataDocumenter):
    """Custom documenter for Operation and FastOperation objects when used with autodata."""

    objtype = "operationdata"
    directivetype = "function"  # Use function directive to avoid "name =" format
    priority = DataDocumenter.priority + 10

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, (ttnn.decorators.Operation, ttnn.decorators.FastOperation))

    def format_signature(self, **kwargs: Any) -> str:
        """Format signature for Operation objects."""
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
