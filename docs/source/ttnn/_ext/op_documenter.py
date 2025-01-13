# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re
import ttnn

from dataclasses import dataclass, field
from typing import Any, ClassVar

from sphinx.ext.autodoc import FunctionDocumenter

import ttnn.decorators


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
        if self.type and self.optional:
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
            default_match = re.search(r"Defaults to [`']?([^`']+)[`']?", description)
            if default_match:
                default = default_match.group(1)

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
        docstrings = self.get_doc()
        if len(docstrings) == 0 or "Overloaded function" in self.object.__doc__:
            return "(*args, ***kwargs)"

        if len(docstrings) > 1:
            raise ValueError("Multiple docstrings found for FastOperation")

        lines = inspect.cleandoc(self.object.__doc__)
        parser = DocstringParser(lines)
        parser.parse()

        return parser.construct_signature()

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, ttnn.decorators.FastOperation)


def setup(app):
    app.add_autodocumenter(FastOperationDocumenter)
