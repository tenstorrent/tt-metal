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
        # Default to ttnn.Tensor for TTNN operations since most return tensors
        return_type = self.returns.to_string() if self.returns else "ttnn.Tensor"
        signature += f") -> {return_type}"

        return signature


class FastOperationDocumenter(FunctionDocumenter):
    objtype = "fastoperation"
    directivetype = "function"
    priority = FunctionDocumenter.priority + 10

    def get_doc(self):
        """Override get_doc to transform bullet list format to Sphinx-compatible format."""
        docstrings = super().get_doc()
        if not docstrings:
            return docstrings

        # Sphinx-recognized field names that should be transformed
        sphinx_fields = {"Parameters", "Args", "Keyword Arguments", "Keyword Args", "Returns", "Raises", "Yields"}

        # Transform each docstring
        transformed = []
        for docstring_lines in docstrings:
            new_lines = []
            current_section = None

            for line in docstring_lines:
                # Check if this is a section header
                section_match = re.match(r"^\s*(\w[\w\s]*):\s*$", line)
                if section_match:
                    current_section = section_match.group(1).strip()
                    # For Returns sections, check if they contain meaningful information
                    if current_section == "Returns":
                        # Collect the Returns section content to analyze
                        returns_content = []
                        content_start_idx = docstring_lines.index(line) + 1
                        content_idx = content_start_idx

                        # Find the end of this Returns section
                        while content_idx < len(docstring_lines):
                            next_line = docstring_lines[content_idx]
                            if re.match(r"^\s*(\w[\w\s]*):\s*$", next_line):  # Next section header
                                break
                            if next_line.strip():  # Non-empty line
                                returns_content.append(next_line.strip())
                            content_idx += 1

                        # Check if this Returns section has meaningful content
                        returns_text = " ".join(returns_content)

                        # Only skip truly generic Returns sections that provide no useful information
                        # Generic patterns: exact matches with minimal descriptive content
                        # Keep List and Tuple returns as they provide useful type information
                        is_truly_generic = returns_text.strip() in [
                            "ttnn.Tensor: the output tensor",
                            "ttnn.Tensor: the output tensor.",
                            "* **ttnn.Tensor**: the output tensor",
                            "* **ttnn.Tensor**: the output tensor.",
                        ]

                        if is_truly_generic:
                            # Skip truly generic Returns sections - don't add anything
                            pass
                        else:
                            # Keep ALL other Returns sections (any detailed description is meaningful)
                            new_lines.append(line)  # Keep the "Returns:" header
                            # Add the content lines as-is
                            for content_line in returns_content:
                                if content_line.strip():
                                    new_lines.append(f"    {content_line}")
                            continue
                    new_lines.append(line)
                    continue

                # Skip content from generic Returns sections (already handled above)
                if current_section == "Returns":
                    continue

                # Only transform if we're in a Sphinx-recognized field
                if current_section in sphinx_fields:
                    # Check for parameter format: "* **name** (type): description"
                    match_param = re.match(r"^(\s*)\*\s+\*\*(\w+)\*\*\s+\(([^)]+)\):\s*(.*)$", line)
                    if match_param:
                        indent = match_param.group(1)
                        param_name = match_param.group(2)
                        param_type = match_param.group(3)
                        description = match_param.group(4)
                        # Remove bullet and bold for Sphinx field lists: "name (type): description"
                        new_lines.append(f"{indent}{param_name} ({param_type}): {description}")
                        continue

                    # Check for return type format: "* **Type**: description"
                    match_return = re.match(r"^(\s*)\*\s+\*\*([^*]+?)\*\*:\s*(.*)$", line)
                    if match_return:
                        indent = match_return.group(1)
                        return_type = match_return.group(2).strip()
                        description = match_return.group(3)
                        # Remove bullet and bold for Sphinx field lists: "Type: description"
                        new_lines.append(f"{indent}{return_type}: {description}")
                        continue

                    # Check for plain return type format: "Type: description"
                    match_plain_return = re.match(r"^(\s*)([^:]+):\s*(.*)$", line)
                    if match_plain_return and current_section == "Returns":
                        indent = match_plain_return.group(1)
                        return_type = match_plain_return.group(2).strip()
                        description = match_plain_return.group(3)
                        # Format for Sphinx field lists: "Type: description"
                        new_lines.append(f"{indent}{return_type}: {description}")
                        continue

                # If not in a Sphinx field or no match, keep original
                new_lines.append(line)

            transformed.append(new_lines)

        return transformed

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

    def get_doc(self):
        """Override get_doc to transform bullet list format to Sphinx-compatible format."""
        docstrings = super().get_doc()
        if not docstrings:
            return docstrings

        # Sphinx-recognized field names that should be transformed
        sphinx_fields = {"Parameters", "Args", "Keyword Arguments", "Keyword Args", "Returns", "Raises", "Yields"}

        # Transform each docstring
        transformed = []
        for docstring_lines in docstrings:
            new_lines = []
            current_section = None

            for line in docstring_lines:
                # Check if this is a section header
                section_match = re.match(r"^\s*(\w[\w\s]*):\s*$", line)
                if section_match:
                    current_section = section_match.group(1).strip()
                    # For Returns sections, check if they contain meaningful information
                    if current_section == "Returns":
                        # Collect the Returns section content to analyze
                        returns_content = []
                        content_start_idx = docstring_lines.index(line) + 1
                        content_idx = content_start_idx

                        # Find the end of this Returns section
                        while content_idx < len(docstring_lines):
                            next_line = docstring_lines[content_idx]
                            if re.match(r"^\s*(\w[\w\s]*):\s*$", next_line):  # Next section header
                                break
                            if next_line.strip():  # Non-empty line
                                returns_content.append(next_line.strip())
                            content_idx += 1

                        # Check if this Returns section has meaningful content
                        returns_text = " ".join(returns_content)

                        # Only skip truly generic Returns sections that provide no useful information
                        # Generic patterns: exact matches with minimal descriptive content
                        # Keep List and Tuple returns as they provide useful type information
                        is_truly_generic = returns_text.strip() in [
                            "ttnn.Tensor: the output tensor",
                            "ttnn.Tensor: the output tensor.",
                            "* **ttnn.Tensor**: the output tensor",
                            "* **ttnn.Tensor**: the output tensor.",
                        ]

                        if is_truly_generic:
                            # Skip truly generic Returns sections - don't add anything
                            pass
                        else:
                            # Keep ALL other Returns sections (any detailed description is meaningful)
                            new_lines.append(line)  # Keep the "Returns:" header
                            # Add the content lines as-is
                            for content_line in returns_content:
                                if content_line.strip():
                                    new_lines.append(f"    {content_line}")
                            continue
                    new_lines.append(line)
                    continue

                # Skip content from generic Returns sections (already handled above)
                if current_section == "Returns":
                    continue

                # Only transform if we're in a Sphinx-recognized field
                if current_section in sphinx_fields:
                    # Check for parameter format: "* **name** (type): description"
                    match_param = re.match(r"^(\s*)\*\s+\*\*(\w+)\*\*\s+\(([^)]+)\):\s*(.*)$", line)
                    if match_param:
                        indent = match_param.group(1)
                        param_name = match_param.group(2)
                        param_type = match_param.group(3)
                        description = match_param.group(4)
                        # Remove bullet and bold for Sphinx field lists: "name (type): description"
                        new_lines.append(f"{indent}{param_name} ({param_type}): {description}")
                        continue

                    # Check for return type format: "* **Type**: description"
                    match_return = re.match(r"^(\s*)\*\s+\*\*([^*]+?)\*\*:\s*(.*)$", line)
                    if match_return:
                        indent = match_return.group(1)
                        return_type = match_return.group(2).strip()
                        description = match_return.group(3)
                        # Remove bullet and bold for Sphinx field lists: "Type: description"
                        new_lines.append(f"{indent}{return_type}: {description}")
                        continue

                    # Check for plain return type format: "Type: description"
                    match_plain_return = re.match(r"^(\s*)([^:]+):\s*(.*)$", line)
                    if match_plain_return and current_section == "Returns":
                        indent = match_plain_return.group(1)
                        return_type = match_plain_return.group(2).strip()
                        description = match_plain_return.group(3)
                        # Format for Sphinx field lists: "Type: description"
                        new_lines.append(f"{indent}{return_type}: {description}")
                        continue

                # If not in a Sphinx field or no match, keep original
                new_lines.append(line)

            transformed.append(new_lines)

        return transformed


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
