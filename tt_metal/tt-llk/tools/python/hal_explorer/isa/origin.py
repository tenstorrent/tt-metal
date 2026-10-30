# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Which source line and which function issued an instruction.

GCC keeps the innermost location of every statement after inlining: for a HAL
call that is the HAL line that issued the word. The enclosing function is found
by walking back from that line and column to the brace that opens the scope.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

CONTROL = re.compile(
    r"^(?:else\b|do\b|try\b|(?:else\s+)?(?:if|for|while|switch|catch)\b)"
)
SCOPE = re.compile(r"\b(?:namespace|class|struct|union|enum)\b|extern\s+\"C")
ATTRIBUTES = re.compile(r"__attribute__\s*\(\(.*?\)\)|\[\[.*?\]\]|\balignas\s*\(.*?\)")


@dataclass
class Origin:
    file: str
    line: int
    text: str
    function: Optional[str]
    in_input: bool


class SourceIndex:
    """Reads each source file once and caches the origin of every location."""

    def __init__(self, llk_root, input_name, input_text):
        self.llk_root = os.path.realpath(str(llk_root))
        self.input_name = input_name
        self.files = {input_name: input_text.split("\n")}
        self.cache = {}

    def lines(self, path):
        if path not in self.files:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    self.files[path] = f.read().split("\n")
            except OSError:
                self.files[path] = []
        return self.files[path]

    def origin(self, location):
        if location is None:
            return None
        if location not in self.cache:
            path, line, column = location
            lines = self.lines(path)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            in_input = path == self.input_name
            function = None if in_input else enclosing_function(lines, line, column)
            self.cache[location] = Origin(
                self.short(path), line, text, function, in_input
            )
        return self.cache[location]

    def short(self, path):
        if path == self.input_name:
            return path
        if "/llk_lib/" in path:
            return path.split("/llk_lib/")[-1]
        real = os.path.realpath(path)
        if real.startswith(self.llk_root + os.sep):
            return os.path.relpath(real, self.llk_root)
        return os.path.basename(path)


def code_only(line):
    """A source line with comments and string or character literals blanked."""
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
    return line.split("//", 1)[0]


def enclosing_function(lines, line, column):
    """Name of the function whose body contains (line, column), or None."""
    code = [code_only(text) for text in lines]
    row, col = line - 1, max(column - 1, 0)
    depth = 0
    while row >= 0:
        text = code[row][:col] if col is not None else code[row]
        for position in range(len(text) - 1, -1, -1):
            char = text[position]
            if char == "}":
                depth += 1
            elif char == "{":
                if depth:
                    depth -= 1
                    continue
                kind, name = classify_scope(header_before(code, row, position))
                if kind == "function":
                    return name
                if kind == "scope":
                    return None
        row, col = row - 1, None
    return None


def header_before(code, row, position, max_lines=16):
    """The declaration text that precedes the brace at (row, position)."""
    parts, text = [], code[row][:position]
    for _ in range(max_lines):
        cut = max(text.rfind(";"), text.rfind("{"), text.rfind("}"))
        if cut >= 0:
            parts.append(text[cut + 1 :])
            break
        parts.append(text)
        row -= 1
        if row < 0:
            break
        text = code[row]
        if text.lstrip().startswith("#"):
            break
    return " ".join(reversed(parts)).strip()


def classify_scope(header):
    """('function', name), ('scope', None) or ('block', None) for a brace's header."""
    header = ATTRIBUTES.sub(" ", header).strip()
    header = re.sub(r"^template\s*<.*?>\s*", "", header)
    if (
        not header
        or CONTROL.match(header)
        or header.endswith("=")
        or header.endswith("return")
    ):
        return "block", None
    if SCOPE.search(header) and "(" not in header:
        return "scope", None
    if re.search(r"\]\s*(\(.*\))?\s*(mutable|constexpr)?\s*(->.*)?$", header):
        return "block", None  # lambda
    m = re.search(
        r"((?:[A-Za-z_]\w*\s*::\s*)*(?:operator\s*(?:\(\)|[^\s(]+)|~?[A-Za-z_]\w*))\s*\(",
        header,
    )
    if not m:
        return "block", None
    name = re.sub(r"\s+", "", m.group(1))
    if name in (
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "sizeof",
        "decltype",
        "static_assert",
    ):
        return "block", None
    return "function", name
