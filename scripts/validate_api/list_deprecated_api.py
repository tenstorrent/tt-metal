#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inventory Metal API symbols marked [[deprecated]].

.github/deprecations.json is the grace-period manifest read by the weekly
deprecation reaper. It is not a dump of every [[deprecated]] attribute.
This script scans tt_metal/api (the Metal host API) and reports each marked
symbol, the removal date written in its message, and whether the manifest
names that symbol.

A symbol is overdue when today is after the date in the message. "Removed
after 2026-11-19" is still in force on that date and overdue starting the
next day. Messages that use DD-MM-YYYY are accepted when the first number
cannot be a month (28-02-2026). Ambiguous pairs such as 02-03-2026 are left
undated.

Usage:
    python3 scripts/validate_api/list_deprecated_api.py
    python3 scripts/validate_api/list_deprecated_api.py --format json
    python3 scripts/validate_api/list_deprecated_api.py --status overdue --tier stable
    python3 scripts/validate_api/list_deprecated_api.py tt_metal/hw/inc/api
    python3 scripts/validate_api/list_deprecated_api.py --self-test
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import re
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

_HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx"}
_SPECIFIERS = {
    "const",
    "constexpr",
    "consteval",
    "constinit",
    "volatile",
    "static",
    "inline",
    "virtual",
    "explicit",
    "friend",
    "extern",
    "mutable",
    "typename",
    "template",
    "class",
    "struct",
    "enum",
    "union",
    "unsigned",
    "signed",
    "long",
    "short",
    "int",
    "char",
    "bool",
    "void",
    "auto",
    "double",
    "float",
    "override",
    "final",
    "noexcept",
    "public",
    "private",
    "protected",
}
_CAST_WORDS = {
    "decltype",
    "sizeof",
    "alignof",
    "noexcept",
    "typeid",
    "static_cast",
    "dynamic_cast",
    "const_cast",
    "reinterpret_cast",
}
_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTH_RE = re.compile(
    r"\b(" + "|".join(calendar.month_name[1:]) + r")" r"\s+(\d{1,2})(?:st|nd|rd|th)?" r"\s*,?\s*" r"(20\d{2})",
    re.IGNORECASE,
)
_ISO_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_DMY_RE = re.compile(r"\b(\d{2})-(\d{2})-(20\d{2})\b")
_STATUS_ORDER = {"overdue": 0, "due": 1, "scheduled": 2, "undated": 3}


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _skip_ws_comments(text: str, i: int) -> int:
    n = len(text)
    while i < n:
        if text[i] in " \t\r\n":
            i += 1
            continue
        if text.startswith("//", i):
            newline = text.find("\n", i)
            i = n if newline < 0 else newline + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        break
    return i


def _skip_back(text: str, i: int) -> int:
    while i > 0:
        prev = text[i - 1]
        if prev in " \t\r":
            i -= 1
            continue
        if prev == "\n":
            line_end = i - 1
            line_start = text.rfind("\n", 0, line_end) + 1
            stripped = text[line_start:line_end].strip()
            if stripped.startswith(("//", "/*", "*")):
                i = line_start
                continue
            i -= 1
            continue
        if i >= 2 and text[i - 2 : i] == "*/":
            start = text.rfind("/*", 0, i - 2)
            if start >= 0:
                i = start
                continue
        break
    return i


def _read_ident(text: str, i: int) -> tuple[str, int]:
    j = i + 1
    n = len(text)
    while j < n and _is_ident_char(text[j]):
        j += 1
    return text[i:j], j


def _peek_ident(text: str, i: int) -> tuple[str, int]:
    i = _skip_ws_comments(text, i)
    if i < len(text) and _is_ident_start(text[i]):
        return _read_ident(text, i)
    return "", i


def _read_ident_back(text: str, i: int) -> tuple[str, int]:
    j = i
    while j > 0 and _is_ident_char(text[j - 1]):
        j -= 1
    if j == i or not _is_ident_start(text[j]):
        return "", i
    return text[j:i], j


def _read_cxx_string(text: str, i: int) -> tuple[str, int]:
    """Decode one ordinary string literal. i points at the opening quote."""
    i += 1
    n = len(text)
    out: list[str] = []
    while i < n:
        ch = text[i]
        if ch == '"':
            return "".join(out), i + 1
        if ch == "\\":
            if i + 1 >= n:
                break
            mapped = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}.get(text[i + 1], text[i + 1])
            out.append(mapped)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out), i


def _skip_char(text: str, i: int) -> int:
    i += 1
    n = len(text)
    if i < n and text[i] == "\\":
        i += 2
    elif i < n:
        i += 1
    if i < n and text[i] == "'":
        i += 1
    return i


def _skip_attribute(text: str, i: int) -> int:
    """Skip one [[...]] attribute. i points at the first '['."""
    n = len(text)
    depth = 0
    while i < n:
        if text[i] in " \t\r\n" or text.startswith("//", i) or text.startswith("/*", i):
            nxt = _skip_ws_comments(text, i)
            if nxt == i:
                break
            i = nxt
            continue
        if text[i] == '"':
            _, i = _read_cxx_string(text, i)
            continue
        if text[i] == "'":
            i = _skip_char(text, i)
            continue
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _skip_parens(text: str, i: int) -> int:
    """Skip a (...) group. i points at '('."""
    n = len(text)
    depth = 0
    while i < n:
        if text[i] in " \t\r\n" or text.startswith("//", i) or text.startswith("/*", i):
            nxt = _skip_ws_comments(text, i)
            if nxt == i:
                break
            i = nxt
            continue
        if text[i] == '"':
            _, i = _read_cxx_string(text, i)
            continue
        if text[i] == "'":
            i = _skip_char(text, i)
            continue
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _skip_braces(text: str, i: int) -> int:
    """Skip a {...} group, including nested braces. i points at '{'."""
    n = len(text)
    depth = 0
    while i < n:
        if text[i] in " \t\r\n" or text.startswith("//", i) or text.startswith("/*", i):
            nxt = _skip_ws_comments(text, i)
            if nxt == i:
                break
            i = nxt
            continue
        if text[i] == '"':
            _, i = _read_cxx_string(text, i)
            continue
        if text[i] == "'":
            i = _skip_char(text, i)
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _skip_template(text: str, i: int) -> int:
    """Skip a <...> template argument list. i points at '<'."""
    n = len(text)
    depth = 0
    while i < n:
        if text[i] in " \t\r\n" or text.startswith("//", i) or text.startswith("/*", i):
            nxt = _skip_ws_comments(text, i)
            if nxt == i:
                break
            i = nxt
            continue
        if text[i] == '"':
            _, i = _read_cxx_string(text, i)
            continue
        if text[i] == "'":
            i = _skip_char(text, i)
            continue
        if text.startswith("<=", i):
            i += 2
            continue
        if text[i] == "<":
            depth += 1
            i += 1
            continue
        if text[i] == ">":
            if i + 1 < n and text[i + 1] == "=":
                i += 2
                continue
            depth -= 1
            i += 1
            if depth == 0:
                return i
            continue
        i += 1
    return i


def _at_bol(text: str, i: int) -> bool:
    k = i
    while k > 0 and text[k - 1] in " \t":
        k -= 1
    return k == 0 or text[k - 1] == "\n"


def _skip_pp(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] != "\n":
        if text[i] == "\\" and i + 1 < n and text[i + 1] == "\n":
            i += 2
            continue
        i += 1
    return min(i + 1, n)


def _consume_angle(text: str, i: int, angle: int, prev_ident: str) -> tuple[int, int]:
    """Advance past one < or > token and update template-argument depth.

    operator<, operator<=, operator<<, operator<=>, operator>, operator>=, and
    comparison operators do not change template depth. A leaked angle inside one
    declaration is cleared by the caller at the next semicolon.
    """
    ch = text[i]
    if prev_ident == "operator" and ch in "<>":
        if text.startswith("<=>", i):
            return i + 3, angle
        if text.startswith(("<=", "<<", ">=", ">>")):
            return i + 2, angle
        return i + 1, angle
    if ch == "<":
        if text.startswith("<=>", i):
            return i + 3, angle
        if text.startswith("<=", i) or (text.startswith("<<", i) and angle == 0):
            return i + 2, angle
        return i + 1, angle + 1
    if i + 1 < len(text) and text[i + 1] == "=":
        return i + 2, angle
    if angle > 0:
        return i + 1, angle - 1
    if text.startswith(">>", i):
        return i + 2, angle
    return i + 1, angle


def _starts_deprecated(text: str, i: int) -> bool:
    if not text.startswith("[[", i):
        return False
    j = _skip_ws_comments(text, i + 2)
    if not text.startswith("deprecated", j):
        return False
    end = j + len("deprecated")
    return end >= len(text) or not _is_ident_char(text[end])


def _parse_deprecated(text: str, i: int) -> tuple[str, int]:
    """Return (message, index after the attribute). i points at '[['."""
    n = len(text)
    j = _skip_ws_comments(text, i + 2)
    if text.startswith("deprecated", j):
        j += len("deprecated")
    j = _skip_ws_comments(text, j)
    parts: list[str] = []
    if j < n and text[j] == "(":
        j += 1
        while j < n:
            j = _skip_ws_comments(text, j)
            if j >= n:
                break
            if text[j] == ")":
                j += 1
                break
            if text[j] == '"':
                piece, j = _read_cxx_string(text, j)
                parts.append(piece)
                continue
            break
    j = _skip_ws_comments(text, j)
    if text.startswith("]]", j):
        j += 2
    else:
        j = _skip_attribute(text, i)
    message = re.sub(r"\s+", " ", "".join(parts)).strip()
    return message, j


def _skip_attr_and_alignas(text: str, i: int) -> int:
    n = len(text)
    while True:
        i = _skip_ws_comments(text, i)
        if i >= n:
            return i
        if text.startswith("[[", i):
            i = _skip_attribute(text, i)
            continue
        ident, j = _peek_ident(text, i)
        if ident == "alignas":
            i = _skip_ws_comments(text, j)
            if i < n and text[i] == "(":
                i = _skip_parens(text, i)
            continue
        return i


def _read_qualified(text: str, i: int) -> tuple[list[str], int]:
    parts: list[str] = []
    i = _skip_ws_comments(text, i)
    if i >= len(text) or not _is_ident_start(text[i]):
        return parts, i
    while True:
        ident, i = _read_ident(text, i)
        parts.append(ident)
        i = _skip_ws_comments(text, i)
        if text.startswith("::", i):
            i = _skip_ws_comments(text, i + 2)
            if i < len(text) and _is_ident_start(text[i]):
                continue
        break
    if i < len(text) and text[i] == "<":
        i = _skip_template(text, i)
    return parts, i


def _scan_to_brace_or_semi(text: str, i: int) -> int | None:
    n = len(text)
    paren = angle = bracket = brace = 0
    prev_ident = ""
    while i < n:
        if text[i] in " \t\r\n" or text.startswith("//", i) or text.startswith("/*", i):
            nxt = _skip_ws_comments(text, i)
            if nxt == i:
                break
            i = nxt
            continue
        if text[i] == '"':
            _, i = _read_cxx_string(text, i)
            prev_ident = ""
            continue
        if text[i] == "'":
            i = _skip_char(text, i)
            prev_ident = ""
            continue
        if text.startswith("[[", i):
            i = _skip_attribute(text, i)
            continue
        ch = text[i]
        if paren == angle == bracket == brace == 0 and ch in "{;":
            return i
        if ch == "(":
            paren += 1
            prev_ident = ""
        elif ch == ")":
            paren = max(0, paren - 1)
            prev_ident = ""
        elif ch == "[":
            bracket += 1
            prev_ident = ""
        elif ch == "]":
            bracket = max(0, bracket - 1)
            prev_ident = ""
        elif ch == "{":
            brace += 1
            prev_ident = ""
        elif ch == "}":
            brace = max(0, brace - 1)
            prev_ident = ""
        elif ch in "<>":
            i, angle = _consume_angle(text, i, angle, prev_ident)
            prev_ident = ""
            continue
        elif _is_ident_start(ch):
            prev_ident, i = _read_ident(text, i)
            continue
        else:
            prev_ident = ""
        i += 1
    return None


def _try_open_scope(text: str, i: int, prev_ident: str) -> tuple[int, list[tuple[str, str]]] | None:
    """If i is a namespace/class/struct/enum definition, return (index after '{', scopes)."""
    ident, j = _read_ident(text, i)
    if ident not in {"namespace", "class", "struct", "union", "enum"}:
        return None
    if prev_ident in {"using", "typename"}:
        return None
    if ident == "namespace":
        return _open_namespace(text, j)
    return _open_type(text, i)


def _open_namespace(text: str, i: int) -> tuple[int, list[tuple[str, str]]] | None:
    i = _skip_ws_comments(text, i)
    if i < len(text) and text[i] == "{":
        return i + 1, [("namespace", "")]
    parts, i = _read_qualified(text, i)
    i = _skip_attr_and_alignas(text, i)
    if i < len(text) and text[i] == "{":
        # One brace closes the whole nested-name namespace, so this is one scope entry.
        # namespace tt::tt_metal { } must not leave "tt" on the stack after the single }.
        joined = "::".join(parts)
        return i + 1, [("namespace", joined)]
    return None


def _open_type(text: str, i: int) -> tuple[int, list[tuple[str, str]]] | None:
    ident, j = _read_ident(text, i)
    j = _skip_ws_comments(text, j)
    if ident == "enum":
        nxt, k = _peek_ident(text, j)
        if nxt in {"class", "struct"}:
            j = k
    j = _skip_attr_and_alignas(text, j)
    parts, j = _read_qualified(text, j)
    j = _skip_ws_comments(text, j)
    nxt, k = _peek_ident(text, j)
    if nxt == "final":
        j = _skip_ws_comments(text, k)
    j = _skip_attr_and_alignas(text, j)
    found = _scan_to_brace_or_semi(text, j)
    if found is None or text[found] != "{":
        return None
    name = parts[-1] if parts else ""
    return found + 1, [("class", name)]


def _try_extern_block(text: str, i: int) -> int | None:
    """If this is extern \"C\" {, return the index after '{'."""
    i = _skip_ws_comments(text, i)
    if i >= len(text) or text[i] != '"':
        return None
    _, i = _read_cxx_string(text, i)
    i = _skip_ws_comments(text, i)
    if i < len(text) and text[i] == "{":
        return i + 1
    return None


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _scan_decl_end(text: str, i: int) -> int:
    n = len(text)
    paren = angle = bracket = brace = 0
    prev_ident = ""
    while i < n:
        if text[i] in " \t\r\n" or text.startswith("//", i) or text.startswith("/*", i):
            nxt = _skip_ws_comments(text, i)
            if nxt == i or nxt >= n:
                return min(nxt, n)
            i = nxt
            continue
        ch = text[i]
        if ch == '"':
            _, i = _read_cxx_string(text, i)
            prev_ident = ""
            continue
        if ch == "'":
            i = _skip_char(text, i)
            prev_ident = ""
            continue
        if paren == angle == bracket == brace == 0 and ch in ";{":
            return i
        if ch == "(":
            paren += 1
            prev_ident = ""
        elif ch == ")":
            paren = max(0, paren - 1)
            prev_ident = ""
        elif ch == "[":
            bracket += 1
            prev_ident = ""
        elif ch == "]":
            bracket = max(0, bracket - 1)
            prev_ident = ""
        elif ch == "{":
            brace += 1
            prev_ident = ""
        elif ch == "}":
            brace = max(0, brace - 1)
            prev_ident = ""
        elif ch in "<>":
            i, angle = _consume_angle(text, i, angle, prev_ident)
            prev_ident = ""
            continue
        elif _is_ident_start(ch):
            prev_ident, i = _read_ident(text, i)
            continue
        else:
            prev_ident = ""
        i += 1
    return i


def _name_before_paren(text: str, paren_i: int) -> str | None:
    i = _skip_back(text, paren_i)
    if i <= 0:
        return None
    if _is_ident_char(text[i - 1]):
        ident, start = _read_ident_back(text, i)
        if not ident or ident in _CAST_WORDS:
            return None
        k = _skip_back(text, start)
        prev = ""
        if k > 0 and _is_ident_char(text[k - 1]):
            prev, _ = _read_ident_back(text, k)
        if prev == "operator":
            return "operator" + ident
        return ident
    if text[i - 1] in "=+-*/%^&|~!<>":
        end = i
        while i > 0 and text[i - 1] in "=+-*/%^&|~!<>[]":
            i -= 1
        token = text[i:end]
        k = _skip_back(text, i)
        prev = ""
        if k > 0 and _is_ident_char(text[k - 1]):
            prev, _ = _read_ident_back(text, k)
        if prev == "operator":
            return "operator" + token
    return None


def _classify(decl: str) -> tuple[str, str]:
    n = len(decl)
    i = _skip_ws_comments(decl, 0)
    ident, j = _peek_ident(decl, i)
    if ident == "template":
        i = _skip_ws_comments(decl, j)
        if i < n and decl[i] == "<":
            i = _skip_template(decl, i)
    paren = angle = bracket = brace = 0
    prev_ident = ""
    last_ident = ""
    while i < n:
        if decl[i] in " \t\r\n" or decl.startswith("//", i) or decl.startswith("/*", i):
            nxt = _skip_ws_comments(decl, i)
            if nxt == i or nxt >= n:
                break
            i = nxt
            continue
        ch = decl[i]
        if paren == angle == bracket == brace == 0 and ch in "=;{":
            break
        if paren == angle == bracket == brace == 0 and ch == "(":
            name = _name_before_paren(decl, i)
            if name:
                return "function", name
            paren += 1
            prev_ident = ""
            i += 1
            continue
        if ch == '"':
            _, i = _read_cxx_string(decl, i)
            prev_ident = ""
            continue
        if ch == "'":
            i = _skip_char(decl, i)
            prev_ident = ""
            continue
        if decl.startswith("[[", i):
            i = _skip_attribute(decl, i)
            continue
        if ch == "(":
            paren += 1
            prev_ident = ""
        elif ch == ")":
            paren = max(0, paren - 1)
            prev_ident = ""
        elif ch == "[":
            bracket += 1
            prev_ident = ""
        elif ch == "]":
            bracket = max(0, bracket - 1)
            prev_ident = ""
        elif ch == "{":
            brace += 1
            prev_ident = ""
        elif ch == "}":
            brace = max(0, brace - 1)
            prev_ident = ""
        elif ch in "<>":
            i, angle = _consume_angle(decl, i, angle, prev_ident)
            prev_ident = ""
            continue
        elif _is_ident_start(ch):
            ident, i = _read_ident(decl, i)
            prev_ident = ident
            if paren == angle == bracket == brace == 0 and ident not in _SPECIFIERS:
                last_ident = ident
            continue
        else:
            prev_ident = ""
        i += 1
    if last_ident:
        return "field", last_ident
    return "unknown", ""


def _declaration_text(text: str, attr_end: int) -> str | None:
    """Declaration following a prefix attribute. None when the attribute is postfix."""
    i = _skip_ws_comments(text, attr_end)
    while text.startswith("[[", i):
        i = _skip_attribute(text, i)
        i = _skip_ws_comments(text, i)
    if i < len(text) and text[i] in "(=;":
        return None
    stop = _scan_decl_end(text, i)
    return text[i:stop]


def _using_alias_name(text: str, attr_start: int) -> str:
    i = _skip_back(text, attr_start)
    name, i = _read_ident_back(text, i)
    if not name:
        return ""
    while True:
        j = _skip_back(text, i)
        if j >= 2 and text[j - 2 : j] == "::":
            j = _skip_back(text, j - 2)
            word, j = _read_ident_back(text, j)
            if not word:
                break
            i = j
            continue
        break
    j = _skip_back(text, i)
    word, _ = _read_ident_back(text, j)
    if word == "using":
        return name
    return ""


def _postfix_name(text: str, attr_start: int) -> str:
    i = _skip_back(text, attr_start)
    name, _ = _read_ident_back(text, i)
    return name


def _qualify(scope: list[tuple[str, str]], name: str) -> str:
    parts: list[str] = []
    for kind, part in scope:
        if part and kind in {"namespace", "class"}:
            parts.extend(piece for piece in part.split("::") if piece)
    if name:
        parts.append(name)
    return "::".join(parts)


@dataclass
class _RawHit:
    line: int
    qualified_name: str
    name: str
    kind: str
    signature: str
    message: str


def scan_text(text: str) -> list[_RawHit]:
    """Return [[deprecated]] symbols in one translation unit."""
    hits: list[_RawHit] = []
    n = len(text)
    i = 0
    scope: list[tuple[str, str]] = []
    paren = angle = bracket = 0
    prev_ident = ""

    while i < n:
        if text[i] in " \t\r\n" or text.startswith("//", i) or text.startswith("/*", i):
            nxt = _skip_ws_comments(text, i)
            if nxt == i:
                break
            i = nxt
            continue
        if text[i] == "#" and _at_bol(text, i):
            i = _skip_pp(text, i)
            prev_ident = ""
            continue
        if text[i] == '"':
            _, i = _read_cxx_string(text, i)
            prev_ident = ""
            continue
        if text[i] == "'":
            i = _skip_char(text, i)
            prev_ident = ""
            continue
        if _starts_deprecated(text, i):
            message, end = _parse_deprecated(text, i)
            hits.append(_hit_at(text, i, end, message, scope))
            i = end
            prev_ident = ""
            continue
        if text.startswith("[[", i):
            i = _skip_attribute(text, i)
            continue

        ch = text[i]
        depths_clear = paren == angle == bracket == 0
        if _is_ident_start(ch) and depths_clear:
            ident, j = _read_ident(text, i)
            if ident == "extern":
                opened = _try_extern_block(text, j)
                if opened is not None:
                    scope.append(("block", ""))
                    prev_ident = ""
                    i = opened
                    continue
            opened_scope = _try_open_scope(text, i, prev_ident)
            if opened_scope is not None:
                new_i, entries = opened_scope
                scope.extend(entries)
                prev_ident = ""
                i = new_i
                continue
            prev_ident = ident
            i = j
            continue
        if ch == "{":
            # Function bodies and brace initializers. Class and namespace braces are
            # consumed when the scope is opened, so a '{' here is not a new scope.
            i = _skip_braces(text, i)
            if paren == 0:
                angle = 0
                bracket = 0
            prev_ident = ""
            continue
        if ch == "}":
            if scope:
                scope.pop()
            angle = 0
            prev_ident = ""
            i += 1
            continue
        if ch == ";" and paren == 0:
            angle = 0
            bracket = 0
            prev_ident = ""
            i += 1
            continue
        if ch == "(":
            paren += 1
            prev_ident = ""
        elif ch == ")":
            paren = max(0, paren - 1)
            prev_ident = ""
        elif ch == "[":
            bracket += 1
            prev_ident = ""
        elif ch == "]":
            bracket = max(0, bracket - 1)
            prev_ident = ""
        elif ch in "<>":
            i, angle = _consume_angle(text, i, angle, prev_ident)
            prev_ident = ""
            continue
        elif _is_ident_start(ch):
            prev_ident, i = _read_ident(text, i)
            continue
        else:
            prev_ident = ""
        i += 1
    return hits


def _hit_at(text: str, attr_start: int, attr_end: int, message: str, scope: list[tuple[str, str]]) -> _RawHit:
    line = text.count("\n", 0, attr_start) + 1
    alias = _using_alias_name(text, attr_start)
    if alias:
        rhs_end = _scan_decl_end(text, attr_end)
        rhs = _collapse(text[attr_end:rhs_end])
        signature = _collapse(f"using {alias} {rhs}")
        return _RawHit(line, _qualify(scope, alias), alias, "alias", signature, message)

    decl = _declaration_text(text, attr_end)
    if decl is None:
        name = _postfix_name(text, attr_start)
        nxt = _skip_ws_comments(text, attr_end)
        kind = "function" if nxt < len(text) and text[nxt] == "(" else "field"
        return _RawHit(line, _qualify(scope, name), name, kind, name, message)

    kind, name = _classify(decl)
    if not name:
        name = "<unknown>"
        kind = "unknown"
    return _RawHit(line, _qualify(scope, name), name, kind, _collapse(decl), message)


def parse_removal(message: str) -> tuple[dt.date | None, str | None, str]:
    """Return (date, raw token, style). style is iso, month, dmy, mdy, ambiguous, or none."""
    match = _ISO_RE.search(message)
    if match:
        try:
            return (
                dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3))),
                match.group(0),
                "iso",
            )
        except ValueError:
            pass
    match = _MONTH_RE.search(message)
    if match:
        month = _MONTHS[match.group(1).lower()]
        try:
            return dt.date(int(match.group(3)), month, int(match.group(2))), match.group(0), "month"
        except ValueError:
            pass
    match = _DMY_RE.search(message)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        raw = match.group(0)
        if day > 12 and month <= 12:
            try:
                return dt.date(year, month, day), raw, "dmy"
            except ValueError:
                return None, raw, "ambiguous"
        if month > 12 and day <= 12:
            try:
                return dt.date(year, day, month), raw, "mdy"
            except ValueError:
                return None, raw, "ambiguous"
        return None, raw, "ambiguous"
    return None, None, "none"


def removal_status(removal: dt.date | None, today: dt.date) -> str:
    if removal is None:
        return "undated"
    if removal < today:
        return "overdue"
    if removal == today:
        return "due"
    return "scheduled"


def manifest_names(description: str) -> set[str]:
    """Identifiers a manifest description names, including {a,b}_suffix expansions."""
    names: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        suffix = match.group(3) or ""
        for part in match.group(2).split(","):
            part = part.strip()
            if part:
                names.add(f"{prefix}{part}{suffix}")
        return " "

    scrubbed = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)?\{([^{}]+)\}(_[A-Za-z0-9_]+)?", replace, description)
    names.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", scrubbed))
    return names


@dataclass
class ManifestEntry:
    id: str
    files: set[str]
    names: set[str]


def load_manifest(path: Path) -> list[ManifestEntry]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: list[ManifestEntry] = []
    for item in data.get("deprecations", []):
        files = {str(item_path).replace("\\", "/") for item_path in item.get("files", [])}
        entries.append(
            ManifestEntry(
                id=str(item.get("id", "")),
                files=files,
                names=manifest_names(str(item.get("description", ""))),
            )
        )
    return entries


def api_tier(relpath: str) -> str:
    marker = "tt_metal/api/"
    rest = relpath[relpath.find(marker) + len(marker) :] if marker in relpath else relpath
    if rest.startswith("tt-metalium/experimental/"):
        return "experimental"
    if rest.startswith("tt-metalium/"):
        return "stable"
    if rest.startswith("internal/"):
        return "internal"
    return "other"


@dataclass
class DeprecatedSymbol:
    file: str
    line: int
    tier: str
    qualified_name: str
    name: str
    kind: str
    signature: str
    message: str
    removal_date: dt.date | None
    removal_raw: str | None
    removal_style: str
    status: str
    manifest_ids: list[str]
    manifest_file_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "tier": self.tier,
            "qualified_name": self.qualified_name,
            "name": self.name,
            "kind": self.kind,
            "signature": self.signature,
            "message": self.message,
            "removal_date": self.removal_date.isoformat() if self.removal_date else None,
            "removal_raw": self.removal_raw,
            "removal_style": self.removal_style,
            "status": self.status,
            "manifest_ids": self.manifest_ids,
            "manifest_file_ids": self.manifest_file_ids,
        }


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def iter_headers(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix in _HEADER_SUFFIXES)


def collect(scan_root: Path, repo_root: Path, manifest: list[ManifestEntry], today: dt.date) -> list[DeprecatedSymbol]:
    symbols: list[DeprecatedSymbol] = []
    for header in iter_headers(scan_root):
        text = header.read_text(encoding="utf-8")
        rel = _rel(header, repo_root)
        tier = api_tier(rel)
        file_entries = [entry for entry in manifest if rel in entry.files]
        for hit in scan_text(text):
            removal, raw, style = parse_removal(hit.message)
            named = [entry.id for entry in file_entries if hit.name in entry.names]
            file_only = [entry.id for entry in file_entries if hit.name not in entry.names]
            symbols.append(
                DeprecatedSymbol(
                    file=rel,
                    line=hit.line,
                    tier=tier,
                    qualified_name=hit.qualified_name,
                    name=hit.name,
                    kind=hit.kind,
                    signature=hit.signature,
                    message=hit.message,
                    removal_date=removal,
                    removal_raw=raw,
                    removal_style=style,
                    status=removal_status(removal, today),
                    manifest_ids=named,
                    manifest_file_ids=file_only,
                )
            )
    symbols.sort(key=lambda item: (_STATUS_ORDER[item.status], item.removal_date or dt.date.max, item.file, item.line))
    return symbols


def find_repo_root(start: Path) -> Path:
    for cand in [start.resolve(), *start.resolve().parents]:
        if (cand / ".github" / "deprecations.json").is_file():
            return cand
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / ".github" / "deprecations.json").is_file():
            return cand
    return start.resolve()


def _tally(symbols: list[DeprecatedSymbol], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in symbols:
        key = str(getattr(symbol, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def format_text(symbols: list[DeprecatedSymbol], scan_root: Path, manifest_path: Path, today: dt.date) -> str:
    lines: list[str] = []
    by_status = _tally(symbols, "status")
    by_tier = _tally(symbols, "tier")
    tracked = sum(1 for symbol in symbols if symbol.manifest_ids)
    lines.append(f"Deprecated Metal API symbols: {len(symbols)}")
    lines.append(f"Scanned: {scan_root}")
    lines.append(f"Today: {today.isoformat()} (overdue means today is after the date in the message)")
    status_bits = ", ".join(f"{name} {by_status.get(name, 0)}" for name in ("overdue", "due", "scheduled", "undated"))
    tier_bits = ", ".join(f"{name} {count}" for name, count in sorted(by_tier.items()))
    lines.append(f"Status: {status_bits}")
    lines.append(f"Tier: {tier_bits if tier_bits else 'none'}")
    lines.append(
        f"Manifest: {manifest_path} names {tracked} of these symbols. "
        "The manifest is the reaper's grace list, not a complete inventory."
    )
    lines.append("")
    for symbol in symbols:
        when = symbol.removal_date.isoformat() if symbol.removal_date else "no date"
        note = ""
        if symbol.removal_date is not None and symbol.status == "overdue":
            note = f", {(today - symbol.removal_date).days}d past"
        elif symbol.removal_date is not None and symbol.status == "scheduled":
            note = f", {(symbol.removal_date - today).days}d left"
        elif symbol.status == "due":
            note = ", removal date is today"
        parsed = ""
        if symbol.removal_style in {"dmy", "mdy"} and symbol.removal_raw:
            parsed = f" (parsed {symbol.removal_raw} as {symbol.removal_style.upper()})"
        elif symbol.removal_style == "ambiguous" and symbol.removal_raw:
            parsed = f" (unparsed ambiguous date {symbol.removal_raw})"
        lines.append(f"{symbol.status:9} {when}{note}{parsed}")
        lines.append(f"  {symbol.qualified_name}")
        lines.append(f"  {symbol.tier}  {symbol.kind}  {symbol.file}:{symbol.line}")
        if symbol.signature:
            signature = symbol.signature if len(symbol.signature) <= 200 else symbol.signature[:197] + "..."
            lines.append(f"  {signature}")
        if symbol.message:
            lines.append(f"  {symbol.message}")
        if symbol.manifest_ids:
            lines.append(f"  manifest: {', '.join(symbol.manifest_ids)}")
        elif symbol.manifest_file_ids:
            listed = ", ".join(symbol.manifest_file_ids)
            lines.append(f"  manifest: header is listed in {listed}; this symbol is not named there")
        else:
            lines.append("  manifest: not tracked")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_json(
    symbols: list[DeprecatedSymbol], scan_root: Path, manifest_path: Path, manifest: list[ManifestEntry], today: dt.date
) -> str:
    matched: dict[str, int] = {}
    for symbol in symbols:
        for entry_id in symbol.manifest_ids:
            matched[entry_id] = matched.get(entry_id, 0) + 1
    payload = {
        "scanned": str(scan_root),
        "today": today.isoformat(),
        "manifest": str(manifest_path),
        "manifest_entries": [
            {"id": entry.id, "symbols_named": matched.get(entry.id, 0), "files": sorted(entry.files)}
            for entry in manifest
        ],
        "counts": {
            "symbols": len(symbols),
            "by_status": _tally(symbols, "status"),
            "by_tier": _tally(symbols, "tier"),
            "named_in_manifest": sum(1 for symbol in symbols if symbol.manifest_ids),
            "not_named_in_manifest": sum(1 for symbol in symbols if not symbol.manifest_ids),
        },
        "symbols": [symbol.to_dict() for symbol in symbols],
    }
    return json.dumps(payload, indent=2) + "\n"


def _self_test() -> int:
    here = str(Path(__file__).resolve().parent)
    suite = unittest.defaultTestLoader.discover(here, pattern="test_list_deprecated_api.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--self-test"]:
        return _self_test()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="Header file or directory. Default: tt_metal/api")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--status", choices=("overdue", "due", "scheduled", "undated"))
    parser.add_argument("--tier", choices=("stable", "experimental", "internal", "other"))
    parser.add_argument("--manifest", type=Path, help="Default: <repo>/.github/deprecations.json")
    parser.add_argument("--today", help="Override today's date as YYYY-MM-DD")
    parser.add_argument("--self-test", action="store_true", help="Run the scanner tests and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    repo = find_repo_root(Path(args.path) if args.path else Path.cwd())
    scan_root = Path(args.path) if args.path else repo / "tt_metal" / "api"
    if not scan_root.exists():
        print(f"path not found: {scan_root}", file=sys.stderr)
        return 2
    manifest_path = args.manifest if args.manifest else repo / ".github" / "deprecations.json"
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    manifest = load_manifest(manifest_path)
    symbols = collect(scan_root, repo, manifest, today)
    if args.status:
        symbols = [symbol for symbol in symbols if symbol.status == args.status]
    if args.tier:
        symbols = [symbol for symbol in symbols if symbol.tier == args.tier]

    if args.format == "json":
        sys.stdout.write(format_json(symbols, scan_root, manifest_path, manifest, today))
    else:
        sys.stdout.write(format_text(symbols, scan_root, manifest_path, today))
    return 0


if __name__ == "__main__":
    sys.exit(main())
