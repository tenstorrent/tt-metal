# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tree-sitter-based C++ Parsing for Kernel Fusion

Provides robust AST-based parsing of C++ kernel source code for the
sequential kernel chaining infrastructure.  Uses tree-sitter for:
  - Kernel body extraction (finding kernel_main and its body)
  - Pre-main code categorization (functions, variables, namespaces, etc.)
  - Code-aware text replacement (skipping strings, comments, literals)

Also contains utilities for:
  - Local include inlining
  - Include/define collection
"""

import dataclasses
import os
import re
from typing import List, Optional, Set, Tuple

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

# =============================================================================
# Tree-sitter Setup
# =============================================================================

_CPP_LANGUAGE = Language(tscpp.language())
_parser = Parser(_CPP_LANGUAGE)


def _get_text(node, src_bytes: bytes) -> str:
    """Get the source text corresponding to an AST node."""
    return src_bytes[node.start_byte : node.end_byte].decode("utf8")


def _normalize_alt_tokens_in_preprocessor(source: str) -> str:
    """Replace C++ alternative tokens in preprocessor directives for tree-sitter.

    Tree-sitter's C++ grammar doesn't handle C++ alternative tokens
    (``and``, ``or``, ``not``) in preprocessor directives.  We normalize
    them to their symbolic equivalents before parsing.  Handles backslash
    line continuations so multi-line directives are fully normalized.
    """
    result = []
    in_directive = False
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#if") or stripped.startswith("#elif"):
            in_directive = True
        if in_directive:
            line = re.sub(r"\bnot\b", "!", line)
            line = re.sub(r"\band\b", "&&", line)
            line = re.sub(r"\bor\b", "||", line)
        if not line.rstrip().endswith("\\"):
            in_directive = False
        result.append(line)
    return "\n".join(result)


def _parse(source: str):
    """Parse C++ source and return (tree, src_bytes)."""
    normalized = _normalize_alt_tokens_in_preprocessor(source)
    src = normalized.encode("utf8")
    tree = _parser.parse(src)
    return tree, src


# =============================================================================
# Data Structures
# =============================================================================


@dataclasses.dataclass
class PreMainBlock:
    """A categorized top-level block from before kernel_main().

    Attributes:
        text: Original source text of the block.
        kind: One of "function", "variable", "namespace", "using",
              "namespace_alias", "struct", "template", "other",
              "preproc_block".
        name: Extracted name (function or variable name), or None for
              blocks where no phase-specific prefixing is needed.
        inner_names: For ``preproc_block`` kind only — function and
              variable names found inside the preprocessor block.
    """

    text: str
    kind: str
    name: Optional[str] = None
    inner_names: Optional[List[str]] = None


# =============================================================================
# Kernel Body Extraction
# =============================================================================


def _find_function_name(node, src_bytes: bytes) -> Optional[str]:
    """Extract the function name from a function_definition node.

    Handles normal functions and ALWI-prefixed functions where tree-sitter
    parses ALWI as a type_identifier with an ERROR on the real return type.
    """
    declarator = node.child_by_field_name("declarator")
    if not declarator:
        return None
    # function_declarator -> declarator field is the name identifier
    name_node = declarator.child_by_field_name("declarator")
    if name_node:
        text = _get_text(name_node, src_bytes)
        # Handle scope-qualified names (Ns::func -> func)
        if "::" in text:
            return text.split("::")[-1]
        return text
    # Fallback: look for an identifier child directly
    for child in declarator.children:
        if child.type == "identifier":
            return _get_text(child, src_bytes)
    return None


def extract_kernel_body(source: str) -> str:
    """Extract the body of kernel_main() using tree-sitter AST.

    Returns the inner body (without outer braces) of the kernel_main
    function definition.  Returns empty string if not found.
    """
    tree, src = _parse(source)
    for child in tree.root_node.children:
        if child.type == "function_definition":
            name = _find_function_name(child, src)
            if name == "kernel_main":
                body = child.child_by_field_name("body")
                if body:
                    body_text = _get_text(body, src)
                    inner = body_text.strip()
                    if inner.startswith("{"):
                        inner = inner[1:]
                    if inner.endswith("}"):
                        inner = inner[:-1]
                    return inner
    return ""


# =============================================================================
# Pre-Main Code Categorization
# =============================================================================

# Node types that are skipped during pre-main categorization
# (collected separately by collect_includes/collect_defines or not needed).
# Note: preproc_ifdef/preproc_if/preproc_ifndef are NOT skipped — they are
# returned as "preproc_block" PreMainBlocks so their inner function/variable
# names can be extracted and prefixed per-phase.
_SKIP_NODE_TYPES = frozenset(
    {
        "preproc_include",
        "preproc_def",
        "preproc_call",
        "preproc_function_def",
        "comment",
    }
)

# Node types that represent preprocessor conditional blocks at the top level
_PREPROC_BLOCK_TYPES = frozenset({"preproc_ifdef", "preproc_if", "preproc_ifndef"})


def _extract_names_from_preproc(node, src_bytes: bytes) -> List[str]:
    """Recursively extract function and variable names from a preprocessor block.

    Walks all descendants of a ``preproc_ifdef``/``preproc_if`` node (including
    ``preproc_else`` branches and nested preproc blocks) and collects function
    definition names and variable declaration names.
    """
    names: List[str] = []
    _walk_for_names(node, src_bytes, names)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def _walk_for_names(node, src_bytes: bytes, names: List[str]) -> None:
    """Walk AST nodes to find function/variable names."""
    if node.type == "function_definition":
        name = _find_function_name(node, src_bytes)
        if name:
            names.append(name)
        return  # Don't recurse into function bodies

    if node.type == "declaration":
        if not _is_function_declaration(node):
            var_name = _extract_variable_name_from_declaration(node, src_bytes)
            if var_name:
                names.append(var_name)
        return

    if node.type == "template_declaration":
        func_name = _extract_template_function_name(node, src_bytes)
        if func_name:
            names.append(func_name)
        return

    for child in node.children:
        _walk_for_names(child, src_bytes, names)


def _extract_variable_name_from_declaration(node, src_bytes: bytes) -> Optional[str]:
    """Extract the variable name from a declaration node.

    For declarations like ``constexpr uint32_t MAX_TILES = 64;``,
    finds the init_declarator or declarator containing the variable name.
    """
    # Check for init_declarator (has = initializer) or plain declarator
    for child in node.children:
        if child.type == "init_declarator":
            # init_declarator -> declarator field is the variable name
            decl = child.child_by_field_name("declarator")
            if decl:
                # Could be pointer_declarator, reference_declarator, or identifier
                if decl.type == "identifier":
                    return _get_text(decl, src_bytes)
                # pointer_declarator: uint32_t* foo -> find identifier
                for sub in decl.children:
                    if sub.type == "identifier":
                        return _get_text(sub, src_bytes)
            return None

    # Fallback: look for declarator field directly on the declaration
    declarator = node.child_by_field_name("declarator")
    if declarator:
        if declarator.type == "identifier":
            return _get_text(declarator, src_bytes)
        if declarator.type in ("pointer_declarator", "reference_declarator"):
            for sub in declarator.children:
                if sub.type == "identifier":
                    return _get_text(sub, src_bytes)
        # Could be a function_declarator (function declaration, not variable)
        if declarator.type == "function_declarator":
            return None

    return None


def _is_function_declaration(node) -> bool:
    """Check if a declaration node is actually a function declaration (not a variable)."""
    declarator = node.child_by_field_name("declarator")
    if declarator and declarator.type == "function_declarator":
        return True
    # Check for init_declarator wrapping a function_declarator
    for child in node.children:
        if child.type == "init_declarator":
            decl = child.child_by_field_name("declarator")
            if decl and decl.type == "function_declarator":
                return True
    return False


def _extract_template_function_name(node, src_bytes: bytes) -> Optional[str]:
    """Extract function name from a template_declaration if it wraps a function."""
    for child in node.children:
        if child.type == "function_definition":
            return _find_function_name(child, src_bytes)
    return None


def _is_kernel_main_func(node, src_bytes: bytes) -> bool:
    """Check if a function_definition node is kernel_main.

    Note: *src_bytes* is needed here because ``_find_function_name``
    extracts the name text from the AST via byte slicing.
    """
    return _find_function_name(node, src_bytes) == "kernel_main"


def categorize_pre_main(source: str) -> List[PreMainBlock]:
    """Categorize all top-level blocks before kernel_main() using tree-sitter.

    Parses the source with tree-sitter and iterates top-level children,
    stopping at kernel_main.  Each node is classified by type and returned
    as a PreMainBlock with extracted name where applicable.

    Preprocessor conditional blocks (``#ifdef``, ``#if``, ``#ifndef``) are
    returned as ``preproc_block`` blocks with ``inner_names`` listing any
    function or variable names inside.  Other preprocessor directives
    (``#include``, ``#define``) and comments are skipped (collected
    separately).
    """
    tree, src = _parse(source)
    blocks: List[PreMainBlock] = []

    for child in tree.root_node.children:
        # Stop at kernel_main
        if child.type == "function_definition" and _is_kernel_main_func(child, src):
            break

        # Skip non-conditional preprocessor and comments
        if child.type in _SKIP_NODE_TYPES:
            continue

        text = _get_text(child, src)

        # Preprocessor conditional blocks — preserve structure, extract inner names
        if child.type in _PREPROC_BLOCK_TYPES:
            inner = _extract_names_from_preproc(child, src)
            blocks.append(PreMainBlock(text=text, kind="preproc_block", inner_names=inner))
            continue

        if child.type == "function_definition":
            name = _find_function_name(child, src)
            blocks.append(PreMainBlock(text=text, kind="function", name=name))

        elif child.type == "declaration":
            if _is_function_declaration(child):
                # Forward declaration — treat as shared/other
                blocks.append(PreMainBlock(text=text, kind="other"))
            else:
                var_name = _extract_variable_name_from_declaration(child, src)
                blocks.append(PreMainBlock(text=text, kind="variable", name=var_name))

        elif child.type == "namespace_definition":
            blocks.append(PreMainBlock(text=text, kind="namespace"))

        elif child.type == "namespace_alias_definition":
            blocks.append(PreMainBlock(text=text, kind="namespace_alias"))

        elif child.type in ("using_declaration", "alias_declaration", "type_alias_declaration"):
            blocks.append(PreMainBlock(text=text, kind="using"))

        elif child.type in ("struct_specifier", "class_specifier", "enum_specifier"):
            blocks.append(PreMainBlock(text=text, kind="struct"))

        elif child.type == "template_declaration":
            func_name = _extract_template_function_name(child, src)
            if func_name:
                blocks.append(PreMainBlock(text=text, kind="function", name=func_name))
            else:
                blocks.append(PreMainBlock(text=text, kind="template"))

        elif child.type == "expression_statement":
            # e.g., ALWI parsed weirdly — check if it's a misparse of a function
            # Skip these (usually ERROR recovery artifacts)
            continue

        elif child.type == "ERROR":
            # tree-sitter error recovery — skip silently
            continue

        else:
            # Anything else (typedefs, etc.)
            if text.strip():
                blocks.append(PreMainBlock(text=text, kind="other"))

    return blocks


# =============================================================================
# Code-Aware Text Replacement
# =============================================================================

# Node types that represent non-code content (strings, comments, etc.)
_NON_CODE_NODE_TYPES = frozenset(
    {
        "string_literal",
        "char_literal",
        "comment",
        "raw_string_literal",
        "string_content",
        "concatenated_string",
        "system_lib_string",
    }
)


def _collect_non_code_ranges(node, ranges: List[Tuple[int, int]]) -> None:
    """Recursively collect byte ranges of string/comment/literal nodes."""
    if node.type in _NON_CODE_NODE_TYPES:
        ranges.append((node.start_byte, node.end_byte))
        return  # Don't recurse into children of non-code nodes
    for child in node.children:
        _collect_non_code_ranges(child, ranges)


def _in_skip_range(pos: int, skip_ranges: List[Tuple[int, int]]) -> bool:
    """Check if a byte position falls within any skip range."""
    for start, end in skip_ranges:
        if start <= pos < end:
            return True
    return False


def replace_in_code_only(source: str, old_name: str, new_name: str) -> str:
    """Replace word-boundary occurrences of old_name only in code context.

    Uses tree-sitter to identify string literals, char literals, and
    comments, then performs regex replacement only outside those ranges.

    Note on byte vs char offsets: tree-sitter skip_ranges are in byte
    offsets.  ``re.finditer`` yields char offsets.  We convert each match
    start to a byte offset for the skip-range check, but all string
    slicing (``source[last_end:match.start()]``) uses the original char
    offsets, which is correct because ``source`` is a Python str.
    """
    tree, src = _parse(source)
    skip_ranges: List[Tuple[int, int]] = []
    _collect_non_code_ranges(tree.root_node, skip_ranges)

    pattern = re.compile(rf"\b{re.escape(old_name)}\b")

    # Build result by processing matches
    result = []
    last_end = 0
    for match in pattern.finditer(source):
        # Convert character offset to byte offset for skip-range comparison
        match_byte_start = len(source[: match.start()].encode("utf8"))
        if _in_skip_range(match_byte_start, skip_ranges):
            # Inside string/comment — keep original
            result.append(source[last_end : match.end()])
        else:
            result.append(source[last_end : match.start()])
            result.append(new_name)
        last_end = match.end()
    result.append(source[last_end:])
    return "".join(result)


# =============================================================================
# Include Inlining (moved from sequential.py)
# =============================================================================


def inline_local_includes(source: str, kernel_dir: Optional[str]) -> str:
    """Inline local includes (same-directory headers) into source.

    For generated SOURCE_CODE kernels, local includes won't resolve because
    the compiler doesn't know the original directory.  We inline them directly
    into the source.

    Supports both local-only includes (no path separator) and relative path
    includes (e.g. ``"subdir/header.h"``), resolving them relative to
    *kernel_dir*.
    """
    if kernel_dir is None:
        return source

    lines = source.split("\n")
    result = []
    inlined: Set[str] = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#include "'):
            match = re.match(r'#include\s+"([^"]+)"', stripped)
            if match:
                inc_path = match.group(1)
                if inc_path not in inlined:
                    full_inc = os.path.normpath(os.path.join(kernel_dir, inc_path))
                    if os.path.exists(full_inc):
                        with open(full_inc, "r") as f:
                            inc_content = f.read()
                        inc_lines = inc_content.split("\n")
                        for inc_line in inc_lines:
                            stripped_inc = inc_line.strip()
                            if stripped_inc.startswith("#pragma once"):
                                continue
                            if stripped_inc.startswith('#include "'):
                                nested_match = re.match(r'#include\s+"([^"]+)"', stripped_inc)
                                if nested_match:
                                    nested = nested_match.group(1)
                                    nested_full = os.path.normpath(os.path.join(os.path.dirname(full_inc), nested))
                                    # Only skip nested includes that resolve to local files
                                    # (those will be inlined separately). Non-existent paths
                                    # are kept as-is — they may be system/SDK includes that
                                    # the compiler resolves via its include path.
                                    if os.path.exists(nested_full):
                                        continue
                            result.append(inc_line)
                        inlined.add(inc_path)
                        continue
        result.append(line)

    return "\n".join(result)


# =============================================================================
# Collection Helpers (moved from sequential.py)
# =============================================================================


def collect_includes(sources: List[str]) -> List[str]:
    """Collect unique #include lines from multiple source strings."""
    includes = set()
    for source in sources:
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#include"):
                includes.add(stripped)
    return sorted(includes)


def collect_defines(sources: List[str]) -> List[str]:
    """Collect unique #define lines from multiple source strings (before kernel_main)."""
    defines = []
    seen = set()
    for source in sources:
        # Use tree-sitter to find where kernel_main starts
        tree, src = _parse(source)
        kernel_main_line = None
        for child in tree.root_node.children:
            if child.type == "function_definition" and _is_kernel_main_func(child, src):
                kernel_main_line = child.start_point[0]
                break

        for line_no, line in enumerate(source.split("\n")):
            if kernel_main_line is not None and line_no >= kernel_main_line:
                break
            stripped = line.strip()
            if stripped.startswith("#define") and stripped not in seen:
                defines.append(line)
                seen.add(stripped)
    return defines


def normalize_block(block: str) -> str:
    """Normalize a code block for deduplication comparison.

    Strips leading/trailing whitespace on each line, collapses
    multiple spaces to single space, and removes empty lines.
    """
    lines = []
    for line in block.split("\n"):
        normalized = " ".join(line.split())
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


__all__ = [
    "PreMainBlock",
    "categorize_pre_main",
    "collect_defines",
    "collect_includes",
    "extract_kernel_body",
    "inline_local_includes",
    "normalize_block",
    "replace_in_code_only",
]
