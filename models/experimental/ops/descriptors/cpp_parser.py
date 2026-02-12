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
  - Selective ifdef resolution (AST-based condition evaluation)

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
              "namespace_alias", "struct", "template", "other".
        name: Extracted name (function or variable name), or None for
              blocks where no phase-specific prefixing is needed.
    """

    text: str
    kind: str
    name: Optional[str] = None


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
# (collected separately by collect_includes/collect_defines or not needed)
_SKIP_NODE_TYPES = frozenset(
    {
        "preproc_include",
        "preproc_def",
        "preproc_ifdef",
        "preproc_if",
        "preproc_ifndef",
        "preproc_call",
        "preproc_function_def",
        "comment",
    }
)


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


def _is_function_declaration(node, src_bytes: bytes) -> bool:
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
    """Check if a function_definition node is kernel_main."""
    return _find_function_name(node, src_bytes) == "kernel_main"


def categorize_pre_main(source: str) -> List[PreMainBlock]:
    """Categorize all top-level blocks before kernel_main() using tree-sitter.

    Parses the source with tree-sitter and iterates top-level children,
    stopping at kernel_main.  Each node is classified by type and returned
    as a PreMainBlock with extracted name where applicable.

    Skips preprocessor directives (#include, #define, #ifdef) and comments,
    which are collected separately.
    """
    tree, src = _parse(source)
    blocks: List[PreMainBlock] = []

    for child in tree.root_node.children:
        # Stop at kernel_main
        if child.type == "function_definition" and _is_kernel_main_func(child, src):
            break

        # Skip preprocessor and comments
        if child.type in _SKIP_NODE_TYPES:
            continue

        text = _get_text(child, src)

        if child.type == "function_definition":
            name = _find_function_name(child, src)
            blocks.append(PreMainBlock(text=text, kind="function", name=name))

        elif child.type == "declaration":
            if _is_function_declaration(child, src):
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
    """
    tree, src = _parse(source)
    skip_ranges: List[Tuple[int, int]] = []
    _collect_non_code_ranges(tree.root_node, skip_ranges)

    pattern = re.compile(rf"\b{re.escape(old_name)}\b")

    # Build result by processing matches
    result = []
    last_end = 0
    for match in pattern.finditer(source):
        # Convert character offset to byte offset for comparison
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
# Ifdef Resolution (tree-sitter AST-based)
# =============================================================================

# Defines that are resolved per-phase in source code (not passed to compiler)
SOURCE_LEVEL_DEFINES = {"RMSNORM", "FUSE_PRE_ADD", "FUSED_PRE_ADD", "FUSE_GAMMA", "FUSE_BETA"}


def _collect_referenced_defines(node, src_bytes: bytes) -> Set[str]:
    """Collect all define names referenced in a preprocessor condition AST."""
    refs: Set[str] = set()
    if node.type == "preproc_defined":
        for child in node.children:
            if child.type == "identifier":
                refs.add(_get_text(child, src_bytes))
    elif node.type == "identifier":
        refs.add(_get_text(node, src_bytes))
    elif node.type in ("binary_expression", "unary_expression", "parenthesized_expression"):
        for child in node.children:
            refs.update(_collect_referenced_defines(child, src_bytes))
    return refs


def _is_resolvable(referenced_defines: Set[str]) -> bool:
    """Check if all referenced defines are known source-level defines."""
    if not referenced_defines:
        return True  # No defines (e.g. #if 0)
    return all(d in SOURCE_LEVEL_DEFINES for d in referenced_defines)


def _eval_condition_node(node, src_bytes: bytes, active_defines: set) -> bool:
    """Recursively evaluate a preprocessor condition expression AST."""
    if node.type == "preproc_defined":
        for child in node.children:
            if child.type == "identifier":
                return _get_text(child, src_bytes) in active_defines
        return False
    elif node.type == "identifier":
        return _get_text(node, src_bytes) in active_defines
    elif node.type == "number_literal":
        return int(_get_text(node, src_bytes)) != 0
    elif node.type == "unary_expression":
        operand = node.children[-1]
        return not _eval_condition_node(operand, src_bytes, active_defines)
    elif node.type == "binary_expression":
        left = node.children[0]
        op = _get_text(node.children[1], src_bytes)
        right = node.children[2]
        if op == "&&":
            return _eval_condition_node(left, src_bytes, active_defines) and _eval_condition_node(
                right, src_bytes, active_defines
            )
        elif op == "||":
            return _eval_condition_node(left, src_bytes, active_defines) or _eval_condition_node(
                right, src_bytes, active_defines
            )
    elif node.type == "parenthesized_expression":
        for child in node.children:
            if child.type not in ("(", ")"):
                return _eval_condition_node(child, src_bytes, active_defines)
    return False


def _get_if_body_range(node, src_bytes: bytes) -> Tuple[int, int]:
    """Get byte range of the if-branch body in a preproc node.

    Returns (start_byte, end_byte) spanning from after the directive line
    (after the ``\\n``) to the start of the first alternative or ``#endif``.
    Includes leading whitespace/indentation on the first body line.
    """
    children = node.children
    if node.type == "preproc_ifdef":
        # Children: #ifdef/#ifndef, identifier, body..., [preproc_else], #endif
        # No \n child — skip newline manually after identifier
        header_end = children[1].end_byte
        if header_end < len(src_bytes) and src_bytes[header_end : header_end + 1] == b"\n":
            header_end += 1
    else:
        # preproc_if/preproc_elif: children[2] is '\n', end_byte is after the \n
        header_end = children[2].end_byte if len(children) > 2 else children[1].end_byte

    # Find where body ends (before next structural element)
    body_end = node.end_byte
    for child in children:
        if child.type in ("preproc_else", "preproc_elif", "#endif"):
            body_end = child.start_byte
            break

    return header_end, body_end


def _get_else_body_range(else_node, src_bytes: bytes, body_end: int) -> Tuple[int, int]:
    """Get byte range of the else-branch body.

    Returns (start_byte, end_byte) spanning from after ``#else\\n`` to
    *body_end* (typically ``#endif.start_byte`` in the parent node).
    The caller provides *body_end* because tree-sitter's ``preproc_else``
    node may not include trailing comments on the last body line.
    """
    else_token = else_node.children[0]  # #else
    body_start = else_token.end_byte
    nl_pos = src_bytes.find(b"\n", body_start)
    if nl_pos >= 0 and nl_pos < body_end:
        body_start = nl_pos + 1
    return body_start, body_end


def _get_body_children(node, body_start: int, body_end: int, *extra_nodes) -> list:
    """Get child nodes that fall within a byte range (for nested resolution).

    Checks direct children of *node* plus children of any *extra_nodes*.
    This is needed for else-branch processing: the body content lives as
    children of ``preproc_else``, but trailing siblings (comments) are
    children of the parent.
    """
    candidates = list(node.children)
    for extra in extra_nodes:
        candidates.extend(extra.children)
    return [child for child in candidates if child.start_byte >= body_start and child.end_byte <= body_end]


def _resolve_body(body_start: int, body_end: int, body_children: list, src_bytes: bytes, active_defines: set) -> str:
    """Extract body text from a byte range and resolve nested preproc nodes within it.

    Recursively walks ALL descendants of body_children (not just direct children)
    to find preproc nodes at any nesting depth — e.g. ``#ifdef`` inside a for loop
    inside the body of an outer ``#if``.
    """
    text = src_bytes[body_start:body_end]

    # Recursively collect all preproc node replacements within the body range
    replacements: List[Tuple[int, int, bytes]] = []
    _collect_body_replacements(body_children, src_bytes, active_defines, replacements, body_start)

    replacements.sort(key=lambda r: r[0], reverse=True)
    for s, e, r in replacements:
        text = text[:s] + r + text[e:]
    return text.decode("utf8")


def _collect_body_replacements(
    children: list, src_bytes: bytes, active_defines: set, replacements: List[Tuple[int, int, bytes]], offset: int
) -> None:
    """Recursively walk children to find and resolve preproc nodes at any depth."""
    for child in children:
        if child.type in ("preproc_ifdef", "preproc_if"):
            resolved = _resolve_preproc_node(child, src_bytes, active_defines)
            if resolved is not None:
                replacements.append((child.start_byte - offset, child.end_byte - offset, resolved.encode("utf8")))
            else:
                # Not resolvable — recurse into it to find nested resolvable nodes
                _collect_nested_replacements(child, src_bytes, active_defines, replacements, offset)
        else:
            # Recurse into non-preproc nodes (for loops, if statements, etc.)
            # to find deeply nested preproc directives
            if child.children:
                _collect_body_replacements(child.children, src_bytes, active_defines, replacements, offset)


def _collect_nested_replacements(node, src_bytes: bytes, active_defines: set, replacements: list, offset: int) -> None:
    """Recurse into a non-resolvable preproc node to find nested resolvable ones."""
    for child in node.children:
        if child.type in ("preproc_ifdef", "preproc_if"):
            resolved = _resolve_preproc_node(child, src_bytes, active_defines)
            if resolved is not None:
                replacements.append((child.start_byte - offset, child.end_byte - offset, resolved.encode("utf8")))
            else:
                _collect_nested_replacements(child, src_bytes, active_defines, replacements, offset)
        elif child.type in ("preproc_else", "preproc_elif"):
            _collect_nested_replacements(child, src_bytes, active_defines, replacements, offset)


def _resolve_preproc_node(node, src_bytes: bytes, active_defines: set) -> Optional[str]:
    """Try to resolve a preproc_ifdef or preproc_if node.

    Returns the replacement text if resolvable, None otherwise.
    """
    if node.type == "preproc_ifdef":
        return _resolve_ifdef_node(node, src_bytes, active_defines)
    elif node.type == "preproc_if":
        return _resolve_if_node(node, src_bytes, active_defines)
    return None


def _find_endif_start(node) -> int:
    """Find the start_byte of the ``#endif`` child in a preproc node."""
    for child in node.children:
        if child.type == "#endif":
            return child.start_byte
    return node.end_byte


def _resolve_ifdef_node(node, src_bytes: bytes, active_defines: set) -> Optional[str]:
    """Resolve a ``preproc_ifdef`` node (covers both ``#ifdef`` and ``#ifndef``)."""
    is_ifndef = False
    name = None
    for child in node.children:
        if child.type == "#ifndef":
            is_ifndef = True
        elif child.type == "identifier" and name is None:
            name = _get_text(child, src_bytes)

    if name is None or name not in SOURCE_LEVEL_DEFINES:
        return None

    is_defined = name in active_defines
    take_if = (is_defined and not is_ifndef) or (not is_defined and is_ifndef)

    if take_if:
        start, end = _get_if_body_range(node, src_bytes)
        children = _get_body_children(node, start, end)
        return _resolve_body(start, end, children, src_bytes, active_defines)
    else:
        endif_start = _find_endif_start(node)
        for child in node.children:
            if child.type == "preproc_else":
                start, end = _get_else_body_range(child, src_bytes, endif_start)
                children = _get_body_children(node, start, end, child)
                return _resolve_body(start, end, children, src_bytes, active_defines)
        return ""  # No else branch, condition was false


def _resolve_if_node(node, src_bytes: bytes, active_defines: set) -> Optional[str]:
    """Resolve a ``preproc_if`` node."""
    if len(node.children) < 2:
        return None
    condition_node = node.children[1]

    refs = _collect_referenced_defines(condition_node, src_bytes)
    if not _is_resolvable(refs):
        return None

    if _eval_condition_node(condition_node, src_bytes, active_defines):
        start, end = _get_if_body_range(node, src_bytes)
        children = _get_body_children(node, start, end)
        return _resolve_body(start, end, children, src_bytes, active_defines)

    # Condition false — check #elif / #else alternatives
    endif_start = _find_endif_start(node)
    for child in node.children:
        if child.type == "preproc_elif":
            return _resolve_elif_chain(child, src_bytes, active_defines, endif_start)
        elif child.type == "preproc_else":
            start, end = _get_else_body_range(child, src_bytes, endif_start)
            children = _get_body_children(node, start, end, child)
            return _resolve_body(start, end, children, src_bytes, active_defines)
    return ""  # No alternative, condition was false


def _resolve_elif_chain(node, src_bytes: bytes, active_defines: set, endif_start: int) -> str:
    """Resolve an ``#elif`` chain by walking until a branch matches or falls through."""
    if len(node.children) < 2:
        return ""
    condition_node = node.children[1]

    refs = _collect_referenced_defines(condition_node, src_bytes)
    if _is_resolvable(refs) and _eval_condition_node(condition_node, src_bytes, active_defines):
        start, end = _get_if_body_range(node, src_bytes)
        children = _get_body_children(node, start, end)
        return _resolve_body(start, end, children, src_bytes, active_defines)

    # This elif didn't match — check further alternatives
    for child in node.children:
        if child.type == "preproc_elif":
            return _resolve_elif_chain(child, src_bytes, active_defines, endif_start)
        elif child.type == "preproc_else":
            start, end = _get_else_body_range(child, src_bytes, endif_start)
            children = _get_body_children(node, start, end, child)
            return _resolve_body(start, end, children, src_bytes, active_defines)
    return ""


def _collect_all_replacements(
    node, src_bytes: bytes, active_defines: set, replacements: List[Tuple[int, int, bytes]]
) -> None:
    """Recursively walk the entire AST collecting byte-range replacements for resolvable preproc nodes.

    Preprocessor directives can appear anywhere in the tree (top-level,
    inside function bodies, etc.), so we must walk all children recursively.
    """
    for child in node.children:
        if child.type in ("preproc_ifdef", "preproc_if"):
            resolved = _resolve_preproc_node(child, src_bytes, active_defines)
            if resolved is not None:
                replacements.append((child.start_byte, child.end_byte, resolved.encode("utf8")))
            else:
                # Not resolvable — recurse to find nested resolvable nodes
                _collect_all_replacements(child, src_bytes, active_defines, replacements)
        else:
            # Recurse into all other nodes (function bodies, compound statements, etc.)
            _collect_all_replacements(child, src_bytes, active_defines, replacements)


def resolve_ifdef_directives(source: str, active_defines: set) -> str:
    """Resolve preprocessor #ifdef/#ifndef/#if defined/#elif directives in source code.

    Uses tree-sitter to parse the preprocessor structure into an AST, then
    recursively resolves directives involving known source-level defines
    (RMSNORM, FUSE_PRE_ADD, FUSED_PRE_ADD, FUSE_GAMMA, FUSE_BETA).
    Other directives are left untouched.

    Supports:
      - ``#ifdef NAME``, ``#ifndef NAME``
      - ``#if defined(NAME)`` / ``#if defined NAME``
      - ``#if !defined(NAME)`` / ``#if !defined NAME``
      - ``#if defined A || defined B``, ``#if defined A && !defined B``
      - ``#if 0``, ``#if 1`` (unconditional)
      - ``#elif`` with any of the above
      - Line continuation with backslash (``\\``)
      - ``#else``, ``#endif``
    """
    normalized = _normalize_alt_tokens_in_preprocessor(source)
    src = normalized.encode("utf8")
    tree = _parser.parse(src)

    replacements: List[Tuple[int, int, bytes]] = []
    _collect_all_replacements(tree.root_node, src, active_defines, replacements)

    # Apply replacements in reverse byte order to preserve offsets
    replacements.sort(key=lambda r: r[0], reverse=True)
    result = src
    for start, end, repl in replacements:
        result = result[:start] + repl + result[end:]
    return result.decode("utf8")


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
                        for il in inc_lines:
                            ils = il.strip()
                            if ils.startswith("#pragma once"):
                                continue
                            if ils.startswith('#include "'):
                                nested_match = re.match(r'#include\s+"([^"]+)"', ils)
                                if nested_match:
                                    nested = nested_match.group(1)
                                    nested_full = os.path.normpath(os.path.join(os.path.dirname(full_inc), nested))
                                    if os.path.exists(nested_full):
                                        continue
                            result.append(il)
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
