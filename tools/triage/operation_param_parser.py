#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from functools import lru_cache

from triage import log_check


class ParameterParser:
    """Parser/pretty-printer for C++ constructor parameter strings from inspector data.

    Handles nested constructors with balanced delimiters and formats them
    into human-readable multi-line output.
    """

    # Parsing constraints
    MAX_RECURSION_DEPTH = 15  # Prevent infinite recursion in nested structures

    # Compiled regex patterns for performance
    TENSOR_PATTERN = re.compile(r"\[(\d+)\]:\s*")
    CONSTRUCTOR_PATTERN = re.compile(r"(\w+)\((.*)\)$", re.DOTALL)
    KEY_VALUE_PATTERN = re.compile(r"(\w+)\s*=\s*")

    @classmethod
    @lru_cache(maxsize=2048)
    def format_multiline(cls, params: str) -> str:
        """Format operation parameters into readable multi-line output."""
        if not params or params == "N/A":
            return params

        # Split by tensor pattern: [0]:, [1]:, etc.
        tensor_parts = cls.TENSOR_PATTERN.split(params)

        result_lines: list[str] = []

        # Process pairs: (tensor_index, tensor_content)
        for idx in range(1, len(tensor_parts), 2):
            if idx + 1 >= len(tensor_parts):
                break

            tensor_index = tensor_parts[idx]
            tensor_content = tensor_parts[idx + 1]

            result_lines.append(f"Tensor[{tensor_index}]")

            # Extract and format all key=value pairs
            try:
                key_value_pairs = cls._parse_constructor(tensor_content)
                for full_key, value in key_value_pairs:
                    # Use only the last component of dotted key paths for cleaner output
                    short_key = full_key.split(".")[-1]
                    result_lines.append(f"  {short_key}: {value}")
            except Exception as e:
                log_check(False, f"Parameter parsing error for tensor {tensor_index}: {e}")
                result_lines.append(f"  (parsing error: {e})")

            result_lines.append("")  # Blank line between tensors

        return "\n".join(result_lines).rstrip()

    @classmethod
    def _parse_constructor(cls, content: str) -> list[tuple[str, str]]:
        """Parse C++ constructor parameter strings."""
        # Extract top-level function name and its content
        match = cls.CONSTRUCTOR_PATTERN.match(content)
        if not match:
            # Maybe it's just key=value pairs without outer constructor
            return cls._parse_kv_pairs(content)

        inner_content = match.group(2)
        return cls._parse_kv_pairs(inner_content)

    @classmethod
    def _parse_kv_pairs(cls, content: str, prefix: str = "", depth: int = 0) -> list[tuple[str, str]]:
        """Parse key=value pairs respecting nested parentheses/brackets/braces."""
        if depth >= cls.MAX_RECURSION_DEPTH:
            log_check(False, f"Parameter parsing exceeded max recursion depth ({cls.MAX_RECURSION_DEPTH})")
            return []

        if not content.strip():
            return []

        pairs: list[tuple[str, str]] = []
        pos = 0

        while pos < len(content):
            # Skip whitespace
            while pos < len(content) and content[pos].isspace():
                pos += 1

            if pos >= len(content):
                break

            # Try to find key=value pattern
            match = cls.KEY_VALUE_PATTERN.match(content[pos:])
            if not match:
                # No more key=value pairs, skip ahead
                pos += 1
                continue

            key = match.group(1)
            pos += len(match.group(0))

            # Extract the value (respecting balanced delimiters)
            value_str, value_len = cls._extract_balanced_value(content, pos)
            if not value_str:
                pos += max(1, value_len)
                continue

            full_key = f"{prefix}.{key}" if prefix else key
            pos += value_len

            # Check if value is a function call (constructor)
            func_match = cls.CONSTRUCTOR_PATTERN.match(value_str)
            if func_match:
                func_name, inner = func_match.groups()

                # Special case for Shape and Alignment - if they contain just an array, keep it simple
                if func_name in ["Shape", "Alignment"] and inner.strip().startswith("["):
                    pairs.append((full_key, inner.strip()))
                else:
                    nested = cls._parse_kv_pairs(inner, full_key, depth + 1)
                    if nested:
                        pairs.extend(nested)
                    else:
                        # Empty constructor or failed to parse
                        pairs.append((full_key, f"{func_name}()"))
            else:
                value_str = cls._clean_value(value_str)
                pairs.append((full_key, value_str))

            # Skip comma and whitespace
            while pos < len(content) and content[pos] in ", \t\n":
                pos += 1

        return pairs

    @staticmethod
    def _clean_value(value: str) -> str:
        """Clean up a value string for better readability."""
        value = value.strip()

        # Replace nullopt with none
        if "nullopt" in value:
            return "none"

        # Extract last component of namespace-qualified names
        if "::" in value:
            return value.split("::")[-1]

        return value

    @staticmethod
    def _extract_balanced_value(content: str, start: int) -> tuple[str, int]:
        """Extract a value from content starting at start, respecting balanced delimiters."""
        if start >= len(content):
            return "", 0

        # Track nesting depth for (), [], {}
        depth = {"(": 0, "[": 0, "{": 0}
        pos = start

        while pos < len(content):
            ch = content[pos]

            if ch in "([{":
                depth[ch] += 1
            elif ch == ")":
                if depth["("] == 0:
                    break
                depth["("] -= 1
            elif ch == "]":
                if depth["["] == 0:
                    break
                depth["["] -= 1
            elif ch == "}":
                if depth["{"] == 0:
                    break
                depth["{"] -= 1
            elif ch == "," and all(d == 0 for d in depth.values()):
                break

            pos += 1

        return content[start:pos].strip(), pos - start
