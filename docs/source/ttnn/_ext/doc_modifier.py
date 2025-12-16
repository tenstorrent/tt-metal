# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import inspect

# Add the project root to Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.ttnn.docs_examples.examples_mapping import FUNCTION_TO_EXAMPLES_MAPPING_DICT


def get_function_body(func: callable) -> str:
    """
    Extract the body of a function as a string, removing the function definition line.
    Args:
        func (callable): The function to extract the body from.
    Returns:
        function_body (str): The body of the function as a string, properly indented.
    """
    # Get the source code
    source = inspect.getsource(func)
    source_lines = source.splitlines()

    # Find where the function signature ends (look for the colon)
    signature_end_idx = None
    paren_count = 0

    for i, line in enumerate(source_lines):
        # Count parentheses to handle multi-line signatures
        paren_count += line.count("(") - line.count(")")

        # Check if we've found the end of the signature
        if ":" in line and paren_count == 0:
            # Check if there's code after the colon on the same line
            after_colon = line.split(":", 1)[1].strip()
            if after_colon and not after_colon.startswith("#"):
                # There's code on the same line, include it
                signature_end_idx = i
            else:
                # Body starts on the next line
                signature_end_idx = i + 1
            break

    if signature_end_idx is None:
        return ""

    # Extract body lines
    body_lines = source_lines[signature_end_idx:]

    # If the first line contains the colon, extract only the part after it
    if signature_end_idx < len(source_lines) and ":" in source_lines[signature_end_idx]:
        first_line = body_lines[0]
        after_colon = first_line.split(":", 1)[1]
        if after_colon.strip():
            body_lines[0] = after_colon
        else:
            body_lines = body_lines[1:] if len(body_lines) > 1 else []

    # Join and dedent to remove the function's original indentation
    body_text = "\n".join(body_lines)

    # Re-indent with 4 spaces
    body_lines = body_text.splitlines()
    indented_lines = ["    " + line if line.strip() else "" for line in body_lines]

    return "\n".join(indented_lines).rstrip()


def process_docstring(app, what, name, obj, options, lines):
    """
    Modify docstrings during Sphinx processing by appending example code blocks.

    This function is designed to be used as a Sphinx autodoc event handler that
    processes docstrings and adds example code sections when available.

    Args:
        app: The Sphinx application instance
        what (str): The type of object being documented ('module', 'class', 'method', 'function', etc.)
        name (str): The fully qualified name of the object (e.g., 'ttnn.sort')
        obj: The actual Python object being documented
        options: Options passed to the autodoc directive
        lines (list): Current docstring lines as a mutable list

    Returns:
        None: Modifies the lines parameter in place

    Notes:
        - Only processes objects whose names are present in the global EXAMPLES_DICT
        - Extracts function body from example functions, removing the function definition
        - Formats the example code as a reStructuredText code block with proper indentation
        - Preserves existing docstring content and appends the example section
        - Uses inspect.getsource() to extract source code from example functions
    """
    if name in FUNCTION_TO_EXAMPLES_MAPPING_DICT:
        # Get current docstring
        current_doc = lines

        # Get the example function
        function = FUNCTION_TO_EXAMPLES_MAPPING_DICT.get(name, None)
        if function is None:
            # No function found for example, skip modification - early exit
            return

        # Get example code as string
        body_code = get_function_body(function)
        additional_doc = f"\n.. admonition:: Example\n\n    .. code-block:: python\n\n{body_code}\n\n"

        # Replace existing lines with updated content
        new_lines = []
        new_lines.extend(current_doc)
        new_lines.extend(additional_doc.splitlines())

        # Modify lines *in place*
        lines[:] = new_lines


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring)
    return {"version": "0.1"}
