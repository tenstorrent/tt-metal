# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import inspect

# Add the project root to Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.ttnn.docs_examples.examples_mapping import EXAMPLES_DICT


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
    if name in EXAMPLES_DICT:
        # Get current docstring
        current_doc = getattr(obj, "__doc__", "") or ""

        # Get the example function
        function = EXAMPLES_DICT.get(name, None)
        if function is None:
            # No function found for example, skip modification
            return

        # Get example code as string
        function_code = inspect.getsource(function)

        # Process example code to extract only the body (skip def and indentation)
        function_lines = function_code.splitlines()
        body_lines = []
        in_body = False
        for line in function_lines:
            if line.startswith("def "):
                in_body = True
                continue
            if in_body:
                body_lines.append(line)
        body_code = "\n\t".join(body_lines)
        additional_doc = f"\n\tExample:\n\n\t.. code-block:: python\n\n\t{body_code}\n"

        # Replace existing lines with updated content
        new_lines = []
        if current_doc.strip():
            new_lines.extend(current_doc.splitlines())
        new_lines.extend(additional_doc.splitlines())

        for line in new_lines:
            print(line)

        # Modify lines *in place*
        lines[:] = new_lines


def setup(app):
    app.connect("autodoc-process-docstring", process_docstring)
    return {"version": "0.1"}
