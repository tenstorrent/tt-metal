import ast
import inspect
import textwrap
from typing import Callable, List

# ====
# Public functions
# ====


def function_to_source(func: Callable, indent: str = "") -> List[str]:
    """
    Convert a function to optimized source code by extracting the original source,
    analyzing for optimization opportunities (placeholder), and generating enhanced source.

    Args:
        func: The callable function to introspect.
        indent: Indentation string.

    Returns:
        List of source code lines.

    Raises:
        ValueError: If the function cannot be inspected or parsed.
    """
    if not callable(func):
        raise ValueError(f"Expected a callable, got {type(func)}")

    try:
        # Get original source
        original_source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise ValueError(f"Could not retrieve source code for function '{func.__name__}': {str(e)}")

    # Dedent to ensure clean parsing if function was nested or indented
    original_source = textwrap.dedent(original_source)

    try:
        # Parse AST for analysis
        tree = ast.parse(original_source)
    except SyntaxError as e:
        raise ValueError(f"Failed to parse source code for function '{func.__name__}': {str(e)}")

    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(f"Parsed AST for '{func.__name__}' does not contain a valid function definition")

    # Extract function details
    func_def = tree.body[0]
    func_name = func_def.name
    args = [arg.arg for arg in func_def.args.args]

    # Generate optimized source with full context
    source_lines = _generate_optimized_source(func_name, args, func_def, indent)

    return source_lines


def class_to_source(cls: type, indent: str = "") -> List[str]:
    """
    Convert a class to source code.

    Args:
        cls: The class to introspect.
        indent: Indentation string.

    Returns:
        List of source code lines.

    Raises:
        ValueError: If the input is not a class or source cannot be retrieved.
    """
    if not isinstance(cls, type):
        raise ValueError(f"Expected a class, got {type(cls)}")

    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError) as e:
        raise ValueError(f"Could not retrieve source code for class '{cls.__name__}': {str(e)}")

    source = textwrap.dedent(source)
    lines = source.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [f"{indent}{line}" for line in lines]


# ====
# Private functions
# ====


def _generate_optimized_source(
    func_name: str, args: List[str], func_def: ast.FunctionDef, indent: str = ""
) -> List[str]:
    """
    Generate complete optimized source code.

    Args:
        func_name: Name of the function to generate.
        args: List of argument names.
        func_def: AST definition of the function body.
        indent: Indentation string to apply to the generated code.

    Returns:
        List of strings representing the source code lines.

    Raises:
        ValueError: If inputs are invalid or inconsistent.
    """
    if not func_name:
        raise ValueError("Function name cannot be empty")

    if func_def is None:
        raise ValueError("Function definition AST cannot be None")

    # Extract function body
    try:
        # unparse usually re-indents to 4 spaces.
        full_code = ast.unparse(func_def)
    except Exception as e:
        raise ValueError(f"Failed to unparse AST for function '{func_name}': {str(e)}")

    body_lines = full_code.split("\n")[1:]  # Skip def line

    local_indent = "    "  # 4 spaces
    source_lines = [
        f'def {func_name}({", ".join(args)}):',
        f'{local_indent}"""Generated from introspected function"""',
    ]

    for line in body_lines:
        source_lines.append(line)

    source_lines = [f"{indent}{line}" for line in source_lines]
    return source_lines
