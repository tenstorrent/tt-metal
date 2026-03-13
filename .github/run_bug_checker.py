#!/usr/bin/env python3
"""Entry point script for bug-checker.

Handles the fact that the package directory uses a hyphen (bug-checker)
which is not a valid Python identifier. Registers it as 'bug_checker'
in sys.modules, then delegates to __main__.
"""

import importlib.util
import sys
from pathlib import Path

pkg_dir = Path(__file__).resolve().parent / "bug-checker"

# Register the package so relative imports work
spec = importlib.util.spec_from_file_location(
    "bug_checker",
    pkg_dir / "__init__.py",
    submodule_search_locations=[str(pkg_dir)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["bug_checker"] = mod
spec.loader.exec_module(mod)

# Register submodules needed by relative imports
for submod_name in ("rules", "llm", "github_client", "output", "orchestrator"):
    submod_path = pkg_dir / f"{submod_name}.py"
    sub_spec = importlib.util.spec_from_file_location(f"bug_checker.{submod_name}", submod_path)
    sub_mod = importlib.util.module_from_spec(sub_spec)
    sys.modules[f"bug_checker.{submod_name}"] = sub_mod
    sub_spec.loader.exec_module(sub_mod)

# Now run __main__
main_spec = importlib.util.spec_from_file_location("bug_checker.__main__", pkg_dir / "__main__.py")
main_mod = importlib.util.module_from_spec(main_spec)
main_spec.loader.exec_module(main_mod)
sys.exit(main_mod.main())
