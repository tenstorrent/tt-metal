"""Bootstrap bug_checker package for tests."""

import importlib.util
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent.parent

if "bug_checker" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "bug_checker",
        _pkg_dir / "__init__.py",
        submodule_search_locations=[str(_pkg_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bug_checker"] = mod
    spec.loader.exec_module(mod)

    for submod_name in ("rules", "llm", "github_client", "output", "orchestrator"):
        submod_path = _pkg_dir / f"{submod_name}.py"
        if submod_path.exists():
            sub_spec = importlib.util.spec_from_file_location(f"bug_checker.{submod_name}", submod_path)
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[f"bug_checker.{submod_name}"] = sub_mod
            sub_spec.loader.exec_module(sub_mod)
