"""Import the repository-owned fixed advisor-challenger harness protocol."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_path = Path(__file__).resolve().parents[5] / ".agents/skills/advisor-challenger/scripts/harness_template.py"
_spec = spec_from_file_location("advisor_challenger_fixed_harness", _path)
assert _spec and _spec.loader
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})
