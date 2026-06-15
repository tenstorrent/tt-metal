import importlib
import pkgutil
from pathlib import Path
from typing import List, Type

from .binary._spec import BinaryOpSpec
from .ternary._spec import TernaryOpSpec
from .unary._spec import UnaryOpSpec

# Roots an SFPU op header may live under, matching the compiler -I paths: the LLK
# common tree, and tt-metal's quasar llk_api (where the sfpi float ops live, e.g.
# "llk_sfpu/ckernel_sfpu_binary.h").
_INC_ROOTS = [
    Path(__file__).resolve().parents[4] / "tt_llk_quasar" / "common" / "inc",
    Path(__file__).resolve().parents[5]
    / "hw"
    / "ckernels"
    / "quasar"
    / "metal"
    / "llk_api",
]


def _discover(package_name: str, spec_type: Type) -> List:
    pkg = importlib.import_module(package_name)
    specs = []
    for module_info in pkgutil.iter_modules(pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        spec = getattr(module, "SPEC", None)
        if spec is None:
            raise ImportError(f"{module.__name__} missing SPEC")
        if not isinstance(spec, spec_type):
            raise TypeError(
                f"{module.__name__} SPEC must be {spec_type.__name__}, got {type(spec).__name__}"
            )
        if not any((root / spec.include_header).is_file() for root in _INC_ROOTS):
            raise FileNotFoundError(
                f"{spec.name} header not found on any SFPU include root: {spec.include_header}"
            )
        specs.append(spec)
    specs.sort(key=lambda s: s.name)
    return specs


UNARY_OP_REGISTRY = _discover(f"{__name__}.unary", UnaryOpSpec)
BINARY_OP_REGISTRY = _discover(f"{__name__}.binary", BinaryOpSpec)
TERNARY_OP_REGISTRY = _discover(f"{__name__}.ternary", TernaryOpSpec)
