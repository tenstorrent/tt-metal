"""Explicit historical monolithic control for diagnostic comparisons only.

Production never imports this module. It does not add a runtime fallback or
configuration switch to the serving generator. The snapshot includes canonical
prefill workspace changes immediately preceding factored decode selection;
its exact hash is retained in diagnostic_sources/factored_before/manifest.json.
"""
import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).parent / "diagnostic_sources/factored_before/generator.py"
NAME = "models.autoports.qwen_qwen3_8_flash_next.tt._whole_decode_control_generator"


def build_generator(*args, **kwargs):
    if NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(NAME, SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(NAME, None)
            raise
    return sys.modules[NAME].build_generator(*args, **kwargs)
