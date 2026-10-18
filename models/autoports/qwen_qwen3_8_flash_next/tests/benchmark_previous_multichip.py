"""Immutable stage-4 implementation in the same current measurement harness."""

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench


def main():
    source = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder/s5_before_l0/source.py.txt"
    name = "models.autoports.qwen_qwen3_8_flash_next.tt.stage4_control"
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(source)))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    bench.MultichipDecoder = module.MultichipDecoder
    bench.SOURCE_OVERRIDE = source
    bench.CANDIDATE = {"name": "immutable_stage4_control", "source": str(source)}
    bench.main()


if __name__ == "__main__":
    main()
