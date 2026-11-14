# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests: real public wrapper, synthetic assets, no native imports."""
import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[3] / "pipelines/ltx/pipeline_ltx25_iclora.py"


def test_public_callable_is_cpu_importable():
    spec = importlib.util.spec_from_file_location("iclora_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.LTX25ICLoRAPipeline.create_pipeline)
