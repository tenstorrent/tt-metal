# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LLK API analyzer: recover the LLK APIs (and their template / runtime args)
used by the compute kernels of a TTNN / tt-metal run.

See ``README.md``. The most commonly used entry points are re-exported here:

    from tt_metal.tools.llk_api_analyzer import LlkAnalyzer, ModelRunner
"""

from __future__ import annotations

from .analyzer import LlkAnalyzer
from .extractor import ExtractorConfig
from .model import ApiCall, ApiLayer, ComputeThread, KernelAnalysis, RunAnalysis
from .report import collapse_rows, render_csv, render_table, render_text, to_json
from .runner import ModelRunner, RunResult

__all__ = [
    "LlkAnalyzer",
    "ExtractorConfig",
    "ModelRunner",
    "RunResult",
    "RunAnalysis",
    "KernelAnalysis",
    "ApiCall",
    "ApiLayer",
    "ComputeThread",
    "render_text",
    "to_json",
    "render_table",
    "render_csv",
    "collapse_rows",
]
