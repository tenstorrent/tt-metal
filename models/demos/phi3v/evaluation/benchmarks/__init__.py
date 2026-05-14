# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.qwen3_vl.evaluation.benchmarks.mcq_benchmarks import (
    AI2DBenchmark,
    MathVistaBenchmark,
    MMMUBenchmark,
)
from models.demos.qwen3_vl.evaluation.benchmarks.chartqa import ChartQABenchmark

from .mmbench import MMBenchBenchmark
from .pope import POPEBenchmark
from .scienceqa import ScienceQABenchmark
from .textvqa import TextVQABenchmark

BENCHMARK_REGISTRY = {
    "mmmu": MMMUBenchmark,
    "mmbench": MMBenchBenchmark,
    "scienceqa": ScienceQABenchmark,
    "mathvista": MathVistaBenchmark,
    "ai2d": AI2DBenchmark,
    "chartqa": ChartQABenchmark,
    "textvqa": TextVQABenchmark,
    "pope": POPEBenchmark,
}

# Reference scores from Phi-3.5-vision-instruct model card
REFERENCE_SCORES = {
    "MMMU": ("Accuracy", 43.0),
    "MMBench": ("Accuracy", 81.9),
    "ScienceQA": ("Accuracy", 91.3),
    "MathVista": ("Accuracy", 43.9),
    "AI2D": ("Accuracy", 78.1),
    "ChartQA": ("RelaxedAcc", 81.8),
    "TextVQA": ("VQA Acc", 72.0),
    "POPE": ("Accuracy", 86.1),
}

__all__ = [
    "MMMUBenchmark",
    "MMBenchBenchmark",
    "ScienceQABenchmark",
    "MathVistaBenchmark",
    "AI2DBenchmark",
    "ChartQABenchmark",
    "TextVQABenchmark",
    "POPEBenchmark",
    "BENCHMARK_REGISTRY",
    "REFERENCE_SCORES",
]
