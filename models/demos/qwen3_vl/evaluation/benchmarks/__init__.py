from .docvqa import DocVQABenchmark, InfoVQABenchmark
from .chartqa import ChartQABenchmark
from .mcq_benchmarks import (
    MMMUBenchmark,
    MMStarBenchmark,
    MathVistaBenchmark,
    AI2DBenchmark,
    RealWorldQABenchmark,
    OCRBenchBenchmark,
)

BENCHMARK_REGISTRY = {
    "docvqa":     DocVQABenchmark,
    "infovqa":    InfoVQABenchmark,
    "chartqa":    ChartQABenchmark,
    "mmmu":       MMMUBenchmark,
    "mmstar":     MMStarBenchmark,
    "mathvista":  MathVistaBenchmark,
    "ai2d":       AI2DBenchmark,
    "realworldqa": RealWorldQABenchmark,
    "ocrbench":   OCRBenchBenchmark,
}

# Reference scores from Qwen3-VL-2B-Instruct model card
REFERENCE_SCORES = {
    "DocVQA":     ("ANLS",        93.3),
    "InfoVQA":    ("ANLS",        72.4),
    "ChartQA":    ("RelaxedAcc",  72.8),
    "MMMU":       ("Accuracy",    53.4),
    "MMStar":     ("Accuracy",    58.3),
    "MathVista":  ("Accuracy",    61.3),
    "AI2D":       ("Accuracy",    76.9),
    "RealWorldQA":("Accuracy",    63.9),
    "OCRBench":   ("Score(×1000)", 881),
}

__all__ = [
    "DocVQABenchmark", "InfoVQABenchmark", "ChartQABenchmark",
    "MMMUBenchmark", "MMStarBenchmark", "MathVistaBenchmark",
    "AI2DBenchmark", "RealWorldQABenchmark", "OCRBenchBenchmark",
    "BENCHMARK_REGISTRY", "REFERENCE_SCORES",
]
