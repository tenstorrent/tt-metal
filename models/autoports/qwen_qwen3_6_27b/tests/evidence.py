# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only provenance, called outside model execution and measured windows."""

import hashlib
import importlib.metadata
import inspect
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def provenance():
    root = Path(__file__).parents[1]
    source = root / "tt" / (os.environ.get("QWEN_DECODER", "functional") + "_decoder.py")
    candidate = os.environ.get("QWEN_FUSION_CANDIDATE")
    if candidate:
        source = root / "doc/fused_decoder/candidates" / (candidate + ".py")
    opt_candidate = os.environ.get("QWEN_OPT_CANDIDATE")
    if opt_candidate:
        source = root / "doc/optimized_decoder/candidates" / (opt_candidate + ".py")
    runner = Path(inspect.currentframe().f_back.f_code.co_filename)
    hf_source = importlib.metadata.distribution("transformers").locate_file(
        "transformers/models/qwen3_5/modeling_qwen3_5.py"
    )
    return {
        "decoder_implementation": os.environ.get("QWEN_DECODER", "functional"),
        "fusion_candidate": candidate,
        "optimization_candidate": opt_candidate,
        "optimization_policy": __import__("json").loads(os.environ.get("QWEN_OPT_POLICY", "{}")),
        "decoder_source": str(source),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": sys.argv,
        "python": sys.executable,
        "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "decoder_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "harness_source_sha256": hashlib.sha256(runner.read_bytes()).hexdigest() if runner.is_file() else None,
        "harness_source": str(runner),
        "hf_reference_source_sha256": hashlib.sha256(hf_source.read_bytes()).hexdigest(),
        "dependency_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "hf_config_sha256": hashlib.sha256((root / "doc/hf_config.json").read_bytes()).hexdigest(),
        "execution_regime": "serial, single 1x1 mesh on physical chip 0",
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_TRACEBACKS",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT",
                "TT_METAL_OPERATION_TIMEOUT_SECONDS",
                "TT_VISIBLE_DEVICES",
            )
        },
    }


def decoder_class():
    implementation = os.environ.get("QWEN_DECODER", "functional")
    if implementation == "optimized" and os.environ.get("QWEN_OPT_CANDIDATE"):
        import importlib

        name = os.environ["QWEN_OPT_CANDIDATE"]
        assert name.isidentifier(), name
        return importlib.import_module(
            "models.autoports.qwen_qwen3_6_27b.doc.optimized_decoder.candidates." + name
        ).OptimizedDecoder
    if implementation == "optimized":
        from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder

        return OptimizedDecoder
    if implementation == "fused" and os.environ.get("QWEN_FUSION_CANDIDATE"):
        import importlib

        name = os.environ["QWEN_FUSION_CANDIDATE"]
        assert name.isidentifier(), name
        return importlib.import_module(
            "models.autoports.qwen_qwen3_6_27b.doc.fused_decoder.candidates." + name
        ).FusedDecoder
    if implementation == "fused":
        from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder

        return FusedDecoder
    if implementation != "functional":
        raise ValueError(implementation)
    from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder

    return FunctionalDecoder


def decoder_kwargs():
    import json

    return (
        {"policy": json.loads(os.environ.get("QWEN_OPT_POLICY", "{}"))}
        if os.environ.get("QWEN_DECODER") == "optimized"
        else {}
    )
