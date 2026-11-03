# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Host-only timing of the actual patched plugin; not serving measurements."""
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["VLLM_PLUGINS"] = ""

import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler


def main():
    root = Path(__file__).resolve().parents[1]
    source = root.parents[2].parent / "vllm-tt-plugin/src/vllm_tt_plugin/host_sampler.py"
    spec = importlib.util.spec_from_file_location("k2_repair_sampler", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(8)
    torch.manual_seed(482)
    rows = []
    for sigma in (1.0, 5.0, 8.0):
        logits = torch.randn(32, 250624) * sigma
        p = torch.full((32,), 0.95)
        native = TopKTopPSampler()
        patched = module.TTTopKTopPSampler()
        timings = {"native": [], "patched": []}
        for iteration in range(12):
            for label, sampler in (("native", native), ("patched", patched)):
                began = time.perf_counter()
                sampler.forward_native(logits.clone(), {}, None, p)
                duration = time.perf_counter() - began
                if iteration >= 2:
                    timings[label].append(duration)
        rows.append(
            dict(
                sigma=sigma,
                seconds=timings,
                median_speedup=statistics.median(timings["native"]) / statistics.median(timings["patched"]),
            )
        )
    imported_device = any(name == "ttnn" or name.startswith("ttnn.") for name in sys.modules)
    if imported_device:
        raise RuntimeError("Host experiment unexpectedly imported TTNN")
    result = dict(
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        batch=32,
        vocabulary=250624,
        top_p=0.95,
        top_k=None,
        request_generators={},
        ttnn_imported=False,
        measurements=rows,
        limitation="Synthetic host logits; does not prove live serving throughput or completion within one hour.",
    )
    output = root / "doc/benchmark/recovery/patched_plugin_sampling.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
