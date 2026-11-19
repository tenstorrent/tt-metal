# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only characterization of the installed HF router's equal-logit policy.

This is a diagnostic, not a replacement router or an acceptance threshold.
The C++ candidate follows the installed ATen TopKImpl.h algorithm through the
host standard library. It compares real layer0 batch32 fixture routes and
additional quantized synthetic logits. It never imports TTNN or opens a device.
"""

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_batch_intermediates import fixture

CPP = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
extern "C" void candidate(const float* input, int64_t* output, int rows) {
    using Item = std::pair<float, int64_t>;
    auto greater = [](const Item& a, const Item& b) {
        return (std::isnan(a.first) && !std::isnan(b.first)) || a.first > b.first;
    };
    std::vector<Item> queue(512);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < 512; ++col) queue[col] = {input[row * 512 + col], col};
        // ATen k=10,n=512 chooses nth_element (10*64 > 512).
        std::nth_element(queue.begin(), queue.begin() + 9, queue.end(), greater);
        // ATen deliberately excludes the nth item from its final sort.
        std::sort(queue.begin(), queue.begin() + 9, greater);
        for (int col = 0; col < 10; ++col) output[row * 10 + col] = queue[col].second;
    }
}
"""


def compare(expected, actual):
    return {
        "rows": expected.shape[0],
        "ordered_mismatch_rows": torch.where((expected != actual).any(-1))[0].tolist(),
        "set_mismatch_rows": torch.where((expected.sort(-1).values != actual.sort(-1).values).any(-1))[0].tolist(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    _, _, _, requests, hashes = fixture(args.snapshot, tuple(range(32)))
    logits, indices, labels = [], [], []
    for request, item in requests.items():
        for phase, values in item["phases"].items():
            logits.append(values["router_logits"].float())
            indices.append(values["router_indices"])
            labels.extend({"request": request, "phase": phase, "row": row} for row in range(logits[-1].shape[0]))
    logits, expected = torch.cat(logits), torch.cat(indices)
    with tempfile.TemporaryDirectory(prefix="qwen-router-ties-") as directory:
        source, library = Path(directory) / "candidate.cpp", Path(directory) / "candidate.so"
        source.write_text(CPP)
        subprocess.run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
        function = ctypes.CDLL(str(library)).candidate
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

        def candidate(value):
            value = value.float().contiguous()
            result = torch.empty((value.shape[0], 10), dtype=torch.int64)
            function(value.data_ptr(), result.data_ptr(), value.shape[0])
            return result

        probabilities = logits.softmax(-1)
        ordered = logits.topk(11, dim=-1).values
        ties = torch.where(ordered[:, 9] == ordered[:, 10])[0].tolist()
        report = {
            "scope": __doc__,
            "torch_version": torch.__version__,
            "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
            "aten_source": str(Path(torch.__file__).parent / "include/ATen/native/TopKImpl.h"),
            "candidate_source_sha256": hashlib.sha256(CPP.encode()).hexdigest(),
            "fixture_input_hashes": hashes,
            "real_probabilities": compare(expected, candidate(probabilities)),
            "real_logits": compare(expected, candidate(logits)),
            "real_torch_logits": compare(expected, logits.topk(10, dim=-1).indices),
            "boundary_ties": [{**labels[row], "hf_indices": expected[row].tolist()} for row in ties],
            "synthetic": {},
        }
        generator = torch.Generator().manual_seed(9827)
        for name, value in {
            "bf16_normal": torch.randn((4096, 512), generator=generator).bfloat16().float(),
            "coarse_ties": torch.randint(-4, 5, (4096, 512), generator=generator).float(),
            "all_equal": torch.ones((1, 512)),
        }.items():
            target = value.softmax(-1).topk(10, dim=-1).indices
            report["synthetic"][name] = {
                "logits": compare(target, candidate(value)),
                "probabilities": compare(target, candidate(value.softmax(-1))),
                "first_indices": target[0].tolist(),
            }
    assert "ttnn" not in sys.modules
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "fixture_input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
