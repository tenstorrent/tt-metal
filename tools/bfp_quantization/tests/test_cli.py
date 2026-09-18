# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import torch
from tt_bfp_quant import gptq_search, search_linear


def test_cli_exports_both_methods_with_metadata(tmp_path):
    torch.manual_seed(19)
    w = torch.randn(40, 17)
    h = torch.eye(17)
    wp, hp = tmp_path / "w.pt", tmp_path / "h.pt"
    torch.save(w, wp)
    torch.save(h, hp)
    for method in ("max-minus-one", "gptq-search"):
        out = tmp_path / (method + ".pt")
        command = [
            sys.executable,
            "-m",
            "tt_bfp_quant.cli",
            "quantize",
            "--weight",
            str(wp),
            "--output",
            str(out),
            "--method",
            method,
            "--backend",
            "numpy",
            "--threads",
            "2",
        ]
        if method == "gptq-search":
            command += ["--hessian", str(hp)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        actual = torch.load(out, weights_only=True)
        expected = (
            gptq_search(w, h, backend="numpy")[0] if method == "gptq-search" else search_linear(w, backend="numpy")[0]
        )
        assert actual.dtype == torch.bfloat16 and torch.equal(actual.float(), expected)
        info = json.loads(out.with_suffix(".pt.json").read_text())
        assert info["validation"]["numerical_repacking_exact"]
        assert subprocess.run(command, capture_output=True).returncode != 0
