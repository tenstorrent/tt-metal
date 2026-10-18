# SPDX-License-Identifier: Apache-2.0
"""Run isolated review candidates through the unchanged real-HF fused contract."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--layers", default="layer0 or layer1 or layer3")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    modules = {
        "weighted_reduce": "fused_weighted_reduce_candidate",
        "attn_weighted_reduce": "fused_weighted_reduce_candidate",
        "signed_sqrt": "review_fusion_candidates",
        "floor_position": "review_fusion_candidates",
        "indexed_cache": "fused_cache_candidate",
        "paged_decode_cache": "fused_cache_candidate",
        "fp32_norm": "fused_hc_precision_candidates",
        "stable_norm": "fused_hc_precision_candidates",
        "sequential_sum": "fused_hc_precision_candidates",
        "fp32_norm_sequential_sum": "fused_hc_precision_candidates",
    }
    module = importlib.import_module("models.autoports.qwen_qwen3_8_flash_next.tests." + modules[args.candidate])
    module.install(args.candidate)
    import pytest

    from models.autoports.qwen_qwen3_8_flash_next.tests import test_fused_decoder as suite

    root = Path(__file__).resolve().parents[1]
    output = root / "doc/fused_decoder" / args.tag
    output.mkdir(parents=True, exist_ok=False)
    source = root / "tt/fused_decoder.py"
    candidate = Path(module.__file__)
    (output / "fused_source.py.txt").write_bytes(source.read_bytes())
    (output / "candidate.py.txt").write_bytes(candidate.read_bytes())
    (output / "provenance.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                layers=args.layers,
                runtime_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                environment={k: v for k, v in os.environ.items() if k.startswith(("QWEN38_", "TT_METAL_"))},
            ),
            indent=2,
        )
        + "\n"
    )
    with patch.object(suite, "EVIDENCE", output):
        return pytest.main(["-s", "-q", "--timeout=0", str(root / "tests/test_fused_decoder.py"), "-k", args.layers])


if __name__ == "__main__":
    raise SystemExit(main())
