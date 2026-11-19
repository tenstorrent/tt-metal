# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Current optimized-path stress using the unchanged fresh-HF fused harness."""

import argparse
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import stress_fused_decoder as stress
from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder


def assert_optimized_path():
    expected = Path(__file__).parents[1] / "tt" / "optimized_decoder.py"
    assert Path(inspect.getfile(OptimizedDecoder)).resolve() == expected.resolve()
    assert Path(inspect.getfile(OptimizedDecoder.from_state_dict)).resolve() == expected.resolve()
    assert OptimizedDecoder.__name__ == "OptimizedDecoder"
    return expected.resolve()


def run(args):
    source = assert_optimized_path()
    # The stage contract gates whole-output PCC; retain every per-row diagnostic.
    args.whole_output_gate = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), "Use a fresh output path so failed setup cannot reuse stale evidence"
    with patch.object(stress, "FusedDecoder", OptimizedDecoder), patch.dict(os.environ, QWEN_DECODER="optimized"):
        assert stress.FusedDecoder is OptimizedDecoder
        try:
            stress.run(args)
        finally:
            if output.exists():
                data = json.loads(output.read_text())
                assert data["decoder"] == "OptimizedDecoder", "Stress exercised the wrong decoder"
                data["measured_decoder_source"] = str(source)
                data["implementation_selection"] = "explicit OptimizedDecoder; policy: " + getattr(
                    args, "build_kwargs", "{}"
                )
                output.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-length", type=int, default=128)
    parser.add_argument("--continuation-length", type=int, default=33)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-cache")
    parser.add_argument("--build-kwargs", default="{}")
    parser.add_argument("--activation-file")
    parser.add_argument("--clean-performance", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
