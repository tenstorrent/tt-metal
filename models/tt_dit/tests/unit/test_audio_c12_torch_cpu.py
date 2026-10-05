# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optional dependency-complete CPU check; exits nonzero if Torch is missing.

Run directly with --upstream /path/to/run106/evidence/upstream-audio_pack.py.
Only the two pinned, device-free reference functions are AST-extracted. Never
imports TTNN or opens a device. This remains a CPU, not native-arithmetic test.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    data = args.upstream.read_bytes()
    expected = "7dfb8e41fbe7c42ce3c058e34bdab65a185a449709d2619063a32138485ed014"
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("upstream impulse reference SHA256 mismatch")
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("c12", Path(__file__).resolve().parents[2] / "utils/c12.py")
    c12 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = c12
    spec.loader.exec_module(c12)
    tree = ast.parse(data)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("packed_weight", "conv1d_same")
    ]
    ns = {"torch": torch, "F": F}
    exec(compile(tree, str(args.upstream), "exec"), ns)
    generator = torch.Generator().manual_seed(52)
    weight = torch.randint(-3, 4, (24, 24, 7), generator=generator).double() / 8
    cases = 0
    for zero in (False, True):
        w = torch.zeros_like(weight) if zero else weight
        for pack in (1, 4):
            half = 3 if pack == 1 else 1
            transformed, _ = c12.transform_weight(w, None, pack)
            impulse = ns["packed_weight"](
                ns["conv1d_same"](w, 1), c_in=24, c_out=24, k_in=pack, k_out=pack, support=7, q_half=half
            )
            torch.testing.assert_close(transformed, impulse.double(), rtol=0, atol=0)
            cases += 1
            for batch in (1, 2):
                for length in (1, 3, 7, 124, 127, 128, 129, 132, 256):
                    x = torch.randint(-4, 5, (batch, 24, length), generator=generator).double() / 8
                    for bias in (None, torch.arange(24).double() / 16):
                        v, b = c12.transform_weight(w, bias, pack)

                        def evaluate(values):
                            padded = F.pad(values, (0, -length % pack))
                            packed = padded.transpose(1, 2).reshape(batch, -1, pack * 24).transpose(1, 2)
                            y = F.conv1d(packed, v, b, padding=half)
                            return y.transpose(1, 2).reshape(batch, -1, 24).transpose(1, 2)[..., :length]

                        expected_y = F.conv1d(x, w, bias, padding=3)
                        torch.testing.assert_close(evaluate(x), expected_y, rtol=0, atol=0)
                        torch.testing.assert_close(evaluate(-x), F.conv1d(-x, w, bias, padding=3), rtol=0, atol=0)
                        torch.testing.assert_close(evaluate(x), expected_y, rtol=0, atol=0)
                        cases += 1
    # Actual-helper Torch sharded convolution, global edges, and partial/multi-shard tails.
    v, bias = c12.transform_weight(weight, torch.arange(24).double() / 16, 4)
    negatives = 0
    for shards in (2, 4, 8):
        for tail in (0, 1, 3, 7, 127, 128, 129):
            length = shards * 128
            x = torch.randint(-4, 5, (2, 24, length), generator=generator).double() / 8
            if tail:
                x[..., -tail:] = 0
            packed = x.transpose(1, 2).reshape(2, -1, 96).transpose(1, 2)
            global_pad = F.pad(packed, (1, 1))
            good, bad = [], []
            for shard in range(shards):
                good.append(F.conv1d(global_pad[..., shard * 32 : shard * 32 + 34], v, bias))
                bad.append(F.conv1d(F.pad(packed[..., shard * 32 : (shard + 1) * 32], (1, 1)), v, bias))
            y = torch.cat(good, -1).transpose(1, 2).reshape(2, length, 24).transpose(1, 2)
            wrong = torch.cat(bad, -1).transpose(1, 2).reshape(2, length, 24).transpose(1, 2)
            reference = F.conv1d(x, weight, torch.arange(24).double() / 16, padding=3)
            torch.testing.assert_close(y, reference, rtol=0, atol=0)
            assert not torch.equal(wrong, reference)
            negatives += 1
            cases += 1
    print(
        json.dumps(
            {
                "torch_cpu_cases": cases,
                "wrong_seam_controls": negatives,
                "upstream_sha256": expected,
                "torch_version": torch.__version__,
                "arithmetic": "dyadic float64 Torch CPU; no native FP32 proof",
            }
        )
    )


if __name__ == "__main__":
    main()
