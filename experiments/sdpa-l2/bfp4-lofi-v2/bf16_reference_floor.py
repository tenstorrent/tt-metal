# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only BF16 output-rounding floors for existing sampled-Q device records.

Regenerates the original inputs/reference; does not download or change outputs.
Compare with a record's original L2/absolute RMS, not a different sample scope.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import fullchip as F

HERE = Path(__file__).resolve().parent
# Verified locally with git show637d956:tests/.../repro_sdpa_l2.py | shasum-a256.
# The remote experimental checkout receives source files, not local Git objects.
CHECKPOINT_REFERENCE_SHA = "8a78370d89e0a11f22e1d835b53b88141cb81b6b7e1a3e963228ea365631ae2d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("records", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    path = HERE / (args.label + ".jsonl")
    assert not path.exists()
    reference_path = Path(F.REPRO.__file__)
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    cache = {}
    with path.open("x") as stream:
        for name in args.records:
            original_path = HERE / name
            record = json.loads(original_path.read_text())
            expected_sha = record["source_sha256"].get(str(reference_path.relative_to(F.ROOT)))
            reference_pin_scope = "Pinned directly by original device record"
            if expected_sha is None:
                # Some early drivers omitted their imported reference helper.
                # Do not pretend those records contained a pin retroactively.
                expected_sha = CHECKPOINT_REFERENCE_SHA
                reference_pin_scope = "Original record omitted reference pin; current helper matches hash independently verified from local locked637d956 checkpoint"
            assert expected_sha == reference_sha
            rows = record["sampled_query_rows"]
            key = (record["length"], record["heads"], record["seed"], record["distribution"],
                   record.get("common_mode", 32), tuple(rows))
            if key not in cache:
                inputs = F.REPRO.make_inputs(key[1], key[0], key[0], 128, key[2], key[3], key[4])
                reference = F.REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
                rounded = reference.bfloat16()
                mean_v = inputs[2].mean(dim=2, keepdim=True, dtype=torch.float64)
                residual = reference - mean_v
                error = rounded.double() - reference
                constant = torch.equal(inputs[2], inputs[2][:, :, :1].expand_as(inputs[2]))
                cache[key] = dict(original=F.REPRO.metrics(rounded, reference),
                    centered_l2_pct=None if constant or residual.norm() == 0 else float(100 * error.norm() / residual.norm()),
                    centered_reference_rms=0.0 if constant else float(residual.square().mean().sqrt()),
                    absolute_error_rms=float(error.square().mean().sqrt()),
                    absolute_error_max=float(error.abs().max()),
                    rounded_reference_unique_values=int(rounded.unique().numel()),
                    rounded_reference_sha256=hashlib.sha256(rounded.view(torch.uint16).numpy().tobytes()).hexdigest(),
                    original_input_sha256=[hashlib.sha256(x.view(torch.uint16).numpy().tobytes()).hexdigest() for x in inputs])
                del inputs, reference, rounded
            result = dict(record=name, record_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                          scope="Same original BF16 inputs, all KV and exact sampled Q rows as device record; FP64 reference rounded to BF16",
                          floor=cache[key], reference_source_sha256=reference_sha,
                          reference_pin_scope=reference_pin_scope,
                          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            line = json.dumps(result, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)


if __name__ == "__main__":
    main()
