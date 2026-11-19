# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record backend verbose output for original real BF16 32-row projection."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .probe_qsa import load_reference
from .probe_qsa_fp64_indexer_control import digest, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    _, model, _, _, provenance = load_reference(args.snapshot)
    assert source["key"]["reference_sha256"] == provenance["reference_sha256"]
    assert source["key"]["config_sha256"] == provenance["config_sha256"]
    weight = model.indexer.index_qk_proj.weight
    records = []
    for start in [105568, 58528, 9184, 7840]:
        hidden = source["hidden"][:, start : start + 32]
        print(
            json.dumps(
                {
                    "literal_projection_begin": start,
                    "hidden_shape": list(hidden.shape),
                    "weight_shape": list(weight.shape),
                }
            ),
            flush=True,
        )
        output = F.linear(hidden, weight)
        repeat = F.linear(hidden, weight)
        assert torch.equal(output, repeat)
        records.append({"start": start, "repeat_exact": True})
        print(json.dumps({"literal_projection_end": start}), flush=True)
    assert "ttnn" not in sys.modules
    write_json(
        args.output,
        {
            "scope": __doc__,
            "accepted": False,
            "device_executed": False,
            "source_key": source["key"],
            "harness_sha256": digest(Path(__file__)),
            "source_sha256": digest(args.source_cache),
            "reference": provenance,
            "verbose_environment": {name: os.environ.get(name) for name in ["ONEDNN_VERBOSE", "DNNL_VERBOSE"]},
            "torch_version": torch.__version__,
            "torch_build_config": torch.__config__.show(),
            "records": records,
            "backend_identity_authority": "raw verbose lines between recorded markers; inspect log, no inferred ISA",
        },
    )


if __name__ == "__main__":
    main()
