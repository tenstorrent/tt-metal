# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small distinct-input fused LoFi qualification; CPU preprocessing excluded from timing."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lofi_streaming", HERE / "streaming.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    opts = parser.parse_args()
    path = HERE / (opts.label + ".jsonl")
    assert not path.exists()
    torch.set_num_threads(8)
    cases = [(n, s, "normal") for n in (4096, 32768, 262144) for s in (1240, 1241)]
    cases += [(32768, s, d) for s in (1240, 1241) for d in ("outliers", "scaled_qk", "scaled_down", "common_k", "common_v")]
    device = ttnn.open_device(device_id=0)
    try:
        with path.open("x") as output:
            for length, seed, distribution in cases:
                inputs = S.V2.V1_NUMERICS.FRONTIER.inputs_for(length, seed, distribution)
                ref = S.repro.reference(*inputs)
                for variant, mode in (("pervalue_rawp", "main"), ("pervalue_rawp", "fast"),
                                      ("pervalue_rawp", "fp32_hifi2_cheap"), ("q7kv8_rawp", "fp32_hifi2_cheap")):
                    label = f"{opts.label}-{length}-{seed}-{distribution}-{variant}-{mode}"
                    args = argparse.Namespace(label=label, variant=variant, mode=mode, length=length,
                                              seed=seed, distribution=distribution, q_repeats=1,
                                              k_chunks=length // 512, distinct_kv=True, iters=0, warmup=0)
                    record, actual = S.run(device, args, inputs, ref)
                    record["kind"] = "fused_attention"
                    output.write(json.dumps(record, allow_nan=False) + "\n")
                    output.flush()
                    print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        ttnn.close_device(device)
