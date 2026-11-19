# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only cancellation diagnostic; the TF32 simulation is not a device oracle.

The exported real-weight core fixture is consumed without TTNN. This reports
reference head-row variance and emulates truncation only in transient dot and
outer products while retaining full precision state decay/accumulation. An
actual device's worst-row index must still be recorded before attributing its
PCC outlier to any particular reference row.
"""

import argparse
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_gdn_numerics import metrics, truncate_tf32


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    fixture = torch.load(args.fixture, weights_only=True)
    expected = fixture["output"].float().reshape(-1, 128)
    std = expected.std(-1)
    report = {
        "scope": "CPU-only expected core magnitude and hypothetical TF32 transient dots; not device row identification",
        "smallest_expected_std": [
            {
                "token": int(index) // 48,
                "head": int(index) % 48,
                "std": std[index].item(),
                "max_abs": expected[index].abs().max().item(),
            }
            for index in std.argsort()[:16]
        ],
    }
    state = torch.zeros_like(fixture["state"])
    outputs = []
    for index in range(fixture["q"].shape[1]):
        decayed = state * fixture["g"][:, index].exp()[..., None, None]
        key = truncate_tf32(fixture["k"][:, index].repeat_interleave(3, 1))
        query = fixture["q"][:, index].repeat_interleave(3, 1) * (128**-0.5)
        read = (truncate_tf32(decayed) * key[..., None]).sum(-2)
        delta = (fixture["v"][:, index] - read) * fixture["beta"][:, index, :, None]
        state = decayed + key[..., None] * truncate_tf32(delta)[..., None, :]
        outputs.append((truncate_tf32(state) * truncate_tf32(query)[..., None]).sum(-2))
    report["hypothetical_transient_tf32_core"] = metrics(fixture["output"], torch.stack(outputs, 1))
    report["hypothetical_transient_tf32_state"] = metrics(fixture["state"], state)
    token, head = divmod(report["hypothetical_transient_tf32_core"]["worst_row"]["flat_index"], 48)
    report.update(
        hypothetical_worst_token=token,
        hypothetical_worst_head=head,
        hypothetical_worst_log_decay=fixture["g"][0, token, head].item(),
        hypothetical_worst_beta=fixture["beta"][0, token, head].item(),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
