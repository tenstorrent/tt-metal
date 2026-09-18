# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Runnable CPU example: collect once, quantize, score separate held-out inputs."""
import json
import torch

from tt_bfp_quant import capture_linear_inputs, factor_hessian, gptq_search, search_linear, validate_repacking


def main():
    torch.manual_seed(7)
    torch.set_num_threads(4)
    model = torch.nn.Sequential(torch.nn.Linear(65, 80, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.normal_(mean=0, std=0.02)
    mixing = torch.randn(12, 65)

    def inputs(n):
        return torch.randn(n, 12) @ mixing + 0.15 * torch.randn(n, 65)

    train, test = inputs(2048), inputs(1024)
    with capture_linear_inputs(model, ["0"]) as stats, torch.inference_mode():
        for batch in train.split(256):
            model(batch)
    weight = model[0].weight.detach()
    # Two output shards of width 40: reset/pad BFP groups at each boundary.
    splits = [40, 40]
    factor = factor_hessian(stats["0"].value())
    variants = {
        "ordinary BFP4": search_linear(weight, 4, (0,), output_splits=splits),
        "max/max-1 BFP4": search_linear(weight, 4, output_splits=splits),
        "GPTQ + search BFP4": gptq_search(weight, factor=factor, output_splits=splits),
        "max/max-1 BFP8": search_linear(weight, 8, output_splits=splits),
    }
    records = []
    for name, (q, info) in variants.items():
        validate_repacking(q, info["bits"], output_splits=splits)
        records.append(
            {
                "method": name,
                "heldout_output_mse": float(((test @ (weight - q).T) ** 2).mean()),
                "seconds": info["seconds"],
                "backend": info["backend"],
            }
        )
    print(
        json.dumps(
            {
                "calibration_rows": stats["0"].count,
                "heldout_rows": len(test),
                "factor_seconds": factor.seconds,
                "results": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
