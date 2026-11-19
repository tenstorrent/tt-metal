# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-only GDN projection blocking candidate through the unchanged batch probe.

Accepts probe_batch's flags. Stores each BF16 checkpoint matrix once in
256-output parent blocks. Single-token passes use those blocks directly;
larger token blocks take device32-output slices. This preserves the exact
per-output FP32 product/addition tree while reducing projection dispatches
eightfold in decode. It changes neither batch size nor tensor precision.

The largest decode product is6MiB, versus24MiB for a32-token prefill block;
two half-slices can temporarily double these figures. The old full product
is released before allocating the reduction result. Persistent weight payload
is unchanged. No global/runtime source is edited by this diagnostic.
"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest import mock


def candidate_class(ttnn, base):
    class HybridGDNProjection(base):
        def __init__(self, loader, name):
            weight = loader.state_dict[name]
            self.output_width, self.input_width = weight.shape
            if (self.output_width, self.input_width) not in ((10240, 2560), (2560, 6144)):
                raise ValueError("Expected a real target GDN QKV/output weight")
            self.parent_block = 256
            self.weights = [
                loader.tensor(weight[start : start + 256].reshape(1, 1, 256, self.input_width))
                for start in range(0, self.output_width, 256)
            ]

        def _project_block(self, value):
            count = value.shape[-2]
            block = 256 if count == 1 else 32
            expanded = ttnn.reshape(ttnn.typecast(value, ttnn.float32), (1, count, 1, self.input_width))
            columns = []
            for parent in self.weights:
                for offset in range(0, self.parent_block, block):
                    weight = parent if block == self.parent_block else parent[:, :, offset : offset + block, :]
                    products = ttnn.multiply(expanded, ttnn.typecast(weight, ttnn.float32))
                    width, tails = self.input_width, []
                    while width > 1:
                        half = width // 2
                        if width % 2:
                            tails.append(products[..., -1:])
                        left, right = products[..., :half], products[..., half : 2 * half]
                        del products
                        products = ttnn.add(left, right)
                        del left, right
                        width = half
                    for tail in tails:
                        products = ttnn.add(products, tail)
                    columns.append(ttnn.reshape(products, (1, 1, count, block)))
            return ttnn.typecast(ttnn.concat(columns, dim=-1), ttnn.bfloat16)

    return HybridGDNProjection


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests import probe_batch
    from models.autoports.qwen_qwen3_8_flash_next.tt import decoder_components

    candidate = candidate_class(ttnn, decoder_components.PreciseGDNProjection)
    try:
        with mock.patch.object(decoder_components, "PreciseGDNProjection", candidate):
            probe_batch.main()
    finally:
        if args.output.exists():
            report = json.loads(args.output.read_text())
            report["diagnostic_projection_policy"] = {
                "source": str(Path(__file__).resolve()),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "persistent_output_block": 256,
                "single_token_output_block": 256,
                "multiple_token_output_block": 32,
                "persistent_weight_dtype": "BF16",
                "arithmetic_dtype": "FP32 SFPU",
                "no_persistent_weight_duplicates": True,
                "scope": __doc__,
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
