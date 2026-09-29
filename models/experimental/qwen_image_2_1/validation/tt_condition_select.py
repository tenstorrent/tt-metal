# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Check device-side selection of timestep and t=0 conditioning rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    args = parser.parse_args()
    base = args.cuda_dir / "step_000/transformer"
    mask = torch.load(base / "transformer_blocks.0/input/target_token_mask.pt", weights_only=True)
    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_dit_components import select_rows_by_mask, to_device, to_host
    from models.experimental.qwen_image_2_1.tt.tt_block import select_modulation_device

    device = None
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        for name in ("time_text_embed", "modulation"):
            rows = torch.load(base / f"{name}.pt", weights_only=True)
            expected = torch.where(mask.reshape(1, -1, 1), rows[:1, None], rows[1:2, None])
            selected = select_rows_by_mask(to_device(rows, device), mask, rows.shape[-1], device)
            actual = to_host(selected, tuple(expected.shape))
            error = (actual.float() - expected.float()).abs()
            print(
                json.dumps(
                    {
                        "component": name,
                        "shape": list(actual.shape),
                        "equal": bool(torch.equal(actual, expected)),
                        "max_abs_error": float(error.max()),
                    }
                ),
                flush=True,
            )
            if name == "modulation":
                chunks = select_modulation_device(to_device(rows, device), mask, device)
                for index, chunk in enumerate(chunks):
                    expected_chunk = expected[..., index * 4096 : (index + 1) * 4096]
                    actual_chunk = to_host(chunk, tuple(expected_chunk.shape))
                    print(
                        json.dumps(
                            {
                                "component": f"modulation_chunk_{index}",
                                "equal": bool(torch.equal(actual_chunk, expected_chunk)),
                            }
                        ),
                        flush=True,
                    )
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
