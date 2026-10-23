# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated FP32 TT Euler/CFG replay against actual native CUDA step captures."""
import argparse
import os
import json
from pathlib import Path
import torch
from models.experimental.moss_image.tt.scheduler import guided_euler_step
from models.experimental.moss_image.validation.dit_validate import readings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", required=True)
    p.add_argument("--device-id", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not os.environ.get("TT_VISIBLE_DEVICES"):
        p.error("Set TT_VISIBLE_DEVICES to the reserved PCI BDF(s) before hardware initialization")
    import ttnn as t

    root = Path(args.capture)

    def load(name):
        return torch.load(root / (name + ".pt"), weights_only=True, map_location="cpu")

    manifest = json.loads((root / "manifest.json").read_text())
    mesh = t.open_mesh_device(
        mesh_shape=t.MeshShape(1, 1), physical_device_ids=[args.device_id], l1_small_size=32768, trace_region_size=0
    )
    try:

        def packed(value):
            return value.flatten(2).transpose(1, 2).contiguous()

        def upload(value):
            return t.from_torch(
                packed(value), dtype=t.float32, layout=t.TILE_LAYOUT, device=mesh, memory_config=t.DRAM_MEMORY_CONFIG
            )

        actual, velocity = guided_euler_step(
            upload(load("steps/000/input_latents")),
            upload(load("steps/000/positive_prediction")),
            upload(load("steps/000/negative_prediction")),
            manifest["guidance_scale"],
            manifest["steps"][0]["next_sigma"] - manifest["steps"][0]["sigma"],
        )
        report = {"activation_injection": True, "results": {}}
        for name, value, reference in [
            ("guided_velocity", velocity, "steps/000/guided_prediction"),
            ("updated_latents", actual, "steps/000/updated_latents"),
        ]:
            report["results"][name] = readings(t.to_torch(t.get_device_tensors(value)[0]), packed(load(reference)))
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if any(r["pcc"] < 0.999999 or r["relative_rms"] > 1e-5 for r in report["results"].values()):
            raise RuntimeError("FP32 scheduler mismatch")
    finally:
        t.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
