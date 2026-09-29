# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""List the small CUDA capture subset needed by the TT denoiser validator.

Pass the resulting file to rsync's --files-from option. Full CUDA hook captures
can be gigabytes, while the TT runner needs only first-step token metadata,
per-step input/output/latent boundaries, and selected complete block chains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def selected_files(root: Path) -> list[str]:
    manifest = json.loads((root / "manifest.json").read_text())
    steps = manifest["steps"]
    if manifest["capture_steps"] != list(range(steps)):
        raise ValueError("the TT validator requires CUDA boundaries from every denoising step")
    files = [
        "manifest.json",
        "schedule.json",
        "step_000/transformer/input/hidden_states.pt",
        "step_000/transformer/input/encoder_hidden_states.pt",
    ]
    block_input = Path("step_000/transformer/transformer_blocks.0/input")
    files.extend(
        (block_input / name).as_posix()
        for name in (
            "target_token_mask.pt",
            "rotary_emb.pt",
            "segments.json",
        )
    )
    key_valid = block_input / "key_valid.pt"
    if (root / key_valid).is_file():
        files.append(key_valid.as_posix())
    full_steps = set(manifest.get("full_capture_steps", manifest["capture_steps"]))
    for step in range(steps):
        base = f"step_{step:03d}"
        files.extend(
            (
                f"{base}/transformer/input/timestep.pt",
                f"{base}/transformer/proj_out.pt",
                f"{base}/scheduler/updated_latents.pt",
            )
        )
        if step in full_steps:
            files.extend(f"{base}/transformer/transformer_blocks.{layer}.pt" for layer in range(32))
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CUDA capture is missing {len(missing)} required files: {missing[:5]}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = selected_files(args.cuda_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(files) + "\n")
    size = sum((args.cuda_dir / name).stat().st_size for name in files)
    print(f"listed {len(files)} files ({size / 1e6:.1f} MB) in {args.output}")


if __name__ == "__main__":
    main()
