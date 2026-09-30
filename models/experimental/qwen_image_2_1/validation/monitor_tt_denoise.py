# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Mirror a native TT run and collect its completed VAE image for the browser pane.

This monitor does not execute inference or start a remote workload. Remote host
and path arguments are restricted before passing them to SSH and rsync.
"""

from __future__ import annotations

import argparse
import json
import shutil
import re
import subprocess
import time
from pathlib import Path


def preview_steps(total_steps: int) -> tuple[int, ...]:
    """Capture a few evenly spread images, always including the final step."""
    if total_steps < 2:
        raise ValueError("a valid Qwen Image 2.1 schedule has at least two steps")
    return tuple(sorted({int((total_steps - 1) * fraction) for fraction in (0, 0.125, 0.25, 0.5, 0.75, 1)}))


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"exit {result.returncode}: {command[0]}")
    return result


def _remote_progress(host: str, remote_root: str) -> dict[str, object]:
    result = _run(
        [
            "timeout",
            "20",
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            host,
            "timeout",
            "--foreground",
            "12",
            "cat",
            remote_root + "/progress.json",
        ],
        25,
    )
    return json.loads(result.stdout)


def _fetch_file(host: str, remote_root: str, relative: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    _run(
        [
            "timeout",
            "45",
            "rsync",
            "-a",
            "--timeout=35",
            "--rsync-path=timeout --foreground 40 rsync",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=8",
            f"{host}:{remote_root}/{relative}",
            str(temporary),
        ],
        50,
    )
    temporary.replace(destination)


def _fetch_latent(host: str, remote_root: str, step: int, destination: Path) -> None:
    _fetch_file(host, remote_root, f"step_{step:03d}/updated_latents.pt", destination)


def _collect_tt_vae(args, total_steps: int) -> None:
    """Publish the native TT image; pair the CUDA image only for an identical latent."""
    import torch
    from models.experimental.qwen_image_2_1.validation.tt_vae_decode import metrics

    target = args.output_dir / "integrated"
    target.mkdir(exist_ok=True)
    for name in ("decoded_image.pt", "vae_progress.json"):
        _fetch_file(args.host, args.remote_root, name, target / name)
    latent = target / "updated_latents.pt"
    _fetch_latent(args.host, args.remote_root, total_steps - 1, latent)
    if args.vae_reference:
        actual_latent = torch.load(latent, map_location="cpu", weights_only=True)
        expected_latent = torch.load(args.vae_reference / "packed_latents.pt", map_location="cpu", weights_only=True)
        identical = torch.equal(actual_latent, expected_latent)
        report = {"same_latent_as_cuda_vae_reference": identical}
        if identical:
            actual = torch.load(target / "decoded_image.pt", map_location="cpu", weights_only=True)
            expected = torch.load(args.vae_reference / "output.pt", map_location="cpu", weights_only=True)
            report.update(metrics(actual, expected))
            shutil.copyfile(args.vae_reference / "cuda_vae.png", target / "cuda_vae.png")
        (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Publish the PNG last so the viewer sees a complete comparison.
    _fetch_file(args.host, args.remote_root, "tt_vae.png", target / "tt_vae.png")
    print(f"native TT VAE image saved: {target / 'tt_vae.png'}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH alias or hostname of the TT runner")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument(
        "--vae-reference", type=Path, help="independent CUDA VAE capture for identical-latent comparison"
    )
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.host) is None:
        parser.error("--host must be a plain SSH alias or hostname")
    if re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_root) is None:
        parser.error("--remote-root must be an absolute path without shell metacharacters")
    manifest = json.loads((args.cuda_dir / "manifest.json").read_text())
    total_steps = manifest["steps"]
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_connection_error = None
    while True:
        try:
            progress = _remote_progress(args.host, args.remote_root)
            temporary = args.output_dir / "progress.json.tmp"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(args.output_dir / "progress.json")
            if last_connection_error:
                print("TT progress connection restored", flush=True)
                last_connection_error = None
            if progress["total_steps"] != total_steps:
                raise ValueError("remote progress step count differs from the CUDA reference")
            if progress["status"] in {"complete", "failed"}:
                if progress["status"] == "complete":
                    _collect_tt_vae(args, total_steps)
                print(f"TT run ended: {progress['status']}", flush=True)
                return
        except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as error:
            message = str(error)
            if message != last_connection_error:
                print(f"monitor waiting: {message}", flush=True)
                last_connection_error = message
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
