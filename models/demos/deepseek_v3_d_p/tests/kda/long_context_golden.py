# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long FP64 KDA golden: the reference layer run segment by segment on CPU, queryable by window.

A golden over a long context (up to 1M tokens) takes hours of CPU, so the sequence is processed in fixed-size
segments. After each segment the run atomically writes that segment's output rows and the full carried state
(recurrent and convolution histories) after it, so an interrupted run resumes from the last complete segment
and any window can be read back with the exact state before it. The run never opens a device: device tests only
load a prepared golden, so the CPU work never holds the device.

``manifest.json`` is the golden's only metadata: what produced it (identity, provenance of each run session),
which segments are complete and how long each took, and, once complete, the token floor for per-token NRMSE.
Example (Kimi-K3 ``layers.0`` on the code tokens of a vLLM trace)::

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kda.long_context_golden \\
        --checkpoint $KIMI_K3_CKPT --tokens /path/to/vllm/trace --count 1044480 --out /path/to/golden
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import golden_floor

MANIFEST = "manifest.json"


def _segment_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"segment_{index:06d}.pt"


def _atomic_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    temporary = run_dir / f"{MANIFEST}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    temporary.replace(run_dir / MANIFEST)


def _provenance() -> dict:
    """What determines the golden's last bits besides its identity: code, torch, and the CPU backend."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return {
        "code_revision": revision,
        "torch": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "machine": platform.machine(),
    }


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(torch.as_tensor(token_ids, dtype=torch.int64).numpy().tobytes()).hexdigest()


def run_golden(
    *,
    run_dir: Path,
    token_ids: Sequence[int],
    layer_input: Callable[[Sequence[int]], torch.Tensor],
    weights: dict[str, torch.Tensor],
    config: KDAConfig,
    segment: int,
    dtype: torch.dtype,
    identity: dict,
) -> dict:
    """Run (or resume) the golden over ``token_ids`` and return the manifest.

    ``identity`` names what produced the inputs and weights. A resume refuses a directory whose identity,
    tokens, segment size, dtype or thread count differ: its snapshots would not continue this run, and the
    thread count changes FP64 summation order, so enforcing it keeps a resumed golden bit-identical to an
    uninterrupted one. Each run session's provenance is recorded with the segment it started from.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "identity": identity,
        "config": asdict(config),
        "tokens": len(token_ids),
        "token_ids_sha256": token_ids_sha256(token_ids),
        "segment": segment,
        "dtype": str(dtype),
        "threads": torch.get_num_threads(),
    }
    manifest_path = run_dir / MANIFEST
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatched = sorted(key for key, value in expected.items() if manifest.get(key) != value)
        if mismatched:
            raise ValueError(f"{run_dir} holds a different run; mismatched fields: {mismatched}")
    else:
        manifest = {**expected, "sessions": [], "segments": []}

    segments = -(-len(token_ids) // segment)
    done = len(manifest["segments"])
    if done < segments:
        manifest["sessions"].append({"first_segment": done, **_provenance()})
        _write_manifest(run_dir, manifest)
    state = None
    if done:
        state = KDAReferenceState(**torch.load(_segment_path(run_dir, done - 1), weights_only=True)["state"])
        print(f"resuming at segment {done}/{segments} (token {manifest['segments'][-1]['end']})", flush=True)

    for index in range(done, segments):
        start, end = index * segment, min((index + 1) * segment, len(token_ids))
        began = time.perf_counter()
        output, state = kda_forward_reference(layer_input(token_ids[start:end]), weights, config, state, dtype=dtype)
        _atomic_save(
            {"start": start, "end": end, "output": output[0], "state": vars(state)}, _segment_path(run_dir, index)
        )
        seconds = time.perf_counter() - began
        manifest["segments"].append({"start": start, "end": end, "seconds": seconds})
        _write_manifest(run_dir, manifest)
        print(f"segment {index + 1}/{segments} tokens [{start}, {end}) {seconds:.2f} s", flush=True)

    if "token_floor" not in manifest:
        manifest["token_floor"] = golden_floor(
            torch.cat(
                [
                    torch.load(_segment_path(run_dir, index), weights_only=True)["output"].norm(dim=-1)
                    for index in range(segments)
                ]
            )
        )
        _write_manifest(run_dir, manifest)
    return manifest


def load_window(run_dir: Path, start: int, end: int) -> tuple[torch.Tensor, KDAReferenceState | None]:
    """Golden output rows ``[start, end)`` and the state before ``start`` (``None`` at token 0).

    ``start`` must fall on a segment boundary, since states are saved only there.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    segment = manifest["segment"]
    if start % segment:
        raise ValueError(f"start {start} is not a multiple of the segment size {segment}")
    available = manifest["segments"][-1]["end"] if manifest["segments"] else 0
    if not start < end <= available:
        raise ValueError(f"window [{start}, {end}) is not inside the {available} golden tokens of {run_dir}")
    first = start // segment
    state = None
    if first:
        state = KDAReferenceState(**torch.load(_segment_path(run_dir, first - 1), weights_only=True)["state"])
    rows = []
    for index in range(first, -(-end // segment)):
        payload = torch.load(_segment_path(run_dir, index), weights_only=True)
        rows.append(
            payload["output"][
                max(start, payload["start"]) - payload["start"] : min(end, payload["end"]) - payload["start"]
            ]
        )
    return torch.cat(rows), state


def main() -> None:
    from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, resolve_trace_dir
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
    from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256, load_kda_layer_state_dict
    from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FIRST_LAYER_INPUT, FirstLayerInput

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Kimi-K3 checkpoint with layers.0 and embed")
    parser.add_argument("--tokens", type=Path, required=True, help="vLLM trace directory holding metadata.json")
    parser.add_argument("--count", type=int, required=True, help="number of tokens to run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--segment", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16, help="CPU threads; a resume must use the same count")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    trace = resolve_trace_dir(args.tokens)
    token_ids = load_trace_token_ids(trace)
    if len(token_ids) < args.count:
        raise ValueError(f"{trace} has {len(token_ids)} tokens, fewer than --count {args.count}")
    config = kimi_k3_kda_config()
    weights = load_kda_layer_state_dict(args.checkpoint, 0, config)
    layer_input = FirstLayerInput(args.checkpoint)
    identity = {
        "model": "Kimi-K3",
        "layer": "model.layers.0",
        "weights_sha256": kda_state_dict_sha256(weights),
        "input": FIRST_LAYER_INPUT,
        "input_weights_sha256": layer_input.weights_sha256(),
        "token_source": f"vllm trace {trace.name}",
    }
    run_golden(
        run_dir=args.out,
        token_ids=token_ids[: args.count],
        layer_input=layer_input,
        weights=weights,
        config=config,
        segment=args.segment,
        dtype=torch.float64,
        identity=identity,
    )


if __name__ == "__main__":
    main()
