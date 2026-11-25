# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long FP64 KDA golden: the reference layer run segment by segment on CPU, queryable by window.

A golden over a long context (up to 1M tokens) takes hours of CPU, so the sequence is processed in fixed-size
segments. After each segment the run atomically writes that segment's output rows and the full carried state
(recurrent and convolution histories) after it, so an interrupted run resumes from the last complete segment
and any window can be read back with the exact state before it. The run never opens a device: device tests only
load a prepared golden, so the CPU work never holds the device.

``manifest.json`` is the golden's only metadata, in the long-context golden registry's format
(``tests/accuracy/golden_registry.py``; the golden's id is its directory name): what produced it (producer,
identity, provenance of each run session), the prompt (``token_ids.pt``), which segments are complete and how long
each took, the products, and, once complete, the token floor for per-token NRMSE. Example (Kimi-K3 ``layers.0``
on the code tokens of a vLLM trace, written straight into the registry as golden ``k3-l0-kda-fp64-1m``)::

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kda.long_context_golden \\
        --checkpoint $KIMI_K3_CKPT --tokens /path/to/vllm/trace --count 1044480 \\
        --out /mnt/weka/model-cache/scratch/accuracy-goldens/k3-l0-kda-fp64-1m
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.accuracy.golden_registry import (
    MANIFEST,
    Golden,
    make_golden_dir,
    manifest_problems,
    open_golden,
    product_schema,
    publish,
    token_ids_sha256,
    write_manifest,
    write_token_ids,
)
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import golden_floor

PRODUCER = "models.demos.deepseek_v3_d_p.tests.kda.long_context_golden"
PRODUCER_VERSION = 1
# Kimi-K3 layers.0 over the first 1,044,480 code tokens of the k3_4_layers_moe_kda_io vLLM trace.
K3_LAYER0_GOLDEN = "k3-l0-kda-fp64-1m"


def _segment_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"segment_{index:06d}.pt"


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
    make_golden_dir(run_dir)
    expected = {
        "golden_id": run_dir.name,
        "producer": PRODUCER,
        "producer_version": PRODUCER_VERSION,
        "identity": identity,
        "config": asdict(config),
        "rows": len(token_ids),
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
        write_token_ids(run_dir, token_ids)
        manifest = {**expected, "sessions": [], "segments": []}

    segments = -(-len(token_ids) // segment)
    done = len(manifest["segments"])
    if done < segments:
        manifest["sessions"].append({"first_segment": done, **_provenance()})
        write_manifest(run_dir, manifest)
    state = None
    if done:
        state = KDAReferenceState(**torch.load(_segment_path(run_dir, done - 1), weights_only=True)["state"])
        print(f"resuming at segment {done}/{segments} (token {manifest['segments'][-1]['end']})", flush=True)

    for index in range(done, segments):
        start, end = index * segment, min((index + 1) * segment, len(token_ids))
        began = time.perf_counter()
        output, state = kda_forward_reference(layer_input(token_ids[start:end]), weights, config, state, dtype=dtype)
        payload = {"start": start, "end": end, "output": output[0], "state": vars(state)}
        publish(lambda path: torch.save(payload, path), _segment_path(run_dir, index))
        if "products" not in manifest:
            manifest["products"] = {
                "output": product_schema("row", output[0]),
                **{f"state.{name}": product_schema("segment_end", t) for name, t in vars(state).items()},
            }
        seconds = time.perf_counter() - began
        manifest["segments"].append({"start": start, "end": end, "seconds": seconds})
        write_manifest(run_dir, manifest)
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
        write_manifest(run_dir, manifest)
    problems = manifest_problems(run_dir.name, manifest)
    if problems:
        raise ValueError(f"the golden in {run_dir} breaks the registry contract: {problems}")
    return manifest


def open_k3_layer0_golden() -> Golden:
    """The registered Kimi-K3 ``layers.0`` golden, checked to be this producer's, on the first-layer input."""
    from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FIRST_LAYER_INPUT

    golden = open_golden(K3_LAYER0_GOLDEN)
    golden.require(
        producer=PRODUCER,
        producer_version=PRODUCER_VERSION,
        identity={"model": "Kimi-K3", "layer": "model.layers.0", "input": FIRST_LAYER_INPUT},
        products=["output", "state.recurrent"],
    )
    return golden


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


def state_after(run_dir: Path, end: int) -> KDAReferenceState:
    """The golden state after token ``end``, which must be the end of a complete segment."""
    run_dir = Path(run_dir)
    ends = [segment["end"] for segment in json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))["segments"]]
    if end not in ends:
        raise ValueError(f"token {end} is not the end of a complete segment of the golden in {run_dir}")
    return KDAReferenceState(**torch.load(_segment_path(run_dir, ends.index(end)), weights_only=True)["state"])


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
