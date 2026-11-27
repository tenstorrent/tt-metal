# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 ``layers.0`` KDA FP64 golden producer: the reference layer run segment by segment on CPU.

A golden over a long context (up to 1M tokens) takes hours of CPU, so the sequence is processed in fixed-size
segments. After each segment the run atomically writes that segment's output rows and the full carried state
(recurrent and convolution histories) after it (``segment_path``), so an interrupted run resumes from the last
complete segment and any window can be read back with the exact state before it (the reader: ``golden.py`` next to
this file). The run never opens a device: device tests only load a prepared golden, so the CPU work never holds the
device.

``manifest.json`` is the golden's only metadata, in the long-context golden registry's format
(``common/prefill/long_context_accuracy/fp64_golden.py``; the golden's id is its directory name): what produced it
(producer, identity, provenance of each run session), the prompt (``token_ids.pt``), which segments are complete and how
long each took, the products, and, once complete, the token floor for per-token NRMSE.

The manifest's ``producer`` is ``PRODUCER``, ``models.demos.deepseek_v3_d_p.tests.kda.long_context_golden``: this
module's path when it wrote the registered golden ``k3-l0-kda-fp64-1m`` (version 1). It names the producer's
numerics, not where the module lives now, so it stays unchanged while the bytes do; the reader requires it.

Example (Kimi-K3 ``layers.0`` on the code tokens of a vLLM trace, written straight into the registry as golden
``k3-l0-kda-fp64-1m``; ``--segments N`` stops after N segments, and a later run resumes)::

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden_producer \\
        --checkpoint $KIMI_K3_CKPT --tokens /path/to/vllm/trace --count 1044480 --segment 1024 --threads 32 \\
        --out /mnt/weka/model-cache/scratch/accuracy-goldens/k3-l0-kda-fp64-1m
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    manifest_problems,
    product_schema,
    publish,
    start_or_resume,
    token_ids_sha256,
    write_manifest,
)
from models.demos.common.prefill.long_context_accuracy.windowed_errors import golden_floor
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference

PRODUCER = "models.demos.deepseek_v3_d_p.tests.kda.long_context_golden"
PRODUCER_VERSION = 1
# Kimi-K3 layers.0 over the first 1,044,480 code tokens of the k3_4_layers_moe_kda_io vLLM trace.
K3_LAYER0_GOLDEN = "k3-l0-kda-fp64-1m"


def segment_path(run_dir: Path, index: int) -> Path:
    """Segment ``index``'s file: ``{"start", "end", "output": [rows, hidden], "state": {name: tensor}}``, the state
    after the segment."""
    return Path(run_dir) / f"segment_{index:06d}.pt"


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
    max_segments: int | None = None,
) -> dict:
    """Run (or resume) the golden over ``token_ids`` and return the manifest; with ``max_segments``, stop once that
    many segments are complete (an incomplete golden a later run resumes).

    ``identity`` names what produced the inputs and weights. A resume refuses a directory whose identity, tokens,
    segment size, dtype or thread count differ (``fp64_golden.start_or_resume``).
    """
    run_dir = Path(run_dir)
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
    segments = -(-len(token_ids) // segment)
    stop = segments if max_segments is None else min(segments, max_segments)
    manifest = start_or_resume(run_dir, expected, token_ids, stop)
    done = len(manifest["segments"])
    state = None
    if done:
        state = KDAReferenceState(**torch.load(segment_path(run_dir, done - 1), weights_only=True)["state"])
        print(f"resuming at segment {done}/{segments} (token {manifest['segments'][-1]['end']})", flush=True)

    for index in range(done, stop):
        start, end = index * segment, min((index + 1) * segment, len(token_ids))
        began = time.perf_counter()
        output, state = kda_forward_reference(layer_input(token_ids[start:end]), weights, config, state, dtype=dtype)
        payload = {"start": start, "end": end, "output": output[0], "state": vars(state)}
        publish(lambda path: torch.save(payload, path), segment_path(run_dir, index))
        if "products" not in manifest:
            manifest["products"] = {
                "output": product_schema("row", output[0]),
                **{f"state.{name}": product_schema("segment_end", t) for name, t in vars(state).items()},
            }
        seconds = time.perf_counter() - began
        manifest["segments"].append({"start": start, "end": end, "seconds": seconds})
        write_manifest(run_dir, manifest)
        print(f"segment {index + 1}/{segments} tokens [{start}, {end}) {seconds:.2f} s", flush=True)

    if len(manifest["segments"]) < segments:
        return manifest  # incomplete: no floor and no registry check yet
    if "token_floor" not in manifest:
        manifest["token_floor"] = golden_floor(
            torch.cat(
                [
                    torch.load(segment_path(run_dir, index), weights_only=True)["output"].norm(dim=-1)
                    for index in range(segments)
                ]
            )
        )
        write_manifest(run_dir, manifest)
    problems = manifest_problems(run_dir.name, manifest)
    if problems:
        raise ValueError(f"the golden in {run_dir} breaks the registry contract: {problems}")
    return manifest


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
    parser.add_argument("--segments", type=int, help="stop after this many complete segments (resumable)")
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
        max_segments=args.segments,
    )


if __name__ == "__main__":
    main()
