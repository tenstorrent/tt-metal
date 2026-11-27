# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 ``layers.0`` KDA FP64 golden producer: the reference layer run segment by segment on CPU.

A golden over a long context (up to 1M tokens) takes hours of CPU, so the sequence is processed in fixed-size
segments. After each segment the run atomically writes that segment's output rows and the full carried state
(recurrent and convolution histories) after it (``segment_path``), so any window can be read back with the state
before it (the reader: ``golden.py`` next to this file). The run never opens a device: device tests only load a
prepared golden, so the CPU work never holds the device.

The layer is computed in FP64 (``dtype``) and its products are stored in ``STORAGE_DTYPE``, FP32
(``storage_dtype``): a device cannot compute in FP64, so storing the FP64 result in FP32 moves a device measurement
only by FP32 rounding (2^-24 relative), at half the bytes to read. The carried state stays FP64 in
``resume_state_path`` (the latest complete segment's only), so an interrupted run resumes bit-identical to an
uninterrupted one and a complete golden can be extended.

``manifest.json`` is the golden's only metadata, in the long-context golden registry's format
(``common/prefill/long_context_accuracy/fp64_golden.py``; the golden's id is its directory name): what produced it
(producer, identity, provenance of each run session), the prompt (``token_ids.pt``), which segments are complete and how
long each took, the products, and, once complete, the token floor for per-token NRMSE.

The manifest's ``producer`` is ``PRODUCER``, ``models.demos.deepseek_v3_d_p.tests.kda.long_context_golden``: this
module's path when it wrote golden ``k3-l0-kda-fp64-1m`` (version 1, products stored FP64). It names the producer's
numerics, not where the module lives now, so it stays unchanged while the bytes do; the reader requires it with
``PRODUCER_VERSION`` (2: products stored FP32). The registered golden ``K3_LAYER0_GOLDEN``
(``k3-l0-kda-fp64-stored-fp32-1m``) is ``k3-l0-kda-fp64-1m`` converted by ``convert_to_fp32_storage``, a pure cast;
its manifest's ``conversion`` names the source and the source manifest's digest. The source was written before the
registry existed and registered by rewriting its manifest (``manifest.orig.json`` holds the original), so both
manifests differ from what this module writes in form only: ``identity.revision`` is the code revision this module
records per run session (``sessions[].code_revision``), the registry added ``migration``, and the segment entries carry
the old producer's ``index`` and ``decay`` statistics.

Example (Kimi-K3 ``layers.0`` on the code tokens of a vLLM trace, written straight into the registry;
``--segments N`` stops after N segments, and a later run resumes)::

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden_producer \\
        --checkpoint $KIMI_K3_CKPT --tokens /path/to/vllm/trace --count 1044480 --segment 1024 --threads 32 \\
        --out /mnt/weka/model-cache/scratch/accuracy-goldens/<golden id>

Converting a version-1 golden (refuses a destination that already holds a manifest)::

    python -m models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden_producer convert \\
        /mnt/weka/model-cache/scratch/accuracy-goldens/k3-l0-kda-fp64-1m \\
        /mnt/weka/model-cache/scratch/accuracy-goldens/k3-l0-kda-fp64-stored-fp32-1m
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    MANIFEST,
    TOKEN_IDS,
    make_golden_dir,
    manifest_problems,
    product_schema,
    publish,
    session_provenance,
    start_or_resume,
    token_ids_sha256,
    write_manifest,
)
from models.demos.common.prefill.long_context_accuracy.windowed_errors import golden_floor
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference

PRODUCER = "models.demos.deepseek_v3_d_p.tests.kda.long_context_golden"
PRODUCER_VERSION = 2  # 1: products stored FP64; 2: computed FP64, products stored STORAGE_DTYPE
STORAGE_DTYPE = torch.float32
# Kimi-K3 layers.0 over the first 1,044,480 code tokens of the k3_4_layers_moe_kda_io vLLM trace.
K3_LAYER0_GOLDEN = "k3-l0-kda-fp64-stored-fp32-1m"
PRODUCER_MODULE = "models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden_producer"  # (this file)


def segment_path(run_dir: Path, index: int) -> Path:
    """Segment ``index``'s file: ``{"start", "end", "output": [rows, hidden], "state": {name: tensor}}``, the state
    after the segment, each tensor in ``STORAGE_DTYPE``."""
    return Path(run_dir) / f"segment_{index:06d}.pt"


def resume_state_path(run_dir: Path, index: int) -> Path:
    """The FP64 state after segment ``index`` (``{name: tensor}``, not a product), kept for the latest complete
    segment only."""
    return Path(run_dir) / f"resume_state_{index:06d}.pt"


def _stored(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(STORAGE_DTYPE)


def _write_segment(run_dir: Path, index: int, start: int, end: int, output: torch.Tensor, state: dict) -> None:
    payload = {
        "start": start,
        "end": end,
        "output": _stored(output),
        "state": {name: _stored(t) for name, t in state.items()},
    }
    publish(lambda path: torch.save(payload, path), segment_path(run_dir, index))


def _remove_resume_states(run_dir: Path, keep: int) -> None:
    """Remove every resume state but segment ``keep``'s. Called once the manifest lists segment ``keep`` (whose state
    was published before the manifest), so after a crash at any point the state of the manifest's last segment is
    there."""
    for path in Path(run_dir).glob("resume_state_*.pt"):
        if path.name != resume_state_path(run_dir, keep).name:
            path.unlink()


def _products(output: torch.Tensor, state: dict[str, torch.Tensor]) -> dict:
    return {
        "output": product_schema("row", _stored(output)),
        **{f"state.{name}": product_schema("segment_end", _stored(t)) for name, t in state.items()},
    }


def _token_floor(run_dir: Path, segments: int) -> float:
    """``golden_floor`` of the stored output rows' norms, taken in FP64."""
    norms = [
        torch.load(segment_path(run_dir, index), weights_only=True, mmap=True)["output"].double().norm(dim=-1)
        for index in range(segments)
    ]
    return golden_floor(torch.cat(norms))


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
    """Run (or resume) the golden over ``token_ids`` in ``dtype``, store its products in ``STORAGE_DTYPE``, and return
    the manifest; with ``max_segments``, stop once that many segments are complete (an incomplete golden a later run
    resumes).

    ``identity`` names what produced the inputs and weights. A resume refuses a directory whose identity, tokens,
    segment size, dtypes or thread count differ (``fp64_golden.start_or_resume``).
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
        "storage_dtype": str(STORAGE_DTYPE).removeprefix("torch."),
        "threads": torch.get_num_threads(),
    }
    segments = -(-len(token_ids) // segment)
    stop = segments if max_segments is None else min(segments, max_segments)
    manifest = start_or_resume(run_dir, expected, token_ids, stop)
    done = len(manifest["segments"])
    state = None
    if done:
        state = KDAReferenceState(**torch.load(resume_state_path(run_dir, done - 1), weights_only=True))
        print(f"resuming at segment {done}/{segments} (token {manifest['segments'][-1]['end']})", flush=True)

    for index in range(done, stop):
        start, end = index * segment, min((index + 1) * segment, len(token_ids))
        began = time.perf_counter()
        output, state = kda_forward_reference(layer_input(token_ids[start:end]), weights, config, state, dtype=dtype)
        _write_segment(run_dir, index, start, end, output[0], vars(state))
        publish(lambda path: torch.save(dict(vars(state)), path), resume_state_path(run_dir, index))
        if "products" not in manifest:
            manifest["products"] = _products(output[0], vars(state))
        seconds = time.perf_counter() - began
        manifest["segments"].append({"start": start, "end": end, "seconds": seconds})
        write_manifest(run_dir, manifest)
        _remove_resume_states(run_dir, index)
        print(f"segment {index + 1}/{segments} tokens [{start}, {end}) {seconds:.2f} s", flush=True)

    if len(manifest["segments"]) < segments:
        return manifest  # incomplete: no floor and no registry check yet
    if "token_floor" not in manifest:
        manifest["token_floor"] = _token_floor(run_dir, segments)
        write_manifest(run_dir, manifest)
    problems = manifest_problems(run_dir.name, manifest)
    if problems:
        raise ValueError(f"the golden in {run_dir} breaks the registry contract: {problems}")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def _same_bits(got: torch.Tensor, want: torch.Tensor) -> bool:
    return got.dtype == want.dtype == STORAGE_DTYPE and torch.equal(got.view(torch.int32), want.view(torch.int32))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def convert_to_fp32_storage(source: Path, destination: Path) -> dict:
    """Write golden ``source`` (this producer's version 1: products stored FP64) as a version-2 golden in
    ``destination``, whose directory name is the new golden id, and return the new manifest.

    A pure cast: every tensor of every segment file becomes ``.to(STORAGE_DTYPE)`` of the source's, checked bit for
    bit on the file read back, whose sha256 goes to its manifest segment entry. The source's last FP64 state becomes
    the resume state, the token ids are copied (digest checked), and the token floor is recomputed from the stored
    rows as ``run_golden`` does (the source's is kept in ``conversion``). The manifest is written last, so a
    destination interrupted before it is no golden (the registry refuses it); a destination that already holds a
    manifest is refused."""
    source, destination = Path(source), Path(destination)
    source_manifest = (source / MANIFEST).read_bytes()
    manifest = json.loads(source_manifest)
    problems = manifest_problems(source.name, manifest)
    if problems or (manifest.get("producer"), manifest.get("producer_version")) != (PRODUCER, 1):
        raise ValueError(f"{source} is not a complete {PRODUCER} version-1 golden: {problems}")
    if (destination / MANIFEST).exists():
        raise ValueError(f"{destination} already holds a golden")
    make_golden_dir(destination)
    if token_ids_sha256(torch.load(source / TOKEN_IDS, weights_only=True)) != manifest["token_ids_sha256"]:
        raise ValueError(f"{source / TOKEN_IDS} does not match the manifest's token_ids_sha256")
    publish(lambda path: shutil.copyfile(source / TOKEN_IDS, path), destination / TOKEN_IDS)

    segments = manifest["segments"]
    began = time.perf_counter()
    for index, entry in enumerate(segments):
        payload = torch.load(segment_path(source, index), weights_only=True)
        if (payload["start"], payload["end"]) != (entry["start"], entry["end"]):
            raise ValueError(f"segment {index} of {source} holds rows {payload['start']}-{payload['end']}, not {entry}")
        _write_segment(destination, index, payload["start"], payload["end"], payload["output"], payload["state"])
        written = torch.load(segment_path(destination, index), weights_only=True)
        want = {"output": payload["output"], **{f"state.{n}": t for n, t in payload["state"].items()}}
        got = {"output": written["output"], **{f"state.{n}": t for n, t in written["state"].items()}}
        if (written["start"], written["end"]) != (entry["start"], entry["end"]) or got.keys() != want.keys():
            raise ValueError(f"segment {index} of {destination} does not hold the source's rows and tensors")
        bad = [n for n, t in want.items() if t.dtype != torch.float64 or not _same_bits(got[n], t.to(STORAGE_DTYPE))]
        if bad:
            raise ValueError(f"segment {index} of {destination}: {bad} are not the FP64 source cast to {STORAGE_DTYPE}")
        entry["sha256"] = _sha256(segment_path(destination, index))
        if index == len(segments) - 1:
            publish(lambda path: torch.save(dict(payload["state"]), path), resume_state_path(destination, index))
            products = _products(payload["output"], payload["state"])
        print(f"{_now()} converted segment {index + 1}/{len(segments)} {time.perf_counter() - began:.1f} s", flush=True)

    manifest["conversion"] = {
        "source_golden_id": source.name,
        "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "source_token_floor": manifest["token_floor"],
        "tool": f"{PRODUCER_MODULE}.convert_to_fp32_storage",
        "at": _now(),
        "verified": f"every tensor of every segment bit-equal to the source's .to({STORAGE_DTYPE})",
        **session_provenance(),
    }
    manifest.update(
        golden_id=destination.name,
        producer_version=PRODUCER_VERSION,
        storage_dtype=str(STORAGE_DTYPE).removeprefix("torch."),
        products=products,
        token_floor=_token_floor(destination, len(segments)),
    )
    problems = manifest_problems(destination.name, manifest)
    if problems:
        raise ValueError(f"the converted golden in {destination} breaks the registry contract: {problems}")
    write_manifest(destination, manifest)
    return manifest


def main() -> None:
    if sys.argv[1:2] == ["convert"]:
        parser = argparse.ArgumentParser(description=convert_to_fp32_storage.__doc__)
        parser.add_argument("source", type=Path)
        parser.add_argument("destination", type=Path)
        args = parser.parse_args(sys.argv[2:])
        convert_to_fp32_storage(args.source, args.destination)
        return

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
