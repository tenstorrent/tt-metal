# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resumable long-sequence KDA golden: the reference layer run segment by segment on CPU.

A long FP64 golden takes hours, so the sequence is processed in fixed-size segments. After each segment the
run atomically writes one file holding that segment's output rows and the full carried state (recurrent and
convolution histories) after it. A segment boundary is invisible to the token-ordered reference
(``tests/kimi_k3/test_kda_golden.py`` asserts this), so a run resumed from the last complete segment
reproduces an uninterrupted one. Any later window can be recomputed from the snapshot before it.

The run never opens a device. Example (Kimi-K3 ``layers.0``, Gutenberg pipeline check)::

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kda.golden_run \\
        --checkpoint $KIMI_K3_CKPT --tokens gutenberg --count 102400 --out /path/to/run --threads 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_gate_reference
from models.demos.deepseek_v3_d_p.tests.kda.real_input import chunk_decay_statistics

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


def _log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


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

    ``identity`` names what produced the inputs and weights; a resume refuses a directory whose identity,
    tokens, segment size, dtype or thread count differ, since its snapshots would not continue this run.
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
        manifest = {**expected, "segments": []}
        _write_manifest(run_dir, manifest)

    segments = -(-len(token_ids) // segment)
    done = len(manifest["segments"])
    state = None
    if done:
        state = KDAReferenceState(**torch.load(_segment_path(run_dir, done - 1), weights_only=True)["state"])
        _log(f"resuming at segment {done}/{segments} (token {manifest['segments'][-1]['end']})")

    for index in range(done, segments):
        start, end = index * segment, min((index + 1) * segment, len(token_ids))
        began = time.perf_counter()
        hidden = layer_input(token_ids[start:end])
        output, state = kda_forward_reference(hidden, weights, config, state, dtype=dtype)
        raw_gate = F.linear(
            F.linear(hidden.to(dtype), weights["f_a_proj.weight"].to(dtype)), weights["f_b_proj.weight"].to(dtype)
        ).reshape(end - start, config.num_heads, config.head_k_dim)
        gate = kda_gate_reference(
            raw_gate.unsqueeze(0), weights["A_log"], weights["dt_bias"], config.gate_lower_bound, dtype=dtype
        )[0]
        _atomic_save(
            {"start": start, "end": end, "output": output[0], "state": vars(state)}, _segment_path(run_dir, index)
        )
        seconds = time.perf_counter() - began
        manifest["segments"].append(
            {"index": index, "start": start, "end": end, "seconds": seconds, "decay": chunk_decay_statistics(gate)}
        )
        _write_manifest(run_dir, manifest)
        _log(
            f"segment {index + 1}/{segments} tokens [{start}, {end}) {seconds:.2f} s "
            f"({(end - start) / seconds:.0f} tok/s), max |G_last| {manifest['segments'][-1]['decay'].get('max', float('nan')):.2f}"
        )
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
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
    from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256, load_kda_layer_state_dict
    from models.demos.deepseek_v3_d_p.tests.kda.real_input import KIMI_K3_REVISION, FirstLayerInput, gutenberg_token_ids

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Kimi-K3 checkpoint with layers.0 and embed_tokens"
    )
    parser.add_argument("--tokens", required=True, help="'gutenberg' or a JSON file holding a list of token ids")
    parser.add_argument(
        "--corpus-cache", type=Path, default=Path.home() / ".cache" / "tt-linear-layers-shared" / "text_corpus"
    )
    parser.add_argument("--count", type=int, required=True, help="number of tokens to run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--segment", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16, help="CPU threads; a resume must use the same count")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.tokens == "gutenberg":
        token_ids, source = gutenberg_token_ids(args.corpus_cache), "gutenberg pg1342"
    else:
        token_ids, source = json.loads(Path(args.tokens).read_text(encoding="utf-8")), str(args.tokens)
    if len(token_ids) < args.count:
        raise ValueError(f"{source} has {len(token_ids)} tokens, fewer than --count {args.count}")
    token_ids = token_ids[: args.count]

    _log(f"loading layers.0 and embeddings from {args.checkpoint}")
    config = kimi_k3_kda_config()
    weights = load_kda_layer_state_dict(args.checkpoint, 0, config)
    identity = {
        "model": "Kimi-K3",
        "revision": KIMI_K3_REVISION,
        "layer": "model.layers.0",
        "weights_sha256": kda_state_dict_sha256(weights),
        "input": "input_layernorm(embed_tokens)",
        "token_source": source,
    }
    run_golden(
        run_dir=args.out,
        token_ids=token_ids,
        layer_input=FirstLayerInput(args.checkpoint),
        weights=weights,
        config=config,
        segment=args.segment,
        dtype=torch.float64,
        identity=identity,
    )
    _log("done")


if __name__ == "__main__":
    main()
