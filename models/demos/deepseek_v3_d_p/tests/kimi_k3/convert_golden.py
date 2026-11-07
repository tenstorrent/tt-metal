# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite a ``chunked_group_a_v1`` vLLM trace as the ``per_stream_safetensors_v1`` golden K3 reads.

K3's golden (``k3_vllm_code_debug_1M``) came out of the tracer already in the per-stream layout that
``golden.GoldenTrace`` reads. The K3.1 traces come out chunked: each stream is a directory of
``rows_<start>_<end>.safetensors`` shards listed in ``index.json``. This concatenates each stream's
shards into one file, keeping the trace's own group directories and the per-stream naming rule of
``golden.py`` (file stem = tensor key, except ``kv_cache/layer_N`` holding ``kv_post_transform_layer_N``)::

    decoder_io/decoder_output_layer_0/rows_*.safetensors  -> decoder_io/decoder_output_layer_0.safetensors
    kv_cache/layer_3/rows_*.safetensors                   -> kv_cache/layer_3.safetensors

Lossless: every stream, every row, every dtype. Rows past ``n_prompt_tokens`` are the recorded decode
steps (teacher-forced on ``completion_token_ids``); a consumer that reads the prompt length never sees
them. ``metadata.json`` is the source's with ``layout`` rewritten, the source index (minus its shard
list) and provenance added; it is written last, so a partial output never resolves as a trace.

Host only::

    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.convert_golden SRC DST
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE_LAYOUT = "chunked_group_a_v1"
LAYOUT = "per_stream_safetensors_v1"
# Bump when the output of the same source changes.
CONVERTER_VERSION = 1
# Small provenance files the tracer writes next to metadata.json; copied verbatim when present.
SIDECARS = ("indexer_verify.json",)


def stream_path(group: str, key: str) -> Path:
    """Where `golden.GoldenTrace` looks for `key`: its file stem is its key, except a KV layer's."""
    if group == "kv_cache":
        return Path(group) / f"layer_{key.rsplit('_', 1)[1]}.safetensors"
    return Path(group) / f"{key}.safetensors"


def _read_stream(src: Path, key: str, stream: dict) -> torch.Tensor:
    parts, row = [], 0
    for chunk in stream["chunks"]:
        if chunk["row_start"] != row:
            raise ValueError(f"{key}: shards are not contiguous at row {row} ({chunk['path']})")
        with safe_open(src / chunk["path"], framework="pt", device="cpu") as handle:
            part = handle.get_tensor(key)
        if part.shape[0] != chunk["row_end"] - chunk["row_start"]:
            raise ValueError(f"{key}: {chunk['path']} holds {part.shape[0]} rows, index says {chunk}")
        parts.append(part)
        row = chunk["row_end"]
    tensor = torch.cat(parts)
    expected = (stream["row_count"], *stream["shape_tail"])
    if tuple(tensor.shape) != expected or str(tensor.dtype).removeprefix("torch.") != stream["dtype"]:
        raise ValueError(f"{key}: got {tuple(tensor.shape)} {tensor.dtype}, index says {expected} {stream['dtype']}")
    return tensor


def convert(src: Path, dst: Path, log=print) -> None:
    index = json.loads((src / "index.json").read_text())
    if index["layout"] != SOURCE_LAYOUT:
        raise ValueError(f"{src} is {index['layout']!r}, expected {SOURCE_LAYOUT!r}")
    metadata = json.loads((src / "metadata.json").read_text())
    if (dst / "metadata.json").exists():
        raise FileExistsError(f"{dst} already holds a trace")

    streams = index["tensor_streams"]
    for n, (key, stream) in enumerate(sorted(streams.items()), 1):
        group = Path(stream["chunks"][0]["path"]).parts[0]
        out = dst / stream_path(group, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({key: _read_stream(src, key, stream)}, str(out))
        # save_file creates 0600 whatever the umask; a golden on a shared tree has to be readable.
        out.chmod(0o664)
        log(f"[{time.strftime('%H:%M:%S')}] {n}/{len(streams)} {out.relative_to(dst)}")

    for name in SIDECARS:
        if (src / name).is_file():
            shutil.copyfile(src / name, dst / name)
    metadata["layout"] = LAYOUT
    metadata["source_index"] = {k: v for k, v in index.items() if k != "tensor_streams"}
    metadata["converted_from"] = {"path": str(src), "layout": SOURCE_LAYOUT, "converter_version": CONVERTER_VERSION}
    (dst / "metadata.json").write_text(json.dumps(metadata, indent=1))
    log(f"[{time.strftime('%H:%M:%S')}] done: {len(streams)} streams -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="chunked trace dir (holds index.json + metadata.json)")
    parser.add_argument("dst", type=Path, help="output dir; must not already hold a metadata.json")
    args = parser.parse_args()
    # Shared Weka tree: group-writable, world-readable (see the Exabox Weka storage note).
    os.umask(0o002)
    convert(args.src, args.dst, log=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()
