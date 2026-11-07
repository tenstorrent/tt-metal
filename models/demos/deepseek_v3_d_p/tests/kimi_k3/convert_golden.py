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

One stream changes meaning on the way, so that the golden honours K3's contract
(``test_golden_contract.py``): K3's ``decoder_output_layer_i`` is the AttnRes *running sum* after layer i,
which the stack's layer tap returns, but the K3.1 tracer records the layer's FFN output alone under that
key. The converter writes ``decoder_output_layer_i`` as K3's post-seal running sum,
``sum(attn_output_j + ffn_j)`` over ``j`` from the block's seal (``j % ATTN_RES_BLOCK_SIZE == 0``, where the
sum restarts) to ``i``, accumulated as vLLM does — in the trace's dtype, attention then FFN — which reproduces
the tracer's sealed block snapshots (``attn_res/residual_layer_*``) bit for bit, and keeps the trace's own stream
as ``ffn_output/ffn_output_layer_i``. So nothing is lost: every other stream, every row, every dtype is
copied as is. Rows past ``n_prompt_tokens`` are the recorded decode steps (teacher-forced on
``completion_token_ids``); a consumer that reads the prompt length never sees them. ``metadata.json`` is the source's with ``layout`` rewritten, the source index (minus its shard
list) and provenance added; it is written last, so a partial output never resolves as a trace.

``--layers N`` keeps only the first N layers' streams (and every stream that is not per layer), for a
golden that only has to serve the layers its consumer checks: K3's long-context check scores layers 0-23,
and the 128K trace's 93 layers are ~500 GB. Recorded as ``converted_from.layers``.

Host only::

    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.convert_golden SRC DST [--layers N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config

SOURCE_LAYOUT = "chunked_group_a_v1"
LAYOUT = "per_stream_safetensors_v1"
# Bump when the output of the same source changes. 2: decoder_output is K3's running sum (module docstring).
CONVERTER_VERSION = 2
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


def _stream_layer(key: str) -> int | None:
    """The model layer a stream belongs to (`..._layer_N`), or None for a stream that is not per layer."""
    match = re.search(r"_layer_(\d+)$", key)
    return int(match[1]) if match else None


def convert(src: Path, dst: Path, log=print, layers: int | None = None) -> None:
    index = json.loads((src / "index.json").read_text())
    if index["layout"] != SOURCE_LAYOUT:
        raise ValueError(f"{src} is {index['layout']!r}, expected {SOURCE_LAYOUT!r}")
    metadata = json.loads((src / "metadata.json").read_text())
    if (dst / "metadata.json").exists():
        raise FileExistsError(f"{dst} already holds a trace")

    streams = {
        key: stream
        for key, stream in index["tensor_streams"].items()
        if layers is None or _stream_layer(key) is None or _stream_layer(key) < layers
    }
    decoder_layers = sorted(int(key.rsplit("_", 1)[1]) for key in streams if key.startswith("decoder_output_layer_"))
    # The running sum needs every layer of each block from its seal, and each layer's attention output.
    missing = [layer for layer in decoder_layers if f"attn_output_layer_{layer}" not in streams]
    if missing or decoder_layers != list(range(len(decoder_layers))):
        raise ValueError(
            f"{src}: decoder_output layers {decoder_layers}, without attn_output {missing}; no running sum"
        )
    count = len(streams) + len(decoder_layers)  # every stream, plus each layer's FFN output kept beside its running sum

    def write(n: int, group: str, key: str, tensor: torch.Tensor) -> None:
        out = dst / stream_path(group, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({key: tensor.contiguous()}, str(out))
        # save_file creates 0600 whatever the umask; a golden on a shared tree has to be readable.
        out.chmod(0o664)
        log(f"[{time.strftime('%H:%M:%S')}] {n}/{count} {out.relative_to(dst)}")

    n = 0
    for key, stream in sorted(streams.items()):
        if not key.startswith("decoder_output_layer_"):
            n += 1
            write(n, Path(stream["chunks"][0]["path"]).parts[0], key, _read_stream(src, key, stream))

    for layer in decoder_layers:
        key = f"decoder_output_layer_{layer}"
        ffn = _read_stream(src, key, streams[key])
        attn = _read_stream(src, f"attn_output_layer_{layer}", streams[f"attn_output_layer_{layer}"])
        if layer % KimiK3Config.ATTN_RES_BLOCK_SIZE == 0:
            running = torch.zeros_like(ffn)  # the seal: the sum restarts
        running = running + attn + ffn  # left to right, in the trace's dtype: vLLM's two adds
        write(n + 1, "ffn_output", f"ffn_output_layer_{layer}", ffn)
        write(n + 2, "decoder_io", key, running)
        n += 2

    for name in SIDECARS:
        if (src / name).is_file():
            shutil.copyfile(src / name, dst / name)
    metadata["layout"] = LAYOUT
    metadata["source_index"] = {k: v for k, v in index.items() if k != "tensor_streams"}
    metadata["converted_from"] = {
        "path": str(src),
        "layout": SOURCE_LAYOUT,
        "converter_version": CONVERTER_VERSION,
        "layers": layers,
    }
    (dst / "metadata.json").write_text(json.dumps(metadata, indent=1))
    log(f"[{time.strftime('%H:%M:%S')}] done: {count} streams -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="chunked trace dir (holds index.json + metadata.json)")
    parser.add_argument("dst", type=Path, help="output dir; must not already hold a metadata.json")
    parser.add_argument("--layers", type=int, default=None, help="keep only the first N layers' streams")
    args = parser.parse_args()
    # Shared Weka tree: group-writable, world-readable (see the Exabox Weka storage note).
    os.umask(0o002)
    convert(args.src, args.dst, log=lambda line: print(line, flush=True), layers=args.layers)


if __name__ == "__main__":
    main()
