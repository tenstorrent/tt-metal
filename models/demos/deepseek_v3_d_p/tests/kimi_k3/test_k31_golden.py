# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The K3.1 golden is K3's golden format, and it is the trace it was converted from. Host only.

`convert_golden.py` rewrites Miguel's chunked vLLM trace into the per-stream layout `golden.GoldenTrace`
reads. Pinned here: the conversion round-trips through both readers that consume a golden — K3's
`GoldenTrace` (the *_matches_golden tests) and the runner producer's KV loaders (the PCC gate) — and the
converted K3.1 golden is the source trace row for row on the streams those gates score, at 24 and at
93 layers, and enters at the K3.1 checkpoint's own token embedding.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from models.demos.common.prefill.runners.runner_utils import load_trace_golden_span, load_trace_token_ids
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.convert_golden import LAYOUT, SOURCE_LAYOUT, convert
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_K31, GoldenTrace, embedding_rows, resolve_trace
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter
from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import (
    _load_golden_kv_post,
    _load_sharded_rows,
    kvpe_golden_present,
)

# The runner's K3.1 legs: sc1 runs stage 0 (layers 0-23), sc4 all 93.
DEPTHS = (24, KimiK3Config.NUM_LAYERS)


def _write_chunked(root: Path, streams: dict[str, tuple[str, torch.Tensor]], shard_rows: int) -> None:
    """A minimal `chunked_group_a_v1` trace: `streams` maps key -> (stream dir under root, tensor)."""
    index = {"format_version": 1, "layout": SOURCE_LAYOUT, "chunk_rows": shard_rows, "tensor_streams": {}}
    for key, (stream_dir, tensor) in streams.items():
        chunks = []
        for start in range(0, tensor.shape[0], shard_rows):
            end = min(start + shard_rows, tensor.shape[0])
            path = f"{stream_dir}/rows_{start:08d}_{end:08d}.safetensors"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            save_file({key: tensor[start:end].contiguous()}, str(root / path))
            chunks.append({"row_start": start, "row_end": end, "path": path})
        index["tensor_streams"][key] = {
            "row_count": tensor.shape[0],
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape_tail": list(tensor.shape[1:]),
            "chunks": chunks,
        }
    (root / "index.json").write_text(json.dumps(index))
    (root / "metadata.json").write_text(json.dumps({"layout": SOURCE_LAYOUT, "token_ids": list(range(10))}))


def test_converter_round_trips_through_both_golden_readers(tmp_path, expect_error):
    """Every stream lands where `GoldenTrace` and the producer look for it, bit for bit, shards joined in order."""
    hidden = torch.randn(10, 8).to(torch.bfloat16)
    kv = torch.randn(10, 6).to(torch.bfloat16)
    topk = torch.randint(-1, 10, (3, 4), dtype=torch.int32)
    src, dst = tmp_path / "chunked", tmp_path / "golden"
    _write_chunked(
        src,
        {
            "decoder_output_layer_0": ("decoder_io/decoder_output_layer_0", hidden),
            "kv_post_transform_layer_3": ("kv_cache/layer_3", kv),
            "topk_indices": ("topk/indices", topk),
        },
        shard_rows=4,
    )
    convert(src, dst, log=lambda _: None)

    assert {path.stat().st_mode & 0o777 for path in dst.rglob("*.safetensors")} == {0o664}
    trace = GoldenTrace(dst)
    assert trace.metadata["layout"] == LAYOUT and trace.metadata["converted_from"]["path"] == str(src)
    assert torch.equal(trace.decoder_output(0).to(torch.bfloat16), hidden)
    assert torch.equal(trace.kv_cache(3).to(torch.bfloat16), kv)
    assert torch.equal(trace.rows("topk", "topk_indices").to(torch.int32), topk)
    assert kvpe_golden_present(dst, 3) and not kvpe_golden_present(dst, 7)
    assert torch.equal(_load_golden_kv_post(dst, 3, 7, start=2).to(torch.bfloat16), kv[2:9])

    with expect_error(FileExistsError, "already holds a trace"):
        convert(src, dst, log=lambda _: None)


@pytest.fixture(scope="module")
def k31_trace() -> GoldenTrace:
    trace = resolve_trace(TRACE_K31)
    if trace is None:
        pytest.skip(f"no K3.1 golden at {TRACE_K31}; set KIMI_K3_GOLDEN_TRACE")
    return trace


def test_k31_golden_enters_at_the_k31_token_embedding(k31_trace):
    """Bit-identical, as K3's `test_decoder_input_is_the_token_embedding`: the checkpoint is the trace's."""
    checkpoint = Path(os.getenv(KimiK31Adapter.env_var) or KimiK31Adapter.hf_model_default)
    if not (checkpoint / "model.safetensors.index.json").is_file():
        pytest.skip(f"no K3.1 checkpoint at {checkpoint}; set {KimiK31Adapter.env_var}")
    rows = 512
    expected = embedding_rows(checkpoint, k31_trace.token_ids(rows)).to(torch.bfloat16)
    assert torch.equal(expected, k31_trace.decoder_input(0, rows).to(torch.bfloat16))


@pytest.mark.parametrize("num_layers", DEPTHS, ids=lambda n: f"L{n}")
def test_k31_golden_serves_the_runner_gate(k31_trace, num_layers):
    """What a `num_layers` run scores is in the golden, and is the source trace's prompt rows unchanged.

    The producer compares the prompt span (`load_trace_golden_span`) of each DSA layer's kvpe; the depth
    tests read the last layer's decoder output. Both through the readers those tests use.
    """
    trace_dir = k31_trace.path
    prompt = k31_trace.metadata["n_prompt_tokens"]
    source = Path(k31_trace.metadata["converted_from"]["path"])
    assert load_trace_golden_span(trace_dir) == (0, prompt) and len(load_trace_token_ids(trace_dir)) == prompt

    dsa_layers = [layer for layer in KimiK3Config.mla_layer_ids() if layer < num_layers]
    for layer in dsa_layers:
        assert kvpe_golden_present(trace_dir, layer), layer
        golden = _load_golden_kv_post(trace_dir, layer, prompt)
        key = f"kv_post_transform_layer_{layer}"
        assert torch.equal(golden, _load_sharded_rows(source / "kv_cache" / f"layer_{layer}", key, 0, prompt)), layer

    last = num_layers - 1
    golden = k31_trace.decoder_output(last, 0, prompt)
    key = f"decoder_output_layer_{last}"
    assert torch.equal(golden, _load_sharded_rows(source / "decoder_io" / key, key, 0, prompt))
