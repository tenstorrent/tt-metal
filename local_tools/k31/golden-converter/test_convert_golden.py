# Round-trip test of the local golden converter, extracted from tt-metal
# models/demos/deepseek_v3_d_p/tests/kimi_k3/test_k31_golden.py @ 050ce1431f9 (tt_metal_tracker-8v1.7).
# Run from a tt-metal checkout root (PYTHONPATH=$PWD, its python_env active):
#   pytest /home/mvasilijevic/repos/k31-local-tools/golden-converter/test_convert_golden.py -p no:cacheprovider
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from convert_golden import LAYOUT, SOURCE_LAYOUT, convert  # noqa: E402

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import GoldenTrace
from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import _load_golden_kv_post, kvpe_golden_present

@pytest.fixture
def expect_error():
    """tt-metal's root conftest fixture, which a test outside the checkout does not load."""
    return lambda error, message: pytest.raises(error, match=message)


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
    """Every stream lands where `GoldenTrace` and the producer look for it, bit for bit, shards joined in order;
    `decoder_output` becomes the running sum of `attn_output` and the FFN output, restarted at the seal (layer 12),
    and the FFN output is kept as `ffn_output`."""
    layers = KimiK3Config.ATTN_RES_BLOCK_SIZE + 2
    ffn = [torch.randn(10, 8).to(torch.bfloat16) for _ in range(layers)]
    attn = [torch.randn(10, 8).to(torch.bfloat16) for _ in range(layers)]
    kv = torch.randn(10, 6).to(torch.bfloat16)
    topk = torch.randint(-1, 10, (3, 4), dtype=torch.int32)
    src, dst = tmp_path / "chunked", tmp_path / "golden"
    streams = {
        "kv_post_transform_layer_3": ("kv_cache/layer_3", kv),
        "topk_indices": ("topk/indices", topk),
    }
    for i in range(layers):
        streams[f"decoder_output_layer_{i}"] = (f"decoder_io/decoder_output_layer_{i}", ffn[i])
        streams[f"attn_output_layer_{i}"] = (f"attn_output/attn_output_layer_{i}", attn[i])
    _write_chunked(src, streams, shard_rows=4)
    convert(src, dst, log=lambda _: None)

    assert {path.stat().st_mode & 0o777 for path in dst.rglob("*.safetensors")} == {0o664}
    trace = GoldenTrace(dst)
    assert trace.metadata["layout"] == LAYOUT and trace.metadata["converted_from"]["path"] == str(src)
    for i in range(layers):
        running = running if i % KimiK3Config.ATTN_RES_BLOCK_SIZE else torch.zeros(10, 8, dtype=torch.bfloat16)
        running = (running + attn[i]) + ffn[i]  # bf16, attention then FFN
        assert torch.equal(trace.decoder_output(i).to(torch.bfloat16), running), i
        assert torch.equal(trace.rows("ffn_output", f"ffn_output_layer_{i}").to(torch.bfloat16), ffn[i]), i
        assert torch.equal(trace.rows("attn_output", f"attn_output_layer_{i}").to(torch.bfloat16), attn[i]), i
    assert torch.equal(trace.kv_cache(3).to(torch.bfloat16), kv)
    assert torch.equal(trace.rows("topk", "topk_indices").to(torch.int32), topk)
    assert kvpe_golden_present(dst, 3) and not kvpe_golden_present(dst, 7)
    assert torch.equal(_load_golden_kv_post(dst, 3, 7, start=2).to(torch.bfloat16), kv[2:9])

    with expect_error(FileExistsError, "already holds a trace"):
        convert(src, dst, log=lambda _: None)
    # `layers` keeps the first N layers' streams and every stream that is not per layer.
    convert(src, tmp_path / "golden_l12", log=lambda _: None, layers=KimiK3Config.ATTN_RES_BLOCK_SIZE)
    short = GoldenTrace(tmp_path / "golden_l12")
    assert short.metadata["converted_from"]["layers"] == KimiK3Config.ATTN_RES_BLOCK_SIZE
    assert short.has("decoder_io", "decoder_output_layer_11") and not short.has("decoder_io", "decoder_output_layer_12")
    assert torch.equal(short.decoder_output(11), trace.decoder_output(11)) and short.has("topk", "topk_indices")
    assert short.has_kv_cache(3)
    # A layer without its attention output has no running sum: refused, not guessed.
    del streams["attn_output_layer_5"]
    _write_chunked(tmp_path / "chunked2", streams, shard_rows=4)
    with expect_error(ValueError, "no running sum"):
        convert(tmp_path / "chunked2", tmp_path / "golden2", log=lambda _: None)


