# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only prefill chunk/sequential state ordering counterfactual."""

import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"


def compare(a, b):
    a, b = a.float(), b.float()
    diff = a - b
    return dict(
        bf16_different=int((a.bfloat16() != b.bfloat16()).sum()),
        fp32_different=int((a != b).sum()),
        max_abs=float(diff.abs().max()),
        relative_l2=float(diff.norm() / a.norm()),
        pcc=reference.pcc(a, b),
    )


def main():
    torch.set_num_threads(4)
    meta = json.loads((DOC / "v5_combined_layer34_stream_probe.json").read_text())
    oracle = torch.load(meta["oracle"], weights_only=False)
    capture_path = DOC / "v5_combined_layer34_stream_probe.json.pos128.pt"
    capture = torch.load(capture_path, weights_only=False)
    hf = reference.layer(reference.weights(34), 34)
    cache = DynamicCache(config=reference.config())
    saved = {}
    original = hf.linear_attn.chunk_gated_delta_rule

    def chunk(q, k, v, **kw):
        saved.update(q=q.clone(), k=k.clone(), v=v.clone(), kw=kw.copy())
        result = original(q, k, v, **kw)
        saved["chunk_state"] = result[1].clone()
        return result

    hf.linear_attn.chunk_gated_delta_rule = chunk
    with torch.no_grad():
        x = hf.input_layernorm(oracle["prefix"])
        hf.linear_attn(x, cache_params=cache)
        q, k, v, kw = saved["q"], saved["k"], saved["v"], saved["kw"]
        seqkw = {k: value for k, value in kw.items() if k != "cu_seqlens"}
        _, sequential = torch_recurrent_gated_delta_rule(q, k, v, **seqkw)
        _, chunk32 = torch_chunk_gated_delta_rule(q, k, v, chunk_size=32, **kw)
    hf_state = capture["hf_values"]["delta_state_input"]
    tt_state = capture["tt_values"]["delta_state_input"]
    assert torch.equal(hf_state, saved["chunk_state"].bfloat16()), "CPU prefix does not reproduce frozen HF cache"
    hf_codes = hf_state.contiguous().view(torch.int16).flatten()
    seq_codes = sequential.bfloat16().contiguous().view(torch.int16).flatten()
    tt_codes = tt_state.bfloat16().contiguous().view(torch.int16).flatten()
    differences = (hf_codes != tt_codes).nonzero().flatten().tolist()
    assert torch.equal(seq_codes, tt_codes)
    result = dict(
        all_different_bf16_entries=[
            dict(flat_index=i, hf_code=int(hf_codes[i]), sequential_code=int(seq_codes[i]), tt_code=int(tt_codes[i]))
            for i in differences
        ],
        all_difference_codes_explained_by_sequential=True,
        measured_decoder_source_sha256=capture["measured_decoder_source_sha256"],
        oracle_prefix_sha256=hashlib.sha256(oracle["prefix"].view(torch.uint8).numpy().tobytes()).hexdigest(),
        classification="CPU same-input chunk/sequential counterfactual; not device acceptance",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hf_source_sha256=hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest(),
        capture=str(capture_path),
        capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        chunk64_vs_sequential=compare(saved["chunk_state"], sequential),
        chunk64_vs_chunk32=compare(saved["chunk_state"], chunk32),
        captured_hf_vs_sequential_cache=compare(hf_state, sequential.bfloat16()),
        captured_hf_vs_tt_cache=compare(hf_state, tt_state.bfloat16()),
        sequential_vs_tt_cache=compare(sequential.bfloat16(), tt_state.bfloat16()),
        sequential_vs_tt_fp32=compare(sequential, tt_state),
        chunk32_vs_tt_cache=compare(chunk32.bfloat16(), tt_state.bfloat16()),
    )
    (DOC / "v6_prefill_state_hypothesis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
