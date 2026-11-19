# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated real-weight layer0 with 128 changing decode inputs, repeated twice.

The default fixture uses the established seeded Gaussian activation distribution.
An explicit embeddings mode uses checkpoint embedding rows when that extra shard
is installed. Neither mode builds a full model or claims language quality.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.test_synthetic_decoder import (
    assert_pcc,
    device,
    host,
    restore,
    snapshot,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder


def exercise(decoder_bundle, single_chip_mesh, sequence, record_property, hidden, decode_steps):
    config, layer_idx, reference, model = decoder_bundle
    indexed = layer_idx == 3
    cache = DynamicCache(config=config)
    with torch.inference_mode():
        if indexed:
            rope = Qwen4ExpTextRotaryEmbedding(config)
            cos, sin = rope(hidden, torch.arange(sequence + decode_steps).reshape(1, -1))
            mask = torch.full((sequence, sequence), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16).triu(1)
            expected_prefill = reference(
                hidden[:, :sequence],
                position_embeddings=(cos[:, :sequence], sin[:, :sequence]),
                attention_mask=mask[None, None],
                past_key_values=cache,
            )
        else:
            expected_prefill = reference(hidden[:, :sequence], position_embeddings=(None, None), past_key_values=cache)
        expected_decode = []
        for position in range(sequence, sequence + decode_steps):
            expected_decode.append(
                reference(
                    hidden[:, position : position + 1],
                    position_embeddings=((cos[:, : position + 1], sin[:, : position + 1]) if indexed else (None, None)),
                    attention_mask=(torch.zeros(1, 1, 1, position + 1, dtype=torch.bfloat16) if indexed else None),
                    past_key_values=cache,
                )
            )
    # 128 is a cache fixture covering these tests; config/context are untouched.
    page_table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    state = model.create_state(capacity=128, page_table=page_table) if indexed else model.create_state()
    if indexed:
        names = ("position", "k_cache", "v_cache", "raw_history", "compressed_keys")
        buffers = {name: getattr(state.attention, name) for name in names}
    else:
        buffers = {"conv": state.conv, "recurrent": state.recurrent}
    initial = snapshot(buffers)
    prompt = device(hidden[:, :sequence].unsqueeze(0), single_chip_mesh)
    current = device(hidden[:, sequence : sequence + 1].unsqueeze(0), single_chip_mesh)
    refresh = [
        host(hidden[:, position : position + 1].unsqueeze(0)) for position in range(sequence, sequence + decode_steps)
    ]
    with runtime_audit():
        output = model.prefill_forward(prompt, state=state)
    actual_prefill = ttnn.to_torch(output)
    record_property("prefill_pcc", assert_pcc(expected_prefill, actual_prefill, label="token activation prefill"))
    ttnn.deallocate(output)
    saved = snapshot(buffers)
    with runtime_audit():
        warm = model.decode_forward(current, state=state)
    ttnn.synchronize_device(single_chip_mesh)
    ttnn.deallocate(warm)
    restore(buffers, saved)
    trace = ttnn.begin_trace_capture(single_chip_mesh, cq_id=0)
    with runtime_audit():
        output = model.decode_forward(current, state=state)
    ttnn.end_trace_capture(single_chip_mesh, trace, cq_id=0)
    # Exact output only: consumed by synchronous readback before any replay.
    trace_allocation_tracker.acknowledge_corruptible(output)
    decoded = []
    try:
        for repetition in range(2):
            restore(buffers, saved)  # Capture also advances device cache/position.
            values = []
            for step, expected in enumerate(expected_decode):
                ttnn.copy_host_to_device_tensor(refresh[step], current)
                with runtime_audit():
                    ttnn.execute_trace(single_chip_mesh, trace, cq_id=0, blocking=True)
                actual = ttnn.to_torch(output).reshape(expected.shape)
                value = assert_pcc(expected, actual, label=f"token activation decode step{step}, repeat{repetition}")
                if repetition == 0:
                    record_property(f"decode_{step}_pcc", value)
                values.append(actual.clone())
            decoded.append(values)
        assert all(torch.equal(a, b) for a, b in zip(*decoded)), "reset trace replay is not deterministic"
    finally:
        ttnn.release_trace(single_chip_mesh, trace)
    restore(buffers, initial)
    with runtime_audit():
        reset_output = model.prefill_forward(prompt, state=state)
    assert torch.equal(actual_prefill, ttnn.to_torch(reset_output)), "fresh state did not reproduce prefill"
    record_property("weights", "real decoder checkpoint; activation source is explicit in the report")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--activation-source", choices=("gaussian", "embeddings"), default="gaussian")
    a = p.parse_args()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    ids = None
    if a.activation_source == "embeddings":
        ids = torch.randint(config.vocab_size, (256,), generator=torch.Generator().manual_seed(81296)).tolist()
        index = json.loads((a.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        name = "model.language_model.embed_tokens.weight"
        with safe_open(a.snapshot / index[name], framework="pt") as f:
            table = f.get_slice(name)
            embedding = torch.cat([table[i : i + 1] for i in ids], dim=0).unsqueeze(0)
        hidden = embedding.repeat(1, 1, config.hc_count)
    else:
        hidden = (torch.randn(1, 256, 10240, generator=torch.Generator().manual_seed(81296)) * 0.02).bfloat16()
    assert hidden.dtype == torch.bfloat16 and hidden.shape == (1, 256, 10240)
    weights = load_weights(a.snapshot, 0, "")
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 0)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    report = dict(
        scope=__doc__,
        token_ids=ids,
        checkpoint=str(a.snapshot),
        layer=0,
        runtime_factory="OptimizedDecoder",
        activation_source=a.activation_source,
        runtime_audit=True,
        rows=[],
        passed=False,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        model = OptimizedDecoder.from_state_dict(weights, hf_config=config, layer_idx=0, mesh_device=mesh)
        assert type(model) is OptimizedDecoder
        for length, steps in ((1, 3), (33, 3), (128, 128)):
            row = dict(prefill_length=length, decode_steps=steps, repeats=2, properties={}, passed=False)
            report["rows"].append(row)
            exercise(
                (config, 0, reference, model),
                mesh,
                length,
                lambda k, v: row["properties"].update({k: v}),
                hidden[:, : length + steps],
                steps,
            )
            row["passed"] = True
            print(length, steps, "PASS", flush=True)
        report["passed"] = all(r["passed"] for r in report["rows"])
    finally:
        ttnn.close_mesh_device(mesh)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
