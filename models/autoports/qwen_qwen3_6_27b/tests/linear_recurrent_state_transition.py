# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill-to-decode regression for optimized linear recurrent-state dtypes.

This deliberately crosses a non-aligned, multi-chunk prefill boundary before
advancing the same physical recurrent cache for several decode steps.  It is
the focused correctness probe for ``linear_state_*`` candidates; traced decode
latency remains measured by ``traced_synthetic_pcc.py``.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_real_pcc import _hf_layer as real_hf_layer
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_real_pcc import _real_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _hf_layer as synthetic_hf_layer
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as synthetic_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, MODEL_REVISION, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder, resolve_policy
from models.common.utility_functions import comp_pcc


def _copy_host(value, destination, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    source = ttnn.from_torch(value, layout=layout, dtype=dtype)
    ttnn.copy_host_to_device_tensor(source, destination, cq_id=0)


def _reference(config, layer, prefix, decode_tokens):
    cache = DynamicCache(config=config)
    outputs = [
        layer(
            prefix,
            position_embeddings=(None, None),
            attention_mask=None,
            past_key_values=cache,
        )
    ]
    for token in decode_tokens:
        outputs.append(
            layer(
                token,
                position_embeddings=(None, None),
                attention_mask=None,
                past_key_values=cache,
            )
        )
    return outputs


def _device_run(config, state, candidate, prefix, decode_tokens):
    batch, sequence, _ = prefix.shape
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER,
            mesh_device=mesh,
            batch=batch,
            max_context=max(64, sequence + len(decode_tokens)),
            page_size=64,
            candidate=candidate,
        )
        page_table = _to_device(
            torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        prefill_positions = _to_device(
            torch.arange(sequence, dtype=torch.int64).to(torch.uint32).reshape(1, -1).expand(batch, -1),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        prefix_tt = _to_device(prefix.unsqueeze(0), mesh_device=mesh)
        prefill_output = decoder.prefill_forward(
            hidden_states=prefix_tt,
            page_table=page_table,
            current_positions=prefill_positions,
        )
        ttnn.synchronize_device(mesh)
        outputs = [ttnn.to_torch(ttnn.get_device_tensors(prefill_output)[0]).squeeze(0)]

        decode_input = _to_device(
            decode_tokens[0].reshape(1, 1, batch, config.hidden_size),
            mesh_device=mesh,
        )
        decode_positions = _to_device(
            torch.full((batch,), sequence, dtype=torch.uint32),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        elapsed_ms = []
        for step, token in enumerate(decode_tokens):
            _copy_host(token.reshape(1, 1, batch, config.hidden_size), decode_input)
            _copy_host(
                torch.full((batch,), sequence + step, dtype=torch.uint32),
                decode_positions,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
            started = time.perf_counter()
            output = decoder.decode_forward(
                hidden_states=decode_input,
                page_table=page_table,
                current_positions=decode_positions,
            )
            ttnn.synchronize_device(mesh)
            elapsed_ms.append((time.perf_counter() - started) * 1000)
            outputs.append(ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape(batch, 1, config.hidden_size))

        recurrent = decoder.caches["recurrent"]
        recurrent_dtype = str(recurrent.dtype)
        recurrent_host = ttnn.to_torch(ttnn.get_device_tensors(recurrent)[0])
        return outputs, recurrent_host, recurrent_dtype, elapsed_ms
    finally:
        ttnn.close_mesh_device(mesh)


@torch.no_grad()
def run(candidate, batch, prefill_sequence, decode_steps, real_weights=False, repeat_runs=1):
    ttnn.CONFIG.throw_exception_on_fallback = True
    if real_weights and batch != 1:
        raise ValueError("official-weight transition evidence is intentionally B1; use synthetic evidence for B32")
    torch.manual_seed(20260730)
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        **({"revision": MODEL_REVISION} if real_weights else {}),
    ).text_config
    state = _real_state() if real_weights else synthetic_state(config)
    hf_layer = real_hf_layer(config, state) if real_weights else synthetic_hf_layer(config, state)
    row_offset = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1) * 0.01
    prefix = ((torch.randn(batch, prefill_sequence, config.hidden_size) * 0.2) + row_offset).bfloat16()
    decode_tokens = [
        ((torch.randn(batch, 1, config.hidden_size) * 0.2) + row_offset + (step + 1) * 0.03).bfloat16()
        for step in range(decode_steps)
    ]
    references = _reference(config, hf_layer, prefix, decode_tokens)

    device_runs = [_device_run(config, state, candidate, prefix, decode_tokens) for _ in range(repeat_runs)]
    actual, recurrent, recurrent_dtype, elapsed_ms = device_runs[0]
    pcc = []
    all_pcc_passed = True
    for phase, reference, observed in zip(
        ["prefill", *[f"decode_{step}" for step in range(decode_steps)]],
        references,
        actual,
        strict=True,
    ):
        passed, message = comp_pcc(reference.float(), observed.float(), 0.995)
        print("LINEAR_STATE_TRANSITION_PCC", f"phase={phase}", message)
        all_pcc_passed = all_pcc_passed and bool(passed)
        pcc.append({"phase": phase, "value": float(message), "passed": bool(passed)})

    deterministic = None
    if repeat_runs == 2:
        second_actual, second_recurrent, second_dtype, _ = device_runs[1]
        deterministic = all(torch.equal(first, second) for first, second in zip(actual, second_actual, strict=True))
        deterministic = deterministic and torch.equal(recurrent, second_recurrent)
        assert recurrent_dtype == second_dtype
        assert deterministic, "prefill-to-decode outputs or final recurrent state are not bit-exact"

    expected_dtype = str(resolve_policy(candidate, "linear_attention").linear_recurrent_state_dtype)
    assert recurrent_dtype == expected_dtype, (recurrent_dtype, expected_dtype)
    assert torch.count_nonzero(recurrent).item() > 0
    result = {
        "command_result": "pass" if all_pcc_passed else "fail",
        "candidate": candidate,
        "batch": batch,
        "prefill_sequence": prefill_sequence,
        "decode_steps": decode_steps,
        "real_weights": real_weights,
        "recurrent_state_dtype": recurrent_dtype,
        "recurrent_nonzero": int(torch.count_nonzero(recurrent).item()),
        "pcc": pcc,
        "decode_median_ms": float(torch.tensor(elapsed_ms).median().item()),
        "decode_min_ms": float(min(elapsed_ms)),
        "repeat_runs": repeat_runs,
        "bit_exact": deterministic,
        "watcher_enabled": os.environ.get("TT_METAL_WATCHER") == "1",
    }
    print(
        "LINEAR_STATE_TRANSITION",
        f"candidate={candidate}",
        f"batch={batch}",
        f"prefill_sequence={prefill_sequence}",
        f"decode_steps={decode_steps}",
        f"state_dtype={recurrent_dtype}",
        f"median_ms={result['decode_median_ms']:.6f}",
        f"bit_exact={deterministic}",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(POLICIES), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), required=True)
    parser.add_argument("--prefill-sequence", type=int, default=129)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--repeat-runs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.prefill_sequence < (65 if args.batch == 1 else 5):
        parser.error("B1 must cross a 64-token chunk boundary; B32 must cover at least five non-aligned tokens")
    if args.decode_steps < 2:
        parser.error("--decode-steps must exercise a multi-step state transition")
    result = run(
        args.candidate,
        args.batch,
        args.prefill_sequence,
        args.decode_steps,
        args.real_weights,
        args.repeat_runs,
    )
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["command_result"] != "pass":
        raise SystemExit(1)
