# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Traced-decode correctness + speed for the dots.ocr TP4 text path.

Builds the text-only model (no vision tower) and runs greedy generation twice on
the same prompt -- once eager, once with the captured-trace decode -- on the same
opened device. Asserts the two token streams are identical (trace must not change
numerics) and prints decode ms/tok for both so the speedup is visible.

This device always reserves a trace region (the traced run needs it), so it does
not depend on the DOTS_OCR_TP4_TRACE env var.

Run::

    MESH_DEVICE=P150x4 pytest -s \\
        models/experimental/dots_ocr_tp4/tests/test_traced_decode.py
"""

import os

import pytest
import torch

import ttnn

from models.experimental.dots_ocr_tp4.tests.common import mesh_num_devices_for_shape, resolve_mesh_shape
from models.experimental.dots_ocr_tp4.tt.dots_ocr_model import DotsOCRModelTP4


def _device_params_trace():
    n = mesh_num_devices_for_shape(resolve_mesh_shape())
    dp = {"trace_region_size": 300_000_000, "num_command_queues": 1}
    dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING if n > 1 else ttnn.FabricConfig.DISABLED
    return dp


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download

    return snapshot_download("rednote-hilab/dots.ocr")


@pytest.mark.parametrize("device_params", [_device_params_trace()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("max_new_tokens", [int(os.environ.get("DOTS_OCR_TP4_DECODE_TOKENS", "32"))])
def test_tp4_traced_decode_matches_eager(mesh_device, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip(f"dots.ocr TP4 vision/decode tuned for Blackhole, got {mesh_device.arch().name}")
    if int(mesh_device.get_num_devices()) != 4:
        pytest.skip(f"dots.ocr TP4 requires a 4-device mesh (P150x4), got {mesh_device.get_num_devices()}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_path = _resolve_model_path()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Text-only: skip the vision tower build entirely.
    model = DotsOCRModelTP4.from_hf(mesh_device, hf_model, build_vision=False)

    messages = [{"role": "user", "content": "What is optical character recognition and how does it work?"}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"]

    # Eager reference (direct forward; unaffected by run mode).
    eager_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, stop_on_eos=False, use_trace=False)
    eager_t = dict(model.last_timings)

    # Traced: warm up (prime + capture prefill/decode traces) then replay.
    # NOTE: actual capture/replay only happens under TT_SYMBIOTE_RUN_MODE=TRACED
    # (bound at import); otherwise the graph path runs eager-via-framework.
    model.warmup(input_ids)
    traced_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, stop_on_eos=False, use_trace=True)
    traced_t = dict(model.last_timings)

    eager_ms = eager_t["decode_ms_per_token"]
    traced_ms = traced_t["decode_ms_per_token"]
    speedup = (eager_ms / traced_ms) if traced_ms > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"[dots_ocr_tp4 TRACE] max_new_tokens={max_new_tokens}")
    print(f"eager  decode: {eager_ms:.2f} ms/tok ({eager_t['decode_tok_per_s']:.1f} tok/s)")
    print(f"traced decode: {traced_ms:.2f} ms/tok ({traced_t['decode_tok_per_s']:.1f} tok/s)  ->  {speedup:.2f}x")
    bd = getattr(model, "last_timings_breakdown", None)
    if bd:
        print(
            f"  traced per-token split: host-write {bd['write_ms_per_token']:.2f} ms | "
            f"device-replay {bd['exec_ms_per_token']:.2f} ms | readback {bd['read_ms_per_token']:.2f} ms"
        )
    print(f"eager  : {eager_ids}")
    print(f"traced : {traced_ids}")
    print("=" * 70)

    # Trace replay must be numerically identical to eager. Compare the overlap
    # (eager has 1 extra token at the tail only if loops differ in count).
    n = min(len(eager_ids), len(traced_ids))
    assert n > 0
    assert (
        eager_ids[:n] == traced_ids[:n]
    ), f"traced decode diverged from eager:\n  eager ={eager_ids}\n  traced={traced_ids}"
