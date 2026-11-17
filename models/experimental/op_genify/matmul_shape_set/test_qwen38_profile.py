# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile the stock Qwen3.8-27B QB2 model for the #65148 matmul shape set.

Mirrors `models/demos/qwen38_27b_qb2/demo/text_demo.py --profile`: same fabric, mesh, generator and
reduced stack (layer 0 = linear attention, layer 3 = full attention), with one signposted section per
prefill length and one per decode step. Every layer of a kind has the same shapes, so per-layer calls
times the layer counts (48 linear + 16 full) give the full model.

Run (needs MODEL_WEIGHTS_DIR and a Tracy build):
    scripts/run_safe_pytest.sh --profile models/experimental/op_genify/matmul_shape_set/test_qwen38_profile.py
"""

from pathlib import Path

import torch

import ttnn
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric

PREFILL_LENGTHS = (128, 512, 1024, 2048, 4096)
LAYERS = [0, 3]


def _signpost(name):
    import tracy

    tracy.signpost(name)


def test_qwen38_matmul_profile():
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(
            Path(__file__).resolve().parents[3] / "demos" / "qwen38_27b_qb2",
            mesh,
            layer_indices=LAYERS,
            sampling_strategy="split",
            head_strategy="dram",
        )
        ttnn.ReadDeviceProfiler(mesh)
        base = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain why the sky is blue."}], tokenize=False, add_generation_prompt=True
        )
        base = gen.tokenizer(base, add_special_tokens=False)["input_ids"]
        gen._ensure_cache(1, max(PREFILL_LENGTHS) + 127)
        gen._ensure_history(127)
        for length in PREFILL_LENGTHS:
            prompt = (base * ((length + len(base) - 1) // len(base)))[:length]
            gen.generate(prompt, 2)  # warm: compile, capture traces
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            _signpost(f"PREFILL_M{length}")
            gen.generate(prompt, 1)
            ttnn.synchronize_device(mesh)
            _signpost(f"PREFILL_M{length}_END")
            ttnn.ReadDeviceProfiler(mesh)
        # One decode step through the captured decode trace (batch 1, M padded to one 32-row tile).
        _signpost("DECODE_B1")
        ttnn.execute_trace(mesh, gen.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        _signpost("DECODE_B1_END")
        ttnn.ReadDeviceProfiler(mesh)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
