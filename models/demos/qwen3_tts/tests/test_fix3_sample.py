# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test generation with sampling (temperature=0.9) to check EOS behavior."""

import os
import traceback

import torch
import soundfile as sf

os.environ["HF_MODEL"] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

import ttnn
from models.demos.qwen3_tts.tt.generator import TTSGenerator

device_ids = ttnn.get_device_ids()
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, len(device_ids)),
    dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
)
try:
    mesh.enable_program_cache()
except AttributeError:
    ttnn.enable_program_cache(mesh)

try:
    gen = TTSGenerator.build("Qwen/Qwen3-TTS-12Hz-1.7B-Base", mesh, max_seq_len=2560)

    tests = [
        ("はい", "hai"),
        ("こんにちは", "konnichiwa"),
        ("今日はいい天気ですね", "tenki"),
    ]

    for text, name in tests:
        wav, sr = gen.generate(
            text, language="japanese",
            max_new_tokens=500, temperature=0.9, top_k=50, top_p=1.0,
            repetition_penalty=1.05,
        )
        path = f"/home/yito/ttwork/tt-metal/demo_ja_output/fix3_{name}.wav"
        sf.write(path, wav, sr)
        print(f"{name}: {len(wav)} samples, {len(wav)/sr:.2f}s audio -> {path}")

except Exception as e:
    traceback.print_exc()
finally:
    ttnn.close_mesh_device(mesh)
