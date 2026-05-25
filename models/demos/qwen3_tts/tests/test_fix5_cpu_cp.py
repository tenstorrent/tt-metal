# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test with CPU Code Predictor for numerical accuracy."""

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

    wav, sr = gen.generate(
        "こんにちは", language="japanese",
        max_new_tokens=500, temperature=0.0,
    )
    os.makedirs("/home/yito/ttwork/tt-metal/demo_ja_output", exist_ok=True)
    sf.write("/home/yito/ttwork/tt-metal/demo_ja_output/fix5_konnichiwa.wav", wav, sr)
    print(f"konnichiwa: {len(wav)} samples, {len(wav)/sr:.2f}s")

except Exception as e:
    traceback.print_exc()
finally:
    ttnn.close_mesh_device(mesh)
