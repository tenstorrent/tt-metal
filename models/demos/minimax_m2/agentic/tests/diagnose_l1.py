#!/usr/bin/env python3
"""Diagnose what Whisper leaves in L1 after a transcription."""
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/home/ubuntu/agentic/tt-metal")

from loguru import logger

import ttnn

mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 2),
    l1_small_size=300_000,
    trace_region_size=100_000_000,
    num_command_queues=2,
)
mesh.enable_program_cache()

num_devices = mesh.get_num_devices() if hasattr(mesh, "get_num_devices") else 1
chip_views = []
if num_devices > 1:
    for i in range(num_devices):
        chip_views.append(mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, i)))
else:
    chip_views = [mesh]


def show_l1(label):
    for i, dev in enumerate(chip_views):
        try:
            addr = (
                dev.lowest_occupied_compute_l1_address()
                if hasattr(dev, "lowest_occupied_compute_l1_address")
                else "N/A"
            )
            logger.info(f"  {label} chip{i} lowest_L1={addr}")
        except Exception as e:
            logger.info(f"  {label} chip{i}: {e}")


show_l1("BEFORE Whisper load")

from models.demos.minimax_m2.agentic.tool_wrappers.whisper_tool import WhisperTool

w = WhisperTool(mesh_device=mesh)
show_l1("AFTER Whisper load (before transcribe)")

t = np.linspace(0, 2, 32000, endpoint=False)
with tempfile.TemporaryDirectory() as d:
    sf.write(f"{d}/t.wav", (np.sin(2 * 3.14159 * 440 * t) * 0.5).astype(np.float32), 16000)
    txt = w.transcribe(f"{d}/t.wav")
    logger.info(f"Transcribed: {txt!r}")

show_l1("AFTER transcribe (trace alive)")

w.close()
logger.info("Whisper closed")
show_l1("AFTER Whisper close (trace released)")

ttnn.close_mesh_device(mesh)
logger.info("Done")
