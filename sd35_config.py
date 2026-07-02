# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os


@dataclass
class SD35Config:
    """Configuration for SD3.5 Large standalone server (2x4 mesh / LoudBox)"""

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    num_workers: int = 1  # Single worker owns entire 2x4 mesh

    # Device settings (LoudBox) - 1 worker with 2x4 mesh
    device_ids: Tuple[int, ...] = (
        0,
        1,
        2,
        3,
    )  # PCIe-attached L chips only; R chips auto-discovered via ethernet fabric
    device_mesh_shape: Tuple[int, int] = (2, 4)

    # Device parameters (from test_pipeline_sd35.py device_params fixture)
    fabric_config_name: str = "FABRIC_1D"  # String (not enum) for pickle compatibility
    l1_small_size: int = 32768
    trace_region_size: int = 25000000

    # Model settings
    model_name: str = "stabilityai/stable-diffusion-3.5-large"
    hf_home: str = field(default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    # Pipeline settings
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    image_width: int = 1024
    image_height: int = 1024
    use_trace: bool = True

    # Parallel config: None = use create_pipeline defaults for 2x4 mesh
    cfg_config: Optional[Tuple[int, int]] = None
    sp_config: Optional[Tuple[int, int]] = None
    tp_config: Optional[Tuple[int, int]] = None
    num_links: Optional[int] = None

    # Queue settings
    max_queue_size: int = 16
    inference_timeout_seconds: int = 600

    # Development mode: reads SD35_DEV_MODE env var
    # When True: num_inference_steps=5, use_trace=False (keeps all 8 devices)
    dev_mode: bool = field(default_factory=lambda: os.getenv("SD35_DEV_MODE", "false").lower() == "true")

    def __post_init__(self):
        """Apply dev mode overrides if enabled"""
        if self.dev_mode:
            self.num_inference_steps = 5  # Fewer steps for faster dev iteration
            self.use_trace = False  # Skip trace capture to avoid lengthy compilation
