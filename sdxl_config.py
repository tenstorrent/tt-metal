# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class SDXLConfig:
    """Configuration for SDXL standalone server"""

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    num_workers: int = 4  # T3K = 4 workers, one per device

    # Device settings (T3K) - 4 workers with 1x1 mesh each
    device_ids: Tuple[int, ...] = (0, 1, 2, 3)
    device_mesh_shape: Tuple[int, int] = (1, 1)
    is_galaxy: bool = False

    # Device parameters
    l1_small_size: int = 23000
    trace_region_size: int = 34000000

    # Model settings
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    hf_home: str = os.getenv("HF_HOME", "/mnt/MLPerf/tt_dnn-models/hf_home")

    # Pipeline settings
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    capture_trace: bool = True  # Re-enabled with in-place embedding updates
    vae_on_device: bool = True
    encoders_on_device: bool = True
    use_cfg_parallel: bool = False

    # Queue settings
    max_queue_size: int = 64
    max_batch_size: int = 1  # Stable for (1,1) mesh
    inference_timeout_seconds: int = 300

    # Development mode settings
    dev_mode: bool = field(default_factory=lambda: os.getenv("SDXL_DEV_MODE", "false").lower() == "true")

    def __post_init__(self):
        """Apply dev mode overrides if enabled"""
        if self.dev_mode:
            self.num_workers = 1  # Single worker for faster startup
            self.num_inference_steps = 20  # Fewer steps for faster warmup
            self.device_ids = (0,)  # Single device only
