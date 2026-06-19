# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from dataclasses import dataclass, field
from typing import Tuple

from device_specs import DeviceClass, get_board_spec, get_deployment, is_galaxy


@dataclass
class SDXLConfig:
    """Configuration for SDXL standalone server.

    Topology fields (num_workers, device_mesh_shape, device_ids) are derived
    from device_specs.DEPLOYMENTS based on `board`. They are filled in during
    __post_init__ — do not set them manually.
    """

    # Required: which Tenstorrent board this server is running on.
    board: DeviceClass = None  # set explicitly via CLI; validated in __post_init__

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    # Topology — derived from device_specs.get_deployment("sdxl", board) in __post_init__.
    num_workers: int = 0
    device_ids: Tuple[int, ...] = ()
    device_mesh_shape: Tuple[int, int] = (0, 0)

    # Device parameters — l1_small_size and trace_region_size are derived from
    # BoardSpec in __post_init__. Pass a non-zero value at construction time to
    # override (ad-hoc tuning).
    l1_small_size: int = 0
    trace_region_size: int = 0

    # Model settings
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    hf_home: str = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # Pipeline settings
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    capture_trace: bool = True  # Re-enabled - trace is required for performance
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
        """Resolve topology from board, then apply dev_mode overrides."""
        if self.board is None:
            raise ValueError("SDXLConfig.board is required (pass --board on the CLI)")

        dep = get_deployment("sdxl", self.board)
        self.num_workers = dep.num_workers
        self.device_mesh_shape = dep.mesh_shape
        self.device_ids = dep.device_ids

        # Derive use_cfg_parallel from mesh layout. Mirrors tt-media-server's
        # base_device_runner.is_tensor_parallel = device_mesh_shape[0] > 1
        # and sdxl_generate_runner_trace.py: use_cfg_parallel=is_tensor_parallel.
        # Multi-chip TP layouts must run in cfg_parallel mode; otherwise the
        # text_embeds reshape gate at test_common.py:944 fires with the wrong
        # batch size.
        self.use_cfg_parallel = self.device_mesh_shape[0] > 1

        # Per-board L1 / trace region sizes (WH vs BH differ); explicit non-zero overrides.
        spec = get_board_spec(self.board)
        if not self.l1_small_size:
            self.l1_small_size = spec.l1_small_size
        if not self.trace_region_size:
            self.trace_region_size = spec.trace_region_size

        if self.dev_mode:
            self.num_workers = 1  # Single worker for faster startup
            self.num_inference_steps = 20  # Fewer steps for faster warmup
            self.device_ids = (self.device_ids[0],) if self.device_ids else (0,)

    @property
    def is_galaxy(self) -> bool:
        """Derived from `board`. Preserved as a property so existing call
        sites (e.g. sdxl_runner.py passing into TtSDXLPipelineConfig) keep
        working without changes."""
        return is_galaxy(self.board)
