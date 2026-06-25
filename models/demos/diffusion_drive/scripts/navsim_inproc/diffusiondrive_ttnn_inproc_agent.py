# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
In-process NavSim agent for the DiffusionDrive TTNN model (single-env path).

Unlike ``scripts/navsim_bridge/`` — which delegates inference to a separate
``ttnn_pdm_server.py`` over a Unix-domain socket because navsim was Python 3.9
and ``ttnn`` is Python 3.10 — this agent runs the TTNN model **in the same
process** as the NavSim eval harness.  Run it in the ``navsim310`` conda env
(Python 3.10), where both the navsim stack and ``ttnn`` import.

Why this is possible now: the upstream DiffusionDrive ``python310`` branch
modernised navsim's dependencies to Python 3.10 (torch 2.0.1+cpu, numpy 1.23.4,
…), materialised as the ``navsim310`` conda env — so navsim and the compiled
``ttnn`` wheel coexist.  No socket, no second process, no server lifecycle.

Device arbitration: one Wormhole = one device handle.  Run the eval with
``worker=single_machine_thread_pool`` (threads share this single agent instance
and the one device behind ``self._lock``; feature-building still parallelises)
or ``worker=sequential``.  Do **not** use ``worker=ray_distributed``: each ray
worker is a separate process and would try to open device 0 concurrently.

Performance note: switching from the bridge to in-process is an operational
simplification, not by itself a throughput win — the IPC it removes (~pickle +
socket round-trip of the feature payload) is <1% of per-scenario time, and the
device forward serialises either way.  The throughput lever is **trace capture**
(01_plan.md §10.4 #1): this agent captures the consolidated backbone-loop trace
once in ``initialize()`` and replays it per scene via ``execute_compiled()``
(~1.34× full-model forward, traced-vs-eager trajectory PCC 1.0).  Set
``DD_TRACE=0`` to fall back to the eager forward (for A/B).

Wire-up (see scripts/navsim_inproc/README.md for the full recipe + caveats):
  1. ``conda activate navsim310``
  2. one-time: install ttnn's pure-Python deps into navsim310 and put the
     tt-metal ttnn *inner-package parent* on PYTHONPATH (just /root/tt/tt-metal
     resolves to an empty namespace package — ttnn.open_device would be missing):
       pip install --no-deps loguru==0.6.0 graphviz==0.21 seaborn==0.13.2 ml_dtypes==0.5.4
       export PYTHONPATH=$BR:/root/tt/tt-metal/ttnn:/root/tt/tt-metal:/root/tt/tt-metal/tools:$NAVSIM_DEVKIT_ROOT
  3. copy ``diffusiondrive_ttnn_inproc_agent.yaml`` into the navsim agent config dir,
  4. run ``run_pdm_score.py`` with ``agent=diffusiondrive_ttnn_inproc_agent``.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any, Dict, List

import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import SensorConfig


class DiffusionDriveTtnnInprocAgent(AbstractAgent):
    """NavSim agent that runs the DiffusionDrive TTNN model in-process on a
    Tenstorrent Wormhole device (no cross-process bridge)."""

    def __init__(
        self,
        config,
        checkpoint_path: str,
        anchors_path: str,
        lr: float = 6e-4,
        device_id: int = 0,
    ):
        super().__init__()
        self._config = config
        self._checkpoint_path = checkpoint_path
        self._anchors_path = anchors_path
        self._lr = lr
        self._device_id = device_id
        # Built lazily in initialize() (once per process) — never at construction,
        # so the agent can be created on the driver before the device is needed.
        self._device = None
        self._model = None
        # Serialises the single device across threads (single_machine_thread_pool);
        # uncontended under worker=sequential.
        self._lock = threading.Lock()
        # Stage 7: True once the backbone-loop trace is captured (set in initialize()).
        self._traced = False

    # -- required hooks --------------------------------------------------
    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """Open the Wormhole device and build the full on-device TTNN stack.

        Called once per worker process; idempotent and thread-safe (a second
        call, or a concurrent call from another thread, is a no-op).
        """
        with self._lock:
            if self._model is not None:
                return
            import ttnn  # local import: only the worker process needs ttnn
            from models.demos.diffusion_drive.tt.config import ModelConfig
            from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

            trace_enabled = os.environ.get("DD_TRACE", "1").strip().lower() not in ("0", "false", "no", "off")
            # DD-1: ttnn.conv2d allocates from the L1_SMALL bank; default 0 → OOM.
            # Trace capture additionally needs a trace region (256 MB validated).
            open_kwargs = {"device_id": self._device_id, "l1_small_size": 32768}
            if trace_enabled:
                open_kwargs["trace_region_size"] = 256 * 1024 * 1024
            self._device = ttnn.open_device(**open_kwargs)
            atexit.register(self._close)

            cfg = ModelConfig()
            cfg.plan_anchor_path = self._anchors_path
            # latent=False → use the real LiDAR BEV the feature builder produces.
            model = TtnnDiffusionDriveModel.from_checkpoint(self._checkpoint_path, cfg, self._device, latent=False)
            # Full on-device stack — identical chain to scripts/ttnn_pdm_server._build_model:
            # backbone (stems + BasicBlocks + FPN + GPT fusion, device-native/consolidated),
            # perception head, DDIM denoiser, agent head; build_stage4 consolidates the
            # perception block + the DDIM decoder loop. Valid at the production resolution
            # the feature builder emits (camera 256×1024, LiDAR 256×256), where the
            # pool/upsample ratios are integer.
            (
                model.build_stage2(self._device)
                .build_stage3(self._device)
                .build_stage3_4(self._device)
                .build_stage3_5(self._device)
                .build_stage3_6(self._device)
                .build_stage3_7(self._device)
                .build_stage4(self._device)
            )
            self._model = model

            # Stage 7: capture the backbone-loop trace once for fast replay
            # (~1.34× full-model vs eager, traced-vs-eager trajectory PCC 1.0).
            # A capture failure falls back gracefully rather than aborting the eval.
            if trace_enabled:
                try:
                    model.compile()
                    self._traced = True
                    logging.getLogger(__name__).info(
                        "DiffusionDrive backbone-loop trace captured — using execute_compiled() path"
                    )
                except Exception as exc:  # never let trace capture break the eval
                    logging.getLogger(__name__).warning(
                        "DiffusionDrive trace capture failed (%s); falling back to eager forward", exc
                    )

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[3])

    def get_feature_builders(self) -> List[TransfuserFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    # -- inference -------------------------------------------------------
    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """Run the TTNN model on-device and return ``{"trajectory": (1,8,3) cpu}``.

        ``AbstractAgent.compute_trajectory`` passes features already carrying a
        batch dim and then does ``predictions["trajectory"].squeeze(0).numpy()``,
        so the returned tensor must be a CPU float tensor.

        DDIM noise is intentionally left **unseeded** (drawn fresh each forward)
        to match the upstream stochastic eval — see scripts/navsim_bridge/README.md.
        """
        if self._model is None:  # defensive: harness should call initialize() first
            self.initialize()
        feats = {
            "camera_feature": features["camera_feature"].float(),
            "lidar_feature": features["lidar_feature"].float(),
            "status_feature": features["status_feature"].float(),
        }
        with self._lock:  # one device → one forward at a time
            with torch.no_grad():
                out = self._model.execute_compiled(feats) if self._traced else self._model(feats)
        return {"trajectory": out["trajectory"].float().cpu()}

    # -- cleanup ---------------------------------------------------------
    def _close(self) -> None:
        if self._device is not None:
            try:
                import ttnn

                ttnn.close_device(self._device)
            except Exception:
                pass
            self._device = None
