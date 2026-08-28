# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Owns the warm mesh device + FLUX.2 pipeline for the lifetime of the server.

One process holds one mesh and one pipeline. Weights are large enough that
reconstructing per request is not viable, and ttnn wants a single owning thread, so
every device-touching call runs on the job store's single executor thread.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from loguru import logger

from .config import ServerConfig
from .device import close_mesh, open_mesh


class PipelineHolder:
    """Holds the mesh device and the one warm pipeline.

    Thread-safety: ``startup``, ``generate`` and ``close`` are only ever called on the
    single executor thread, so the pipeline needs no extra locking. ``is_ready`` is
    queried from the event-loop thread, so the readiness flag is guarded by a lock.
    """

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self._mesh = None
        self._fabric_config = None
        self._pipeline = None
        self._ready = False
        self._ready_lock = threading.Lock()

    # --- lifecycle (executor thread) -----------------------------------------
    def startup(self) -> None:
        """Open the mesh, build the pipeline, and warm/compile it.

        Blocks until the pipeline is resident and ready to serve. The first ready can
        take minutes: weights convert to device layout, and with ``run_warmup`` the
        pipeline additionally runs a short generation to capture its trace.
        """
        cfg = self.cfg
        t0 = time.time()
        os.makedirs(cfg.output_dir, exist_ok=True)

        logger.info(f"PipelineHolder.startup: opening mesh {cfg.mesh_shape}")
        self._mesh, self._fabric_config = open_mesh(cfg)

        pipeline_class = cfg.pipeline_class()
        logger.info(
            f"Creating FLUX.2 pipeline from {cfg.checkpoint} "
            f"(fsdp={cfg.is_fsdp}, dynamic_load={cfg.dynamic_load}, traced={cfg.traced}). "
            "This converts tens of GB of weights and may take several minutes."
        )
        # Pass topology explicitly so the device fabric and the pipeline CCL topology
        # can never disagree.
        self._pipeline = pipeline_class.create_pipeline(
            mesh_device=self._mesh,
            checkpoint_name=cfg.checkpoint,
            sp_axis=cfg.sp_axis,
            tp_axis=cfg.tp_axis,
            encoder_tp_axis=cfg.encoder_tp_axis,
            vae_tp_axis=cfg.vae_tp_axis,
            vae_h_axis=1 - cfg.vae_tp_axis,
            vae_w_axis=None,
            num_links=cfg.num_links,
            topology=cfg.ttnn_topology(),
            width=cfg.width,
            height=cfg.height,
            is_fsdp=cfg.is_fsdp,
            dynamic_load=cfg.dynamic_load,
            trace_warmup=cfg.run_warmup and cfg.traced,
            shard_prompt=True,
        )

        with self._ready_lock:
            self._ready = True
        logger.info(f"PipelineHolder.startup complete in {time.time() - t0:.1f}s — ready to serve.")

    def is_ready(self) -> bool:
        with self._ready_lock:
            return self._ready

    def close(self) -> None:
        """Release traces and close the mesh. Best-effort: teardown must not raise."""
        if self._pipeline is not None and hasattr(self._pipeline, "release_traces"):
            try:
                self._pipeline.release_traces()
            except Exception as exc:  # noqa: BLE001 — teardown is best-effort
                logger.warning(f"Error releasing traces: {exc}")
        self._pipeline = None
        close_mesh(self._mesh, self._fabric_config)
        self._mesh = None
        with self._ready_lock:
            self._ready = False

    # --- inference (executor thread) -----------------------------------------
    def generate(self, job_id: str, params: dict[str, Any]) -> str:
        """Run one generation and return the written PNG path."""
        if self._pipeline is None:
            msg = "Pipeline is not initialized"
            raise RuntimeError(msg)

        cfg = self.cfg
        output_path = os.path.join(cfg.output_dir, f"{job_id}.png")

        kwargs = {
            "prompts": [params["prompt"]],
            "num_inference_steps": params.get("num_inference_steps") or cfg.default_steps,
            "guidance_scale": params.get("guidance_scale") or cfg.default_guidance,
            "traced": cfg.traced,
        }
        seed = params.get("seed")
        if seed is not None:
            kwargs["seed"] = seed

        t0 = time.time()
        logger.info(f"job {job_id}: generating ({kwargs['num_inference_steps']} steps)")
        images = self._pipeline(**kwargs)
        images[0].save(output_path)
        logger.info(f"job {job_id}: done in {time.time() - t0:.1f}s -> {output_path}")
        return output_path
