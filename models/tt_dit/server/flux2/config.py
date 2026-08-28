# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Server configuration for the standalone FLUX.2 text-to-image server.

Everything is resolved from environment variables at process startup via
:meth:`ServerConfig.from_env`, so the container image needs no config file and the
launcher passes everything through the environment.

This module is import-safe on a machine with no Tenstorrent device: ``ttnn`` and the
pipeline class are imported lazily inside the methods that need them, never at module
scope. That lets the image's build-time verify step import this without a card.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger

_HF_CKPT_REPO = "black-forest-labs/FLUX.2-dev"

#: FLUX.2 needs BOTH sequence and tensor parallelism above 1: the attention path raises
#: on sp_factor == 1 (an unset `spatial`) and on tp_factor == 1 ("fused addcmul needs the
#: all-gather-matmul path"). On a 4-chip box 2x2 is therefore the only legal geometry.
_DEFAULT_MESH = (2, 2)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_mesh(name: str, default: tuple[int, int]) -> tuple[int, int]:
    """Parse a ``RxC`` mesh shape, e.g. ``2x2``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        rows, cols = (int(part) for part in raw.lower().split("x", 1))
    except ValueError as err:
        msg = f"{name} must look like '2x2' (got {raw!r})"
        raise ValueError(msg) from err
    return rows, cols


@dataclass
class ServerConfig:
    """Resolved server configuration. See :meth:`from_env` for the variable names."""

    checkpoint: str = _HF_CKPT_REPO
    mesh_shape: tuple[int, int] = _DEFAULT_MESH
    sp_axis: int = 0
    tp_axis: int = 1
    encoder_tp_axis: int = 1
    vae_tp_axis: int = 1
    num_links: int = 2
    topology: str = "linear"

    # Memory / execution strategy. On a 4-chip box the 32B transformer, the 24B text
    # encoder and the VAE cannot be co-resident, so dynamic_load is required and FSDP
    # shards the transformer's weights across the sequence axis.
    is_fsdp: bool = True
    dynamic_load: bool = True
    traced: bool = True
    run_warmup: bool = True

    # Warmed generation shape. Requests may not change it: the pipeline compiles and
    # traces for one resolution, and a different one would silently recompile.
    height: int = 1024
    width: int = 1024
    default_steps: int = 50
    default_guidance: float = 4.0

    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = "/tmp/flux2-server"
    max_jobs: int = 32
    job_retention_seconds: int = 3600

    @classmethod
    def from_env(cls) -> ServerConfig:
        cfg = cls(
            checkpoint=os.environ.get("FLUX2_CHECKPOINT", _HF_CKPT_REPO),
            mesh_shape=_env_mesh("FLUX2_MESH_SHAPE", _DEFAULT_MESH),
            sp_axis=_env_int("FLUX2_SP_AXIS", 0),
            tp_axis=_env_int("FLUX2_TP_AXIS", 1),
            encoder_tp_axis=_env_int("FLUX2_ENCODER_TP_AXIS", 1),
            vae_tp_axis=_env_int("FLUX2_VAE_TP_AXIS", 1),
            num_links=_env_int("FLUX2_NUM_LINKS", 2),
            topology=os.environ.get("FLUX2_TOPOLOGY", "linear").strip().lower(),
            is_fsdp=_env_bool("FLUX2_FSDP", True),
            dynamic_load=_env_bool("FLUX2_DYNAMIC_LOAD", True),
            traced=_env_bool("FLUX2_TRACED", True),
            run_warmup=_env_bool("FLUX2_WARMUP", True),
            height=_env_int("FLUX2_HEIGHT", 1024),
            width=_env_int("FLUX2_WIDTH", 1024),
            default_steps=_env_int("FLUX2_STEPS", 50),
            host=os.environ.get("FLUX2_HOST", "0.0.0.0"),
            port=_env_int("FLUX2_PORT", 8000),
            output_dir=os.environ.get("FLUX2_OUTPUT_DIR", "/tmp/flux2-server"),
            max_jobs=_env_int("FLUX2_MAX_JOBS", 32),
            job_retention_seconds=_env_int("FLUX2_JOB_RETENTION_S", 3600),
        )
        cfg.validate()
        logger.info(
            f"ServerConfig: mesh={cfg.mesh_shape} sp={cfg.sp_axis} tp={cfg.tp_axis} "
            f"topology={cfg.topology} fsdp={cfg.is_fsdp} dynamic_load={cfg.dynamic_load} "
            f"traced={cfg.traced} shape={cfg.width}x{cfg.height} steps={cfg.default_steps}"
        )
        return cfg

    def validate(self) -> None:
        """Reject configurations the pipeline would only fail on deep inside device init."""
        rows, cols = self.mesh_shape
        if rows < 1 or cols < 1:
            msg = f"mesh_shape must be positive, got {self.mesh_shape}"
            raise ValueError(msg)
        if self.topology not in ("linear", "ring"):
            msg = f"topology must be 'linear' or 'ring', got {self.topology!r}"
            raise ValueError(msg)
        if self.sp_axis == self.tp_axis:
            msg = f"sp_axis and tp_axis must differ (both {self.sp_axis})"
            raise ValueError(msg)

        # The constraint that cost a day of debugging: flux2 attention requires both
        # factors > 1, so a line mesh silently picks a code path that raises.
        sp_factor = self.mesh_shape[self.sp_axis]
        tp_factor = self.mesh_shape[self.tp_axis]
        if sp_factor < 2 or tp_factor < 2:
            msg = (
                f"FLUX.2 needs sequence and tensor parallel factors both > 1, but mesh "
                f"{self.mesh_shape} with sp_axis={self.sp_axis}/tp_axis={self.tp_axis} gives "
                f"sp={sp_factor}, tp={tp_factor}. Use a 2-D mesh such as 2x2."
            )
            raise ValueError(msg)

        for name, value in (("height", self.height), ("width", self.width)):
            if value % 16 != 0:
                msg = f"{name} must be divisible by 16 (got {value})"
                raise ValueError(msg)

    def ttnn_topology(self):
        """The ttnn topology enum matching :attr:`topology`."""
        import ttnn

        return ttnn.Topology.Ring if self.topology == "ring" else ttnn.Topology.Linear

    def pipeline_class(self):
        """Imported lazily so this module stays usable without a device."""
        from ...pipelines.flux2.pipeline_flux2 import Flux2Pipeline

        return Flux2Pipeline
