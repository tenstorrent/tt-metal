# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Prefill's stage hang detector: ttnn.stream_hang_detection's detector, started with a prefill rank's services and
watcher and with settings from TT_PREFILL_STAGE_HANG_* environment variables."""

from __future__ import annotations

import os

from ttnn.stream_hang_detection import AckStream, HangSettings, Stream, StreamHangDetector, describe_devices

import ttnn
from models.demos.common.prefill.runners.hang_watcher import PrefillStreamHangWatcher

ENABLED_ENV = "TT_PREFILL_STAGE_HANG_DETECTION_ENABLED"
TIMEOUT_ENV = "TT_PREFILL_STAGE_HANG_TIMEOUT_SECONDS"
FIRST_CHUNK_TIMEOUT_ENV = "TT_PREFILL_STAGE_HANG_FIRST_CHUNK_TIMEOUT_SECONDS"
POLL_ENV = "TT_PREFILL_STAGE_HANG_POLL_SECONDS"
COMMAND_ENV = "TT_PREFILL_STAGE_HANG_COMMAND_TO_EXECUTE"
COMMAND_TIMEOUT_ENV = "TT_PREFILL_STAGE_HANG_COMMAND_TIMEOUT_SECONDS"
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_FIRST_CHUNK_TIMEOUT_S = 100.0  # covers trace capture and cold first-chunk compiles
DEFAULT_POLL_S = 2.0
DEFAULT_COMMAND_TIMEOUT_S = 300.0  # room for a full tt-triage run


def arm_hang_detector(
    mesh_device: ttnn.MeshDevice,
    rank: int,
    *,
    h2d: ttnn.H2DStreamService | None,
    d2d_in: ttnn.D2DStreamServiceReceiver | None,
    d2d_out: ttnn.D2DStreamServiceSender | None,
    d2h: ttnn.D2HStreamService | None,
    idx_in: ttnn.D2DStreamServiceReceiver | None,
    idx_out: ttnn.D2DStreamServiceSender | None,
    acks_per_chunk: int,
    sends_warmup: bool,
    warmup_acks: int,
    layers: str,
) -> StreamHangDetector | None:
    """Start this rank's watcher when TT_PREFILL_STAGE_HANG_DETECTION_ENABLED=1 and a D2D output or the D2H layer
    acks say a chunk finished.

    sends_warmup: before the first real chunk, every rank with a D2D output sends a warm-up chunk to the next.
    warmup_acks: layer acks the trace capture's warm pass fires.
    """
    if os.environ.get(ENABLED_ENV, "0") != "1":
        return None
    inputs = (("h2d", h2d), ("activation", d2d_in), ("indices", idx_in))
    outputs = (("activation", d2d_out), ("indices", idx_out))
    devices = describe_devices(
        mesh_device,
        inputs=[Stream(name, service) for name, service in inputs if service is not None],
        outputs=[Stream(name, service) for name, service in outputs if service is not None],
        acks=None if d2h is None else AckStream("layer acks", d2h, acks_per_chunk),
    )
    if devices is None:
        return None
    # Chunks out and in before the first real one: the warm-up send, or the warm pass's acks without a D2D output.
    warmup_out = int(sends_warmup) if d2d_out is not None else (warmup_acks // acks_per_chunk if warmup_acks else 0)
    watcher = PrefillStreamHangWatcher(
        pid=os.getpid(),
        settings=HangSettings(
            timeout_s=float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S)),
            first_chunk_timeout_s=float(os.environ.get(FIRST_CHUNK_TIMEOUT_ENV, DEFAULT_FIRST_CHUNK_TIMEOUT_S)),
            poll_s=float(os.environ.get(POLL_ENV, DEFAULT_POLL_S)),
            command=os.environ.get(COMMAND_ENV, ""),
            command_timeout_s=float(os.environ.get(COMMAND_TIMEOUT_ENV, DEFAULT_COMMAND_TIMEOUT_S)),
        ),
        devices=devices,
        stage_idx=rank,
        layers=layers,
        warmup_out=warmup_out,
        warmup_in=int(sends_warmup and rank > 0),
    )
    return StreamHangDetector(watcher)
