# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in traces of the existing unconditioned LTX Euler tail."""

import math

import ttnn

from .tracing import Tracer


def euler_tail(video_lat, audio_lat, video_velocity, audio_velocity, video_mask, audio_mask, dt):
    """Keep the baseline operation order, rounding, masks and in-place updates."""
    v_vel = ttnn.typecast(video_velocity, ttnn.bfloat16)
    ttnn.multiply_(v_vel, video_mask)
    ttnn.multiply_(v_vel, dt)
    ttnn.add_(video_lat, v_vel)
    ttnn.multiply_(video_lat, video_mask)
    a_vel = ttnn.typecast(audio_velocity, ttnn.bfloat16)
    ttnn.multiply_(a_vel, audio_mask)
    ttnn.multiply_(a_vel, dt)
    ttnn.add_(audio_lat, a_vel)
    ttnn.multiply_(audio_lat, audio_mask)
    # Cast results are trace scratch, never persistent outputs of this owner.
    return None


class EulerTail:
    """One stage's tail traces, borrowing its persistent DiT inputs and outputs.

    The producer must already own a real trace. No new persistent device input
    is allocated here: latents/masks predate DiT capture, and velocities are the
    retained declared outputs of that capture. Reject address changes instead of
    copying into stale producer buffers. Only Python ``dt`` scalars are cached.
    Release this owner before releasing the producer or any of its buffers.
    """

    def __init__(self, device):
        self._device = device
        self._tracers = {}
        self._signature = None
        self._inputs = None
        self._closed = False
        self._failed = False

    def __call__(self, video_lat, audio_lat, video_velocity, audio_velocity, video_mask, audio_mask, dt, blocking=True):
        assert not self._closed and not self._failed, "release and recreate the Euler owner after cleanup/failure"
        assert isinstance(dt, float) and math.isfinite(dt) and dt < 0, "Euler dt must be finite and negative"
        args = (video_lat, audio_lat, video_velocity, audio_velocity, video_mask, audio_mask)
        signature = tuple(
            (
                value.device().id(),
                value.buffer_address(),
                tuple(value.shape),
                value.dtype,
                value.layout,
                value.memory_config(),
            )
            for value in args
        )
        assert all(value.device() == self._device for value in args)
        if self._signature is None:
            self._signature, self._inputs = signature, args
        else:
            assert signature == self._signature, "Euler trace inputs changed; release producer and tail traces first"
        tracer = self._tracers.get(dt)
        if tracer is None:
            # Preparation advances CLONED inputs only. Capture records commands
            # without executing them; Tracer executes the captured tail once.
            tracer = Tracer(euler_tail, device=self._device, prep_run=True, clone_prep_inputs=True)
            self._tracers[dt] = tracer
        try:
            return tracer(*args, dt=dt, tracer_blocking_execution=blocking)
        except BaseException:
            # A partial device failure cannot be rolled back by host bookkeeping.
            self._failed = True
            raise

    def release(self):
        for tracer in self._tracers.values():
            tracer.release_trace()
        self._tracers.clear()
        self._inputs = None
        self._signature = None
        self._closed = True
