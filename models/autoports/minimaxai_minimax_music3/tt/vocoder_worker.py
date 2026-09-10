# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The host fp32 vocoder in a separate process (stage 07).

Running the torch vocoder in a *thread* next to the DiT loop starves it: the loop's host side sits in blocking ttnn
read-backs that hold the GIL, so the vocoder's ~200 small torch ops per window wait for it (31.6 s instead of 6.8 s for a
window overlapped with denoising, doc/optimize/README.md). A separate process has its own interpreter and thread pool;
the pipeline hands it the ``[1, 128, L]`` fp32 latents of window k (350 KB) while the chip denoises window k + 1 and
collects the ``[1, 2, 512 L]`` waveforms (2.8 MB) in order afterwards. The worker never imports ttnn or touches the
device; it is started with the ``spawn`` method so no device state is inherited.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import torch

_VOCODER = None


def _init_worker(weights_dir: str, threads: int) -> None:
    global _VOCODER
    from models.autoports.minimaxai_minimax_music3.reference import vocoder_ref as V

    torch.set_num_threads(int(threads))
    _VOCODER = V.load_vocoder(Path(weights_dir), dtype=torch.float32)


def _vocode(latents: torch.Tensor):
    t0 = time.perf_counter()
    with torch.no_grad():
        wav = _VOCODER(latents.float())
    return wav, time.perf_counter() - t0


class VocoderProcess:
    """One spawned worker holding its own fp32 vocoder; ``submit`` returns a Future of ``(waveform, seconds)``."""

    def __init__(self, weights_dir: Path, threads: int = 10):
        self.weights_dir = str(weights_dir)
        self.threads = threads
        self._pool: Optional[ProcessPoolExecutor] = None

    def start(self) -> None:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.weights_dir, self.threads),
            )

    def warm(self) -> float:
        """Force the worker to start and load its weights (returns the seconds it took)."""
        self.start()
        t0 = time.perf_counter()
        self._pool.submit(_vocode, torch.zeros(1, 128, 4)).result()
        return time.perf_counter() - t0

    def submit(self, latents: torch.Tensor) -> Future:
        self.start()
        return self._pool.submit(_vocode, latents.detach().float().contiguous())

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
