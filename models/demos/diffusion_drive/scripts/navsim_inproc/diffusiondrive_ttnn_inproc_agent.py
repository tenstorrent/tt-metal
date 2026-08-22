# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
In-process NavSim agent for the DiffusionDrive TTNN model (single-env path).

Unlike ``scripts/navsim_bridge/`` — which delegates inference to a separate
``ttnn_pdm_server.py`` over a Unix-domain socket because navsim was Python 3.9
and ``ttnn`` is Python 3.10 — this agent runs the TTNN model **in the same
process** as the NavSim eval harness.  Run it in the ``navsim`` conda env
(Python 3.10), where both the navsim stack and ``ttnn`` import.

Why this is possible now: the upstream DiffusionDrive ``python310`` branch
modernised navsim's dependencies to Python 3.10 (torch 2.0.1+cpu, numpy 1.23.4,
…), materialised as the ``navsim`` conda env — so navsim and the compiled
``ttnn`` wheel coexist.  No socket, no second process, no server lifecycle.

Device arbitration — one Wormhole = one device handle, the TTNN device context is
**thread-affine**, AND navsim's ``run_pdm_score`` instantiates a FRESH agent and
calls ``initialize()`` **per worker chunk** (not once, shared).  The agent picks
the right path **automatically from the worker type — no env var** (see
:func:`_is_pool_worker_thread`):

  * ``worker=sequential``: one chunk runs on the MAIN thread → one agent → opens +
    uses the device on that thread.  Simplest; the validated path.
  * ``worker=single_machine_thread_pool``: the N worker chunks each build a (cheap)
    agent on a pool thread, but all of them delegate to a **process-wide singleton**
    (:class:`_DeviceWorker`) that owns the device + model on ONE dedicated thread and
    serves forwards off a queue.  Worker threads build features in parallel (CPU) and
    enqueue, overlapping that CPU with the serialized device forward (~1.59× vs
    sequential).  Required because (a) per-chunk agents would otherwise each
    ``open_device(0)`` → conflict, and (b) the device context is thread-affine so all
    device ops must run on the owner thread.

Do **not** use ``worker=ray_distributed``: each ray worker is a separate process
and would try to open device 0 concurrently.

Performance note: in-process vs the bridge is an operational simplification, not
by itself a throughput win.  The levers are **trace capture** (the singleton/
sequential build captures the consolidated backbone-loop trace and replays it via
``execute_compiled()``; ~1.34× forward, PCC 1.0; ``DD_TRACE=0`` to disable) and,
because the in-process per-scene cost is dominated by navsim CPU (feature-build +
PDM scoring), the ``single_machine_thread_pool`` funnel above (~1.59×, the
dominant eval lever).

Wire-up (see scripts/navsim_inproc/README.md for the full recipe + caveats):
  1. ``conda activate navsim``
  2. one-time: install ttnn's pure-Python deps into navsim and put the
     tt-metal ttnn *inner-package parent* on PYTHONPATH (just $TT_METAL_HOME
     resolves to an empty namespace package — ttnn.open_device would be missing):
       pip install --no-deps loguru==0.6.0 graphviz==0.21 seaborn==0.13.2 ml_dtypes==0.5.4
       export PYTHONPATH=$BR:$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools:$NAVSIM_DEVKIT_ROOT
  3. copy ``diffusiondrive_ttnn_inproc_agent.yaml`` into the navsim agent config dir,
  4. run ``run_pdm_score.py`` with ``agent=diffusiondrive_ttnn_inproc_agent``.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
from typing import Any, Dict, List

import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import SensorConfig


def _trace_enabled() -> bool:
    return os.environ.get("DD_TRACE", "1").strip().lower() not in ("0", "false", "no", "off")


def _is_pool_worker_thread() -> bool:
    """True on a thread-pool worker, False on the main thread.

    Auto-selects the device path from the worker type with **no env var**: nuplan's
    Sequential executor runs ``run_pdm_score`` (hence ``initialize()``) inline on the
    MAIN thread, while ``single_machine_thread_pool`` runs it on ThreadPoolExecutor
    worker threads.  Main → native (device on this thread); worker → shared singleton
    funnel.
    """
    return threading.current_thread() is not threading.main_thread()


def _open_build_trace(checkpoint_path: str, anchors_path: str, device_id: int):
    """Open the device, build the full TTNN stack, and (optionally) capture the trace.

    Runs on whichever thread calls it — the device context becomes affine to that
    thread, so every later device op must run on it too.  Returns
    ``(device, model, traced)``.
    """
    import ttnn  # local import: only the worker process needs ttnn
    from models.demos.diffusion_drive.tt.config import ModelConfig
    from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

    # DD-1: ttnn.conv2d allocates from the L1_SMALL bank; default 0 → OOM.
    # Trace capture additionally needs a trace region (256 MB validated).
    open_kwargs = {"device_id": device_id, "l1_small_size": 32768}
    if _trace_enabled():
        open_kwargs["trace_region_size"] = 256 * 1024 * 1024
    device = ttnn.open_device(**open_kwargs)

    cfg = ModelConfig()
    cfg.plan_anchor_path = anchors_path
    # latent=False → use the real LiDAR BEV the feature builder produces.
    # Full on-device stack — identical chain to scripts/ttnn_pdm_server._build_model:
    # backbone (stems + BasicBlocks + FPN + GPT fusion, consolidated), perception
    # head, DDIM denoiser, agent head. Valid at the production resolution the feature
    # builder emits (camera 256×1024, LiDAR 256×256).
    model = TtnnDiffusionDriveModel.from_checkpoint(checkpoint_path, cfg, device, latent=False)
    (
        model.build_stage2(device)
        .build_stage3(device)
        .build_stage3_4(device)
        .build_stage3_5(device)
        .build_stage3_6(device)
        .build_stage3_7(device)
        .build_stage4(device)
    )

    traced = False
    if _trace_enabled():
        # Stage 7: capture the backbone-loop trace once for fast replay (PCC 1.0).
        # A capture failure falls back gracefully rather than aborting the eval.
        try:
            model.compile()
            traced = True
            logging.getLogger(__name__).info(
                "DiffusionDrive backbone-loop trace captured — using execute_compiled() path"
            )
        except Exception as exc:  # never let trace capture break the eval
            logging.getLogger(__name__).warning(
                "DiffusionDrive trace capture failed (%s); falling back to eager forward", exc
            )
    return device, model, traced


def _run_forward(model, feats, traced):
    with torch.no_grad():
        out = model.execute_compiled(feats) if traced else model(feats)
    return {"trajectory": out["trajectory"].float().cpu()}


# ---------------------------------------------------------------------------
# Process-wide device-owner singleton (worker=single_machine_thread_pool)
# ---------------------------------------------------------------------------
# navsim's run_pdm_score instantiates a FRESH agent + initialize() per worker
# chunk, so with worker=single_machine_thread_pool the N chunks would each open
# device 0 (conflict) and would touch the thread-affine context from N threads.
# This singleton owns the device + model on ONE thread; every per-chunk agent
# funnels its forwards to it.
_SINGLETON_LOCK = threading.Lock()
_SINGLETON = None  # type: _DeviceWorker | None


class _DeviceWorker:
    """One thread that owns the device + model and serves forwards off a queue."""

    def __init__(self, checkpoint_path: str, anchors_path: str, device_id: int) -> None:
        self._ckpt = checkpoint_path
        self._anch = anchors_path
        self._dev_id = device_id
        self._device = None
        self._model = None
        self._traced = False
        self._req_queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._setup_exc: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="dd-device", daemon=True)
        self._thread.start()
        self._ready.wait()  # block until setup finishes (or fails)
        if self._setup_exc is not None:
            raise self._setup_exc
        atexit.register(self._shutdown)

    def _loop(self) -> None:
        try:
            self._device, self._model, self._traced = _open_build_trace(self._ckpt, self._anch, self._dev_id)
        except BaseException as exc:  # surface setup failure to the constructor
            self._setup_exc = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._req_queue.get()
            if item is None:  # shutdown sentinel
                self._close()
                return
            feats, box, ev = item
            try:
                box["out"] = _run_forward(self._model, feats, self._traced)
            except BaseException as exc:  # propagate to the waiting worker thread
                box["exc"] = exc
            finally:
                ev.set()

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        box: Dict[str, Any] = {}
        ev = threading.Event()
        self._req_queue.put((feats, box, ev))
        ev.wait()
        if "exc" in box:
            raise box["exc"]
        return box["out"]

    def _close(self) -> None:
        if self._device is not None:
            try:
                import ttnn

                ttnn.close_device(self._device)
            except Exception:
                pass
            self._device = None

    def _shutdown(self) -> None:
        # Close the device on its owner thread (the context is thread-affine).
        self._req_queue.put(None)
        self._thread.join(timeout=60)
        # tt_metal registers a libc on_exit handler that destroys the MetalContext/
        # Cluster on the MAIN thread at process exit; because the cluster is affine to
        # the (now-closed) device thread, that teardown SIGABRTs in umd::Cluster::
        # close_device. The PDM results CSV is already written by this point, so bypass
        # the crashing C++ static teardown with an immediate clean exit. Only reached on
        # the thread-pool funnel path (this atexit is registered by _DeviceWorker).
        os._exit(0)


def _get_singleton(checkpoint_path: str, anchors_path: str, device_id: int) -> "_DeviceWorker":
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = _DeviceWorker(checkpoint_path, anchors_path, device_id)
        return _SINGLETON


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
        # Sequential path: per-instance device/model, built in initialize().
        self._device = None
        self._model = None
        self._traced = False
        self._lock = threading.Lock()
        # Thread-pool path: shared process-wide device-owner singleton.
        self._worker = None  # type: _DeviceWorker | None
        self._initialized = False

    # -- required hooks --------------------------------------------------
    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """Build (or attach to) the on-device TTNN stack; idempotent + thread-safe.

        Auto-selected by worker type (no env var).  Main thread (``sequential``):
        opens the device + builds + captures the trace on this thread (one chunk →
        one agent, so it's safe).  Pool worker (``single_machine_thread_pool``):
        attaches to the process-wide :class:`_DeviceWorker` singleton (created by
        the first agent across all worker chunks), which owns the device on one
        thread — see the module docstring for why per-chunk agents require this.
        """
        with self._lock:
            if self._initialized:
                return
            if _is_pool_worker_thread():
                # thread-pool: per-chunk agents on worker threads → shared singleton.
                self._worker = _get_singleton(self._checkpoint_path, self._anchors_path, self._device_id)
            else:
                # sequential: one chunk on the main thread → device on this thread.
                self._device, self._model, self._traced = _open_build_trace(
                    self._checkpoint_path, self._anchors_path, self._device_id
                )
                atexit.register(self._close)
            self._initialized = True

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[3])

    def get_feature_builders(self) -> List[TransfuserFeatureBuilder]:
        return [TransfuserFeatureBuilder(config=self._config)]

    # -- inference -------------------------------------------------------
    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """Run the model and return ``{"trajectory": (1,8,3) cpu}``.

        ``AbstractAgent.compute_trajectory`` passes batched features and then does
        ``predictions["trajectory"].squeeze(0).numpy()``, so the return must be a CPU
        float tensor.  DDIM noise is left **unseeded** (fresh per forward) to match the
        upstream stochastic eval — see scripts/navsim_bridge/README.md.
        """
        if not self._initialized:  # defensive: harness should call initialize() first
            self.initialize()
        feats = {
            "camera_feature": features["camera_feature"].float(),
            "lidar_feature": features["lidar_feature"].float(),
            "status_feature": features["status_feature"].float(),
        }
        if self._worker is not None:
            # Funnel to the device-owner thread; feature-building already ran on this
            # (worker) thread, overlapping the serialized device forward.
            return self._worker.forward(feats)
        with self._lock:  # sequential path: one device → one forward at a time
            return _run_forward(self._model, feats, self._traced)

    # -- cleanup ---------------------------------------------------------
    def _close(self) -> None:
        if self._device is not None:
            try:
                import ttnn

                ttnn.close_device(self._device)
            except Exception:
                pass
            self._device = None
