# SPDX-License-Identifier: Apache-2.0
"""Device-free close-before-mesh contracts for the TT worker wrapper."""

from __future__ import annotations

from laguna_vllm_ext.lifecycle import ADAPTER_MARKER, WORKER_PATCH_MARKER, _patch_worker_lifecycle


def test_worker_shutdown_and_destructor_close_laguna_before_mesh():
    events = []

    class Model:
        _closed = False

        def close(self):
            if not self._closed:
                events.append("adapter-close")
                self._closed = True

    setattr(Model, ADAPTER_MARKER, True)

    class Runner:
        model = Model()

    class Worker:
        model_runner = Runner()

        def shutdown(self):
            events.append("worker-shutdown")

        def __del__(self):
            events.append("mesh-close")

    assert _patch_worker_lifecycle(Worker)
    assert not _patch_worker_lifecycle(Worker)
    assert Worker.__dict__[WORKER_PATCH_MARKER]

    worker = Worker()
    worker.shutdown()
    worker.__del__()

    assert events == ["adapter-close", "worker-shutdown", "mesh-close"]


def test_worker_lifecycle_wrapper_is_inert_for_other_models():
    events = []

    class Model:
        def close(self):
            events.append("wrong-model-close")

    class Runner:
        model = Model()

    class Worker:
        model_runner = Runner()

        def shutdown(self):
            events.append("worker-shutdown")

    assert _patch_worker_lifecycle(Worker)
    Worker().shutdown()

    assert events == ["worker-shutdown"]
