# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for pipeline residency (no device needed)."""

import pytest

from models.tt_dit.pipelines import residency as residency_module
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.pipelines.residency import Offload, Residency, Shard
from models.tt_dit.utils.tracing import Tracer


class _StubModule:
    def __init__(self, name: str, log: list, *, loaded: bool = False) -> None:
        self.name = name
        self._log = log
        self._loaded = loaded
        self._sharded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def is_transiently_sharded(self) -> bool:
        return self._sharded

    def deallocate_weights(self) -> None:
        self._log.append(("deallocate", self.name))
        self._loaded = False

    def shard(self, axis: int) -> None:
        self._log.append(("shard", self.name, axis))
        self._loaded = False
        self._sharded = True

    def gather(self, ccl_manager) -> None:
        self._log.append(("gather", self.name, ccl_manager))
        self._loaded = True
        self._sharded = False


class _StubPipeline(PipelineAPIMixin):
    def __init__(self, log: list) -> None:
        self.log = log

    def release_traces(self) -> None:
        self.log.append(("release",))


@pytest.fixture
def log(monkeypatch):
    entries = []

    def load_model(module, **spec):
        entries.append(("load_model", module.name, spec))
        module._loaded = True

    monkeypatch.setattr(residency_module.cache, "load_model", load_model)
    return entries


def test_mixin_residency_is_cached(log):
    pipeline = _StubPipeline(log)
    assert isinstance(pipeline.residency, Residency)
    assert pipeline.residency is pipeline.residency


def test_load_evicts_in_order_and_runs_on_evict_after_each(log):
    pipeline = _StubPipeline(log)
    a, b, c = (_StubModule(n, log) for n in "abc")
    first, second = (lambda: log.append(("cb", 1))), (lambda: log.append(("cb", 2)))
    pipeline.residency.register(a, {"subfolder": "a"}, evicts=[b, c])
    pipeline.residency.register(b, {"subfolder": "b"})
    pipeline.residency.register(c, {"subfolder": "c"}, on_evict=[first, second])
    b._loaded = c._loaded = True

    pipeline.residency.load(a)

    assert log == [
        ("deallocate", "b"),
        ("release",),
        ("deallocate", "c"),
        ("cb", 1),
        ("cb", 2),
        ("load_model", "a", {"subfolder": "a"}),
    ]


def test_evicted_module_is_not_evicted_again(log):
    pipeline = _StubPipeline(log)
    a, b = _StubModule("a", log), _StubModule("b", log)
    pipeline.residency.register(a, {}, evicts=[b])
    pipeline.residency.register(b, {})

    pipeline.residency.load(a)

    assert log == [("load_model", "a", {})]


def test_no_load_releases_traces(log):
    pipeline = _StubPipeline(log)
    unlisted, a, b = (_StubModule(n, log) for n in ("unlisted", "a", "b"))
    pipeline.residency.register(unlisted, {})
    pipeline.residency.register(a, {}, evicts=[b])
    pipeline.residency.register(b, {}, evicts=[a])

    pipeline.residency.load(unlisted)
    pipeline.residency.load(unlisted)
    pipeline.residency.load(b)

    assert ("release",) not in log
    assert log == [("load_model", "unlisted", {}), ("load_model", "b", {})]


def test_shard_mode_shards_then_gathers(log):
    pipeline = _StubPipeline(log)
    a = _StubModule("a", log)
    decoder = _StubModule("decoder", log, loaded=True)
    ccl = object()
    pipeline.residency.register(a, {}, evicts=[decoder])
    pipeline.residency.register(decoder, {}, evicts=[a], mode=Shard(1, ccl), on_evict=())

    pipeline.residency.load(a)
    pipeline.residency.load(decoder)

    assert log == [
        ("shard", "decoder", 1),
        ("load_model", "a", {}),
        ("deallocate", "a"),
        ("release",),
        ("gather", "decoder", ccl),
    ]


def test_shard_mode_loads_when_not_sharded(log):
    pipeline = _StubPipeline(log)
    decoder = _StubModule("decoder", log)
    pipeline.residency.register(decoder, {"subfolder": "vae"}, mode=Shard(0, object()))

    pipeline.residency.load(decoder)

    assert log == [("load_model", "decoder", {"subfolder": "vae"})]


def test_load_of_resident_module_only_evicts(log):
    pipeline = _StubPipeline(log)
    a = _StubModule("a", log, loaded=True)
    b = _StubModule("b", log, loaded=True)
    pipeline.residency.register(a, {}, evicts=[b], mode=Offload())
    pipeline.residency.register(b, {}, on_evict=())

    pipeline.residency.load(a)

    assert log == [("deallocate", "b")]


def test_default_release_traces_releases_every_tracer(monkeypatch):
    calls = []
    monkeypatch.setattr(Tracer, "release_all", staticmethod(lambda: calls.append("all")))
    PipelineAPIMixin().release_traces()
    assert calls == ["all"]
