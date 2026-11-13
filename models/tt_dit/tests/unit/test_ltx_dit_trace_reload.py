# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU: a dynamic-load DiT reload that moves the weights releases the traces that baked the old addresses.

On a 2x4 (dynamic_load) with TT_DIT_CACHE_DIR unset, each request rebuilt the DiT from the state dict at
other addresses and replayed the previous request's denoise traces over them: every generation after the
capturing one was noise. The production methods are run here with fake modules and tracers (no ttnn).
"""
import ast
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pipelines/ltx/pipeline_ltx.py"
DISTILLED = ROOT / "pipelines/ltx/pipeline_ltx_distilled.py"


def _load(path, names, namespace, class_name=None):
    body = ast.parse(path.read_text()).body
    if class_name is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), "missing production method"
    for n in nodes:
        n.decorator_list = []
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(nodes, type_ignores=[]))
    exec(compile(source, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class Buffer:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class Param:
    def __init__(self, address):
        self._data = None if address is None else Buffer(address)


class Mod:
    def __init__(self, params=(), children=()):
        self.params = [Param(a) for a in params]
        self.children = list(children)

    def named_children(self):
        return [(str(i), c) for i, c in enumerate(self.children)]

    def named_parameters(self):
        return [(str(i), p) for i, p in enumerate(self.params)]

    def move(self, offset):
        for p in self.params:
            if p._data is not None:
                p._data.address += offset
        for c in self.children:
            c.move(offset)


class Tracer:
    def __init__(self, captured=True):
        self.trace_captured = captured
        self.released = 0

    def release_trace(self):
        self.released += 1
        self.trace_captured = False


def _namespace():
    keyed = weakref.WeakKeyDictionary()
    loads = []
    ns = {
        "weakref": weakref,
        "logger": Mock(),
        "LTX_FUSE_GATE_ON_DEVICE": False,
        "LTXTransformerModel": SimpleNamespace(inner_step=SimpleNamespace(_tracers_keyed=keyed)),
        "cache_module": SimpleNamespace(load_model=lambda model, **kw: loads.append(model)),
    }
    _load(PIPE, ["_parameter_addresses"], ns)
    base = _load(
        PIPE,
        ["_prepare_transformer", "_release_dit_traces_if_weights_moved", "_tracers_replayed_beside_dit"],
        ns,
        "LTXPipeline",
    )
    return ns, base, keyed, loads


def _pipeline(base, model, *, dynamic_load=True, traced=True, beside=()):
    class Pipe:
        _prepare_transformer = base["_prepare_transformer"]
        _release_dit_traces_if_weights_moved = base["_release_dit_traces_if_weights_moved"]

        def _tracers_replayed_beside_dit(self):
            return list(beside)

    p = Pipe()
    p.dynamic_load, p._traced = dynamic_load, traced
    p.checkpoint_name = "ckpt"
    p.transformer_states = [
        SimpleNamespace(model=model, lora_specs=[], cache_name="dit", state_dict_provider=None, **{"_": None})
    ]
    p.transformer_states[0].model.weight_cache_subfolder = lambda: "sub"
    p.parallel_config, p.mesh_device, p.is_fsdp = None, SimpleNamespace(shape=(2, 4)), False
    p._trace_state = {}
    p._dit_weight_addresses = weakref.WeakKeyDictionary()
    return p


def _dit():
    return Mod(params=[0x1000, None], children=[Mod(params=[0x2000, 0x3000]), Mod(children=[Mod(params=[0x4000])])])


def test_parameter_addresses_walks_children_and_skips_unloaded():
    ns, *_ = _namespace()
    assert ns["_parameter_addresses"](_dit()) == (0x2000, 0x3000, 0x4000, 0x1000)


def test_reload_at_same_addresses_keeps_traces():
    _, base, keyed, loads = _namespace()
    dit = _dit()
    pipe = _pipeline(base, dit)
    pipe._prepare_transformer(0)  # first load: nothing captured yet
    s1, s2 = Tracer(), Tracer()
    keyed[dit] = {"s1": s1, "s2": s2}  # gen 0 captures
    pipe._prepare_transformer(0)  # gen 1 reload, same placement
    assert len(loads) == 2 and pipe.transformer is dit
    assert (s1.released, s2.released) == (0, 0) and s1.trace_captured


def test_reload_that_moves_weights_releases_dit_tail_and_transition_traces():
    _, base, keyed, _ = _namespace()
    dit = _dit()
    t12 = Tracer()
    pipe = _pipeline(base, dit, beside=[t12])
    tail = Mock()
    pipe._trace_state = {"s1": SimpleNamespace(_euler_tail=tail), "s2": SimpleNamespace(_euler_tail=None)}
    pipe._prepare_transformer(0)
    s1, s2 = Tracer(), Tracer()
    keyed[dit] = {"s1": s1, "s2": s2}
    dit.move(0x100000)  # the reload lands elsewhere
    pipe._prepare_transformer(0)
    assert (s1.released, s2.released, t12.released) == (1, 1, 1)
    assert not s1.trace_captured and not s2.trace_captured
    tail.release.assert_called_once()
    assert pipe._trace_state["s1"]._euler_tail is None
    # The new placement is the record now: a further reload there keeps the recaptured traces.
    s1.trace_captured = s2.trace_captured = True
    pipe._prepare_transformer(0)
    assert (s1.released, s2.released) == (1, 1)


def test_moved_weights_with_nothing_captured_release_nothing():
    _, base, keyed, _ = _namespace()
    dit = _dit()
    t12 = Tracer()
    pipe = _pipeline(base, dit, beside=[t12])
    pipe._prepare_transformer(0)
    keyed[dit] = {"s1": Tracer(captured=False)}
    dit.move(0x100000)
    pipe._prepare_transformer(0)
    assert t12.released == 0


def test_traces_captured_before_any_recorded_load_are_released():
    _, base, keyed, _ = _namespace()
    dit = _dit()
    pipe = _pipeline(base, dit)
    s1 = Tracer()
    keyed[dit] = {"s1": s1}
    pipe._prepare_transformer(0)
    assert s1.released == 1


def test_static_or_untraced_load_never_releases():
    for dynamic_load, traced in ((False, True), (True, False)):
        _, base, keyed, _ = _namespace()
        dit = _dit()
        pipe = _pipeline(base, dit, dynamic_load=dynamic_load, traced=traced)
        s1 = Tracer()
        keyed[dit] = {"s1": s1}
        pipe._prepare_transformer(0)
        dit.move(0x100000)
        pipe._prepare_transformer(0)
        assert s1.released == 0 and len(pipe._dit_weight_addresses) == 0


def test_distilled_pipeline_releases_its_stage_transition_beside_the_dit():
    ns = {}
    fn = _load(DISTILLED, ["_tracers_replayed_beside_dit"], ns, "LTXDistilledPipeline")["_tracers_replayed_beside_dit"]
    t12 = Tracer()

    class Pipe:
        _stage_transition = SimpleNamespace(_tracers_keyed=weakref.WeakKeyDictionary())
        _tracers_replayed_beside_dit = fn

    p = Pipe()
    assert p._tracers_replayed_beside_dit() == []
    Pipe._stage_transition._tracers_keyed[p] = {"t12": t12}
    assert p._tracers_replayed_beside_dit() == [t12]
