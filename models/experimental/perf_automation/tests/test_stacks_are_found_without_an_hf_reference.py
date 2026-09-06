# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A model with no HF lineage still gets its stacks found, sized and bounded.

THE REFERENCE CENSUS ONLY SERVES MODELS THAT CARRY A REFERENCE. It compares the torch stacks a
pipeline holds for weight loading against the stacks the device side exposes, which is a real and
cheap witness -- and entirely absent for a model trained in-house, exported from a research repo, or
shipped as a bare checkpoint. Those models had exactly one statement of how many stacks they have:
the walk, which is the thing being checked.

THREE WITNESSES, NONE OF THEM NEEDING HF.

  fingerprint   Same weight shapes means same block, whatever the classes are called. This is what
                the two failed similarity attempts should have compared: attribute NAMES are shared
                by every torch module (which is why everything scored identical and unrelated
                submodules shadowed the real stacks), while tensor SHAPES are what a layer is. Goes
                into the walk, accept-only, so it can never displace a stack found the old way.

  caller        Boundaries come from identity, not from ops. Two towers of identical layers emit
                indistinguishable op streams -- 64 identical motifs with nothing marking where the
                first tower ended -- but they are different containers holding different objects.
                Recording which object emitted each op gives the boundary exactly, with no
                classification step at all: a container whose elements each ran and each emitted the
                same op subsequence IS a stack.

  checkpoint    Weight keys are paths and a repeated block prints its index into every one of them,
                so grouping keys gives one entry per stack and the depth of each -- from a file,
                needing no config, no transformers, no torch and no device. Read from the
                safetensors JSON header and the torch zip directory, keys only, never values.
"""

import ast
import json
import struct
import sys
import tempfile
import types
import zipfile
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _walk():
    """find_all_stacks lifted out of the probe, so the walk under test is the real one."""
    src = (_PA / "cc_optimize" / "_op_sig_probe.py").read_text()
    tree = ast.parse(src)
    want = {
        "_shape_sig", "_is_atomic", "_stack_members", "_stack_tier", "_shared_base", "_is_composite",
        "_is_block_stack", "_uniform_kind", "_child_nodes", "_node_sequence", "_largest_repeated_stack",
        "_enclosing_stack", "StackInfo", "_dominant_type", "_walk_for_stacks", "find_all_stacks",
    }  # fmt: skip
    keep = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)) or getattr(n, "name", None) in want
    ]
    mod = types.ModuleType("_walk_only")
    mod.__dict__["__file__"] = str(_PA / "cc_optimize" / "_op_sig_probe.py")
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<probe>", "exec"), mod.__dict__)
    return mod.find_all_stacks


class _T:
    """A tensor-like: anything with .shape, so no framework is required to exercise this."""

    def __init__(self, *dims):
        self.shape = dims
        self.dtype = "bf16"


def _layer(kind="enc", sink=None):
    """A block owning weights, wrapped in a class of its own -- the shape that defeats the walk.

    `sink` is the op stream: a real block dispatches ttnn ops, and a block that dispatches NOTHING is
    deliberately not a stack, so anything exercising the caller witness has to emit.
    """
    w = {"enc": (1280, 1280), "dec": (3072, 3072)}[kind]

    class _Sub:
        def __init__(self):
            self.weight = _T(*w)

        def __call__(self, x):
            return x

    def _init(s):
        s.attn, s.mlp = _Sub(), _Sub()

    def _call(s, x):
        if sink is not None:
            sink.append("matmul(%s,)" % (w,))
            sink.append("layernorm(%s,)" % (w,))
        return x

    cls = type("W%d" % _layer.n, (), {"__init__": _init, "__call__": _call})
    _layer.n += 1
    return cls()


_layer.n = 0


# ---------------------------------------------------------------- fingerprint (goes into the walk)


def test_same_weights_means_same_block_however_the_classes_are_named():
    from agent.block_fingerprint import fingerprint, same_kind, uniform_kind

    a, b = _layer("enc"), _layer("enc")
    assert type(a) is not type(b), "the test must exercise differently-typed wrappers"
    assert same_kind(a, b), "two wrappers around the same layer are not recognised"
    assert uniform_kind([_layer("enc") for _ in range(3)])
    assert fingerprint(a), "a block owning weights fingerprints as empty"


def test_the_pair_that_broke_both_earlier_attempts_separates_here():
    """An audio tower beside a language model: attribute names are identical (every torch module has
    _parameters, _modules, training), which is why comparing names grouped them and shadowed the real
    stacks. Their weights are not the same size, so shapes separate them immediately."""
    from agent.block_fingerprint import same_kind

    assert not same_kind(_layer("enc"), _layer("dec"))


def test_objects_owning_no_tensors_never_group():
    """Absence of evidence is not a match. Without this, a list of stubs -- or any two bare objects --
    would fingerprint identically as () and register as a stack."""
    from agent.block_fingerprint import same_kind, uniform_kind

    class Bare:
        def __init__(self):
            self.name = "x"

    assert not same_kind(Bare(), Bare())
    assert not uniform_kind([Bare(), Bare(), Bare()])


def test_a_back_reference_does_not_fingerprint_the_whole_model():
    """A block that reaches the model through a parent pointer would otherwise fingerprint AS the
    model, so every block would match every other one."""
    from agent.block_fingerprint import fingerprint

    class Model:
        def __init__(self):
            self.big = _T(50000, 4096)
            self.blocks = [_layer("enc") for _ in range(4)]
            for b in self.blocks:
                b.parent = self

    m = Model()
    assert (50000, 4096) not in [d for d, _ in fingerprint(m.blocks[0])], "the block absorbed the model"


def test_the_walk_now_sees_a_stack_of_differently_typed_wrappers():
    """THE ORIGINAL DEFECT, end to end, with no reference and no base class anywhere.

    This is the shape that produced one visible stack out of three on Voxtral: each layer wrapped in
    its own class, no shared base, held in a plain list and run in sequence.
    """
    find_all_stacks = _walk()

    class Pipe:
        def __init__(self):
            self.lm = [_layer("dec") for _ in range(6)]

    paths = {s.path for s in find_all_stacks(Pipe())}
    assert "lm" in paths, "a stack of differently-typed wrappers is still invisible"


def test_the_walk_sees_it_at_three_blocks_too():
    """The hybrid base-class rule needs 4+ elements, which is a real trap: these lists get short
    exactly when the profiler caps the depth. Same weights has no such bound."""
    find_all_stacks = _walk()

    class Pipe:
        def __init__(self):
            self.lm = [_layer("dec") for _ in range(3)]

    assert "lm" in {s.path for s in find_all_stacks(Pipe())}


def test_the_new_rule_can_only_add_never_displace():
    """THE CONTAINMENT THAT BOTH REGRESSIONS LACKED.

    Widening by attribute-name similarity took the walk from 5 stacks to 3 and lost an encoder,
    because the new rule matched things that then WON the selection. This clause is reachable only
    after the same-class test has already failed, so a list the old rules accepted is decided exactly
    as before.
    """
    src = (_PA / "cc_optimize" / "_op_sig_probe.py").read_text()
    i = src.index("def _is_block_stack(")
    body = src[i : i + 2500]
    assert body.index("if len(kinds) == 1:") < body.index("_uniform_kind(members)"), "the new rule can preempt"
    assert "accept" in body.lower()


def test_a_model_the_walk_already_handled_is_unchanged():
    """Regression guard for the shape that was lost twice: unrelated submodules under one parent
    must not group into a stack."""
    find_all_stacks = _walk()

    class Tower:
        def __init__(self, kind, n):
            self.layers = [_layer(kind) for _ in range(n)]

        def __call__(self, x):
            return x

    class Pipe:
        def __init__(self):
            self.audio = Tower("enc", 8)
            self.text = Tower("dec", 6)

    paths = {s.path for s in find_all_stacks(Pipe())}
    assert "audio.layers" in paths and "text.layers" in paths, "two towers did not both survive"
    assert not any(p in ("root", "") for p in paths), "the top-level submodules grouped into a stack"


# ------------------------------------------------------------------------------- caller identity


def test_two_identical_towers_are_two_stacks_not_one_long_one():
    """WHAT THE OP STREAM CANNOT DO. Both towers run the same layer, so their ops are identical and
    periodicity reads 8 repeats, not 2 stacks of 4. Identity has the boundary exactly."""
    from agent.caller_stacks import instrument, stacks_that_ran

    seq = []

    class Tower:
        def __init__(self):
            self.layers = [_layer("enc", seq) for _ in range(4)]

        def __call__(self, x):
            for lyr in self.layers:
                x = lyr(x)
            return x

    class Pipe:
        def __init__(self):
            self.a, self.b = Tower(), Tower()

        def run(self):
            return self.b(self.a(0))

    p = Pipe()
    assert instrument(p, seq.append) >= 8
    p.run()
    found = {s["path"]: s for s in stacks_that_ran(seq)}
    assert "a.layers" in found and "b.layers" in found, "the two towers were not separated: %s" % list(found)
    assert found["a.layers"]["depth"] == 4 and found["b.layers"]["depth"] == 4


def test_a_container_that_never_runs_is_not_a_stack():
    """Instrumentation is deliberately generous and the strictness lives here: anything that did not
    execute emits no marker, so a spare list of callables cannot invent a stack."""
    from agent.caller_stacks import instrument, stacks_that_ran

    seq = []

    class Pipe:
        def __init__(self):
            self.live = [_layer("enc", seq) for _ in range(3)]
            self.spare = [_layer("dec", seq) for _ in range(3)]

        def run(self):
            for lyr in self.live:
                lyr(0)

    p = Pipe()
    instrument(p, seq.append)
    p.run()
    paths = {s["path"] for s in stacks_that_ran(seq)}
    assert "live" in paths and "spare" not in paths


def test_the_ops_a_block_dispatched_are_attributed_to_it():
    from agent.caller_stacks import instrument, observed_stacks

    seq = []

    class Pipe:
        def __init__(self):
            self.layers = [_layer("dec", seq) for _ in range(3)]

        def run(self):
            for lyr in self.layers:
                lyr(0)

    p = Pipe()
    instrument(p, seq.append)
    p.run()
    st = {s["path"]: s for s in observed_stacks(seq)}["layers"]
    assert st["depth"] == 3 and st["uniform"], "identical blocks did not read as repeats"


def test_a_block_that_dispatches_nothing_is_not_a_stack():
    """The strictness that keeps this from inventing structure: a container can be instrumented, run,
    and still not be a stack if its elements did no device work. A wrapper chain or a list of
    bookkeeping objects passes every structural test and fails this one."""
    from agent.caller_stacks import instrument, observed_stacks, stacks_that_ran

    seq = []

    class Pipe:
        def __init__(self):
            self.layers = [_layer("dec") for _ in range(3)]  # no sink: dispatches nothing

        def run(self):
            for lyr in self.layers:
                lyr(0)

    p = Pipe()
    instrument(p, seq.append)
    p.run()
    assert {s["path"] for s in observed_stacks(seq)} == {"layers"}, "the container was not even observed"
    assert stacks_that_ran(seq) == [], "a stack that dispatched no ops was reported as real"


def test_instrumentation_hooks_the_class_because_python_ignores_an_instance_dunder():
    """obj() resolves __call__ on the TYPE. A wrapper installed on the instance is silently dead and
    every stack then reads as never having run -- which looks exactly like a model with no stacks."""
    src = (_PA / "agent" / "caller_stacks.py").read_text()
    assert "cls.__call__ = wrapped" in src, "the hook is not installed on the class"
    assert "TAG THE INSTANCE, WRAP THE CLASS" in src


def test_broad_instrumentation_is_flagged_as_probe_only():
    """Broad instrumentation is what exceeded tracy's 32K source-location limit and left a pytest
    process that never exited. Safe in the op-signature probe, which runs without tracy."""
    src = (_PA / "agent" / "caller_stacks.py").read_text()
    assert "must NOT leak into a profiling run" in src


# ----------------------------------------------------------------------------- checkpoint witness


def _safetensors(path, keys):
    head = {k: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]} for k in keys}
    blob = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 32)


def test_sections_and_depths_come_straight_off_the_weights():
    """No config.json, no transformers, no torch, no device -- and it answers both questions the HF
    config answered: how many stacks, and how deep each one is."""
    from agent.checkpoint_sections import declared_sections

    d = Path(tempfile.mkdtemp())
    keys = ["audio_tower.layers.%d.self_attn.q_proj.weight" % i for i in range(32)]
    keys += ["language_model.layers.%d.mlp.down_proj.weight" % i for i in range(30)]
    keys += ["lm_head.weight", "embed_tokens.weight"]
    _safetensors(d / "model.safetensors", keys)

    sec = declared_sections(d)
    assert sec == {"audio_tower.layers": 32, "language_model.layers": 30}, sec


def test_a_sharded_checkpoint_reports_the_models_depth_not_the_shards():
    """Depth is max index + 1, not the count, so a shard holding layers 16..31 still says 32."""
    from agent.checkpoint_sections import sections_from_keys

    keys = ["m.layers.%d.w" % i for i in (0, 16, 31)]
    assert sections_from_keys(keys) == {"m.layers": 32}


def test_an_index_that_is_not_a_stack_is_not_reported():
    """A lone `adapter.0` is a naming choice; a section needs at least two indices and must start
    at 0, or any numbered attribute in the graph becomes a phantom stack."""
    from agent.checkpoint_sections import sections_from_keys

    assert sections_from_keys(["adapter.0.weight", "head.weight"]) == {}
    assert sections_from_keys(["m.layers.3.w", "m.layers.4.w"]) == {}, "a run not starting at 0 is not a stack"


def test_keys_are_read_without_materialising_a_weight():
    """This runs during discovery. Loading a 3B model to count its layers would cost more than the
    check saves -- and unpickling a checkpoint to answer a structural question executes code from it."""
    src = (_PA / "agent" / "checkpoint_sections.py").read_text()
    assert "KEYS ONLY, NEVER VALUES" in src
    assert "torch.load" not in src, "a checkpoint is being loaded to read its keys"


def test_a_torch_zip_checkpoint_is_read_from_its_directory():
    from agent.checkpoint_sections import declared_sections

    d = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(d / "model.pt", "w") as zf:
        for i in range(4):
            zf.writestr("archive/blocks.%d.attn.weight" % i, b"")
        zf.writestr("archive/data/0", b"")
    assert declared_sections(d) == {"blocks": 4}


def test_no_checkpoint_means_no_evidence_not_no_stacks():
    """Same discipline as the reference census: silence rather than a manufactured verdict."""
    from agent.checkpoint_sections import declared_sections, section_count

    d = Path(tempfile.mkdtemp())
    assert declared_sections(d) == {}
    assert section_count(d) == 0
    assert declared_sections(d / "nope") == {}


# ------------------------------------------------------------------- weights are actually present


def test_weights_in_the_shared_cache_count_as_present():
    """A TT-METAL DEMO SHIPS CODE, NOT WEIGHTS.

    Measured 2026-08-13: ZERO weight files under the Voxtral demo directory, ~9 GB in
    ~/.cache/huggingface/hub/models--mistralai--Voxtral-Mini-3B-2507/. Looking only beside the model
    finds nothing for every model in this repo -- which is what the section reader did, silently
    reporting "no sections declared" and sending the survey off to build the model instead.
    """
    import json
    import os
    import tempfile

    from agent.checkpoint_sections import declared_sections, hf_cache_dir

    cache = Path(tempfile.mkdtemp())
    snap = cache / "hub" / "models--acme--tiny" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"encoder.layers.%d.w" % i: "s.safetensors" for i in range(8)}})
    )
    old = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = str(cache)
    try:
        assert hf_cache_dir("acme/tiny") == snap
        # the model directory itself has no weights at all
        model = Path(tempfile.mkdtemp())
        assert declared_sections(model) == {}
        assert declared_sections(model, "acme/tiny") == {"encoder.layers": 8}
    finally:
        os.environ.pop("HF_HOME", None)
        if old is not None:
            os.environ["HF_HOME"] = old


def test_a_dataset_repo_is_not_reported_as_missing_weights():
    """Pipelines load BOTH: weights from a model repo and sample inputs from a dataset repo. Voxtral
    pulls hf-internal-testing/dummy-audio-samples for its test audio, and looking only under
    models-- reported a present dataset as missing -- a readiness gate that cries wolf."""
    import os
    import tempfile

    from agent.checkpoint_sections import hf_cache_dir

    cache = Path(tempfile.mkdtemp())
    snap = cache / "hub" / "datasets--acme--samples" / "snapshots" / "d1"
    snap.mkdir(parents=True)
    (snap / "x.json").write_text("{}")
    old = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = str(cache)
    try:
        assert hf_cache_dir("acme/samples") == snap
    finally:
        os.environ.pop("HF_HOME", None)
        if old is not None:
            os.environ["HF_HOME"] = old


def test_the_gate_fires_before_the_device_is_opened():
    """Without weights the run reaches perf-test generation (~10 min of agent work) and a device open
    before the build tries to load them -- then dies far from the cause, or downloads gigabytes
    mid-profile and records the resulting timing as a measurement."""
    from agent.model_contract import CLAUSES, check

    assert any(name == "weights-present" for name, _fn in CLAUSES), "the gate is not registered"

    import tempfile

    d = Path(tempfile.mkdtemp())
    (d / "tt").mkdir()
    (d / "tt" / "pipeline.py").write_text(
        'PIPELINE_STAGES = ["decode"]\n'
        "def load():\n"
        '    return Model.from_pretrained("acme/definitely-not-cached")\n'
        "def build_pipeline(device, layers=None):\n    return None\n"
    )
    found = [f for f in check(d) if f.clause == "weights-present"]
    assert found, "a model naming an uncached repo passes the gate"
    assert found[0].severity == "error" and found[0].kind == "compatibility"
    assert "acme/definitely-not-cached" in found[0].detail


def test_a_path_configured_model_is_not_falsely_flagged():
    """Not every model names a hub repo: tt-metal also has models that read a weights directory from
    the environment. Checking the env vars the model ITSELF reads keeps this general with no naming
    heuristic -- if one currently points at weights, the model is provisioned."""
    import json
    import os
    import struct
    import tempfile

    from agent.model_contract import check

    weights = Path(tempfile.mkdtemp())
    head = {"layers.%d.w" % i: {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]} for i in range(4)}
    blob = json.dumps(head).encode()
    (weights / "w.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 8)

    d = Path(tempfile.mkdtemp())
    (d / "tt").mkdir()
    (d / "tt" / "pipeline.py").write_text(
        'PIPELINE_STAGES = ["decode"]\n'
        "import os\n"
        'WEIGHTS = os.environ.get("ACME_MODEL_DIR")\n'
        "def build_pipeline(device, layers=None):\n    return None\n"
    )
    os.environ["ACME_MODEL_DIR"] = str(weights)
    try:
        assert not [f for f in check(d) if f.clause == "weights-present"], "a provisioned model was flagged"
    finally:
        os.environ.pop("ACME_MODEL_DIR", None)
