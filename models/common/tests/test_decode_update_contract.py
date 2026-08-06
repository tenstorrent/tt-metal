# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host coverage for the decode input-update contract.

vLLM owns decode reload policy and sends four commands plus ``slot_remap`` on every
decode; generators execute them exactly. The contract is described in
``models/common/decode_contract.py`` (command semantics),
``models/common/sampling/README.md`` (sampling-side ordering) and the paired vLLM
document ``plugins/vllm-tt-plugin/docs/decode-reload-contract.md``.

These tests pin, without a device:

* which trace inputs each input command stages,
* that the commands are required and have no legacy ``reset_batch`` alias,
* that every adapter advertising version 1 accepts all four,
* that adapters rejecting a subset do so loudly and name the offending command,
* that the slot remap reaches exactly one sampling owner, and only after a
  successful decode when the sampler is dormant (the remap is not idempotent, so a
  retry must not apply it twice).
"""

import ast
import inspect
import pathlib
import random
from types import SimpleNamespace

import pytest
import torch

from models.common.decode_contract import (
    DecodeInputStaging,
    decode_input_staging,
    per_layer_page_tables_need_upload,
    rank_local_slot_remap,
    require_full_input_reload,
)
from models.common.sampling.generator import SamplingGenerator, SeedManager
from models.tt_transformers.tt.common import Mode


def _fake_sampling_generator():
    calls = []
    fake = SimpleNamespace(
        tt_sampling=SimpleNamespace(max_batch_size=4),
        reset_sampling_params=lambda params: calls.append(("params", params)),
        reset_prompt_tokens=lambda tokens: calls.append(("prompt", tokens)),
        reset_output_state=lambda tokens: calls.append(("output", tokens)),
    )
    return fake, calls


def test_sampling_state_can_reset_without_reloading_params():
    fake, calls = _fake_sampling_generator()

    SamplingGenerator.apply_decode_state(
        fake,
        [object()],
        reload_sampling_params=False,
        reset_sampling_state=True,
        prompt_tokens="prompt",
        output_tokens="output",
    )

    assert calls == [("prompt", "prompt"), ("output", "output")]


def test_a_state_reset_without_prompt_history_keeps_the_prompt_mask():
    """Penalties cannot rebuild the mask without the real tokens, so it is left alone.

    Warmup and some demos request a state reset with no prompt history. Clearing the
    output counters is still unconditional; only the prompt half is caller-owned.
    """
    fake, calls = _fake_sampling_generator()

    SamplingGenerator.apply_decode_state(
        fake,
        [object()],
        reload_sampling_params=False,
        reset_sampling_state=True,
        prompt_tokens=None,
        output_tokens=None,
    )

    assert calls == [("output", None)]


def test_no_sampling_updates_is_a_true_noop():
    fake, calls = _fake_sampling_generator()

    SamplingGenerator.apply_decode_state(
        fake,
        [object()],
        reload_sampling_params=False,
        reset_sampling_state=False,
    )

    assert calls == []


def test_sampling_update_commands_are_required_and_have_no_legacy_alias():
    params = inspect.signature(SamplingGenerator.apply_decode_state).parameters

    assert params["reload_sampling_params"].default is inspect.Parameter.empty
    assert params["reset_sampling_state"].default is inspect.Parameter.empty
    assert "reset_batch" not in params


class _RecordingSeedManager:
    def __init__(self, max_batch_size=4):
        self.max_batch_size = max_batch_size
        self.events = []

    def apply_slot_remap(self, remap):
        self.events.append(("remap", list(remap)))

    def reset_seed_from_slots(self, seeds, user_ids):
        self.events.append(("reset", seeds, user_ids))

    def reset_seed_from_slots_if_needed(self, seeds, user_ids):
        self.events.append(("conditional-reset", seeds, user_ids))

    def align_seed_counters_to_positions(self, seeds, user_ids, positions):
        self.events.append(("align", positions))

    def get_new_values(self, user_slots):
        self.events.append(("advance", user_slots))


def _recording_sampling_generator():
    fake = SimpleNamespace(
        tt_sampling=SimpleNamespace(max_batch_size=4),
        seed_manager=_RecordingSeedManager(),
        apply_decode_state=lambda chunks, **kwargs: None,
    )
    return fake, fake.seed_manager.events


@pytest.mark.parametrize(
    "reload_sampling_params, reset_sampling_state, expected",
    [
        # A state reset rebuilds the stream, and only a reloading step ever carries
        # one, so this is the one case where host positions may be consulted.
        (True, True, [("reset", [7], [0]), ("align", [3]), ("advance", [0])]),
        (False, True, [("reset", [7], [0]), ("align", [3]), ("advance", [0])]),
        # A params-only update keeps the resident counter: aligning here would tie the
        # RNG stream to a host position that lags the device under async decode.
        (True, False, [("conditional-reset", [7], [0]), ("advance", [0])]),
        (False, False, [("advance", [0])]),
    ],
)
def test_seed_alignment_happens_only_on_a_state_reset(reload_sampling_params, reset_sampling_state, expected):
    fake, events = _recording_sampling_generator()

    SamplingGenerator.apply_decode_update(
        fake,
        None,
        reload_sampling_params=reload_sampling_params,
        reset_sampling_state=reset_sampling_state,
        seeds=[7],
        active_slots=[0],
        positions=[3],
    )

    assert events == expected


def test_seed_state_is_untouched_when_no_slot_samples():
    fake, events = _recording_sampling_generator()

    SamplingGenerator.apply_decode_update(
        fake,
        None,
        reload_sampling_params=True,
        reset_sampling_state=True,
        seeds=[7],
        active_slots=[],
        positions=[3],
    )

    assert events == [("advance", [])]


def test_seed_advance_can_be_deferred_to_the_sampling_call():
    fake, events = _recording_sampling_generator()

    SamplingGenerator.apply_decode_update(
        fake,
        None,
        reload_sampling_params=False,
        reset_sampling_state=True,
        seeds=[7],
        active_slots=[0],
        positions=[3],
        advance_seeds=False,
    )

    assert events == [("reset", [7], [0]), ("align", [3])]


@pytest.mark.parametrize(
    "module_path, class_name, method",
    [
        ("models.tt_transformers.tt.generator", "Generator", "sample_decode_on_device"),
        ("models.demos.llama3_70b_galaxy.tt.generator", "Generator", "sample_decode_on_device"),
        ("models.demos.deepseek_v3.tt.generator", "DeepseekGenerator", "_apply_sampling_state"),
    ],
)
def test_every_generator_routes_its_seed_state_through_the_shared_protocol(module_path, class_name, method):
    # Seed alignment must have exactly one implementation: a per-family copy changes
    # reproducibility silently, with no crash and no wrong shape to notice.
    import importlib

    owner = getattr(importlib.import_module(module_path), class_name)
    source = inspect.getsource(getattr(owner, method))

    assert "apply_decode_update(" in source
    for direct_call in ("align_seed_counters_to_positions", "reset_seed_from_slots"):
        assert direct_call not in source, f"{class_name}.{method} still drives {direct_call} itself"


def _staging_events(monkeypatch, module):
    """Record which decode trace inputs the commands actually copy to device."""
    events = []
    monkeypatch.setattr(
        module,
        "copy_host_to_device",
        lambda host_tensors, device_tensors=None, **kwargs: events.append(("all", tuple(host_tensors))),
    )
    monkeypatch.setattr(
        module.ttnn,
        "copy_host_to_device_tensor",
        lambda host, device: events.append(("one", host, device)),
    )
    monkeypatch.setattr(module.ttnn, "execute_trace", lambda *args, **kwargs: None)
    return events


_HOST_DECODE_INPUTS = ("h_tokens", "h_pos", "h_rope", "h_page_table")
_DEVICE_DECODE_INPUTS = ["d_tokens", "d_pos", "d_rope", "d_page_table"]


@pytest.mark.parametrize(
    "reload_inputs, reload_page_table, expected",
    [
        (True, False, [("all", _HOST_DECODE_INPUTS)]),
        # reload_inputs subsumes reload_page_table: one full restage, not two copies.
        (True, True, [("all", _HOST_DECODE_INPUTS)]),
        (False, True, [("one", "h_page_table", "d_page_table")]),
        (False, False, []),
    ],
)
def test_shared_generator_stages_only_what_the_commands_request(
    monkeypatch, reload_inputs, reload_page_table, expected
):
    from models.tt_transformers.tt import generator as generator_module

    events = _staging_events(monkeypatch, generator_module)
    fake = SimpleNamespace(
        data_parallel=1,
        model=[SimpleNamespace(prepare_decode_inputs_host=lambda *args: _HOST_DECODE_INPUTS)],
        model_args=[SimpleNamespace(mesh_device="mesh")],
        trace_ids_decode={False: {0: "trace"}},
        trace_inputs_decode={False: [_DEVICE_DECODE_INPUTS]},
        trace_output_decode={False: "output"},
    )

    result = generator_module.Generator._decode_forward_trace_text(
        fake,
        ["tokens"],
        ["current_pos"],
        page_table=["page_table"],
        on_device_sampling=False,
        reload_inputs=reload_inputs,
        reload_page_table=reload_page_table,
    )

    assert events == expected
    assert result == "output"


@pytest.mark.parametrize(
    "reload_inputs, reload_page_table, expected",
    [
        (True, False, [("all", _HOST_DECODE_INPUTS)]),
        (True, True, [("all", _HOST_DECODE_INPUTS)]),
        (False, True, [("one", "h_page_table", "d_page_table")]),
        (False, False, []),
    ],
)
def test_galaxy_generator_stages_only_what_the_commands_request(
    monkeypatch, reload_inputs, reload_page_table, expected
):
    from models.demos.llama3_70b_galaxy.tt import generator as generator_module

    events = _staging_events(monkeypatch, generator_module)
    fake = SimpleNamespace(
        model=SimpleNamespace(
            prepare_decode_inputs_host=lambda *args: _HOST_DECODE_INPUTS,
            prepare_decode_shard_configs=lambda *args: None,
        ),
        mesh_device="mesh",
        _decode_forward_trace_text=lambda *args, **kwargs: "output",
        trace_ids_decode={False: "trace"},
        trace_inputs_decode={False: _DEVICE_DECODE_INPUTS},
        trace_output_decode={False: "output"},
    )

    result = generator_module.Generator._decode_easy_trace_text(
        fake,
        torch.zeros((1, 1), dtype=torch.int64),
        torch.zeros((1,), dtype=torch.int64),
        page_table="page_table",
        on_device_logits=False,
        reload_inputs=reload_inputs,
        reload_page_table=reload_page_table,
    )

    assert events == expected
    assert result == "output"


# A command may carry a default, but only the host-authoritative value: a caller that
# omits one must get a full reload and no sampling update, never the reverse.
_SAFE_COMMAND_DEFAULTS = {
    "reload_inputs": True,
    "reload_page_table": False,
    "reload_sampling_params": False,
    "reset_sampling_state": False,
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _adapter_decode_forwards():
    """Every ``decode_forward`` defined in a serving-adapter module, read not imported.

    Most of these modules import vLLM at module scope, which a tt-metal deployment
    does not have, so importing them would skip this check exactly where adapters are
    written. Parsing the source keeps it running everywhere, and the property being
    checked is a property of the source.

    Covers the sglang bridge too: its wrappers call the same strict
    ``Generator.decode_forward`` and are equally able to drop a command.
    """
    repo_root = _repo_root()
    paths = sorted(set((repo_root / "models").rglob("*_vllm.py")) | set((repo_root / "models").rglob("*_sglang.py")))
    for path in paths:
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "decode_forward":
                    if _is_abstract_stub(item):
                        continue
                    yield f"{path.relative_to(repo_root)}::{node.name}", item


def _is_abstract_stub(func: ast.FunctionDef) -> bool:
    """Whether ``func`` only tells a subclass to override it.

    A ``NotImplementedError`` placeholder declares an interface rather than executing
    commands, so requiring it to accept them would say nothing about conformance.
    """
    body = [node for node in func.body if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Constant)]
    return len(body) == 1 and isinstance(body[0], ast.Raise)


def _delegates_kwargs_to_decode_forward(func: ast.FunctionDef) -> bool:
    """Whether ``func`` forwards its ``**kwargs`` into another ``decode_forward``.

    A bare ``**kwargs`` proves nothing on its own: an adapter can accept the commands
    and drop them on the floor. What makes absorbing them safe is passing them on to a
    base that declares them, so look for that call rather than for the parameter.
    """
    kwarg = func.args.kwarg
    if kwarg is None:
        return False
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not (isinstance(target, ast.Attribute) and target.attr == "decode_forward"):
            continue
        for keyword in node.keywords:
            if keyword.arg is None and isinstance(keyword.value, ast.Name) and keyword.value.id == kwarg.arg:
                return True
    return False


def _argument_defaults(func: ast.FunctionDef) -> dict:
    """Map every named argument of ``func`` to its default, or to Ellipsis if required."""
    defaults = {}
    positional = func.args.posonlyargs + func.args.args
    padding = [...] * (len(positional) - len(func.args.defaults))
    for arg, default in zip(positional, padding + list(func.args.defaults)):
        defaults[arg.arg] = default
    for arg, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
        defaults[arg.arg] = ... if default is None else default
    return defaults


# Every adapter definition the sweep is expected to find. An exact set, not a floor:
# a new adapter has to be added here deliberately, and a definition that disappears
# because a glob or a layout moved cannot pass unnoticed.
_EXPECTED_ADAPTER_DECODE_FORWARDS = {
    "models/demos/blackhole/qwen36/tt/qwen36_vllm.py::Qwen36ForCausalLM",
    "models/demos/deepseek_v3/tt/generator_vllm.py::DeepseekV3ForCausalLM",
    "models/demos/gemma4/tt/generator_vllm.py::Gemma4ForCausalLM",
    "models/demos/llama3_70b_galaxy/tt/generator_vllm.py::LlamaForCausalLM",
    "models/demos/llama3_70b_galaxy/tt/generator_vllm.py::QwenForCausalLM",
    "models/demos/qwen25_vl/tt/generator_vllm.py::Qwen2_5_VLForConditionalGeneration",
    "models/demos/qwen3_vl/tt/generator_vllm.py::Qwen3VLForConditionalGeneration",
    "models/demos/t3000/llama2_70b/tt/generator_vllm.py::TtLlamaForCausalLM",
    "models/tt_transformers/tt/generator_sglang.py::LlamaForCausalLM",
    "models/tt_transformers/tt/generator_sglang.py::QwenForCausalLM",
    "models/tt_transformers/tt/generator_sglang.py::MistralForCausalLM",
    "models/tt_transformers/tt/generator_sglang.py::GptOssForCausalLM",
    "models/tt_transformers/tt/generator_vllm.py::Gemma3ForConditionalGeneration",
    "models/tt_transformers/tt/generator_vllm.py::GptOssForCausalLM",
    # ``HybridAttentionForCausalLM.decode_forward`` is absent on purpose: it only raises
    # NotImplementedError to force its subclasses to route per-layer page tables.
    "models/tt_transformers/tt/generator_vllm.py::LlamaForCausalLM",
    "models/tt_transformers/tt/generator_vllm.py::Mistral3ForConditionalGeneration",
    "models/tt_transformers/tt/generator_vllm.py::MistralForCausalLM",
    "models/tt_transformers/tt/generator_vllm.py::MllamaForConditionalGeneration",
    "models/tt_transformers/tt/generator_vllm.py::QwenForCausalLM",
}


def test_the_adapter_sweep_covers_every_definition():
    assert {label for label, _ in _adapter_decode_forwards()} == _EXPECTED_ADAPTER_DECODE_FORWARDS


def test_every_adapter_decode_forward_accepts_the_four_commands():
    # An adapter that advertises the contract but drops a command fails at the first
    # decode with a bare TypeError, and one that defaults a command the wrong way is
    # worse: it silently ignores the scheduler. Sweep the sources instead of trusting
    # a manual audit.
    for label, func in _adapter_decode_forwards():
        defaults = _argument_defaults(func)
        delegates = _delegates_kwargs_to_decode_forward(func)
        assert "reset_batch" not in defaults, f"{label} still takes reset_batch"
        for command, safe in _SAFE_COMMAND_DEFAULTS.items():
            # Declared here, or handed to a base that declares it. Merely having
            # ``**kwargs`` is not enough: the commands have to reach an implementation.
            assert (
                command in defaults or delegates
            ), f"{label} neither declares {command} nor forwards **kwargs to a base decode_forward"
            default = defaults.get(command, ...)
            if isinstance(default, ast.Constant):
                assert default.value is safe, f"{label} defaults {command} unsafely"


def test_version_zero_executor_adapters_stay_unmarked():
    """The executor-backed adapters are holdouts, and nothing may mark them converted.

    ``models/common/models/*/generator.py`` forward ``**kwargs`` straight into
    ``EagerLLMExecutor``/``TracedLLMExecutor.decode_forward``, which accept none of the
    four commands. They survive only because they never advertise the contract, so the
    day one of them does, its first vLLM decode is a bare ``TypeError``. That is the
    pairing this checks, since the sweep above cannot see these modules.
    """
    repo_root = _repo_root()
    family_root = repo_root / "models" / "common" / "models"
    # The shared base is what makes this family vLLM-reachable; the leaves subclass it.
    assert "initialize_vllm_model" in (family_root / "generator.py").read_text()
    generators = sorted(family_root.rglob("generator.py"))
    assert len(generators) >= 8, f"expected the executor-backed generator family, found {len(generators)}"
    for path in generators:
        assert "decode_input_update_contract" not in path.read_text(), (
            f"{path.relative_to(repo_root)} advertises the contract, but its executor "
            "accepts none of the four commands; convert the executor first"
        )

    executor = ast.parse((repo_root / "models" / "common" / "models" / "executor.py").read_text())
    for node in ast.walk(executor):
        if not (isinstance(node, ast.ClassDef) and node.name.endswith("LLMExecutor")):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "decode_forward":
                declared = set(_argument_defaults(item))
                assert not declared & set(_SAFE_COMMAND_DEFAULTS), (
                    f"{node.name}.decode_forward now takes contract commands; the "
                    "holdout note and this test need updating together"
                )


def test_models_without_token_feedback_have_no_async_decode_adapter():
    """A model that cannot feed its sampled token back must not be async-decode capable.

    ``decode_forward`` rejects ``reload_inputs=False`` for these models, so the only
    thing keeping vLLM from planning it is the adapter leaving the capability off. That
    pairing spans two files per model and had nothing enforcing it.
    """
    repo_root = _repo_root()
    checked = 0
    for model_path in sorted((repo_root / "models").rglob("model.py")):
        source = model_path.read_text()
        if "_tt_supports_decode_token_feedback = False" not in source:
            continue
        checked += 1
        adapters = [p for p in model_path.parent.glob("*_vllm.py") if not p.name.startswith("test_")]
        assert adapters, f"{model_path.relative_to(repo_root)} has no adapter beside it"
        for adapter in adapters:
            assert '"supports_async_decode": True' not in adapter.read_text(), (
                f"{adapter.relative_to(repo_root)} advertises supports_async_decode while "
                f"{model_path.name} has no decode token feedback"
            )
    assert checked >= 4, f"expected the known no-feedback models, found {checked}"


def test_the_base_generators_advertise_the_contract():
    # The marker lives on the generators that execute the commands, so every adapter
    # inherits it. These modules hold no vLLM import.
    from models.demos.llama3_70b_galaxy.tt.generator import Generator as GalaxyGenerator
    from models.tt_transformers.tt.generator import Generator as SharedGenerator

    for generator in (SharedGenerator, GalaxyGenerator):
        assert int(generator.decode_input_update_contract) >= 1
        params = inspect.signature(generator.decode_forward).parameters
        for command, safe in _SAFE_COMMAND_DEFAULTS.items():
            assert params[command].kind is inspect.Parameter.KEYWORD_ONLY
            assert params[command].default is safe


@pytest.mark.parametrize(
    "module_path", ["models.tt_transformers.tt.generator", "models.demos.llama3_70b_galaxy.tt.generator"]
)
def test_the_legacy_reset_batch_signal_is_rejected(module_path, expect_error):
    import importlib

    generator_cls = importlib.import_module(module_path).Generator
    # Both base generators, not just the shared one: the same mistake against Galaxy
    # used to surface as a bare TypeError from an unexpected keyword.
    fake = SimpleNamespace(
        mode=Mode.DECODE,
        model=[SimpleNamespace(switch_mode=lambda mode: None)],
        data_parallel=1,
        _disable_decode_tracing=False,
    )

    # The commands carry defaults, so a pre-contract vLLM would otherwise be accepted
    # with its layout changes silently reinterpreted as the default full reload.
    with expect_error(ValueError, "predates it"):
        generator_cls.decode_forward(
            fake,
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1,), dtype=torch.int64),
            reset_batch=True,
        )


def test_the_two_input_commands_collapse_to_one_decision(expect_error):
    """Reading them as independent switches is the misread that loses a page table.

    An adapter whose page-table copy hangs off ``reload_page_table`` alone never
    refreshes it on vLLM's every-transition shape, and the device then addresses the
    previous batch's KV blocks with nothing to show for it.
    """
    assert decode_input_staging(True, False) is DecodeInputStaging.ALL
    assert decode_input_staging(False, True) is DecodeInputStaging.PAGE_TABLE_ONLY
    assert decode_input_staging(False, False) is DecodeInputStaging.NONE
    with expect_error(ValueError, "meaningless with"):
        decode_input_staging(True, True)


@pytest.mark.parametrize(
    "commands, offender",
    [
        ({"reload_inputs": False}, "reload_inputs=False"),
        ({"reload_page_table": True}, "reload_page_table=True"),
        ({"reload_sampling_params": True}, "reload_sampling_params=True"),
        ({"reset_sampling_state": True}, "reset_sampling_state=True"),
    ],
)
def test_full_reload_only_adapters_name_the_rejected_command(commands, offender, expect_error):
    accepted = {
        "reload_inputs": True,
        "reload_page_table": False,
        "reload_sampling_params": False,
        "reset_sampling_state": False,
    }
    require_full_input_reload("Adapter", **accepted)

    with expect_error(ValueError, offender):
        require_full_input_reload("Adapter", **{**accepted, **commands})


def test_qwen_vl_slot_remap_moves_persistent_rope_deltas():
    from models.demos.qwen3_vl.tt.generator import Generator as Qwen3Generator
    from models.demos.qwen25_vl.tt.generator import Generator as Qwen25Generator

    for generator_cls in (Qwen25Generator, Qwen3Generator):
        generator = SimpleNamespace(
            model=SimpleNamespace(
                rope_setup=SimpleNamespace(
                    batch_size=4,
                    rope_deltas=torch.tensor([10, 20, 30, 40]),
                )
            )
        )

        generator_cls.remap_rope_deltas(generator, [3, 1, 2, 3])

        assert generator.model.rope_setup.rope_deltas.tolist() == [40, 20, 30, 40]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"reload_inputs": True, "reload_page_table": False}, True),
        ({"reload_inputs": False, "reload_page_table": True}, True),
        ({"reload_inputs": False, "reload_page_table": False}, False),
        # A caller that omits the commands gets the conservative refresh here and, for a
        # partial reload it cannot execute, the error from decode_forward.
        ({}, True),
    ],
)
def test_hybrid_per_layer_page_tables_follow_the_input_commands(kwargs, expected):
    assert per_layer_page_tables_need_upload(kwargs) is expected


def test_every_per_layer_page_table_upload_is_gated_by_the_commands():
    """The behavioural half: no adapter may upload per-layer tables unconditionally.

    On a ``reload_inputs=False, reload_page_table=False`` step the device holds the
    tables it needs, and re-uploading them mid-trace is exactly the model-local refresh
    the contract removes. The adapters that do this import vLLM at module scope, which
    this test image has no copy of, so read the sources: what matters is that the guard
    is present at every call site, which is a property of the source.
    """
    repo_root = _repo_root()
    guarded = 0
    for path in sorted((repo_root / "models").rglob("*_vllm.py")):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "decode_forward"):
                continue
            body = ast.unparse(node)
            if "update_persistent_per_layer_page_tables" not in body:
                continue
            assert "per_layer_page_tables_need_upload" in body, (
                f"{path.relative_to(repo_root)}::{node.name} uploads per-layer page tables "
                "without asking whether the commands request it"
            )
            guarded += 1
    assert guarded == 3, f"expected the three hybrid adapters, found {guarded}"


def test_partial_reload_is_refused_without_decode_token_feedback(expect_error):
    from models.tt_transformers.tt.generator import Generator

    model = SimpleNamespace(switch_mode=lambda mode: None, _tt_supports_decode_token_feedback=False)
    fake = SimpleNamespace(mode=Mode.DECODE, model=[model], data_parallel=1)

    # Nothing would write the sampled token into the traced input, so a steady-state step
    # would replay the previous token instead of the new one. Match on the model-naming
    # half of the message: "requires reload_inputs=True" also appears in the untraced
    # raise below, so a looser match would pass on the wrong path.
    with expect_error(ValueError, "no on-device decode token feedback"):
        Generator.decode_forward(
            fake,
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1,), dtype=torch.int64),
            sampling_params=object(),
            reload_inputs=False,
            reload_page_table=True,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )


def test_untraced_decode_refuses_a_partial_reload(expect_error):
    from models.tt_transformers.tt.generator import Generator

    fake = SimpleNamespace(mode=Mode.DECODE, model=[SimpleNamespace(switch_mode=lambda mode: None)], data_parallel=1)

    # Without a trace there are no persistent device inputs to preserve, so the whole
    # forward is rebuilt from host and a partial reload cannot mean anything.
    with expect_error(ValueError, "Non-traced decode rebuilds"):
        Generator.decode_forward(
            fake,
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1,), dtype=torch.int64),
            enable_trace=False,
            reload_inputs=False,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )


def test_shared_generator_routes_slot_remap_to_exactly_one_sampling_owner():
    from models.tt_transformers.tt.generator import Generator

    def run(*, sampling_params=None, defer_device_sampling=False, fail_readback=False):
        events = []
        fake = SimpleNamespace(
            mode=Mode.DECODE,
            model=[SimpleNamespace(switch_mode=lambda mode: None)],
            data_parallel=1,
            _decode_forward_trace_text=lambda **kwargs: events.append("decode") or "logits",
            sample_decode_on_device=lambda output, **kwargs: events.append(("sample", kwargs["slot_remap"]))
            or "tokens",
            read_decode_output=lambda output: events.append("read") or output,
            process_decode_output_host=lambda output, **kwargs: (_ for _ in ()).throw(RuntimeError("readback"))
            if fail_readback
            else events.append("process") or output,
            _apply_sampling_slot_remap=lambda remap: events.append(("host-remap", remap)),
        )

        try:
            result = Generator.decode_forward(
                fake,
                torch.zeros((1, 1), dtype=torch.int64),
                torch.zeros((1,), dtype=torch.int64),
                sampling_params=sampling_params,
                slot_remap=[0],
                defer_device_sampling=defer_device_sampling,
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=False,
                reset_sampling_state=False,
            )
        except RuntimeError:
            result = None
        return events, result

    host_events, _ = run()
    device_events, _ = run(sampling_params=object())
    deferred_events, _ = run(defer_device_sampling=True)
    failed_host_events, _ = run(fail_readback=True)

    assert host_events == ["decode", "read", "process", ("host-remap", [0])]
    assert device_events == ["decode", ("sample", [0]), "read", "process"]
    assert deferred_events == ["decode"]
    assert failed_host_events == ["decode", "read"]


def test_galaxy_generator_routes_slot_remap_to_exactly_one_sampling_owner():
    from models.demos.llama3_70b_galaxy.tt.generator import Generator

    def run(*, sampling_params=None, defer_device_sampling=False):
        events = []
        fake = SimpleNamespace(
            model=SimpleNamespace(is_decode_setup=True),
            _decode_easy_trace_text=lambda **kwargs: events.append("decode") or ("logits", None),
            sample_decode_on_device=lambda output, **kwargs: events.append(("sample", kwargs["slot_remap"]))
            or ("tokens", "logprobs"),
            read_decode_output=lambda output, **kwargs: events.append("read") or output,
            process_decode_output_host=lambda output, **kwargs: events.append("process") or output,
            _apply_sampling_slot_remap=lambda remap: events.append(("host-remap", remap)),
        )

        result = Generator.decode_forward(
            fake,
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1,), dtype=torch.int64),
            kv_cache=[object()],
            sampling_params=sampling_params,
            slot_remap=[0],
            defer_device_sampling=defer_device_sampling,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        return events, result

    host_events, _ = run()
    device_events, _ = run(sampling_params=object())
    deferred_events, _ = run(defer_device_sampling=True)

    assert host_events == ["decode", "read", "process", ("host-remap", [0])]
    assert device_events == ["decode", ("sample", [0]), "read", "process"]
    assert deferred_events == ["decode"]


def test_unseeded_decode_reset_loads_fresh_device_seed(monkeypatch):
    manager = SeedManager.__new__(SeedManager)
    manager.max_batch_size = 1
    manager.seeds = [None]
    manager.seed_counters = [4]
    manager.rngs = [random.Random(1)]
    manager._seed_active = False
    manager._reseted = False
    manager._needs_skip = False
    manager._active_request_seed = False
    manager._seed_mapper = None
    seed_buffer = object()
    manager.tt_sampling = SimpleNamespace(seeds_tt_tensor=seed_buffer)
    before = manager.rngs[0].getstate()
    host_seed_tensor = object()
    uploads = []

    monkeypatch.setattr(manager, "_next_unseeded_device_seed", lambda: 123)
    monkeypatch.setattr(
        "models.common.sampling.generator.ttnn.from_torch",
        lambda *args, **kwargs: host_seed_tensor,
    )
    monkeypatch.setattr(
        "models.common.sampling.generator.ttnn.copy_host_to_device_tensor",
        lambda host, device: uploads.append((host, device)),
    )

    # The conditional path would see None == None and do nothing. A decode
    # state reset must be unconditional so the following get_new_values()
    # enters its init state and uploads a fresh device seed.
    manager.reset_seed_from_slots([None], [0])
    manager.get_new_values([0])

    assert manager.seed_counters == [0]
    assert manager.rngs[0].getstate() != before
    assert uploads == [(host_seed_tensor, seed_buffer)]
    assert manager._needs_skip
    assert not manager._reseted


# ---------------------------------------------------------------------------
# Slot-remap index spaces
# ---------------------------------------------------------------------------


def test_rank_local_slot_remap_rebases_each_rank_onto_its_own_slots():
    # vLLM offsets each rank's local mapping by rank*slots_per_rank; per-rank state
    # indexes its own slots, so rank 1's values must come back down into [0, 4).
    global_remap = [1, 0, 2, 3, 5, 4, 6, 7]

    assert rank_local_slot_remap(global_remap, rank=0, slots_per_rank=4, data_parallel=2) == [1, 0, 2, 3]
    assert rank_local_slot_remap(global_remap, rank=1, slots_per_rank=4, data_parallel=2) == [1, 0, 2, 3]


def test_rank_local_slot_remap_takes_the_stride_from_the_mapping():
    """The stride is the scheduler's per-rank batch, not the sampler's slot count.

    Serving fewer requests than the sampler has slots is the normal case
    (``--max_num_seqs 1`` against a 32-slot device sampler), so a rule that required the
    two to match would reject it outright. Deriving the stride is also what keeps rank 0
    from slicing up its neighbour's entries: with a wider stride assumed, every one of
    them lands in range and rank 1 is left an empty slice.
    """
    # world=2 at max_num_seqs=16 against a 32-slot sampler: 32 entries, stride 16.
    two_ranks_of_sixteen = list(range(16)) + [16 + i for i in range(16)]

    assert rank_local_slot_remap(two_ranks_of_sixteen, rank=0, slots_per_rank=32, data_parallel=2) == list(range(16))
    assert rank_local_slot_remap(two_ranks_of_sixteen, rank=1, slots_per_rank=32, data_parallel=2) == list(range(16))

    # A single request against the same sampler: one entry, not 32.
    assert rank_local_slot_remap([0], rank=0, slots_per_rank=32) == [0]


def test_rank_local_slot_remap_rejects_a_batch_wider_than_the_sampler(expect_error):
    # The other direction is a genuine mismatch: those slots have nowhere to land.
    with expect_error(ValueError, "wider than the sampler"):
        rank_local_slot_remap(list(range(8)), rank=0, slots_per_rank=4)

    with expect_error(ValueError, "not divisible"):
        rank_local_slot_remap(list(range(3)), rank=0, slots_per_rank=4, data_parallel=2)


def test_rank_local_slot_remap_rejects_a_cross_rank_move(expect_error):
    # Moving a request between ranks would index another rank's state.
    with expect_error(ValueError, "across DP rank 1"):
        rank_local_slot_remap([0, 1, 0, 3], rank=1, slots_per_rank=2, data_parallel=2)


def test_shared_generator_rebases_the_remap_for_every_dp_rank():
    from models.tt_transformers.tt.generator import Generator

    managers = [_RecordingSeedManager(max_batch_size=2) for _ in range(2)]
    fake = SimpleNamespace(
        data_parallel=2,
        model=[SimpleNamespace(sampling=SimpleNamespace(seed_manager=m)) for m in managers],
    )

    # Rank 1's half is offset by 2 in the global namespace.
    Generator._apply_sampling_slot_remap(fake, [1, 0, 3, 2])

    assert managers[0].events == [("remap", [1, 0])]
    assert managers[1].events == [("remap", [1, 0])]


def test_galaxy_generator_validates_the_remap_it_slices(expect_error):
    from models.demos.llama3_70b_galaxy.tt.generator import Generator as GalaxyGenerator

    manager = _RecordingSeedManager(max_batch_size=2)
    fake = SimpleNamespace(model=SimpleNamespace(sampling=SimpleNamespace(seed_manager=manager)))

    GalaxyGenerator._apply_sampling_slot_remap(fake, [1, 0])
    assert manager.events == [("remap", [1, 0])]

    # One model, so a mapping wider than its own slots is a mismatch, not a slice.
    with expect_error(ValueError, "wider than the sampler"):
        GalaxyGenerator._apply_sampling_slot_remap(fake, [1, 0, 3, 2])


# ---------------------------------------------------------------------------
# sglang bridge
# ---------------------------------------------------------------------------


def _sglang_commands(*args, **kwargs):
    """Load the sglang command table without importing the bridge module.

    ``generator_sglang.py`` imports ttnn and the model, so read the two functions out
    of the source and bind them to the real ``Generator.decode_forward`` signature.
    """
    import importlib

    source = pathlib.Path(_repo_root() / "models/tt_transformers/tt/generator_sglang.py").read_text()
    tree = ast.parse(source)
    wanted = {"_decode_forward_sampling_params", "sglang_decode_forward_commands"}
    body = [n for n in tree.body if getattr(n, "name", None) in wanted]
    namespace = {
        "inspect": inspect,
        "Generator": importlib.import_module("models.tt_transformers.tt.generator").Generator,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "<sglang>", "exec"), namespace)
    return namespace["sglang_decode_forward_commands"](args, kwargs)


def test_sglang_commands_are_host_authoritative_on_every_step():
    commands = _sglang_commands(torch.zeros((1, 1)), torch.zeros((1,)))

    assert commands == {
        "reload_inputs": True,
        "reload_page_table": False,
        "reload_sampling_params": False,
        "reset_sampling_state": False,
    }


def test_sglang_finds_sampling_params_passed_positionally():
    """The bridge forwards ``*args`` verbatim, so a positional call must be seen.

    Missing it reports "no sampling params", skips the parameter upload while device
    sampling is active, and yields wrong token distributions rather than an error.
    """
    by_keyword = _sglang_commands(torch.zeros((1, 1)), sampling_params=object())
    # tokens, start_pos, page_table, kv_cache, enable_trace, read_from_device, sampling_params
    positional = _sglang_commands(torch.zeros((1, 1)), torch.zeros((1,)), None, None, True, True, object())

    assert by_keyword["reload_sampling_params"] is True
    assert positional["reload_sampling_params"] is True


def test_mllama_rejects_partial_reloads_and_stays_synchronous():
    """A shipped multimodal adapter: both halves of its contract stance pinned.

    It rebuilds all host inputs per decode, so it rejects the partial combinations, and
    it must keep ``supports_async_decode`` off, which is what stops vLLM from planning
    one. Read from source: the module imports vLLM.
    """
    source = pathlib.Path(_repo_root() / "models/tt_transformers/tt/generator_vllm.py").read_text()
    tree = ast.parse(source)
    mllama = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MllamaForConditionalGeneration"
    )
    body = ast.unparse(mllama)

    assert '"supports_async_decode": False' in body
    decode = next(n for n in mllama.body if isinstance(n, ast.FunctionDef) and n.name == "decode_forward")
    assert "require_full_input_reload" in ast.unparse(decode)
    # The helper names the offending command; check it is handed the adapter's own name.
    assert "'Mllama'" in ast.unparse(decode) or '"Mllama"' in ast.unparse(decode)
