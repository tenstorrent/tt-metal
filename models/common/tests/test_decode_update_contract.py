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

from models.common.decode_contract import per_layer_page_tables_need_upload, require_full_input_reload
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
    # Three families used to carry three different alignment policies. Requiring the
    # single implementation is what keeps them from drifting apart again.
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


def _adapter_decode_forwards():
    """Every ``decode_forward`` defined in a vLLM adapter module, read not imported.

    Most of these modules import vLLM at module scope, which a tt-metal deployment
    does not have, so importing them would skip this check exactly where adapters are
    written. Parsing the source keeps it running everywhere, and the property being
    checked is a property of the source.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    for path in sorted((repo_root / "models").rglob("*_vllm.py")):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "decode_forward":
                    yield f"{path.relative_to(repo_root)}::{node.name}", item


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


def test_every_adapter_decode_forward_accepts_the_four_commands():
    # An adapter that advertises the contract but drops a command fails at the first
    # decode with a bare TypeError, and one that defaults a command the wrong way is
    # worse: it silently ignores the scheduler. Sweep the sources instead of trusting
    # a manual audit.
    checked = 0
    for label, func in _adapter_decode_forwards():
        accepts_kwargs = func.args.kwarg is not None
        defaults = _argument_defaults(func)
        assert "reset_batch" not in defaults, f"{label} still takes reset_batch"
        for command, safe in _SAFE_COMMAND_DEFAULTS.items():
            assert command in defaults or accepts_kwargs, f"{label} drops {command}"
            default = defaults.get(command, ...)
            if isinstance(default, ast.Constant):
                assert default.value is safe, f"{label} defaults {command} unsafely"
        checked += 1
    # A floor, not a census: it only has to catch the sweep finding nothing because the
    # glob or the layout moved.
    assert checked >= 8, f"only found {checked} adapter decode_forward definitions"


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


def test_the_legacy_reset_batch_signal_is_rejected(expect_error):
    from models.tt_transformers.tt.generator import Generator

    fake = SimpleNamespace(mode=Mode.DECODE, model=[SimpleNamespace(switch_mode=lambda mode: None)], data_parallel=1)

    # The commands carry defaults, so a pre-contract vLLM would otherwise be accepted
    # with its layout changes silently reinterpreted as the default full reload.
    with expect_error(ValueError, "predates the contract"):
        Generator.decode_forward(
            fake,
            torch.zeros((1, 1), dtype=torch.int64),
            torch.zeros((1,), dtype=torch.int64),
            reset_batch=True,
        )


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
    assert (
        per_layer_page_tables_need_upload(kwargs.get("reload_inputs", True), kwargs.get("reload_page_table", False))
        is expected
    )


def test_partial_reload_is_refused_without_decode_token_feedback(expect_error):
    from models.tt_transformers.tt.generator import Generator

    model = SimpleNamespace(switch_mode=lambda mode: None, _tt_supports_decode_token_feedback=False)
    fake = SimpleNamespace(mode=Mode.DECODE, model=[model], data_parallel=1)

    # Nothing would write the sampled token into the traced input, so a steady-state step
    # would replay the previous token instead of the new one.
    with expect_error(ValueError, "requires reload_inputs=True"):
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
