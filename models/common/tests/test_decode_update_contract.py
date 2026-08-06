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

import inspect
import random
from types import SimpleNamespace

import pytest
import torch

from models.common.decode_contract import require_full_input_reload
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


def _contract_v1_adapters():
    """Every built-in vLLM adapter class that advertises the explicit contract."""
    pytest.importorskip("vllm")

    from models.demos.blackhole.qwen36.tt import qwen36_vllm
    from models.demos.deepseek_v3.tt import generator_vllm as deepseek_vllm
    from models.demos.gemma4.tt import generator_vllm as gemma4_vllm
    from models.demos.llama3_70b_galaxy.tt import generator_vllm as galaxy_vllm
    from models.demos.qwen3_vl.tt import generator_vllm as qwen3_vl_vllm
    from models.demos.qwen25_vl.tt import generator_vllm as qwen25_vl_vllm
    from models.demos.t3000.llama2_70b.tt import generator_vllm as t3000_vllm
    from models.tt_transformers.tt import generator_vllm as tt_transformers_vllm

    modules = [
        tt_transformers_vllm,
        galaxy_vllm,
        gemma4_vllm,
        qwen36_vllm,
        qwen25_vl_vllm,
        qwen3_vl_vllm,
        deepseek_vllm,
        t3000_vllm,
    ]
    adapters = {}
    for module in modules:
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or not hasattr(obj, "initialize_vllm_model"):
                continue
            if int(getattr(obj, "decode_input_update_contract", 0)) < 1:
                continue
            adapters[f"{module.__name__}.{name}"] = obj
    return adapters


def test_every_marked_adapter_accepts_all_four_commands():
    # The marker is what makes vLLM stop sending reset_batch, so an adapter that
    # advertises version 1 without accepting the commands fails at the first decode
    # with a bare TypeError. Enumerate the registry instead of trusting a manual audit.
    adapters = _contract_v1_adapters()
    assert adapters, "no contract-v1 adapters found — did the marker move?"

    for label, adapter in adapters.items():
        params = inspect.signature(adapter.decode_forward).parameters
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        for command in (
            "reload_inputs",
            "reload_page_table",
            "reload_sampling_params",
            "reset_sampling_state",
        ):
            named = params.get(command)
            assert named is not None or accepts_kwargs, f"{label}.decode_forward drops {command}"
            if named is not None:
                assert named.default is inspect.Parameter.empty, f"{label}.decode_forward defaults {command}"
        assert "reset_batch" not in params, f"{label}.decode_forward still takes reset_batch"


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
