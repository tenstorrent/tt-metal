# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from models.common.sampling.generator import SamplingGenerator, SeedManager
from models.common.warmup.warmup_utils import WarmupForwardMixin
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


def test_decode_warmup_does_not_reset_absent_request_history():
    calls = []
    fake = SimpleNamespace(
        _create_sampling_params=lambda *args, **kwargs: [object()],
        _create_decode_warmup_inputs=lambda *args: (
            torch.zeros((1, 1)),
            torch.zeros((1,)),
            torch.zeros((1, 1)),
        ),
        decode_forward=lambda **kwargs: calls.append(kwargs),
    )

    WarmupForwardMixin.warmup_model_decode(
        fake,
        kv_cache=object(),
        enable_trace=True,
        max_batch_size=1,
        num_blocks=1,
        can_sample_on_device=True,
    )

    assert len(calls) == 1
    assert calls[0]["reload_sampling_params"] is True
    assert calls[0]["reset_sampling_state"] is False


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


def test_qwen_vl_generator_forwards_slot_remap_to_shared_sampling_owner():
    from models.demos.qwen3_vl.tt.generator import Generator as Qwen3Generator
    from models.demos.qwen25_vl.tt.generator import Generator as Qwen25Generator

    for generator_cls in (Qwen25Generator, Qwen3Generator):
        calls = []
        generator = SimpleNamespace(
            _ttt_generator=SimpleNamespace(decode_forward=lambda **kwargs: calls.append(kwargs) or "output")
        )

        result = generator_cls.decode_forward(
            generator,
            tokens="tokens",
            start_pos="positions",
            slot_remap=[3, 1, 2, 3],
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )

        assert result == "output"
        assert calls[0]["slot_remap"] == [3, 1, 2, 3]


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


def test_shared_generator_rebases_and_pads_lane_sampling_remaps():
    from models.tt_transformers.tt.generator import Generator

    calls = [[], []]
    models = []
    for lane in range(2):
        seed_manager = SimpleNamespace(
            max_batch_size=4,
            apply_slot_remap=lambda remap, lane=lane: calls[lane].append(torch.as_tensor(remap).tolist()),
        )
        models.append(SimpleNamespace(sampling=SimpleNamespace(seed_manager=seed_manager)))
    fake = SimpleNamespace(data_parallel=2, model=models)

    Generator._apply_sampling_slot_remap(fake, torch.tensor([1, 0, 3, 2]))

    assert calls == [[[1, 0, 2, 3]], [[1, 0, 2, 3]]]


def test_shared_generator_rejects_cross_lane_sampling_remap(expect_error):
    from models.tt_transformers.tt.generator import Generator

    seed_manager = SimpleNamespace(max_batch_size=2, apply_slot_remap=lambda remap: None)
    fake = SimpleNamespace(
        data_parallel=2,
        model=[
            SimpleNamespace(sampling=SimpleNamespace(seed_manager=seed_manager)),
            SimpleNamespace(sampling=SimpleNamespace(seed_manager=seed_manager)),
        ],
    )

    with expect_error(ValueError, "outside its global range"):
        Generator._apply_sampling_slot_remap(fake, torch.tensor([0, 2, 2, 3]))


def test_sglang_bridge_explicitly_requests_host_authoritative_decode(monkeypatch):
    from models.tt_transformers.tt.generator import Generator
    from models.tt_transformers.tt.generator_sglang import LlamaForCausalLM

    calls = []
    monkeypatch.setattr(
        Generator,
        "decode_forward",
        lambda self, *args, **kwargs: calls.append(kwargs) or "output",
    )

    result = LlamaForCausalLM.decode_forward(object(), tokens="tokens", start_pos="positions")

    assert result == "output"
    assert calls == [
        {
            "tokens": "tokens",
            "start_pos": "positions",
            "reload_inputs": True,
            "reload_page_table": False,
            "reload_sampling_params": False,
            "reset_sampling_state": False,
        }
    ]


def test_lfm_demo_supplies_every_decode_update_command():
    source_path = Path("models/demos/multimodal/lfm25_vl/demo/vision_demo.py")
    tree = ast.parse(source_path.read_text())
    required = {
        "reload_inputs",
        "reload_page_table",
        "reload_sampling_params",
        "reset_sampling_state",
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "decode_forward"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "generator"
    ]

    assert len(calls) == 2
    for call in calls:
        assert required <= {keyword.arg for keyword in call.keywords}


def test_galaxy_reset_only_formats_seed_slots(monkeypatch):
    from models.demos.llama3_70b_galaxy.tt import generator as galaxy_generator

    formatted = SimpleNamespace(seed=[17, 17, 17, 17])
    monkeypatch.setattr(
        galaxy_generator,
        "format_sampling_params",
        lambda params, max_batch_size: formatted,
    )
    monkeypatch.setattr(
        galaxy_generator,
        "_fill_inactive_params_from_active",
        lambda params, active_slots, max_batch_size: params,
    )
    reset_calls = []
    seed_manager = SimpleNamespace(
        max_batch_size=4,
        deactivate_slots_except=lambda slots: None,
        reset_seed_from_slots=lambda seeds, slots: reset_calls.append((seeds, slots)),
        align_seed_counters_to_positions=lambda *args: None,
        reset_seed_from_slots_if_needed=lambda *args: None,
        get_new_values=lambda slots: None,
    )
    sampling = SimpleNamespace(
        seed_manager=seed_manager,
        reset_sampling_params=lambda params: (_ for _ in ()).throw(
            AssertionError("reset-only must not upload sampling parameters")
        ),
        reset_prompt_tokens=lambda tokens: None,
        reset_output_state=lambda tokens: None,
        sample=lambda **kwargs: "tokens",
    )
    fake = SimpleNamespace(
        trace_inputs_decode={True: None},
        model=SimpleNamespace(sampling=sampling),
        model_args=SimpleNamespace(max_batch_size=4),
        _apply_sampling_slot_remap=lambda remap: None,
    )

    result = galaxy_generator.Generator.sample_decode_on_device(
        fake,
        tt_logits="logits",
        sampling_params=SimpleNamespace(seed=[17]),
        start_pos=torch.tensor([0, -1, -1, -1]),
        reload_sampling_params=False,
        reset_sampling_state=True,
    )

    assert result == "tokens"
    assert reset_calls == [([17, 17, 17, 17], [0])]


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
    manager.seed_salts = [0]
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
