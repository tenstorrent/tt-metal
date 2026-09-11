# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from models.common.sampling.generator import SamplingGenerator, SamplingParams, SeedManager
from models.common.warmup.warmup_utils import WarmupForwardMixin
from models.tt_transformers.tt.common import Mode


def _fake_sampling_generator():
    calls = []
    fake = SimpleNamespace(
        tt_sampling=SimpleNamespace(max_batch_size=32),
        seed_manager=SimpleNamespace(
            max_batch_size=32,
            apply_slot_remap=lambda remap: calls.append(("remap", list(remap))),
        ),
        _slot_state_requires_authoritative_reload=False,
        reset_sampling_params=lambda params: calls.append(("params", params)),
        reset_prompt_tokens=lambda tokens, slots=None: calls.append(("prompt", tokens, slots)),
        reset_output_state=lambda tokens, slots=None: calls.append(("output", tokens, slots)),
    )
    fake.validate_decode_state_commands = lambda **kwargs: SamplingGenerator.validate_decode_state_commands(
        fake, **kwargs
    )
    fake.commit_decode_state_commands = lambda **kwargs: SamplingGenerator.commit_decode_state_commands(fake, **kwargs)
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

    assert calls == [("prompt", "prompt", None), ("output", "output", None)]


def test_sampling_state_reset_can_preserve_unlisted_slots():
    fake, calls = _fake_sampling_generator()

    SamplingGenerator.apply_decode_state(
        fake,
        [object()],
        reload_sampling_params=False,
        reset_sampling_state=True,
        prompt_tokens="prompt",
        output_tokens=None,
        sampling_state_slots=[1, 3],
    )

    assert calls == [("prompt", "prompt", [1, 3]), ("output", None, [1, 3])]


def test_non_identity_slot_remap_requires_authoritative_device_state_reload(expect_error):
    fake, calls = _fake_sampling_generator()
    remap = [1, 0, *range(2, 32)]

    SamplingGenerator.apply_slot_remap(fake, remap)

    assert calls == [("remap", remap)]
    assert fake._slot_state_requires_authoritative_reload
    for reload_sampling_params, reset_sampling_state in ((False, False), (True, False), (False, True)):
        with expect_error(ValueError, "requires reload_sampling_params=True and reset_sampling_state=True"):
            SamplingGenerator.apply_decode_state(
                fake,
                [object()],
                reload_sampling_params=reload_sampling_params,
                reset_sampling_state=reset_sampling_state,
            )

    params = SamplingParams(temperature=1.0, top_k=1, top_p=1.0)
    SamplingGenerator.apply_decode_state(
        fake,
        [params],
        reload_sampling_params=True,
        reset_sampling_state=True,
        prompt_tokens="prompt",
        output_tokens="output",
    )

    assert calls[1][0] == "params"
    assert calls[2:] == [("prompt", "prompt", None), ("output", "output", None)]
    assert not fake._slot_state_requires_authoritative_reload


def test_partial_sampling_state_rebuild_does_not_clear_slot_remap_invalidation(expect_error):
    fake, _ = _fake_sampling_generator()
    remap = [1, 0, *range(2, 32)]
    params = SamplingParams(temperature=1.0, top_k=1, top_p=1.0)

    SamplingGenerator.apply_slot_remap(fake, remap)
    SamplingGenerator.apply_decode_state(
        fake,
        [params],
        reload_sampling_params=True,
        reset_sampling_state=True,
        prompt_tokens="prompt",
        output_tokens="output",
        sampling_state_slots=[1, 3],
    )

    assert fake._slot_state_requires_authoritative_reload
    with expect_error(ValueError, "requires reload_sampling_params=True and reset_sampling_state=True"):
        SamplingGenerator.apply_decode_state(
            fake,
            [params],
            reload_sampling_params=False,
            reset_sampling_state=False,
        )

    SamplingGenerator.apply_decode_state(
        fake,
        [params],
        reload_sampling_params=True,
        reset_sampling_state=True,
        prompt_tokens="prompt",
        output_tokens="output",
    )

    assert not fake._slot_state_requires_authoritative_reload


def test_identity_slot_remap_keeps_device_sampling_state_valid():
    fake, calls = _fake_sampling_generator()
    remap = list(range(32))

    SamplingGenerator.apply_slot_remap(fake, remap)

    assert calls == [("remap", remap)]
    assert not fake._slot_state_requires_authoritative_reload


def test_output_penalty_reset_masks_only_selected_slots(monkeypatch):
    from models.common.sampling import tt_penalties

    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self.deallocated = False

        def deallocate(self):
            self.deallocated = True

    allocated = []
    multiplies = []
    penalty_state = SimpleNamespace(
        _total_batch=4,
        _shard_dims_gathered=(0, None),
        _op_kwargs={},
        output_mask=FakeTensor("mask"),
        output_counts=FakeTensor("counts"),
        output_counts_gathered=FakeTensor("gathered"),
    )

    def allocate(*, host, shard_dims):
        result = FakeTensor("keep")
        allocated.append((host.clone(), shard_dims, result))
        return result

    penalty_state._alloc_int_buffer = allocate
    monkeypatch.setattr(
        tt_penalties.ttnn,
        "mul",
        lambda value, keep, *, output_tensor, **kwargs: multiplies.append((value, keep, output_tensor))
        or output_tensor,
    )

    tt_penalties.TTPenalties.reset_output_tokens(penalty_state, slots=[3, 1])

    assert allocated[0][0].reshape(-1).tolist() == [1, 0, 1, 0]
    assert allocated[0][1] == (0, None)
    keep = allocated[0][2]
    assert [(value.name, mask is keep, output.name) for value, mask, output in multiplies] == [
        ("mask", True, "mask"),
        ("counts", True, "counts"),
        ("gathered", True, "gathered"),
    ]
    assert keep.deallocated


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
        sampling = SimpleNamespace(
            seed_manager=SimpleNamespace(max_batch_size=4),
            apply_slot_remap=lambda remap, lane=lane: calls[lane].append(torch.as_tensor(remap).tolist()),
        )
        models.append(SimpleNamespace(sampling=sampling))
    fake = SimpleNamespace(data_parallel=2, model=models)

    Generator._apply_sampling_slot_remap(fake, torch.tensor([1, 0, 3, 2]))

    assert calls == [[[1, 0, 2, 3]], [[1, 0, 2, 3]]]


def test_shared_generator_rejects_cross_lane_sampling_remap(expect_error):
    from models.tt_transformers.tt.generator import Generator

    sampling = SimpleNamespace(seed_manager=SimpleNamespace(max_batch_size=2), apply_slot_remap=lambda remap: None)
    fake = SimpleNamespace(
        data_parallel=2,
        model=[
            SimpleNamespace(sampling=sampling),
            SimpleNamespace(sampling=sampling),
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


def test_all_known_shared_generator_callers_supply_every_decode_update_command():
    required = {
        "reload_inputs",
        "reload_page_table",
        "reload_sampling_params",
        "reset_sampling_state",
    }
    expected_calls = {
        Path("tt-train/sources/examples/grpo_remote_rollout/utils/ttt_generation_worker.py"): 1,
        Path("models/experimental/ops/quasar/gpt_oss/demo/text_demo.py"): 2,
        Path("models/experimental/ops/quasar/gpt_oss/tests/accuracy/test_model.py"): 1,
        Path("models/experimental/ops/quasar/qwen3_vl/demo/demo.py"): 1,
    }

    for source_path, expected_count in expected_calls.items():
        tree = ast.parse(source_path.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "decode_forward"
        ]
        assert len(calls) == expected_count, source_path
        for call in calls:
            assert required <= {keyword.arg for keyword in call.keywords}, (source_path, call.lineno)


def test_shared_generator_preserves_explicit_commands_after_mainline_seed_fix():
    source_path = Path("models/tt_transformers/tt/generator.py")
    source_text = source_path.read_text()
    tree = ast.parse(source_text)
    generator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Generator")
    decode = next(
        node for node in generator.body if isinstance(node, ast.FunctionDef) and node.name == "decode_forward"
    )
    trace_decode = next(
        node
        for node in generator.body
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_forward_trace_text"
    )

    required = {"reload_inputs", "reload_page_table", "reload_sampling_params", "reset_sampling_state"}
    assert required <= {arg.arg for arg in decode.args.kwonlyargs}
    assert {"reload_inputs", "reload_page_table"} <= {arg.arg for arg in trace_decode.args.kwonlyargs}

    decode_source = ast.get_source_segment(source_text, decode)
    trace_source = ast.get_source_segment(source_text, trace_decode)
    assert decode_source is not None
    assert trace_source is not None
    assert "reset_batch" not in decode_source
    assert "_prev_on_device_sampling" not in decode_source
    assert "_tt_vllm_always_refresh_decode_trace_inputs" not in trace_source
    assert "torch.equal" not in trace_source


def test_gemma4_override_uses_only_explicit_decode_update_commands():
    source_path = Path("models/demos/gemma4/tt/generator.py")
    tree = ast.parse(source_path.read_text())
    mixin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChunkedPrefillPageTableGuardMixin"
    )
    decode = next(node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name == "decode_forward")
    trace_decode = next(
        node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name == "_decode_forward_trace_text"
    )
    required = {
        "reload_inputs",
        "reload_page_table",
        "reload_sampling_params",
        "reset_sampling_state",
    }

    decode_params = {arg.arg for arg in decode.args.kwonlyargs}
    trace_params = {arg.arg for arg in trace_decode.args.kwonlyargs}
    assert required <= decode_params
    assert {"reload_inputs", "reload_page_table"} <= trace_params
    assert "reset_batch" not in {arg.arg for arg in decode.args.args}
    assert "reset_batch" not in {arg.arg for arg in trace_decode.args.args}

    source = ast.get_source_segment(source_path.read_text(), trace_decode)
    assert source is not None
    assert "_prev_on_device_sampling" not in source
    assert "_tt_vllm_always_refresh_decode_trace_inputs" not in source
    assert "torch.equal" not in source


def test_gemma4_pli_explicitly_disables_decode_token_feedback():
    source = Path("models/demos/gemma4/tt/model.py").read_text()

    assert "_tt_supports_decode_token_feedback = False" in source
    assert "self._tt_supports_decode_token_feedback = not self._tt_vllm_always_refresh_decode_trace_inputs" in source


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
        _slot_state_requires_authoritative_reload=False,
        reset_sampling_params=lambda params: (_ for _ in ()).throw(
            AssertionError("reset-only must not upload sampling parameters")
        ),
        reset_prompt_tokens=lambda tokens: None,
        reset_output_state=lambda tokens: None,
        sample=lambda **kwargs: "tokens",
    )
    sampling.validate_decode_state_commands = lambda **kwargs: SamplingGenerator.validate_decode_state_commands(
        sampling, **kwargs
    )
    sampling.commit_decode_state_commands = lambda **kwargs: SamplingGenerator.commit_decode_state_commands(
        sampling, **kwargs
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


def test_galaxy_slot_remap_moves_parameter_shadow_with_seed_state():
    from models.demos.llama3_70b_galaxy.tt.generator import Generator

    seed_remaps = []
    fake = SimpleNamespace(
        model=SimpleNamespace(
            sampling=SimpleNamespace(
                seed_manager=SimpleNamespace(max_batch_size=4),
                apply_slot_remap=lambda remap: seed_remaps.append(remap),
            )
        ),
        model_args=SimpleNamespace(max_batch_size=4),
        _slot_sampling_params={
            "temperature": [0.1, 0.2, 0.3, 0.4],
            "top_k": [1, 2, 3, 4],
        },
    )

    Generator._apply_sampling_slot_remap(fake, [2, 0, 1, 3])

    assert seed_remaps == [[2, 0, 1, 3]]
    assert fake._slot_sampling_params == {
        "temperature": [0.3, 0.1, 0.2, 0.4],
        "top_k": [3, 1, 2, 4],
    }


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
