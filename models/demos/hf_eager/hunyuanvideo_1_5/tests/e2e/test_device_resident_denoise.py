# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import (
    DeviceResidentFlowMatchScheduler,
    HunyuanVideo15Pipeline,
    TTTransformerAdapter,
)


class _FakeResidentTT:
    """Torch oracle for testing the diffusers adapter contract without hardware."""

    def __init__(self, latent, predictions):
        self.latent = latent
        self.predictions = iter(predictions)

    def denoise_resident_update(self, _output, *, sigma, sigma_next, guidance_scale, original_cfg, delta=None):
        pred_cond, pred_uncond = next(self.predictions)
        if pred_uncond is None:
            guided = pred_cond
        else:
            base = pred_cond if original_cfg else pred_uncond
            guided = base + guidance_scale * (pred_cond - pred_uncond)
        dt = sigma_next - sigma if delta is None else delta
        delta_velocity = dt * guided
        self.latent = (self.latent.float() + delta_velocity).to(guided.dtype)

    def denoise_resident_to_torch(self):
        return self.latent


class _FakeTraceTT:
    """Execution-count oracle for the adapter's first traced CFG group."""

    def __init__(self):
        self.calls = []
        self.trace_output = object()
        self.first_output = self.trace_output

    def denoise_resident_trace_setup(self, inputs, *, num_conditions, torch_dtype):
        self.calls.append(
            ("setup", num_conditions, torch_dtype, inputs["timestep"].clone(), inputs.get("use_joint_mask", False))
        )

    def denoise_trace_capture(self, *, blocking):
        self.calls.append(("capture_execute", blocking))
        return 17, self.trace_output, self.first_output


class _FakeSeparateResidentTT:
    """Execution/shape oracle for mixed-length resident CFG."""

    def __init__(self):
        self.calls = []
        self.outputs = [object(), object()]

    @staticmethod
    def _lengths(inputs):
        return [
            (
                int(item["encoder_attention_mask"].sum()),
                int(item["encoder_attention_mask_2"].sum()),
            )
            for item in inputs
        ]

    def denoise_resident_eager_setup_conditions(self, inputs, *, num_conditions, batch, torch_dtype):
        self.calls.append(("eager_setup", num_conditions, batch, torch_dtype, self._lengths(inputs)))
        return self.outputs

    def denoise_resident_eager_step_conditions(self, timestep):
        self.calls.append(("eager_step", timestep.clone()))
        return self.outputs

    def denoise_resident_trace_setup_conditions(self, inputs, *, num_conditions, batch, torch_dtype):
        self.calls.append(("trace_setup", num_conditions, batch, torch_dtype, self._lengths(inputs)))
        return [41, 42], self.outputs

    def denoise_resident_trace_step_conditions(self, timestep, trace_ids, outputs):
        self.calls.append(("trace_step", timestep.clone(), list(trace_ids)))
        return outputs


def _adapter_inputs(qwen_length, byt5_length):
    return dict(
        hidden_states=torch.randn(1, 65, 1, 2, 2, dtype=torch.bfloat16),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 4, 8),
        encoder_attention_mask=torch.tensor([[1] * qwen_length + [0] * (4 - qwen_length)]),
        encoder_hidden_states_2=torch.randn(1, 3, 6),
        encoder_attention_mask_2=torch.tensor([[1] * byt5_length + [0] * (3 - byt5_length)]),
        image_embeds=torch.randn(1, 2, 7),
    )


@pytest.mark.parametrize("use_trace", [False, True])
def test_mixed_length_resident_cfg_uses_exact_condition_executions_or_fails_closed(monkeypatch, use_trace):
    """Unequal CFG rows run exactly in eager; unsafe trace fails without fallback."""
    monkeypatch.setenv("HY_CFG_PADDING_POLICY", "separate")
    fake_tt = _FakeSeparateResidentTT()
    adapter = TTTransformerAdapter(
        SimpleNamespace(config=SimpleNamespace(), dtype=torch.bfloat16),
        fake_tt,
        SimpleNamespace(num_conditions=2),
        use_trace=use_trace,
        device_resident=True,
        task="i2v",
        counters={},
    )

    first = adapter(**_adapter_inputs(4, 3)).sample
    if use_trace:
        with pytest.raises(ValueError, match="use HY_TRACE=0 to preserve exact conditioning"):
            adapter(**_adapter_inputs(2, 1))
        assert first.numel() == 0
        assert adapter.__dict__["_counters"] == {"n": 2}
        assert fake_tt.calls == []
        return
    second = adapter(**_adapter_inputs(2, 1)).sample

    assert first.numel() == second.numel() == 0
    assert adapter.__dict__["_counters"] == {"n": 2, "device_runs": 2}
    assert fake_tt.calls == [
        (
            "eager_setup",
            2,
            1,
            torch.bfloat16,
            [(4, 3), (2, 1)],
        )
    ]
    assert adapter.__dict__["_device_output"] == fake_tt.outputs


def test_mixed_length_masked_cfg_uses_one_fixed_trace_execution(monkeypatch):
    """The opt-in masked path batches unequal positive/unconditional rows."""
    monkeypatch.setenv("HY_CFG_PADDING_POLICY", "masked")
    fake_tt = _FakeTraceTT()
    counters = {}
    adapter = TTTransformerAdapter(
        SimpleNamespace(config=SimpleNamespace(), dtype=torch.bfloat16),
        fake_tt,
        SimpleNamespace(num_conditions=2),
        use_trace=True,
        device_resident=True,
        task="i2v",
        counters=counters,
    )

    adapter(**_adapter_inputs(4, 3))
    adapter(**_adapter_inputs(2, 1))

    setup = fake_tt.calls[0]
    assert setup[0] == "setup"
    assert setup[1:3] == (2, torch.bfloat16)
    assert setup[3].shape == (2,)
    assert setup[4] is True
    assert fake_tt.calls[1] == ("capture_execute", True)
    assert counters == {"n": 2, "device_runs": 1}


def test_mixed_length_batch_is_split_condition_major_per_row():
    group = []
    for qwen_masks, byt5_masks in (
        ([[1, 1, 1, 0], [1, 1, 0, 0]], [[1, 1, 0], [1, 0, 0]]),
        ([[1, 0, 0, 0], [1, 1, 1, 1]], [[1, 0, 0], [1, 1, 1]]),
    ):
        item = _adapter_inputs(4, 3)
        item = {
            key: value.repeat(2, *([1] * (value.ndim - 1))) if torch.is_tensor(value) else value
            for key, value in item.items()
        }
        item["encoder_attention_mask"] = torch.tensor(qwen_masks)
        item["encoder_attention_mask_2"] = torch.tensor(byt5_masks)
        group.append((item, torch.bfloat16, torch.empty(0)))

    separated, batch = TTTransformerAdapter._separate_group_inputs(group)

    assert batch == 2
    assert TTTransformerAdapter._padding_signatures(group) == [(3, 2), (2, 1), (1, 1), (4, 3)]
    assert [
        (int(item["encoder_attention_mask"].sum()), int(item["encoder_attention_mask_2"].sum())) for item in separated
    ] == [(3, 2), (2, 1), (1, 1), (4, 3)]


def test_mixed_trace_compiles_all_shapes_before_grouped_capture(monkeypatch):
    pipe = object.__new__(HunyuanVideo15Pipeline)
    pipe.device = object()
    pipe._write_event = None
    pipe._op_event = None
    pipe._capturing = False
    events = []
    residents = iter(
        [
            {"hidden_sp": object(), "logical_n": 4, "shape": "positive"},
            {"hidden_sp": object(), "logical_n": 4, "shape": "negative"},
        ]
    )
    pipe.denoise_trace_setup = lambda item: next(residents)
    pipe._setup_separate_resident_denoise = lambda *args, **kwargs: None
    pipe._activate_resident_trace_slot = lambda slot: setattr(pipe, "_active_shape", slot["resident"]["shape"])
    pipe._save_resident_trace_slot = lambda slot: None
    pipe.denoise_trace_step = (
        lambda: events.append(("capture" if pipe._capturing else "compile", pipe._active_shape)) or object()
    )
    monkeypatch.setattr(
        ttnn,
        "begin_trace_capture",
        lambda device, cq_id: setattr(pipe, "_capturing", True) or events.append(("begin",)) or 71,
    )
    monkeypatch.setattr(
        ttnn,
        "end_trace_capture",
        lambda device, trace_id, cq_id: setattr(pipe, "_capturing", False) or events.append(("end",)),
    )
    monkeypatch.setattr(ttnn, "execute_trace", lambda device, trace_id, cq_id, blocking: events.append(("execute",)))
    monkeypatch.setattr(ttnn, "record_event", lambda device, cq_id: object())

    trace_ids, outputs = pipe.denoise_resident_trace_setup_conditions(
        [{}, {}], num_conditions=2, batch=1, torch_dtype=torch.bfloat16
    )

    assert trace_ids == [71]
    assert len(outputs) == 2
    assert events == [
        ("compile", "positive"),
        ("compile", "negative"),
        ("begin",),
        ("capture", "positive"),
        ("capture", "negative"),
        ("end",),
        ("execute",),
    ]


def test_trace_capture_executes_captured_first_step_after_compile(monkeypatch):
    """Compile warmup is followed by capture and one real trace execution."""
    calls = []
    pipe = object.__new__(HunyuanVideo15Pipeline)
    pipe.device = object()
    pipe._write_event = None
    compile_output = object()
    captured_output = object()
    outputs = iter((compile_output, captured_output))
    pipe.denoise_trace_step = lambda: calls.append("step") or next(outputs)
    pipe.denoise_trace_execute = lambda trace_id, blocking: calls.append(("execute", trace_id, blocking))

    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda device, cq_id: calls.append("begin") or 23)
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda device, trace_id, cq_id: calls.append("end"))

    trace_id, trace_output, first_output = pipe.denoise_trace_capture(blocking=True)

    assert trace_id == 23
    assert trace_output is captured_output
    assert first_output is captured_output
    assert calls == ["step", "begin", "step", "end", ("execute", 23, True)]


def test_first_cfg_group_has_one_capture_execution():
    """Two diffusers condition calls collapse to one returned traced DiT step."""
    fake_tt = _FakeTraceTT()
    real = SimpleNamespace(config=SimpleNamespace(), dtype=torch.bfloat16)
    guider = SimpleNamespace(num_conditions=2)
    counters = {}
    adapter = TTTransformerAdapter(
        real,
        fake_tt,
        guider,
        use_trace=True,
        device_resident=True,
        task="i2v",
        counters=counters,
    )
    common = dict(
        hidden_states=torch.randn(1, 65, 1, 2, 2, dtype=torch.bfloat16),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 3, 4),
        encoder_attention_mask=torch.ones(1, 3, dtype=torch.long),
        encoder_hidden_states_2=torch.randn(1, 2, 5),
        encoder_attention_mask_2=torch.ones(1, 2, dtype=torch.long),
        image_embeds=torch.randn(1, 2, 6),
    )

    adapter(**common)
    adapter(**common)

    assert [call[0] for call in fake_tt.calls] == ["setup", "capture_execute"]
    assert counters == {"n": 2, "device_runs": 1}
    assert adapter.__dict__["_device_output"] is fake_tt.first_output
    assert adapter.__dict__["_trace_out"] is fake_tt.trace_output


def test_adapter_release_trace_frees_and_resets_state(monkeypatch):
    released = []
    ttpipe = SimpleNamespace(
        device=object(),
        _resident=object(),
        _resident_denoise=object(),
        _write_event=object(),
        _op_event=object(),
    )
    adapter = TTTransformerAdapter(
        SimpleNamespace(config=SimpleNamespace(), dtype=torch.bfloat16),
        ttpipe,
        SimpleNamespace(num_conditions=1),
        use_trace=True,
    )
    adapter.__dict__["_trace_id"] = 31
    adapter.__dict__["_trace_ids"] = [32, 33]
    adapter.__dict__["_trace_out"] = object()
    adapter.__dict__["_resident_started"] = True
    adapter.__dict__["_resident_conditions"] = 1
    adapter.__dict__["_device_output"] = object()
    monkeypatch.setattr(ttnn, "release_trace", lambda device, trace_id: released.append((device, trace_id)))

    adapter.release_trace()

    assert released == [(ttpipe.device, 31), (ttpipe.device, 32), (ttpipe.device, 33)]
    assert adapter.__dict__["_trace_id"] is None
    assert adapter.__dict__["_trace_ids"] == []
    assert adapter.__dict__["_resident_started"] is False
    assert ttpipe._resident is None
    assert ttpipe._resident_denoise is None


@pytest.mark.parametrize("shape", [(1, 32, 2, 2, 2), (2, 32, 1, 3, 4)])
@pytest.mark.parametrize("cfg_enabled", [False, True])
@pytest.mark.parametrize("original_cfg", [False, True])
@pytest.mark.parametrize("shift", [5.0, 9.0])
def test_resident_scheduler_contract_matches_diffusers(shape, cfg_enabled, original_cfg, shift):
    """Indexing, dtype, terminal sigma, CFG, and final-only handoff match diffusers."""
    generator = torch.Generator().manual_seed(17)
    latent = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
    predictions = []
    for _ in range(5):
        cond = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
        uncond = torch.randn(shape, generator=generator, dtype=torch.bfloat16) if cfg_enabled else None
        predictions.append((cond, uncond))

    reference_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    reference_scheduler.set_timesteps(5)
    expected = latent.clone()
    scale = 4.5
    for index, (cond, uncond) in enumerate(predictions):
        if uncond is None:
            guided = cond
        else:
            base = cond if original_cfg else uncond
            guided = base + scale * (cond - uncond)
        expected = reference_scheduler.step(guided, reference_scheduler.timesteps[index], expected, return_dict=False)[
            0
        ]

    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    scheduler.set_timesteps(5)
    ttpipe = _FakeResidentTT(latent.clone(), predictions)
    adapter = SimpleNamespace(_tt=ttpipe, _device_output=object())
    guider = SimpleNamespace(
        guidance_scale=scale,
        guidance_rescale=0.0,
        use_original_formulation=original_cfg,
        _start=0.0,
        _stop=1.0,
        _enabled=cfg_enabled,
    )
    resident_scheduler = DeviceResidentFlowMatchScheduler(scheduler, adapter, guider)

    host_placeholder = latent.clone()
    for index, timestep in enumerate(scheduler.timesteps):
        adapter._device_output = object()
        result = resident_scheduler.step(
            torch.empty(0, dtype=latent.dtype),
            timestep,
            host_placeholder,
            return_dict=False,
        )[0]
        if index < len(scheduler.timesteps) - 1:
            assert result.data_ptr() == host_placeholder.data_ptr()
        else:
            torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("separate_outputs", [False, True])
@pytest.mark.parametrize("cfg_enabled", [False, True])
def test_resident_device_cfg_euler_matches_diffusers(device, cfg_enabled, separate_outputs):
    """The actual TTNN CFG + Euler kernels match host diffusers on one device."""
    generator = torch.Generator().manual_seed(29)
    batch, sequence, channels = 1, 64, 32
    latent = torch.randn(batch, sequence, channels, generator=generator, dtype=torch.bfloat16)
    cond = torch.randn(batch, sequence, channels, generator=generator, dtype=torch.bfloat16)
    uncond = torch.randn(batch, sequence, channels, generator=generator, dtype=torch.bfloat16)
    model_output = torch.cat([cond, uncond], dim=0) if cfg_enabled else cond
    scale = 6.0

    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    scheduler.set_timesteps(7)
    step_index = 3
    guided = uncond + scale * (cond - uncond) if cfg_enabled else cond
    expected = scheduler.step(
        guided,
        scheduler.timesteps[step_index],
        latent,
        return_dict=False,
    )[0]

    latent_tt = ttnn.from_torch(latent, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tt = ttnn.from_torch(model_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pipe = object.__new__(HunyuanVideo15Pipeline)
    pipe.out_channels = channels
    pipe._resident = None
    pipe._resident_denoise = dict(
        ctx={},
        num_conditions=2 if cfg_enabled else 1,
        batch=batch,
        shard_n=sequence,
        in_channels=channels,
        latent=latent_tt,
        latent_dtype=ttnn.bfloat16,
        static_cond=None,
        traced=False,
        out_shape=(batch, channels, 1, 1, sequence),
    )
    device_output = (
        [ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)]
        if separate_outputs and not cfg_enabled
        else (
            [
                ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(uncond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            ]
            if separate_outputs
            else output_tt
        )
    )
    actual_tt = pipe.denoise_resident_update(
        device_output,
        sigma=float(scheduler.sigmas[step_index]),
        sigma_next=float(scheduler.sigmas[step_index + 1]),
        guidance_scale=scale,
    )
    actual = ttnn.to_torch(actual_tt)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_resident_scheduler_rejects_dynamic_or_rescaled_cfg():
    scheduler = FlowMatchEulerDiscreteScheduler()
    adapter = SimpleNamespace()
    dynamic = SimpleNamespace(
        guidance_rescale=0.0,
        _start=0.2,
        _stop=1.0,
        _enabled=True,
    )
    with pytest.raises(ValueError, match="start=0 and stop=1"):
        DeviceResidentFlowMatchScheduler(scheduler, adapter, dynamic)

    rescaled = SimpleNamespace(
        guidance_rescale=0.5,
        _start=0.0,
        _stop=1.0,
        _enabled=True,
    )
    with pytest.raises(ValueError, match="guidance_rescale"):
        DeviceResidentFlowMatchScheduler(scheduler, adapter, rescaled)

    stochastic = FlowMatchEulerDiscreteScheduler(stochastic_sampling=True)
    with pytest.raises(ValueError, match="stochastic_sampling=False"):
        DeviceResidentFlowMatchScheduler(stochastic, adapter, dynamic)


def test_resident_scheduler_rejects_per_token_timesteps():
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    adapter = SimpleNamespace(_tt=SimpleNamespace(), _device_output=object())
    guider = SimpleNamespace(
        guidance_scale=1.0,
        guidance_rescale=0.0,
        use_original_formulation=False,
        _start=0.0,
        _stop=1.0,
        _enabled=False,
    )
    resident = DeviceResidentFlowMatchScheduler(scheduler, adapter, guider)
    with pytest.raises(ValueError, match="per_token_timesteps"):
        resident.step(
            torch.empty(0, dtype=torch.bfloat16),
            scheduler.timesteps[0],
            torch.zeros(1, dtype=torch.bfloat16),
            per_token_timesteps=torch.ones(1),
        )
