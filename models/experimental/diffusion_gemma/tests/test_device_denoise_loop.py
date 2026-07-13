# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the assembled DiffusionGemma denoise loop step (#47463)."""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block as ref_denoise_block
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories
from models.experimental.diffusion_gemma.tt.denoise_loop import (
    _ids_to_torch,
    denoise_block,
    denoise_step,
    renoise,
    run_fixed_denoise_steps,
    temperature_at_step,
)
from models.experimental.diffusion_gemma.tt.sampling import ChunkedGumbelNoise
from models.experimental.diffusion_gemma.tt.traced_denoise import TracedDenoiseController
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,
]


def _to_device(device, value, *, dtype=ttnn.float32):
    return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _structured_logits(length: int, vocab_size: int):
    """Logits with stable argmax and well-separated entropy ordering."""
    logits = torch.full((1, length, vocab_size), -4.0, dtype=torch.float32)
    token_ids = torch.arange(length) % vocab_size
    sharpness = torch.linspace(0.25, 2.0, length)
    logits[0, torch.arange(length), token_ids] = sharpness
    logits += torch.randn_like(logits) * 1.0e-3
    return logits


def _budget_for_accept_count(entropy: torch.Tensor, count: int):
    sorted_entropy = torch.sort(entropy, dim=-1).values
    exclusive = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
    return float((exclusive[0, count - 1] + exclusive[0, count]) / 2)


class _ResettableStaticLogits:
    def __init__(self, logits):
        self.logits = logits
        self.reset_calls = 0

    def __call__(self, canvas, step):
        return self.logits

    def reset(self):
        self.reset_calls += 1
        if self.logits is not None:
            self.logits.deallocate(True)
            self.logits = None


class _TraceStaticLogits:
    """Minimal trace-controller adapter over one persistent synthetic logits tensor."""

    def __init__(self, device, logits, start_pos=32):
        self.tt_model = type("_TtModel", (), {"mesh_device": device})()
        self.logits = logits
        self.q_rope_offset = start_pos
        self.use_canvas_rope = False

    def __call__(self, canvas, step):
        del canvas, step
        return self.logits

    def owns_logits(self, value):
        return value is self.logits

    def reset(self):
        pass

    def prepare_trace_safe_self_conditioning(self, *, canvas_len):
        del canvas_len

    def prepare_canvas_rope_buffers(self, *, canvas_len):
        del canvas_len

    def update_canvas_rope_buffers(self, start_pos):
        self.q_rope_offset = start_pos

    def reset_signal_buffer(self):
        pass

    def sharded_terminal_context(self):
        return None


def test_single_denoise_step_matches_reference(device):
    torch.manual_seed(11)
    length = 256
    vocab_size = 256
    max_steps = 48
    step = 3
    temperature = temperature_at_step(step, max_steps, 0.8, 0.4)

    logits = _structured_logits(length, vocab_size)
    gumbel_noise = torch.zeros_like(logits)
    noise_tokens = torch.randint(0, vocab_size, (1, length), dtype=torch.long)
    ref_entropy = S.token_entropy(logits, temperature=temperature)
    accept_count = 96
    budget = _budget_for_accept_count(ref_entropy, accept_count)
    ref = S.denoise_step(
        logits,
        temperature=temperature,
        entropy_budget=budget,
        vocab_size=vocab_size,
        sampler=S.SAMPLER_GUMBEL,
        gumbel_noise=gumbel_noise,
        noise_tokens=noise_tokens,
        min_accept=0,
    )

    tt = denoise_step(
        _to_device(device, logits.unsqueeze(1)),
        temperature=temperature,
        entropy_budget=budget,
        gumbel_noise=_to_device(device, gumbel_noise.unsqueeze(1)),
        noise_tokens=_to_device(device, noise_tokens.view(1, 1, length, 1).to(torch.int32), dtype=ttnn.uint32),
    )

    out_entropy = ttnn.to_torch(tt.entropy).squeeze(1).squeeze(-1).float()
    out_accept = ttnn.to_torch(tt.accept_mask).squeeze(1).squeeze(1) > 0.5
    out_sampled = ttnn.to_torch(tt.sampled).squeeze(1).squeeze(-1).to(torch.long)
    out_argmax = ttnn.to_torch(tt.argmax).squeeze(1).squeeze(-1).to(torch.long)
    out_canvas = ttnn.to_torch(tt.canvas).squeeze(1).squeeze(-1).to(torch.long)

    passing, message = assert_with_pcc(ref.entropy.float(), out_entropy.float(), 0.99)
    assert passing, message
    assert torch.equal(out_accept, ref.accept_mask)
    assert torch.equal(out_sampled, ref.sampled)
    assert torch.equal(out_argmax, ref.argmax)
    assert torch.equal(out_canvas, ref.canvas)
    assert int(out_accept.sum()) == accept_count


def test_uint32_renoise_preserves_full_vocab_token_ids(device):
    sampled = torch.tensor([0, 1, 65535, 65536, 131071, 131072, 262143, 17], dtype=torch.int32).view(1, 1, 8, 1)
    noise_tokens = torch.tensor([262143, 131072, 131071, 65536, 65535, 1, 0, 2048], dtype=torch.int32).view(1, 1, 8, 1)
    accept = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32).view(1, 1, 8, 1)
    ref = torch.where(accept.bool(), sampled, noise_tokens).view(1, 8).to(torch.long)

    out = renoise(
        _to_device(device, accept, dtype=ttnn.bfloat16),
        _to_device(device, sampled, dtype=ttnn.uint32),
        _to_device(device, noise_tokens, dtype=ttnn.uint32),
    )

    assert torch.equal(ttnn.to_torch(out).squeeze(1).squeeze(-1).to(torch.long), ref)


def test_multi_step_denoise_control_flow_smoke_matches_reference(device):
    """Synthetic controller smoke; real canvas->W2 logits cycling is covered in the integration suite."""
    torch.manual_seed(17)
    batch = 1
    length = 256
    vocab_size = 256
    max_steps = 4

    logits = _structured_logits(length, vocab_size)
    step0_temperature = temperature_at_step(0, max_steps, 0.8, 0.4)
    ref_entropy = S.token_entropy(logits, temperature=step0_temperature)
    budget = _budget_for_accept_count(ref_entropy, 96)
    cfg = DiffusionConfig(
        max_denoise_steps=max_steps,
        entropy_stop_threshold=10.0,
        stable_steps_to_halt=1,
        entropy_budget=budget,
    )
    init_canvas = torch.randint(0, vocab_size, (batch, length), dtype=torch.long)
    gumbel_noise = [torch.zeros_like(logits) for _ in range(max_steps)]
    noise_tokens = [torch.randint(0, vocab_size, (batch, length), dtype=torch.long) for _ in range(max_steps)]

    ref = ref_denoise_block(
        lambda canvas, step: logits,
        init_canvas,
        cfg,
        vocab_size,
        gumbel_noise_fn=lambda step: gumbel_noise[step],
        noise_tokens_fn=lambda step: noise_tokens[step],
    )

    tt_logits = _ResettableStaticLogits(_to_device(device, logits.unsqueeze(1)))
    tt_gumbel_noise = [_to_device(device, noise.unsqueeze(1)) for noise in gumbel_noise]
    tt_noise_tokens = [
        _to_device(device, noise.view(batch, 1, length, 1).to(torch.int32), dtype=ttnn.uint32) for noise in noise_tokens
    ]
    tt = denoise_block(
        tt_logits,
        _to_device(device, init_canvas.view(batch, 1, length, 1).to(torch.int32), dtype=ttnn.uint32),
        cfg,
        gumbel_noise_fn=lambda step: tt_gumbel_noise[step],
        noise_tokens_fn=lambda step: tt_noise_tokens[step],
    )

    comparison = compare_trajectories(ref, tt, max_entropy_abs_err_threshold=0.2)
    accept_flips = sum(int((ra.accept_mask != rb.accept_mask).sum()) for ra, rb in zip(ref.per_step, tt.per_step))
    assert comparison.passed, comparison
    assert ref.halted and tt.halted
    assert ref.num_steps == tt.num_steps == 2
    assert accept_flips == 0
    assert tt_logits.reset_calls == 1


def test_traced_materialized_gumbel_refresh_matches_fixed_loop_across_blocks(device):
    """One stable Gumbel trace-input buffer must reproduce fresh per-step/block noise."""
    torch.manual_seed(29)
    length = 256
    vocab_size = 256
    steps = 2
    cfg = DiffusionConfig(
        canvas_length=length,
        max_denoise_steps=steps,
        entropy_stop_threshold=-1.0,
        stable_steps_to_halt=1,
    )
    logits_torch = _structured_logits(length, vocab_size).unsqueeze(1)
    block_inputs = []
    for block in range(2):
        generator = torch.Generator().manual_seed(1000 + block)
        block_inputs.append(
            {
                "init": torch.randint(0, vocab_size, (1, 1, length, 1), generator=generator, dtype=torch.int32),
                "gumbel": [
                    -torch.log(
                        -torch.log(
                            torch.rand((1, 1, length, vocab_size), generator=generator, dtype=torch.float32) + 1.0e-10
                        )
                        + 1.0e-10
                    )
                    for _ in range(steps)
                ],
                "noise": [
                    torch.randint(
                        0,
                        vocab_size,
                        (1, 1, length, 1),
                        generator=generator,
                        dtype=torch.int32,
                    )
                    for _ in range(steps)
                ],
            }
        )

    def run_fixed(inputs):
        logits = _to_device(device, logits_torch)
        adapter = _TraceStaticLogits(device, logits)
        committed = run_fixed_denoise_steps(
            adapter,
            _to_device(device, inputs["init"], dtype=ttnn.uint32),
            cfg,
            gumbel_noise_fn=lambda step: _to_device(device, inputs["gumbel"][step]),
            noise_tokens_fn=lambda step: _to_device(device, inputs["noise"][step], dtype=ttnn.uint32),
        )
        host = _ids_to_torch(committed)
        committed.deallocate(True)
        logits.deallocate(True)
        return host

    expected = [run_fixed(inputs) for inputs in block_inputs]

    traced_logits = _to_device(device, logits_torch)
    traced_adapter = _TraceStaticLogits(device, traced_logits)
    controller = TracedDenoiseController(device, cfg)
    actual = []
    try:
        for block, inputs in enumerate(block_inputs):
            traced_adapter.q_rope_offset = 32 + block * length
            trajectory = controller.denoise_block(
                traced_adapter,
                _to_device(device, inputs["init"], dtype=ttnn.uint32),
                cfg,
                gumbel_noise_fn=lambda step, inputs=inputs: _to_device(device, inputs["gumbel"][step]),
                noise_tokens_fn=lambda step, inputs=inputs: _to_device(
                    device, inputs["noise"][step], dtype=ttnn.uint32
                ),
            )
            actual.append(trajectory.committed)
    finally:
        controller.release()
        traced_logits.deallocate(True)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert controller.capture_events == 1
    assert controller.traces_captured == steps
    assert controller.replay_blocks == 2


def test_traced_chunked_gumbel_dynamic_seed_matches_fixed_loop_across_blocks(device):
    """Chunked trace replay must refresh its base seed and preserve per-step offsets."""
    torch.manual_seed(31)
    length = 256
    vocab_size = 512
    chunk_size = 256
    steps = 2
    cfg = DiffusionConfig(
        canvas_length=length,
        max_denoise_steps=steps,
        entropy_stop_threshold=-1.0,
        stable_steps_to_halt=1,
    )
    logits_torch = _structured_logits(length, vocab_size).unsqueeze(1)
    block_inputs = []
    for block in range(2):
        generator = torch.Generator().manual_seed(2000 + block)
        block_inputs.append(
            {
                "init": torch.randint(0, vocab_size, (1, 1, length, 1), generator=generator, dtype=torch.int32),
                "noise": [
                    torch.randint(
                        0,
                        vocab_size,
                        (1, 1, length, 1),
                        generator=generator,
                        dtype=torch.int32,
                    )
                    for _ in range(steps)
                ],
                "seed": 37 + block * 1_000_003,
            }
        )

    def gumbel_for(inputs):
        return lambda step: ChunkedGumbelNoise(
            seed=inputs["seed"] + step,
            vocab_chunk_size=chunk_size,
        )

    def run_fixed(inputs):
        logits = _to_device(device, logits_torch)
        adapter = _TraceStaticLogits(device, logits)
        committed = run_fixed_denoise_steps(
            adapter,
            _to_device(device, inputs["init"], dtype=ttnn.uint32),
            cfg,
            gumbel_noise_fn=gumbel_for(inputs),
            noise_tokens_fn=lambda step: _to_device(device, inputs["noise"][step], dtype=ttnn.uint32),
        )
        host = _ids_to_torch(committed)
        committed.deallocate(True)
        logits.deallocate(True)
        return host

    expected = [run_fixed(inputs) for inputs in block_inputs]

    traced_logits = _to_device(device, logits_torch)
    traced_adapter = _TraceStaticLogits(device, traced_logits)
    controller = TracedDenoiseController(device, cfg)
    actual = []
    try:
        for block, inputs in enumerate(block_inputs):
            traced_adapter.q_rope_offset = 32 + block * length
            trajectory = controller.denoise_block(
                traced_adapter,
                _to_device(device, inputs["init"], dtype=ttnn.uint32),
                cfg,
                gumbel_noise_fn=gumbel_for(inputs),
                noise_tokens_fn=lambda step, inputs=inputs: _to_device(
                    device, inputs["noise"][step], dtype=ttnn.uint32
                ),
            )
            actual.append(trajectory.committed)
    finally:
        controller.release()
        traced_logits.deallocate(True)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert controller.gumbel_chunked_state is None
    assert controller.capture_events == 1
    assert controller.traces_captured == steps
    assert controller.replay_blocks == 2
