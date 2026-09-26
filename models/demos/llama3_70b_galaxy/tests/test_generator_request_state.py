# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host regressions for Galaxy request completion and prefill result packing."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ttnn
from models.common.sampling import SamplingParams, SeedManager
from models.demos.llama3_70b_galaxy.tt.generator import Generator


@pytest.mark.parametrize("flags", [[True, False, True], [False, True, False], [True] * 3, [False] * 3])
def test_seeded_prefill_logprobs_keep_request_rows(monkeypatch, flags):
    # Run the actual prefill coordinator; replace only device/model execution.
    # Non-monotonic slots and distinct values expose both compaction and scatter errors.
    sampling = SimpleNamespace(
        reset_prompt_tokens=Mock(),
        reset_output_state=Mock(),
        seed_manager=Mock(),
    )

    def reset_params(params):
        sampling.params = params

    def sample(logits, **kwargs):
        token = logits.reshape(-1)[0].to(torch.int32)
        lp = -token.float() / 100 if sampling.params.enable_log_probs[0] else None
        return token, lp

    sampling.reset_sampling_params = reset_params
    sampling.sample = sample
    args = SimpleNamespace(max_batch_size=32)
    generator = SimpleNamespace(
        already_warmed_up_prefill=True,
        warming_up_prefill=False,
        model_args=args,
        mesh_device=object(),
        model=SimpleNamespace(
            is_prefill_setup=True,
            sampling=sampling,
            tt_ccl=SimpleNamespace(support_seqlens=[]),
            switch_mode=Mock(),
            process_output_prefill_logits=lambda tensor, **kw: [tensor],
        ),
        tt_logits_accumulated=[torch.zeros(1, 1, 1, 1) for _ in range(32)],
        _slot_sampling_params={},
        prefill_forward_single_user_text=lambda tokens, **kw: tokens[0, 0].float().reshape(1, 1, 1, 1),
    )
    generator._remember_slot_params = MethodType(Generator._remember_slot_params, generator)
    monkeypatch.setattr(ttnn, "copy", lambda input_a, input_b: input_b.copy_(input_a))
    monkeypatch.setattr(ttnn, "concat", lambda tensors, dim, **kw: torch.cat(tensors, dim=dim))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda tensor: [tensor])
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor: tensor)
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: None)

    result = Generator.prefill_forward_text(
        generator,
        torch.tensor([[101, 1], [202, 2], [303, 3]]),
        kv_cache=[None],
        empty_slots=[31, 0, 15],
        enable_trace=False,
        sampling_params=SamplingParams(
            temperature=[0.8] * 3, top_k=[20] * 3, top_p=[0.9] * 3, seed=[7, 8, 9], enable_log_probs=flags
        ),
    )

    if any(flags):
        tokens, logprobs = result
        assert logprobs.shape == tokens.shape == (3,)
        for row, enabled in enumerate(flags):
            if enabled:
                assert logprobs[row].item() == pytest.approx(-(row + 1) * 1.01)
            else:
                assert torch.isnan(logprobs[row])
    else:
        tokens = result
        assert isinstance(tokens, torch.Tensor)
    assert tokens.tolist() == [101, 202, 303]


def test_completion_releases_tail_seed_without_resetting_survivor():
    manager = SeedManager(
        max_batch_size=32,
        salt_duplicate_seeds=False,
        seed_buffer=SimpleNamespace(source=torch.arange(32, dtype=torch.int64), update=Mock()),
    )
    manager.reset_seed([7, 9], [0, 31])
    manager.get_new_values([0, 31])
    survivor_rng = manager.rngs[0].getstate()
    generator = SimpleNamespace(
        model_args=SimpleNamespace(max_batch_size=32),
        model=SimpleNamespace(sampling=SimpleNamespace(seed_manager=manager)),
    )

    Generator.release_request(generator, 31)
    Generator.release_request(generator, 31)  # Completion is idempotent.
    assert manager.seeds[31] is None
    assert manager.seed_counters[31] == manager.seed_salts[31] == 0
    assert manager.seeds[0] == 7
    assert manager.seed_counters[0] == 1
    assert manager.rngs[0].getstate() == survivor_rng
    assert manager.has_active_request_seed()

    assert not manager._reseted
    Generator.release_request(generator, 0)
    # Before any decode reconciliation or replacement prefill, the last seed is gone.
    assert not manager._seed_active
    assert manager._reseted
    manager.get_new_values([0])
    assert not manager.has_active_request_seed()


@pytest.mark.parametrize("slot", [-1, 32])
def test_completion_rejects_invalid_slots(slot, expect_error):
    generator = SimpleNamespace(model_args=SimpleNamespace(max_batch_size=32))
    with expect_error(ValueError, "slot"):
        Generator.release_request(generator, slot)
