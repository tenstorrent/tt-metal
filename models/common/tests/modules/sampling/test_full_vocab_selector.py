import pytest
import torch

import models.common.sampling.generator as generator_module
from models.common.sampling.generator import SamplingGenerator, _make_full_vocab_selector


def test_full_vocabulary_capability_is_a_narrow_parameter_domain():
    domain = SamplingGenerator.full_vocabulary_sampling_capabilities()

    assert domain == {
        "abi": "tt-metal.common.sampling.full-vocabulary/v1",
        "sampling_dp": (1,),
        "top_p": (1.0,),
        "sampled_logprobs": False,
        "topk_logprobs": False,
        "traced": False,
        "mixed_rows": True,
    }


@pytest.mark.parametrize("batch_size", [1, 32])
def test_full_vocab_selector_has_persistent_buffer_shape(batch_size):
    selector = _make_full_vocab_selector(batch_size, {0, batch_size - 1})

    assert selector.shape == (1, 1, 1, batch_size)
    assert selector.dtype == torch.uint32
    assert selector[0, 0, 0, 0] == 1
    assert selector[0, 0, 0, batch_size - 1] == 1


def test_full_vocab_selector_marks_only_unrestricted_mixed_rows():
    selector = _make_full_vocab_selector(32, {8, 17, 31})

    assert selector.flatten().tolist() == [
        1 if slot in {8, 17, 31} else 0 for slot in range(32)
    ]


def test_selector_copy_failure_releases_native_output(monkeypatch):
    native_tokens = object()
    native_log_probs = (object(), object())

    class FakeSampling:
        max_batch_size = 32
        temp_tensor = object()

        def __call__(self, logits, tt_out_tok=None):
            return native_tokens, native_log_probs

    generator = SamplingGenerator.__new__(SamplingGenerator)
    generator.tt_sampling = FakeSampling()
    generator.seed_manager = type(
        "SeedManagerStub",
        (),
        {"next_managed_draw_seed_plan": lambda self, draws: type("Plan", (), {"seeds_by_subdraw": ([7] * 32,)})()},
    )()
    generator._full_vocab_selector = object()
    generator._full_vocab_invalid_tokens = object()
    released = []
    def collect(tensors, protect=()):
        for tensor in tensors:
            released.extend(tensor if isinstance(tensor, tuple) else [tensor])

    generator._deallocate_tensors = collect

    monkeypatch.setattr(generator_module.ttnn, "from_torch", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        generator_module.ttnn,
        "copy_host_to_device_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected copy failure")),
    )

    with pytest.raises(RuntimeError, match="injected copy failure"):
        generator._run_mixed_full_vocab_sampling(
            object(),
            contract=type("Contract", (), {"unrestricted_slots": (0,)})(),
            tt_out_tok=None,
        )

    assert released == [native_tokens, *native_log_probs]


def test_deallocate_tensors_flattens_optional_logprob_outputs(monkeypatch):
    class Allocation:
        def __init__(self):
            self.allocated = True

        def is_allocated(self):
            return self.allocated

    token = Allocation()
    logprob = Allocation()
    released = []

    def deallocate(tensor):
        released.append(tensor)
        tensor.allocated = False

    monkeypatch.setattr(generator_module.ttnn, "deallocate", deallocate)
    SamplingGenerator._deallocate_tensors([token, (logprob, None), logprob])

    assert released == [logprob, token]
