import pytest
import torch
from types import SimpleNamespace

import models.common.sampling.generator as generator_module
from models.common.sampling.generator import SamplingGenerator, _make_full_vocab_selector
import models.common.sampling.tt_sampling as tt_sampling_module
from models.common.sampling.tt_sampling import TTSampling


def test_full_vocabulary_capability_is_a_narrow_parameter_domain():
    domain = SamplingGenerator.full_vocabulary_sampling_capabilities(vocab_size=201088)

    assert domain == {
        "abi": "tt-metal.common.sampling.full-vocabulary/v1",
        "sampling_dp": (1,),
        "mesh_shapes": ((1, 4),),
        "top_p_one": True,
        "max_nucleus_top_p": pytest.approx(2048 / 201088),
        "max_nucleus_candidates": 2048,
        "stable_topk_max_local_width": 65504,
        "sampled_logprobs": True,
        "max_top_logprobs": 0,
        "topk_logprobs": False,
        "traced": False,
        "mixed_rows": True,
    }


def test_full_vocabulary_capability_bound_is_conservative_and_vocab_derived():
    domain = SamplingGenerator.full_vocabulary_sampling_capabilities(vocab_size=4096)

    assert domain["max_nucleus_top_p"] < 0.5
    assert domain["max_nucleus_top_p"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="positive integer"):
        SamplingGenerator.full_vocabulary_sampling_capabilities(vocab_size=0)

    non_tile_vocab = SamplingGenerator.full_vocabulary_sampling_capabilities(vocab_size=1001)
    assert non_tile_vocab["max_nucleus_top_p"] < 992 / 1001
    assert non_tile_vocab["max_nucleus_top_p"] == pytest.approx(992 / 1001)


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
        vocab_size = 128

        def gather_full_vocab_logits(self, logits):
            return logits, ()

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
    generator._full_vocab_inverse_temperature = object()
    generator._full_vocab_top_p = object()
    generator._full_vocab_row_scratch = ()
    generator._full_vocab_nucleus_max_candidates = 32
    generator._full_vocab_stable_topk_max_local_width = 0
    released = []
    def collect(tensors, protect=()):
        for tensor in tensors:
            released.extend(tensor if isinstance(tensor, tuple) else [tensor])

    generator._deallocate_tensors = collect

    monkeypatch.setattr(generator_module.ttnn, "from_torch", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        generator_module,
        "sample_unrestricted_top_p_one",
        lambda *args, **kwargs: SimpleNamespace(owned_tensors=()),
    )
    monkeypatch.setattr(
        generator_module.ttnn,
        "copy_host_to_device_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected copy failure")),
    )

    with pytest.raises(RuntimeError, match="injected copy failure"):
        generator._run_mixed_full_vocab_sampling(
            object(),
            contract=type(
                "Contract",
                (),
                {"unrestricted_slots": (0,), "unrestricted_nucleus_slots": ()},
            )(),
            tt_out_tok=None,
        )

    # Native logprob outputs are persistent calculator-owned buffers; an
    # unrelated selector failure must not deallocate them.
    assert released == [native_tokens]


def test_bounded_penalty_sampling_uses_pre_penalty_raw_snapshot_for_sampled_logprob():
    original_logits = SimpleNamespace(value=7)
    selected_tokens = object()
    native_logprob = object()
    raw_logprob = object()
    released = []
    observed = []

    generator = SamplingGenerator.__new__(SamplingGenerator)
    generator._full_vocab_contract = lambda: SimpleNamespace(needs_full_vocabulary=False)
    generator._raw_sampled_logprob_slots = lambda: (0,)
    generator._copy_warmup_logits = lambda logits: SimpleNamespace(value=logits.value)
    generator.tt_penalties = SimpleNamespace(
        apply=lambda logits: (setattr(logits, "value", 99) or logits),
        update_output_tokens=lambda tokens: None,
    )
    generator.tt_sampling = lambda logits, tt_out_tok=None: (selected_tokens, native_logprob)

    def calculate(raw_logits, tokens):
        observed.append((raw_logits.value, original_logits.value, tokens))
        return raw_logprob

    generator._calculate_raw_sampled_logprobs = calculate
    generator._deallocate_tensors = lambda tensors, protect=(): released.extend(tensors)

    tokens, logprob = generator._run_sampling(
        original_logits,
        penalties_on=True,
        tt_out_tok=None,
        count_tokens=False,
    )

    assert tokens is selected_tokens and logprob is raw_logprob
    assert observed == [(7, 99, selected_tokens)]
    assert len(released) == 1 and released[0].value == 7


def test_penalty_failure_releases_pre_penalty_raw_snapshot():
    original_logits = SimpleNamespace(value=7)
    snapshot = SimpleNamespace(value=7)
    released = []

    generator = SamplingGenerator.__new__(SamplingGenerator)
    generator._full_vocab_contract = lambda: SimpleNamespace(needs_full_vocabulary=False)
    generator._raw_sampled_logprob_slots = lambda: (0,)
    generator._copy_warmup_logits = lambda logits: snapshot
    generator.tt_penalties = SimpleNamespace(
        apply=lambda logits: (_ for _ in ()).throw(RuntimeError("injected penalty failure"))
    )
    generator._deallocate_tensors = lambda tensors, protect=(): released.extend(tensors)

    with pytest.raises(RuntimeError, match="injected penalty failure"):
        generator._run_sampling(original_logits, penalties_on=True, tt_out_tok=None)

    assert released == [snapshot]


def test_mixed_top_p_one_and_nucleus_rows_merge_sequentially(monkeypatch):
    native_tokens = object()
    p1_tokens = object()
    final_tokens = object()
    full_logits = object()
    selector_copies = []
    merge_inputs = []
    group_seed_values = []

    class FakeSampling:
        max_batch_size = 4
        temp_tensor = object()
        vocab_size = 64

        def __call__(self, logits, tt_out_tok=None):
            return native_tokens, None

        def gather_full_vocab_logits(self, logits):
            return full_logits, ()

    generator = SamplingGenerator.__new__(SamplingGenerator)
    generator.tt_sampling = FakeSampling()
    generator.seed_manager = SimpleNamespace(
        next_managed_draw_seed_plan=lambda draws: SimpleNamespace(
            seeds_by_subdraw=((11, 22, 2**32 - 1, 2**32 - 1),)
        )
    )
    generator._full_vocab_selector = object()
    generator._full_vocab_invalid_tokens = object()
    generator._full_vocab_inverse_temperature = object()
    generator._full_vocab_top_p = object()
    generator._full_vocab_row_scratch = ()
    generator._full_vocab_nucleus_max_candidates = 32
    generator._full_vocab_stable_topk_max_local_width = 64
    generator._deallocate_tensors = lambda *args, **kwargs: None

    p1_categorical = SimpleNamespace(owned_tensors=())
    nucleus_categorical = SimpleNamespace(owned_tensors=())
    def sample_p1(*args, **kwargs):
        group_seed_values.append(tuple(kwargs["seed_values"]))
        return p1_categorical

    def sample_nucleus(*args, **kwargs):
        group_seed_values.append(tuple(kwargs["seed_values"]))
        return nucleus_categorical

    monkeypatch.setattr(generator_module, "sample_unrestricted_top_p_one", sample_p1)
    monkeypatch.setattr(generator_module, "sample_unrestricted_nucleus", sample_nucleus)
    monkeypatch.setattr(generator_module.ttnn, "from_torch", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(
        generator_module.ttnn,
        "copy_host_to_device_tensor",
        lambda update, target: selector_copies.append(update.flatten().tolist()),
    )

    def merge(categorical, current_tokens, **kwargs):
        merge_inputs.append((categorical, current_tokens))
        return SimpleNamespace(
            token_ids=p1_tokens if categorical is p1_categorical else final_tokens,
            owned_tensors=(),
        )

    monkeypatch.setattr(generator_module, "merge_unrestricted_rows", merge)
    contract = SimpleNamespace(
        unrestricted_slots=(0, 1),
        unrestricted_nucleus_slots=(1,),
        rows=(
            SimpleNamespace(top_p=1.0),
            SimpleNamespace(top_p=0.49),
            SimpleNamespace(top_p=1.0),
            SimpleNamespace(top_p=1.0),
        ),
    )

    tokens, logprobs = generator._run_mixed_full_vocab_sampling(
        object(), contract=contract, tt_out_tok=None
    )

    assert tokens is final_tokens and logprobs is None
    assert merge_inputs == [(p1_categorical, native_tokens), (nucleus_categorical, p1_tokens)]
    assert selector_copies == [[1, 0, 0, 0], [0, 1, 0, 0]]
    assert group_seed_values == [
        (11, 2**32 - 1, 2**32 - 1, 2**32 - 1),
        (2**32 - 1, 22, 2**32 - 1, 2**32 - 1),
    ]


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


class _Allocation:
    def __init__(self, width):
        self.shape = (1, 1, 1, width)
        self.allocated = True

    def is_allocated(self):
        return self.allocated


@pytest.mark.parametrize("failure", ["gather", "shape"])
def test_full_vocab_gather_releases_local_intermediates_on_failure(monkeypatch, failure):
    borrowed = _Allocation(16)
    masked = _Allocation(16)
    gathered = _Allocation(15 if failure == "shape" else 16)
    released = []

    sampling = TTSampling.__new__(TTSampling)
    sampling._line_all_gather = None
    sampling.mesh_device = type("Mesh", (), {"get_num_devices": lambda self: 4})()
    sampling.padded_vocab_size = 16
    sampling.num_gather_links = 1
    sampling._mask_invalid_vocab_logits = lambda logits: masked
    sampling._get_sampling_cluster_axis = lambda: None

    def gather(*args, **kwargs):
        if failure == "gather":
            raise RuntimeError("injected gather failure")
        return gathered

    sampling._perform_all_gather = gather

    def deallocate(tensor):
        released.append(tensor)
        tensor.allocated = False

    monkeypatch.setattr(tt_sampling_module.ttnn, "deallocate", deallocate)

    match = "injected gather failure" if failure == "gather" else "produced width"
    with pytest.raises(RuntimeError, match=match):
        sampling.gather_full_vocab_logits(borrowed)

    assert borrowed.is_allocated()
    assert released == ([masked] if failure == "gather" else [gathered, masked])
