"""Host-only translation checks for the plugin DECODE_RELOAD_CONTRACT.

No model is constructed. Device execution/replay correctness requires the separate
serving tests; these tests catch loss of scheduler authority at the adapter boundary.
"""

from types import SimpleNamespace

import pytest
import torch

from models.autoports.qwen_qwen3_8_flash_next.tt import generator_vllm as adapter_module


class RecordingGenerator:
    def __init__(self, batch_size):
        self.calls = []
        self.parameter_updates = []
        self.counters = {"token_readbacks": 0}
        self.device_output = object()
        self.prefill_output = torch.arange(batch_size).reshape(batch_size, 1)
        self._active = [True] * batch_size
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def set_serving_sampling_params(self, params):
        self.parameter_updates.append(params)
        return "greedy"

    def decode_forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if kwargs.get("active_rows") is not None:
            self._active = list(kwargs["active_rows"])
        return self.device_output

    def prefill_forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.prefill_output


@pytest.fixture
def adapter(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A host adapter test attempted to open a device")

    for name in ("open_device", "open_mesh_device"):
        monkeypatch.setattr(adapter_module.ttnn, name, forbidden, raising=False)
    obj = adapter_module.QwenFlashNextForCausalLM()
    obj.batch_size = 3
    obj.context = 1024
    obj.mesh = object()
    obj.slot_map = [0, 1, 2]
    obj.allow_host_sampling = False
    obj.generator = RecordingGenerator(obj.batch_size)
    obj.kv_cache = object()
    return obj


def decode(adapter, **overrides):
    kwargs = dict(
        tokens=torch.tensor([[101], [202], [303]], dtype=torch.int32),
        start_pos=torch.tensor([31, 63, -1], dtype=torch.int32),
        page_table=torch.tensor([[5, 9, 11], [7, 8, 12], [0, 0, 0]], dtype=torch.int32),
        kv_cache=adapter.kv_cache,
        sampling_params=object(),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        read_from_device=False,
    )
    kwargs.update(overrides)
    return adapter.decode_forward(**kwargs)


@pytest.mark.parametrize("page_only", [False, True])
def test_resident_decode_preserves_token_position_and_cache_identity(adapter, page_only):
    decode(adapter)
    updates = len(adapter.generator.parameter_updates)
    result = decode(adapter, reload_inputs=False, reload_page_table=page_only)
    args, kwargs = adapter.generator.calls[-1]
    assert args == (None, None)
    assert kwargs["kv_cache"] is adapter.kv_cache
    assert kwargs["slots"] is None
    assert kwargs["active_rows"] is None
    assert kwargs["read_tokens"] is False
    assert (kwargs["page_table"] is not None) == page_only
    assert len(adapter.generator.parameter_updates) == updates
    assert result is adapter.generator.device_output


@pytest.mark.parametrize("page_only", [False, True])
def test_resident_decode_does_not_inspect_stale_host_tokens_or_positions(adapter, page_only):
    decode(adapter)
    # The plugin supplies stale tensors. Unconvertible sentinels make any read
    # observable, establishing that authority comes from the explicit command.
    decode(adapter, tokens=object(), start_pos=object(), reload_inputs=False, reload_page_table=page_only)
    assert adapter.generator.calls[-1][0] == (None, None)


def test_authoritative_reload_passes_logical_positions_and_active_rows(adapter):
    tokens = torch.tensor([[19], [23], [0]], dtype=torch.int32)
    decode(adapter, tokens=tokens)
    args, kwargs = adapter.generator.calls[-1]
    assert args[0] is tokens
    assert args[1].tolist() == [31, 63, 0]
    assert kwargs["active_rows"] == [True, True, False]
    assert kwargs["slots"] == [0, 1, 2]
    assert kwargs["page_table"][:, :3].tolist() == [[5, 9, 11], [7, 8, 12], [0, 0, 0]]


@pytest.mark.parametrize("legacy_value", [False, True])
def test_legacy_reset_batch_rejected_before_submission(adapter, legacy_value):
    with pytest.raises(ValueError, match="contract"):  # allow-pytest.raises: host-only contract error, no device log
        decode(adapter, reset_batch=legacy_value)
    assert not adapter.generator.calls


@pytest.mark.parametrize(
    "overrides",
    [
        {"reload_inputs": True, "reload_page_table": True},
        {"reload_inputs": False, "reset_sampling_state": True},
        {"enable_trace": False},
        {"rope_deltas_all_users": [0, 1, 0]},
        {"slot_remap": [1, 1, 0]},
    ],
)
def test_invalid_decode_commands_rejected_without_submission(adapter, overrides):
    with pytest.raises(ValueError):  # allow-pytest.raises: host-only contract error, no device log
        decode(adapter, **overrides)
    assert not adapter.generator.calls


def test_remap_composes_once_and_prefill_uses_current_scheduler_slots(adapter):
    decode(adapter, slot_remap=[2, 0, 1])
    assert adapter.slot_map == [2, 0, 1]
    assert adapter.generator.calls[-1][1]["slots"] == [2, 0, 1]
    decode(adapter, reload_inputs=False)
    assert adapter.slot_map == [2, 0, 1]
    assert adapter.generator.calls[-1][1]["slots"] is None
    decode(adapter, slot_remap=[1, 0, 2])
    assert adapter.slot_map == [0, 2, 1]
    # Plugin prompt_lens are absolute chunk ends; input_tokens retains each
    # full prefix through the largest end, even for a continuation submission.
    tokens = torch.arange(2 * 644, dtype=torch.int32).reshape(2, 644)
    output, deltas = adapter.prefill_forward(
        tokens,
        torch.tensor([[8, 12], [9, 13]], dtype=torch.int32),
        adapter.kv_cache,
        prompt_lens=[386, 644],
        empty_slots=[1, 2],
        start_pos=[257, 513],
        sampling_params=object(),
    )
    args, kwargs = adapter.generator.calls[-1]
    assert args[0].shape == (3, 131)
    assert torch.equal(args[0][0, :129], tokens[0, 257:386])
    assert torch.count_nonzero(args[0][0, 129:]) == 0
    assert torch.equal(args[0][1], tokens[1, 513:644])
    assert torch.count_nonzero(args[0][2]) == 0
    assert kwargs["prompt_lens"] == [129, 131, 0]
    assert kwargs["start_pos"].tolist() == [257, 513, 0]
    assert kwargs["slots"] == [2, 1, 0]
    assert kwargs["kv_cache"] is adapter.kv_cache
    assert output.shape == (2, 1)
    assert deltas.tolist() == [0, 0]
    assert adapter.slot_map == [0, 2, 1]


@pytest.mark.parametrize("start,end", [(-1, 33), (34, 33), (0, -1), (0, 132)])
def test_prefill_invalid_absolute_bounds_rejected_before_submission(adapter, start, end):
    with pytest.raises(ValueError):  # allow-pytest.raises: host-only contract error, no device log
        adapter.prefill_forward(
            torch.arange(131, dtype=torch.int32).reshape(1, 131),
            torch.tensor([[8, 12, 13, 14, 15]], dtype=torch.int32),
            adapter.kv_cache,
            prompt_lens=[end],
            start_pos=[start],
            sampling_params=object(),
        )
    assert not adapter.generator.calls


def test_rejected_submission_does_not_consume_state_remap(adapter, monkeypatch):
    def rejected(*args, **kwargs):
        raise RuntimeError("Rejected before device submission")

    monkeypatch.setattr(adapter.generator, "decode_forward", rejected)
    with pytest.raises(RuntimeError, match="Rejected before"):  # allow-pytest.raises: mocked submission error
        decode(adapter, slot_remap=[2, 0, 1])
    assert adapter.slot_map == [0, 1, 2]


def test_async_minimal_read_returns_same_submission_without_forward_mutation(adapter, monkeypatch):
    submission = decode(adapter)
    forward_count = len(adapter.generator.calls)
    copied = object()
    event = object()
    reads = []

    def cpu(*, blocking):
        reads.append(blocking)
        return copied

    def get_device_tensors(value):
        if value is copied:
            return [copied]
        assert value is submission
        return [SimpleNamespace(cpu=cpu)]

    def record_event(mesh, cq_id):
        assert mesh is adapter.mesh and cq_id == 0
        return event

    monkeypatch.setattr(adapter_module.ttnn, "get_device_tensors", get_device_tensors)
    monkeypatch.setattr(adapter_module.ttnn, "record_event", record_event)
    monkeypatch.setattr(
        adapter_module.ttnn, "to_torch", lambda value: torch.tensor([11, 22, 33]) if value is copied else None
    )
    host, events = adapter.read_decode_output(submission, async_read=True)
    assert host is copied and events == [event]
    assert reads == [False]
    assert adapter.generator.counters["token_readbacks"] == 1
    assert adapter.process_decode_output_host(host, is_tokens=True).tolist() == [[11], [22], [33]]
    assert len(adapter.generator.calls) == forward_count


def test_synchronous_replicated_device_output_formats_exactly_one_shard(adapter, monkeypatch):
    submission = decode(adapter)
    replicas = [object() for _ in range(4)]
    conversions = []

    def get_device_tensors(value):
        assert value is submission
        return replicas

    def to_torch(value):
        # A replicated tensor cannot be passed directly to to_torch. Using all
        # replicas would duplicate user rows and break the plugin output shape.
        assert value is replicas[0]
        conversions.append(value)
        return torch.tensor([[[[41, 42, 43]]]], dtype=torch.int32)

    monkeypatch.setattr(adapter_module.ttnn, "get_device_tensors", get_device_tensors)
    monkeypatch.setattr(adapter_module.ttnn, "to_torch", to_torch)
    output = adapter.process_decode_output_host(submission, is_tokens=True)
    assert output.tolist() == [[41], [42], [43]]
    assert output.dtype == torch.long
    assert conversions == [replicas[0]]
    assert adapter.generator.counters["token_readbacks"] == 0
    assert len(adapter.generator.calls) == 1


def test_host_sampling_is_explicit_and_host_output_identity_preserved(adapter):
    with pytest.raises(  # allow-pytest.raises: host-only policy error
        ValueError, match="QWEN_FLASH_ALLOW_HOST_SAMPLING"
    ):
        decode(adapter, sampling_params=None)
    host = torch.tensor([[7], [8], [9]])
    assert adapter.process_decode_output_host(host) is host
    async_host, events = adapter.read_decode_output(host, async_read=True)
    assert async_host is host and events == []
