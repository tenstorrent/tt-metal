# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two users through the prefill runner's path on one rank: the V4 adapter builds the runtime with the
parameters prefill_runner derives, and every chunk goes through ``TtV4Runtime.prefill_chunk`` with the
runner's arguments, so the runtime alone decides which slot's attention state to reset.

The schedule is test_transformer_chunked's two-user one: two requests interleaved chunk by chunk, a third
reusing slot 0 while slot 1's request is mid-prefill, then the slot 1 request again with nothing between its
chunks. Real weights for the first few layers of the checkpoint, random tokens.

Graded device to device only, for the reason test_transformer_chunked gives: a chunk that reads another
slot's state still lands near the CPU reference, while a device rerun of the same chunks is exact. The
reused slot must reproduce the first request's opening chunk, and the interleaved slot 1 request must
reproduce its uninterrupted rerun. Accuracy against a reference is the chunked and depth tests' job.

On one rank ``prefill_chunk`` returns None and discards the model's output, so the test wraps the model's
``forward``, which the runtime calls through the instance, to read each layer's streams and the output.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.v4.helpers import (
    USER_ISOLATED,
    USER_REQUESTS,
    USER_SCHEDULE,
    assert_same_on_device,
    hidden_to_host,
    mesh_params,
    streams_to_host,
)
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

CHUNK = PREFILL_CHUNK_TOKENS
_NUM_USERS = 2
_SEED = 7
_REQUESTS = USER_REQUESTS[_NUM_USERS]
_SCHEDULE = USER_SCHEDULE[_NUM_USERS]
_ISOLATED = USER_ISOLATED[_NUM_USERS]
_MAX_SEQ_LEN = max(n for _, n, _ in _REQUESTS) * CHUNK


class _AckRecorder:
    """The runner's host layer-completion sink (PREFILL_LAYER_ACK_D2H=0), keeping the records."""

    def __init__(self):
        self.records = []

    def layers_completed(self, layer_start, layer_end, slot_id, pos_start, pos_end) -> None:
        self.records.append((layer_start, layer_end, slot_id, pos_start, pos_end))


def _build(mesh_device, adapter, num_layers, num_links):
    """The runtime and its (empty) KV handle as prefill_runner.main builds them on a single rank."""
    hf_config = adapter.load_hf_config()
    hf_config.max_seq_len = _MAX_SEQ_LEN
    params = PrefillRunParams(
        mesh_shape=tuple(mesh_device.shape),
        num_layers=num_layers,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=_MAX_SEQ_LEN,
        chunk_size=CHUNK,
        num_users=_NUM_USERS,
        capacity_factor=8,
        num_links=num_links,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path(tuple(mesh_device.shape)),
    )
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    kv_caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    return runtime, kv_caches


def _tap_forward(runtime, mesh_device, sink):
    """Every model forward appends ``(per-layer streams, output)`` on host to ``sink``."""
    forward = runtime.model.forward
    hc_mult = runtime.hf_config.hc_mult

    def tapped(*args, **kwargs):
        layers = []
        out = forward(*args, layer_tap=lambda i, h: layers.append(streams_to_host(mesh_device, h, hc_mult)), **kwargs)
        sink.append((layers, hidden_to_host(mesh_device, out)))
        ttnn.deallocate(out)
        return out

    runtime.model.forward = tapped


def run_runtime_two_users(mesh_device, num_links, adapter_name, num_layers):
    adapter = get_adapter(adapter_name)
    runtime, kv_caches = _build(mesh_device, adapter, num_layers, num_links)
    gen = torch.Generator().manual_seed(_SEED)
    input_ids = []
    for _, n, replays in _REQUESTS:
        if replays is None:
            input_ids.append(torch.randint(0, runtime.hf_config.vocab_size, (1, n * CHUNK), generator=gen))
        else:
            input_ids.append(input_ids[replays][:, : n * CHUNK])

    try:
        runtime.compile(kv_caches)
        acks = _AckRecorder()
        runtime.set_layer_completion_sink(acks)
        taps = []
        _tap_forward(runtime, mesh_device, taps)
        programs = mesh_device.num_program_cache_entries()
        expected_acks = []

        def prefill(req, chunk, per_layer, outs):
            """One chunk of request ``req`` as the runner hands it to the runtime."""
            slot, start = _REQUESTS[req][0], chunk * CHUNK
            ids = input_ids[req][0, start : start + CHUNK].tolist()
            ret = runtime.prefill_chunk(
                runtime.make_chunk_input(ids),
                kv_caches,
                slot_id=slot,
                actual_start=start,
                actual_end=start + CHUNK,
                request_id=len(expected_acks),
                d2h_service=None,
                metadata_msg=None,
            )
            assert ret is None, "a single-rank runtime returns no activation"
            expected_acks.append([(i, i + 1, slot, start, start + CHUNK) for i in range(num_layers)])
            layers, out = taps.pop()
            assert not taps, "one prefill_chunk ran more than one model forward"
            assert torch.isfinite(out).all() and out.abs().max() > 0, f"request {req} chunk {chunk} output degenerate"
            for i, h in enumerate(layers):
                per_layer[i].append(h)
            outs.append(out)
            now = mesh_device.num_program_cache_entries()
            assert now == programs, f"request {req} chunk {chunk} compiled {now - programs} new program(s)"
            logger.info(f"  request {req} chunk {chunk} done (slot={slot}, start={start})")

        per_layer = [[[] for _ in range(num_layers)] for _ in _REQUESTS]
        outs = [[] for _ in _REQUESTS]
        for req, chunk in _SCHEDULE:
            prefill(req, chunk, per_layer[req], outs[req])

        isolated = {}
        for req in _ISOLATED:
            # The runtime resets the request's own slot at its head; the others are emptied here.
            slot = _REQUESTS[req][0]
            for other in range(_NUM_USERS):
                if other != slot:
                    runtime.model.reset_streams(other)
            isolated[req] = ([[] for _ in range(num_layers)], [])
            for chunk in range(_REQUESTS[req][1]):
                prefill(req, chunk, *isolated[req])

        assert acks.records == [a for chunk_acks in expected_acks for a in chunk_acks], "layer acks out of contract"

        name = f"{adapter.model_config.__name__} L{num_layers} runtime"
        for req, (slot, n_chunks, replays) in enumerate(_REQUESTS):
            label = f"{name} request {req} (slot {slot}, x{n_chunks} chunks)"
            if replays is not None:
                assert_same_on_device(
                    f"{label} vs request {replays} on device",
                    (per_layer[replays], outs[replays]),
                    (per_layer[req], outs[req]),
                )
            if req in isolated:
                assert_same_on_device(
                    f"{label} interleaved vs alone on device", (per_layer[req], outs[req]), isolated[req]
                )
    finally:
        runtime.release_trace()


def _galaxy(payload):
    return [p for p in mesh_params(payload) if p.id == "torus-xy-8x4"]


# Flash's first four layers hold every attention kind (sliding, CSA, HCA) and both gates (hash, top-k);
# Pro's first three hold both of its kinds (HCA, CSA).
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _galaxy(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_runtime_two_users(mesh_device, device_params, num_links):
    run_runtime_two_users(mesh_device, num_links, "deepseek_v4_pro", num_layers=3)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _galaxy(DeepSeekV4FlashConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_runtime_two_users(mesh_device, device_params, num_links):
    run_runtime_two_users(mesh_device, num_links, "deepseek_v4_flash", num_layers=4)
