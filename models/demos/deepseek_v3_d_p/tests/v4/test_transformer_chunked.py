# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked-prefill PCC for TtV4Transformer: each prompt in 5120-token chunks against one unchunked CPU
pass of that prompt.

Random weights. Chunk c+1 of every layer attends over the state chunk c left, so the per-layer and
output PCC over the whole prompt grade the attention state carried across chunk boundaries, at model
level. Every chunk drives the same device shapes, whichever user slot it belongs to, so the program
cache must not grow after the first chunk.

With two user slots, requests interleave chunk by chunk as the runner sends them, and a third request
reuses slot 0 while slot 1's request is mid-prefill. Each request is graded against its own reference,
and the chunks that continue a carried state are graded again on their own.

That grading cannot see one user's state leaking into another's: with random weights and tokens, a
chunk that attends the wrong request's state entirely still lands within ~0.002 PCC of the reference,
far above the floors. Isolation is checked on device instead, where a rerun of the same chunks is
exact. The third request replays the first one's opening chunk on the reused slot, and once the
schedule is done the second request runs again alone, every other slot empty. Both must reproduce the
interleaved outputs to ``_REPLAY_FLOOR``, which a full leak misses on every layer of both models.
"""

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.model import (
    build_v4_model_reference,
    v4_model_forward,
    v4_model_state_dict,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.v4.helpers import (
    assert_layers_and_output,
    hidden_to_host,
    mesh_params,
    streams_to_host,
    upload_tokens,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Transformer
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

CHUNK = PREFILL_CHUNK_TOKENS
_PRO_FLOOR, _FLASH_FLOOR = 0.98, 0.99
_NUM_LAYERS = 3
_SEED = 7
# Device-to-device agreement for a rerun chunk: same weights, tokens and programs over the same state. Reruns
# measure 1.000000; a chunk that attends another request's state entirely measures 0.9996-0.9999 on
# Flash's sliding layers and lower everywhere else.
_REPLAY_FLOOR = 0.99999

# (slot, chunks, replays) per request: ``replays`` names an earlier request whose opening tokens this one
# repeats, or None for fresh tokens. The third request reuses slot 0 and is shorter than the one before
# it there, so the rows that request left past the new one's end are still in slot 0's index key cache
# and have to go unread.
_REQUESTS = {1: [(0, 2, None)], 2: [(0, 2, None), (1, 2, None), (0, 1, 0)]}
# (request, chunk) in the order the chunks reach the device.
_SCHEDULE = {1: [(0, 0), (0, 1)], 2: [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]}
# Requests run again on their own slot once the schedule is done, every other slot empty.
_ISOLATED = {1: [], 2: [1]}
# Every slot's state spans the longest request.
_MAX_CHUNKS = max(n for requests in _REQUESTS.values() for _, n, _ in requests)


def _assert_same_on_device(label, first, again):
    """Two device runs of the same chunks, each ``(per_layer, outs)``, agree to ``_REPLAY_FLOOR``."""
    (first_layers, first_outs), (again_layers, again_outs) = first, again
    for c in range(len(again_outs)):
        taps = [(f"layer {i}", first_layers[i][c], again_layers[i][c]) for i in range(len(again_layers))]
        for name, a, b in [*taps, ("output", first_outs[c], again_outs[c])]:
            _, pcc = comp_pcc(a.float(), b.float())
            logger.info(f"[{label}] chunk {c} {name}: PCC {pcc:.6f}")
            assert pcc >= _REPLAY_FLOOR, f"[{label}] chunk {c} {name} {pcc:.6f} < {_REPLAY_FLOOR}"


def run_transformer_chunked(mesh_device, device_params, num_links, num_users, model_config, floor):
    requests, schedule = _REQUESTS[num_users], _SCHEDULE[num_users]
    config = v4_hf_config(model_config, _NUM_LAYERS, max_seq_len=_MAX_CHUNKS * CHUNK)
    ref = build_v4_model_reference(config, seed=_SEED)
    input_ids, refs = [], []
    for _, n, replays in requests:
        if replays is None:
            input_ids.append(torch.randint(0, config.vocab_size, (1, n * CHUNK)))
            refs.append(v4_model_forward(ref, config, input_ids[-1]))
        else:
            # The reference is causal, so a prefix's pass is the prefix of the full one.
            length = n * CHUNK
            src_out, src_layers = refs[replays]
            input_ids.append(input_ids[replays][:, :length])
            refs.append((src_out[:, :length], [h[:, :length] for h in src_layers]))

    # The reference's bf16 experts (~50 GB per Pro layer) are shared with the state dict, not copied;
    # dropping the reference frees the rest of it.
    state_dict = v4_model_state_dict(ref, config)
    del ref
    model = TtV4Transformer(
        mesh_device,
        config,
        model_config,
        state_dict,
        _NUM_LAYERS,
        CHUNK,
        max_seq_len=_MAX_CHUNKS * CHUNK,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        slot_num=num_users,
    )
    del state_dict

    def prefill(req, chunk, per_layer, outs):
        """One chunk of request ``req`` on its slot, its layer taps and output appended to the sinks."""
        slot, start = requests[req][0], chunk * CHUNK
        if chunk == 0:
            # What TtV4Runtime.prefill_chunk does at the head of a request.
            model.reset_streams(slot)
        out = model(
            upload_tokens(mesh_device, input_ids[req][:, start : start + CHUNK]),
            actual_isl=CHUNK,
            actual_start=start,
            cache_user_id=slot,
            layer_tap=lambda i, h: per_layer[i].append(streams_to_host(mesh_device, h, config.hc_mult)),
        )
        outs.append(hidden_to_host(mesh_device, out))
        logger.info(f"  request {req} chunk {chunk} done (slot={slot}, start={start})")

    per_layer = [[[] for _ in range(_NUM_LAYERS)] for _ in requests]
    outs = [[] for _ in requests]
    mesh_device.enable_program_cache()
    programs = None
    for req, chunk in schedule:
        prefill(req, chunk, per_layer[req], outs[req])
        now = mesh_device.num_program_cache_entries()
        assert programs in (
            None,
            now,
        ), f"request {req} chunk {chunk} compiled {now - programs} new program(s) ({programs} -> {now})"
        programs = now

    isolated = {}
    for req in _ISOLATED[num_users]:
        for slot in range(num_users):
            model.reset_streams(slot)
        isolated[req] = ([[] for _ in range(_NUM_LAYERS)], [])
        for chunk in range(requests[req][1]):
            prefill(req, chunk, *isolated[req])
    assert mesh_device.num_program_cache_entries() == programs, "the isolated reruns compiled new programs"

    for req, ((slot, n_chunks, replays), (ref_out, ref_layers)) in enumerate(zip(requests, refs)):
        label = f"{model_config.__name__} L{_NUM_LAYERS} request {req} (slot {slot}, x{n_chunks} chunks)"
        assert_layers_and_output(
            config,
            [torch.cat(chunks, dim=1) for chunks in per_layer[req]],
            ref_layers,
            torch.cat(outs[req], dim=1),
            ref_out,
            floor,
            label,
        )
        if n_chunks > 1:
            assert_layers_and_output(
                config,
                [torch.cat(chunks[1:], dim=1) for chunks in per_layer[req]],
                [h[:, CHUNK:] for h in ref_layers],
                torch.cat(outs[req][1:], dim=1),
                ref_out[:, CHUNK:],
                floor,
                f"{label} carried chunks",
            )
        if replays is not None:
            _assert_same_on_device(
                f"{label} vs request {replays} on device",
                (per_layer[replays], outs[replays]),
                (per_layer[req], outs[req]),
            )
        if req in isolated:
            _assert_same_on_device(
                f"{label} interleaved vs alone on device", (per_layer[req], outs[req]), isolated[req]
            )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_users", [1, 2], ids=["users1", "users2"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_transformer_chunked(mesh_device, device_params, num_links, num_users):
    run_transformer_chunked(mesh_device, device_params, num_links, num_users, DeepSeekV4ProConfig, _PRO_FLOOR)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4FlashConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_users", [1, 2], ids=["users1", "users2"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_transformer_chunked(mesh_device, device_params, num_links, num_users):
    run_transformer_chunked(mesh_device, device_params, num_links, num_users, DeepSeekV4FlashConfig, _FLASH_FLOOR)
