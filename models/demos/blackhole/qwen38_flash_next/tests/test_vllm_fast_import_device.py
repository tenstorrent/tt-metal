# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3c device test: the device-direct lane import against the host-pager import, byte for byte.

``Qwen38PlainLaneDecoder.move_into_lane`` imports the B=1 chain's generic state into a lane two ways: the device-direct
``Qwen38TTNNLaneImporter`` with no keep-mask reset (the default) and the P1/P2 host-pager path (evict, rotate the ring
slots, ``admit_lane`` with its reset, readmit; ``QWEN38_IMPORT_HOST=1``).  This test builds the vLLM adapter at B=8 in
batch mode, fills three other lanes with live prefills and the target lane with a stale one, prefills a prompt of
length ``1024 + p`` on the B=1 chain (ring residue ``p``), then imports it into the target lane by BOTH paths from the
same lane state (the host snapshot of every lane tensor is restored in between; a control import at a wrong ring phase must differ in the ring
slots only) and compares every lane-state family
(12 compressed caches, 12 staging tiles, 12 raw-key rings, 36 recurrent states, 144 ring slots, 9 PLE slots: all 48
layers, all 4 devices) byte for byte, the other lanes against the state before the import (unchanged), the position
row and the PLE contexts.  The KV pages are written by the same ``kv_page_scatter`` in both paths and are not compared.

Ids: ``p{0..3}_lane{0,7}`` (one case per process), ``all_cases`` (the eight in one process).

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_vllm_fast_import_device.py -v -s -k p1_lane7
Env:  as ``test_vllm_batch_contract_device.py``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_prompt_ids, load_context_text
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    PROMPT_FILE,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_contract_device import (
    MAX_SEQ_LEN,
    initialize,
    plugin_num_blocks,
    plugin_warmups,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import page_table_rows
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import greedy_batch_sampling_params
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import LANE_AXIS

LANES = 8
BLOCK = 64
TARGET_BASE = 1024
FILLER_LENS = (300, 777, 1500)  # live lanes: prompts of other lengths (and ring residues 0, 1, 3 -> 300 % 4 == 0)
STALE_LEN = 450
CASES = [(p, lane) for lane in (0, LANES - 1) for p in range(4)]
FAMILIES = ("compressed", "staging", "ring", "recurrent", "conv", "ple")


def build_prompts(checkpoint: Path, lengths):
    template = Qwen38OfficialChatTemplate(checkpoint)
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context")
    render = lambda content: template.render(
        [{"role": "user", "content": content}], enable_thinking=False
    ).text  # noqa: E731
    return {n: build_prompt_ids(template.tokenizer, render, config, context_text, n)[0] for n in sorted(set(lengths))}


def lane_images(layout):
    """Host copies of every lane tensor by family (the exact device bytes)."""

    return {name: [ttnn.from_device(t) for t in getattr(layout, name)] for name in FAMILIES}


def as_bits(host_tensor):
    """Per device: the tensor's logical contents as integer bit patterns (``-0.0`` and ``+0.0`` differ)."""

    images = []
    for device_tensor in ttnn.get_device_tensors(host_tensor):
        torch_tensor = ttnn.to_torch(device_tensor)
        images.append(torch_tensor.view(torch.int32 if torch_tensor.dtype == torch.float32 else torch.int16))
    return images


def restore(layout, images):
    for name in FAMILIES:
        for tensor, host in zip(getattr(layout, name), images[name]):
            ttnn.copy_host_to_device_tensor(host, tensor)


def family_differences(a, b, lane=None, *, only_other_lanes=False):
    """Tensors (family, index, device) whose bits differ between two images: the whole tensor, the lane's slice only
    (``lane``), or everything but the lane's slice (``only_other_lanes``)."""

    bad = []
    for name in FAMILIES:
        axis = LANE_AXIS[name]
        for index, (ha, hb) in enumerate(zip(a[name], b[name])):
            for device, (xa, xb) in enumerate(zip(as_bits(ha), as_bits(hb))):
                if only_other_lanes:
                    keep = [u for u in range(xa.shape[axis]) if u != lane]
                    xa, xb = xa.index_select(axis, torch.tensor(keep)), xb.index_select(axis, torch.tensor(keep))
                if not torch.equal(xa, xb):
                    bad.append((name, index, device))
    return bad


def run_case(model, prompts, lane: int, p: int):
    decoder, lanes_state = model.decoder, model.decoder.ls
    layout = decoder.pager_l.layout
    sampling = greedy_batch_sampling_params(1)
    width = -(-MAX_SEQ_LEN // BLOCK)
    pages_of = {}
    next_page = 1

    def pages_for(length):
        nonlocal next_page
        count = -(-length // BLOCK)
        pages = list(range(next_page, next_page + count))
        next_page += count
        return pages

    def import_by_prefill(target_lane, ids):
        pages = pages_for(len(ids))
        pages_of[target_lane] = pages
        row = pages + [0] * (width - len(pages))
        model.prefill_forward(
            tokens=torch.tensor([ids], dtype=torch.int32),
            page_table=torch.tensor([row], dtype=torch.int32),
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(ids)],
            start_pos=torch.zeros(1, dtype=torch.int32),
            empty_slots=[target_lane],
            sampling_params=sampling,
        )

    others = [u for u in range(LANES) if u != lane]
    live = [others[1], others[len(others) // 2], others[-1]]
    for u, n in zip(live, FILLER_LENS):
        import_by_prefill(u, prompts[n])
    import_by_prefill(lane, prompts[STALE_LEN])  # stale state in the target lane
    decoder.release_lane_external(lane)
    model.lane_slots[lane] = type(model.lane_slots[lane])()

    ids = prompts[TARGET_BASE + p]
    assert len(ids) % 4 == p
    model._prefill_one(ids, greedy_token=True)  # leaves the chain's generic state after len(ids) tokens
    pages = pages_for(len(ids))
    decoder.page_rows[lane] = []
    decoder.set_page_row(lane, pages)
    model.chain.built_target.model.write_lane_page_table(lanes_state, page_table_rows(decoder.page_rows, LANES))
    ttnn.synchronize_device(decoder.mesh)

    before = lane_images(layout)
    contexts_before = [tuple(layers.ple.token_contexts) for layers in lanes_state.layers if layers.ple is not None]
    result = {}
    for label, host in (("new", False), ("old", True)):
        restore(layout, before)
        ttnn.synchronize_device(decoder.mesh)
        lanes_state.position.positions[lane] = 7  # a stale mirror entry the import must overwrite
        decoder.host_import = host
        decoder.move_into_lane(lane, len(ids))
        ttnn.synchronize_device(decoder.mesh)
        result[label] = lane_images(layout)
        result[label + "_position"] = (lanes_state.position.positions[lane], lanes_state.position.shifts[lane])
        result[label + "_contexts"] = [
            tuple(layers.ple.token_contexts) for layers in lanes_state.layers if layers.ple is not None
        ]
    decoder.host_import = False
    # control: a wrong ring phase must be seen by the comparison (ring slots only; nothing else depends on the phase)
    restore(layout, before)
    ttnn.synchronize_device(decoder.mesh)
    decoder.importer.import_lane(lane, len(ids) + 1)
    ttnn.synchronize_device(decoder.mesh)
    wrong = family_differences(lane_images(layout), result["old"])
    assert wrong and {name for name, _, _ in wrong} == {"conv"}, f"the comparison missed a rotated ring: {wrong[:5]}"
    assert result["new_position"] == result["old_position"] == (len(ids), 0)
    assert result["new_contexts"] == result["old_contexts"]
    assert all(c[lane] is None for c in result["new_contexts"])
    assert all(
        c[u] == contexts_before[i][u] for i, c in enumerate(result["new_contexts"]) for u in range(LANES) if u != lane
    )
    bad = family_differences(result["new"], result["old"])
    assert not bad, f"lane {lane} p={p}: new vs old import differ in {len(bad)} tensors, first {bad[:5]}"
    moved = family_differences(result["new"], before, lane, only_other_lanes=True)
    assert not moved, f"lane {lane} p={p}: the import changed other lanes in {len(moved)} tensors, first {moved[:5]}"
    changed = family_differences(result["new"], before, lane)
    assert changed, "the import changed nothing: the test is vacuous"
    decoder.release_lane_external(lane)
    for u in live:
        decoder.release_lane_external(u)
    model.chain.reset_and_seed(1000)
    logger.info(f"lane {lane} p={p}: new import == host import, bit for bit ({len(changed)} tensors changed)")


def run_cases(mesh_device, cases):
    checkpoint = Path(os.environ["MODEL_WEIGHTS_DIR"])
    mesh_device.enable_program_cache()
    prompts = build_prompts(checkpoint, [TARGET_BASE + p for p in range(4)] + [STALE_LEN, *FILLER_LENS])
    model = initialize(mesh_device, checkpoint, LANES)
    try:
        blocks = plugin_num_blocks(LANES, BLOCK)
        model.allocate_kv_cache((blocks, 1, BLOCK, 128), ttnn.bfloat16, 12)
        plugin_warmups(model, LANES, BLOCK, blocks)
        for p, lane in cases:
            run_case(model, prompts, lane, p)
    finally:
        model.release_persistent_capture()


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=f"p{c[0]}_lane{c[1]}", marks=pytest.mark.timeout(3600)) for c in CASES]
    + [pytest.param(None, id="all_cases", marks=pytest.mark.timeout(10800))],
)
def test_vllm_fast_import(mesh_device, case):
    if not os.environ.get("MODEL_WEIGHTS_DIR") or not os.environ.get("QWEN38_CACHE_ROOT"):
        pytest.fail("set MODEL_WEIGHTS_DIR and QWEN38_CACHE_ROOT")
    run_cases(mesh_device, CASES if case is None else [case])
