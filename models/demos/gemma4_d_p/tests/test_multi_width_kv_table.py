# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-request chunk width, end to end on the KV side.

One model built with several prefill widths and one KV cache with one slot per width. Each slot is prefilled with
the GPU-golden prompt at its own width, so each slot's ring cache has a different block-cyclic layout (period =
width). One migration address table then describes every slot with that slot's width, and each slot is read back
the way the migration worker sees it, block by block through the table, and compared with the GPU golden. A passing
slot means a reader that only knows (layer, position, slot) gets correct KV regardless of the width that wrote it.

GEMMA4_KVW_WIDTHS: widths, one slot each (default "2048,8192"). GEMMA4_KVW_CONTEXT: tokens per slot (default 32768).
GEMMA4_KVW_WRONG_SLOT: describe this slot with another slot's width (negative control; that slot should fail).
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import _enumerate_devices
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table
from models.demos.gemma4_d_p.tt.runners.kv_validation import read_slot_kv_and_check_pcc

MIN_PER_HEAD_PCC = 0.91
TRACE_REGION_SIZE = int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", 256_000_000))


def _tokens_mapper(mesh_device):
    return ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=(1, None))


class _Width:
    """Pinned staging tensors and one captured trace for one prefill width."""

    def __init__(self, mesh_device, model, tokens, width):
        self.mesh_device, self.model, self.tokens, self.width = mesh_device, model, tokens, width
        self.input_tokens = ttnn.to_device(self._host(self.tokens[:width]), mesh_device)
        self.positions = ttnn.to_device(self._host(torch.arange(width)), mesh_device)
        self.trace_id = None

    def _host(self, values):
        return ttnn.from_torch(
            torch.as_tensor(values, dtype=torch.int64).reshape(1, self.width),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_tokens_mapper(self.mesh_device),
        )

    def stage(self, slot, chunk_idx):
        start = chunk_idx * self.width
        ttnn.copy_host_to_device_tensor(self._host(self.tokens[start : start + self.width]), self.input_tokens)
        self.model.prefill_metadata.update(slot_idx=slot, kv_actual_global=start)
        ttnn.copy_host_to_device_tensor(self._host(torch.arange(start, start + self.width)), self.positions)

    def forward(self):
        self.model.set_prefill_rope_positions(self.positions)
        embeddings = self.model.transform_and_embed_prefill_inputs_device(self.input_tokens)
        return self.model(hidden_states=embeddings)

    def warm(self, slot):
        self.stage(slot, 0)
        out = self.forward()
        ttnn.synchronize_device(self.mesh_device)
        out.deallocate(True)

    def capture(self, slot):
        self.stage(slot, 0)
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.output = self.forward()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

    def prefill(self, slot, context_len):
        for chunk_idx in range(context_len // self.width):
            self.stage(slot, chunk_idx)
            ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_multi_width_kv_table(mesh_device, reset_seeds):
    widths = [int(w) for w in os.environ.get("GEMMA4_KVW_WIDTHS", "2048,8192").split(",")]
    context_len = int(os.environ.get("GEMMA4_KVW_CONTEXT", "32768"))
    wrong_slot = os.environ.get("GEMMA4_KVW_WRONG_SLOT")
    trace_dir = Path(os.environ["PREFILL_TRACE_DIR"])
    hf_model_id = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    tokens = torch.tensor(json.loads((trace_dir / "metadata.json").read_text())["token_ids"][:context_len])
    assert len(tokens) == context_len and all(context_len % w == 0 for w in widths)

    mesh_config = MeshConfig(mesh_device)
    hf_config = Gemma4ModelArgs.load_hf_config(hf_model_id)
    kv_caches = allocate_ring_kv_caches(
        mesh_config,
        Gemma4ModelArgs.from_hf_config(hf_config),
        num_users=len(widths),
        max_seq_len=context_len,
        prefill_chunk_size=max(widths),
    )
    _, model, _, _ = create_tt_model(
        mesh_config=mesh_config,
        hf_model_id=hf_model_id,
        prefill_chunk_size=tuple(widths),
        max_seq_len=context_len,
        max_batch_size=1,
        ring_kv_caches=kv_caches,
    )
    model._prefill_metadata_external = True

    # Warm every width, then capture every trace (capturing one width before warming the next corrupts it).
    runners = {slot: _Width(mesh_device, model, tokens, width) for slot, width in enumerate(widths)}
    for slot, runner in runners.items():
        runner.warm(slot)
    for slot, runner in runners.items():
        runner.capture(slot)
    try:
        for slot, runner in runners.items():
            t0 = time.time()
            runner.prefill(slot, context_len)
            logger.info(f"[kvw] slot {slot} prefilled at width {runner.width} in {time.time() - t0:.2f}s")
    finally:
        for runner in runners.values():
            ttnn.release_trace(mesh_device, runner.trace_id)

    slot_widths = dict(enumerate(widths))
    if wrong_slot is not None:
        slot = int(wrong_slot)
        slot_widths[slot] = next(w for w in widths if w != widths[slot])
        logger.info(f"[kvw] negative control: slot {slot} described with width {slot_widths[slot]}")
    table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_caches=kv_caches, chunk_size=slot_widths)
    device_map = {(mesh, chip): unique_id for unique_id, mesh, chip in _enumerate_devices(mesh_device)}

    results = {}
    for slot, width in enumerate(widths):
        minima = read_slot_kv_and_check_pcc(table, device_map, slot, context_len, trace_dir)
        results[slot] = min(minima.values())
        logger.info(
            f"[kvw] slot {slot} width {width} (table width {slot_widths[slot]}): min per-head PCC {results[slot]:.6f} {minima}"
        )
    for slot, worst in results.items():
        if wrong_slot is not None and slot == int(wrong_slot):
            assert worst < MIN_PER_HEAD_PCC, f"negative control passed unexpectedly: slot {slot} PCC {worst:.6f}"
        else:
            assert worst >= MIN_PER_HEAD_PCC, f"slot {slot} at width {widths[slot]}: min per-head PCC {worst:.6f}"
