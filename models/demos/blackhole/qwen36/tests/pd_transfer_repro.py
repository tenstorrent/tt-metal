# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process round trip of the prefill/decode disaggregation state transfer (pd_transfer.py).

Prefill a request into decode slot 0 (its blocks = page-table row 0) with the GDN capture on, export its
paged-KV blocks and GDN snapshot to the host, then import both into decode slot 3 / page-table row 3 and
greedy-decode row 3. The continuation must equal the continuation decoded from slot 0 itself: what a decode
instance receives from a prefill instance is exactly the state it would have produced locally.

Run: MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=2,3,4,5 pytest models/demos/blackhole/qwen36/tests/pd_transfer_repro.py -x -s
Env: QWEN36_REPRO_STEPS (24), QWEN36_REPRO_PROMPT_TOKENS (0 = a chat prompt), QWEN36_PD_BLOCKS_PER_USER (8).
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt import pd_transfer
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BMAX = 8
STEPS = int(os.environ.get("QWEN36_REPRO_STEPS", "24"))
BPU = int(os.environ.get("QWEN36_PD_BLOCKS_PER_USER", "8"))
SRC_SLOT, DST_SLOT = 0, 3


def _prefill_slot(model, ids, page_tables, slot):
    T = len(ids)
    logits = model.prefill_paged_slots(
        [torch.tensor([ids], dtype=torch.int32)], page_tables[slot : slot + 1], [slot], valid_lens=[T]
    )
    return int(logits[0].reshape(-1)[: model.vocab_size].float().argmax())


def _decode_row(model, row, first, T, page_tables, steps):
    """Greedy-decode decode-slot `row` for `steps` steps at full width (other rows = dummy users)."""
    out_ids, tok = [], first
    for s in range(steps):
        tokens = torch.zeros((BMAX, 1), dtype=torch.int32)
        positions = torch.zeros((BMAX,), dtype=torch.int32)
        for u in range(BMAX):
            tokens[u, 0], positions[u] = 100 + u, 64
        tokens[row, 0], positions[row] = tok, T + s
        dev = model.prepare_inputs_decode(tokens, positions, page_tables)
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        lg = model.process_output_decode(out, BMAX)[:, 0, : model.vocab_size].float()
        tok = int(lg[row].argmax())
        out_ids.append(tok)
    return out_ids


@run_for_blackhole()
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_pd_export_import_round_trip(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    n_tok = int(os.environ.get("QWEN36_REPRO_PROMPT_TOKENS", "0"))
    if n_tok:
        ids = [int(x) for x in torch.randint(1000, 50000, (n_tok,))]
    else:
        text = os.environ.get("QWEN36_REPRO_PROMPT_TEXT") or (
            "List the planets of the solar system in order from the sun, one line each."
        )
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = list(ids["input_ids"])
        ids = [int(x) for x in ids]
    T = len(ids)
    assert T <= BPU * BLOCK_SIZE - STEPS, f"prompt {T} + {STEPS} steps exceeds {BPU} blocks"
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    try:
        # --- prefill side: prefill into SRC_SLOT with the capture on, export KV + GDN ---
        model.pd_gdn_capture = {}
        first = _prefill_slot(model, ids, page_tables, SRC_SLOT)
        assert SRC_SLOT in model.pd_gdn_capture, "prefill_paged_slots did not park the GDN snapshot"
        rec_snap, conv_snap = model.pd_gdn_capture.pop(SRC_SLOT)
        model.pd_gdn_capture = None
        src_blocks = [int(b) for b in page_tables[SRC_SLOT]]
        t0 = time.perf_counter()
        kv = pd_transfer.export_kv_blocks(model, src_blocks)
        t_exp = time.perf_counter() - t0
        logger.info(
            f"[pd] export: KV {pd_transfer.kv_nbytes(kv) / 2**20:.1f} MiB ({len(src_blocks)} blocks x {len(kv)} layers) "
            f"in {1e3 * t_exp:.1f} ms; GDN {pd_transfer.gdn_state_nbytes(rec_snap, conv_snap) / 2**20:.1f} MiB "
            f"(rec {tuple(rec_snap[0].shape)} {rec_snap[0].dtype}, taps {tuple(conv_snap[0][0].shape)} {conv_snap[0][0].dtype})"
        )
        # reference continuation from the slot the prefill wrote
        ref = _decode_row(model, SRC_SLOT, first, T, page_tables, STEPS)
        logger.info(f"[pd] src slot {SRC_SLOT}: {tok.decode([first] + ref)!r}")

        # --- decode side: import into DST_SLOT / row DST_SLOT, decode from there ---
        dst_blocks = [int(b) for b in page_tables[DST_SLOT]]
        t0 = time.perf_counter()
        pd_transfer.import_kv_blocks(model, dst_blocks, kv)
        t_kv = time.perf_counter() - t0
        t0 = time.perf_counter()
        pd_transfer.import_gdn_slot(model, DST_SLOT, rec_snap, conv_snap)
        t_gdn = time.perf_counter() - t0
        ttnn.synchronize_device(device)
        logger.info(f"[pd] import: KV {1e3 * t_kv:.1f} ms, GDN {1e3 * t_gdn:.1f} ms")
        # warm timings (first calls above include program compiles)
        t0 = time.perf_counter()
        pd_transfer.import_kv_blocks(model, dst_blocks, kv)
        t_kv2 = time.perf_counter() - t0
        t0 = time.perf_counter()
        pd_transfer.import_gdn_slot(model, DST_SLOT, rec_snap, conv_snap)
        t_gdn2 = time.perf_counter() - t0
        t0 = time.perf_counter()
        kv2 = pd_transfer.export_kv_blocks(model, src_blocks)
        t_exp2 = time.perf_counter() - t0
        ttnn.synchronize_device(device)
        logger.info(
            f"[pd] WARM: export KV {1e3 * t_exp2:.1f} ms, import KV {1e3 * t_kv2:.1f} ms, import GDN {1e3 * t_gdn2:.1f} ms"
        )
        got = _decode_row(model, DST_SLOT, first, T, page_tables, STEPS)
        same = sum(1 for a, b in zip(got, ref) if a == b)
        logger.info(f"[pd] dst slot {DST_SLOT}: {tok.decode([first] + got)!r} -- {same}/{STEPS} tokens match")

        # --- remap check: move the imported state to other rows (even and odd offsets) and decode there ---
        remap_bad = []
        for dst_row in (1, 0, 6):
            pd_transfer.import_kv_blocks(model, dst_blocks, kv)  # KV is per page table row; reuse dst_blocks' row
            pd_transfer.import_gdn_slot(model, DST_SLOT, rec_snap, conv_snap)
            perm = list(range(BMAX))
            perm[dst_row], perm[DST_SLOT] = DST_SLOT, dst_row  # row dst_row <- slot DST_SLOT (swap)
            model._remap_gdn_slots(torch.tensor(perm, dtype=torch.int32))
            pt2 = page_tables.clone()
            pt2[dst_row] = page_tables[DST_SLOT]  # the moved request keeps its blocks
            got2 = _decode_row(model, dst_row, first, T, pt2, min(STEPS, 12))
            same2 = sum(1 for a, b in zip(got2, ref) if a == b)
            logger.info(
                f"[pd] remap slot {DST_SLOT} -> row {dst_row}: {tok.decode([first] + got2)!r} -- {same2}/{min(STEPS, 12)} match"
            )
            if same2 != min(STEPS, 12):
                remap_bad.append((dst_row, same2))
            # undo the swap so the next iteration starts clean
            model._remap_gdn_slots(torch.tensor(perm, dtype=torch.int32))
        # --- negative control: a slot that got only the KV (no GDN state) must NOT reproduce it ---
        ctl_slot = 5
        pd_transfer.import_kv_blocks(model, [int(b) for b in page_tables[ctl_slot]], kv)
        ctl = _decode_row(model, ctl_slot, first, T, page_tables, min(STEPS, 8))
        logger.info(f"[pd] control (KV only, zero GDN) slot {ctl_slot}: {tok.decode([first] + ctl)!r}")

        assert same == STEPS, f"imported state diverges from the source: {same}/{STEPS} tokens match"
        assert not remap_bad, f"remapped state diverges: {remap_bad}"
    finally:
        model.pd_gdn_capture = None
        model.free_kv_caches()
