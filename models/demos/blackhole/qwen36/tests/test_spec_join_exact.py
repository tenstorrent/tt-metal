# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""prefill_for_spec's DEVICE-SIDE GDN slot write (device_copy=True: ttnn.fill_cache on the recurrent state + a masked
ttnn.where per conv tap, the QWEN36_GDN_SLOT_DEVICE_COPY=2 mechanism ported from prefill_paged_slots) must leave the
batched GDN decode state BIT-IDENTICAL to the host round trip it replaces (device_copy=False: ttnn.to_torch of every
layer's B=1 scratch, from_torch, write_slot) -- for the written row AND every other row.

Scenario, on a B=4 model (the served slot count): the same three joins in the same order under each path --
prompt A -> slot 0, a >1-chunk prompt C -> slot 3 (the chunked prefill), prompt B -> slot 1 (a row between two live
rows) -- with the batched state zeroed in place between the two runs. Every GDN layer's full [B, Nv, Dk, Dv] recurrent
state and its K [1, B, C] conv taps are read back after each run and compared with torch.equal, as are prompt B's
logits. Also checks the flags a join leaves (taps current, window mirror stale, packed plain-decode history invalid),
that the untouched slot stays zero, and that the single-row spec seed (seed_spec_state_user == seed_spec_row('default',
u, u): slice / concat / in-place copy, no selector matmul) writes E_prev row u from the slot's taps exactly with a zero
tail -- and that the old cached one-hot selector (_spec_seed_sel / _spec_seed_selector) is gone.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_spec_join_exact.py -v -s
Needs the full model (all 48 GDN layers are compared); no drafter weights are needed (the taps are not captured).
"""

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_spec_batched import _batch_prompts, _blocks_per_user, _release
from models.demos.blackhole.qwen36.tests.test_spec_lossless import MAX_NEW, NUM_BLOCKS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

B = 4  # the served slot count
K_DRAFT = 7  # DFlash2 block 8 -> K = 7 draft tokens, T = K + 1 verify rows
T_VERIFY = K_DRAFT + 1
LONG_LEN = 2048 + 53  # one full 2048 chunk + a masked tail: the chunked spec prefill
UNTOUCHED_SLOT = 2


def _gdn_layers(model):
    return [layer.attention for layer in model.layers if not layer.is_full_attention]


def _join(model, prompt_ids, slot, page_tables, device_copy):
    """One request joining ``slot`` through prefill_for_spec (no drafter: on_chunk ignores the taps). Returns the host
    [1, vocab] float logits of its last position."""
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.int32)
    T = len(prompt_ids)
    pt_u = page_tables[slot : slot + 1].contiguous()

    def on_chunk(hidden, chunk_start, valid_len):
        pass  # the caller frees ``hidden``; nothing to ingest without a drafter

    logits = model.prefill_for_spec(prompt, pt_u, T, on_chunk, slot=slot, device_copy=device_copy)
    lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    ttnn.deallocate(logits)
    return lt.reshape(-1, model.vocab_size)[:1].float().clone()


def _read_layer(model, dn):
    """Host copies of one GDN layer's batched decode state: rec [n_dev * B, Nv, Dk, Dv], taps K x [n_dev, B, C]."""
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    rec = ttnn.to_torch(dn.rec_state, mesh_composer=comp)
    convs = [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states]
    return rec, convs


def _snapshot(model):
    ttnn.synchronize_device(model.mesh_device)
    return [_read_layer(model, dn) for dn in _gdn_layers(model)]


def _zero_batched_state(model):
    """Zero every GDN layer's batched decode buffers in place (the bound state: reset_state_inplace)."""
    for dn in _gdn_layers(model):
        dn.reset_state_inplace()
    ttnn.synchronize_device(model.mesh_device)


def _run_joins(model, prompts, page_tables, device_copy):
    """The scenario: A -> slot 0, long C -> slot 3, B -> slot 1. Returns prompt B's logits."""
    _join(model, prompts["A"], 0, page_tables, device_copy)
    _join(model, prompts["C"], 3, page_tables, device_copy)
    logits_b = _join(model, prompts["B"], 1, page_tables, device_copy)
    ttnn.synchronize_device(model.mesh_device)
    return logits_b


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_join_device_copy_is_bit_exact(mesh_device):
    if not _MULTI:
        pytest.skip("prefill_for_spec is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=B, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    model.set_gdn_fused_decode(True)  # what the serving path sets (fused GDN decode/verify math)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    short = _batch_prompts(3, tokenizer)  # 130 / 147 / 165 tokens, distinct content
    prompts = {"A": short[0], "B": short[1], "C": (short[2] * (LONG_LEN // len(short[2]) + 1))[:LONG_LEN]}
    assert len(prompts["C"]) == LONG_LEN > 2048
    bpu = _blocks_per_user(LONG_LEN, K_DRAFT, MAX_NEW + 8)
    page_tables = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    kv_shape = [B * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)
    gdns = _gdn_layers(model)
    assert len(gdns) >= 40, f"expected the full model's 48 GDN layers, got {len(gdns)}"
    n_dev = model.num_devices
    logger.info(
        f"[join-exact] B={B} n_dev={n_dev} GDN layers={len(gdns)} prompts A={len(prompts['A'])} B={len(prompts['B'])} C={LONG_LEN}"
    )

    try:
        # ---- run 1: the host round trip (the reference) ---------------------------------------------- #
        logits_host = _run_joins(model, prompts, page_tables, device_copy=False)
        ref = _snapshot(model)
        rec0, convs0 = ref[0]
        assert rec0.shape[0] == n_dev * B and convs0[0].shape[1] == B, (rec0.shape, convs0[0].shape)
        rec0_rows = rec0.view(n_dev, B, *rec0.shape[1:])
        for slot in (0, 1, 3):
            assert rec0_rows[:, slot].abs().sum() > 0, f"host path left slot {slot} zero (nothing to compare)"
        assert rec0_rows[:, UNTOUCHED_SLOT].abs().sum() == 0, "the untouched slot should be zero"
        logger.info("[join-exact] host-path reference captured")

        # ---- zero the batched state in place, and prove it (else run 2 could coast on run 1's rows) --- #
        _zero_batched_state(model)
        rec_z, convs_z = _read_layer(model, gdns[0])
        assert rec_z.abs().sum() == 0 and all(c.abs().sum() == 0 for c in convs_z), "in-place reset did not zero"

        # ---- run 2: the device-side row write --------------------------------------------------------- #
        logits_dev = _run_joins(model, prompts, page_tables, device_copy=True)
        ttnn.synchronize_device(device)
        worst = 0.0
        for li, dn in enumerate(gdns):
            rec, convs = _read_layer(model, dn)
            rec_ref, convs_ref = ref[li]
            if not torch.equal(rec, rec_ref):
                d = (rec.float() - rec_ref.float()).abs()
                rows = d.view(n_dev, B, -1).amax(dim=(0, 2))
                pytest.fail(f"layer {li}: rec_state differs from the host path (per-slot max|delta| {rows.tolist()})")
            for m in range(dn.K):
                if not torch.equal(convs[m], convs_ref[m]):
                    d = (convs[m].float() - convs_ref[m].float()).abs().amax(dim=(0, 2))
                    pytest.fail(
                        f"layer {li}: conv tap {m} differs from the host path (per-slot max|delta| {d.tolist()})"
                    )
            worst = max(worst, float((rec.float() - rec_ref.float()).abs().max()))
            # The flags a join must leave: taps current, window mirror behind them, packed plain history invalid.
            assert dn._conv_taps_stale is False, f"layer {li}: taps marked stale after the join"
            assert dn._conv_win_stale is True, f"layer {li}: window mirror not marked stale after the join"
            assert dn._hist_packed_valid is False, f"layer {li}: packed conv history marked valid after a spec join"
        assert worst == 0.0
        assert torch.equal(logits_dev, logits_host), "prompt B's logits differ between the two paths"
        logger.info(
            f"[join-exact] device path == host path on all {len(gdns)} GDN layers (rec + {gdns[0].K} taps) and logits"
        )

        # ---- the single-row seed: E_prev row 1 <- slot 1's taps exactly, zero tail; no cached selector ------ #
        dn0 = gdns[0]
        dn0.prepare_spec_verify(B, T_VERIFY)  # == prepare_spec_cfg("default", B, T_VERIFY)
        for name in ("_spec_seed_sel", "_spec_seed_selector"):
            assert not hasattr(dn0, name), f"the one-hot seed selector ({name}) was removed with the per-row seed"
        cfg = dn0.spec_cfg("default")
        assert cfg.shape == (B, T_VERIFY), cfg.shape
        dn0.seed_spec_state_user(1)  # seed_spec_row("default", 1, 1)
        ttnn.synchronize_device(device)
        win = ttnn.to_torch(ttnn.get_device_tensors(dn0.verify_win_cur())[0])  # [B, R, C], device 0
        assert win.shape[0] == B and win.shape[1] == dn0.K - 1 + T_VERIFY, win.shape
        taps = torch.stack([ttnn.to_torch(ttnn.get_device_tensors(c)[0])[0, 1] for c in dn0.conv_states])  # [K, C]
        assert torch.equal(win[1, : dn0.K], taps), "single-row seed: E_prev row 1 != the slot's taps"
        assert win[1, dn0.K :].abs().sum() == 0, "single-row seed: E_prev row 1 tail not zero"
        logger.info("[join-exact] seed_spec_state_user(1) writes E_prev row 1 from the taps exactly (zero tail)")
    finally:
        _release(model)
    del model
    gc.collect()
