# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device test: the 16-row verify pass (TargetVerifier) matches 16 plain decode steps (tokens, taps, KV rows)."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer_30b.tt.attention import kv_page_table
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import DEFAULT_CKPT_DIR, MuseGlimmerModelArgs

CKPT = os.environ.get("HF_MODEL") or DEFAULT_CKPT_DIR

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "config.json")), reason="checkpoint not available"
)

_mesh_params = mesh_test_params()

S = 100
ROWS = 16
TAP_LAYER = 1
KV_LAYERS = (0, 3)  # 0 is sliding, 3 is global


def _replicated(t, mesh, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _read_kv_rows(model, layer, table_row, positions):
    """-> {'k': [n_dev, P, kv_local, hd], 'v': ...} float32; each device holds its own KV head(s)."""
    bs = model.args.block_size
    out = {}
    for name, cache in zip(("k", "v"), model.kv_caches[layer]):
        devs = []
        for d in ttnn.get_device_tensors(cache):
            t = ttnn.to_torch(d).float()  # [n_blocks, kv_local, block, hd]
            devs.append(torch.stack([t[int(table_row[p // bs]), :, p % bs, :] for p in positions]))
        out[name] = torch.stack(devs)
    return out


def _zero_kv(model, layers):
    for li in layers:
        for cache in model.kv_caches[li]:
            h = ttnn.from_torch(
                torch.zeros(list(cache.shape)),
                dtype=cache.dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(h, cache)


@pytest.mark.timeout(1800)
@_mesh_params
def test_verify_matches_plain_decode(mesh_device, tp):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.verify import TargetVerifier, set_greedy_sampling
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, num_layers=4, max_batch_size=1, max_seq_len=8192)
    assert [args.is_sliding(i) for i in range(4)] == [True, True, True, False]
    model = MuseGlimmerModel(args, sub)
    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = _replicated(pt, sub, ttnn.int32)
    sliding_row = kv_page_table(args, 1, sliding=True)[0]

    prompt = torch.randint(1000, 100000, (S,), generator=torch.Generator().manual_seed(0))
    T = torch.randint(1000, 100000, (ROWS,), generator=torch.Generator().manual_seed(1)).tolist()
    positions = list(range(S, S + ROWS))
    set_greedy_sampling(model)

    def kv_rows():
        return {
            li: _read_kv_rows(model, li, sliding_row if args.is_sliding(li) else pt[0], positions) for li in KV_LAYERS
        }

    # ---- path A: 16 plain decode steps
    model.prefill_forward(prompt, 0, pt_tt)
    buf_a = _replicated(torch.zeros(1, 1, 32, args.hidden_size), sub, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    dev_in = model.allocate_decode_inputs(1, nb)
    arg_a, tap_a = [], []
    for i in range(ROWS):
        host = model.prepare_decode_inputs_host(torch.tensor([T[i]]), torch.tensor([S + i]), pt)
        model.copy_decode_inputs(host, dev_in)
        logits = model.decode_forward(*dev_in, taps={TAP_LAYER: buf_a})
        tok = model.sample_on_device(logits)
        ttnn.deallocate(logits)
        arg_a.append(int(model.tokens_to_torch(tok, 1)[0]))
        ttnn.deallocate(tok)
        tap_a.append(ttnn.to_torch(ttnn.get_device_tensors(buf_a)[0])[0, 0, 0].float())
    tap_a = torch.stack(tap_a)
    kv_a = kv_rows()

    # ---- path B: fresh prefill, one verify pass. Zero the compared caches first so a missed write shows up.
    _zero_kv(model, KV_LAYERS)
    model.prefill_forward(prompt, 0, pt_tt)
    verifier = TargetVerifier(model, [TAP_LAYER], pt[0])
    arg_b = verifier.run(T, S)
    tap_b = ttnn.to_torch(ttnn.get_device_tensors(verifier.tap_list()[0])[0])[0, 0, :ROWS].float()
    kv_b = kv_rows()

    # ---- compare
    matches = 0
    for i in range(ROWS):
        if arg_a[i] == arg_b[i]:
            matches += 1
        else:
            logger.info(f"token mismatch row {i}: plain={arg_a[i]} verify={arg_b[i]}")
    logger.info(f"token agreement {matches}/{ROWS}")
    pccs = []
    for i in range(ROWS):
        ok, msg = comp_pcc(tap_a[i], tap_b[i], 0.99)
        logger.info(f"tap row {i}: {msg}")
        pccs.append(ok)
    kv_ok = []
    for li in KV_LAYERS:
        for name in ("k", "v"):
            ok, msg = comp_pcc(kv_a[li][name], kv_b[li][name], 0.999)
            logger.info(f"kv layer {li} {name}: {msg}")
            kv_ok.append(ok)
    assert matches >= 15, f"only {matches}/{ROWS} tokens agree"
    assert all(pccs), "tap PCC below 0.99 on some row"
    assert all(kv_ok), "KV PCC below 0.999"


@pytest.mark.timeout(1800)
@_mesh_params
def test_verify_topk(mesh_device, tp):
    """Sample mode returns the per-device-merged top-32 of the target logits for the 16 verify rows."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.verify import TargetVerifier
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, num_layers=4, max_batch_size=1, max_seq_len=8192)
    model = MuseGlimmerModel(args, sub)
    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = _replicated(pt, sub, ttnn.int32)

    prompt = torch.randint(1000, 100000, (S,), generator=torch.Generator().manual_seed(0))
    T = torch.randint(1000, 100000, (ROWS,), generator=torch.Generator().manual_seed(1)).tolist()
    model.prefill_forward(prompt, 0, pt_tt)
    verifier = TargetVerifier(model, [TAP_LAYER], pt[0])

    verifier.prepare(T, S)
    logits = model.decode_forward(*verifier.decode_inputs, verify=verifier, taps=verifier.taps)
    ref = model.logits_to_torch(logits, ROWS)
    ttnn.deallocate(logits)
    ref_vals, ref_ids = torch.topk(ref, 33)  # the 33rd value exposes a tie across the top-32 boundary
    ref_next = ref_vals[:, 32]
    ref_vals, ref_ids = ref_vals[:, :32], ref_ids[:, :32]

    vals, ids = verifier.run(T, S, mode="topk")
    greedy = verifier.run(T, S)
    assert vals.shape == (ROWS, 32) and ids.shape == (ROWS, 32)

    val_bad, id_bad, id_checked, top_bad, top_ties = 0, 0, 0, 0, 0
    for r in range(ROWS):
        if not torch.equal(vals[r].float(), ref_vals[r].float()):
            val_bad += 1
            logger.info(f"row {r}: values differ, max abs {(vals[r].float() - ref_vals[r].float()).abs().max()}")
        uniq = {v: c for v, c in zip(*[x.tolist() for x in torch.unique(ref_vals[r], return_counts=True)])}
        for j in range(32):
            if uniq[ref_vals[r, j].item()] == 1 and not (j == 31 and ref_vals[r, 31] == ref_next[r]):
                id_checked += 1
                if int(ids[r, j]) != int(ref_ids[r, j]):
                    id_bad += 1
                    logger.info(f"row {r} col {j}: id {int(ids[r, j])} != ref {int(ref_ids[r, j])}")
        if ref_vals[r, 0] == ref_vals[r, 1]:
            top_ties += 1
        elif int(ids[r, 0]) != int(greedy[r]):
            top_bad += 1
            logger.info(f"row {r}: top id {int(ids[r, 0])} != greedy {greedy[r]}")
    logger.info(
        f"rows with value mismatch {val_bad}/{ROWS}; unique-value id mismatches {id_bad}/{id_checked}; "
        f"top-vs-greedy mismatches {top_bad} (exact ties skipped: {top_ties})"
    )
    assert val_bad == 0, "top-32 values differ from the reference"
    assert id_bad == 0, "top-32 ids differ from the reference"
    assert top_bad == 0, "top-1 id differs from the greedy token"


@pytest.mark.timeout(1800)
@_mesh_params
def test_verify_logits(mesh_device, tp):
    """Logits mode returns the full target logits of the 16 verify rows, equal to the plain decode_forward logits."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.verify import TargetVerifier
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, num_layers=4, max_batch_size=1, max_seq_len=8192)
    model = MuseGlimmerModel(args, sub)
    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = _replicated(pt, sub, ttnn.int32)

    prompt = torch.randint(1000, 100000, (S,), generator=torch.Generator().manual_seed(0))
    T = torch.randint(1000, 100000, (ROWS,), generator=torch.Generator().manual_seed(1)).tolist()
    model.prefill_forward(prompt, 0, pt_tt)
    verifier = TargetVerifier(model, [TAP_LAYER], pt[0])

    verifier.prepare(T, S)
    logits = model.decode_forward(*verifier.decode_inputs, verify=verifier, taps=verifier.taps)
    ref = model.logits_to_torch(logits, ROWS)
    ttnn.deallocate(logits)

    lg = verifier.run(T, S, mode="logits")
    greedy = verifier.run(T, S)
    assert lg.shape == ref.shape == (ROWS, args.vocab_size)
    assert torch.equal(lg, ref), f"logits differ from the reference, max abs {(lg - ref).abs().max()}"
    am = lg.argmax(-1).tolist()
    bad, ties = 0, 0
    for r in range(ROWS):
        top = torch.topk(lg[r], 2).values
        if top[0] == top[1]:
            ties += 1
        elif am[r] != int(greedy[r]):
            bad += 1
            logger.info(f"row {r}: logits argmax {am[r]} != greedy {greedy[r]}")
    logger.info(f"logits-vs-greedy mismatches {bad} (exact ties skipped: {ties})")
    assert bad == 0, "logits argmax differs from the greedy token"
