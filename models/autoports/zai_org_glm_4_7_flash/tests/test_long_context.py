# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context evidence for the GLM-4.7-Flash functional decoder.

The HF-advertised context is max_position_embeddings = 202752. A full HF fp32
CPU reference is infeasible at that length (the S^2 attention alone is ~1e15
FLOP on CPU), so the evidence ladder is:

1. ``test_absorbed_reference_matches_hf`` (CPU): certify the absorbed-MLA torch
   window reference against the actual HF layer at a short length.
2. ``test_prefill_8k_vs_hf`` (device): non-aligned 8191-token prefill + decode
   against the full HF fp32 reference.
3. ``test_full_context_202k`` (device): 202751-token prefill (the longest valid
   non-aligned length: positions 0..202751 with the last one decoded) with
   * full latent-cache PCC vs the exact linear reference,
   * full-layer output PCC on 32-row windows at the start / middle / end of the
     sequence vs the certified absorbed reference (the end window attends the
     whole ~202k-token cache),
   * a traced-decode-path decode step at position 202751 (the maximum valid
     position) vs the absorbed reference,
   * wall-clock timings.
   Evidence JSON: doc/functional_decoder/long_context_{moe,dense}.json.

Run: pytest -q -s -m long models/autoports/zai_org_glm_4_7_flash/tests/test_long_context.py

Set GLM47_DECODER=fused to run the identical evidence ladder against the
fused decoder (tt/fused_decoder.py); JSON then goes to doc/fused_decoder/.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tests import utils
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import FunctionalDecoder, PagedCacheConfig

if os.environ.get("GLM47_DECODER", "functional") == "fused":
    from models.autoports.zai_org_glm_4_7_flash.tt.fused_decoder import FusedDecoder as DecoderCls

    DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "fused_decoder"
else:
    DecoderCls = FunctionalDecoder
    DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"
PCC_BAR = 0.995
FULL_CONTEXT = 202752

pytestmark = pytest.mark.long


@pytest.fixture(scope="module")
def cfg():
    return utils.hf_config()


def test_absorbed_reference_matches_hf(cfg):
    """CPU-only: the absorbed-MLA window reference is an exact refactoring of
    the HF layer; certify it before trusting it at 202k."""
    sd = utils.synth_layer_state_dict(cfg, 1)
    layer = utils.build_hf_layer(cfg, 1, sd)
    S = 256
    x = utils.synth_activations(cfg, 1, S, seed=7)
    ref = utils.hf_forward(cfg, layer, x)[0]
    kvpe = utils.torch_latent_cache_reference(cfg, sd, x[0])
    rows = list(range(0, 8)) + list(range(120, 128)) + list(range(248, 256))
    got = utils.torch_absorbed_window_reference(cfg, sd, layer, x[0], kvpe, rows)
    p = utils.pcc(ref[rows], got)
    print(f"absorbed-reference certification PCC={p:.8f}")
    assert p >= 0.9999


def test_prefill_8k_vs_hf(cfg):
    """Mid-length anchor: full HF reference at a non-aligned 8191 tokens."""
    S, n_decode = 8191, 2
    sd = utils.synth_layer_state_dict(cfg, 1)
    layer = utils.build_hf_layer(cfg, 1, sd)
    x = utils.synth_activations(cfg, 1, S + n_decode, seed=7)
    t0 = time.perf_counter()
    ref = utils.hf_forward(cfg, layer, x)
    print(f"HF fp32 reference at S={S + n_decode}: {time.perf_counter() - t0:.1f}s")

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    try:
        paged = PagedCacheConfig.for_context(16384, 1)
        dec = DecoderCls.from_state_dict(
            sd,
            hf_config=cfg,
            layer_idx=1,
            mesh_device=device,
            max_batch_size=1,
            max_context=16384,
            paged_config=paged,
            prefill_chunk_size=2048,
        )
        cache = dec.allocate_kv_cache()
        pt_torch = utils.make_page_table(1, paged.max_num_blocks, seed=3)
        pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        t0 = time.perf_counter()
        out = dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S)
        ttnn.synchronize_device(device)
        wall = time.perf_counter() - t0
        got = ttnn.to_torch(out).float()[0, 0]
        p = utils.pcc(ref[0, :S], got)
        print(f"prefill S={S} PCC={p:.6f} wall={wall:.1f}s ({S / wall:.0f} tok/s, includes compile)")
        assert p >= PCC_BAR

        ties = utils.router_tie_positions(cfg, layer, x)
        for i in range(n_decode):
            pos = S + i
            xd = ttnn.from_torch(
                x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            cur = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
            rot = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32), device=device)
            outd = dec.decode_forward(xd, kv_cache=cache, page_table=pt, cur_pos_tensor=cur, rot_idxs=rot)
            pd = utils.pcc(ref[0, pos], ttnn.to_torch(outd).float()[0, 0, 0])
            print(f"decode pos={pos} PCC={pd:.6f}")
            if pd < PCC_BAR:
                assert pos in ties, f"pos {pos} PCC {pd} not a router tie"
    finally:
        ttnn.close_device(device)


@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_full_context_202k(cfg, kind):
    """Full-context capability proof for one layer kind.

    dense is the numerics control: no routing discreteness, so every window row
    must meet the bar directly - this isolates the MLA/flash/rope path at full
    length. moe additionally exhibits discrete top-4 routing flips when the
    (synthetic, diffuse) attention output at ~1e5 keys perturbs router scores
    by more than a bf16 ulp; every below-bar moe row must therefore be exactly
    reproduced by an alternate top-4 subset of the reference top-6 experts
    (utils.explain_row_as_routing_flip) - proving the row is a routing flip,
    not an attention/cache numerics bug.
    """
    S = FULL_CONTEXT - 1  # 202751: longest valid non-aligned prefill; the last
    # position (202751) is exercised through the decode path below.
    layer_idx = utils.LAYER_KINDS[kind]
    sd = utils.synth_layer_state_dict(cfg, layer_idx)
    layer = utils.build_hf_layer(cfg, layer_idx, sd)
    x = utils.synth_activations(cfg, layer_idx, FULL_CONTEXT, seed=7)
    evidence = {"kind": kind, "advertised_context": FULL_CONTEXT, "prefill_seq_len": S}
    failures = []

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    try:
        paged = PagedCacheConfig.for_context(FULL_CONTEXT, 1)
        dec = DecoderCls.from_state_dict(
            sd,
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=device,
            max_batch_size=1,
            max_context=FULL_CONTEXT,
            paged_config=paged,
            prefill_chunk_size=2048,
        )
        cache = dec.allocate_kv_cache()
        pt_torch = utils.make_page_table(1, paged.max_num_blocks, seed=3)
        pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        t_start = time.perf_counter()
        last = [t_start]

        def progress(i, n):
            now = time.perf_counter()
            if i % 10 == 0 or i == n - 1:
                print(f"  chunk {i + 1}/{n} t={now - t_start:.0f}s (+{now - last[0]:.1f}s)", flush=True)
            last[0] = now

        out = dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S, progress_cb=progress)
        ttnn.synchronize_device(device)
        wall = time.perf_counter() - t_start
        evidence["prefill_wall_s"] = round(wall, 1)
        evidence["prefill_tokens_per_s"] = round(S / wall, 1)
        print(f"[{kind}] full-context prefill S={S}: {wall:.0f}s ({S / wall:.0f} tok/s)")
        got = ttnn.to_torch(out).float()[0, 0]
        ttnn.deallocate(out)
        ttnn.deallocate(x_tt)

        # 1. latent cache vs exact linear reference
        kvpe_ref = utils.torch_latent_cache_reference(cfg, sd, x[0])
        cache_torch = ttnn.to_torch(cache).float()
        cache_rows = utils.gather_user_cache(cache_torch, pt_torch, 0, S, paged.block_size)
        del cache_torch
        p_cache = utils.pcc(kvpe_ref[:S], cache_rows)
        evidence["cache_pcc_vs_linear_ref"] = p_cache
        print(f"[{kind}] latent cache PCC vs linear reference: {p_cache:.6f}")
        if p_cache < 0.999:
            failures.append(f"cache PCC {p_cache:.6f} < 0.999")
        del cache_rows

        # 2. full-layer window analysis (start / middle / end-of-sequence rows)
        windows = {"start": 0, "middle": 101376, "end": S - 32}
        evidence["windows"] = {}
        for name, r0 in windows.items():
            rows = list(range(r0, r0 + 32))
            ref_rows, tie, res_rows, h2_rows = utils.torch_absorbed_window_reference(
                cfg, sd, layer, x[0], kvpe_ref, rows, return_parts=True
            )
            row_pcc = [utils.pcc(ref_rows[i], got[rows[i]]) for i in range(32)]
            below = [i for i in range(32) if row_pcc[i] < PCC_BAR]
            explained, unexplained = [], []
            for i in below:
                # Every below-bar moe row must be exactly reproduced by an
                # alternate top-4 expert subset (tie status is an annotation,
                # not a bypass: a tie flip IS a 4th<->5th routing flip and must
                # pass the same reconstruction proof).
                if kind == "moe":
                    p_alt, subset = utils.explain_row_as_routing_flip(
                        cfg, sd, h2_rows[i], res_rows[i], got[rows[i]], PCC_BAR
                    )
                    if subset is not None:
                        explained.append(
                            {
                                "row": rows[i],
                                "pcc": row_pcc[i],
                                "why": "routing flip" + (" (sub-ulp tie)" if tie[i] else ""),
                                "alt_expert_set_pcc": p_alt,
                            }
                        )
                        continue
                    unexplained.append({"row": rows[i], "pcc": row_pcc[i], "best_alt_pcc": p_alt, "tie": bool(tie[i])})
                else:
                    unexplained.append({"row": rows[i], "pcc": row_pcc[i]})
            ok_rows = [p for i, p in enumerate(row_pcc) if i not in below]
            evidence["windows"][name] = {
                "rows": [r0, r0 + 31],
                "pcc_all_rows": utils.pcc(ref_rows, got[rows]),
                "pcc_min_ok_row": min(ok_rows) if ok_rows else None,
                "rows_at_bar": 32 - len(below),
                "explained": explained,
                "unexplained": unexplained,
            }
            print(
                f"[{kind}] window {name} rows {r0}..{r0+31}: agg={evidence['windows'][name]['pcc_all_rows']:.6f} "
                f"rows>=bar={32-len(below)}/32 explained={len(explained)} unexplained={len(unexplained)}"
            )
            if unexplained:
                failures.append(f"window {name}: unexplained rows {unexplained}")
            if 32 - len(below) < 26:
                failures.append(f"window {name}: only {32 - len(below)}/32 rows at bar")
        del got

        # 3. decode at the maximum position, attending the full cache
        pos = S  # 202751
        xd = ttnn.from_torch(
            x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        cur = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
        rot = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32), device=device)
        t0 = time.perf_counter()
        outd = dec.decode_forward(xd, kv_cache=cache, page_table=pt, cur_pos_tensor=cur, rot_idxs=rot)
        ttnn.synchronize_device(device)
        t_decode = time.perf_counter() - t0
        got_d = ttnn.to_torch(outd).float()[0, 0, 0]
        ref_d, tie_d, res_d, h2_d = utils.torch_absorbed_window_reference(
            cfg, sd, layer, x[0], kvpe_ref, [pos], return_parts=True
        )
        p_dec = utils.pcc(ref_d[0], got_d)
        dec_entry = {
            "position": pos,
            "pcc": p_dec,
            "router_tie": bool(tie_d[0]),
            "first_call_wall_s": round(t_decode, 3),
        }
        if p_dec < PCC_BAR:
            # Same rule as the window analysis: tie status is an annotation,
            # not a bypass; a below-bar moe decode row must pass the
            # alternate-top-4 reconstruction proof.
            if kind == "moe":
                p_alt, subset = utils.explain_row_as_routing_flip(cfg, sd, h2_d[0], res_d[0], got_d, PCC_BAR)
                dec_entry["routing_flip_explained"] = subset is not None
                dec_entry["alt_expert_set_pcc"] = p_alt
                if subset is None:
                    failures.append(f"decode at {pos}: PCC {p_dec:.6f}, no explanation")
            else:
                failures.append(f"decode at {pos}: PCC {p_dec:.6f}")
        evidence["decode_max_position"] = dec_entry
        print(f"[{kind}] decode at pos {pos} (full 202k cache): PCC={p_dec:.6f} tie={bool(tie_d[0])}")
    finally:
        ttnn.close_device(device)

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / f"long_context_{kind}.json").write_text(json.dumps(evidence, indent=1))
    print(f"wrote {DOC_DIR / f'long_context_{kind}.json'}")
    assert not failures, failures


def test_full_context_aligned_202752(cfg):
    """Letter-of-the-contract check: prefill at exactly S=202752 (the aligned
    advertised maximum; every row is a real token). Validates the cache and the
    final 32 output rows (positions 202720..202751) with the same per-row
    routing-flip proof as the 202751 run."""
    S = FULL_CONTEXT
    sd = utils.synth_layer_state_dict(cfg, 1)
    layer = utils.build_hf_layer(cfg, 1, sd)
    x = utils.synth_activations(cfg, 1, S, seed=7)
    failures = []
    evidence = {"kind": "moe", "prefill_seq_len": S}

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    try:
        paged = PagedCacheConfig.for_context(S, 1)
        dec = DecoderCls.from_state_dict(
            sd,
            hf_config=cfg,
            layer_idx=1,
            mesh_device=device,
            max_batch_size=1,
            max_context=S,
            paged_config=paged,
            prefill_chunk_size=2048,
        )
        cache = dec.allocate_kv_cache()
        pt_torch = utils.make_page_table(1, paged.max_num_blocks, seed=3)
        pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        x_tt = ttnn.from_torch(x.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        t0 = time.perf_counter()
        out = dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S)
        ttnn.synchronize_device(device)
        wall = time.perf_counter() - t0
        evidence["prefill_wall_s"] = round(wall, 1)
        print(f"[moe] aligned full-context prefill S={S}: {wall:.0f}s ({S / wall:.0f} tok/s)")
        got = ttnn.to_torch(out).float()[0, 0]
        assert got.shape[0] == S

        kvpe_ref = utils.torch_latent_cache_reference(cfg, sd, x[0])
        cache_torch = ttnn.to_torch(cache).float()
        cache_rows = utils.gather_user_cache(cache_torch, pt_torch, 0, S, paged.block_size)
        del cache_torch
        p_cache = utils.pcc(kvpe_ref, cache_rows)
        evidence["cache_pcc_vs_linear_ref"] = p_cache
        print(f"[moe] aligned 202752 cache PCC: {p_cache:.6f}")
        if p_cache < 0.999:
            failures.append(f"cache PCC {p_cache}")

        rows = list(range(S - 32, S))
        ref_rows, tie, res_rows, h2_rows = utils.torch_absorbed_window_reference(
            cfg, sd, layer, x[0], kvpe_ref, rows, return_parts=True
        )
        row_pcc = [utils.pcc(ref_rows[i], got[rows[i]]) for i in range(32)]
        below = [i for i in range(32) if row_pcc[i] < PCC_BAR]
        explained, unexplained = [], []
        for i in below:
            p_alt, subset = utils.explain_row_as_routing_flip(cfg, sd, h2_rows[i], res_rows[i], got[rows[i]], PCC_BAR)
            entry = {"row": rows[i], "pcc": row_pcc[i], "tie": bool(tie[i])}
            if subset is not None:
                explained.append({**entry, "alt_expert_set_pcc": p_alt})
            else:
                unexplained.append({**entry, "best_alt_pcc": p_alt})
        evidence["final_window"] = {
            "rows": [S - 32, S - 1],
            "pcc_all_rows": utils.pcc(ref_rows, got[rows]),
            "rows_at_bar": 32 - len(below),
            "explained": explained,
            "unexplained": unexplained,
        }
        print(
            f"[moe] final window rows {S-32}..{S-1}: agg={evidence['final_window']['pcc_all_rows']:.6f} "
            f"rows>=bar={32-len(below)}/32 explained={len(explained)} unexplained={len(unexplained)}"
        )
        if unexplained:
            failures.append(f"unexplained rows {unexplained}")
        if 32 - len(below) < 26:
            failures.append(f"only {32 - len(below)}/32 rows at bar")
    finally:
        ttnn.close_device(device)

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "long_context_aligned_202752.json").write_text(json.dumps(evidence, indent=1))
    print(f"wrote {DOC_DIR / 'long_context_aligned_202752.json'}")
    assert not failures, failures
