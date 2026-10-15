# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel prefill (tt/dp_prefill.py, GPT_OSS_DP_PREFILL=1) vs today's TP batch-1 prefill, Blackhole 1x4.

S1 test_dp_attention_kv_handoff: one layer's DP attention + K/V handoff (layer 0 sliding window, layer 23 full) for
   4 users (user c on device c), inputs = the users' real layer-0 input (text embedding -> layer-0 input norm). Per user: the
   DP attention output vs the TP attention of that user alone (PCC >= ATTN_MIN_PCC), and the user's TP cache pages
   (K and V, all 8 kv heads, through the true page table) written by the DP handoff vs those the TP batch-1 prefill
   writes (PCC >= KV_MIN_PCC, non-zero). mode: exact (TP's o_proj + sum, dp_prefill.tp_exact_sum) or local (one
   full-width o_proj, no collective). plant=pt_swap swaps page-table rows 0 and 1 of the DP call only: the KV
   check must FAIL (the test asserts that it does).
S2 test_dp_prefill_model: the full model (prefill_common.build_prefill_generator_b32, KV for 32 x max L) with the DP
   stack; Model.dp_prefill of the 32 b32 text prompts vs a TP batch-1 Generator.prefill_forward_text of each checked
   user: top-1 equal, logits PCC >= LOGIT_MIN_PCC, KV pages of layers 0 and 23 PCC >= KV_MIN_PCC. Same plant.
S2-perf test_dp_prefill_perf: warm wall time of Model.dp_prefill for 32 users (N=1, device synced) per length, and
   the DRAM view after the build. Results to DP_PREFILL_RESULT (JSON, one key per case) if set.

Run (sjc2-t3020, after baseline/env.sh): pytest models/demos/gpt_oss/tests/unit/test_dp_prefill.py -k "<test> and 1x4"
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ...tt.attention import Attention, AttentionConfig
from ...tt.attention_configs import GPTOSSAttentionProgramConfig
from ...tt.batched_prefill import batched_reduce_links, fill_batch_idx
from ...tt.dp_prefill import DP_PREFILL_ENV, dp_attention_prefill, dp_attention_weights, dp_prefill_mesh_supported
from ...tt.model import create_rope_setup
from ...tt.rms_norm import RMSNorm
from ..test_factory import TestFactory, parametrize_mesh_with_fabric

ATTN_MIN_PCC = 0.999
KV_MIN_PCC = 0.999
LOGIT_MIN_PCC = 0.99
BLOCK = 64
N_DEV = 4


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def _record(key, value):
    path = os.environ.get("DP_PREFILL_RESULT")
    if not path:
        return
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))


def _skip_unless_bh_1x4(mesh_device):
    if not dp_prefill_mesh_supported(mesh_device):
        pytest.skip("DP prefill runs on Blackhole 1x4")


def _read_pages(mesh_device, cache, page_table, users, seq_len):
    """{u: [8 kv heads, L, 64]} of a TP paged cache ([blocks, 2, 64, 64] per device, device d = heads 2d, 2d + 1)."""
    full = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).float()
    nb = seq_len // BLOCK
    out = {}
    for u in users:
        pages = full[page_table[u, :nb].long()]  # [nb, 8, 64, 64]
        out[u] = pages.permute(1, 0, 2, 3).reshape(pages.shape[1], nb * BLOCK, -1)[:, :seq_len]
    return out


def _kv_fails(tag, kv_dp, kv_tp, metrics):
    fails = []
    for key in kv_tp:
        a, b = kv_dp[key], kv_tp[key]
        pcc = _pcc(a, b)
        nz = bool(a.abs().max() > 0) and bool(b.abs().max() > 0)
        metrics[f"kv_{key}"] = {"pcc": pcc, "bit_exact": bool(torch.equal(a, b)), "nonzero": nz}
        if not nz:
            fails.append(f"{tag} {key}: KV pages all zero")
        elif not pcc >= KV_MIN_PCC:
            fails.append(f"{tag} {key}: KV PCC {pcc:.6f} < {KV_MIN_PCC}")
    return fails


# ---- S1 -----------------------------------------------------------------------------------------------------------


@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize("plant", ["none", "pt_swap"])
@pytest.mark.parametrize("mode", ["exact", "local"])
@pytest.mark.parametrize("seq_len", [128, 1024, 8192], ids=["L128", "L1024", "L8192"])
@pytest.mark.parametrize("layer_idx", [0, 23], ids=["layer0_sliding", "layer23_full"])
def test_dp_attention_kv_handoff(mesh_device, device_params, layer_idx, seq_len, mode, plant, reset_seeds):
    _skip_unless_bh_1x4(mesh_device)
    from ..perf.prefill_common import b32_prompt_tokens

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    hf_config = setup["config"]
    mesh_config = setup["mesh_config"]
    ccl_manager = setup["ccl_manager"]
    cache = Path(setup["model_args"].weight_cache_path(ttnn.bfloat8_b))
    layer_cache = cache / f"model.layers.{layer_idx}"
    assert (layer_cache / "self_attn").is_dir(), f"needs the TP weight cache at {layer_cache}"
    H = hf_config.hidden_size

    # The TP layer's attention + input norm, from the weight cache (no HF load, no cache writes).
    rope_setup = create_rope_setup(mesh_device, hf_config)
    tmats = rope_setup.get_both_trans_mats()
    attn_config = AttentionConfig(
        hidden_size=H,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.head_dim,
        sliding_window=hf_config.sliding_window,
        max_seq_len=1024,
        max_local_batch_size=1,
    )
    attn = Attention(
        mesh_device=mesh_device,
        config=attn_config,
        state_dict={},
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=GPTOSSAttentionProgramConfig(),
        layer_idx=layer_idx,
        transformation_mats=tmats,
        tensor_cache_path=str(layer_cache / "self_attn"),
        create_kv_cache=False,
    )
    assert (attn.config.sliding_window is not None) == (layer_idx % 2 == 0)
    # Input: the users' real layer-0 input (embedding -> layer-0 input norm), for both layers. The layer-23 input
    # norm applied to a raw embedding gives extreme activations (o_proj output RMS ~2e4) on which the SDPA op itself
    # is only 0.98 (16 heads / device) to 0.997 (64 heads / device) vs fp32 torch, so it cannot tell a mapping error
    # from kernel rounding (dp/logs/s1_diag_layer23.txt); layer 23 on real activations is checked by S2.
    norm = RMSNorm(
        mesh_device,
        hf_config,
        {},
        tensor_cache_path=str(cache / "model.layers.0" / "input_layernorm"),
        mesh_config=mesh_config,
    )
    emb = ttnn.as_tensor(
        None,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        cache_file_name=str(cache / "model.embed_tokens.weight"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dp_w = dp_attention_weights(attn.weights, attn.config, mesh_device)
    rope = [rope_setup.cos_matrix_prefill[:, :, :seq_len, :], rope_setup.sin_matrix_prefill[:, :, :seq_len, :]]

    # Paged TP cache for the 4 users: blocks shuffled, user u = page-table row u.
    nb = seq_len // BLOCK
    torch.manual_seed(0)
    num_blocks = N_DEV * nb + 8
    page_table = torch.randperm(num_blocks)[: N_DEV * nb].reshape(N_DEV, nb).to(torch.int32)
    kv = [
        ttnn.from_torch(
            torch.zeros(num_blocks, hf_config.num_key_value_heads // N_DEV, BLOCK, hf_config.head_dim),
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(2)
    ]

    def zero_kv():
        for t in kv:
            ttnn.mul(t, 0, output_tensor=t)

    def snapshot():
        return {
            (u, name): pages
            for name, t in zip("kv", kv)
            for u, pages in _read_pages(mesh_device, t, page_table, range(N_DEV), seq_len).items()
        }

    tokens = b32_prompt_tokens(seq_len, "text")[:N_DEV]

    def normed_input(tok, mapper):
        tt = ttnn.from_torch(
            tok.to(torch.int32), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
        )
        e = ttnn.embedding(tt, emb, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        tt.deallocate(True)
        if len(e.shape) == 3:
            e = ttnn.unsqueeze_to_4D(e)
        x = norm(e)
        e.deallocate(True)
        return x

    def pt_tt(rows):
        return ttnn.from_torch(
            rows,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # TP batch-1 reference, one user at a time (disjoint pages, so one snapshot holds all 4).
    zero_kv()
    tp_out = {}
    for u in range(N_DEV):
        x = normed_input(tokens[u].reshape(1, 1, 1, seq_len), ttnn.ReplicateTensorToMesh(mesh_device))
        ptu = pt_tt(page_table[u : u + 1])
        o = attn(x, rope, page_table=ptu, kv_cache=kv, is_decode=False, user_id=0)
        tp_out[u] = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(seq_len, H)
        o.deallocate(True)
        ptu.deallocate(True)
    kv_tp = snapshot()

    # DP: user c on device c, one call for the 4 users.
    zero_kv()
    pt_dp = page_table.clone()
    if plant == "pt_swap":
        pt_dp[[0, 1]] = pt_dp[[1, 0]]
    x = normed_input(
        tokens.reshape(N_DEV, 1, 1, seq_len),
        ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_device.shape),
    )
    ptd = pt_tt(pt_dp)
    links = batched_reduce_links(ccl_manager, 1)
    o = dp_attention_prefill(
        x, rope, dp_w, attn, kv, ptd, fill_batch_idx(ccl_manager, N_DEV), links, exact=(mode == "exact")
    )
    dp_out = [ttnn.to_torch(t).float().reshape(seq_len, H) for t in ttnn.get_device_tensors(o)]
    o.deallocate(True)
    kv_dp = snapshot()

    fails = []
    result = {"layer": layer_idx, "L": seq_len, "mode": mode, "plant": plant, "a2a_links": links, "users": {}}
    for u in range(N_DEV):
        m = {"attn_pcc": _pcc(dp_out[u], tp_out[u]), "attn_bit_exact": bool(torch.equal(dp_out[u], tp_out[u]))}
        if not m["attn_pcc"] >= ATTN_MIN_PCC:
            fails.append(f"user {u}: attention PCC {m['attn_pcc']:.6f} < {ATTN_MIN_PCC}")
        fails += _kv_fails(
            f"user {u}",
            {k[1]: v for k, v in kv_dp.items() if k[0] == u},
            {k[1]: v for k, v in kv_tp.items() if k[0] == u},
            m,
        )
        result["users"][u] = m
        logger.info(f"[S1] layer {layer_idx} L={seq_len} mode={mode} plant={plant} user {u}: {m}")
    result["fails"] = fails
    _record(f"s1_layer{layer_idx}_L{seq_len}_{mode}_{plant}", result)
    if plant == "none":
        assert not fails, "S1 FAIL:\n" + "\n".join(fails)
    else:
        # The plant must bite: users 0 and 1 wrote each other's pages, so their KV checks fail.
        assert any(f.startswith("user 0") for f in fails) and any(f.startswith("user 1") for f in fails), fails
        logger.info(f"[S1] plant {plant} bites: {fails}")


# ---- S2 -----------------------------------------------------------------------------------------------------------

S2_USERS = [0, 1, 17, 31]
S2_KV_LAYERS = [0, 23]


def _build_dp_model(mesh_device, kv_len, monkeypatch):
    from ..perf.prefill_common import build_prefill_generator_b32, dram_view

    monkeypatch.setenv(DP_PREFILL_ENV, "1")
    t0 = time.perf_counter()
    ctx = build_prefill_generator_b32(mesh_device, kv_len)
    ctx["build_s"] = time.perf_counter() - t0
    ttnn.synchronize_device(mesh_device)
    ctx["dram_after_build"] = dram_view(mesh_device)
    model = ctx["model"]
    ctx["tt_model"] = model[0] if isinstance(model, (list, tuple)) else model  # one model per data-parallel group
    assert ctx["tt_model"].dp_prefill_stack is not None
    return ctx


def _dp_call(ctx, tokens, plant=None):
    model = ctx["tt_model"]
    pt = ctx["page_table"]
    hook = None
    if plant == "pt_swap":
        # users 0 and 1 run in step 0 on devices 0 and 1: swap their page-table rows there.
        def hook(step, rows):
            if step == 0:
                rows = rows.clone()
                rows[[0, 1]] = rows[[1, 0]]
            return rows

    return model.dp_prefill_stack.prefill(tokens, pt, ctx["kv_cache"][0], page_table_plant=hook)


@pytest.mark.timeout(3 * 3600)
@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize("plant", ["none", "pt_swap"])
def test_dp_prefill_model(mesh_device, device_params, plant, monkeypatch):
    _skip_unless_bh_1x4(mesh_device)
    from ..perf.prefill_common import b32_prompt_tokens, compare_logits, run_prefill_batch

    lengths = [int(v) for v in os.getenv("DP_PREFILL_S2_LENGTHS", "1024,8192").split(",")]
    ctx = _build_dp_model(mesh_device, max(lengths), monkeypatch)
    pt = ctx["page_table"]
    layers_kv = ctx["kv_cache"][0]

    def zero_kv():
        for k, v in layers_kv:
            ttnn.mul(k, 0, output_tensor=k)
            ttnn.mul(v, 0, output_tensor=v)
        ttnn.synchronize_device(mesh_device)

    def read_kv(users, L):
        out = {u: {} for u in users}
        for layer in S2_KV_LAYERS:
            for name, t in zip("kv", layers_kv[layer]):
                for u, pages in _read_pages(mesh_device, t, pt, users, L).items():
                    out[u][f"L{layer}_{name}"] = pages
        return out

    fails = []
    report = {"plant": plant, "build_s": ctx["build_s"], "kv_len_per_user": ctx["kv_len_per_user"], "lengths": {}}
    for L in lengths:
        tokens = b32_prompt_tokens(L, "text")
        zero_kv()
        _dp_call(ctx, tokens, plant)  # compile pass for this length
        zero_kv()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        logits_dp = _dp_call(ctx, tokens, plant)
        ttnn.synchronize_device(mesh_device)
        dp_ms = (time.perf_counter() - t0) * 1e3
        kv_dp = read_kv(S2_USERS, L)
        entry = {"dp_ms_32users": dp_ms, "users": {}}
        for u in S2_USERS:
            zero_kv()
            _, logits1 = run_prefill_batch(ctx, tokens[u : u + 1], page_table=pt[u : u + 1])
            kv1 = read_kv([u], L)[u]
            m = compare_logits(logits_dp[u], logits1[0])
            m["bit_exact"] = bool(torch.equal(logits_dp[u], logits1[0]))
            if not m["top1_match"]:
                fails.append(f"L={L} user {u}: top1 DP {m['cur_top1']} != TP b1 {m['ref_top1']}")
            if not m["pcc"] >= LOGIT_MIN_PCC:
                fails.append(f"L={L} user {u}: logits PCC {m['pcc']:.5f} < {LOGIT_MIN_PCC}")
            fails += _kv_fails(f"L={L} user {u}", kv_dp[u], kv1, m)
            entry["users"][str(u)] = m
            logger.info(
                f"[S2] L={L} plant={plant} u={u}: logits pcc={m['pcc']:.6f} top1 {m['cur_top1']}/{m['ref_top1']} | "
                + " ".join(f"{k}:{v['pcc']:.6f}" for k, v in m.items() if k.startswith("kv_"))
            )
        report["lengths"][str(L)] = entry
    report["fails"] = fails
    _record(f"s2_model_{plant}", report)
    if plant == "none":
        assert not fails, "S2 FAIL:\n" + "\n".join(fails)
    else:
        assert any("user 0" in f for f in fails) and any("user 1" in f for f in fails), fails
        logger.info(f"[S2] plant {plant} bites: {fails}")


@pytest.mark.timeout(3 * 3600)
@parametrize_mesh_with_fabric([(1, 4)])
def test_dp_prefill_perf(mesh_device, device_params, monkeypatch):
    _skip_unless_bh_1x4(mesh_device)
    from ..perf.prefill_common import b32_prompt_tokens, dram_view

    lengths = [int(v) for v in os.getenv("DP_PREFILL_PERF_LENGTHS", "2048,4096,8192,16384").split(",")]
    kv_len = int(os.getenv("DP_PREFILL_PERF_KV_LEN", str(max(lengths))))
    ctx = _build_dp_model(mesh_device, kv_len, monkeypatch)
    result = {
        "kv_len_per_user": kv_len,
        "kv_bytes_per_device": ctx["kv_bytes_per_device"],
        "build_s": ctx["build_s"],
        "dram_after_build": ctx["dram_after_build"],
        "lengths": {},
    }
    logger.info(f"[S2-perf] DRAM after build (KV 32 x {kv_len}): {ctx['dram_after_build']}")
    for L in lengths:
        tokens = b32_prompt_tokens(L, "text")
        _dp_call(ctx, tokens)  # compile
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        _dp_call(ctx, tokens)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) * 1e3
        result["lengths"][str(L)] = {"dp_ms_32users": ms, "dram_after_run": dram_view(mesh_device)}
        logger.info(f"[S2-perf] L={L}: dp_prefill 32 users {ms:.1f} ms")
        _record(f"s2_perf_kv{kv_len}", result)
