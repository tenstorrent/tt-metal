# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 end-to-end through the GENERATOR: looped per-user prefill -> single B=32 decode pass.

Architecture under test: prefill is LOOPED per-user (generator.prefill_forward_text iterates
empty_slots, writing each user's paged KV + GDN/conv state to its own row); decode is a SINGLE
B=32 forward pass (generator.decode_forward(batch_size=32) -> ttnn_decode_forward(batch_size=32) ->
forward(batch_size=32), one traced replay producing all 32 users' logits at once).

Oracle (CPU-free, correct by construction): prefill 32 IDENTICAL users -> a single B=32 decode MUST
produce IDENTICAL logits for all 32 rows. Any per-row divergence is a real batch bug (KV/GDN cross-talk,
the full-attn row-collapse at batch_size==1, or a trace-key collision). This exercises the generator
threading added for batch-32 (the GDN recurrent kernel itself is covered by
test_gdn_recurrent_batch32_micro.py; the model.forward B=32 path by test_decode_batch32_smoke.py).

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    QWEN36_N_LAYERS=8 python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_batch32_prefill_decode_e2e.py -v -s
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn

_N_USERS = int(os.environ.get("QWEN36_E2E_USERS", "32"))
_STEPS = int(os.environ.get("QWEN36_E2E_STEPS", "4"))


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200_000_000)
    yield mesh
    # ROBUST TEARDOWN: on a test error, close_mesh_device can itself throw
    # ("SubDeviceManagerTracker not initialized / remote devices only"). If that
    # propagates, set_fabric_config(DISABLED) never runs -> the fabric is left
    # ENABLED/dirty -> the NEXT process inherits a bad fabric (ethernet-core
    # timeout / IndexError / hang). Always disable the fabric, even if close fails.
    try:
        ttnn.close_mesh_device(mesh)
    except Exception as _e:
        print(f"[teardown] close_mesh_device raised (continuing to disable fabric): {_e}", flush=True)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.hardware
def test_batch32_prefill_decode_identical_users(bh_glx_mesh):
    from models.common.utility_functions import comp_pcc
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
    from models.demos.qwen3_6_galaxy_v2.demo import text_demo_qwen36 as D
    from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
    from models.demos.qwen3_6_galaxy_v2.tt.generator_vllm import allocate_vllm_kv_cache

    # Known-good decode-CCL/perf config (same as the batch-1 generator demo).
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FULLATTN_WO_TUNED": "1",
        "QWEN36_DELTA_OP_TUNED": "1",
        "QWEN36_CCL_NUM_LINKS_DELTA": "2",
    }.items():
        os.environ.setdefault(_k, _v)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(D._SNAPSHOT), trust_remote_code=True)
    prompt = D._load_prompt_for_isl(D._T_PREFILL)
    ids = tok(prompt, return_tensors="pt").input_ids[:, : D._T_PREFILL].to(torch.long)
    T = int(ids.shape[-1])
    print(f"\n[b32-e2e] users={_N_USERS} ISL={T} layers={D._N_LAYERS}")

    state_dict = D._load_full_state_dict(D._SNAPSHOT)
    pac = PagedAttentionConfig(block_size=D._PAGED_BLOCK_SIZE, max_num_blocks=D._PAGED_MAX_NUM_BLOCKS)
    # round max_num_blocks to a multiple of N_USERS for the [N, nblk/N] page-table reshape
    if pac.max_num_blocks % _N_USERS != 0:
        import math as _m

        pac.max_num_blocks = int(_m.ceil(pac.max_num_blocks / _N_USERS) * _N_USERS)
    model, args = D._build_tt_model_paged_kv(
        bh_glx_mesh, state_dict, D._PATTERN, D._N_LAYERS, pac, max_batch_size=_N_USERS
    )

    permutation = torch.randperm(pac.max_num_blocks)
    page_table = torch.argsort(permutation).reshape(args.max_batch_size, pac.max_num_blocks // args.max_batch_size)

    _kv_shape = (pac.max_num_blocks, 1, pac.block_size, args.head_dim)
    tt_kv_cache = allocate_vllm_kv_cache(
        _kv_shape, torch.bfloat16, args.n_layers, model, args.weight_cache_path(ttnn.bfloat8_b)
    )

    generator = Generator(model, args, bh_glx_mesh, tokenizer=tok)
    generator._disable_prefill_tracing = True
    generator.prefill_warmup_completed = True

    # ---- LOOPED per-user prefill: N IDENTICAL users (same prompt) -> each fills its own KV+GDN row ----
    tokens_NB = ids.repeat(_N_USERS, 1)  # [N, T] identical rows
    print("[b32-e2e] prefill_forward_text (looped per-user, return_logits) ...")
    prefill_logits = generator.prefill_forward_text(
        tokens_NB,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[T] * _N_USERS,
        empty_slots=list(range(_N_USERS)),
        enable_trace=False,
        sampling_params=None,
    )
    # First decode token = argmax of user 0's prefill logits (all users identical -> same token).
    pl = torch.as_tensor(prefill_logits).float().reshape(-1)[: args.vocab_size]
    first_tok = int(pl.argmax().item())
    print(f"[b32-e2e] first decode token = {first_tok} ({tok.decode([first_tok])!r})")

    # ---- SINGLE B=32 decode pass(es), host-sample (return_logits) so we can compare per-user rows ----
    cur = first_tok
    worst = 1.0
    for step in range(_STEPS):
        toks = torch.full((_N_USERS, 1), cur, dtype=torch.long)
        current_pos = torch.tensor([T + step] * _N_USERS, dtype=torch.long)
        out = generator.decode_forward(
            toks,
            current_pos,
            enable_trace=False,  # eager first; trace covered separately
            page_table=page_table,
            kv_cache=tt_kv_cache,
            read_from_device=True,
            sampling_params=None,  # return logits -> host compare
            reset_inputs=True,
            batch_size=_N_USERS,
        )
        logits = out[0] if isinstance(out, (tuple, list)) else out
        lt = torch.as_tensor(logits).float().reshape(-1)
        # The host-sampled return is per-user logits; reshape to [N, vocab] (trailing padded vocab).
        vocab = args.vocab_size
        n_full = lt.numel() // _N_USERS
        rows = lt.reshape(_N_USERS, n_full)[:, :vocab]
        row0 = rows[0:1]
        step_worst = 1.0
        for u in range(1, _N_USERS):
            ok, m = comp_pcc(row0, rows[u : u + 1], 0.99)
            mv = float(str(m).split()[-1]) if not isinstance(m, float) else float(m)
            step_worst = min(step_worst, mv)
        worst = min(worst, step_worst)
        nxt = int(rows[0].argmax().item())
        print(
            f"[b32-e2e] step {step}: all-rows-vs-row0 worst PCC={step_worst:.5f}  next_tok={nxt} ({tok.decode([nxt])!r})"
        )
        cur = nxt

    print(f"[b32-e2e] WORST per-user PCC over {_STEPS} steps = {worst:.5f}")
    assert (
        worst > 0.99
    ), f"batch-{_N_USERS} decode rows diverge (worst PCC {worst}) -> per-user cross-talk / row-collapse bug"
    print(f"[b32-e2e] PASS: single B={_N_USERS} decode pass isolates all {_N_USERS} users.")
