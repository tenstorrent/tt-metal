# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated repro of the vLLM adapter decode call-sequence WITHOUT the server, on the reduced
[0,1,4] model. Mimics exactly what TTModelRunner/AsyncDecodeController do: build model,
allocate_kv_cache (vLLM-owned), warmup_model_decode (pre-capture trace), prefill_forward (device
sampling), then a decode_forward loop with padded batch + reset_batch + per-step page tables. Prints
each phase so a hang/crash is localized.

Run:
  cd /tmp && TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
    PYTHONPATH=/home/ttuser/dev/tt-metal:/home/ttuser/.local/lib/model-bringup/tt-metal/vllm \
    /home/ttuser/.tenstorrent-venv/bin/python -u -m \
    models.autoports.poolside_laguna_xs_2_1.doc.vllm_integration.scripts.adapter_repro --batch 32
"""
from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch

import ttnn


def sp(batch, temperature=0.0, top_k=0, top_p=1.0, seed=None, n_active=None):
    """Build a TTSamplingParams-like object (lists) for `batch` rows. Padded rows (>= n_active) get
    neutral greedy defaults (matches _sampling_params_for_padded_decode)."""
    n_active = batch if n_active is None else n_active
    return SimpleNamespace(
        temperature=[temperature if i < n_active else 0.0 for i in range(batch)],
        top_k=[top_k if i < n_active else 0 for i in range(batch)],
        top_p=[top_p if i < n_active else 1.0 for i in range(batch)],
        seed=[seed for _ in range(batch)],
        num_logprobs=None,
        enable_log_probs=torch.zeros(batch, dtype=torch.bool),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=17)  # non-aligned
    ap.add_argument("--gen", type=int, default=12)
    ap.add_argument("--layers", type=str, default="0,1,4")
    ap.add_argument(
        "--interleave",
        action="store_true",
        help="Interleave new-request prefills between decode steps (continuous-batching hazard).",
    )
    ap.add_argument(
        "--server-path",
        action="store_true",
        help="Use the server sync path: decode_forward(read_from_device=False) + "
        "process_decode_output_host (non-blocking execute_trace), instead of "
        "read_from_device=True (blocking).",
    )
    args = ap.parse_args()

    def do_decode(model, cur_tok, positions, page_table, kv, sampling_params, reset):
        if args.server_path:
            dev = model.decode_forward(
                cur_tok,
                positions,
                page_table=page_table,
                kv_cache=kv,
                enable_trace=True,
                read_from_device=False,
                sampling_params=sampling_params,
                reset_batch=reset,
            )
            host = model.process_decode_output_host(dev, is_tokens=True)
            return torch.as_tensor(host).reshape(-1)
        out = model.decode_forward(
            cur_tok,
            positions,
            page_table=page_table,
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=sampling_params,
            reset_batch=reset,
        )
        return torch.as_tensor(out).reshape(-1)

    from transformers import AutoConfig

    from models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm import LagunaForCausalLM

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=400_000_000)
    try:
        hf = AutoConfig.from_pretrained("poolside/Laguna-XS-2.1", trust_remote_code=True)
        n_layers = [int(x) for x in args.layers.split(",")]
        print(f"[phase] building model layers={n_layers}", flush=True)
        model = LagunaForCausalLM.initialize_vllm_model(
            hf, mesh, max_batch_size=args.batch, max_seq_len=args.max_model_len, n_layers=n_layers
        )
        B = args.batch
        bs = args.block_size
        nblocks = math.ceil(args.max_model_len / bs)
        num_gpu_blocks = B * nblocks + 16
        print(f"[phase] allocate_kv_cache blocks={num_gpu_blocks} shape=({num_gpu_blocks},2,{bs},128)", flush=True)
        kv = model.allocate_kv_cache((num_gpu_blocks, 2, bs, 128), torch.bfloat16, len(n_layers))

        # Mimic the plugin's TWO-PHASE warmup exactly (model_runner.py): Phase 1 compiles all
        # prefill+decode ops with NO trace; Phase 2 re-runs prefill warmup (allocating persistent
        # prefill buffers, incl. the serving-width page table now that decode warmup has recorded
        # max_num_blocks_per_req) and THEN captures the decode trace. All prefill allocation therefore
        # happens before any resident trace exists, so serving-time prefill is allocation-free.
        print("[phase] warmup phase-1 (compile, no trace)", flush=True)
        model.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
        model.warmup_model_decode(
            kv_cache=kv, enable_trace=False, max_batch_size=B, num_blocks=nblocks, can_sample_on_device=True
        )
        model.already_warmed_up_prefill = False  # plugin resets this between phases
        print("[phase] warmup phase-2 (prefill buffers pre-trace, then capture decode trace)", flush=True)
        model.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
        model.warmup_model_decode(
            kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=nblocks, can_sample_on_device=True
        )
        ttnn.synchronize_device(mesh)
        print("[phase] warmup done", flush=True)

        # ---- prefill user 0 with a non-aligned prompt ----
        P = args.prompt_len
        torch.manual_seed(0)
        prompt = torch.randint(0, model.vocab, (P,), dtype=torch.int64)
        # user 0 gets blocks [0, nblocks); build a [1, nblocks] page table.
        user0_blocks = list(range(0, nblocks))
        pt_prefill = torch.tensor([user0_blocks], dtype=torch.int32)
        print(f"[phase] prefill_forward P={P} (device sampling)", flush=True)
        first = model.prefill_forward(
            prompt.reshape(1, P),
            page_table=pt_prefill,
            kv_cache=kv,
            prompt_lens=[P],
            start_pos=[0],
            sampling_params=sp(1, n_active=1),
        )
        tok0 = int(first[0].reshape(-1)[0]) if isinstance(first, tuple) else int(first[0].reshape(-1)[0])
        if isinstance(first, tuple):
            tok0 = int(first[0].reshape(-1)[0])
        print(f"[ok] prefill sampled token0={tok0}", flush=True)

        # ---- decode loop: row 0 active, rows 1..B-1 padded (pos -1). page table [B, nblocks]. ----
        pt_decode = torch.zeros((B, nblocks), dtype=torch.int32)
        pt_decode[0] = torch.tensor(user0_blocks, dtype=torch.int32)
        toks = [tok0]
        cur_tok = torch.zeros((B, 1), dtype=torch.int64)
        cur_tok[0, 0] = tok0
        positions = torch.full((B,), -1, dtype=torch.int32)
        pos0 = P
        gen_params = sp(B, n_active=1)  # greedy
        for step in range(args.gen):
            positions[0] = pos0
            reset = step == 0  # first decode after prefill resets layout
            print(f"[phase] decode step={step} pos0={pos0} reset={reset} server_path={args.server_path}", flush=True)
            out = do_decode(model, cur_tok, positions, pt_decode, kv, gen_params, reset)
            t = int(out[0])
            toks.append(t)
            cur_tok[0, 0] = t  # vLLM would pass this back (stale-safe: device already fed it)
            pos0 += 1
            print(f"[ok] decode step={step} -> token={t}", flush=True)

        print("REPRO_TOKENS", toks, flush=True)
        # crude degeneracy check
        distinct = len(set(toks))
        print("REPRO_RESULT", "OK" if distinct > 2 else "DEGENERATE", "distinct=", distinct, flush=True)

        # ---- mixed-params multi-active-row decode (mimics test_mixed_params_batch device path) ----
        # Prefill 3 more users into disjoint block ranges, then decode 4 active rows with DIFFERENT
        # per-row sampling params (greedy / top_k / top_p+temp / different seeds), several steps.
        n_act = 4
        assert nblocks * n_act + 1 <= num_gpu_blocks
        pt_multi = torch.zeros((B, nblocks), dtype=torch.int32)
        starts_pos = []
        cur_tok_m = torch.zeros((B, 1), dtype=torch.int64)
        for u in range(n_act):
            blocks = list(range(u * nblocks, (u + 1) * nblocks))
            pt_multi[u] = torch.tensor(blocks, dtype=torch.int32)
            Pu = 13 + u * 5  # 13,18,23,28 — non-aligned, distinct lengths
            torch.manual_seed(100 + u)
            promptu = torch.randint(0, model.vocab, (Pu,), dtype=torch.int64)
            fu = model.prefill_forward(
                promptu.reshape(1, Pu),
                page_table=torch.tensor([blocks], dtype=torch.int32),
                kv_cache=kv,
                prompt_lens=[Pu],
                start_pos=[0],
                sampling_params=sp(1, n_active=1),
            )
            t0 = int(fu[0].reshape(-1)[0]) if isinstance(fu, tuple) else int(fu.reshape(-1)[0])
            cur_tok_m[u, 0] = t0
            starts_pos.append(Pu)
            print(f"[ok] multi prefill user{u} Pu={Pu} tok={t0}", flush=True)
        # mixed params: row0 greedy, row1 top_k=5, row2 top_p=0.9 temp=0.8, row3 top_k=20 temp=0.7 seed=7
        mixed = SimpleNamespace(
            temperature=[0.0, 1.0, 0.8, 0.7] + [0.0] * (B - n_act),
            top_k=[0, 5, 0, 20] + [0] * (B - n_act),
            top_p=[1.0, 1.0, 0.9, 1.0] + [1.0] * (B - n_act),
            seed=[None, 1, 2, 7] + [None] * (B - n_act),
            num_logprobs=None,
            enable_log_probs=torch.zeros(B, dtype=torch.bool),
        )
        positions_m = torch.full((B,), -1, dtype=torch.int32)
        multi_out = {u: [] for u in range(n_act)}
        for step in range(8):
            for u in range(n_act):
                positions_m[u] = starts_pos[u] + step
            reset = step == 0
            print(f"[phase] mixed decode step={step} reset={reset}", flush=True)
            outt = do_decode(model, cur_tok_m, positions_m, pt_multi, kv, mixed, reset)
            for u in range(n_act):
                tu = int(outt[u])
                multi_out[u].append(tu)
                cur_tok_m[u, 0] = tu
            print(f"[ok] mixed step={step} row_tokens={[multi_out[u][-1] for u in range(n_act)]}", flush=True)
        print("REPRO_MIXED", {u: multi_out[u] for u in range(n_act)}, flush=True)
        # row0 is greedy → should be deterministic; rows differ from row0 given different params usually
        print("REPRO_MIXED_RESULT", "OK", flush=True)

        if args.interleave:
            # ---- interleaved prefill DURING active decode-trace serving (continuous batching) ----
            # Reuse block ranges from the multi phase. Admit users one at a time: prefill user k
            # (NEW buffer allocation while the decode trace exists), then decode all admitted rows a
            # few steps with reset_batch=True (layout changed). This is the exact hazard the
            # "Allocating device buffers unsafe due to active trace" warning flags.
            print("[phase] INTERLEAVE begin", flush=True)
            admitted = 0
            pos_by_row = {}
            cur_tok_i = torch.zeros((B, 1), dtype=torch.int64)
            pt_i = torch.zeros((B, nblocks), dtype=torch.int32)
            positions_i = torch.full((B,), -1, dtype=torch.int32)
            gp = sp(B)  # greedy for all
            for k in range(6):
                blocks = list(range(k * nblocks, (k + 1) * nblocks))
                pt_i[k] = torch.tensor(blocks, dtype=torch.int32)
                Pk = 11 + k * 7
                torch.manual_seed(500 + k)
                pk = torch.randint(0, model.vocab, (Pk,), dtype=torch.int64)
                print(f"[phase] INTERLEAVE prefill new user{k} Pk={Pk} (alloc during active trace)", flush=True)
                fk = model.prefill_forward(
                    pk.reshape(1, Pk),
                    page_table=torch.tensor([blocks], dtype=torch.int32),
                    kv_cache=kv,
                    prompt_lens=[Pk],
                    start_pos=[0],
                    sampling_params=sp(1, n_active=1),
                )
                t0 = int(fk[0].reshape(-1)[0]) if isinstance(fk, tuple) else int(fk.reshape(-1)[0])
                cur_tok_i[k, 0] = t0
                pos_by_row[k] = Pk
                admitted += 1
                print(f"[ok] INTERLEAVE prefilled user{k} tok={t0}; decoding {admitted} active rows", flush=True)
                for s in range(3):
                    for u in range(admitted):
                        positions_i[u] = pos_by_row[u]
                    reset = s == 0  # layout changed (new row admitted)
                    o = do_decode(model, cur_tok_i, positions_i, pt_i, kv, gp, reset)
                    for u in range(admitted):
                        tu = int(o[u])
                        cur_tok_i[u, 0] = tu
                        pos_by_row[u] += 1
                    print(f"[ok] INTERLEAVE decode after user{k} step{s} row0_tok={int(o[0])}", flush=True)
            print("REPRO_INTERLEAVE_RESULT OK", flush=True)
    finally:
        try:
            model.gen.teardown()
        except Exception:
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    import os as _os
    import sys as _sys

    _sys.stdout.flush()
    _os._exit(0)
