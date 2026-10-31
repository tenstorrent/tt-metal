# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Light batch-32 correctness check for GPT-OSS prefill on a 1x4 mesh. Exit code 0 = PASS, 1 = FAIL.

    python models/demos/gpt_oss/tests/perf/prefill_gate_b32.py --result <json>                 # 128, 1K, 8K
    python models/demos/gpt_oss/tests/perf/prefill_gate_b32.py --plant pt_swap --result <json>  # bite check: FAIL
    python models/demos/gpt_oss/tests/perf/prefill_gate_b32.py --plant out_swap --result <json> # bite check: FAIL

One model built for batch 32 (prefill_common.build_prefill_generator_b32, KV for 32 x max(lengths)). For each length
L: the 32 text prompts of the batch-32 harness (b32_prompt_tokens) go through ONE Generator.prefill_forward_text call;
then, for each checked user u in USERS, the same prompt goes through a batch-1 Generator.prefill_forward_text call on
the same model with that user's page-table row. The KV cache is zeroed before every call, so each snapshot holds only
what that call wrote. Checks per (L, u):
  logits: batch-32 last-token logits of user u vs the batch-1 logits: top-1 equal and PCC >= LOGIT_MIN_PCC.
  KV: layers KV_LAYERS, K and V, the pages of user u read through the TRUE page table (the first L / 64 entries of
      row u, all 8 kv heads gathered from the 4 devices): batch-32 vs batch-1 PCC >= KV_MIN_PCC and non-zero.
Bit-exactness of logits and KV is reported for every pair (not required).

--plant (bite proof, never for real runs): pt_swap swaps page-table rows 0 and 1 for the batch-32 call only, so users
0 and 1 write each other's pages (the KV check must fail); out_swap swaps the batch-32 logits rows of users 0 and 1
(the logits check must fail).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

USERS = [0, 1, 17, 31]
LENGTHS = [128, 1024, 8192]
KV_LAYERS = [0, 23]  # layer 0: sliding-window attention, layer 23: full attention
LOGIT_MIN_PCC = 0.99
KV_MIN_PCC = 0.999
PLANTS = ("none", "pt_swap", "out_swap")
REPO = Path(__file__).resolve().parents[5]


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


if __name__ != "__main__":
    import pytest

    from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

    @pytest.mark.timeout(2 * 3600)
    @parametrize_mesh_with_fabric([(1, 4)])
    def test_prefill_gate_b32(mesh_device, device_params):
        import torch
        from loguru import logger

        import ttnn
        from models.demos.gpt_oss.tests.perf.prefill_common import (
            B32_BATCH,
            PAGE_BLOCK_SIZE,
            b32_prompt_tokens,
            build_prefill_generator_b32,
            compare_logits,
            env_snapshot,
            git_info,
            parse_lengths,
            run_prefill_batch,
        )

        lengths = parse_lengths(os.getenv("PREFILL_GATE_B32_LENGTHS", ",".join(map(str, LENGTHS))))
        plant = os.getenv("PREFILL_GATE_B32_PLANT", "none")
        assert plant in PLANTS, plant
        result_path = os.getenv("PREFILL_GATE_B32_RESULT")
        os.environ["PREFILL_PROMPT"] = "text"

        t0 = time.perf_counter()
        ctx = build_prefill_generator_b32(mesh_device, max(lengths))
        build_s = time.perf_counter() - t0
        pt = ctx["page_table"]
        layers_kv = ctx["kv_cache"][0]  # per layer [k, v]

        def zero_kv():
            for k, v in layers_kv:
                ttnn.mul(k, 0, output_tensor=k)
                ttnn.mul(v, 0, output_tensor=v)
            ttnn.synchronize_device(mesh_device)

        def read_kv(users, L):
            """{u: {(layer, 'k'|'v'): [8 heads, L, 64]}} through the true page table rows."""
            nb = L // PAGE_BLOCK_SIZE
            out = {u: {} for u in users}
            for layer in KV_LAYERS:
                for name, t in zip("kv", layers_kv[layer]):
                    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
                    for u in users:
                        pages = full[pt[u, :nb].long()]  # [nb, 8, 64, 64]
                        out[u][(layer, name)] = pages.permute(1, 0, 2, 3).reshape(pages.shape[1], nb * 64, -1)[:, :L]
                    del full
            return out

        report = {
            "check": "models/demos/gpt_oss/tests/perf/prefill_gate_b32.py",
            "git": git_info(),
            "env": env_snapshot(),
            "plant": plant,
            "users": USERS,
            "kv_layers": KV_LAYERS,
            "thresholds": {"logit_min_pcc": LOGIT_MIN_PCC, "kv_min_pcc": KV_MIN_PCC, "top1": "equal"},
            "batch": B32_BATCH,
            "kv_len_per_user": ctx["kv_len_per_user"],
            "kv_dtype": ctx["kv_dtype"],
            "model_build_s": build_s,
            "lengths": {},
        }
        fails = []
        for L in lengths:
            t1 = time.perf_counter()
            assert L % PAGE_BLOCK_SIZE == 0
            tokens = b32_prompt_tokens(L, "text")
            pt_run = pt.clone()
            if plant == "pt_swap":
                pt_run[[0, 1]] = pt_run[[1, 0]]
            zero_kv()
            b32_s, logits32 = run_prefill_batch(ctx, tokens, page_table=pt_run)
            if plant == "out_swap":
                logits32[[0, 1]] = logits32[[1, 0]]
            kv32 = read_kv(USERS, L)
            entry = {"b32_prefill_s": b32_s, "users": {}}
            for u in USERS:
                zero_kv()
                _, logits1 = run_prefill_batch(ctx, tokens[u : u + 1], page_table=pt[u : u + 1])
                kv1 = read_kv([u], L)[u]
                m = compare_logits(logits32[u], logits1[0])
                m["bit_exact"] = bool(torch.equal(logits32[u], logits1[0]))
                ue = {"logits": m, "kv": {}}
                if not m["top1_match"]:
                    fails.append(f"L={L} user {u}: top1 b32 {m['cur_top1']} != b1 {m['ref_top1']}")
                if not m["pcc"] >= LOGIT_MIN_PCC:
                    fails.append(f"L={L} user {u}: logits PCC {m['pcc']:.5f} < {LOGIT_MIN_PCC}")
                for key, a in kv32[u].items():
                    b = kv1[key]
                    kname = f"L{key[0]}_{key[1]}"
                    pcc = _pcc(a, b)
                    nz = bool(a.abs().max() > 0) and bool(b.abs().max() > 0)
                    ue["kv"][kname] = {
                        "pcc": pcc,
                        "bit_exact": bool(torch.equal(a, b)),
                        "max_abs_diff": float((a.float() - b.float()).abs().max()),
                        "nonzero": nz,
                    }
                    if not nz:
                        fails.append(f"L={L} user {u} {kname}: KV pages all zero (b32 or b1)")
                    elif not pcc >= KV_MIN_PCC:
                        fails.append(f"L={L} user {u} {kname}: KV PCC {pcc:.6f} < {KV_MIN_PCC}")
                entry["users"][str(u)] = ue
                logger.info(
                    f"[gate_b32] L={L} u={u}: logits pcc={m['pcc']:.6f} top1 {m['cur_top1']}/{m['ref_top1']} "
                    f"exact={m['bit_exact']} | KV "
                    + " ".join(f"{k}:{v['pcc']:.6f}{'=' if v['bit_exact'] else ''}" for k, v in ue["kv"].items())
                )
            entry["length_s"] = time.perf_counter() - t1
            report["lengths"][str(L)] = entry
        report["fails"] = fails
        report["verdict"] = "FAIL" if fails else "PASS"
        if result_path:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(result_path).write_text(json.dumps(report, indent=2))
        assert not fails, "batch-32 gate FAIL:\n" + "\n".join(fails)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)), help="comma list (default 128,1024,8192)")
    ap.add_argument("--plant", default="none", choices=PLANTS, help="bite proof only")
    ap.add_argument("--result", default=None, help="JSON report path")
    args = ap.parse_args()
    env = dict(os.environ)
    env["PREFILL_GATE_B32_LENGTHS"] = args.lengths
    env["PREFILL_GATE_B32_PLANT"] = args.plant
    if args.result:
        env["PREFILL_GATE_B32_RESULT"] = str(Path(args.result).resolve())
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", str(Path(__file__).resolve()) + "::test_prefill_gate_b32"]
    cmd += ["-k", "1x4"]
    t0 = time.time()
    print(f"[gate_b32] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    print(f"[gate_b32] plant={args.plant} rc={rc} in {time.time() - t0:.0f}s", flush=True)
    print(f"[gate_b32] RESULT: {'PASS' if rc == 0 else 'FAIL'}", flush=True)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
