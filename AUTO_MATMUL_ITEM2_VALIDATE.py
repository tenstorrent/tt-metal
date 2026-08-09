# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Item #2 -- correctness gate, hardware validation on an N300 (single chip + 1x2 mesh).

What this asserts (all verified on the N300 Tracy build):

  SINGLE-DEVICE (fix A -- fp32-golden gate, no collectives needed):
    * the selector picks a real tuned program_config (NOT the default),
    * the fp32-golden relative gate does NOT over-reject valid bf16 configs
      (incorrect == 0 -- the fix-A backfire guard, live on device), and
    * the winner's output matches a float32 torch golden within bf16 tolerance.

  DISTRIBUTED (fix B -- the correctness gate on a 1x2 mesh):
    * the fp32 golden reconstructed across the mesh matches a pure-torch full
      matmul exactly (rel-L2 == 0), and
    * the real GPT-OSS o_proj reduce-scatter recipe genuinely FAILS on this 2-chip
      build (reduce_scatter_minimal_async TT_FATAL -- the real "Group B" crash);
      the gate catches it and selection FAILS CLOSED to the base op -- it does NOT
      cache a crashing recipe on timing alone.  This is the defensive direction the
      gate exists for.

  HONEST COVERAGE LIMIT: on this 2-chip N300 build the distributed collectives
  (reduce_scatter_minimal_async / all_gather) do not execute for these shapes, so a
  "distributed recipe WINS and verifies" positive is not demonstrable here -- which
  is itself the reason item #2 (reject-broken-recipe, fail-closed) is needed.  The
  numerical-gating and all-shards-agree paths are proven by the host-only tests.

Run (on the Tracy build, edited _selector.py overlaid):
    python3 AUTO_MATMUL_ITEM2_VALIDATE.py
"""
import torch

import ttnn
from ttnn import ShardTensorToMesh
from ttnn._experimental.auto_config import _selector as s


def _rel_l2(ref, cand):
    ref, cand = ref.float().flatten(), cand.float().flatten()
    return (torch.linalg.vector_norm(cand - ref) / torch.linalg.vector_norm(ref)).item()


def validate_single_device():
    print("=== Single-device fp32-golden gate (fix A) ===")
    dev = ttnn.open_device(device_id=0)
    try:
        for tag, (M, K, N) in {
            "square": (1024, 1024, 1024),
            "nonpow2": (2048, 2880, 2880),
            "wide": (512, 4096, 4096),
        }.items():
            torch.manual_seed(0)
            a = torch.randn(M, K, dtype=torch.bfloat16)
            w = torch.randn(K, N, dtype=torch.bfloat16)
            a_dev = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev)
            w_dev = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev)
            ex = s.explain_matmul(a_dev, w_dev)
            win = ex.get("winner") or {}
            out = ttnn.experimental.auto_config.matmul(a_dev, w_dev)
            rel = _rel_l2(a.float() @ w.float(), ttnn.to_torch(out))
            n_incorrect = sum(1 for t in ex.get("candidate_timings_us", []) if t.get("status") == "incorrect")
            n_ok = sum(1 for t in ex.get("candidate_timings_us", []) if t.get("status") == "ok")
            print(
                f"[{tag} {M}x{K}x{N}] winner={win.get('kind')}/{win.get('mode')} ok={n_ok} incorrect={n_incorrect} rel_L2={rel:.4f}"
            )
            assert win.get("kind") == "program_config", f"expected a tuned config, got {win.get('kind')}"
            assert n_incorrect == 0, f"fp32 gate over-rejected {n_incorrect} valid configs (fix-A backfire)"
            assert rel <= 0.05, f"winner output {rel:.4f} off the fp32 golden"
        print("PASS: tuned config selected, 0 false rejects, output within tolerance.\n")
    finally:
        ttnn.close_device(dev)


def validate_distributed_gate():
    print("=== Distributed correctness gate (fix B) -- 1x2 mesh ===")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    try:
        # Real GPT-OSS-20b o_proj: [M,4096] @ [4096,2880], both K-sharded (row-parallel).
        M, K, N = 128, 4096, 2880
        torch.manual_seed(0)
        a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        w = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        a_dev = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ShardTensorToMesh(mesh, dim=3))
        w_dev = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ShardTensorToMesh(mesh, dim=2))
        sig = s._build_signature(
            a_dev,
            w_dev,
            bias=None,
            transpose_a=False,
            transpose_b=False,
            memory_config=None,
            dtype=None,
            activation=None,
            is_linear=False,
        )
        plan = s._infer_distributed_plan(sig)
        prepared = s._prepare_inputs(a_dev, w_dev, None)

        # (1) the mesh-reconstructed fp32 golden matches a pure-torch full matmul exactly.
        golden = s._fp32_reference_distributed(prepared, sig, plan)
        full = (a.reshape(M, K).float() @ w.reshape(K, N).float()).flatten()
        gerr = _rel_l2(full, golden)
        print(f"[o_proj RS] plan={plan.kind} golden_vs_torch_rel_L2={gerr:.6f}")
        assert plan.kind == "matmul_before_reduce_scatter"
        assert golden is not None and gerr < 1e-3, "distributed fp32 golden reconstruction is wrong"

        # (2) the real reduce-scatter recipe is broken on this 2-chip build (Group B
        # TT_FATAL); the gate must reject it and selection must fail closed.
        cands = s._build_candidates(sig, prepared, {}, base_operation=s._get_cpp_base_operation(False))
        matched = [s._distributed_candidate_matches(c, mesh, golden, plan) for c in cands]
        print(f"[o_proj RS] candidates={[c.descriptor.get('kind') for c in cands]} gate_match={matched}")
        assert not any(matched), "a broken reduce-scatter recipe unexpectedly passed the gate"

        ex = s.explain_matmul(a_dev, w_dev)
        winner = (ex.get("winner") or {}).get("kind")
        print(f"[o_proj RS] winner={winner} (fail-closed to base op -- crashing recipe NOT cached)")
        assert winner == s.WINNER_KIND_BASE_OP_FALLBACK, f"expected fail-closed base op, got {winner}"
        print("PASS: golden correct; the real Group-B crash is caught and bypassed, not cached.\n")
        print("NOTE: reduce_scatter_minimal_async does not execute on this 2-chip build for any")
        print("      shape tried, so a 'distributed recipe wins + verified' positive is not")
        print("      demonstrable here; that path is covered by the host-only tests.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    validate_single_device()
    validate_distributed_gate()
    print("\nALL HARDWARE ASSERTIONS PASSED.")
