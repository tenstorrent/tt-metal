# CCL op-gen Phase 2 — overnight report (2026-06-25, session 10)

Run autonomously overnight on bh-33 (same machine/reservation). Everything below is **pushed to
remote git**; the detached pipeline runs auto-pushed their output. Branch tips at the end.

## Headline
**The op-gen pipeline can generate the *harder* CCL op (`all_gather`, a ring collective) from
scratch, and the verifier's OWN run goes GREEN end-to-end on the multi-device simulator.** This
extends the Phase-1 p2p result to a second, substantially more complex op + a second architecture
(Wormhole), and it validates the test↔topology coupling fix from Phase 1.

## What was achieved (in order, each pushed)

### 1. WH `all_gather` validated on the multi-device runner — `3b469987a49`
Realigned the runner's WH topology entry to the existing `all_gather` test's `(1,8)` mesh (the
`t3k_1x8` mesh-graph descriptor — added a `{repo}` placeholder so in-tree descriptors resolve — +
the all-MMIO cluster desc, `mesh_shape:[1,8]`, `FABRIC_1D`). Ran the existing migrated `all_gather`
through the runner → **PASS** (8 sim WH chips, PCC asserted). The runner (`run_multidevice_sim_pytest.py`)
is now a proven **multi-arch, multi-op** CCL verification harness: BH `point_to_point` + WH
`all_gather`, both `required`, both green. The Phase-1 coupling lesson was applied *proactively* (I
caught the `(2,4)` vs `(1,8)` mismatch before running).

### 2. Dashboard wiring implemented — `0e30fb7` (prompts branch)
Added a CCL branch to `run_eval.py`'s `run_golden_tests`: if the op is in the multidevice topology
matrix (the deterministic detector), it grades on the multichip craq-sim via the runner (with a
**clean fast-dispatch env** — a single-chip slow-dispatch `--sim` env would conflict) and maps the
result into the same `golden_results.txt`/junit the existing scoring + ingest path consumes → a real
multichip PASS/FAIL can land on the dashboard. Degrades to single-chip when the matrix is absent.
**The heavy validated `run_eval` clone+build+ingest is left for you to kick off** — I was conservative
about writing to the shared dashboard DB unattended. Plan: `DASHBOARD_WIRING_PLAN.md`.

### 3. `all_gather` GENERATED + verifier-green (the capstone) — `2f20bdc3643`
`run_op.py` on `all_gather_phase2.txt` (detached, ~90min): planner (37 turns) → implementer (66) →
verifier (62), all exit 0. The pipeline produced a **self-contained Python `generic_op` +
`MeshProgramDescriptor`** op (NOT a wrap), with newly-authored ring kernels
(`all_gather_writer.cpp`/`all_gather_reader.cpp`) using the `FabricStreamSender` typestate helper
(`arm_unicast_write`/`arm_multicast_inc` + local `noc_semaphore_wait_min`), a bidirectional
**store-and-forward ring**, the **W4 op-internal semaphore** (`mesh_program_descriptor.semaphores = [sem]`,
created once, no per-call barrier), and — crucially — a generated acceptance test with
**`mesh_device=[(1,8)]` + `FABRIC_1D` matching the runner topology**. Because the test matched the
topology (the W1 coupling fix landed in the planner), **the verifier's own runner run went GREEN:
22/22 acceptance + 8/8 precision on the WH sim, aggregate exit 0 — the cross-device gather actually
executed and PCC asserted.** The verifier even noted the contrast with p2p ("unlike point_to_point
whose BH sim was fabric-blocked, all_gather's WH sim works"). Program-cache reuse was tested.

### 4. p2p RING topology validated — `d11102e839f`
`point_to_point` with `Topology.Ring` + `FABRIC_1D_RING`, ring-wraparound (0,0)→(0,3) on the
`(2,4)` `blackhole_8xP150_torus_x` descriptor → **PASS** via the runner. RING path now exercised.

## The green matrix (all on the multi-device craq-sim, real cross-device execution + PCC)
| Op | Arch | Topology | Mesh | Result |
|---|---|---|---|---|
| `point_to_point` | Blackhole | Linear (`FABRIC_1D`) | (2,4) | ✅ PASS |
| `point_to_point` | Blackhole | Ring (`FABRIC_1D_RING`) | (2,4) torus_x | ✅ PASS |
| `all_gather` (pipeline-generated) | Wormhole | Linear (`FABRIC_1D`) | (1,8) | ✅ 22/22 + 8/8 |

## Status of the p2p verifier gap (from Phase 1)
**Functionally closed.** The p2p op is proven green on the runner (`test_p2p_confirm_topology.py`),
and the coupling fix that makes the verifier's *own* run green is now proven end-to-end by
`all_gather` (its generated test matched the topology → verifier green). The committed p2p artifact
still carries its pre-fix auto-generated `(1,2)` test; a p2p re-regen with the now-fixed prompt would
make its verifier green automatically (as all_gather demonstrated), but that is purely confirmatory,
so I did not spend ~90min re-running it.

## Branch tips (pushed — source of truth)
- **tt-metal** `origin/wransom/ccl_pipeline_phase1` @ `d11102e839f` (off `ccl_help` @ `b7d848979e1`):
  W4 + runner (BH p2p + WH all_gather + p2p RING) + the pipeline-generated p2p & all_gather ops + confirmations.
- **pipeline** `origin/wransom/ccl_pipeline_prompts` @ `0e30fb7`: W1 prompts + verifier routing + dashboard wiring.

## Remaining Phase-2 (sized; deliberately NOT chased unattended to avoid a messy state)
1. **Dashboard ingest** (you kick off): one validated `run_eval` clone+build+grade+ingest of a CCL
   op via the now-wired multichip path → a real multichip run on http://bgdepyc01:8090. Code is done.
2. **`all_gather` RING + 2-D**: needs a ring-capable / 2-D WH mesh-graph descriptor (`t3k_1x8` is
   line-only; the BH torus is ring-capable but all_gather is `@skip_for_blackhole`).
3. **Full topology-matrix expansion** (Galaxy 1×8/1×16, BH 4×4 quietbox): add entries + matching
   tests; the tree ships the descriptors.
4. **Build the CCL golden suite** (`CCL_GOLDEN_TESTS_DESIGN.md`): mesh_device fixture + shard→collective→gather
   oracle. NOTE: the golden harness had a `_op_contract` import gap (run 572: "0 golden cells for every op")
   — verify/fix that first; building on a broken harness was too risky to attempt unattended.

## Verdict
Phase 2's core question — *can the automated pipeline generate AND verify multi-device CCL ops
end-to-end?* — is answered **yes**, now for both a unicast op (p2p) and a ring collective
(all_gather), on both arches, with the verifier's own run green for all_gather. The remaining items
are extensions/ingest, not blockers.
