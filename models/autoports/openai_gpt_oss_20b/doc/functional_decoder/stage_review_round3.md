# Functional Decoder Stage Review — Round 3

Verdict: **CLEAN PASS**

Review date: 2026-07-25 UTC

## Required Work

- None.

## Other Concerns

- None. The round-2 documentation contradiction is corrected. The README now
  distinguishes the source TP4 axes accurately: Q/K/V are column-parallel, O
  is row-parallel, and every gate/up/down expert weight and bias is
  expert-parallel over the 32-to-8 expert axis.

## Hard-Check Gaps

- This independent review did not open a TT device, run pytest, rerun the
  capacity probes, reset hardware, or start a server. The persisted JUnit XML
  is newer than the final implementation and test edits, parses successfully,
  and records 6 tests, 0 failures, 0 errors, and 0 skips. The detailed PCC
  values and capacity output are retained in the README and work log rather
  than JUnit `system-out`.
- The official-weight test passes already-dequantized dense expert tensors to
  `from_state_dict`. The fallback that accepts raw packed MXFP4 blocks and
  scales is statically present but is not directly exercised. This is residual
  loading-path risk, not a gap in the required real-weight dense-math control.

## Anomaly Ledger

- **Observed anomaly:** Round 2 found that the README called gate/up expert
  tensors column-parallel and expert-down tensors row-parallel.
  **Evidence:** The corrected README lines 44-49 now says only Q/K/V are
  column-parallel, only O is row-parallel, and all expert tensors are
  expert-parallel. Source placeholders show Q `[1024, 2880]` versus full
  `[4096, 2880]`, K/V `[128, 2880]` versus full `[512, 2880]`, and O
  `[2880, 1024]` versus full `[2880, 4096]`. In contrast, gate/up
  `[8, 2880, 5760]` and down `[8, 2880, 2880]` retain both complete feature
  dimensions and reduce only the expert count from 32 to 8.
  **Affected path:** TP4 provenance handoff; not the dense runtime.
  **Control or comparison:** The O partial is followed by a sum all-reduce,
  while expert routing is partitioned on its expert dimension, eight complete
  local experts are evaluated, and their routed local sum is all-reduced.
  `multichip_provenance.json` independently records Q/K/V as
  `tensor_column`, O as `tensor_row`, and all four expert weights/biases as
  `parallel_kind: expert`, tensor axis 0.
  **Likely subsystem:** Documentation of the TP4 collapse.
  **Investigation performed:** Reconciled the README against both prefill and
  decode layer-12 weight placeholders, consteval partitioning, runtime
  collectives, and the structured parameter/transient inventories.
  **Resolution:** Fixed.

- **Observed anomaly:** The first beyond-window decode control failed because a
  changing rotary `token_idx` was omitted from the TTNN operation's program
  hash.
  **Evidence:** `AUTODEBUG.md` traces the hash/runtime-argument mismatch. The
  implementation now slices the requested cosine/sine row on device and calls
  rotary with constant index zero. Final position-256 PCC is 0.9999343 for the
  new K row, 0.9999488 for the new V row, 0.9994681 after attention, and
  0.9994802 for the full layer.
  **Affected path:** Same-shape decode calls at distinct absolute positions.
  **Control or comparison:** Prefill cache rows 129-255 separately match the HF
  sliding cache above 0.99994, the test retains the passing position-17 decode,
  and position 256 excludes old keys outside the 128-token window.
  **Likely subsystem:** Rotary program-cache keying.
  **Investigation performed:** Cache, explicit-mask, sink-disabled, and source
  controls were recorded before applying the emitted-form constant-index
  workaround.
  **Resolution:** Fixed and controlled.

- **Observed anomaly:** Round 1 found broad HiFi4/FP32-accumulation overrides
  where the emit uses default compute configuration.
  **Evidence:** The current runtime supplies a custom compute configuration
  only to RMSNorm call sites; projections, SDPA, router, and expert operations
  omit it. The final JUnit suite postdates this implementation.
  **Affected path:** Prefill and decode numerical policy.
  **Control or comparison:** The representative emit applies HiFi4/FP32
  accumulation to its two RMSNorm sites and uses `compute_kernel_config=None`
  for the other material operations.
  **Likely subsystem:** Per-operation emit translation.
  **Investigation performed:** Compared current AST call sites with the two
  selected emit ranges and final recorded PCC rows.
  **Resolution:** Fixed.

## Scope Inspected

- **Goal/skill paths:** `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/forge-functional-decoder/SKILL.md`,
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md`, and
  `.agents/skills/tt-device-usage/SKILL.md`.
- **Artifact paths:** `doc/context_contract.json`; functional-decoder README,
  work log, both prior stage reviews, AutoDebug report, JUnit XML, and
  `multichip_provenance.json`.
- **Code paths:** `tt/functional_decoder.py`, both package initializers,
  `tests/test_functional_decoder.py`, and
  `tests/functional_decoder_capacity_probe.py`.
- **Emit sources:** Layer-12 prefill lines 3879-4089, decode lines 3318-3488,
  layer-12 weight creation, and representative consteval transforms under
  `/home/mvasiljevic/emit-gptoss/g0_prefill` and `g1_decode`.
- **Repository state:** Live worktree on branch
  `mvasiljevic/gpt-oss-pipeline-progress` at
  `dd34ac32928d704bf0aff87fd25f047c5fbb6af0`; all visible untracked stage
  files are confined to `models/autoports/openai_gpt_oss_20b/`.
- **Commands run:** Read-only source/artifact inspection; AST operation audit;
  source-hash verification; JSON/XML parsing; transient-alias reconciliation;
  Python syntax compilation; `git status`; `git diff --check`; and both exact
  and HF-aware strict context-contract checker invocations. Both context checks
  passed with target 131072, supported 21248, and the DRAM-limited
  classification. No TT hardware command was run.

## Verified Gate Evidence

- Both supplied pre-generated EmitPy packages are translated directly; all
  four recorded SHA-256 values match the current source files and no generated
  MLIR/IR conversion artifact is present.
- Layer 12 is a valid middle layer of the 24-layer flat graph. The documented
  ranges contain the two norms, biased Q-K-V attention with RoPE/sinks/cache,
  O projection and residual, FP32 router/top-4 routing, complete expert-axis
  SwiGLU computation, and final residual.
- `FunctionalDecoder` subclasses `LightweightModule`, loads full canonical HF
  tensors, implements both emitted paths, preserves batch one, and enforces a
  dense 1x1 mesh contract. Q-K-V fusion order, projection transposes, expert
  tensor orientation, biases, sink scaling, and residual ordering agree with
  the emit and HF reference.
- Runtime methods contain no Torch execution, host transfer/fallback,
  layout/memory conversion, reshard, collective, or mesh-partition call.
  Decode-only L1 head layout is the allowed minimal workload-derived
  requirement.
- Structured provenance contains 17 representative parameters, 13
  cache/constant tensors, 19 sharded transient groups, both boundary partials,
  and all 16 representative prefill/decode consteval/runtime collectives. All
  86 transient source aliases resolve in their documented layer ranges.
- Final evidence exceeds the required 0.99 floor: three full-shape synthetic
  prefill lengths, official real-weight prefill, official real-weight decode at
  positions 17 and 256, and boundary cache/attention controls all pass. The
  measured dense-functional capacity is S=21248 with the first S=21249 probe
  failing for a recorded device-DRAM allocation.
- The context contract passes both checker modes and honestly distinguishes
  the 21248 functional DRAM limit from the model's advertised 131072 context.
  README and work log contain the runtime signatures, shapes, provenance,
  commands, PCC table, limitations, and no unsupported performance claim.

## Residual Risk

- Real prefill PCC (0.9933186 at S=17 and 0.9913088 at S=256) clears the
  authoritative 0.99 threshold but is below the aspirational 0.995 target.
  Decode and cache controls are above 0.9993, and no evidence indicates a
  semantic defect.
- Packed-MXFP4 ingestion remains less directly covered than the canonical
  dense-weight path. Performance, reduced-precision selection, full advertised
  context, and multi-device execution are correctly deferred to later stages
  and are not claimed here.
