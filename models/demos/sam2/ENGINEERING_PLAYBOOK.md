# SAM2 Bounty #48311 — Engineering Playbook

## Core Principle
Stop trying to prove the current implementation is good. Start trying to falsify it against the official pretrained model. Every discrepancy discovered and removed makes the PR stronger.

## Requirements Matrix

| Gate | Status | Evidence Required |
|---|---|---|
| Official HF architecture faithfully implemented | **Fail/Unproven** | Official component inventory and HF intermediate comparisons |
| Real pretrained weights | **Fail** | Complete checkpoint mapping report |
| Patch embedding | **Likely wrong** | Exact reference op plus PCC |
| Hiera blocks | **Unproven** | Local/global block and stage PCC |
| Native stage transitions | **Fail** | No host fallback and transition PCC |
| Image neck/FPN | **Missing or undocumented** | Exact implementation and multi-scale PCC |
| Point prompts | **Partial** | Official embeddings and end-to-end PCC |
| Box prompts | **Fail/Unproven** | Box-only and mixed-prompt PCC |
| Mask prompts | **Fail** | Dense mask path and PCC |
| Complete mask decoder | **Unproven** | Token, transformer, upscaling, hypernetwork and IoU tests |
| Real sample image | **Fail** | HF/TT image validation artifacts |
| Entire model on device | **Fail** | Instrumented no-fallback execution |
| N150/N300 run | **Unproven** | Actual hardware logs |
| Stage 2 sharding | **Fail** | ShardSpec/CoreGrid/layout plan and measurements |
| Stage 2 fusion/L1 | **Partial at best** | Supported fusions and measured L1 decisions |
| Stage 3 utilization | **Fail** | Profiler evidence and improvements |
| Performance report/header | **Fail** | Repository-standard report and header |
| PR ready for final review | **No** | All previous gates passing |

## Phase Order (Must Follow)

### Phase 0: Reset Status Honestly
- Classify every requirement as PASS, FAIL, PARTIAL, UNTESTED, BLOCKED
- Mark HF checkpoint loading as FAIL
- Mark official HF numerical validation as FAIL
- Mark box prompts as FAIL
- Mark mask prompts as FAIL
- Mark all-device execution as FAIL (host avg_pool2d)
- Mark Stage 2/3 as incomplete
- Mark N150/N300 as UNTESTED
- Remove unsupported claims from PR description and README
- Be willing to replace any part that doesn't match official graph

### Phase 1: Source-of-Truth Inspection
- Pin HF version and model revision
- Trace actual image-mode forward call
- List every participating submodule
- Compare real graph with current implementation
- Identify every approximation and missing module
- **Deliverable**: architecture-difference table

### Phase 2: Graph-Difference Audit
- For every current layer: classify A (faithful+validated) through E (incorrect)
- Immediately replace all C, D, E paths before optimization
- Verify: linear patch projection, avg_pooling transitions, full SDPA per stage, Stage-4-only decoder input, linear point projection, zero-prompt fallback, direct mask projection

### Phase 3: Real Checkpoint Mapping
- Load Sam2Model.from_pretrained("facebook/sam2-hiera-tiny").eval()
- Create complete TTNN parameter preprocessing from that model
- Generate machine-readable mapping report
- Add hard assertions for missing parameters, duplicate mappings, random weights
- **Deliverable**: passing parameter-mapping test and mapping log

### Phase 4: Official HF Intermediate Validation Harness
- Build hooks/submodule calls against the actual HF model
- Validate in order: preprocessing → patch embed → pos embed → attention blocks → stages → neck → prompt encoder → decoder → upscaling → IoU → final masks
- Use repository's accepted PCC helper (comp_pcc)
- Do NOT use custom mirror model as final evidence

### Phase 5: Complete Prompt Support
- Points (positive, negative, padding)
- Boxes (corner encoding)
- Mask prompts (dense downscaling pathway)
- Metamorphic checks: changing prompt changes output
- No zero fallbacks for missing learned embeddings

### Phase 6: Remove All Host Compute Fallbacks
- Instrument forward: identify all ttnn.to_torch, torch.nn calls, CPU pooling
- Replace each with exact TTNN-native operation
- No device-to-host-to-device round trips during model execution

### Phase 7: Real-Image End-to-End Validation
- Deterministic sample image
- Run HF and TT with same prompts
- Compare logits, IoU scores, postprocessed masks
- Run separate cases for points, boxes, mask prompts, combined

### Phase 8-13: Hardware, Optimization, Perf Report, Audit
- Actual N150/N300 run
- Baseline measurement
- Stage 2: sharding decisions per tensor
- Stage 3: profiling + core utilization
- Performance report with prep_perf_report
- Final adversarial review
- PR cleanup with maintainer-preferred commit structure

## Priority Order
- **P0**: Official architecture inventory, real checkpoint mapping, official HF reference validation, removal of architectural approximations
- **P1**: Point/box/mask prompt completeness, full decoder fidelity, no host activation fallback, real-image output, N150/N300 correctness
- **P2**: Intentional memory layouts and sharding, profiling, core utilization, manipulation-overhead reduction
- **P3**: Performance report/header, final documentation, PR cleanup

## Work Cycle Response Format
After every cycle:
1. Acceptance item addressed
2. Sources inspected
3. Facts established
4. Hypothesis
5. Files changed
6. Implementation
7. Tests executed (exact commands)
8. Actual results (pass/fail, shapes, PCC/errors, hardware)
9. First divergence if any
10. Evidence artifact
11. Regressions
12. Acceptance matrix changes (evidence-supported only)
13. Remaining blockers
14. Next highest-severity task

## Non-Negotiable Rules
- Do not invent TTNN APIs. Find working examples in current repo.
- Do not invent SAM2 architecture details. Inspect exact pinned transformers source.
- Do not fabricate PCC, latency, throughput, CI, or hardware results.
- Do not lower thresholds merely to make tests pass.
- Do not compare final results only against custom mirror model.
- Do not use torch.nn/torch.nn.functional for intermediate model computation inside TTNN forward.
- Do not use ttnn.to_torch followed by CPU computation and upload back.
- Host preprocessing before device, output retrieval after device — isolated and documented.
- Do not implement arbitrary operation only because it produces expected shape.
- Do not optimize architecturally incorrect layer.
- Do not hide unsupported behavior behind zeros or dummy embeddings.
- Do not claim prompt type supported unless changing that prompt changes output and matches HF.
- Do not mark hardware work complete without real N150/N300 run.
- Do not perform more public history rewrites unless requested.
- Preserve exact command output and logs as review artifacts.
