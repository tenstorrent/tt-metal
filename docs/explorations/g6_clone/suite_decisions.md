# g6_clone: suite decisions

Decisions the golden-tests author took where `golden_brief.md` left a choice open. The spec
`eval/prompts/g6_clone.txt` is authoritative and was not changed.

| Id | Open point (brief) | Decision | Why |
|---|---|---|---|
| S1 | Q2, §2, §9: project `dtype` from the scenario dict | Accepted: `dtype` and `layout` come from the op's `tag_dtype` / `tag_layout`; 29 cells, no cartesian | the spec fixes these two taggers; BB-22 limits the suite to the five production (dtype, layout) pairs |
| S2 | Q3, §1: can host-to-device carry NaN payloads and subnormals | `create_ttnn_input_tensor` passes `preserve_nan_values=True`, so the tensor is built on the host (TILE: a byte rearrangement) and written; without it `from_torch` tilizes on the device, which is not guaranteed to keep NaN. Every cell reads x back before the call and fails as `HARNESS PRECONDITION` when its bits differ from the host's | a corrupted upload would blame the op. Not confirmed on a device (none allowed here): the first device run answers Q3. If NaN payloads cannot be carried, the lead or pipeline turns that case into an expected failure with the reason, as the brief says, instead of dropping it |
| S3 | §1: uint32 on the host | uint32 inputs are int32 patterns viewed as `torch.uint32`, so `from_torch` performs no typecast; outputs are compared through same-width int views | a typecast int32 to uint32 clamps negative values to 0 |
| S4 | §5: the poison value | NaN only, as the canonical quiet NaN of the dtype (float32 `0x7FC00000`, bfloat16 `0x7FC0`); the 1e30 fallback is not added; the harness checks that the poison landed | torch's own float32 to bfloat16 cast turns NaN into `0xFFFF`; any misaddressed padding element differs in bits from the logical value anyway |
| S5 | §5: where the flag goes | the single `# attention: PERF FOCUS` line is the line directly above the p1 entry; model profile and goal lines sit above it. p2 to p4 carry `math_fidelity: None`, `gate: "exact"` like p1, with no time | only the flagged comment may hold a target |
| S6 | §1, §6: new storage on every cell | every golden and loose cell also checks that out does not share x's buffer and that x keeps its bits; the regression additionally overwrites x and rereads out | a cheap check for rule R2; the regression alone would leave 28 production shapes unchecked |
| S7 | §6: repeat | after the two identical calls, a third call on other values at another address is graded against its own oracle | BB-07: a cached program that keeps stale addresses passes two identical calls |
| S8 | §1: generator of the poisoned cases | the same random bit patterns as every cell | the brief makes random patterns the product draw; the bit comparison still separates a leaked poison |
| S9 | §9: a case that holds the rest of L1 (BB-05) | not added | the suite cannot size an L1 hog to (L1 - 1347776 B) per core through public ttnn without guessing the allocator base and the device's L1 size; a hog of the wrong size proves nothing. BB-05 stays with the verifier's L1-footprint audit |
| S10 | step 3: `KNOWN_MODES` | no change to `eval/features_table.py` | `dtype` and `layout` are mapped already; `unmapped_axes` is empty |
| S11 | step 3: loose test ids | `extras["case"]` leads the id of every loose case; a custom shape id reads `input_shape` | the default shape id joins the scenario dict's keys, giving every cell the same id |
| S12 | metrics | the PCC/RMS record is observe-only, best effort, and only for finite references | the gate is the bit comparison |

Left for the lead: Q1 of the dossier (no `dtype` / `memory_config` keyword) is unchanged and blocks
nothing; S2 needs the first device run.
