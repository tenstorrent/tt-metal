# Movement28: preserve flat topology in gather-first comparisons

CPU-only update of root's26-job `long_movement27.json`. [long_movement28.json](long_movement28.json) contains28 fresh names: the original26 rows with all gather-first candidates explicitly selecting **`flat_gather_prefill=true`**, plus two final unflattened SFPU diagnostics (L0/L3 B32/S513). All nine original control policies remain unchanged. References, activation files, minimal decode norm and GDN column1 are preserved. MMRS placement rows are unchanged. Every row records `comparison_to`; unflattened rows explicitly prohibit rejecting the matched flat family from their result.

The old manifest is preserved. New artifact names prevent `--resume` from confusing earlier unflattened measurements with corrected candidates. For B1, `_can_flatten` remains false by design; adding the admission flag does not change that existing dispatch. For B32/S513, bulk S512 remains flat and the one-token tail remains legacy. For B32/S33, one S33 chunk remains flat.

CPU execution of the isolated current `_can_flatten` function checked40 job/temporal-chunk combinations. Both gather-first choices now match their control's flattening; only the two declared diagnostics differ. All reference files exist. [CPU evidence](long_movement28_cpu.json) records checks and hashes. No device, TTNN import, build, or production edit was performed.

Root-owned execution:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_search models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/long_movement28.json --resume
```

`gather_norm=true,flat_gather_prefill=true` changes normalization placement while retaining the flat projection/cache/MLP pipeline. Adding `sfpu_prefill_norm=true` selects the existing fused local prefill norm. None of these jobs silently selects new SwiGLU, direct-QKV, precision or geometry candidates. The old-topology diagnostics retain `flat_gather_prefill=false`; they may expose a scheduler interaction but cannot decide the flat family.

## Applied-patch compatibility

Combined `git apply --reverse --check` succeeds for the now-applied `direct_prefill_qkv27.patch`, `prefill_swiglu27.patch`, and `flat_gather_prefill27.patch`. Source inspection confirms independent hooks:

- Flat admission changes `_can_flatten`, preserving `_flat_chunk` and its `logical_shape=(batch,seq)` QKV call.
- Direct QKV receives the existing packed flat `[1,B*S,3584]` projection and reconstructs per-user head axes. Its `prefill and not compact` scope remains intact; compact decode and the legacy B32 one-token tail stay on their original path. Logical S33 produces B×33 projection rows, while the head tensors separately own per-user tile padding.
- SwiGLU bulk dispatch is in `_finish_flat`; its helper consumes the same flat normalized width5120. Tail dispatch is in `_finish`. Their off-default flags are absent from this movement manifest, so they do not alter the comparison.
- Minimal decode norm still branches before gather-first policy handling. No prefill proposal reuses its capacity32 resource views or changes its input/post phase sequence.

This establishes source compatibility only; simultaneous candidate composition still needs whole-layer numerical, trace/ownership and profiler evidence. It is not justified by patch application or individual passing families alone.

## S33 correction and proof impact

Corrected `long_prefill_geometry27_audit.md` in place: B32/S33 is **M1056**, with c640 pieces640+416, and c1280 one1056-row piece. It is not M1024 plus32-row temporal tail. Attention SDPA/head storage can pad each user's33 positions to64 independently; that does not change flat projection's logical row count. S513's M16384+M32 description remains correct.

No matrix26 CPU proof result changes: its loop already computed `logical=b*s`, explicitly included B32/S33, and checked padded1056 row coverage. The9,104 proof count, geometry27 policy entries and references remain valid. The prior mistake was prose and timing attribution, not generated geometry. The geometry27 audit now also names the required flat-admission flag in its norm proposal.
