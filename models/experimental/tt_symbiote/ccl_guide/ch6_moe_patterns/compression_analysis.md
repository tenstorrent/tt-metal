# Compression Analysis: Ch6 MoE Patterns

## Summary

- Total files analyzed: 4
- Estimated current line count: ~895
- Estimated post-compression line count: ~780
- Estimated reduction: ~13%

---

## CRUCIAL Suggestions

### `moe_overview.md` lines 114–130 — Tensor shapes table duplicates dispatch/combine API output comments

**Issue:** The "Tensor Shapes and the Dimension Convention" section (lines 114–130, ~17 lines) includes two tables:
- Input tensor shapes: `[B, S, 1, H]`, `[B, S, 1, K]`, `[1, 1, E, D]`
- Output tensor shapes from dispatch: `[1, B×D[A], S, H]` and `[1, B×D[A], S, K]`

The input shapes are re-stated in the API blocks in `dispatch_combine.md` inline comments (lines 17–20). The output shapes are documented verbatim in the dispatch API return comments (lines 29–31) and in the output structure section (lines 52–63). The full tables in `moe_overview.md` pre-explain what `dispatch_combine.md` then explains again in greater detail with the same numbers.

**Suggestion:** Remove the two shape tables from `moe_overview.md` (lines 116–128). Keep only the symbol abbreviation table (lines 102–112) since `B`, `S`, `H`, `K`, `D[A]` are genuinely needed as shorthand throughout the chapter. The shapes themselves belong in §6.2 where the APIs are defined. Replace the two tables with: "See §6.2 for the concrete input and output shapes; the symbols are defined above."

**Estimated savings:** ~14 lines

---

### `moe_overview.md` lines 197–211 — "How tt-metal's AllToAll Differs" table duplicates §6.2 output structure section

**Issue:** The comparison table (lines 197–211, ~14 lines) contrasting standard `MPI_Alltoall` with `all_to_all_dispatch` on 6 properties (routing, communication volume, output, metadata, padding, result) covers:
- Data-dependent routing vs. fixed
- K×tokens bandwidth vs. D×D blocks
- Sparse output with placeholder rows
- `expert_metadata_tensor` encodes row validity

All of these points are made again in `dispatch_combine.md` §Output structure: sparse tokens (lines 50–63), in the `dispatch_combine.md` intro paragraph (line 3), and in `moe_overview.md` §The routing asymmetry problem (lines 29–42) which already explains the same contrast in prose. The table adds no information not already present in the surrounding prose.

**Suggestion:** Remove the comparison table (lines 199–210) and its 2-line header (lines 197–198). The `moe_overview.md` §routing asymmetry prose already makes the same points more clearly. If a table summary is wanted, a forward ref suffices: "See §6.2 §Output structure for how placeholder rows work in practice."

**Estimated savings:** ~13 lines

---

### `dispatch_combine.md` lines 158–179 — Kernel Structure section re-explains Ch3 §3.2 Under the Hood content

**Issue:** The Kernel Structure section (lines 158–179, ~22 lines) documents:
- `AllToAllDispatchSparse` program factory: ternary reader kernel, binary writer kernel, two GlobalSemaphore objects (`init_semaphore`, `cross_device_semaphore`)
- `AllToAllTransferType` enum: `FullPacket` vs `PageByPage` modes and their selection logic
- `AllToAllCombineFromSparse` program factory: ternary reader, unary writer, same semaphore pattern

Ch3 §3.2 (`all_to_all.md`) already covers the kernel structure under "Under the Hood": it documents both `GlobalSemaphore` types used by dispatch, the `AllToAllTransferType` enum with `FullPacket`/`PageByPage` modes, the reader/writer kernel roles, and program caching. This section is a direct re-explanation of Ch3 content that `index.md` line 36 explicitly promises to avoid ("Cross-references to §3.2 are used rather than re-explaining the API basics").

**Suggestion:** Replace the Kernel Structure section (22 lines) with a 2-line forward ref: "For the kernel implementation details (`AllToAllDispatchSparse`, `AllToAllTransferType`, `AllToAllCombineFromSparse`, `GlobalSemaphore` types), see [Ch3 §3.2 — Under the Hood](../ch3_intermediate_operations/all_to_all.md#under-the-hood)."

**Estimated savings:** ~20 lines

---

### `deepseek_patterns.md` lines 29–41 — Python/C++ parameter name table adds no information

**Issue:** The "Python argument names vs C++ parameter names" table (lines 29–41, ~13 lines) for `deepseek_minimal_broadcast` lists all 7 parameters mapped from Python to C++. Every entry has identical names on both sides (`input_tensor → input_tensor`, `sender_coord → sender_coord`, etc.) with no renaming or reordering. The Notes column only says "positional" for the first two parameters — information already clear from their position in the API block on lines 16–26.

This table addresses a documentation concern that does not exist: there are no Python/C++ naming discrepancies to document here (unlike `all_gather_concat` where `multi_device_global_semaphore` maps to C++ `global_semaphore`, which warranted a note).

**Suggestion:** Remove the entire table (13 lines). Move the only useful content — "first two parameters are positional" — into a one-line note after the API block: "`input_tensor` and `sender_coord` are positional; all remaining parameters are keyword."

**Estimated savings:** ~11 lines

---

## MINOR Suggestions

### `deepseek_patterns.md` lines 249–258 — Summary table at end of §6.3 duplicates index.md ToC

**File:** `deepseek_patterns.md`, lines 249–258

The 3-row summary table (Namespace, Use case, Topology default for all three ops) encodes information already in `index.md` lines 11–16 (ToC) and inferrable from the API blocks. The "All three ops require `cluster_axis`..." footer note is the only unique content. Move the footer note into each op's parameter notes or Gotcha block, and remove the table. Saves ~8 lines.

---

### `moe_overview.md` lines 54–95 — Dispatch→Compute→Combine pipeline overview partially redundant with §6.2 worked example structure

**File:** `moe_overview.md`, lines 54–95 (~42 lines)

The 5-stage pipeline diagram and ASCII token journey are first-occurrence conceptual content — appropriate for an overview. However, the ASCII token journey (lines 79–94) is elaborate (15 lines for 3 tokens) for an overview section. The same information is shown more precisely in the §6.2 worked example code. Consider trimming the ASCII to a 4-line sketch and removing the detailed 3-device example, saving ~10 lines. This is MINOR because the journey diagram has genuine pedagogical value as a first exposure.

---

### `dispatch_combine.md` lines 50–63 — Output structure section partially overlaps with API return comments

**File:** `dispatch_combine.md`, lines 50–63

The prose description of the sparse output shape (14 lines) largely re-states what the API return comment on lines 29–31 already captures. The only unique content is the dimension-by-dimension breakdown (lines 53–60) and the placeholder/real row distinction (lines 58–62). Keep those 8 lines; remove the shape re-statement (lines 52–54). Saves ~5 lines.

---

## VERDICT

- Crucial updates: yes
- Summary: 4 CRUCIAL issues identified (~58 lines saveable), 3 MINOR issues (~23 lines). The largest wins are the Ch3 kernel structure re-explanation (C3, ~20 lines), the tensor shapes duplication across files (C1, ~14 lines), and the standard AllToAll comparison table (C2, ~13 lines).

---
# Compression Analysis: Ch6 MoE Patterns — Pass 2

## CRUCIAL Suggestions

None. All 4 pass-1 CRUCIAL issues have been applied and confirmed:
- C1: `moe_overview.md` two shape tables replaced with forward ref to §6.2 (line 114).
- C2: `moe_overview.md` standard AllToAll comparison table replaced with a §6.2 hyperlink (line 182).
- C3: `dispatch_combine.md` Kernel Structure section replaced with a 1-line forward ref to Ch3 §3.2 (line 158).
- C4: `deepseek_patterns.md` Python/C++ parameter name table replaced with one-line positional note (line 28).

No new CRUCIAL redundancies identified on re-read. `moe_overview.md` §Expert Metadata Tensor and §Expert Mapping Tensor are first-occurrence canonical explanations referenced by later sections, not duplicates. The `local_reduce` section in `dispatch_combine.md` (45 lines) is unique to Ch6. The `deepseek_minimal_broadcast` Operational semantics and the How-it-differs-from-broadcast table contain unique content with no counterpart in earlier chapters.

## VERDICT

- Crucial updates: no
