# Compression Analysis: Ch2 Basic Operations

## Summary
- Total files analyzed: 4 (`index.md`, `all_gather.md`, `all_reduce.md`, `broadcast.md`)
- Estimated current line count: ~931 lines
- Estimated post-compression line count: ~680 lines
- Estimated reduction: ~27%

---

## CRUCIAL Suggestions

### [index.md] ~lines 28–50 (Common API Conventions section)
**Issue:** The "Common API Conventions" section explains `topology`, `num_links`, `cluster_axis`, `memory_config`, and `subdevice_id` in full — 22 lines of parameter definitions. Each of the three operation files (`all_gather.md`, `all_reduce.md`, `broadcast.md`) then re-explains every one of these parameters in its own "Parameter notes" section. The index section is never referenced or deferred to — it is a fifth copy of the same material that saves no space.
**Suggestion:** Remove the Common API Conventions section from `index.md` entirely. The three operation files already cover these parameters in context. If a shared reference is desired, replace the section with one sentence pointing to AllGather parameter notes as the canonical source.

### [all_reduce.md] ~lines 225–243 (Input layout impact table + sharding code snippet)
**Issue:** The "Input layout impact" table (`all_reduce.md` lines 226–231) is near-identical to the table in `all_gather.md` lines 250–254 — same three rows, same text, same conclusions. `all_reduce.md` even explicitly acknowledges this: "AllReduce inherits the same layout sensitivity as AllGather." Yet it reproduces the full 3-row table and an 8-line code snippet anyway.
**Suggestion:** Replace the entire "Input layout impact" subsection in `all_reduce.md` (table + code snippet, ~18 lines) with two sentences: "AllReduce inherits the same layout sensitivity as AllGather (see Section 2.1). Convert to L1-sharded layout before calling AllReduce when the input comes from DRAM." The reader has the full explanation 80 lines earlier in the same chapter.

### [all_reduce.md] ~lines 169–195 (Data Flow: Ring AllReduce section)
**Issue:** The "ReduceScatter phase" sub-section (~lines 175–185) describes the same N-round ring rotation algorithm already explained in detail in `all_gather.md` Data Flow section, simply adding "and accumulate" to the forward step. The "AllGather phase" sub-section (lines 187–189) contains only two lines of new content: it says "see Section 2.1" and then restates the formula. The "Linear AllReduce" sub-section (lines 193–195) is three lines. The full section is 27 lines for ~5 lines of genuinely new information.
**Suggestion:** Collapse the entire "Data Flow: Ring AllReduce" section to ~6 lines: state the two-phase decomposition formula (ReduceScatter N-1 rounds + AllGather N-1 rounds), state the total bandwidth formula (`2*(N-1)/N * tensor_size`), note that Linear AllReduce serializes both sweeps, and forward to Section 2.1 for the round-by-round diagram. Remove the ReduceScatter phase ASCII block — it adds nothing the AllGather diagram doesn't already show.

### [all_gather.md + all_reduce.md + broadcast.md] — Topology hang Gotcha repeated 4 times
**Issue:** The warning "specifying `Topology.Ring` without a physical wrap-around cable causes an indefinite hang" appears in:
- `all_gather.md` line 199 (Gotcha callout, full)
- `all_reduce.md` line 97 (prose in Parameter notes, full)
- `all_reduce.md` line 255 (error table row)
- `broadcast.md` line 279 (error table row)
Each instance is worded slightly differently but conveys the exact same fact.
**Suggestion:** Keep the full Gotcha callout in `all_gather.md` (it is the first time a reader encounters it and the detail is warranted there). In `all_reduce.md` Parameter notes, shorten to one clause: "must match physical cabling — see Section 2.1 Gotcha." Remove the duplicate error table row from `all_reduce.md` (the AllGather error table already has it; AllReduce inherits the same behaviour). In `broadcast.md`, keep the error table row since Broadcast is a separate operation, but shorten it to match the AllGather table's phrasing.

### [all_gather.md + all_reduce.md] — Program caching Gotcha repeated twice
**Issue:** The "changing tensor shape invalidates the program cache and triggers expensive recompilation" warning appears as:
- A Gotcha callout in `all_gather.md` lines 240–241
- A "Program caching" prose paragraph in `all_reduce.md` lines 215–217
- An error table row in both `all_gather.md` line 310 and `all_reduce.md` line 257
The warning is stated four times across two files.
**Suggestion:** Keep the full Gotcha in `all_gather.md` (first occurrence, right context). In `all_reduce.md`, remove the standalone "Program caching" prose paragraph (replace with one sentence: "Program cache behaviour is identical to AllGather — keep tensor shapes constant."). The error table rows in both files can remain since they serve quick-reference lookup, but the `all_reduce.md` prose paragraph is pure duplication.

### [all_gather.md + all_reduce.md] — `cluster_axis` out-of-range error row in all three error tables
**Issue:** The row "`cluster_axis` out of range | Passing `cluster_axis=1` to a 1×N mesh | 1-D meshes have only one axis; omit `cluster_axis`" is word-for-word identical in `all_gather.md` line 311, `all_reduce.md` line 258, and `broadcast.md` line 283. This is not an operation-specific error — it is a mesh-level constraint that applies uniformly. Repeating it verbatim in three tables adds no information.
**Suggestion:** Keep the row in `all_gather.md` (first occurrence). In `all_reduce.md` and `broadcast.md`, replace the row with a note at the table footer: "See also: Section 2.1 error table for `cluster_axis` and topology errors common to all ops." This saves 2 full table rows (each row is ~2 rendered lines).

### [all_gather.md + all_reduce.md + broadcast.md] — L1-sharding performance snippet triplicated
**Issue:** All three operation files contain a nearly identical code block showing `ttnn.create_sharded_memory_config` → `ttnn.to_memory_config` → CCL op call for L1-sharding performance optimisation. The blocks differ only in variable names and comments:
- `all_gather.md` lines 258–266
- `all_reduce.md` lines 235–243
- `broadcast.md` lines 251–258
The sharding setup code is not operation-specific — it is generic TTNN memory layout API.
**Suggestion:** Keep the code snippet in `all_gather.md` (fullest context, first encounter). In `all_reduce.md`, replace the snippet with "The L1-sharding setup is identical to AllGather (see Section 2.1); substitute `partial_output` for `input_tensor`." In `broadcast.md`, likewise replace with a one-line forward reference. This removes ~16 lines of duplicate code across two files.

---

## MINOR Suggestions

### [all_gather.md] ~lines 27–36 ("When to use AllGather" table)
**Issue:** The table has 4 rows. Row 3 ("Small weight tensor replicated after initial sharding — Cheaper to gather once than to keep it sharded through non-shardable operations") is the weakest — it describes a general principle rather than a specific situation with a clear trigger condition. A reader who understands the other three rows can infer this case.
**Suggestion:** Remove row 3. The remaining 3 rows are concrete and distinct.

### [all_gather.md] ~lines 60–78 (Parameter notes — `output_tensor` and `sub_core_grids`)
**Issue:** `output_tensor` and `sub_core_grids` are advanced parameters that are rarely used. Their parameter note paragraphs (~4 lines each) add length to an introductory section. The API signature already shows they exist and default to `None`.
**Suggestion:** Merge the two notes into one sentence each, or move them to a "Advanced parameters" collapsed note at the end of the section. Combined saving: ~4 lines.

### [all_reduce.md] ~lines 59–67 ("When to use AllReduce" table — row 4)
**Issue:** Row 4 ("Embedding lookup with partitioned embedding table | Each device fetches some rows; the scatter-reduce produces the combined embedding") is the least common case and the most confusingly worded (scatter-reduce is ReduceScatter terminology, not AllReduce). It could mislead a reader.
**Suggestion:** Remove row 4. The remaining 3 rows cover the canonical use cases clearly.

### [broadcast.md] ~lines 109–117 (`all_broadcast` Parameter notes)
**Issue:** The Parameter notes for `all_broadcast` repeat constraint information already stated in the API signature comments (`# MUST be 1`, `# MUST be Linear`) and the preceding limitation Gotcha. `num_links` and `topology` each get their own note paragraph to say "don't change this" — which is already clear from the API block.
**Suggestion:** Remove the separate `num_links` and `topology` note paragraphs (lines 113–115). The API signature comments and the limitation Gotcha above the API block already carry this information. Keep only the `cluster_axis` note.

### [all_gather.md] ~lines 185–197 (Linear AllGather section)
**Issue:** The prose description of linear AllGather is accurate but the growing-data ASCII diagram (Dev 0 sends [A], Dev 1 sends [A,B], Dev 2 sends [A,B,C]) restates what the preceding prose already described clearly. The diagram adds visual confirmation but the same concept was already shown in a simpler form in Ch1.
**Suggestion:** Remove the 5-line ASCII diagram in the Linear AllGather section. Keep the prose description and the performance note. Saves ~7 lines.

### [index.md] ~lines 9–13 (bullet list in Overview)
**Issue:** The three-bullet description of the chapter files ("AllGather — the foundational...", "AllReduce — the standard tool...", "Broadcast / AllBroadcast — Less symmetric...") largely duplicates the Table of Contents descriptions immediately below it. The ToC already has a "Description" column.
**Suggestion:** Remove the bullet list (5 lines). The Overview paragraph above and the ToC below together cover the same content.

---

## VERDICT
- Crucial updates: yes

---
# Compression Analysis: Ch2 Basic Operations — Pass 2

## CRUCIAL Suggestions

None. Pass 1 eliminated all cross-file redundancy of significant size. Specific checks performed:

- `all_reduce.md` Concept ASCII block (ReduceScatter phase diagram): distinct from the AllGather rotation diagram — shows partial-sum accumulation, not forwarding. Not duplicate.
- `all_reduce.md` Under the Hood / Program structure: covers implementation detail (scratch region, EDM channel reuse) distinct from the Concept section's decomposition explanation. Not duplicate.
- `broadcast.md` GlobalSemaphore callout (3 lines): brief operational note, not a re-explanation of Ch1's full type definition.
- `broadcast.md` AllBroadcast output memory snippet: unique to AllBroadcast's multi-tensor return type; not present elsewhere.
- `all_gather.md` / `all_reduce.md` "Output layout" sections: `all_reduce.md` was already trimmed to 2 lines of prose in pass 1; no remaining duplication.
- All remaining open items (Linear AllGather ASCII diagram, "When to use" table rows, `index.md` bullet list, `all_broadcast` param note paragraphs) are MINOR and were already logged in pass 1. None rises to CRUCIAL.

## VERDICT
- Crucial updates: no
