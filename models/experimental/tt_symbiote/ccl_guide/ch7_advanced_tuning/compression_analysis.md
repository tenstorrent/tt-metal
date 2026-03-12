# Compression Analysis: Ch7 Advanced Tuning

## Summary

- Total files analyzed: 4
- Estimated current line count: ~966
- Estimated post-compression line count: ~780
- Estimated reduction: ~19%

---

## CRUCIAL Suggestions

### `topology_and_links.md` lines 17–57 — Ring/Linear topology re-explains Ch1 §1.2 content

**Issue:** The "Linear topology" and "Ring topology" subsections (lines 17–57, ~40 lines) each contain an ASCII device chain diagram, a bullet list of properties, and a usage sentence. Ch1 §1.2 (`hardware_topology.md`) already covers both topologies at the same level of detail: the same one-directional chain vs. bidirectional ring semantics, the same "requires wrap-around cable" constraint, the same bandwidth trade-off reasoning. The content is not just similar — it repeats the same facts with the same ASCII art.

The "When to choose each" table (lines 46–57) adds modest value as a synthesis, but 5 of its 7 rows are derivable directly from the two paragraphs being deleted.

**Suggestion:** Replace the two subsection bodies (lines 17–44) with a single forward ref: "For the physical Ring and Linear topology definitions and cable requirements, see [Ch1 §1.2 — Hardware Topology](../ch1_introduction/hardware_topology.md#inter-chip-communication-erisc-and-edm)." Keep the "When to choose each" table (12 lines) as it is a useful synthesis. Keep the `Topology` enum listing (lines 9–15) as it shows the full enum values not in Ch1.

**Estimated savings:** ~28 lines

---

### `subdevice_and_semaphores.md` lines 7–100 — SubDevice concept + setup re-explains Ch4 §4.1

**Issue:** The "What is a SubDevice?" section (lines 7–46, ~40 lines) includes the C++ class definition from `sub_device.hpp`, the `SubDeviceId` strong-type definition, the partitioning ASCII grid, and a prose explanation of core partitioning. "SubDevice setup (host-side)" (lines 75–100, ~26 lines) shows the full Python setup code including `create_sub_device_manager`, `load_sub_device_manager`, and `get_sub_device_ids`.

Ch4 §4.1 (`why_async.md`) has a dedicated "SubDevice: Partitioning Tensix Cores" section with: the same concept explanation, an equivalent core partition ASCII (`SubDevice 0: cores 0..7` / `SubDevice 1: cores 8..119`), and substantially the same Python setup code (`create_sub_device_manager`, `load_sub_device_manager`, `SubDeviceId`). The content overlap is ~55 lines.

**Suggestion:** Replace both the concept section and the setup code with a 3-line summary: "A SubDevice partitions the Tensix core grid so CCL and compute can be dispatched concurrently. For concept and creation code, see [Ch4 §4.1 — SubDevice: Partitioning Tensix Cores](../ch4_async_overlap/why_async.md#subdevice-partitioning-tensix-cores). This section focuses on `subdevice_id` parameter semantics and the common mistakes specific to advanced use." Then keep the `subdevice_id parameter` section (lines 49–74) which explains the dispatch semantics — that is unique synthesis content.

**Estimated savings:** ~55 lines

---

### `subdevice_and_semaphores.md` lines 104–167 — GlobalSemaphore lifecycle re-explains Ch1 §1.3 + Ch4 §4.1

**Issue:** The GlobalSemaphore section (lines 104–167, ~64 lines) covers: the concept (lines 104–110), creation API with two overloads (lines 112–133), address querying (lines 135–141), reset between iterations (lines 143–152), and a 7-row table of how many semaphores each op needs (lines 154–166).

- The creation API and reset Gotcha are already in Ch4 §4.1 (`why_async.md`) "Semaphores in Async Operations" (the creation code snippet with 3 semaphores and the reset Gotcha).
- The concept ("L1 memory location that multiple Tensix cores can atomically read/increment/wait on") is from Ch1 §1.3.
- The address querying API (`ttnn.get_global_semaphore_address`) is used internally and not needed at the user-guide level.

The semaphore-count table (lines 154–166) is the only content not duplicated elsewhere — it is a useful synthesis unique to Ch7.

**Suggestion:** Replace the concept/creation/address/reset prose (lines 104–152, ~49 lines) with a 3-line summary: "GlobalSemaphore objects are created with `ttnn.create_global_semaphore(device, cores, initial_value)` and must be reset with `ttnn.reset_global_semaphore_value(sem, 0)` between iterations. For the full creation and reset pattern, see [Ch4 §4.1 — Semaphores in Async Operations](../ch4_async_overlap/why_async.md#semaphores-in-async-operations). The reset rules below apply regardless of which op is using the semaphore." Keep the semaphore-count table (lines 154–166), the Multi-semaphore patterns section (lines 170–202), and the Semaphore reset timing rules (lines 206–227) — all are unique synthesis content.

**Estimated savings:** ~46 lines

---

### `subdevice_and_semaphores.md` lines 230–276 — Async overlap section re-explains Ch4 §4.1 + §4.3

**Issue:** The "Async CCL + SubDevice: overlapping CCL with compute" section (lines 230–276, ~47 lines) includes:
- A motivating statement (lines 230–232): already in Ch4 §4.1 intro
- Two timeline ASCIIs (lines 233–248): near-identical to the Ch4 §4.1 and §4.3 Pattern 1 timelines
- The `t_AG ≈ t_MM` tuning insight (line 248): restates Ch4 §4.3 Pattern 1 Gotcha
- A full dispatch code block (lines 250–275): substantively the same as Ch4 §4.1 SubDevice setup code + §4.3 Pattern 4 correct dispatch pattern

Ch4 §4.1 has the exact same pipeline timeline diagram and the condition "`t_AG` and `t_MM` should be balanced." Ch4 §4.3 Pattern 4 has the correct/incorrect code pair for SubDevice dispatch. This section adds no new information.

**Suggestion:** Replace the entire section (47 lines) with 2 sentences: "For the overlap timeline and dispatch pattern, see [Ch4 §4.3 — Pattern 4: SubDevice Dispatch](../ch4_async_overlap/overlap_patterns.md#pattern-4-subdevice-dispatch). The key rule is that both SubDevice ops must be dispatched before either is waited on — waiting between dispatches eliminates overlap."

**Estimated savings:** ~45 lines

---

### `kernel_internals.md` lines 361–376 — Trace mode constraints re-list Ch4 §4.3 Pattern 5

**Issue:** The "Constraints in trace mode" numbered list (lines 363–371, ~9 lines) and the preceding paragraph (lines 361–362) re-state the 5 constraints from Ch4 §4.3 Pattern 5 ("Traced Overlap"):
1. No host-initiated operations during capture → Ch4 §4.3 Pattern 5: "no synchronize_devices inside the trace"
2. Tensor shapes must be fixed → Ch4 §4.3 constraint 2
3. GlobalSemaphore must be created before capture → Ch4 §4.1 constraint: "Persistent buffers must be allocated before the trace is captured"
4. `reset_global_semaphore_value` must be outside the trace → Ch4 §4.3 Pattern 5 Gotcha
5. Persistent output buffer addresses must be stable → Ch4 §4.3 constraint 5

The "Trace mode and SubDevices" note (lines 373–376) is unique (4 lines).

**Suggestion:** Replace the "What trace mode is" paragraph + constraints list (lines 323–371, ~49 lines) with a 4-line summary: "Trace mode records Metal program dispatches for replay with minimal host overhead — critical for decode loops. For the full capture/replay pattern and constraints with async CCL, see [Ch4 §4.3 — Pattern 5: Traced Overlap](../ch4_async_overlap/overlap_patterns.md#pattern-5-traced-overlap). The persistent buffer pattern from that section applies here." Keep the "Persistent output buffers" code block (lines 329–358, ~30 lines) as it is the only Ch7-specific implementation detail — the Ch4 §4.3 pattern is a simpler example and §7.3 adds the `ttnn.capture_trace` context manager form. Also keep the SubDevices note (lines 373–376).

Wait — re-reading: lines 323–358 contain the "What trace mode is" paragraph and the persistent buffers code. Lines 361–371 are the constraints. The persistent buffers code block (30 lines) is worth keeping as it adds the `with ttnn.capture_trace(...)` pattern not shown in Ch4. So trim only the constraints restatement.

**Revised suggestion:** Replace only the "Constraints in trace mode" section (lines 361–371, ~11 lines) with a 2-line forward ref: "For the full set of trace mode constraints with async CCL, see [Ch4 §4.3 — Pattern 5: Traced Overlap](../ch4_async_overlap/overlap_patterns.md#pattern-5-traced-overlap). Constraint 4 (semaphore reset outside the trace) is most frequently violated — reset in the replay loop, not inside the capture block."

**Estimated savings:** ~9 lines

---

### `subdevice_and_semaphores.md` lines 281–292 — Common mistakes 1–3 duplicate Ch4 Gotchas

**Issue:** Common mistakes 1, 2, and 3 (lines 281–292, ~12 lines):
- Mistake 1 ("Forgetting to reset the semaphore between iterations"): duplicates the Gotcha in Ch4 §4.1 Semaphores section and the §7.2 reset section just above it (lines 143–152).
- Mistake 2 ("Wrong core range for the semaphore"): duplicates Ch4 §4.1 Semaphores Gotcha about core range.
- Mistake 3 ("Resetting before the async op completes"): same timing rule stated on lines 210–212 in this same file.

Mistakes 4 and 5 (lines 293–299) are unique (L1 leak from creating semaphores in loop; core range mismatch with fused ops).

**Suggestion:** Remove mistakes 1–3 (12 lines) entirely. They are triple-documented: in Ch4 §4.1, in the §7.2 Semaphore reset section, and here. The remaining mistakes 4 and 5 are the only new content; renumber them 1 and 2.

**Estimated savings:** ~12 lines

---

## MINOR Suggestions

### `topology_and_links.md` lines 127–149 — cluster_axis 2D mesh ASCII

**File:** `topology_and_links.md`, lines 127–149 (~23 lines)

The full 4×8 grid ASCII (`D00`–`D37`) repeats conceptual content from Ch2 §2.1 `cluster_axis` parameter notes. The ASCII here is more detailed (shows actual device labels) but the concept has already been introduced. Consider trimming to 8 lines showing just one example ring path rather than the full 32-device grid. Saves ~10 lines.

---

### `kernel_internals.md` lines 151–170 — EDM compile-time args table

**File:** `kernel_internals.md`, lines 151–170 (~20 lines)

The `get_compile_time_args()` listing shows 10 internal EDM kernel arg slots with comments. This is low-level firmware internals unlikely to be directly actionable for tuning. The only actionable items (`num_buffers_per_channel` and `num_links` → recompile) are documented in the surrounding prose and the program caching table. Consider removing lines 153–170 and keeping only the prose note that compile-time arg changes trigger recompilation. Saves ~15 lines.

---

### `kernel_internals.md` lines 174–192 — EDM runtime args listing

**File:** `kernel_internals.md`, lines 174–192 (~19 lines)

The raw runtime arg layout (`handshake_addr`, `receiver_channels_offset`, per-channel fields) is internal implementation detail. The only reader-facing conclusion is: "runtime args can be updated without recompile" (line 192), already stated in the Program Caching section. Remove the arg listing; keep the conclusion sentence. Saves ~17 lines.

---

## VERDICT

- Crucial updates: yes
- Summary: 6 CRUCIAL issues (~195 lines saveable), 3 MINOR issues (~42 lines). Chapter 7 has the highest redundancy of any chapter, primarily because `subdevice_and_semaphores.md` re-explains Ch4 §4.1 material across three separate sections (SubDevice concept, GlobalSemaphore lifecycle, and async overlap). The largest wins are the SubDevice re-explanation (C2, ~55 lines), the GlobalSemaphore lifecycle re-explanation (C3, ~46 lines), and the async overlap section (C4, ~45 lines).

---
# Compression Analysis: Ch7 Advanced Tuning — Pass 2

## CRUCIAL Suggestions

None. All 6 pass-1 CRUCIAL compressions are confirmed applied:

- C1: `topology_and_links.md` — Ring/Linear subsection bodies replaced with 1-line forward ref to Ch1 §1.2 (confirmed at line 17); "When to choose each" table and enum listing retained.
- C2: `subdevice_and_semaphores.md` — SubDevice concept + setup replaced with 3-line forward ref to Ch4 §4.1 (confirmed at line 9); `subdevice_id` parameter semantics retained.
- C3: `subdevice_and_semaphores.md` — GlobalSemaphore lifecycle replaced with 3-line forward ref to Ch4 §4.1 (confirmed at line 41); semaphore-count table, multi-semaphore patterns, and reset timing rules retained.
- C4: `subdevice_and_semaphores.md` — Async overlap section replaced with 2-sentence forward ref to Ch4 §4.3 (confirmed at lines 119–121).
- C5: `kernel_internals.md` — Trace mode constraints list replaced with 2-line forward ref to Ch4 §4.3 Pattern 5 (confirmed at line 363); persistent buffer code block retained.
- C6: `subdevice_and_semaphores.md` — Common mistakes 1–3 removed; remaining 2 unique mistakes renumbered (confirmed: only 2 mistakes present).

No new CRUCIAL redundancies identified in the post-pass file states. The EDM deep-dive, NOC transfers, program caching table, trace mode persistent buffer pattern, and L1 budget sections all contain unique content not present in earlier chapters.

## VERDICT

- Crucial updates: no
