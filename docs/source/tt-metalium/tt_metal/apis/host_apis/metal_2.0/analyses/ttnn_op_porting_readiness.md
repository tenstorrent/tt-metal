# TTNN Op Porting Readiness

**Author:** Audrey and Claude

**Purpose:** Give an auditing Claude the live, per-factory porting-readiness data — Diego's "Operations analysis" sheet — plus how to fetch a fresh copy and how to read it. The [audit recipe](../ai/audit/metal2_audit.md) references this doc when it needs the data.

> **Fetch this data yourself, in your main session, once per session.** It is deliberately **not** checked into the repo — the sheet is edited continuously, so a committed copy goes stale fast. Re-download it **even if a `.csv` is already sitting in this folder**; a stale local copy is worse than none. What you must never do is reuse a checked-in copy or one from an earlier session — but a copy you pulled yourself an hour ago is fine, so a session auditing several ops fetches once and reuses it.

## Source (live)

Google Sheet — *"Operations analysis"*, owned by Diego (`dgomez@tenstorrent.com`):
https://docs.google.com/spreadsheets/d/1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw/edit?usp=sharing

- **File ID:** `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`
- Access is through the **claude.ai Google Drive MCP connector**. The human authorizes it once — see [`../human/READ_ME_FIRST.md`](../human/READ_ME_FIRST.md) → *Google Drive MCP setup*. You **cannot** authorize it from inside a session.

## Fetch it (once per session)

Run from your checkout root. Target the CSV at the folder this doc lives in:

```
docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/ttnn_op_porting_readiness.csv
```

1. **Load the MCP tool.** It's a deferred tool — load its schema before you can call it:
   ToolSearch with query `select:mcp__claude_ai_Google_Drive__download_file_content`.
2. **Download as CSV.** Call `mcp__claude_ai_Google_Drive__download_file_content` with:
   - `fileId`: `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`
   - `exportMimeType`: `text/csv`

   The sheet is large (~150 KB), so instead of returning inline the harness **saves the result to a tool-results file and prints its path**. You'll see a message that the result exceeded the token limit and was saved — **this is expected, not a failure.** Use that saved path in the next step. (The saved file is JSON of the shape `{"content": "<base64>", ...}`. If a smaller export ever *does* come back inline, it's the same shape — write it to a file first.)
3. **Decode into the analyses folder.** Extract the base64 `content` field and decode it:
   ```bash
   jq -r '.content' <SAVED_TOOL_RESULT_PATH> \
     | base64 -d > docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/ttnn_op_porting_readiness.csv
   ```
   Use the exact `<SAVED_TOOL_RESULT_PATH>` the tool reported. (No `jq`? Any base64-decode of the `content` field works.)
4. **Look up your op** by grepping the CSV for the op path — e.g. `grep -i 'data_movement/slice,' <csv>`. Grep the *value* (it's distinctive), not a fixed column position. Pull only the row(s) you need; then map each cell to its column by reading the **header row** (see *Reading the CSV*), never by absolute position. Don't read the whole file in.

> **Do not use `read_file_content` for this sheet.** It truncates large sheets — you'll silently miss every op past roughly the "d"s — and renders ~15 phantom empty columns. `download_file_content` as CSV is complete and clean.

> **Do not delegate the fetch to a subagent.** The claude.ai connector authorizes only in the main interactive session; a spawned subagent hits the OAuth wall even though the tool schema loads. Fetch in your main session.

> **Do not commit the CSV.** It's an ephemeral local copy; committing it re-introduces the staleness this whole flow exists to avoid.

## Reading the CSV

**Reference every column by its header name, not its position.** The sheet evolves — columns get added and reordered — so read the **header row** (row 1) to find a column by name; never hard-code "column N." **Match on the distinctive stem of the header, not the whole string**: the parenthetical suffixes are edited from time to time (`Override runtime args method?` has carried more than one), so an exact-string match on a full header can miss a column that is right there. Columns are added far more often than removed, but they *do* get retired, so never assume a name you remember is still present — resolve it against the header row you just fetched. A grep-by-op-path lookup gives you the row; the header row tells you which cell is which. (Do not reproduce a positional column list here — it goes stale the moment a column is inserted.)

One row per **(op, DeviceOperation, ProgramFactory variant)** — an op with several factories has several rows. The columns the audit reads, by name:

> **This list is for *finding and reading* the columns.** What to *do* with a value — which ones gate, and who a block routes to — lives in the [audit's TTNN factory concept prerequisite](../ai/audit/metal2_audit.md#ttnn-factory-concept-prerequisite), and is deliberately not repeated here.

- **`Op`** — op path (`data_movement/slice`); the lookup key.
- **`Device operation`** / **`Factory (variant)`** — which DeviceOperation and ProgramFactory the row describes.
- **`Concept`** — the factory's *current* concept: `descriptor`, `WorkloadDescriptor`, `legacy device-op`, or `MetalV2` (already ported).
- **`Custom hash (…)`** — declares a custom `compute_program_hash`? A companion column tracks the *backdoor* route (a hand-written `attribute_values` / `to_hash`).
- **`Runtime-args update (get_dynamic_runtime_args)`** — has the deprecated `get_dynamic_runtime_args` hook? (Possible only on `descriptor` / `WorkloadDescriptor` concepts — a cross-column invariant.) The hook is deprecated and nearly retired. The hook lives on the *device-op* and may fire for only some of its factories — see the [audit cross-check](../ai/audit/metal2_audit.md#ttnn-factory-concept-prerequisite).
- **`Override runtime args method?`** — has an `override_runtime_arguments` method? Read it together with `Concept`: on a `descriptor`/PD op this is the Metal 2.0 target-concept signal, while on a *legacy* op the same method name is just part of the legacy-concept signature. Distinct from `get_dynamic_runtime_args` above — the two are TTNN's successive runtime-arg-update mechanisms.
- **`Pybind descriptor (…)`** — pybinds factory / device-op internals (`create_descriptor`)?
- **`Smuggled pointer (…)`** — an un-annotated pointer argument (a PD-migration bug).
- **`Is able to port?`** — the derived verdict, and the cell the audit reads. **You will not be able to see how it was reached**: the CSV carries values, not formulas, so a verdict that doesn't follow from the columns in front of you is normal — the derivation can turn on things the audit has no view of. Read the cell; don't vet it. See the [audit's routing rules](../ai/audit/metal2_audit.md#ttnn-factory-concept-prerequisite) for what to do with a `no` you can't attribute.
- **`TensorParameter relaxation`** — whether the op needs a relaxation of the strict `TensorSpec` match. Values are short tags (`none`, `dynamic`, and analysis-state markers); **quote the cell verbatim** rather than paraphrasing it, since the vocabulary distinguishes work that is queued from work already scheduled.
- **`Known op issues`** — free text, reserved for problems that must be cleared before a port. Read the cell; it names its own owner.
- **`Op Classification`** — a derived summary of an op's overall state, including whether it reads as *broken*.
- **`Op-owned tensors?`** / **`Secretly SPMD Workload?`** — feed the target concept and the `WorkloadDescriptor` escape respectively.
- **`Factory definition path`** / **`Declared in`** — the source files, for the cross-check.

The sheet may carry other, informational columns (e.g. `Model`); find any of them by header too.

Notes:

- Cell values are mostly `yes` / `no`, but some are `warning`, `PR` (handled in an in-flight PR), blank, or other short tags. **Diego owns these classifications.** The sheet is a shortcut to work you'd otherwise do by hand — **trust it, but cross-check the cheaply-checkable columns against the code** (per the audit subject). On a code-vs-sheet **conflict on one of those columns, or a missing op, the sheet is broken → gate the port** and route it to the readiness-sheet owner to reconcile. That claim is only available where you hold independent evidence — a *derived* cell you can't explain is not a conflict, and the audit routes it differently.
- **Ignore the trailing summary block.** The last rows aren't ops — they're category totals (`With Smuggled pointers, 66`, and similar) and stray labels. A grep-by-op-path lookup skips them naturally.
- CSV export covers only the sheet's first tab (`Sheet1`), which today holds all the data.
- **CSV flattens formulas to values.** Several columns are computed, not hand-entered, and you see only what they evaluated to. That is the intended working set — derived cells are read, not vetted (see `Is able to port?` above) — but know that it's what you're holding, so an unexplained value reads as *out of my view* rather than *wrong*.

## Troubleshooting

- **"requires authorization" / "token expired":** the human hasn't authorized — or needs to re-authorize — the Google Drive connector. Point them to [`../human/READ_ME_FIRST.md`](../human/READ_ME_FIRST.md) → *Google Drive MCP setup*. It's a claude.ai account-level action; it cannot be done from inside the session.
- **File not found / permission denied:** the sheet isn't shared with the human's Google account. Ask the owner (Diego) to broaden the share.
- **Tool name not found when you call it:** load its schema first (step 1); until then it's deferred and uncallable.
