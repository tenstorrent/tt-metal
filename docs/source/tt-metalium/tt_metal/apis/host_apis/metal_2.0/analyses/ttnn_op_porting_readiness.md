# TTNN Op Porting Readiness

**Author:** Audrey and Claude

**Purpose:** Give an auditing Claude the live, per-factory porting-readiness data — Diego's "Operations analysis" sheet — plus how to fetch a fresh copy and how to read it. The [audit recipe](../ai/port_op_to_metal2_audit.md) references this doc when it needs the data.

> **Fetch this data yourself, in your main session, every run.** It is deliberately **not** checked into the repo — the sheet is edited continuously, so a committed copy goes stale fast. Re-download it **even if a `.csv` is already sitting in this folder**; a stale local copy is worse than none.

## Source (live)

Google Sheet — *"Operations analysis"*, owned by Diego (`dgomez@tenstorrent.com`):
https://docs.google.com/spreadsheets/d/1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw/edit?usp=sharing

- **File ID:** `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`
- Access is through the **claude.ai Google Drive MCP connector**. The human authorizes it once — see [`../human/user_orientation.md`](../human/user_orientation.md) → *Google Drive MCP setup*. You **cannot** authorize it from inside a session.

## Fetch it (do this every run)

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

**Reference every column by its header name, not its position.** The sheet evolves — Diego adds and reorders columns — but two guarantees hold: **existing column names never change, and no column is ever deleted.** So read the **header row** (row 1) to find a column by name; never hard-code "column N." A grep-by-op-path lookup gives you the row; the header row tells you which cell is which. (Do not reproduce a positional column list here — it goes stale the moment a column is inserted.)

One row per **(op, DeviceOperation, ProgramFactory variant)** — an op with several factories has several rows. The columns the audit reads, by name:

- **`Op`** — op path (`data_movement/slice`); the lookup key.
- **`Device operation`** / **`Factory (variant)`** — which DeviceOperation and ProgramFactory the row describes.
- **`Concept`** — the factory's *current* concept: `descriptor`, `WorkloadDescriptor`, `legacy device-op`, or `MetalV2` (already ported).
- **`Custom hash (…)`** — declares a custom `compute_program_hash`?
- **`Runtime-args update (…)`** — has the `get_dynamic_runtime_args` hook? (Possible only on `descriptor` / `WorkloadDescriptor` concepts — a cross-column invariant.)
- **`Pybind descriptor (…)`** — pybinds factory / device-op internals (`create_descriptor`)?
- **`Smuggled pointer (…)`** — an un-annotated pointer argument (a PD-migration bug); feeds `Is safe to port?`.
- **`Is safe to port?`** — Diego's correctness call (`yes` / `no` / `warning` / blank): did the prior PD migration introduce a bug?
- **`Is able to port?`** — the derived Metal-2.0 **gate** verdict. Its derivation is documented in the [audit's TTNN factory concept prerequisite](../ai/port_op_to_metal2_audit.md#ttnn-factory-concept-prerequisite).
- **`TensorParameter relaxation`** — proposed relaxation, if any (e.g. `dynamic_tensor_shape`, `match_padded_shape_only`, `none`).
- **`Op-owned tensors?`** / **`Secretly SPMD Workload?`** — feed the target concept and the `WorkloadDescriptor` escape respectively.
- **`Factory definition path`** / **`Declared in`** — the source files, for the cross-check.

The sheet may carry other, informational columns (e.g. `Model`); find any of them by header too.

Notes:

- Cell values are mostly `yes` / `no`, but some are `warning`, `PR` (handled in an in-flight PR), blank, or other short tags. **Diego owns these classifications.** The sheet is a shortcut to work you'd otherwise do by hand — **trust it, but cross-check the cheaply-checkable columns against the code** (per the audit subject). On a code-vs-sheet **conflict, or a missing op, the sheet is broken → gate the port** and route it to the readiness-sheet owner to reconcile.
- **Ignore the trailing summary block.** The last rows aren't ops — they're category totals (`With Smuggled pointers, 66`, and similar) and stray labels. A grep-by-op-path lookup skips them naturally.
- CSV export covers only the sheet's first tab (`Sheet1`), which today holds all the data.

## Troubleshooting

- **"requires authorization" / "token expired":** the human hasn't authorized — or needs to re-authorize — the Google Drive connector. Point them to [`../human/user_orientation.md`](../human/user_orientation.md) → *Google Drive MCP setup*. It's a claude.ai account-level action; it cannot be done from inside the session.
- **File not found / permission denied:** the sheet isn't shared with the human's Google account. Ask the owner (Diego) to broaden the share.
- **Tool name not found when you call it:** load its schema first (step 1); until then it's deferred and uncallable.
