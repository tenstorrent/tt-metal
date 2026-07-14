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
4. **Look up your op** by grepping the CSV on the op path (column 1) — e.g. `grep -i '^data_movement/slice,' <csv>`. Pull only the row(s) you need into context; don't read the whole file in.

> **Do not use `read_file_content` for this sheet.** It truncates large sheets — you'll silently miss every op past roughly the "d"s — and renders ~15 phantom empty columns. `download_file_content` as CSV is complete and clean.

> **Do not delegate the fetch to a subagent.** The claude.ai connector authorizes only in the main interactive session; a spawned subagent hits the OAuth wall even though the tool schema loads. Fetch in your main session.

> **Do not commit the CSV.** It's an ephemeral local copy; committing it re-introduces the staleness this whole flow exists to avoid.

## Reading the CSV

One row per **(op, DeviceOperation, ProgramFactory variant)**. The 14 columns:

| # | Column | What it holds |
|---|---|---|
| 1 | `Op` | op path, e.g. `data_movement/slice` |
| 2 | `Device operation` | the `DeviceOperation` class |
| 3 | `Factory (variant)` | the ProgramFactory this row describes |
| 4 | `Concept` | factory concept (`descriptor`, `WorkloadDescriptor`, `legacy device-op`, …) |
| 5 | `Custom hash (compute_program_hash)` | declares a custom hash? |
| 6 | `Runtime-args update (override_runtime_arguments / get_dynamic_runtime_args)` | custom override / dynamic-RTA hook? |
| 7 | `Pybind descriptor (nb::class_ of device op)` | pybinds factory / device-op internals? |
| 8 | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | raw buffer address routed through an RTA/CRTA? |
| 9 | `Is safe to port?` | Diego's safety call |
| 10 | `Is able to port?` | Audrey's feasibility call |
| 11 | `Model` | which model surfaced it (`llama`, `resnet`, `other`) |
| 12 | `TensorParameter relaxation` | proposed relaxation, if any (`Structural`, `Coarsening`, `Address`, …) |
| 13 | `Factory definition path` | file the factory is defined in |
| 14 | `Declared in` | file the device-op is declared in |

Notes:

- Cell values are mostly `yes` / `no`, but some are `PR` (being handled in an in-flight PR) or other short tags. **Diego owns these classifications and their exact semantics** — treat the sheet as his cross-team readiness view, a useful prior, **not** ground truth that overrides your own audit findings. Where the sheet and your evidence disagree, your evidence wins; note the discrepancy in your report.
- **Ignore the trailing summary block.** The last few rows aren't ops — they're category totals (`With Smuggled pointers, 66`, and similar). A grep-by-op-path lookup skips them naturally.
- CSV export covers only the sheet's first tab (`Sheet1`), which today holds all the data.

## Troubleshooting

- **"requires authorization" / "token expired":** the human hasn't authorized — or needs to re-authorize — the Google Drive connector. Point them to [`../human/user_orientation.md`](../human/user_orientation.md) → *Google Drive MCP setup*. It's a claude.ai account-level action; it cannot be done from inside the session.
- **File not found / permission denied:** the sheet isn't shared with the human's Google account. Ask the owner (Diego) to broaden the share.
- **Tool name not found when you call it:** load its schema first (step 1); until then it's deferred and uncallable.
