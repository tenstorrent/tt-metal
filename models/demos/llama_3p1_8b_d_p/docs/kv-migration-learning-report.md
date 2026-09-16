<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# KV migration learning guide report

## Result

The live guide includes final K512 Task 6 evidence and completed isolated decoder preparation.

Created files:

- `status/attention-diagnostic.html`
- `status/kv-migration-learning.html`
- `status/kv-migration-learning-sources.md`
- `status/kv-migration-learning-report.md`
- `notes/kv-migration-learning.md`
- `notes/kv-migration-learning-sources.md`
- `evidence/kv-migration-learning/language-check.txt`
- `notes/kv-migration-learning-report.md`

No production code, dashboard, index, commit, remote, or branch changed.

## Guide contents

The page includes a five-Galaxy overview, a 1,033-token worked example, and three inline SVG diagrams.

It explains prefill, native KVM movement, prompt-tail replay, first-token ownership, and SC4 decode.

It defines all three size units. It distinguishes prefill chunks, KV transfer chunks, and physical tiles.

It defines PCC, NL2, pre-O, and post-O before presenting attention evidence.

It includes cache geometry, payload arithmetic, completion ordering, slot lifecycle, failures, retry risks, tests, and exercises.

It also includes sticky navigation, expandable detail, responsive styling, print styling, and accessible SVG labels.

The page links directly to the source manifest and review report.

The top notice separates passing production stock parity from independent correctness and acceptance.

The layout section maps TP8 source heads into TP2 destination shards.

It maps SP4 source positions into the destination's full sequence axis.

Its logical walk covers head 5, position 1,024, layer 13, and example destination slot 1.

The walk stops before physical placement because the pinned Llama decode allocation has one slot.

## Important verified findings

The source prefill engine computes all 1,033 prompt tokens in two prefill chunks.

The destination caps reusable KV at the last 32-token boundary below the prompt end.

The example therefore migrates `[0,1024)` and replays positions 1,024 through 1,032 on decode.

The final prompt forward produces the first generated token at position 1,033.

Historical repeated-token pre-O heads fail on the original ring path.

Task 027 worst NL2 is 0.12608 for BF16 and 0.09887 for BF8_B. Task 028 residual is 0.10081.

Task 029 stock FP32 passes its independent SOURCE-HF pre-O gate at every valid chip.

Its minimum PCC is 0.99989028, and its maximum NL2 is 0.01513274.

Task 030 passes one repeated-token continuation for both cache dtypes.

Task 031 passes 12 exact boundary cases and all six historical Task 027 cases.

All 1,760 per-chip scalar metrics are finite. Exact cache invariants pass on all 32 chips.

Task 034 passed stock parity. Independent gates were still open at that stage.

Task 032 raw characterization completed. The candidate still misses original synthetic hash limits.

Worst cache-relative NL2 is about 9.3% at `[224,257)` for both cache dtypes.

Task 033 stock causal control matches the explicit-local-mask path on the worst row.

Both paths report PCC 0.9957571199646015 and NL2 0.0929282984724009 on chip 8, row 256.

This parity rules out mask introduction for that row. It does not identify one underlying operation.

The broader parity test covers ten intervals and two cache dtypes.

Task 034 completes that parity test. All 20 original interval-and-dtype comparisons pass.

BF16 minimum PCC is 0.9999989380946639, with maximum NL2 0.0015097182062556041.

BF8_B minimum PCC is 0.9999989379186667, with maximum NL2 0.0015106677367818795.

The guide explains the mixed-precision path and the scale sensitivity of NL2.

Task 035 passes 20 real-weight and 12 token-stream cases at unchanged gates.

Four validation groups pass. The K128 pulse PCC fails, so attention remained unaccepted at that stage.

BF16 oracle rounding lowers pulse PCC to about 0.992 while NL2 stays near 0.0023.

This baseline does not explain the observed device minimum PCC 0.97959.

Attempt 036 passes its first three pulses but fails last-pulse PCC.

Attempt 037 matches production and stock outputs exactly on all eight valid chips for both cache dtypes.

Its source minimum PCC is 0.9975303 for BF16 and 0.9974257 for BF8_B.

Its maximum source NL2 is about 0.01065. The source gate remains failed.

All 136,048 numeric values are finite. Devices closed cleanly at 12:45:50.624 UTC.

The control finds no production-specific mask or gather difference for this case.

A bounded host partial-rounding replay does not explain the actual error.

Its error-direction cosine is weak or negative.

Attempt 038 directly tests Q128 and K512. Both cache dtypes pass the original source and cache pulse gates.

Production and stock outputs match exactly on all eight valid chips.

Production source hash is `ab1808733e9d5ed4a515cdd94c35c5b5cd848a157878de3020ac991fc7fc10e2`.

Actual circular-buffer allocation is 1,241,088 bytes per core. The conservative gate is 1,273,856 bytes.

Attempt 041 passes the final suite with unchanged numerical gates.

Actual and verified exits are zero. Eight tests pass without skips on all 32 chips.

All 32 real-weight and token-stream cases pass. K512 pulse checks pass for both cache dtypes.

Devices close cleanly at 13:19:12.853 UTC.

The final suite supplies Task 6 acceptance evidence.

Original periodic-hash source failures remain characterization evidence. The exact internal cause remains unresolved.

Task 7 isolated preparation is complete. Root copied the exact three files into the canonical tree.

The launch contract is open. Device validation is starting in this order: `residual001`, `smoke002`, `full003`, then `Watcher004`.

No decoder device result exists yet.

Host decoder parity is 1.1920928955078125e-07 maximum absolute difference.

The guide labels this as host oracle parity, not device accuracy.

Current main's `MigrationKvManagerClient` uses the vendored legacy migration layer.

PR head `7ee35d8` adds the native `KvmClient` and registry key `kvm` through `EngineAdapter`.

Native DMK completion uses a final NoC write barrier before it publishes its device completion counter.

The device write session waits until all planned writes retire and every channel reports zero in-flight work.

The drain path writes credit after device settlement. The proposed client sends landing after all native command outcomes.

Pinned Blaze uses a 1x2 decoder mesh and shards eight K/V heads across two TP devices.

Global head 5 therefore maps to destination TP shard 1 and local head 1.

Pinned Llama decode sets `BATCH_SIZE=1` and `n_slots=1`. A destination slot 1 stride remains unimplemented.

## Exact checks

### HTML structure and local assets

Command:

```bash
python3 build_page.py
python3 - <<'PY'
from pathlib import Path
import re
from html.parser import HTMLParser
p = Path('kv-migration-learning.html').read_text()
HTMLParser().feed(p)
ids = set(re.findall(r'id="([^"]+)"', p))
refs = re.findall(r'href="#([^"]+)"', p)
print('details', p.count('<details>'), p.count('</details>'))
print('missing', sorted(set(refs) - ids))
print('external_scripts', bool(re.search(r'<script[^>]+src=', p, re.I)))
print('external_styles', bool(re.search(r'<link[^>]+(?:stylesheet|preload)', p, re.I)))
print('svg', p.count('<svg'), p.count('</svg>'))
PY
```

Output:

```text
details 24 24
missing []
external_scripts False
external_styles False
svg 3 3
```

### Language screening

The full command output is in `evidence/kv-migration-learning/language-check.txt`.

Output:

```text
Descriptive sentences above 25 words: 0
Numbered procedural steps above 20 words: 0
HTML parse: PASS
Internal link targets missing: []
Inline SVG pairs: 3 3
Expandable detail pairs: 24 24
External script tags: False
External stylesheet tags: False
Manual term review: PASS
```

This scan is a heuristic. It does not certify formal ASD-STE100 compliance.

### Arithmetic

Command:

```bash
python3 -c 'print(32*16*32*4352, (32*16*32*4352)/(1024**2)); print(64*16*32*4352, (64*16*32*4352)/(1024**2))'
```

Output:

```text
71303168 68.0
142606336 136.0
```

### File sizes

Command:

```bash
wc -l -c notes/kv-migration-learning.md status/kv-migration-learning.html notes/kv-migration-learning-sources.md
```

Output before final transfer:

```text
804 39954 notes/kv-migration-learning.md
577 61805 status/kv-migration-learning.html
307 20702 notes/kv-migration-learning-sources.md
```

The final Markdown and HTML sizes appear in the transfer evidence.

## Evidence boundaries

No device or model test ran for this guide.

Completed module results came from the task's recorded tt-metal evidence through revision `4cf42fb`.

The native manager source defines completion ordering. This work did not prove that ordering on five Galaxies.

The Llama-specific prefill and decode table exporters remain pending.

The pinned Llama decode entry has no `kv_migration_spec` hook.

No verified two-slot decode stride or destination NoC address exists yet.

Production K512 passes the final Task 6 suite. Decoder and full-model validation remain pending.

The old ring path still fails. Raw synthetic hash accuracy remains unresolved.

Task 033 does not explain all raw hash failures. Attention remained unaccepted at that stage.

Task 034 proves stock parity on identical inputs. It does not replace independent float-source or model gates.

Task 035 fails K128 pulse PCC. Task 036 fails its last K128 pulse.

Task 037 matches K128 stock but fails its source gate.

Task 038 passes the direct K512 pulse contrast. Task 041 passes the final K512 suite.

Original periodic-hash limitations remain characterization evidence.

Decoder device validation is starting. No result exists yet.

The Llama full model, runtime integration, native cross-endpoint movement, and semantic decode comparison remain pending.

Chrome was unavailable in the documentation agent environment. Static HTML parsing and structure checks passed.

The parent reviewer must perform the final visual review before linking the page from the dashboard.

## Review corrections applied

- Phase D now states that replay does not overwrite migrated positions below 1,024.
- The evidence table reports Task 027 and Task 028 pre-O failures.
- Published `4cf42fb` module evidence remains separate from later Task 6 attention evidence.
- The cancellation section separates UUID release from source slot and buffer safety.
- The native source slot lifecycle remains a required integration audit.
- The timeline labels the migrated prefix and keeps the final-token label inside its view box.
- The page links directly to this report and the source manifest.
- The layout extension separates config identity from physical placement.
- The head mapping follows the pinned TP2 cache shard mapper.
- The address walk stops before the unimplemented two-slot destination stride.
- The manifest includes direct pinned links for request planning, barriers, the PR client, and Blaze layout sources.
- The evidence now includes verified Task 030 and Task 031 prototype results.
- Historical ring failures remain separate from the passing test-only candidate.
- Task 032 keeps unresolved raw synthetic hash accuracy visible.
- Task 033 records the narrow stock-causal parity result without claiming a complete cause.
- Task 034 records all 20 passing production-to-stock parity cases.
- The guide now explains mixed precision and reference-scale effects on NL2.
- Attempt 035 keeps its passing groups and failing pulse PCC separate.
- Pulse conditioning is labeled a baseline, not a ceiling or full explanation.
- Attempts 036 and 037 preserve the historical K128 source PCC failure.
- Attempts 038 and 041 record the passing K512 path.
- Decoder preparation is canonical, while device validation has no result yet.

## Final hashes

The root handoff records the remote hash for every changed guide file.

The report cannot embed its own stable hash because that value would change the report.

## Root review

Root checked the pinned request-planning, tail-replay, device-barrier, device-session and drain-completion source paths. Browser review verified the page layout and found one clipped timeline label, now corrected. Root required explicit failing Task6 evidence and separation from published module results. Native Llama migration remains an integration gate.
