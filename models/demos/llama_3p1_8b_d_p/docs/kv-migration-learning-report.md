<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# KV migration learning guide report

## Result

The live guide includes the reviewed layout extension.

Created files:

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

The top notice states that attention accuracy remains under investigation.

The layout section maps TP8 source heads into TP2 destination shards.

It maps SP4 source positions into the destination's full sequence axis.

Its logical walk covers head 5, position 1,024, layer 13, and example destination slot 1.

The walk stops before physical placement because the pinned Llama decode allocation has one slot.

## Important verified findings

The source prefill engine computes all 1,033 prompt tokens in two prefill chunks.

The destination caps reusable KV at the last 32-token boundary below the prompt end.

The example therefore migrates `[0,1024)` and replays positions 1,024 through 1,032 on decode.

The final prompt forward produces the first generated token at position 1,033.

Repeated-token pre-O heads fail in uncommitted Task 6 evidence.

Task 027 worst NL2 is 0.12608 for BF16 and 0.09887 for BF8_B. Task 028 residual is 0.10081.

Task 029 stock FP32 passes its independent SOURCE-HF pre-O gate at every valid chip.

Its minimum PCC is 0.99989028, and its maximum NL2 is 0.01513274.

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
details 21 21
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
Expandable detail pairs: 21 21
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
613 30968 notes/kv-migration-learning.md
461 50893 status/kv-migration-learning.html
236 15022 notes/kv-migration-learning-sources.md
```

The final Markdown and HTML sizes appear in the transfer evidence.

## Evidence boundaries

No device or model test ran for this guide.

Completed module results came from the task's recorded tt-metal evidence through revision `4cf42fb`.

The native manager source defines completion ordering. This work did not prove that ordering on five Galaxies.

The Llama-specific prefill and decode table exporters remain pending.

The pinned Llama decode entry has no `kv_migration_spec` hook.

No verified two-slot decode stride or destination NoC address exists yet.

Task 6 attention remains unaccepted despite passing post-O output.

The Llama full model, runtime integration, native cross-endpoint movement, and semantic decode comparison remain pending.

Chrome was unavailable in the documentation agent environment. Static HTML parsing and structure checks passed.

The parent reviewer must perform the final visual review before linking the page from the dashboard.

## Review corrections applied

- Phase D now states that replay does not overwrite migrated positions below 1,024.
- The evidence table reports Task 027 and Task 028 pre-O failures.
- Published `4cf42fb` module evidence is separate from uncommitted Task 6 attention evidence.
- The cancellation section separates UUID release from source slot and buffer safety.
- The native source slot lifecycle remains a required integration audit.
- The timeline labels the migrated prefix and keeps the final-token label inside its view box.
- The page links directly to this report and the source manifest.
- The layout extension separates config identity from physical placement.
- The head mapping follows the pinned TP2 cache shard mapper.
- The address walk stops before the unimplemented two-slot destination stride.
- The manifest includes direct pinned links for request planning, barriers, the PR client, and Blaze layout sources.

## Final hashes

The root handoff records the remote hash for every changed guide file.

The report cannot embed its own stable hash because that value would change the report.

## Root review

Root checked the pinned request-planning, tail-replay, device-barrier, device-session and drain-completion source paths. Browser review verified the page layout and found one clipped timeline label, now corrected. Root required explicit failing Task6 evidence and separation from published module results. Native Llama migration remains an integration gate.
