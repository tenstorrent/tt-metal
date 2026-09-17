<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# KV migration learning guide report

## Result

The guide includes accepted K512 attention, decoder evidence, documented 2K full-prefill contracts, performance through 64K, and the prefill-only native migration boundary.

Created publication candidates:

- `kv-migration-learning.md`
- `kv-migration-learning-sources.md`
- `kv-migration-learning.html`
- `kv-migration-learning-report.md`
- `kv-migration-learning-language-check.txt`

No production code, dashboard, index, commit, remote, or branch changed.


## Current publication status

The documented 2K local-layer and final-output contracts pass for both cache types.

Shared-runtime live 2K readiness and table checks pass at `8d051437`.

Raw accumulated-FP32 comparisons still include misses. The guide preserves them as characterization evidence.

Full-model execution and performance are published through 64K. The 128K device run is pending.

The 128K host-only native table test passes 4,194,304 entries with zero mismatches. It does not open a device or prove manager startup.

The independent 2K address oracle checks all 65,536 physical pages. Local packed-byte copy also passes.

The first native startup attempt seeded 576 nonzero pages and passed exact compute on 32 chips. Its manager configuration then used 64-bit ASIC identities where unsigned 32-bit device IDs were required.

That parsing failure occurred before manager construction. Recovery completed on all 32 chips, and a fresh health check passed.

Corrected IDs let the native manager open 32-device DMK and Mooncake. Startup then stopped on the etcd 3.3.25 v3 API HTTP 404. Native cleanup completed without a reset. Startup acceptance and cross-endpoint transfer remain pending.

Current migration work is prefill-only. Decode-side migration and SC4 tests are outside this scope.

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

Historical Task 032 recorded misses against the original synthetic hash limits.

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

Original periodic-hash source failures remain historical characterization evidence. The exact internal cause was not identified.

Task 7 isolated preparation is complete. Root copied the exact three files into the canonical tree.

At that preparation snapshot, the launch contract was open. The planned order was `residual001`, `smoke002`, `full003`, then `Watcher004`.

Later evidence accepted decoder revision `b18a9763`. The `full007` suite and `normal010` recovery smoke passed.

Host decoder parity is 1.1920928955078125e-07 maximum absolute difference.

The guide labels this as host oracle parity, not device accuracy.

At reviewed tt-d-gen revision `9a8c531`, `MigrationKvManagerClient` uses the vendored legacy migration layer.

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

The full command output is in `kv-migration-learning-language-check.txt`.

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
wc -l -c kv-migration-learning.md kv-migration-learning.html kv-migration-learning-sources.md
```

Output:

```text
  935  48382 kv-migration-learning.md
  684  71714 kv-migration-learning.html
  363  25506 kv-migration-learning-sources.md
 1982 145602 total
```

## Evidence boundaries

No device or model test ran for this guide.

Completed module results came from the task's recorded tt-metal evidence through revision `4cf42fb`.

The native manager source defines completion ordering. This work did not prove that ordering on five Galaxies.

The Llama prefill exporter and independent address oracle pass at 2K. Decode-side table validation remains pending.

The pinned Llama decode entry has no `kv_migration_spec` hook.

No verified two-slot decode stride or destination NoC address exists yet.

Production K512 attention and the decoder device suite pass. The documented 2K full-prefill contracts also pass.

The old ring path and original raw synthetic hash misses remain historical evidence. No ongoing investigation is claimed.

Historical Task 033 did not explain every original raw hash miss. Attention remained unaccepted at that stage.

Task 034 proves stock parity on identical inputs. It does not replace independent float-source or model gates.

Task 035 fails K128 pulse PCC. Task 036 fails its last K128 pulse.

Task 037 matches K128 stock but fails its source gate.

Task 038 passes the direct K512 pulse contrast. Task 041 passes the final K512 suite.

Original periodic-hash limitations remain characterization evidence.

Decoder revision `b18a9763`, the `full007` suite and the `normal010` recovery smoke are accepted evidence.

Native cross-endpoint movement, decode-side migration, SC4 placement and five-Galaxy semantic comparison remain pending.

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
- Tasks 032 through 041 remain labeled as historical attention evidence, with the original misses retained.
- Task 033 records the narrow stock-causal parity result without claiming a complete cause.
- Task 034 records all 20 passing production-to-stock parity cases.
- The guide now explains mixed precision and reference-scale effects on NL2.
- Attempt 035 keeps its passing groups and failing pulse PCC separate.
- Pulse conditioning is labeled a baseline, not a ceiling or full explanation.
- Attempts 036 and 037 preserve the historical K128 source PCC failure.
- Attempts 038 and 041 record the passing K512 path.
- Decoder preparation is historical; later `full007` and `normal010` device evidence is accepted.
- Current migration work is prefill-only; native transfer, decode-side migration and SC4 validation remain pending.
- The three device-ID namespaces are separate and tied to their actual APIs.
- The 128K table-size claims are limited to the pinned host-only export/import result.

## Final hashes

The root handoff records the remote hash for every changed guide file.

The report cannot embed its own stable hash because that value would change the report.

## Root review

Root checked the pinned request-planning, tail-replay, device-barrier, device-session and drain-completion source paths. Browser review verified the page layout and found one clipped timeline label, now corrected. Root required explicit failing Task6 evidence and separation from published module results. Native cross-endpoint Llama migration remains an integration gate.
