# Optional suffix512 fixture review

**PASS — fixture validity only. Not admitted or executed.**

All 36 candidates match their corresponding admitted85 anomaly onsets: exactly the last 512 conditioning IDs, unchanged chosen continuation IDs, and byte-equivalent metadata except the added optional-control provenance. Source/candidate hashes and suffix ranges match. There are no missing sources, duplicate cases, or prefixes of 512 tokens or fewer; original prefixes span 1175–76441 tokens.

Sampling metadata and diagnostic seeds 42/43/44 are preserved; absent native seeds remain absent. Each job retains chosen-only teacher forcing and 3 free trials. The free-generation cap intentionally increases from 256 to 512; this is documented, not identical generation length. Stop-related source metadata and all chosen stop IDs are unchanged. Both paths use the same runner/generator EOS handling; no stop override is added. Total forced positions: 2019; free cap:55,296 tokens per precision.

The suffix may begin inside a turn/protocol sequence and loses earlier context. No BOS or template text was inserted. These checks do not establish native equivalence, semantic reproduction or correctness. Native IDs remain unavailable, including the unresolved one-token prompt-count gap. This is a direct mechanical verification by the fixture-preparation agent, not an independent stage signoff.

Gate: first establish persistence at full-TT native-length conditioning; then obtain root admission for matched served/full-TT suffix tests. Conditional HF is allowed only if the identical short TT input reproduces the relevant anomaly. A negative short result is inconclusive for the full prefix. Parser-valid tool markers alone are not anomalous, and parser validity does not excuse repetition or stray text.

Manifest SHA256: `06ac265b5bedf25f66790b3750ba47458999914d2aa93ff09a08dc2de97a86df`. Case-list SHA256: `7eed047a0fb8ffa351351d69055b209a630502c27298b9120b7d79c5e57b530c`. Detailed hash-bound check record: [JSON](swe85-optional-short-fixtures-review.json).
