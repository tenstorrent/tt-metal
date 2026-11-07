# Effort ledger serialization repair

Coordinator goal: repair check-large-files failure with semantic-preserving compact JSON serialization of doc/agentic_evals/efforts.json only. Inspect actual hook threshold and existing bytes, compare parsed/canonical hashes before and after, run real hook. No field/history edits or archive unless compact serialization cannot fit. Own this serialization and effort evidence only; no staging/commit, other central edits or devices. Device box0h; host box0.1h. Final evidence once; coordinator subsequently closes/reranks ledger using compact serialization.
