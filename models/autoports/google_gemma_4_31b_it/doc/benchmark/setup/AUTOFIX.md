# Stage 11 accounting integration

The failing contract requires lossless full-phase asynchronous completion events, request identity and matching work accounting. AutoDebug confirms the adapter has neither a completion export nor the final host-sampling/token-handling boundary.

Hypotheses checked: direct adapter return timers (rejected: decode enqueue, not completion); existing host-conversion callback (partial: plugin sampling follows); existing plugin completion timestamps (not exported); generic request OTel (review in completion_export_review.md). The pinned stock runtime requirement forbids applying a new plugin observer. No synchronization or profiler was added.

Permitted changes: observed autoport module path and device IDs in startup registration; an owned server-selection hook; a fail-closed collector readiness check; separate latency-only evidence collection within the original stage deadline. No partial phase timing is represented as full_phase_wall_time. A complete collector remains unresolved, with a proposed optional observer requiring operator authorization.

Client build failures were repaired with separate environments: upstream lm-eval0.4.13/transformers4.57.6 for accuracy; vLLM0.26.0 empty target for performance, isolated package with existing server dependency packages available. CLI help passes. GPQA dataset authentication is unavailable; no replacement documents or fabricated scores.
