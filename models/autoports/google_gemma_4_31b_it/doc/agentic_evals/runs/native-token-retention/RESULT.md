# Native generated-token retention

Supported for successfully parsed mini native responses: merge `return_token_ids: true` into only the autoport SWE `agent_kwargs.config.model.model_kwargs.extra_body`. Preserve existing sampling, thinking, output cap and timeout. Keep nonstreaming and default `include_reasoning=true`.

vLLM returns the entire engine output token sequence, including reasoning/control/tool tokens, independently of visible parser fields. Exact CI LiteLLM1.108.1 places unknown choice fields under `provider_specific_fields`; mini saves `response.model_dump()`. Expected stored path: `messages[i].extra.response.choices[0].provider_specific_fields.token_ids`. The exact package source was downloaded privately without installing it; provenance is in `bringup/artifacts/native-token-retention/package-source.json`.

This does **not** solve the11 discarded FormatError responses: mini parses actions before saving the raw response, so a parser exception loses it. That gap needs a separate generic retention change; no model request option fixes the ordering. A capped loop that still parses into a native tool action can retain its exact raw onset with this option.

Streaming vLLM supports delta IDs even on parser-empty/control-token chunks, but current mini is nonstreaming; changing transport is unnecessary. Hiding reasoning suppresses IDs, so preserve the existing default. IDs add private response storage, not sampling changes. A short same-seed generic native pair plus exact LiteLLM/mini serialization check is proposed before acceptance, separately authorized by the coordinator. No hardware, publication or model changes occurred. Detailed source references and limitations are in result.json.
