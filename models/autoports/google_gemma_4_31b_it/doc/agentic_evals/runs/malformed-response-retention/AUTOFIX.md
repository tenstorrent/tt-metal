# AutoFix disposition

AutoDebug confirmed the first inconsistent state transition: mini2.2.8 parses actions while constructing extra, before the full response dump becomes durable. FormatError unwinds to the existing agent loop, which stores only exception feedback. Verified directly against LitellmModel.query, InterruptAgentFlow.messages, DefaultAgent.run/save and Harbor usage extraction.

Preferred upstream repair: catch only FormatError around _parse_actions, attach response.model_dump under existing feedback extra.failed_response, re-raise the same exception; valid path unchanged. Keep the dedicated key (not extra.response) so Harbor usage/cost accounting does not change. Existing request preparation strips extra, preserving subsequent model input. Preserve original feedback even if diagnostic export fails, recording failure explicitly. Do not append malformed assistant content to history or change cost-limit accounting.

No safe TTI-only source fix exists for pinned mini2.2.8. Harbor installs mini-swe-agent==version inside the task container, accepts no immutable mini source selector, and rejects unknown kwargs. Native config is written after installation. Host-venv changes would not reach that process. AutoFix therefore made no source edit, wrapper or runtime monkeypatch.

Supported consumption: upstream mini publishes a repaired version; TTI repins agent_kwargs.version and runs paired malformed/valid host regression tests. If unreleased source is required, separately review a generic Harbor immutable-source installer capability, pin that Harbor SHA through existing metadata.harbor_git_ref, and pin a repaired mini SHA. Neither external artifact is available/proven here. No publication authorized or attempted.

Host behavioral acceptance remains pending a real upstream patch: compare complete request sequences/APIkwargs, feedback/actions/errors, counters/limits and saved raw IDs across malformed→valid→submit fixtures, including diagnostic-export failure. Existing source tests only assert FormatError and cannot confirm retention. Device work is unnecessary. Status: diagnosis confirmed, repair blocked by external pinned dependency, problem open.
