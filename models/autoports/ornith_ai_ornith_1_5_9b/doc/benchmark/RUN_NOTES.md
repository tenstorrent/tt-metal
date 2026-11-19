# Ornith-1.5-9B final benchmark work log

Owner: the person who launched this bringup. This stage reports measurements and protocol facts without an accuracy verdict. No push is authorized or performed.

## Preserved starting state

Stage 10 checkpoint: `3d82102e1b493143bbc29f50ec320938a0048bad`; documentation checkpoint: `a67deb64`. Existing untracked Stage 9/10 artifacts were present at entry and are excluded from stage-owned commits. `setup_start.json` retains the initial context-contract hash. No context-capacity change is planned.

Selected serving path: generated `models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm:OrnithForCausalLM`, all 32 layers, TP4 / P300x2 / mesh1x4 / four Blackhole p300c chips, selected `bfp8_all_lofi`, context262144, block32, scheduling budget4096, device sampling integrated64, decode-only traces, asynchronous scheduling, responsive interleave1prefill:2decode. The Stage 10 model capability explicitly excludes automatic prefix caching because recurrent state cannot be reconstructed from KV blocks; effective prefix caching remains false for evals and performance.

Stage 10 separately measured server slots1 and32. Slots8 and16 were not separately validated there; they are reported as skipped, not as failed hardware configurations. Requested concurrency is recorded separately from server slots.

## Ownership and setup

No device holder or serving process was present at entry. `hardware-list.txt` records a bounded serial `tt-smi -ls --local`, with all four assigned Blackhole devices visible. No reset occurred. Hardware jobs are serialized over all four chips. Every launched server is bounded by timeout21600 and remains owned through `owned_server.json`; server transitions require two fresh idle metrics observations. No unrelated process is stopped.

The earlier Stage 10 server had already stopped. Fresh launch, compile, and pilot setup before the timed invocation are recorded separately from the benchmark runner's client/setup clocks. Client installation commands, wall times, package identities and upstream source commit are in `client_setup/`. Eval and performance clients are separate from the working server environment. The performance client is upstream vLLM0.26.0, built for the empty target; the server remains Stage10 vLLM0.31.0+empty.

## Protocol selection

The pinned exact-model card publishes GPQA Diamond86.4%, averaged over five runs, without a complete matching evaluation recipe: https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/489cb97981b8654bcfcf30ce1f94ed1b62e07b53/README.md . Upstream GPQA dataset access was denied by the Hub gate during setup; the failed preparation log is retained in `pilot_subset/prepare.log`. No score is fabricated for unavailable questions.

The native template is applied exactly once by the server to structured chat messages. Eval explicitly requests native thinking because the inherited Stage10 default is false. Protocol, full generation overrides and references are retained in `protocol.json`; normal EOS is preserved, and textual benchmark stops are cleared to prevent an incidental `Question:` in reasoning from truncating an answer. Upstream few-shot examples, prompts and scorers are retained. Samples are selected by the packaged common seed, never by scores. Pilot results are used for runtime/format planning only; all final frozen documents remain in the denominator.

## Full-phase accounting

The optional model-owned recorder observes existing completion/readback boundaries without added device synchronization or the device profiler. The collector must prove full prefill/decode phases, exact measured request/token membership, executed storage/shapes, source identity and hardware peaks before any final run is accepted. Missing accounting is an incomplete stage, not a zero utilization result.

## Attempts and completion

No timed `benchmark_stage run` invocation yet. Final report target: `run/REPORT.md`. Failed attempts, if any, will be preserved separately. Evidence/context checks and local commit SHAs will be appended after collection.

## Pre-invocation pilot and selection

Setup repairs and failed evidence are documented in `AUTOFIX.md`. The repaired32-slot pilot validates integer-ID performance transport and full-phase accounting. `setup_probe.json` records bounded server selection, check, warmup, measured profile, collection and accuracy pilot commands/wall times; these happened before `benchmark_stage run`. Client installations and initial server launch are separately retained. The final setup elapsed wall time is recorded at invocation, including investigation and idle time.

The18-question native-thinking pilot took152.218 seconds and generated14886 tokens. All18 stopped normally, with no cap hits or blank finals. MMLU and IFEval extraction is100%; GSM8K flexible extraction is100%, strict50% because `$7` fails upstream's strict answer regex. Retain strict and flexible upstream scores separately; use flexible extraction to distinguish this formatting loss. No scoring rule or benchmark prompt is changed.

Task-weighted pilot projection is14016 seconds for the full280/256/256 subset, exceeding the7200-second client budget. Final140/128/64 subsets project3885 seconds; `sample_reduction` records this speed-based choice. Native thinking,32768 output cap, upstream few-shot examples and all selected questions remain. Reduced IDs/document/few-shot hashes are verified against the full common subset before final invocation. This is sample sizing, not selection based on answer correctness.

Local setup commits: `3dedb7085e539b83a8c6026ee4eb22d53d41d713`, `391f81a143ca327bdb9a576d4bc090a843445683`. Further preparation and final evidence commits are recorded in the completion section and external commit logs. No pushes.

## Completed attempt1

Measurement source commit: `af2350b0493fbd9be8b36942a59415a7c5f4fec4` (normal repository hooks passed). Attempt1 produced `run/REPORT.md`, all332 upstream-scored responses, exact4K/128 batch profiles for1 and32 slots, both largest-server single-request rows, and full-phase roofline evidence for both batch profiles. No second timed attempt was required. Runner client707.824s /7200s, setup114.162s /3600s, total821.985s. Accuracy shared pass574.3s. All332 accuracy requests stopped normally; no cap hits or blank finals. Strict GSM8K extraction113/128 is the sole advisory; flexible extraction128/128 and all15 formatting misses are preserved and reported without replacing upstream scores.

Evidence check exit1 means valid with one advisory, not a critical failure. Context check exit0 verifies262144; the context contract hash is unchanged. `run/validation.json` records checks and `run/gsm8k_format_audit.json` records the format audit. No accuracy acceptance verdict or threshold is applied.

Both phase collections completed before stopping their servers. The final owned1-slot server stopped after the required two idle observations; `run/cleanup.json` records successful cleanup, absent ownership state and no remaining device holders. No reset, unrelated process termination or reservation mutation occurred. EngineDeadError lines during intentional shutdown are after all measurements and idle checks.

Large raw files remain intact under this persistent checkout, including every response/question, input-ID list, full configuration metadata and phase timeline. The repository hook rejects files over500KiB, and tt-device-usage says not to compress or archive individual artifacts. Source, report and compact evidence are committed locally; `run/artifact_manifest.json` inventories the retained raw files and explicitly lists those above the commit limit. No raw evidence was deleted or compressed. Local commit SHAs and final reporting time are also written to `/home/ttuser/ornith-9b/artifacts/stage11_completion.json` after the final commit. No push was performed.

The final commit also includes the post-measurement newline repair in `tools/benchmark_client.py`; it changes output serialization after upstream timing, not model execution or metrics. Original/normalized byte hashes and exact JSON equivalence are retained in `run/hook_normalization.json`. Roofline links were repaired and the evidence gate rerun: valid with the same single strict-GSM8K advisory.
