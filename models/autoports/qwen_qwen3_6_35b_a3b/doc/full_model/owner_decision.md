# Stage 6 owner decision

Date: 2026-11-16. Source: `/home/ttuser/qwen36-35b-a3b/multigoal_logs/06-06-full-model.resume-1.message.txt`. Original goal: `/home/ttuser/qwen36-35b-a3b/multigoal_logs/06-06-full-model.prompt.txt`.

> Owner decision: accept the resumed-prefill PCC failure as a known limitation. The original HF model reproduces it (doc/full_model/qualitative/hf_resume_control/verdict.json, original_model=same) and all TT top-1 checks pass. Record it in limitations with this decision and the evidence, finish the review and complete stage 6. No further work on this gate.

This supersedes the requirement to repair the three recorded resumed-prefill PCC failures. It accepts a known numerical limitation; it does not change measured values, claim numerical repair, relax top-1 or negative-control checks, or enable automatic KV-only prefix caching. The retained-state continuation implementation already exists. Its four blocked serving rows can now be accepted with this explicit scope. No further device work on this gate is requested.
