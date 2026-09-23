# Performance and accuracy scripts, 2026-09-22 session

Scripts behind the numbers in `BRINGUP_STATUS.md`'s 2026-09-22 update: traced LLM decode
profile, the cached/traced flow encoder (found not to work -- see below), the traced CFM
Euler-step solver, the CFM trace-lifecycle experiment, the WER scoring fix, and the
steps=10-vs-5 RTF comparison. Written as one-off measurements against the real checkpoints
on an N150, copied here so they survive the session. They are not tests and are not run in
CI -- the actual correctness regression tests for the traced CFM/encoder live in
`tests/pcc/test_flow_decoder.py` and `tests/pcc/test_upsample_conformer_encoder.py`.

Setup: run with `/opt/venv/bin/python` from a directory other than the repo root, with
`PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal`.
Set `COSYVOICE2_SCRATCH` (checkpoint-derived cache dir, default `/tmp/cosyvoice2_stage1_eval`)
and `OUT_DIR` (results and wavs, default `/tmp/cosyvoice2_perf_2026_09_22`). Wrap device jobs
in `timeout -s KILL N` so a hung card cannot stall the session (recover with `tt-smi -r`).

| Script | What it measures |
|---|---|
| `llm_decode_profile.py` | Traced LLM decode's 9.8 ms/token broken into device / readback / host-write / sampling phases |
| `traced_encoder_and_cfm.py` | First pass at items 2-4: encoder trace (found broken -- see its own findings below), CFM capture/warm-up/replay/release costs, full 4-sentence warm regression |
| `trace_lifecycle_experiment.py` | Synthetic (random-weight, fast) isolation of the CFM trace's lifecycle: single-slot cache confirmed, cold-vs-warm-kernel capture cost isolated from tracing's own overhead |
| `wer_repro_and_stage_breakdown.py` | utt1 WER reproducibility (LLM+flow determinism, HiFT's unseeded noise draw) + fresh per-stage (LLM/encoder/CFM/HiFT) time breakdown with CFM traced |
| `rescore_wer_fix.py` | First WER re-score attempt using jiwer's built-in `ExpandCommonEnglishContractions` -- kept for the record; superseded by `rescore_wer_final.py` once that transform's possessive-`'s` false positive was found (see below) |
| `rescore_original_stage1.py` | Faithful reproduction of the original Stage 1 eval (`scripts/vocoder_debug_2026_09_20/stage1_eval_v4_noisefix.py`'s exact config) that produced the doc's reported 4.17% WER, for re-scoring |
| `rescore_wer_final.py` | The corrected re-score: a precise contraction-only transform (jiwer.SubstituteRegexes, the same rule list as `ExpandCommonEnglishContractions` minus the ambiguous bare-`'s` rule) applied to both this round's four sentences and the original Stage 1 eval |
| `cfm_only_steps_compare.py` | Real, isolated (no LLM/HiFT) CFM-only measurement at steps=10 vs steps=5, at the four real T values this round's sentences produce |

## Findings this round (see `BRINGUP_STATUS.md` for the full writeup)

- **Traced CFM solver**: works. Ported from the CosyVoice1 reference's `tt/flow/cfm.py`
  recipe (one Euler step captured, replayed N times, `t`/`dt` as device tensors). Found and
  fixed a real trace-reuse corruption bug along the way (pre-building per-step device
  tensors as a list before the reuse/capture dispatch corrupted a second solve on a cached
  trace) -- see `tt/flow/decoder.py`'s `TtCausalConditionalCFM` class docstring and
  `_temb_device`'s docstring for the full measured detail.
- **Traced flow encoder**: does NOT work. `TtRelPositionMultiHeadedAttention._rel_shift`
  does a deliberate host round-trip once per Conformer layer, so capture always fails
  (`TT_FATAL: Reads are not supported during trace capture`). The fallback is correct
  (falls back to eager, now remembers the failure so it doesn't keep re-paying a doomed
  attempt) but the module is not currently traceable. See `tt/flow/encoder.py`'s
  `TtUpsampleConformerEncoder` class docstring.
- **WER scoring bug**: jiwer's standard contraction-expansion transform also expands
  possessive `'s`. Fixed with a precise, contraction-only substitution. This round's utt1
  (9.52% -> 0.00%, a genuine contraction bug) and the original Stage 1 eval (4.17% -> 4.17%,
  confirmed unaffected -- "Leighton's"/"Layton's" is a real ASR miss) were both re-scored.
- **Step count**: held at 10 (decided; see `BRINGUP_STATUS.md`). Measured real RTF
  reduction from steps=10 to steps=5 (projected via real per-stage data, not the old
  pre-trace estimate) was only 8-11% -- not worth the unvalidated quality risk given
  current priorities.
