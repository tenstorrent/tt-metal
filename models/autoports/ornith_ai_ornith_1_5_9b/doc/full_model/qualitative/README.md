# Full-model qualitative evidence

The reusable implementation is `tests/full_model_qualitative_suite.py`, relative
to the model directory. `run.py` provides the stage-local entry point requested
by the full-model workflow. Run it from the checkout root with the recorded
arguments. It renders the original HF tokenizer chat template, records settings
and source identity, and exports anonymous samples for an independent reader.

- `../qualitative_shared_seed0/`: all six shared prompts, HF and TT, 256-token
  cap, model-card sampling, seed 0. Blinded screen and subsequent unblinding
  are preserved. This screen does not establish a comparative failure rate.
- `../qualitative_entropy_control/`: ten matched seeds for prompt04, using the
  same sampling and cap. A fresh reader receives only `blind_samples.json`
  and `rubric.json`; backend attribution happens after scores are saved.
- `../qualitative_entropy_hf_commands.json`: initial timed-out CPU workers;
  `../qualitative_entropy_hf_retry_commands.json` and per-worker command files
  record the successful exact-cache retry.
- `../final_performance_commands.json`: serial TT quality/performance commands
  and completion status.

The six-prompt screen identified one potential entropy-law wording problem in
unfinished thinking. It was unblinded as HF output. The ten-seed control is
required before any port diagnosis or model-limitation conclusion. Its completed blind comparison found HF 4/10 and TT 1/10 scientific
overstatements; all 20 samples were cap-limited incomplete. The fixed scores and
unblinding are in `../qualitative_entropy_control/unblinded_results.json`;
`../qualitative_verdict.json` records the passing gate and shared limitations.
