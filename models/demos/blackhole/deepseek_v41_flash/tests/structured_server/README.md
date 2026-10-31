Device-server acceptance for structured outputs on the speculative verify round (M1 / M2, see ../../MODEL_MASK_NOTES.md and OPTIMIZATIONS.md 8.22).
Clients (stdlib + jsonschema, run against a live server on 127.0.0.1:8000):
- matrix.py: 28-cell matrix (case x stream x greedy/sampled); --workers/--filler for a mixed batch
- prestream.py: pre-stream validation (14 cells); stream_check.py: streaming deltas / finish reasons / usage vs non-stream, --par for cross-talk
- mode_ab.py: plain-user tok/s/user next to S looping structured workers (or unstructured control), modes none/m0/m1/m2/ctl_tok/ctl_dur (live VLLM_TT_STRUCTURED_MODE_FILE)
- gsm_check.py: GSM8K spot check without structured requests; canary.py: long-context needle canary with structured requests
Scripts (paths are the dev-host ones): hold.sh boots the server, run_load_*.sh are the sequences run by it. They are records of what was run, not CI.
