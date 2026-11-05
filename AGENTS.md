# Gemma 4 31B fresh bringup

This checkout is dedicated to the twelve-stage model-bringup run. Read
`bringup/run_contract.json` and `bringup/agentic_evals/inputs.json` at startup.

Implement text-only google/gemma-4-31B-it at HF revision
842da3794eaa0b77d5f08bae87a17459d91ff475 in
models/autoports/google_gemma_4_31b_it. Pin reference and tokenizer loads to that
revision. Preserve the advertised context, with only evidence-backed physical
hardware limitations allowed by the stage contract.

The operator requested a new implementation. Existing Gemma 4 implementations
were removed from models/demos/gemma4, gemma4_31b_qb2, and gemma4_d_p.
Do not restore or copy their source from Git history, other local experiments,
remote sources, or installed packages. Use the Hugging Face reference and generic
TTNN operations/utilities. Do not copy Gemma-specific implementation branches
from the generic tt_transformers framework into the autoport. Validate that serving
registers and imports this run's autoport adapter, rather than a stock adapter.

Use this machine for local TT work; it is the primary device. Serialize all local
device commands and check for device owners before reset. Up to six QB2 CI machines
are granted as a secondary device (`qb2-ci` in `bringup/agentic_evals/inputs.json`):
dispatch, track and collect those runs yourself, following its instructions file, and
balance work between the local device and CI as the agentic-serving skill says. Use
default batches and evals from the selected skills, with the goal-12 inputs in
`bringup/agentic_evals/inputs.json`. Read current release eval names, versions,
timeouts and output limits from the pinned release spec and its harness sources; the
model spec alone may not contain the eval limits. Do not reuse the old
implementation's layout/precision/serving settings as evidence.

The operator authorizes pushing the dedicated branches `mvasiljevic/gemma4-31b-goal12-v3`
in tt-metal and tt-inference-server for CI. Publish from separate checkouts with
remotes; keep this execution checkout without a remote. Use source-only commits,
dispatch immutable SHAs, and keep weights, credentials, private logs, profiler dumps
and collected results out of publication commits. Never update main or stable. Do not
create issue comments or send messages.

Pipeline worker: gpt-6-astra, medium effort, personal ChatGPT login in
/home/mvasiljevic/.codex-gemma4-v3. Do not substitute enterprise credentials or
ambient OpenAI/Codex API keys. Use enabled workflow skills from the requested skills
branch. Stage templates authorize subagents where the selected skill requests them.
