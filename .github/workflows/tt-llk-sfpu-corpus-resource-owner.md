# SFPU corpus resource-owner workflow

`tt-llk-sfpu-corpus-resource-owner.yaml` connects the canonical host corpus
gate to the lab-owned P150 silicon sweep. It does not replace or weaken the
checks in `tt_metal/tt-llk/tests/corpus/sweep.sh`.

## Deployment contract

The scheduled triggers become active only after the workflow is present on the
repository's default branch. Before enabling them, the resource owner must:

1. Provide an online runner with the existing
   `tt-ubuntu-2204-P150b-stable` label and `/dev/tenstorrent` access.
2. Create the `sfpu-corpus-silicon` GitHub environment. Environment protection
   rules are the approval boundary for access to the lab runner.
3. Define these repository or environment variables:

   | Variable | Required value |
   | --- | --- |
   | `SFPU_CORPUS_SETUP_STATE` | Absolute path to the `SETUP-STATE.env` produced by the CRAQ four-repository setup. |
   | `SFPU_CORPUS_SIM_BH` | Absolute path to the pinned Blackhole `libttsim.so`. |
   | `SFPU_CORPUS_SIM_WH` | Absolute path to the pinned Wormhole `libttsim.so`. |

4. Keep the four checkouts named by `SETUP-STATE.env` pristine and at their
   recorded revisions. Pristine means no tracked changes and no non-ignored
   untracked files; ignored build and virtual-environment outputs remain
   allowed. The recorded and actual tt-metal revision must equal the workflow's
   `github.sha`; the workflow refuses rather than updating a shared checkout.
   `CRAQ_COMPILER` in that state file must be executable and
   resolve to
   `CRAQ_METAL_DIR/tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-g++`.
   The workflow passes this exact compiler to the sweep; it never falls back
   to the legacy quietbox defaults in `sweep_2x2.conf`.
5. Configure `LLK_SLACK_REPORTING_BOT_TOKEN` and
   `LLK_CI_STATUS_CHANNEL_ID` for scheduled-run notifications.

The runner, environment, variables, and secrets are infrastructure owned
outside this branch. Their absence cannot be provisioned by the workflow.

## Trigger and status semantics

- Monday through Saturday at 02:00 UTC, the workflow runs the canonical host
  gate and a serialized `nightly` P150 sweep.
- Sunday at 04:00 UTC, it runs the host gate and serialized `weekly` sweep.
- Manual dispatch supports `host-only`, `headline`, `nightly`, and `weekly`.
  `host-only` is the only mode where silicon is intentionally not required;
  the summary records it as `NOT_REQUESTED` rather than passed.
- The host gate calls `tt-llk-sfpu-corpus.yaml`, so inventory validation,
  hardware-free sweep/campaign self-tests, plan publication, and every mapped
  BH/WH compile must pass before lab dispatch.
- P150 jobs share the non-cancelling
  `tt-llk-sfpu-corpus-p150-silicon` concurrency group. This serializes workflow
  runs; the corpus wrapper retains its own process and device locks.
- Missing variables, simulators, setup state, device access, exact revision,
  or a clean checkout fail the silicon job. A missing runner leaves the run
  queued, not green. Requested silicon that is skipped, cancelled, or failed
  makes the required status fail.
- Scheduled runs notify the LLK status channel after the required status job.
  Manual runs rely on the Actions check and step summary to avoid ad-hoc Slack
  noise.

The invoked wrapper follows its current owner-ratified straight-silicon path:
`classify,silicon,report`, with per-cell device-golden correctness gating each
performance cell and `--skip-craq-gate` recorded as taint. The pinned simulator
files are still identity-checked and recorded, but this workflow does not run a
full CRAQ campaign, the formal campaign, exhaustive streamers, or galaxy
replication. Those remain separate validation campaigns and must not be
inferred from a green resource-owner run.

## Evidence and retention

The reusable host workflow uploads plans for BH, WH, and QSR plus per-arch
compile run directories for 14 days. The resource-owner workflow uploads the
complete silicon evidence root, dispatch metadata, preflight refusal, and
sweep log for 30 days. Upload steps run under `always()` so a refusal retains
the evidence available at the point of failure.

The workflow never edits checked-in baselines. A baseline change still
requires a separately reviewed update backed by a green evidence artifact.

## Activation check

After the environment is configured, first dispatch `host-only`. Then sync the
lab checkout and dispatch `headline`. Confirm the host and silicon artifacts,
the required status summary, and the Slack notification path before relying on
the scheduled runs.
