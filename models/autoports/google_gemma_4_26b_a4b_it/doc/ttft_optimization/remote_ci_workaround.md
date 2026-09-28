# Remote CI source-publication workaround

## Why this exists

The locally reviewed serving integration is commit
`7f72b1c6e905f5137fe3377f2e7b42738d3f271d` in `tenstorrent/vllm`, but the
personal GitHub account has read-only access to that repository and the commit
is not remotely fetchable. The user requested that remote CI use the existing
`tt-agentic-bringup-qb2` dispatcher and only TT-Metal plus
tt-inference-server branches, without creating a personal vLLM fork or relying
on a new QB2 branch.

## Workaround

TT-Metal carries an exact runtime-only snapshot of
`plugins/vllm-tt-plugin` from the unavailable commit at:

`models/autoports/google_gemma_4_26b_a4b_it/vllm_plugin_snapshot/`

The tt-inference-server source-build path detects that snapshot only for the
exact pinned TT-Metal publication commit and installs it on top of the verified
upstream `vllm==0.26.0` empty-target engine recipe. The normal standalone
plugin path remains unchanged for every other TT-Metal commit.

This changes source transport, not the reviewed implementation. Snapshot files
are compared byte-for-byte with the original vLLM commit before publication.
The image records both the TT-Metal commit and the original vLLM source commit
in OCI labels and retained provenance files.

## CI execution policy

1. Prefer an already published image only when its labels match the exact
   TT-Metal and tt-inference-server commits plus the original plugin source
   commit.
2. If none exists, dispatch `benchmarks` with a blank image once so CI builds
   the exact two source branches.
3. Reuse the resulting image for separate `evals` and `agentic` dispatches.
4. Monitor each run to a terminal result before dispatching the next one.
5. Preserve run URLs, inputs, image labels, reports, and any repair commits in
   `readiness_vllm/ttft_optimization_remote/`.

This workaround is temporary and should be removed after the original vLLM
commit becomes remotely accessible.
