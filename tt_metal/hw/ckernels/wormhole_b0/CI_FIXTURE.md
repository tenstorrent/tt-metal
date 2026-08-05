CI fixture — green-path probe for `tt-umd-simulators` `polling_agent_mr`.

This marker file lives under `tt_metal/hw/ckernels/wormhole_b0/` on purpose:
that path matches the `wh` simulator rule in tt-umd-simulators'
`scripts/simulator_paths.yml`, so the polling agent routes PRs touching it
to `build_metal` + `metal_unit_test_wh` only (skipping the heavier bh/qsr
suites). The downstream tt-umd-simulators pipeline runs in ~10 min.

This PR is a long-lived fixture, parallel to `tenstorrent/tt-umd` #2672 on
the tt-umd side. The polling_agent_mr workflow in tt-umd-simulators
discovers it via `get_prs_to_branch()` on every MR pipeline event.

Not user-facing — safe to ignore unless you're maintaining the
tt-umd-simulators CI infrastructure.
