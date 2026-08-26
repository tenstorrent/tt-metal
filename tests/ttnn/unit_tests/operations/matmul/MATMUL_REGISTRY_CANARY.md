<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Matmul registry silicon canary

This canary validates the exact checked-in registry table against a fresh
one-chip Blackhole process. It runs the same bounded entry set in three fresh
processes because registry mode is frozen at the first public matmul dispatch.

It covers:

- every public-operation/scalar semantic represented by the checked lock:
  `ttnn.matmul`, no-bias/no-activation `ttnn.linear`, and/or `ttnn.addmm` with
  `alpha=1` and whichever positive/negative-zero `beta` keys are present;
- Off baseline correctness, Shadow certified lookup without parameter
  selection, and On selection plus successful completion;
- exact-key misses and unsupported public-call variants falling back without
  opening a circuit breaker;
- output PCC and stable named telemetry deltas; and
- one registry resolution at most for a public validation error. The native
  CPU contract additionally injects an error after selection and proves that
  it propagates, records no completion, and circuit-breaks without retry.

An empty table, a lock with no valid dense entry, or a represented semantic
with no bounded topology-compatible entry skips during ordinary device testing.
The release launcher sets the fail-closed requirement, so the same conditions
fail the canary instead of manufacturing absent domains or synthetic hits.

Run directly on an allocated Blackhole node:

```bash
tests/scripts/single_card/run_bh_matmul_registry_canary.sh
```

The launcher applies an independent 13-minute timeout and writes JUnit reports
plus a digest receipt under node-local `${TMPDIR:-/tmp}` by default. To submit a
bounded Slurm canary, use the site's approved one-node partition and exclusion
policy; for example:

```bash
sbatch --time=00:15:00 --nodes=1 \
  --output="$HOME/.cache/ttnn/matmul-registry-canary-%j.log" \
  --wrap='cd /path/to/tt-metal && tests/scripts/single_card/run_bh_matmul_registry_canary.sh'
```

Do not add a pending array or an automatic retry. A failure leaves the node at
process exit and the receipt directory contains the completed mode reports.
