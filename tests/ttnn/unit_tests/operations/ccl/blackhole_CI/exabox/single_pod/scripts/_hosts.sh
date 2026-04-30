#!/bin/bash
# Shared host list for the single-pod scripts. Sourced by reset_chips.sh,
# recover_hung_run.sh, run_chain_test.sh, run_pipeline_test.sh.
#
# Override per-shell via SINGLE_POD_HOSTS="..." (space-separated) if the
# 4-host bindings change.
SINGLE_POD_HOSTS="${SINGLE_POD_HOSTS:-bh-glx-110-c07u02 bh-glx-110-c07u08 bh-glx-110-c08u02 bh-glx-110-c08u08}"
