#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# DO NOT MERGE -- hang-injection experiment: what the prefill stage hang watcher runs on a verdict.
echo "HANG WATCHER COMMAND: stage=${PREFILL_HANG_STAGE} pid=${PREFILL_HANG_PID}" >&2
echo "HANG WATCHER COMMAND: detail=${PREFILL_HANG_DETAIL}" >&2
echo "HANG WATCHER COMMAND: counters=${PREFILL_HANG_COUNTERS}" >&2
# End the producer so run_multirank_pcc.sh reaches its cleanup and prints the rank logs.
# The [_] keeps this script's own command line from matching.
pkill -f 'prefill[_]producer' || true
