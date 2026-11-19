#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=models/autoports/ornith_ai_ornith_1_5_9b
bash "$root/tests/run_optimized_evidence.sh"
bash "$root/tests/run_optimized_contracts.sh" "$@"
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_perf_reports
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.summarize_optimized
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.summarize_optimized_final
