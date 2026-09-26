#!/usr/bin/env bash
# Parent-owned, serialized device execution; this script has not been run by its author.
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
gemma_doc=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder

python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_headnorm \
    --headnorm-mode native --defaults --layer 0 --length 4096 --real \
    --input-fixture "$gemma_doc/actual_text_layer0_4096_128.pt" \
    --decode --steps 128 --timing --verify-program-cache \
    --output "$gemma_doc/native_headnorm_headline_layer0.json" \
    2>&1 | tee "$gemma_doc/native_headnorm_headline_layer0.log"

python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_headnorm \
    --headnorm-mode native --defaults --layer 0 --length 1025 --real \
    --input-fixture "$gemma_doc/actual_text_layer0_1025_512.pt" \
    --decode --steps 512 --timing --verify-program-cache \
    --output "$gemma_doc/native_headnorm_stress_layer0.json" \
    2>&1 | tee "$gemma_doc/native_headnorm_stress_layer0.log"
