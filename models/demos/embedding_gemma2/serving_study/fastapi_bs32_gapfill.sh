#!/bin/bash
# FastAPI --max-batch 32 load points missing from the 2026-10-29 rerun (hung), one fresh server per point.
S="$(dirname "${BASH_SOURCE[0]}")/fastapi_bs32_repro.sh"
for pt in "8 128" "1 512" "8 512" "32 512"; do set -- $pt; echo "### bs32 conc=$1 tok=$2 $(date -u +%H:%M:%SZ)"
  CONCS=$1 TOK=$2 $S 2>&1 | grep -E "^\{|load exit|forwards logged|chip 31|died|busy|Thread|to_torch|^->"; done
echo "gapfill done $(date -u +%H:%M:%SZ)"
