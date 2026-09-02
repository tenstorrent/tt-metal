#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The full-model stage's evidence sweep, committed so the exact ordering and
# flags behind doc/full_model/ are reproducible rather than reconstructable
# from a work-log prose list (work log FM-017).
#
#   bash models/autoports/zai_org_glm_4_7_flash/tests/run_evidence_sweep.sh
#
# Run it from the repo root with a clean stage tree and the source already
# committed: the first and last things it does are record HEAD, the tracked
# status, the *untracked* files and a sha256 snapshot of every stage source
# file, into doc/full_model/logs/sweep_provenance.log. Recording untracked
# files needs --others explicitly, because .git/info/exclude hides
# models/autoports/ from plain `git status`.
#
# About 85 minutes on one Blackhole p150, of which test_full_context.py (last,
# on purpose) is 40. Every step prints "<name>=<exit code>"; anything other
# than 0 means the sweep's evidence is incomplete. Those exit codes and the
# per-run watcher fault counts are the sweep's own verdict, so its stdout is
# teed into doc/full_model/logs/sweep_run.log rather than left in a terminal.
set -x
exec > >(tee "$(git rev-parse --show-toplevel)/models/autoports/zai_org_glm_4_7_flash/doc/full_model/logs/sweep_run.log") 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$(git rev-parse --show-toplevel)" || exit 1
D=models/autoports/zai_org_glm_4_7_flash
L=$D/doc/full_model/logs
P=./python_env/bin/python
REAL_HOME=$HOME
SNAPSHOT="${GLM47_FLASH_SNAPSHOT:-$REAL_HOME/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67}"

provenance() {
  {
    echo "=== sweep $1 $(date -Is)"
    echo "HEAD=$(git rev-parse HEAD)"
    echo "--- tracked changes in the stage directory:"
    git status --porcelain -- $D
    # Deliberately without --exclude-standard: .git/info/exclude lists
    # models/autoports/, so a plain `git status` shows no untracked file here
    # at all and the previous provenance step was blind to new ones.
    echo "--- untracked files in the stage directory:"
    git ls-files --others -- $D | grep -v "__pycache__" || true
    echo "--- sha256 of every source file this sweep depends on:"
    sha256sum $D/tt/*.py $D/tests/*.py $D/probe/*.py \
        $D/tests/run_evidence_sweep.sh \
        models/common/readiness_check/*.py \
        .agents/scripts/check_context_contract.py
  } >> $L/sweep_provenance.log 2>&1
}

: > $L/sweep_provenance.log
provenance started

$P $D/probe/dram_capacity_probe.py > $L/dram_capacity_probe.log 2>&1; echo "dram=$?"

# Unsafe-allocation accounting. Needs the env var: without it the tracker is a
# no-op and the artifact says so in `tracking_enabled`.
TT_METAL_TRACE_ALLOC_TRACKING=1 $P $D/probe/trace_alloc_probe.py > $L/trace_alloc_probe.log 2>&1; echo "tracealloc=$?"
TT_METAL_TRACE_ALLOC_TRACKING=1 $P $D/tests/dev_full_model.py capacity --seq 128 > $L/trace_alloc_full_model.log 2>&1; echo "tracealloc_full=$?"

$P $D/tests/measure_cold_compile.py --repeats 4 > $L/compile_cost_warm.log 2>&1; echo "warmcompile=$?"
cp $D/doc/full_model/compile_cost.json $D/doc/full_model/compile_cost_warm.json

# A throwaway HOME is what makes the JIT cache cold; the run log records
# "JIT cache stats: 0/N hits (0.0%)" so a warm run cannot be mistaken for it.
rm -rf /tmp/glm47_coldhome && mkdir -p /tmp/glm47_coldhome
HOME=/tmp/glm47_coldhome GLM47_FLASH_SNAPSHOT="$SNAPSHOT" \
  $P $D/tests/measure_cold_compile.py --repeats 4 > $L/compile_cost_cold.log 2>&1; echo "coldcompile=$?"
rm -rf /tmp/glm47_coldhome

# Standalone as well as combined: the standalone log is what the report's
# per-suite row cites, the combined one proves they share a session cleanly.
$P -m pytest $D/tests/test_full_model.py -q -s -p no:randomly > $L/pytest_full_model_only.log 2>&1; echo "mainonly=$?"
$P -m pytest $D/tests/test_full_model.py $D/tests/test_prefill_padding.py -q -s -p no:randomly > $L/pytest_full_model.log 2>&1; echo "main=$?"
$P -m pytest $D/tests/test_prefill_padding.py -q -s -p no:randomly > $L/pytest_prefill_padding.log 2>&1; echo "padding=$?"
$P -m pytest $D/tests/test_full_model_perf.py -q -s -p no:randomly > $L/pytest_full_model_perf.log 2>&1; echo "perf=$?"
GLM47_FM_BATCH=32 GLM47_FM_BATCH_SEQ=8192 $P -m pytest $D/tests/test_full_model_batch.py -q -s -p no:randomly > $L/pytest_full_model_batch32.log 2>&1; echo "batch=$?"

$P $D/probe/decode_position_scaling_probe.py > $L/decode_position_scaling.log 2>&1; echo "posscale=$?"
$P $D/probe/decode_cache_scaling_probe.py > $L/cache_scaling.log 2>&1; echo "cachescale=$?"
$P $D/probe/full_model_head_probe.py > $L/head_probe.log 2>&1; echo "head=$?"
$P $D/probe/logits_memory_ab_probe.py > $L/logits_memory_ab.log 2>&1; echo "logitsab=$?"
$P $D/probe/first_use_ttft_probe.py > $L/first_use_ttft.log 2>&1; echo "firstuse=$?"

$P -m models.common.readiness_check.run_prefill_check --model-dir $D --reference $D/readiness_aime24_chat.refpt --mesh-device N150 --trace-region-size 350000000 --l1-small-size 32768 > $L/run_prefill_check.log 2>&1; echo "prefillcheck=$?"
$P -m models.common.readiness_check.run_teacher_forcing --model-dir $D --reference $D/readiness_aime24_chat.refpt --mesh-device N150 --trace-region-size 350000000 --l1-small-size 32768 > $L/run_teacher_forcing.log 2>&1; echo "tf=$?"
$P -m models.common.readiness_check.run_autoregressive --model-dir $D --hf-model zai-org/GLM-4.7-Flash --prompt-file $D/doc/full_model/autoregressive_prompt_chat.txt --mesh-device N150 --trace-region-size 350000000 --l1-small-size 32768 --max-new-tokens 256 > $L/run_autoregressive.log 2>&1; echo "autoreg=$?"
$P models/common/readiness_check/check_degenerate_output.py --model-dir $D --missing-artifacts critical --scope autoregressive --json $D/doc/full_model/degenerate_check.json > $L/check_degenerate_output.log 2>&1; echo "degen=$?"

# --skip-hf reuses the committed CPU HF control: it is a property of the
# checkpoint, not of the port, and costs ~25 minutes to regenerate.
$P $D/tests/run_qualitative_suite.py --max-new-tokens 128 --skip-hf > $L/qualitative_suite.log 2>&1; echo "qual=$?"

rm -rf $D/doc/full_model/tracy && mkdir -p $D/doc/full_model/tracy
$P -m tracy -r -p -v -m pytest $D/tests/test_full_model_profile.py -q -s -p no:randomly > /tmp/tracy_run.log 2>&1; echo "tracy=$?"
CSV=$(grep -a "OPs csv generated at" /tmp/tracy_run.log | tail -1 | sed 's/.*at: //')
test -s "$CSV"; echo "tracycsv=$?"
for w in DECODE_MODEL DECODE_TOKENOUT PREFILL; do
  out=$D/doc/full_model/tracy/$(echo $w | tr 'A-Z' 'a-z')_perf_report
  ./python_env/bin/tt-perf-report --arch p150 "$CSV" --start-signpost PERF_FM_$w --end-signpost PERF_FM_${w}_END --csv ${out}.csv > ${out}.txt 2>&1
  echo "perfreport_$w=$?"
done
gzip -c /tmp/tracy_run.log > $L/tracy_profile_run.log.gz; echo "gzip_tracy=$?"
$P $D/tests/summarize_perf_report.py --tracy-dir $D/doc/full_model/tracy > $L/summarize_perf_report.log 2>&1; echo "summary=$?"
gzip -f $D/doc/full_model/tracy/decode_model_perf_report.csv $D/doc/full_model/tracy/decode_tokenout_perf_report.csv; echo "gzip_csv=$?"

# Watcher and the profiler are kept in separate runs on purpose.
rm -rf generated/watcher $L/watcher; mkdir -p $L/watcher
TT_METAL_WATCHER=2 $P $D/tests/dev_full_model.py smoke > $L/watcher_reduced_smoke.log 2>&1; echo "wsmoke=$? faults=$(grep -aicE 'error|assert|sanitize|fatal|exception' generated/watcher/watcher.log)"
gzip -c generated/watcher/watcher.log > $L/watcher/watcher_reduced_smoke.log.gz; echo "gzip_wsmoke=$?"; rm -rf generated/watcher
TT_METAL_WATCHER=2 $P $D/tests/dev_full_model.py trace --layers probe --seq-cap 202752 > $L/watcher_reduced_trace.log 2>&1; echo "wtrace=$? faults=$(grep -aicE 'error|assert|sanitize|fatal|exception' generated/watcher/watcher.log)"
gzip -c generated/watcher/watcher.log > $L/watcher/watcher_reduced_traced_decode.log.gz; echo "gzip_wtrace=$?"; rm -rf generated/watcher
TT_METAL_WATCHER=2 $P $D/tests/dev_full_model.py capacity --seq 128 > $L/watcher_full_model.log 2>&1; echo "wfull=$? faults=$(grep -aicE 'error|assert|sanitize|fatal|exception' generated/watcher/watcher.log)"
gzip -c generated/watcher/watcher.log > $L/watcher/watcher_full_model.log.gz; echo "gzip_wfull=$?"; rm -rf generated/watcher

# Last: 40 minutes on its own.
PYTHONUNBUFFERED=1 $P -m pytest $D/tests/test_full_context.py -q -s -p no:randomly > $L/pytest_full_context.log 2>&1; echo "fullctx=$?"

# Last two, after every artifact they read: the contract check, and the
# report-figure check that fails when a number in README.md or work_log.md has
# no matching value in a committed artifact.
$P .agents/scripts/check_context_contract.py --model-dir $D --hf-model zai-org/GLM-4.7-Flash --stage full-model --require-contract > $L/check_context_contract.log 2>&1; echo "ctx=$?"
$P $D/tests/check_report_numbers.py > $L/check_report_numbers.log 2>&1; echo "reportnums=$?"

provenance finished
echo SWEEP_DONE
