#!/usr/bin/env bash
# The prefill-trace discriminator matrix, as one runnable script.
#
# These probes are the load-bearing evidence for not shipping traced serving prefill, so the
# commands belong in the repo rather than in a shell history. Each is one adapter-probe
# invocation against the real 52-layer build; none of them starts a server. Results are folded
# into ../prefill_trace_discriminators.json (the folding snippet is in the work log).
#
# The question each one answers:
#
#   controls    tracing off at every length the traced runs use -- and the reference for the
#               two lengths (1024, 8192) that the vLLM-integration stage's committed probe does
#               not contain
#   20bucket    is the "correct at 20 buckets" result real, or an artifact of an older revision?
#   4097only    is it the capture or a replay?  (that run contains no traced request at all)
#   bucket96    is it specific to one bucket value?
#   8192_*      is it an unwarmed shape?  (8192 IS in PREFILL_WARMUP_LENGTHS)
#   bucket1024  does a LARGE single bucket behave differently from a small one?
#   128_1024    the configuration worth wanting: two traces, largest 1024, keeping the bucket
#               the 1.29x was measured on.  Correct at 128/1024/4097 -- and wrong at 8192,
#               which is what settles it
#
# Reading them: a prompt whose padded length is in the bucket set is served by a trace replay;
# every other length takes the eager path.  All the failures are on the eager path.
set -u
REPO=/home/ttuser/dev/muse-glimmer/tt-metal
D=$REPO/models/autoports/meta_models_muse_glimmer_30b/doc/optimized_vllm
P=$REPO/models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/bench/adapter_probe.py
cd "$REPO"
B20=$(python3 -c "print(','.join(str(32*i) for i in range(1,17)))"),640,768,896,1024

run () { name=$1; shift; echo "=== $name $(date -u +%H:%M:%S) ==="; env "$@" > "$D/logs/probe_disc_$name.log" 2>&1; echo "  rc=$?"; }

# --- controls: tracing off ---------------------------------------------------------------
run repro_eager         MUSE_GLIMMER_VLLM_PREFILL_TRACE=0 \
    python "$P" --prompt-lens 128,100,37,4097 --decode-steps 8 --out "$D/probe_repro_eager.json"
run 4097only_eager      MUSE_GLIMMER_VLLM_PREFILL_TRACE=0 \
    python "$P" --prompt-lens 4097 --decode-steps 8 --out "$D/probe_disc_4097only_eager.json"
run 8192_eager          MUSE_GLIMMER_VLLM_PREFILL_TRACE=0 \
    python "$P" --prompt-lens 8192 --decode-steps 8 --out "$D/probe_disc_8192_eager.json"
run 1024_eager          MUSE_GLIMMER_VLLM_PREFILL_TRACE=0 \
    python "$P" --prompt-lens 1024,4097 --decode-steps 8 --out "$D/probe_disc_1024_eager.json"

# --- one small bucket --------------------------------------------------------------------
run repro_traced        MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128 \
    python "$P" --prompt-lens 128,100,37,4097 --decode-steps 8 --out "$D/probe_repro_traced.json"
run 4097only_traced     MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128 \
    python "$P" --prompt-lens 4097 --decode-steps 8 --out "$D/probe_disc_4097only_traced.json"
run 8192_traced         MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128 \
    python "$P" --prompt-lens 8192 --decode-steps 8 --out "$D/probe_disc_8192_traced.json"
run 1024_bucket128      MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128 \
    python "$P" --prompt-lens 1024,4097 --decode-steps 8 --out "$D/probe_disc_1024_bucket128.json"
run bucket96            MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=96 \
    python "$P" --prompt-lens 128,100,37,4097 --decode-steps 8 --out "$D/probe_disc_bucket96.json"

# --- one large bucket, and the two-bucket set --------------------------------------------
run bucket1024          MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=1024 \
    python "$P" --prompt-lens 1024,4097 --decode-steps 8 --out "$D/probe_disc_bucket1024.json"
run bucket128_1024      MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128,1024 \
    python "$P" --prompt-lens 128,1024,4097 --decode-steps 8 --out "$D/probe_disc_bucket128_1024.json"
run 8192_bucket128_1024 MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS=128,1024 \
    python "$P" --prompt-lens 8192 --decode-steps 8 --out "$D/probe_disc_8192_bucket128_1024.json"

# --- the wide set --------------------------------------------------------------------------
run 20bucket            MUSE_GLIMMER_VLLM_PREFILL_TRACE=1 MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS="$B20" \
    python "$P" --prompt-lens 128,100,37,4097 --decode-steps 8 --out "$D/probe_disc_20bucket.json"

# Two further rows in the matrix come from probes run at 16 decode steps before this script
# existed and are not re-run here: probe_full_shipped.json (bucket [128]) and
# probe_full_prefill_traced.json (20 buckets). Their 16-step counters are quoted in the README's
# contract-evidence section; their token rows duplicate the 8-step ones above.
#
# NOT MEASURED, and stated as the matrix's coverage limit rather than left implicit: bucket
# [1024] ALONE at 8192, and any bucket size between 128 and 1024. Both observed 8192 failures
# contain bucket 128, so "a large largest bucket is what helps" and "any small resident bucket
# poisons long eager prefills" are not separated here. The ship decision does not rest on that
# cell: [128,1024] failing 8192 settles it either way.
echo "DISCRIMINATORS_DONE $(date -u +%H:%M:%S)"
