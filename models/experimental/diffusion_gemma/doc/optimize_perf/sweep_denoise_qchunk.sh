#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Bit-exactness A/B for the denoise SDPA q-chunk size (#51080 roadmap item 1).
#
# ``_denoise_sdpa_program_config`` (tt/diffusion_attention.py:71-92) reads
# GEMMA4_PREFILL_SDPA_QCHUNK and feeds it through _largest_tile_divisor(q_seq_len=256, ...),
# so for the 256-row canvas:
#
#     QCHUNK=32  -> q_chunk_size=32,  q_num_chunks=8, 32 work units over an 8-core grid
#     QCHUNK=64  -> q_chunk_size=64,  q_num_chunks=4, 16 work units
#     QCHUNK=128 -> q_chunk_size=128, q_num_chunks=2,  8 work units
#     QCHUNK=256 -> q_chunk_size=256, q_num_chunks=1,  4 work units  <-- half the grid idle
#
# q-chunking changes how the Q axis is GROUPED across cores; it does not change the flash
# K-reduction order (k_chunk_size is untouched). So the committed tokens are EXPECTED to be
# bit-identical -- this script exists because that must be verified, not assumed.
#
# Uses the eager serving smoke: the q-chunk program config is shared by the eager and the
# up-front traced denoise paths, so eager is a valid and much cheaper vehicle for the
# bit-exactness question. Eager wall-clock is NOT representative of traced serving, so this
# script deliberately does not claim a speedup -- that belongs on the traced path.
#
# Usage:
#   models/experimental/diffusion_gemma/doc/optimize_perf/sweep_denoise_qchunk.sh [QCHUNK...]
# Defaults to "32 64 128" (32 is the current shipped value = the golden).

set -uo pipefail

TT_METAL_ROOT="${TT_METAL_ROOT:-/home/zni/tt-metal}"
MODEL_VENV="${MODEL_VENV:-/home/zni/venvs/tt-diffusion-gemma}"
DG_CKPT="${DG_CKPT:-/home/zni/dg_models/diffusiongemma-26B-A4B-it}"
MESH="${MESH:-P150x4}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
STEPS="${STEPS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
OUT_DIR="${OUT_DIR:-/tmp/dg_qchunk_sweep}"

REPS="${REPS:-2}"

CHUNKS=("${@:-}")
if [ -z "${CHUNKS[0]:-}" ]; then CHUNKS=(32 64 128); fi

mkdir -p "$OUT_DIR"
PY="${MODEL_VENV}/bin/python"

echo "DG_QCHUNK_SWEEP_BEGIN chunks=${CHUNKS[*]} blocks=${NUM_BLOCKS} steps=${STEPS} reps=${REPS} out=${OUT_DIR}"

# Repetitions are INTERLEAVED (32,64,128,32,64,128,...) rather than grouped, so any device
# drift (thermal, fragmentation, cache warming) is spread across configs instead of being
# absorbed entirely by whichever config ran first.
for rep in $(seq 1 "${REPS}"); do
    for qc in "${CHUNKS[@]}"; do
        tag="qchunk_${qc}_rep${rep}"
        log="${OUT_DIR}/${tag}.log"
        echo "--- rep=${rep} QCHUNK=${qc} -> ${log}"
        # gumbel-mode argmax keeps the run deterministic so any sha256 delta is attributable
        # to the program config alone. --disable-eos-stop forces the full block count so the
        # steady-state (block 0 discarded) latency is measurable at all.
        env \
            TT_METAL_HOME="${TT_METAL_ROOT}" \
            PYTHONPATH="${TT_METAL_ROOT}" \
            MESH_DEVICE="${MESH}" \
            GEMMA4_PREFILL_SDPA_QCHUNK="${qc}" \
            "${PY}" -u -m models.experimental.diffusion_gemma.demo.serving_smoke \
            --checkpoint "${DG_CKPT}" \
            --mesh "${MESH}" \
            --max-seq-len "${MAX_SEQ_LEN}" \
            --num-blocks "${NUM_BLOCKS}" \
            --max-denoising-steps "${STEPS}" \
            --gumbel-mode argmax \
            --disable-eos-stop \
            --local-files-only \
            --metrics-json "${OUT_DIR}/${tag}.json" \
            >"${log}" 2>&1
        rc=$?
        # committed_sha256 is reported in the metrics JSON, not on the success line.
        sha=$("${PY}" -c "import json,sys;print(json.load(open(sys.argv[1]))['committed_sha256'])" \
            "${OUT_DIR}/${tag}.json" 2>/dev/null)
        echo "rep=${rep} QCHUNK=${qc} rc=${rc} committed_sha256=${sha:-MISSING}"
    done
done

echo
echo "=== verdict ==="
REPS="${REPS}" "${PY}" - "$OUT_DIR" "${CHUNKS[@]}" <<'PYEOF'
import json, os, statistics, sys, pathlib
out = pathlib.Path(sys.argv[1]); chunks = sys.argv[2:]
reps = int(os.environ.get("REPS", "1"))

runs = {}   # qc -> list of blobs
missing = []
for qc in chunks:
    for rep in range(1, reps + 1):
        tag = f"qchunk_{qc}_rep{rep}"
        log, meta = out / f"{tag}.log", out / f"{tag}.json"
        text = log.read_text(errors="replace") if log.exists() else ""
        if "DG_VLLM_SERVING_SMOKE_SUCCESS" not in text or not meta.exists():
            missing.append(tag); continue
        try:
            runs.setdefault(qc, []).append(json.loads(meta.read_text()))
        except Exception:
            missing.append(tag)
for tag in missing:
    print(f"  !! run did not complete: {tag}")

# --- correctness: committed tokens must be bit-identical across every config and rep ---
shas = {qc: {b["committed_sha256"] for b in blobs} for qc, blobs in runs.items()}
emitted = {qc: {b["blocks_emitted"] for b in blobs} for qc, blobs in runs.items()}
all_shas = set().union(*shas.values()) if shas else set()
all_emitted = set().union(*emitted.values()) if emitted else set()
if len(all_emitted) > 1:
    print(f"WARNING: configs emitted different block counts {emitted} -- not a comparable A/B")
print(f"committed_sha256 distinct values across all runs: {len(all_shas)}")
for qc in chunks:
    if qc in shas:
        print(f"  QCHUNK={qc:>4}: {sorted(s[:16] for s in shas[qc])}")
bitexact = len(all_shas) == 1

# --- performance: STEADY-STATE only. Block 0 carries program compilation, so it is
# discarded; my own prior method note is steady = mean(blocks[1:]). ---
print()
print(f"{'QCHUNK':>7} {'n':>3} {'steady_mean_s':>14} {'per-rep steady means':<34} {'block0_s':>10}")
steady = {}
for qc in chunks:
    if qc not in runs: continue
    per_rep, block0 = [], []
    for b in runs[qc]:
        lat = b.get("per_block_latency_s") or []
        block0.append(lat[0] if lat else float("nan"))
        tail = lat[1:]
        if tail: per_rep.append(statistics.mean(tail))
    if not per_rep:
        print(f"{qc:>7} {0:>3}  NO STEADY-STATE SAMPLE (need >=2 blocks; use --disable-eos-stop)")
        continue
    steady[qc] = statistics.mean(per_rep)
    print(f"{qc:>7} {len(per_rep):>3} {steady[qc]:>14.3f} "
          f"{str([round(x,3) for x in per_rep]):<34} {statistics.mean(block0):>10.3f}")

if steady:
    base_qc = chunks[0]
    base = steady.get(base_qc)
    print()
    if base:
        for qc in chunks:
            if qc in steady:
                d = (steady[qc] / base - 1.0) * 100.0
                print(f"  QCHUNK={qc:>4} vs {base_qc}: {d:+.1f}%")
    best = min(steady, key=steady.get)
    print(f"  fastest: QCHUNK={best}")
print("DG_QCHUNK_SWEEP_RESULT " + ("ALL_BIT_IDENTICAL" if bitexact else "DIVERGED"))
PYEOF
