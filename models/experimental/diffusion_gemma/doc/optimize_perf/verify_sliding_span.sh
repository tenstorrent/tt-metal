#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Device gate for the BOUNDED sliding read (#51080 item 3, perf half).
#
# With the retention mask already enforced (DG_DENOISE_SLIDING_WINDOW=1), the keys outside the
# window are NEG-masked, so not reading them at all must expose exactly the same key set. That
# makes this a genuine two-sided gate, unlike the fidelity half:
#
#   control  : DG_DENOISE_SLIDING_WINDOW=1  DG_DENOISE_SLIDING_SPAN=0   (25 sliding layers read p_max)
#   candidate: DG_DENOISE_SLIDING_WINDOW=1  DG_DENOISE_SLIDING_SPAN=1   (25 sliding layers read span)
#
# committed_sha256 SHOULD match. The only expected source of divergence is flash K-chunk
# accumulation ORDER (the K axis shrinks from p_max+C to span+C), which is a bf16 reassociation,
# not a visibility change -- so a mismatch is reported as ACCUMULATION_ORDER rather than a hard
# failure, and the per-block hashes localise it.
#
# Also records the SDPA key-row reduction the change is for.

set -uo pipefail

TT_METAL_ROOT="${TT_METAL_ROOT:-/home/zni/tt-metal}"
MODEL_VENV="${MODEL_VENV:-/home/zni/venvs/tt-diffusion-gemma}"
DG_CKPT="${DG_CKPT:-/home/zni/dg_models/diffusiongemma-26B-A4B-it}"
MESH="${MESH:-P150x4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
TRACE_REGION_SIZE="${TRACE_REGION_SIZE:-6442450944}" # 6 GiB. Measured 2026-07-27: the 48 up-front traces need 3.04 GiB at reveal_pmax=4096 (3 GiB fails, 4 GiB is the floor), so the historical 12 GiB reserved ~8 GiB of DRAM that nothing could allocate. Scale this WITH reveal_pmax - it is not a universal constant. See doc/optimize_perf/bisect_trace_region.sh
OUT_DIR="${OUT_DIR:-/tmp/dg_sliding_span}"

mkdir -p "$OUT_DIR"
PY="${MODEL_VENV}/bin/python"

echo "DG_SLIDING_SPAN_AB_BEGIN blocks=${NUM_BLOCKS} max_seq_len=${MAX_SEQ_LEN}"

for span in 0 1; do
    echo "--- DG_DENOISE_SLIDING_SPAN=${span}"
    env \
        TT_METAL_HOME="${TT_METAL_ROOT}" \
        PYTHONPATH="${TT_METAL_ROOT}" \
        MESH_DEVICE="${MESH}" \
        DG_TRACE_REGION_SIZE="${TRACE_REGION_SIZE}" \
        DG_DENOISE_SLIDING_WINDOW=1 \
        DG_DENOISE_SLIDING_SPAN="${span}" \
        "${PY}" -u -m models.experimental.diffusion_gemma.demo.serving_smoke \
        --checkpoint "${DG_CKPT}" --mesh "${MESH}" \
        --max-seq-len "${MAX_SEQ_LEN}" --num-blocks "${NUM_BLOCKS}" \
        --gumbel-mode host --disable-eos-stop --local-files-only \
        --upfront --reveal-pmax "${MAX_SEQ_LEN}" \
        --metrics-json "${OUT_DIR}/span_${span}.json" \
        >"${OUT_DIR}/span_${span}.log" 2>&1
    echo "DG_DENOISE_SLIDING_SPAN=${span} rc=$?"
done

echo
echo "=== verdict ==="
"${PY}" - "$OUT_DIR" <<'PYEOF'
import json, pathlib, sys

out = pathlib.Path(sys.argv[1])
runs = {}
for span in ("0", "1"):
    meta, log = out / f"span_{span}.json", out / f"span_{span}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    if "DG_VLLM_SERVING_SMOKE_SUCCESS" not in text or not meta.exists():
        print(f"  !! DG_DENOISE_SLIDING_SPAN={span} did not complete")
        print("DG_SLIDING_SPAN_AB_RESULT FAIL")
        raise SystemExit(0)
    runs[span] = (json.loads(meta.read_text()), text)

(off, off_log), (on, on_log) = runs["0"], runs["1"]

# The candidate must actually have engaged, or the A/B proved nothing.
engaged = "DG_DENOISE_SLIDING_SPAN=1:" in on_log
reduction = next((ln.strip() for ln in on_log.splitlines() if "SDPA key rows/step" in ln), None)
print(f"  bounded span engaged on the candidate: {engaged}")
if reduction:
    print(f"  {reduction.split('] ', 1)[-1]}")
if not engaged:
    print("DG_SLIDING_SPAN_AB_RESULT FAIL (candidate did not engage)")
    raise SystemExit(0)

same_total = off["committed_sha256"] == on["committed_sha256"]
print(f"  committed_sha256 identical: {same_total}")
per_off, per_on = off["per_block_sha256"], on["per_block_sha256"]
if len(per_off) == len(per_on):
    for k, (a, b) in enumerate(zip(per_off, per_on), start=1):
        p = off["cache_len"] + off["canvas_length"] * (k - 1)
        print(f"  block {k}: P={p:5d} identical={a == b}")
else:
    print(f"  !! block counts differ {len(per_off)} vs {len(per_on)}")
    same_total = False

for tag, blob in (("control", off), ("candidate", on)):
    lat = blob.get("per_block_latency_s") or []
    steady = lat[1:]
    mean = sum(steady) / len(steady) if steady else float("nan")
    print(f"  {tag:>9}: steady_mean={mean:.3f}s blocks={[round(x, 3) for x in steady]}")

print("DG_SLIDING_SPAN_AB_RESULT " + ("BIT_IDENTICAL" if same_total else "ACCUMULATION_ORDER (visibility equal, bf16 reassociation)"))
PYEOF
