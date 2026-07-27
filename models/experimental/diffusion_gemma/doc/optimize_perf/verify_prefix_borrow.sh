#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Device A/B for the borrowed fixed-span prefix read (#51080 roadmap item 2).
#
# The up-front traced denoise path reads a FIXED p_max-row prefix every step. When that span
# covers the whole KV cache, the read used to ttnn.clone() the cache — ~2 whole-cache copies per
# layer per step of data that is bit-identical across all 48 steps of a block. Borrowing hands
# back the cache itself instead.
#
# Borrowing is bit-exact by construction (the downstream concat copies the bytes it needs), so
# this A/B asserts exactly that: committed_sha256 with DG_PREFIX_BORROW=1 (default) must equal
# DG_PREFIX_BORROW=0 (the clone).
#
# Requires the traced path, so it uses serving_smoke --upfront (the cheap stand-in for the vLLM
# wrapper's fail-loud startup contract).

set -uo pipefail

TT_METAL_ROOT="${TT_METAL_ROOT:-/home/zni/tt-metal}"
MODEL_VENV="${MODEL_VENV:-/home/zni/venvs/tt-diffusion-gemma}"
DG_CKPT="${DG_CKPT:-/home/zni/dg_models/diffusiongemma-26B-A4B-it}"
MESH="${MESH:-P150x4}"
NUM_LAYERS="${NUM_LAYERS:-}"     # empty = full 30
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
PMAX="${PMAX:-${MAX_SEQ_LEN}}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
TRACE_REGION_SIZE="${TRACE_REGION_SIZE:-6442450944}" # 6 GiB. Measured 2026-07-27: the 48 up-front traces need 3.04 GiB at reveal_pmax=4096 (3 GiB fails, 4 GiB is the floor), so the historical 12 GiB reserved ~8 GiB of DRAM that nothing could allocate. Scale this WITH reveal_pmax - it is not a universal constant. See doc/optimize_perf/bisect_trace_region.sh
OUT_DIR="${OUT_DIR:-/tmp/dg_prefix_borrow}"

mkdir -p "$OUT_DIR"
PY="${MODEL_VENV}/bin/python"
LAYER_ARG=()
[ -n "${NUM_LAYERS}" ] && LAYER_ARG=(--num-layers "${NUM_LAYERS}")

echo "DG_PREFIX_BORROW_AB_BEGIN layers=${NUM_LAYERS:-30} max_seq_len=${MAX_SEQ_LEN} p_max=${PMAX} blocks=${NUM_BLOCKS}"

for borrow in 1 0; do
    tag="borrow_${borrow}"
    echo "--- DG_PREFIX_BORROW=${borrow}"
    env \
        TT_METAL_HOME="${TT_METAL_ROOT}" \
        PYTHONPATH="${TT_METAL_ROOT}" \
        MESH_DEVICE="${MESH}" \
        DG_TRACE_REGION_SIZE="${TRACE_REGION_SIZE}" \
        DG_PREFIX_BORROW="${borrow}" \
        "${PY}" -u -m models.experimental.diffusion_gemma.demo.serving_smoke \
        --checkpoint "${DG_CKPT}" --mesh "${MESH}" "${LAYER_ARG[@]}" \
        --max-seq-len "${MAX_SEQ_LEN}" --num-blocks "${NUM_BLOCKS}" \
        --gumbel-mode host --disable-eos-stop --local-files-only \
        --upfront --reveal-pmax "${PMAX}" \
        --metrics-json "${OUT_DIR}/${tag}.json" \
        >"${OUT_DIR}/${tag}.log" 2>&1
    echo "DG_PREFIX_BORROW=${borrow} rc=$?"
done

echo
echo "=== verdict ==="
"${PY}" - "$OUT_DIR" <<'PYEOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
res = {}
for borrow in ("1", "0"):
    meta, log = out / f"borrow_{borrow}.json", out / f"borrow_{borrow}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    ok = "DG_VLLM_SERVING_SMOKE_SUCCESS" in text
    blob = json.loads(meta.read_text()) if meta.exists() else {}
    res[borrow] = (ok, blob)
    engaged = None
    for line in text.splitlines():
        if "borrow_full_span=" in line:
            engaged = line.split("borrow_full_span=", 1)[1].strip()
    print(f"  DG_PREFIX_BORROW={borrow}: success={ok} sha={blob.get('committed_sha256', 'MISSING')} "
          f"blocks={blob.get('blocks_emitted')} owns_result={blob.get('prefix_owns_result')} "
          f"steady={blob.get('per_block_latency_s') and [round(x,3) for x in blob['per_block_latency_s'][1:]]}")
    if engaged:
        print(f"      reader: borrow_full_span={engaged}")
ok_all = all(r[0] for r in res.values())
shas = {b: r[1].get("committed_sha256") for b, r in res.items()}
same = ok_all and len(set(shas.values())) == 1 and None not in shas.values()
# The ON run must actually have borrowed, or the A/B proved nothing.
engaged_on = res["1"][1].get("prefix_owns_result") is False
print(f"  borrow actually engaged on the ON run: {engaged_on}")
print("DG_PREFIX_BORROW_AB_RESULT " + ("BIT_IDENTICAL" if (same and engaged_on) else "FAIL"))
PYEOF
