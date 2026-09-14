#!/usr/bin/env bash
# Chunk-size sweep: does moving off the hardcoded 5,120-token prefill chunk buy anything?
#
# Same runner + producer as run_matrix.sh, so CHUNK SIZE IS THE ONLY VARIABLE within a
# (config, ISL, mode) row. Background, model and predictions:
#   debug-docs/mistral4_prefill_planning-noissue/followup/CHUNK_SIZE_FINDINGS.md
#
# PREREQUISITE, and it is not optional: MLA_SDPA_CONFIG must carry rows for every swept width's
# seq_len_local (chunk/sp). Without them ttMLA falls back to k_chunk_size=32 and every non-5,120
# cell measures that fallback instead of the chunk. mla_config.py has 320 / 1280 / 2560 for exactly
# this; if you add a width, add its row.
#
# WHY BOTH MODES, and why they should disagree:
#   thru  - steady state, pipeline stays full across requests. Chunk size acts only on per-chunk
#           work, so this is where the ring-gather saving (~1/C) can show.
#   ttft  - ONE request. A PP=4 pipeline pays (S-1) chunks of fill, so a bigger chunk means FEWER
#           chunks to hide the fill behind: at ISL 102,400 the fill share goes 3/23 -> 3/13 -> 3/8
#           as the chunk goes 5,120 -> 10,240 -> 20,480. Expect the sign to INVERT against thru.
#           That is the point of running both, not redundancy. The 1rank rows have no pipeline and
#           serve as the control that separates the two effects.
#
# ISLs are chosen to divide by every swept width: 102,400 = 40x2,560 = 20x5,120 = 10x10,240 =
# 5x20,480, and 245,760 (in place of the matrix's 261,120, which is not divisible by 10,240) =
# 96/48/24/12. Cache WIDTH likewise.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/env.sh"
cd "${TT_METAL_HOME}" || exit 1
OUT="${OUT:-${TT_METAL_HOME}/mistral4_chunk_sweep_$(hostname)}"
mkdir -p "$OUT"
TOPO=models/demos/common/prefill/runners/topology_configuration
pick_binding(){ local b="$1"; local h="${b%.yaml}.$(hostname).yaml"; [ -f "$h" ] && echo "$h" || echo "$b"; }

declare -A BIND=(
  [1rank]="$TOPO/pipeline_prefill_request_1rank.yaml"
  [pp4]="$(pick_binding "$TOPO/pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.yaml")"
)
declare -A RANKS=( [1rank]=1 [pp4]=4 )
declare -A CACHE=( [1rank]=${M4_CACHE_8x4} [pp4]=${M4_CACHE_8x1} )
# Per-user KV cache width. Must be a multiple of EVERY swept chunk and >= the ISL.
declare -A WIDTH=( [102400]=122880 [245760]=245760 )

CONFIGS=${CONFIGS:-"pp4 1rank"}
ISLS=${ISLS:-"102400 245760"}
CHUNKSIZES=${CHUNKSIZES:-"5120 10240 20480"}
MODES=${MODES:-"thru ttft"}
# Keep total chunks per throughput cell in the 40-100 band however the chunk splits the ISL, so
# every cell sees a comparable amount of steady state.
TARGET_CHUNKS=${TARGET_CHUNKS:-60}

for cfg in $CONFIGS; do
 for isl in $ISLS; do
  for cs in $CHUNKSIZES; do
    if (( isl % cs != 0 )); then echo "[sweep] skip ${cfg}_${isl}_c${cs}: ISL not a multiple of chunk"; continue; fi
    w=${WIDTH[$isl]}
    if (( w % cs != 0 )); then echo "[sweep] skip ${cfg}_${isl}_c${cs}: cache width $w not a multiple of chunk"; continue; fi
    nchunks=$(( isl / cs ))
    for mode in $MODES; do
      tag="${cfg}_${isl}_c${cs}_${mode}"
      if [ -s "$OUT/$tag/runner.log" ] && [ -z "${FORCE:-}" ]; then
        echo "[sweep] skip $tag (log exists; FORCE=1 to redo)"; continue
      fi
      case "$mode" in
        ttft) req=1; users=1; kvonly=1 ;;
        *)    req=$(( (TARGET_CHUNKS + nchunks - 1) / nchunks )); [ "$req" -lt 2 ] && req=2
              users=2; kvonly=1 ;;
      esac
      # A single long request needs one slot; two at these depths would double the KV budget.
      [ "$isl" -ge 102400 ] && users=1
      echo "[sweep] === $tag (ranks=${RANKS[$cfg]} chunk=$cs nchunks=$nchunks width=$w req=$req users=$users) $(date -Is)"
      RUN_TAG="$tag" \
      PP_BINDING="${BIND[$cfg]}" PP_RANKS="${RANKS[$cfg]}" PP_TTNN_CACHE="${CACHE[$cfg]}" \
      PP_REQUESTS="$req" PP_USERS="$users" PP_CHUNKS="$nchunks" PP_CHUNK_SIZE="$cs" \
      PP_MAX_SEQ_LEN="$w" PP_KV_ONLY_LAST_LAYER="$kvonly" PREFILL_USE_TRACE=1 \
        "$S/run_pp4_model.sh" > "$OUT/${tag}.driver.log" 2>&1
      rc=$?
      mkdir -p "$OUT/$tag"
      cp "$S/logs/$tag/runner.log" "$S/logs/$tag/producer.log" "$OUT/$tag/" 2>/dev/null
      echo "[sweep] $tag rc=$rc"
      if [ "$rc" != "0" ]; then
        echo "[sweep] --- tail ---"; tail -15 "$OUT/${tag}.driver.log"
        # Same recovery discipline as run_matrix.sh: a hard failure can leave the fabric
        # un-mappable, and cascading through the rest of the sweep wastes the whole run.
        if ! "$S/check_board.sh" >/dev/null 2>&1; then
          echo "[sweep] board unmappable; attempting reset"
          tt-smi -r >/dev/null 2>&1 || tt-smi -glx_reset >/dev/null 2>&1
          sleep 30
          "$S/check_board.sh" >/dev/null 2>&1 || { echo "[sweep] ABORT: board still unmappable"; exit 1; }
        fi
      fi
    done
  done
 done
done
echo "[sweep] done -> $OUT"
