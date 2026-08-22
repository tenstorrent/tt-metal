#!/bin/bash
# Serve Laguna-XS-2.1 on vLLM: stock vLLM 0.24.0 + public vllm-tt-plugin + this model's vllm_ext.
# Builds the env on first use (setup_vllm.sh), then backgrounds the server (setsid) and streams
# the FULL raw build+server log to a per-run timestamped file under $LAGUNA_LOG_DIR.
#   Launch:  ./serve_vllm.sh
#   Profile: LAGUNA_PROFILE=p150|p150x2|p150x4 ./serve_vllm.sh
#   Inspect: LAGUNA_PROFILE=p150 ./serve_vllm.sh config   (no device access)
#   Watch:   tail -f ~/laguna-logs/latest.log   (ready at "Application startup complete", ~10 min)
#   Stop:    ./serve_vllm.sh stop               (TERM/KILL + tt-smi -r all)
set +e

MODEL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$MODEL_DIR/../../.." && pwd)
VLLM_ENV="${VLLM_ENV:-$MODEL_DIR/.venv}"
VLLM_ENV_BIN="${VLLM_ENV_BIN:-$VLLM_ENV/bin}"
PIDF="${TMPDIR:-/tmp}/laguna_vllm_srv.$(id -u).pid"

die() {
  echo "[serve_vllm] ERROR: $*" >&2
  exit 2
}

is_positive_integer() {
  [[ "$1" =~ ^[0-9]+$ ]] && ((10#$1 > 0))
}

stop() {
  g=$(cat "$PIDF" 2>/dev/null)
  [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null
  sleep 10; [ -n "$g" ] && kill -KILL -"$g" 2>/dev/null
  pkill -9 -f "vllm serve poolside" 2>/dev/null; sleep 3
  # Hard-killing a FABRIC_1D_RING server dirties eth cores — reset before reopening the mesh.
  TT_SMI=$(command -v tt-smi 2>/dev/null)
  if [ -n "$TT_SMI" ]; then
    "$TT_SMI" -r all >/dev/null 2>&1
    echo "stopped + mesh reset"
  else
    echo "stopped — WARNING: no tt-smi found, mesh NOT reset (run 'tt-smi -r all' before rebooting)"
  fi
}
[ "$1" = "stop" ] && { stop; exit 0; }
case "${1:-}" in
  ""|config) ;;
  *) die "unknown command '$1' (expected: config or stop)" ;;
esac

# The qualified production default is two P150 ASICs. Keep D4 as an explicit regression profile.
DEFAULT_LAGUNA_PROFILE=p150x2
LAGUNA_PROFILE="${LAGUNA_PROFILE:-$DEFAULT_LAGUNA_PROFILE}"

case "$LAGUNA_PROFILE" in
  p150)
    PROFILE_DEVICE_COUNT=1
    PROFILE_VISIBLE_DEVICES=0
    PROFILE_MESH_DEVICE=P150
    PROFILE_MAX_MODEL_LEN=65536
    PROFILE_MAX_NUM_SEQS=1
    PROFILE_FABRIC_CONFIG=DISABLED
    PROFILE_CCL_TOPOLOGY=none
    PROFILE_CCL_NUM_LINKS=0
    PROFILE_DECODE_SDPA_PC=0
    PROFILE_PREFIX_CACHE_DEFAULT=0
    PROFILE_PREFIX_CACHE_POLICY=experimental_only
    ;;
  p150x2)
    PROFILE_DEVICE_COUNT=2
    PROFILE_VISIBLE_DEVICES=0,1
    PROFILE_MESH_DEVICE=P150x2
    PROFILE_MAX_MODEL_LEN=131072
    PROFILE_MAX_NUM_SEQS=1
    # Qualification may select linear/1-link later; ring/2 preserves the known multichip path now.
    PROFILE_FABRIC_CONFIG=FABRIC_1D_RING
    PROFILE_CCL_TOPOLOGY=ring
    PROFILE_CCL_NUM_LINKS=2
    PROFILE_DECODE_SDPA_PC=1
    # Qualified only for this exact D2 serving envelope. TT_LAGUNA_PREFIX_CACHE=0 remains the
    # fail-closed operator rollback and never requires an experimental acknowledgement.
    PROFILE_PREFIX_CACHE_DEFAULT=1
    PROFILE_PREFIX_CACHE_POLICY=qualified
    ;;
  p150x4)
    PROFILE_DEVICE_COUNT=4
    PROFILE_VISIBLE_DEVICES=0,1,2,3
    PROFILE_MESH_DEVICE=P150x4
    PROFILE_MAX_MODEL_LEN=131072
    PROFILE_MAX_NUM_SEQS=8
    PROFILE_FABRIC_CONFIG=FABRIC_1D_RING
    PROFILE_CCL_TOPOLOGY=ring
    PROFILE_CCL_NUM_LINKS=2
    PROFILE_DECODE_SDPA_PC=1
    PROFILE_PREFIX_CACHE_DEFAULT=0
    PROFILE_PREFIX_CACHE_POLICY=experimental_only
    ;;
  *)
    die "invalid LAGUNA_PROFILE '$LAGUNA_PROFILE' (expected p150, p150x2, or p150x4)"
    ;;
esac

# A singleton selected from this P300_X2 host is reported by UMD as ClusterType.CUSTOM and needs an
# explicit 1x1 graph. Multi-device profiles use normal discovery; require a deliberate opt-in before
# passing them any inherited/custom graph so the singleton descriptor cannot leak into D2 or D4.
ALLOW_CUSTOM_MESH_GRAPH_DESC="${LAGUNA_ALLOW_CUSTOM_MESH_GRAPH_DESC:-0}"
case "$ALLOW_CUSTOM_MESH_GRAPH_DESC" in
  0|1) ;;
  *) die "LAGUNA_ALLOW_CUSTOM_MESH_GRAPH_DESC must be 0 or 1" ;;
esac
if [ "$PROFILE_DEVICE_COUNT" -eq 1 ]; then
  if [ -z "${TT_MESH_GRAPH_DESC_PATH+x}" ]; then
    TT_MESH_GRAPH_DESC_PATH="$REPO_ROOT/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
  fi
  [ -f "$TT_MESH_GRAPH_DESC_PATH" ] ||
    die "p150 TT_MESH_GRAPH_DESC_PATH is not a file: '$TT_MESH_GRAPH_DESC_PATH'"
  TT_MESH_GRAPH_DESC_PATH=$(realpath -- "$TT_MESH_GRAPH_DESC_PATH") ||
    die "could not resolve p150 TT_MESH_GRAPH_DESC_PATH: '$TT_MESH_GRAPH_DESC_PATH'"
  export TT_MESH_GRAPH_DESC_PATH
elif [ -n "${TT_MESH_GRAPH_DESC_PATH+x}" ]; then
  [ "$ALLOW_CUSTOM_MESH_GRAPH_DESC" -eq 1 ] ||
    die "profile '$LAGUNA_PROFILE' does not use TT_MESH_GRAPH_DESC_PATH by default; unset it or explicitly set LAGUNA_ALLOW_CUSTOM_MESH_GRAPH_DESC=1"
  [ -f "$TT_MESH_GRAPH_DESC_PATH" ] ||
    die "TT_MESH_GRAPH_DESC_PATH is not a file: '$TT_MESH_GRAPH_DESC_PATH'"
  TT_MESH_GRAPH_DESC_PATH=$(realpath -- "$TT_MESH_GRAPH_DESC_PATH") ||
    die "could not resolve TT_MESH_GRAPH_DESC_PATH: '$TT_MESH_GRAPH_DESC_PATH'"
  export TT_MESH_GRAPH_DESC_PATH
else
  unset TT_MESH_GRAPH_DESC_PATH
fi

# Pin visibility before vLLM imports TTNN. A caller may choose different devices (including BDFs),
# but the explicit identifier count must agree with the selected 1xD profile. Duplicate identifiers
# are rejected because UMD deduplicates them and would otherwise open fewer devices than requested.
if [ -z "${TT_VISIBLE_DEVICES+x}" ]; then
  TT_VISIBLE_DEVICES="$PROFILE_VISIBLE_DEVICES"
fi
TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES//[[:space:]]/}"
case "$TT_VISIBLE_DEVICES" in
  ""|,*|*,|*,,*) die "TT_VISIBLE_DEVICES must contain $PROFILE_DEVICE_COUNT non-empty identifier(s)" ;;
esac
IFS=',' read -r -a VISIBLE_DEVICE_IDS <<< "$TT_VISIBLE_DEVICES"
[ "${#VISIBLE_DEVICE_IDS[@]}" -eq "$PROFILE_DEVICE_COUNT" ] ||
  die "profile '$LAGUNA_PROFILE' requires $PROFILE_DEVICE_COUNT TT_VISIBLE_DEVICES identifier(s), got '${TT_VISIBLE_DEVICES}'"
for ((i = 0; i < ${#VISIBLE_DEVICE_IDS[@]}; i++)); do
  for ((j = i + 1; j < ${#VISIBLE_DEVICE_IDS[@]}; j++)); do
    [ "${VISIBLE_DEVICE_IDS[$i]}" != "${VISIBLE_DEVICE_IDS[$j]}" ] ||
      die "TT_VISIBLE_DEVICES contains duplicate identifier '${VISIBLE_DEVICE_IDS[$i]}'"
  done
done
export TT_VISIBLE_DEVICES

# MESH_DEVICE is derived, not an independent topology selector. Reject a stale inherited value rather
# than silently opening a shape different from the requested profile.
if [ -n "${MESH_DEVICE+x}" ] && [ "$MESH_DEVICE" != "$PROFILE_MESH_DEVICE" ]; then
  die "MESH_DEVICE '$MESH_DEVICE' conflicts with profile '$LAGUNA_PROFILE' (expected $PROFILE_MESH_DEVICE)"
fi
export MESH_DEVICE="$PROFILE_MESH_DEVICE"

MAX_MODEL_LEN="${LAGUNA_MAX_MODEL_LEN:-$PROFILE_MAX_MODEL_LEN}"
is_positive_integer "$MAX_MODEL_LEN" || die "LAGUNA_MAX_MODEL_LEN must be a positive integer"
((10#$MAX_MODEL_LEN <= PROFILE_MAX_MODEL_LEN)) ||
  die "LAGUNA_MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds the verified $LAGUNA_PROFILE limit $PROFILE_MAX_MODEL_LEN"

MAX_NUM_SEQS="${LAGUNA_MAX_NUM_SEQS:-$PROFILE_MAX_NUM_SEQS}"
is_positive_integer "$MAX_NUM_SEQS" || die "LAGUNA_MAX_NUM_SEQS must be a positive integer"
((10#$MAX_NUM_SEQS <= PROFILE_MAX_NUM_SEQS)) ||
  die "LAGUNA_MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds the $LAGUNA_PROFILE limit $PROFILE_MAX_NUM_SEQS"

TRACE_REGION_SIZE="${LAGUNA_TRACE_REGION_SIZE:-1500000000}"
is_positive_integer "$TRACE_REGION_SIZE" || die "LAGUNA_TRACE_REGION_SIZE must be a positive integer"

# CCL topology and fabric routing must move together. Setting either variable alone derives the other,
# which makes D2 qualification concise while still rejecting mismatched combinations.
FABRIC_CONFIG_IS_SET=0
CCL_TOPOLOGY_IS_SET=0
[ -n "${LAGUNA_FABRIC_CONFIG+x}" ] && FABRIC_CONFIG_IS_SET=1
[ -n "${TT_LAGUNA_CCL_TOPOLOGY+x}" ] && CCL_TOPOLOGY_IS_SET=1
FABRIC_CONFIG="${LAGUNA_FABRIC_CONFIG:-$PROFILE_FABRIC_CONFIG}"
CCL_TOPOLOGY="${TT_LAGUNA_CCL_TOPOLOGY:-$PROFILE_CCL_TOPOLOGY}"

if [ "$PROFILE_DEVICE_COUNT" -eq 1 ]; then
  [ "$FABRIC_CONFIG" = DISABLED ] || die "p150 requires LAGUNA_FABRIC_CONFIG=DISABLED"
  [ "$CCL_TOPOLOGY" = none ] || die "p150 requires TT_LAGUNA_CCL_TOPOLOGY=none"
  CCL_NUM_LINKS="${TT_LAGUNA_CCL_NUM_LINKS:-$PROFILE_CCL_NUM_LINKS}"
  [ "$CCL_NUM_LINKS" = 0 ] || die "p150 requires TT_LAGUNA_CCL_NUM_LINKS=0"
else
  if [ "$CCL_TOPOLOGY_IS_SET" -eq 1 ] && [ "$FABRIC_CONFIG_IS_SET" -eq 0 ]; then
    case "$CCL_TOPOLOGY" in
      linear) FABRIC_CONFIG=FABRIC_1D ;;
      ring) FABRIC_CONFIG=FABRIC_1D_RING ;;
      *) die "TT_LAGUNA_CCL_TOPOLOGY must be 'linear' or 'ring' for $LAGUNA_PROFILE" ;;
    esac
  elif [ "$FABRIC_CONFIG_IS_SET" -eq 1 ] && [ "$CCL_TOPOLOGY_IS_SET" -eq 0 ]; then
    case "$FABRIC_CONFIG" in
      FABRIC_1D) CCL_TOPOLOGY=linear ;;
      FABRIC_1D_RING) CCL_TOPOLOGY=ring ;;
      *) die "LAGUNA_FABRIC_CONFIG must be FABRIC_1D or FABRIC_1D_RING for $LAGUNA_PROFILE" ;;
    esac
  fi

  case "$CCL_TOPOLOGY:$FABRIC_CONFIG" in
    linear:FABRIC_1D|ring:FABRIC_1D_RING) ;;
    *) die "incompatible CCL/fabric pair '$CCL_TOPOLOGY/$FABRIC_CONFIG'" ;;
  esac
  CCL_NUM_LINKS="${TT_LAGUNA_CCL_NUM_LINKS:-$PROFILE_CCL_NUM_LINKS}"
  case "$CCL_NUM_LINKS" in
    1|2) ;;
    *) die "TT_LAGUNA_CCL_NUM_LINKS must be 1 or 2 for $LAGUNA_PROFILE" ;;
  esac
fi
export LAGUNA_FABRIC_CONFIG="$FABRIC_CONFIG"
export TT_LAGUNA_CCL_TOPOLOGY="$CCL_TOPOLOGY"
export TT_LAGUNA_CCL_NUM_LINKS="$CCL_NUM_LINKS"

HF_MODEL="${HF_MODEL:-poolside/Laguna-XS-2.1}"
[ "$HF_MODEL" = "poolside/Laguna-XS-2.1" ] ||
  die "HF_MODEL must be poolside/Laguna-XS-2.1; the adapter and cached weights are model-specific"
export HF_MODEL

# Resolve prefix caching from the profile before collecting experimental overrides. Enabling an
# unqualified profile must be acknowledged, while an explicit 0 is always a safe rollback if a
# profile is later promoted to a qualified default of 1.
PREFIX_CACHE_ENV_IS_SET=0
[ -n "${TT_LAGUNA_PREFIX_CACHE+x}" ] && PREFIX_CACHE_ENV_IS_SET=1
TT_LAGUNA_PREFIX_CACHE="${TT_LAGUNA_PREFIX_CACHE:-$PROFILE_PREFIX_CACHE_DEFAULT}"
case "$TT_LAGUNA_PREFIX_CACHE" in
  0|1) ;;
  *) die "TT_LAGUNA_PREFIX_CACHE must be 0 or 1" ;;
esac
case "$PROFILE_PREFIX_CACHE_POLICY:$PROFILE_PREFIX_CACHE_DEFAULT" in
  experimental_only:0|qualification_candidate:0|qualified:1) ;;
  *) die "internal prefix-cache policy mismatch for '$LAGUNA_PROFILE': $PROFILE_PREFIX_CACHE_POLICY/default=$PROFILE_PREFIX_CACHE_DEFAULT" ;;
esac
PREFIX_CACHE_QUANTUM=8192
PREFIX_CACHE_BLOCK_SIZE=64
PREFIX_CACHE_ADMISSION_POLICY=complete_canonical_prompt_chunks
PREFIX_CACHE_KV_GROUP_POLICY=single_uniform_full_attention
PREFIX_CACHE_SCHEDULER_POLICY=max_num_seqs_1_no_chunked_prefill
PREFIX_CACHE_SPEC_DECODE_POLICY=disabled
PREFIX_CACHE_EXTERNAL_KV_POLICY=disabled

# Production profiles are reproducible only if inherited bring-up knobs cannot silently change the
# layer count, precision, attention geometry, warm set, or speculative path. Exact qualified values
# are accepted; any deviation (and any presence-only debug knob) requires an explicit diagnostic opt-in
# and is printed in `config` output / the launch log. The opt-in does not make the result qualified.
ALLOW_EXPERIMENTAL_OVERRIDES="${LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES:-0}"
case "$ALLOW_EXPERIMENTAL_OVERRIDES" in
  0|1) ;;
  *) die "LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES must be 0 or 1" ;;
esac

EXPERIMENTAL_OVERRIDES=()
record_if_set() {
  local name="$1"
  if declare -p "$name" >/dev/null 2>&1; then
    EXPERIMENTAL_OVERRIDES+=("$name=${!name}")
  fi
}
record_if_nondefault() {
  local name="$1" expected="$2"
  if declare -p "$name" >/dev/null 2>&1 && [ "${!name}" != "$expected" ]; then
    EXPERIMENTAL_OVERRIDES+=("$name=${!name} (qualified=$expected)")
  fi
}

for name in \
  TT_LAGUNA_ADVERTISED_CONTEXT \
  TT_LAGUNA_VLLM_NUM_LAYERS \
  TT_LAGUNA_PREFILL_WARM_CAP \
  TT_LAGUNA_PRECISION_CONFIG \
  TT_LAGUNA_SPEC_DECODE \
  TT_LAGUNA_SPEC_K \
  TT_LAGUNA_SPEC_LOGITS \
  TT_LAGUNA_SPEC_NGRAM_MAX \
  TT_LAGUNA_SPEC_SINGLE \
  TT_LAGUNA_SPEC_TRACED \
  TT_LAGUNA_SPEC_DEBUG \
  TT_LAGUNA_SPEC_STEP_LOG \
  TT_LAGUNA_NO_ROPE_HOIST \
  TT_LAGUNA_FUSED_REDUCE \
  TT_LAGUNA_FUSED_ROPE \
  TT_LAGUNA_FUSE_QKV_DECODE \
  TT_LAGUNA_ROPE_DEBUG \
  TT_LAGUNA_WEIGHT_CACHE_DISABLE; do
  record_if_set "$name"
done
record_if_nondefault TT_LAGUNA_PIPE_CHUNK 2048
# Enabled caching on a non-qualified profile remains diagnostic-only. A disable is intentionally not
# recorded: TT_LAGUNA_PREFIX_CACHE=0 is the emergency rollback for a qualified/default-on profile and
# must never require an experimental acknowledgement.
if [ "$TT_LAGUNA_PREFIX_CACHE" -eq 1 ] && [ "$PROFILE_PREFIX_CACHE_POLICY" != qualified ]; then
  EXPERIMENTAL_OVERRIDES+=(
    "TT_LAGUNA_PREFIX_CACHE=1 (profile_policy=$PROFILE_PREFIX_CACHE_POLICY; default=$PROFILE_PREFIX_CACHE_DEFAULT)"
  )
fi
record_if_nondefault TT_LAGUNA_PREFILL_FAST 1
record_if_nondefault TT_LAGUNA_PREFILL_FAST_CHUNK 8192
record_if_nondefault TT_LAGUNA_PREFILL_SDPA_CHUNK 8192
record_if_nondefault TT_LAGUNA_DECODE_SDPA_PC "$PROFILE_DECODE_SDPA_PC"
record_if_nondefault TT_LAGUNA_DECODE_K 64
record_if_nondefault TT_LAGUNA_DECODE_EXP 0
record_if_nondefault TT_LAGUNA_DECODE_MAXCORES 16
record_if_nondefault TT_LAGUNA_VERIFY_K 64
record_if_nondefault TT_LAGUNA_ENFORCE_MEMORY_MARGIN 1
record_if_nondefault TT_LAGUNA_MIN_DRAM_FREE_FRACTION 0.10
record_if_nondefault TT_LAGUNA_MIN_CONTIGUOUS_MIB 128

if [ "${#EXPERIMENTAL_OVERRIDES[@]}" -gt 0 ]; then
  printf -v EXPERIMENTAL_OVERRIDE_SUMMARY '%s; ' "${EXPERIMENTAL_OVERRIDES[@]}"
  EXPERIMENTAL_OVERRIDE_SUMMARY="${EXPERIMENTAL_OVERRIDE_SUMMARY%; }"
  if [ "$ALLOW_EXPERIMENTAL_OVERRIDES" -ne 1 ]; then
    die "unqualified inherited/debug override(s): $EXPERIMENTAL_OVERRIDE_SUMMARY. Unset them, or set LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1 for a diagnostic-only run"
  fi
  echo "[serve_vllm] WARNING: diagnostic-only experimental override(s): $EXPERIMENTAL_OVERRIDE_SUMMARY" >&2
else
  EXPERIMENTAL_OVERRIDE_SUMMARY="<none>"
fi

export TT_LAGUNA_PIPE_CHUNK="${TT_LAGUNA_PIPE_CHUNK:-2048}"
export TT_LAGUNA_PREFIX_CACHE
export TT_LAGUNA_PREFILL_FAST="${TT_LAGUNA_PREFILL_FAST:-1}"
export TT_LAGUNA_PREFILL_FAST_CHUNK="${TT_LAGUNA_PREFILL_FAST_CHUNK:-8192}"
export TT_LAGUNA_PREFILL_SDPA_CHUNK="${TT_LAGUNA_PREFILL_SDPA_CHUNK:-8192}"
export TT_LAGUNA_DECODE_SDPA_PC="${TT_LAGUNA_DECODE_SDPA_PC:-$PROFILE_DECODE_SDPA_PC}"
export TT_LAGUNA_DECODE_K="${TT_LAGUNA_DECODE_K:-64}"
export TT_LAGUNA_DECODE_EXP="${TT_LAGUNA_DECODE_EXP:-0}"
export TT_LAGUNA_DECODE_MAXCORES="${TT_LAGUNA_DECODE_MAXCORES:-16}"
export TT_LAGUNA_VERIFY_K="${TT_LAGUNA_VERIFY_K:-64}"
export TT_LAGUNA_ENFORCE_MEMORY_MARGIN="${TT_LAGUNA_ENFORCE_MEMORY_MARGIN:-1}"
export TT_LAGUNA_MIN_DRAM_FREE_FRACTION="${TT_LAGUNA_MIN_DRAM_FREE_FRACTION:-0.10}"
export TT_LAGUNA_MIN_CONTIGUOUS_MIB="${TT_LAGUNA_MIN_CONTIGUOUS_MIB:-128}"
is_positive_integer "$TT_LAGUNA_PIPE_CHUNK" || die "TT_LAGUNA_PIPE_CHUNK must be a positive integer"
((10#$TT_LAGUNA_PIPE_CHUNK >= 64 && 10#$TT_LAGUNA_PIPE_CHUNK % 64 == 0)) ||
  die "TT_LAGUNA_PIPE_CHUNK must be a positive multiple of the production KV block size 64"
is_positive_integer "$TT_LAGUNA_PREFILL_FAST_CHUNK" || die "TT_LAGUNA_PREFILL_FAST_CHUNK must be a positive integer"
((10#$TT_LAGUNA_PREFILL_FAST_CHUNK % 64 == 0)) || die "TT_LAGUNA_PREFILL_FAST_CHUNK must be a multiple of 64"
is_positive_integer "$TT_LAGUNA_PREFILL_SDPA_CHUNK" || die "TT_LAGUNA_PREFILL_SDPA_CHUNK must be a positive integer"
((10#$TT_LAGUNA_PREFILL_SDPA_CHUNK % 64 == 0)) || die "TT_LAGUNA_PREFILL_SDPA_CHUNK must be a multiple of 64"
case "$TT_LAGUNA_PREFIX_CACHE:$TT_LAGUNA_PREFILL_FAST:$TT_LAGUNA_ENFORCE_MEMORY_MARGIN" in
  [01]:[01]:[01]) ;;
  *) die "TT_LAGUNA_PREFIX_CACHE, TT_LAGUNA_PREFILL_FAST, and TT_LAGUNA_ENFORCE_MEMORY_MARGIN must be 0 or 1" ;;
esac
if [ "$TT_LAGUNA_PREFIX_CACHE" -eq 1 ]; then
  PREFIX_CACHE_CLI_ARG=--enable-prefix-caching
  PROMPT_TOKENS_DETAILS_CLI_ARG=--enable-prompt-tokens-details
  PREFIX_CACHE_CLI_ARGS=("$PREFIX_CACHE_CLI_ARG" "$PROMPT_TOKENS_DETAILS_CLI_ARG")
  CHUNKED_PREFILL_CLI_ARG=--no-enable-chunked-prefill
  CHUNKED_PREFILL_CLI_ARGS=("$CHUNKED_PREFILL_CLI_ARG")
  case "$PROFILE_PREFIX_CACHE_POLICY" in
    qualified) PREFIX_CACHE_STATUS=production_qualified ;;
    qualification_candidate) PREFIX_CACHE_STATUS=qualification_candidate ;;
    experimental_only) PREFIX_CACHE_STATUS=experimental_unqualified ;;
  esac
else
  # Be explicit: vLLM's CLI default is version/config dependent (currently None). This branch covers
  # profiles that remain cache-off and the p150x2 operator rollback; neither may schedule a suffix hit.
  PREFIX_CACHE_CLI_ARG=--no-enable-prefix-caching
  PROMPT_TOKENS_DETAILS_CLI_ARG='<none>'
  PREFIX_CACHE_CLI_ARGS=("$PREFIX_CACHE_CLI_ARG")
  CHUNKED_PREFILL_CLI_ARG='<vllm-default>'
  CHUNKED_PREFILL_CLI_ARGS=()
  if [ "$PREFIX_CACHE_ENV_IS_SET" -eq 1 ] && [ "$PROFILE_PREFIX_CACHE_DEFAULT" -eq 1 ]; then
    PREFIX_CACHE_STATUS=operator_rollback_disabled
  else
    PREFIX_CACHE_STATUS=production_safe_disabled
  fi
fi
case "$TT_LAGUNA_DECODE_EXP" in
  0|1) ;;
  *) die "TT_LAGUNA_DECODE_EXP must be 0 or 1" ;;
esac
case "$TT_LAGUNA_DECODE_K:$TT_LAGUNA_VERIFY_K" in
  32:32|32:64|32:128|64:32|64:64|64:128|128:32|128:64|128:128) ;;
  *) die "TT_LAGUNA_DECODE_K and TT_LAGUNA_VERIFY_K must each be 32, 64, or 128" ;;
esac
is_positive_integer "$TT_LAGUNA_DECODE_MAXCORES" || die "TT_LAGUNA_DECODE_MAXCORES must be a positive integer"
case "$TT_LAGUNA_DECODE_SDPA_PC" in
  0|1) ;;
  *) die "TT_LAGUNA_DECODE_SDPA_PC must be 0 or 1" ;;
esac
if [ "$PROFILE_DEVICE_COUNT" -eq 1 ] && [ "$TT_LAGUNA_DECODE_SDPA_PC" -eq 1 ]; then
  echo "[serve_vllm] WARNING: TT_LAGUNA_DECODE_SDPA_PC=1 is unqualified and known inaccurate for long-context p150 decode; use only for explicit debugging" >&2
  DECODE_SDPA_PC_STATUS=unsafe_p150_override
else
  DECODE_SDPA_PC_STATUS=profile_safe
fi
[[ "$TT_LAGUNA_MIN_DRAM_FREE_FRACTION" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] ||
  die "TT_LAGUNA_MIN_DRAM_FREE_FRACTION must be between 0 and 1"
is_positive_integer "$TT_LAGUNA_MIN_CONTIGUOUS_MIB" ||
  die "TT_LAGUNA_MIN_CONTIGUOUS_MIB must be a positive integer"
if [ -n "${TT_LAGUNA_HYBRID_KV+x}" ] && [ "$TT_LAGUNA_HYBRID_KV" != 0 ]; then
  die "TT_LAGUNA_HYBRID_KV must remain 0; hybrid KV is not qualified for Laguna serving"
fi
export TT_LAGUNA_HYBRID_KV=0
if [ "$TT_LAGUNA_PREFIX_CACHE" -eq 1 ]; then
  [ "$MAX_NUM_SEQS" -eq 1 ] ||
    die "Laguna canonical prefix caching requires LAGUNA_MAX_NUM_SEQS=1"
  [ "$TT_LAGUNA_PREFILL_FAST" -eq 1 ] ||
    die "Laguna canonical prefix caching requires TT_LAGUNA_PREFILL_FAST=1"
  [ "$TT_LAGUNA_PREFILL_FAST_CHUNK" -eq "$PREFIX_CACHE_QUANTUM" ] ||
    die "Laguna canonical prefix caching requires TT_LAGUNA_PREFILL_FAST_CHUNK=$PREFIX_CACHE_QUANTUM"
  [ "$TT_LAGUNA_PREFILL_SDPA_CHUNK" -eq "$PREFIX_CACHE_QUANTUM" ] ||
    die "Laguna canonical prefix caching requires TT_LAGUNA_PREFILL_SDPA_CHUNK=$PREFIX_CACHE_QUANTUM"
  [ -z "${TT_LAGUNA_SPEC_DECODE:-}" ] ||
    die "Laguna canonical prefix caching does not support TT_LAGUNA_SPEC_DECODE"
fi

printf -v VLLM_ADDITIONAL_CONFIG \
  '{"tt": {"sample_on_device_mode": "all", "trace_region_size": %s, "fabric_config": "%s"}}' \
  "$TRACE_REGION_SIZE" "$FABRIC_CONFIG"

if [ "$1" = "config" ]; then
  printf '%s\n' \
    "profile=$LAGUNA_PROFILE" \
    "mesh_device=$MESH_DEVICE" \
    "mesh_graph_desc_path=${TT_MESH_GRAPH_DESC_PATH:-<unset>}" \
    "tt_visible_devices=$TT_VISIBLE_DEVICES" \
    "device_count=$PROFILE_DEVICE_COUNT" \
    "max_model_len=$MAX_MODEL_LEN" \
    "max_num_seqs=$MAX_NUM_SEQS" \
    "fabric_config=$FABRIC_CONFIG" \
    "ccl_topology=$TT_LAGUNA_CCL_TOPOLOGY" \
    "ccl_num_links=$TT_LAGUNA_CCL_NUM_LINKS" \
    "decode_sdpa_pc=$TT_LAGUNA_DECODE_SDPA_PC" \
    "decode_sdpa_pc_status=$DECODE_SDPA_PC_STATUS" \
    "decode_k=$TT_LAGUNA_DECODE_K" \
    "decode_exp=$TT_LAGUNA_DECODE_EXP" \
    "decode_maxcores=$TT_LAGUNA_DECODE_MAXCORES" \
    "verify_k=$TT_LAGUNA_VERIFY_K" \
    "pipe_chunk=$TT_LAGUNA_PIPE_CHUNK" \
    "prefix_cache=$TT_LAGUNA_PREFIX_CACHE" \
    "prefix_cache_profile_default=$PROFILE_PREFIX_CACHE_DEFAULT" \
    "prefix_cache_profile_policy=$PROFILE_PREFIX_CACHE_POLICY" \
    "prefix_cache_env_explicit=$PREFIX_CACHE_ENV_IS_SET" \
    "prefix_cache_status=$PREFIX_CACHE_STATUS" \
    "prefix_cache_cli_arg=$PREFIX_CACHE_CLI_ARG" \
    "prompt_tokens_details_cli_arg=$PROMPT_TOKENS_DETAILS_CLI_ARG" \
    "prefix_cache_cli_args=${PREFIX_CACHE_CLI_ARGS[*]}" \
    "prefix_cache_quantum=$PREFIX_CACHE_QUANTUM" \
    "prefix_cache_block_size=$PREFIX_CACHE_BLOCK_SIZE" \
    "prefix_cache_admission_policy=$PREFIX_CACHE_ADMISSION_POLICY" \
    "prefix_cache_kv_group_policy=$PREFIX_CACHE_KV_GROUP_POLICY" \
    "prefix_cache_scheduler_policy=$PREFIX_CACHE_SCHEDULER_POLICY" \
    "prefix_cache_spec_decode_policy=$PREFIX_CACHE_SPEC_DECODE_POLICY" \
    "prefix_cache_external_kv_policy=$PREFIX_CACHE_EXTERNAL_KV_POLICY" \
    "chunked_prefill_cli_arg=$CHUNKED_PREFILL_CLI_ARG" \
    "prefill_fast=$TT_LAGUNA_PREFILL_FAST" \
    "prefill_fast_chunk=$TT_LAGUNA_PREFILL_FAST_CHUNK" \
    "prefill_sdpa_chunk=$TT_LAGUNA_PREFILL_SDPA_CHUNK" \
    "allow_experimental_overrides=$ALLOW_EXPERIMENTAL_OVERRIDES" \
    "experimental_overrides=$EXPERIMENTAL_OVERRIDE_SUMMARY" \
    "trace_region_size=$TRACE_REGION_SIZE" \
    "enforce_memory_margin=$TT_LAGUNA_ENFORCE_MEMORY_MARGIN" \
    "min_dram_free_fraction=$TT_LAGUNA_MIN_DRAM_FREE_FRACTION" \
    "min_contiguous_mib=$TT_LAGUNA_MIN_CONTIGUOUS_MIB" \
    "additional_config=$VLLM_ADDITIONAL_CONFIG"
  exit 0
fi

# One file per run, never overwritten, so a boot's log survives the next boot. `latest.log` is a
# stable symlink to the current run: `tail -f` it and you always follow the run in progress.
# NOT a single reused path — that made a hours-long first build look "ready", because the previous
# run's "Application startup complete" was still sitting in the file the whole time.
LOG_DIR="${LAGUNA_LOG_DIR:-$HOME/laguna-logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/laguna_serve_$(date +%Y%m%d-%H%M%S).log"
LATEST="$LOG_DIR/latest.log"

# Open this run's log before anything else, so `tail -f latest.log` follows the build too — on a
# first run that is hours of real progress, not a silent file.
: > "$LOG"
ln -sfn "$LOG" "$LATEST"
echo "[serve_vllm] log: $LOG" | tee -a "$LOG"

# One command from a fresh clone: build the env if it isn't there yet.
if [ ! -x "$VLLM_ENV_BIN/vllm" ]; then
  echo "[serve_vllm] no env at $VLLM_ENV — running setup_vllm.sh first (first run: hours)" | tee -a "$LOG"
  VLLM_ENV="$VLLM_ENV" "$MODEL_DIR/setup_vllm.sh" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[serve_vllm] setup failed" | tee -a "$LOG"; exit 1; }
fi

# setup_vllm.sh installs TTNN from this checkout in editable mode. Keep TT_METAL_HOME unset so an
# inherited path cannot redirect the runtime to a different tt-metal tree and mismatched kernels.
unset TT_METAL_HOME
export PYTHONPATH="$REPO_ROOT"          # so EXTRA_MODELS_DIR's main_class (generator_vllm) resolves
export EXTRA_MODELS_DIR="$MODEL_DIR/vllm_ext/extra_models"

echo "[serve_vllm] vllm $("$VLLM_ENV_BIN/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null) | env: $VLLM_ENV | log: $LOG" | tee -a "$LOG"
echo "[serve_vllm] profile: $LAGUNA_PROFILE | mesh: $MESH_DEVICE | devices: $TT_VISIBLE_DEVICES | context: $MAX_MODEL_LEN | seqs: $MAX_NUM_SEQS | prefix cache: $TT_LAGUNA_PREFIX_CACHE ($PREFIX_CACHE_STATUS; quantum=$PREFIX_CACHE_QUANTUM; admission=$PREFIX_CACHE_ADMISSION_POLICY; scheduler=$PREFIX_CACHE_SCHEDULER_POLICY) | fabric: $FABRIC_CONFIG | CCL: $TT_LAGUNA_CCL_TOPOLOGY/$TT_LAGUNA_CCL_NUM_LINKS | decode SDPA PC/k/exp/maxcores: $TT_LAGUNA_DECODE_SDPA_PC/$TT_LAGUNA_DECODE_K/$TT_LAGUNA_DECODE_EXP/$TT_LAGUNA_DECODE_MAXCORES | verify k: $TT_LAGUNA_VERIFY_K | experimental: $EXPERIMENTAL_OVERRIDE_SUMMARY" | tee -a "$LOG"
cd /tmp
setsid "$VLLM_ENV_BIN/vllm" serve "$HF_MODEL" \
  --trust-remote-code --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --block-size "$PREFIX_CACHE_BLOCK_SIZE" \
  --additional-config "$VLLM_ADDITIONAL_CONFIG" \
  "${PREFIX_CACHE_CLI_ARGS[@]}" "${CHUNKED_PREFILL_CLI_ARGS[@]}" --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --port 8000 >> "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[serve_vllm] booting (pid $(cat "$PIDF")). Ready at 'Application startup complete' (~10 min)."
echo "  tail -f $LATEST"
