#!/usr/bin/env bash
# Start the Qwen3.8-Flash-Next chat server on one 1x4 mesh of Blackhole chips, from this checkout's build.
#
#   tools/run_qwen38_chat_server.sh --profile tt-quietbox|p150-line|qb2 --checkpoint DIR --cache-root DIR [options]
#
#   --profile tt-quietbox   QuietBox, 4x p150b in an ethernet ring (verified; opened as a 1x4 line through the shipped
#                           mesh graph descriptor)
#   --profile p150-line     4x p150 in one host as an ethernet line, ttnn's default descriptor; the route is
#                           derived at start
#   --profile qb2           p300-based box (QuietBox 2, or a 4x p300 host with --instance 0|1) -- UNTESTED, see README
#   --devices A,B,C,D       run p150-line on these four KMD device nodes (four chips of a larger host)
#   --checkpoint DIR        the ModelScope checkpoint directory (tools/download_checkpoint.py)
#   --cache-root DIR        the converted weights, the BF4 expert cache, the model I/O cache, the JIT cache and the run
#                           directories (about 23 GB for 32k plus 107 GB of BF4 experts on the first start)
#   --allocated-context N   32768 (default) | 65536 | 131072 | 262144
#   --mtp K                 multi-token-prediction drafting depth, 3, 4 or 5 (off by default; greedy chunked-mode requests draft)
#   --long-chunks           accepted for compatibility: the 128-row chunks are the server's default (combines with --mtp)
#   --prefill-slab ROWS     prefill in slabs of ROWS rows (a multiple of 128, 256..4096; 2048 is the measured form)
#                           ahead of the 128-row chunks (off by default; implies --long-chunks; not with --mtp)
#   --no-sampling           serve greedy requests only (the default server takes --sampling: a request naming no
#                           sampling field is still the bitwise greedy stream, temperature > 0 samples)
#   --stall-seconds N       a request with no completed device step for N seconds ends the server with exit 1 so a
#                           supervisor restarts it (default 300; every decode step, prefill event and admission from
#                           the queue restarts the clock, so a long prompt never trips it); 0 disables the watchdog
#   --port N --host ADDR    default 8000 on 0.0.0.0
#   --acceptance            replay the shipped CPU greedy records (tools/acceptance/greedy-prompts) at startup
#   --acceptance-prompts D  replay the records in D instead
#   --require-json-96       refuse to serve unless the json record matches the CPU 96/96
#   --acceptance-only       replay acceptance, write evidence and stop (requires --acceptance or --acceptance-prompts)
#   --prepare-only          build the BF4 expert cache (convert the missing layers) and stop; --bf4-stage-limit N
#                           converts at most N layers per run (resumable)
#   --bf4-corpus DIR --bf4-corpus-verification FILE
#                           a CPU-staged corpus (tools/stage_full_bf4_cpu.py) instead of the first-start conversion
#   --serve-seconds N       stop after N seconds (default: until SIGTERM)
#   --python PATH           the interpreter that imports ttnn (default: <checkout>/python_env/bin/python)
#   --validate-only         run the checks and the CPU preparation, do not open the mesh
#   --prefill-mode M        chunked (default) | teacher_forced (every prompt token through the decode step)
#   --agreement-reference F --agreement-parts "P Q" --agreement-items "ID ID" --agreement-full-logits DIR
#                           teacher-force the reference corpus Q38-REF-v1 through the chain before READY and write
#                           the agreement records, the device column and its score against F (needs the sampled
#                           server; see tools/qwen38_reference_corpus.py)
#   QWEN38_DENSE_WEIGHT_DTYPE=bf8|bf16|bf4 (environment; default bf8; bf16 = the previous production format): the resident dense matmul
#                           weights' dtype (ttnn/decode_matmul.py); the launcher passes it through and prints it
#
# The checkout this script lives in must be built (build_metal.sh, create_venv.sh); the server admits only a ttnn
# imported from it and records the checkout's commit, tree and extension digest as the runtime identity.  No locks, no
# archives, no seals.  Start it from any directory, the model directory included: relative path arguments are the
# caller's, and the interpreter runs from the repository root so the model's own ttnn/ package never shadows ttnn.
set -euo pipefail

readonly HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly MODEL_DIR="$(cd -- "$HERE/.." && pwd -P)"
readonly REPO_ROOT="$(cd -- "$MODEL_DIR/../../../.." && pwd -P)"
readonly SERVER="$HERE/qwen38_chat_server.py"

die() { printf 'run_qwen38_chat_server: %s\n' "$*" >&2; exit 2; }
usage() { sed -n '2,47p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

profile= instance=0 devices= checkpoint= cache_root= allocated_context=32768 mtp= port=8000 host=0.0.0.0 long_chunks=
prefill_slab=
acceptance= acceptance_prompts= require_json_96= acceptance_only= prepare_only= bf4_stage_limit= bf4_corpus= bf4_corpus_verification=
serve_seconds= python= validate_only= prefill_mode=chunked sampling=1 stall_seconds=300
agreement_reference= agreement_parts= agreement_items= agreement_full_logits=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) profile=${2-}; shift 2 ;;
        --instance) instance=${2-}; shift 2 ;;
        --devices) devices=${2-}; shift 2 ;;
        --checkpoint) checkpoint=${2-}; shift 2 ;;
        --cache-root) cache_root=${2-}; shift 2 ;;
        --allocated-context) allocated_context=${2-}; shift 2 ;;
        --mtp) mtp=${2-}; shift 2 ;;
        --long-chunks) long_chunks=1; shift ;;
        --prefill-slab) prefill_slab=${2-}; shift 2 ;;
        --no-sampling) sampling=; shift ;;
        --stall-seconds) stall_seconds=${2-}; shift 2 ;;
        --port) port=${2-}; shift 2 ;;
        --host) host=${2-}; shift 2 ;;
        --acceptance) acceptance=1; shift ;;
        --acceptance-prompts) acceptance_prompts=${2-}; shift 2 ;;
        --require-json-96) require_json_96=1; shift ;;
        --acceptance-only) acceptance_only=1; shift ;;
        --prepare-only) prepare_only=1; shift ;;
        --bf4-stage-limit) bf4_stage_limit=${2-}; shift 2 ;;
        --bf4-corpus) bf4_corpus=${2-}; shift 2 ;;
        --bf4-corpus-verification) bf4_corpus_verification=${2-}; shift 2 ;;
        --serve-seconds) serve_seconds=${2-}; shift 2 ;;
        --python) python=${2-}; shift 2 ;;
        --prefill-mode) prefill_mode=${2-}; shift 2 ;;
        --validate-only) validate_only=1; shift ;;
        --agreement-reference) agreement_reference=${2-}; shift 2 ;;
        --agreement-parts) agreement_parts=${2-}; shift 2 ;;
        --agreement-items) agreement_items=${2-}; shift 2 ;;
        --agreement-full-logits) agreement_full_logits=${2-}; shift 2 ;;
        -h|--help) usage ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done
[[ -n "$profile" && -n "$checkpoint" && -n "$cache_root" ]] || usage
[[ -d "$checkpoint" ]] || die "--checkpoint $checkpoint is not a directory"
[[ "$allocated_context" =~ ^(32768|65536|131072|262144)$ ]] || die "--allocated-context must be 32768, 65536, 131072 or 262144"
[[ "$instance" =~ ^[01]$ ]] || die "--instance must be 0 or 1"
[[ -z "$devices" || "$devices" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || die "--devices must be four device nodes, e.g. 4,5,6,7"
[[ -z "$bf4_stage_limit" || "$bf4_stage_limit" =~ ^[1-9][0-9]*$ ]] || die "--bf4-stage-limit must be a positive integer"
[[ "$stall_seconds" =~ ^[0-9]+$ ]] || die "--stall-seconds must be a whole number of seconds (0 disables the watchdog)"
[[ -z "$bf4_corpus" && -z "$bf4_corpus_verification" || -n "$bf4_corpus" && -n "$bf4_corpus_verification" ]] \
    || die "--bf4-corpus and --bf4-corpus-verification go together"
[[ -z "$acceptance" || -z "$acceptance_prompts" ]] || die "--acceptance and --acceptance-prompts are alternatives"
[[ -z "${TT_METAL_HOME:-}" || "$(cd -- "$TT_METAL_HOME" && pwd -P)" == "$REPO_ROOT" ]] \
    || die "TT_METAL_HOME=$TT_METAL_HOME is not this checkout ($REPO_ROOT); unset it or run that checkout's launcher"
for name in checkpoint cache_root acceptance_prompts bf4_corpus bf4_corpus_verification python agreement_reference agreement_full_logits; do
    [[ -z "${!name}" || "${!name}" == /* ]] || printf -v "$name" '%s/%s' "$PWD" "${!name}"
done
cd "$REPO_ROOT"  # from here on the real ttnn package comes first, not the model's own ttnn/
python=${python:-$REPO_ROOT/python_env/bin/python}
[[ -x "$python" ]] || die "no python at $python: build this checkout (build_metal.sh, create_venv.sh) or pass --python"

case "$profile" in
    tt-quietbox)
        [[ "$instance" == 0 && -z "$devices" ]] || die "--instance and --devices do not apply to --profile tt-quietbox"
        hardware_profile=tt-quietbox visible_devices=0,1,2,3
        descriptor="$HERE/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto" ;;
    p150-line)
        [[ "$instance" == 0 ]] || die "--instance applies to --profile qb2"
        hardware_profile=p150-line visible_devices=${devices:-0,1,2,3} descriptor= ;;
    qb2)
        [[ -z "$devices" ]] || die "--devices applies to --profile p150-line"
        printf 'run_qwen38_chat_server: the qb2 profile is UNTESTED (no QuietBox 2 was available); the first run prints the derived route to pin\n' >&2
        if [[ "$instance" == 0 ]]; then hardware_profile=tt-quietbox-2 visible_devices=0,1,2,3; else hardware_profile=tt-quietbox-2-instance-1 visible_devices=4,5,6,7; fi
        descriptor="$HERE/qb2_p300_1x4_line_mesh_graph_descriptor.textproto" ;;
    *) die "--profile must be tt-quietbox, p150-line or qb2" ;;
esac

# -- the checkout identity the server admits (printed here, proven there) -----------------------------------------
head=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null) || die "$REPO_ROOT is not a git checkout"
dirty=$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=no | wc -l | tr -d ' ')
extension=$("$python" -c 'import ttnn._ttnn as m; print(m.__file__)') || die "$python cannot import ttnn"
case "$(readlink -f -- "$extension")" in
    "$REPO_ROOT"/*) ;;
    *) die "$python imports ttnn from $extension, not from this checkout; build it or pass the checkout's --python" ;;
esac

# -- caches, run directory, environment ---------------------------------------------------------------------------------
label="c$allocated_context-$hardware_profile"
caches="$cache_root/caches/$label"
run_dir="$cache_root/runs/q38-chat-server-$hardware_profile-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$caches/components" "$cache_root/caches/bf4-experts" "$caches/model-io" "$cache_root/jit-cache" "$run_dir/tmp"

export TT_METAL_HOME="$REPO_ROOT"
export TT_VISIBLE_DEVICES="$visible_devices"
[[ -z "$descriptor" ]] || export TT_MESH_GRAPH_DESC_PATH="$descriptor"
export QWEN38_HARDWARE_MODE=diagnostic_non_promoting
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_CACHE="$cache_root/jit-cache"
export TT_METAL_LOGS_PATH="$run_dir/metal-logs"
export TMPDIR="$run_dir/tmp"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# the BF4 expert cache is keyed by the build identity inside; one root serves every allocated context
args=(
    --hardware-profile "$hardware_profile"
    --checkpoint "$checkpoint"
    --component-cache-root "$caches/components"
    --routed-bf4-scratch-root "$cache_root/caches/bf4-experts"
    --model-io-cache-root "$caches/model-io"
    --phase-log "$run_dir/phase-markers.jsonl"
    --evidence "$run_dir"
    --allocated-context "$allocated_context"
    --prefill-mode "$prefill_mode"
    --host "$host" --port "$port"
)
[[ -z "$devices" ]] || args+=(--device-nodes "$devices")
if [[ -n "$mtp" ]]; then
    [[ "$mtp" == 3 || "$mtp" == 4 || "$mtp" == 5 ]] || die "--mtp takes 3, 4 or 5, got $mtp"
    args+=(--mtp "$mtp")
fi
[[ -z "$long_chunks" ]] || args+=(--long-chunks)
if [[ -n "$prefill_slab" ]]; then
    [[ "$prefill_slab" =~ ^[0-9]+$ && $((prefill_slab % 128)) == 0 && "$prefill_slab" -ge 256 && "$prefill_slab" -le 4096 ]] \
        || die "--prefill-slab takes a multiple of 128 in 256..4096, got $prefill_slab"
    args+=(--prefill-slab "$prefill_slab")
fi
[[ -z "$sampling" ]] || args+=(--sampling)
[[ "$stall_seconds" == 0 ]] || args+=(--stall-seconds "$stall_seconds")
[[ -z "$acceptance" ]] || args+=(--acceptance-prompts "$HERE/acceptance/greedy-prompts")
[[ -z "$acceptance_prompts" ]] || args+=(--acceptance-prompts "$acceptance_prompts")
[[ -z "$acceptance_only" ]] || args+=(--acceptance-only)
[[ -z "$require_json_96" ]] || args+=(--require-json-96)
[[ -z "$prepare_only" ]] || args+=(--prepare-only)
[[ -z "$bf4_stage_limit" ]] || args+=(--bf4-stage-limit "$bf4_stage_limit")
[[ -z "$bf4_corpus" ]] || args+=(--bf4-corpus "$bf4_corpus" --bf4-corpus-verification "$bf4_corpus_verification")
[[ -z "$validate_only" ]] || args+=(--validate-only)
if [[ -n "$agreement_reference" ]]; then
    [[ -n "$sampling" ]] || die "--agreement-reference needs the sampled server (drop --no-sampling)"
    [[ -f "$agreement_reference" ]] || die "--agreement-reference $agreement_reference is not a file"
    args+=(--agreement-reference "$agreement_reference")
    if [[ -n "$agreement_parts" ]]; then read -r -a agreement_part_list <<<"$agreement_parts"; args+=(--agreement-parts "${agreement_part_list[@]}"); fi
    if [[ -n "$agreement_items" ]]; then read -r -a agreement_item_list <<<"$agreement_items"; args+=(--agreement-items "${agreement_item_list[@]}"); fi
    [[ -z "$agreement_full_logits" ]] || args+=(--agreement-full-logits "$agreement_full_logits")
else
    [[ -z "$agreement_parts$agreement_items$agreement_full_logits" ]] || die "--agreement-parts, --agreement-items and --agreement-full-logits need --agreement-reference"
fi

printf 'run_qwen38_chat_server: checkout %s (%s) head %s%s\n' "$REPO_ROOT" "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)" "$head" "$([[ "$dirty" == 0 ]] || printf ' (%s modified files)' "$dirty")" >&2
printf 'run_qwen38_chat_server: python %s, ttnn extension %s\n' "$python" "$extension" >&2
printf 'run_qwen38_chat_server: profile %s devices %s context %s run %s\n' "$hardware_profile" "$visible_devices" "$allocated_context" "$run_dir" >&2
[[ -z "${QWEN38_DENSE_WEIGHT_DTYPE:-}" ]] || printf 'run_qwen38_chat_server: QWEN38_DENSE_WEIGHT_DTYPE=%s (resident dense matmul weights; default bf8, bf16 = the previous production format)\n' "$QWEN38_DENSE_WEIGHT_DTYPE" >&2
if [[ -n "$serve_seconds" ]]; then
    exec timeout --signal=TERM "$serve_seconds" "$python" "$SERVER" "${args[@]}"
fi
exec "$python" "$SERVER" "${args[@]}"
