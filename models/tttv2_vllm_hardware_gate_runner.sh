#!/usr/bin/env bash
# Canonical TTTv2 vLLM hardware qualification runner.
#
# Examples:
#   ./tttv2_vllm_hardware_gate_runner.sh --tier smoke --expectations tttv2_llama3_8b_vllm_expectations_bh.json --artifact-root /tmp/smoke
#   ./tttv2_vllm_hardware_gate_runner.sh --tier benchmark --expectations tttv2_qwen3_32b_vllm_expectations_bh.json --artifact-root /tmp/bench --row p150x4_dp1_decode_only --row p150x4_dp1_all
#   ./tttv2_vllm_hardware_gate_runner.sh --tier quality --expectations tttv2_llama33_70b_vllm_expectations_bh.json --artifact-root /tmp/quality
#   ./tttv2_vllm_hardware_gate_runner.sh --tier quality --expectations tttv2_llama33_70b_vllm_expectations_bh.json --artifact-root /tmp/quality --validate-only

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="${VLLM_DIR:-/localdev/gwang/vllm_duo/vllm}"
PY="${PY:-/localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/python}"
VALIDATOR=""
RUNNER_PATH="$(realpath "$0")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8167}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-3600}"
RESET_TIMEOUT_SECONDS="${RESET_TIMEOUT_SECONDS:-120}"
RESET_KILL_AFTER_SECONDS="${RESET_KILL_AFTER_SECONDS:-10}"
TT_DEVICE_RECOVERY_MODE="${TT_DEVICE_RECOVERY_MODE:-reset}"
PROMPT=""

TIER=""
EXPECTATIONS=""
ROOT=""
DRY_RUN=0
VALIDATE_ONLY=0
declare -a REQUESTED_ROWS=()
declare -a SELECTED_ROWS=()
declare -a AGGREGATE_ROOTS=()
SERVER_PID=""
SERVER_PGID=""
CURRENT_CASE=""
JOURNAL_OPEN=0
RUN_JOURNAL_OPEN=0
HARDWARE_PREPARED=0

usage() {
    sed -n '2,9p' "$0"
    cat <<'EOF'

Required:
  --tier smoke|benchmark|quality
  --expectations FILE
  --artifact-root DIR

Selection and control:
  --row ROW_ID       Select an exact expectations row; repeat as needed.
                     Benchmark/quality default to all rows. Smoke always
                     uses smoke.row_id and rejects overrides.
  --dry-run          Resolve and print the commands; do not create files,
                     reset hardware, start a server, or validate artifacts.
                     Expectations schema/provenance preflight still runs.
  --validate-only    Validate an existing canonical benchmark/quality root.
  --aggregate-from DIR
                     Repeat to combine disjoint subset roots into one fresh,
                     canonical root and run final validation. No hardware.
  --help

Environment overrides: HOST, PORT, HEALTH_TIMEOUT_SECONDS,
RESET_TIMEOUT_SECONDS, and RESET_KILL_AFTER_SECONDS. Acceptance always uses
reset. VLLM_DIR and PY must match W0; the validator is expectations-pinned.

Quality runs create quality_review.json with each pair unaccepted. A human
must review every decode_only/all pair, fill accepted/reviewer/note, and rerun
with --validate-only. Thus the first quality run normally exits 2 after all
hardware rows pass their machine checks.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 64
}

while (($#)); do
    case "$1" in
        --tier) [[ $# -ge 2 ]] || die "--tier requires a value"; TIER="$2"; shift 2 ;;
        --expectations) [[ $# -ge 2 ]] || die "--expectations requires a value"; EXPECTATIONS="$2"; shift 2 ;;
        --artifact-root) [[ $# -ge 2 ]] || die "--artifact-root requires a value"; ROOT="$2"; shift 2 ;;
        --row) [[ $# -ge 2 ]] || die "--row requires a value"; REQUESTED_ROWS+=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --validate-only) VALIDATE_ONLY=1; shift ;;
        --aggregate-from) [[ $# -ge 2 ]] || die "--aggregate-from requires a value"; AGGREGATE_ROOTS+=("$2"); shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ "$TIER" == "smoke" || "$TIER" == "benchmark" || "$TIER" == "quality" ]] || die "--tier must be smoke, benchmark, or quality"
[[ -n "$EXPECTATIONS" && -f "$EXPECTATIONS" ]] || die "--expectations must name an existing file"
[[ -n "$ROOT" ]] || die "--artifact-root is required"
[[ "$TT_DEVICE_RECOVERY_MODE" == "reset" ]] || die "TT_DEVICE_RECOVERY_MODE must be reset; health-check-only runs are not acceptance evidence"
EXPECTATIONS="$(realpath "$EXPECTATIONS")"
ROOT="$(realpath -m "$ROOT")"
[[ -x "$PY" ]] || die "Python environment is not executable: $PY"
VALIDATOR="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["execution"]["validator"]["path"])
PY
)" || die "cannot read canonical validator path"
EXPECTED_VALIDATOR_SHA256="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["execution"]["validator"]["sha256"])
PY
)" || die "cannot read canonical validator hash"
[[ -f "$VALIDATOR" && "$(realpath "$VALIDATOR")" == "$VALIDATOR" ]] || die "canonical validator path is missing or unresolved: $VALIDATOR"
[[ "$(sha256sum "$VALIDATOR" | awk '{print $1}')" == "$EXPECTED_VALIDATOR_SHA256" ]] || die "canonical validator hash mismatch"
if ((DRY_RUN && VALIDATE_ONLY)); then
    die "--dry-run and --validate-only are mutually exclusive"
fi
if (( ${#AGGREGATE_ROOTS[@]} > 0 )) && ((DRY_RUN || VALIDATE_ONLY)); then
    die "--aggregate-from cannot be combined with --dry-run or --validate-only"
fi
if (( ${#AGGREGATE_ROOTS[@]} > 0 )) && [[ "$TIER" == "smoke" ]]; then
    die "smoke evidence is never aggregated"
fi
if (( ${#AGGREGATE_ROOTS[@]} > 0 )) && (( ${#REQUESTED_ROWS[@]} > 0 )); then
    die "--row cannot be combined with --aggregate-from"
fi
if ((VALIDATE_ONLY)); then
    [[ "$TIER" != "smoke" ]] || die "--validate-only is only for benchmark or quality"
    (( ${#REQUESTED_ROWS[@]} == 0 )) || die "--row cannot be combined with --validate-only; selection is pinned in run_contract.json"
    [[ -d "$ROOT" ]] || die "artifact root does not exist: $ROOT"
    "$PY" "$VALIDATOR" --expectations "$EXPECTATIONS" --check-expectations || die "expectations schema preflight failed"
    validation_extra=()
    validation_scope="$($PY - "$ROOT/run_contract.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["acceptance_scope"])
PY
    )" || die "cannot read validate-only acceptance scope"
    [[ "$validation_scope" != "subset" ]] || validation_extra+=(--subset-evidence)
    "$PY" "$VALIDATOR" --artifact-root "$ROOT" --expectations "$EXPECTATIONS" --tier "$TIER" "${validation_extra[@]}" 2>&1 | tee "$ROOT/validation_final.log"
    validation_statuses=("${PIPESTATUS[@]}")
    ((validation_statuses[1] == 0)) || die "cannot persist validate-only output"
    exit "${validation_statuses[0]}"
fi
if ((!DRY_RUN)) && [[ -e "$ROOT" ]]; then
    die "artifact root must be fresh and absent: $ROOT"
fi

selection_output="$("$PY" - "$EXPECTATIONS" "$TIER" "${REQUESTED_ROWS[@]}" 2>&1 <<'PY'
import json
import sys

path, tier, *requested = sys.argv[1:]
data = json.load(open(path))
rows = data.get("rows")
if not isinstance(rows, list) or not rows:
    raise SystemExit("expectations.rows must be a non-empty list")
if any(not isinstance(row, dict) for row in rows):
    raise SystemExit("every expectations row must be an object")
ids = [row.get("id") for row in rows]
if any(not isinstance(row_id, str) or not row_id for row_id in ids) or len(ids) != len(set(ids)):
    raise SystemExit("expectations row ids must be unique non-empty strings")
if data.get("schema_version") != 2 or data.get("canonical_row_ids") != ids:
    raise SystemExit("schema_version 2 and an exact immutable canonical_row_ids list are required")
smoke_row_id = data.get("smoke", {}).get("row_id")
if smoke_row_id not in ids:
    raise SystemExit("smoke.row_id must name a canonical row")
smoke_row = rows[ids.index(smoke_row_id)]
if smoke_row.get("manifest", {}).get("trace_mode") != "decode_only":
    raise SystemExit("smoke.row_id must select a decode_only row")
if requested:
    if len(requested) != len(set(requested)):
        raise SystemExit("duplicate --row selection")
    unknown = sorted(set(requested) - set(ids))
    if unknown:
        raise SystemExit(f"unknown row selection(s): {', '.join(unknown)}")
    selected = [row_id for row_id in ids if row_id in set(requested)]
elif tier == "smoke":
    selected = [smoke_row_id]
else:
    selected = ids
if tier == "smoke" and selected != [smoke_row_id]:
    raise SystemExit(f"smoke tier requires declared row {smoke_row_id}")
if tier == "quality":
    selected_pairs = {}
    available_pairs = {}
    by_id = {row["id"]: row for row in rows}
    for row in rows:
        manifest = row["manifest"]
        pair = (manifest["platform"], manifest["dp"])
        available_pairs.setdefault(pair, set()).add(manifest["trace_mode"])
    for row_id in selected:
        manifest = by_id[row_id]["manifest"]
        pair = (manifest["platform"], manifest["dp"])
        selected_pairs.setdefault(pair, set()).add(manifest["trace_mode"])
    incomplete = [
        f"{platform.lower()}_dp{dp}"
        for (platform, dp), modes in selected_pairs.items()
        if available_pairs[(platform, dp)] == {"decode_only", "all"}
        and modes != {"decode_only", "all"}
    ]
    if incomplete:
        raise SystemExit("quality selection requires complete decode_only/all pairs: " + ", ".join(incomplete))
print(*selected, sep="\n")
PY
)"
selection_exit=$?
((selection_exit == 0)) || die "$selection_output"
mapfile -t SELECTED_ROWS <<<"$selection_output"
(( ${#SELECTED_ROWS[@]} > 0 )) || die "row selection is empty"

row_json() {
    local row_id="$1"
    "$PY" - "$EXPECTATIONS" "$row_id" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
matches = [row for row in data["rows"] if row["id"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one row named {sys.argv[2]}")
print(json.dumps(matches[0], separators=(",", ":")))
PY
}

json_field() {
    local document="$1" expression="$2"
    "$PY" - "$document" "$expression" <<'PY'
import json
import sys
value = json.loads(sys.argv[1])
for key in sys.argv[2].split("."):
    value = value[key]
if value is None:
    print("")
elif isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, separators=(",", ":")))
else:
    print(value)
PY
}

EXPECTATIONS_ABS="$(realpath "$EXPECTATIONS")" || die "cannot resolve expectations path"
MODEL="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["model"])
PY
)" || die "cannot read model from expectations"
HF_HOME_PIN="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["hf_cache"]["hf_home"])
PY
)" || die "cannot read HF_HOME from expectations"
HF_SNAPSHOT="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["hf_cache"]["snapshot"])
PY
)" || die "cannot read pinned HF snapshot from expectations"
QUALITY_TOKENS="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["quality"]["quality_tokens"])
PY
)" || die "cannot read quality token budget"
SMOKE_TOKENS="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["smoke"]["tokens"])
PY
)" || die "cannot read smoke token budget"
PROMPT="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["execution"]["prompt"])
PY
)" || die "cannot read quality prompt"
CANONICAL_COUNT="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["canonical_row_ids"]))
PY
)" || die "cannot read canonical row count"
ACCEPTANCE_SCOPE=subset
if [[ "$TIER" == "smoke" ]]; then
    ACCEPTANCE_SCOPE=smoke
elif (( ${#SELECTED_ROWS[@]} == CANONICAL_COUNT )); then
    ACCEPTANCE_SCOPE=complete
fi

check_hf_cache_and_ref() {
"$PY" - "$EXPECTATIONS" "${SELECTED_ROWS[@]}" <<'PY'
import json
import pathlib
import sys
data = json.load(open(sys.argv[1]))
selected = set(sys.argv[2:])
cache = data.get("hf_cache", {})
try:
    hf_home = pathlib.Path(cache.get("hf_home", "")).resolve(strict=True)
    snapshot = pathlib.Path(cache.get("snapshot", "")).resolve(strict=True)
    ref_path = pathlib.Path(cache.get("ref_path", "")).resolve(strict=True)
except (OSError, RuntimeError) as error:
    raise SystemExit(f"cannot resolve HF cache paths: {error}")
if not snapshot.is_relative_to(hf_home) or not ref_path.is_relative_to(hf_home):
    raise SystemExit(f"HF snapshot/ref_path must resolve beneath hf_home={hf_home}")
if not snapshot.is_dir():
    raise SystemExit(f"pinned HF snapshot directory is missing: {snapshot}")
missing = [name for name in cache.get("verified_files", ()) if not (snapshot / name).is_file()]
if missing:
    raise SystemExit(f"pinned HF snapshot is incomplete; missing: {', '.join(missing)}")
revision = cache.get("revision")
if not revision or snapshot.name != revision:
    raise SystemExit(f"HF snapshot/revision mismatch: snapshot={snapshot.name!r}, revision={revision!r}")
ref_revision = cache.get("ref_revision", revision)
if not ref_revision:
    raise SystemExit("hf_cache.ref_revision must be a non-empty revision when provided")
for row in data["rows"]:
    if row["id"] not in selected:
        continue
    manifest = row["manifest"]
    if manifest.get("revision") != revision or manifest.get("tokenizer_revision") != revision:
        raise SystemExit(f"{row['id']}: model/tokenizer revision does not match hf_cache.revision")
try:
    ref_value = ref_path.read_text().strip()
except OSError as error:
    raise SystemExit(f"cannot read pinned HF ref {ref_path}: {error}")
if ref_value != ref_revision:
    raise SystemExit(
        f"HF {cache.get('ref')} moved: expected {ref_revision}, got {ref_value}"
    )
PY
}
check_hf_cache_and_ref || die "HF cache/revision/ref preflight failed"

EXPECTED_VLLM_DIR="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["execution"]["vllm_dir"])
PY
)" || die "cannot read pinned vLLM path"
EXPECTED_PY="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["execution"]["python"])
PY
)" || die "cannot read pinned Python path"
[[ "$VLLM_DIR" == "$EXPECTED_VLLM_DIR" ]] || die "VLLM_DIR differs from the W0-pinned execution path"
[[ "$PY" == "$EXPECTED_PY" ]] || die "PY differs from the W0-pinned execution path"
[[ -d "$VLLM_DIR/.git" ]] || die "vLLM checkout is missing: $VLLM_DIR"
EXPECTED_SERVER_SCRIPT="$($PY - "$EXPECTATIONS" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["execution"]["server_script"])
PY
)" || die "cannot read pinned server entrypoint"
if [[ "$EXPECTED_SERVER_SCRIPT" = /* ]]; then
    [[ -f "$EXPECTED_SERVER_SCRIPT" ]] || die "pinned server entrypoint is missing: $EXPECTED_SERVER_SCRIPT"
else
    [[ -f "$VLLM_DIR/$EXPECTED_SERVER_SCRIPT" ]] || die "pinned server entrypoint is missing beneath vLLM_DIR: $EXPECTED_SERVER_SCRIPT"
fi
TT_METAL_DIR="$(realpath "$SCRIPT_DIR/..")"
[[ -z "$(git -C "$TT_METAL_DIR" status --porcelain --untracked-files=no)" ]] || die "tt-metal has tracked changes; pause before hardware"
[[ -z "$(git -C "$VLLM_DIR" status --porcelain --untracked-files=no)" ]] || die "vLLM has tracked changes; zero-code-change qualification requires a clean checkout"
for required_command in curl timeout tt-smi ps awk sha256sum setsid flock git find; do
    command -v "$required_command" >/dev/null 2>&1 || die "required command is missing: $required_command"
done
"$PY" - "$VALIDATOR" <<'PY' || die "validator does not compile"
import pathlib, sys
path = pathlib.Path(sys.argv[1])
compile(path.read_text(), str(path), "exec")
PY
"$PY" "$VALIDATOR" --expectations "$EXPECTATIONS" --check-expectations || die "expectations schema preflight failed"

# Only these ambient values can cross the env -i boundary. Every other server
# and client value is constructed from the frozen expectations document.
BASE_ENV_JSON="$($PY - <<'PY'
import json, os
allowed = ("PATH", "HOME", "LD_LIBRARY_PATH", "TT_METAL_HOME")
env = {name: os.environ[name] for name in allowed if name in os.environ}
env["PYTHONNOUSERSITE"] = "1"
print(json.dumps(env, separators=(",", ":"), sort_keys=True))
PY
)" || die "cannot construct ambient environment allowlist"

preflight_cache_roots() {
    "$PY" - "$EXPECTATIONS" "${SELECTED_ROWS[@]}" <<'PY'
import json, os, pathlib, sys
data = json.load(open(sys.argv[1])); selected = set(sys.argv[2:])
for row in data["rows"]:
    if row["id"] not in selected:
        continue
    raw = row["manifest"].get("cache_root")
    if not isinstance(raw, str) or not pathlib.Path(raw).is_absolute():
        raise SystemExit(f"{row['id']}: cache_root must be absolute")
    path = pathlib.Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        probe = resolved / f".tttv2-write-probe-{os.getpid()}"
        fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd); probe.unlink()
    except OSError as error:
        raise SystemExit(f"{row['id']}: cache_root is not resolvable/writable: {error}")
    if str(resolved) != raw:
        raise SystemExit(f"{row['id']}: cache_root must already be resolved: declared={raw}, resolved={resolved}")
PY
}
if ((!DRY_RUN)) && ((!VALIDATE_ONLY)) && (( ${#AGGREGATE_ROOTS[@]} == 0 )); then
    preflight_cache_roots || die "cache-root preflight failed"
fi

write_run_contract() {
    local destination="$1" scope="$2" host="$3" port="$4"
    "$PY" - "$EXPECTATIONS" "$destination" "$TIER" "$scope" "$host" "$port" "$TT_DEVICE_RECOVERY_MODE" "$RUNNER_PATH" "$VALIDATOR" "$VLLM_DIR" "$BASE_ENV_JSON" "$ROOT" "${SELECTED_ROWS[@]}" <<'PY'
import json
import pathlib
import subprocess
import sys
import hashlib
source, destination, tier, scope, host, port, recovery_mode, runner, validator, vllm_dir, base_env_raw, root_raw, *selected = sys.argv[1:]
data = json.load(open(source))
def digest(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def repository(path):
    root = subprocess.check_output(["git", "-C", path, "rev-parse", "--show-toplevel"], text=True).strip()
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], text=True)
    return {"path": str(pathlib.Path(root).resolve()), "head": head, "dirty": bool(status), "tracked_status": status.splitlines()}
root = pathlib.Path(root_raw)
repositories = {"tt_metal": repository(str(pathlib.Path(runner).parent.parent)), "vllm": repository(vllm_dir)}
record = {
    "schema_version": 2,
    "tier": tier,
    "model_id": data["model_id"],
    "architecture": data["architecture"],
    "generator": data["generator"],
    "canonical_expectations_sha256": hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest(),
    "canonical_row_ids": data["canonical_row_ids"],
    "selected_row_ids": selected,
    "acceptance_scope": scope,
    "host": host,
    "port": int(port),
    "tt_device_recovery_mode": recovery_mode,
    "base_env": json.loads(base_env_raw),
    "tested_code_sha": repositories["tt_metal"]["head"],
    "vllm_sha": repositories["vllm"]["head"],
    "repositories": repositories,
    "tools": {"runner": {"path": runner, "sha256": digest(runner)}, "validator": {"path": validator, "sha256": digest(validator)}},
    "inputs": data["provenance"],
    "resolved_cache_roots": {row["id"]: str(pathlib.Path(row["manifest"]["cache_root"]).resolve(strict=True)) for row in data["rows"] if row["id"] in selected},
    "launch_sha256_by_row": {row_id: digest(root / row_id / "launch.json") for row_id in selected},
}
target = pathlib.Path(destination)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2) + "\n")
temporary.replace(target)
PY
}

list_vllm_processes() {
    ps -eo pid=,pgid=,stat=,args= | awk '
        /plugins\/vllm-tt-plugin\/examples\/server_example_tt.py|vllm\.entrypoints\.cli\.main (bench )?serve|(^|[[:space:]])vllm serve([[:space:]]|$)|vllm\.entrypoints\.openai\.api_server|EngineCore|python.*multiprocessing\.(spawn|resource_tracker)|ray::.*[vV][lL][lL][mM]|[vV][lL][lL][mM].*[wW]orker|[wW]orker.*[vV][lL][lL][mM]/ &&
        $0 !~ /awk/ {print}
    '
}

process_group_exists() {
    local pgid="$1"
    ps -e -o pgid=,stat= | awk -v wanted="$pgid" '$1 == wanted && $2 !~ /^Z/ {found=1} END {exit !found}'
}

cleanup_server() {
    local pid="${1:-}" pgid="${2:-}" evidence="${3:-/dev/null}" attempt
    [[ -n "$pid" && -n "$pgid" ]] || return 0
    {
        printf 'pid=%s pgid=%s\n' "$pid" "$pgid"
        ps -e -o pid=,ppid=,pgid=,stat=,args= | awk -v wanted="$pgid" '$3 == wanted'
    } >>"$evidence" 2>&1
    if process_group_exists "$pgid"; then
        kill -INT -- "-$pgid" >>"$evidence" 2>&1 || true
        for attempt in $(seq 1 60); do
            process_group_exists "$pgid" || break
            sleep 1
        done
    fi
    if process_group_exists "$pgid"; then
        kill -TERM -- "-$pgid" >>"$evidence" 2>&1 || true
        sleep 5
    fi
    if process_group_exists "$pgid"; then
        kill -KILL -- "-$pgid" >>"$evidence" 2>&1 || true
        sleep 2
    fi
    if process_group_exists "$pgid"; then
        printf 'cleanup_status=failed\n' >>"$evidence"
        return 1
    fi
    printf 'cleanup_status=ok\n' >>"$evidence"
    SERVER_PID=""
    SERVER_PGID=""
    return 0
}

cleanup_on_exit() {
    if [[ -n "$SERVER_PID" && -n "$SERVER_PGID" ]]; then
        cleanup_server "$SERVER_PID" "$SERVER_PGID" "${CURRENT_CASE:-$ROOT}/trap_cleanup.log" || true
    fi
    if ((HARDWARE_PREPARED)); then
        reset_tt "${CURRENT_CASE:-$ROOT}/trap_reset_after.log" || true
        HARDWARE_PREPARED=0
    fi
    if ((JOURNAL_OPEN)) && [[ -n "$CURRENT_CASE" && -n "$ROOT" && -f "$ROOT/attempt_journal.jsonl" ]]; then
        "$PY" - "$ROOT/attempt_journal.jsonl" "$(basename "$CURRENT_CASE")" "$TIER" <<'PY' || true
import datetime, json, pathlib, sys
record = {"event": "row_aborted", "row_id": sys.argv[2], "tier": sys.argv[3], "status": "aborted", "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with pathlib.Path(sys.argv[1]).open("a") as out: out.write(json.dumps(record, sort_keys=True) + "\n")
PY
        JOURNAL_OPEN=0
    fi
    if ((RUN_JOURNAL_OPEN)) && [[ -n "$ROOT" && -f "$ROOT/attempt_journal.jsonl" ]]; then
        "$PY" - "$ROOT/attempt_journal.jsonl" "$TIER" <<'PY' || true
import datetime, json, pathlib, sys
record = {"event": "run_aborted", "tier": sys.argv[2], "status": "aborted", "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with pathlib.Path(sys.argv[1]).open("a") as out: out.write(json.dumps(record, sort_keys=True) + "\n")
PY
        RUN_JOURNAL_OPEN=0
    fi
}
trap cleanup_on_exit EXIT
trap 'cleanup_on_exit; exit 130' INT
trap 'cleanup_on_exit; exit 143' TERM

reset_tt() {
    local log="$1" attempt exit_code=1
    : >"$log"
    printf 'tt_device_recovery_mode=%s\n' "$TT_DEVICE_RECOVERY_MODE" >>"$log"
    for attempt in 1 2; do
        timeout --signal=TERM --kill-after="${RESET_KILL_AFTER_SECONDS}s" "${RESET_TIMEOUT_SECONDS}s" tt-smi -r >>"$log" 2>&1
        exit_code=$?
        ((exit_code == 0)) && return 0
        ((attempt == 1)) && sleep 10
    done
    return "$exit_code"
}

wait_for_health() {
    local log="$1" elapsed=0 state
    while ((elapsed < HEALTH_TIMEOUT_SECONDS)); do
        if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        state="$(ps -o stat= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
        if [[ -z "$state" || "$state" == Z* ]]; then
            printf 'server exited before health\n' >>"$log"
            return 1
        fi
        sleep 1
        ((elapsed += 1))
    done
    printf 'health timeout after %s seconds\n' "$HEALTH_TIMEOUT_SECONDS" >>"$log"
    return 1
}

write_manifest() {
    local row="$1" destination="$2" status="$3" error_hits="$4"
    "$PY" - "$row" "$destination" "$TIER" "$status" "$error_hits" <<'PY'
import json
import pathlib
import sys
row, destination, tier, status, error_hits = sys.argv[1:]
record = dict(json.loads(row)["manifest"])
record.update({"tier": tier, "row_id": json.loads(row)["id"], "status": status, "error_hits": int(error_hits)})
target = pathlib.Path(destination)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2) + "\n")
temporary.replace(target)
PY
}

record_exit() {
    local destination="$1" code="$2"
    printf '%s\n' "$code" >"$destination" || die "cannot write lifecycle evidence: $destination"
}

scan_error_hits() {
    local case_dir="$1" context_log
    local -a logs=("$case_dir/server.log" "$case_dir/client.log")
    while IFS= read -r -d '' context_log; do
        logs+=("$context_log")
    done < <(find "$case_dir/context_subcases" -type f -name client.log -print0 2>/dev/null)
    "$PY" - "${logs[@]}" <<'PY'
import pathlib
import re
import sys
patterns = (
    re.compile(r"\bERROR\b", re.I), re.compile(r"\bCRITICAL\b", re.I),
    re.compile(r"Traceback"), re.compile(r"RuntimeError"), re.compile(r"EngineDead", re.I),
    re.compile(r"index_cpu", re.I), re.compile(r"sampled[- ]token[^\n]*(?:shape|dtype)", re.I),
    re.compile(r"invalid[- ]token", re.I), re.compile(r"SIGKILL", re.I),
    re.compile(r"\bOOM\b", re.I), re.compile(r"OutOfMemory", re.I), re.compile(r"\bKilled\b", re.I),
)
hits = []
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    text = path.read_text(errors="replace") if path.exists() else ""
    for line in text.splitlines():
        if "| warning  |" in line and "hard error in a future release" in line:
            continue
        if (
            " WARNING " in line
            and "Encountered invalid prefix detokenization error" in line
            and "resetting decode stream" in line
        ):
            continue
        for pattern in patterns:
            for match in pattern.finditer(line):
                hits.append(f"{path.name}:{pattern.pattern}:{match.group(0)}")
failure_scan = pathlib.Path(sys.argv[1]).parent / "failure_scan.log"
failure_scan.write_text("\n".join(hits) + ("\n" if hits else ""))
print(len(hits))
PY
}

write_launch() {
    local row="$1" case_dir="$2" destination
    destination="$case_dir/launch.json"
    [[ "$case_dir" == "-" ]] && destination="-"
    "$PY" - "$EXPECTATIONS" "$row" "$destination" "$case_dir" "$TIER" "$HOST" "$PORT" "$BASE_ENV_JSON" <<'PY'
import json
import pathlib
import sys
expectations_path, row_raw, destination, case_dir_raw, tier, host, port, base_env_raw = sys.argv[1:]
data = json.load(open(expectations_path))
row = json.loads(row_raw)
manifest = row["manifest"]
execution = data["execution"]
base_env = json.loads(base_env_raw)
tt = {"trace_mode": manifest["trace_mode"], "trace_region_size": manifest["trace_region_size"], "sample_on_device_mode": manifest["sample_on_device_mode"]}
if manifest.get("fabric_config") is not None:
    tt["fabric_config"] = manifest["fabric_config"]
server_argv = [
    execution["python"], execution["server_script"], "--model", manifest["model"],
    "--revision", manifest["revision"], "--tokenizer-revision", manifest["tokenizer_revision"],
    "--host", host, "--port", port, "--max-model-len", str(manifest["max_model_len"]),
    "--data-parallel-size", str(manifest["dp"]), "--max_num_seqs", str(manifest["max_num_seqs_per_rank"]),
    "--additional-config", json.dumps({"tt": tt}, separators=(",", ":")),
]
server_argv.append("--async-scheduling" if manifest["async_scheduling"] else "--no-async-scheduling")
server_argv.append("--enable-prefix-caching" if manifest["prefix_caching"] else "--no-enable-prefix-caching")
server_env = dict(base_env)
server_env.update({
    "MESH_DEVICE": manifest["platform"], "HF_MODEL": manifest["model"], manifest["family_var"]: manifest["family_version"],
    "HF_HOME": data["hf_cache"]["hf_home"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "TT_VISIBLE_DEVICES": ",".join(str(value) for value in manifest["visible_devices"]),
})
if manifest.get("cache_root"):
    server_env["TT_CACHE_PATH"] = manifest["cache_root"]
record = {"schema_version": 1, "server": {"kind": "process", "cwd": execution["vllm_dir"], "argv": server_argv, "env": server_env}}
if tier == "benchmark":
    common = data["common"]
    client_argv = [
        execution["python"], "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", common["backend"], "--endpoint", common["endpoint"], "--model", data["model"],
        "--tokenizer", data["hf_cache"]["snapshot"],
        "--host", host, "--port", port, "--dataset-name", "random", "--random-input-len", str(common["input_tokens"]),
        "--random-output-len", str(common["output_tokens"]), "--num-prompts", str(common["num_prompts"]),
        "--max-concurrency", str(common["max_concurrency"]), "--request-rate", str(common["request_rate"]),
        "--temperature", str(common["temperature"]), "--ignore-eos", "--save-result", "--save-detailed",
        "--result-filename", str(pathlib.Path(case_dir_raw) / "result.json"),
    ]
    client_env = dict(base_env)
    client_env.update({"HF_HOME": data["hf_cache"]["hf_home"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    record["client"] = {"kind": "process", "cwd": execution["vllm_dir"], "argv": client_argv, "env": client_env}
    context_program = r'''import json,pathlib,sys,urllib.request
spec=json.loads(sys.argv[1]); host=sys.argv[2]; port=sys.argv[3]; model=sys.argv[4]; output=pathlib.Path(sys.argv[5])
def call(prompt):
    payload={"model":model,"prompt":prompt,"max_tokens":spec["output_tokens"],"temperature":0,"ignore_eos":True}
    request=urllib.request.Request(f"http://{host}:{port}/v1/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(request,timeout=1800) as response:return {"request":payload,"response":json.loads(response.read())}
count=spec["input_tokens"]
if spec["kind"]=="cached_prefill":
    common=spec["common_prefix_tokens"]; first=[42]*common+[43]*(count-common); second=[42]*common+[44]*(count-common)
    calls=[call(first),call(second)]
else:calls=[call([42]*count)]
output.write_text(json.dumps({"schema_version":1,"subcase":spec,"calls":calls},indent=2)+"\n")'''
    context_clients = {}
    for subcase in manifest["context_subcases"]:
        kind = subcase["kind"]
        result_path = pathlib.Path(case_dir_raw) / "context_subcases" / kind / "result.json"
        context_clients[kind] = {
            "kind": "process", "cwd": execution["vllm_dir"], "env": client_env,
            "argv": [execution["python"], "-c", context_program, json.dumps(subcase, separators=(",", ":")), host, port, manifest["model"], str(result_path)],
        }
    record["context_clients"] = context_clients
else:
    record["client"] = {"kind": "http", "method": "POST", "url": f"http://{host}:{port}/v1/completions", "request_file": "request.json", "response_file": "response.json"}
if destination == "-":
    print(json.dumps(record, indent=2))
    raise SystemExit(0)
target = pathlib.Path(destination)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2) + "\n")
temporary.replace(target)
PY
}

exec_process_spec() {
    local launch_file="$1" member="$2"
    exec "$PY" - "$launch_file" "$member" <<'PY'
import json, os, sys
launch_file, member = sys.argv[1:]
spec = json.load(open(launch_file))
for component in member.split("."):
    spec = spec[component]
if spec.get("kind") != "process":
    raise SystemExit(f"{member} is not a ProcessSpec")
os.chdir(spec["cwd"])
if member == "server":
    os.setsid()
os.execvpe(spec["argv"][0], spec["argv"], spec["env"])
PY
}

capture_live_process() {
    local pid="$1" destination="$2" launch_file="$3" member="${4:-server}"
    "$PY" - "$pid" "$destination" "$launch_file" "$member" <<'PY'
import json, os, pathlib, sys, time
pid = int(sys.argv[1]); destination = pathlib.Path(sys.argv[2]); launch = json.load(open(sys.argv[3])); member=sys.argv[4]
for component in member.split("."): launch=launch[component]
proc = pathlib.Path("/proc") / str(pid)
for _ in range(500):
    try:
        argv = [part.decode(errors="surrogateescape") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        if argv == launch["argv"]:
            break
    except OSError:
        pass
    time.sleep(0.01)
else:
    raise SystemExit("launched process never exposed the frozen argv in /proc")
env_parts = (proc / "environ").read_bytes().split(b"\0")
env = {}
for part in env_parts:
    if not part: continue
    key, value = part.split(b"=", 1)
    env[key.decode(errors="surrogateescape")] = value.decode(errors="surrogateescape")
record = {
    "pid": pid,
    "pgid": os.getpgid(pid),
    "cwd": os.readlink(proc / "cwd"),
    "executable": os.readlink(proc / "exe"),
    "argv": argv,
    "env": env,
}
if record["cwd"] != launch["cwd"] or record["argv"] != launch["argv"] or record["env"] != launch["env"]:
    raise SystemExit("live process differs from immutable server ProcessSpec")
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
PY
}

run_process_spec_with_proof() {
    local launch_file="$1" member="$2" proof="$3" log="$4" pid exit_code
    (exec_process_spec "$launch_file" "$member") >"$log" 2>&1 &
    pid=$!
    capture_live_process "$pid" "$proof" "$launch_file" "$member" || {
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 97
    }
    wait "$pid"
    exit_code=$?
    return "$exit_code"
}

write_quality_review() {
    local destination="$1"
    "$PY" - "$EXPECTATIONS" "$ROOT" "$destination" <<'PY'
import hashlib
import json
import pathlib
import sys
expectations_path, root_raw, destination = sys.argv[1:]
data = json.load(open(expectations_path))
root = pathlib.Path(root_raw)
pairs = {}
for row in data["rows"]:
    if not (root / row["id"]).is_dir():
        continue
    manifest = row["manifest"]
    pair = f"{manifest['platform'].lower()}_dp{manifest['dp']}"
    record = pairs.setdefault(pair, {"rows": {}, "response_sha256": {}, "accepted": False, "reviewer": "", "note": ""})
    mode = manifest["trace_mode"]
    record["rows"][mode] = row["id"]
    response = root / row["id"] / "response.json"
    if response.is_file():
        record["response_sha256"][mode] = hashlib.sha256(response.read_bytes()).hexdigest()
pairs = {pair: record for pair, record in pairs.items() if set(record["rows"]) == {"decode_only", "all"}}
target = pathlib.Path(destination)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps({"pairs": pairs}, indent=2) + "\n")
temporary.replace(target)
PY
}

run_final_validation() {
    local log="$ROOT/validation_final.log" validator_exit tee_exit non_review_failures
    "$PY" "$VALIDATOR" --artifact-root "$ROOT" --expectations "$EXPECTATIONS" --tier "$TIER" 2>&1 | tee "$log"
    validator_statuses=("${PIPESTATUS[@]}")
    validator_exit=${validator_statuses[0]}
    tee_exit=${validator_statuses[1]}
    ((tee_exit == 0)) || die "cannot persist final validator output"
    ((validator_exit == 0)) && return 0
    if [[ "$TIER" == "quality" ]]; then
        non_review_failures="$(awk '
            /^FAIL: quality pair .*: accepted: expected True, got False$/ {next}
            /^FAIL: quality pair .*: reviewer must be a non-empty string$/ {next}
            /^FAIL: quality pair .*: note must be a non-empty string$/ {next}
            /^FAILED: [0-9]+ violation\(s\)$/ {next}
            /^FAIL:/ {print}
        ' "$log")" || die "cannot inspect quality validator output"
        if [[ -z "$non_review_failures" ]]; then
            printf 'PENDING: complete %s/quality_review.json, then rerun with --validate-only\n' "$ROOT" >&2
            return 2
        fi
    fi
    return "$validator_exit"
}

if (( ${#AGGREGATE_ROOTS[@]} > 0 )); then
    mkdir -p "$(dirname "$ROOT")" || die "cannot create artifact parent"
    mkdir "$ROOT" || die "cannot create fresh aggregate root"
    "$PY" - "$EXPECTATIONS" "$ROOT" "$TIER" "${AGGREGATE_ROOTS[@]}" <<'PY' || die "aggregation failed"
import hashlib
import json
import pathlib
import shutil
import sys
expectations_path, destination_raw, tier, *source_roots = sys.argv[1:]
expectations_file = pathlib.Path(expectations_path)
expectations = json.loads(expectations_file.read_text())
destination = pathlib.Path(destination_raw)
expected_hash = hashlib.sha256(expectations_file.read_bytes()).hexdigest()
canonical = expectations["canonical_row_ids"]
seen = {}
host_port = None
shared_provenance = None
summary_lines = []
journal_by_row = {}
launch_hash_by_row = {}
for source_raw in source_roots:
    source = pathlib.Path(source_raw).resolve()
    contract = json.loads((source / "run_contract.json").read_text())
    if contract.get("tier") != tier or contract.get("canonical_expectations_sha256") != expected_hash:
        raise SystemExit(f"incompatible aggregate source: {source}")
    if contract.get("acceptance_scope") not in ("subset", "complete"):
        raise SystemExit(f"source is not matrix evidence: {source}")
    if contract.get("schema_version") != 2 or contract.get("tt_device_recovery_mode") != "reset":
        raise SystemExit(f"source lacks schema-v2 reset-only contract: {source}")
    provenance = {key: contract.get(key) for key in ("repositories", "tools", "inputs", "base_env")}
    if shared_provenance is None:
        shared_provenance = provenance
    elif provenance != shared_provenance:
        raise SystemExit("aggregate sources have different repository/tool/input/environment provenance")
    process_check = source / "process_check_after.log"
    if not process_check.is_file() or process_check.read_text().strip():
        raise SystemExit(f"source has missing/nonempty final process inventory: {source}")
    current_host_port = (contract.get("host"), contract.get("port"))
    if host_port is None:
        host_port = current_host_port
    elif current_host_port != host_port:
        raise SystemExit("aggregate sources used different host/port launch contracts")
    for row_id in contract.get("selected_row_ids", []):
        if row_id in seen:
            raise SystemExit(f"duplicate row {row_id} in {source} and {seen[row_id]}")
        row_dir = source / row_id
        if not row_dir.is_dir():
            raise SystemExit(f"source contract row directory missing: {row_dir}")
        expected_launch_hash = contract.get("launch_sha256_by_row", {}).get(row_id)
        actual_launch_hash = hashlib.sha256((row_dir / "launch.json").read_bytes()).hexdigest()
        if expected_launch_hash != actual_launch_hash:
            raise SystemExit(f"source launch hash mismatch for {row_id} in {source}")
        shutil.copytree(row_dir, destination / row_id)
        seen[row_id] = str(source)
        launch_hash_by_row[row_id] = expected_launch_hash
    summary = source / "summary.jsonl"
    if summary.is_file():
        summary_lines.extend(summary.read_text().splitlines())
    journal = source / "attempt_journal.jsonl"
    if not journal.is_file():
        raise SystemExit(f"source attempt journal missing: {source}")
    for line in journal.read_text().splitlines():
        if not line.strip(): continue
        entry = json.loads(line)
        if entry.get("row_id") in contract.get("selected_row_ids", []):
            journal_by_row.setdefault(entry["row_id"], []).append(line)
if set(seen) != set(canonical) or len(seen) != len(canonical):
    missing = [row for row in canonical if row not in seen]
    extra = [row for row in seen if row not in canonical]
    raise SystemExit(f"aggregate is not canonical; missing={missing}, extra={extra}")
contract = {
    "schema_version": 2, "tier": tier, "model_id": expectations["model_id"],
    "architecture": expectations["architecture"], "generator": expectations["generator"],
    "canonical_expectations_sha256": expected_hash, "canonical_row_ids": canonical,
    "selected_row_ids": canonical, "acceptance_scope": "complete",
    "host": host_port[0], "port": host_port[1], "aggregate_sources": [str(pathlib.Path(p).resolve()) for p in source_roots],
    "row_sources": seen,
    "tt_device_recovery_mode": "reset",
    **shared_provenance,
    "tested_code_sha": shared_provenance["repositories"]["tt_metal"]["head"],
    "vllm_sha": shared_provenance["repositories"]["vllm"]["head"],
    "launch_sha256_by_row": {row_id: launch_hash_by_row[row_id] for row_id in canonical},
    "resolved_cache_roots": {row_id: json.loads((destination / row_id / "manifest.json").read_text())["cache_root"] for row_id in canonical},
}
(destination / "run_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
(destination / "canonical_expectations.json").write_bytes(expectations_file.read_bytes())
(destination / "summary.jsonl").write_text("\n".join(summary_lines) + ("\n" if summary_lines else ""))
(destination / "attempt_journal.jsonl").write_text("\n".join(line for row_id in canonical for line in journal_by_row.get(row_id, [])) + "\n")
(destination / "process_check_after.log").write_text("")
PY
    if [[ "$TIER" == "quality" ]]; then
        write_quality_review "$ROOT/quality_review.json" || die "cannot create aggregate quality review"
    fi
    run_final_validation
    exit $?
fi

if ((DRY_RUN)); then
    printf 'DRY RUN tier=%s model=%s expectations=%s artifact_root=%s\n' "$TIER" "$MODEL" "$EXPECTATIONS_ABS" "$ROOT"
    for row_id in "${SELECTED_ROWS[@]}"; do
        dry_row="$(row_json "$row_id")" || die "$row_id: cannot load dry-run row"
        printf 'ROW %s immutable LaunchSpec:\n' "$row_id"
        write_launch "$dry_row" "-" || die "$row_id: cannot render immutable LaunchSpec"
    done
    exit 0
fi

mkdir -p "$(dirname "$ROOT")" || die "cannot create artifact parent"
mkdir "$ROOT" || die "cannot create fresh artifact root"
cp "$EXPECTATIONS" "$ROOT/canonical_expectations.json" || die "cannot persist canonical expectations"
printf '%s\n' "$EXPECTATIONS_ABS" >"$ROOT/source_expectations.path" || die "cannot persist expectations source path"
: >"$ROOT/run.log" || die "cannot initialize run.log"
: >"$ROOT/summary.jsonl" || die "cannot initialize summary.jsonl"
: >"$ROOT/attempt_journal.jsonl" || die "cannot initialize attempt journal"
"$PY" - "$ROOT/attempt_journal.jsonl" "$TIER" <<'PY' || die "cannot journal run start"
import datetime, json, pathlib, sys
record = {"event": "run_started", "tier": sys.argv[2], "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with pathlib.Path(sys.argv[1]).open("a") as out: out.write(json.dumps(record, sort_keys=True) + "\n")
PY
RUN_JOURNAL_OPEN=1
for row_id in "${SELECTED_ROWS[@]}"; do
    setup_row="$(row_json "$row_id")" || die "$row_id: cannot load row for LaunchSpec"
    mkdir "$ROOT/$row_id" || die "cannot create row directory: $ROOT/$row_id"
    write_manifest "$setup_row" "$ROOT/$row_id/manifest.json" pending 0 || die "cannot write pending manifest"
    write_launch "$setup_row" "$ROOT/$row_id" || die "cannot freeze LaunchSpec"
done
write_run_contract "$ROOT/run_contract.json" "$ACCEPTANCE_SCOPE" "$HOST" "$PORT" || die "cannot persist run contract"

# Hold the same host-wide lock as TT pytest fixtures for the complete
# reset/server/client/cleanup interval. It is intentionally non-reentrant.
exec 9>"${TT_DEVICE_LOCK_PATH:-/tmp/tt_device.lock}" || die "cannot open TT device lock"
flock -w "${TT_DEVICE_LOCK_TIMEOUT:-600}" 9 || die "cannot acquire host-wide TT device lock"

write_quality_request() {
    local destination="$1" max_tokens="$2"
    "$PY" - "$destination" "$MODEL" "$PROMPT" "$max_tokens" <<'PY'
import json
import pathlib
import sys
destination, model, prompt, max_tokens = sys.argv[1:]
payload = {"model": model, "prompt": prompt, "max_tokens": int(max_tokens), "temperature": 0, "ignore_eos": True}
pathlib.Path(destination).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

execute_http_client_spec() {
    local launch_json="$1"
    "$PY" - "$launch_json" <<'PY'
import json
import pathlib
import sys
import urllib.request
launch_path = pathlib.Path(sys.argv[1])
spec = json.loads(launch_path.read_text())["client"]
if set(spec) != {"kind", "method", "url", "request_file", "response_file"}:
    raise SystemExit("HTTP client spec has unknown or missing fields")
if spec["kind"] != "http" or spec["method"] != "POST":
    raise SystemExit("HTTP client spec must be a POST")
request_path = launch_path.parent / spec["request_file"]
response_path = launch_path.parent / spec["response_file"]
request = urllib.request.Request(
    spec["url"],
    data=pathlib.Path(request_path).read_bytes(),
    headers={"Content-Type": "application/json"},
    method=spec["method"],
)
with urllib.request.urlopen(request, timeout=900) as response:
    body = json.loads(response.read().decode("utf-8"))
pathlib.Path(response_path).write_text(json.dumps(body, indent=2) + "\n")
choice = body.get("choices", [{}])[0]
print(f"finish_reason={choice.get('finish_reason')}")
print(f"completion_tokens={body.get('usage', {}).get('completion_tokens')}")
print("TEXT_BEGIN")
print(choice.get("text", ""))
print("TEXT_END")
PY
}

validate_completion_response() {
    local response_json="$1" token_budget="$2"
    "$PY" - "$EXPECTATIONS" "$response_json" "$token_budget" "$TIER" <<'PY'
import json
import sys
expectations = json.load(open(sys.argv[1]))
response = json.load(open(sys.argv[2]))
budget = int(sys.argv[3])
tier = sys.argv[4]
choice = response["choices"][0]
text = choice["text"]
if response.get("object") != "text_completion":
    raise SystemExit(f"response object is not text_completion: {response.get('object')!r}")
if not isinstance(text, str) or not text.strip():
    raise SystemExit("completion text is empty")
if choice.get("finish_reason") != "length":
    raise SystemExit(f"finish_reason is not length: {choice.get('finish_reason')!r}")
if response.get("usage", {}).get("completion_tokens") != budget:
    raise SystemExit(f"completion token count is not {budget}")
lowered = text.lower()
semantic_groups = expectations.get(tier, {}).get(
    "semantic_term_groups",
    expectations.get("quality", {}).get("semantic_term_groups", ()),
)
for index, alternatives in enumerate(semantic_groups):
    if not any(term.lower() in lowered for term in alternatives):
        raise SystemExit(f"semantic term group {index} absent: {alternatives!r}")
PY
}

run_context_subcases() {
    local case_dir="$1" manifest="$2" kind start end exit_code
    mapfile -t context_kinds < <("$PY" - "$manifest" <<'PY'
import json, sys
for item in json.loads(sys.argv[1])["context_subcases"]: print(item["kind"])
PY
    ) || return 97
    for kind in "${context_kinds[@]}"; do
        mkdir -p "$case_dir/context_subcases/$kind" || return 97
        start="$(stat -c %s "$case_dir/server.log")" || return 97
        run_process_spec_with_proof "$case_dir/launch.json" "context_clients.$kind" \
            "$case_dir/context_subcases/$kind/live_process.json" "$case_dir/context_subcases/$kind/client.log"
        exit_code=$?
        record_exit "$case_dir/context_subcases/$kind/client.exit" "$exit_code"
        end="$(stat -c %s "$case_dir/server.log")" || return 97
        "$PY" - "$case_dir" "$kind" "$start" "$end" <<'PY' || return 97
import hashlib, json, pathlib, re, sys
case=pathlib.Path(sys.argv[1]); kind=sys.argv[2]; start=int(sys.argv[3]); end=int(sys.argv[4])
segment=(case/"server.log").read_bytes()[start:end]
result_path=case/"context_subcases"/kind/"result.json"
result=json.loads(result_path.read_text()) if result_path.exists() else None
counts={
 "chunk_events":len(re.findall(rb"(?:chunked[^\n]*prefill|prefill[^\n]*chunk)",segment,re.I)),
 "cache_hits":len(re.findall(rb"(?:cache[^\n]*hit|prefix[^\n]*cache[^\n]*hit)",segment,re.I)),
}
evidence={"schema_version":1,"server_log_start":start,"server_log_end":end,"server_segment_sha256":hashlib.sha256(segment).hexdigest(),"instrumentation_counts":counts,"result":result}
(case/"context_subcases"/kind/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
PY
        ((exit_code == 0)) || return "$exit_code"
    done
}

run_client() {
    local case_dir="$1"
    case "$TIER" in
        benchmark)
            local manifest
            manifest="$($PY - "$case_dir/manifest.json" <<'PY'
import json,sys
print(json.dumps(json.load(open(sys.argv[1])),separators=(",",":")))
PY
            )" || return 97
            run_context_subcases "$case_dir" "$manifest" || return $?
            run_process_spec_with_proof "$case_dir/launch.json" client "$case_dir/client_live_process.json" "$case_dir/client.log"
            ;;
        smoke|quality)
            local token_budget="$QUALITY_TOKENS" completion_exit
            [[ "$TIER" == "smoke" ]] && token_budget="$SMOKE_TOKENS"
            write_quality_request "$case_dir/request.json" "$token_budget"
            execute_http_client_spec "$case_dir/launch.json" >"$case_dir/client.log" 2>&1
            completion_exit=$?
            if ((completion_exit == 0)); then
                validate_completion_response "$case_dir/response.json" "$token_budget" >>"$case_dir/client.log" 2>&1
                completion_exit=$?
            fi
            return "$completion_exit"
            ;;
    esac
}

write_summary() {
    local case_dir="$1" row="$2" status="$3" error_hits="$4"
    "$PY" - "$case_dir" "$row" "$TIER" "$status" "$error_hits" >>"$ROOT/summary.jsonl" <<'PY'
import json
import pathlib
import re
import sys
case_dir, row_raw, tier, status, error_hits = sys.argv[1:]
case = pathlib.Path(case_dir)
row = json.loads(row_raw)
server = (case / "server.log").read_text(errors="replace") if (case / "server.log").exists() else ""
trace_counts = {"prefill": len(re.findall(r"Captured prefill trace", server)), "decode": len(re.findall(r"Captured decode trace", server))}
program_counts = {
    "decode_compiles": len(re.findall(r"Compiled decode", server)),
    "sampling_compiles": len(re.findall(r"Compiled on-device sampling", server)),
}
record = {
    "row_id": row["id"], "tier": tier, "status": status, "error_hits": int(error_hits),
    "prefill_traces": trace_counts["prefill"], "decode_traces": trace_counts["decode"],
    "sampling_compiles": program_counts["sampling_compiles"],
    "case_dir": str(case),
}
metrics = {}
result_path = case / "result.json"
if result_path.exists():
    result = json.loads(result_path.read_text())
    for key in ("completed", "failed", "duration", "request_throughput", "output_throughput", "total_token_throughput", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms"):
        record[key] = result.get(key)
    for key in ("request_throughput", "output_throughput", "total_token_throughput", "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"):
        metrics[key] = result.get(key)
response_path = case / "response.json"
if response_path.exists():
    response = json.loads(response_path.read_text())
    choice = response.get("choices", [{}])[0]
    record.update({"finish_reason": choice.get("finish_reason"), "completion_tokens": response.get("usage", {}).get("completion_tokens"), "text_prefix": choice.get("text", "")[:180]})
evidence = {
    "trace_counts": trace_counts,
    "program_counts": program_counts,
    "trace_region_config_hits": len(re.findall(rf"(?:['\"])?trace_region_size(?:['\"])?\s*[:=]\s*{row['manifest']['trace_region_size']}\b", server)),
    "metrics": metrics,
}
evidence_path = case / "evidence.json"
temporary = evidence_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(evidence, indent=2) + "\n")
temporary.replace(evidence_path)
print(json.dumps(record, sort_keys=True))
PY
}

run_row() {
    local row_id="$1" row manifest cache_root case_dir
    local reset_before=99 reset_after=99 client_exit=99 cleanup_exit=99 error_hits status server_exit preexisting=0 preparation_failed=0
    row="$(row_json "$row_id")" || die "$row_id: cannot load row"
    manifest="$(json_field "$row" manifest)" || die "$row_id: cannot load manifest"
    cache_root="$("$PY" - "$row" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["manifest"].get("cache_root") or "")
PY
    )" || die "$row_id: cannot read cache_root"
    [[ -n "$cache_root" ]] || die "$row_id: manifest.cache_root is required (per-row TT cache isolation)"
    case_dir="$ROOT/$row_id"
    CURRENT_CASE="$case_dir"
    write_manifest "$row" "$case_dir/manifest.json" running 0 || die "cannot write initial manifest"
    "$PY" - "$ROOT/attempt_journal.jsonl" "$row_id" "$TIER" <<'PY' || die "cannot journal row start"
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
record = {"event": "row_started", "row_id": sys.argv[2], "tier": sys.argv[3], "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with path.open("a") as out: out.write(json.dumps(record, sort_keys=True) + "\n")
PY
    JOURNAL_OPEN=1
    printf '=== %s ===\n' "$row_id" | tee -a "$ROOT/run.log"

    list_vllm_processes >"$case_dir/process_check_before.log" || die "cannot write pre-launch process inventory"
    if [[ -s "$case_dir/process_check_before.log" ]]; then
        preexisting=1
        printf 'pre-existing vLLM processes; refusing to share hardware\n' >"$case_dir/client.log"
        record_exit "$case_dir/client.exit" 96
        : >"$case_dir/server.log"
    else
        reset_tt "$case_dir/reset_before.log"
        reset_before=$?
        record_exit "$case_dir/reset_before.exit" "$reset_before"
        if ((reset_before == 0)); then
            HARDWARE_PREPARED=1
            check_hf_cache_and_ref || die "$row_id: HF refs/main or cache changed immediately before launch"
            (
                exec_process_spec "$case_dir/launch.json" server
            ) >"$case_dir/server.log" 2>&1 &
            SERVER_PID=$!
            record_exit "$case_dir/server.pid" "$SERVER_PID"
            if ! capture_live_process "$SERVER_PID" "$case_dir/live_process.json" "$case_dir/launch.json"; then
                kill "$SERVER_PID" 2>/dev/null || true
                wait "$SERVER_PID" 2>/dev/null || true
                SERVER_PID=""
                die "$row_id: live process does not match LaunchSpec"
            fi
            SERVER_PGID="$($PY - "$case_dir/live_process.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["pgid"])
PY
            )" || die "$row_id: cannot read proven server process group"
            if [[ "$SERVER_PGID" != "$SERVER_PID" ]]; then
                kill "$SERVER_PID" 2>/dev/null || true
                wait "$SERVER_PID" 2>/dev/null || true
                SERVER_PID=""
                SERVER_PGID=""
                die "$row_id: server launcher did not create an owned session/process group"
            fi
            record_exit "$case_dir/server.pgid" "$SERVER_PGID"
            if wait_for_health "$case_dir/server.log"; then
                run_client "$case_dir"
                client_exit=$?
            else
                printf 'health failed\n' >"$case_dir/client.log"
                client_exit=98
            fi
            record_exit "$case_dir/client.exit" "$client_exit"
            local server_pid="$SERVER_PID"
            cleanup_server "$SERVER_PID" "$SERVER_PGID" "$case_dir/cleanup.log"
            cleanup_exit=$?
            record_exit "$case_dir/cleanup.exit" "$cleanup_exit"
            wait "$server_pid" 2>/dev/null
            server_exit=$?
            record_exit "$case_dir/server.exit" "$server_exit"
        else
            : >"$case_dir/server.log"
            preparation_failed=1
            printf 'TT device preparation failed (mode=%s)\n' "$TT_DEVICE_RECOVERY_MODE" >"$case_dir/client.log"
            record_exit "$case_dir/client.exit" 97
            record_exit "$case_dir/cleanup.exit" 97
        fi
    fi

    if [[ ! -f "$case_dir/reset_before.exit" ]]; then record_exit "$case_dir/reset_before.exit" "$reset_before"; fi
    if [[ ! -f "$case_dir/cleanup.exit" ]]; then record_exit "$case_dir/cleanup.exit" "$cleanup_exit"; fi
    if ((preexisting)); then
        printf 'reset skipped because pre-existing vLLM processes were detected\n' >"$case_dir/reset_after.log"
        reset_after=96
    elif ((preparation_failed)); then
        printf 'TT device post-run action skipped because preparation failed before server launch (mode=%s)\n' "$TT_DEVICE_RECOVERY_MODE" >"$case_dir/reset_after.log"
        reset_after=95
    elif ((cleanup_exit != 0)); then
        printf 'reset skipped because owned process-group cleanup failed\n' >"$case_dir/reset_after.log"
        reset_after=95
    else
        reset_tt "$case_dir/reset_after.log"
        reset_after=$?
    fi
    record_exit "$case_dir/reset_after.exit" "$reset_after"
    ((reset_after == 0)) && HARDWARE_PREPARED=0
    list_vllm_processes >"$case_dir/process_check_after.log" || die "cannot write post-row process inventory"
    error_hits="$(scan_error_hits "$case_dir")" || die "$row_id: cannot scan failure logs"
    status=failed
    if [[ "$(<"$case_dir/client.exit")" == 0 && "$reset_before" == 0 && "$reset_after" == 0 && "$cleanup_exit" == 0 && "$error_hits" == 0 && ! -s "$case_dir/process_check_after.log" ]]; then
        status=ok
    fi
    write_manifest "$row" "$case_dir/manifest.json" "$status" "$error_hits" || die "cannot finalize manifest"
    write_summary "$case_dir" "$row" "$status" "$error_hits" || die "cannot write evidence/summary"
    "$PY" - "$ROOT/attempt_journal.jsonl" "$row_id" "$TIER" "$status" "$error_hits" <<'PY' || die "cannot journal row completion"
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
record = {"event": "row_finished", "row_id": sys.argv[2], "tier": sys.argv[3], "status": sys.argv[4], "error_hits": int(sys.argv[5]), "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with path.open("a") as out: out.write(json.dumps(record, sort_keys=True) + "\n")
PY
    JOURNAL_OPEN=0
    CURRENT_CASE=""
    [[ "$status" == ok ]]
}

run_status=0
for row_id in "${SELECTED_ROWS[@]}"; do
    run_row "$row_id" || run_status=1
done
list_vllm_processes >"$ROOT/process_check_after.log" || die "cannot write final process inventory"
"$PY" - "$ROOT/attempt_journal.jsonl" "$TIER" "$run_status" <<'PY' || die "cannot journal run completion"
import datetime, json, pathlib, sys
record = {"event": "run_finished", "tier": sys.argv[2], "status": "ok" if int(sys.argv[3]) == 0 else "failed", "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
with pathlib.Path(sys.argv[1]).open("a") as out: out.write(json.dumps(record, sort_keys=True) + "\n")
PY
RUN_JOURNAL_OPEN=0

if [[ "$TIER" == "quality" ]]; then
    write_quality_review "$ROOT/quality_review.json" || die "cannot create quality review provenance"
fi

if ((run_status != 0)); then
    printf 'FAIL: one or more hardware rows failed; inspect %s\n' "$ROOT" >&2
    exit 1
fi
if [[ "$ACCEPTANCE_SCOPE" == "subset" ]]; then
    "$PY" "$VALIDATOR" --artifact-root "$ROOT" --expectations "$EXPECTATIONS" --tier "$TIER" --subset-evidence 2>&1 | tee "$ROOT/validation_subset.log"
    subset_statuses=("${PIPESTATUS[@]}")
    subset_validation_exit=${subset_statuses[0]}
    ((subset_statuses[1] == 0)) || die "cannot persist subset validator output"
    if ((subset_validation_exit != 3)); then
        printf 'FAIL: subset evidence validation failed; inspect %s\n' "$ROOT/validation_subset.log" >&2
        exit 1
    fi
    "$PY" - "$ROOT/subset_status.json" "$TIER" "${SELECTED_ROWS[@]}" <<'PY' || die "cannot persist subset status"
import json, pathlib, sys
destination, tier, *rows = sys.argv[1:]
pathlib.Path(destination).write_text(json.dumps({"status": "execution_complete", "acceptance": False, "tier": tier, "selected_row_ids": rows, "next": "aggregate all canonical rows with --aggregate-from"}, indent=2) + "\n")
PY
    printf 'SUBSET_COMPLETE_NOT_ACCEPTED: %s row(s); aggregate all canonical rows for acceptance\n' "${#SELECTED_ROWS[@]}" >&2
    exit 3
fi
run_final_validation
exit $?
