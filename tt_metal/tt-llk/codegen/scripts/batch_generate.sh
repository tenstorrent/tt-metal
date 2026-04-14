#!/bin/bash
# Batch LLK kernel generation for Quasar
# Usage:
#   ./scripts/batch_generate.sh                        # list all kernels
#   ./scripts/batch_generate.sh --wave 1               # run all Wave 1 sequentially
#   ./scripts/batch_generate.sh --wave 1 --parallel    # run all Wave 1 in parallel
#   ./scripts/batch_generate.sh --kernel abs           # run a single kernel
#   ./scripts/batch_generate.sh --from 5               # resume from kernel #5
#   ./scripts/batch_generate.sh --wave 1 --dry-run     # show what would run
#   ./scripts/batch_generate.sh --parallel -j 4        # limit to 4 concurrent jobs
#   ./scripts/batch_generate.sh --model sonnet          # use a different model
#
# Waves are ordered to maximize test feedback early:
#   Wave 1: Testable simple SFPU (4)     — have golden generators
#   Wave 2: Testable medium SFPU (5)     — have golden generators
#   Wave 3: Remaining simple SFPU (6)    — compile-only
#   Wave 4: Remaining medium SFPU (9)    — compile-only
#   Wave 5: Complex SFPU w/ tests (4)    — have test potential
#   Wave 6: Remaining complex SFPU (8)   — compile-only
#   Wave 7: Specialized SFPU (6)         — compile-only
#   Wave 8: LLK Submodule core (10)      — math wrappers, pack, unpack
#   Wave 9: LLK Submodule experimental (13) — low priority
#
# Parallel safety:
#   - Each kernel writes to its own SFPU file — no conflicts
#   - Compilation uses per-PID build dirs — no conflicts
#   - runs.jsonl uses file locking for safe concurrent appends
#   - ckernel_sfpu.h is NOT edited during generation — update it after
#   - Wave 8-9 depends on Waves 1-7; do NOT run Wave 8-9 in parallel with earlier waves
#   - Simulator tests are serialized via flock /tmp/tt-llk-test-simulator.lock
#     (each agent acquires lock before running pytest --run-simulator)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODEGEN_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/tmp/codegen_logs_$(date +%Y%m%d_%H%M%S)"

# CI environment — tells the orchestrator this is an automated batch run
export CODEGEN_BATCH_ID="${CODEGEN_BATCH_ID:-$(date +%Y-%m-%d)_batch}"
export CODEGEN_MODEL="${CODEGEN_MODEL:-opus}"

# --- Kernel definitions ---
# Format: "number|wave|kernel_name|type|notes"
# Ordered by wave to maximize testable kernels early.

KERNELS=(
  # Wave 0: Regression smoke test — simplest kernels for quick validation
  "1|0|abs|sfpu|regression smoke test"
  "2|0|fill|sfpu|regression smoke test"

  # Wave 1: Testable simple SFPU — have golden generators in test infra
  "3|1|abs|sfpu|has golden: _abs"
  "4|1|negative|sfpu|has golden: _neg"
  "5|1|fill|sfpu|has golden: _fill"
  "6|1|threshold|sfpu|has golden: _threshold"

  # Wave 2: Testable medium SFPU — have golden generators
  "7|2|elu|sfpu|has golden: _elu, uses exp"
  "8|2|exp2|sfpu|has golden: _exp2"
  "9|2|log|sfpu|has golden: _log, LUT-based"
  "10|2|trigonometry|sfpu|has golden: _cos/_sin"
  "11|2|activations|sfpu|has golden: multiple"

  # Wave 3: Remaining simple SFPU — compile-only
  "12|3|sign|sfpu|"
  "13|3|hardtanh|sfpu|"
  "14|3|clamp|sfpu|"
  "15|3|dropout|sfpu|"
  "16|3|is_fp16_zero|sfpu|"
  "17|3|where|sfpu|conditional select"

  # Wave 4: Remaining medium SFPU — compile-only
  "18|4|cdf|sfpu|uses exp"
  "19|4|tanh_derivative|sfpu|uses tanh"
  "20|4|rsqrt_compat|sfpu|"
  "21|4|rounding_ops|sfpu|"
  "22|4|polyval|sfpu|polynomial eval"
  "23|4|load_config|sfpu|config loading"
  "24|4|cast_fp32_to_fp16a|sfpu|format conversion"
  "25|4|converter|sfpu|format conversion"
  "26|4|typecast|sfpu|multi-type cast"

  # Wave 5: Complex SFPU with test potential
  "27|5|comp|sfpu|has cross-arch test"
  "28|5|topk|sfpu|has cross-arch test"
  "29|5|quant|sfpu|has cross-arch test"
  "30|5|binary|sfpu|test via eltwise_binary after math wrapper"

  # Wave 6: Remaining complex SFPU — compile-only
  "31|6|binary_bitwise|sfpu|bitwise ops"
  "32|6|add_int|sfpu|integer arithmetic"
  "33|6|sub_int|sfpu|integer arithmetic"
  "34|6|mul_int|sfpu|integer arithmetic"
  "35|6|shift|sfpu|bit shifting"
  "36|6|isinf_isnan|sfpu|"
  "37|6|cumsum|sfpu|cumulative sum"
  "38|6|ema|sfpu|exponential moving avg"

  # Wave 7: Specialized SFPU — compile-only
  "39|7|welfords|sfpu|online variance"
  "40|7|reduce|sfpu|SFPU reduction"
  "41|7|reduce_custom|sfpu|custom reduction"
  "42|7|max_pool_indices|sfpu|pooling indices"
  "43|7|add_top_row|sfpu|row manipulation"
  "44|7|reshuffle_rows|sfpu|row reordering"

  # Wave 8: LLK Submodule — core (math wrappers, pack, unpack)
  "45|8|math_eltwise_unary_sfpu_params|math|depends on sfpu_common (exists)"
  "46|8|math_eltwise_binary_sfpu|math|depends on SFPU kernels"
  "47|8|math_eltwise_binary_sfpu_params|math|depends on #46"
  "48|8|math_eltwise_ternary_sfpu|math|depends on SFPU kernels"
  "49|8|math_eltwise_ternary_sfpu_params|math|depends on #48"
  "50|8|math_welfords_sfpu|math|depends on sfpu_welfords"
  "51|8|math_welfords_sfpu_params|math|depends on #50"
  "52|8|math_transpose_dest|math|dest register transpose"
  "53|8|pack_rows|pack|row-based packing"
  "54|8|unpack_untilize|unpack|untilize on unpack"

  # Wave 9: LLK Submodule — experimental (low priority)
  "55|9|math_eltwise_binary_custom|math|experimental"
  "56|9|math_eltwise_unary_datacopy_custom|math|experimental"
  "57|9|math_matmul_custom_no_mop|math|experimental"
  "58|9|math_mul_reduce_scalar|math|experimental"
  "59|9|math_reduce_custom|math|experimental"
  "60|9|math_reduce_runtime_custom|math|experimental"
  "61|9|pack_custom|pack|experimental"
  "62|9|unpack_A_custom|unpack|experimental"
  "63|9|unpack_AB_matmul_custom|unpack|experimental"
  "64|9|unpack_AB_reduce_custom|unpack|experimental"
  "65|9|unpack_AB_reduce_custom_runtime|unpack|experimental"
  "66|9|unpack_AB_sub_bcast_col_custom|unpack|experimental"
  "67|9|unpack_mul_reduce_scalar|unpack|experimental"
)

# --- Parse args ---
WAVE=""
KERNEL=""
FROM=1
DRY_RUN=false
PARALLEL=false
MAX_JOBS=0  # 0 = unlimited
MODEL="${CODEGEN_MODEL}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --wave)   WAVE="$2"; shift 2 ;;
    --kernel) KERNEL="$2"; shift 2 ;;
    --from)   FROM="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --parallel) PARALLEL=true; shift ;;
    -j)       MAX_JOBS="$2"; PARALLEL=true; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--wave N] [--kernel NAME] [--from N] [--model MODEL] [--parallel] [-j N] [--dry-run]"
      echo ""
      echo "  --wave N      Run all kernels in wave N (1-9)"
      echo "  --kernel NAME Run a single kernel by name"
      echo "  --from N      Resume from kernel number N"
      echo "  --model MODEL Claude model to use (default: opus)"
      echo "  --parallel    Run kernels in parallel (within a wave)"
      echo "  -j N          Max concurrent jobs (default: unlimited)"
      echo "  --dry-run     Show prompts without running"
      echo ""
      echo "Waves (ordered for maximum test feedback):"
      echo "  0  Regression smoke test (2)     — abs + fill only"
      echo "  1  Testable simple SFPU (4)      — have golden generators"
      echo "  2  Testable medium SFPU (5)      — have golden generators"
      echo "  3  Remaining simple SFPU (6)     — compile-only"
      echo "  4  Remaining medium SFPU (9)     — compile-only"
      echo "  5  Complex SFPU w/ tests (4)     — have test potential"
      echo "  6  Remaining complex SFPU (8)    — compile-only"
      echo "  7  Specialized SFPU (6)          — compile-only"
      echo "  8  LLK Submodule core (10)       — math wrappers, pack, unpack"
      echo "  9  LLK Submodule experimental (13) — low priority"
      echo ""
      echo "Parallel notes:"
      echo "  - Safe to run all kernels within Waves 1-7 in parallel"
      echo "  - Wave 8-9 depends on Waves 1-7 completing first"
      echo "  - Within Wave 8, #45 depends on #44, #47 on #46, #49 on #48"
      echo "  - After parallel runs, update ckernel_sfpu.h with new #includes"
      echo ""
      echo "With no args, lists all kernels."
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Update model env var for orchestrator
export CODEGEN_MODEL="$MODEL"

# --- List mode ---
if [[ -z "$WAVE" && -z "$KERNEL" ]]; then
  echo "=== Quasar LLK Generation Plan: 67 kernels (2 regression + 52 core + 13 experimental) ==="
  echo "=== Ordered by testability (testable kernels first) ==="
  echo "=== Model: $MODEL ==="
  echo ""
  current_wave=""
  for entry in "${KERNELS[@]}"; do
    IFS='|' read -r num wave name type notes <<< "$entry"
    if [[ "$wave" != "$current_wave" ]]; then
      current_wave="$wave"
      case $wave in
        0) echo "--- Wave 0: Regression Smoke Test (2) — abs + fill ---" ;;
        1) echo "--- Wave 1: Testable Simple SFPU (4) — have golden generators ---" ;;
        2) echo "--- Wave 2: Testable Medium SFPU (5) — have golden generators ---" ;;
        3) echo "--- Wave 3: Remaining Simple SFPU (6) — compile-only ---" ;;
        4) echo "--- Wave 4: Remaining Medium SFPU (9) — compile-only ---" ;;
        5) echo "--- Wave 5: Complex SFPU w/ Tests (4) — have test potential ---" ;;
        6) echo "--- Wave 6: Remaining Complex SFPU (8) — compile-only ---" ;;
        7) echo "--- Wave 7: Specialized SFPU (6) — compile-only ---" ;;
        8) echo "--- Wave 8: LLK Submodule Core (10) — math wrappers, pack, unpack ---" ;;
        9) echo "--- Wave 9: LLK Submodule Experimental (13) — low priority ---" ;;
      esac
    fi
    printf "  %2s. %-40s [%s] %s\n" "$num" "$name" "$type" "$notes"
  done
  echo ""
  echo "Run with: $0 --wave 1                    # all wave 1 sequentially"
  echo "          $0 --wave 1 --parallel          # all wave 1 in parallel"
  echo "          $0 --wave 1 -j 4                # wave 1, max 4 concurrent"
  echo "          $0 --kernel abs                 # single kernel"
  echo "          $0 --from 5                     # resume from #5"
  echo "          $0 --model sonnet --wave 1      # use sonnet model"
  echo "          $0 --wave 1 --dry-run           # preview"
  exit 0
fi

# --- Build run list ---
run_list=()
for entry in "${KERNELS[@]}"; do
  IFS='|' read -r num wave name type notes <<< "$entry"

  # Filter by wave
  if [[ -n "$WAVE" && "$wave" != "$WAVE" ]]; then continue; fi

  # Filter by kernel name
  if [[ -n "$KERNEL" && "$name" != "$KERNEL" ]]; then continue; fi

  # Filter by --from
  if [[ "$num" -lt "$FROM" ]]; then continue; fi

  run_list+=("$entry")
done

if [[ ${#run_list[@]} -eq 0 ]]; then
  echo "No matching kernels found."
  exit 1
fi

mode="sequentially"
if $PARALLEL; then mode="in parallel"; fi
echo "=== Will generate ${#run_list[@]} kernel(s) ${mode} ==="
echo "=== Model: $MODEL ==="
if $PARALLEL && [[ $MAX_JOBS -gt 0 ]]; then
  echo "=== Max concurrent jobs: ${MAX_JOBS} ==="
fi
echo ""

# --- Save CLI JSON output and patch token/cost data into runs.jsonl + run.json ---
save_cli_output() {
  local name="$1" json_file="$2"
  python3 -c "
import json, shutil, sys, os, fcntl, tempfile

RUNS_JSONL = '/proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl'
RUNS_BASE  = '/proj_sw/user_dev/llk_code_gen/quasar'

name, json_file = sys.argv[1], sys.argv[2]

def resolve_log_dir(log_dir):
    \"\"\"Resolve relative log_dir paths against the runs base directory.\"\"\"
    if os.path.isabs(log_dir):
        return log_dir
    # Relative paths like 'logs/2026-...' — try under RUNS_BASE
    candidate = os.path.join(RUNS_BASE, log_dir)
    if os.path.isdir(candidate):
        return candidate
    # Also try stripping 'logs/' prefix (orchestrator sometimes uses logs/ prefix)
    if log_dir.startswith('logs/'):
        candidate = os.path.join(RUNS_BASE, log_dir[5:])
        if os.path.isdir(candidate):
            return candidate
    return log_dir  # return as-is, caller checks existence

def extract_tokens(cli_json_path):
    \"\"\"Extract token counts and cost from CLI JSON output.\"\"\"
    try:
        with open(cli_json_path) as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            return None
        last = data[-1]
        if not isinstance(last, dict):
            return None
        # Prefer modelUsage (cumulative across all turns) over usage (last turn only)
        model_usage = last.get('modelUsage', {})
        if model_usage:
            # Sum across all models (typically just one)
            total_input = sum(v.get('inputTokens', 0) for v in model_usage.values())
            total_output = sum(v.get('outputTokens', 0) for v in model_usage.values())
            total_cache_read = sum(v.get('cacheReadInputTokens', 0) for v in model_usage.values())
            total_cache_creation = sum(v.get('cacheCreationInputTokens', 0) for v in model_usage.values())
            cost = last.get('total_cost_usd', 0)
            return {
                'input': total_input,
                'output': total_output,
                'cache_read': total_cache_read,
                'cache_creation': total_cache_creation,
                'total': total_input + total_output,
                'cost_usd': round(cost, 6) if cost else 0,
            }
        # Fallback to top-level usage (last turn only — less accurate but better than 0)
        usage = last.get('usage', {})
        if usage:
            inp = usage.get('input_tokens', 0)
            out = usage.get('output_tokens', 0)
            cache_read = usage.get('cache_read_input_tokens', 0)
            return {
                'input': inp,
                'output': out,
                'cache_read': cache_read,
                'cache_creation': usage.get('cache_creation_input_tokens', 0),
                'total': inp + out,
                'cost_usd': round(last.get('total_cost_usd', 0), 6),
            }
        return None
    except Exception:
        return None

try:
    # 1. Find the last matching entry in runs.jsonl
    last_entry = None
    last_line_idx = None
    lines = []
    with open(RUNS_JSONL) as f:
        for i, line in enumerate(f):
            lines.append(line)
            try:
                entry = json.loads(line)
                if entry.get('kernel') == name:
                    last_entry = entry
                    last_line_idx = i
            except:
                pass

    if not last_entry:
        print(f'  Warning: no runs.jsonl entry found for kernel \"{name}\"', file=sys.stderr)
        sys.exit(0)

    # 2. Resolve log_dir and copy CLI output
    log_dir = resolve_log_dir(last_entry.get('log_dir', ''))
    if os.path.isdir(log_dir):
        shutil.copy2(json_file, os.path.join(log_dir, 'cli_output.json'))
        print(f'  Saved CLI output to {log_dir}/cli_output.json')
    else:
        print(f'  Warning: log_dir not found: {last_entry.get(\"log_dir\", \"\")}', file=sys.stderr)

    # 3. Extract tokens from CLI output and patch runs.jsonl + run.json
    tokens = extract_tokens(json_file)
    if tokens:
        last_entry['tokens'] = tokens
        # Rewrite the matching line in runs.jsonl (atomic via temp file)
        lines[last_line_idx] = json.dumps(last_entry, separators=(',', ':')) + '\n'
        tmp_fd, tmp_path = tempfile.mkstemp(dir=RUNS_BASE, suffix='.jsonl')
        try:
            with os.fdopen(tmp_fd, 'w') as tmp_f:
                tmp_f.writelines(lines)
            os.replace(tmp_path, RUNS_JSONL)
            print(f'  Patched runs.jsonl: input={tokens[\"input\"]} output={tokens[\"output\"]} cache_read={tokens[\"cache_read\"]} cost=\${tokens[\"cost_usd\"]}')
        except Exception as e:
            # Clean up temp file on failure, leave original intact
            try: os.unlink(tmp_path)
            except: pass
            print(f'  Warning: could not patch runs.jsonl: {e}', file=sys.stderr)

        # Also patch run.json in log_dir if it exists
        if os.path.isdir(log_dir):
            run_json_path = os.path.join(log_dir, 'run.json')
            if os.path.isfile(run_json_path):
                try:
                    with open(run_json_path) as f:
                        run_data = json.load(f)
                    run_data['tokens'] = tokens
                    with open(run_json_path, 'w') as f:
                        json.dump(run_data, f, indent=2)
                        f.write('\n')
                    print(f'  Patched {run_json_path}')
                except Exception as e:
                    print(f'  Warning: could not patch run.json: {e}', file=sys.stderr)
    else:
        print(f'  Warning: could not extract tokens from CLI output', file=sys.stderr)

except Exception as e:
    print(f'  Warning: save_cli_output failed: {e}', file=sys.stderr)
" "$name" "$json_file"
}

# --- Run a single kernel ---
run_one_kernel() {
  local num="$1" name="$2" total="$3"
  local prompt="Generate ${name} for Quasar. CODEGEN_BATCH_ID=${CODEGEN_BATCH_ID} CODEGEN_MODEL=${MODEL}"
  local logfile="${LOG_DIR}/${name}.log"
  local jsonfile="${LOG_DIR}/${name}.json"

  echo "[$num/$total] START: $prompt (model: $MODEL)"

  cd "$CODEGEN_DIR"
  claude -p "$prompt" --dangerously-skip-permissions --effort max --verbose --model "$MODEL" --output-format json > "$jsonfile" 2>"$logfile"
  local exit_code=$?

  save_cli_output "$name" "$jsonfile"

  if [[ $exit_code -ne 0 ]]; then
    echo "[$num/$total] FAILED: $name (exit code $exit_code) — see $logfile"
    return 1
  else
    echo "[$num/$total] DONE: $name"
    return 0
  fi
}

# --- Sequential run ---
run_sequential() {
  for entry in "${run_list[@]}"; do
    IFS='|' read -r num wave name type notes <<< "$entry"

    prompt="Generate ${name} for Quasar. CODEGEN_BATCH_ID=${CODEGEN_BATCH_ID} CODEGEN_MODEL=${MODEL}"

    echo "[$num/${#KERNELS[@]}] $prompt"

    if $DRY_RUN; then
      echo "  (dry run — skipping)"
      echo ""
      continue
    fi

    cd "$CODEGEN_DIR"
    # JSON output -> file, verbose stderr -> tee (terminal + log)
    claude -p "$prompt" --dangerously-skip-permissions --effort max --verbose --model "$MODEL" --output-format json 2>&1 1>"${LOG_DIR}/${name}.json" | tee "${LOG_DIR}/${name}.log"

    exit_code=${PIPESTATUS[0]}

    save_cli_output "$name" "${LOG_DIR}/${name}.json"

    if [[ $exit_code -ne 0 ]]; then
      echo "  FAILED (exit code $exit_code) — stopping. Resume with: $0 --from $((num + 1))"
      exit 1
    fi

    echo "  Done."
    echo ""
  done
}

# --- Parallel run ---
run_parallel() {
  local pids=()
  local names=()
  local active=0

  for entry in "${run_list[@]}"; do
    IFS='|' read -r num wave name type notes <<< "$entry"

    if $DRY_RUN; then
      echo "[$num/${#KERNELS[@]}] Generate ${name} for Quasar (dry run — skipping)"
      continue
    fi

    # Throttle if max jobs reached
    if [[ $MAX_JOBS -gt 0 ]]; then
      while [[ $active -ge $MAX_JOBS ]]; do
        # Wait for any child to finish
        wait -n 2>/dev/null || true
        # Recount active
        active=0
        for pid in "${pids[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            ((active++))
          fi
        done
      done
    fi

    run_one_kernel "$num" "$name" "${#KERNELS[@]}" &
    pids+=($!)
    names+=("$name")
    ((active++))
  done

  if $DRY_RUN; then return; fi

  # Wait for all and collect results
  echo ""
  echo "=== Waiting for ${#pids[@]} parallel job(s) to complete ==="
  echo ""

  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      ((failed++))
    fi
  done

  echo ""
  if [[ $failed -gt 0 ]]; then
    echo "=== ${failed} kernel(s) FAILED — check logs in ${LOG_DIR}/ ==="
    return 1
  fi
}

# --- Main ---
mkdir -p "$LOG_DIR"

# --- Pre-flight: clean stale simulator processes ---
echo "=== Checking for stale simulator processes on port 5556 ==="
STALE_PIDS=$(lsof -ti :5556 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "WARNING: Found stale processes on port 5556: $STALE_PIDS"
    echo "Killing stale processes..."
    echo "$STALE_PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
    echo "Done."
else
    echo "No stale processes found."
fi
pkill -9 -f "tt-exalens.*--port=5556" 2>/dev/null || true

if $PARALLEL; then
  run_parallel
else
  run_sequential
fi

echo "=== All ${#run_list[@]} kernel(s) complete ==="
echo "=== Logs: ${LOG_DIR}/ ==="

# --- Post-run reminder ---
if ! $DRY_RUN; then
  echo ""
  echo "=== POST-RUN: Update ckernel_sfpu.h ==="
  echo "Add #include lines for newly generated kernels to:"
  echo "  tt_llk_quasar/common/inc/ckernel_sfpu.h"
  echo ""
  echo "Generated SFPU kernels:"
  for entry in "${run_list[@]}"; do
    IFS='|' read -r num wave name type notes <<< "$entry"
    if [[ "$type" == "sfpu" ]]; then
      echo "  #include \"sfpu/ckernel_sfpu_${name}.h\""
    fi
  done
fi
