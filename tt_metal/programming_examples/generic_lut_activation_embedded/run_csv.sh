#!/bin/bash
# =============================================================================
# run_csv.sh — Run an arbitrary coefficient CSV through the embedded flow
#
# Usage:
#   ./run_csv.sh <csv_file> --activation <name> [--precision fp32|bf16|both] [--tiles N] [--range-min X] [--range-max X] [--skip-build] [--dump-csv <path>|--dump-npz <path>]
#
# Example:
#   ./run_csv.sh /path/to/sigmoid_16_6_uniform_any_ulp.csv --activation sigmoid
#   ./run_csv.sh my_coeffs.csv --activation gelu --precision bf16 --tiles 64
# =============================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
WORK_DIR="tt_metal/programming_examples/generic_lut_activation_embedded"
BUILD_DIR="${TT_METAL_RUNTIME_ROOT:-$REPO_ROOT}/build_Release"
BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc"
KERNEL_DIR="$SCRIPT_DIR/kernels/compute/adhoc"

source "$SCRIPT_DIR/profiler_helpers.sh"
source "$SCRIPT_DIR/sweep_helpers.sh"
init_arch_detection

# TT_POLY_FIT_DIR needed for extract_accuracy.py (ground truth + ULP computation)
if [[ -z "${TT_POLY_FIT_DIR:-}" ]]; then
    for candidate in \
        "$REPO_ROOT/../tt-polynomial-fitter" \
        "$HOME/tt-polynomial-fitter" \
        "$HOME/workspace/tt-polynomial-fitter" \
        "/localdev/$USER/tt-polynomial-fitter" \
        "/localdev/nkapre/tt-polynomial-fitter"; do
        if [[ -f "$candidate/extract_accuracy.py" ]]; then
            TT_POLY_FIT_DIR="$(cd "$candidate" && pwd)"
            break
        fi
    done
fi
TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/$USER/tt-polynomial-fitter}"
ACCURACY_SCRIPT="$TT_POLY_FIT_DIR/extract_accuracy.py"
# Use system python for accuracy (python_env's torch has broken BF16 ULP spacing)
ACCURACY_PYTHON="/usr/bin/python3"
HAS_ACCURACY=false
if [[ -f "$ACCURACY_SCRIPT" ]]; then
    HAS_ACCURACY=true
fi

# Standard test shapes (same as sweep scripts)
declare -a TEST_SHAPES=(
    "single_tile:32:32"
    "8_tiles:64:128"
    "256_tiles:512:512"
    "height_sharded:25600:128"
    "yolov4:5120:320"
)

NUM_RUNS=3

# --- Parse args ---
CSV_FILE=""
ACTIVATION=""
PRECISION="fp32"
TILES_OVERRIDE=""
RANGE_MIN=""
RANGE_MAX=""
SKIP_BUILD=false
DUMP_CSV=""
DUMP_NPZ=""
EXTRA_BINARY_ARGS=()

show_help() {
    echo "Usage: $0 <csv_file> --activation <name> [OPTIONS]"
    echo ""
    echo "Run an arbitrary coefficient CSV through the embedded kernel flow."
    echo "Auto-detects polynomial degree and segment count from the CSV."
    echo "Runs all 5 standard shapes with Tracy profiler timing (3 runs, takes min)."
    echo ""
    echo "Required:"
    echo "  <csv_file>                Path to coefficient CSV (segment_id,lo,hi,c0,c1,...)"
    echo "  --activation <name>       Activation function name (for ground truth comparison)"
    echo ""
    echo "Optional:"
    echo "  --precision <fp32|bf16|both>  Precision mode (default: fp32). 'both' runs bf16 then fp32."
    echo "  --tiles <N>               Override: run only this tile count (skip standard shapes)"
    echo "  --range-min <X>           Override input range min (default: from CSV)"
    echo "  --range-max <X>           Override input range max (default: from CSV)"
    echo "  --runs <N>                Number of timing runs per shape (default: 3)"
    echo "  --skip-build              Skip build (reuse last binary)"
    echo "  --dump-csv <path>         Copy the first measured shape/precision raw output CSV"
    echo "                            Use --tiles and one --precision for deterministic dumps"
    echo "  --dump-npz <path>         Same as --dump-csv, but compressed NPZ with input/output arrays"
    echo "  --no-dual-eval            Disable dual x-vector evaluation"
    echo "  --no-adaptive-degree      Disable per-segment degree optimization"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Standard shapes (run by default):"
    echo "  single_tile    32x32      (1 tile)"
    echo "  8_tiles        64x128     (8 tiles)"
    echo "  256_tiles      512x512    (256 tiles)"
    echo "  height_sharded 25600x128  (3200 tiles)"
    echo "  yolov4         5120x320   (51200 tiles)"
    echo ""
    echo "Examples:"
    echo "  $0 coeffs/sigmoid_16_6_uniform_any_ulp.csv --activation sigmoid"
    echo "  $0 my_gelu.csv --activation gelu --precision bf16"
    echo "  $0 my_gelu.csv --activation gelu --tiles 256          # single shape only"
    echo "  $0 exp_coeffs.csv --activation exp --dump-csv /tmp/hw_out.csv"
    exit 0
}

# First positional arg is CSV file
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
fi

if [[ "$1" != --* ]]; then
    CSV_FILE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --activation|-a) ACTIVATION="$2"; shift 2 ;;
        --precision|-p)  PRECISION="$2"; shift 2 ;;
        --tiles)         TILES_OVERRIDE="$2"; shift 2 ;;
        --range-min)     RANGE_MIN="$2"; shift 2 ;;
        --range-max)     RANGE_MAX="$2"; shift 2 ;;
        --runs)          NUM_RUNS="$2"; shift 2 ;;
        --skip-build)    SKIP_BUILD=true; shift ;;
        --dump-csv)      DUMP_CSV="$2"; shift 2 ;;
        --dump-npz)      DUMP_NPZ="$2"; shift 2 ;;
        --no-dual-eval)  EXTRA_BINARY_ARGS+=("--no-dual-eval"); shift ;;
        --no-adaptive-degree) EXTRA_BINARY_ARGS+=("--no-adaptive-degree"); shift ;;
        -h|--help)       show_help ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CSV_FILE" ]]; then
    echo "Error: CSV file path required as first argument"
    exit 1
fi
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi
if [[ -z "$ACTIVATION" ]]; then
    echo "Error: --activation is required"
    exit 1
fi

# Make CSV path absolute
CSV_FILE="$(cd "$(dirname "$CSV_FILE")" && pwd)/$(basename "$CSV_FILE")"

# If --tiles is given, replace standard shapes with a single custom shape
if [[ -n "$TILES_OVERRIDE" ]]; then
    TEST_SHAPES=("custom:1:$((TILES_OVERRIDE * 1024))")
    # Actually, tiles = rows/32 * cols/32. Simplest: use rows=32, cols=TILES*32
    cols=$((TILES_OVERRIDE * 32))
    TEST_SHAPES=("custom_${TILES_OVERRIDE}t:32:${cols}")
fi

# --- Step 1: Generate embedded kernel from CSV ---
echo "================================================================================"
echo "  run_csv.sh — Embedded flow for arbitrary coefficient CSV"
echo "================================================================================"
echo "CSV:        $CSV_FILE"
echo "Activation: $ACTIVATION"
echo "Precision:  $PRECISION"
echo "Runs/shape: $NUM_RUNS"
echo ""

mkdir -p "$KERNEL_DIR"
rm -f "$KERNEL_DIR/adhoc.cpp"

# Inline Python: parse CSV, auto-detect degree/segments, write kernel .cpp
RUN_CSV_GEN_LOG="$(mktemp "${TMPDIR:-/tmp}/run_csv_gen.XXXXXX.log")"
cleanup_gen_log() {
    rm -f "$RUN_CSV_GEN_LOG"
}
trap cleanup_gen_log EXIT

python3 -c "
import csv, sys, os

csv_path = '$CSV_FILE'
kernel_path = '$KERNEL_DIR/adhoc.cpp'
kernel_tmp_path = f'{kernel_path}.tmp.{os.getpid()}'
range_min_override = '$RANGE_MIN' or None
range_max_override = '$RANGE_MAX' or None

# Parse CSV
import math
boundaries = []
coefficients = []
segment_degrees = []
asymptotic_flags = []  # per-segment: True if asymptotic
dominant_factors = []  # per-segment: dominant factor string
metadata = {}
degree = 0

with open(csv_path) as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames

    # Auto-detect: polynomial (c0,c1,...) vs rational (n0,n1,...,d0,d1,...)
    num_cols = sorted([h for h in headers if h.startswith('n') and h[1:].isdigit()], key=lambda h: int(h[1:]))
    den_cols = sorted([h for h in headers if h.startswith('d') and h[1:].isdigit()], key=lambda h: int(h[1:]))
    coeff_cols = sorted([h for h in headers if h.startswith('c') and h[1:].isdigit()], key=lambda h: int(h[1:]))

    is_rational = len(num_cols) > 0 and len(den_cols) > 0
    num_degree = 0
    den_degree = 0

    if is_rational:
        num_degree = len(num_cols) - 1
        den_degree = len(den_cols) - 1
        degree = num_degree  # for compatibility with degree_macros logic below
        print(f'Auto-detected RATIONAL approximation: n{num_degree}/d{den_degree} (num: {num_cols[0]}..{num_cols[-1]}, den: {den_cols[0]}..{den_cols[-1]})')
    else:
        degree = len(coeff_cols) - 1
        print(f'Auto-detected polynomial degree: {degree} (columns: {coeff_cols[0]}..{coeff_cols[-1]})')

    num_coefficients = []  # rational numerator
    den_coefficients = []  # rational denominator

    for row in reader:
        if row.get('segment_id', '').upper() == 'METADATA':
            key = row.get('lo', '')
            val = row.get('hi', '')
            if key:
                metadata[key] = val
            continue

        if len(boundaries) == 0:
            boundaries.append(float(row['lo']))
        boundaries.append(float(row['hi']))

        if is_rational:
            seg_num = [float(row[col]) for col in num_cols]
            seg_den = [float(row[col]) for col in den_cols]
            num_coefficients.extend(seg_num)
            den_coefficients.extend(seg_den)
            coefficients.extend(seg_num + seg_den)
            # Effective degree = max of num and den effective degrees
            seg_deg = 0
            for d in range(num_degree, -1, -1):
                if seg_num[d] != 0.0:
                    seg_deg = d
                    break
            segment_degrees.append(seg_deg)
        else:
            seg_coeffs = [float(row[col]) for col in coeff_cols]
            coefficients.extend(seg_coeffs)
            # Detect effective degree (highest non-zero coefficient)
            seg_deg = 0
            for d in range(degree, -1, -1):
                if seg_coeffs[d] != 0.0:
                    seg_deg = d
                    break
            segment_degrees.append(seg_deg)

        # Asymptotic factoring metadata
        is_asym = row.get('is_asymptotic', '').strip().lower() == 'true'
        dom_factor = row.get('dominant_factor', '').strip() if is_asym else ''
        asymptotic_flags.append(is_asym)
        dominant_factors.append(dom_factor)

num_segments = len(boundaries) - 1
rr_enabled = str(metadata.get('range_reduction_enabled', '')).strip().lower() in ('true', '1', 'yes')
rr_has_original_domain = (
    metadata.get('range_reduction_original_min', '') != '' and
    metadata.get('range_reduction_original_max', '') != ''
)
default_input_min = float(metadata['range_reduction_original_min']) if rr_enabled and rr_has_original_domain else boundaries[0]
default_input_max = float(metadata['range_reduction_original_max']) if rr_enabled and rr_has_original_domain else boundaries[-1]
input_min = float(range_min_override) if range_min_override else default_input_min
input_max = float(range_max_override) if range_max_override else default_input_max

print(f'Segments: {num_segments}')
print(f'Range: [{input_min}, {input_max}]')
print(f'LUT entries: {len(boundaries)} boundaries + {len(coefficients)} coefficients = {len(boundaries) + len(coefficients)} total')

# Clamp float32
def clamp(v):
    if abs(v) > 3.4028234663852886e38:
        return 3.4028234663852886e38 if v > 0 else -3.4028234663852886e38
    if abs(v) < 1.4e-45:
        return 0.0
    return v

if is_rational:
    # Rational LUT layout: boundaries + [num_coeffs_seg0 + den_coeffs_seg0 + num_coeffs_seg1 + ...]
    lut_values = boundaries + coefficients
else:
    lut_values = boundaries + coefficients
lut_size = len(lut_values)
lut_str = ',\n    '.join(f'{clamp(v):.10e}f' for v in lut_values)

# Per-segment adaptive degree (skip wasted FMA on zero high-order coefficients)
# Two mechanisms:
#   1. SEGi_DEGREE macros — used by hand-written 4/8/16/32 specializations
#   2. SEGMENT_DEGREES[] constexpr array — used by recursive _N unroller
degree_macros = ''
if any(d < degree for d in segment_degrees):
    # SEGi_DEGREE macros for hand-written specializations (4/8/16/32)
    seg_macro_lines = [f'#define SEG{i}_DEGREE {d}' for i, d in enumerate(segment_degrees)]
    # constexpr array for recursive unroller (any segment count)
    deg_array = ', '.join(str(d) for d in segment_degrees)
    degree_macros = (
        '\n#ifndef DISABLE_ADAPTIVE_DEGREE\n'
        + '\n'.join(seg_macro_lines) + '\n'
        + f'#define HAS_SEGMENT_DEGREES\n'
        + f'constexpr uint32_t SEGMENT_DEGREES[] = {{{deg_array}}};\n'
        + '#endif\n'
    )
    reduced = sum(1 for d in segment_degrees if d < degree)
    avg_deg = sum(segment_degrees) / len(segment_degrees)
    print(f'Adaptive degree: {reduced}/{num_segments} segments reduced (avg effective degree: {avg_deg:.1f} vs max {degree})')

# Explicit evaluator basis metadata. This is a kernel eval contract, not an
# activation-name special case. odd_factored accepts either expanded
# P(u)=u*Q(u) coefficients or quotient Q(u) coefficients; only Q form needs the
# kernel to multiply by abs(x) before post-processing.
eval_basis_macro = ''
basis_kind = metadata.get('basis_kind', '').strip()
basis_kind_norm = basis_kind.lower().replace('-', '_')
basis_clamp_max = metadata.get('basis_clamp_max', '').strip()
plain_basis_kinds = ('', 'natural', 'monomial', 'power', 'polynomial', 'poly', 'trig_odd_residual')

def _meta_norm(*keys):
    for key in keys:
        val = metadata.get(key, '').strip()
        if val:
            return val.lower().replace('-', '_').replace(' ', '_')
    return ''

def _meta_truthy(*keys):
    return _meta_norm(*keys) in ('1', 'true', 'yes', 'y', 'q')

def _basis_coeff_form():
    form = _meta_norm(
        'basis_coefficients',
        'basis_coeffs',
        'basis_coefficient_basis',
        'basis_coeff_form',
        'coefficient_basis',
        'coeff_basis',
        'coefficients_basis',
    )
    q_forms = {
        'q',
        'q_abs',
        'q_abs_x',
        'q_absx',
        'q_coefficients',
        'q_coeffs',
        'factored',
        'factored_q',
        'odd_factored',
        'odd_factored_q',
        'odd_factor_q',
        'quotient',
    }
    p_forms = {
        '',
        'p',
        'p_abs',
        'p_abs_x',
        'p_absx',
        'p_coefficients',
        'p_coeffs',
        'expanded',
        'expanded_p',
        'u_times_q',
        'abs_x_times_q',
        'abs_times_q',
    }
    if (
        basis_kind_norm in ('odd_factored_q', 'odd_factor_q') or
        form in q_forms or
        _meta_truthy('basis_coefficients_are_q', 'coefficients_are_q', 'basis_q_coefficients')
    ):
        return form, True, True
    if form in p_forms:
        return form, False, True
    return form, False, False

basis_coeff_form, basis_coeffs_are_q, basis_coeff_form_supported = _basis_coeff_form()

has_abs_sign_basis = (not is_rational) and basis_kind_norm in (
    'signed_abs_poly',
    'signed_abs',
    'abs_signed_poly',
    'odd_factored',
    'odd_factored_poly',
    'odd_factored_q',
    'odd_factor_q',
)
has_affine_even_basis = (not is_rational) and basis_kind_norm in (
    'affine_even',
    'affine_even_poly',
    'affine_even_cdf',
)
if has_abs_sign_basis:
    is_odd_factored = basis_kind_norm.startswith('odd_factored') or basis_kind_norm == 'odd_factor_q'
    odd_factored_q_coeffs = is_odd_factored and basis_coeffs_are_q
    if is_odd_factored and not basis_coeff_form_supported:
        raise ValueError(
            f'unsupported odd_factored coefficient form {basis_coeff_form!r}; '
            'expected expanded/P(abs(x)) or quotient/Q(abs(x)) metadata'
        )

    basis_comment = 'basis_kind=signed_abs_poly: y = copysign(clamp(P(abs(x))), x)'
    if is_odd_factored:
        if odd_factored_q_coeffs:
            basis_comment = 'basis_kind=odd_factored: y = copysign(post(abs(x) * Q(abs(x))), x); CSV coeffs are Q'
        else:
            basis_comment = 'basis_kind=odd_factored: y = copysign(post(P(abs(x))), x); CSV coeffs are expanded P(u)=u*Q(u)'

    eval_basis_macro = (
        f'\n// {basis_comment}\n'
        '#define BASIS_SIGNED_ABS_POLY\n'
        '#define BASIS_INPUT_ABS_X\n'
        '#define BASIS_POST_SIGN_X\n'
    )
    if is_odd_factored:
        eval_basis_macro += '#define BASIS_ODD_FACTORED\n'
    if odd_factored_q_coeffs:
        eval_basis_macro += '#define BASIS_MUL_ABS_X_BEFORE_POST\n'
    if basis_clamp_max:
        eval_basis_macro += (
            '#define BASIS_CLAMP_MAX\n'
            f'constexpr float BASIS_CLAMP_MAX_VALUE = {float(basis_clamp_max):.16e}f;\n'
        )
    coeff_label = 'Q(abs(x))' if odd_factored_q_coeffs else 'P(abs(x))'
    clamp_label = basis_clamp_max if basis_clamp_max else 'none'
    print(f'Basis: {basis_kind_norm} coeffs={coeff_label} clamp_max={clamp_label}')
elif has_affine_even_basis:
    def _meta_float(default, *keys):
        for key in keys:
            val = metadata.get(key, '').strip()
            if val:
                return float(val)
        return float(default)

    affine_bias = _meta_float(0.0, 'basis_bias', 'affine_bias', 'basis_affine_bias')
    affine_scale = _meta_float(1.0, 'basis_scale_x', 'affine_scale', 'basis_affine_scale')
    affine_even_scale = _meta_float(1.0, 'affine_even_scale', 'basis_even_scale', 'even_scale')
    left_tail_mode = _meta_norm('basis_left_tail_mode', 'left_tail_mode')
    right_tail_mode = _meta_norm('basis_right_tail_mode', 'right_tail_mode')
    left_tail_zero = metadata.get(
        'basis_left_tail_max',
        metadata.get('left_tail_zero', metadata.get('basis_left_tail_zero', '')),
    ).strip() if left_tail_mode in ('', 'zero') else ''
    right_tail_identity = metadata.get(
        'basis_right_tail_min',
        metadata.get('right_tail_identity', metadata.get('basis_right_tail_identity', '')),
    ).strip() if right_tail_mode in ('', 'identity') else ''
    if left_tail_mode not in ('', 'zero'):
        raise ValueError(f'unsupported affine_even left tail mode {left_tail_mode!r}')
    if right_tail_mode not in ('', 'identity'):
        raise ValueError(f'unsupported affine_even right tail mode {right_tail_mode!r}')

    eval_basis_macro = (
        f'\n// basis_kind={basis_kind_norm}: y = bias + scale*x + even_scale*abs(x)*P(abs(x))\n'
        '#define BASIS_AFFINE_EVEN\n'
        '#define BASIS_INPUT_ABS_X\n'
        f'#define BASIS_AFFINE_BIAS {clamp(affine_bias):.10e}f\n'
        f'#define BASIS_AFFINE_SCALE {clamp(affine_scale):.10e}f\n'
        f'#define BASIS_AFFINE_EVEN_SCALE {clamp(affine_even_scale):.10e}f\n'
    )
    if left_tail_zero:
        eval_basis_macro += (
            '#define BASIS_LEFT_TAIL_ZERO\n'
            f'#define BASIS_LEFT_TAIL_ZERO_THRESHOLD {clamp(float(left_tail_zero)):.10e}f\n'
        )
    if right_tail_identity:
        eval_basis_macro += (
            '#define BASIS_RIGHT_TAIL_IDENTITY\n'
            f'#define BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD {clamp(float(right_tail_identity)):.10e}f\n'
        )
    left_tail_zero_label = left_tail_zero or 'none'
    right_tail_identity_label = right_tail_identity or 'none'
    print(
        f'Basis: {basis_kind_norm} affine_bias={affine_bias:.6g} '
        f'affine_scale={affine_scale:.6g} even_scale={affine_even_scale:.6g} '
        f'left_tail_zero={left_tail_zero_label} '
        f'right_tail_identity={right_tail_identity_label}'
    )
elif basis_kind_norm not in plain_basis_kinds:
    raise ValueError(f'unsupported basis_kind={basis_kind!r}; refusing to generate a normal polynomial kernel')

# Detect parity from coefficient values
poly_parity_macro = ''
threshold = 1e-30
if has_abs_sign_basis or has_affine_even_basis:
    # Do not infer parity: abs/sign basis coefficients are dense in abs(x),
    # not expanded odd/even coefficients in x.
    pass
elif is_rational and num_degree >= 2:
    # Rational parity: check num (odd-index) and den (even-index) separately
    num_even_zero = True  # even-index num coeffs zero → odd numerator
    den_odd_zero = True   # odd-index den coeffs zero → even denominator
    ncps = num_degree + 1
    dcps = den_degree + 1
    for s in range(num_segments):
        seg_num = num_coefficients[s * ncps : (s + 1) * ncps]
        seg_den = den_coefficients[s * dcps : (s + 1) * dcps]
        for i in range(ncps):
            if i % 2 == 0 and abs(seg_num[i]) > threshold:
                num_even_zero = False
        for i in range(dcps):
            if i % 2 == 1 and abs(seg_den[i]) > threshold:
                den_odd_zero = False
    if num_even_zero and den_odd_zero:
        poly_parity_macro = '\n// Rational parity: odd num / even den -> x^2-Horner\n#define RATIONAL_NUM_PARITY_ODD\n#define RATIONAL_DEN_PARITY_EVEN\n'
        print(f'Rational parity: odd num / even den -> x^2-Horner enabled')
elif not is_rational and degree >= 2:
    # Polynomial parity
    cps = degree + 1
    seg_coeffs_list = []
    for s in range(num_segments):
        seg_coeffs_list.append(coefficients[s * cps : (s + 1) * cps])

    even_all_zero = True
    odd_all_zero = True
    for seg in seg_coeffs_list:
        for i in range(degree + 1):
            if i % 2 == 0 and abs(seg[i]) > threshold:
                even_all_zero = False
            if i % 2 == 1 and abs(seg[i]) > threshold:
                odd_all_zero = False

    if even_all_zero:
        poly_parity_macro = '\n// Polynomial parity: odd function (c0=c2=c4=...=0) -> x^2-Horner\n#define POLY_PARITY_ODD\n'
        print(f'Polynomial parity: ODD (c0=c2=c4=...=0) -> x^2-Horner enabled')
    elif odd_all_zero:
        poly_parity_macro = '\n// Polynomial parity: even function (c1=c3=c5=...=0) -> x^2-Horner\n#define POLY_PARITY_EVEN\n'
        print(f'Polynomial parity: EVEN (c1=c3=c5=...=0) -> x^2-Horner enabled')

# Check for range reduction
rr_macro = ''
rr_method = metadata.get('range_reduction_method', '')
# Standalone evaluator tags: exponent_alu (exp2/log2/pow) and the FIRST-CLASS
# newton_root method. newton_root is no longer nested under exponent_alu in the
# fitter tag; accept both the first-class 'newton_root' and the legacy
# 'exponent_alu_newton_root' for back-compat. All standalone paths share the
# constant-pool hoisting + bypass the cascade.
if rr_method.startswith('exponent_alu_') or rr_method == 'newton_root':
    # Hardware-exponent-ALU backend (exp2 / log2 / pow) or newton_root. The
    # fitted poly (exp/log/pow) lives in the first segment's coefficients; the
    # kernel owns the exman/exexp/setexp decompose + scale fold + recombine.
    # newton_root fits no poly (magic-seed + Newton). Disable adaptive-degree /
    # parity (cascade is bypassed) to keep defines clean.
    degree_macros = ''
    poly_parity_macro = ''
    kind = 'newton_root' if rr_method == 'newton_root' else rr_method[len('exponent_alu_'):]
    cps = degree + 1
    seg0 = coefficients[0:cps]

    # ---- GENERIC constant-pool hoisting (data-driven, all kinds, any degree) ----
    # Collect every loop-invariant constexpr the kernel reads for this kind, ranked
    # by reuse (touched-every-element first). Top-3 -> vConstFloatPrgm0/1/2; the next
    # HOIST_BUDGET -> pre-loop hoisted vFloat LREGs; anything beyond -> in-body
    # literals, and we LOG the count (never silently dropped).
    PRGM_SLOTS = 3
    HOIST_BUDGET = 5  # iteration-invariant LREGs the compiler can keep across the loop
    def _build_pool(kind, degree):
        pool = []  # (name, reuse_rank) high rank = hotter
        if kind == 'exp2':
            pool.append(('MULT(1/ln2)', 100))
            for k in range(degree, -1, -1):
                pool.append((f'c{k}', 50 + k))     # higher coeffs slightly hotter (Horner head)
            pool.append(('clamp255', 40))
        elif kind == 'log2':
            pool.append(('LOG_HW_SCALE', 100))
            for k in range(degree, -1, -1):
                pool.append((f'c{k}', 50 + k))
        elif kind == 'pow':
            pool.append(('SQRT2', 90))
            pool.append(('round_magic', 60))
            for k in range(degree, -1, -1):
                pool.append((f'c{k}', 50 + k))
        elif kind == 'newton_root':
            # Newton path: loop-invariant seed/correction constants in prgm const
            # regs where the selected algorithm uses them. No polynomial coeffs
            # are touched on this path.
            pool.append(('NEWTON_MAGIC', 100))
            pool.append(('NEWTON_C0', 95))
            pool.append(('NEWTON_C1', 90))
            pool.append(('NEWTON_C2', 80))
        pool.sort(key=lambda t: -t[1])
        return [n for n, _ in pool]
    _pool = _build_pool(kind, degree)
    _budget = PRGM_SLOTS + HOIST_BUDGET
    _spilled = _pool[_budget:]
    # HW_PRELOAD_DISABLE=1 lets A/B measure the non-preload baseline on identical HW.
    hw_preload_macro = '' if os.environ.get('HW_PRELOAD_DISABLE') == '1' else '#define HW_PRELOAD\n'
    if _spilled:
        print(f'// HW_PRELOAD constant-pool: {len(_pool)} constants, {len(_spilled)} spilled to in-body load: {_spilled}')
        hw_preload_macro += f'// HW_PRELOAD_SPILL: {len(_spilled)} constants spilled to in-body load: {_spilled}\n'
    else:
        print(f'// HW_PRELOAD constant-pool: {len(_pool)} constants, 0 spilled (all in prgm/hoist budget)')

    if kind == 'exp2':
        # Full degree-N natural [0,1) coeffs; kernel normalizes f then Horner.
        # Honor the log2-domain multiplier + optional compose post-transform.
        mult = metadata.get('expalu_log2_multiplier', '1.4426950408889634')
        compose = metadata.get('expalu_compose', '').strip()
        coeff_str = ', '.join(f'{clamp(v):.10e}f' for v in seg0)
        compose_macro = ''
        if compose == 'sigmoid':
            compose_macro = '#define EXP_HW_COMPOSE_SIGMOID\n'
        elif compose in ('sigmoid_product', 'silu', 'swish'):
            compose_macro = '#define EXP_HW_COMPOSE_SIGMOID_PRODUCT\n'
        elif compose == 'minus_one':
            compose_macro = '#define EXP_HW_COMPOSE_MINUS_ONE\n'
        if compose in ('sigmoid', 'sigmoid_product', 'silu', 'swish') and hw_preload_macro:
            print('// HW_PRELOAD disabled for sigmoid compose: reciprocal temporaries exceed SFPI reload budget')
            hw_preload_macro = ''
        # TTNN exp_21f instruction-count cuts (see piecewise_generic.cpp):
        #   EXP_HW_FUSED       : fold 2^-23 normalize into coeffs (removes 1 SFPMUL).
        #                        Safe unless a scaled coeff c[DEG]*2^-23*DEG underflows fp32.
        #   EXP_HW_BARE_SETEXP : drop the pe correction (removes 1 SFPEXEXP + 2 SFPIADD).
        #                        Safe when g(f) in [1,2) on f in [0,1), i.e. c0 >= 1.
        c0v = float(seg0[0]) if len(seg0) else 0.0
        c_topv = float(seg0[degree]) if len(seg0) > degree else 0.0
        scaled_topv = abs(c_topv) * (2.0 ** (-23.0 * degree))
        fused_safe = scaled_topv == 0.0 or scaled_topv >= 1e-37
        bare_safe = c0v >= 1.0
        fused_macro = '#define EXP_HW_FUSED\n' if fused_safe else ''
        bare_macro = '#define EXP_HW_BARE_SETEXP\n' if bare_safe else ''
        rr_macro = (
            '\n// eval_method: exponent_alu / exp2 (exman/exexp/setexp), natural [0,1) coeffs. STANDALONE\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_EXPONENT_ALU\n'
            '#define EVAL_METHOD_EXPONENT_ALU\n'
            '#define EXPONENT_ALU_EXP2\n'
            f'#define EXP_HW_MULT {clamp(float(mult)):.10e}f\n'
            f'{compose_macro}'
            f'{fused_macro}'
            f'{bare_macro}'
            f'{hw_preload_macro}'
            f'constexpr uint32_t EXP_HW_DEGREE = {degree};\n'
            f'constexpr float EXP_HW_COEFFS[] = {{{coeff_str}}};\n'
        )
        print(f'Range reduction: HW exponent-ALU exp2 (degree {degree}, mult {mult}, compose {compose or \"none\"}, fused {\"ON\" if fused_safe else \"off\"}, bare_setexp {\"ON\" if bare_safe else \"off\"})')
    elif kind == 'log2':
        scale = metadata.get('expalu_log_scale', metadata.get('log_scale', '1.0'))
        basis = metadata.get('expalu_log2_basis', 'natural')
        coeff_str = ', '.join(f'{clamp(v):.10e}f' for v in seg0)
        log_basis_macro = '#define LOG_HW_BASIS_M_MINUS_1\n' if basis == 'm_minus_1' else ''
        # log1p decomposes (x + 1) before log2 -> expalu_input_offset = 1.0.
        offset = float(metadata.get('expalu_input_offset', '0.0') or '0.0')
        offset_macro = f'#define LOG_HW_INPUT_OFFSET {clamp(offset):.10e}f\n' if offset != 0.0 else ''
        rr_macro = (
            '\n// eval_method: exponent_alu / log2 (exexp -> e, exman -> m), natural coeffs. STANDALONE\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_EXPONENT_ALU\n'
            '#define EVAL_METHOD_EXPONENT_ALU\n'
            '#define EXPONENT_ALU_LOG2\n'
            f'{log_basis_macro}'
            f'{offset_macro}'
            f'{hw_preload_macro}'
            f'constexpr uint32_t LOG_HW_DEGREE = {degree};\n'
            f'constexpr float LOG_HW_COEFFS[] = {{{coeff_str}}};\n'
            f'#define LOG_HW_SCALE {scale}f\n'
        )
        print(f'Range reduction: HW exponent-ALU log2 (degree {degree}, scale {scale}, basis {basis}, offset {offset})')
    elif kind == 'pow':
        coeff_str = ', '.join(f'{clamp(v):.10e}f' for v in seg0)
        # root order N, optional final reciprocal (rsqrt), per-r scale constants.
        root_n = int(float(metadata.get('expalu_root_n', '2') or '2'))
        recip = str(metadata.get('expalu_reciprocal', 'False')).strip().lower() in ('true', '1')
        recip_macro = '#define POW_HW_RECIPROCAL\n' if recip else ''
        scale_macros = f'#define POW_HW_ROOT_N {root_n}\n'
        for r in range(root_n):
            key = f'expalu_pow_scale_c{r}'
            if key in metadata:
                scale_macros += f'#define POW_HW_SCALE_C{r} {clamp(float(metadata[key])):.10e}f\n'
        rr_macro = (
            '\n// eval_method: exponent_alu / pow/root_N (exexp -> e, exman -> m), natural [1,2) coeffs. STANDALONE\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_EXPONENT_ALU\n'
            '#define EVAL_METHOD_EXPONENT_ALU\n'
            '#define EXPONENT_ALU_POW\n'
            f'{scale_macros}'
            f'{recip_macro}'
            f'{hw_preload_macro}'
            f'constexpr uint32_t POW_HW_DEGREE = {degree};\n'
            f'constexpr float POW_HW_COEFFS[] = {{{coeff_str}}};\n'
        )
        print(f'Range reduction: HW exponent-ALU pow (degree {degree}, root_n {root_n}, reciprocal {recip})')
    elif kind == 'newton_root':
        # Metadata-declared magic-root integer root. The fitter owns all
        # constants and the algorithm subtag; the kernel does not dispatch by
        # activation name.
        def parse_float_literal(value):
            text = str(value).strip()
            try:
                return float.fromhex(text)
            except ValueError:
                return float(text)

        root_n = int(float(metadata.get('newton_root_n', metadata.get('expalu_root_n', '2')) or '2'))
        recip = str(metadata.get('newton_root_reciprocal',
                                 metadata.get('expalu_reciprocal', 'False'))).strip().lower() in ('true', '1')
        iters = int(float(metadata.get('newton_root_iters', '3') or '3'))
        algorithm = metadata.get(
            'newton_root_algorithm',
            metadata.get(
                'magic_root_algorithm',
                metadata.get('newton_root_lowering', metadata.get('lowering', ''))
            )
        ).strip().lower()
        cbrt_magic_aliases = (
            'cbrt_magic', 'magic_cbrt', 'native_cbrt', 'moroz_cbrt', 'cube_root_native',
            'cbrt_magic_root', 'magic_root_cbrt', 'cbrt-magic-root', 'magic-root-cbrt'
        )
        use_cbrt_magic = root_n == 3 and algorithm in cbrt_magic_aliases
        if use_cbrt_magic:
            magic = metadata.get('newton_root_magic', '0x548c2b4b')
            c0 = parse_float_literal(metadata.get('newton_root_c0', '0x1.c09806p0'))
            c1 = parse_float_literal(metadata.get('newton_root_c1', '-0x1.403e6cp0'))
            c2 = parse_float_literal(metadata.get('newton_root_c2', '0x1.04cdb2p-1'))
        else:
            magic = metadata.get('newton_root_magic', '0x5f1110a0')
            c0 = parse_float_literal(metadata.get('newton_root_c0', '0.0'))
            c1 = parse_float_literal(metadata.get('newton_root_c1', '2.2825186'))
            c2 = parse_float_literal(metadata.get('newton_root_c2', '2.2533049'))
        recip_macro = '#define NEWTON_ROOT_RECIPROCAL\n' if recip else ''
        algorithm_macro = '#define NEWTON_ROOT_ALGORITHM_CBRT_MAGIC\n' if use_cbrt_magic else ''
        rr_macro = (
            '\n// eval_method: newton_root (metadata-declared magic-root, NO poly fit). FIRST-CLASS standalone.\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_NEWTON_ROOT\n'
            '#define EVAL_METHOD_NEWTON_ROOT\n'
            f'#define NEWTON_ROOT_MAGIC {magic}\n'
            f'#define NEWTON_ROOT_C0 {c0:.10e}f\n'
            f'#define NEWTON_ROOT_C1 {c1:.10e}f\n'
            f'#define NEWTON_ROOT_C2 {c2:.10e}f\n'
            f'#define NEWTON_ROOT_N {root_n}\n'
            f'#define NEWTON_ROOT_ITERS {iters}\n'
            f'{recip_macro}'
            f'{algorithm_macro}'
            # POLY_DEGREE template arg is unused on this path but the kernel
            # signature still takes it; the LUT/coeffs are ignored.
        )
        print(f'Range reduction: metadata magic-root (algorithm {algorithm or \"default\"}, magic {magic}, '
              f'c0 {c0}, c1 {c1}, c2 {c2}, root_n {root_n}, reciprocal {recip}, iters {iters})')
    else:
        print(f'WARNING: unknown exponent_alu kind {kind}')
elif rr_method == 'exp':
    rr_macro = '\n// eval_method: reduced_poly / exp\n#define TT_ACT_EVAL_KIND TT_ACT_EVAL_REDUCED_POLY\n#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_EXP\n'
    print(f'Range reduction: exp')
elif rr_method == 'trig':
    rr_macro = '\n// eval_method: reduced_poly / trig\n#define TT_ACT_EVAL_KIND TT_ACT_EVAL_REDUCED_POLY\n#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_TRIG\n'
    print(f'Range reduction: trig')
elif rr_method in ('trig_residual', 'trig_standalone'):
    if is_rational:
        raise ValueError('trig_residual standalone evaluator requires polynomial coefficients')
    if num_segments != 1:
        raise ValueError('trig_residual standalone evaluator currently requires one segment')
    trig_phase = _meta_norm('trig_phase', 'trig_reduction_phase', 'range_reduction_phase')
    cosine_phases = ('cosine_pi2_odd', 'cos_pi2_odd', 'cosine_shifted_sine', 'cos_shifted_sine')
    sine_phases = ('sine_pi_odd', 'sin_pi_odd')
    if trig_phase not in cosine_phases + sine_phases:
        raise ValueError(f'trig_residual requires supported trig_phase, got {trig_phase!r}')
    cps = degree + 1
    seg0 = coefficients[0:cps]
    coeff_str = ', '.join(f'{clamp(v):.10e}f' for v in seg0)
    c1_is_one_macro = '#define TRIG_RESIDUAL_C1_IS_ONE\n' if len(seg0) > 1 and abs(float(seg0[1]) - 1.0) < 1e-12 else ''
    trig_phase_macro = 'TRIG_RESIDUAL_PHASE_COSINE_PI2_ODD' if trig_phase in cosine_phases else 'TRIG_RESIDUAL_PHASE_SINE_PI_ODD'
    degree_macros = ''
    poly_parity_macro = ''
    rr_macro = (
        '\n// eval_method: trig_residual standalone, odd residual polynomial. STANDALONE\n'
        '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_TRIG_RESIDUAL\n'
        '#define EVAL_METHOD_TRIG_RESIDUAL\n'
        f'#define {trig_phase_macro}\n'
        f'{c1_is_one_macro}'
        f'constexpr uint32_t TRIG_RESIDUAL_DEGREE = {degree};\n'
        f'constexpr float TRIG_RESIDUAL_COEFFS[] = {{{coeff_str}}};\n'
    )
    print(f'Range reduction: trig_residual standalone (phase {trig_phase}, degree {degree})')
elif rr_method == 'log':
    expand_const = metadata.get('log_ln2_constant', '0.6931471805599453')
    rr_macro = (f'\n// eval_method: reduced_poly / log\n#define TT_ACT_EVAL_KIND TT_ACT_EVAL_REDUCED_POLY\n#define EVAL_METHOD_REDUCED_POLY\n'
                f'#define REDUCE_LOG\n#define LOG_EXPAND_CONSTANT {expand_const}f\n')
    print(f'Range reduction: log (expand_const={expand_const})')
elif rr_method == 'tan':
    if not is_rational and num_segments == 1:
        cps = degree + 1
        seg0 = coefficients[0:cps]
        coeff_str = ', '.join(f'{clamp(v):.10e}f' for v in seg0)
        degree_macros = ''
        poly_parity_macro = ''
        rr_macro = (
            '\n// eval_method: tan standalone, reduced odd polynomial plus quadrant reciprocal. STANDALONE\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_TAN_STANDALONE\n'
            '#define EVAL_METHOD_TAN_STANDALONE\n'
            f'constexpr uint32_t TAN_STANDALONE_DEGREE = {degree};\n'
            f'constexpr float TAN_STANDALONE_COEFFS[] = {{{coeff_str}}};\n'
        )
        print(f'Range reduction: tan standalone (degree {degree})')
    else:
        rr_macro = '\n// eval_method: reduced_poly / tan\n#define TT_ACT_EVAL_KIND TT_ACT_EVAL_REDUCED_POLY\n#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_TAN\n'
        print(f'Range reduction: tan')
elif rr_method == 'cbrt':
    rr_macro = '\n// eval_method: reduced_poly / cbrt\n#define TT_ACT_EVAL_KIND TT_ACT_EVAL_REDUCED_POLY\n#define EVAL_METHOD_REDUCED_POLY\n#define REDUCE_CBRT\n'
    print(f'Range reduction: cbrt')

# Detect asymptotic factoring from CSV columns
DOMINANT_FACTOR_MAP = {
    '-exp(-x^2/2) / sqrt(2*pi)': ('EXP_QUADRATIC', -0.5, -1.0 / math.sqrt(2 * math.pi)),
    'exp(-x^2/2) / sqrt(2*pi)': ('EXP_QUADRATIC', -0.5, 1.0 / math.sqrt(2 * math.pi)),
    'exp(x)': ('EXP_LINEAR', 1.0, 1.0),
    'exp(-x)': ('EXP_LINEAR', -1.0, 1.0),
    '-exp(-x)': ('EXP_LINEAR', -1.0, -1.0),
    'x * exp(x)': ('X_EXP_LINEAR', 1.0, 1.0),
    'x': ('X', 0.0, 1.0),
}

asymptotic_macro = ''
if any(asymptotic_flags):
    active_factors = [f for f, a in zip(dominant_factors, asymptotic_flags) if a and f]
    unique_factors = set(active_factors)
    if len(unique_factors) == 1 and active_factors[0] in DOMINANT_FACTOR_MAP:
        dom_str = active_factors[0]
        factor_class, arg_scale, output_scale = DOMINANT_FACTOR_MAP[dom_str]
        asymptotic_macro = f'\n// Asymptotic factoring: {dom_str}\n#define ASYMPTOTIC_FACTOR_{factor_class}\n'
        if factor_class != 'X':
            asymptotic_macro += f'constexpr float ASYMPTOTIC_EXP_ARG_SCALE = {arg_scale:.16e}f;\n'
        asymptotic_macro += f'constexpr float ASYMPTOTIC_SCALE = {output_scale:.16e}f;\n'
        # Determine bound: left tail (x < bound) or right tail (x > bound)
        first_asym = next(i for i, a in enumerate(asymptotic_flags) if a)
        last_asym = next(i for i in range(num_segments - 1, -1, -1) if asymptotic_flags[i])
        if first_asym == 0 and last_asym < num_segments - 1:
            bound = boundaries[last_asym + 1]
            asymptotic_macro += f'constexpr float ASYMPTOTIC_UPPER_BOUND = {bound:.16e}f;\n'
            print(f'Asymptotic factoring: {dom_str} (left tail, x < {bound})')
        elif last_asym == num_segments - 1 and first_asym > 0:
            bound = boundaries[first_asym]
            asymptotic_macro += f'constexpr float ASYMPTOTIC_LOWER_BOUND = {bound:.16e}f;\n'
            print(f'Asymptotic factoring: {dom_str} (right tail, x > {bound})')
        elif first_asym == 0 and last_asym == num_segments - 1:
            asymptotic_macro += 'constexpr float ASYMPTOTIC_UPPER_BOUND = 1.0e38f;\n'
            print(f'Asymptotic factoring: {dom_str} (all segments)')
        else:
            asymptotic_macro = ''
            print(f'WARNING: non-contiguous asymptotic segments, skipping')
    elif len(unique_factors) > 1:
        print(f'WARNING: mixed dominant factors not supported: {unique_factors}')

# Whole-function algebraic collapses, in priority order. These are intentionally
# coefficient-pattern recognizers, not activation-name branches:
#   1. affine/identity:        y = c0 + c1*x
#   2. clamped affine:         y = min(max(c0 + c1*x, low), high)
#
# More granular segment_kind lowering belongs inside the cascade family later;
# these collapses bypass the segment selector only when the whole CSV proves the
# simpler algebra exactly.
affine_macro = ''
if (not is_rational) and (not has_abs_sign_basis) and (not has_affine_even_basis) and rr_method in ('', 'none') and num_segments == 1:
    seg0_coeffs = coefficients[0:degree + 1]
    higher_zero = all(c == 0.0 for c in seg0_coeffs[2:])
    if higher_zero:
        c0 = float(seg0_coeffs[0]) if len(seg0_coeffs) >= 1 else 0.0
        c1 = float(seg0_coeffs[1]) if len(seg0_coeffs) >= 2 else 0.0
        if c0 == 0.0 and c1 == 1.0:
            affine_macro = '\n// eval_method: identity. fit is y = x. Pure tile copy, no SFPU eval.\n#define TT_ACT_EVAL_KIND TT_ACT_EVAL_IDENTITY\n#define EVAL_METHOD_AFFINE_COLLAPSE\n#define AFFINE_COLLAPSE\n#define AFFINE_IDENTITY\n'
            print('AFFINE COLLAPSE: identity (c0=0, c1=1) -> pure-copy bypass (no SFPU eval)')
        else:
            affine_macro = (
                '\n// eval_method: affine. fit is y = c0 + c1*x. One SFPMAD per element.\n'
                '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_AFFINE\n'
                '#define EVAL_METHOD_AFFINE_COLLAPSE\n'
                '#define AFFINE_COLLAPSE\n'
                f'#define TT_ACT_AFFINE_B {clamp(c0):.10e}f\n'
                f'#define TT_ACT_AFFINE_A {clamp(c1):.10e}f\n'
                '#define AFFINE_C0 TT_ACT_AFFINE_B\n'
                '#define AFFINE_C1 TT_ACT_AFFINE_A\n'
            )
            print(f'AFFINE COLLAPSE: y = {c0:.6g} + {c1:.6g}*x -> single SFPMAD bypass')

# Clamped affine collapse: same whole-function policy as affine collapse, with
# either clamp bound optional. Constant regions before/after the affine region
# become lower/upper clamps based on coefficient slope and segment order.
clamped_affine_macro = ''
if (not is_rational) and (not has_abs_sign_basis) and (not has_affine_even_basis) and rr_method in ('', 'none') and not affine_macro and num_segments >= 2:
    cps = degree + 1
    segs = []
    all_affine = True
    tol = 1.0e-5
    for s in range(num_segments):
        coeff = coefficients[s * cps:(s + 1) * cps]
        if any(abs(c) > tol for c in coeff[2:]):
            all_affine = False
            break
        c0 = float(coeff[0]) if len(coeff) >= 1 else 0.0
        c1 = float(coeff[1]) if len(coeff) >= 2 else 0.0
        segs.append((boundaries[s], boundaries[s + 1], c0, c1))

    if all_affine:
        candidates = [seg for seg in segs if abs(seg[3]) > tol]

        def eval_piece(seg, x):
            return seg[2] + seg[3] * x

        for cand in candidates:
            cand_lo, cand_hi, c0, c1 = cand
            lo_values = []
            hi_values = []
            for seg in segs:
                if abs(seg[3]) > tol:
                    continue
                before = seg[1] <= cand_lo + tol
                after = seg[0] >= cand_hi - tol
                if c1 >= 0.0:
                    if before:
                        lo_values.append(seg[2])
                    elif after:
                        hi_values.append(seg[2])
                else:
                    if before:
                        hi_values.append(seg[2])
                    elif after:
                        lo_values.append(seg[2])
            lo = lo_values[0] if lo_values else None
            hi = hi_values[0] if hi_values else None
            if lo_values and any(abs(v - lo) > tol for v in lo_values):
                continue
            if hi_values and any(abs(v - hi) > tol for v in hi_values):
                continue
            if lo is not None and hi is not None and lo > hi + tol:
                continue

            def clamp_affine(x):
                y = c0 + c1 * x
                if lo is not None:
                    y = max(lo, y)
                if hi is not None:
                    y = min(hi, y)
                return y

            ok = True
            for seg in segs:
                for x in (seg[0], (seg[0] + seg[1]) * 0.5, seg[1]):
                    if abs(eval_piece(seg, x) - clamp_affine(x)) > 5.0e-4:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                clamped_affine_macro = (
                    '\n// eval_method: clamped_affine. fit is y = min(max(c0 + c1*x, min), max).\n'
                    '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_CLAMPED_AFFINE\n'
                    '#define EVAL_METHOD_CLAMPED_AFFINE_COLLAPSE\n'
                    '#define CLAMPED_AFFINE_COLLAPSE\n'
                    f'#define TT_ACT_CLAMPED_AFFINE_B {clamp(c0):.10e}f\n'
                    f'#define TT_ACT_CLAMPED_AFFINE_A {clamp(c1):.10e}f\n'
                    '#define CLAMPED_AFFINE_C0 TT_ACT_CLAMPED_AFFINE_B\n'
                    '#define CLAMPED_AFFINE_C1 TT_ACT_CLAMPED_AFFINE_A\n'
                )
                if lo is not None:
                    clamped_affine_macro += f'#define CLAMPED_AFFINE_HAS_MIN\n#define TT_ACT_CLAMPED_AFFINE_MIN {clamp(lo):.10e}f\n#define CLAMPED_AFFINE_MIN TT_ACT_CLAMPED_AFFINE_MIN\n'
                if hi is not None:
                    clamped_affine_macro += f'#define CLAMPED_AFFINE_HAS_MAX\n#define TT_ACT_CLAMPED_AFFINE_MAX {clamp(hi):.10e}f\n#define CLAMPED_AFFINE_MAX TT_ACT_CLAMPED_AFFINE_MAX\n'
                low_label = '-inf' if lo is None else f'{lo:.6g}'
                high_label = '+inf' if hi is None else f'{hi:.6g}'
                print(
                    f'CLAMPED AFFINE COLLAPSE: y = clamp({c0:.6g} + {c1:.6g}*x, '
                    f'{low_label}, {high_label}) -> SFPMAD + min/max bypass'
                )
                break

# Threshold identity collapse: piecewise y=x outside a dead zone and y=0 inside.
# This is the algebraic shape behind shrink/threshold-style functions. The
# threshold owns equality, so x == +/-threshold returns zero instead of identity.
threshold_select_macro = ''
if (not is_rational) and (not has_abs_sign_basis) and (not has_affine_even_basis) and rr_method in ('', 'none') and not affine_macro and not clamped_affine_macro and num_segments == 3:
    cps = degree + 1
    tol = 1.0e-5
    segs = [coefficients[s * cps:(s + 1) * cps] for s in range(num_segments)]
    def _is_identity(coeff):
        return abs((coeff[0] if len(coeff) > 0 else 0.0)) <= tol and abs((coeff[1] if len(coeff) > 1 else 0.0) - 1.0) <= tol and all(abs(c) <= tol for c in coeff[2:])
    def _is_zero(coeff):
        return all(abs(c) <= tol for c in coeff)
    lo_thr = abs(boundaries[1])
    hi_thr = abs(boundaries[2])
    if _is_identity(segs[0]) and _is_zero(segs[1]) and _is_identity(segs[2]) and abs(lo_thr - hi_thr) <= tol:
        threshold_select_macro = (
            '\n// eval_method: threshold_identity. y=x outside |x|<=threshold, else 0.\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_THRESHOLD_IDENTITY\n'
            '#define EVAL_METHOD_THRESHOLD_IDENTITY_SELECT\n'
            '#define THRESHOLD_IDENTITY_SELECT\n'
            f'#define TT_ACT_THRESHOLD_LAMBDA {clamp(hi_thr):.10e}f\n'
            '#define THRESHOLD_IDENTITY_LAMBDA TT_ACT_THRESHOLD_LAMBDA\n'
        )
        print(f'THRESHOLD IDENTITY SELECT: y=x when |x|>{hi_thr:.6g}, else 0 -> threshold bypass')

# Absolute-value collapse: y = abs(x), recognized from two affine pieces.
abs_value_macro = ''
if (not is_rational) and (not has_abs_sign_basis) and (not has_affine_even_basis) and rr_method in ('', 'none') and not affine_macro and not clamped_affine_macro and not threshold_select_macro and num_segments == 2:
    cps = degree + 1
    tol = 1.0e-5
    segs = [coefficients[s * cps:(s + 1) * cps] for s in range(num_segments)]
    def _coeff(coeff, idx):
        return float(coeff[idx]) if len(coeff) > idx else 0.0
    def _is_affine(coeff, c0, c1):
        return abs(_coeff(coeff, 0) - c0) <= tol and abs(_coeff(coeff, 1) - c1) <= tol and all(abs(c) <= tol for c in coeff[2:])
    if _is_affine(segs[0], 0.0, -1.0) and _is_affine(segs[1], 0.0, 1.0) and abs(boundaries[1]) <= tol:
        abs_value_macro = (
            '\n// eval_method: abs_value. y=abs(x).\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_ABS_VALUE\n'
            '#define EVAL_METHOD_ABS_VALUE\n'
            '#define ABS_VALUE\n'
        )
        print('ABS VALUE: y=abs(x) -> sign-clear bypass')

# Soft-threshold collapse: y=sign(x)*(abs(x)-lambda) outside |x|<=lambda.
threshold_softshift_macro = ''
if (not is_rational) and (not has_abs_sign_basis) and (not has_affine_even_basis) and rr_method in ('', 'none') and not affine_macro and not clamped_affine_macro and not threshold_select_macro and not abs_value_macro and num_segments == 3:
    cps = degree + 1
    tol = 1.0e-5
    segs = [coefficients[s * cps:(s + 1) * cps] for s in range(num_segments)]
    def _coeff(coeff, idx):
        return float(coeff[idx]) if len(coeff) > idx else 0.0
    def _is_affine(coeff, c0, c1):
        return abs(_coeff(coeff, 0) - c0) <= tol and abs(_coeff(coeff, 1) - c1) <= tol and all(abs(c) <= tol for c in coeff[2:])
    def _is_zero(coeff):
        return all(abs(c) <= tol for c in coeff)
    lo_thr = abs(boundaries[1])
    hi_thr = abs(boundaries[2])
    if _is_affine(segs[0], lo_thr, 1.0) and _is_zero(segs[1]) and _is_affine(segs[2], -hi_thr, 1.0) and abs(lo_thr - hi_thr) <= tol:
        threshold_softshift_macro = (
            '\n// eval_method: threshold_softshift. y=sign(x)*(abs(x)-lambda) outside dead zone.\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_THRESHOLD_SOFTSHIFT\n'
            '#define EVAL_METHOD_THRESHOLD_SOFTSHIFT\n'
            '#define THRESHOLD_SOFTSHIFT_SELECT\n'
            f'#define TT_ACT_THRESHOLD_LAMBDA {clamp(hi_thr):.10e}f\n'
            '#define THRESHOLD_SOFTSHIFT_LAMBDA TT_ACT_THRESHOLD_LAMBDA\n'
        )
        print(f'THRESHOLD SOFTSHIFT: y=sign(x)*(abs(x)-{hi_thr:.6g}) outside dead zone -> threshold bypass')

# Gated affine-product collapse: y = x * clamp(q0 + q1*x, low, high).
# This captures hardmish/hardswish-like exact piecewise polynomials without
# naming the op.
gated_affine_product_macro = ''
if (not is_rational) and (not has_abs_sign_basis) and (not has_affine_even_basis) and rr_method in ('', 'none') and not affine_macro and not clamped_affine_macro and not threshold_select_macro and not abs_value_macro and not threshold_softshift_macro and num_segments == 3:
    cps = degree + 1
    tol = 1.0e-5
    segs = [coefficients[s * cps:(s + 1) * cps] for s in range(num_segments)]
    def _coeff(coeff, idx):
        return float(coeff[idx]) if len(coeff) > idx else 0.0
    def _is_const(coeff, value):
        return abs(_coeff(coeff, 0) - value) <= tol and all(abs(c) <= tol for c in coeff[1:])
    def _is_identity(coeff):
        return abs(_coeff(coeff, 0)) <= tol and abs(_coeff(coeff, 1) - 1.0) <= tol and all(abs(c) <= tol for c in coeff[2:])
    middle_high_zero = all(abs(c) <= tol for c in segs[1][3:])
    # middle y = x*(q0 + q1*x) => c0=0, c1=q0, c2=q1
    if _is_const(segs[0], 0.0) and _is_identity(segs[2]) and abs(_coeff(segs[1], 0)) <= tol and middle_high_zero:
        q0 = _coeff(segs[1], 1)
        q1 = _coeff(segs[1], 2)
        gated_affine_product_macro = (
            '\n// eval_method: gated_affine_product. y = x * clamp(q0 + q1*x, 0, 1).\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_GATED_AFFINE_PRODUCT\n'
            '#define EVAL_METHOD_GATED_QUADRATIC_COLLAPSE\n'
            '#define GATED_AFFINE_PRODUCT\n'
            '#define GATED_QUADRATIC_COLLAPSE\n'
            f'#define TT_ACT_GATE_B {clamp(q0):.10e}f\n'
            f'#define TT_ACT_GATE_A {clamp(q1):.10e}f\n'
            '#define GATED_QUADRATIC_Q0 TT_ACT_GATE_B\n'
            '#define GATED_QUADRATIC_Q1 TT_ACT_GATE_A\n'
        )
        print(f'GATED QUADRATIC COLLAPSE: y = x * clamp({q0:.6g} + {q1:.6g}*x, 0, 1) -> clamp template')

# Abs-denominator rational collapse: y = x / (1 + abs(x)). Recognized from the
# exact two-segment rational coefficient pattern, not from activation name.
abs_den_rational_macro = ''
if is_rational and rr_method in ('', 'none') and num_segments == 2 and num_degree == 1 and den_degree == 1:
    tol = 1.0e-5
    n_cps = num_degree + 1
    d_cps = den_degree + 1
    seg0_num = num_coefficients[0:n_cps]
    seg1_num = num_coefficients[n_cps:2 * n_cps]
    seg0_den = den_coefficients[0:d_cps]
    seg1_den = den_coefficients[d_cps:2 * d_cps]
    numer_ok = (
        abs(seg0_num[0]) <= tol and abs(seg0_num[1] - 1.0) <= tol and
        abs(seg1_num[0]) <= tol and abs(seg1_num[1] - 1.0) <= tol
    )
    den_ok = (
        abs(seg0_den[0] - 1.0) <= tol and abs(seg0_den[1] + 1.0) <= tol and
        abs(seg1_den[0] - 1.0) <= tol and abs(seg1_den[1] - 1.0) <= tol and
        abs(boundaries[1]) <= tol
    )
    if numer_ok and den_ok:
        abs_den_rational_macro = (
            '\n// rational template: abs_denominator_rational. y = x / (1 + abs(x)).\n'
            '#define TT_ACT_EVAL_KIND TT_ACT_EVAL_ABS_DENOMINATOR_RATIONAL\n'
            '#define EVAL_METHOD_ABS_DENOMINATOR_RATIONAL\n'
            '#define ABS_DENOMINATOR_RATIONAL\n'
        )
        print('ABS DENOMINATOR RATIONAL: y = x / (1 + abs(x)) -> abs+reciprocal bypass')

# Optional postcompose hook. This is explicit CSV metadata for algebraic wrappers
# such as acos(x) = pi/2 - asin(x). It is generic affine-in-output composition.
postcompose_macro = ''
postcompose = _meta_norm('postcompose', 'post_compose', 'post_transform', 'output_transform')
if postcompose in ('pi_over_2_minus_y', 'half_pi_minus_y'):
    postcompose_macro = (
        '\n// postcompose: y = pi/2 - y\n'
        '#define POSTCOMPOSE_AFFINE_Y\n'
        '#define POSTCOMPOSE_A 1.5707963267948966e+00f\n'
        '#define POSTCOMPOSE_B -1.0000000000e+00f\n'
    )
    print('Postcompose: y = pi/2 - y')
elif postcompose in ('', 'none', 'identity'):
    pass
else:
    raise ValueError(f'unsupported postcompose={postcompose!r}')

# Explicit algebraic eval methods are metadata requests. Coefficient-pattern
# recognizers above are the proof gate: an explicit request that does not prove
# out is a hard error, not a silent cascade fallback.
declared_eval_method = _meta_norm('eval_method')
declared_to_macro = {
    'identity': affine_macro if '#define AFFINE_IDENTITY' in affine_macro else '',
    'affine': affine_macro,
    'affine_collapse': affine_macro,
    'clamped_affine': clamped_affine_macro,
    'clamped_affine_collapse': clamped_affine_macro,
    'abs_value': abs_value_macro,
    'threshold_identity': threshold_select_macro,
    'threshold_identity_select': threshold_select_macro,
    'threshold_softshift': threshold_softshift_macro,
    'softshrink_select': threshold_softshift_macro,
    'gated_affine_product': gated_affine_product_macro,
    'gated_quadratic_collapse': gated_affine_product_macro,
    'abs_denominator_rational': abs_den_rational_macro,
}
if declared_eval_method in declared_to_macro and not declared_to_macro[declared_eval_method]:
    raise ValueError(
        f'eval_method={declared_eval_method!r} requested an algebraic lowering, '
        'but CSV coefficients/boundaries did not prove that lowering'
    )

# Exactly one evaluator selector. Metadata-driven methods (rr_macro) and
# whole-function collapses emit TT_ACT_EVAL_KIND; otherwise default to cascade.
# Parity / dual / adaptive / blend are orthogonal modifiers.
eval_method_macro = ''
has_codegen_eval_method = any(
    'TT_ACT_EVAL_KIND' in macro for macro in (
        rr_macro,
        affine_macro,
        clamped_affine_macro,
        abs_value_macro,
        threshold_select_macro,
        threshold_softshift_macro,
        gated_affine_product_macro,
        abs_den_rational_macro,
    )
)
if not is_rational:
    if not has_codegen_eval_method:
        eval_method_macro = '\n// eval_method: poly_cascade (default piecewise polynomial cascade)\n#define TT_ACT_EVAL_KIND TT_ACT_EVAL_POLY_CASCADE\n#define EVAL_METHOD_POLY_CASCADE\n'
else:
    # rational_cascade is the base method; reduced_poly may layer on via rr_macro.
    if not has_codegen_eval_method:
        eval_method_macro = '\n// eval_method: rational_cascade (piecewise P(x)/Q(x))\n#define TT_ACT_EVAL_KIND TT_ACT_EVAL_RATIONAL_CASCADE\n#define EVAL_METHOD_RATIONAL_CASCADE\n'

# Write kernel .cpp
if is_rational:
    kernel = f'''// Auto-generated by run_csv.sh from: {os.path.basename(csv_path)}
// Rational n{num_degree}/d{den_degree}, {num_segments} segments, range [{input_min}, {input_max}]
#include <array>
#include <cstdint>

#define EMBEDDED_LUT
constexpr uint32_t NUM_DEGREE = {num_degree};
constexpr uint32_t DEN_DEGREE = {den_degree};
constexpr uint32_t NUM_SEGMENTS = {num_segments};

constexpr float INPUT_MIN = {input_min:.10e}f;
constexpr float INPUT_MAX = {input_max:.10e}f;

constexpr uint32_t LUT_SIZE = {lut_size};
constexpr std::array<float, LUT_SIZE> LUT_DATA = {{{{
    {lut_str}
}}}};
{eval_method_macro}{poly_parity_macro}{rr_macro}{asymptotic_macro}{abs_den_rational_macro}{postcompose_macro}
#include \"../piecewise_rational.cpp\"
'''
    # Override degree-related output variables for rational
    degree = num_degree
else:
    kernel = f'''// Auto-generated by run_csv.sh from: {os.path.basename(csv_path)}
// Degree {degree}, {num_segments} segments, range [{input_min}, {input_max}]
#include <array>
#include <cstdint>

#define EMBEDDED_LUT
constexpr uint32_t POLY_DEGREE = {degree};
constexpr uint32_t NUM_SEGMENTS = {num_segments};

constexpr float INPUT_MIN = {input_min:.10e}f;
constexpr float INPUT_MAX = {input_max:.10e}f;

constexpr uint32_t LUT_SIZE_BF16 = {lut_size};
constexpr std::array<float, LUT_SIZE_BF16> LUT_DATA_BF16 = {{{{
    {lut_str}
}}}};

constexpr uint32_t LUT_SIZE_FP32 = {lut_size};
constexpr std::array<float, LUT_SIZE_FP32> LUT_DATA_FP32 = {{{{
    {lut_str}
}}}};

#ifdef USE_BF16
    constexpr auto& LUT_DATA = LUT_DATA_BF16;
    constexpr uint32_t LUT_SIZE = LUT_SIZE_BF16;
#else
    constexpr auto& LUT_DATA = LUT_DATA_FP32;
    constexpr uint32_t LUT_SIZE = LUT_SIZE_FP32;
#endif
{eval_method_macro}{degree_macros}{eval_basis_macro}{poly_parity_macro}{rr_macro}{asymptotic_macro}{affine_macro}{clamped_affine_macro}{abs_value_macro}{threshold_select_macro}{threshold_softshift_macro}{gated_affine_product_macro}{postcompose_macro}
#include \"../piecewise_generic.cpp\"
'''

try:
    with open(kernel_tmp_path, 'w') as f:
        f.write(kernel)
    os.replace(kernel_tmp_path, kernel_path)
finally:
    if os.path.exists(kernel_tmp_path):
        os.unlink(kernel_tmp_path)

print(f'Generated: {kernel_path}')

# Write detected values to stdout for bash to capture
print(f'DETECTED_RANGE_MIN={input_min}')
print(f'DETECTED_RANGE_MAX={input_max}')
print(f'DETECTED_DEGREE={degree}')
print(f'DETECTED_SEGMENTS={num_segments}')
print(f'DETECTED_DEGREE_SUM={sum(segment_degrees)}')
print(f'DETECTED_IS_RATIONAL={1 if is_rational else 0}')
if is_rational:
    print(f'DETECTED_NUM_DEGREE={num_degree}')
    print(f'DETECTED_DEN_DEGREE={den_degree}')
" 2>&1 | tee "$RUN_CSV_GEN_LOG"

# Extract detected values from Python output
if [[ -z "$RANGE_MIN" ]]; then
    RANGE_MIN=$(grep '^DETECTED_RANGE_MIN=' "$RUN_CSV_GEN_LOG" | tail -1 | cut -d= -f2)
fi
if [[ -z "$RANGE_MAX" ]]; then
    RANGE_MAX=$(grep '^DETECTED_RANGE_MAX=' "$RUN_CSV_GEN_LOG" | tail -1 | cut -d= -f2)
fi
POLY_DEGREE=$(grep '^DETECTED_DEGREE=' "$RUN_CSV_GEN_LOG" | tail -1 | cut -d= -f2)
NUM_SEGMENTS=$(grep '^DETECTED_SEGMENTS=' "$RUN_CSV_GEN_LOG" | tail -1 | cut -d= -f2)
COEFFS=$(grep '^DETECTED_DEGREE_SUM=' "$RUN_CSV_GEN_LOG" | tail -1 | cut -d= -f2)
IS_RATIONAL=$(grep '^DETECTED_IS_RATIONAL=' "$RUN_CSV_GEN_LOG" | tail -1 | cut -d= -f2)
[[ -z "$COEFFS" || "$COEFFS" == "0" ]] && COEFFS=$((POLY_DEGREE * NUM_SEGMENTS))

if [[ "$IS_RATIONAL" == "1" ]]; then
    NUM_DEG=$(grep '^DETECTED_NUM_DEGREE=' "$RUN_CSV_GEN_LOG" | tail -1 | cut -d= -f2)
    DEN_DEG=$(grep '^DETECTED_DEN_DEGREE=' "$RUN_CSV_GEN_LOG" | tail -1 | cut -d= -f2)
    CONFIG_NAME="n${NUM_DEG}d${DEN_DEG}_s${NUM_SEGMENTS}"
    COEFFS=$(( (NUM_DEG + 1 + DEN_DEG + 1) * NUM_SEGMENTS ))
else
    CONFIG_NAME="p${POLY_DEGREE}_s${NUM_SEGMENTS}"
fi

echo "Config:     $CONFIG_NAME ($COEFFS coeffs/segment)"
echo ""

# Build list of precisions to iterate for accuracy/reporting
if [[ "$PRECISION" == "both" ]]; then
    PRECISION_LIST=(bf16 fp32)
else
    PRECISION_LIST=("$PRECISION")
fi

# --- Step 2: Build ---
if [[ "$SKIP_BUILD" == true ]]; then
    echo "Skipping build (--skip-build)"
else
    echo "Building adhoc target..."
    cd "$REPO_ROOT"
    ninja -C "$BUILD_DIR" programming_examples_generic_lut_activation_embedded_adhoc
    echo "Build complete."
fi

echo ""

# --- Step 3: Run all shapes with profiler timing and accuracy ---
cd "$REPO_ROOT"

# Hardware output directory for accuracy CSVs
output_dir=$(get_hardware_output_dir "$ACTIVATION" "$WORK_DIR")

# Build batch tile list from TEST_SHAPES
batch_tiles=""
declare -A shape_name_by_tiles=()
for test_shape in "${TEST_SHAPES[@]}"; do
    parse_shape "$test_shape"
    batch_tiles="${batch_tiles:+$batch_tiles,}$tile_count"
    shape_name_by_tiles[$tile_count]="$shape_name"
done

# Determine if we should use batch mode (multiple shapes → single binary invocation)
use_batch=false
if [[ ${#TEST_SHAPES[@]} -gt 1 ]]; then
    use_batch=true
fi

# Print table header (matching sweep_best.sh format)
echo "=== $ACTIVATION ($PRECISION, range: $RANGE_MIN to $RANGE_MAX) ==="
printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "Config" "DegSum" "MAE" "MaxErr" "MaxULP" "MeanULP" "Prof(µs)" "Host(ms)"
printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "-------------------------" "--------" "------------" "------------" "------------" "------------" "----------" "----------"

# Per-shape result accumulators (associative arrays keyed by shape_name)
declare -A profiler_times_csv=()   # shape_name → "t1,t2,t3" (comma-separated across runs)
declare -A host_times_csv=()       # shape_name → "t1,t2,t3"
declare -A csv_output_paths=()     # shape_name → path to hardware output CSV

# Precompute per-shape, per-precision CSV output paths
# Must match what the C++ binary produces from DUMP_OUTPUT_CSV base path:
#   --precision both:  base (no precision) + _${prec}_tiles${N}.csv
#   single precision:  base (with precision) + _tiles${N}.csv  (backward compat)
for prec in "${PRECISION_LIST[@]}"; do
    for test_shape in "${TEST_SHAPES[@]}"; do
        parse_shape "$test_shape"
        if [[ "$PRECISION" == "both" && "$use_batch" == true ]]; then
            # Binary base: ${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}
            # Binary suffix: _${prec}_tiles${N} (is_multi_precision + is_batch_mode)
            csv_output_paths["${prec}_${shape_name}"]="${output_dir}/${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}_${prec}_tiles${tile_count}.csv"
        elif [[ "$PRECISION" == "both" ]]; then
            # Binary base: ${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}
            # Binary suffix: _${prec} (is_multi_precision only, no batch)
            csv_output_paths["${prec}_${shape_name}"]="${output_dir}/${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}_${prec}.csv"
        elif [[ "$use_batch" == true ]]; then
            # Single precision batch: base has precision, binary adds _tiles${N}
            csv_output_paths["${prec}_${shape_name}"]="${output_dir}/${ACTIVATION}_${PRECISION}_${NUM_SEGMENTS}_${POLY_DEGREE}_tiles${tile_count}.csv"
        else
            # Single precision, single shape: exact path (no suffix added by binary)
            csv_output_paths["${prec}_${shape_name}"]="${output_dir}/${ACTIVATION}_${PRECISION}_${NUM_SEGMENTS}_${POLY_DEGREE}_tiles${tile_count}.csv"
        fi
    done
done

# ===== STEP 1: Profiler + host timing + accuracy (single pass) =====
# First run also dumps hardware output for accuracy computation
profiler_success=true
host_success=true

for run in $(seq 1 $NUM_RUNS); do
    PROFILER_BASE="$WORK_DIR/profiler_results/reports/adhoc_${ACTIVATION}_run${run}"
    mkdir -p "$PROFILER_BASE"

    # Dump hardware output on first run (for accuracy computation)
    if [[ "$run" -eq 1 ]]; then
        if [[ "$PRECISION" == "both" ]]; then
            # Binary inserts _bf16/_fp32 and _tilesN before .csv (is_multi_precision + is_batch_mode)
            export DUMP_OUTPUT_CSV="${output_dir}/${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}.csv"
        elif [[ "$use_batch" == true ]]; then
            # Single precision, binary inserts _tilesN before .csv
            export DUMP_OUTPUT_CSV="${output_dir}/${ACTIVATION}_${PRECISION}_${NUM_SEGMENTS}_${POLY_DEGREE}.csv"
        else
            parse_shape "${TEST_SHAPES[0]}"
            export DUMP_OUTPUT_CSV="${csv_output_paths[${PRECISION_LIST[0]}_${shape_name}]}"
        fi
    else
        unset DUMP_OUTPUT_CSV
    fi

    set +e
    if [[ "$use_batch" == true ]]; then
        run_output=$(TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$PROFILER_BASE" \
            "$BINARY" --activation "$ACTIVATION" --precision "$PRECISION" \
            --range-min "$RANGE_MIN" --range-max "$RANGE_MAX" \
            --batch-tiles "$batch_tiles" "${EXTRA_BINARY_ARGS[@]}" 2>&1)
    else
        parse_shape "${TEST_SHAPES[0]}"
        run_output=$(TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$PROFILER_BASE" \
            "$BINARY" --activation "$ACTIVATION" --precision "$PRECISION" \
            --range-min "$RANGE_MIN" --range-max "$RANGE_MAX" --tiles "$tile_count" \
            "${EXTRA_BINARY_ARGS[@]}" 2>&1)
    fi
    run_exit=$?
    set -e

    if [[ $run_exit -ne 0 ]]; then
        # Surface the binary's stderr/stdout — DON'T swallow it. A non-zero exit
        # here (e.g. a post-dump exception from generic_lut_activation.cpp's
        # try/catch) was previously invisible, turning every binary failure into a
        # black-box "FAILED" row with no cause.
        echo "  [run_csv] binary exited $run_exit (activation=$ACTIVATION); output follows:" >&2
        echo "$run_output" >&2
        profiler_success=false
        host_success=false
        break
    fi

    # Extract per-shape profiler times from single CSV (batch clustering)
    PROFILER_CSV_PATH="${PROFILER_BASE}/.logs/profile_log_device.csv"
    if [[ -f "$PROFILER_CSV_PATH" ]]; then
        total_shapes=$(( ${#PRECISION_LIST[@]} * ${#TEST_SHAPES[@]} ))
        batch_times=$(extract_batch_profiler_times "$PROFILER_CSV_PATH" "$total_shapes" "$WORK_DIR")
        i=0
        IFS=',' read -ra _ptimes <<< "$batch_times"
        for prec in "${PRECISION_LIST[@]}"; do
            for test_shape in "${TEST_SHAPES[@]}"; do
                parse_shape "$test_shape"
                pkey="${prec}_${shape_name}"
                ptime="${_ptimes[$i]:-0}"
                if [[ -n "$ptime" && "$ptime" != "0" && "$ptime" != "0.00" ]]; then
                    profiler_times_csv[$pkey]="${profiler_times_csv[$pkey]:+${profiler_times_csv[$pkey]},}$ptime"
                fi
                i=$((i + 1))
            done
        done
    fi

    # Extract per-shape, per-precision host timing from same output
    for prec in "${PRECISION_LIST[@]}"; do
        for test_shape in "${TEST_SHAPES[@]}"; do
            parse_shape "$test_shape"
            pkey="${prec}_${shape_name}"
            if [[ "$PRECISION" == "both" ]]; then
                # Multi-precision format: BATCH[bf16,tiles=N]:TIMING_KERNEL_EXECUTION:
                kernel_exec=$(echo "$run_output" | grep "BATCH\[${prec},tiles=${tile_count}\]:TIMING_KERNEL_EXECUTION:" | awk -F': ' '{print $2}')
            elif [[ "$use_batch" == true ]]; then
                # Single-precision batch: BATCH[tiles=N]:TIMING_KERNEL_EXECUTION:
                kernel_exec=$(echo "$run_output" | grep "BATCH\[tiles=${tile_count}\]:TIMING_KERNEL_EXECUTION:" | awk -F': ' '{print $2}')
            else
                # Single shape, single precision: TIMING_KERNEL_EXECUTION:
                kernel_exec=$(echo "$run_output" | grep "TIMING_KERNEL_EXECUTION:" | head -1 | awk -F': ' '{print $2}')
            fi
            if [[ -n "$kernel_exec" ]]; then
                host_times_csv[$pkey]="${host_times_csv[$pkey]:+${host_times_csv[$pkey]},}$kernel_exec"
            fi
        done
    done
done
unset DUMP_OUTPUT_CSV

# ===== STEP 3: Compute results per shape and print table =====
# Arrays to collect results for summary table
declare -a result_configs=()
declare -a result_coeffs=()
declare -a result_mae=()
declare -a result_max_err=()
declare -a result_max_ulp=()
declare -a result_mean_ulp=()
declare -a result_profiler_us=()
declare -a result_host_ms=()

all_runs_passed=$( [[ "$profiler_success" == true && "$host_success" == true ]] && echo true || echo false )

for prec in "${PRECISION_LIST[@]}"; do
    for test_shape in "${TEST_SHAPES[@]}"; do
        parse_shape "$test_shape"
        pkey="${prec}_${shape_name}"
        csv_output="${csv_output_paths[$pkey]}"

        # Export exactly one raw dump: first requested precision, first requested shape.
        if [[ -n "$DUMP_CSV" && "${test_shape}" == "${TEST_SHAPES[0]}" && "$prec" == "${PRECISION_LIST[0]}" && -f "$csv_output" ]]; then
            cp "$csv_output" "$DUMP_CSV"
        fi
        if [[ -n "$DUMP_NPZ" && "${test_shape}" == "${TEST_SHAPES[0]}" && "$prec" == "${PRECISION_LIST[0]}" && -f "$csv_output" ]]; then
            mkdir -p "$(dirname "$DUMP_NPZ")"
            /usr/bin/python3 - "$csv_output" "$DUMP_NPZ" <<'PY'
import sys
from pathlib import Path
import numpy as np

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = np.genfromtxt(src, delimiter=",", names=True, invalid_raise=False)
if data.size == 0:
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
else:
    data = np.atleast_1d(data)
    names = data.dtype.names or ()
    if "input" not in names or "output" not in names:
        raise SystemExit(f"{src} must have input,output columns")
    x = np.asarray(data["input"], dtype=np.float32)
    y = np.asarray(data["output"], dtype=np.float32)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
np.savez_compressed(dst, input=x, output=y)
PY
        fi

        if [[ "$all_runs_passed" == true ]]; then
            # Compute profiler timing (in us)
            profiler_min_us="0"
            if [[ -n "${profiler_times_csv[$pkey]:-}" ]]; then
                profiler_min_us=$(compute_kernel_exec_min "${profiler_times_csv[$pkey]}")
            fi

            # Compute host timing (in ms)
            host_min_ms="0"
            if [[ -n "${host_times_csv[$pkey]:-}" ]]; then
                host_min_ms=$(compute_kernel_exec_min "${host_times_csv[$pkey]}")
            fi

            # Compute accuracy metrics
            mae_hw="0" max_hw="0" max_ulp_hw="0" mean_ulp_hw="0"
            if [[ "$HAS_ACCURACY" == true && -f "$csv_output" ]]; then
                accuracy_stats=$($ACCURACY_PYTHON "$ACCURACY_SCRIPT" "$ACTIVATION" "$csv_output" 2>/dev/null) || true
                if [[ -n "$accuracy_stats" ]]; then
                    mae_hw=$(echo "$accuracy_stats" | cut -d',' -f1)
                    max_hw=$(echo "$accuracy_stats" | cut -d',' -f3)
                    max_ulp_hw=$(echo "$accuracy_stats" | cut -d',' -f5)
                    mean_ulp_hw=$(echo "$accuracy_stats" | cut -d',' -f6)
                fi
            fi

            # Compress hardware output CSV to save disk space (original deleted by gzip)
            if [[ -f "$csv_output" ]]; then
                gzip -f "$csv_output"
            fi

            # Print row — include precision prefix when running both
            if [[ ${#PRECISION_LIST[@]} -gt 1 ]]; then
                config_display="${prec}_${shape_name}_${CONFIG_NAME}"
            else
                config_display="${shape_name}_${CONFIG_NAME}"
            fi
            [[ "$shape_name" == "yolov4" ]] && printf "${GREEN}"
            printf "%-25s %8d %12.2e %12.2e %12.2f %12.2f %9.2fµs %9.2fms\n" \
                "$config_display" "$COEFFS" "$mae_hw" "$max_hw" "$max_ulp_hw" "$mean_ulp_hw" "$profiler_min_us" "$host_min_ms"
            [[ "$shape_name" == "yolov4" ]] && printf "${NC}"

            # Store for summary
            result_configs+=("$config_display")
            result_coeffs+=("$COEFFS")
            result_mae+=("$mae_hw")
            result_max_err+=("$max_hw")
            result_max_ulp+=("$max_ulp_hw")
            result_mean_ulp+=("$mean_ulp_hw")
            result_profiler_us+=("$profiler_min_us")
            result_host_ms+=("$host_min_ms")
        else
            if [[ ${#PRECISION_LIST[@]} -gt 1 ]]; then
                config_display="${prec}_${shape_name}_${CONFIG_NAME}"
            else
                config_display="${shape_name}_${CONFIG_NAME}"
            fi
            printf "%-25s %8d %12s %12s %12s %12s %10s\n" "$config_display" "$COEFFS" "-" "-" "-" "-" "FAILED"

            result_configs+=("$config_display")
            result_coeffs+=("$COEFFS")
            result_mae+=("-")
            result_max_err+=("-")
            result_max_ulp+=("-")
            result_mean_ulp+=("-")
            result_profiler_us+=("-")
            result_host_ms+=("-")
        fi
    done
done

# --- Summary table ---
echo ""
echo "================================================================================"
echo "  SUMMARY: $ACTIVATION  $PRECISION  $(basename "$CSV_FILE")"
echo "================================================================================"
printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "Config" "DegSum" "MAE" "MaxErr" "MaxULP" "MeanULP" "Prof(µs)" "Host(ms)"
printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "-------------------------" "--------" "------------" "------------" "------------" "------------" "----------" "----------"

for i in "${!result_configs[@]}"; do
    if [[ "${result_mae[$i]}" == "-" ]]; then
        printf "%-25s %8s %12s %12s %12s %12s %10s\n" \
            "${result_configs[$i]}" "${result_coeffs[$i]}" "-" "-" "-" "-" "FAILED"
    else
        [[ "${result_configs[$i]}" == yolov4_* ]] && printf "${GREEN}"
        printf "%-25s %8d %12.2e %12.2e %12.2f %12.2f %9.2fµs %9.2fms\n" \
            "${result_configs[$i]}" "${result_coeffs[$i]}" \
            "${result_mae[$i]}" "${result_max_err[$i]}" \
            "${result_max_ulp[$i]}" "${result_mean_ulp[$i]}" \
            "${result_profiler_us[$i]}" "${result_host_ms[$i]}"
        [[ "${result_configs[$i]}" == yolov4_* ]] && printf "${NC}"
    fi
done

echo "================================================================================"
