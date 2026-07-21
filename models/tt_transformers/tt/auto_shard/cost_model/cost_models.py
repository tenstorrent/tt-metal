# cost models to pass into sharding.py
#
# The workload is one prefill followed by DECODE_STEPS single-token decodes per layer. The two
# regimes cost very differently, so we model them separately: total = prefill + DECODE_STEPS * decode.
# Prefill is throughput-bound, so a split genuinely divides the work; decode is latency-bound and
# falls off more slowly (see DECODE_SPLIT_EXPONENT).
#
# Constants live in params.py so cost_models and the stack tests stay in sync; re-run
# calibrate_cost_model.py to recalibrate. The ranking depends on the prefill/decode mix, so
# select_sharding forwards the real workload; PREFILL_LEN/DECODE_STEPS are fallbacks.

from toy_problem.utils import fabric_link_report

from models.tt_transformers.tt.params import (  # noqa: F401
    ALPHA,
    BETA,
    BYTES,
    DECODE_SPLIT_EXPONENT,
    DECODE_STEPS,
    PREFILL_LEN,
    TFLOPS_DECODE,
    TFLOPS_PREFILL,
)


def cost_model(mesh_device, placements, shapes=None, mesh_shape=None, prefill_len=PREFILL_LEN, decode_steps=DECODE_STEPS, debug=False, interconnect_aware=True):
    return cost_model_analytical(mesh_device, placements, shapes, mesh_shape, prefill_len, decode_steps, debug, interconnect_aware)


def _axis_size(mesh_shape, axis):
    return 1 if axis is None else mesh_shape[axis]


def _flops_and_comm(shapes):
    """Per-token flops and the two collective payloads for one forward pass of a block.

    Attention and MLP have the same shape: a column-parallel matmul feeding a row-parallel one.
    MLP just has two column matmuls (w1, w3), which doubles the column flops and its all-reduce.

    Returns (flops_per_token, col_out, row_out, n_col_reduces), where col_out/row_out are the
    column and row matmul output widths and n_col_reduces is 1 for attention, 2 for MLP.
    """
    if hasattr(shapes, "hidden_dim"):  # MLP: w1, w3 [dim, hidden_dim]; w2 [hidden_dim, dim]
        flops = 2 * (2 * shapes.dim * shapes.hidden_dim) + 2 * shapes.hidden_dim * shapes.dim
        return flops, shapes.hidden_dim, shapes.dim, 2
    # attention: wqkv [dim, qkv_size]; wo [n_heads*head_dim, dim]
    flops = 2 * shapes.dim * shapes.qkv_size + 2 * shapes.n_heads * shapes.head_dim * shapes.dim
    return flops, shapes.qkv_size, shapes.dim, 1


def placement_cost(p, shapes, mesh_shape, beta, prefill_len=PREFILL_LEN, decode_steps=DECODE_STEPS):
    """Seconds for one layer under placement `p`, given an explicit per-axis beta (bytes/s). No device.

    The device-free seam: cost_model_analytical reads beta off the live fabric to rank placements
    within a mesh, and mesh_selection.mesh_cost builds beta from a static table pre-open to rank
    whole meshes. Same arithmetic either way, so the two cost models can't drift.

    Per-layer cost only. The fixed per-token work outside the block stack (embedding, final norm,
    lm_head, sampling) doesn't scale with layer count and belongs in
    mesh_selection.fixed_decode_cost, which mesh_cost adds on.
    """
    prefill = _regime(p, shapes, mesh_shape, prefill_len, TFLOPS_PREFILL, latency_bound=False, beta=beta)
    decode = _regime(p, shapes, mesh_shape, 1, TFLOPS_DECODE, latency_bound=True, beta=beta)
    return prefill + decode_steps * decode


def _regime(p, shapes, mesh_shape, tokens, tflops, latency_bound, beta):
    """Seconds for one column + row matmul pass at `tokens` tokens under placement `p`.

    `beta` is a per-axis bandwidth (bytes/s): each all-reduce is charged at the speed of the axis it
    runs on. Pass a uniform beta for the interconnect-agnostic baseline.
    """
    intermediate = _axis_size(mesh_shape, p.intermediate_axis)
    model = _axis_size(mesh_shape, p.model_axis)

    flops_per_token, col_out, row_out, n_col_reduces = _flops_and_comm(shapes)
    flops = tokens * flops_per_token

    # Product of the two axis sizes, not the chip count: i1/mNone on a 2x2 splits 2 ways and
    # replicates twice.
    split = intermediate * model

    # Prefill's split genuinely divides the work. Decode falls off more slowly (measured ~1.3x over
    # a 2x split), so it gets a fitted exponent; see params.py.
    if latency_bound:
        compute = flops / tflops * split**-DECODE_SPLIT_EXPONENT
    else:
        compute = flops / split / tflops

    # 0, 1, or 2 all-reduces: one per axis the split contracts over, each at that axis's speed
    comm = 0.0
    if model > 1:  # column matmuls contract over the model axis
        comm += n_col_reduces * (ALPHA + 2 * (model - 1) / model * tokens * (col_out / intermediate) * BYTES / beta[p.model_axis])
    if intermediate > 1:  # row matmul contracts over the intermediate axis
        comm += ALPHA + 2 * (intermediate - 1) / intermediate * tokens * (row_out / model) * BYTES / beta[p.intermediate_axis]

    return compute + comm


def cost_model_analytical(
    mesh_device, placements, shapes=None, mesh_shape=None, prefill_len=PREFILL_LEN, decode_steps=DECODE_STEPS, debug=False, interconnect_aware=True
):
    report = fabric_link_report(mesh_device)
    axis_bw = [report["axes"][a]["bandwidth"] for a in range(len(mesh_shape))]  # GB/s per axis
    axis_wires = [report["axes"][a]["wires"] for a in range(len(mesh_shape))]  # used wires per axis
    faster_axis = report["faster_axis"]

    # axis_bw is relative, not absolute, so scale the calibrated BETA by each axis's share of the
    # fastest. A linkless axis falls back to BETA, but a split over it never fires a comm term
    # anyway. Calibrating _CABLE_BW in real bytes/s would let this go absolute.
    ref = max(axis_bw) or 1.0
    beta = [BETA * bw / ref if bw > 0 else BETA for bw in axis_bw]
    if debug:
        print(f"per-axis wires={axis_wires} bandwidth={axis_bw} beta={beta} faster_axis={faster_axis} interconnect_aware={interconnect_aware}")

    # Interconnect-aware charges each all-reduce at its axis's bandwidth; the agnostic baseline
    # uses one global BETA everywhere.
    beta_used = beta if interconnect_aware else [BETA] * len(mesh_shape)

    def regime(p, tokens, tflops, latency_bound):
        return _regime(p, shapes, mesh_shape, tokens, tflops, latency_bound, beta_used)

    def cost(p):
        return placement_cost(p, shapes, mesh_shape, beta_used, prefill_len, decode_steps)

    if debug:
        print(f"cost model ranking on mesh {mesh_shape} (prefill_len={prefill_len}, decode_steps={decode_steps}):")
        for p in sorted(placements, key=cost):
            prefill = regime(p, prefill_len, TFLOPS_PREFILL, latency_bound=False)
            decode = regime(p, 1, TFLOPS_DECODE, latency_bound=True)
            print(
                f"  intermediate={str(p.intermediate_axis):>4} model={str(p.model_axis):>4}  "
                f"prefill={prefill*1e3:7.3f} ms + {decode_steps}*decode={decode_steps*decode*1e3:7.3f} ms  "
                f"= {cost(p)*1e3:7.3f} ms/step"
            )

    return min(placements, key=cost)
