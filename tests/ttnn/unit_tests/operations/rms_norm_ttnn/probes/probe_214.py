# Perf experiment `stream_regime`.
#
# The STREAM regime (X_RESIDENT == 0) is the L1 fallback the op takes when neither
# RESIDENT nor ROW_RESIDENT can hold a tile-row.  It pays twice:
#   1. the activations cross DRAM TWICE (pass B re-reads x and the residual);
#   2. the per-channel operands are re-staged per pass-B chunk of EVERY row-block
#      (num_blocks x NUM_W_CHUNKS times per core instead of once).
#
# Variants
#   base    the shipped op, kernels copied byte-for-byte into k_base
#   ablpc   RMS_ABLATE_PER_CHANNEL -- the per-channel DRAM read deleted.  UPPER
#           BOUND on option (1).  Output intentionally wrong (perf only).
#   ablx    RMS_ABLATE_READ_X -- the activation DRAM read deleted.  UPPER BOUND on
#           options (2)/(3).  Output intentionally wrong (perf only).
#   cand    the candidate kernels (k_cand) + pd_stream.py host solve.
import importlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(os.environ.get("RMS_EXP_DIR") or Path(__file__).resolve().parent)
REPO = Path(__file__).resolve().parents[6] if "perf_experiments" not in str(Path(__file__)) else HERE.parents[4]
REPO = Path("/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal")
sys.path.insert(0, str(REPO / "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes"))

import bench_r3  # noqa: E402
import ttnn  # noqa: E402
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD  # noqa: E402

OP = importlib.import_module("ttnn.operations.rms_norm_ttnn.rms_norm_ttnn")

_ML = ttnn.TensorMemoryLayout

# ---- the cases this experiment owns ---------------------------------------
# name: shape, shard|"auto"|None, memory_layout, mode, fp32_dest, ceiling_ns
bench_r3.CASES.update(
    {
        # THE TARGET: perf case #15, ratio 0.859, STREAM.
        "T15_7168_gbr32": ((1, 1, 8192, 7168), None, _ML.INTERLEAVED, "gamma_bias_residual", True, 1837678),
        # domain sweep
        "D1_7168_g": ((1, 1, 8192, 7168), None, _ML.INTERLEAVED, "gamma", False, 589591),
        "D2_5120_gbr32": ((1, 1, 8192, 5120), None, _ML.INTERLEAVED, "gamma_bias_residual", True, 0),
        "D3_5120_g": ((1, 1, 8192, 5120), None, _ML.INTERLEAVED, "gamma", False, 0),
        # THE ROUND FOCUS SHAPE -- RESIDENT, must not regress
        "F_2304_g": ((1, 1, 8192, 2304), None, _ML.INTERLEAVED, "gamma", False, 0),
    }
)

SHIPPED = OP.create_program_descriptor
SHIPPED_KDIR = PD.KERNEL_DIR


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _use_shipped(kdir):
    OP.create_program_descriptor = SHIPPED
    PD.KERNEL_DIR = kdir


def _use_cand(**knobs):
    mod = _load("pd_stream", HERE / "pd_stream.py")
    OP.create_program_descriptor = mod.create_program_descriptor
    mod.KERNEL_DIR = HERE / "k_cand"
    for k, v in knobs.items():
        setattr(mod, k, v)


VARIANTS = {
    "base": lambda: _use_shipped(HERE / "k_base"),
    "ablpc": lambda: _use_shipped(HERE / "k_ablpc"),
    "ablx": lambda: _use_shipped(HERE / "k_ablx"),
    "pd_off": lambda: _use_cand(PC_COMPACT_HOLD=False, ROW_RESIDENT_COMPACT_PC=False),
    # option 1 alone: the compact cache, but the ROW_RESIDENT hold still priced tiled
    "hold": lambda: _use_cand(PC_COMPACT_HOLD=True, ROW_RESIDENT_COMPACT_PC=False, PC_COMPACT_LAZY=False),
    "hold_lz": lambda: _use_cand(PC_COMPACT_HOLD=True, ROW_RESIDENT_COMPACT_PC=False, PC_COMPACT_LAZY=True),
    # options 1+2: the compact price lets ROW_RESIDENT fit -> pass B re-reads nothing
    "rowres_eg": lambda: _use_cand(PC_COMPACT_HOLD=True, ROW_RESIDENT_COMPACT_PC=True, PC_COMPACT_LAZY=False),
    "rowres": lambda: _use_cand(PC_COMPACT_HOLD=True, ROW_RESIDENT_COMPACT_PC=True, PC_COMPACT_LAZY=True),
}

DEFAULT_VARIANTS = ["base", "ablpc", "ablx"]
DEFAULT_CASES = ["T15_7168_gbr32"]


def main():
    cases = (os.environ.get("RMS_CASES") or ",".join(DEFAULT_CASES)).split(",")
    labels = (os.environ.get("RMS_VARIANTS") or ",".join(DEFAULT_VARIANTS)).split(",")
    reps = int(os.environ.get("RMS_BREPS", "1"))
    res = {}
    device = ttnn.open_device(device_id=0)
    try:
        for _ in range(reps):
            for label in labels:
                VARIANTS[label]()
                for case in cases:
                    ns, p, r = bench_r3.measure(device, case)
                    res.setdefault((case, label), []).append((ns, p, r))
                    print(f"RESULT.raw {case:16s} {label:8s} ns={ns:10.0f} pcc={p:.6f} relrms={r:.3e}", flush=True)
    finally:
        _use_shipped(SHIPPED_KDIR)
        ttnn.close_device(device)

    print("RESULT " + f"{'case':16s}" + "".join(f"{l:>11s}" for l in labels))
    for case in cases:
        row = f"{case:16s}"
        for label in labels:
            row += f"{min(x[0] for x in res[(case, label)]):11.0f}"
        print("RESULT " + row)
    print("RESULT --- speedup vs " + labels[0] + " (>1 = faster) ---")
    for case in cases:
        b = min(x[0] for x in res[(case, labels[0])])
        row = f"{case:16s}"
        for label in labels:
            row += f"{b / min(x[0] for x in res[(case, label)]):11.3f}"
        pcs = " ".join(f"{l}:{min(x[1] for x in res[(case, l)]):.6f}" for l in labels)
        print("RESULT " + row + "  pcc " + pcs)


if __name__ == "__main__":
    main()
