"""Rank matmul program configs for one BGE-M3 row with the tt-optimization-loop sweep.

The sweep measures each configuration in isolation in a subprocess and reports
device kernel time. It never edits the model and never runs the gate, so a
winner still has to be applied and gated separately.

Run from the tt-metal root with its python_env:

  PYTHONPATH=/local/ttuser/gtobar/tt-optimization-loop/src:$PWD \
  python_env/bin/python /local/ttuser/gtobar/sweep_bge.py wi

`m_tiles` in registry.Shape is the PER-CORE M, not the whole output height:
cb_bytes() multiplies it by the block width to check one core's L1. Passing the
full 128 tiles rejects every configuration.
"""

import sys
from pathlib import Path

from tt_opt_loop.evidence.models import Dtypes, MemoryConfigs
from tt_opt_loop.sweep.registry import Shape
from tt_opt_loop.sweep.runner import run_sweep

FRAMEWORK = Path("/local/ttuser/gtobar/tt-metal")
RUN_DIR = Path("/local/ttuser/gtobar/sweep_runs")
GRID = (13, 10)  # p150 compute_with_storage_grid_size
BATCH, SEQ = 8, 512
M_TILES = BATCH * SEQ // 32  # 128 output rows of tiles

# Each row: k_tiles, n_tiles, the in-model us/call from the B8 Tracy capture,
# and the config the model runs today (grid 11x10, from optimizations.py).
ROWS = {
    "wi": dict(k=1024 // 32, n=4096 // 32, in_model_us=103.9, ibw=8, sub_w=4, label="MLP wi 1024->4096"),
    "wo": dict(k=4096 // 32, n=1024 // 32, in_model_us=106.6, ibw=8, sub_w=1, label="MLP wo 4096->1024"),
    "qkv": dict(k=1024 // 32, n=3072 // 32, in_model_us=102.9, ibw=8, sub_w=3, label="QKV 1024->3072"),
    "ao": dict(k=1024 // 32, n=1024 // 32, in_model_us=34.6, ibw=8, sub_w=1, label="attn-out 1024->1024"),
}

which = sys.argv[1] if len(sys.argv) > 1 else "wi"
budget_s = float(sys.argv[2]) if len(sys.argv) > 2 else 600.0
row = ROWS[which]

grid_y = GRID[1]
per_core_m = (M_TILES + grid_y - 1) // grid_y

# B8 runs ttnn.linear with the 2D mcast class. DRAM-sharded needs its activation
# width-sharded on the same grid, which this row does not have, so pin the class
# and keep the sweep inside the kernel the model actually launches.
PROGRAM_CONFIG_CLASS = "MatmulMultiCoreReuseMultiCastProgramConfig"

shape = Shape(
    m_tiles=per_core_m,
    k_tiles=row["k"],
    n_tiles=row["n"],
    weight_dtype="BFLOAT8_B",
    max_cores=GRID[0] * GRID[1],
    core_grid=GRID,
    program_config_class=PROGRAM_CONFIG_CLASS,
)

# What the model runs now, so the sweep can report the delta against it.
cur_grid_x, cur_grid_y = 11, 10
current = {
    "program_config_class": "MatmulMultiCoreReuseMultiCastProgramConfig",
    "cores": cur_grid_x * cur_grid_y,
    "grid_x": cur_grid_x,
    "grid_y": cur_grid_y,
    "in0_block_w": min(row["ibw"], row["k"]),
    "out_subblock_h": 1,
    "out_subblock_w": row["sub_w"],
    "out_block_h": 1,
    "out_block_w": row["sub_w"],
    "per_core_M": (M_TILES + cur_grid_y - 1) // cur_grid_y,
    "per_core_N": (row["n"] + cur_grid_x - 1) // cur_grid_x,
}

# What the row states. run_sweep refuses geometry alone, because m_tiles*32
# rebuilds a padded height and measures a different op. These are the real
# operand extents of the B8/S512 row: in0 is [1, 1, B*S, K], in1 is [K, N].
shape_request = {
    "in0_dtype": "BFLOAT8_B",
    "in1_dtype": "BFLOAT8_B",
    "input_source": "exact",
    "logical": [1, 1, BATCH * SEQ, row["k"] * 32],
    "w": 1,
    "z": 1,
    "m": BATCH * SEQ,
    "k": row["k"] * 32,
    "n": row["n"] * 32,
    "in1_layout": "TILE",
}

print("row        : %s (B%d/S%d)" % (row["label"], BATCH, SEQ))
print("shape      : per_core_M=%d k_tiles=%d n_tiles=%d bf8 grid=%dx%d" % (per_core_m, row["k"], row["n"], *GRID))
print("in model   : %.1f us/call" % row["in_model_us"])
print("budget     : %.0f s" % budget_s)

result = run_sweep(
    op_key="bge_m3_b%d_s%d_%s" % (BATCH, SEQ, which),
    family="matmul",
    shape=shape,
    run_dir=RUN_DIR,
    framework_root=FRAMEWORK,
    current_config=current,
    dtypes=Dtypes(in0="BFLOAT8_B", in1="BFLOAT8_B", out="BFLOAT8_B"),
    # The row reads its activation from L1 and writes its output to L1
    # (optimizations.py _mlp_wi_output_memory_config). A DRAM-resident trial
    # measures a different op: reproduction_ratio came out 4.54 against a 2%
    # tolerance.
    memory=MemoryConfigs(in0="L1", in1="DRAM", out="L1"),
    in_model_us=row["in_model_us"],
    budget_s=budget_s,
    best_n=8,
    label=row["label"],
    # Keep the surface inside the kernel the model launches. B8's activation is
    # interleaved, so the DRAM-sharded class cannot run here, and the 1D class
    # is a different kernel from the 2D mcast the row uses.
    enumerate_extra={"classes": [PROGRAM_CONFIG_CLASS]},
    shape_request=shape_request,
)

print("\n=== RESULT ===")
print(
    "legal %d  tried %d  complete=%s  stopped=%s"
    % (result.legal, result.tried, result.complete, result.stopped_because)
)
if result.refusal:
    print("REFUSAL: %s" % result.refusal)
cur = result.current
if cur is not None:
    print("current config  : %.1f us  (in model %.1f)" % (cur.kernel_us, row["in_model_us"]))
for i, t in enumerate((result.best or [])[:8], 1):
    cfg = t.config or {}
    gain = "" if cur is None else "  %+.1f%%" % (100.0 * (t.kernel_us - cur.kernel_us) / cur.kernel_us)
    print(
        "  %d. %8.1f us%s  %-34s cores=%-4s ibw=%-3s sub=%sx%s"
        % (
            i,
            t.kernel_us,
            gain,
            str(cfg.get("program_config_class", ""))
            .replace("MatmulMultiCoreReuse", "")
            .replace("ProgramConfig", "")[:34],
            cfg.get("cores"),
            cfg.get("in0_block_w"),
            cfg.get("out_subblock_h"),
            cfg.get("out_subblock_w"),
        )
    )
out = RUN_DIR / ("sweep_%s.json" % which)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(result.model_dump_json(indent=2))
print("\nwrote %s" % out)
