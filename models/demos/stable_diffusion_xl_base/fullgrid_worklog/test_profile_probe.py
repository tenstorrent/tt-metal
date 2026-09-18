# Profile any pytest command the way the CI device-perf wrapper does (python -m tracy -p -r), then print the
# conv / GN / layout rows. Usage:
#   PROBE_CMD='pytest <nodeid>' scripts/run_safe_pytest.sh --no-precompile \
#       models/demos/stable_diffusion_xl_base/fullgrid_worklog/test_profile_probe.py
import csv
import glob
import os
import re


def test_profile_probe():
    cmd = os.environ["PROBE_CMD"]
    subdir = os.environ.get("PROBE_SUBDIR", "fullgrid_probe")
    from tracy.process_model_log import run_device_profiler

    run_device_profiler(cmd, subdir)
    csvs = sorted(glob.glob(f"generated/profiler/{subdir}/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
    path = csvs[-1]
    rows = [r for r in csv.DictReader(open(path)) if r.get("DEVICE KERNEL DURATION [ns]")]
    print(f"\nPROBE_CSV {path}")
    keep = tuple(
        os.environ.get(
            "PROBE_OPS",
            "Conv2d,GenericOp,Halo,Unary,InterleavedToSharded,ShardedToInterleaved,Untilize,BinaryNg,Matmul,LayerNorm",
        ).split(",")
    )
    tot = 0.0
    for r in rows:
        ns = float(r["DEVICE KERNEL DURATION [ns]"])
        tot += ns
        if not r["OP CODE"].startswith(keep):
            continue
        a = r["ATTRIBUTES"]
        abh = re.search(r"act_block_h_ntiles=(\d+)", a)
        g = re.search(r"grid_size=(\d+)-(\d+)", a)
        risc = " ".join(
            f"{k}={float(r[c])/1e3:.1f}"
            for k, c in (
                ("BR", "DEVICE BRISC KERNEL DURATION [ns]"),
                ("NC", "DEVICE NCRISC KERNEL DURATION [ns]"),
                ("T0", "DEVICE TRISC0 KERNEL DURATION [ns]"),
                ("T1", "DEVICE TRISC1 KERNEL DURATION [ns]"),
                ("T2", "DEVICE TRISC2 KERNEL DURATION [ns]"),
            )
            if r.get(c)
        )
        print(
            ("PROBE %-24s %4s %9.1f us  in %sx%s %s -> C=%s abh=%s grid=%s  " + risc)
            % (
                r["OP CODE"][:24],
                r["CORE COUNT"],
                ns / 1e3,
                r["INPUT_0_Y_PAD[LOGICAL]"].split("[")[0],
                r["INPUT_0_X_PAD[LOGICAL]"].split("[")[0],
                r["INPUT_0_MEMORY"][6:],
                r["OUTPUT_0_X_PAD[LOGICAL]"].split("[")[0],
                abh.group(1) if abh else "-",
                "x".join(g.groups()) if g else "-",
            )
        )
    print(f"PROBE_TOTAL {tot/1e6:.3f} ms over {len(rows)} ops")
