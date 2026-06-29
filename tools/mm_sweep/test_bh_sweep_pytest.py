"""
Repro the intermittent BH deadlock as a SWEEP under run_safe_pytest. One device session (the sustained-
load condition that triggers it — per-config resets mask it), looping the branch (S,Pk,block) configs of
the two hang-prone large shapes back-to-back. run_safe_pytest provides the 5s dispatch-timeout (-> raises
on a real hang) + triage + auto tt-smi -r. Run:

  scripts/run_safe_pytest.sh tools/mm_sweep/test_bh_sweep_pytest.py --timeout=100000000

On a hang: pytest fails at the offending config (printed to /tmp/bh_pyswp_progress.txt) and run_safe_pytest
dumps triage (now meaningful — the benign K-par assert is fixed). On completion: NO repro this run.
"""
import os, sys, torch, ttnn, pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_sweep_mp import gen_configs, spk_combos_for, percore

SHAPES = [(2048, 6144, 4608), (512, 6144, 9216)]
REPS = int(os.environ.get("SWEEP_REPS", "3"))  # invocations per config (sustained load)
PROG = "/tmp/bh_pyswp_progress.txt"


def test_bh_sweep(device):
    gs = device.compute_with_storage_grid_size()
    GX, GY = gs.x, gs.y
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    work = []
    for M, K, N in SHAPES:
        Kt = K // 32
        for S, Pk in spk_combos_for(Kt, GY, GX):
            pcM, pcN = percore(M, N, S, Pk, GX, GY)
            for mb, kb, nb, sbh, sbw in gen_configs(pcM, pcN, Kt // max(1, Pk)):
                work.append((S, Pk, M, K, N, mb, kb, nb, sbh, sbw))
    open(PROG, "w").write(f"total {len(work)} configs\n")

    cur_spk = cur_shape = None
    a = b = None
    for i, (S, Pk, M, K, N, mb, kb, nb, sbh, sbw) in enumerate(work):
        if (S, Pk) != cur_spk:
            for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED"):
                os.environ.pop(k, None)
            os.environ["TT_MM_NUM_SLICES"] = str(S)
            if Pk > 1:
                os.environ["TT_MM_K_SLICES"] = str(Pk)
                os.environ["TT_MM_K_FUSED"] = "1"
            cur_spk = (S, Pk)
            cur_shape = None
        if (M, K, N) != cur_shape:
            for t in (a, b):
                if t is not None:
                    t.deallocate()
            a = ttnn.from_torch(
                torch.randn(M, K, dtype=torch.bfloat16), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )
            b = ttnn.from_torch(
                torch.randn(K, N, dtype=torch.bfloat16), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )
            cur_shape = (M, K, N)
        with open(PROG, "a") as f:
            f.write(f"[{i}/{len(work)}] S{S}Pk{Pk} {M}x{K}x{N} mb{mb}kb{kb}nb{nb} sb{sbh}x{sbw}\n")
        device.clear_program_cache()
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sbh,
            subblock_w=sbw,
            compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
        )
        try:
            for _ in range(REPS):
                o = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc, config=cfg)
                ttnn.synchronize_device(device)
                o.deallocate()
            if os.environ.get("PROFILE") == "1":
                ttnn.ReadDeviceProfiler(device)  # confirmation test: does enabling the profiler trigger hangs?
        except Exception as e:
            msg = str(e)
            # Benign L1/block program-build throw (no dispatch happened -> device clean): skip + continue.
            # A real hang is a dispatch timeout (system_memory_manager.cpp:757) -> re-raise so safe_pytest
            # triages + resets (that is the deadlock we are hunting).
            if "program.cpp" in msg:
                with open(PROG, "a") as f:
                    f.write(f"    build-skip: {msg[:80]}\n")
                continue
            with open(PROG, "a") as f:
                f.write(f"    HANG/device-error: {msg[:100]}\n")
            raise
    open(PROG, "a").write("COMPLETED ALL — no hang\n")
