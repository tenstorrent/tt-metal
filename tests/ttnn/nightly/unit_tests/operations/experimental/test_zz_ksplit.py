import os
import pytest
import torch
import ttnn


@pytest.mark.timeout(300)
def test_ksplit(device):
    # Split-K (A2) correctness: minimal_matmul with TT_MM_K_SLICES=Pk writes Pk partial MxN results
    # stacked along M into [Pk*M, N]; host reshapes -> [Pk,M,N] and sums -> [M,N]. PCC vs torch.
    M = int(os.environ.get("FL_M", "32"))
    K = int(os.environ.get("FL_K", "6144"))
    N = int(os.environ.get("FL_N", "512"))
    Pk = int(os.environ.get("TT_MM_K_SLICES", "8"))
    os.environ["TT_MM_K_SLICES"] = str(Pk)
    os.environ["TT_MM_NUM_SLICES"] = "1"  # avoid the auto N-slicer overflowing Pk*S > grid.y
    torch.manual_seed(0)
    ti = torch.randn((M, K), dtype=torch.bfloat16)
    wi = torch.randn((K, N), dtype=torch.bfloat16)
    tt_i = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_w = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out = ttnn.experimental.minimal_matmul(tt_i, tt_w, compute_kernel_config=cc)
    got = ttnn.to_torch(out).float()
    print(f"KSPLIT M{M}K{K}N{N} Pk{Pk}: raw out shape {tuple(got.shape)}")
    assert got.shape[-2] == M * Pk, f"expected stacked M={M*Pk}, got {got.shape}"
    summed = got.reshape(Pk, M, N).sum(0)
    ref = ti.float() @ wi.float()
    pcc = torch.corrcoef(torch.stack([ref.flatten(), summed.flatten()]))[0, 1].item()
    print(f"KSPLIT pcc={pcc:.5f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
    out.deallocate()


@pytest.mark.timeout(600)
def test_ksplit_perf(device):
    # Perf bench (env-driven, no pinned config so the K-par auto path engages). Runs the op FC_REPS
    # times under tracy; the orchestrator parses DEVICE KERNEL DURATION. Drive split-K via TT_MM_K_*
    # env BEFORE launch. Baseline = no env (Pk=1). B = TT_MM_K_FUSED=1 TT_MM_K_SLICES=Pk.
    M = int(os.environ.get("FL_M", "32"))
    K = int(os.environ.get("FL_K", "6144"))
    N = int(os.environ.get("FL_N", "256"))
    reps = int(os.environ.get("FC_REPS", "20"))
    torch.manual_seed(0)
    ti = torch.randn((M, K), dtype=torch.bfloat16)
    wi = torch.randn((K, N), dtype=torch.bfloat16)
    tt_i = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_w = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    for _ in range(reps):
        out = ttnn.experimental.minimal_matmul(tt_i, tt_w, compute_kernel_config=cc)
        ttnn.synchronize_device(device)
        out.deallocate()


@pytest.mark.timeout(120)
def test_ksplit_fused(device):
    # Split-K plan B (TT_MM_K_FUSED=1): bands reduce the running sum UP the K-band column on-device, so
    # the op returns the final [M, N] directly (no host sum). Set TT_MM_NUM_SLICES=1 so Pk*S <= grid.y
    # (the auto N-slicer would otherwise pick S>1 on skinny shapes and overflow the row partition).
    M = int(os.environ.get("FL_M", "32"))
    K = int(os.environ.get("FL_K", "6144"))
    N = int(os.environ.get("FL_N", "512"))
    Pk = int(os.environ.get("TT_MM_K_SLICES", "8"))
    os.environ["TT_MM_K_SLICES"] = str(Pk)
    os.environ["TT_MM_K_FUSED"] = "1"
    os.environ["TT_MM_NUM_SLICES"] = "1"
    torch.manual_seed(0)
    ti = torch.randn((M, K), dtype=torch.bfloat16)
    wi = torch.randn((K, N), dtype=torch.bfloat16)
    tt_i = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_w = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out = ttnn.experimental.minimal_matmul(tt_i, tt_w, compute_kernel_config=cc)
    got = ttnn.to_torch(out).float()
    print(f"KFUSED M{M}K{K}N{N} Pk{Pk}: out shape {tuple(got.shape)}")
    assert tuple(got.shape[-2:]) == (M, N), f"expected [M,N]=[{M},{N}], got {got.shape}"
    ref = ti.float() @ wi.float()
    pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item()
    print(f"KFUSED pcc={pcc:.5f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
    out.deallocate()
