# Scratch, not for merge: the compressed DRAM matmul at the production gate/up call (kt 56, ct 1, 2 cores per bank).
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_matmul_custom_compressed import _run_dram_matmul_custom_compressed as run

def test_cd_mix_n2048(device):
    run(device, 1, 7168, 2048, formats=["bfp4", "bfp2"], cores_per_bank=2)

def test_cd_mix_n4096(device):
    run(device, 1, 7168, 4096, formats=["bfp4", "bfp2"], cores_per_bank=2)

def test_cd_bfp4_n2048(device):
    run(device, 1, 7168, 2048, formats=["bfp4"], cores_per_bank=2)

def test_cd_bfp4_n4096(device):
    run(device, 1, 7168, 4096, formats=["bfp4"], cores_per_bank=2)

def test_cd_mix024_n2048(device):
    run(device, 1, 7168, 2048, formats=["bfp4", "bfp2", "bfp0"], cores_per_bank=2)
