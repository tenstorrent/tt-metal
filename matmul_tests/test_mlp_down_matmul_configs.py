import pytest
import torch

import ttnn
from matmul_tests._matmul_config_sweep import MatmulSweepSpec, build_configs, run_matmul_sweep_test


SPEC = MatmulSweepSpec(
    m=32,
    k=8960,
    n=1536,
    in0_dtype=ttnn.bfloat8_b,
    in1_dtype=ttnn.bfloat4_b,
    out_dtype=ttnn.bfloat8_b,
    torch_in0_dtype=torch.bfloat16,
    torch_in1_dtype=torch.bfloat16,
    math_fidelity=ttnn.MathFidelity.LoFi,
    pcc=0.94,
)

WIDTH_COMBOS = [
    (1, 1, 48, 7),
    (1, 1, 48, 5),
    (2, 1, 24, 7),
    (2, 1, 24, 5),
    (4, 1, 12, 7),
    (4, 1, 12, 5),
    (8, 1, 6, 7),
    (8, 1, 6, 5),
]
HEIGHT_COMBOS = [(1, 1, 48, 280)]
BLOCK_COMBOS = [
    (1, 1, 48, 7),
    (1, 1, 48, 5),
    (2, 1, 24, 7),
    (2, 1, 24, 5),
    (4, 1, 12, 7),
    (4, 1, 12, 5),
    (8, 1, 6, 7),
    (8, 1, 6, 5),
]
MCAST_COMBOS = [
    (8, 1, 6, 6, 8),
    (8, 2, 3, 3, 8),
    (8, 3, 2, 2, 8),
    (8, 6, 1, 1, 8),
]
DRAM_CONFIGS = [
    ("dram_sharded_12banks_8cores_ibw5", 8, 6, 5),
    ("dram_sharded_12banks_8cores_ibw7", 8, 6, 7),
    ("mixed_in0_l1_in1_dram_outl1_8cores_ibw5", 8, 6, 5),
    ("mixed_in0_l1_in1_dram_outl1_8cores_ibw7", 8, 6, 7),
    ("in0_l1_sharded_in1_dram_sharded_l1_out", 48, 1, 1),
]

CONFIGS = build_configs(
    SPEC, WIDTH_COMBOS, HEIGHT_COMBOS, BLOCK_COMBOS, MCAST_COMBOS, DRAM_CONFIGS, include_auto_default=True
)
CONFIG_IDS = [name for name, _ in CONFIGS]
SKIP_CONFIGS = {
    name: "HEIGHT_SHARDED M=32 has one M tile and K_TILES=280 makes the verified L1 footprint too large"
    for name, _ in CONFIGS
    if name.startswith("height_sharded_")
}
XFAIL_CONFIGS = {}
ALLOW_PCC_FAILURE_CONFIGS = set()


@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_mlp_down_matmul_configs(device, config_name, cfg_builder):
    run_matmul_sweep_test(
        device, SPEC, config_name, cfg_builder, SKIP_CONFIGS, XFAIL_CONFIGS, ALLOW_PCC_FAILURE_CONFIGS
    )
