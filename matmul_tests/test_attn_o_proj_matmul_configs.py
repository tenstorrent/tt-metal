import pytest
import torch

import ttnn
from matmul_tests._matmul_config_sweep import MatmulSweepSpec, build_configs, run_matmul_sweep_test


SPEC = MatmulSweepSpec(
    m=32,
    k=1536,
    n=1536,
    in0_dtype=ttnn.bfloat16,
    in1_dtype=ttnn.bfloat4_b,
    out_dtype=ttnn.bfloat8_b,
    torch_in0_dtype=torch.bfloat16,
    torch_in1_dtype=torch.bfloat16,
    math_fidelity=ttnn.MathFidelity.LoFi,
    pcc=0.95,
)

WIDTH_COMBOS = [
    (1, 1, 48, 8),
    (2, 1, 24, 8),
    (3, 1, 16, 8),
    (4, 1, 12, 6),
    (4, 1, 12, 3),
    (6, 1, 8, 8),
    (8, 1, 6, 6),
    (8, 1, 6, 3),
    (8, 2, 3, 3),
    (8, 2, 3, 1),
    (8, 3, 2, 2),
    (8, 3, 2, 1),
    (8, 6, 1, 1),
]
HEIGHT_COMBOS = [(1, 1, 48, 48)]
BLOCK_COMBOS = [
    (1, 1, 48, 8),
    (2, 1, 24, 8),
    (3, 1, 16, 8),
    (4, 1, 12, 6),
    (4, 1, 12, 3),
    (6, 1, 8, 8),
    (8, 1, 6, 6),
    (8, 1, 6, 3),
]
MCAST_COMBOS = [
    (8, 1, 6, 6, 8),
    (8, 2, 3, 3, 8),
    (8, 3, 2, 2, 8),
    (8, 6, 1, 1, 8),
]
DRAM_CONFIGS = [
    ("dram_sharded_12banks_8cores", 8, 6, 6),
    ("dram_sharded_12banks_16cores", 16, 3, 3),
    ("dram_sharded_12banks_24cores", 24, 2, 2),
    ("dram_sharded_12banks_48cores", 48, 1, 1),
    ("mixed_in0_l1_in1_dram_outl1_8cores", 8, 6, 6),
    ("mixed_in0_l1_in1_dram_outl1_16cores", 16, 3, 3),
    ("mixed_in0_l1_in1_dram_outl1_24cores", 24, 2, 2),
    ("mixed_in0_l1_in1_dram_outl1_48cores", 48, 1, 1),
    ("in0_l1_sharded_in1_dram_sharded_l1_out", 48, 1, 1),
]

CONFIGS = build_configs(
    SPEC, WIDTH_COMBOS, HEIGHT_COMBOS, BLOCK_COMBOS, MCAST_COMBOS, DRAM_CONFIGS, include_auto_default=True
)
CONFIG_IDS = [name for name, _ in CONFIGS]
SKIP_CONFIGS = {
    name: "HEIGHT_SHARDED M=32 has one M tile and this one-core L1 footprint exceeds the verified budget"
    for name, _ in CONFIGS
    if name.startswith("height_sharded_")
}
XFAIL_CONFIGS = {}
ALLOW_PCC_FAILURE_CONFIGS = {
    "dram_sharded_12banks_16cores",
    "dram_sharded_12banks_24cores",
    "dram_sharded_12banks_48cores",
    "mixed_in0_l1_in1_dram_outl1_16cores",
    "mixed_in0_l1_in1_dram_outl1_24cores",
    "mixed_in0_l1_in1_dram_outl1_48cores",
}


@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_attn_o_proj_matmul_configs(device, config_name, cfg_builder):
    run_matmul_sweep_test(
        device, SPEC, config_name, cfg_builder, SKIP_CONFIGS, XFAIL_CONFIGS, ALLOW_PCC_FAILURE_CONFIGS
    )
