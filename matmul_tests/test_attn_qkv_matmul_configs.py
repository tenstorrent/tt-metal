import pytest
import torch

import ttnn
from matmul_tests._matmul_config_sweep import MatmulSweepSpec, build_configs, cfg_block_explicit, run_matmul_sweep_test


SPEC = MatmulSweepSpec(
    m=32,
    k=1536,
    n=2048,
    in0_dtype=ttnn.bfloat16,
    in1_dtype=ttnn.bfloat8_b,
    out_dtype=ttnn.bfloat16,
    torch_in0_dtype=torch.bfloat16,
    torch_in1_dtype=torch.bfloat16,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    pcc=0.98,
)

WIDTH_COMBOS = [
    (1, 1, 64, 8),
    (2, 1, 32, 8),
    (4, 1, 16, 6),
    (4, 1, 16, 3),
    (8, 1, 8, 6),
    (8, 1, 8, 3),
    (8, 2, 4, 3),
    (8, 2, 4, 1),
]
HEIGHT_COMBOS = [(1, 1, 64, 48)]
BLOCK_COMBOS = [
    (1, 1, 64, 8),
    (2, 1, 32, 8),
    (4, 1, 16, 6),
    (4, 1, 16, 3),
    (8, 1, 8, 6),
    (8, 1, 8, 3),
]
MCAST_COMBOS = [
    (8, 1, 8, 8, 8),
    (8, 2, 4, 4, 8),
    (8, 4, 2, 2, 8),
    (8, 8, 1, 1, 8),
]
DRAM_CONFIGS = [
    ("dram_sharded_12banks_8cores", 8, 8, 6),
    ("dram_sharded_12banks_16cores", 16, 4, 3),
    ("mixed_in0_l1_in1_dram_outl1_8cores", 8, 8, 6),
    ("mixed_in0_l1_in1_dram_outl1_16cores", 16, 4, 3),
    ("in0_l1_sharded_in1_dram_sharded_l1_out", 16, 4, 3),
]

CONFIGS = build_configs(
    SPEC, WIDTH_COMBOS, HEIGHT_COMBOS, BLOCK_COMBOS, MCAST_COMBOS, DRAM_CONFIGS, include_auto_default=True
)
CONFIGS.append(("block_sharded_1x1_pcn64_ibw8_in1height", cfg_block_explicit(SPEC, 1, 1, 64, 8, "height")))
CONFIG_IDS = [name for name, _ in CONFIGS]
SKIP_CONFIGS = {}
for _name, _ in CONFIGS:
    if _name.startswith("height_sharded_"):
        SKIP_CONFIGS[
            _name
        ] = "HEIGHT_SHARDED M=32 has one M tile and this one-core L1 footprint exceeds the verified budget"
    if _name.startswith("block_sharded_1x1_"):
        SKIP_CONFIGS[_name] = "BLOCK_SHARDED 1x1/pCN64 exceeds the verified L1 footprint budget"
    if _name == "width_sharded_1x1_pcn64_ibw8":
        SKIP_CONFIGS[_name] = "WIDTH_SHARDED 1x1/pCN64 exceeds the verified L1 footprint budget"
XFAIL_CONFIGS = {}
ALLOW_PCC_FAILURE_CONFIGS = {
    "dram_sharded_12banks_16cores",
    "mixed_in0_l1_in1_dram_outl1_16cores",
}


@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_attn_qkv_matmul_configs(device, config_name, cfg_builder):
    run_matmul_sweep_test(
        device, SPEC, config_name, cfg_builder, SKIP_CONFIGS, XFAIL_CONFIGS, ALLOW_PCC_FAILURE_CONFIGS
    )
