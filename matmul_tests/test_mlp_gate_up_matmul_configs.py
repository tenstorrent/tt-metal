import pytest
import torch

import ttnn
from matmul_tests._matmul_config_sweep import MatmulSweepSpec, build_configs, cfg_block_explicit, run_matmul_sweep_test


SPEC = MatmulSweepSpec(
    m=32,
    k=1536,
    n=17920,
    in0_dtype=ttnn.bfloat16,
    in1_dtype=ttnn.bfloat4_b,
    out_dtype=ttnn.bfloat8_b,
    torch_in0_dtype=torch.bfloat16,
    torch_in1_dtype=torch.bfloat16,
    math_fidelity=ttnn.MathFidelity.LoFi,
    pcc=0.94,
)

WIDTH_COMBOS = [
    (1, 1, 560, 8),
    (2, 1, 280, 8),
    (4, 1, 140, 6),
    (4, 1, 140, 3),
    (8, 1, 70, 6),
    (8, 1, 70, 3),
    (8, 2, 35, 3),
    (8, 2, 35, 1),
]
HEIGHT_COMBOS = [(1, 1, 560, 48)]
BLOCK_COMBOS = [
    (1, 1, 560, 8),
    (2, 1, 280, 8),
    (4, 1, 140, 6),
    (4, 1, 140, 3),
    (8, 1, 70, 6),
    (8, 1, 70, 3),
]
MCAST_COMBOS = [
    (8, 1, 70, 7, 3),
    (8, 2, 35, 7, 4),
    (8, 5, 14, 7, 4),
    (8, 7, 10, 5, 4),
]
DRAM_CONFIGS = [
    ("width_sharded_l1_in1_dram12_outl1_16cores_ibw3", 16, 35, 3),
    ("baseline_6banks_16cores", 16, 35, 3, 6),
    ("dram_sharded_12banks_16cores", 16, 35, 3),
    ("dram_sharded_12banks_8cores_ibw6", 8, 70, 6),
    ("dram_sharded_12banks_8cores_ibw3", 8, 70, 3),
    ("mixed_in0_l1_in1_dram6_outl1_16cores", 16, 35, 3, 6),
    ("mixed_in0_l1_in1_dram12_outl1_16cores", 16, 35, 3),
    ("mixed_in0_l1_in1_dram12_outl1_8cores_ibw6", 8, 70, 6),
    ("mixed_in0_l1_in1_dram12_outl1_8cores_ibw3", 8, 70, 3),
    ("in0_l1_sharded_in1_dram_sharded_l1_out", 16, 35, 3),
]

CONFIGS = build_configs(
    SPEC,
    WIDTH_COMBOS,
    HEIGHT_COMBOS,
    BLOCK_COMBOS,
    MCAST_COMBOS,
    DRAM_CONFIGS,
    exact_gate=True,
    include_auto_default=True,
)
CONFIGS.extend(
    [
        ("block_sharded_1x1_pcn560_ibw8_in1height", cfg_block_explicit(SPEC, 1, 1, 560, 8, "height")),
        ("block_sharded_2x1_pcn280_ibw8_in1height", cfg_block_explicit(SPEC, 2, 1, 280, 8, "height")),
        ("block_sharded_4x1_pcn140_ibw6_in1height", cfg_block_explicit(SPEC, 4, 1, 140, 6, "height")),
    ]
)
CONFIG_IDS = [name for name, _ in CONFIGS]
SKIP_CONFIGS = {}
for _name, _ in CONFIGS:
    if _name.startswith("height_sharded_"):
        SKIP_CONFIGS[
            _name
        ] = "HEIGHT_SHARDED M=32 has one M tile and this one-core L1 footprint exceeds the verified budget"
    if _name.startswith("block_sharded_1x1_") or _name.startswith("block_sharded_2x1_"):
        SKIP_CONFIGS[_name] = "BLOCK_SHARDED pCN is too large for the verified L1 footprint budget"
    if _name.startswith("block_sharded_4x1_pcn140_ibw6_"):
        SKIP_CONFIGS[_name] = "BLOCK_SHARDED 4x1/pCN140/in0_block_w=6 exceeds the verified L1 footprint budget"
    if _name in {
        "width_sharded_1x1_pcn560_ibw8",
        "width_sharded_2x1_pcn280_ibw8",
        "width_sharded_4x1_pcn140_ibw6",
    }:
        SKIP_CONFIGS[_name] = "WIDTH_SHARDED pCN/in0_block_w combination exceeds the verified L1 footprint budget"

XFAIL_CONFIGS = {}
ALLOW_PCC_FAILURE_CONFIGS = {
    "width_sharded_l1_in1_dram12_outl1_16cores_ibw3",
}


@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_mlp_gate_up_matmul_configs(device, config_name, cfg_builder):
    run_matmul_sweep_test(
        device, SPEC, config_name, cfg_builder, SKIP_CONFIGS, XFAIL_CONFIGS, ALLOW_PCC_FAILURE_CONFIGS
    )
