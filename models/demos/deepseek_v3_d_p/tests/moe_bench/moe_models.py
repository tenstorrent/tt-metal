# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""MoE shapes for the BH Galaxy MoE benchmark (prefill: TtMoe on 8x4; decode: ttnn.experimental.moe_compute).

Every row is a public model's routed-expert shape. Edit a row, or add one through the environment, to benchmark
any other MoE:

    MOE_BENCH_MODELS=deepseek_v3,kimi_k2_7          # run a subset (default: all rows below)
    MOE_BENCH_CUSTOM="mymoe:H=5120,N=1536,E=160,k=6" # add a row; optional Hr= (latent routed width), act=,
                                                     # shared=, cf= (dispatch capacity factor)

Custom rows run prefill on the DeepSeek-V3 code path with a host gate (any E/k), and decode with experts_per_chip =
E / 32 (expert parallel over the whole Galaxy).
"""

import os
from dataclasses import dataclass, field

GALAXY_CHIPS = 32

# Peak numbers for a Blackhole Galaxy chip; roofline.py reads them from here. The clock and grid are measured in the
# job (tt-smi snapshot and the device's compute grid) and override these when available.
BH_DRAM_GBPS = 512.0  # 8 GDDR6 channels at 16 Gbps x 32 bit
BH_AICLK_MHZ = 1350.0
BH_FLOP_PER_CYCLE_PER_CORE_LOFI = 4096  # 8x16x16 MACs x 2 FLOP; HiFi2 halves it, HiFi4 quarters it

# Bytes per element of the formats the two paths store expert weights in.
BYTES_PER_ELEM = {"bfloat4_b": 576 / 1024, "bfloat8_b": 1088 / 1024, "bfloat16": 2.0}


@dataclass(frozen=True)
class MoEShape:
    name: str
    hidden: int  # model hidden (embedding) size
    inter: int  # routed expert intermediate size (N)
    experts: int  # routed experts (E)
    k: int  # experts per token
    routed_hidden: int | None = None  # LatentMoE routed width (Kimi K3); None = hidden
    shared_experts: int = 1
    shared_inter: int | None = None  # total shared-expert intermediate; None = inter * shared_experts
    routed_act: str = "silu"  # prefill RoutedExpertActivation name
    decode_act: str = "SILU"  # MoEActivationFunction name for moe_compute
    decode_act_limit: float | None = None
    # Prefill wiring: which TtMoe variant / config class / gate to use.
    prefill_variant: str = "deepseek_v3_d_p"
    prefill_cfg: str = "DeepSeekV3Config"
    prefill_gate: str = "DEVICE_FP32"
    capacity_factor: int = 8
    prefill_kwargs: dict = field(default_factory=dict)
    note: str = ""

    @property
    def hr(self) -> int:
        return self.routed_hidden or self.hidden

    @property
    def sh_inter(self) -> int:
        return self.shared_inter if self.shared_inter is not None else self.inter * self.shared_experts

    @property
    def experts_per_chip(self) -> int:
        return max(1, self.experts // GALAXY_CHIPS)


MODELS = {
    m.name: m
    for m in [
        MoEShape("deepseek_v3", 7168, 2048, 256, 8, prefill_cfg="DeepSeekV3Config", note="DeepSeek V3 / R1"),
        MoEShape(
            "deepseek_v4_flash",
            4096,
            2048,
            256,
            6,
            routed_act="clamped_silu_glu",
            decode_act="CLAMPED_SILU",
            decode_act_limit=10.0,
            prefill_cfg="DeepSeekV4FlashConfig",
            prefill_gate="HOST_ALL",
            prefill_kwargs={"shared_activation": "clamped_silu_glu", "clamped_silu_glu_limit": 10.0},
            note="host gate (V4 sqrtsoftplus router is benchmarked separately)",
        ),
        MoEShape(
            "deepseek_v4_pro",
            7168,
            3072,
            384,
            6,
            routed_act="clamped_silu_glu",
            decode_act="CLAMPED_SILU",
            decode_act_limit=10.0,
            prefill_cfg="DeepSeekV4ProConfig",
            prefill_gate="HOST_ALL",
            prefill_kwargs={"shared_activation": "clamped_silu_glu", "clamped_silu_glu_limit": 10.0},
            note="host gate",
        ),
        MoEShape(
            "kimi_k2_7",
            7168,
            2048,
            384,
            8,
            prefill_variant="kimi_k2_7",
            prefill_cfg="KimiK27Config",
            capacity_factor=5,
            note="routed expert overlapped with combine (production setting)",
        ),
        MoEShape(
            "kimi_k3",
            7168,
            3072,
            896,
            16,
            routed_hidden=3584,
            shared_experts=2,
            routed_act="situ",
            decode_act="SILU",
            prefill_variant="kimi_k3",
            prefill_cfg="KimiK3Config",
            capacity_factor=5,
            note="LatentMoE (routed experts at 3584); decode uses SiLU in place of SiTU",
        ),
        MoEShape(
            "minimax_m3",
            6144,
            3072,
            128,
            4,
            routed_act="swiglu_oai",
            decode_act="SWIGLU",
            prefill_cfg="MiniMaxM3Config",
            prefill_gate="HOST_ALL",
            note="host gate",
        ),
        MoEShape(
            "glm_5_3",
            6144,
            2048,
            256,
            8,
            prefill_cfg="GLM53Config",
            note="fabric2d dispatch/combine + overlap (production setting)",
        ),
    ]
}


def _parse_custom(spec: str) -> MoEShape:
    name, _, kv = spec.partition(":")
    vals = dict(item.split("=", 1) for item in kv.split(",") if item)
    return MoEShape(
        name=name.strip(),
        hidden=int(vals["H"]),
        inter=int(vals["N"]),
        experts=int(vals["E"]),
        k=int(vals["k"]),
        routed_hidden=int(vals["Hr"]) if "Hr" in vals else None,
        shared_experts=int(vals.get("shared", 1)),
        routed_act=vals.get("act", "silu"),
        decode_act={"silu": "SILU", "swiglu_oai": "SWIGLU", "gelu": "GELU"}.get(vals.get("act", "silu"), "SILU"),
        prefill_gate="HOST_ALL",
        capacity_factor=int(vals.get("cf", 8)),
        note="custom",
    )


def selected_models() -> list[MoEShape]:
    models = dict(MODELS)
    for spec in filter(None, os.environ.get("MOE_BENCH_CUSTOM", "").split(";")):
        m = _parse_custom(spec)
        models[m.name] = m
    names = [n.strip() for n in os.environ.get("MOE_BENCH_MODELS", "").split(",") if n.strip()]
    if names:
        unknown = set(names) - set(models)
        assert not unknown, f"unknown MOE_BENCH_MODELS {sorted(unknown)}; known: {sorted(models)}"
        return [models[n] for n in names]
    return list(models.values())


def decode_users_per_chip() -> list[int]:
    return [int(x) for x in os.environ.get("MOE_BENCH_DECODE_USERS", "1,8,32,128").split(",")]
