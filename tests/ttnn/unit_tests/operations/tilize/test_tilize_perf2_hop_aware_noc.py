# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 / hop_aware_noc harness (opt-in: TILIZE_PERF_EXPERIMENTS=1), run under --profile.

HOP_CASES="0,7,w256"   LOOSE_CASES indices, or an extra key of EXTRA below
HOP_VARIANTS="head,hop_aware_noc/kernels_dyn_hop"
                        kernel dirs relative to perf_experiments/ ("head" = the op's kernels/).
                        A dir whose basename starts with kernels_dyn_ runs with
                        pd.WRITE_NOC_SPLIT = 1 (both DM kernels in DM_DYNAMIC_NOC, write_noc_split = 1
                        passed to store_rows, which the variant reads as "use the hop mask").
                        A dir whose basename starts with kernels_dynr_ runs with pd.READ_NOC_SPLIT = 1
                        (the StickProducer reads' NoC chosen per bank; disables bank_coalesced).
                        A dir whose basename starts with kernels_duo_ runs with the two-RISC-V
                        host patch (hop_aware_noc/host_patch.py); kernels_dedf_ adds its flag semaphore;
                        kernels_dedg_ (the graduation candidate) gets host_patch.install_gated.
                        "grad" = graduated_descriptor.py (the patch applied to the host) + kernels_dedg_hop.
                        "grad2" = hop_aware_noc/graduate/ (the rebase onto be093526489: its descriptor
                        + kernels/). "head" / "grad2" take descriptor-knob overrides after "@", joined
                        by "+": "grad2@HOP_WRITE_MIN_SAVING=0" (values are Python literals).
HOP_CHECK=0            skip the golden contract check (ablated variants)
HOP_REPEAT=1           runs per (case, variant), interleaved variant-major inside each repeat
Prints `P2 case=<c> variant=<v> done` per run (p2_breakdown/label_ns.py pairs them with the report).
"""
import ast
import importlib.util
import os
from pathlib import Path

import pytest

import ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from eval.feature_matrix import cartesian
from eval.golden_tests.tilize import helpers
from eval.golden_tests.tilize.feature_spec import LOOSE_CASES, TARGET
from ttnn.operations.tilize import INPUT_TAGGERS  # type: ignore

EXP = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments"
_DRAM_IL = {"kind": "interleaved", "buffer": ttnn.BufferType.DRAM}
_BF = dict(dtype=ttnn.bfloat16, output_dtype=ttnn.bfloat16)
EXTRA = {
    "w256": {
        "inputs": ({"input_shape": [1, 1, 8192, 256], "shard_api": "none", "in": _DRAM_IL, "out": _DRAM_IL},),
        **_BF,
    },
    "w1024": {
        "inputs": ({"input_shape": [1, 1, 2048, 1024], "shard_api": "none", "in": _DRAM_IL, "out": _DRAM_IL},),
        **_BF,
    },
    "w128": {
        "inputs": ({"input_shape": [1, 1, 16384, 128], "shard_api": "none", "in": _DRAM_IL, "out": _DRAM_IL},),
        **_BF,
    },
}
_L1_IL = {"kind": "interleaved", "buffer": ttnn.BufferType.L1}
_G64 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
_G16 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})


def _sh(grid, shard, scheme, buffer=ttnn.BufferType.L1):
    return {
        "kind": "sharded",
        "buffer": buffer,
        "grid": grid,
        "shard_shape": shard,
        "orientation": ttnn.ShardOrientation.ROW_MAJOR,
        "scheme": scheme,
    }


_HS, _BS, _WS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
)


def _x(shape, i, o, **dt):
    return {
        "inputs": (
            {
                "input_shape": shape,
                "shard_api": "legacy_2d" if "sharded" in (i["kind"], o["kind"]) else "none",
                "in": i,
                "out": o,
            },
        ),
        **(dt or _BF),
    }


EXTRA.update(
    {
        # resident output (BRISC idle) <- DRAM interleaved input: the reader twin's regime
        "o_hs256": _x([1, 1, 8192, 256], _DRAM_IL, _sh(_G64, (128, 256), _HS)),
        "o_hs1024": _x([1, 1, 2048, 1024], _DRAM_IL, _sh(_G64, (32, 1024), _HS)),
        "o_bs1024": _x([1, 1, 1024, 1024], _DRAM_IL, _sh(_G64, (128, 128), _BS)),
        "o_hs_k4": _x(
            [1, 1, 128, 512],
            _DRAM_IL,
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}), (32, 512), _HS),
        ),
        # resident input (NCRISC idle) -> DRAM interleaved output: the writer half's regime
        "hs64": _x([1, 1, 16384, 64], _sh(_G64, (256, 64), _HS), _DRAM_IL),
        "hs256": _x([1, 1, 8192, 256], _sh(_G64, (128, 256), _HS), _DRAM_IL),
        "hs1024": _x([1, 1, 2048, 1024], _sh(_G64, (32, 1024), _HS), _DRAM_IL),
        "hs512f": _x(
            [1, 1, 2048, 512], _sh(_G64, (32, 512), _HS), _DRAM_IL, dtype=ttnn.float32, output_dtype=ttnn.float32
        ),
        "hs512s": _x([1, 1, 512, 512], _sh(_G16, (32, 512), _HS), _DRAM_IL),  # 16 Tensix cores, small
        "bs1024": _x([1, 1, 1024, 1024], _sh(_G64, (128, 128), _BS), _DRAM_IL),
        "ws2048": _x([1, 1, 256, 2048], _sh(_G64, (256, 32), _WS), _DRAM_IL),
        "hs_tiny": _x(
            [1, 1, 64, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}), (32, 512), _HS),
            _DRAM_IL,
        ),
        "hs_k4": _x(
            [1, 1, 128, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}), (32, 512), _HS),
            _DRAM_IL,
        ),
        "hs_k8": _x(
            [1, 1, 256, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}), (32, 512), _HS),
            _DRAM_IL,
        ),
        "hs_k32": _x(
            [1, 1, 1024, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}), (32, 512), _HS),
            _DRAM_IL,
        ),
        "hs_2c_big": _x(
            [1, 1, 512, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}), (256, 512), _HS),
            _DRAM_IL,
        ),
        "hs_acc": _x(
            [1, 1, 2048, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}), (64, 512), _HS),
            _DRAM_IL,
        ),
        # L1 interleaved input (reads on NoC0 from L1 banks) -> DRAM
        "l1_256": _x([1, 1, 2048, 256], _L1_IL, _DRAM_IL),
        "l1_64": _x([1, 1, 16384, 64], _L1_IL, _DRAM_IL),
        "l1_pad": {
            "inputs": (
                {
                    "input_shape": [1, 1, 1000, 100],
                    "shard_api": "none",
                    "in": _L1_IL,
                    "out": _DRAM_IL,
                    "pad_mode": "auto",
                },
            ),
            **_BF,
        },
        "l1_retile": {
            "inputs": (
                {
                    "input_shape": [1, 1, 4096, 64],
                    "shard_api": "none",
                    "in": _L1_IL,
                    "out": _DRAM_IL,
                    "in_tile_height": 32,
                    "tile_height": 16,
                },
            ),
            **_BF,
        },
        "l1_lowl1": {
            "inputs": (
                {"input_shape": [1, 1, 16384, 64], "shard_api": "none", "in": _L1_IL, "out": _DRAM_IL, "low_l1": True},
            ),
            **_BF,
        },
        "l1_1024": _x([1, 1, 2048, 1024], _L1_IL, _DRAM_IL),
        "l1_pad_big": {
            "inputs": (
                {
                    "input_shape": [1, 1, 4000, 500],
                    "shard_api": "none",
                    "in": _L1_IL,
                    "out": _DRAM_IL,
                    "pad_mode": "auto",
                },
            ),
            **_BF,
        },
        "l1_128": _x([1, 1, 128, 512], _L1_IL, _DRAM_IL),  # 64 tiles, 64 Tensix cores
        "l1_512": _x([1, 1, 512, 512], _L1_IL, _DRAM_IL),  # 256 tiles
        # DRAM -> DRAM interleaved sweep (the DRAM-input carve-out)
        "d8192x32": _x([1, 1, 8192, 32], _DRAM_IL, _DRAM_IL),
        "d4096x32": _x([1, 1, 4096, 32], _DRAM_IL, _DRAM_IL),
        "d4096x64": _x([1, 1, 4096, 64], _DRAM_IL, _DRAM_IL),
        "d2048x128": _x([1, 1, 2048, 128], _DRAM_IL, _DRAM_IL),
        "d64x8192": _x([1, 1, 64, 8192], _DRAM_IL, _DRAM_IL),
        "d32x16384": _x([1, 1, 32, 16384], _DRAM_IL, _DRAM_IL),
        "d16384x32f": _x([1, 1, 16384, 32], _DRAM_IL, _DRAM_IL, dtype=ttnn.float32, output_dtype=ttnn.float32),
        "d4096x32f": _x([1, 1, 4096, 32], _DRAM_IL, _DRAM_IL, dtype=ttnn.float32, output_dtype=ttnn.float32),
        "d4096x64f": _x([1, 1, 4096, 64], _DRAM_IL, _DRAM_IL, dtype=ttnn.float32, output_dtype=ttnn.float32),
        # writer Tensix-core-count sweep, resident HEIGHT_SHARDED input -> DRAM (the small-output carve-out)
        **{
            f"hsb_k{k}": _x(
                [1, 1, 256 * k, 512],
                _sh(
                    ttnn.CoreRangeSet(
                        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(k, 8) - 1, max(1, k // 8) - 1))}
                    ),
                    (256, 512),
                    _HS,
                ),
                _DRAM_IL,
            )
            for k in (4, 8, 16, 32, 64)
        },
        **{
            f"hs_k{k}": _x(
                [1, 1, 32 * k, 512],
                _sh(
                    ttnn.CoreRangeSet(
                        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(k, 8) - 1, max(1, k // 8) - 1))}
                    ),
                    (32, 512),
                    _HS,
                ),
                _DRAM_IL,
            )
            for k in (16, 48)
        },
        "hs_k16w": _x([1, 1, 512, 2048], _sh(_G16, (32, 2048), _HS), _DRAM_IL),  # 16 Tensix cores x 64 tiles
        "l1_k64": _x([1, 1, 64, 1024], _L1_IL, _DRAM_IL),  # 64 tiles
        "l1_4096x32": _x([1, 1, 4096, 32], _L1_IL, _DRAM_IL),  # 128 tiles, one tile-column
        # confirmation points (not used to fit the carve-outs)
        "c_d2048x64": _x([1, 1, 2048, 64], _DRAM_IL, _DRAM_IL),  # 4 KiB / core
        "c_d1024x256": _x([1, 1, 1024, 256], _DRAM_IL, _DRAM_IL),  # 8 KiB / core
        "c_d2048x64f": _x([1, 1, 2048, 64], _DRAM_IL, _DRAM_IL, dtype=ttnn.float32, output_dtype=ttnn.float32),  # 8 KiB
        "c_d32x4096": _x([1, 1, 32, 4096], _DRAM_IL, _DRAM_IL),  # 4 KiB, one position
        "c_d8192x64": _x([1, 1, 8192, 64], _DRAM_IL, _DRAM_IL),  # 16 KiB
        "c_d2048x256": _x([1, 1, 2048, 256], _DRAM_IL, _DRAM_IL),  # 16 KiB
        "c_d4096x128": _x([1, 1, 4096, 128], _DRAM_IL, _DRAM_IL),  # 16 KiB
        "c_d1024x1024": _x([1, 1, 1024, 1024], _DRAM_IL, _DRAM_IL),  # 32 KiB
        "c_bs16": _x([1, 1, 512, 512], _sh(_G16, (128, 128), _BS), _DRAM_IL),  # 16 Tensix cores
        "c_bs32": _x(
            [1, 1, 1024, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))}), (128, 128), _BS),
            _DRAM_IL,
        ),
        "c_hs64_k24": _x(
            [1, 1, 1536, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))}), (64, 512), _HS),
            _DRAM_IL,
        ),
        "c_hs_k40": _x(
            [1, 1, 1280, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}), (32, 512), _HS),
            _DRAM_IL,
        ),
        "c_ws16": _x([1, 1, 256, 512], _sh(_G16, (256, 32), _WS), _DRAM_IL),
        "c_l1_256x64": _x([1, 1, 256, 64], _L1_IL, _DRAM_IL),  # 16 tiles
        "c_l1_4096x256": _x([1, 1, 4096, 256], _L1_IL, _DRAM_IL),
        "c_hs_k20": _x(
            [1, 1, 640, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}), (32, 512), _HS),
            _DRAM_IL,
        ),
        "c_hsb_k20": _x(
            [1, 1, 5120, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}), (256, 512), _HS),
            _DRAM_IL,
        ),
        "c_hsb_k24": _x(
            [1, 1, 6144, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))}), (256, 512), _HS),
            _DRAM_IL,
        ),
        "c_bs24": _x(
            [1, 1, 512, 768],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 3))}), (128, 128), _BS),
            _DRAM_IL,
        ),
        "c_hs_k24": _x(
            [1, 1, 768, 512],
            _sh(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))}), (32, 512), _HS),
            _DRAM_IL,
        ),
        "l1_64_t16": {
            "inputs": (
                {
                    "input_shape": [1, 1, 16384, 64],
                    "shard_api": "none",
                    "in": _L1_IL,
                    "out": _DRAM_IL,
                    "tile_height": 16,
                },
            ),
            **_BF,
        },
    }
)
CASES = os.environ.get("HOP_CASES", "0").split(",")
VARIANTS = os.environ.get("HOP_VARIANTS", "head").split(",")
CHECK = os.environ.get("HOP_CHECK", "1") != "0"
REPEAT = int(os.environ.get("HOP_REPEAT", "1"))
_GRAD2 = None


def _case(c):
    return EXTRA[c] if c in EXTRA else LOOSE_CASES[int(c)]


def _axes(case):
    inputs = case["inputs"]
    dt = case.get("dtype", ttnn.bfloat16)
    odt = case.get("output_dtype", dt)
    return next(a for a in cartesian(TARGET, INPUT_TAGGERS, inputs) if a["dtype"] == dt and a["output_dtype"] == odt)


def _host_patch():
    spec = importlib.util.spec_from_file_location("hop_host_patch", EXP / "hop_aware_noc/host_patch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Order: repeat (slowest) -> case -> variant slot (fastest); odd repeats run the variants in reverse
# order, so every (case, variant) pair is measured back to back in both orders.
@pytest.mark.parametrize("slot", range(len(VARIANTS)))
@pytest.mark.parametrize("idx", CASES)
@pytest.mark.parametrize("rep", range(REPEAT))
def test_hop_variant(device, monkeypatch, idx, slot, rep):
    variant = (VARIANTS if rep % 2 == 0 else VARIANTS[::-1])[slot]
    base, _, knobs = variant.partition("@")
    if base in ("head", "grad2"):
        if base == "grad2":
            global _GRAD2
            if _GRAD2 is None:
                spec = importlib.util.spec_from_file_location(
                    "hop_grad2_pd", EXP / "hop_aware_noc/graduate/tilize_program_descriptor.py"
                )
                _GRAD2 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_GRAD2)
            mod = _GRAD2
            import sys

            def _cpd(*a, **k):
                desc = mod.create_program_descriptor(*a, **k)
                try:
                    defs = [tuple(d) for d in desc.kernels[1].defines]
                except Exception as e:  # noqa: BLE001
                    defs = [("ERR", str(e))]
                hop = [v for n, v in defs if n == "TILIZE_HOP_WRITE_MIN_SAVING"]
                sb = [v for n, v in defs if n == "TILIZE_SUB_BLOCK_TILES"]
                try:
                    rct = list(desc.kernels[0].compile_time_args)
                    wct = list(desc.kernels[1].compile_time_args)
                    ncores = desc.kernels[1].core_ranges.num_cores()
                    info = f"cores={ncores} bw={wct[1]} rpq={wct[10]} coalesce={rct[26]} co_read={rct[28]} resident_in={rct[9]}"
                except Exception as e:  # noqa: BLE001
                    info = f"info_err={e}"
                print(
                    f"HOP_ENGAGED case={idx} variant={variant} hop={hop[0] if hop else 0} sub_block={sb[0] if sb else 0} {info}"
                )
                return desc

            monkeypatch.setattr(sys.modules["ttnn.operations.tilize.tilize"], "create_program_descriptor", _cpd)
        else:
            mod = pd
        for knob in filter(None, knobs.split("+")):
            name, _, value = knob.partition("=")
            assert hasattr(mod, name), name
            value = ast.literal_eval(value)
            if name == "KERNEL_DIR":  # a dir relative to perf_experiments/
                value = EXP / value
            monkeypatch.setattr(mod, name, value)
    elif variant == "grad":
        # the graduation patch end to end: patched descriptor + kernels_dedg_hop (graduate_writer_hop.patch)
        spec = importlib.util.spec_from_file_location("hop_grad_pd", EXP / "hop_aware_noc/graduated_descriptor.py")
        gpd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gpd)
        gpd.KERNEL_DIR = EXP / "hop_aware_noc/kernels_dedg_hop"
        import sys

        monkeypatch.setattr(
            sys.modules["ttnn.operations.tilize.tilize"], "create_program_descriptor", gpd.create_program_descriptor
        )
    else:
        d = Path(variant)
        d = d if d.is_absolute() else EXP / d
        monkeypatch.setattr(pd, "KERNEL_DIR", d)
        if d.name.startswith("kernels_dyn_"):
            monkeypatch.setattr(pd, "WRITE_NOC_SPLIT", 1)
        if d.name.startswith("kernels_dynr_"):
            monkeypatch.setattr(pd, "READ_NOC_SPLIT", 1)
        if d.name.startswith("kernels_duo_"):
            _host_patch().install(monkeypatch, pd)
        if d.name.startswith("kernels_dedf_"):
            _host_patch().install_flag(monkeypatch, pd)
        if d.name.startswith("kernels_dedg_"):
            _host_patch().install_gated(monkeypatch, pd)
        if d.name.startswith("kernels_dedr_"):
            _host_patch().install_dedr(monkeypatch, pd)
    case = _case(idx)
    axes = _axes(case)
    if not CHECK:
        monkeypatch.setattr(helpers, "check_output", lambda *a, **k: None)
    helpers.run_tilize(case["inputs"], device=device, extras=case.get("extras"), **axes)
    ttnn.synchronize_device(device)
    print(f"P2 case={idx} variant={variant} done")


# Back-to-back programs on one device (HOP_B2B=1): the hop handshake touches firmware-visible NoC
# counters, so run engaged / non-engaged / DRAM -> DRAM tilizes and a non-tilize op interleaved,
# with repeated program-cache hits, every output checked. Run under --dev (idle ASSERTs, watcher).
B2B_SEQ = os.environ.get(
    "HOP_B2B_SEQ", "7,hs_k8,0,7,7,l1_64,hs64,8,l1_64,7,eltwise,hs256,0,hs256,bs1024,l1_256,7"
).split(",")


@pytest.mark.skipif(os.environ.get("HOP_B2B", "0") != "1", reason="HOP_B2B=1")
def test_hop_back_to_back(device, monkeypatch):
    import sys

    import torch

    spec = importlib.util.spec_from_file_location(
        "hop_grad2_pd_b2b", EXP / "hop_aware_noc/graduate/tilize_program_descriptor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(
        sys.modules["ttnn.operations.tilize.tilize"], "create_program_descriptor", mod.create_program_descriptor
    )
    before = device.num_program_cache_entries()
    for step, c in enumerate(B2B_SEQ):
        if c == "eltwise":
            a = torch.randn(1, 1, 1024, 1024, dtype=torch.bfloat16)
            ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
            out = ttnn.to_torch(ttnn.add(ta, ta))
            assert torch.equal(out, a + a)
        else:
            case = _case(c)
            helpers.run_tilize(case["inputs"], device=device, extras=case.get("extras"), **_axes(case))
        ttnn.synchronize_device(device)
        print(f"B2B step={step} case={c} ok cache_entries={device.num_program_cache_entries() - before}")
