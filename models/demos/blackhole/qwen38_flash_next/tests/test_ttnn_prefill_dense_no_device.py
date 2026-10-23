# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab's dense-linear switches (``ttnn/prefill_dense``) without a device: the four switches, their
defaults and refusals; grid today resolving to the modules' own compute config, today's program config, no
resident weight and ``decode_matmul.prefill_linear``; the default (the wide grid on today's arithmetic: the two
fused siblings resident in bf16, nothing else); the arm configs (fidelity, fp32 accumulation, the wide grid per
shape); the resident-weight plans (nothing under grid today with dtype bf16, the bfloat8_b copies only under dtype
bf8, the fused siblings only under grid wide) and their bytes; the exact set (router, index projections) carrying no
policy in the code; the DRAM admission; and the wiring pins (decode_matmul and the weight loaders untouched, the
builder attaching after the resident experts, the slab MoE instance sharing the layer's object)."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import builder as builder_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import prefill_dense as pd
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import shared_expert

MODEL_DIR = Path(__file__).resolve().parents[1]
SLAB = 2048
DTYPE, FIDELITY, GRID, FP32 = pd.DTYPE_ENV, pd.FIDELITY_ENV, pd.GRID_ENV, pd.FP32_ACC_ENV
# The slab's dense shapes (local K x N) and the arms' wide configs: (grid columns, out_subblock_w, per_core_N), DERIVED
# on the 11 x 10 grid at 2048 rows (the prefill dense design note).
WIDE = {
    (2560, 4160): (11, 4, 12),  # GDN in-proj: full already
    (1536, 2560): (10, 4, 8),  # GDN / QSA out-proj: the 100 columns today's (11, 10) also runs
    (384, 2560): (10, 4, 8),  # GR up
    (160, 2560): (10, 4, 8),  # shared down
    (2560, 3072): (11, 3, 9),  # QSA query-gate: 80 -> 110 cores
    (2560, 512): (8, 2, 2),  # [k | v]: 2 x 20 -> 80 cores
    (2560, 384): (6, 2, 2),  # GR down+inject: 30 -> 60 cores
    (2560, 320): (5, 2, 2),  # [gate | up]: 2 x 20 -> 50 cores
}
BF8_TILE, BF16_TILE = 1088, 2048


class _Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def config_class(monkeypatch):
    monkeypatch.setattr(pd.ttnn, "MatmulMultiCoreReuseMultiCastProgramConfig", _Config)
    return _Config


def _mesh(arch="ARCH"):
    grid = SimpleNamespace(x=11, y=10)
    return SimpleNamespace(compute_with_storage_grid_size=lambda: grid, arch=lambda: arch)


# --------------------------------------------------------------------------- the switches


def test_switch_names_defaults_and_refusals(expect_error) -> None:
    assert pd.SWITCHES == (
        "QWEN38_PREFILL_DENSE_DTYPE",
        "QWEN38_PREFILL_DENSE_FIDELITY",
        "QWEN38_PREFILL_DENSE_GRID",
        "QWEN38_PREFILL_DENSE_FP32_ACC",
    )
    assert pd.DEFAULTS == {DTYPE: "bf16", FIDELITY: "hifi4", GRID: "wide", FP32: "1"}
    assert pd.CHOICES == {
        DTYPE: ("bf16", "bf8"),
        FIDELITY: ("hifi4", "hifi2", "lofi"),
        GRID: ("today", "wide"),
        FP32: ("1", "0"),
    }
    default = pd.Qwen38PrefillDensePolicy.from_environ({})
    assert default == pd.Qwen38PrefillDensePolicy() == pd.Qwen38PrefillDensePolicy(grid="wide") and default.is_default
    # the default: the wide grid (the fused siblings resident) on today's arithmetic (the modules' own compute config)
    assert default.resident_weights and default.fused_siblings and default.module_compute_config
    assert (default.dtype, default.fidelity, default.grid, default.fp32_acc) == ("bf16", "hifi4", "wide", True)
    assert "default" in default.describe() and "grid=wide" in default.describe()
    today = pd.Qwen38PrefillDensePolicy.from_environ({GRID: "today"})
    assert today == pd.Qwen38PrefillDensePolicy(grid="today") and not today.is_default
    assert not today.resident_weights and not today.fused_siblings and today.module_compute_config
    assert "default" not in today.describe()
    arm3 = pd.Qwen38PrefillDensePolicy.from_environ({DTYPE: "bf8", FIDELITY: "lofi", GRID: "wide"})
    assert (arm3.dtype, arm3.fidelity, arm3.grid, arm3.fp32_acc) == ("bf8", "lofi", "wide", True)
    assert arm3.resident_weights and arm3.fused_siblings and not arm3.module_compute_config and not arm3.is_default
    arm4 = pd.Qwen38PrefillDensePolicy.from_environ({DTYPE: "bf8", FIDELITY: "lofi", GRID: "wide", FP32: "0"})
    assert arm4.fp32_acc is False and arm4.math_fidelity == ttnn.MathFidelity.LoFi
    assert pd.Qwen38PrefillDensePolicy.from_environ({DTYPE: " BF8 ", FIDELITY: "HiFi2"}).fidelity == "hifi2"
    assert pd.Qwen38PrefillDensePolicy.from_environ({DTYPE: "", GRID: "  "}) == default  # empty = unset
    for variable, bad in ((DTYPE, "bf4"), (FIDELITY, "hifi3"), (GRID, "wider"), (FP32, "yes"), (FP32, "true")):
        with expect_error(ValueError, match=variable):  # allow-pytest.raises: pure contract test
            pd.Qwen38PrefillDensePolicy.from_environ({variable: bad})
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        pd.Qwen38PrefillDensePolicy(dtype="fp8")
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        pd.Qwen38PrefillDensePolicy(fp32_acc=1)
    # bf16 with the wide grid (the default) holds the fused siblings but no bfloat8_b copy
    wide16 = pd.Qwen38PrefillDensePolicy(grid="wide")
    assert wide16.resident_weights and wide16.fused_siblings and wide16.dtype == "bf16"


def test_resolve_reads_the_environment_once_and_rejects_aliases(expect_error, monkeypatch) -> None:
    mesh = _mesh()
    for variable in pd.SWITCHES:
        monkeypatch.delenv(variable, raising=False)
    assert pd.Qwen38TTNNPrefillDense.resolve(None, mesh).policy.is_default
    monkeypatch.setenv(GRID, "today")
    resolved = pd.Qwen38TTNNPrefillDense.resolve(None, mesh)
    assert resolved.policy.grid == "today" and resolved.mesh_device is mesh
    given = pd.Qwen38TTNNPrefillDense(pd.Qwen38PrefillDensePolicy(), mesh)
    assert pd.Qwen38TTNNPrefillDense.resolve(given, mesh) is given
    with expect_error(TypeError):  # allow-pytest.raises: pure contract test
        pd.Qwen38TTNNPrefillDense.resolve("wide", mesh)
    with expect_error(TypeError):  # allow-pytest.raises: pure contract test
        pd.Qwen38TTNNPrefillDense("bf8", mesh)


# --------------------------------------------------------------------------- grid today: the earlier slab, bitwise


def test_today_policy_resolves_to_the_modules_objects_and_todays_config(config_class) -> None:
    mesh = _mesh(arch=None)
    dense = pd.Qwen38TTNNPrefillDense(pd.Qwen38PrefillDensePolicy(grid="today"), mesh)
    module_default = object()
    assert dense.compute_config(module_default) is module_default  # the same object, no config is built
    for k, n in list(WIDE) + [(2560, 160), (2560, 1), (2560, 512), (2560, 128)]:
        assert (
            dense.program_config(SLAB, k, n).kwargs
            == decode_matmul.prefill_matmul_program_config(mesh, SLAB, k, n).kwargs
        )
    assert dense.program_config(SLAB, 2560, 384) is dense.program_config(SLAB, 2560, 384)  # built once
    assert len(dense.weights) == 0 and dense.resident("qg") is None and dense.weights.names == ()
    policy = dense.policy
    assert all(policy.resident_plan(module) == () for module in pd.MODULES) and policy.plan_bytes_per_device() == 0


def test_default_policy_is_the_wide_grid_on_todays_arithmetic(expect_error, config_class) -> None:
    # The default (grid wide, bf16, HiFi4, fp32 accumulation): the modules' own compute config, the wide program
    # configs, the two fused siblings resident in bf16 and nothing else.  MEASURED on the 4-chip line (2026-09-25):
    # +110,100,480 B of DRAM per device over grid today, the device column identical at every scored position.
    mesh = _mesh(arch=None)
    dense = pd.Qwen38TTNNPrefillDense(pd.Qwen38PrefillDensePolicy(), mesh)
    module_default = object()
    assert dense.policy.grid == "wide" and dense.compute_config(module_default) is module_default
    wide = pd.Qwen38TTNNPrefillDense(pd.Qwen38PrefillDensePolicy(grid="wide"), mesh)
    for k, n in list(WIDE) + [(2560, 160), (2560, 1), (2560, 512), (2560, 128)]:
        assert dense.program_config(SLAB, k, n).kwargs == wide.program_config(SLAB, k, n).kwargs
    assert dense.program_config(SLAB, 2560, 3072).kwargs["compute_with_storage_grid_size"] == (11, 10)
    policy = dense.policy
    assert {module: tuple(spec.name for spec in policy.resident_plan(module)) for module in pd.MODULES} == {
        "gr": (),
        "gdn": (),
        "qsa": ("kv",),
        "moe": ("shared_gate_up",),
    }
    assert all(spec.dtype == "bf16" for module in pd.MODULES for spec in policy.resident_plan(module))
    assert policy.plan_bytes_per_device() == 110_100_480
    # The siblings take the module's dense weight format (QWEN38_DENSE_WEIGHT_DTYPE, bf8 by default since the stack
    # landed after the measurement): the fused linear reads the same tiles the separate K / V and gate / up read.
    assert [spec.dtype for spec in policy.resident_plan("qsa", "bf8")] == ["bf8"]
    assert [spec.dtype for spec in policy.resident_plan("moe", "bf4")] == ["bf4"]
    assert policy.plan_bytes_per_device({"qsa": "bf8", "moe": "bf8"}) == 12 * 1280 * 1088 + 48 * 800 * 1088
    assert policy.plan_bytes_per_device({"qsa": "bf8", "moe": "bf8"}) == 58_490_880
    assert policy.plan_bytes_per_device({"gdn": "bf8", "gr": "bf4"}) == 110_100_480  # no siblings there
    # Under dtype bf8 the siblings are bf8 like the other copies, whatever the module's format.
    bf8_wide = pd.Qwen38PrefillDensePolicy(dtype="bf8")
    assert {spec.dtype for spec in bf8_wide.resident_plan("qsa", "bf4")} == {"bf8"}
    assert bf8_wide.plan_bytes_per_device({"qsa": "bf4", "moe": "bf16"}) == bf8_wide.plan_bytes_per_device()
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        policy.resident_plan("qsa", "fp8")
    assert pd.weight_dtype_name(ttnn.bfloat8_b) == "bf8" and pd.weight_dtype_name(ttnn.bfloat4_b) == "bf4"
    assert pd.weight_dtype_name(ttnn.bfloat16) == "bf16"
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        pd.weight_dtype_name(ttnn.float32)
    assert len(dense.weights) == 0  # attached by the builder after the resident experts, behind the DRAM admission


def test_prefill_linear_default_is_decode_matmuls_helper_and_the_resident_form_copies_nothing(monkeypatch) -> None:
    # decode_matmul is untouched: its helper keeps today's signature and knows nothing of resident weights (checked
    # before the helper is patched below).
    assert tuple(inspect.signature(decode_matmul.prefill_linear).parameters) == (
        "activation",
        "weight",
        "program_config",
        "compute_kernel_config",
        "dtype",
    )
    assert "resident_weight" not in inspect.getsource(decode_matmul)
    assert "prefill_dense" not in inspect.getsource(decode_matmul)
    calls = []
    monkeypatch.setattr(
        pd.decode_matmul, "prefill_linear", lambda *args, **kwargs: calls.append(("decode", args, kwargs)) or "copy"
    )
    monkeypatch.setattr(pd.ttnn, "linear", lambda *args, **kwargs: calls.append(("linear", args, kwargs)) or "resident")
    monkeypatch.setattr(
        pd.ttnn, "to_memory_config", lambda *args, **kwargs: pytest.fail("no copy on the resident path")
    )
    monkeypatch.setattr(
        pd.ttnn, "deallocate", lambda *args, **kwargs: pytest.fail("nothing to release on the resident path")
    )
    assert pd.prefill_linear("act", "w", "cfg", compute_kernel_config="cc") == "copy"
    assert calls == [("decode", ("act", "w", "cfg"), {"compute_kernel_config": "cc", "dtype": None})]
    calls.clear()
    assert (
        pd.prefill_linear("act", "w", "cfg", compute_kernel_config="cc", dtype="f32", resident_weight="rw")
        == "resident"
    )
    assert calls == [
        (
            "linear",
            ("act", "rw"),
            {
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "program_config": "cfg",
                "compute_kernel_config": "cc",
                "dtype": "f32",
            },
        )
    ]


# --------------------------------------------------------------------------- the arms


def test_arm_compute_configs(monkeypatch) -> None:
    built = []

    def init(arch, **kwargs):
        built.append((arch, kwargs))
        return f"config-{len(built)}"

    monkeypatch.setattr(pd.ttnn, "init_device_compute_kernel_config", init)
    mesh = _mesh(arch="BH")
    module_default, other_default = object(), object()
    hifi2 = pd.Qwen38TTNNPrefillDense(pd.Qwen38PrefillDensePolicy(fidelity="hifi2"), mesh)
    assert hifi2.compute_config(module_default) == "config-1" and hifi2.compute_config(module_default) == "config-1"
    assert hifi2.compute_config(other_default) == "config-2"  # one per module default, built once
    assert built[0] == (
        "BH",
        {
            "math_fidelity": ttnn.MathFidelity.HiFi2,
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": False,
        },
    )
    arm4 = pd.Qwen38TTNNPrefillDense(
        pd.Qwen38PrefillDensePolicy(dtype="bf8", fidelity="lofi", grid="wide", fp32_acc=False), mesh
    )
    arm4.compute_config(module_default)
    assert built[-1][1] == {
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "math_approx_mode": False,
        "fp32_dest_acc_en": False,
        "packer_l1_acc": False,
    }
    # fp32 accumulation off with HiFi4 is still a new config (the flag moved), HiFi4 + fp32 is the module's object
    assert pd.Qwen38PrefillDensePolicy(fp32_acc=False).module_compute_config is False
    assert pd.Qwen38PrefillDensePolicy(dtype="bf8", grid="wide").module_compute_config is True


def test_wide_grid_runs_the_most_columns_with_two_tile_subblocks(expect_error, config_class) -> None:
    mesh = _mesh()
    dense = pd.Qwen38TTNNPrefillDense(pd.Qwen38PrefillDensePolicy(grid="wide"), mesh)
    for (k, n), (cols, subblock, per_core_n) in WIDE.items():
        config = dense.program_config(SLAB, k, n).kwargs
        n_tiles, k_tiles = -(-n // 32), k // 32
        assert config["compute_with_storage_grid_size"] == (cols, 10), (k, n, config)
        assert (config["out_subblock_w"], config["per_core_N"]) == (subblock, per_core_n), (k, n, config)
        assert config["out_subblock_h"] == 1 and config["per_core_M"] == 7 and config["transpose_mcast"] is False
        assert config["in0_block_w"] == next(d for d in range(8, 0, -1) if k_tiles % d == 0)
        assert -(-n_tiles // per_core_n) == cols  # the grid width is what the program factory runs
        assert per_core_n % subblock == 0 and subblock >= 2
        assert config["fused_activation"] is None and config["fuse_batch"] is False
    # A one-tile N (the scalar gate) has no two-tile subblock: today's config.
    assert (
        dense.program_config(SLAB, 2560, 1).kwargs
        == decode_matmul.prefill_matmul_program_config(mesh, SLAB, 2560, 1).kwargs
    )
    # The narrow shapes gain cores against today's rule; the full-width ones keep their 110 / 100 columns.
    today = {shape: decode_matmul.prefill_matmul_program_config(mesh, SLAB, *shape).kwargs for shape in WIDE}
    for shape, (cols, _sub, per_core_n) in WIDE.items():
        n_tiles = -(-shape[1] // 32)
        running_today = -(-n_tiles // today[shape]["per_core_N"])
        assert cols >= running_today, shape
    assert {shape for shape in WIDE if WIDE[shape][0] > -(-(-(-shape[1] // 32)) // today[shape]["per_core_N"])} == {
        (2560, 3072),
        (2560, 512),
        (2560, 384),
        (2560, 320),
    }
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        pd.wide_prefill_matmul_program_config(mesh, 2000, 2560, 512)


# --------------------------------------------------------------------------- the resident weights


def test_resident_plans_hold_bf8_copies_only_under_bf8_and_fused_siblings_only_under_wide(expect_error) -> None:
    names = lambda policy, module: tuple(spec.name for spec in policy.resident_plan(module))  # noqa: E731
    bf8 = pd.Qwen38PrefillDensePolicy(dtype="bf8", grid="today")  # the copies alone: the grid default is wide
    assert names(bf8, "gr") == ("down_inject", "up") and names(bf8, "gdn") == ("qkvzab", "out")
    assert names(bf8, "qsa") == ("qg", "out", "k", "v")
    assert names(bf8, "moe") == ("shared_gate", "shared_up", "shared_down", "shared_scalar_gate")
    assert all(
        spec.dtype == "bf8" and spec.ttnn_dtype == ttnn.bfloat8_b for m in pd.MODULES for spec in bf8.resident_plan(m)
    )
    wide16 = pd.Qwen38PrefillDensePolicy(grid="wide")
    assert names(wide16, "gr") == () and names(wide16, "gdn") == ()
    assert names(wide16, "qsa") == ("kv",) and names(wide16, "moe") == ("shared_gate_up",)
    assert all(
        spec.dtype == "bf16" and spec.ttnn_dtype == ttnn.bfloat16
        for m in pd.MODULES
        for spec in wide16.resident_plan(m)
    )
    wide8 = pd.Qwen38PrefillDensePolicy(dtype="bf8", grid="wide")
    assert names(wide8, "qsa") == ("qg", "out", "kv")
    assert names(wide8, "moe") == ("shared_gate_up", "shared_down", "shared_scalar_gate")
    # The exact set is in no plan under any policy.
    assert pd.EXACT_LINEARS == {"moe": ("router",), "qsa": ("index_q", "index_k")}
    for policy in (
        bf8,
        wide16,
        wide8,
        pd.Qwen38PrefillDensePolicy(dtype="bf8", fidelity="lofi", grid="wide", fp32_acc=False),
    ):
        for module, exact in pd.EXACT_LINEARS.items():
            assert not set(exact) & set(names(policy, module)), (policy, module)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        bf8.resident_plan("ple")
    # Shapes are the modules' constants; the fused siblings cut on whole tiles into the widths the modules validate.
    assert (
        pd.HIDDEN
        == gr_module.FLAT_LOCAL_WIDTH
        == gdn_module.HIDDEN_SIZE
        == qsa_module.HIDDEN_SIZE
        == moe_module.HIDDEN_SIZE
    )
    assert (
        pd.GR_PARTIAL_WIDTH == gr_module.PARTIAL_WIDTH
        and pd.GDN_PROJECTION_WIDTH == gdn_module.PROJECTION_WIDTH_PER_DEVICE
    )
    assert (
        pd.GDN_VALUE_WIDTH == gdn_module.VALUE_WIDTH_PER_DEVICE
        and pd.QSA_LOCAL_QUERY_WIDTH == qsa_module.LOCAL_QUERY_WIDTH
    )
    assert pd.QSA_HEAD_DIM == qsa_module.HEAD_DIM
    assert pd.MOE_LOCAL_INTERMEDIATE == moe_module.INTERMEDIATE_SIZE // 4 == shared_expert.LOCAL_INTERMEDIATE
    assert (
        2 * pd.MOE_LOCAL_INTERMEDIATE == shared_expert.SCALAR_COLUMN
    )  # [gate | up] = the decode concat's first 320 columns
    assert pd.KV_COLUMNS == (("k", 0, 256), ("v", 256, 256)) and pd.SHARED_COLUMNS == (
        ("shared_gate", 0, 160),
        ("shared_up", 160, 160),
    )
    for spec in (wide8.resident_plan("qsa")[-1], wide8.resident_plan("moe")[0]):
        assert sum(width for _name, _start, width in spec.columns) == spec.n
        assert all(start % 32 == 0 and width % 32 == 0 for _name, start, width in spec.columns)
    # The scalar gate stays its own [2560, 1] replicated linear: its output feeds a [rows, 1] sigmoid broadcast.
    scalar = wide8.resident_plan("moe")[-1]
    assert (scalar.name, scalar.k, scalar.n, scalar.shard_dim) == ("shared_scalar_gate", 2560, 1, None)


def test_resident_bytes_and_cache_names(expect_error) -> None:
    spec = pd.ResidentSpec("qkvzab", 2560, 4160, 3, "bf8")
    assert (
        spec.tiles == 80 * 130
        and spec.bytes == 80 * 130 * BF8_TILE
        and spec.cache_name == "qkvzab_prefill_interleaved.bf8b"
    )
    assert pd.ResidentSpec("kv", 2560, 512, 3, "bf16", pd.KV_COLUMNS).cache_name == "kv_prefill_interleaved.bf16"
    bf4 = pd.ResidentSpec("kv", 2560, 512, 3, "bf4", pd.KV_COLUMNS)
    assert (bf4.cache_name, bf4.ttnn_dtype, bf4.bytes) == ("kv_prefill_interleaved.bf4b", ttnn.bfloat4_b, 1280 * 576)
    assert pd.ResidentSpec("kv", 2560, 512, 3, "bf8", pd.KV_COLUMNS).ttnn_dtype == ttnn.bfloat8_b
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        pd.ResidentSpec("kv", 2560, 512, 3, "fp8")
    assert pd.TILE_BYTES == {"bf16": BF16_TILE, "bf8": BF8_TILE, "bf4": 576}
    assert pd.LAYER_COUNTS == {"gr": 96, "gdn": 36, "qsa": 12, "moe": 48}
    bf8 = pd.Qwen38PrefillDensePolicy(dtype="bf8", grid="today")
    gdn = 36 * (80 * 130 + 48 * 80) * BF8_TILE
    qsa = 12 * (80 * 96 + 48 * 80 + 2 * 80 * 8) * BF8_TILE
    gr = 96 * (80 * 12 + 12 * 80) * BF8_TILE
    moe = 48 * (80 * 5 + 80 * 5 + 5 * 80 + 80 * 1) * BF8_TILE
    assert bf8.plan_bytes_per_device() == gdn + qsa + gr + moe == 992_256_000  # 946 MiB per device (DERIVED)
    wide16 = pd.Qwen38PrefillDensePolicy(grid="wide")
    # 110,100,480 B: the MEASURED +13,762,560 B per bank (x 8) of the wide grid on the 4-chip line
    assert wide16.plan_bytes_per_device() == 12 * 80 * 16 * BF16_TILE + 48 * 80 * 10 * BF16_TILE == 110_100_480
    assert pd.Qwen38PrefillDensePolicy(dtype="bf8", grid="wide").plan_bytes_per_device() == bf8.plan_bytes_per_device()
    # A weights table sums its specs and refuses a second attach of the same name.
    table = pd.Qwen38TTNNPrefillDenseWeights({"up": ("tensor", pd.ResidentSpec("up", 384, 2560, 3, "bf8"))})
    assert table.bytes == 12 * 80 * BF8_TILE and "up" in table and table.tensor("up") == "tensor" and len(table) == 1
    dense = pd.Qwen38TTNNPrefillDense(bf8, _mesh())
    dense.attach(table)
    assert dense.resident("up") == "tensor" and dense.resident("down_inject") is None
    with expect_error(ValueError, match="attached twice"):  # allow-pytest.raises: pure contract test
        dense.attach(table)


def test_fused_linear_is_one_linear_and_the_callers_cut_it_with_literal_bounds(expect_error, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(pd.ttnn, "linear", lambda *args, **kwargs: calls.append(("linear", args, kwargs)) or "fused")
    dense = pd.Qwen38TTNNPrefillDense(pd.Qwen38PrefillDensePolicy(grid="wide"), _mesh())
    dense.attach(
        pd.Qwen38TTNNPrefillDenseWeights({"kv": ("W", pd.ResidentSpec("kv", 2560, 512, 3, "bf16", pd.KV_COLUMNS))})
    )
    dense._program_configs[(SLAB, 2560, 512)] = "wide-config"
    assert dense.fused_linear("hidden", "kv", SLAB, compute_kernel_config="module-config") == "fused"
    assert calls == [
        (
            "linear",
            ("hidden", "W"),
            {
                "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "program_config": "wide-config",
                "compute_kernel_config": "module-config",
            },
        )
    ]
    dense.attach(pd.Qwen38TTNNPrefillDenseWeights({"qg": ("Q", pd.ResidentSpec("qg", 2560, 3072, 3, "bf8"))}))
    with expect_error(ValueError, match="not a fused sibling"):  # allow-pytest.raises: pure contract test
        dense.fused_linear("hidden", "qg", SLAB, compute_kernel_config="c")
    # The slices take the reference's distribution and coordinates, sharded on the column axis; no bytes move.
    monkeypatch.setattr(pd.ttnn, "TensorTopology", lambda shape, placements, coords: (shape, tuple(placements), coords))
    monkeypatch.setattr(pd.ttnn, "PlacementReplicate", lambda: "replicate")
    monkeypatch.setattr(pd.ttnn, "PlacementShard", lambda dim: ("shard", dim))
    topology = SimpleNamespace(distribution_shape=lambda: (1, 4), mesh_coords=lambda: "coords")
    tagged = []
    parts = [SimpleNamespace(update_tensor_topology=lambda t, i=i: tagged.append((i, t))) for i in range(2)]
    dense.retag_sharded(*parts, reference=SimpleNamespace(tensor_topology=lambda: topology), shard_dim=3)
    assert tagged == [(i, ((1, 4), ("replicate", ("shard", 3)), "coords")) for i in range(2)]
    # The callers cut the fused output with literal bounds: the captured decode body reaches the shared expert's slab
    # and admits no host-integer shape op (test_ttnn_generic_body_trace_contract_static); the QSA cut is a helper the
    # projection's pinned op walk does not enter.
    shared = _flat(inspect.getsource(moe_module.Qwen38TTNNMoE._shared_partial_slab))
    assert "gate=ttnn.slice(fused,(0,0,0,0),(1,1,self.rows,local_intermediate),memory_config=dram)" in shared
    assert (
        "up=ttnn.slice(fused,(0,0,0,local_intermediate),(1,1,self.rows,2*local_intermediate),memory_config=dram)"
        in shared
    )
    assert "_deallocate(fused)dense.retag_sharded(gate,up,reference=full_hidden,shard_dim=3)" in shared
    kv = _flat(inspect.getsource(qsa_module.Qwen38TTNNQSA._fused_kv_rows))
    assert "k=ttnn.slice(kv,(0,0,0,0),(1,1,rows,HEAD_DIM),memory_config=ttnn.DRAM_MEMORY_CONFIG)" in kv
    assert "v=ttnn.slice(kv,(0,0,0,HEAD_DIM),(1,1,rows,2*HEAD_DIM),memory_config=ttnn.DRAM_MEMORY_CONFIG)" in kv
    assert "_deallocate(kv)self.prefill_dense.retag_sharded(k,v,reference=full_hidden,shard_dim=3)" in kv
    assert qsa_module.HEAD_DIM == pd.QSA_HEAD_DIM == pd.KV_COLUMNS[1][1] and pd.SHARED_COLUMNS[1][1] == 160


def test_dram_admission_models_the_slab_working_set_with_the_measured_band(expect_error, monkeypatch) -> None:
    # The term (prefill_dense.SLAB_WORKING_SET_BYTES) is the band's upper end, the only post-build free space the
    # slab is known to fit, less the context state at 32k and the margin the admission adds on its own.
    context_32k = builder_module.Qwen38ResidentContext().context_state_bytes_per_device
    assert builder_module.Qwen38ResidentContext().allocated_context == 32_768 and context_32k == 436_797_440
    assert pd.DRAM_MARGIN_BYTES == 512 << 20
    assert pd.SLAB_WORKING_SET_BYTES == 2_010_176_000 - context_32k - pd.DRAM_MARGIN_BYTES == 1_036_507_648
    views = {}
    monkeypatch.setattr(pd.ttnn, "get_memory_view", lambda mesh, buffer_type: views[buffer_type])

    def free_per_bank(value: int) -> None:
        views[ttnn.BufferType.DRAM] = SimpleNamespace(total_bytes_free_per_bank=value, num_banks=8)

    wide = pd.Qwen38PrefillDensePolicy().plan_bytes_per_device()  # the default: 110,100,480
    bf8_wide = pd.Qwen38PrefillDensePolicy(dtype="bf8", grid="wide").plan_bytes_per_device()  # 992,256,000
    # MEASURED on the 4-chip line at 32k under the device profiler's 64,000-program reservation: 265,034,560 B free
    # per bank before any plan; the wide plan left 251,272,000 (the slab ran), the bf8 + wide plan 141,002,560 (OOM).
    free_per_bank(265_034_560)
    numbers = pd.admit_prefill_dense_dram("mesh", plan_bytes=wide, context_state_bytes=context_32k)
    assert numbers["free_bytes_per_device"] == 265_034_560 * 8 and numbers["plan_bytes_per_device"] == wide
    assert numbers["free_bytes_per_device"] - wide == 2_010_176_000  # admitted at the band's upper end exactly
    assert numbers["slab_working_set_bytes_per_device"] == pd.SLAB_WORKING_SET_BYTES
    assert numbers["needed_bytes_per_device"] == wide + context_32k + pd.SLAB_WORKING_SET_BYTES + (512 << 20)
    assert numbers["margin_bytes"] == 512 << 20 and numbers["num_banks"] == 8
    with pytest.raises(ValueError, match="refused") as refusal:  # allow-pytest.raises: pure contract test
        pd.admit_prefill_dense_dram("mesh", plan_bytes=bf8_wide, context_state_bytes=context_32k)
    message = str(refusal.value)
    for number in ("946 MiB", "417 MiB", "988 MiB", "512 MiB", "2863 MiB", "2022 MiB", "141.0 MB per bank"):
        assert number in message, (number, message)
    # The same plan under a 32,000-program reservation (153,600,000 B per bank more) ran: admitted.
    free_per_bank(265_034_560 + 153_600_000)
    admitted = pd.admit_prefill_dense_dram("mesh", plan_bytes=bf8_wide, context_state_bytes=context_32k)
    assert admitted["free_bytes_per_device"] - bf8_wide == 294_602_560 * 8
    # The served process at 32k (no reservation; 443,402,560 B per bank after the bf8 + wide plan) ran: admitted.
    free_per_bank(443_402_560 + bf8_wide // 8)
    pd.admit_prefill_dense_dram("mesh", plan_bytes=bf8_wide, context_state_bytes=context_32k)
    # The slab8k derivation's 145 MB (one stage's transient peak) would have admitted the run that OOM'd.
    free_per_bank(265_034_560)
    pd.admit_prefill_dense_dram(
        "mesh", plan_bytes=bf8_wide, context_state_bytes=context_32k, slab_working_set=145_000_000
    )
    # Below the OOM point every plan with the bf8 copies is refused, and the wide default at its own edge too.
    free_per_bank(265_034_560 - 8)
    with expect_error(ValueError, match="refused"):  # allow-pytest.raises: pure contract test
        pd.admit_prefill_dense_dram("mesh", plan_bytes=wide, context_state_bytes=context_32k)


def test_weight_table_builder_allocates_nothing_without_a_plan() -> None:
    source = inspect.getsource(pd.build_prefill_dense_weights)
    assert source.index("if not plan:\n        return Qwen38TTNNPrefillDenseWeights()") < source.index("_host_blocks(")
    assert source.count("_upload(") == 1 and inspect.getsource(pd).count("ttnn.as_tensor(") == 1
    upload = inspect.getsource(pd._upload)
    assert "dtype=spec.ttnn_dtype" in upload and "memory_config=ttnn.DRAM_MEMORY_CONFIG" in upload
    assert "cache_file_name=cache_dir / spec.cache_name" in upload  # a new tensorbin name per layout and dtype
    hosts = inspect.getsource(pd._host_blocks)
    # the same host sources and packers as the decode loaders, which are not edited
    for packer in (
        "gr._prepare_host_weights(source)",
        "gdn.pack_projection_columns(shards)",
        "qsa._expanded_pair_kv(",
        "Qwen38MoEWeights(checkpoint, placement, layer_index=layer_index)",
    ):
        assert packer in hosts, packer
    for loader in (
        gr_module.Qwen38TTNNGatedResidualWeights.from_checkpoint,
        gdn_module.Qwen38TTNNGDNWeights.from_checkpoint,
        moe_module.Qwen38TTNNMoEWeights.from_checkpoint,
        qsa_module.Qwen38TTNNQSAWeights.from_checkpoint,
    ):
        assert "prefill" not in inspect.getsource(loader), loader.__qualname__


# --------------------------------------------------------------------------- the exact set and the call sites


def test_exact_set_carries_no_policy() -> None:
    router = inspect.getsource(moe_module.Qwen38TTNNMoE._slab_router_logits)
    assert "self._slab_program_config(HIDDEN_SIZE, ROUTED_EXPERTS, exact=True)" in router
    assert "compute_kernel_config=self.compute_config," in router
    assert "resident_weight" not in router and "prefill_dense" not in router
    index = inspect.getsource(qsa_module.Qwen38TTNNQSA._index_projection_rows)
    assert "self._linear_rows(full_hidden, self.weights.index_q, self.index_program_config, hidden_tiles)" in index
    assert "self._linear_rows(full_hidden, self.weights.index_k, self.index_program_config, hidden_tiles)" in index
    assert "dense=" not in index and "prefill_dense" not in index
    linear_rows = inspect.getsource(qsa_module.Qwen38TTNNQSA._linear_rows)
    exact_branch = linear_rows.split("if dense is None:", 1)[1].split("return prefill_linear(", 2)[1]
    assert "exact=True" in exact_branch and "compute_kernel_config=self.projection_compute_config," in exact_branch
    assert "resident_weight" not in exact_branch
    policy_branch = linear_rows.rsplit("return prefill_linear(", 1)[1]
    assert "resident_weight=self.prefill_dense.resident(dense)" in policy_branch
    assert "compute_kernel_config=self.prefill_dense.compute_config(self.projection_compute_config)" in policy_branch
    assert inspect.signature(qsa_module.Qwen38TTNNQSA._linear_rows).parameters["dense"].default is None
    for module in (moe_module.Qwen38TTNNMoE, qsa_module.Qwen38TTNNQSA):
        config = inspect.getsource(module._slab_program_config)
        assert 'if exact or self.prefill_dense.policy.grid == "today":' in config
        assert config.index("prefill_matmul_program_config(") < config.index("self.prefill_dense.program_config(")
        assert inspect.signature(module._slab_program_config).parameters["exact"].default is False
    for module in (gr_module.Qwen38TTNNGatedResidual, gdn_module.Qwen38TTNNGDN):
        config = inspect.getsource(module._slab_program_config)
        assert 'if self.prefill_dense.policy.grid == "today":' in config and "prefill_matmul_program_config(" in config


def test_every_other_slab_linear_takes_the_policy() -> None:
    expected = {
        gr_module: (2, ("down_inject", "up")),
        gdn_module: (2, ("qkvzab", "out")),
        qsa_module: (2, ("out",)),  # plus _linear_rows' ``dense`` name for qg / k / v
        moe_module: (4, ("shared_gate", "shared_up", "shared_down", "shared_scalar_gate")),
    }
    for module, (count, names) in expected.items():
        source = inspect.getsource(module)
        assert source.count("resident_weight=") == count, module.__name__
        for name in names:
            assert f'.resident("{name}")' in source, (module.__name__, name)
        assert module.Qwen38TTNNPrefillDense is pd.Qwen38TTNNPrefillDense
        assert module.prefill_linear is pd.prefill_linear
        assert "    prefill_linear,\n" not in source.split("decode_matmul import (", 1)[1].split(")", 1)[0]
        assert (
            inspect.signature(module.__dict__[_module_class(module)].__init__).parameters["prefill_dense"].default
            is None
        )
    projection = inspect.getsource(qsa_module.Qwen38TTNNQSA._main_projection_rows)
    for name in ("qg", "k", "v"):
        assert f'dense="{name}"' in projection
    assert 'self.prefill_dense.resident("kv") is not None' in projection
    assert "k,v=self._fused_kv_rows(full_hidden,rows)" in _flat(projection)
    assert 'fused_linear(full_hidden,"kv",rows,' in _flat(inspect.getsource(qsa_module.Qwen38TTNNQSA._fused_kv_rows))
    shared = inspect.getsource(moe_module.Qwen38TTNNMoE._shared_partial_slab)
    assert 'dense.resident("shared_gate_up") is not None' in shared
    assert 'fused_linear(full_hidden,"shared_gate_up",self.rows,' in _flat(shared)
    # The slab MoE instance shares the layer's object; the builder hands every module one and attaches the weights
    # after the resident experts, behind the DRAM admission.
    allocate = inspect.getsource(layer_module.Qwen38TTNNDecoderLayer.allocate_chunk_state)
    assert "prefill_dense=self.mlp.prefill_dense," in allocate
    builder = inspect.getsource(builder_module.Qwen38TTNNBuilder)
    assert builder.count("prefill_dense=self._prefill_dense(),") == 4  # GR, MoE, GDN, QSA
    assert "self.prefill_dense_policy = Qwen38PrefillDensePolicy.from_environ()" in inspect.getsource(
        builder_module.Qwen38TTNNBuilder.__init__
    )
    components = inspect.getsource(builder_module.Qwen38TTNNBuilder._build_target_components)
    assert components.index("self.expert_streamer.preload_backbone()") < components.index(
        "self._attach_prefill_dense_weights(layers)"
    )
    attach = inspect.getsource(builder_module.Qwen38TTNNBuilder._attach_prefill_dense_weights)
    assert "if self.prefill_slab_rows is None or not policy.resident_weights:\n            return" in attach
    assert attach.index("admit_prefill_dense_dram(") < attach.index("build_prefill_dense_weights(")
    assert "context_state_bytes=context.context_state_bytes_per_device" in attach
    assert "slab_working_set=slab_working_set_bytes(self.prefill_slab_rows)" in attach
    # the siblings in their module's dense weight format, for the admission's bytes and the uploads alike
    assert '"qsa": weight_dtype_name(self.dense_weight_plan.dtype("qsa"))' in attach
    assert '"moe": weight_dtype_name(self.dense_weight_plan.dtype("shared_expert"))' in attach
    assert "plan_bytes=policy.plan_bytes_per_device(weight_dtypes)" in attach
    assert 'weight_dtype=weight_dtypes.get(kind, "bf16")' in attach
    assert builder_module.Qwen38TTNNBuilder.prefill_dense_policy.is_default


def test_builder_builds_the_plan_and_admits_only_for_a_build_that_runs_a_slab(expect_error, monkeypatch) -> None:
    # No slab (the class default, every build that never calls enable_prefill_slab: the chunked 256k server among
    # them): nothing is admitted, nothing is built, no device is touched, whatever the switches say.
    builder_class = builder_module.Qwen38TTNNBuilder
    assert builder_class.prefill_slab_rows is None and builder_class.prefill_dense_admission is None
    monkeypatch.setattr(builder_module, "admit_prefill_dense_dram", lambda *a, **k: pytest.fail("admitted"))
    monkeypatch.setattr(builder_module, "build_prefill_dense_weights", lambda *a, **k: pytest.fail("built"))
    no_slab = object.__new__(builder_class)
    for policy in (pd.Qwen38PrefillDensePolicy(), pd.Qwen38PrefillDensePolicy(dtype="bf8", fidelity="lofi")):
        no_slab.prefill_dense_policy = policy
        assert policy.resident_weights
        no_slab._attach_prefill_dense_weights(())
    assert no_slab.prefill_dense_admission is None
    # enable_prefill_slab: a slab row count, before the target is built.
    builder = object.__new__(builder_class)
    builder._target_components, builder._built_target = None, False
    for bad in (0, 100, 2048.0, 8192, None):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            builder.enable_prefill_slab(bad)
    builder.enable_prefill_slab(2048)
    assert builder.prefill_slab_rows == 2048
    builder._built_target = True
    with expect_error(RuntimeError, match="before the target is built"):  # allow-pytest.raises: contract
        builder.enable_prefill_slab(2048)
    # With a slab the admission runs before any plan is built, with the slab's working-set term.
    admissions = []
    monkeypatch.setattr(
        builder_module, "admit_prefill_dense_dram", lambda mesh, **k: admissions.append((mesh, k)) or {"ok": 1}
    )
    slab = object.__new__(builder_class)
    slab.prefill_dense_policy, slab.prefill_slab_rows = pd.Qwen38PrefillDensePolicy(), 2048
    slab.mesh_device, slab.qsa_cache_capacity = "mesh", 32_768
    slab.dense_weight_plan = decode_matmul.default_dense_weight_plan({})  # the landed default: bf8 dense weights
    assert slab.dense_weight_plan.dtype("qsa") == slab.dense_weight_plan.dtype("shared_expert") == ttnn.bfloat8_b
    slab._attach_prefill_dense_weights(())
    assert slab.prefill_dense_admission == {"ok": 1} and len(admissions) == 1
    assert admissions[0] == (
        "mesh",
        {
            "plan_bytes": 58_490_880,  # the two siblings as bf8, the format of the k / v and gate / up they replace
            "context_state_bytes": builder_module.Qwen38ResidentContext().context_state_bytes_per_device,
            "slab_working_set": pd.SLAB_WORKING_SET_BYTES,
        },
    )
    # The term: the measured 2048-row band up to 2048 rows, rows-linear above it.
    assert pd.MEASURED_SLAB_ROWS == 2048
    assert pd.slab_working_set_bytes(2048) == pd.slab_working_set_bytes(256) == pd.SLAB_WORKING_SET_BYTES
    assert pd.slab_working_set_bytes(4096) == 2 * pd.SLAB_WORKING_SET_BYTES
    assert pd.slab_working_set_bytes(3072) == pd.SLAB_WORKING_SET_BYTES * 3072 // 2048
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        pd.slab_working_set_bytes(100)
    # The chat chain tells the builder the slab before it builds the target (the other slab-building tools are
    # pinned by the development-side companion test).
    chain = (MODEL_DIR / "tools" / "qwen38_chat_session.py").read_text()
    assert chain.count("builder.enable_prefill_slab(slab_rows)") == 1
    assert chain.index("builder.enable_prefill_slab(slab_rows)") < chain.index("built_target = builder.build_target()")


def _flat(text: str) -> str:
    return "".join(text.split())


def _module_class(module) -> str:
    return {
        gr_module: "Qwen38TTNNGatedResidual",
        gdn_module: "Qwen38TTNNGDN",
        qsa_module: "Qwen38TTNNQSA",
        moe_module: "Qwen38TTNNMoE",
    }[module]


# --------------------------------------------------------------------------- the note and the manifest


def test_design_note_reference_and_docs() -> None:
    # The design note is a development document: named without its path here and in the module (the export forbids
    # naming the development directories in public code); its presence and the gate tool's slab option are checked
    # by the development-side companion test.
    assert "the prefill dense design note" in pd.__doc__
    docs = (MODEL_DIR / "docs" / "PREFILL.md").read_text()
    assert all(switch in docs for switch in pd.SWITCHES)
    assert "| `QWEN38_PREFILL_DENSE_GRID` | `today`, `wide` | `wide` |" in docs  # the documented default
    assert "| `QWEN38_PREFILL_DENSE_DTYPE` | `bf16`, `bf8` | `bf16` |" in docs
