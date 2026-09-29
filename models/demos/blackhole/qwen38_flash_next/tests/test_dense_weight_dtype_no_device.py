# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The dense weight dtype switch without a device: ``QWEN38_DENSE_WEIGHT_DTYPE`` parsing and the per-module overrides,
the dtype -> tag / fidelity / tile-byte tables, the cache names of converted weights (bf16 names unchanged), the
two-reader qualification set and the builder's fallback, and that every module's converted-weight linears run the
weight format's compute config while the rest keep theirs."""

import inspect
import re
from pathlib import Path

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import builder as builder_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as final_mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp as mtp_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module

BF16, BF8, BF4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b
MODEL_DIR = Path(gdn_module.__file__).resolve().parents[1]


def test_the_switch_defaults_to_bf8_and_admits_the_three_formats(expect_error) -> None:
    assert dm.DENSE_DTYPE_ENV == "QWEN38_DENSE_WEIGHT_DTYPE" and dm.DEFAULT_DENSE_DTYPE_NAME == "bf8"
    plan = dm.default_dense_weight_plan({})
    assert plan.name == "bf8" and all(plan.dtype(m) == BF8 for m in dm.DENSE_MODULES)
    assert dm.default_dense_weight_plan({dm.DENSE_DTYPE_ENV: "bf16"}).name == "bf16"
    assert all(dm.default_dense_weight_plan({dm.DENSE_DTYPE_ENV: "bf16"}).dtype(m) == BF16 for m in dm.DENSE_MODULES)
    assert dm.default_dense_weight_plan({dm.DENSE_DTYPE_ENV: " BF4 "}).dtype("gdn") == BF4
    assert dm.default_dense_weight_plan({dm.DENSE_DTYPE_ENV: ""}).name == "bf8"
    for bad in ("bf12", "bfp8", "fp8", "1", "bf8,bf4"):
        with expect_error(ValueError, match=dm.DENSE_DTYPE_ENV):
            dm.default_dense_weight_plan({dm.DENSE_DTYPE_ENV: bad})


def test_per_module_overrides(expect_error) -> None:
    env = {dm.DENSE_DTYPE_ENV: "bf4", f"{dm.DENSE_DTYPE_ENV}_GDN": "bf8"}
    plan = dm.default_dense_weight_plan(env)
    assert plan.name == "mixed" and plan.dtype("gdn") == BF8 and plan.dtype("qsa") == BF4
    assert plan.describe()["modules"]["gdn"] == {"dtype": "bf8b", "math_fidelity": "HiFi2"}
    assert plan.describe()["modules"]["lm_head"] == {"dtype": "bf4b", "math_fidelity": "LoFi"}
    with expect_error(ValueError, match=f"{dm.DENSE_DTYPE_ENV}_QSA"):
        dm.default_dense_weight_plan({f"{dm.DENSE_DTYPE_ENV}_QSA": "int8"})
    assert dm.DENSE_MODULES == ("gdn", "qsa", "shared_expert", "lm_head", "final_mixer", "mtp")
    with expect_error(ValueError):
        dm.DenseWeightPlan({"gdn": BF16})
    with expect_error(ValueError):
        dm.DenseWeightPlan(dict.fromkeys(dm.DENSE_MODULES, ttnn.float32))


def test_tags_fidelities_tile_bytes_and_cache_names(expect_error) -> None:
    assert dm.DENSE_DTYPE_TAGS == {BF16: "bf16", BF8: "bf8b", BF4: "bf4b"}
    assert dm.DENSE_MATH_FIDELITY_NAMES == {BF16: "HiFi4", BF8: "HiFi2", BF4: "LoFi"}
    assert dm.DENSE_TILE_BYTES == {BF16: 2048, BF8: 1088, BF4: 576}
    for dtype in (BF16, BF8, BF4):
        assert hasattr(ttnn.MathFidelity, dm.dense_math_fidelity_name(dtype))
    assert dm.dense_weight_name("qg_dram_sharded", BF16) == "qg_dram_sharded"  # the production tensorbin name
    assert dm.dense_weight_name("qg_dram_sharded", BF8) == "qg_dram_sharded.bf8b"
    assert dm.dense_weight_name("shared_down_dram_sharded", BF4) == "shared_down_dram_sharded.bf4b"
    for bad in (ttnn.float32, ttnn.uint16, None, "bf8"):
        with expect_error(ValueError):
            dm.dense_dtype_tag(bad)


def test_two_reader_layouts_are_tile_granular_and_the_qualified_set_gates_them() -> None:
    assert dm.TWO_READER_QUALIFIED_DTYPES == {BF16, BF8, BF4}  # qualified bitwise on a p150b, 2026-09-24
    assert dm.TWO_READER_QUALIFIED_DTYPES <= set(dm.DENSE_DTYPE_TAGS)
    # the bank shard is counted in tiles: the same layout tag for every dtype
    mesh = type("Mesh", (), {"dram_grid_size": staticmethod(lambda: ttnn.CoreCoord(8, 1))})()
    tag = dm.weight_layout_tag(mesh, 2560, 4160, num_workers_per_dram_bank=2)
    assert tag == "_bank576" and dm.bank_tiles(mesh, 2560, 4160, 2) * 32 * dm.DENSE_TILE_BYTES[BF8] == 18 * 32 * 1088
    loader = inspect.getsource(gdn_module.Qwen38TTNNGDNWeights.from_checkpoint)
    assert "projection_dtype not in TWO_READER_QUALIFIED_DTYPES" in loader
    assert "dtype_tag = dense_dtype_tag(projection_dtype)" in loader
    assert "projection_dtype not in DENSE_DTYPE_TAGS" in loader
    builder = inspect.getsource(builder_module.Qwen38TTNNBuilder.__init__)
    assert 'for module in ("gdn", "qsa", "lm_head")' in builder
    assert "not in TWO_READER_QUALIFIED_DTYPES" in builder and "decode_dram_workers_per_bank = 1" in builder
    assert (
        builder.index("qualify_decode_dram_workers(")
        < builder.index("TWO_READER_QUALIFIED_DTYPES")
        < builder.index("live_identity = Qwen38LiveBuildIdentity(")
    )


def _call_blocks(source: str, names: tuple[str, ...]) -> list[str]:
    blocks = []
    for match in re.finditer("|".join(re.escape(n) + r"\(" for n in names), source):
        depth, i = 0, match.end() - 1
        while True:
            c = source[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        blocks.append(source[match.start() : i + 1])
    return blocks


def _check_module(
    module, weights: tuple[str, ...], attribute: str, expected: int, calls=("ttnn.linear", "prefill_linear")
):
    source = inspect.getsource(module)
    converted = [b for b in _call_blocks(source, calls) if any(w in b for w in weights)]
    assert len(converted) == expected, (len(converted), expected)
    # The decode linears take the module's config; the prefill slab's dense linears take it through the slab policy
    # (ttnn/prefill_dense: the same object under the default QWEN38_PREFILL_DENSE_* switches, the policy's fidelity
    # otherwise; test_ttnn_prefill_dense_no_device pins that resolution).
    forms = (
        f"compute_kernel_config=self.{attribute}",
        f"compute_kernel_config=self.prefill_dense.compute_config(self.{attribute})",
    )
    assert all(any(form in b for form in forms) for b in converted), [
        b for b in converted if not any(form in b for form in forms)
    ]
    assert not any("compute_kernel_config=self.compute_config" in b for b in converted)
    return source


def test_gdn_projection_linears_run_the_weight_fidelity() -> None:
    # the one-row forms, the MTP rows forms and the lanes forms (_project_lanes, _gate_and_project_lanes)
    source = _check_module(gdn_module, ("self.weights.qkvzab", "self.weights.out"), "projection_compute_config", 9)
    init = inspect.getsource(gdn_module.Qwen38TTNNGDN.__init__)
    assert "dense_math_fidelity_name(weights.projection_dtype)" in init
    assert (
        "compute_kernel_config=" not in init and "recurrent_read_compute_config" in init
    )  # the step kernels keep HiFi2
    assert "compute_kernel_config=self.compute_config" in source  # the selects and norms keep HiFi4


def test_qsa_projection_linears_run_the_weight_fidelity() -> None:
    weights = (
        "self.weights.qg",
        "self.weights.k_pair_grouped",
        "self.weights.v_pair_grouped",
        "self.weights.out",
        "self.weights.index_q",
        "self.weights.index_k",
    )
    # 15 chain / slab / decode-fused sites + the verify rows step's three (qg, k, v at the projection fidelity, the
    # decode _main_tail_step's blocks over the 32-row tile; qsa_rows program 2, 2026-09-26)
    _check_module(qsa_module, weights, "projection_compute_config", 18)
    rows = inspect.getsource(qsa_module.Qwen38TTNNQSA._linear_rows)
    assert rows.count("compute_kernel_config=self.projection_compute_config") == 3
    assert "compute_kernel_config=self.compute_config" not in rows
    loader = inspect.getsource(qsa_module.Qwen38TTNNQSAWeights.from_checkpoint)
    # the six projection uploads and the merged projections shard (the served fused path's one linear)
    assert len(re.findall(r"(?<!\w)dtype=weight_dtype,", loader)) == 7
    assert "cache / dense_weight_name(name, dtype)" in loader
    validate = inspect.getsource(qsa_module.Qwen38TTNNQSAWeights.validate)
    assert "tensor.dtype != self.weight_dtype" in validate and 'if name == "index_k"' in validate
    assert qsa_module.Qwen38TTNNQSAWeights.__dataclass_fields__["weight_dtype"].default == BF16


def test_shared_expert_linears_run_the_weight_fidelity_and_the_router_keeps_hifi4() -> None:
    # 9 chain / slab / fused-shared-expert sites + the MoE dense composite's two (its [gate | up | scalar] linear and
    # the composite program, which runs the down linear at the config's fidelity)
    source = _check_module(
        moe_module,
        ("self.weights.shared_",),
        "shared_compute_config",
        11,
        calls=("ttnn.linear", "prefill_linear", "fused.shared_expert.shared_expert", "fused.moe_dense.moe_dense"),
    )
    router = [b for b in _call_blocks(source, ("ttnn.linear", "prefill_linear")) if "self.weights.router" in b]
    assert router and all("compute_kernel_config=self.compute_config" in b for b in router)
    loader = inspect.getsource(moe_module.Qwen38TTNNMoEWeights.from_checkpoint)
    assert (
        loader.count("dtype=shared_dtype,") == 5
        and '"router_replicated_dram_sharded",\n            replicate_mapper' in loader
    )
    assert moe_module.Qwen38TTNNMoEWeights.__dataclass_fields__["shared_dtype"].default == BF16


def test_final_mixer_lm_head_and_mtp_linears_run_the_weight_fidelity() -> None:
    _check_module(
        final_mixer_module, ("self.weights.down", "self.weights.up"), "weight_compute_config", 4, calls=("ttnn.linear",)
    )
    mixer = inspect.getsource(final_mixer_module.Qwen38TTNNFinalMixer.__init__)
    assert 'fused_kernels.enabled("final_mixer") and weights.weight_dtype != ttnn.bfloat16' in mixer
    loader = inspect.getsource(final_mixer_module.Qwen38TTNNFinalMixerWeights.from_checkpoint)
    assert '"down-dram-sharded.bf16" if weight_dtype == ttnn.bfloat16 else f"down-dram-sharded.{weight_tag}"' in loader
    head = inspect.getsource(embedding_module.Qwen38TTNNLMHead.__call__)
    assert head.count("compute_kernel_config=self.weight_compute_config") == 1
    assert "compute_kernel_config=self.compute_config" not in head
    io_loader = inspect.getsource(embedding_module.Qwen38TTNNModelIOWeights.from_checkpoint)
    assert (
        "lm-head-chunk-dram-sharded-{lm_head_tag}-{chunk_index:02d}" in io_loader and "dtype=lm_head_dtype" in io_loader
    )
    assert (
        "dtype=ttnn.bfloat16,\n            layout=ttnn.ROW_MAJOR_LAYOUT" in io_loader
    )  # the embedding table stays BF16
    # The one-row mixer's two direct linears; the rows form projects both weights through _project_rows (the 32-row
    # call at 32 rows, per 32-row tile at 128), whose one linear carries the same fidelity.
    _check_module(
        mtp_module,
        ("self.weights.fc_embedding", "self.weights.fc_hidden"),
        "projection_compute_config",
        2,
        calls=("ttnn.linear",),
    )
    rows = inspect.getsource(mtp_module.Qwen38TTNNMTPInput.rows)
    assert "self._project_rows(full_embedding, self.weights.fc_embedding, rows=rows)" in rows
    assert "self._project_rows(full_hidden, self.weights.fc_hidden, rows=rows)" in rows
    project = inspect.getsource(mtp_module.Qwen38TTNNMTPInput._project_rows)
    assert project.count("ttnn.linear(") == 1 and "compute_kernel_config=self.projection_compute_config" in project
    assert "compute_kernel_config=self.compute_config" not in project


def test_the_builder_hands_every_module_its_dtype() -> None:
    source = inspect.getsource(builder_module.Qwen38TTNNBuilder)
    for call in (
        'projection_dtype=self.dense_weight_plan.dtype("gdn")',
        'weight_dtype=self.dense_weight_plan.dtype("qsa")',
        'shared_dtype=self.dense_weight_plan.dtype("shared_expert")',
        'lm_head_dtype=self.dense_weight_plan.dtype("lm_head")',
        'weight_dtype=self.dense_weight_plan.dtype("final_mixer")',
        'projection_dtype=self.dense_weight_plan.dtype("mtp")',
    ):
        assert call in source, call
    assert source.count('weight_dtype=self.dense_weight_plan.dtype("qsa")') == 2  # the backbone and the MTP QSA
    parameters = inspect.signature(builder_module.Qwen38TTNNBuilder.__init__).parameters
    assert parameters["dense_weight_plan"].default is None
    assert "default_dense_weight_plan()" in inspect.getsource(builder_module.Qwen38TTNNBuilder.__init__)
    # the GR mixers are not on the switch: their fused read program streams BF16 tiles
    gr_build = inspect.getsource(builder_module.Qwen38TTNNBuilder._build_gr)
    assert "dense_weight_plan" not in gr_build
    builder_module.validate_builder_constructor_contract()


def test_the_launcher_passes_the_switch_and_the_agreement_flags_through() -> None:
    launcher = (MODEL_DIR / "tools" / "run_qwen38_chat_server.sh").read_text(encoding="utf-8")
    assert "QWEN38_DENSE_WEIGHT_DTYPE" in launcher
    for flag in ("--agreement-reference", "--agreement-parts", "--agreement-full-logits"):
        assert f"        {flag})" in launcher
    assert 'args+=(--agreement-reference "$agreement_reference")' in launcher
    assert "--agreement-reference needs the sampled server" in launcher
    server = (MODEL_DIR / "tools" / "qwen38_chat_server.py").read_text(encoding="utf-8")
    assert '"dense_weight_dtype": report["chain"]["dense_weight_dtype"]' in server
