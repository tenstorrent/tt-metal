#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit and regression tests for `check_compute_api_order.py` and `compute_api_rules.yaml`."""

import contextlib
import io
import json
import os
import re
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
INFRA = os.path.normpath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.normpath(os.path.join(INFRA, "..", "..", ".."))
SCRIPT = os.path.join(INFRA, "check_compute_api_order.py")
RULES_YAML = os.path.join(INFRA, "compute_api_rules.yaml")
SANITIZER_HEADER = os.path.join(
    INFRA, "..", "tools", "include", "sanitizer", "operation.h"
)

sys.path.insert(0, INFRA)
import check_compute_api_order as checker_mod  # noqa: E402


def run_cli(*args):
    """Run `check_compute_api_order.main()` in-process with cached rule table."""
    argv = [*map(str, args)]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = checker_mod.main(argv)
    return types.SimpleNamespace(
        returncode=code, stdout=out.getvalue(), stderr=err.getvalue()
    )


def write_kernel(tmp_path, body: str, name: str = "kernel.cpp"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def parse_sanitizer_operations_from_header(header_path: str):
    """Parse `Operation*` structs from `tt_metal/tt-llk/tools/include/sanitizer/operation.h`."""
    src = open(header_path, "r", encoding="utf-8").read()
    struct_re = re.compile(
        r"struct\s+(Operation\w+)\s*:\s*Operation<\s*Exu::(\w+)\s*,\s*Hoistable::(\w+)\s*>\s*\{(.*?)\n\};",
        re.DOTALL,
    )
    ops = {}
    for m in struct_re.finditer(src):
        op_name, exu, hoist, body = m.group(1), m.group(2), m.group(3), m.group(4)
        fields = re.findall(r"struct\s+([A-Za-z_]\w*)\s*:\s*Field<", body)
        ops[op_name] = {
            "exu": exu,
            "hoistable": hoist == "Yes",
            "state_fields": fields,
        }
    return ops


# ---------------------------------------------------------------------------
# 1. Rule Table & LLK Sanitizer Consistency Tests
# ---------------------------------------------------------------------------


def test_rule_table_matches_llk_sanitizer_operation_header():
    """Verify `compute_api_rules.yaml` agrees 1-to-1 with `sanitizer/operation.h`."""
    rules = checker_mod.RuleTable(RULES_YAML)
    header_ops = parse_sanitizer_operations_from_header(SANITIZER_HEADER)
    assert len(header_ops) >= 14
    assert set(rules.sanitizer_operations.keys()) == set(header_ops.keys())

    for op_name, hdr_spec in header_ops.items():
        yaml_spec = rules.sanitizer_operations[op_name]
        assert yaml_spec["exu"] == hdr_spec["exu"], f"{op_name}: exu mismatch"
        assert (
            yaml_spec["hoistable"] == hdr_spec["hoistable"]
        ), f"{op_name}: hoistable mismatch"
        assert (
            yaml_spec["state_fields"] == hdr_spec["state_fields"]
        ), f"{op_name}: state_fields mismatch"


def test_every_rule_entry_cites_existing_source_and_deprecations_manifest():
    """Verify every entry in `compute_api_rules.yaml` cites an existing repo file or issue."""
    rules = checker_mod.RuleTable(RULES_YAML)
    dep_manifest = os.path.join(REPO_ROOT, rules.raw["deprecations_manifest"])
    assert os.path.isfile(dep_manifest)
    dep_data = json.load(open(dep_manifest, "r", encoding="utf-8"))
    assert "deprecations" in dep_data

    sections = [
        rules.sanitizer_operations,
        rules.llk_operations,
        rules.sfpu_operations,
        rules.reconfig_operations,
        rules.compute_api_wrappers,
        rules.deprecated_apis,
    ]
    for sec in sections:
        for key, spec in sec.items():
            src = spec.get("source", "")
            assert src, f"{key} is missing a `source` citation"
            path_part = src.split(":")[0].strip()
            full_path = os.path.join(REPO_ROOT, path_part)
            assert os.path.isfile(
                full_path
            ), f"{key} cites non-existent file `{path_part}`"


# ---------------------------------------------------------------------------
# 2. Rule 1: `missing-init` Tests
# ---------------------------------------------------------------------------


def test_valid_eltwise_add_kernel_passes(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_binary.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 1, 16);
            add_init(0, 1);
            for (std::uint32_t i = 0; i < 4; ++i) {
                tile_regs_acquire();
                add_tiles(0, 1, i, i, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, 16);
                tile_regs_release();
            }
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["total_findings"] == 0


def test_missing_init_detected_on_unpack_and_math(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_binary.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 1, 16);
            // Missing add_init(0, 1)!
            add_tiles(0, 1, 0, 0, 0);
            pack_tile(0, 16);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert data["counts_by_rule"]["missing-init"] > 0
    threads_flagged = {
        f["thread"] for f in data["findings"] if f["rule"] == "missing-init"
    }
    assert threads_flagged == {"UNPACK", "MATH"}


def test_missing_init_on_partial_branch_path(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_binary.h"
        void kernel_main() {
            std::uint32_t cond = get_arg_val<std::uint32_t>(0);
            compute_kernel_hw_startup(0, 1, 16);
            if (cond) {
                add_tiles_init(0, 1);
            }
            add_tiles(0, 1, 0, 0, 0);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert data["counts_by_rule"]["missing-init"] >= 2


def test_matching_inits_on_both_branches_passes(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/matmul.h"
        void kernel_main() {
            std::uint32_t cond = get_arg_val<std::uint32_t>(0);
            if (cond) {
                mm_init(0, 1, 16);
            } else {
                compute_kernel_hw_startup(0, 1, 16);
                mm_init_short(0, 1);
            }
            matmul_tiles(0, 1, 0, 0, 0);
            pack_tile(0, 16);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 0


# ---------------------------------------------------------------------------
# 3. Rule 2: `arg-mismatch` & Historical Bug Reproductions (#59276, #39945)
# ---------------------------------------------------------------------------


def test_historical_bug_59276_unpack_a_acc_to_dest_mismatch(tmp_path):
    """Reproduce #59276: `llk_unpack_A_init` with `acc_to_dest=true` vs `llk_unpack_A` with `acc_to_dest=false`."""
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/transpose.h"
        void kernel_main() {
            UNPACK((llk_unpack_A_init<BroadcastType::NONE, true>(true, true, 0)));
            UNPACK((llk_unpack_A<BroadcastType::NONE, false>(0, 0, false)));
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    mismatches = [f for f in data["findings"] if f["rule"] == "arg-mismatch"]
    assert len(mismatches) >= 1
    assert any(
        "AccumulateToDest" in f["message"] and "#59276" in f["source_ref"]
        for f in mismatches
    )


def test_historical_bug_39945_reconfig_data_format_matmul_operand_reversal(tmp_path):
    """Reproduce #39945 (`llk_asserts.rst`): `reconfig_data_format(in0, in1)` before `matmul_tiles(in0, in1, ...)`."""
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/matmul.h"
        #include "api/compute/reconfig_data_format.h"
        void kernel_main() {
            constexpr std::uint32_t cb_in0 = 0;
            constexpr std::uint32_t cb_in1 = 1;
            mm_init(cb_in0, cb_in1, 16);
            reconfig_data_format(cb_in0, cb_in1);
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    mismatches = [f for f in data["findings"] if f["rule"] == "arg-mismatch"]
    assert any("#39945" in f["message"] for f in mismatches)


def test_reconfig_data_format_reversed_operands_before_matmul_passes(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/matmul.h"
        #include "api/compute/reconfig_data_format.h"
        void kernel_main() {
            constexpr std::uint32_t cb_in0 = 0;
            constexpr std::uint32_t cb_in1 = 1;
            mm_init(cb_in0, cb_in1, 16);
            reconfig_data_format(cb_in1, cb_in0);
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 0


def test_bcast_type_and_binary_type_arg_mismatch(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/bcast.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 1, 16);
            add_bcast_rows_init_short(0, 1);
            add_tiles_bcast_cols(0, 1, 0, 0, 0);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert data["counts_by_rule"]["arg-mismatch"] >= 2


def test_runtime_unknown_arguments_are_never_guessed(tmp_path):
    """Verify runtime arguments produce `unknown(...)` rather than false-positive guesses."""
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/matmul.h"
        void kernel_main() {
            std::uint32_t runtime_ct_dim = get_arg_val<std::uint32_t>(0);
            std::uint32_t runtime_rt_dim = get_arg_val<std::uint32_t>(1);
            std::uint32_t runtime_kt_dim = get_arg_val<std::uint32_t>(2);
            mm_block_init(0, 1, 16, 0, runtime_ct_dim, runtime_rt_dim, runtime_kt_dim);
            matmul_block(0, 1, 0, 0, 0, 0, runtime_ct_dim, runtime_rt_dim, runtime_kt_dim);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 0
    env = {}
    val = checker_mod.normalize_const("runtime_ct_dim", env)
    assert not val.is_known
    assert val.format_for_diag() == "unknown(runtime_ct_dim)"


# ---------------------------------------------------------------------------
# 4. Rule 3: `init-conflict` & Historical Bug Reproductions (#61146, #39184, #50381)
# ---------------------------------------------------------------------------


def test_historical_bug_61146_blackhole_pack_reconfig_after_tilize_init(tmp_path):
    """Reproduce #61146: `pack_reconfig_data_format` after `tilize_init` before `tilize_block` on Blackhole."""
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/tilize.h"
        #include "api/compute/reconfig_data_format.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 16);
            tilize_init(0, 4, 16);
            pack_reconfig_data_format(17);
            tilize_block(0, 4, 17);
            tilize_uninit(0, 17);
        }
        """,
    )
    # Must fail on Blackhole with #61146 `init-conflict`
    res_bh = run_cli(k, "--arch", "blackhole", "--json")
    assert res_bh.returncode == 1
    data_bh = json.loads(res_bh.stdout)
    conflicts_bh = [f for f in data_bh["findings"] if f["rule"] == "init-conflict"]
    assert len(conflicts_bh) == 1
    assert conflicts_bh[0]["thread"] == "PACK"
    assert "#61146" in conflicts_bh[0]["message"]

    # Wormhole's packer stride setup is unaffected by #61146
    res_wh = run_cli(k, "--arch", "wormhole_b0", "--json")
    assert res_wh.returncode == 0


def test_historical_bug_39184_tilize_ab_unpacker_init_overwritten(tmp_path):
    """Reproduce #39184 (`llk_asserts.rst`): unpacker reconfigured to A mode before `unpack_tilizeA_B_block`."""
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/tilize.h"
        #include "api/compute/tile_move_copy.h"
        void kernel_main() {
            tilizeA_B_reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(0, 1, 4, 16);
            copy_tile_init(0);
            unpack_tilizeA_B_block(0, 1, 4, 0);
            reduce_uninit();
            tilize_uninit(0, 16);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    conflicts = [f for f in data["findings"] if f["rule"] == "init-conflict"]
    assert any(
        f["thread"] == "UNPACK" and f["earlier_call"] == "copy_tile_init"
        for f in conflicts
    )


def test_historical_bug_50381_sfpu_init_clobbers_constants(tmp_path):
    """Reproduce #50381 (`exp.h`): `recip_tile_init()` between `exp_tile_init()` and `exp_tile()`."""
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_unary/exp.h"
        #include "api/compute/eltwise_unary/recip.h"
        void kernel_main() {
            exp_tile_init();
            recip_tile_init();
            exp_tile(0);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    conflicts = [f for f in data["findings"] if f["rule"] == "init-conflict"]
    assert len(conflicts) >= 1
    assert conflicts[0]["exu"] == "Sfpu"
    assert conflicts[0]["earlier_call"] == "recip_tile_init"


def test_fpu_and_sfpu_inits_do_not_conflict_on_math_thread(tmp_path):
    """Verify `copy_tile_init` (`Exu::Fpu`) and `exp_tile_init` (`Exu::Sfpu`) coexist without conflict."""
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/tile_move_copy.h"
        #include "api/compute/eltwise_unary/exp.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 16);
            copy_tile_init(0);
            exp_tile_init();
            for (std::uint32_t i = 0; i < 4; ++i) {
                copy_tile(0, i, 0);
                exp_tile(0);
                pack_tile(0, 16);
            }
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 0


# ---------------------------------------------------------------------------
# 5. Rule 4: `missing-uninit` & Historical Bug #52175
# ---------------------------------------------------------------------------


def test_historical_bug_52175_missing_uninit_for_tilize_untilize_and_reduce(tmp_path):
    """Reproduce #52175: missing `pack_untilize_uninit` / `tilize_uninit` / `reduce_uninit`."""
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/pack_untilize.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 16);
            pack_untilize_init<4>(0, 16);
            pack_untilize_block<4>(0, 1, 16);
            // Missing pack_untilize_uninit(16)!
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert data["counts_by_rule"]["missing-uninit"] >= 1


def test_tilize_and_reduce_with_proper_uninit_passes(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/tilize.h"
        #include "api/compute/reduce.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 16);
            tilize_init(0, 4, 16);
            tilize_block(0, 4, 16);
            tilize_uninit(0, 16);

            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(0, 1, 16);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(0, 1, 0, 0, 0);
            reduce_uninit();
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 0


# ---------------------------------------------------------------------------
# 6. Rule 5: `deprecated-init` Tests
# ---------------------------------------------------------------------------


def test_deprecated_init_apis_flagged(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/untilize.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 16);
            untilize_init_short(0);
            untilize_uninit(0);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert data["counts_by_rule"]["deprecated-init"] >= 1
    assert any(
        "untilize_init" in f["message"]
        for f in data["findings"]
        if f["rule"] == "deprecated-init"
    )


# ---------------------------------------------------------------------------
# 7. Rule 6: `loop-invariant-init` (Warning-Only) Tests
# ---------------------------------------------------------------------------


def test_loop_invariant_init_emits_warning_and_exits_zero(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_binary.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 1, 16);
            for (std::uint32_t i = 0; i < 8; ++i) {
                add_tiles_init(0, 1);
                add_tiles(0, 1, i, i, 0);
                pack_tile(0, 16);
            }
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["counts_by_rule"]["loop-invariant-init"] >= 2
    assert all(f["severity"] == "warning" for f in data["findings"])


def test_alternating_inits_in_loop_not_flagged_as_loop_invariant(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_binary.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 1, 16);
            for (std::uint32_t i = 0; i < 8; ++i) {
                add_tiles_init(0, 1);
                add_tiles(0, 1, i, i, 0);
                mul_tiles_init(0, 1);
                mul_tiles(0, 1, i, i, 0);
            }
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["counts_by_rule"]["loop-invariant-init"] == 0


# ---------------------------------------------------------------------------
# 8. Helper Indirection, `#include` Resolution, `if constexpr`, and `TT_KERNEL`
# ---------------------------------------------------------------------------


def test_follows_calls_through_included_helper_and_if_constexpr(tmp_path):
    write_kernel(
        tmp_path,
        """
        #pragma once
        template <bool DO_INIT, bool USE_SUB>
        inline void run_binary_step(std::uint32_t cb0, std::uint32_t cb1) {
            if constexpr (DO_INIT) {
                if constexpr (USE_SUB) {
                    sub_tiles_init(cb0, cb1);
                } else {
                    add_tiles_init(cb0, cb1);
                }
            }
            add_tiles(cb0, cb1, 0, 0, 0);
        }
        """,
        name="my_helper.h",
    )

    k_ok = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_binary.h"
        #include "my_helper.h"
        TT_KERNEL(void, my_entry, (std::uint32_t n)) {
            compute_kernel_hw_startup(0, 1, 16);
            run_binary_step<true, false>(0, 1);
        }
        """,
        name="kernel_ok.cpp",
    )
    assert run_cli(k_ok).returncode == 0

    k_mismatch = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_binary.h"
        #include "my_helper.h"
        TT_KERNEL(void, my_entry, (std::uint32_t n)) {
            compute_kernel_hw_startup(0, 1, 16);
            run_binary_step<true, true>(0, 1);
        }
        """,
        name="kernel_mismatch.cpp",
    )
    res_mm = run_cli(k_mismatch, "--json")
    assert res_mm.returncode == 1
    assert json.loads(res_mm.stdout)["counts_by_rule"]["arg-mismatch"] >= 1


def test_matmul_block_init_and_matmul_block_matching_dims_pass(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/matmul.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 1, 16);
            matmul_block_init(0, 1, false, 4, 2, 1);
            matmul_block(0, 1, 0, 0, 0, false, 4, 2, 1);
            pack_tile(0, 16);
        }
        """,
    )
    assert run_cli(k).returncode == 0


def test_matmul_block_init_and_matmul_block_ct_dim_mismatch_flagged(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/matmul.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 1, 16);
            matmul_block_init(0, 1, false, 4, 2, 1);
            matmul_block(0, 1, 0, 0, 0, false, 2, 2, 1);
        }
        """,
    )
    res = run_cli(k, "--json")
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert data["counts_by_rule"]["arg-mismatch"] >= 2


def test_pack_untilize_uninit_restores_pack_init_for_subsequent_pack_tile(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/pack_untilize.h"
        #include "api/compute/tile_move_copy.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 16);
            pack_untilize_dest_init<4, 4>(16);
            pack_untilize_dest<4, 4>(16, 1, 0);
            pack_untilize_uninit(16);
            copy_init(0);
            copy_tile(0, 0, 0);
            pack_tile(0, 16);
        }
        """,
    )
    assert run_cli(k).returncode == 0


def test_reduce_init_pack_tile_and_reduce_uninit_passes(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/reduce.h"
        void kernel_main() {
            compute_kernel_hw_startup(0, 1, 16);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(0, 1, 16);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(0, 1, 0, 0, 0);
            pack_tile(0, 16);
            reduce_uninit();
        }
        """,
    )
    assert run_cli(k).returncode == 0


def test_tt_kernel_prefix_attribute_entry_point_with_preceding_helpers(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/eltwise_binary.h"
        #include "experimental/kernel_args.h"
        void helper_pack(std::uint32_t out_cb) {
            pack_tile(0, out_cb);
        }
        TT_KERNEL void compute(std::uint32_t n) {
            compute_kernel_hw_startup(0, 1, 16);
            add_init(0, 1);
            add_tiles(0, 1, 0, 0, 0);
            helper_pack(16);
        }
        """,
    )
    assert run_cli(k).returncode == 0


def test_correlated_if_constexpr_and_first_iter_loop_guard_pass(tmp_path):
    k = write_kernel(
        tmp_path,
        """
        #include "api/compute/reduce.h"
        #include "api/compute/tile_move_copy.h"
        void kernel_main() {
            for (std::uint32_t x = 0; x < 2; ++x) {
                if (x == 0) {
                    compute_kernel_hw_startup(0, 1, 16);
                } else {
                    reconfig_data_format(0, 1);
                    pack_reconfig_data_format(16);
                }
                if constexpr (unknown_compile_flag) {
                    copy_init(0);
                } else {
                    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(0, 1, 16);
                }
                if constexpr (unknown_compile_flag) {
                    copy_tile(0, 0, 0);
                    pack_tile(0, 16);
                } else {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(0, 1, 0, 0, 0);
                    pack_tile(0, 16);
                    reduce_uninit();
                }
            }
        }
        """,
    )
    assert run_cli(k).returncode == 0
