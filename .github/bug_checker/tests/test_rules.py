# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for rule loading and targeting."""

from bug_checker.rules import Rule, load_manifest, load_rules, select_rules


def _rule(paths=None, labels=None) -> Rule:
    return Rule(
        id="test",
        file="test.md",
        severity="warning",
        suggest_fix=False,
        model=None,
        paths=paths or [],
        labels=labels or [],
    )


def test_load_manifest():
    manifest = load_manifest()
    assert "rules" in manifest
    assert "ccl-ring-buffer-mismatch" in manifest["rules"]
    assert "reshape-dim-check" in manifest["rules"]


def test_load_rules():
    rules = load_rules()
    assert len(rules) >= 2
    ids = {r.id for r in rules}
    assert "ccl-ring-buffer-mismatch" in ids
    assert "reshape-dim-check" in ids
    for rule in rules:
        assert rule.content, f"Rule {rule.id} has no content"


def test_rule_matches_by_path():
    rule = _rule(paths=["ttnn/cpp/ttnn/operations/ccl/**"])
    assert rule.matches_pr(["ttnn/cpp/ttnn/operations/ccl/all_gather/foo.cpp"], [])
    assert not rule.matches_pr(["ttnn/cpp/ttnn/operations/data_movement/bar.cpp"], [])


def test_rule_matches_by_label():
    rule = _rule(labels=["area:ccl"])
    assert rule.matches_pr([], ["area:ccl"])
    assert not rule.matches_pr([], ["area:ops"])


# --- match_reason ---


def test_match_reason_returns_none_when_no_match():
    rule = _rule(paths=["foo/**"], labels=["area:foo"])
    assert rule.match_reason(["bar/x.cpp"], ["area:bar"]) is None


def test_match_reason_identifies_path_and_pattern():
    rule = _rule(paths=["ttnn/cpp/ttnn/operations/ccl/**"])
    reason = rule.match_reason(["ttnn/cpp/ttnn/operations/ccl/foo.cpp"], [])
    assert reason is not None
    assert "ttnn/cpp/ttnn/operations/ccl/foo.cpp" in reason
    assert "ttnn/cpp/ttnn/operations/ccl/**" in reason


def test_match_reason_identifies_label():
    rule = _rule(labels=["area:ccl"])
    reason = rule.match_reason([], ["area:ccl"])
    assert reason is not None
    assert "area:ccl" in reason


def test_match_reason_prefers_path_over_label():
    rule = _rule(paths=["foo/**"], labels=["area:foo"])
    # Both match — path should appear in the reason (checked first)
    reason = rule.match_reason(["foo/bar.cpp"], ["area:foo"])
    assert "foo/bar.cpp" in reason


def test_matches_pr_delegates_to_match_reason():
    rule = _rule(paths=["foo/**"])
    assert rule.matches_pr(["foo/bar.cpp"], []) is True
    assert rule.matches_pr(["baz/bar.cpp"], []) is False


# --- orphan rule ---


def test_orphan_rule_never_matches():
    """A rule with no paths and no labels can never be selected."""
    rule = _rule()
    assert rule.match_reason(["any/file.cpp"], ["any:label"]) is None
    assert not rule.matches_pr(["any/file.cpp"], ["any:label"])


def test_select_rules():
    rules = load_rules()
    selected = select_rules(
        rules,
        changed_files=["ttnn/cpp/ttnn/operations/ccl/something.cpp"],
        pr_labels=[],
    )
    assert any(r.id == "ccl-ring-buffer-mismatch" for r in selected)
    assert not any(r.id == "reshape-dim-check" for r in selected)


def test_select_rules_by_label():
    rules = load_rules()
    selected = select_rules(rules, changed_files=[], pr_labels=["area:ops"])
    assert any(r.id == "reshape-dim-check" for r in selected)


# --- real-world path coverage regressions ---
#
# These pin the naming families that actually occur under
# ttnn/cpp/ttnn/operations/**/device/. The blocking rules below were
# originally scoped to only the modern *_device_operation.cpp /
# *_program_factory.cpp names and silently skipped every op using a legacy
# or one-off name. Each path here is a real file on main.


def test_hash_rule_matches_legacy_op_filenames():
    """program-cache-hash-collision must cover legacy *_op.{cpp,hpp} files.

    These all implement a custom compute_program_hash() but do not match
    the *device_operation* glob.
    """
    rules = load_rules()
    for path in [
        "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp",
        "ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp",
        "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.cpp",
        "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp",
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp",
        "ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp",
        # modern convention must keep working
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_device_operation.cpp",
    ]:
        selected = select_rules(rules, changed_files=[path], pr_labels=[])
        assert any(r.id == "program-cache-hash-collision" for r in selected), path


def test_smuggled_rta_rule_matches_non_program_factory_filenames():
    """smuggled-buffer-runtime-arg must cover every host-side factory family.

    Each of these writes a tensor buffer address into runtime args but does
    not match the original *program_factory* glob.
    """
    rules = load_rules()
    for path in [
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_multicast_factory.cpp",
        "ttnn/cpp/ttnn/operations/index_fill/device/index_fill_multi_core_factory.cpp",
        "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp",
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp",
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp",
        "ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp",
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp",
        # modern convention must keep working
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp",
    ]:
        selected = select_rules(rules, changed_files=[path], pr_labels=[])
        assert any(r.id == "smuggled-buffer-runtime-arg" for r in selected), path


def test_host_rules_do_not_match_device_kernel_sources():
    """Kernel sources under device/kernels/ are device-side code.

    The host-side program-construction rules must not fire on them, or every
    kernel-only PR pays for an irrelevant blocking LLM analysis.
    """
    rules = load_rules()
    for path in [
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_sharded_gn.cpp",
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp",
    ]:
        ids = {r.id for r in select_rules(rules, changed_files=[path], pr_labels=[])}
        assert "smuggled-buffer-runtime-arg" not in ids, path
        assert "program-cache-hash-collision" not in ids, path
        assert "override-rebuild-in-cache-hit" not in ids, path


def test_llk_rule_matches_vendored_llk_tree():
    """tt-llk is vendored in-tree (a plain directory, not a submodule)."""
    rules = load_rules()
    for path in [
        "tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_common.h",
        "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_common_api.h",
    ]:
        selected = select_rules(rules, changed_files=[path], pr_labels=[])
        assert any(r.id == "llk-stale-hw-config-state" for r in selected), path
