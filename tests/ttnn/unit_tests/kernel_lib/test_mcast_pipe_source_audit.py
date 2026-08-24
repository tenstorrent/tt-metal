# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER_PATH = REPO_ROOT / "helper_design/mcast_pipe/migration/ledger.json"


def test_mcast_args_owns_its_compile_time_presence_tag():
    helper = (REPO_ROOT / "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp").read_text()
    host = (REPO_ROOT / "ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp").read_text()

    assert "OptionalMcastArgs" not in helper
    assert "McastArgsImpl<(get_compile_time_arg_val(CT_BASE) != 0)" in helper
    assert "return CT_BASE + 7;" in helper
    assert "return CT_BASE + 1;" in helper
    assert "McastArgs::sender() cannot be used when the presence tag is false" in helper
    assert "McastArgs::receiver() cannot be used when the presence tag is false" in helper
    assert "return {0u};" in host
    assert "if (compact_wire_)" in host
    assert host.count("                1u,\n                has_receivers_ ? 1u : 0u,") == 1
    assert host.count("            2u,\n            has_receivers_ ? 1u : 0u,") == 1


def test_mcast_args_has_one_template_owned_runtime_base():
    helper = (REPO_ROOT / "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp").read_text()
    assert "runtime_args_base_" not in helper
    assert "runtime_args_base()" not in helper
    assert "runtime_args_end()" not in helper
    assert "McastArgsImpl(uint32_t" not in helper

    kernels, _ = _migrated_sources()
    dynamic_construction = re.compile(r"\b(?:McastArgs|\w+McastArgs)\s+\w+\s*\(\s*\w+")
    violations = [
        str(kernel.relative_to(REPO_ROOT)) for kernel in kernels if dynamic_construction.search(kernel.read_text())
    ]
    assert not violations, "McastArgs runtime bases must come only from the RT_BASE template argument:\n" + "\n".join(
        violations
    )


def _migrated_sources():
    ledger = json.loads(LEDGER_PATH.read_text())
    kernels = [REPO_ROOT / entry["kernel"] for entry in ledger["entries"] if entry["status"] == "migrated"]
    factories = {
        REPO_ROOT / binding["factory"] for binding in ledger["host_bindings"] if binding["status"] == "migrated"
    }
    return kernels, sorted(factories)


def test_migrated_host_bindings_treat_helper_outputs_as_opaque_ranges():
    _, factories = _migrated_sources()
    declaration = re.compile(r"\b(?:const\s+)?auto\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", re.DOTALL)

    violations = []
    for factory in factories:
        if factory.suffix != ".cpp":
            continue
        source = factory.read_text()
        helper_outputs = {
            name
            for name, expression in declaration.findall(source)
            if "compile_time_args(" in expression or "runtime_args(" in expression
        }
        for name in helper_outputs:
            if re.search(rf"\b{re.escape(name)}\s*\[", source):
                violations.append(f"{factory.relative_to(REPO_ROOT)} indexes helper output `{name}`")

    assert not violations, "Helper argument vectors must be copied as complete ranges:\n" + "\n".join(violations)


def test_migrated_host_bindings_use_append_style_helper_tails():
    _, factories = _migrated_sources()
    allowed_queries = {
        "ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp": 2,
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp": 4,
        "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp": 1,
    }

    violations = []
    for factory in factories:
        if factory.suffix != ".cpp":
            continue
        source = factory.read_text()
        getter_calls = len(re.findall(r"\.(?:compile_time_args|runtime_args)\(", source))
        if "kernel_lib::host::Mcast" not in source and getter_calls == 0:
            continue
        relative = str(factory.relative_to(REPO_ROOT))
        if getter_calls != allowed_queries.get(relative, 0):
            violations.append(f"{relative} has {getter_calls} non-append helper getter calls")
        if "append_compile_time_args_to" not in source or "append_runtime_args_to" not in source:
            violations.append(f"{relative} does not emit both helper tails through append APIs")

    assert not violations, "Migrated producers must append complete helper tails:\n" + "\n".join(violations)


def test_migrated_kernels_do_not_repeat_rotating_span_as_a_template_argument():
    kernels, _ = _migrated_sources()
    third_template_argument = re.compile(r"\bMcastArgs\s*<\s*[^,>]+\s*,\s*[^,>]+\s*,")

    violations = [
        str(kernel.relative_to(REPO_ROOT)) for kernel in kernels if third_template_argument.search(kernel.read_text())
    ]

    assert not violations, "McastArgs rotating span must come only from the v10 CT wire:\n" + "\n".join(violations)


def test_mixed_role_kernels_use_direct_mcast_pipe_aliases():
    helper = (REPO_ROOT / "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp").read_text()
    assert re.search(r"using SenderPipe\s*=\s*SenderPipeFor<noc_index>;", helper)
    assert re.search(r"using ReceiverPipe\s*=\s*dataflow_kernel_lib::ReceiverPipe<", helper)

    kernels, _ = _migrated_sources()
    expression_recovery = re.compile(r"decltype\([^\n]*\.(?:sender|receiver)\(")
    violations = [
        str(kernel.relative_to(REPO_ROOT)) for kernel in kernels if expression_recovery.search(kernel.read_text())
    ]
    assert not violations, "Mixed-role pipe storage must use McastArgs aliases:\n" + "\n".join(violations)

    mixed_role_kernels = [
        REPO_ROOT / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
        REPO_ROOT / "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp",
        REPO_ROOT / "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp",
        REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/"
        "reader_mcast_transformer_group_attn_matmul.cpp",
    ]
    for kernel in mixed_role_kernels:
        source = kernel.read_text()
        assert "::SenderPipe" in source
        assert "::ReceiverPipe" in source
        assert "std::optional<" in source

    group_attention = mixed_role_kernels[-1].read_text()
    assert "if (in1_mcast_args.can_send())" in group_attention
    assert "if (in1_mcast_args.can_receive())" in group_attention
    assert "if (in1_mcast_args.should_send(tile_row_id))" in group_attention
    assert "else if (in1_mcast_args.can_receive())" in group_attention
    assert "in1_sender_pipe.emplace(noc, in1_mcast_args.rect(), in1_mcast_num_dests);" in group_attention
    assert "in1_sender_in_receiver_grid" not in group_attention
    assert "sender_x(tile_row_id)" not in group_attention
    assert "sender_y(tile_row_id)" not in group_attention


def test_migrated_kernels_use_objects_for_offsets_and_reserve_aliases_for_pipe_types():
    kernels, _ = _migrated_sources()
    violations = []
    alias_pattern = re.compile(r"using\s+(\w+)\s*=\s*(?:dataflow_kernel_lib::\s*)?McastArgs<")

    for path in kernels:
        source = path.read_text()
        relative = str(path.relative_to(REPO_ROOT))
        for alias_match in alias_pattern.finditer(source):
            alias = alias_match.group(1)
            if not re.search(rf"\b{alias}::(?:SenderPipe|ReceiverPipe)\b", source):
                violations.append(f"{relative} retains alias {alias} without a nested pipe-type use")
            if re.search(rf"\b{alias}::next_(?:compile_time|runtime)_args_offset\(\)", source):
                violations.append(f"{relative} type-qualifies an offset through {alias}")

    assert (
        not violations
    ), "McastArgs aliases must be reserved for pipe types and offsets chained through objects:\n" + (
        "\n".join(violations)
    )


def test_sort_row_start_readiness_is_pipe_owned():
    kernels, factories = _migrated_sources()
    sort_kernels = [path for path in kernels if "single_row_multi_core.cpp" in path.name]
    sort_factories = [path for path in factories if path.name == "sort_program_factory.cpp"]

    assert len(sort_factories) == 1
    violations = [
        str(path.relative_to(REPO_ROOT)) for path in sort_kernels if "cores_to_coordinator_ready" in path.read_text()
    ]
    factory_source = sort_factories[0].read_text()
    assert ".handshake = true" in factory_source
    assert "row_start_mcast" in factory_source and "substage_mcast" in factory_source
    assert not violations, "Sort row-start readiness must remain inside the handshaked Pipe:\n" + "\n".join(violations)


OPAQUE_BOUNDARY_RULES = {
    "reader_bmm_tile_layout_in1_sender_writer_padding.cpp": [
        (r"rt_args_idx\s*\+=\s*4", "manual runtime skip assumes the helper wire width"),
    ],
    "reader_mcast_sender_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-2])\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "reader_mcast_receiver_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|20)\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "welford_reader_mcast_sender_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-5])\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-3])\)", "fixed operation tail bypasses the mcast chain"),
    ],
}


@pytest.mark.parametrize("kernel,rules", OPAQUE_BOUNDARY_RULES.items(), ids=OPAQUE_BOUNDARY_RULES)
def test_migrated_kernels_resume_from_named_mcast_boundaries(kernel, rules):
    kernels, _ = _migrated_sources()
    matching_paths = [path for path in kernels if path.name == kernel]
    assert len(matching_paths) == 1, f"Expected one migrated ledger entry for {kernel}, found {matching_paths}"

    source = matching_paths[0].read_text()
    violations = [message for pattern, message in rules if re.search(pattern, source)]
    assert not violations, f"{matching_paths[0].relative_to(REPO_ROOT)}: " + "; ".join(violations)


MATMUL_MCAST_SOURCES = [
    REPO_ROOT / "ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp",
    REPO_ROOT / "ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp",
    REPO_ROOT / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
    REPO_ROOT
    / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
    REPO_ROOT
    / "ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp",
]


def test_matmul_migration_has_one_unconditional_mcast_abi():
    violations = [str(path.relative_to(REPO_ROOT)) for path in MATMUL_MCAST_SOURCES if "MCAST_ARGS" in path.read_text()]
    assert not violations, "Matmul must use McastArgs without a conditional legacy ABI:\n" + "\n".join(violations)


def test_matmul_in0_padding_appends_mcast_after_fixed_operation_args():
    kernel = MATMUL_MCAST_SOURCES[2].read_text()
    accessor = kernel.index("constexpr auto in0_args = TensorAccessorArgs<24>();")
    operation_end = kernel.index("constexpr uint32_t operation_ct_args_end")
    helper = kernel.index("McastArgs<operation_ct_args_end, 4>")
    assert accessor < operation_end < helper
    assert "constexpr dataflow_kernel_lib::McastArgs<operation_ct_args_end, 4> in0_mcast_args;" in kernel
    assert "rt_args_idx = in0_mcast_args.next_runtime_args_offset();" in kernel

    factory_sources = "\n".join(path.read_text() for path in MATMUL_MCAST_SOURCES[:2])
    assert "begin() + 15" not in factory_sources


def test_matmul_1d_inactive_operands_emit_only_tagged_absent_mcast_blocks():
    source = MATMUL_MCAST_SOURCES[0].read_text()
    assert source.count("const ttnn::kernel_lib::host::Mcast2D in0_mcast") == 2
    assert source.count("const ttnn::kernel_lib::host::Mcast2D in1_mcast") == 2
    assert source.count("in1_mcast.append_compile_time_args_to(") == 4
    assert source.count("in1_mcast.append_runtime_args_to(") == 4
    assert source.count("append_absent_mcast_compile_time_args_to(") == 4
    assert "in1_mcast_dest_noc_start_x" not in source
    assert "in1_mcast_num_dests" not in source

    in0_kernel = MATMUL_MCAST_SOURCES[2].read_text()
    in1_kernel = MATMUL_MCAST_SOURCES[3].read_text()
    assert "McastArgs<operation_ct_args_end, 4>" in in0_kernel
    assert "McastArgs<operation_ct_args_end, operation_rt_args_end>" in in1_kernel
    assert "OptionalMcastArgs" not in in0_kernel + in1_kernel
    all_sources = "".join(path.read_text() for path in MATMUL_MCAST_SOURCES)
    assert "mcast_args_present" not in all_sources
    assert all_sources.count("append_absent_mcast_compile_time_args_to(") == 5
    assert "#ifndef SKIP_MCAST\n    constexpr auto in0_mcast_args" not in in0_kernel
    assert "#ifndef SKIP_MCAST\n    constexpr auto in1_mcast_args" not in in1_kernel


def test_matmul_1d_partial_rectangles_preserve_divergent_ack_counts():
    source = MATMUL_MCAST_SOURCES[0].read_text()
    assert source.count("in0_mcast_config, num_cores - 1") == 2
    assert len(re.findall(r"McastConfig\{\.noc = in1_noc, \.base_sem_id = 0\},\s*num_cores - 1", source)) == 2


def test_matmul_in0_common_helper_tails_are_appended_once_after_each_branch():
    append = "in0_mcast.append_compile_time_args_to(in0_sender_compile_time_args);"
    after_conditional = re.compile(r"^    }\n    " + re.escape(append), re.MULTILINE)

    for path in MATMUL_MCAST_SOURCES[:2]:
        source = path.read_text()
        assert source.count(append) == 2
        assert len(after_conditional.findall(source)) == 2


def test_sparse_matmul_bindings_put_accessors_before_the_active_helper_tail():
    source = MATMUL_MCAST_SOURCES[4].read_text()

    in0_ct_helper = source.index("in0_mcast.append_compile_time_args_to(in0_sender_compile_time_args)")
    fixed_tail_matches = [
        match.start()
        for match in re.finditer(r"\(std::uint32_t\)false,\s*// fuse_op", source)
        if match.start() < in0_ct_helper
    ]
    assert fixed_tail_matches
    in0_ct_fixed_tail = fixed_tail_matches[-1]
    in0_ct_accessor = source.index("TensorAccessorArgs(*in0_buffer)")
    assert in0_ct_fixed_tail < in0_ct_accessor < in0_ct_helper

    in0_rt_fixed_tail = source.index("(std::uint32_t)sparsity_buffer->address()};")
    in0_rt_helper = source.index("in0_mcast.append_runtime_args_to(mm_in0_sender_args, core)")
    assert in0_rt_fixed_tail < in0_rt_helper

    assert "const auto in1_mcast_compile_time_args = in1_mcast.compile_time_args();" not in source
    assert "const auto in1_mcast_runtime_args = in1_mcast.runtime_args(core);" not in source
    assert "in1_mcast_dest_noc_start_x" not in source
    assert "in1_mcast_num_dests" not in source


def test_matmul_mcast_objects_reuse_their_kernel_descriptor_nocs():
    for path in (MATMUL_MCAST_SOURCES[0], MATMUL_MCAST_SOURCES[1], MATMUL_MCAST_SOURCES[4]):
        source = path.read_text()
        assert not re.search(
            r"McastConfig\s*\{\s*\.noc\s*=\s*tt::tt_metal::detail::preferred_noc_for_dram_",
            source,
        ), f"{path.relative_to(REPO_ROOT)} recomputes a NoC inside McastConfig"
        assert "McastConfig{.noc = in0_noc" in source or ".noc = in0_noc," in source
        assert "McastConfig{.noc = in1_noc" in source or ".noc = in1_noc," in source


def test_block_sharded_matmul_keeps_receiver_geometry_separate_from_sender_span():
    factory_1d = MATMUL_MCAST_SOURCES[0].read_text()
    factory_2d = MATMUL_MCAST_SOURCES[1].read_text()

    assert "CoreRange in0_mcast_rect = all_cores_with_work.bounding_box();" in factory_1d
    assert "in0_mcast_senders" in factory_1d
    assert factory_2d.count("device, output_work_grid, in0_tensor.shard_spec()->grid") == 2
    assert factory_2d.count("in0_mcast.participating_cores()") == 2
    assert factory_2d.count("in0_mcast.sender_only_cores()") == 2
    assert "in0_mcast_sender_lines" not in factory_2d
    assert not re.search(r"Mcast[12]D\([^;]+CoreRangeSet\(all_cores\)", factory_1d, re.DOTALL)
    assert not re.search(r"Mcast[12]D\([^;]+CoreRangeSet\(all_cores\)", factory_2d, re.DOTALL)


def test_conv3d_mcast_precedes_the_dynamic_operation_tail_and_scopes_weight_share_modes():
    factory = (REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp").read_text()
    kernel = (REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp").read_text()

    assert "constexpr McastArgs<bias_args.next_compile_time_args_offset(), 21> weights_mcast_args;" in kernel
    assert "argidx = weights_mcast_args.next_runtime_args_offset();" in kernel
    assert kernel.index("const uint32_t mcast_num_iters") < kernel.index("weights_mcast_args;")
    assert "weights_mcast.append_runtime_args_to(writer_args, core);" in factory
    assert factory.index("writer_args.push_back(cw.mcast_num_iters);") < factory.index(
        "weights_mcast.append_runtime_args_to(writer_args, core);"
    )
    assert factory.index("weights_mcast.append_runtime_args_to(writer_args, core);") < factory.index(
        "if (num_workers > 0)"
    )
    assert ": weights_mcast_template;" in factory
    assert "mcast_bbox_start_x" not in factory + kernel
    assert "mcast_num_dests" not in factory + kernel


def test_sender_pipe_degenerate_copy_preserves_async_write_semantics():
    source = (REPO_ROOT / "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl").read_text()
    start = source.index("    local_copy_(uint32_t src_l1")
    body = source[start : source.index("// ReceiverPipe", start)]
    assert "noc_.async_write(" in body
    assert "async_read" not in body
    assert "barrier" not in body


def test_groupnorm_uses_one_family_wire_and_preserves_legacy_phases():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/normalization/groupnorm/device"
    factories = [
        (base / "groupnorm_mcast_program_factory.cpp").read_text(),
        (base / "groupnorm_no_mcast_program_factory.cpp").read_text(),
        (base / "groupnorm_sharded_program_factory.cpp").read_text(),
    ]
    kernel_base = base / "kernels/dataflow"
    legacy_sender = (kernel_base / "reader_mcast_sender_unary_gn.cpp").read_text()
    legacy_receiver = (kernel_base / "reader_mcast_receiver_unary_gn.cpp").read_text()
    welford_sender = (kernel_base / "welford_reader_mcast_sender_unary_gn.cpp").read_text()
    welford_receiver = (kernel_base / "welford_reader_mcast_receiver_unary_gn.cpp").read_text()
    sharded_legacy_sender = (kernel_base / "reader_mcast_sender_unary_sharded_gn_v2.cpp").read_text()
    sharded_legacy_receiver = (kernel_base / "reader_mcast_receiver_unary_sharded_gn_v2.cpp").read_text()
    sharded_welford_sender = (kernel_base / "welford_reader_mcast_sender_unary_sharded_gn_v2.cpp").read_text()
    sharded_welford_receiver = (kernel_base / "welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp").read_text()

    senders = (legacy_sender, welford_sender, sharded_legacy_sender, sharded_welford_sender)
    receivers = (legacy_receiver, welford_receiver, sharded_legacy_receiver, sharded_welford_receiver)
    for source in senders + receivers:
        assert "mcast_dest_noc" not in source
        assert "num_mcast_cores_mid_group" not in source

    for source in senders + receivers:
        assert len(re.findall(r"constexpr\s+dataflow_kernel_lib::\s*McastArgs<", source)) == 1
        assert "reduction_mcast_args" in source

    assert legacy_sender.count("reduction_pipe.send_signal();") == 1
    assert legacy_sender.count("reduction_pipe.send(l1_read_addr_ex") == 1
    assert "reduction_pipe.receive_signal();" in legacy_receiver
    assert "reduction_pipe.receive(dfb_ex_global.get_write_ptr(), single_tile_size_bytes);" in legacy_receiver
    assert "reduction_pipe.receive(dfb_ex2_global.get_write_ptr(), single_tile_size_bytes);" in legacy_receiver
    assert "reduce_receiver_sem.up(" in legacy_receiver
    assert welford_sender.count("reduction_pipe.send(global_means_ptr") == 1
    assert "reduction_pipe.receive(global_means_ptr, 2 * single_tile_size_bytes);" in welford_receiver
    assert sharded_legacy_sender.count("reduction_pipe.send(l1_read_addr_ex") == 1
    assert "reduction_pipe.receive(dfb_ex_global.get_write_ptr(), single_tile_size_bytes);" in sharded_legacy_receiver
    assert sharded_welford_sender.count("reduction_pipe.send(global_means_ptr") == 1
    assert "reduction_pipe.receive(global_means_ptr, 2 * single_tile_size_bytes);" in sharded_welford_receiver

    mcast_factory, no_mcast_factory, sharded_factory = factories
    assert "/*pre_handshake=*/use_welford" in mcast_factory
    assert "/*pre_handshake=*/false" in no_mcast_factory
    assert "/*pre_handshake=*/true" in sharded_factory
    for source in factories:
        assert "ttnn::kernel_lib::host::McastFamily" in source
        assert "reduction_family.append_runtime_args_to" in source
        assert "ttnn::kernel_lib::host::Mcast2D" not in source
        assert "mcast_dest_noc" not in source


def test_sdpa_decode_read_k_uses_opaque_fixed_star_and_keeps_blackhole_completion():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device"
    common = (base / "kernels/dataflow/dataflow_common.hpp").read_text()
    reader = (base / "kernels/dataflow/reader_decode_all.cpp").read_text()
    factory = (base / "sdpa_decode_program_factory.cpp").read_text()
    read_k = common[
        common.index("uint32_t read_k(") : common.index("// Non-multicast path", common.index("uint32_t read_k("))
    ]

    assert "async_write_multicast" not in read_k
    assert "SourceL1Guard::CallerManaged" in read_k
    assert "k_pipe.receive();" in read_k
    assert "noc.async_write_barrier();" in read_k
    assert "noc.async_atomic_barrier();" in read_k
    assert "McastArgs<attention_sink_args.next_compile_time_args_offset(), 16 + 2 * num_output_cores>" in reader
    assert "mcast_x" not in reader + factory
    assert "std::vector<ttnn::kernel_lib::host::Mcast1D> k_mcasts" in factory
    assert "Mcast1DShape::PerColumn" in factory
    assert "CoreRangeSet(CoreRange({0, y}, {grid_size.x - 1, y + q_heads_parallel_factor - 1}))" in factory
    assert "core.y / q_heads_parallel_factor" in factory
    assert "core.x * (grid_size.y / q_heads_parallel_factor)" not in factory
    assert "k_mcasts[mcast_index].append_runtime_args_to(reader_rt_args, core);" in factory


def test_argmax_multicore_composes_two_counter_wires_and_keeps_done_fanin():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/reduction/argmax/device"
    kernel = (base / "kernels/reader_argmax_interleaved_multicore.cpp").read_text()
    factory = (base / "argmax_multi_core_program_factory.cpp").read_text()

    assert "McastArgs<operation_compile_time_args_end, 7>" in kernel
    assert kernel.index("TensorAccessorArgs<18>()") < kernel.index("McastArgs<operation_compile_time_args_end, 7>")
    assert "group0_start_args.next_compile_time_args_offset()" in kernel
    assert "group0_start_args.next_runtime_args_offset()" in kernel
    assert kernel.count("send_signal();") == 2
    assert kernel.count("start_receiver->receive_signal();") == 2
    assert "set_multicast" not in kernel
    assert "done_sem.up(" in kernel and "done_sem.wait(num_cores)" in kernel
    assert factory.count("DataReadyMode::Counter") == 1
    assert factory.count("group0_start_mcast.append_runtime_args_to(reader_runtime_args, core);") == 2
    assert factory.count("group1_start_mcast.append_runtime_args_to(reader_runtime_args, core);") == 2


def test_migrated_kernels_keep_fixed_operation_prefixes_before_helper_decoders():
    kernels, _ = _migrated_sources()
    violations = []
    optional_compile_time_tails = {
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp": (
            "config_dram_addr_index = act_mcast_args.next_compile_time_args_offset();"
        ),
    }
    variable_runtime_tails = {
        "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp": "argidx = weights_mcast_args.next_runtime_args_offset();",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_padding.cpp": "rt_args_idx = in0_mcast_args.next_runtime_args_offset();",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp": "operation_rt_args_idx = in0_mcast_args.next_runtime_args_offset();",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_writer_padding.cpp": "rt_args_idx = in1_mcast_args.next_runtime_args_offset();",
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp": "rt_args_idx = in1_mcast_args.next_runtime_args_offset();",
    }

    for path in kernels:
        source = path.read_text()
        if "McastArgs<" not in source:
            continue

        helper = source.index("McastArgs<")
        first_pipe_matches = list(re.finditer(r"\.(?:sender|receiver)\(", source))
        setup_end = first_pipe_matches[0].start() if first_pipe_matches else len(source)
        positional_ct = [match.start() for match in re.finditer(r"get_compile_time_arg_val\(", source[:setup_end])]
        positional_rt = [match.start() for match in re.finditer(r"get_arg_(?:val|addr)\(", source[:setup_end])]

        relative = str(path.relative_to(REPO_ROOT))
        if positional_ct and helper < positional_ct[-1]:
            derived_tail = optional_compile_time_tails.get(relative)
            if derived_tail is None:
                violations.append(f"{relative} declares McastArgs before an unregistered operation CT tail")
            elif derived_tail not in source:
                violations.append(f"{relative} does not derive its optional operation CT tail from McastArgs")
        if positional_rt and helper < positional_rt[-1]:
            derived_tail = variable_runtime_tails.get(relative)
            if derived_tail is None:
                violations.append(f"{relative} declares McastArgs before an unregistered operation RT tail")
            elif derived_tail not in source:
                violations.append(f"{relative} does not derive its variable operation RT tail from McastArgs")

    assert (
        not violations
    ), "Fixed operation prefixes and registered optional CT/variable RT tails must bound helpers:\n" + "\n".join(
        violations
    )


def test_conv_width_mcast_precedes_the_optional_config_tensor_tail():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/conv/conv2d/device"
    kernel = (base / "kernels/activation_reader_width_sharded.cpp").read_text()
    factory = (base / "conv2d_op_width_sharded_program_factory.cpp").read_text()

    assert "McastArgs<operation_ct_args_end, 3>" in kernel
    assert "config_dram_addr_index = act_mcast_args.next_compile_time_args_offset();" in kernel
    assert "TensorAccessorArgs<23>" not in kernel
    helper_append = factory.index("activation_mcast.append_compile_time_args_to(activation_kernel_compile_args);")
    optional_tail = factory.index("if (config_tensors_in_dram) {", helper_append)
    assert helper_append < optional_tail
    assert "TensorAccessorArgs(static_cast<const Buffer*>(nullptr))" not in factory


def test_migrated_conv_weight_kernels_preserve_terminal_write_barriers():
    kernel_dir = REPO_ROOT / "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels"
    kernels = [
        "reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp",
        "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp",
        "writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp",
        "writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp",
    ]

    for kernel in kernels:
        source = (kernel_dir / kernel).read_text()
        closing_barrier = source.rindex("noc.async_write_barrier();")
        assert source[closing_barrier:].strip() == "noc.async_write_barrier();\n}"


def test_conv_weight_helpers_do_not_duplicate_or_ambiguously_name_sender_roles():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/conv/conv2d/device"
    kernels = [
        (base / "kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp").read_text(),
        (base / "kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp").read_text(),
    ]
    factory = (base / "conv2d_op_sharded_program_factory.cpp").read_text()

    for source in kernels:
        assert "is_sender_core" not in source
        assert "has_sharded_input" in source
    assert factory.count("const bool has_sharded_input = input_cores.contains(core);") == 2


def test_conv_weight_sender_preserves_original_source_lifetime_policy():
    kernel = (
        REPO_ROOT / "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
        "reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp"
    ).read_text()

    assert "weight_sources_are_persistent" not in kernel
    assert "async_writes_flushed" not in kernel
    assert kernel.count("SourceL1Guard::CallerManaged") == 2
    closing_barrier = kernel.rindex("noc.async_write_barrier();")
    assert kernel[closing_barrier:].strip() == "noc.async_write_barrier();\n}"


def test_conv_ack_counts_are_geometry_derived_only_for_dense_families():
    conv2d_factory = (
        REPO_ROOT / "ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp"
    ).read_text()
    width_factory = (
        REPO_ROOT / "ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_width_sharded_program_factory.cpp"
    ).read_text()
    conv3d_factory = (
        REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp"
    ).read_text()

    # Dense block-sharded weights use Mcast1D's span-minus-sender default.
    block_start = conv2d_factory.index("weights_mcast_1d.emplace(")
    block_end = conv2d_factory.index("    }", block_start)
    block_binding = conv2d_factory[block_start:block_end]
    assert "weights_mcast_sender_semaphore_id}});" in block_binding

    # Dense Conv3D rectangles make otherwise-idle members passive participants,
    # so both the template and per-group Mcast2D bindings use area-minus-sender.
    assert "CoreCoord{0, 0}, weights_mcast_config);" in conv3d_factory
    assert "device, group_rect, CoreCoord{sender_x_log, sender_y_log}, weights_mcast_config);" in conv3d_factory

    # These two geometries deliberately have fewer acknowledging participants
    # than landing cores and must retain their explicit operation-owned counts.
    assert "total_active_num_cores - 1);" in conv2d_factory
    assert "std::max(input_num_cores, output_num_cores) - 1);" in width_factory

    # The raw block-sharded activation family remains deferred; do not silently
    # remove its geometric scalar before that family is migrated as its own unit.
    assert "act_mcast_num_dests" in conv2d_factory


def test_move_overlap_composes_three_release_wires_and_keeps_return_counter():
    base = REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/move/device"
    kernels = [
        (base / "kernels/dataflow/move_interleaved_with_overlap.cpp").read_text(),
        (base / "kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp").read_text(),
    ]
    factory = (base / "move_overlap_program_factory.cpp").read_text()
    cache_override = (base / "move_sharded_program_factory.cpp").read_text()

    for kernel in kernels:
        assert kernel.count("McastArgs<") == 3
        assert "using Release" not in kernel
        assert "release0_args.next_compile_time_args_offset()" in kernel
        assert "release1_args.next_compile_time_args_offset()" in kernel
        assert kernel.count("send_signal();") == 3
        assert kernel.count("receive_signal();") == 3
        assert "return_sem.up(" in kernel and "return_sem.wait(num_workers)" in kernel
        assert "set_multicast" not in kernel

    assert "std::vector<ttnn::kernel_lib::host::Mcast2D> release_mcasts" in factory
    assert factory.count("mcast.append_compile_time_args_to(compile_time_args);") == 1
    assert factory.count("mcast.append_runtime_args_to(runtime_args, core);") == 1
    assert "mcast_dest_noc" not in factory
    assert "case MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP" in cache_override
    assert "a[0] = src_addr;" in cache_override and "a[1] = dst_addr;" in cache_override
