# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.chip_architecture import ChipArchitecture
from helpers.constraints import (
    is_valid_quasar_packer_conversion,
    is_valid_quasar_unpack_to_dest,
    quasar_sfpu_hardware_format,
)
from helpers.data_format_inference import (
    infer_math_format,
    infer_pack_in,
    infer_unpack_out,
    is_format_combination_outlier,
    validate_quasar_data_formats,
)
from helpers.format_config import ARCH_DATA_FORMAT_ENUM_VALUES, DataFormat
from helpers.llk_params import EltwiseBinaryReuseDestType

from .fpu_node import FpuNode
from .operand import L1_PACKERS
from .pack_node import PackNode
from .sfpu_node import SfpuNode


def supported_operand_formats(architecture):
    return set(ARCH_DATA_FORMAT_ENUM_VALUES[architecture]) & L1_PACKERS.keys()


def _hardware_format(fmt, config):
    if config.architecture == ChipArchitecture.QUASAR:
        return quasar_sfpu_hardware_format(fmt)
    return fmt


def _dest_format(fmt, config):
    if config.dest_acc.value:
        return DataFormat.Int32 if fmt.is_integer() else DataFormat.Float32
    if fmt.is_32_bit():
        raise ValueError(f"{fmt.name} in dest requires dest_acc: true")
    if fmt.is_block_float():
        return DataFormat.Float16_b if fmt.is_exponent_B() else DataFormat.Float16
    return fmt


def _single_format(formats):
    if len(formats) != 1:
        raise ValueError(f"mixed dest formats {sorted(f.name for f in formats)}")
    return next(iter(formats))


def _infer_fpu(node, sources, output, config):
    if node.src_a is None or node.reuse_dest != EltwiseBinaryReuseDestType.NONE:
        dest = _single_format({source.dest_format for source in sources})
        reused = infer_unpack_out(
            _hardware_format(dest, config), output, config.dest_acc
        )
    if node.src_a is None:
        # No unpack: SrcA/SrcB are filled from dest in its source-register format.
        node.dest_format = dest
        node.unpack_A_dst = node.unpack_B_dst = reused
        return

    a = _hardware_format(node.src_a.data_format, config)
    b = _hardware_format(node.src_b.data_format, config) if node.src_b else a
    if node.reuse_dest != EltwiseBinaryReuseDestType.NONE:
        a, b = (
            (reused, a)
            if node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA
            else (a, reused)
        )

    direct = node.unpack_to_dest.value
    unpack_a = infer_unpack_out(a, output, config.dest_acc, direct)
    unpack_b = infer_unpack_out(b, output, config.dest_acc, direct)
    math = infer_math_format(unpack_a, unpack_b)
    if not direct and (unpack_a.is_32_bit() or unpack_b.is_32_bit()):
        raise ValueError("32-bit integer inputs require unpack_to_dest: true")
    dest = _dest_format(math, config)
    if config.architecture == ChipArchitecture.QUASAR:
        validate_quasar_data_formats(
            unpack_a, unpack_b, _hardware_format(dest, config), direct
        )
        if (
            direct
            and node.src_a.data_format == DataFormat.UInt16
            and not config.dest_acc.value
        ):
            dest = DataFormat.UInt16
        if direct and not is_valid_quasar_unpack_to_dest(a, unpack_a, config.dest_acc):
            raise ValueError(f"Quasar cannot unpack {a.name} to {unpack_a.name} dest")
    node.dest_format = dest
    node.unpack_A_src, node.unpack_B_src = node.unpacker.physical_order(a, b)
    node.unpack_A_dst, node.unpack_B_dst = node.unpacker.physical_order(
        unpack_a, unpack_b
    )


def _infer_sfpu(node, sources, output, config):
    # Sfpu Fill may initialize dest without an earlier producer.
    fill = _dest_format(
        infer_math_format(infer_unpack_out(output, output, config.dest_acc, True)),
        config,
    )
    formats = {fill if source is node else source.dest_format for source in sources}
    node.dest_format = dest = _single_format(formats)
    if not node.sfpu.supports_dest_format(dest):
        raise ValueError(f"{node.sfpu.operation.name} cannot use dest {dest.name}")


def _quasar_integer_overflows(values: DataFormat, target: DataFormat) -> bool:
    return target.size < values.size or {values, target} == {
        DataFormat.Int16,
        DataFormat.UInt16,
    }


def _infer_pack(node, sources, config):
    target = node.output.data_format
    output = _hardware_format(target, config)
    formats = set()
    for source in sources:
        if isinstance(source, FpuNode):
            math = source.src_format
        else:
            math = _hardware_format(source.dest_format, config)
        if math.is_integer() != output.is_integer():
            raise ValueError(f"Packing {math.name} to {output.name} needs a typecast")
        values = source.dest_format
        if isinstance(source, FpuNode) and source.src_a is not None:
            values = source.src_a.data_format
        if (
            config.architecture == ChipArchitecture.QUASAR
            and output.is_integer()
            and _quasar_integer_overflows(values, target)
        ):
            raise ValueError(f"Quasar packer cannot fit {values.name} in {target.name}")
        if (
            config.architecture != ChipArchitecture.QUASAR
            and is_format_combination_outlier(math, output, config.dest_acc)
        ):
            raise ValueError(f"{math.name} -> {output.name} requires dest_acc: true")
        direct = not isinstance(source, FpuNode) or source.unpack_to_dest.value
        pack_src = infer_pack_in(
            math, output, math, config.dest_acc, direct, config.architecture
        )
        if output == DataFormat.Fp8_e4m3:
            pack_src = (
                DataFormat.Float16_b if math.is_exponent_B() else DataFormat.Float16
            )
        if (
            config.architecture == ChipArchitecture.QUASAR
            and not is_valid_quasar_packer_conversion(pack_src, output)
        ):
            raise ValueError(f"Quasar cannot pack {pack_src.name} to {output.name}")
        formats.add(pack_src)
    node.pack_src_format = _single_format(formats)
    node.pack_dst_format = output


def _node_label(section, index, node):
    if section != "Math node":
        return f"{section} {index}"
    kind = "Fpu" if isinstance(node, FpuNode) else type(node.sfpu).__name__
    return f"{section} {index} ({kind})"


def _infer_nodes(nodes, section, sources, output, config):
    for index, node in enumerate(nodes, 1):
        try:
            producers = sources.get(node, set())
            if isinstance(node, FpuNode):
                _infer_fpu(node, producers, output, config)
            elif isinstance(node, SfpuNode):
                _infer_sfpu(node, producers, output, config)
            elif isinstance(node, PackNode):
                _infer_pack(node, producers, config)
        except ValueError as error:
            raise ValueError(f"{_node_label(section, index, node)}\n    {error}")


def infer_operation_formats(operation, config, blocks):
    sources = {}
    for block in blocks:
        for node, producers in block.dest_sources.items():
            sources.setdefault(node, set()).update(producers)
    output = operation._get_pack_nodes()[0].output.data_format
    _infer_nodes(operation.math_nodes, "Math node", sources, output, config)
    _infer_nodes(operation.pack_nodes, "Pack entry", sources, output, config)


def validate_pipeline_formats(pipeline, config):
    if config.dest_acc.value:
        return
    pack_sfpu = [
        node for op in pipeline for node in op.pack_nodes if isinstance(node, SfpuNode)
    ]
    if not pack_sfpu:
        return
    formats = {node.dest_format for node in pack_sfpu} | {
        node.dest_format
        for op in pipeline
        for node in op.math_nodes
        if isinstance(node, FpuNode) and node.src_a is not None
    }
    if len(formats) > 1:
        raise ValueError("pack-thread SFPU with a 16-bit dest needs one math format")
