# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integrate one DRAM-reader role with production precision and common padding."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_dram_readers import (
    COMPUTE_FIELDS,
    RUNTIME,
    digest,
    geometry,
    write_json,
)


def compute_description(config):
    return {name: str(getattr(config, name)) for name in COMPUTE_FIELDS}


def setup_qkv(decoder, mesh, readers, block, ttnn, torch):
    projection = decoder.layer.self_attn.source.weights.wqkv
    while hasattr(projection, "decode_source"):
        projection = projection.decode_source
    if not projection.dram or len(projection.weights) != 1:
        raise ValueError("Expected the factory-built packed direct DRAM projection")
    old = projection.weights[0]
    k = projection.source.weight.shape[-2]
    n = projection.source.weight.shape[-1]
    banks = mesh.dram_grid_size()
    geo = geometry(k, n, block, banks.x * banks.y, mesh.compute_with_storage_grid_size().x, "bfloat8_b")
    if old.dtype != ttnn.bfloat8_b or projection.input_dtype != ttnn.float32:
        raise ValueError("QKV production precision changed")
    values = ttnn.to_torch(old).float()[..., :n]
    values = torch.nn.functional.pad(values, (0, geo["physical_MKN"][-1] - n))
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks.x - 1, banks.y - 1))}),
            geo["weight_shard_shape"],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    weight = ttnn.from_torch(values, device=mesh, dtype=old.dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory)
    if not torch.equal(ttnn.to_torch(weight).float(), values):
        raise AssertionError("Common-padding QKV repack changed quantized values")
    program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=block,
        per_core_M=1,
        per_core_N=geo["physical_MKN"][-1] // 32 // geo["input_storage_cores"],
        num_workers_per_dram_bank=readers,
    )
    projection.weights = projection.prefill_source.weights = (weight,)
    projection.programs = projection.prefill_source.programs = (program,)
    projection.precision_policy.update(
        weight_shapes=[list(weight.shape)],
        weight_memory=[str(weight.memory_config())],
        programs=[
            dict(in0_block_w=block, per_core_M=1, per_core_N=program.per_core_N, num_workers_per_dram_bank=readers)
        ],
    )
    decoder.precision_policy["qkv_dram_geometry"] = geo
    return dict(
        geometry=geo,
        program=str(program),
        compute=compute_description(projection.compute),
        input_dtype=str(projection.input_dtype),
        weight_dtype=str(weight.dtype),
        prefill="Original minimal prefill delegate and packed source weight unchanged",
    )


def setup_output(decoder, mesh, readers, ttnn):
    from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_output import OutputProjectionProbe

    attention = decoder.layer.self_attn
    native_compute = attention.output_compute
    args = argparse.Namespace(
        output_mode="dram",
        output_memory="l1",
        output_input_dtype="bfloat16",
        output_k_block=16,
        output_fidelity=str(native_compute.math_fidelity).split(".")[-1],
        output_weight_dtype=None,
        # Existing probe's three-reader alignment gives common N3072. Change
        # only the program reader count after the single setup-time repack.
        output_readers=3,
        output_input_cores=8,
        output_grid=(8, 8),
        output_subblock=0,
    )
    metadata = {}
    probe = OutputProjectionProbe(attention, mesh, args, metadata)
    if probe.physical_n != 3072 or probe.width != 2816:
        raise ValueError("Unexpected output width: this controlled experiment requires logical2816/padded3072")
    probe.compute = native_compute
    probe.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=16,
        per_core_M=1,
        per_core_N=probe.program.per_core_N,
        num_workers_per_dram_bank=readers,
    )
    args.output_readers = readers
    metadata.update(program=str(probe.program), compute=compute_description(native_compute), common_padding=True)
    metadata["geometry"]["readers_per_bank"] = readers
    metadata["geometry"]["in1_triple_buffer_bytes_per_reader"] = 12 // readers * 16 * 3 * 1088
    attention.project = probe
    return metadata


def setup_shared(decoder, role, readers, ttnn):
    shared = decoder.layer.shared_mlp
    target = shared.gate_up if role == "shared_gate_up" else shared.down
    old = target.program
    if not target.dram or old.in0_block_w != 11:
        raise ValueError("Expected the selected K11 DRAM shared projection")
    target.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=old.in0_block_w,
        per_core_M=old.per_core_M,
        per_core_N=old.per_core_N,
        num_workers_per_dram_bank=readers,
    )
    return {
        name: dict(
            readers=getattr(shared, name).program.num_workers_per_dram_bank,
            program=str(getattr(shared, name).program),
            compute=compute_description(getattr(shared, name).compute),
            weight_dtype=str(getattr(shared, name).weight.dtype),
            weight_shape=list(getattr(shared, name).weight.shape),
        )
        for name in ("gate_up", "down")
    }


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--reader-role", choices=("baseline", "qkv", "output", "shared_gate_up", "shared_down"), required=True
    )
    parser.add_argument("--readers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--reader-qkv-block", type=int, choices=(1, 11), default=11)
    args, rest = parser.parse_known_args()
    if "--defaults" not in rest:
        parser.error("This experiment requires --defaults to preserve the selected cumulative precision policy")
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(output)
    import torch

    import ttnn
    from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    torch.set_num_threads(4)
    factory = OptimizedDecoder.from_state_dict.__func__
    source_hash = digest(RUNTIME)
    metadata = {}

    def build(cls, *a, **kw):
        if args.reader_role == "qkv":
            kw.update(qkv_dram=True, qkv_dram_readers=args.readers, qkv_dram_block=args.reader_qkv_block)
        decoder = factory(cls, *a, **kw)
        if args.reader_role == "qkv":
            metadata.update(setup_qkv(decoder, kw["mesh_device"], args.readers, args.reader_qkv_block, ttnn, torch))
        elif args.reader_role == "output":
            metadata.update(setup_output(decoder, kw["mesh_device"], args.readers, ttnn))
        elif args.reader_role.startswith("shared_"):
            metadata.update(setup_shared(decoder, args.reader_role, args.readers, ttnn))
        metadata["prefill_qkv_policy"] = decoder.precision_policy.get("prefill_qkv_projection")
        metadata["prefill_output_policy"] = decoder.precision_policy.get("prefill_output_projection")
        return decoder

    failure = None
    try:
        with (
            patch.object(OptimizedDecoder, "from_state_dict", classmethod(build)),
            patch.object(torch, "set_num_threads", lambda _: None),
            patch.object(sys, "argv", [sys.argv[0], *rest]),
        ):
            run_optimized_decoder.main()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report.update(
            reader_layer_candidate=vars(args),
            reader_layer_runtime=metadata,
            reader_layer_probe_sha256=digest(__file__),
            reader_layer_runtime_sha256=source_hash,
        )
        if failure:
            report["reader_layer_error"] = failure
        write_json(output, report)
        if digest(RUNTIME) != source_hash:
            raise RuntimeError("Runtime changed during whole-layer reader control")


if __name__ == "__main__":
    main()
