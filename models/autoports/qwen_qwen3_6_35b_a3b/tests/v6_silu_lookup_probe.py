# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-operand BF16-domain SiLU lookup; setup table is independent of the fixture."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_gated_norm_same_operand_probe import compare

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"


def device_lookup(ttnn, z, table):
    indices = ttnn.bitcast(z, ttnn.uint16)
    indices = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, 4096))
    values = ttnn.gather(table, 1, indices)
    return ttnn.to_layout(ttnn.reshape(values, (1, 32, 128)), ttnn.TILE_LAYOUT)


def run(args):
    assert not args.output.exists()
    torch.set_num_threads(4)
    capture = torch.load(E / "v6_prefix_control_stream_probe.json.pos3271.pt", weights_only=False, map_location="cpu")
    values = torch.load(E / "v6_gated_norm_3271_device.json.tensors.pt", weights_only=False, map_location="cpu")
    expected = values["3271"]["cpu_reference"]
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    lut = torch.nn.functional.silu(codes.float()).reshape(1, 65536)
    z = capture["tt_values"]["projection_z"].reshape(1, 32, 128).contiguous()
    torch_lookup = lut.flatten()[z.view(torch.int16).long().bitwise_and(65535)]
    assert torch.equal(torch_lookup, expected["silu_standalone_fp32"])
    result = dict(
        classification="same-operand component diagnostic, not model acceptance",
        status="started",
        device_closed=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        table_sha256=hashlib.sha256(lut.view(torch.uint8).numpy().tobytes()).hexdigest(),
        table_bytes=lut.numel() * lut.element_size(),
        table_contract="CPU SiLU evaluated once for all 65536 BF16 bit patterns during setup; no fixture-specific entries or host runtime computation",
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    record(args.output)
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)

        def upload(x, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x, device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        table = upload(lut, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
        dz = upload(z, ttnn.bfloat16)
        weighted = upload(expected["weighted_fp32_carrier"], ttnn.float32)
        with device_only():
            silu = device_lookup(ttnn, dz, table)
            gated = ttnn.multiply(
                weighted,
                silu,
                dtype=ttnn.bfloat16,
                fast_and_approximate_mode=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        hs, hg = ttnn.to_torch(silu), ttnn.to_torch(gated)
        result.update(silu=compare(expected["silu_standalone_fp32"], hs), gated=compare(expected["gated_bf16"], hg))
        assert torch.equal(hs, expected["silu_standalone_fp32"])
        assert torch.equal(hg, expected["gated_bf16"])
        result["status"] = "component_pass"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()
    print(result["status"], result["silu"], result["gated"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=E / "v6_silu_lookup_device.json")
    run(parser.parse_args())
