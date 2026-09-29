"""Hold real layer-2 TT Q/K/V fixed to isolate SDPA recurrence/fidelity."""

import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import K2Generator

DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")


def metric(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return {
        "pcc": float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
        "relative_l2": float((a - b).norm() / a.norm()),
        "max_abs": float((a - b).abs().max()),
    }


def main():
    torch.set_num_threads(4)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(mesh, override_num_layers=3)
        gen._ensure_owned_cache(1, 384)
        gen.reset()
        original = ttnn.transformer.chunked_scaled_dot_product_attention
        calls = 0
        result = {}

        def read(x):
            return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).double()

        def inspected(q, k, v, table, **kw):
            nonlocal calls
            calls += 1
            output = original(q, k, v, table, **kw)
            if calls != 3:
                return output
            query = read(q)
            keys = read(k)[:9].permute(1, 0, 2, 3).reshape(1, 8, 288, 128).repeat_interleave(4, dim=1)
            values = read(v)[:9].permute(1, 0, 2, 3).reshape(1, 8, 288, 128).repeat_interleave(4, dim=1)
            scores = query @ keys.transpose(-1, -2) / (128**0.5)
            mask = torch.arange(288)[None, :] > torch.arange(288)[:, None]
            scores.masked_fill_(mask, -torch.inf)
            probs = scores.softmax(-1)
            reference = probs @ values
            result["reference_bos_attention_mass_last"] = probs[0, :, 256, ::13].sum(-1).tolist()
            outputs = {"LoFi_full_K256": read(output)}
            for fidelity in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi4):
                compute = ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity=fidelity,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
                for chunk in (32, 256):
                    if fidelity == ttnn.MathFidelity.LoFi and chunk == 256:
                        continue
                    out = original(
                        q,
                        k,
                        v,
                        table,
                        chunk_start_idx=0,
                        compute_kernel_config=compute,
                        program_config=ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=(11, 10),
                            q_chunk_size=32,
                            k_chunk_size=chunk,
                            exp_approx_mode=False,
                        ),
                    )
                    outputs[f"{fidelity.name}_full_K{chunk}"] = read(out)
                out = original(
                    q[:, :, 32:, :],
                    k,
                    v,
                    table,
                    chunk_start_idx=32,
                    compute_kernel_config=compute,
                    program_config=ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=(11, 10), q_chunk_size=32, k_chunk_size=32, exp_approx_mode=False
                    ),
                )
                outputs[f"{fidelity.name}_suffix_K32"] = read(out)
            for name, out in outputs.items():
                offset = 32 if "suffix" in name else 0
                ref = reference[:, :, offset:257, :]
                actual = out[:, :, : 257 - offset, :]
                result[name] = {
                    "all": metric(ref, actual),
                    "last": metric(ref[:, :, -1], actual[:, :, -1]),
                    "last_per_head": [metric(ref[:, h, -1], actual[:, h, -1]) for h in range(32)],
                }
            path = DOC / "attention_fixed_input.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "last_per_head"}
                        for k, v in result.items()
                    },
                    indent=2,
                ),
                flush=True,
            )
            return output

        ids = (gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. ") * 30)[:257]
        with patch.object(ttnn.transformer, "chunked_scaled_dot_product_attention", side_effect=inspected):
            gen.prefill_forward(
                torch.tensor([ids]),
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=[257],
                sampling_mode="host",
            )
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
