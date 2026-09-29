"""AutoFix discriminator: long-prefix attention without a complete prefill run."""

import argparse
import json

import torch

import ttnn

from .run_functional import DOC, pcc, to_device


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=524288)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--accurate", action="store_true")
    parser.add_argument("--uniform", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(674)
    n = args.context
    q = torch.randn(1, 32, 128, 128).bfloat16() * 0.5
    k = torch.randn(1, 8, n, 128).bfloat16() * 0.5
    v = torch.randn(1, 8, n, 128).bfloat16() * 0.5
    if args.uniform:
        q.zero_()
        k.zero_()
        v.zero_()
        v[:, :, :128, :] = 1
    mask = torch.where(
        torch.arange(n)[None, :] <= torch.arange(n - 128, n)[:, None], 0.0, torch.finfo(torch.bfloat16).min
    ).bfloat16()[None, None]
    reference = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, -32:, :], k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), attn_mask=mask[:, :, -32:, :]
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    results = []
    try:
        tq = to_device(q, mesh)
        tk = to_device(k.reshape(8, n // 32, 32, 128).permute(1, 0, 2, 3).contiguous(), mesh)
        tv = to_device(v.reshape(8, n // 32, 32, 128).permute(1, 0, 2, 3).contiguous(), mesh)
        pt = to_device(torch.arange(n // 32, dtype=torch.int32)[None], mesh, True)
        offset = to_device(torch.tensor([n - 128], dtype=torch.int32), mesh, True)
        for fp32, kchunk, flex, qchunk, packer in [
            (True, 128, True, 32, True),
            (True, 512, True, 32, True),
            (True, 1024, True, 32, True),
        ]:
            cfg = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=packer,
            )
            kwargs = {"chunk_start_idx_tensor": offset} if flex else {"chunk_start_idx": n - 128}
            try:
                if args.accurate:
                    from ..tt.accurate_attention import accurate_attention

                    out = accurate_attention(
                        tq, tk, tv, pt, chunk_start_idx_tensor=offset, q_chunk_size=qchunk, k_chunk_size=kchunk
                    )
                    actual = ttnn.to_torch(out)[:, :, -32:, :]
                    ref = reference
                elif args.decode:
                    dq = to_device(q[:, :, -1:, :].transpose(1, 2).contiguous(), mesh)
                    pos = to_device(torch.tensor([n - 1], dtype=torch.int32), mesh, True)
                    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        dq,
                        tk,
                        tv,
                        page_table_tensor=pt,
                        cur_pos_tensor=pos,
                        compute_kernel_config=cfg,
                        program_config=ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=(8, 8),
                            q_chunk_size=32,
                            k_chunk_size=kchunk,
                            exp_approx_mode=False,
                        ),
                    )
                    actual = ttnn.to_torch(out).transpose(1, 2)
                    ref = reference[:, :, -1:, :]
                else:
                    out = ttnn.transformer.chunked_scaled_dot_product_attention(
                        tq,
                        tk,
                        tv,
                        pt,
                        compute_kernel_config=cfg,
                        program_config=ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=(8, 8),
                            q_chunk_size=qchunk,
                            k_chunk_size=kchunk,
                            exp_approx_mode=False,
                        ),
                        **kwargs,
                    )
                    actual = ttnn.to_torch(out)[:, :, -32:, :]
                    ref = reference
                row = {
                    "fp32_dest": fp32,
                    "k_chunk_size": kchunk,
                    "flexible": flex,
                    "q_chunk_size": qchunk,
                    "packer_l1_acc": packer,
                    "pcc": None if args.uniform else pcc(actual, ref),
                    "relative_l2": ((actual.float() - ref.float()).norm() / ref.float().norm()).item(),
                }
                out.deallocate(True)
            except Exception as e:
                row = {
                    "fp32_dest": fp32,
                    "k_chunk_size": kchunk,
                    "flexible": flex,
                    "q_chunk_size": qchunk,
                    "packer_l1_acc": packer,
                    "error": str(e).split("backtrace:")[0],
                }
            print(json.dumps(row), flush=True)
            results.append(row)
    finally:
        ttnn.close_mesh_device(mesh)
        (
            DOC
            / (
                "debug_long/attention_probe_"
                + ("accurate_" if args.accurate else "")
                + str(n)
                + "_"
                + ("decode" if args.decode else "prefill")
                + ("_uniform" if args.uniform else "")
                + ".json"
            )
        ).write_text(json.dumps({"context": n, "results": results}, indent=2) + "\n")

    if args.accurate:
        assert all(
            "error" not in row and row["relative_l2"] < 0.01 and (args.uniform or row["pcc"] >= 0.995)
            for row in results
        ), results


if __name__ == "__main__":
    main()
