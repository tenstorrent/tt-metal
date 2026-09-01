# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device contract probes for the fused-decoder graph rewrites.

Each probe checks one op contract the fused decoder wants to rely on, with a
tiny numeric check against torch. Run:

    python models/autoports/zai_org_glm_4_7_flash/probe/fusing_contract_probe.py
"""

import torch

import ttnn


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.std() == 0 or b.std() == 0:
        return float(torch.equal(a, b))
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    results = {}
    try:
        # ---------------------------------------------------------------- 1
        # matmul fused activation "silu"
        x = torch.randn(1, 1, 32, 2048)
        w = torch.randn(2048, 1536) * 0.02
        xt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        wt = ttnn.from_torch(w, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        try:
            got = ttnn.to_torch(ttnn.linear(xt, wt, activation="silu")).float()
            ref = torch.nn.functional.silu(x.to(torch.bfloat16).float() @ w.to(torch.bfloat16).float())
            results["matmul_silu"] = f"PCC={pcc(ref, got):.6f}"
        except Exception as e:
            results["matmul_silu"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

        # ---------------------------------------------------------------- 2
        # binary multiply with lhs silu activation
        a = torch.randn(1, 4, 32, 1536)
        b = torch.randn(1, 4, 32, 1536)
        at = ttnn.from_torch(a, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        bt = ttnn.from_torch(b, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        for arg in (["silu"], [ttnn.UnaryOpType.SILU]):
            try:
                got = ttnn.to_torch(ttnn.multiply(at, bt, input_tensor_a_activations=arg)).float()
                ref = torch.nn.functional.silu(a.to(torch.bfloat16).float()) * b.to(torch.bfloat16).float()
                results[f"mul_lhs_silu[{arg}]"] = f"PCC={pcc(ref, got):.6f}"
                break
            except Exception as e:
                results[f"mul_lhs_silu[{arg}]"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

        # ---------------------------------------------------------------- 3
        # topk index dtype + indexed sparse matmul (gate-like and down-like)
        E, H, I, k = 64, 2048, 1536, 4
        scores = torch.randn(1, 1, 1, E)
        st = ttnn.from_torch(scores, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        vals, idx = ttnn.topk(st, k=k, dim=-1, sorted=True)
        results["topk_idx_dtype"] = f"{idx.dtype}, shape={tuple(idx.shape)}, layout={idx.layout}"
        # convert topk idx -> single-stick RM uint16 [1,1,1,k]
        try:
            idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
            if idx_rm.dtype != ttnn.uint16:
                idx_rm = ttnn.typecast(idx_rm, ttnn.uint16)
            results["topk_idx_to_rm"] = f"OK shape={tuple(idx_rm.shape)} dtype={idx_rm.dtype}"
        except Exception as e:
            results["topk_idx_to_rm"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
            idx_rm = None

        xg = torch.randn(1, 1, 32, H)
        wg = torch.randn(1, E, H, I) * 0.02
        xgt = ttnn.from_torch(xg, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        wgt = ttnn.from_torch(wg, device=dev, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ones = ttnn.from_torch(torch.ones(1, 1, 1, E), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        def sparse_pc(m, n, kdim, cores=(8, 4), in0_block_w=8):
            core_x, core_y = cores
            num_cores = core_x * core_y
            Nt = -(-n // 32)
            per_core_N = -(-Nt // num_cores)
            Kt = -(-kdim // 32)
            if Kt % in0_block_w != 0:
                divisors = [d for d in range(2, in0_block_w + 1) if Kt % d == 0]
                in0_block_w = max(divisors) if divisors else Kt
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=max(32, m) // 32,
                per_core_N=per_core_N,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )

        if idx_rm is not None:
            try:
                gate = ttnn.sparse_matmul(
                    xgt,
                    wgt,
                    sparsity=ones,
                    indices=idx_rm,
                    is_input_b_sparse=True,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=sparse_pc(32, I, H),
                    dtype=ttnn.bfloat16,
                )
                results["indexed_gate_shape"] = f"{tuple(gate.shape)}"
                got = ttnn.to_torch(gate).float().reshape(k, 32, I)
                idx_host = ttnn.to_torch(idx_rm).int().flatten().tolist()
                wg8 = ttnn.to_torch(wgt).float()
                ok = all(pcc(xg[0, 0].float() @ wg8[0, e], got[i]) > 0.98 for i, e in enumerate(idx_host))
                results["indexed_gate_pcc"] = f"{'OK' if ok else 'MISMATCH'} ids={idx_host}"

                # down-like: compact A + indexed B
                wd = torch.randn(1, E, I, H) * 0.02
                wdt = ttnn.from_torch(wd, device=dev, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
                a_compact = ttnn.reshape(gate, (1, k, 32, I))
                down = ttnn.sparse_matmul(
                    a_compact,
                    wdt,
                    sparsity=ones,
                    indices=idx_rm,
                    is_input_a_sparse=True,
                    is_input_b_sparse=True,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=sparse_pc(32, H, I),
                    dtype=ttnn.bfloat16,
                )
                results["indexed_down_shape"] = f"{tuple(down.shape)}"
                gotd = ttnn.to_torch(down).float().reshape(k, 32, H)
                wd8 = ttnn.to_torch(wdt).float()
                gotg = ttnn.to_torch(a_compact).float()
                ok = all(pcc(gotg[0, i] @ wd8[0, e], gotd[i]) > 0.98 for i, e in enumerate(idx_host))
                results["indexed_down_pcc"] = "OK" if ok else "MISMATCH"
            except Exception as e:
                results["indexed_sparse_mm"] = f"FAIL: {type(e).__name__}: {str(e)[:300]}"

        # ---------------------------------------------------------------- 4
        # gather of fp32 scores by topk indices
        try:
            sf = torch.randn(1, 1, 1, E)
            sft = ttnn.from_torch(sf, device=dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            g = ttnn.gather(sft, dim=3, index=idx)
            goth = ttnn.to_torch(g).float()
            idx_host = ttnn.to_torch(idx).long()
            ref = torch.gather(sf, 3, idx_host)
            results["gather_scores"] = f"PCC={pcc(ref, goth):.6f} shape={tuple(g.shape)}"
        except Exception as e:
            results["gather_scores"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

        # ---------------------------------------------------------------- 5
        # batched wq_b: in0 [1,1,S,768] x in1 [1,20,768,256] broadcast over batch
        S, nh, qlr, hd = 128, 20, 768, 256
        q = torch.randn(1, 1, S, qlr)
        wq = torch.randn(1, nh, qlr, hd) * 0.02
        qt = ttnn.from_torch(q, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        wqt = ttnn.from_torch(wq, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        try:
            o = ttnn.matmul(qt, wqt)
            got = ttnn.to_torch(o).float()
            ref = q.to(torch.bfloat16).float() @ wq.to(torch.bfloat16).float()
            results["batched_wq_b"] = f"PCC={pcc(ref, got):.6f} shape={tuple(o.shape)}"
        except Exception as e:
            results["batched_wq_b"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

        # ---------------------------------------------------------------- 6
        # concatenate_heads prefill: [1, nh, S, d] -> [1, S, nh*d]
        try:
            v = torch.randn(1, nh, S, hd)
            vt = ttnn.from_torch(v, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            c = ttnn.transformer.concatenate_heads(vt)
            got = ttnn.to_torch(c).float()
            ref = v.permute(0, 2, 1, 3).reshape(1, S, nh * hd).to(torch.bfloat16).float()
            results["concatenate_heads"] = f"PCC={pcc(ref, got):.6f} shape={tuple(c.shape)}"
        except Exception as e:
            results["concatenate_heads"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

        # ---------------------------------------------------------------- 7
        # nlp_concat_heads_decode: [1, B, nh(padded 32), d] height-sharded -> [1,1,B,nh*d]
        try:
            B = 1
            vd = torch.randn(1, B, 32, hd)  # padded heads
            vdt = ttnn.from_torch(vd, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            grid = dev.compute_with_storage_grid_size()
            shard = ttnn.create_sharded_memory_config(
                shape=(32, hd),
                core_grid=ttnn.num_cores_to_corerangeset(B, grid, row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            vds = ttnn.to_memory_config(vdt, shard)
            c = ttnn.experimental.nlp_concat_heads_decode(vds, num_heads=nh)
            got = ttnn.to_torch(c).float()
            ref = vd[:, :, :nh, :].reshape(1, 1, B, nh * hd).to(torch.bfloat16).float()
            results["nlp_concat_heads_decode"] = f"PCC={pcc(ref, got[:, :, :B]):.6f} shape={tuple(c.shape)}"
        except Exception as e:
            results["nlp_concat_heads_decode"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

        # ---------------------------------------------------------------- 8
        # slice_write: chunk into preallocated DRAM tensor
        try:
            out = ttnn.zeros((1, 1, 128, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            ch = torch.randn(1, 1, 64, 64)
            cht = ttnn.from_torch(ch, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.experimental.slice_write(cht, out, [0, 0, 64, 0], [1, 1, 128, 64], [1, 1, 1, 1])
            got = ttnn.to_torch(out).float()
            ok = pcc(ch.to(torch.bfloat16).float(), got[:, :, 64:]) > 0.9999 and got[:, :, :64].abs().max() == 0
            results["slice_write"] = "OK" if ok else "MISMATCH"
        except Exception as e:
            results["slice_write"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

        # ---------------------------------------------------------------- 9
        # fp32 linear with bf16 input (router without typecast)
        try:
            xb = torch.randn(1, 1, 32, 256)
            wf = torch.randn(256, 64)
            xbt = ttnn.from_torch(xb, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            wft = ttnn.from_torch(wf, device=dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            o = ttnn.linear(xbt, wft, dtype=ttnn.float32)
            got = ttnn.to_torch(o).float()
            ref = xb.to(torch.bfloat16).float() @ wf
            results["mixed_dtype_router_mm"] = f"PCC={pcc(ref, got):.6f}"
        except Exception as e:
            results["mixed_dtype_router_mm"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

        # ---------------------------------------------------------------- 10
        # block-union sparsity mask via max over 32-token blocks
        try:
            r = torch.rand(1, 1, 128, E)
            rt = ttnn.from_torch(r, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            rg = ttnn.reshape(rt, (1, 4, 32, E))
            m = ttnn.max(rg, dim=2, keepdim=True)
            mrm = ttnn.to_layout(m, ttnn.ROW_MAJOR_LAYOUT)
            got = ttnn.to_torch(mrm).float()
            ref = r.reshape(1, 4, 32, E).to(torch.bfloat16).float().max(dim=2, keepdim=True).values
            results["block_union_mask"] = f"PCC={pcc(ref, got):.6f} shape={tuple(mrm.shape)}"
        except Exception as e:
            results["block_union_mask"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

    finally:
        ttnn.close_device(dev)

    # ------------------------------------------------------------------ 11
    # nlp_create_qkv_heads_decode with num_kv_heads=0 (the q-only head-split
    # candidate) crashes host-side (core dump), so it runs in a subprocess.
    # Kept as the reproducible record of the op-contract blocker.
    import subprocess
    import sys

    try:
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, ttnn\n"
                "dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)\n"
                "x = ttnn.from_torch(torch.randn(1, 1, 1, 20 * 256), device=dev, dtype=ttnn.bfloat16,"
                " layout=ttnn.TILE_LAYOUT)\n"
                "q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(x, num_heads=20, num_kv_heads=0)\n"
                "ttnn.close_device(dev)\n"
                "print('UNEXPECTED_SUCCESS')\n",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if "UNEXPECTED_SUCCESS" in child.stdout:
            results["create_qkv_heads_kv0"] = "UNEXPECTED SUCCESS (blocker gone; revisit the q-only head split)"
        else:
            results[
                "create_qkv_heads_kv0"
            ] = f"CRASHES as expected (returncode {child.returncode}) - op-contract blocker"
    except subprocess.TimeoutExpired:
        # Observed both as a fast host-side core dump and as a hang; either
        # way the q-only split is unusable. The killed child can leave the
        # board needing a reset before the next open.
        results["create_qkv_heads_kv0"] = "HANGS (killed at 120s) - op-contract blocker"

    print("\n===== PROBE RESULTS =====")
    for k_, v_ in results.items():
        print(f"{k_:32s} {v_}")


if __name__ == "__main__":
    main()
