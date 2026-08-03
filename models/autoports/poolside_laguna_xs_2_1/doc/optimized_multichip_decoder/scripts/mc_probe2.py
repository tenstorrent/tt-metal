"""Focused candidate probe (single weight-load, layer 4 full+MoE).
Candidates: router LoFi fidelity, SiLU fused into gate matmuls, fused matmul_reduce_scatter (WO).
Reports traced decode ms/tok + decode-output PCC vs baseline."""
import time
import types

import numpy as np
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt import optimized_decoder as OD
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder

LAYER = 4
ITERS = 50
HIDDEN = 2048
PREFILL = 512


def pcc(a, b):
    a = a.flatten().float().numpy()
    b = b.flatten().float().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    try:
        cfg = R.build_config()
        raw = W.load_layer_tensors(LAYER)
        dec = MultichipDecoder.from_state_dict(
            raw, hf_config=cfg, layer_idx=LAYER, mesh_device=dev, max_seq_len=PREFILL + 64
        )
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=PREFILL + 64, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        mm = ttnn.ReplicateTensorToMesh(dev)
        torch.manual_seed(0)
        xt = ttnn.from_torch(
            torch.randn(1, PREFILL, HIDDEN) * 0.5,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        dec.prefill_forward(xt, kv, pt, user_id=0, start_pos=0)
        ttnn.synchronize_device(dev)
        x_dev = ttnn.from_torch(
            torch.randn(1, 1, 1, HIDDEN) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mm
        )
        cur = ttnn.from_torch(
            torch.tensor([PREFILL], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        ridx = ttnn.from_torch(
            torch.tensor([[PREFILL]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )

        def meas(n=ITERS):
            out = dec.decode_forward(x_dev, cur, ridx, pt, kv)
            ttnn.synchronize_device(dev)
            ref = (
                ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[0:1].float().reshape(1, 1, HIDDEN)
            )
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            dec.decode_forward(x_dev, cur, ridx, pt, kv)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.synchronize_device(dev)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            ms = (time.perf_counter() - t0) * 1e3 / n
            ttnn.release_trace(dev, tid)
            return ms, ref

        results = []
        ms, ref0 = meas()
        results.append(("baseline", ms, 1.0))
        print("baseline", ms)

        # C1: router LoFi
        dec._ck_router = dec._ck_lofi
        try:
            ms, ref = meas()
            results.append(("router_lofi", ms, pcc(ref, ref0)))
            print("router_lofi", ms, pcc(ref, ref0))
        except Exception as e:
            print("router_lofi ERR", str(e)[:150])
        dec._ck_router = dec._ck_hifi2

        # C2: SiLU fused into gate sparse matmul + shared/dense gate dram_mm
        # patch _sparse_pc to add SiLU fused activation ONLY for gate (n==H means down; gate n==I=512)
        # Simpler: patch _moe to fuse silu in gate sparse matmul.
        orig_moe = dec._moe

        def moe_silu(self, ln_flat, m, sharded):
            cfg = self.cfg
            GE, LE = self.global_experts, self.local_experts
            H, I, K = cfg.hidden, cfg.moe_intermediate, cfg.top_k
            T = ln_flat.shape[2]
            logits = ttnn.linear(ln_flat, self.w["gate_w"], compute_kernel_config=self._ck_router)
            scores = ttnn.sigmoid(logits)
            sel = ttnn.add(scores, self.w["e_bias"])
            _, idx = ttnn.topk(ttnn.typecast(sel, ttnn.bfloat16), k=K, dim=-1, sorted=True)
            wsel = ttnn.gather(scores, dim=3, index=idx)
            if cfg.norm_topk_prob:
                wsel = ttnn.div(wsel, ttnn.sum(wsel, dim=3, keepdim=True))
            if cfg.routed_scaling != 1.0:
                wsel = ttnn.multiply(wsel, cfg.routed_scaling)
            dense = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=idx, src=wsel)
            dense_local = ttnn.matmul(dense, self.w["ep_sel"], compute_kernel_config=self._ck_router)
            union = ttnn.sum(dense_local, dim=2, keepdim=True)
            sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
            a = ttnn.reshape(ln_flat, (1, 1, T, H))
            moe_mem = ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG
            otile = ttnn.Tile([32, 32])
            gu_pc = OD._sparse_pc(I, T, H)
            gate_pc = OD._sparse_pc(I, T, H)
            gate_pc.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
            gate_o = ttnn.sparse_matmul(
                a,
                self.w["exp_gate"],
                sparsity=sparsity,
                program_config=gate_pc,
                compute_kernel_config=self._ck_moe,
                memory_config=moe_mem,
                output_tile=otile,
            )
            up_o = ttnn.sparse_matmul(
                a,
                self.w["exp_up"],
                sparsity=sparsity,
                program_config=gu_pc,
                compute_kernel_config=self._ck_moe,
                memory_config=moe_mem,
                output_tile=otile,
            )
            gate_o = ttnn.reshape(gate_o, (1, LE, T, I))
            up_o = ttnn.reshape(up_o, (1, LE, T, I))
            glu = ttnn.mul(gate_o, up_o)
            dn_pc = OD._sparse_pc(H, T, I)
            down_o = ttnn.sparse_matmul(
                glu,
                self.w["exp_down"],
                sparsity=sparsity,
                is_input_a_sparse=True,
                program_config=dn_pc,
                compute_kernel_config=self._ck_moe,
                memory_config=moe_mem,
                output_tile=otile,
            )
            wv = ttnn.reshape(dense_local, (1, T, LE))
            wv = ttnn.permute(wv, (0, 2, 1))
            wv = ttnn.reshape(wv, (1, LE, T, 1))
            routed_local = ttnn.reshape(ttnn.sum(ttnn.mul(down_o, wv), dim=1), (1, 1, T, H))
            shared_partial = self._glu_mlp(ln_flat, "sh", cfg.hidden, cfg.shared_intermediate, self._ck_shared, sharded)
            return self._reduce(ttnn.add(routed_local, ttnn.reshape(shared_partial, (1, 1, T, H))))

        dec._moe = types.MethodType(moe_silu, dec)
        try:
            ms, ref = meas()
            results.append(("moe_gate_silu_fused", ms, pcc(ref, ref0)))
            print("moe_gate_silu_fused", ms, pcc(ref, ref0))
        except Exception as e:
            print("moe_silu ERR", type(e).__name__, str(e)[:200])
        dec._moe = orig_moe

        print("\n=== PROBE2 RESULTS layer", LAYER, "===")
        for n, ms, p in results:
            print(f"  {n:24s} {ms:.4f} ms/tok  pcc {p:.6f}")
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
