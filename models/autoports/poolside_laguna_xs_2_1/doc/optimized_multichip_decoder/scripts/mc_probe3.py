"""Probe: packed MoE gate+up sparse_matmul vs separate (P1 from stage review).
Single weight-load, layer 4. Builds exp_gate_up = concat(exp_gate,exp_up,dim=-1) on device,
runs one sparse_matmul over [1,64,H,1024], splits on device, silu(gate)*up.
Reports traced decode ms/tok + decode-output PCC vs the separate baseline."""
import sys
import time
import types

import numpy as np
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt import optimized_decoder as OD
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
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
        # build packed gate+up weight on device (dim -1 = intermediate I: 512+512=1024)
        dec.w["exp_gate_up"] = ttnn.concat([dec.w["exp_gate"], dec.w["exp_up"]], dim=-1)
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
        results.append(("baseline_separate", ms, 1.0))
        print("baseline", ms)

        LE = dec.local_experts
        H = dec.cfg.hidden
        I = dec.cfg.moe_intermediate
        K = dec.cfg.top_k

        def moe_packed(self, ln_flat, m, sharded):
            cfg = self.cfg
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
            gu_pc = OD._sparse_pc(2 * I, T, H)  # packed N=1024
            gu = ttnn.sparse_matmul(
                a,
                self.w["exp_gate_up"],
                sparsity=sparsity,
                program_config=gu_pc,
                compute_kernel_config=self._ck_moe,
                memory_config=moe_mem,
                output_tile=otile,
            )
            gu = ttnn.reshape(gu, (1, LE, T, 2 * I))
            gate_o = ttnn.slice(gu, [0, 0, 0, 0], [1, LE, T, I])
            up_o = ttnn.slice(gu, [0, 0, 0, I], [1, LE, T, 2 * I])
            glu = ttnn.mul(ttnn.silu(gate_o), up_o)
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

        dec._moe = types.MethodType(moe_packed, dec)
        try:
            ms, ref = meas()
            results.append(("packed_gate_up", ms, pcc(ref, ref0)))
            print("packed_gate_up", ms, pcc(ref, ref0))
        except Exception as e:
            print("packed ERR", type(e).__name__, str(e)[:300])
        print("\n=== PROBE3 RESULTS layer", LAYER, "===")
        for n, ms, p in results:
            print(f"  {n:20s} {ms:.4f} ms/tok  pcc {p:.6f}")
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
