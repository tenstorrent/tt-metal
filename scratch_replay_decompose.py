"""Decisive, final-tensor-only diagnostic for the sparse_sdpa_msa diag-mask bug.

No async CB dumps. Run the op once, then decide between hypotheses purely from the
output tensor + inputs:

  H1 (leak):    device tracks the NON-CAUSAL (mask-ignored) reference -> -inf columns
                are not being applied / not becoming 0 -> they leak into the numerator.
  H0 (correct): device tracks the CAUSAL reference -> mask works; bug is elsewhere.

Focus offset==96 queries: boundary_col=1, full_neginf=0 -> ONLY the vmask partial tile,
zero full -inf tiles. This is the minimal bug surface (31 masked cols in ONE tile) and
the sharpest causal-vs-noncausal contrast.

  M3_REPLAY_DUMP=1 pytest tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa_msa.py -k replay -s   (for env)
  ... or run directly via the test's device fixture. Here: standalone with a device.
"""
import os
import torch
import ttnn
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests/ttnn/unit_tests/operations/sdpa"))
from sparse_sdpa_msa_test_utils import sparse_attention_ref_msa, BLK_KV  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    dev_id = 0
    device = ttnn.open_device(device_id=dev_id)
    try:
        blob = torch.load("/tmp/m3_opdump/L31_worst.pt")
        q, k, v = blob["q"].float(), blob["k"].float(), blob["v"].float()
        block_ids = blob["block_ids"].to(torch.int64)
        cs, scale = int(blob["chunk_start_idx"]), float(blob["scale"])
        Hql = q.shape[1]
        print(f"[dec] shard chunk_start={cs} scale={scale:.6f} q={tuple(q.shape)} k={tuple(k.shape)}")

        q_t = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_t = ttnn.from_torch(k.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_t = ttnn.from_torch(v.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        bids_t = ttnn.from_torch((block_ids & 0xFFFFFFFF).to(torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out = ttnn.transformer.sparse_sdpa_msa(q_t, k_t, v_t, bids_t, scale=scale, block_size=BLK_KV, chunk_start_idx=cs, cluster_axis=None)
        dev = ttnn.to_torch(out)[:, :Hql].float()

        idx_ref = block_ids.clone()
        idx_ref[idx_ref == 0xFFFFFFFF] = -1
        idx_ref = idx_ref.to(torch.int32)

        # References on identical inputs:
        ref_causal = sparse_attention_ref_msa(q, k, v, idx_ref, scale, causal=True, chunk_start_idx=cs)
        ref_nomask = sparse_attention_ref_msa(q, k, v, idx_ref, scale, causal=False, chunk_start_idx=cs)  # mask IGNORED

        # PROBE ref: diagonal block fully excluded (attend past blocks only). Remove the diag block from each
        # query's selection.
        idx_nodiag = idx_ref.clone()
        BLK = BLK_KV
        for s in range(idx_nodiag.shape[2]):
            diag = (cs + s) // BLK
            rowsel = idx_nodiag[0, 0, s]
            idx_nodiag[0, 0, s] = torch.where(rowsel == diag, torch.full_like(rowsel, -1), rowsel)
        ref_nodiag = sparse_attention_ref_msa(q, k, v, idx_nodiag, scale, causal=False, chunk_start_idx=cs)

        print(f"[dec] overall PCC  dev-vs-CAUSAL(correct)     = {pcc(dev, ref_causal):.5f}")
        print(f"[dec] overall PCC  dev-vs-NOMASK(mask-ignored) = {pcc(dev, ref_nomask):.5f}")
        print(f"[dec] overall PCC  dev-vs-NODIAG(diag-excluded)= {pcc(dev, ref_nodiag):.5f}   <-- PROBE: force-all-inf")

        S = dev.shape[2]
        for target in (0, 32, 64, 96, 127):
            ss = [s for s in range(S) if (cs + s) % BLK_KV == target]
            if not ss:
                continue
            idx = torch.tensor(ss)
            d = dev[:, :, idx, :]
            rc = ref_causal[:, :, idx, :]
            rn = ref_nomask[:, :, idx, :]
            # how different are the two refs here (0 => can't distinguish)
            sep = pcc(rc, rn)
            print(f"  offset=={target:>3}: n={len(ss):>2}  dev-vs-CAUSAL={pcc(d, rc):.4f}  dev-vs-NOMASK={pcc(d, rn):.4f}  (ref_causal-vs-nomask={sep:.4f})")

        # For offset==96 (pure vmask): per-head, is the error aligned with the masked-key V mean?
        s96 = [s for s in range(S) if (cs + s) % BLK_KV == 96]
        if s96:
            s = s96[0]
            pos = cs + s
            blk = pos // BLK_KV
            # masked keys = positions (pos+1 .. blk*128+127) within the diagonal block, IF that block is selected.
            masked_kv = list(range(pos + 1, (blk + 1) * BLK_KV))
            print(f"[dec] offset-96 query s={s} pos={pos} diag_blk={blk} n_masked_keys={len(masked_kv)}")
            # V for group 0 (single kv group on-device shard). v: [1,1,T,128]
            v0 = v[0, 0]  # [T,128]
            vm = v0[torch.tensor(masked_kv)].mean(0)  # [128] mean masked-key V
            d96 = dev[0, :, s, :]  # [Hq,128]
            rc96 = ref_causal[0, :, s, :]
            err = d96 - rc96  # [Hq,128]
            # cosine( error , (vm - rc) )  -> +1 means output pulled toward masked-V (leak)
            cos_leak = torch.nn.functional.cosine_similarity(err, (vm.unsqueeze(0) - rc96), dim=1)
            print(f"  |err| per head: min={err.norm(dim=1).min():.3f} max={err.norm(dim=1).max():.3f}")
            print(f"  cos(err, vm-ref) per head: min={cos_leak.min():.3f} max={cos_leak.max():.3f} mean={cos_leak.mean():.3f}")
            print(f"  (cos~+1 across heads => leak toward masked-key V confirmed)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
