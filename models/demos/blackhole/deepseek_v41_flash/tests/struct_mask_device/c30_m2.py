# M2: per-position masks, structured rows that carry drafts. needs c00_lib.py. Run with DSV41_STRUCTURED_M2=1
import os as _os

M2 = _os.environ.get("DSV41_STRUCTURED_M2") == "1"


def mask_block_pos(allowed_by_slot_pos, width=1 + K):
    """{slot: {j: bool[V]}} -> int32 [W, width, 4040] all ones elsewhere"""
    blk = torch.full((W, width, WORDS), -1, dtype=torch.int32)
    for s, d in allowed_by_slot_pos.items():
        for j, al in d.items():
            blk[s, j] = pack_bool(al)
    return blk


def case_m2(name, kv=None):
    if not M2:
        log(f"RESULT {name} skipped (DSV41_STRUCTURED_M2 not set)")
        return True
    gen = torch.Generator().manual_seed(21)
    slots = list(range(W))
    res = {}
    st = fresh(slots)
    tokens, start, nv = make_block(st, kv=kv)
    base, _ = verify(tokens, start, nv, None, sp_greedy())
    n = base.shape[1]
    r = runner_for_round(int(nv.max()))
    L0 = logits_rows(r, slots)
    S = [s for s in (2, 6, 10, 13) if s < W] or [0]
    # (a) the baseline argmax of every position is allowed (plus a random set): output identical to the unmasked round, drafts keep being accepted
    allowed_keep = {}
    for s in S:
        allowed_keep[s] = {}
        for j in range(n):
            al = rand_allowed(0.0005, gen)
            al[int(base[s, j])] = True
            allowed_keep[s][j] = al
    st = fresh(slots)
    tokens, start, nv = make_block(st, kv=kv)
    out_a, _ = verify(tokens, start, nv, mask_block_pos(allowed_keep, width=1 + K), sp_greedy())
    res["keep_equals_base"] = bool(torch.equal(out_a[slots], base[slots]))
    # (b) the baseline argmax of every position is EXCLUDED: every position equals the masked host argmax (rows j of the buffer line up with block position j)
    allowed_ex = {}
    for s in S:
        allowed_ex[s] = {}
        for j in range(n):
            al = rand_allowed(0.0005, gen)
            al[int(base[s, j])] = False
            if not bool(al.any()):
                al[(int(base[s, j]) + 1) % V] = True
            allowed_ex[s][j] = al
    st = fresh(slots)
    tokens, start, nv = make_block(st, kv=kv)
    out_b, _ = verify(tokens, start, nv, mask_block_pos(allowed_ex, width=1 + K), sp_greedy())
    L1 = logits_rows(r, slots)
    ok = True
    for s in S:
        for j in range(n):
            # row j's logits depend on the inputs of rows <= j (the offered drafts): they are the same block as in the baseline, so the unmasked logits are L0
            ref = masked_ref_argmax(L1[s][j], allowed_ex[s][j])
            if int(out_b[s, j]) != ref:
                ok = False
                log(f"  MISMATCH slot {s} pos {j}: got {int(out_b[s, j])} ref {ref}")
    res["excluded_positions_equal_host_ref"] = ok
    res["excluded_differs_from_base"] = all(int(out_b[s, 0]) != int(base[s, 0]) for s in S)
    others = [s for s in slots if s not in S]
    res["nonstructured_identical"] = bool(torch.equal(out_b[others], base[others]))
    log(f"RESULT {name} " + json.dumps(res))
    return all(res.values())


def masked_ref_argmax(lrow, al):
    return int(torch.where(al, lrow, torch.full_like(lrow, float("-inf"))).argmax())


ok1 = case_m2("m2_k5")
ok2 = case_m2("m2_k3", kv=3) if model.spec_adapt is not None else True
log(f"C30 DONE ok={ok1 and ok2}")
