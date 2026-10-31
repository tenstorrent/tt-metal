# greedy masked-vs-unmasked checks (M1). needs c00_lib.py exec'd first
def masked_ref_argmax(lrow, al):
    return int(torch.where(al, lrow, torch.full_like(lrow, float("-inf"))).argmax())


def run_case(name, kv=None, topk_far=True, sp=None):
    gen = torch.Generator().manual_seed(11 + (kv or 0))
    slots = list(range(W))
    res = {}
    st = fresh(slots)
    tokens, start, nv = make_block(st, kv=kv)
    orig = GM.has_structured_row
    GM.has_structured_row = lambda b, live: True
    try:  # a masked (all ones) round: split trace A keeps the logits; the greedy single trace does not
        ones0, _ = verify(tokens, start, nv, mask_block({}), sp or sp_greedy())
    finally:
        GM.has_structured_row = orig
    r = runner_for_round(max(int(nv.max()), 0))
    L0 = logits_rows(r, slots)
    st = fresh(slots)
    tokens, start, nv = make_block(st, kv=kv)
    base, _ = verify(tokens, start, nv, None, sp or sp_greedy())
    log(f"CASE {name}: runner n={r.n} kv={kv} nv={nv.tolist()}")
    S = [s for s in (1, 3, 5, 9, 14) if s < W]
    allowed = {s: rand_allowed(0.001, gen) for s in S}
    topk_far = topk_far and 3 in S
    far_tok = None
    if topk_far:
        far_tok = int(L0[3][0].argsort(descending=True)[500])
        allowed[3] = torch.zeros(V, dtype=torch.bool)
        allowed[3][far_tok] = True
    # all-ones mask through the masked trace (forced): identical to base
    st = fresh(slots)
    tokens, start, nv = make_block(st, kv=kv)
    orig = GM.has_structured_row
    GM.has_structured_row = lambda b, live: True
    try:
        ones, dt1 = verify(tokens, start, nv, mask_block({}), sp or sp_greedy())
    finally:
        GM.has_structured_row = orig
    res["allones_equal_base"] = bool(torch.equal(ones[slots], base[slots]))
    # random masks on S, nv = 0 for those rows
    st = fresh(slots)
    tokens, start, nv = make_block(st, structured=S, kv=kv)
    out, dt2 = verify(tokens, start, nv, mask_block(allowed), sp or sp_greedy())
    L1 = logits_rows(r, slots)
    ok_ref, ok_allowed = True, True
    for s in S:
        t = int(out[s, 0])
        ref = masked_ref_argmax(L1[s][0], allowed[s])
        ok_ref &= t == ref
        ok_allowed &= bool(allowed[s][t])
        if not t == ref:
            log(f"  MISMATCH slot {s}: got {t} ref {ref} (allowed {int(allowed[s].sum())})")
    res["structured_equals_host_ref"] = ok_ref
    res["structured_in_allowed"] = ok_allowed
    if topk_far:
        res["far_token_outside_top32_chosen"] = int(out[3, 0]) == far_tok and int(base[3, 0]) != far_tok
    others = [s for s in slots if s not in S]
    res["nonstructured_rows_identical_to_base"] = bool(torch.equal(out[others], base[others]))
    log(f"RESULT {name} " + json.dumps(res) + f" masked_round_s={dt2:.3f} ones_round_s={dt1:.3f}")
    return all(res.values())


okA = run_case("greedy_k5")
okB = run_case("greedy_k3", kv=3) if model.spec_adapt is not None else True
log(f"C10 DONE ok={okA and okB}")
